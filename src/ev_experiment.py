import argparse
import json
import logging
import multiprocessing
import os
import subprocess
from contextlib import ExitStack

import yaml
from pyntcloud import PyntCloud

from map_color import run_mapcolor
from utils import mpeg_parsing
from utils.experiment import assert_exists
from utils.parallel_process import parallel_process, Popen
from utils.pc_metric import avail_opt_metrics, validate_opt_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def flatten(l):
    return [item for sublist in l for item in sublist]


def print_progress(from_path, to_path, comment=''):
    if not isinstance(from_path, list):
        from_path = [from_path]
    if not isinstance(to_path, list):
        to_path = [to_path]
    from_path_str = ', '.join(from_path)
    to_path_str = ', '.join(to_path)
    logger.info(f'[{from_path_str}] -> [{to_path_str}] {comment}')


def run_pcerror(decoded_pc, input_norm, input_pc, pcerror_cfg_params, pcerror_path, pcerror_result):
    f = open(pcerror_result, 'w')
    return Popen([pcerror_path,
                  '-a', input_pc, '-b', decoded_pc, '-n', input_norm] + pcerror_cfg_params,
                 stdout=f, stderr=f)


def run_experiment(output_dir, model_dir, model_config, pc_name, pcerror_path, pcerror_cfg_path, input_pc, input_norm,
                   opt_metrics, max_deltas, fixed_threshold, no_merge_coding, num_parallel, no_stream_redirection=False):
    for f in [model_dir, pcerror_path, pcerror_cfg_path, input_pc, input_norm]:
        assert_exists(f)
    validate_opt_metrics(opt_metrics, with_normals=input_norm is not None)
    with open(pcerror_cfg_path, 'r') as f:
        pcerror_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    opt_group = ['d1', 'd2']
    enc_pc_filenames = [f'{pc_name}_{x}.ply.bin' for x in opt_group]
    dec_pc_filenames = [f'{x}.ply' for x in enc_pc_filenames]
    dec_pc_color_filenames = [f'{x}.color.ply' for x in dec_pc_filenames]
    pcerror_result_filenames = [f'{x}.pc_error' for x in dec_pc_filenames]
    enc_pcs = [os.path.join(output_dir, x) for x in enc_pc_filenames]
    dec_pcs = [os.path.join(output_dir, x) for x in dec_pc_filenames]
    dec_pcs_color = [os.path.join(output_dir, x) for x in dec_pc_color_filenames]
    pcerror_results = [os.path.join(output_dir, x) for x in pcerror_result_filenames]
    exp_reports = [os.path.join(output_dir, f'report_{x}.json') for x in opt_group]

    compress_log = os.path.join(output_dir, 'compress.log')
    decompress_log = os.path.join(output_dir, 'decompress.log')

    # Create folder
    os.makedirs(output_dir, exist_ok=True)

    resolution = pcerror_cfg['resolution']

    # Encoding or Encoding/Decoding with merge_coding option
    if all(os.path.exists(x) for x in enc_pcs) and (no_merge_coding or all(os.path.exists(x) for x in dec_pcs)):
        print_progress(input_pc, enc_pcs, '(exists)')
    else:
        print_progress(input_pc, enc_pcs)
        with ExitStack() as stack:
            if no_stream_redirection:
                f = None
            else:
                f = open(compress_log, 'w')
                stack.enter_context(f)
            additional_params = []
            if not no_merge_coding:
                additional_params += ['--dec_files', *dec_pcs]
            if fixed_threshold:
                additional_params += ['--fixed_threshold']
            subprocess.run(['python', 'compress_octree.py',  # '--debug',
                            '--input_files', input_pc,
                            '--input_normals', input_norm,
                            '--output_files', *enc_pcs,
                            '--checkpoint_dir', model_dir,
                            '--opt_metrics', *opt_metrics,
                            '--max_deltas', *map(str, max_deltas),
                            '--resolution', str(resolution + 1),
                            '--model_config', model_config] + additional_params, stdout=f, stderr=f, check=True)

    # Decoding, skipped with merge_coding option
    if all(os.path.exists(x) for x in dec_pcs):
        print_progress(enc_pcs, dec_pcs, '(exists)')
    elif not no_merge_coding:
        print_progress(enc_pcs, dec_pcs, '(merge_coding)')
    else:
        print_progress(enc_pcs, dec_pcs)
        with ExitStack() as stack:
            if no_stream_redirection:
                f = None
            else:
                f = open(decompress_log, 'w')
                stack.enter_context(f)
            subprocess.run(['python', 'decompress_octree.py',  # '--debug',
                            '--input_files', *enc_pcs,
                            '--output_files', *dec_pcs,
                            '--checkpoint_dir', model_dir,
                            '--model_config', model_config], stdout=f, stderr=f, check=True)

    # Color mapping
    mc_params = []
    if all(os.path.exists(x) for x in dec_pcs_color):
        print_progress(dec_pcs, dec_pcs_color, '(exists)')
    else:
        for dp, dpc in zip(dec_pcs, dec_pcs_color):
            print_progress(dp, dpc)
            mc_params.append((input_pc, dp, dpc))
    parallel_process(run_mapcolor, mc_params, num_parallel)

    pcerror_cfg_params = [[f'--{k}', str(v)] for k, v in pcerror_cfg.items()]
    pcerror_cfg_params = flatten(pcerror_cfg_params)
    params = []
    for pcerror_result, decoded_pc in zip(pcerror_results, dec_pcs):
        if os.path.exists(pcerror_result):
            print_progress(decoded_pc, pcerror_result, '(exists)')
        else:
            print_progress(decoded_pc, pcerror_result)
            params.append((decoded_pc, input_norm, input_pc, pcerror_cfg_params, pcerror_path, pcerror_result))
    parallel_process(run_pcerror, params, num_parallel)

    for pcerror_result, enc_pc, decoded_pc, experiment_report in zip(pcerror_results, enc_pcs, dec_pcs, exp_reports):
        if os.path.exists(experiment_report):
            print_progress('all', experiment_report, '(exists)')
        else:
            print_progress('all', experiment_report)
            pcerror_data = mpeg_parsing.parse_pcerror(pcerror_result)

            pos_total_size_in_bytes = os.stat(enc_pc).st_size
            input_point_count = len(PyntCloud.from_file(input_pc).points)
            data = {
                'pos_total_size_in_bytes': pos_total_size_in_bytes,
                'pos_bits_per_input_point': pos_total_size_in_bytes * 8 / input_point_count,
                'input_point_count': input_point_count
            }
            data = {**data, **pcerror_data}
            with open(experiment_report, 'w') as f:
                json.dump(data, f, sort_keys=True, indent=4)

            # Debug
            with open(enc_pc + '.enc.metric.json', 'r') as f:
                enc_metrics = json.load(f)
            diff = abs(enc_metrics['d1_psnr'] - data['d1_psnr'])
            logger.info(f'D1 PSNR diff between encoder and decoder: {diff}')
            assert diff < 0.01, f'encoded {enc_pc} with D1 {enc_metrics["d1_psnr"]} but decoded {decoded_pc} with D1 {data["d1_psnr"]}dB'

    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ev_experiment.py', description='Run experiment for a point cloud.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', help='Output directory', required=True)
    parser.add_argument('--model_dir', help='Model directory', required=True)
    parser.add_argument('--model_config', help='Model configuration', required=True)
    parser.add_argument('--pc_name', help='Point cloud name', required=True)
    parser.add_argument('--input_pc', help='Path to input point cloud', required=True)
    parser.add_argument('--input_norm', help='Path to input point cloud normals', required=True)
    parser.add_argument('--pcerror_path', help='Path to pcerror executable', required=True)
    parser.add_argument('--pcerror_cfg_path', help='Path to pcerror configuration', required=True)
    parser.add_argument('--opt_metrics', nargs='+', help=f'Optimization metrics used. Available: {avail_opt_metrics}', required=True)
    parser.add_argument('--max_deltas', nargs='+', help=f'Max deltas tested during optimization.', required=True)
    parser.add_argument('--fixed_threshold', help='Enable fixed thresholding.', default=False, action='store_true')
    parser.add_argument('--num_parallel', help='Number of parallel jobs', default=multiprocessing.cpu_count(), type=int)
    parser.add_argument('--no_stream_redirection', help='Disable stdout and stderr redirection.', default=False, action='store_true')
    parser.add_argument('--no_merge_coding', help='Do not merge encoding and decoding.', default=False, action='store_true')
    args = parser.parse_args()

    run_experiment(args.output_dir, args.model_dir, args.model_config, args.pc_name, args.pcerror_path,
                   args.pcerror_cfg_path, args.input_pc, args.input_norm, args.opt_metrics, args.max_deltas,
                   args.fixed_threshold, args.no_merge_coding, args.num_parallel, args.no_stream_redirection)
