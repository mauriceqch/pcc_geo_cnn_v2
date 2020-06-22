import argparse
import logging
import multiprocessing
import os

import yaml
from pyntcloud import PyntCloud

from map_color import run_mapcolor
from utils.experiment import assert_exists
from utils.parallel_process import parallel_process, Popen

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def get_n_points(f):
    return len(PyntCloud.from_file(f).points)


def run_mpeg_experiment(current_mpeg_output_dir, mpeg_cfg_path, input_pc, input_norm):
    input_pc_full = os.path.join(MPEG_DATASET_DIR, input_pc)
    input_norm_full = os.path.join(MPEG_DATASET_DIR, input_norm)
    os.makedirs(current_mpeg_output_dir, exist_ok=True)

    assert_exists(input_pc_full)
    assert_exists(input_norm_full)
    assert_exists(mpeg_cfg_path)

    return Popen(['make',
                  '-f', f'{MPEG_TMC13_DIR}/scripts/Makefile.tmc13-step',
                  '-C', current_mpeg_output_dir,
                  f'VPATH={mpeg_cfg_path}',
                  f'ENCODER={TMC13}',
                  f'DECODER={TMC13}',
                  f'PCERROR={PCERROR}',
                  f'SRCSEQ={input_pc_full}',
                  f'NORMSEQ={input_norm_full}'])


def run_gen_report(folder_path):
    return Popen(['python', 'mp_report.py', folder_path])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mp_run.py', description='Run MPEG experiments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment_path', help='Experiments file path.')
    parser.add_argument('--num_parallel', help='Number of parallel jobs.', default=multiprocessing.cpu_count(), type=int)
    args = parser.parse_args()

    with open(args.experiment_path, 'r') as f:
        experiments = yaml.load(f.read(), Loader=yaml.FullLoader)
    keys = ['MPEG_TMC13_DIR', 'PCERROR', 'MPEG_DATASET_DIR', 'EXPERIMENT_DIR', 'mpeg_modes', 'rates']
    MPEG_TMC13_DIR, PCERROR, MPEG_DATASET_DIR, EXPERIMENT_DIR, mpeg_modes, rates = [experiments[k] for k in keys]
    TMC13 = f'{MPEG_TMC13_DIR}/build/tmc3/tmc3'

    assert_exists(TMC13)
    assert_exists(PCERROR)
    assert_exists(MPEG_DATASET_DIR)

    output_path = os.path.join(EXPERIMENT_DIR, 'gpcc')

    logger.info('Starting GPCC experiments')
    params = []
    for mpeg_mode in mpeg_modes:
        mpeg_id = mpeg_mode['id']
        for experiment in experiments['data']:
            pc_name, cfg_name, input_pc, input_norm = \
                [experiment[x] for x in ['pc_name', 'cfg_name', 'input_pc', 'input_norm']]
            mpeg_output_dir = os.path.join(output_path, mpeg_id, pc_name)
            for rate in rates:
                current_mpeg_output_dir = os.path.join(mpeg_output_dir, rate)
                mpeg_cfg_path = f'{MPEG_TMC13_DIR}/cfg/{mpeg_id}/{cfg_name}/{rate}'
                pc_path = os.path.join(EXPERIMENT_DIR, 'gpcc', mpeg_id, pc_name, rate, f'{pc_name}.ply.bin.decoded.ply')
                if not os.path.exists(pc_path):
                    params.append((current_mpeg_output_dir, mpeg_cfg_path, input_pc, input_norm))
                else:
                    logger.info(f'{pc_path} exists')
    logger.info('Started GPCC experiments')
    # An SSD is highly recommended, extremely slow when running in parallel on an HDD due to parallel writes
    # If HDD, set parallelism to 1
    parallel_process(run_mpeg_experiment, params, args.num_parallel)

    logger.info('Finished GPCC experiments')

    logger.info('Starting point cloud recoloring')
    params = []
    for experiment in experiments['data']:
        pc_name, input_pc = [experiment[x] for x in ['pc_name', 'input_pc']]
        input_pc_full = os.path.join(MPEG_DATASET_DIR, input_pc)
        for model_config in mpeg_modes:
            mpeg_id = model_config['id']
            for rate in rates:
                pc_path = os.path.join(EXPERIMENT_DIR, 'gpcc', mpeg_id, pc_name, rate, f'{pc_name}.ply.bin.decoded.ply')
                cur_output_path = pc_path + '.color.ply'
                if not os.path.exists(cur_output_path):
                    if not os.path.exists(pc_path):
                        raise RuntimeError(f'Point clouds not found at {pc_path}')
                    else:
                        params.append((input_pc_full, pc_path, cur_output_path))
                else:
                    logger.info(f'{cur_output_path} exists')
    parallel_process(run_mapcolor, params, args.num_parallel)
    logger.info('Finished point cloud recoloring')

    logger.info('Generating GPCC experimental reports')
    params = []
    for mpeg_mode in mpeg_modes:
        mpeg_id = mpeg_mode['id']
        for experiment in experiments['data']:
            pc_name = experiment['pc_name']
            mpeg_output_dir = os.path.join(output_path, mpeg_id, pc_name)
            for rate in rates:
                current_mpeg_output_dir = os.path.join(mpeg_output_dir, rate)
                report_path = os.path.join(current_mpeg_output_dir, 'report.json')
                if not os.path.exists(report_path):
                    params.append((current_mpeg_output_dir,))
                else:
                    logger.info(f'{report_path} exists')
    logger.info('Generated GPCC experimental reports')
    parallel_process(run_gen_report, params, args.num_parallel)

    logger.info('Done')
