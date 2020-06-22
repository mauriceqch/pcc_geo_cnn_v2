import argparse
import json
import logging
import multiprocessing
import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib import rcParams
from glob import glob
from utils.experiment import assert_exists, index_by_id
from utils.matplotlib_utils import default_rc_params, load_rc_params, render_legend
from utils.parallel_process import parallel_process, Popen
from utils.pc_metric import validate_opt_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def run_compare(paths, patterns, ids, labels, pc_output_dir, name, no_stream_redirection, path_filter, modes,
                cur_rcParams, bd_ignore, no_legend, lims):
    os.makedirs(pc_output_dir, exist_ok=True)
    additional_args = []

    if path_filter is not None:
        additional_args += ['--path_filter', path_filter]
    if modes is not None and len(modes) > 0:
        additional_args += ['--modes', *modes]
    if cur_rcParams is not None:
        additional_args += ['--rcParams', json.dumps(cur_rcParams)]
    if len(bd_ignore) > 0:
        additional_args += ['--bd_ignore', *bd_ignore]
    if no_legend:
        additional_args += ['--no_legend']
    if lims is not None:
        additional_args += ['--lims', *[str(x) for x in lims]]

    if no_stream_redirection:
        f = None
    else:
        f = open(os.path.join(pc_output_dir, f'ev_compare_{name}opt.log'), 'w')
    return Popen(['python', 'ev_compare.py',
                  '--paths', *paths,
                  '--patterns', *patterns,
                  '--labels', *labels,
                  '--mode_ids', *ids,
                  '--output_path', pc_output_dir,
                  '--output_prefix', name + '_opt_'] + additional_args, stdout=f, stderr=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ev_run_compare.py', description='Run eval compare between experiments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment_path', help='Experiments file path.')
    parser.add_argument('--num_parallel', help='Number of parallel jobs.', default=multiprocessing.cpu_count(), type=int)
    parser.add_argument('--no_stream_redirection', help='Disable stdout and stderr redirection.', default=False, action='store_true')
    parser.add_argument('--path_filter', help='Path based result filtering.')
    args = parser.parse_args()

    with open(args.experiment_path, 'r') as f:
        experiments = yaml.load(f.read(), Loader=yaml.FullLoader)
    keys = ['MPEG_DATASET_DIR', 'EXPERIMENT_DIR', 'mpeg_modes', 'model_configs', 'opt_metrics', 'eval_modes', 'bd_ignore']
    MPEG_DATASET_DIR, EXPERIMENT_DIR, mpeg_modes, model_configs, opt_metrics, eval_modes, bd_ignore = [experiments[k] for k in keys]
    mpeg_path = os.path.join(EXPERIMENT_DIR, 'gpcc')
    mpeg_modes = index_by_id(mpeg_modes)
    model_configs = index_by_id(model_configs)

    validate_opt_metrics(opt_metrics, with_normals=True)

    logger.info('Rendering legends')

    # Render legend
    for eval_mode in eval_modes:
        # Init
        eval_id, cur_modes = eval_mode['id'], eval_mode['modes']
        cur_modes = index_by_id(cur_modes)
        cur_rcParams = eval_mode.get('rcParams')
        default_rc_params(rcParams)
        load_rc_params(cur_rcParams, rcParams)

        # Get labels
        labels = []
        for id, cur_mode in cur_modes.items():
            if id in mpeg_modes:
                mpeg_mode = mpeg_modes[id]
                label = cur_mode.get('label', mpeg_mode.get('label', id))
            elif id in model_configs:
                model_config = model_configs[id]
                label = cur_mode.get('label', model_config.get('label', id))
            else:
                raise RuntimeError(f'Unknown mode {id} {cur_mode}')
            labels.append(label)

        # Render legend
        figlegend = render_legend(labels)
        legend_path = os.path.join(EXPERIMENT_DIR, 'results', eval_id)
        os.makedirs(legend_path, exist_ok=True)
        for ext in ['.pdf', '.png']:
            figlegend.savefig(os.path.join(legend_path, 'legend' + ext))

    logger.info('Starting comparisons')
    conditions = []
    params = []
    for experiment in experiments['data']:
        opt_groups = ['d1', 'd2']
        opt_metrics_groups = [[x for x in opt_metrics if x.startswith(group)] for group in opt_groups]
        for eval_mode in eval_modes:
            eval_id, cur_modes = eval_mode['id'], eval_mode['modes']
            no_legend = eval_mode.get('no_legend', False)
            lims = eval_mode.get('lims', [None] * len(opt_groups))
            cur_modes = index_by_id(cur_modes)
            cur_rcParams = eval_mode.get('rcParams')

            # Group metrics by prefix
            for opt_group, opt_metric_group, cur_lims in zip(opt_groups, opt_metrics_groups, lims):
                pc_name = experiment['pc_name']

                data = []
                for id, cur_mode in cur_modes.items():
                    if id in mpeg_modes:
                        mpeg_mode = mpeg_modes[id]
                        mpeg_label = cur_mode.get('label', mpeg_mode.get('label', id))
                        mpeg_output_dir = os.path.join(mpeg_path, id, pc_name)
                        assert_exists(mpeg_output_dir)
                        data.append((mpeg_output_dir, '**/report.json', id, mpeg_label))
                    elif id in model_configs:
                        model_config = model_configs[id]
                        model_label = cur_mode.get('label', model_config.get('label', id))
                        pattern = f'**/report_{opt_group}.json'
                        model_output_dir = os.path.join(EXPERIMENT_DIR, pc_name, id)
                        reports_glob = os.path.join(model_output_dir, pattern)
                        reports = glob(reports_glob, recursive=True)
                        if not os.path.exists(model_output_dir):
                            logger.warning(f'Model folder {model_output_dir} was not found: omitting model')
                        elif len(reports) == 0:
                            logger.warning(f'No reports found in {reports_glob}')
                        else:
                            data.append((model_output_dir, pattern, id, model_label))
                    else:
                        raise RuntimeError(f'Unknown id {id}')

                if len(data) > 0:
                    pc_output_dir = os.path.join(EXPERIMENT_DIR, pc_name, 'results', eval_id)
                    paths, patterns, ids, labels = zip(*data)
                    params.append((paths, patterns, ids, labels, pc_output_dir, opt_group, args.no_stream_redirection,
                                   args.path_filter, [opt_group], cur_rcParams, bd_ignore, no_legend, cur_lims))
                    conditions.append({'pc_name': pc_name, 'eval_id': eval_id, 'opt_group': opt_group})
                else:
                    logger.warning(f'Ommiting eval condition {eval_id} for {pc_name}')
    parallel_process(run_compare, params, args.num_parallel)

    logger.info('Merging data')
    merged_output_dir = os.path.join(EXPERIMENT_DIR, 'results')
    os.makedirs(merged_output_dir, exist_ok=True)
    data_types = ['data', 'bdrate', 'bdsnr']
    for data_type in data_types:
        df_list = []
        for cond in conditions:
            csv_file = os.path.join(EXPERIMENT_DIR, cond['pc_name'], 'results', cond['eval_id'],
                                    f"{cond['opt_group']}_opt_rd_curve_{cond['opt_group']}_{data_type}.csv")
            df = pd.read_csv(csv_file)
            for k, v in cond.items():
                df.insert(0, k, v)
            df['csv_file'] = csv_file
            df_list.append(df)
        output = pd.concat(df_list, ignore_index=True, sort=True)
        output.to_csv(os.path.join(merged_output_dir, data_type + '.csv'))
    logger.info('Finished')
