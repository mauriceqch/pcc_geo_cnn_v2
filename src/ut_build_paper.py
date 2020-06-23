import argparse
import logging
import os
import shutil

import yaml
import pandas as pd
import numpy as np

from utils.experiment import assert_exists, index_by_id
from utils.pc_metric import validate_opt_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def write_table_main(pc_names, opt_groups, mode_ids, ref, query_df, with_deltas=False):
    table = ''
    query_df = query_df.set_index('mode_id')
    table_data = []
    for pc_name in pc_names:
        for i, opt_group in enumerate(opt_groups):
            cur_pc_data = query_df.query(f'metric == "{opt_group}_psnr" & opt_group == "{opt_group}" & pc_name == "{pc_name}"')
            assert len(cur_pc_data) > 0
            values = cur_pc_data.loc[mode_ids, ref].values
            values = np.reshape(values, (-1, len(ref)))
            table_data.append({'name': pc_name, 'opt_group': opt_group, 'values': values})

    for opt_group in opt_groups:
        opt_values = np.array([x['values'] for x in table_data if x['opt_group'] == opt_group])
        opt_means = np.mean(opt_values, axis=0)
        opt_means = np.reshape(opt_means, (-1, len(ref)))
        table_data.append({'name': 'Average', 'opt_group': opt_group, 'values': opt_means})

    for i, row_data in enumerate(table_data):
        if i % 2 == 0:
            if row_data['name'] == 'Average':
                table += '\\hline\n'
            table += '\\multirow{2}{*}{' + row_data['name'].split('_')[0] + '}'
        table += ' & ' + row_data['opt_group'].upper() + ' & '
        str_values = []
        row_values = np.round(row_data['values'], decimals=2)
        # Keep ref dimension only (second)
        subcol_maxs1 = np.max(row_values, axis=0, keepdims=True)[0]
        subcol_maxs2 = [np.max(row_values[:, col][row_values[:, col] != subcol_maxs1[col]], axis=0, keepdims=True)[0]
                        for col in range(row_values.shape[1])]
        if with_deltas:
            deltas = [row_values[j] - row_values[j + 1] for j in range(len(row_values) - 1)]
            max1_delt = np.max(deltas, axis=0)
            max2_delt = np.max(deltas[deltas != max1_delt])
        for j in range(row_values.shape[0]):
            metric_vals = row_values[j]
            subcol_indexes = range(len(metric_vals))
            cell_str = []
            for subcol_idx in subcol_indexes:
                metric_val = metric_vals[subcol_idx]
                s = f'{metric_val:.2f}'
                if metric_val == subcol_maxs2[subcol_idx]:
                    s = '\\mathit{' + s + '}'
                if metric_val == subcol_maxs1[subcol_idx]:
                    s = '\\bm{' + s + '}'
                s = f'${s}$'
                if with_deltas and j < len(row_values) - 1:
                    delt = deltas[j][subcol_idx]
                    delt_s = ' ($' + f'{deltas[j]:+.2f}' + '$)'
                    if delt == max1_delt[subcol_idx]:
                        delt_s = ' ($\\bm{' + f'{deltas[j]:+.2f}' + '}$)'
                    if delt == max2_delt[subcol_idx]:
                        delt_s = ' ($\\mathit{' + f'{deltas[j]:+.2f}' + '}$)'
                    s += delt_s
                cell_str.append(s)
            str_values.append(' / '.join(cell_str))
        table += ' & '.join(str_values)
        if i % 2 == 0:
            line_end = '\\\\ \\cline{2-' + str(len(mode_ids) + 2) + '}\n'
        else:
            line_end = '\\\\ \\hline\n'
        table += line_end
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ut_build_paper.py', description='Build figures and tables.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment_path', help='Experiments file path.')
    parser.add_argument('output_path', help='Output folder.')
    args = parser.parse_args()

    with open(args.experiment_path, 'r') as f:
        experiments = yaml.load(f.read(), Loader=yaml.FullLoader)
    keys = ['MPEG_DATASET_DIR', 'EXPERIMENT_DIR', 'mpeg_modes', 'model_configs', 'opt_metrics', 'eval_modes']
    MPEG_DATASET_DIR, EXPERIMENT_DIR, mpeg_modes, model_configs, opt_metrics, eval_modes = [experiments[k] for k in keys]
    mpeg_path = os.path.join(EXPERIMENT_DIR, 'gpcc')
    eval_modes = index_by_id(eval_modes)

    validate_opt_metrics(opt_metrics, with_normals=True)
    os.makedirs(args.output_path, exist_ok=True)

    opt_groups = ['d1', 'd2']

    logger.info('Build tables')
    # eval_id, label, metric, mode_id, opt_group, pc_name
    bdsnr_df = pd.read_csv(os.path.join(EXPERIMENT_DIR, 'results', 'bdsnr.csv'))

    # Alpha table
    alpha_bdsnr_ref = ['trisoup-predlift/lossy-geom-lossy-attrs']
    alpha_modes = index_by_id(eval_modes['alpha']['modes'])
    alpha_mode_ids = [x for x in alpha_modes if x not in alpha_bdsnr_ref]
    alpha_df = bdsnr_df.query(f'eval_id == "alpha" & mode_id in {alpha_mode_ids}')
    pc_names = [x['pc_name'] for x in experiments['data']]

    alpha_table = '\\hline\n'
    alpha_table += 'Point cloud & Metric & ' + ' & '.join([alpha_modes[x]['label'] for x in alpha_mode_ids]) + '\\\\ \\hline\n'
    alpha_table += write_table_main(pc_names, opt_groups, alpha_mode_ids, alpha_bdsnr_ref, alpha_df)
    logger.info(f'Table for alpha comparisons with {alpha_bdsnr_ref} \n{alpha_table}')

    # Main table
    main_bdrate_ref = ['trisoup-predlift/lossy-geom-lossy-attrs', 'octree-predlift/lossy-geom-lossy-attrs']
    main_modes = index_by_id(eval_modes['main']['modes'])
    main_mode_ids = [x for x in main_modes if x not in main_bdrate_ref]
    main_df = bdsnr_df.query(f'eval_id == "main" & mode_id in {main_mode_ids}')

    main_table = '\\hline\n'
    main_table += 'Point cloud & Metric & '
    main_table += ' & '.join([main_modes[x].get('label', main_modes[x]['id']) for x in main_mode_ids]) + '\\\\ \\hline\n'
    main_table += write_table_main(pc_names, opt_groups, main_mode_ids, main_bdrate_ref, main_df)
    logger.info(f'Table for main comparisons with {main_bdrate_ref} \n{main_table}')

    logger.info('Loading figures')
    params = []
    for experiment in experiments['data']:
        opt_metrics_groups = [[x for x in opt_metrics if x.startswith(group)] for group in opt_groups]
        for eval_id, eval_mode in eval_modes.items():
            cur_modes = index_by_id(eval_mode['modes'])
            # Group metrics by prefix
            for opt_group, opt_metric_group in zip(opt_groups, opt_metrics_groups):
                pc_name = experiment['pc_name']
                pc_output_dir = os.path.join(EXPERIMENT_DIR, pc_name, 'results', eval_id)

                if os.path.exists(pc_output_dir):
                    input_fig = os.path.join(pc_output_dir, f'{opt_group}_opt_rd_curve_{opt_group}.pdf')
                    output_fig = os.path.join(args.output_path, f'fig_{eval_id}_{pc_name}_{opt_group}.pdf')
                    assert_exists(input_fig)
                    shutil.copyfile(input_fig, output_fig)
                else:
                    logger.warning(f'Ommiting eval condition {eval_id}, pc {pc_name}, opt_group {opt_group}')
    logger.info('Finished')
