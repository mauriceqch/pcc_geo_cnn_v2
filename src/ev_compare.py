import argparse
import os
import json
import logging
import pprint
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from utils import bd
from glob import glob

from utils.matplotlib_utils import default_rc_params, linestyles_cycle, markers_cycle, load_rc_params, set_lims

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

x_col = 'pos_bits_per_input_point'
rcParams = default_rc_params(rcParams)


def read_json(file):
    with open(file, 'r') as f:
        return json.load(f)


def build_curves(data, ylabel, column, filename, output_path, ylim=None, xlim=None, legend_loc='lower right',
                 no_legend=False, lims=None):
    logger.info(f'Building curves with {ylabel}')

    data = [d.copy() for d in data]
    for d in data:
        d['reports'] = d['reports'].copy()
    fig, ax = plt.subplots()

    markers = markers_cycle()
    linestyles = linestyles_cycle()
    for d, marker, linestyle in zip(data, markers, linestyles):
        d['marker'] = marker
        d['linestyle'] = linestyle

    data_summary = []
    for d in data:
        df = d['reports']
        df['finite_mask'] = np.isfinite(df[column].values)
        logger.debug(f'{column} {df}')
        # if not np.all(data_finite):
        #     data_lossless = cur_data[~data_finite]
        #     data_lossless_bpp = np.min(data_lossless[:, 0])
        #     ax.axvline(x=data_lossless_bpp, label='_nolegend_', linestyle=data['linestyle'])

        df_finite = df[df['finite_mask']]
        ax.plot(df_finite[x_col], df_finite[column],
                label=d['label'], linestyle=d['linestyle'], marker=d['marker'])

        for _, row in df_finite.iterrows():
            data_summary.append({'mode_id': d['mode_id'], 'label': d['label'], 'metric': column,
                                 'ylabel': ylabel, 'x': row[x_col], 'y': row[column]})

    pd.DataFrame(data_summary).to_csv(os.path.join(output_path, filename + '_data.csv'))

    ax.set(xlabel='bits per input point', ylabel=ylabel)
    ax.set_xlim(left=0)
    set_lims(ax, lims)
    if not no_legend:
        ax.legend(loc=legend_loc)
    ax.locator_params(axis='x', nbins=6)
    ax.locator_params(axis='y', nbins=6)
    ax.grid(True)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    for ext in ['.pdf', '.png']:
        fig.savefig(os.path.join(output_path, filename + ext))

    message = ''
    for bdf, bdname in zip((bd.bdrate, bd.bdsnr), ('bdrate', 'bdsnr')):
        bddf = [{'metric': column, 'mode_id': d['mode_id'], 'label': d['label']} for d in data]

        for i, d1 in enumerate(data):
            for j, d2 in enumerate(data):
                df1 = d1['reports']
                df1 = df1.query('bd_mask and finite_mask')
                df2 = d2['reports']
                df2 = df2.query('bd_mask and finite_mask')
                bd_result = bdf(df2[[x_col, column]].values, df1[[x_col, column]].values)
                bddf[i][d2['mode_id']] = bd_result
        bddf = pd.DataFrame(bddf)

        bd_str = bddf.to_string()
        message += bd_str + '\n'
        print(bd_str)
        bddf.to_csv(os.path.join(output_path, filename + '_' + bdname + '.csv'))
    with open(os.path.join(output_path, filename + '.log'), 'w') as f:
        f.write(message)


def run(paths, patterns, labels, mode_ids, output_path, output_prefix, path_filter=None, modes=('d1', 'd2'),
        bd_ignore=(), no_legend=False, lims=None):
    for path in paths:
        assert os.path.exists(path), f'{path} does not exist'

    data = [{'reports': [{'path': gpath} for gpath in glob(os.path.join(path, pattern), recursive=True)],
             'path': path, 'pattern': pattern, 'label': label.replace('_', ' '), 'mode_id': mode_id}
            for path, pattern, label, mode_id in zip(paths, patterns, labels, mode_ids)]

    # Filtering paths and data
    filtered_data = []
    for d in data:
        if path_filter is not None:
            regexp = re.compile(path_filter)
            filtered_reports = []
        else:
            filtered_reports = d['reports']
        for report in d['reports']:
            path = report['path']
            if path_filter is not None:
                mask_value = regexp.search(report)
                if not mask_value:
                    logger.info(f'Ignoring {report}')
                else:
                    filtered_reports.append(report)
            bd_mask_value = not any(bdi in path for bdi in bd_ignore)
            report['bd_mask'] = bd_mask_value
            if not bd_mask_value:
                logger.info(f'Ignoring {path} for BD computations')

        if len(filtered_reports) == 0:
            logger.info(f'Ignoring {d["path"]} {d["pattern"]}')
        else:
            d['reports'] = filtered_reports
            filtered_data.append(d)
    data = filtered_data

    for d in data:
        reports = d['reports']
        logger.info(f'reports: {reports}')
        for i in range(len(reports)):
            report = reports[i]
            json_data = read_json(report['path'])
            rkeys = set(report.keys())
            jkeys = set(json_data.keys())
            assert len(rkeys.intersection(jkeys)) == 0, f'Key conflict rkeys {rkeys} jkeys {jkeys}'
            report = {**report, **json_data}
            reports[i] = report
        reports = pd.DataFrame(data=reports)
        d['reports'] = reports.sort_values(by=x_col)

    curves = {
        'd1': (output_prefix + 'rd_curve_d1', 'd1_psnr', 'D1 PSNR (dB)', 'lower right'),
        'd2': (output_prefix + 'rd_curve_d2', 'd2_psnr', 'D2 PSNR (dB)', 'lower right')
    }
    curves = [curves[m] for m in modes]

    for (filename, column, ylabel, legend_loc) in curves:
        build_curves(data, ylabel, column, filename, output_path, legend_loc=legend_loc, no_legend=no_legend,
                     lims=lims)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ev_compare.py', description='Gathers reports and produces summary.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--paths', help='Input paths.', nargs='+', required=True)
    parser.add_argument('--patterns', help='Search patterns (ex: **/report.json).', nargs='+', required=True)
    parser.add_argument('--labels', help='Labels.', nargs='+', required=True)
    parser.add_argument('--mode_ids', help='Identifiers.', nargs='+', required=True)
    parser.add_argument('--output_path', help='Output directory path.', required=True)
    parser.add_argument('--output_prefix', help='Prefix for output files.', default='')
    parser.add_argument('--path_filter', help='Path based result filtering.')
    parser.add_argument('--modes', help='Modes to use for output: d1, d2 or both.', default=['d1', 'd2'], nargs='+')
    parser.add_argument('--rcParams', help='Dictionary of parameters to pass to rcParams (JSON format).', type=json.loads)
    parser.add_argument('--bd_ignore', help='Ignore certain reports (usually to make BD metrics comparables).', nargs='+')
    parser.add_argument('--no_legend', help='Remove legend.', default=False, action='store_true')
    parser.add_argument('--lims', help='xmin xmax ymin ymax. None for auto.', nargs='+')
    args = parser.parse_args()
    lims = args.lims
    if lims is not None:
        lims = [None if x == 'None' else float(x) for x in lims]

    if args.rcParams is not None:
        logger.info('Loaded rcParams configuration')
        pprint.pprint(args.rcParams)
        rcParams = load_rc_params(args.rcParams, rcParams)

    run(args.paths, args.patterns, args.labels, args.mode_ids, args.output_path, args.output_prefix, args.path_filter,
        args.modes, args.bd_ignore, args.no_legend, lims)
