import argparse
import logging
import os
from glob import glob

import numpy as np
import yaml
import tensorflow as tf
from utils.experiment import index_by_id, assert_exists
import matplotlib.pyplot as plt
from matplotlib import rcParams

from utils.matplotlib_utils import default_rc_params, set_lims, load_rc_params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def get_model_dir(exp_dir, model_id, lmbda_str):
    return os.path.join(exp_dir, 'models', model_id, lmbda_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ut_run_render.py', description='Run experiments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment_path', help='Experiments file path.')
    parser.add_argument('output_path', help='Output folder.')
    args = parser.parse_args()

    with open(args.experiment_path, 'r') as f:
        experiments = yaml.load(f.read(), Loader=yaml.FullLoader)
    keys = ['MPEG_DATASET_DIR', 'EXPERIMENT_DIR', 'model_configs', 'tensorboard_plots']
    MPEG_DATASET_DIR, EXPERIMENT_DIR, model_configs, tensorboard_plots = [experiments[x] for x in keys]
    model_configs = index_by_id(model_configs)

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    rcParams = default_rc_params(rcParams)
    custom_rc_params = tensorboard_plots.get('rcParams')
    if custom_rc_params is not None:
        rcParams = load_rc_params(custom_rc_params, rcParams)

    logger.info('Build tensorboard plots')
    params = []
    for tb_plot in tensorboard_plots['plots']:
        tb_plot_id = tb_plot['id']
        subplots = index_by_id(tb_plot['subplots'])
        model_id = tb_plot['model_id']

        model_config = model_configs[model_id]
        lambdas = tb_plot.get('lambdas', model_config['lambdas'])
        checkpoint_id = model_config.get('checkpoint_id', model_id)

        figax = {}
        for subplot_id, subplot in subplots.items():
            fig, ax = plt.subplots()
            figax[subplot_id] = (fig, ax)

        tb_data = {}
        for lmbda in lambdas:
            lmbda_str = f'{lmbda:.2e}'
            model_dir = get_model_dir(EXPERIMENT_DIR, checkpoint_id, lmbda_str)
            done_path = os.path.join(model_dir, 'done')
            if not os.path.exists(done_path):
                logger.info(f'Ignoring {model_dir}')
                pass
            logger.info(f'Processing {model_dir}')

            event_files = glob(os.path.join(model_dir, 'train', '*'))
            assert len(event_files) > 0
            logger.info(f'Event files {event_files}')
            event_data = {}

            # Init
            for subplot_id in subplots:
                event_data[subplot_id] = []

            # Gather event data
            for event_file in event_files:
                events_it = tf.train.summary_iterator(event_file)
                for event in events_it:
                    step = event.step
                    summaries = event.summary.value
                    if len(summaries) > 0:
                        for summary in summaries:
                            for subplot_id, subplot in subplots.items():
                                tag = subplot['tag']
                                if summary.tag == tag:
                                    event_data[subplot_id].append([step, summary.simple_value])

            # Process and plot
            for subplot_id, subplot in subplots.items():
                arr = np.asarray(event_data[subplot_id])
                event_data[subplot_id] = arr
                fig, ax = figax[subplot_id]
                ax.plot(arr[:, 0], arr[:, 1], label=lmbda_str)

            tb_data[lmbda] = event_data

        for subplot_id, subplot in subplots.items():
            fig, ax = figax[subplot_id]
            ax.set(xlabel='Training steps', ylabel=subplot['label'])
            ax.locator_params(axis='x', nbins=6)
            ax.locator_params(axis='y', nbins=6)
            ax.legend(loc='upper right')
            ax.grid(True)
            yscale = subplot.get('yscale')
            if yscale is not None:
                ax.set_yscale(yscale)
            lims = subplot.get('lims')
            if lims is not None:
                set_lims(ax, lims)
            fig.tight_layout()

            for ext in ['.pdf', '.png']:
                fig.savefig(os.path.join(output_path, tb_plot_id + '_' + subplot_id + ext))

    logger.info('Done')
