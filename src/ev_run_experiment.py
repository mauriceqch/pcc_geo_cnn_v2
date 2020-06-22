import argparse
import logging
import os

import yaml

from utils.experiment import assert_exists
from utils.parallel_process import parallel_process, Popen
from utils.pc_metric import validate_opt_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def run_experiment(output_dir, model_dir, model_config, pc_name, pcerror_path, pcerror_cfg_path, input_pc, input_norm,
                   opt_metrics, max_deltas, fixed_threshold, no_stream_redirection=False):
    os.makedirs(output_dir, exist_ok=True)
    additional_params = []
    if fixed_threshold:
        additional_params += ['--fixed_threshold']
    if no_stream_redirection:
        f = None
        additional_params += ['--no_stream_redirection']
    else:
        f = open(os.path.join(output_dir, 'experiment.log'), 'w')
    return Popen(['python', 'ev_experiment.py',
                  '--output_dir', output_dir,
                  '--model_dir', model_dir,
                  '--model_config', model_config,
                  '--opt_metrics', *opt_metrics,
                  '--max_deltas', *map(str, max_deltas),
                  '--pc_name', pc_name,
                  '--pcerror_path', pcerror_path,
                  '--pcerror_cfg_path', pcerror_cfg_path,
                  '--input_pc', input_pc,
                  '--input_norm', input_norm] + additional_params, stdout=f, stderr=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ev_run_experiment.py', description='Run experiments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment_path', help='Experiments file path.')
    parser.add_argument('--num_parallel', help='Number of parallel jobs. Adjust according to GPU memory.', default=16, type=int)
    parser.add_argument('--no_stream_redirection', help='Disable stdout and stderr redirection.', default=False, action='store_true')
    args = parser.parse_args()
    with open(args.experiment_path, 'r') as f:
        experiments = yaml.load(f.read(), Loader=yaml.FullLoader)
    keys = ['MPEG_TMC13_DIR', 'PCERROR', 'MPEG_DATASET_DIR', 'EXPERIMENT_DIR', 'pcerror_mpeg_mode',
            'model_configs', 'opt_metrics', 'max_deltas', 'fixed_threshold']
    MPEG_TMC13_DIR, PCERROR, MPEG_DATASET_DIR, EXPERIMENT_DIR, pcerror_mpeg_mode, model_configs, opt_metrics,\
        max_deltas, fixed_threshold = [experiments[x] for x in keys]

    assert_exists(PCERROR)
    assert_exists(MPEG_DATASET_DIR)
    assert_exists(EXPERIMENT_DIR)
    validate_opt_metrics(opt_metrics, with_normals=True)

    logger.info('Starting our method\'s experiments')
    params = []
    for experiment in experiments['data']:
        pc_name, cfg_name, input_pc, input_norm = \
            [experiment[x] for x in ['pc_name', 'cfg_name', 'input_pc', 'input_norm']]
        opt_output_dir = os.path.join(EXPERIMENT_DIR, pc_name)
        for model_config in model_configs:
            model_id = model_config['id']
            config = model_config['config']
            lambdas = model_config['lambdas']
            cur_opt_metrics = model_config.get('opt_metrics', opt_metrics)
            cur_max_deltas = model_config.get('max_deltas', max_deltas)
            cur_fixed_threshold = model_config.get('fixed_threshold', fixed_threshold)
            for lmbda in lambdas:
                lmbda_str = f'{lmbda:.2e}'
                checkpoint_id = model_config.get('checkpoint_id', model_id)
                model_dir = os.path.join(EXPERIMENT_DIR, 'models', checkpoint_id, lmbda_str)
                current_output_dir = os.path.join(opt_output_dir, model_id, lmbda_str)

                pcerror_cfg_path = f'{MPEG_TMC13_DIR}/cfg/{pcerror_mpeg_mode}/{cfg_name}/r06/pcerror.cfg'
                input_pc_full = os.path.join(MPEG_DATASET_DIR, input_pc)
                input_norm_full = os.path.join(MPEG_DATASET_DIR, input_norm)
                if not os.path.exists(os.path.join(model_dir, 'done')):
                    logger.warning(f'Model training is not finished: skipping {model_dir} for {pc_name}')
                else:
                    opt_groups = ['d1', 'd2']
                    if not all(os.path.exists(os.path.join(current_output_dir, f'report_{g}.json')) for g in opt_groups):
                        params.append((current_output_dir, model_dir, config, pc_name, PCERROR, pcerror_cfg_path,
                                       input_pc_full, input_norm_full, cur_opt_metrics, cur_max_deltas,
                                       cur_fixed_threshold, args.no_stream_redirection))
    parallel_process(run_experiment, params, args.num_parallel)
    logger.info('Done')
