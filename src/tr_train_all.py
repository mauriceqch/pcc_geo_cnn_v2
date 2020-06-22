import os
import subprocess
import argparse
import yaml
from utils.experiment import assert_exists, build_logger


def lmbda_to_str(lmbda):
    return f'{lmbda:.2e}'


def get_model_dir(exp_dir, model_id, lmbda_str):
    return os.path.join(exp_dir, 'models', model_id, lmbda_str)


def get_log_path(exp_dir, model_id, lmbda_str):
    return os.path.join(exp_dir, 'models', model_id, f'{lmbda_str}.log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='tr_train_all.py', description='Train all models for an experimental setup.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment_path', help='Experiments file path.')
    args = parser.parse_args()

    with open(args.experiment_path, 'r') as f:
        experiments = yaml.load(f.read(), Loader=yaml.FullLoader)
    keys = ['TRAIN_DATASET_PATH', 'EXPERIMENT_DIR', 'TRAIN_RESOLUTION', 'model_configs', 'alpha', 'gamma', 'batch_size', 'train_mode']
    TRAIN_DATASET_PATH, EXPERIMENT_DIR, TRAIN_RESOLUTION, model_configs, alpha, gamma, batch_size, train_mode = [experiments[x] for x in keys]
    assert train_mode in ['independent', 'warm_seq']

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    assert_exists(EXPERIMENT_DIR)

    logger = build_logger(__name__, os.path.join(EXPERIMENT_DIR, 'tr_train_all.log'))

    logger.info('Starting training')
    for model_config in model_configs:
        model_id = model_config['id']
        config = model_config['config']
        lambdas = model_config['lambdas']
        cur_alpha = model_config.get('alpha', alpha)
        cur_gamma = model_config.get('gamma', gamma)
        cur_batch_size = model_config.get('batch_size', batch_size)
        cur_train_mode = model_config.get('train_mode', train_mode)
        assert cur_train_mode in ['independent', 'warm_seq']
        for i, lmbda in enumerate(lambdas):
            lmbda_str = lmbda_to_str(lmbda)
            logger.info(f'Training model {model_id} {i}/{len(lambdas)-1} for lambda {lmbda_str} with train_mode {cur_train_mode}')
            checkpoint_id = model_config.get('checkpoint_id', model_id)
            model_dir = get_model_dir(EXPERIMENT_DIR, checkpoint_id, lmbda_str)
            log_path = get_log_path(EXPERIMENT_DIR, checkpoint_id, lmbda_str)
            done_path = os.path.join(model_dir, 'done')
            if not os.path.exists(done_path):
                os.makedirs(model_dir, exist_ok=True)
                additional_params = []
                if cur_train_mode == 'warm_seq' and i > 0:
                    warm_lmbda_str = lmbda_to_str(lambdas[i-1])
                    warm_model_dir = get_model_dir(EXPERIMENT_DIR, checkpoint_id, warm_lmbda_str)
                    additional_params += ['--warm_start', warm_model_dir]
                    logger.info(f'Warm start using {warm_lmbda_str}')
                with open(log_path, 'w') as f:
                    subprocess.run(['python', 'tr_train.py',
                                    TRAIN_DATASET_PATH,
                                    model_dir,
                                    '--resolution', str(TRAIN_RESOLUTION),
                                    '--lmbda', str(lmbda_str),
                                    '--alpha', str(cur_alpha),
                                    '--gamma', str(cur_gamma),
                                    '--batch_size', str(cur_batch_size),
                                    '--model_config', config] + additional_params,
                                   stdout=f, stderr=f, check=True)
            assert os.path.exists(done_path)
    logger.info('Done')
