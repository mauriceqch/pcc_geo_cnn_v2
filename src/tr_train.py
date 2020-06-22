import logging
from contextlib import ExitStack
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from utils import pc_io
from tqdm import tqdm, trange
from model_types import input_fn
from model_configs import ModelConfigType

np.random.seed(42)
tf.set_random_seed(42)


def train():
    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(args.resolution, args.data_format)
    files = pc_io.get_files(args.train_glob)
    assert len(files) > 0
    points = pc_io.load_points(files)

    files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in files])
    points_train = points[files_cat == 'train']
    points_val = points[files_cat == 'test']

    train_ds = input_fn(points_train, args.batch_size, dense_tensor_shape, args.data_format, repeat=True, shuffle=True)
    val_ds = input_fn(points_val, args.batch_size, dense_tensor_shape, args.data_format, repeat=True, shuffle=True)
    train_iterator = tf.data.make_one_shot_iterator(train_ds)
    val_iterator = tf.data.make_one_shot_iterator(val_ds)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, tf.data.get_output_types(train_ds), tf.data.get_output_shapes(train_ds))
    x = iterator.get_next()

    model = ModelConfigType[args.model_config].build()
    model.train(x, args.gamma, args.alpha, args.lmbda)

    # Summary writers
    train_writer = tf.summary.FileWriter(os.path.join(args.checkpoint_dir, 'train'))
    val_writer = tf.summary.FileWriter(os.path.join(args.checkpoint_dir, 'val'))
    # Checkpoints
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, save_relative_paths=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model.ckpt')
    # Init
    init = tf.global_variables_initializer()

    validation_interval = 500
    early_stop_patience = validation_interval * 4
    validation_steps = 10
    summary_interval = 100

    logger.info('Starting session')
    with ExitStack() as stack:
        if args.profiling:
            builder = tf.profiler.ProfileOptionBuilder
            opts = builder(builder.time_and_memory()).order_by('micros').build()
            pctx = tf.contrib.tfprof.ProfileContext('./profiler', trace_steps=[], dump_steps=[])
            stack.enter_context(pctx)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        stack.enter_context(sess)

        logger.info('Init session')
        sess.run(init)

        train_handle, test_handle = sess.run([train_iterator.string_handle(), val_iterator.string_handle()])

        checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        if checkpoint is not None:
            saver.restore(sess, checkpoint)
        elif args.warm_start is not None:
            warm_checkpoint = tf.train.latest_checkpoint(args.warm_start)
            saver.restore(sess, warm_checkpoint)
        train_writer.add_graph(sess.graph)

        step_val = sess.run(model.step)
        first_step_val = step_val
        pbar = tqdm(total=args.max_steps)
        logger.info(f'Starting training')
        best_loss = float('inf')
        best_loss_step = step_val
        while step_val <= args.max_steps:
            pbar.update(step_val - pbar.n)

            # Validation
            if step_val != first_step_val and step_val % validation_interval == 0:
                logger.info('Executing validation')
                losses = []
                for i in trange(validation_steps):
                    summary, vloss = sess.run(
                        [model.merged_summary, model.train_loss],
                        feed_dict={handle: test_handle}
                    )
                    losses.append(vloss)
                    val_writer.add_summary(summary, step_val + i)
                loss = np.mean(losses)
                print('')

                # Early stopping
                if loss < best_loss:
                    logger.info(f'Val loss {loss:.3E}@{step_val} lower than previous best {best_loss:.3E}@{best_loss_step}')
                    best_loss_step = step_val
                    best_loss = loss
                    save_path = saver.save(sess, checkpoint_path, global_step=step_val)
                    logger.info(f'Model saved to {save_path}')
                elif step_val - best_loss_step >= early_stop_patience:
                    save_path = saver.save(sess, checkpoint_path, global_step=step_val)
                    logger.info(f'Val loss {loss:.3E}@{step_val} higher than previous best {best_loss:.3E}@{best_loss_step}')
                    logger.info(f'Early stopping: model saved to {save_path}')
                    break
                else:
                    logger.info(f'Val loss {loss:.3E}@{step_val} higher than previous best {best_loss:.3E}@{best_loss_step}')

            if args.profiling:
                pctx.trace_next_step()
                pctx.dump_next_step()

            # Training
            get_summary = step_val % summary_interval == 0
            if get_summary:
                sess_args = [model.merged_summary, model.train_op, model.train_fl, model.train_mbpov, model.train_loss]
            else:
                sess_args = model.train_op

            sess_output = sess.run(sess_args, feed_dict={handle: train_handle})

            if args.profiling:
                pctx.profiler.profile_operations(options=opts)

            step_val += 1
            if get_summary:
                train_writer.add_summary(sess_output[0], step_val)
                pbar.set_description(f"fl: {sess_output[2]:.3E}, mbpov: {sess_output[3]:.3E},"
                                     + f" loss: {sess_output[4]:.3E}")

    Path(os.path.join(args.checkpoint_dir, 'done')).touch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='tr_train.py',
        description='Train network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'train_glob',
        help='Glob pattern for identifying training data.')
    parser.add_argument(
        'checkpoint_dir',
        help='Directory where to save/load model checkpoints.')
    parser.add_argument(
        '--model_config',
        help=f'Model used: {ModelConfigType.keys()}.', required=True)
    parser.add_argument(
        '--warm_start',
        help='Checkpoint path for warm start')
    parser.add_argument(
        '--resolution',
        type=int, help='Dataset resolution.', default=64)
    parser.add_argument(
        '--num_filters', type=int, default=32,
        help='Number of filters per layer.')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for training.')
    parser.add_argument(
        '--lmbda', type=float, default=0.0001,
        help='Lambda for rate-distortion tradeoff.')
    parser.add_argument(
        '--alpha', type=float, default=0.9,
        help='Focal loss alpha.')
    parser.add_argument(
        '--gamma', type=float, default=2.0,
        help='Focal loss gamma.')
    parser.add_argument(
        '--max_steps', type=int, default=100000,
        help='Train up to this number of steps.')
    parser.add_argument(
        '--profiling', default=False, action='store_true',
        help='Enable profiling')
    parser.add_argument(
        '--data_format', default='channels_first',
        help='Data format used: channels_first or channels_last')

    args = parser.parse_args()

    os.makedirs(os.path.split(args.checkpoint_dir)[0], exist_ok=True)
    assert args.resolution > 0, 'resolution must be positive'
    assert args.batch_size > 0, 'batch_size must be positive'
    assert args.data_format in ['channels_first', 'channels_last']
    assert args.model_config in ModelConfigType.keys()

    train()
