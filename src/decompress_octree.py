import gzip
import logging
import time

from model_syntax import load_compressed_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow.compat.v1 as tf
import argparse
from model_configs import ModelConfigType
from utils import pc_io
from utils.octree_coding import departition_octree

np.random.seed(42)
tf.set_random_seed(42)


def read_pcs(length, folder):
    return [pc_io.load_pc(os.path.join(folder, f'{j}.ply')) for j in range(length)]


def decompress():
    model = ModelConfigType[args.model_config].build()
    logger.info('Files loading')
    compressed_data = []
    for file in args.input_files:
        with gzip.open(file, "rb") as f:
            compressed_data.append(load_compressed_file(f))
    logger.info('Files loaded')
    model.decompress()

    # Checkpoints
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    # Init
    init = tf.global_variables_initializer()

    len_files = len(args.input_files)
    i = 0
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        logger.info('Init session')
        sess.run(init)

        checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        assert checkpoint is not None, f'Checkpoint {args.checkpoint_dir} was not found'
        saver.restore(sess, checkpoint)

        for (resolution, level, binstr, blocks), ori_file, output_file in zip(compressed_data, args.input_files, args.output_files):
            logger.info(f'{i}/{len_files} - Writing {ori_file} to {output_file} with {len(blocks)} blocks')
            output_dir, _ = os.path.split(output_file)
            os.makedirs(output_dir, exist_ok=True)

            x_shape = np.array([resolution, resolution, resolution], dtype=np.uint32) // (2 ** level)

            if args.debug:
                dec_blocks_enc = read_pcs(len(blocks), ori_file + '.enc.blocks')
                debug_data = np.load(ori_file + '.enc.data.npz', allow_pickle=True)
                data_list_enc, debug_t_list_enc = debug_data['data'], debug_data['debug_t_list']
                dec_blocks, debug_t_list = model.decompress_blocks(sess, blocks, x_shape, debug=args.debug)
                max_retries = 100
                # Sometimes, GPU computations are wrong... debug mode detects computation errors and resolves them
                for j, (db, dbe) in enumerate(zip(dec_blocks, dec_blocks_enc)):
                    logger.info(f'{i}/{len_files} - Verifying block {j}/{len(blocks)} {db.shape} {dbe.shape}')
                    error = True
                    retries = 0
                    new_block = db
                    new_debug = debug_t_list[j]
                    # Checking inputs consistency
                    np.testing.assert_equal(blocks[j], data_list_enc[j])
                    while error and retries < max_retries:
                        try:
                            intermediate_error = False
                            # Checking intermediate results consistency
                            for key in new_debug:
                                try:
                                    v1 = new_debug[key]
                                    v_len = np.prod(v1.shape)
                                    v2 = debug_t_list_enc[j][key]
                                    if v1.dtype == object:
                                        np.testing.assert_equal(v1, v2, err_msg=f'Values did not match for key {key}')
                                        err_n = np.sum(v1 != v2)
                                        err_percent = 100 * (err_n / v_len)
                                        logger.info(f'{key} {err_n}/{v_len} ({err_percent:.2f}%)')
                                    elif np.issubdtype(v1.dtype, np.number):
                                        atol = 0.001
                                        rtol = 1e-7
                                        aerr = abs(v1 - v2)
                                        aerr_n = np.sum(aerr > rtol * abs(v2))
                                        err_percent = 100 * (aerr_n / v_len)
                                        logger.info(f'{key} {aerr_n}/{v_len} ({err_percent:.2f}%) min {np.min(aerr)} mean {np.mean(aerr)} max {np.max(aerr)}')
                                        np.testing.assert_allclose(v1, v2, rtol=rtol, atol=atol,
                                                                   err_msg=f'Values did not match for key {key}')
                                    else:
                                        raise RuntimeError(f'Unsupported type {v1.dtype} for key {key} and value {v1}')
                                except AssertionError as e:
                                    if np.issubdtype(v1.dtype, np.number):
                                        idxs = np.argwhere(abs(v1 - v2) > atol + rtol * abs(v1))
                                        for idx in idxs[:500]:
                                            idx = tuple(idx)
                                            logger.error(f'{key} mismatch {idx} expected {v2[idx]} but got {v1[idx]}')
                                        if len(idxs) > 500:
                                            logger.error(f'Remaining entries truncated...')
                                    intermediate_error = True
                                    print(e)
                            if intermediate_error:
                                np.savez_compressed(output_file + '.dec.dump.npz', dec=new_debug, enc=debug_t_list_enc[j])
                                raise AssertionError('Intermediate results are wrong')

                            # Checking final result consistency
                            np.testing.assert_equal(new_block, dbe)
                            dec_blocks[j] = new_block
                            error = False
                            if retries > 0:
                                logger.info(f'{i}/{len_files} - Verifying block {j}/{len(blocks)} error resolved {retries}/{max_retries} retries {new_block.shape}')
                        except AssertionError as e:
                            logger.warning(f'{i}/{len_files} - Verifying block {j}/{len(blocks)} {retries}/{max_retries} retries\n{str(e)}')
                            error = True
                            new_block, new_debug = model.decompress_blocks(sess, [blocks[j]], x_shape, debug=args.debug)
                            new_block, new_debug = (new_block[0], new_debug[0])
                        retries += 1
                    if retries == max_retries:
                        raise RuntimeError(f'{i}/{len_files} - Verifying {ori_file} failed after {retries} retries')

            else:
                dec_blocks, _ = model.decompress_blocks(sess, blocks, x_shape, debug=args.debug)

            # Hardcode bbox_min
            bbox_min = [0, 0, 0]
            # Assume blocks of equal size
            bbox_max = x_shape * (2 ** level)
            dec_blocks = departition_octree(dec_blocks, binstr, bbox_min, bbox_max, level)
            pa = np.vstack(dec_blocks)

            pc_io.write_df(output_file, pc_io.pa_to_df(pa))
            i += 1
    logger.info('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='decompress_octree.py',
        description='Decompress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_files', nargs='+',
        help='Input files.', required=True)
    parser.add_argument(
        '--output_files', nargs='+',
        help='Input files.', required=True)
    parser.add_argument(
        '--checkpoint_dir',
        help='Directory where to save/load model checkpoints.', required=True)
    parser.add_argument(
        '--model_config',
        help=f'Model used: {ModelConfigType.keys()}.', required=True)
    parser.add_argument(
        '--num_filters', type=int, default=32,
        help='Number of filters per layer.')
    parser.add_argument(
        '--data_format', default='channels_first',
        help='Data format used: channels_first or channels_last')
    parser.add_argument(
        '--debug', default=False, action='store_true',
        help='Use debug data to check results.')

    args = parser.parse_args()

    assert args.data_format in ['channels_first', 'channels_last']
    assert len(args.input_files) == len(args.output_files)
    assert args.model_config in ModelConfigType.keys()

    decompress()
