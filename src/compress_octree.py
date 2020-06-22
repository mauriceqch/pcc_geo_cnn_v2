import json
import logging

from model_syntax import save_compressed_file

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
import gzip
from tqdm import trange
from model_configs import ModelConfigType
from utils import pc_io
from utils.octree_coding import partition_octree
from utils.pc_metric import avail_opt_metrics, validate_opt_metrics
from pyntcloud import PyntCloud


np.random.seed(42)
tf.set_random_seed(42)


def write_pcs(pcs, folder):
    os.makedirs(folder, exist_ok=True)
    for j, points in enumerate(pcs):
        pc_io.write_df(os.path.join(folder, f'{j}.ply'), pc_io.pa_to_df(points))


def compress():
    assert args.resolution > 0, 'resolution must be positive'
    assert args.data_format in ['channels_first', 'channels_last']
    with_normals = args.input_normals is not None
    validate_opt_metrics(args.opt_metrics, with_normals=with_normals)

    files_mult = 1
    if with_normals:
        files_mult *= 2
        assert files_mult * len(args.input_files) == len(args.output_files)
        assert files_mult * len(args.input_normals) == len(args.output_files)
    else:
        assert files_mult * len(args.input_files) == len(args.output_files)
    decode_files = args.dec_files is not None
    if decode_files:
        assert files_mult * len(args.input_files) == len(args.dec_files)

    assert args.model_config in ModelConfigType.keys()

    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(args.resolution, args.data_format)
    points = pc_io.load_points(args.input_files, batch_size=args.read_batch_size)
    if with_normals:
        normals = [PyntCloud.from_file(x).points[['nx', 'ny', 'nz']].values for x in args.input_normals]
        points = [np.hstack((p, n)) for p, n in zip(points, normals)]

    logger.info('Performing octree partitioning')
    # Hardcode bbox_min
    bbox_min = [0, 0, 0]
    if args.data_format == 'channels_first':
        bbox_max = dense_tensor_shape[1:].copy()
        dense_tensor_shape[1:] = dense_tensor_shape[1:] // (2 ** args.octree_level)
    else:
        bbox_max = dense_tensor_shape[:3].copy()
        dense_tensor_shape[:3] = dense_tensor_shape[:3] // (2 ** args.octree_level)
    blocks_list, binstr_list = zip(*[partition_octree(p, bbox_min, bbox_max, args.octree_level) for p in points])
    blocks_list_flat = [y for x in blocks_list for y in x]
    logger.info(f'Processing resolution {args.resolution} with octree level {args.octree_level} resulting in '
                + f'dense_tensor_shape {dense_tensor_shape} and {len(blocks_list_flat)} blocks')

    batch_size = 1
    x_shape = np.concatenate(((batch_size,), dense_tensor_shape))

    model = ModelConfigType[args.model_config].build()
    model.compress(x_shape)

    # Checkpoints
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    init = tf.global_variables_initializer()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        logger.info('Init session')
        sess.run(init)

        checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        assert checkpoint is not None, f'Checkpoint {args.checkpoint_dir} was not found'
        saver.restore(sess, checkpoint)

        for i in trange(len(args.input_files)):
            ori_file, cur_points, blocks, binstr = [x[i] for x in (args.input_files, points, blocks_list, binstr_list)]
            n_blocks = len(blocks)

            cur_output_files = [args.output_files[i*files_mult+j] for j in range(files_mult)]
            if decode_files:
                cur_dec_files = [args.dec_files[i*files_mult+j] for j in range(files_mult)]
            assert len(set(cur_output_files)) == len(cur_output_files), f'{cur_output_files} should have no duplicates'
            logger.info(f'Starting {ori_file} to {", ".join(cur_output_files)} with {n_blocks} blocks')
            data_list, data, debug_t_list = model.compress_blocks(sess, blocks, binstr, cur_points, args.resolution,
                                                                  args.octree_level, with_normals=with_normals,
                                                                  opt_metrics=args.opt_metrics, max_deltas=args.max_deltas,
                                                                  fixed_threshold=args.fixed_threshold, debug=args.debug)
            assert len(data_list) == files_mult

            for j in range(len(cur_output_files)):
                of, cur_data_list, cur_data = [x[j] for x in (cur_output_files, data_list, data)]
                os.makedirs(os.path.split(of)[0], exist_ok=True)
                with gzip.open(of, "wb") as f:
                    ret = save_compressed_file(binstr, cur_data_list, args.resolution, args.octree_level)
                    f.write(ret)
                if decode_files:
                    pc_io.write_df(cur_dec_files[j], pc_io.pa_to_df(cur_data['blocks_full']))
                with open(of + '.enc.metric.json', 'w') as f:
                    json.dump(cur_data['metrics'], f, sort_keys=True, indent=4)
                if args.debug:
                    pc_io.write_df(of + '.enc.ply', pc_io.pa_to_df(cur_data['blocks_full']))

                    write_pcs(blocks, of + '.ori.blocks')
                    write_pcs(cur_data['x_hat_list'], of + '.enc.blocks')
                    write_pcs(cur_data['blocks_depart'], of + '.enc.blocks.depart')
                    np.savez_compressed(of + '.enc.data.npz', data=cur_data_list, debug_t_list=debug_t_list)

            logger.info(f'Finished {ori_file} to {", ".join(cur_output_files)} with {n_blocks} blocks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='compress_octree.py',
        description='Compress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_files', nargs='+',
        help='Input files.', required=True)
    parser.add_argument(
        '--output_files', nargs='+',
        help='Output files. If input normals are provided, specify two output files per input file.', required=True)
    parser.add_argument(
        '--input_normals', nargs='+',
        help='Input normals. If provided, two output paths are needed for each input file for D1 and D2 optimization.')
    parser.add_argument(
        '--dec_files', nargs='*',
        help='Decoded files. Allows compression/decompression in a single execution. If input normals are provided, '
             + 'specify two decoded files per input file.')
    parser.add_argument(
        '--checkpoint_dir',
        help='Directory where to save/load model checkpoints.', required=True)
    parser.add_argument(
        '--model_config',
        help=f'Model used: {ModelConfigType.keys()}.', required=True)
    parser.add_argument(
        '--opt_metrics', nargs='+', default=['d1_psnr'],
        help=f'Optimization metrics used. Available: {avail_opt_metrics}')
    parser.add_argument(
        '--max_deltas', nargs='+', default=[np.inf], type=float,
        help=f'Max deltas tested during optimization.')
    parser.add_argument(
        '--fixed_threshold', default=False, action='store_true',
        help='Enable fixed thresholding.')
    parser.add_argument(
        '--read_batch_size', type=int, default=1,
        help='Batch size for parallel reading.')
    parser.add_argument(
        '--resolution',
        type=int, help='Dataset resolution.', default=64)
    parser.add_argument(
        '--octree_level',
        type=int, help='Octree level.', default=4)
    parser.add_argument(
        '--num_filters', type=int, default=32,
        help='Number of filters per layer.')
    parser.add_argument(
        '--data_format', default='channels_first',
        help='Data format used: channels_first or channels_last')
    parser.add_argument(
        '--debug', default=False, action='store_true',
        help='Output debug data for point cloud.')

    args = parser.parse_args()

    compress()
