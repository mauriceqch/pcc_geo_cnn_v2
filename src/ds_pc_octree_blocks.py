import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

import os
from os.path import join, basename, split, splitext
from os import makedirs
from glob import glob
from pyntcloud import PyntCloud
import pandas as pd
import argparse
import functools
from tqdm import tqdm
from multiprocessing import Pool
from utils.octree_coding import partition_octree


def arr_to_pc(arr, cols, types):
    d = {}
    for i in range(arr.shape[1]):
        col = cols[i]
        dtype = types[i]
        d[col] = arr[:, i].astype(dtype)
    df = pd.DataFrame(data=d)
    pc = PyntCloud(df)
    return pc


def process(path, args):
    ori_path = join(args.source, path)
    target_path, _ = splitext(join(args.dest, path))
    target_folder, _ = split(target_path)
    makedirs(target_folder, exist_ok=True)

    pc = PyntCloud.from_file(ori_path)
    points = pc.points.values
    bbox_min = [0, 0, 0]
    bbox_max = [args.vg_size, args.vg_size, args.vg_size]
    blocks, _ = partition_octree(points, bbox_min, bbox_max, args.level)

    for i, block in enumerate(blocks):
        final_target_path = target_path + f'_{i:03d}{args.target_extension}'
        logger.debug(f"Writing PC {ori_path} to {final_target_path}")
        cur_pc = arr_to_pc(block, pc.points.columns, pc.points.dtypes)
        cur_pc.to_file(final_target_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ds_pc_octree_blocks.py',
        description='Converts a folder containing meshes to point clouds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('source', help='Source directory')
    parser.add_argument('dest', help='Destination directory')
    parser.add_argument('--vg_size', type=int, help='Voxel Grid resolution for x, y, z dimensions', default=64)
    parser.add_argument('--level', type=int, help='Octree decomposition level.', default=3)
    parser.add_argument('--source_extension', help='Mesh files extension', default='.ply')
    parser.add_argument('--target_extension', help='Point cloud extension', default='.ply')

    args = parser.parse_args()

    assert os.path.exists(args.source), f'{args.source} does not exist'
    assert args.vg_size > 0, f'vg_size must be positive'

    paths = glob(join(args.source, '**', f'*{args.source_extension}'), recursive=True)
    files = [x[len(args.source) + 1:] for x in paths]
    files_len = len(files)
    assert files_len > 0
    logger.info(f'Found {files_len} models in {args.source}')

    with Pool() as p:
        process_f = functools.partial(process, args=args)
        list(tqdm(p.imap(process_f, files), total=files_len))
        # Without parallelism
        # list(tqdm((process_f(f) for f in files), total=files_len))

    logger.info(f'{files_len} models written to {args.dest}')
