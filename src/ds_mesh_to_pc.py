#!/usr/bin/env python

################################################################################
### Init
################################################################################
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
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import functools

################################################################################
### Definitions
################################################################################
def process(path, args):
    ori_path = join(args.source, path)
    target_path, _ = splitext(join(args.dest, path))
    target_path += args.target_extension
    target_folder, _ = split(target_path)
    makedirs(target_folder, exist_ok=True)

    logger.debug(f"Writing PC {ori_path} to {target_path}")
    pc_mesh = PyntCloud.from_file(ori_path)
    mesh = pc_mesh.mesh
    pc_mesh.points = pc_mesh.points.astype('float64', copy=False)
    pc_mesh.mesh = mesh

    pc = pc_mesh.get_sample("mesh_random", n=args.n_samples, as_PyntCloud=True)
    coords = ['x', 'y', 'z']
    points = pc.points.values
    points = points - np.min(points)
    points = points / np.max(points)
    points = points * (args.vg_size - 1)
    points = np.round(points)
    pc.points[coords] = points
    if len(set(pc.points.columns) - set(coords)) > 0:
        pc.points = pc.points.groupby(by=coords, sort=False).mean()
    else:
        pc.points = pc.points.drop_duplicates()

    pc.to_file(target_path)
    
################################################################################
### Script
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ds_mesh_to_pc.py',
        description='Converts a folder containing meshes to point clouds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('source', help='Source directory')
    parser.add_argument('dest', help='Destination directory')
    parser.add_argument('--vg_size', type=int, help='Voxel Grid resolution for x, y, z dimensions', default=64)
    parser.add_argument('--n_samples', type=int, help='Number of samples', default=500000)
    parser.add_argument('--source_extension', help='Mesh files extension', default='.off')
    parser.add_argument('--target_extension', help='Point cloud extension', default='.ply')

    args = parser.parse_args()

    assert os.path.exists(args.source), f'{args.source} does not exist'
    assert not os.path.exists(args.dest), f'{args.dest} already exists'
    assert args.vg_size > 0, f'vg_size must be positive'
    assert args.n_samples > 0, f'n_samples must be positive'

    paths = glob(join(args.source, '**', f'*{args.source_extension}'), recursive=True)
    files = [x[len(args.source) + 1:] for x in paths]
    files_len = len(files)
    assert files_len > 0
    logger.info(f'Found {files_len} models in {args.source}')

    with Pool() as p:
        process_f = functools.partial(process, args=args)
        list(tqdm(p.imap(process_f, files), total=files_len)) 
        # Without parallelism
        #list(tqdm((process_f(f) for f in files), total=files_len))
 
    logger.info(f'{files_len} models written to {args.dest}')
