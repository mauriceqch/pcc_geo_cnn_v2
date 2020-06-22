import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

import os
import argparse
import shutil
from os import makedirs
from glob import glob
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ds_select_largest.py',
        description='Converts a folder containing meshes to point clouds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('source', help='Source directory')
    parser.add_argument('dest', help='Destination directory')
    parser.add_argument('n', help='Number of largest files to keep.', type=int)

    args = parser.parse_args()

    assert os.path.exists(args.source), f'{args.source} does not exist'
    assert args.n > 0

    paths = glob(os.path.join(args.source, '**', f'*'), recursive=True)
    paths = [x for x in paths if os.path.isfile(x)]
    files = [x[len(args.source) + 1:] for x in paths]
    files_len = len(files)
    assert files_len > 0
    logger.info(f'Found {files_len} models in {args.source}')

    sizes = [os.stat(x).st_size for x in paths]
    files_with_sizes = list(zip(files, paths, sizes))
    files_sorted_by_size = sorted(files_with_sizes, key=lambda x: -x[2])

    for file, path, _ in tqdm(files_sorted_by_size[:args.n]):
        target_path = os.path.join(args.dest, file)
        target_folder, _ = os.path.split(target_path)
        makedirs(target_folder, exist_ok=True)
        # shutil.copyfile(path, target_path)
        os.symlink(path, target_path)

    logger.info(f'{files_len} models to {args.dest}')
