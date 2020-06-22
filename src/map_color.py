import argparse
import logging

from utils.parallel_process import Popen

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

import pandas as pd
from pyntcloud import PyntCloud

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='map_color.py',
                                     description='Map colors from one PC to another.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('ori_file', help='Original directory.')
    parser.add_argument('target_file', help='Mesh detection pattern.')
    parser.add_argument('output_file', help='Decompressed directory.')
    args = parser.parse_args()

    ori_pc = PyntCloud.from_file(args.ori_file)
    ori_points = ori_pc.points
    target_pc = PyntCloud.from_file(args.target_file)
    target_points = target_pc.points[['x', 'y', 'z']]

    kdid = ori_pc.add_structure('kdtree')
    kdtree = ori_pc.structures[kdid]

    mapped_indices = kdtree.query(target_pc.points[['x', 'y', 'z']].values, k=2, n_jobs=-1)[1][:, 1]
    mapped_colors = ori_points[['red', 'green', 'blue']].iloc[mapped_indices]
    mapped_colors.index = target_points.index

    output_points = pd.concat([target_points, mapped_colors], axis=1)
    output_pc = PyntCloud(output_points)
    output_pc.to_file(args.output_file)


def run_mapcolor(input_pc, target_file, output_file):
    return Popen(['python', 'map_color.py', input_pc, target_file, output_file])