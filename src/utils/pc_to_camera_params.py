import argparse
import open3d as o3d
from utils.o3d import pc_to_camera_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='pc_to_camera_params.py',
        description='Generates camera parameters for a point cloud.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_path',
        help='Input point cloud path (ply).')
    parser.add_argument(
        'output_path',
        help='Output camera params path.')
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.input_path)
    camera_params = pc_to_camera_params(pcd)
    o3d.io.write_pinhole_camera_parameters(args.output_path, camera_params)
