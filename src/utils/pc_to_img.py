import logging
import numpy as np
import argparse
import open3d as o3d
from utils.o3d import pc_to_img, trim_img_bbox
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pc_to_img.py',
                                     description='Converts a point cloud to an image.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_path', help='Input point cloud path (ply).')
    parser.add_argument('output_path', help='Output image path.')
    parser.add_argument('camera_params_path', help='Camera params path.')
    parser.add_argument('--point_size', help='Point size.', default=1.0, type=float)
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.input_path)
    camera_params = o3d.io.read_pinhole_camera_parameters(args.camera_params_path)
    img = pc_to_img(pcd, camera_params, point_size=args.point_size)
    img = Image.fromarray(np.floor(img * 255.0).astype(np.uint8))
    img = img.crop(trim_img_bbox(img))
    img.save(args.output_path)
