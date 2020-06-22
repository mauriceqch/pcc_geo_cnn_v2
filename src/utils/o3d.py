import io
import logging
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import open3d as o3d
from PIL import Image, ImageChops

# Value defined in Open3D
# https://github.com/intel-isl/Open3D/blob/master/src/Open3D/Visualization/Visualizer/ViewControl.cpp
ROTATION_RADIAN_PER_PIXEL = 0.003

logger = logging.getLogger(__name__)


def np_to_o3d(arr):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(arr[:, 3:6])
    return pcd


def pc_to_camera_params(pcd, width=1024, height=1024, interactive=True):
    """ Take an Open3D point cloud as input and returns pinhole camera parameters """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    if interactive:
        vis.run()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    vis.destroy_window()
    return camera_params


def pc_to_img(pcd, camera_params, width=1024, height=1024, point_size=1.0):
    f = io.StringIO()
    with redirect_stdout(f):
        with redirect_stderr(f):
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=width, height=height)
            vis.add_geometry(pcd)
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(camera_params)
            # rot_x = -(3.14 / 180) / ROTATION_RADIAN_PER_PIXEL
            # rot_y = -(3.14 / 3) / ROTATION_RADIAN_PER_PIXEL
            # ctr.rotate(rot_x, rot_y)
            rdr_opt = vis.get_render_option()
            rdr_opt.point_size = point_size
            img = vis.capture_screen_float_buffer(True)
            img = np.asarray(img)
            vis.destroy_window()
    out = f.getvalue()
    if len(out) > 0:
        raise RuntimeError(out)
        # logger.warning(f'Image render failed: retrying')
        # return pc_to_img(img)
    return img


def trim_img_bbox(im, border=(255, 255, 255)):
    bg = Image.new(im.mode, im.size, border)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if not bbox:
        raise RuntimeError('Empty image')
    return bbox
