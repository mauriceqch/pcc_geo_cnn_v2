import argparse
import itertools
import json
import logging
import os
import shutil
import yaml
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.spatial.ckdtree import cKDTree
from utils.colorbar import get_colorbar
from utils.experiment import index_by_id, assert_exists
from utils.matplotlib_utils import default_rc_params
from utils.o3d import pc_to_camera_params, pc_to_img, trim_img_bbox
from PIL import Image
from utils.pc_metric import compute_d1_res_ba

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

MPEG = 'mpeg'
MODEL = 'model'

rcParams = default_rc_params(rcParams)

def mpeg_path_color(exp_dir, model_id, pc_name, rate):
    return os.path.join(exp_dir, 'gpcc', model_id, pc_name, rate, f'{pc_name}.ply.bin.decoded.ply.color.ply')


def model_path_color(exp_dir, pc_name, model_id, lmbda, opt_group):
    lmbda_str = f'{lmbda:.2e}'
    return os.path.join(exp_dir, pc_name, model_id, lmbda_str, f'{pc_name}_{opt_group}.ply.bin.ply.color.ply')


def arr_to_pil(img_arr):
    return Image.fromarray(np.floor(img_arr * 255.0).astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ut_run_render.py', description='Run experiments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment_path', help='Experiments file path.')
    args = parser.parse_args()
    with open(args.experiment_path, 'r') as f:
        experiments = yaml.load(f.read(), Loader=yaml.FullLoader)
    keys = ['MPEG_DATASET_DIR', 'EXPERIMENT_DIR', 'model_configs', 'mpeg_modes', 'rates', 'vis_comps']
    MPEG_DATASET_DIR, EXPERIMENT_DIR, model_configs, mpeg_modes, rates, vis_comps = [experiments[x] for x in keys]

    logger.info('Gathering camera params and paths')
    opt_groups = ['d1', 'd2']
    params = []
    for experiment in experiments['data']:
        pc_name, cfg_name, input_pc, input_norm = \
            [experiment[x] for x in ['pc_name', 'cfg_name', 'input_pc', 'input_norm']]
        input_pc_full = os.path.join(MPEG_DATASET_DIR, input_pc)
        cur_output_dir = os.path.join(EXPERIMENT_DIR, pc_name)
        pc_data = []
        for model_config in model_configs:
            model_id = model_config['id']
            lambdas = model_config['lambdas']
            for lmbda in lambdas:
                lmbda_str = f'{lmbda:.2e}'
                for g in opt_groups:
                    pc_path = model_path_color(EXPERIMENT_DIR, pc_name, model_id, lmbda, g)
                    if not os.path.exists(pc_path):
                        logger.warning(f'Colored point clouds not found at {pc_path}')
                    else:
                        pc_data.append({'path': pc_path, 'opt_group': g, 'lambda': lmbda, 'model_id': model_id, 'type': MODEL})

        for model_config in mpeg_modes:
            mpeg_id = model_config['id']
            for rate in rates:
                pc_path = mpeg_path_color(EXPERIMENT_DIR, mpeg_id, pc_name, rate)
                if not os.path.exists(pc_path):
                    logger.warning(f'Colored point clouds not found at {pc_path}')
                else:
                    pc_data.append({'path': pc_path, 'rate': rate, 'type': MPEG, 'model_id': mpeg_id})

        camera_params_path = os.path.join(cur_output_dir, 'camera_params.json')
        full_pcd = o3d.io.read_point_cloud(input_pc_full)
        if os.path.exists(camera_params_path):
            logger.info(f'Loading camera parameters for {pc_name} and {len(pc_data)} dependent point clouds from {camera_params_path}')
            camera_params = o3d.io.read_pinhole_camera_parameters(camera_params_path)
        else:
            logger.info(f'Computing camera parameters for {pc_name} and {len(pc_data)} dependent point clouds')
            camera_params = pc_to_camera_params(full_pcd)
            o3d.io.write_pinhole_camera_parameters(camera_params_path, camera_params)

        save_path = os.path.join(cur_output_dir, os.path.split(input_pc_full)[1])
        params.append({'path': input_pc_full, 'pcd': full_pcd, 'camera_params': camera_params, 'save_path': save_path,
                       'pc_data': pc_data, 'pc_name': pc_name, 'id': pc_name})

    logger.info('Starting rendering')
    for p in params:
        logger.info(f'Rendering {p["pc_name"]}')
        bbox_path = p['save_path'] + '.bbox.json'
        img_path = p['save_path'] + '.png'
        if all(os.path.exists(x) for x in (bbox_path, img_path)):
            img = Image.open(img_path)
            img_arr = np.asarray(img)
            with open(bbox_path, 'r') as f:
                p['bbox'] = json.load(f)
            logger.info(f'Loaded img {p["pc_name"]} with {img_arr.shape} with trim bbox {p["bbox"]}')
        else:
            img_arr = pc_to_img(p['pcd'], p['camera_params'])
            img = arr_to_pil(img_arr)
            p['bbox'] = trim_img_bbox(img)
            img = img.crop(p['bbox'])
            if not os.path.exists(p['save_path']):
                shutil.copyfile(p['path'], p['save_path'])
            with open(bbox_path, 'w') as f:
                json.dump(p['bbox'], f)
            img.save(img_path)
            logger.info(f'Rendered {p["pc_name"]} with {img_arr.shape} with trim bbox {p["bbox"]}')

        for i, data in enumerate(p['pc_data']):
            path = data['path']
            save_path = path + '.png'
            if not os.path.exists(save_path):
                logger.info(f'{i + 1}/{len(p["pc_data"])} Rendering {path}')
                pcd = o3d.io.read_point_cloud(path)
                # Rendering can sometimes fail, retry if results are incorrect
                retries = 0
                while True:
                    try:
                        retries += 1
                        cur_img_arr = pc_to_img(pcd, p['camera_params'])
                        np.testing.assert_equal(cur_img_arr.shape, img_arr.shape)
                        mse = np.mean(np.square(cur_img_arr - img_arr))
                        assert mse < 0.3, f'mse={mse}'
                    except AssertionError as e:
                        if retries > 20:
                            raise e
                        logger.warning(f'Retrying {retries} {e}')
                        continue
                    break
                img = arr_to_pil(cur_img_arr)
                img = img.crop(p['bbox'])
                img.save(save_path)
                logger.info(f'{i + 1}/{len(p["pc_data"])} Rendered {path} with {cur_img_arr.shape}')
            # else:
            #     logger.info(f'{i + 1}/{len(p["pc_data"])} Exists {path}')

    logging.info('Rendering visual comparisons')
    indexed_params = index_by_id(params)
    for vc in vis_comps:
        pc_name = vc['pc_name']
        vc_id = vc['id']
        pc_param = indexed_params[pc_name]
        camera_params = pc_param['camera_params']

        logger.info(f'{vc_id} {pc_name}')
        vc_folder = os.path.join(EXPERIMENT_DIR, 'vis_comps', vc_id)
        os.makedirs(vc_folder, exist_ok=True)
        shutil.copyfile(pc_param['path'], os.path.join(vc_folder, pc_name + '.ply'))
        shutil.copyfile(pc_param['save_path'] + '.png', os.path.join(vc_folder, pc_name + '.ply.png'))

        pcd = pc_param['pcd']
        pcd_points = np.asarray(pcd.points)
        t1 = cKDTree(pcd_points, balanced_tree=False)

        # Reading point clouds and computing errors
        compared_pcds = []
        compared_err = []
        for comp in vc['compared']:
            # Compute path
            if comp['type'] == MPEG:
                path = mpeg_path_color(EXPERIMENT_DIR, comp['model_id'], pc_name, comp['rate'])
                suffix = ''
            else:
                path = model_path_color(EXPERIMENT_DIR, pc_name, comp['model_id'], comp['lambda'], comp['opt_group'])
                suffix = f'_{comp["opt_group"]}'

            # Gather report and image
            report_path = os.path.join(os.path.split(path)[0], f'report{suffix}.json')
            with open(report_path, 'r') as f:
                report = f.read()
            shutil.copyfile(report_path, os.path.join(vc_folder, comp['id'] + '.report.json'))
            shutil.copyfile(path + '.png', os.path.join(vc_folder, comp['id'] + '.ply.png'))

            # Read point cloud
            logger.info(f'Loading {comp["id"]}\n{report}')
            assert_exists(path)
            cur_pcd = o3d.io.read_point_cloud(path)
            cur_pcd_points = np.asarray(cur_pcd.points)

            compared_pcds.append(cur_pcd)
            compared_err.append(compute_d1_res_ba(pcd_points, cur_pcd_points, t1=t1))

        # Computing min, max, target percentile
        all_res = np.concatenate(compared_err)
        percentile = 99
        global_max = np.max(all_res)
        global_pmax = np.percentile(all_res, percentile)
        global_min = 0.0

        # Build colorbar
        fig, cmap = get_colorbar(global_min, global_pmax, orientation='vertical')
        # Change last label
        ax = fig.gca()
        fig.canvas.draw()
        yticklabels = ax.get_yticklabels()
        yticklabels[-1].set_text(yticklabels[-1].get_text() + r'\small{+}')
        ax.set_yticklabels(yticklabels)
        fig.savefig(os.path.join(vc_folder, 'colorbar.pdf'))

        # Remove alpha channel
        compared_cols = [cmap(x)[:, :3] for x in compared_err]
        # Save errors in ply and png
        for comp, ce, cc, cp in zip(vc['compared'], compared_err, compared_cols, compared_pcds):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cp.points)
            pcd.colors = o3d.utility.Vector3dVector(cc)
            save_path = os.path.join(vc_folder, comp['id'] + '.ply')
            o3d.io.write_point_cloud(save_path, pcd)

            img = pc_to_img(pcd, camera_params)
            img = arr_to_pil(img)
            img = img.crop(pc_param['bbox'])
            img.save(save_path + '.res.png')

        # Gather labels
        labels = []
        for comp in vc['compared']:
            model_id = comp['model_id']
            if comp['type'] == MPEG:
                mode = index_by_id(mpeg_modes)[model_id]
            else:
                mode = index_by_id(model_configs)[model_id]
            label = mode.get('label', mode['id'])
            labels.append(label)

        # Computing bins
        logger.info(f'Min {global_min} {percentile} {global_pmax} Max {global_max}')
        _, bins = np.histogram([global_min, global_max], bins=32)

        # Plotting errors histogram
        fig, ax = plt.subplots()
        ax.hist(compared_err, bins=bins, label=labels, edgecolor='black')
        ax.set(xlabel='Squared error', ylabel='Occurences')
        ax.legend(loc='upper right')
        ax.set_yscale('log')
        ax.grid(True)
        fig.tight_layout()
        for ext in ['.pdf', '.png']:
            fig.savefig(os.path.join(vc_folder, 'errors.hist' + ext))

    logger.info('Done')
