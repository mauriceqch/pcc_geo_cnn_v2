import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from pyntcloud import PyntCloud
from glob import glob


def df_to_pc(df):
    points = df[['x','y','z']].values
    return points


def pa_to_df(points):
    cols = ['x', 'y', 'z', 'red', 'green', 'blue']
    types = (['float32'] * 3) + (['uint8'] * 3)
    d = {}
    assert 3 <= points.shape[1] <= 6
    for i in range(points.shape[1]):
        col = cols[i]
        dtype = types[i]
        d[col] = points[:, i].astype(dtype)
    df = pd.DataFrame(data=d)
    return df


def pc_to_df(pc):
    points = pc.points
    return pa_to_df(points)


def load_pc(path):
    logger.debug(f"Loading PC {path}")
    pc = PyntCloud.from_file(path)
    ret = df_to_pc(pc.points)
    logger.debug(f"Loaded PC {path}")

    return ret


def write_pc(path, pc):
    df = pc_to_df(pc)
    write_df(path, df)


def write_df(path, df):
    pc = PyntCloud(df)
    pc.to_file(path)


def get_shape_data(resolution, data_format):
    assert data_format in ['channels_last', 'channels_first']
    bbox_min = 0
    bbox_max = resolution
    p_max = np.array([bbox_max, bbox_max, bbox_max])
    p_min = np.array([bbox_min, bbox_min, bbox_min])
    if data_format == 'channels_last':
        dense_tensor_shape = np.concatenate([p_max, [1]]).astype('int64')
    else:
        dense_tensor_shape = np.concatenate([[1], p_max]).astype('int64')

    return p_min, p_max, dense_tensor_shape


def get_files(input_glob):
    return np.array(glob(input_glob, recursive=True))


def load_points(files, batch_size=32):
    files_len = len(files)

    with multiprocessing.Pool() as p:
        logger.info('Loading PCs into memory (parallel reading)')
        points = np.array(list(tqdm(p.imap(load_pc, files, batch_size), total=files_len)))

    return points
