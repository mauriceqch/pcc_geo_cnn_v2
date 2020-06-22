import numpy as np
from pyntcloud import PyntCloud
from scipy.spatial import cKDTree
from numba import njit
from utils.experiment import timing


@njit
def assign_attr(attr1, idx1, idx2):
    """Given point sets x1 and x2, transfers attributes attr1 from x1 to x2.
    idx1: N2 array containing the nearest neighbors indices of x2 in x1
    idx2: N1 array containing the nearest neighbors indices of x1 in x2
    """
    counts = np.zeros(idx1.shape[0])
    attr_sums = np.zeros((idx1.shape[0], attr1.shape[1]))
    for i, idx in enumerate(idx2):
        counts[idx] += 1
        attr_sums[idx] += attr1[i]
    for i, idx in enumerate(idx1):
        if counts[i] == 0:
            counts[i] += 1
            attr_sums[i] += attr1[idx]
    counts = np.expand_dims(counts, -1)
    attr2 = attr_sums / counts
    return attr2


def d1_res(x, y):
    return np.sum((x - y) ** 2, axis=1)


def d1(x, y):
    return np.mean(d1_res(x, y))


def sum_d1(x, y):
    return np.sum(d1_res(x, y))


def d2_res(x, y, n):
    return np.sum((x - y) * n, axis=1) ** 2


def d2(x, y, n):
    return np.mean(d2_res(x, y, n))


def sum_d2(x, y, n):
    return np.sum(np.sum((x - y) * n, axis=1) ** 2)


def psnr(x, max_energy):
    return 10 * np.log10(max_energy / x)


# No PSNR as minimizing MSE is equivalent
avail_opt_metrics = [y for x in zip(*[(f'd1_{x}', f'd2_{x}') for x in ['sum_AB', 'sum_BA', 'sum_max', 'sum_mean',
                                                                       'mse_AB', 'mse_BA', 'mse']]) for y in x]


def validate_opt_metrics(opt_metrics, with_normals=False):
    for opt_metric in opt_metrics:
        assert opt_metric in avail_opt_metrics, f'{opt_metric} not found in {avail_opt_metrics}'
        if not with_normals:
            assert not opt_metric.startswith('d2'), f'{opt_metric} not available without normals'


def compute_d1_res_ba(p1, p2, t1=None):
    if t1 is None:
        t1 = cKDTree(p1, balanced_tree=False)
    _, idx1 = t1.query(p2, n_jobs=-1)
    p2_ngb = p1[idx1]
    return d1_res(p2, p2_ngb)


def compute_metrics(p1, p2, r, p1_n=None, t1=None):
    if t1 is None:
        t1 = cKDTree(p1, balanced_tree=False)
    t2 = cKDTree(p2, balanced_tree=False)
    _, idx2 = t2.query(p1, n_jobs=-1)
    _, idx1 = t1.query(p2, n_jobs=-1)

    max_energy = 3 * r * r
    p1_ngb = p2[idx2]
    p2_ngb = p1[idx1]
    d1_sum_AB = sum_d1(p1, p1_ngb)
    d1_sum_BA = sum_d1(p2, p2_ngb)
    d1_sum_max = max(d1_sum_AB, d1_sum_BA)
    d1_sum_mean = (d1_sum_AB + d1_sum_BA) / 2
    d1_mse_AB = d1_sum_AB / p1.shape[0]
    d1_mse_BA = d1_sum_BA / p2.shape[0]
    d1_mse = max(d1_mse_AB, d1_mse_BA)
    d1_psnr_AB = psnr(d1_mse_AB, max_energy)
    d1_psnr_BA = psnr(d1_mse_BA, max_energy)
    d1_psnr = min(d1_psnr_AB, d1_psnr_BA)
    metrics = {
        'd1_sum_AB': d1_sum_AB,
        'd1_sum_BA': d1_sum_BA,
        'd1_sum_max': d1_sum_max,
        'd1_sum_mean': d1_sum_mean,
        'd1_mse_AB': d1_mse_AB,
        'd1_mse_BA': d1_mse_BA,
        'd1_mse': d1_mse,
        'd1_psnr_AB': d1_psnr_AB,
        'd1_psnr_BA': d1_psnr_BA,
        'd1_psnr': d1_psnr
    }

    if p1_n is not None:
        # Compute normals in p2 from normals in p1
        p2_n = assign_attr(p1_n, idx1, idx2)
        p1_ngb_n = p2_n[idx2]
        p2_ngb_n = p1_n[idx1]
        # D2 may not exactly match mpeg-pcc-dmetric because of variations in nearest neighbors chosen when at equal distances
        d2_sum_AB = sum_d2(p1, p1_ngb, p1_ngb_n)
        d2_sum_BA = sum_d2(p2, p2_ngb, p2_ngb_n)
        d2_sum_max = max(d2_sum_AB, d2_sum_BA)
        d2_sum_mean = (d2_sum_AB + d2_sum_BA) / 2
        d2_mse_AB = d2_sum_AB / p1.shape[0]
        d2_mse_BA = d2_sum_BA / p2.shape[0]
        d2_mse = max(d2_mse_AB, d2_mse_BA)
        d2_psnr_AB = psnr(d2_mse_AB, max_energy)
        d2_psnr_BA = psnr(d2_mse_BA, max_energy)
        d2_psnr = min(d2_psnr_AB, d2_psnr_BA)
        d2_metrics = {
            'd2_sum_AB': d2_sum_AB,
            'd2_sum_BA': d2_sum_BA,
            'd2_sum_max': d2_sum_max,
            'd2_sum_mean': d2_sum_mean,
            'd2_mse_AB': d2_mse_AB,
            'd2_mse_BA': d2_mse_BA,
            'd2_mse': d2_mse,
            'd2_psnr_AB': d2_psnr_AB,
            'd2_psnr_BA': d2_psnr_BA,
            'd2_psnr': d2_psnr
        }
        metrics = {**metrics, **d2_metrics}
    return metrics


if __name__ == '__main__':
    pc1 = PyntCloud.from_file('C:/Users/User/Downloads/longdress_vox10_1300.ply')
    pc1_n = PyntCloud.from_file('C:/Users/User/Downloads/longdress_vox10_1300_n.ply')
    # r04 trisoup-predlift v9.1
    # PCC quality measurement software, version 0.12.3a
    #
    # infile1: /home/quachmau/data/datasets/mpeg_pcc/Static_Objects_and_Scenes/People/longdress_vox10_1300/longdress_vox10_1300.ply
    # infile2: longdress_vox10_1300.ply.bin.decoded.ply
    # normal1: /home/quachmau/data/datasets/mpeg_pcc/Static_Objects_and_Scenes/People/longdress_vox10_1300_n/longdress_vox10_1300_n.ply
    #
    # Verifying if the data is loaded correctly.. The last point is: 256 902 320
    # Reading file 1 done.
    # Verifying if the data is loaded correctly.. The last point is: 256 902 320
    # Reading normal 1 done.
    # Verifying if the data is loaded correctly.. The last point is: 416 665 303
    # Reading file 2 done.
    # Imported intrinsic resoluiton: 1023
    # Peak distance for PSNR: 1023
    # Point cloud sizes for org version, dec version, and the scaling ratio: 857966, 642798, 0.749211
    # Normals prepared.
    #
    # WARNING: no reflectance property in input files, disabling reflectance metrics.
    # 1. Use infile1 (A) as reference, loop over A, use normals on B. (A->B).
    # sse_dist_b_c2p 326824
    # c2p_debug 614568
    #    mae1      (p2point): 0.716308
    #    mse1      (p2point): 0.789544
    #    mse1,PSNR (p2point): 65.995
    #    mse1      (p2plane): 0.380929
    #    mse1,PSNR (p2plane): 69.1603
    #    c[0],    1         : 0.000782137
    #    c[1],    1         : 0.000160144
    #    c[2],    1         : 0.000187521
    #    c[0],PSNR1         : 31.0672
    #    c[1],PSNR1         : 37.9549
    #    c[2],PSNR1         : 37.2695
    # 2. Use infile2 (B) as reference, loop over B, use normals on A. (B->A).
    # sse_dist_b_c2p 252720
    # c2p_debug 331529
    #    mae2      (p2point): 0.515759
    #    mse2      (p2point): 0.585377
    #    mse2,PSNR (p2point): 67.2944
    #    mse2      (p2plane): 0.393156
    #    mse2,PSNR (p2plane): 69.0231
    #    c[0],    2         : 0.00068769
    #    c[1],    2         : 0.000147418
    #    c[2],    2         : 0.000175277
    #    c[0],PSNR2         : 31.6261
    #    c[1],PSNR2         : 38.3145
    #    c[2],PSNR2         : 37.5627
    # 3. Final (symmetric).
    #    maeF      (p2point): 0.716308
    #    mseF      (p2point): 0.789544
    #    mseF,PSNR (p2point): 65.995
    #    mseF      (p2plane): 0.393156
    #    mseF,PSNR (p2plane): 69.0231
    #    c[0],    F         : 0.000782137
    #    c[1],    F         : 0.000160144
    #    c[2],    F         : 0.000187521
    #    c[0],PSNRF         : 31.0672
    #    c[1],PSNRF         : 37.9549
    #    c[2],PSNRF         : 37.2695
    # Job done! 15.136 seconds elapsed (excluding the time to load the point clouds).
    pc2 = PyntCloud.from_file('C:/Users/User/Downloads/longdress_vox10_1300.ply.bin.decoded.ply')

    r = 1023

    p1 = pc1.points.values[:, :3]
    p1_n = pc1_n.points[['nx', 'ny', 'nz']].values
    p2 = pc2.points.values[:, :3]

    metrics = timing(compute_metrics)(p1, p2, r, p1_n)

    import pprint
    pprint.pprint(metrics)
