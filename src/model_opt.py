import numpy as np
import logging
from scipy.spatial.ckdtree import cKDTree
from utils.pc_metric import validate_opt_metrics, compute_metrics

logger = logging.getLogger(__name__)


def build_points_threshold(x_hat, thresholds, len_block, max_delta=np.inf):
    pa_list = []
    for i, t in enumerate(thresholds):
        pa = np.argwhere(x_hat > t).astype('float32')
        if len(pa) == 0:
            break
        len_ratio = len(pa) / len_block
        if (1 / max_delta) < len_ratio < max_delta:
            pa_list.append((i, pa))
    return pa_list


def compute_optimal_thresholds(block, x_hat, thresholds, resolution, normals=None, opt_metrics=['d1_mse'],
                               max_deltas=[np.inf], fixed_threshold=False):
    validate_opt_metrics(opt_metrics, with_normals=normals is not None)
    assert len(max_deltas) > 0
    best_thresholds = []
    ret_opt_metrics = [f'{opt_metric}_{max_delta}' for max_delta in max_deltas for opt_metric in opt_metrics]
    if fixed_threshold:
        half_thr = len(thresholds) // 2
        half_pa = np.argwhere(x_hat > thresholds[half_thr]).astype('float32')
        logger.info(f'Fixed threshold {half_thr}/{len(thresholds)} with {len(half_pa)}/{len(block)} points (ratio {len(half_pa)/len(block):.2f})')
        return ret_opt_metrics, [half_thr] * len(max_deltas) * len(opt_metrics)

    pa_list = build_points_threshold(x_hat, thresholds, len(block))
    max_threshold_idx = len(thresholds) - 1
    if len(pa_list) == 0:
        return ret_opt_metrics, [max_threshold_idx] * len(opt_metrics)

    t1 = cKDTree(block[:, :3], balanced_tree=False)
    pa_metrics = [compute_metrics(block[:, :3], pa, resolution - 1, p1_n=normals, t1=t1) for _, pa in pa_list]

    log_message = f'Processing max_deltas {max_deltas} on block with {len(block)} points'
    for max_delta in max_deltas:
        if max_delta is not None:
            cur_pa_list = build_points_threshold(x_hat, thresholds, len(block), max_delta)
            if len(cur_pa_list) > 0:
                idx_mask = [x[0] for x in cur_pa_list]
                cur_pa_metrics = [pa_metrics[i] for i in idx_mask]
            else:
                cur_pa_list = pa_list
                cur_pa_metrics = pa_metrics
        else:
            cur_pa_list = pa_list
            cur_pa_metrics = pa_metrics
        log_message += f'\n{len(cur_pa_list)}/{len(thresholds)} thresholds eligible for max_delta {max_delta}'
        for opt_metric in opt_metrics:
            best_threshold_idx = np.argmin([x[opt_metric] for x in cur_pa_metrics])
            cur_best_metric = cur_pa_metrics[best_threshold_idx][opt_metric]

            # Check for failure scenarios
            mean_point_metric = compute_metrics(block[:, :3],
                                                np.round(np.mean(block[:, :3], axis=0))[np.newaxis, :],
                                                resolution - 1, p1_n=normals, t1=t1)[opt_metric]
            # In case a single point is better than the network output, this is a failure case
            # Do not output any points
            if cur_best_metric > mean_point_metric:
                best_threshold_idx = max_threshold_idx
                final_idx = best_threshold_idx
                log_message += f', {opt_metric} {final_idx} 0/{len(block)}, metric {cur_best_metric:.2e} > mean point metric {mean_point_metric:.2e}'
            else:
                final_idx = cur_pa_list[best_threshold_idx][0]
                cur_n_points = len(cur_pa_list[best_threshold_idx][1])
                log_message += f', {opt_metric} {final_idx} {cur_n_points}/{len(block)} points (ratio {cur_n_points/len(block):.2f}) {cur_best_metric :.2e} < mean point metric {mean_point_metric:.2e}'
            best_thresholds.append(final_idx)
    logger.info(log_message)
    assert len(ret_opt_metrics) == len(best_thresholds)

    return ret_opt_metrics, best_thresholds