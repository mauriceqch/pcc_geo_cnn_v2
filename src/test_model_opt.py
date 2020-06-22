import numpy as np
from numpy.testing import assert_array_equal
from model_opt import build_points_threshold, compute_optimal_thresholds


def assert_pa_list_equal(pa_list, pa_list_expected):
    for i in range(len(pa_list)):
        assert_array_equal(pa_list[i][1], pa_list_expected[i][1])
        assert pa_list[i][0] == pa_list_expected[i][0]


def test_build_points_threshold():
    x_hat = np.array([[0, 2, 4, 6],
                      [2, 4, 6, 0]])
    thresholds = np.array([1, 3, 5, 7])
    len_block = 2
    pa_list = build_points_threshold(x_hat, thresholds, len_block)
    pa_list_expected = list(enumerate([
        [[0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2]],
        [[0, 2], [0, 3], [1, 1], [1, 2]],
        [[0, 3], [1, 2]]]))
    assert_pa_list_equal(pa_list, pa_list_expected)
    pa_list_2 = build_points_threshold(x_hat, thresholds, len_block, max_delta=2.5)
    assert_pa_list_equal(pa_list_2, pa_list_expected[1:])
    pa_list_3 = build_points_threshold(x_hat, thresholds, len_block, max_delta=2)
    assert_pa_list_equal(pa_list_3, pa_list_expected[2:])


def test_compute_optimal_thresholds():
    block = np.array([[0, 0]])
    x_hat = np.array([[0, 1]])
    thresholds = np.array([0, 1.5, 3.0])
    resolution = np.sqrt(2)
    ret_opt_metrics, best_thresholds = compute_optimal_thresholds(block, x_hat, thresholds, resolution,
                                                                  opt_metrics=['d1_mse'], max_deltas=[np.inf])
    assert ret_opt_metrics == ['d1_mse_inf']
    assert best_thresholds == [2]
    ret_opt_metrics, best_thresholds = compute_optimal_thresholds(block, x_hat, thresholds, resolution,
                                                                  opt_metrics=['d1_mse'], max_deltas=[np.inf],
                                                                  fixed_threshold=True)
    assert ret_opt_metrics == ['d1_mse_inf']
    assert best_thresholds == [1]

    thresholds = np.array([0, 1.5, 3.0, 4.5, 6.0])
    ret_opt_metrics, best_thresholds = compute_optimal_thresholds(block, x_hat, thresholds, resolution,
                                                                  opt_metrics=['d1_mse'], max_deltas=[np.inf],
                                                                  fixed_threshold=True)
    assert ret_opt_metrics == ['d1_mse_inf']
    assert best_thresholds == [2]

