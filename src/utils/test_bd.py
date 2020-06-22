from utils.bd import bdsnr, bdrate
import numpy as np

eps = 1e-8


def exp_x(test_arr):
    return [(np.exp(x), y) for x, y in test_arr]


def test_bdrate():
    expected_1 = 100.0
    test_1 = [[1.0, 1.0], [2.0, 2.0]]
    test_2 = [[2.0, 1.0], [4.0, 2.0]]
    assert 3.31 < bdrate(test_1, test_2, pchip=False) - expected_1 < 3.32
    assert abs(bdrate(test_1, test_2, pchip=True) - expected_1) < eps


def test_bdsnr():
    expected_1 = 1.0
    test_1 = exp_x([[1.0, 1.0], [2.0, 2.0], [3.0, 2.0]])
    test_2 = exp_x([[1.0, 2.0], [2.0, 3.0], [3.0, 3.0]])
    assert abs(bdsnr(test_1, test_2, pchip=False) - expected_1) < eps
    assert abs(bdsnr(test_1, test_2, pchip=True) - expected_1) < eps

    # In this test, the saw-like function cannot be well approximated by a cubic fit
    expected_2 = -0.583
    test_3 = exp_x([[1.0, 2.0], [2.0, 3.0], [3.0, 2.0], [4.0, 3.0], [5.0, 2.0]])
    test_4 = exp_x([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0], [5.0, 2.0]])
    assert bdsnr(test_3, test_4, pchip=False) - expected_2 < 0.09
    assert abs(bdsnr(test_3, test_4, pchip=True) - expected_2) < 0.0004

