import numpy as np
from numpy.testing import assert_allclose

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from helbigwindparam.slope_average import mu_average


def test_mu_helbig_average():
    """
    test passes 01/12/2021
    test passes 16/10/2022
    """

    array_test_2 = np.array([[1, 3, 5, 7],
                             [2, 4, 6, 8],
                             [9, 11, 13, 15],
                             [10, 12, 14, 16]])
    expected_result_2 = np.array([[2.37170825, 2.37170825, 2.37170825, 2.37170825],
                                  [2.63231, 2.63231, 2.63231, 2.63231],
                                  [2.63231, 2.63231, 2.63231, 2.63231],
                                  [2.37170825, 2.37170825, 2.37170825, 2.37170825]], dtype=np.float32)

    array_test_3 = np.random.randint(0, 20, (100, 200))

    result_3 = mu_average(array_test_2, 1, x_win=1, y_win=1, library_mu_avg="tensorflow")
    result_4 = mu_average(array_test_2, 1, x_win=1, y_win=1, library_mu_avg="numba")
    result_5 = mu_average(array_test_2, 1, x_win=1, y_win=1, library_mu_avg="numpy")
    result_6 = mu_average(array_test_3, 1, idx_x=50, idx_y=50, x_win=1, y_win=1)
    result_7 = mu_average(array_test_3, 1, x_win=1, y_win=1)[50, 50]

    assert_allclose(result_3[1:-1, 1:-1], expected_result_2[1:-1, 1:-1], rtol=1e-03, atol=0.001)
    assert_allclose(result_4[1:-1, 1:-1], expected_result_2[1:-1, 1:-1], rtol=1e-03, atol=0.001)
    assert_allclose(result_5[1:-1, 1:-1], expected_result_2[1:-1, 1:-1], rtol=1e-03, atol=0.001)
    assert_allclose(result_6, result_7, rtol=1e-03, atol=0.001)
