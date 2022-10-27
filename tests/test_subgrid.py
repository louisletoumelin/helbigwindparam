import numpy as np
from numpy.testing import assert_allclose

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from helbigwindparam.subgrid import xsi
from helbigwindparam.slope_average import mu_average


def test_xsi_helbig_map():
    """test passes 02/12/2021"""

    array_test_2 = np.array([[1, 3, 5, 7],
                             [2, 4, 6, 8],
                             [9, 11, 13, 15],
                             [10, 12, 14, 16]])

    std_expected = np.array([[1.118033, 1.707825, 1.707825, 1.118033],
                             [3.696845, 3.915780, 3.915780, 3.696845],
                             [3.696845, 3.915780, 3.915780, 3.696845],
                             [1.118033, 1.707825, 1.707825, 1.118033]])

    mu_expected = np.array([[2.37170825, 2.37170825, 2.37170825, 2.37170825],
                            [2.63231, 2.63231, 2.63231, 2.63231],
                            [2.63231, 2.63231, 2.63231, 2.63231],
                            [2.37170825, 2.37170825, 2.37170825, 2.37170825]], dtype=np.float32)

    array_test_3 = np.random.randint(0, 20, (100, 200))
    mu_3 = mu_average(array_test_3, 1, idx_x=50, idx_y=50, x_win=1, y_win=1)

    result_expected_2 = np.sqrt(2) * std_expected / mu_expected
    mu = mu_average(array_test_2, 1, x_win=1, y_win=1, library_mu_avg="tensorflow")
    result_2 = xsi(array_test_2, mu, x_win=1, y_win=1, library_std_avg="numba")
    result_3 = xsi(array_test_2, mu, x_win=1, y_win=1, library_std_avg="tensorflow")
    result_4 = xsi(array_test_2, mu, x_win=1, y_win=1, library_std_avg="numpy")

    result_5 = mu_average(array_test_3, 1, x_win=1, y_win=1, library_mu_avg="tensorflow")[50, 50]
    result_6 = mu_average(array_test_3, 1, x_win=1, y_win=1, library_mu_avg="numba")[50, 50]
    result_7 = mu_average(array_test_3, 1, x_win=1, y_win=1, library_mu_avg="numpy")[50, 50]
    result_8 = mu_average(array_test_3, 1, idx_x=50, idx_y=50, x_win=1, y_win=1)

    assert_allclose(result_2[1:-1, 1:-1], result_expected_2[1:-1, 1:-1], rtol=1e-03, atol=0.001)
    assert_allclose(result_2[1:-1, 1:-1], result_3[1:-1, 1:-1], rtol=1e-03, atol=0.001)
    assert_allclose(result_2[1:-1, 1:-1], result_4[1:-1, 1:-1], rtol=1e-03, atol=0.001)

    assert_allclose(result_5, result_8, rtol=1e-03, atol=0.001)
    assert_allclose(result_6, result_8, rtol=1e-03, atol=0.001)
    assert_allclose(result_7, result_8, rtol=1e-03, atol=0.001)
