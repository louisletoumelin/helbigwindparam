import numpy as np
from numpy.testing import assert_array_almost_equal,\
    assert_almost_equal
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from helbigwindparam.slope import slope_mu_map, slope_mu_idx


def test_mu_helbig_map():
    """
    test passes 30/11/2021
    test passes 23/10/2022
    """

    array_test_1 = np.array([[1, 2],
                             [3, 4]])

    expected_result_1 = np.sqrt(np.array([[2.5, 2.5],
                                          [2.5, 2.5]]))

    array_test_2 = np.array([[1, 3, 5, 7],
                             [2, 4, 6, 8],
                             [9, 11, 13, 15],
                             [10, 12, 14, 16]])

    expected_result_2 = np.array([[1.5811388, 1.5811388, 1.5811388, 1.5811388],
                                  [3.1622777, 3.1622777, 3.1622777, 3.1622777],
                                  [3.1622777, 3.1622777, 3.1622777, 3.1622777],
                                  [1.5811388, 1.5811388, 1.5811388, 1.5811388]], dtype=np.float32)

    result_1 = slope_mu_map(array_test_1, 1)
    result_2 = slope_mu_map(array_test_2, 1)

    assert_array_almost_equal(result_1, expected_result_1)
    assert_array_almost_equal(result_2, expected_result_2)


def test_mu_helbig_map_idx():
    """
    test passes 30/11/2021
    test passes 23/10/2022
    """
    array_test = np.array([[1, 3, 5, 7],
                           [2, 4, 6, 8],
                           [9, 11, 13, 15],
                           [10, 12, 14, 16]])
    expected_result = np.array([[1.5811388, 1.5811388, 1.5811388, 1.5811388],
                                [3.1622777, 3.1622777, 3.1622777, 3.1622777],
                                [3.1622777, 3.1622777, 3.1622777, 3.1622777],
                                [1.5811388, 1.5811388, 1.5811388, 1.5811388]],
                               dtype=np.float32)
    result = slope_mu_idx(array_test,
                          1,
                          np.array([1]),
                          np.array([2]))
    assert_almost_equal(result, expected_result[2, 1])

