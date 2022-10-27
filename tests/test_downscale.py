import numpy as np
from numpy.testing import assert_array_almost_equal

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from helbigwindparam.laplacian import laplacian_map, laplacian_idx


def test_laplacian_classic():
    """
    test passes 30/11/2021
    test passes 23/10/2022
    """
    array_test_1 = np.array([[1, 2],
                             [3, 4]])

    expected_result_1 = np.array([[3, 1],
                                  [-1, -3]])

    array_test_2 = np.array([[12, 14, 28, 32],
                             [15, 27, 42, 53],
                             [41, 40, 21, 13],
                             [18, 12, 21, 42]])

    expected_result_2 = np.array([[5, 25, 4, 17],
                                  [35, 3, -39, -72],
                                  [-50, -59, 32, 77],
                                  [17, 43, 12, -50]])

    result_np_1 = laplacian_map(array_test_1, 1, library_laplacian="numpy", helbig=False, verbose=False)
    result_np_2 = laplacian_map(array_test_2, 1, library_laplacian="numpy", helbig=False, verbose=False)
    result_nb_1 = laplacian_map(array_test_1, 1, library_laplacian="numba", helbig=False, verbose=False)
    result_nb_2 = laplacian_map(array_test_2, 1, library_laplacian="numba", helbig=False, verbose=False)
    result_tf_2 = laplacian_map(array_test_2, 1, library_laplacian="tensorflow", helbig=False, verbose=False)
    assert_array_almost_equal(result_np_1, expected_result_1)
    assert_array_almost_equal(result_np_2, expected_result_2)
    assert_array_almost_equal(result_nb_1, expected_result_1)
    assert_array_almost_equal(result_nb_2, expected_result_2)

    # We are not testing tensorflow on the small array because of problems at the borders of the image
    # For the bigger array, we test only in the inner portion of the array to avoid border issues
    assert_array_almost_equal(result_tf_2[1:-1, 1:-1], expected_result_2[1:-1, 1:-1])


def test_laplacian_helbig():
    """
    test passes 30/11/2021
    test passes 23/10/2022
    """
    array_test_1 = np.array([[1, 2],
                             [3, 4]])

    expected_result_1 = np.array([[3, 1],
                                  [-1, -3]]) / 4

    array_test_2 = np.array([[12, 14, 28, 32],
                             [15, 27, 42, 53],
                             [41, 40, 21, 13],
                             [18, 12, 21, 42]])

    expected_result_2 = np.array([[5, 25, 4, 17],
                                  [35, 3, -39, -72],
                                  [-50, -59, 32, 77],
                                  [17, 43, 12, -50]]) / 4

    result_np = laplacian_map(array_test_1, 1, library_laplacian="numpy", helbig=True, verbose=False)
    result_nb = laplacian_map(array_test_2, 1, library_laplacian="numba", helbig=True, verbose=False)
    result_tf = laplacian_map(array_test_2, 1, library_laplacian="tensorflow", helbig=True, verbose=False)
    assert_array_almost_equal(result_np, expected_result_1)
    assert_array_almost_equal(result_nb, expected_result_2)

    # We test only in the inner portion of the array to avoid border issues
    assert_array_almost_equal(result_tf[1:-1, 1:-1], expected_result_2[1:-1, 1:-1])


def test_laplacian_idx_and_map_give_same_result():
    """
    test passes 30/11/2021
    test passes 23/10/2022
    """
    array_test = np.array([[12, 14, 28, 32],
                           [15, 27, 42, 53],
                           [41, 40, 21, 13],
                           [18, 12, 21, 42]])

    result_idx_1 = laplacian_idx(array_test, np.array([1]), np.array([2]), 1, library_laplacian="numpy", helbig=False)
    result_map_1 = laplacian_map(array_test, 1, library_laplacian="numpy", helbig=False, verbose=False)[2, 1]

    result_idx_2 = laplacian_idx(array_test, np.array([1]), np.array([2]), 1, library_laplacian="numba", helbig=False)
    result_map_2 = laplacian_map(array_test, 1, library_laplacian="numba", helbig=False, verbose=False)[2, 1]

    result_idx_tf_no_helbig = laplacian_idx(array_test, np.array([1]), np.array([2]), 1, library_laplacian="numba", helbig=False)
    result_map_tf_no_helbig = laplacian_map(array_test, 1, library_laplacian="numba", helbig=False, verbose=False)[2, 1]

    result_idx_3 = laplacian_idx(array_test, np.array([1]), np.array([2]), 1, library_laplacian="numpy", helbig=True)
    result_map_3 = laplacian_map(array_test, 1, library_laplacian="numpy", helbig=True, verbose=False)[2, 1]

    result_idx_tf_helbig = laplacian_idx(array_test, np.array([1]), np.array([2]), 1, library_laplacian="numba", helbig=True)
    result_map_tf_helbig = laplacian_map(array_test, 1, library_laplacian="numba", helbig=True, verbose=False)[2, 1]

    result_idx_4 = laplacian_idx(array_test, np.array([1]), np.array([2]), 1, library_laplacian="numba", helbig=True)
    result_map_4 = laplacian_map(array_test, 1, library_laplacian="numba", helbig=True, verbose=False)[2, 1]

    assert_array_almost_equal(result_map_1, result_idx_1)
    assert_array_almost_equal(result_map_2, result_idx_2)
    assert_array_almost_equal(result_idx_3, result_map_3)
    assert_array_almost_equal(result_idx_4, result_map_4)
    assert_array_almost_equal(result_idx_tf_helbig, result_map_tf_helbig)
    assert_array_almost_equal(result_idx_tf_no_helbig, result_map_tf_no_helbig)


def test_laplacian_idx_result():
    """
    test passes 30/11/2021
    test passes 23/10/2022
    """
    array_test = np.array([[12, 14, 28, 32],
                           [15, 27, 42, 53],
                           [41, 40, 21, 13],
                           [18, 12, 21, 42]])
    expected_result_2 = np.array([[5, 25, 4, 17],
                                  [35, 3, -39, -72],
                                  [-50, -59, 32, 77],
                                  [17, 43, 12, -50]]) / 4

    result_idx_np = laplacian_idx(array_test, np.array([1]), np.array([2]), 1, library_laplacian="numpy", helbig=True)
    result_idx_nb = laplacian_idx(array_test, np.array([1]), np.array([2]), 1, library_laplacian="numba", helbig=True)
    assert_array_almost_equal(result_idx_np, expected_result_2[2, 1])
    assert_array_almost_equal(result_idx_nb, expected_result_2[2, 1])