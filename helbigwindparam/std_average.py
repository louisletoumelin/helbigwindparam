import numpy as np

from typing import Union, List

from helbigwindparam.utils_func import print_func_executed_decorator, \
    timer_decorator, \
    change_several_dtype_if_required
from helbigwindparam.dispatch import get_window_idx_boundaries, \
    get_and_control_idx_boundary, \
    idx_from_array_shape

try:
    from helbigwindparam.config import config
except ModuleNotFoundError:
    pass

try:
    from numba import jit, guvectorize, vectorize, prange, float64, float32, int32, int64

    _numba = True
except ModuleNotFoundError:
    _numba = False

try:
    import tensorflow as tf

    _tensorflow = True
except ModuleNotFoundError:
    _tensorflow = False


def std_moving_window_numpy(array: np.ndarray,
                            y_left: Union[list, np.ndarray],
                            y_right: Union[list, np.ndarray],
                            x_left: Union[list, np.ndarray],
                            x_right: Union[list, np.ndarray]
                            ) -> np.ndarray:
    """Compute std with a moving window using numpy."""
    result = np.empty(y_left.shape)
    for index, (i1, j1, i2, j2) in enumerate(zip(y_left, y_right, x_left, x_right)):
        result[index] = np.std(array[i1:j1 + 1, i2:j2 + 1])
    return result.astype(np.float32)


@print_func_executed_decorator("std_slicing_numba",
                               level_begin="___",
                               level_end="___",
                               end="",
                               verbose=config["verbose"])
@timer_decorator("std_slicing_numba", unit='minute', level="....", verbose=config["verbose"])
def std_moving_window_numba(mnt: np.ndarray,
                            y_left: Union[list, np.ndarray],
                            y_right: Union[list, np.ndarray],
                            x_left: Union[list, np.ndarray],
                            x_right: Union[list, np.ndarray]
                            ) -> np.ndarray:
    """Compute std with a moving window using numba."""
    mnt, y_left, y_right, x_left, x_right = change_several_dtype_if_required(
        [mnt, y_left, y_right, x_left, x_right],
        [np.float32, np.int32, np.int32, np.int32, np.int32])
    _std_slicing_numba = jit([float32[:](float32[:, :], int32[:], int32[:], int32[:], int32[:])], nopython=True)(
        std_moving_window_numpy)
    return _std_slicing_numba(mnt, y_left, y_right, x_left, x_right)


@print_func_executed_decorator("std_slicing_numpy_list_comprehension",
                               level_begin="___",
                               level_end="___",
                               end="",
                               verbose=config["verbose"])
@timer_decorator("std_slicing_numpy_list_comprehension", unit='minute', level="....", verbose=config["verbose"])
def std_moving_window_numpy0(mnt: np.ndarray,
                             y_left: Union[list, np.ndarray],
                             y_right: Union[list, np.ndarray],
                             x_left: Union[list, np.ndarray],
                             x_right: Union[list, np.ndarray]
                             ) -> np.ndarray:
    """Compute std with a moving window with numpy."""
    list_indexes = change_several_dtype_if_required([y_left, y_right, x_left, x_right],
                                                    [np.int32, np.int32, np.int32, np.int32])
    for idx_file, file in enumerate(list_indexes):
        if file.ndim == 0:
            list_indexes[idx_file] = np.expand_dims(file, axis=0)

    y_left, y_right, x_left, x_right = list_indexes
    std_idx = np.empty(y_left.shape)
    std_idx = np.expand_dims(std_idx) if std_idx.ndim == 0 else std_idx
    for index, (i1, j1, i2, j2) in enumerate(zip(y_left, y_right, x_left, x_right)):
        std_idx[index] = np.std(mnt[i1:j1 + 1, i2:j2 + 1])

    return std_idx


def std_moving_window_tensorflow(mnt: np.ndarray,
                                 x_win: float = 69//2,
                                 y_win: float = 79//2
                                 ) -> np.ndarray:
    """Compute std with a moving window with tensorflow."""
    # reshape for tensorflow
    x_length = x_win * 2 + 1
    y_length = y_win * 2 + 1
    mnt = mnt.reshape((1, mnt.shape[0], mnt.shape[1], 1)).astype(np.float32)

    # filter
    filter_mean = np.ones((1, y_length, x_length, 1), dtype=np.float32) / (y_length * x_length)
    filter_mean = filter_mean.reshape((y_length, x_length, 1, 1))

    # convolution
    term_1 = tf.nn.convolution(mnt ** 2, filter_mean, strides=[1, 1, 1, 1], padding="SAME").numpy()[0, :, :, 0]
    term_2 = tf.nn.convolution(mnt, filter_mean, strides=[1, 1, 1, 1], padding="SAME").numpy()[0, :, :, 0]

    return np.sqrt(term_1 - term_2 ** 2)


def std_moving_window_num(mnt: np.ndarray,
                          idx_x: Union[List, np.ndarray],
                          idx_y: Union[List, np.ndarray],
                          x_win: Union[List, np.ndarray],
                          y_win: Union[List, np.ndarray],
                          verbose: Union[bool, None] = None
                          ) -> np.ndarray:
    """Compute std with a moving window with numpy or numba."""
    y_left, y_right, x_left, x_right = get_and_control_idx_boundary(mnt, idx_x, idx_y,
                                                                    x_win=x_win,
                                                                    y_win=y_win)

    if config["library_std_avg"] == "numba" and _numba:
        if verbose:
            print("std_avg_map library: numba")
        return std_moving_window_numba(mnt, y_left, y_right, x_left, x_right)
    else:
        if verbose:
            print("std_avg_map library: numpy")
        return std_moving_window_numpy(mnt, y_left, y_right, x_left, x_right)


@print_func_executed_decorator("std_avg_map", level_begin="___", level_end="___", end="", verbose=config["verbose"])
@timer_decorator("std_avg_map", unit='second', level="....", verbose=config["verbose"])
def std_moving_window_map(mnt: np.ndarray,
                          x_win: float = 69//2,
                          y_win: float = 79//2,
                          idx_x: Union[List, np.ndarray, None] = None,
                          idx_y: Union[List, np.ndarray, None] = None,
                          library_std_avg: Union[str, None] = None,
                          verbose: Union[bool, None] = None
                          ) -> np.ndarray:
    """Compute standard deviation with a moving window on a map."""
    verbose = config["verbose"] if verbose is None else verbose

    if idx_x is None and idx_y is None:
        idx_x, idx_y = idx_from_array_shape(mnt)
    library_std_avg = library_std_avg if library_std_avg is not None else config["library_std_avg"]
    if library_std_avg == "tensorflow" and _tensorflow:
        # std_avg is flat
        if verbose:
            print("std_avg_map library: tensorflow")
        std_avg = std_moving_window_tensorflow(mnt, x_win=x_win, y_win=y_win)
    else:
        std_avg = std_moving_window_num(mnt, idx_x, idx_y, x_win, y_win, verbose=verbose)

    std_avg = std_avg.reshape((mnt.shape[0], mnt.shape[1]))
    return std_avg


def std_moving_window_idx(mnt: np.ndarray,
                          x_win: float = 69//2,
                          y_win=79//2,
                          idx_x=None,
                          idx_y=None
                          ) -> np.ndarray:
    """Compute standard deviation with a moving window using indexes."""
    y_left, y_right, x_left, x_right = get_window_idx_boundaries(idx_x, idx_y, x_win=x_win, y_win=y_win)
    return std_moving_window_numpy0(mnt, y_left, y_right, x_left, x_right)
