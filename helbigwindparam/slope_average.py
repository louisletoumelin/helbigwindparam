import numpy as np

from typing import Union, List

from helbigwindparam.utils_func import print_func_executed_decorator, \
    timer_decorator, \
    change_several_dtype_if_required
from helbigwindparam.dispatch import detect_type_input, \
    get_window_idx_boundaries,\
    idx_from_array_shape,\
    get_and_control_idx_boundary
from helbigwindparam.slope import slope_mu_map

try:
    from helbigwindparam.config import config
except ModuleNotFoundError:
    print("config not imported")


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


@print_func_executed_decorator("mu_average_numba",
                               level_begin="____",
                               level_end="____",
                               end="",
                               verbose=config["verbose"])
@timer_decorator("mu_average_numba", unit='minute', level=".... ", verbose=config["verbose"])
def mu_average_numba(mu: np.ndarray,
                     y_left: Union[List, np.ndarray],
                     y_right: Union[List, np.ndarray],
                     x_left: Union[List, np.ndarray],
                     x_right: Union[List, np.ndarray]
                     ) -> np.ndarray:
    """Compute mean slope with numba."""

    mu, y_left, y_right, x_left, x_right = change_several_dtype_if_required(
        [mu, y_left, y_right, x_left, x_right], [np.float32, np.int32, np.int32, np.int32, np.int32])

    inputs_outputs_types = [float32[:](float32[:, :], int32[:], int32[:], int32[:], int32[:])]
    jit_mean = jit(inputs_outputs_types, nopython=True)(mean_slicing_numpy)

    return jit_mean(mu, y_left, y_right, x_left, x_right)


@print_func_executed_decorator("mu_average_numpy_list_comprehension",
                               level_begin="__",
                               level_end="__",
                               end="",
                               verbose=config["verbose"])
@timer_decorator("mu_average_numpy_list_comprehension", unit='minute', level=".... ", verbose=config["verbose"])
def mu_average_numpy_list_comprehension(mu: np.ndarray,
                                        y_left: Union[List, np.ndarray],
                                        y_right: Union[List, np.ndarray],
                                        x_left: Union[List, np.ndarray],
                                        x_right: Union[List, np.ndarray]
                                        ) -> np.ndarray:
    """Compute mean slope with numpy using list comprehension."""
    return np.array([np.mean(mu[i1:j1 + 1, i2:j2 + 1]) for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])


@timer_decorator("mu_average_idx", unit='minute', level=".... ", verbose=config["verbose"])
def mu_average_idx(mnt: np.ndarray,
                   dx: float,
                   idx_x: Union[List, np.ndarray],
                   idx_y: Union[List, np.ndarray],
                   x_win: float = 69//2,
                   y_win: float = 79//2
                   ) -> np.ndarray:
    """Compute mean slope around indexes."""

    assert idx_x is not None
    assert idx_y is not None

    y_left, y_right, x_left, x_right = get_window_idx_boundaries(idx_x, idx_y, x_win=x_win, y_win=y_win)

    list_indexes = change_several_dtype_if_required([y_left, y_right, x_left, x_right],
                                                    [np.int32, np.int32, np.int32, np.int32])
    for idx_file, file in enumerate(list_indexes):
        if file.ndim == 0:
            list_indexes[idx_file] = np.expand_dims(file, axis=0)

    y_left, y_right, x_left, x_right = list_indexes

    mu_idx = np.empty(y_left.shape)
    if mu_idx.ndim == 0:
        mu_idx = np.expand_dims(mu_idx)

    for index, (i1, j1, i2, j2) in enumerate(zip(y_left, y_right, x_left, x_right)):
        mu_map_i = slope_mu_map(mnt[i1 - 1:j1 + 1 + 1, i2 - 1:j2 + 1 + 1], dx)
        mu_idx[index] = np.mean(mu_map_i[1:-1, 1:-1])

    return mu_idx


@print_func_executed_decorator("mu_average_tensorflow",
                               level_begin="____",
                               level_end="____",
                               end="",
                               verbose=config["verbose"])
@timer_decorator("mu_average_tensorflow", unit='minute', level=".... ", verbose=config["verbose"])
def mu_average_tensorflow(mu: np.ndarray,
                          x_win: float = 69//2,
                          y_win: float = 79//2
                          ) -> np.ndarray:
    """Compute mean slope with tensorflow."""

    # reshape for tensorflow
    mu = mu.reshape((1, mu.shape[0], mu.shape[1], 1)).astype(np.float32)

    # filter
    x_length = x_win * 2 + 1
    y_length = y_win * 2 + 1
    filter_mean = np.ones((1, y_length, x_length, 1), dtype=np.float32) / (x_length * y_length)
    filter_mean = filter_mean.reshape((y_length, x_length, 1, 1))

    # convolution
    return tf.nn.convolution(mu, filter_mean, strides=[1, 1, 1, 1], padding="SAME").numpy()[0, :, :, 0]


def mu_average_map(mnt: np.ndarray,
                   dx: float,
                   x_win: float = 69//2,
                   y_win: float = 79//2,
                   library_mu_avg: Union[str, None] = None,
                   verbose: Union[bool, None] = None
                   ) -> np.ndarray:
    """Compute mean slope on a map."""

    verbose = config["verbose"] if verbose is None else verbose
    library_mu_avg = config["library_mu_avg"] if library_mu_avg is None else library_mu_avg
    
    mu = slope_mu_map(mnt, dx)

    if library_mu_avg == "tensorflow" and _tensorflow:
        # mu_avg is not flat (I think)
        if verbose:
            print("mu_average_map library: tensorflow")
        return mu_average_tensorflow(mu, x_win=x_win, y_win=y_win)
    else:
        idx_x, idx_y = idx_from_array_shape(mnt)
        y_left, y_right, x_left, x_right = get_and_control_idx_boundary(mnt, idx_x, idx_y, x_win=x_win, y_win=y_win)
        # mu_avg is flat
        if library_mu_avg == 'numba' and _numba:
            print("mu_average_map library: numba")
            mu_avg = mu_average_numba(mu, y_left, y_right, x_left, x_right)
        elif library_mu_avg == "numpy":
            print("mu_average_map library: numpy")
            mu_avg = mu_average_numpy_list_comprehension(mu, y_left, y_right, x_left, x_right)
        return mu_avg.reshape(mnt.shape[0], mnt.shape[1])


@print_func_executed_decorator("mu_average", level_begin="__", level_end="__", end="", verbose=config["verbose"])
@timer_decorator("mu_average", unit='second', level="....", verbose=config["verbose"])
def mu_average(mnt: np.ndarray,
               dx: float,
               idx_x: Union[np.ndarray, List, float, None] = None,
               idx_y: Union[np.ndarray, List, float, None] = None,
               x_win: float = 69//2,
               y_win: float = 79//2,
               library_mu_avg: Union[str, None] = None
               ) -> np.ndarray:
    """Compute mean slope."""
    if idx_x is not None and np.ndim(idx_x) == 0:
        idx_x = np.expand_dims(idx_x, axis=-1)
        idx_y = np.expand_dims(idx_y, axis=-1)

    type_input = detect_type_input(idx_x, idx_y)
    if type_input == "map":
        return mu_average_map(mnt, dx, x_win=x_win, y_win=y_win, library_mu_avg=library_mu_avg)
    elif type_input == "idx":
        return mu_average_idx(mnt, dx, idx_x, idx_y, x_win=x_win, y_win=y_win)


def mean_slicing_numpy(array: np.ndarray,
                       y_left: Union[List, np.ndarray],
                       y_right: Union[List, np.ndarray],
                       x_left: Union[List, np.ndarray],
                       x_right: Union[List, np.ndarray]
                       ) -> np.ndarray:
    """Compute mean with a moving window on a map"""
    result = np.empty(y_left.shape)
    for index, (i1, j1, i2, j2) in enumerate(zip(y_left, y_right, x_left, x_right)):
        result[index] = np.mean(array[i1:j1 + 1, i2:j2 + 1])
    return result.astype(np.float32)
