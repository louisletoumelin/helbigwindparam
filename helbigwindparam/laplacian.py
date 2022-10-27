import numpy as np

from typing import Union, List

from helbigwindparam.utils_func import change_dtype_if_required, \
    print_func_executed_decorator, \
    timer_decorator

try:
    from helbigwindparam.config import config
except ModuleNotFoundError:
    pass

try:
    import tensorflow as tf

    _tensorflow = True
except ModuleNotFoundError:
    _tensorflow = False

try:
    from numba import jit, guvectorize, vectorize, prange, float64, float32, int32, int64

    _numba = True
except ModuleNotFoundError:
    _numba = False


@print_func_executed_decorator("laplacian_map_tensorflow",
                               level_begin="____",
                               level_end="____",
                               end="",
                               verbose=config["verbose"])
@timer_decorator("laplacian_map_tensorflow", unit='second', level="....", verbose=config["verbose"])
def laplacian_map_tensorflow(mnt: np.ndarray,
                             dx: float,
                             helbig: bool = True
                             ) -> np.ndarray:
    """Compute map of laplacian using tensorflow."""
    mnt = mnt.reshape((1, mnt.shape[0], mnt.shape[1], 1)).astype(np.float32)

    # filter
    filter_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    # convolution
    filter_laplacian = filter_laplacian.reshape((3, 3, 1, 1))
    laplacian = np.float32(tf.nn.convolution(mnt, filter_laplacian, strides=[1, 1, 1, 1], padding="SAME").numpy())
    laplacian = laplacian[0, :, :, 0] / (dx ** 2)

    if helbig:
        return laplacian * (dx / 4)
    else:
        return laplacian


@print_func_executed_decorator("laplacian_map", level_begin="__", level_end="__", end="", verbose=config["verbose"])
@timer_decorator("laplacian_map", unit='second', level="....", verbose=config["verbose"])
def laplacian_map(mnt: np.ndarray,
                  dx: float,
                  helbig: bool = True,
                  verbose: Union[bool, None] = None,
                  library_laplacian=None
                  ) -> np.ndarray:
    """Compute laplacian on a map."""
    library_laplacian = config["library_laplacian"] if library_laplacian is None else library_laplacian
    verbose = config["verbose"] if verbose is None else verbose

    if library_laplacian == "tensorflow" and _tensorflow:
        return laplacian_map_tensorflow(mnt, dx, helbig=helbig)
    if mnt.ndim > 2:
        return np.vectorize(_laplacian_map, signature='(m,n),(),(),(),()->(m,n)')(mnt,
                                                                                  dx,
                                                                                  library_laplacian,
                                                                                  helbig,
                                                                                  verbose)
    else:
        return _laplacian_map(mnt, dx, helbig, library_laplacian, verbose)


def _laplacian_map(mnt: np.ndarray,
                   dx: float,
                   helbig: bool,
                   library_laplacian: str,
                   verbose: Union[bool, None]
                   ) -> np.ndarray:
    """Computes laplacian using numpy."""
    # Pad mnt to compute laplacian on edges
    mnt_padded = np.pad(mnt, (1, 1), "edge").astype(np.float32)
    if verbose:
        print("__MNT padded for laplacian computation") if verbose else None
    shape = mnt_padded.shape

    # Use meshgrid to create indexes with mnt size and use numpy broadcasting when selecting indexes
    xx, yy = np.array(np.meshgrid(list(range(shape[1])), list(range(shape[0])))).astype(np.int32)

    # Compute laplacian on indexes using an index for every grid point (meshgrid)
    return laplacian_idx(mnt_padded,
                         xx[1:-1, 1:-1],
                         yy[1:-1, 1:-1],
                         dx,
                         library_laplacian=library_laplacian,
                         helbig=helbig)


@print_func_executed_decorator("laplacian_idx", level_begin="____", level_end="____", end="", verbose=config["verbose"])
@timer_decorator("laplacian_idx", unit='second', level="....", verbose=config["verbose"])
def laplacian_idx(mnt: np.ndarray,
                  idx_x: Union[np.ndarray, List, float],
                  idx_y: Union[np.ndarray, List, float],
                  dx: float,
                  helbig: bool = True,
                  library_laplacian: Union[str, None] = None,
                  verbose: Union[bool, None] = None
                  ) -> np.ndarray:
    """Compute laplacian around indexes (idx) that indicate the position of a location on an array."""

    library_laplacian = config["library_laplacian"] if library_laplacian is None else library_laplacian
    verbose = config["verbose"] if verbose is None else verbose

    if library_laplacian == 'numba' and _numba:
        if verbose:
            print(f"Library laplacian: numba")
        return _laplacian_numba_idx(mnt, idx_x, idx_y, dx, helbig=helbig)
    else:
        if verbose:
            print(f"Library laplacian: numpy")
        return _laplacian_numpy_idx(mnt, idx_x, idx_y, dx, helbig=helbig)


def _laplacian_numpy_idx(mnt: np.ndarray,
                         idx_x: Union[np.ndarray, List, float],
                         idx_y: Union[np.ndarray, List, float],
                         dx: float,
                         helbig: bool = True
                         ) -> np.ndarray:
    """Compute laplacian around indexes using numpy."""
    a = np.float32((mnt[idx_y - 1, idx_x]
                    + mnt[idx_y + 1, idx_x]
                    + mnt[idx_y, idx_x - 1]
                    + mnt[idx_y, idx_x + 1]
                    - 4 * mnt[idx_y, idx_x]) / dx ** 2)
    c = np.float32(dx / 4) if helbig else 1
    return a * c


def _laplacian_loop_numpy_1d_helbig(mnt: np.ndarray,
                                    idx_x: np.ndarray,
                                    idx_y: np.ndarray,
                                    dx: float
                                    ) -> np.ndarray:
    """Compute laplacian using Helbig formula on a 1D array using loop and numby (for numba jit optimization)."""
    laplacian = np.empty(idx_x.shape, np.float32)
    for i in range(idx_x.shape[0]):
        a = (mnt[idx_y[i] - 1, idx_x[i]]
             + mnt[idx_y[i] + 1, idx_x[i]]
             + mnt[idx_y[i], idx_x[i] - 1]
             + mnt[idx_y[i], idx_x[i] + 1]
             - 4 * mnt[idx_y[i], idx_x[i]]) / dx ** 2
        c = dx / 4
        laplacian[i] = a * c
    return laplacian


def _laplacian_loop_numpy_1d(mnt: np.ndarray,
                             idx_x: np.ndarray,
                             idx_y: np.ndarray,
                             dx: float
                             ) -> np.ndarray:
    """Compute laplacian using standard formula on a 1D array using loop and numby (for numba jit optimization)."""
    laplacian = np.empty(idx_x.shape, np.float32)
    for i in range(idx_x.shape[0]):
        laplacian[i] = (mnt[idx_y[i] - 1, idx_x[i]]
                        + mnt[idx_y[i] + 1, idx_x[i]]
                        + mnt[idx_y[i], idx_x[i] - 1]
                        + mnt[idx_y[i], idx_x[i] + 1]
                        - 4 * mnt[idx_y[i], idx_x[i]]) / dx ** 2
    return laplacian


def _laplacian_loop_numpy_2d_helbig(mnt: np.ndarray,
                                    idx_x: np.ndarray,
                                    idx_y: np.ndarray,
                                    dx: float
                                    ) -> np.ndarray:
    """Compute laplacian using Helbig formula on a 2D array using loop and numby (for numba jit optimization)."""
    laplacian = np.empty(idx_x.shape, np.float32)
    for j in range(idx_x.shape[0]):
        for i in range(idx_x.shape[1]):
            a = (mnt[idx_y[j, i] - 1, idx_x[j, i]]
                 + mnt[idx_y[j, i] + 1, idx_x[j, i]]
                 + mnt[idx_y[j, i], idx_x[j, i] - 1]
                 + mnt[idx_y[j, i], idx_x[j, i] + 1]
                 - 4 * mnt[idx_y[j, i], idx_x[j, i]]) / dx ** 2
            c = dx / 4
            laplacian[j, i] = a * c
    return laplacian


def _laplacian_loop_numpy_2d(mnt: np.ndarray,
                             idx_x: np.ndarray,
                             idx_y: np.ndarray,
                             dx: float
                             ) -> np.ndarray:
    """Compute laplacian using standard formula on a 2D array using loop and numby (for numba jit optimization)."""
    laplacian = np.empty(idx_x.shape, np.float32)
    for j in range(idx_x.shape[0]):
        for i in range(idx_x.shape[1]):
            a = (mnt[idx_y[j, i] - 1, idx_x[j, i]]
                 + mnt[idx_y[j, i] + 1, idx_x[j, i]]
                 + mnt[idx_y[j, i], idx_x[j, i] - 1]
                 + mnt[idx_y[j, i], idx_x[j, i] + 1]
                 - 4 * mnt[idx_y[j, i], idx_x[j, i]]) / dx ** 2
            laplacian[j, i] = a
    return laplacian


def _laplacian_numba_idx(mnt: np.ndarray,
                         idx_x: np.ndarray,
                         idx_y: np.ndarray,
                         dx: float,
                         helbig: bool = True
                         ) -> np.ndarray:
    """Compute laplacian on indexes using numba."""
    mnt = change_dtype_if_required(mnt, np.float32)
    idx_x = change_dtype_if_required(idx_x, np.int32)
    idx_y = change_dtype_if_required(idx_y, np.int32)

    if helbig:
        laplacian_1d = _laplacian_loop_numpy_1d_helbig
        laplacian_2d = _laplacian_loop_numpy_2d_helbig
    else:
        laplacian_1d = _laplacian_loop_numpy_1d
        laplacian_2d = _laplacian_loop_numpy_2d

    if idx_x.ndim <= 1:
        lapl_vect = jit([(float32[:, :], int32[:], int32[:], int64)], nopython=True)(laplacian_1d)

    if idx_x.ndim == 2:
        lapl_vect = jit([(float32[:, :], int32[:, :], int32[:, :], int64)], nopython=True)(laplacian_2d)

    return lapl_vect(mnt, idx_x, idx_y, dx)
