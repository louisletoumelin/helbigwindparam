import numpy as np

from typing import Union, MutableSequence, Tuple, List

from helbigwindparam.utils_func import change_several_dtype_if_required


def detect_type_input(idx_x: Union[None, np.ndarray],
                      idx_y: Union[None, np.ndarray]
                      ) -> str:
    """Detect if inputs correspond to gridded inputs (map) or coordinates based inputs (idx)"""
    if idx_x is None and idx_y is None:
        type_input = "map"
    else:
        if idx_x.ndim > 1:
            type_input = "map"
        else:
            type_input = "idx"
    return type_input


def get_window_idx_boundaries(idx_x: Union[float, np.ndarray, List],
                              idx_y: Union[float, np.ndarray, List],
                              x_win: float = 69//2,
                              y_win: float = 79 // 2
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the indexes of boundaries box given their center and extent."""
    y_left = np.int32(idx_y - y_win)
    y_right = np.int32(idx_y + y_win)
    x_left = np.int32(idx_x - x_win)
    x_right = np.int32(idx_x + x_win)

    idx_x = np.int32(idx_x)
    idx_y = np.int32(idx_y)

    if idx_x.ndim > 1:
        flat_shape = idx_y.shape[0] * idx_y.shape[1]
        return y_left.reshape(flat_shape), y_right.reshape(flat_shape), x_left.reshape(flat_shape), x_right.reshape(
            flat_shape)

    return y_left, y_right, x_left, x_right


def control_idx_boundaries(idx: Union[float, MutableSequence[float]],
                           min_idx: Union[float, MutableSequence[float]] = 0,
                           max_idx: Union[float, MutableSequence[float], None] = None
                           ) -> Union[float, MutableSequence[np.ndarray], np.ndarray]:
    """Control that index of the boundaries of bounding boxes don't reach rejected values (e.g. negative values)."""
    if isinstance(idx, List) or isinstance(idx, np.ndarray):
        result = []
        for index, x in enumerate(idx):
            x = np.where(x < min_idx[index], 0, x)
            x = np.where(x > max_idx[index], max_idx[index], x)
            result.append(x)
        return result
    else:
        idx = np.where(idx < min_idx, 0, idx)
        idx = np.where(idx > max_idx, max_idx, idx)
        return idx


def get_and_control_idx_boundary(mnt: np.ndarray,
                                 idx_x: Union[np.ndarray, List, None],
                                 idx_y: Union[np.ndarray, List, None],
                                 x_win: float = 69//2,
                                 y_win: float = 79//2
                                 ) -> Union[float, Tuple[MutableSequence[np.ndarray], ...], Tuple[np.ndarray, ...]]:
    """Get indexes of boundary boxes and (eventually) modify it to force it to be in a range of accepted values."""
    shape = mnt.shape
    y_left, y_right, x_left, x_right = get_window_idx_boundaries(idx_x, idx_y, x_win=x_win, y_win=y_win)

    mnt, y_left, y_right, x_left, x_right = change_several_dtype_if_required(
        [mnt, y_left, y_right, x_left, x_right], [np.float32, np.int32, np.int32, np.int32, np.int32])
    boundaries_mnt = [shape[0], shape[0], shape[1], shape[1]]
    y_left, y_right, x_left, x_right = control_idx_boundaries([y_left, y_right, x_left, x_right],
                                                              min_idx=[0, 0, 0, 0],
                                                              max_idx=boundaries_mnt)
    return y_left, y_right, x_left, x_right


def idx_from_array_shape(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a grid of indexes given a 2-D array."""
    shape = array.shape
    idx_x = range(shape[1])
    idx_y = range(shape[0])
    idx_x, idx_y = np.array(np.meshgrid(idx_x, idx_y)).astype(np.int32)
    return idx_x, idx_y
