import numpy as np

from typing import Union, List

from helbigwindparam.utils_func import print_func_executed_decorator, timer_decorator
from helbigwindparam.config import config


def slope_mu_map(mnt: np.ndarray,
                 dx: float,
                 ) -> np.ndarray:
    """Compute square of mean squared slope onn a map"""
    if mnt.ndim > 2:
        return np.vectorize(_slope_mu_map, signature='(m,n),(),()->(m,n)')(mnt, dx)
    else:
        return _slope_mu_map(mnt, dx)


@print_func_executed_decorator("_slope_mu_map numpy",
                               level_begin="____",
                               level_end="____",
                               end="",
                               verbose=config["verbose"])
@timer_decorator("_slope_mu_map numpy", unit='minute', level="....", verbose=config["verbose"])
def _slope_mu_map(mnt: np.ndarray,
                  dx: float
                  ) -> np.ndarray:
    """mu = sum((slope**2))**(1/2) """
    mu = np.sqrt(np.sum(np.array(np.gradient(mnt, dx)) ** 2, axis=0) / 2)
    return mu


@print_func_executed_decorator("mu_idx", level_begin="__", level_end="__", end="", verbose=config["verbose"])
@timer_decorator("mu_idx", unit='minute', level="....", verbose=config["verbose"])
def slope_mu_idx(mnt: np.ndarray,
                 dx: float,
                 idx_x: Union[float, List, np.ndarray],
                 idx_y: Union[float, List, np.ndarray]
                 ) -> np.ndarray:
    """This function can not be directly written with numba"""
    if np.ndim(idx_x) == 0:
        idx_x = np.expand_dims(idx_x)
        idx_y = np.expand_dims(idx_y)
    mu = [slope_mu_map(mnt[y - 1:y + 2, x - 1:x + 2], dx)[1, 1] for (x, y) in zip(idx_x, idx_y)]
    return np.array(mu)


