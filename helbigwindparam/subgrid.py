import numpy as np

from typing import Union, List

from helbigwindparam.utils_func import print_func_executed_decorator, \
    timer_decorator
from helbigwindparam.dispatch import detect_type_input, \
    idx_from_array_shape
from helbigwindparam.std_average import std_moving_window_map, \
    std_moving_window_idx
from helbigwindparam.slope_average import mu_average

try:
    from helbigwindparam.config import config
except ModuleNotFoundError:
    print("config not imported")
    pass

try:
    import numexpr as ne

    _numexpr = True
except ModuleNotFoundError:
    _numexpr = False


@print_func_executed_decorator("xsi_numpy", level_begin="____", level_end="____", end="", verbose=config["verbose"])
@timer_decorator("xsi_numpy", unit='second', level="....", verbose=config["verbose"])
def xsi_numpy(std_avg: np.ndarray,
              mu_avg: np.ndarray
              ) -> np.ndarray:
    """Compute xsi parameter using numpy."""
    return np.sqrt(2) * std_avg / mu_avg


@print_func_executed_decorator("xsi_numexpr", level_begin="____", level_end="____", end="", verbose=config["verbose"])
@timer_decorator("xsi_numexpr", unit='second', level="....", verbose=config["verbose"])
def xsi_numexpr(std_avg: np.ndarray,
                mu_avg: np.ndarray
                ) -> np.ndarray:
    """Compute xsi parameter using numexpr."""
    return ne.evaluate("sqrt(2) * std_avg / mu_avg")


@print_func_executed_decorator("xsi", level_begin="__", level_end="__", end="", verbose=config["verbose"])
@timer_decorator("xsi", unit='minute', level="....", verbose=config["verbose"])
def xsi(mnt: np.ndarray,
        mu_avg: np.ndarray,
        idx_x: Union[List, np.ndarray, None] = None,
        idx_y: Union[List, np.ndarray, None] = None,
        x_win: float = 69 // 2,
        y_win: float = 79 // 2,
        library_std_avg: Union[str, None] = None,
        verbose: Union[str, None] = None
        ) -> np.ndarray:
    """Compute xsi parameter."""

    library_std_avg = config["library_std_avg"] if library_std_avg is None else library_std_avg
    verbose = config["verbose"] if verbose is None else verbose

    type_input = detect_type_input(idx_x, idx_y)
    if type_input == "map":
        std_avg = std_moving_window_map(mnt,
                                        x_win=x_win,
                                        y_win=y_win,
                                        idx_x=idx_x,
                                        idx_y=idx_y,
                                        library_std_avg=library_std_avg,
                                        verbose=verbose)
    elif type_input == "idx":
        std_avg = std_moving_window_idx(mnt,
                                        x_win=x_win,
                                        y_win=y_win,
                                        idx_x=idx_x,
                                        idx_y=idx_y)

    if config["library_xsi"] == "numpy":
        return xsi_numpy(std_avg, mu_avg)
    elif config["library_xsi"] == "numexpr" and _numexpr:
        return xsi_numexpr(std_avg, mu_avg)


@print_func_executed_decorator("x_sgp_numpy", level_begin="__", level_end="__", end="", verbose=config["verbose"])
@timer_decorator("x_sgp_numpy", unit='second', level="....", verbose=config["verbose"])
def x_sgp_numpy(mu: np.ndarray,
                xsi_: np.ndarray,
                l: float,
                a: float,
                b: float,
                c: float,
                d: float
                ) -> np.ndarray:
    """Compute x_sgp (subgrid parameter) using numpy."""
    return 1 - (1 - (1 / (1 + a * mu ** b)) ** c) * np.exp(-d * (l / xsi_) ** (-2))


@print_func_executed_decorator("x_sgp_numexpr", level_begin="__", level_end="__", end="", verbose=config["verbose"])
@timer_decorator("x_sgp_numexpr", unit='second', level="....", verbose=config["verbose"])
def x_sgp_numexpr(mu: np.ndarray,
                  xsi_: np.ndarray,
                  l: float,
                  a: float,
                  b: float,
                  c: float,
                  d: float
                  ) -> np.ndarray:
    """Compute x_sgp (subgrid parameter) using numexpr."""
    return ne.evaluate("1 - (1 - (1 / (1 + a * mu ** b)) ** c) * exp(-d * (l / xsi_) ** (-2))")


@print_func_executed_decorator("Subgrid", level_begin="_", level_end="_", end="", verbose=config["verbose"])
def x_sgp(mnt: np.ndarray,
          idx_x: Union[List, np.ndarray, None] = None,
          idx_y: Union[List, np.ndarray, None] = None,
          dx: float = 30,
          l: float = 2_000,
          x_win: float = 69//2,
          y_win: float = 79//2,
          a: Union[float, None] = None,
          b: Union[float, None] = None,
          c: Union[float, None] = None,
          d: Union[float, None] = None,
          library_x_sgp: Union[str, None] = None
          ) -> np.ndarray:
    """Compute x_sgp (subgrid parameter)."""

    a = config["a_sgp"] if a is None else a
    b = config["b_sgp"] if b is None else b
    c = config["c_sgp"] if c is None else c
    d = config["d_sgp"] if d is None else d
    library_x_sgp = config["library_x_sgp"] if library_x_sgp is None else library_x_sgp

    if idx_x is None and idx_y is None:
        idx_x, idx_y = idx_from_array_shape(mnt)

    mu = mu_average(mnt, dx, idx_x, idx_y, x_win=x_win, y_win=y_win)
    xsi_ = xsi(mnt, mu, idx_x, idx_y, x_win=x_win, y_win=y_win)

    if library_x_sgp == "numpy":
        return x_sgp_numpy(mu, xsi_, l, a, b, c, d)
    elif library_x_sgp == "numexpr":
        return x_sgp_numexpr(mu, xsi_, l, a, b, c, d)
