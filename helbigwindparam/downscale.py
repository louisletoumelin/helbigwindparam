import numpy as np

from typing import Union

from helbigwindparam.utils_func import print_func_executed_decorator, change_dtype_if_required_decorator, timer_decorator
from helbigwindparam.dispatch import detect_type_input
from helbigwindparam.slope import slope_mu_map, slope_mu_idx
from helbigwindparam.laplacian import laplacian_map, laplacian_idx
from helbigwindparam.subgrid import x_sgp

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


@print_func_executed_decorator("x_dsc_numpy", level_begin="__", level_end="__", end="", verbose=config["verbose"])
@timer_decorator("x_dsc_numpy", unit='second', level="....", verbose=config["verbose"])
def x_dsc_numpy(laplacian: Union[np.ndarray, float],
                mu: Union[np.ndarray, float],
                a: float,
                b: float,
                c: float,
                d: float,
                e: float
                ) -> Union[np.ndarray, float]:
    """Compute x_dsc using numpy."""
    return (1 - a * laplacian / (1 + a * np.abs(laplacian) ** b)) * (c / (1 + d * mu ** e))


@print_func_executed_decorator("x_dsc_numexpr", level_begin="__", level_end="__", end="", verbose=config["verbose"])
@timer_decorator("x_dsc_numexpr", unit='second', level="....", verbose=config["verbose"])
def x_dsc_numexpr(laplacian: Union[np.ndarray, float],
                  mu: Union[np.ndarray, float],
                  a: float,
                  b: float,
                  c: float,
                  d: float,
                  e: float
                  ):
    """Compute x_dsc using numexpr."""
    return ne.evaluate("(1 - a * laplacian / (1 + a * abs(laplacian) ** b)) * (c / (1 + d * mu ** e))")


@print_func_executed_decorator("downscaling",
                               level_begin="_",
                               level_end="_",
                               end="",
                               verbose=config["verbose"])
@change_dtype_if_required_decorator(np.float32)
def x_dsc(mnt: Union[np.ndarray, float],
          dx: float = 25,
          idx_x: Union[np.ndarray, float, None] = None,
          idx_y: Union[np.ndarray, float, None] = None,
          library_dsc: Union[str, None] = None,
          a: Union[float, None] = None,
          b: Union[float, None] = None,
          c: Union[float, None] = None,
          d: Union[float, None] = None,
          e: Union[float, None] = None,
          verbose: Union[bool, None] = None
          ) -> Union[float, np.ndarray]:
    """Compute x_dsc."""

    # Get value from the config or arguments
    a = config["a_dsc"] if a is None else a
    b = config["b_dsc"] if b is None else b
    c = config["c_dsc"] if c is None else c
    d = config["d_dsc"] if d is None else d
    e = config["e_dsc"] if e is None else e
    library_dsc = config["library_dsc"] if library_dsc is None else library_dsc
    verbose = config["verbose"] if verbose is None else verbose

    type_input = detect_type_input(idx_x, idx_y)

    if type_input == "map":

        laplacian = laplacian_map(mnt, dx, helbig=True)
        mu = slope_mu_map(mnt, dx)

    elif type_input == "indexes":

        if idx_x.ndim == 0:
            idx_x = np.expand_dims(idx_x, axis=0)
        if idx_y.ndim == 0:
            idx_y = np.expand_dims(idx_y, axis=0)

        laplacian = laplacian_idx(mnt, idx_x, idx_y, dx, helbig=True)
        mu = slope_mu_idx(mnt, dx, idx_x, idx_y)

    if library_dsc == "numpy" or library_dsc == "numba":
        if library_dsc == "numba":
            print("numba not available for x_dsc, use numpy instead")
        return x_dsc_numpy(laplacian, mu, a, b, c, d, e)
    elif library_dsc == "numexpr":
        return x_dsc_numexpr(laplacian, mu, a, b, c, d, e)


@timer_decorator("downscale + subgrid", unit='second', level="", verbose=config["verbose"])
@print_func_executed_decorator("Downscaling + subgrid from Helbig et al. 2017",
                               level_begin="\n",
                               level_end="",
                               end="",
                               verbose=config["verbose"])
def downscale(mnt: np.ndarray,
              dx: float = 30,
              l: float = 2_000,
              idx_x: Union[np.ndarray, float, None] = None,
              idx_y: Union[np.ndarray, float, None] = None,
              x_win: float = 69//2,
              y_win: float = 79//2
              ) -> np.ndarray:
    """Downscaling method from Helbig et al. 2017 (subgrid + downscale)"""
    # todo check if there is problems at the borders of the mnt or if the mnt size is reduced
    return x_sgp(mnt, idx_x, idx_y, dx, l, x_win, y_win) * x_dsc(mnt, dx, idx_x, idx_y)
