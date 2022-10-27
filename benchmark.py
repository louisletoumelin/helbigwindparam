import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import timeit

from helbigwindparam import downscale, laplacian_map, xsi
from helbigwindparam.slope_average import mu_average
from helbigwindparam.subgrid import std_moving_window_map, xsi_numpy, xsi_numexpr, x_sgp_numpy, x_sgp_numexpr
from helbigwindparam.slope import slope_mu_map
from helbigwindparam.downscale import x_dsc_numpy, x_dsc_numexpr
path_to_file = "/home/letoumelinl/bias_correction/Data/1_Raw/DEM/DEM_ALPES_L93_30m.nc"
test = xr.open_dataset(path_to_file)
launch_computation = False

"""
Benchmark

24/10/2022

30_000 x 30_000 px (30 m grid spacing)

No crash observed.

Tensorflow: 13.01s
Numba (run 0): 39.32s
Numpy: 39.84s
Numba (run 1): 23.12s
"""

if launch_computation:
    alti = test.alti.values[:30_000, :30_000]
    laplacian_map(alti, 30, helbig=True, library_laplacian="tensorflow")
    laplacian_map(alti, 30, helbig=True, library_laplacian="numba")
    laplacian_map(alti, 30, helbig=True, library_laplacian="numpy")
    laplacian_map(alti, 30, helbig=True, library_laplacian="numba")

"""
Benchmark

24/10/2022

3500 x 3500 px (30 m grid spacing)

Code crashes for topographies larger than 3500px by 3500px

Tensorflow: 8.53s
Numba (run 0): 192.53s
Numpy (list comprehension): 164.14s
Numba (run 1): 193s
"""

if launch_computation:
    alti = test.alti.values[10_000:13_500, 10_000:13_500]
    mu_avg_tf = mu_average(alti, 30, x_win=69//2, y_win=79//2, library_mu_avg="tensorflow")
    _ = mu_average(alti, 30, x_win=69//2, y_win=79//2, library_mu_avg="numba")
    _ = mu_average(alti, 30, x_win=69//2, y_win=79//2, library_mu_avg="numpy")
    _ = mu_average(alti, 30, x_win=69//2, y_win=79//2, library_mu_avg="numba")

"""
Benchmark

24/10/2022

3500 x 3500 px (30 m grid spacing)

Code crashes for topographies larger than 3500px by 3500px

Tensorflow: 21.11s
Numba (run 0): 354.8s
Numpy (list comprehension): 352.45s
Numba (run 1): 346.72s
"""

if launch_computation:
    alti = test.alti.values[10_000:13_500, 10_000:13_500]
    std_avg_tf = std_moving_window_map(alti, x_win=69 // 2, y_win=79 // 2, library_std_avg="tensorflow")
    std_avg_nb = std_moving_window_map(alti, x_win=69 // 2, y_win=79 // 2, library_std_avg="numba")
    _ = std_moving_window_map(alti, x_win=69 // 2, y_win=79 // 2, library_std_avg="numpy")
    _ = std_moving_window_map(alti, x_win=69 // 2, y_win=79 // 2, library_std_avg="numba")

    plt.figure()
    plt.imshow(std_avg_tf)

    plt.figure()
    plt.imshow(std_avg_nb)

"""
Benchmark

24/10/2022

3500 x 3500 px (30 m grid spacing)

There seems to be a scale effect: large arrays are faster with numexpr than with numpy.
Numpy seems to be better for small arrays.

Numpy: 1.30s (100 iterations)
Numexpr: 2.40s (100 iterations)

For an array of shape 40_000px x 40_000px
Numpy: 6.7s
Numexpr: 2.79s
"""

if launch_computation:

    alti = test.alti.values[10_000:13_500, 10_000:13_500]
    mu_avg_tf = mu_average(alti, 30, x_win=69 // 2, y_win=79 // 2, library_mu_avg="tensorflow")
    std_avg_tf = std_moving_window_map(alti, x_win=69 // 2, y_win=79 // 2, library_std_avg="tensorflow")

    timeit.timeit('xsi_numpy(std_avg_tf, mu_avg_tf)', number=100, globals=globals())
    timeit.timeit('xsi_numexpr(std_avg_tf, mu_avg_tf)', number=100, globals=globals())

"""
Benchmark

24/10/2022

3500 x 3500 px (30 m grid spacing)

There seems to be a scale effect: large arrays are faster with numexpr than with numpy.
Numpy seems to be better for small arrays.

Numpy: 1.30s (100 iterations)
Numexpr: 2.40s (100 iterations)

For an array of shape 40_000px x 40_000px
Numpy: 6.7s
Numexpr: 2.79s
"""

if launch_computation:

    alti = test.alti.values[10_000:13_500, 10_000:13_500]
    mu_avg_tf = mu_average(alti, 30, x_win=69 // 2, y_win=79 // 2, library_mu_avg="tensorflow")
    std_avg_tf = std_moving_window_map(alti, x_win=69 // 2, y_win=79 // 2, library_std_avg="tensorflow")

    timeit.timeit('xsi_numpy(std_avg_tf, mu_avg_tf)', number=100, globals=globals())
    timeit.timeit('xsi_numexpr(std_avg_tf, mu_avg_tf)', number=100, globals=globals())

    test = np.ones((40_000, 40_000))
    xsi_numpy(test, test)
    xsi_numexpr(test, test)


"""
Benchmark

24/10/2022

3500 x 3500 px (30 m grid spacing)

There seems to be a scale effect: large arrays are faster with numexpr than with numpy.
Numpy seems to be better for small arrays.

Numpy: 0.67s
Numexpr: 0.14s
"""

if launch_computation:

    alti = test.alti.values[10_000:13_500, 10_000:13_500]
    mu_avg_tf = mu_average(alti, 30, x_win=69 // 2, y_win=79 // 2, library_mu_avg="tensorflow")
    xsi_ = xsi(alti, mu_avg_tf, x_win=69//2, y_win=79//2, library_std_avg="tensorflow")
    a = 3.354688
    b = 1.998767
    c = 0.20286
    d = 5.951
    l = 2_000

    x_sgp_numpy(mu_avg_tf, xsi_, l, a, b, c, d)
    x_sgp_numexpr(mu_avg_tf, xsi_, l, a, b, c, d)

"""
Benchmark

24/10/2022

Domain: full DEM

Numpy: 20.52s
Numexpr: 5.05s
"""

if launch_computation:
    alti = test.alti.values
    laplacian = laplacian_map(alti, 30, helbig=True)
    mu = slope_mu_map(alti, 30)
    a = 17.0393
    b = 0.737
    c = 1.0234
    d = 0.3794
    e = 1.9821

    x_dsc_numpy(laplacian, mu, a, b, c, d, e)
    x_dsc_numexpr(laplacian, mu, a, b, c, d, e)

"""
Benchmark

24/10/2022

Domain: 3500 x 3500px

Optimized configuration: 31.04s
config["library_dsc"] = "numexpr"  # "numpy" or "numexpr"
config["library_std_avg"] = "tensorflow"  # "tensorflow", "numpy", "numba"
config["library_mu_avg"] = "tensorflow"  # "tensorflow", "numpy", "numba"
config["library_xsi"] = "numexpr"  # "numpy", "numexpr"
config["library_laplacian"] = "tensorflow"  # "numba", "numpy"
config["library_x_sgp"] = "numexpr"  # "numpy", "numexpr"

Numpy configuration: 524.15s
config["library_dsc"] = "numpy"  # "numpy" or "numexpr"
config["library_std_avg"] = "numpy"  # "tensorflow", "numpy", "numba"
config["library_mu_avg"] = "numpy"  # "tensorflow", "numpy", "numba"
config["library_xsi"] = "numpy"  # "numpy", "numexpr"
config["library_laplacian"] = "numpy"  # "numba", "numpy"
config["library_x_sgp"] = "numpy"  # "numpy", "numexpr"
"""

if launch_computation:

    alti = test.alti.values[10_000:13_500, 10_000:13_500]
    downscale(alti, dx=30, l=2_000, x_win=69//2, y_win=79//2)
