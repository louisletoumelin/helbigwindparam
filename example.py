import xarray as xr
import matplotlib.pyplot as plt

from helbigwindparam import downscale, x_dsc, x_sgp, laplacian_map

"# todo add README which describes use and installation
# todo add tests
# todo add coverage
# todo check test passes
# todo add .gitignore file
# todo push on Github

path_to_file = "/home/letoumelinl/bias_correction/Data/1_Raw/DEM/DEM_ALPES_L93_30m.nc"
test = xr.open_dataset(path_to_file)
alti = test.alti.values[10_000:12_000, 10_000:12_000]


plt.figure()
plt.imshow(alti)
plt.colorbar()
plt.title("MNT")

map_sgp = x_sgp(alti, idx_x=None, idx_y=None, dx=30, l=2_000, x_win=69//2, y_win=79//2)
plt.figure()
plt.imshow(map_sgp)
plt.colorbar()
plt.title("x_sgp")

map_dsc = x_dsc(alti, dx=30)
plt.figure()
plt.imshow(map_dsc)
plt.colorbar()
plt.title("x_dsc")

map_laplacian = laplacian_map(alti, 30, helbig=True)
plt.figure()
plt.imshow(map_laplacian, vmin=-1, vmax=1)
plt.colorbar()
plt.title("laplacian")

map_dwnc = downscale(alti, dx=30, l=2_000, idx_x=None, idx_y=None, x_win=69//2, y_win=79//2)
plt.figure()
plt.imshow(map_dwnc)
plt.colorbar()
plt.title("Downscale")
