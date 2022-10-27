Ongoing project.

## Use
See example in ``example.py``.
```python
import xarray as xr
import matplotlib.pyplot as plt

from helbigwindparam import downscale

# Open file
path_to_file = "/home/letoumelinl/bias_correction/Data/1_Raw/DEM/DEM_ALPES_L93_30m.nc"
alti = xr.open_dataset(path_to_file).alti.values

# Compute downscaling factor
map_dwnc = downscale(alti, dx=30, l=2_000, x_win=69//2, y_win=79//2)

# Plot downscaling pfactor
plt.figure()
plt.imshow(map_dwnc)
plt.colorbar()
plt.title("Map of the downscaling factor")
```
## Installation
In a terminal, use the following commands:

##### Optional: creates and activate a virtual environment (ex: virtualenv, conda)
```shell
python -m venv helbigwind_env
```

```shell
source helbigwind_env/bin/activate
```

##### Clone repository
```shell
git clone https://github.com/louisletoumelin/helbigwindparam.git
```


##### Install requirements
```shell
pip install -r helbigwindparam/requirements.txt
```
##### Install helbigwindparam
```shell
pip install helbigwindparam
```
