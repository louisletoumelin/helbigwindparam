from distutils.core import setup
import re


def get_version():
    version_file = "_version.py"
    verstrline = open(version_file, "rt").read()
    vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(vsre, verstrline, re.M)
    if mo:
        return mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (version_file,))


with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='helbigwindparam',
      version=get_version(),
      description='Unofficial implementation of wind parametrization from Helbig et al. 2017',
      long_description=long_description,
      author='Louis Le Toumelin',
      author_email='louis.letoumelin@gmail.com',
      packages=["helbigwindparam"],
      install_requires=["numpy"],
      license="MIT",
      extras_require={"tf": "tensorflow",
                      "nb": "numba",
                      "nexpr": "numexpr",
                      "plt": "matplotlib",
                      "xr": "xarray",
                      "nc": "netCDF4",
                      "pd": "pandas"}
      )
