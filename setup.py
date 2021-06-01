from setuptools import setup
setup(
    name="CAMS_aerosol_props",
    version="0.1",
    description="A Python interface for loading CAMS netCDF data. Includes spatial and temporal interpolation and Aerosol optical properties calculation from model level data.",
    url="https://github.com/jonas-witthuhn/CAMS_aerosol_props",
    license="CC BY-NC",
    author="Jonas Witthuhn",
    author_email="witthuhn@tropos.de",
    packages=["CAMS_aerosol_props"],
    package_dir={"":"src"},
    install_requires=["numpy",
                      "xarray",
                      "netcdf4",
                      "scipy",
                      "trosat-base @ git+https://github.com/hdeneke/trosat-base.git#egg=trosat-base"],
        )
