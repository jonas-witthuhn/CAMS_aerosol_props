# CAMS_aerosol_props

Load CAMS RA netcdf files and calculate spectral aerosol properties. (AOD, EXT, SSA, G)

## install
`pip install git+https://github.com/jonas-witthuhn/CAMS_aerosol_props.git#egg=CAMS_aerosol_props`

## usage
load up the cams netcdf files:
```
from CAMS_aerosol_props.load_cams_data import CAMS
mlfile = "ecmwf/data/nc/cams-ra_2015-01-01_ml.nc"
sfcfile = "ecmwf/data/nc/cams-ra_2015-01-01_sfc.nc"
c = CAMS(sfcfile,mlfile)

print(c._init__.__doc__)
```


>        This class loads up the cams surface (sfc) and model level (ml) files.
>         Required parameters in files:
>             sfc:
>                 time,lat,lon - time, latitude, longitude
>                 psfc - surface pressure [Pa] ID: 134.128
>                 geop - geopotential [m2 s-2] ID: 129.128
>                 tsfc - skin temperature [k]  ID: 235.128
>             ml:
>                 time,lat,lon -time, latitude, longitude (same as sfc)
>                 hyam,hybm,hyai,hybi - factors for model level to pressure
>                 q - specific humidity [kg kg-1] ID: 133
>                 t - temperature [K] ID: 130
>                 aermr01 - mixing ratio Sea salt aerosol (0.03 - 0.5um)    ID:210001
>                 aermr02 - mixing ratio Sea salt aerosol (0.55 - 0.9um)    ID:210002
>                 aermr03 - mixing ratio Sea salt aerosol (0.9 - 20um)      ID:210003
>                 aermr04 - mixing ratio Dust Aerosol (0.03-0.55um)         ID:210004
>                 aermr05 - mixing ratio Dust Aerosol (0.55 - 0.9um)        ID:210005
>                 aermr06 - mixing ratio Dust Aerosol (0.9 - 20um)          ID:210006
>                 aermr07 - mixing ratio hydrophilic organic matter aerosol ID:210007
>                 aermr08 - mixing ratio hydrophobic organic matter aerosol ID:210008
>                 aermr09 - mixing ratio hydrophilic black carbon aerosol   ID:210009
>                 aermr10 - mixing ratio hydrophobic black carbon aerosol   ID:210010
>                 aermr11 - mixing ratio Sulphate aerosol                   ID:210011
> 
>         Parameters
>         ----------
>         cams_sfc_file : str
>             path and filename to CAMS RA surface netcdf file
>         cams_ml_file : str
>             path and filename to CAMS RA model level netcdf file


## Atmosphere props:
You can lookup e.g. geopotential (Q), geometrical heigth (z), relative (rh) and specific (q) humidity, virtual temperature (Tv), pressure (P), temperature (T). At layer midpoints (mlvl) and layer interfaces (ilvl) and surface (sfc).

For example:
```
z_mlvl = c.z_mlvl # geometrical heigth at layer midpoints
z_ilvl = c.z_ilvl # geometrical heigth at layer interface

```



## Aerosol props:
Also you can calculate aerosol properties at desired wavelengths:

```
AP_sfc,AP_ml = c.aerosol_optprop([469.,550,670,865,1240])
print(c.aerosol_optprop.__doc__)
print(AP_sfc.info())
print(AP_ml.info())
```

>         calculate spectral optical properties of CAMS aerosol.
>         aod - spectral aerosol optical depth
>         ext - spectral extinction coefficient
>         ssa - spectral single scattering albedo
>         g   - spectral asymmetry parameter
> 
>         Parameters
>         ----------
>         wvl : float, array(n)
>             spectral wavelength [nm]
>         delta_eddington. bool, (optional)
>             switch on delta eddington scaling of the phase function if True.
>             The default value is: False.
>             
>         Returns
>         -------
>         ds_sfc : xarray.Dataset
>             This dataset includes the column integrated spectral optical
>             properties of aerosol at [wvl]
>         ds_ml : xarray.Dataset
>             This dataset includes the spectral optical properties per column
>             at [wvl]      
>         
>         ### AP_sfc
>         xarray.Dataset {
>         dimensions:
>         column = 1568 ;
>         wvl = 5 ;
> 
>         variables:
>         float64 g(column, wvl) ;
>         float64 ssa(column, wvl) ;
>         float64 aod(column, wvl) ;
>         float64 ext(column, wvl) ;
>         datetime64[ns] time(column) ;
>         float64 lat(column) ;
>         float64 lon(column) ;
>         float64 wvl(wvl) ;
> 
>         // global attributes:
>         }
>         ### AP_ml
>         xarray.Dataset {
>         dimensions:
>         column = 1568 ;
>         mlvl = 60 ;
>         wvl = 5 ;
> 
>         variables:
>         float64 g(column, mlvl, wvl) ;
>         float64 ssa(column, mlvl, wvl) ;
>         float64 aod(column, mlvl, wvl) ;
>         float64 ext(column, mlvl, wvl) ;
>         datetime64[ns] time(column) ;
>         float64 lat(column) ;
>         float64 lon(column) ;
>         float64 wvl(wvl) ;
> 
>         // global attributes:
>         }None
> 
