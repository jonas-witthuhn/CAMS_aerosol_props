#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:14:44 2020

@author: walther
"""
import os

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d

# Hartwigs sunpos routine
import sunpos as sp

class CAMS:
    def __init__(self, cams_sfc_file, cams_ml_file, opt_prop_file = None, scale_altitude = None, scale_sfcpressure=None):
        """
        This class loads up the cams surface (sfc) and model level (ml) files.
        Required parameters in files:
            sfc:
                time,lat,lon - time, latitude, longitude
                psfc - surface pressure [Pa] ID: 134.128
                geop - geopotential [m2 s-2] ID: 129.128
                tsfc - skin temperature [k]  ID: 235.128
            ml:
                time,lat,lon -time, latitude, longitude (same as sfc)
                hyam,hybm,hyai,hybi - factors for model level to pressure
                q - specific humidity [kg kg-1] ID: 133
                t - temperature [K] ID: 130
                aermr01 - mixing ratio Sea salt aerosol (0.03 - 0.5um)    ID:210001
                aermr02 - mixing ratio Sea salt aerosol (0.55 - 0.9um)    ID:210002
                aermr03 - mixing ratio Sea salt aerosol (0.9 - 20um)      ID:210003
                aermr04 - mixing ratio Dust Aerosol (0.03-0.55um)         ID:210004
                aermr05 - mixing ratio Dust Aerosol (0.55 - 0.9um)        ID:210005
                aermr06 - mixing ratio Dust Aerosol (0.9 - 20um)          ID:210006
                aermr07 - mixing ratio hydrophilic organic matter aerosol ID:210007
                aermr08 - mixing ratio hydrophobic organic matter aerosol ID:210008
                aermr09 - mixing ratio hydrophilic black carbon aerosol   ID:210009
                aermr10 - mixing ratio hydrophobic black carbon aerosol   ID:210010
                aermr11 - mixing ratio Sulphate aerosol                   ID:210011

        Parameters
        ----------
        cams_sfc_file : str
            path and filename to CAMS RA surface netcdf file
        cams_ml_file : str
            path and filename to CAMS RA model level netcdf file
        """

        ### aerosol prop config
        if opt_prop_file == None:
            pfx = os.path.split(os.path.realpath(__file__))[0]
            fname = os.path.join(pfx, "aerosol_cams_ifs_optics.nc")
            self.AERCFG = xr.open_dataset(fname)
        else:
            self.AERCFG = xr.open_dataset(opt_prop_file)

        ### cams files
        cams_ml = xr.open_dataset(cams_ml_file)
        cams_sfc = xr.open_dataset(cams_sfc_file)
        self.cams_ml = cams_ml
        self.cams_sfc = cams_sfc

        # coordinates
        times, lats, lons = np.meshgrid(cams_ml.time, cams_ml.lat, cams_ml.lon,
                                        indexing='ij')
        self.times = times.flatten()
        self.lats = lats.flatten()
        self.lons = lons.flatten()

        ## calculate cosine of zenith angle
        jd = sp.datetime2julday(self.times)
        self.sza, self.azi = sp.zenith_azimuth(jd, self.lats, self.lons)
        self.mu0 = np.cos(np.deg2rad(self.sza))

        # scale to sfc pressure if needed
        # scaling is done only if both scale_altitude and scale_sfcpressure is known
        # _scale at the end of __init__ will calculate the missing and run __init__ again
        if type(scale_altitude) != type(None) and type(scale_sfcpressure) != type(None):
            self.P_sfc = scale_sfcpressure.flatten()
        else:
            ## calculate pressure [Pa] at level interfaces and midpoints
            self.P_sfc = cams_sfc.psfc.data.flatten() # surface pressure
        # pressure at half-level -> shape(col, half_level)
        self.P_ilvl = self.calc_lvl_pressure(cams_ml.hyai.data,
                                             cams_ml.hybi.data)
        self.P_mlvl = self.calc_lvl_pressure(cams_ml.hyam.data,
                                             cams_ml.hybm.data)

        # geopotential at surface scaled to altitude if needed
        # scaling is done only if both scale_altitude and scale_sfcpressure is known
        # _scale at the end of __init__ will calculate the missing and run __init__ again
        if type(scale_altitude) != type(None) and type(scale_sfcpressure) != type(None):
            # scale to altitude
            self.Q_sfc = scale_altitude.flatten()*9.80665
        else:
            self.Q_sfc = cams_sfc.geop.values.flatten()

        # temperature [K]
        self.T_mlvl = flatten_coords(cams_ml.t.data, 1)
        self.T_sfc = cams_sfc.tsfc.values.flatten()

        # specific humidity [kg/kg]
        self.q = flatten_coords(cams_ml.q.values, idx_keep=1)

        ### derive furhter quantities
        # relative humidity [0-1]
        self.calc_rh()
        # virtual temperature [K]
        self.calc_Tv()
        # geopotential at layer interfaces and layer mid points [m2/s2]
        self.calc_Q()
        # geometric hights [m]
        self.z_mlvl = self.Q_mlvl / 9.80665
        self.z_ilvl = self.Q_ilvl / 9.80665

        self._scale(cams_sfc_file, cams_ml_file, opt_prop_file, scale_altitude,scale_sfcpressure)

    def _scale(self,cams_sfc_file, cams_ml_file, opt_prop_file,scale_altitude,scale_sfcpressure):
        # if we dont know the sfc pressure but scale to altitude
        # we have to find the appropriate sfc pressure by interpolating
        # interface pressure to geometric altitude
        if (type(scale_altitude) != type(None)) and (type(scale_sfcpressure) == type(None)):
            altitude = scale_altitude.flatten()
            # interpolate sfcpressure from altitude difference 
            psfc = []
            # iterate over all columns
            for i in range(len(self.times)):
                f = interp1d(self.z_ilvl[i,1:],self.P_ilvl[i,1:],kind='linear',fill_value='extrapolate')
                psfc.append(f(altitude[i]))
            psfc=np.array(psfc).reshape(self.cams_sfc.psfc.data.shape)
            # run init again, now with known sfc pressure
            self.__init__(cams_sfc_file, cams_ml_file, opt_prop_file = opt_prop_file, scale_altitude = scale_altitude, scale_sfcpressure=psfc)
        # if we dont know the altitude but want to scale sfc pressure,
        # we have to interpolate the altitude from the geometric hights
        elif (type(scale_altitude) == type(None)) and (type(scale_sfcpressure) != type(None)):
            sfcpressure = scale_sfcpressure.flatten()
            # interpolate altitude from pressure levels
            alts = []
            # iterate of all columns
            for i in range(len(self.times)):
                f = interp1d(self.P_ilvl[i,1:],self.z_ilvl[i,1:],kind='linear',fill_value='extrapolate')
                alts.append(f(sfc_pressure[i]))
            alts = np.array(alts).reshape(self.cams_sfc.psfc.data.shape)
            # run init again, now with known altitude
            self.__init__(cams_sfc_file, cams_ml_file, opt_prop_file = opt_prop_file, scale_altitude = alts, scale_sfcpressure=scale_sfcpressure)


    def _area_density(self):
        ## calculate area density f
        # factor converting  mmr [kg/kg] to path mixing ratio [kg/m2]
        # dp = rho *g *dz
        # tau = mext * rho * dz
        #     = mext * f
        # f = dp / g
        return np.diff(self.P_ilvl)/9.80665

    def calc_lvl_pressure(self, A, B):
        """calculate model level pressure (interface or half level) from defined
        constants A and B from grib metadata. Pk = Ak [Pa] + p0 [Pa]*Bk"""
        p0 = self.P_sfc
        if np.isscalar(p0):
            P = A + p0*B
        else:
            P = A + p0[:, np.newaxis]*B
        return P

    def calc_rh(self):
        """
        Calculate relative humidity from specific humidity (q) [kg/kg]
        """
        P = self.P_mlvl
        T = self.T_mlvl
        #saturation pressure
        e_sat = 6.11e2 * np.exp(17.269 * (T-273.16) / (T-35.86))
        h2o_sat_liq = 0.622 * e_sat / P
        h2o_sat_liq[h2o_sat_liq > 1] = 1.
        self.rh = self.q / h2o_sat_liq
        return 0

    def calc_Q(self):
        """calculate geopotential Q at layer interfaces.
        Q_sfc - geopotential at surface
        P_ilvl - pressure at layer interfaces
        T_mlvl - temperature at layer midpoints
        q_mlvl - relative humidity at layer midpoints
        ilvl[0]  = TOA, ilvl[-1] = sfc
        """

        Tv = self.Tv
        Qi = np.zeros(self.P_ilvl.shape) # at layer interfaces
        Qm = np.zeros(self.T_mlvl.shape) # at layer midpoints
        P = self.P_ilvl

        Rd = CONSTANTS.Rdry

        # initialyze with surface value
        Qi[:, -1] = self.Q_sfc.copy()
        # calculate layer for layer
        N = len(Qi[0, :])
        ilow = -1
        iup = -2
        imid = -1
        while N+iup >= 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                logP = np.log(P[:, ilow]/P[:, iup])
                Qi[:, iup] = Qi[:, ilow] + Rd*Tv[:, imid]*logP
                dP = P[:, ilow]-P[:, iup]
                a = 1. - (P[:, iup]/dP)*logP
            if N+iup == 0:
                a = np.log(2)
            Qm[:, imid] = Qi[:, ilow] + a*Rd*Tv[:, imid]
            ilow -= 1
            iup -= 1
            imid -= 1
        self.Q_ilvl = Qi
        self.Q_mlvl = Qm
        return 0

    def calc_Tv(self):
        RR = CONSTANTS.Rwv / CONSTANTS.Rdry
        # virtual temperature
        self.Tv = self.T_mlvl*(1. + (RR-1)*self.q)
        return 0

    def _scale_delta_eddington(self, od, ssa, g):
        """Perform in-place delta-Eddington scaling of the phase function
        """
        f = g*g
        od = od*(1.-ssa*f)
        ssa = ssa*(1.-f)/(1.-ssa*f)
        g = g/(1.+g)
        return od, ssa, g

    def get_aerosol_props(self, aerosol_type, wavel, mono):
        """
        Calculate mass extinction, single scattering albedo and asymmetry parameter

        Parameters
        ----------
        aerosol_type: str
            aerosol type name
                SS1 -> Sea salt 0.03-0.5        OPAC
                SS2 -> Sea salt 0.5-5           OPAC
                SS3 -> Sea salt 5-20            OPAC
                DU1 -> Dust 0.03-0.55           Dubovic et al 2002
                DU2 -> Dust 0.55-0.9            Woodward et al 2001
                DU3 -> Dust 0.9-20              Fouquart et al 1987
                OM2 -> Organic Matter hydrophilic    OPAC-Mixture
                Om1 -> Organic Matter hydrophobic    OPAC Mixture at 20% humidity
                BC1 -> Black Carbon hydrophilic      OPAC (SOOT) (same as BC2 in CAMS)
                BC2 -> Black Carbon hydrophobic      OPAC (SOOT)
                SU  -> Sulfates                      Lacis et al (GACP)
        wavel: array(n), float
            wavelength [nm]
        mono: bool,
            If True use single wavelenghts for interpolation (340nm - 2130nm).
            If False use spectral channel (ECRAD) integrated optical properties
            to interpolate the optical props. (232nm - 25000nm)

        Returns
        -------
        mext: array(k0,..kn,n)
            mass extiction coefficient [m2/kg]
        ssa: array(k0,..kn,n)
            single scattering albedo [-]
        g: array(k0,..kn,n)
            asymmetry parameter[-]
        """
        def get_hydrophob(i, rh, wavel, mono):
            AERCFG = self.AERCFG
            if mono:
                cwvl = 1e9*AERCFG.wavelength_mono.values #[nm]
            else:
                channels1 = np.concatenate((1./AERCFG.wavenumber1_lw[:-1],
                                            1./AERCFG.wavenumber1_sw[:-1]),axis=0)
                channels2 = np.concatenate((1./AERCFG.wavenumber2_lw[:-1],
                                            1./AERCFG.wavenumber2_sw[:-1]),axis=0)
                cwvl=np.mean(np.vstack((channels1,channels2)),axis=0)
                cwvl*= 1e7 #[nm]

            if mono:
                Y = AERCFG.mass_ext_mono_hydrophobic.values[i, :]
            else:
                Y = np.concatenate((AERCFG.mass_ext_lw_hydrophobic.values[i,:-1],
                                    AERCFG.mass_ext_sw_hydrophobic.values[i,:-1]),
                                   axis=0)
            f = interp1d(cwvl, Y,
                         kind='cubic',
                         # kind='nearest',
                         fill_value='extrapolate')
            mext = f(wavel)

            if mono:
                Y = AERCFG.ssa_mono_hydrophobic.values[i, :]
            else:
                Y = np.concatenate((AERCFG.ssa_lw_hydrophobic.values[i,:-1],
                                    AERCFG.ssa_sw_hydrophobic.values[i,:-1]),
                                   axis=0)
            f = interp1d(cwvl, Y,
                         kind='cubic',
                         # kind='nearest',
                         fill_value='extrapolate')
            ssa = f(wavel)

            if mono:
                Y = AERCFG.asymmetry_mono_hydrophobic.values[i, :]
            else:
                Y = np.concatenate((AERCFG.asymmetry_lw_hydrophobic.values[i,:-1],
                                    AERCFG.asymmetry_sw_hydrophobic.values[i,:-1]),
                                   axis=0)
            f = interp1d(cwvl, Y,
                         kind='cubic',
                         # kind='nearest',
                         fill_value='extrapolate')
            g = f(wavel)
            if np.isscalar(ssa):
                mext = np.array([mext])
                ssa = np.array([ssa])
                g = np.array([g])
            for i in range(len(rh.shape)):
                mext = np.expand_dims(mext, axis=0)
                ssa = np.expand_dims(ssa, axis=0)
                g = np.expand_dims(g, axis=0)
            return mext, ssa, g

        def get_hydrophil(i, rh, wavel, mono):
            AERCFG = self.AERCFG

            if mono:
                cwvl = 1e9*AERCFG.wavelength_mono.values #[nm]
            else:
                channels1 = np.concatenate((1./AERCFG.wavenumber1_lw[:-1],
                                            1./AERCFG.wavenumber1_sw[:-1]),axis=0)
                channels2 = np.concatenate((1./AERCFG.wavenumber2_lw[:-1],
                                            1./AERCFG.wavenumber2_sw[:-1]),axis=0)
                cwvl=np.mean(np.vstack((channels1,channels2)),axis=0)
                cwvl*= 1e7 #[nm]

            rh1 = AERCFG.relative_humidity1.values
            ihum = np.searchsorted(rh1, rh)-1

            if mono:
                Y = AERCFG.mass_ext_mono_hydrophilic.values[i, ihum, :]
            else:
                Y = np.concatenate((AERCFG.mass_ext_lw_hydrophilic.values[i,ihum,:-1],
                                    AERCFG.mass_ext_sw_hydrophilic.values[i,ihum,:-1]),
                                   axis=-1)
            f = interp1d(cwvl, Y,
                         kind='cubic',
                         # kind='nearest',
                         axis=-1,
                         fill_value='extrapolate')
            mext = f(wavel)

            if mono:
                Y = AERCFG.ssa_mono_hydrophilic.values[i, ihum, :]
            else:
                Y = np.concatenate((AERCFG.ssa_lw_hydrophilic.values[i,ihum,:-1],
                                    AERCFG.ssa_sw_hydrophilic.values[i,ihum,:-1]),
                                   axis=-1)
            f = interp1d(cwvl, Y,
                         kind='cubic',
                         # kind='nearest',
                         axis=-1,
                         fill_value='extrapolate')
            ssa = f(wavel)

            if mono:
                Y = AERCFG.asymmetry_mono_hydrophilic.values[i, ihum, :]
            else:
                Y = np.concatenate((AERCFG.asymmetry_lw_hydrophilic.values[i,ihum,:-1],
                                    AERCFG.asymmetry_sw_hydrophilic.values[i,ihum,:-1]),
                                   axis=-1)
            f = interp1d(cwvl, Y,
                         kind='cubic',
                         # kind='nearest',
                         axis=-1,
                         fill_value='extrapolate')
            g = f(wavel)
            return mext, ssa, g

        rh = self.rh
        if np.isscalar(wavel):
            wavel = np.array([wavel])
        else:
            wavel = np.array(wavel)
        if np.isscalar(rh):
            rh = np.array([rh])
        else:
            rh = np.array(rh)
        AEROSOL = aerosol_type

        # SS1 -> Sea salt 0.03-0.5        OPAC
        # SS2 -> Sea salt 0.5-5           OPAC
        # SS3 -> Sea salt 5-20            OPAC
        # DU1 -> Dust 0.03-0.55           Dubovic et al 2002
        # DU2 -> Dust 0.55-0.9            Woodward et al 2001
        # DU3 -> Dust 0.9-20              Fouquart et al 1987
        # OM2 -> Organic Matter hydrophilic    OPAC-Mixture
        # Om1 -> Organic Matter hydrophobic    OPAC Mixture at 20% humidity
        # BC1 -> Black Carbon hydrophilic      OPAC (SOOT) (same as BC2 in CAMS)
        # BC2 -> Black Carbon hydrophobic      OPAC (SOOT)
        # SU  -> Sulfates                      Lacis et al (GACP)
        if AEROSOL == 'DU1':
            mext, ssa, g = get_hydrophob(0, rh, wavel,mono)
            mmr_key = 'aermr04'
        elif AEROSOL == 'DU2':
            mext, ssa, g = get_hydrophob(7, rh, wavel,mono)
            mmr_key = 'aermr05'
        elif AEROSOL == 'DU3':
            mext, ssa, g = get_hydrophob(5, rh, wavel,mono)
            mmr_key = 'aermr06'
        elif AEROSOL == 'BC1':
            mext, ssa, g = get_hydrophob(10, rh, wavel,mono)
            mmr_key = 'aermr09'
        elif AEROSOL == 'BC2':
            mext, ssa, g = get_hydrophob(10, rh, wavel,mono)
            mmr_key = 'aermr10'
        elif AEROSOL == 'SS1':
            mext, ssa, g = get_hydrophil(0, rh, wavel,mono)
            mmr_key = 'aermr01'
        elif AEROSOL == 'SS2':
            mext, ssa, g = get_hydrophil(1, rh, wavel,mono)
            mmr_key = 'aermr02'
        elif AEROSOL == 'SS3':
            mext, ssa, g = get_hydrophil(2, rh, wavel,mono)
            mmr_key = 'aermr03'
        elif AEROSOL == 'SU':
            mext, ssa, g = get_hydrophil(4, rh, wavel,mono)
            mmr_key = 'aermr11'
        elif AEROSOL == 'OM1':
            mext, ssa, g = get_hydrophob(9, rh, wavel,mono)
            mmr_key = 'aermr08'
        elif AEROSOL == 'OM2':
            mext, ssa, g = get_hydrophil(3, rh, wavel,mono)
            mmr_key = 'aermr07'
        return mmr_key, mext, ssa, g

    def aerosol_optprop(self, wvl, delta_eddington=False,mono=True):
        """
        calculate spectral optical properties of CAMS aerosol.
        aod - spectral aerosol optical depth
        ext - spectral extinction coefficient [km-1]
        ssa - spectral single scattering albedo
        g   - spectral asymmetry parameter

        Parameters
        ----------
        wvl : float, array(n)
            spectral wavelength [nm]
        delta_eddington : bool, (optional)
            switch on delta eddington scaling of the phase function if True.
            The default value is: False.
        mono : bool, (optional)
            If True, use optical properties calculated for single wavelengths.
            This is preferable, but only useable for shortwave calculation [340nm - 2130nm].
            For wavelenghts >2130nm consider switching to band wise optical 
            properties [232nm - 25000nm](False). The default is True.

        Returns
        -------
        ds_sfc : xarray.Dataset
            This dataset includes the column integrated spectral optical
            properties of aerosol at [wvl]
        ds_ml : xarray.Dataset
            This dataset includes the spectral optical properties per column
            at [wvl]
        """
        if np.isscalar(wvl):
            wvl = np.array([wvl])
        else:
            wvl = np.array(wvl)
        f = self._area_density()
        od_ext_ml = np.zeros((f.shape[0], f.shape[1], len(wvl)))
        od_scat_ml = np.zeros(od_ext_ml.shape)
        scatg_ml = np.zeros(od_ext_ml.shape)

        for AEROSOL in ['SS1', 'SS2', 'SS3', 'DU1', 'DU2', 'DU3', 'OM1', 'OM2', 'BC1', 'BC2', 'SU']:
            mmr_key, mext, ssa, g = self.get_aerosol_props(AEROSOL, wvl, mono)
            mmr = flatten_coords(self.cams_ml[mmr_key].values, idx_keep=1)
            local_od = mmr[:, :, np.newaxis] * f[:, :, np.newaxis] * mext
            od_ext_ml += local_od.copy()
            od_scat_ml += local_od.copy() * ssa
            scatg_ml += local_od.copy() * ssa * g

        # optical properties of individual layers
        g_ml = np.zeros(od_scat_ml.shape)
        ssa_ml = np.zeros(od_scat_ml.shape)

        # calculate g and ssa only if denominator is not zero
        # else they default to zero
        idx = od_ext_ml!=0
        g_ml[idx] = scatg_ml[idx] / od_scat_ml[idx]
        ssa_ml[idx] = od_scat_ml[idx] / od_ext_ml[idx] 
        aod_ml = od_ext_ml # aerosol optical depth [-]
        dz = np.abs(np.diff(self.z_ilvl))*1e-3 # layer depth [km]
        ext_ml = aod_ml / dz[:, :, np.newaxis] # extinction coefficient [km-1]

        if delta_eddington:
            aod_ml, ssa_ml, g_ml = self._scale_delta_eddington(aod_ml,
                                                               ssa_ml,
                                                               g_ml)

        ds_ml = xr.Dataset({'g': (('column', 'mlvl', 'wvl'), g_ml),
                            'ssa': (('column', 'mlvl', 'wvl'), ssa_ml),
                            'aod': (('column', 'mlvl', 'wvl'), aod_ml),
                            'ext': (('column', 'mlvl', 'wvl'), ext_ml)},
                           coords={
                                   'time': ('column', self.times),
                                   'lat': ('column', self.lats),
                                   'lon': ('column', self.lons),
                                   'wvl': ('wvl', wvl)})

        # column props
        aod_sfc = np.sum(aod_ml, axis=1)
        ext_sfc = aod_sfc / self.z_mlvl[:, 0][:, np.newaxis]
        ssa_sfc = np.sum(od_scat_ml, axis=1)/aod_sfc
        g_sfc = np.sum(scatg_ml, axis=1)/np.sum(od_scat_ml, axis=1)

        ds_sfc = xr.Dataset({'g': (('column', 'wvl'), g_sfc),
                             'ssa': (('column', 'wvl'), ssa_sfc),
                             'aod': (('column', 'wvl'), aod_sfc),
                             'ext': (('column', 'wvl'), ext_sfc)},
                            coords={
                                   'time': ('column', self.times),
                                   'lat':  ('column', self.lats),
                                   'lon':  ('column', self.lons),
                                   'wvl':  ('wvl', wvl)})
        return ds_sfc, ds_ml


class SI:
    """ SI constants according to 9th edition of the SI Brochure, BIPM 2019
    Hz = s-1
    J = kg m2 s-2
    C = A s
    lm = cd m2 m-2 = cd sr
    W = kg m2 s-3
    """
    # Avogadro constant
    N_A = 6.02214076e+23 # [mol-1]
    # Planck constant
    h = 6.62607015e-34 # [J s]
    # speed of light in vacuum
    c = 299792458 # [m s-1]
    # unpertubed ground state hyperfine transition frequency of the caesium 133 atom
    Dv_Cs = 9192631770 # [Hz]
    # elementary charge
    e = 1.602176634e-19 # [C] [A s]
    # Boltzmann constant
    k = 1.380649e-23 # [J K-1]
    # luminous efficacy of  monochromatic radiation of frequency 540e12 Hz
    K_cd = 683 # [lm W-1]

class CONSTANTS:
    """ specific constants
    """
    # gas constant
    R = SI.N_A * SI.k
    # molar mass
    Mm_dry = 28.9645 #[g mol-1]
    Mm_wv = 2.*1.008 + 15.999 # H2O [g mol-1]
    # specific gas constants
    Rdry = R / (Mm_dry/1000.) # [J kg-1 K-1]
    Rwv = R / (Mm_wv/1000.)


def flatten_coords(A, idx_keep=1):
    """ flattening the data of an array but keeping one axis.
    Example:
        >>> a = np.array([[1,2,3],
                          [4,5,6]])
        >>> a.flatten()
            array([1, 2, 3, 4, 5, 6])
        >>> A = np.zeros((a.shape[0],2,a.shape[1]))
        >>> A[:,0,:] = a.copy()
        >>> A[:,1,:] = a*10
        >>> flatten_coords(A,1)
            array([[ 1., 10.],
                   [ 2., 20.],
                   [ 3., 30.],
                   [ 4., 40.],
                   [ 5., 50.],
                   [ 6., 60.]])
    """
    N = A.shape[idx_keep]
    B = np.zeros(A.reshape(-1, N).shape)
    for i in range(N):
        B[:, i] = A.take(i, axis=idx_keep).flatten()
    return B
