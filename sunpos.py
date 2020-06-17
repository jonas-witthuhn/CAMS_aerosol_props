#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# sunpos.py is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors/Copyright(2012-2018):
# -Hartwig Deneke (deneke@tropos.de)
# -Jonas Witthuhn (witthuhn@tropos.de)
# -Carola Barrientos (barrientos@tropos.de)

'''
sunpos.py is a library for calculation of the sun position, based on
an algorithm from the Astronomical Alamanac. This algorithm was
compared in Michalsky (1988a,1988b) with other popular approximate
formulae and was found to be the most accurate. It is thus recommended
by the WMO Guide to Meteorological Instruments and Methods of
Observations for practical application.

References
----------
.. [1] United States Naval Observatory, 1993: The Astronomical Almanac, 
       Nautical Almanac Office, Washington DC.
.. [2] Michalsky, J.J., 1988a: The Astronomical Almanac’s algorithm for 
       approximate solar position (1950–2050).
.. [3] Michalsky, J.J., 1988b: Errata. The astronomical almanac’s algorithm 
       for approximate solar position (1950–2050).
.. [4] World Meteorological Organization, 2014: Guide to Meteorological 
       Instruments and Methods of Observation. Geneva, Switzerland, World 
       Meteorological Organization, 1128p. (WMO-No.8, 2014).
'''

import numpy as np
from numpy import sin, cos
from numpy import arcsin as asin, arccos as acos, arctan2 as atan2
from numpy import deg2rad, rad2deg
import datetime, pytz

#: Default epoch used for sun position calculation
EPOCH_J2000_0 = np.datetime64("2000-01-01T12:00:00")
#: UNIX standard epoch
EPOCH_UNIX    = np.datetime64("1970-01-01T00:00:00")

def sincos(x):
    '''
    Evaluate sin/cos simultaneously.

    Parameters
    ----------
    x: float or ndarray of float
        Angle in radians.

    Returns
    -------
    sin : float or ndarray of float
        the sine of x
    cos : float or ndarray of float
        the cosine of x
    '''
    return sin(x),cos(x)

def datetime2julday(dt, epoch=EPOCH_J2000_0):
    '''
    Convert datetime to Julian day number.
    
    Parameters
    ---------

    Returns
    -------
    jd: double or ndarray of double
        The 
    '''
    jd = (dt-epoch)/np.timedelta64(1,"D")
    return jd


def julday2datetime(jd,epoch=EPOCH_J2000_0):
    '''
    Convert Julian day to datetime.

    Parameters
    ----------
    jd: double or ndarray of double: 
        The Julian day, relative to the epoch
    epoch: datetime64
        The epoch used as reference

    Returns
    -------
    dt: datetime64 or ndarray of datetime64
        The datetime64 
    '''
    return epoch + np.timedelta64(1,'D')*jd

def julday2gmst(jd):
    '''
    Convert Julian day to Greenwich mean sideral time.

    Parameters
    ----------
    jd:  double or ndarray of double

    Returns
    -------
    gmst : double or ndarray of double
        The Greenwich mean sideral time [in hours].

    Examples
    --------
    >>> import numpy as np, solpos
    >>> dt = np.datetime64('2012-07-01T12:00')
    >>> jd = sunpos.datetime2julday(dt)
    >>> gmst = sunpos.julday2gmst(jd)
    '''
    hh   = np.remainder(jd-0.5,1.0)*24.0
    gmst = 6.697375 + 0.0657098242 *jd + hh
    return np.remainder(gmst,24.0)

def mean_longitude(jd):

    '''
    Mean solar longitude.

    Parameters
    ----------
       jd(double): the Julian day.

    Returns
    -------
    mnlon:  double or ndarray
        The mean solar longitude (in degrees).
    '''
    mnlon  = 280.460 + 0.9856474*jd; 
    return np.remainder(mnlon,360.0)

def mean_anomaly(jd):
    '''
    Mean solar anomaly.

    Args:
       jd(double): the Julian day.

    Returns:
       double: the mean solar anomaly [in degrees].
    '''
    mnanom = 357.528 + 0.9856003*jd
    return np.remainder(mnanom,360.0)

def ecliptic_longitude(jd):
    '''
    Calculate the ecliptic longitude of the sun.

    Parameters
    ----------
    jd : double or ndarray
       The Julian day.

    Returns
    ------
    eclon : double or ndarray
       The ecliptic longitude [in degrees].
    '''
    eclon  = 280.460 + 0.9856474*jd
    mnanom = 357.528 + 0.9856003*jd
    mnanom = np.deg2rad(mnanom)
    eclon += 1.915*sin(mnanom)+ 0.020*sin(2.0*mnanom)
    return np.remainder(eclon,360.0)

def oblq_ecliptic(jd):

    '''
    Get obliquity of ecliptic.

    Parameters
    ----------
    jd : double or ndarray
       The Julian day.

    Returns
    -------
    ep: double or ndarray
        The obliquity of ecliptic [in degrees].
    '''
    oblqec = 23.439 - 0.0000004*jd;
    return oblqec

def celestial_coords(jd):
    '''
    Get celestial coordinates of sun.

    Parameters
    ----------
    jd : double or ndarray
       The Julian day.

    Returns
    -------
    (dec,ra): tuple of doubles or ndarrays
        The declination/right ascension of the sun [in radians].
    '''

    # get ecliptic longitude and obliquity of ecliptic
    # and convert to radians
    eclon  = deg2rad(ecliptic_longitude(jd))
    oblqec = deg2rad(oblq_ecliptic(jd))

    # get trig. functions 
    (sin_eclon,cos_eclon)   = sincos(eclon)
    (sin_oblqec,cos_oblqec) = sincos(oblqec)
    
    # Calculate declination
    dec = asin(sin_oblqec*sin_eclon)

    # Calculate right ascension
    num = cos_oblqec*sin_eclon
    den = cos_eclon
    ra = atan2(num,den)
    ra = np.remainder(ra,2.0*np.pi)
    return (dec,ra)

def zenith_azimuth(jd, lat, lon):
    '''
    Get solar zenith/azimuth angle for a specific geographic location.

    Parameters
    ----------
    jd: double or array_like
        The Julian day
    lat: double or array_like
        The latitude [in degrees]
    lon: double or array_like
        The longitude [in degrees]

    Returns
    -------
       tuple of zenith/azimuth angle [in degrees].
    '''

    # Get celestial coordinates and
    # Greenwich mean sideral time
    (dec,ra) = celestial_coords(jd)
    (sin_dec,cos_dec) = sincos(dec)
    gmst = julday2gmst(jd)

    # Calculate Greenwich hour angle, trig. funcs.
    gha = deg2rad((gmst*15.0))-ra
    (sin_gha,cos_gha) = sincos(gha)

    # Calc. trig functions of lat/lon
    (sin_lat,cos_lat) = sincos(deg2rad(lat))
    (sin_lon,cos_lon) = sincos(deg2rad(lon))

    # Calculate trig. functions of hour angle
    sin_ha = sin_gha*cos_lon + cos_gha*sin_lon
    cos_ha = cos_gha*cos_lon - sin_gha*sin_lon

    # Calculate cos(sun zenith) 
    mu0 = sin_dec*sin_lat + cos_dec*cos_lat*cos_ha

    # Calculate azimuth
    #azi = asin(-cos_dec*sin_ha/np.sqrt(1.0-mu0**2))
    #if np.isscalar(azi):
    #    if sin_dec > mu0*sin_lat:
    #        if azi<0.0:
    #            azi += 2.0*pi
    #    else:
    #        azi = pi-azi
    #else:
    #    i = sin_dec<=mu0*sin_lat
    #    azi[i]=pi-azi[i]
    #    i = np.logical_and(azi<0.0,np.logical_not(i))
    #    azi[i]+=2.0*pi
    sin_azi = -cos_dec*sin_ha                  ## skip divide by cos_el 
    cos_azi = (sin_dec-mu0*sin_lat)/cos_lat
    azi = atan2(sin_azi,cos_azi)
    azi = np.remainder(azi,2.0*np.pi)
    return (np.rad2deg(acos(mu0)),np.rad2deg(azi))

def earth_sun_distance(jd):
    '''
    Calculate the sun-earth distance

    Args:
       jd(double): the Julian day.

    Returns:
       double the Earth-sun distance [in AE].
    '''

    g = deg2rad(mean_anomaly(jd))
    esd = 1.00014-0.01671*cos(g)+0.00014*cos(2.0*g)
    return esd

def hour_angle(jd,lon):
    '''
    Calculate the sun hour angle.

    Args:
       jd(double): the Julian day.

    Returns:
       double the hour angle [in degree].
    '''

    gmst = julday2gmst(jd)
    (dec,ra) = celestial_coords(jd)
    ha = np.remainder(gmst*15.0+lon-rad2deg(ra),360.0)-180.0
    return ha


def noon(jd,lon):

    '''
    Calculate the time of local noon.

    Args:
       jd(double): date given as Julian day.
       lon(double): the longitude of the location [in degrees].

    Returns:
       double: the date and time of local noon.
    '''

    # convergence limit, gives msec resolution
    eps = 1.0e-8 
    # first guess based on longitude
    noon = np.double(np.round(jd))
    noon = noon-lon/360.0
    # iterate noon till convergence
    for i in np.arange(0,10):
        prev = noon
        ha = hour_angle(noon,lon)
        noon = noon-ha/360.0
        if np.fabs(prev-noon)<eps: break
    return noon

def sunrise(jd,lat,lon,mu0=np.cos((90.0+34.0/60.0)*np.pi/180.0)):

    '''
    Calculate the time of sunrise.

    Args:
       jd(double): date given as Julian day.
       lat(double): the latitude of the location [in degrees].
       lon(double): the longitude of the location [in degrees].

    Returns:
       double: the date and time of sunrise.
    '''

    # get noon time
    jd_noon = noon(jd,lon)
    # get min/max mu0
    (sin_lat,cos_lat) = sincos(np.pi/180.0*lat)
    (dec,ra_noon) = celestial_coords(jd_noon)
    (sin_dec,cos_dec) = sincos(dec)
    # Check if we do have a sunset ...
    mu0_min = sin_dec*sin_lat-cos_dec*cos_lat
    mu0_max = sin_dec*sin_lat+cos_dec*cos_lat
    if mu0_max<mu0 or mu0_min>mu0: return None
    # Iteratively adjust hour angle at sunset/sunset time
    dra = 0.0
    for i in np.arange(0,5):
        # Calculate hour angle at sunset
        cos_ha = (mu0-(sin_dec*sin_lat))/(cos_dec*cos_lat)
        ha = -np.arccos(cos_ha)*12.0/np.pi
        # relation: dha = (1.0+0.0657098242/24.0)*delta - dra
        delta = (ha+dra)/(1.0+0.0657098242/24.0)
        (dec,ra) = celestial_coords(jd_noon+delta/24.0)
        (sin_dec,cos_dec) = sincos(dec)
        dra = (ra-ra_noon)*12.0/np.pi
    return jd_noon+delta/24.0

def sunset(jd,lat,lon,mu0=np.cos((90.0+34.0/60.0)*np.pi/180.0)):

    '''
    Calculate the time of sunset.

    Args:
       jd(double): the date given as Julian day.
       lat(double): the latitude of the location [in degrees].
       lon(double): the longitude of the location [in degrees].

    Returns:
       double: the date and time of sunset.
    '''

    # get noon time
    jd_noon = noon(jd,lon)
    # get min/max mu0
    (sin_lat,cos_lat) = sincos(np.pi/180.0*lat)
    (dec,ra_noon) = celestial_coords(jd_noon)
    (sin_dec,cos_dec) = sincos(dec)
    # Check if we do have a sunset ...
    mu0_min = sin_dec*sin_lat-cos_dec*cos_lat
    mu0_max = sin_dec*sin_lat+cos_dec*cos_lat
    if mu0_max<mu0 or mu0_min>mu0: return None
    # Iteratively adjust hour angle at sunset/sunset time
    dra = 0.0
    for i in np.arange(0,5):
        # Calculate hour angle at sunset
        cos_ha = (mu0-(sin_dec*sin_lat))/(cos_dec*cos_lat)
        ha = np.arccos(cos_ha)*12.0/np.pi
        # relation: dha = (1.0+0.0657098242/24.0)*delta - dra
        delta = (ha+dra)/(1.0+0.0657098242/24.0)
        (dec,ra) = celestial_coords(jd_noon+delta/24.0)
        (sin_dec,cos_dec) = sincos(dec)
        dra = (ra-ra_noon)*12.0/np.pi
    return jd_noon+delta/24.0

