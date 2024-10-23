from astropy.coordinates import get_sun, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u
from datetime import datetime, timedelta

def compute_sun_position(lat, lon, time_step, start_time):
    """
    Take a given point P (lat, lon) and time t. Find the Sun's position (azimuth, elevation) at P.
    """
    time = start_time + timedelta(hours=time_step)
    t = Time(time.isoformat())

    moon_loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=0*u.m)
    sun_loc = get_sun(t).transform_to(AltAz(obstime=t, location=moon_loc))

    sun_azimuth = sun_loc.az.deg
    sun_elev = sun_loc.alt.deg

    return sun_azimuth, sun_elev

def compute_horizon_elevation(df):
    """
    Take a given point P (coords: lat, lon, elevation) and a Sun azimuth angle. Find the height
    of the horizon at P in the direction of the Sun. This is the elevation angle at which the Sun
    will rise above the horizon at P.
    """
    lats = df['Latitude'].values
    lons = df['Longitude'].values
    elevs = df['Elevation'].values

    return c
