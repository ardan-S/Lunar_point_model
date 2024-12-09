import numpy as np
from astropy.time import Time
from astropy.coordinates import get_sun, AltAz, ITRS
from astropy.coordinates import CartesianRepresentation
from datetime import timedelta


MOON_RADIUS = 1737.4 * 1000  # Convert to meters
GRID_RES = 240  # Resolution of the grid in meters
RES_DEG = (GRID_RES / MOON_RADIUS) * (180 / np.pi)  # Resolution in degrees


def compute_sun_position(lats, lons, time_step, start_time):
    """
    Take a given point P (lat, lon) and time t. Find the Sun's position (azimuth, elevation) at P.
    """
    time = start_time + timedelta(hours=time_step)
    t = Time(time.isoformat())

    sun_azimuths = np.zeros(len(lats))
    sun_elevs = np.zeros(len(lats))

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        x = MOON_RADIUS * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
        y = MOON_RADIUS * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
        z = MOON_RADIUS * np.sin(np.radians(lat))

        moon_loc = ITRS(CartesianRepresentation(x=x, y=y, z=z), obstime=t)
        sun_loc = get_sun(t).transform_to(AltAz(obstime=t, location=moon_loc))

        # sun_azimuths[i] = sun_loc.az.deg
        # sun_elevs[i] = sun_loc.alt.deg

    return sun_azimuths, sun_elevs