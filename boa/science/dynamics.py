import numpy as np
from .deriv import pderiv3D

__all__ = ['avg_eqlat', 'epflux_eddyterms', 'epflux_all', 'theta']
    
def avg_equivlat(in_field, pv_field, n_lon, n_lat):
    """
    Average a 2-D input field along equivalent latitude bands 
    using global 2-D field of potential vorticity.
    
    Equivalent latitude is defined as 

    .. math::
    \phi_e = \mathrm{arcsin}[1- (A/(2*\pi*R^2))].

    where A is the area enclosed by the equivalent latitude band 
    and R is the radius of the Earth.
    This function uses the 'piece-wise constant' method 
    where PV is assumed to be constant within each grid box.

    Parameters
    ----------
    in_field: array_like
        The data to be averaged along equivalent latitude bands. 
        It must be a 2-D array or data that can be converted to such. 
    pv_field: array_like
        The data along the isolines of which the in_field data is to 
        be averaged. In atmospheric sciences this is usually the potential
        vorticity. It must be a 2-D array with exactly the same dimensions
        as in_field.
    n_lon: int (or sequence)
        Longitude values, can be an integer or sequence. If integer 
        assume that longitude values are divided evenly across the 
        globe. i.e. d_lon = 2*PI/n_lon
    n_lat: int (or sequence)
        Number of latitudes (or a sequence of latitudes) If integer then
        assume latitudes are equally divided between 90S and 90N
        
        FIXME!!!! Currently code only works if n_lon and n_lat are integers.
               Need to generalise to take 1-D arrays

    Returns
    -------
    latitude: list
        Equivalent latitude values
    infield_eq: list
        Values of in_field averaged along equivalent latitude bands.

    """
    # constants
    PI = np.pi

    # grid characteristics
    n_grid = int(n_lon)*int(n_lat)
    phi = PI/n_lat
    phih = 0.5*PI - phi*np.arange(n_lat+1)

    area_field = np.zeros([n_lon, n_lat])
    for j in range(n_lat):
        area_field[:, j] = 2*PI*(np.sin(phih[j]) - np.sin(phih[j+1]))/n_lon

    # reorder the fields
    ord_ind         = np.argsort(pv_field, axis=None)[::-1]
    infield_ordered = in_field.flatten()[ord_ind]
    pv_ordered      = pv_field.flatten()[ord_ind]
    area_ordered    = area_field.flatten()[ord_ind]

    # areas of equivalent latitude bands for output
    # sum area along latitude bands
    area_band = np.sum(area_field, axis = 0)
    infield_eq = np.zeros(n_lat)

    ll = 0
    area_now = 0.0
    infield_tot = 0.0

    # loop to average in equivalent latitude bands
    for nn in range(n_grid):
        area_now += area_ordered[nn]
        infield_tot += area_ordered[nn]*infield_ordered[nn]
        if (area_now >= area_band[ll] or (nn == n_grid-1)):
            infield_tot -= (area_now - area_band[ll])*infield_ordered[nn]
            infield_eq[ll] = infield_tot/area_band[ll]
            infield_tot = (area_now - area_band[ll])*infield_ordered[nn]
            area_now -= area_band[ll]
            ll += 1
            
    # in field is averaged along eq. latitude bands from 90N - 90S
    # legacy from times when we were mostly interested in NH 
    lat = PI/2 - np.arange(n_lat)*phi            
    return (lat, infield_eq)

def epflux_all(U, V, W, T, longitude, latitude, press, boa=None):
    """
    Calculate the Eliassen-Palm flux and related variables
    Basic equations are 3.53a, 3.53b of "Middle Atmospheric Dnamics", 
    by Andrews, Holton and Leovy (1987). 
    
    Parameters
    -----------
    U, V, T on pressure levels, 3D, one file at a time

    nc: boolean (optional)
        If set, output data as boa object

    Output
    ------
    epflux_eddyterms: <U>, <V>, <T>, <V'T'>, <U'V'>

    epflux_all:
    """
    pass

def epflux_eddyterms():
    """
    Return the eddy terms only
    """
    pass

def epflux_boa(netcdf = False):
    """
    Input boa variables U, V, T
    Ouput boa variable. Save to netcdf file if netcdf is true
    """
    pass

def theta():
    """
    Convert values on pressure levels to theta (potential temperature)
    """
    pass
