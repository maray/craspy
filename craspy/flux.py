from astropy.nddata import support_nddata

from astropy.units import Quantity
import numpy as np

from .data import Data
from .axes import fov, features, axes_units
from .transform import _fix_mask, _standarize, _unstandarize, _add

def _rms(data, mask=None):
    """
    Compute the RMS of data. If mask != None, then we use that mask.

    Parameters
    ----------
    data : (M,N,Z) numpy.ndarray or astropy.nddata.NDData or or astropy.nddata.NDDataRef
        Astronomical data cube.

    mask : numpy.ndarray (default = None)

    Returns
    -------
    RMS of the data (float)
    """
    # TODO: check photutils background estimation for using that if possible
    if mask is not None:
        data = _fix_mask(data, mask)
    mm = data * data
    rms = np.sqrt(mm.sum() * 1.0 / mm.size)
    return rms

@support_nddata
def rms(data, mask=None, unit=None):
    """
        Compute the RMS of data.

        Parameters
        ----------
        data : (M,N) numpy.ndarray or astropy.nddata.NDData or astropy.nddata.NDDataRef
        mask : numpy.ndarray
            mask for the data
        unit : astropy.units.Unit
            Astropy Unit (http://docs.astropy.org/en/stable/units/)

        Returns
        -------
        rms : float
            RMS of data
    """

    # TODO: check photutils background estimation for using that if possible
    if unit is None:
        return _rms(data, mask)
    else:
        return _rms(data, mask) * unit

#TODO: candidate for deprecation
@support_nddata
def standarize(data, wcs=None, unit=None, mask=None, meta=None):
    """
        Standarize data:

        Parameters
        ----------
        data : (M,N) numpy.ndarray or astropy.nddata.NDData or astropy.nddata.NDDataRef
        wcs : World Coordinate System data (http://docs.astropy.org/en/stable/wcs/)
        mask : numpy.ndarray
            mask for the data
        unit : astropy.units.Unit
            Astropy Unit (http://docs.astropy.org/en/stable/units/)
        meta : FITS metadata

        Returns
        -------
        Standarized data where data = a * res + b
    """
    if mask is not None:
        data = _fix_mask(data, mask)
    (res, a, b) = _standarize(data)
    res = Data(res, uncertainty=None, mask=mask, wcs=wcs, meta=meta, unit=unit)
    return res, a, b

#TODO: candidate for deprecation
@support_nddata
def unstandarize(data, a, b, wcs=None, unit=None, mask=None, meta=None):
    """
        Unstandarize data: res = a * data + b


        Parameters
        ----------
        data : (M,N) numpy.ndarray or astropy.nddata.NDData or astropy.nddata.NDDataRef
        a : float
            slope of straight
        b : float
            Intercept of straight
        wcs : World Coordinate System data (http://docs.astropy.org/en/stable/wcs/)
        mask : numpy.ndarray
            mask for the data
        unit : astropy.units.Unit
            Astropy Unit (http://docs.astropy.org/en/stable/units/)
        meta : FITS metadata

        Returns
        -------
        NDDataRef: Unstandarized data: res = a * data + b
    """
    if mask is not None:
        data = _fix_mask(data, mask)
    res = _unstandarize(data, a, b)
    return Data(res, uncertainty=None, mask=mask, wcs=wcs, meta=meta, unit=unit)

#TODO: candidate for deprecation
@support_nddata
def add(data, flux, lower=None, upper=None, wcs=None, unit=None, meta=None, mask=None):
    """
        Create a new data with the new flux added.

        Lower and upper are bounds for data. This operation is border-safe and creates a new object at each call.

        Parameters
        ----------
        data : (M,N) numpy.ndarray or astropy.nddata.NDData or astropy.nddata.NDDataRef
        flux : float
            Flux of data
        lower : numpy.ndarray
        upper : numpy.ndarray
            Bounds for data
        wcs : World Coordinate System data (http://docs.astropy.org/en/stable/wcs/)
        mask : numpy.ndarray
            mask for the data
        unit : astropy.units.Unit
            Astropy Unit (http://docs.astropy.org/en/stable/units/)
        meta : FITS metadata

        Returns
        -------
        NDDataRef: structure with new flux added

    """

    # Please use the OO version data.add(flux) for modifying the data itself.
    res = data.copy()
    _add(res, flux, lower, upper)
    return Data(res, uncertainty=None, mask=mask, wcs=wcs, meta=meta, unit=unit)

#TODO: candidate for deprecation --> need to move to algorithms
@support_nddata
def denoise(data, wcs=None, mask=None, unit=None, threshold=0.0):
    """
        Simple denoising given a threshold (creates a new object)

        Parameters
        ----------
        data : (M,N) numpy.ndarray or astropy.nddata.NDData or astropy.nddata.NDDataRef
        wcs : World Coordinate System data (http://docs.astropy.org/en/stable/wcs/)
        mask : numpy.ndarray
            mask for the data
        unit : astropy.units.Unit
            Astropy Unit (http://docs.astropy.org/en/stable/units/)
        threshold : float

        Returns
        -------
        NDDataRef: Data denoised

    """
    if isinstance(threshold,Quantity):
        threshold=threshold.value
    newdata = _denoise(data, threshold)
    return Data(newdata, uncertainty=None, mask=mask, wcs=wcs, meta=None, unit=unit)

@support_nddata
def snr_estimation(data, mask=None, noise=None, points=1000, full_output=False, max_rms=3.0):
    """
    Heurustic that uses the inflexion point of the thresholded RMS to estimate where signal is dominant w.r.t. noise

    Parameters
    ----------
    data : (M,N,Z) numpy.ndarray or astropy.nddata.NDData or astropy.nddata.NDDataRef
        Astronomical data cube.

    mask : numpy.ndarray (default = None)

    noise : float (default=None)
        Noise level, if not given will use rms of the data.

    points : (default=1000)

    full_output : boolean (default=False)
        Gives verbose results if True

    Returns
    --------

    "Signal to Noise Radio" value

    """
    if noise is None:
        noise = rms(data, mask)
    x = []
    y = []
    n = []
    sdata = data[data > noise]
    for i in range(1, int(points)):
        val = 1.0 + (max_rms-1.0)* i / points
        sdata = sdata[sdata > val * noise]
        if sdata.size < 2:
            break
        n.append(sdata.size)
        yval = sdata.mean() / noise
        x.append(val)
        y.append(yval)
    y = np.array(y)
    v = y[1:] - y[0:-1]
    p = v.argmax() + 1
    snrlimit = x[p]
    if full_output == True:
        return snrlimit, noise, x, y, v, n, p
    return snrlimit

def gaussian_function(mu, P, feat, peak):
    """
    Generates an N-dimensional Gaussian using the feature matrix feat,
    centered at mu, with precision matrix P and with intensity peak.

    Parameters
    ----------
    mu : numpy.ndarray
        Centers of gaussians array.
    P : numpy.ndarray
        Precision matrix.
    feat : numpy.ndarray.
        Features matrix.
    peak : float
        Peak value of the resulting evaluation.

    Returns
    -------
    result: 2D numpy.ndarray
        Returns the gaussian function evaluated at the value on feat. 
    """

    cent_feat = np.empty_like(feat)
    for i in range(len(mu)):
        cent_feat[i] = feat[i] - mu[i]
    qform = (P.dot(cent_feat)) * cent_feat
    quad = qform.sum(axis=0)
    res = np.exp(-quad / 2.0)
    res = peak * (res / res.max())
    return res


def create_mould(P, delta):
    """
    Creates a Gaussian mould with precision matrix P, using the already computed values of delta.

    Parameters
    ----------
    P : numpy.ndarray
        Precision matrix.
    delta : list or numpy.ndarray
        Delta values used to generate the mould.

    Returns
    -------
    result : numpy.ndarray
        Mould matrix.
    """
    n = len(delta)
    ax = []
    elms = []
    for i in range(n):
        lin = np.linspace(-delta[i], delta[i], delta[i] * 2 + 1)
        elms.append(len(lin))
        ax.append(lin)
    grid = np.meshgrid(*ax, indexing='ij')
    feat = np.empty((n, np.product(elms)))
    for i in range(n):
        feat[i] = grid[i].ravel()
    mould = gaussian_function(np.zeros(n), P, feat, 1)
    mould = mould.reshape(*elms)
    return mould


def eighth_mould(P,delta):
    """This function creates an eighth of a symetrical Gaussian mould with precision matrix P, using the already computed values of delta
    """
    n=len(delta)
    ax=[]
    elms=[]
    for i in range(n):
        lin=np.linspace(0,delta[i],delta[i]+1)
        elms.append(len(lin))
        ax.append(lin)
    grid=np.meshgrid(*ax,indexing='ij')
    feat=np.empty((n,np.product(elms)))
    for i in range(n):
        feat[i]=grid[i].ravel()
    mould=gaussian_function(np.zeros(n),P,feat,1)
    return mould,feat.T

#TODO: really needed? it was for GaussClumps
@support_nddata
def world_gaussian(data, mu, P, peak, cutoff, wcs=None):
    """
        Creates a gaussian flux at mu position (WCS), with P shape, with a maximum value equal to peak,
        and with compact support up to the cutoff contour

        Parameters
        ----------
        data : (M,N) numpy.ndarray or astropy.nddata.NDData or astropy.nddata.NDDataRef
        mu : float
        P : tuple
            Shape of result
        peak : float
            maximum value

        cutoff :

        wcs : World Coordinate System data (http://docs.astropy.org/en/stable/wcs/)

        Returns
        -------

        Tuple of gaussian flux and borders

    """
    Sigma = np.linalg.inv(P)
    window = np.sqrt(2 * np.log(peak / cutoff) * np.diag(Sigma)) * axes_units(data, wcs=wcs)
    lower, upper = fov(data, mu, window, wcs=wcs)
    if np.any(np.array(upper - lower) <= 0):
        return None, lower, upper
    feat = features(data, wcs=wcs, lower=lower, upper=upper)
    feat = np.array(feat.columns.values())
    mu = np.array([x.value for x in mu])
    res = _gaussian_function(mu, P, feat, peak)
    # TODO Not generic
    res = res.reshape(upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2])
    return res, lower, upper
