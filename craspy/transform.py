import numpy as np
from .index import _matching_slabs

def _fix_mask(data, mask):
    """

    Parameters
    ----------
    data : numpy.ndarray or numpy.ma.MaskedArray
        Astronomical data cube.
    mask : numpy.ndarray
        Boolean that will be applied.
    
    Returns
    -------
    result : numpy.ma.MaskedArray
        Masked astronomical data cube.
    """

    ismasked = isinstance(data, np.ma.MaskedArray)
    if ismasked and mask is None:
        return data
    else:
        return np.ma.MaskedArray(data, mask)

def _standarize(data):
    """
    Standarize astronomical data cubes in the 0-1 range.

    Parameters
    ----------
    data : numpy.ndarray or astropy.nddata.NDData or or astropy.nddata.NDData
        Astronomical data cube.

    Returns
    -------
    result : tuple
        Tuple containing the standarized numpy.ndarray or astropy.nddata.NDData cube, the factor scale y_fact and the shift y_min.
    """
    y_min = data.min()
    res = data - y_min
    y_fact = res.sum()
    res = res / y_fact
    return (res, y_fact, y_min)


def _unstandarize(data, a, b):
    """
    Unstandarize the astronomical data cube: :math:`a \cdot data + b`.

    Parameters
    ----------
    data : numpy.ndarray or astropy.nddata.NDData or or astropy.nddata.NDData
        Astronomical data cube.
    a : float
        Scale value.
    b : float
        Shift value.

    Returns
    -------
    result : numpy.ndarray or astropy.nddata.NDData
        Unstandarized astronomical cube.
    """
    return a*data+b


def _add(data, flux, lower, upper):
    """
    Adds flux to a sub-cube of an astronomical data cube.

    Parameters
    ----------
    data : numpy.ndarray or astropy.nddata.NDData or or astropy.nddata.NDData
        Astronomical data cube.
    flux : numpy.ndarray
        Flux added to the cube.
    lower : float
        Lower bound of the sub-cube to which flux will be added.
    upper : float
        Upper bound of the sub-cube to which flux will be added.
    """

    data_slab, flux_slab = _matching_slabs(data, flux, lower, upper)
    data[data_slab] += flux[flux_slab]


def _denoise(data, threshold):
    """
    Performs denoising of data cube, thresholding over the threshold value.

    Parameters
    ----------
    data : numpy.ndarray or astropy.nddata.NDData or or astropy.nddata.NDData
        Astronomical data cube.
    threshold : float
        Threshold value used for denoising.

    Returns
    -------
    result : numpy.ndarray
        Denoised (thresholded) astronomical data cube.
    """

    elms = data > threshold
    newdata = np.zeros(data.shape)
    newdata[elms] = data[elms]
    return newdata


# TODO: Candidate for deprecation
def _integrate(data, mask=None, axis=(0)):
    """
    Sums the slices of a cube of data given an axis.

    Parameters
    ----------
    data : (M,N,Z) numpy.ndarray or astropy.nddata.NDData or astropy.nddata.NDDataRef
        Astronomical data cube.

    mask : numpy.ndarray (default = None)

    axis : int (default=(0))

    Returns
    -------
     A numpy array with the integration results.

    """
    if mask is not None:
        data = fix_mask(data, mask)
    newdata = np.sum(data, axis=axis)
    mask = np.isnan(newdata)
    return newdata