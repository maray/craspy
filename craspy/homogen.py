from astropy import log
from astropy.table import Table
import numpy as np
from .index import _matching_slabs, _fix_limits, _slab
from .flux import create_mould, eighth_mould
from .transform import _add


def _update_min_energy(energy,mat,ub,lb,delta):
    """Updates the minimum energies of energy from mat defaced by delta. 
       ub and lb bounds are provided to shrink the mat matrix when required (out of bounds, or partial update)
    """
    # Numpyfy everithing
    ub = np.array(ub)
    lb = np.array(lb)
    delta=np.array(delta,dtype=int)
    # Create energy indices
    eub=ub + delta
    elb=lb + delta
    eslab,mslab = _matching_slabs(energy,mat,elb,eub)
    # Obtain a reduced view of the matrices
    eview=energy[eslab]
    mview=mat[mslab]
    cmat=mview < eview
    eview[cmat]=mview[cmat]


def _update_energies_sym(residual,energy,ev,ef,lb,ub):
    """Update the energies, only from the lb to the ub points. 
    """
    lb=_fix_limits(residual, lb)
    ub=_fix_limits(residual, ub)
    mcb=residual[_slab(residual, lb, ub)]
    #Obtain the reference of the eighth of the bubble.
    # Iterates for every point in the eighth of the bubble
    for i in range(0,ev.size):
        mat=mcb/ev[i]
        d=ef[i]
        # This is dimension specific
        if len(residual.shape) == 2:
            dset=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])*d
        else:
            dset=np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]])*d
        dset=np.vstack({tuple(row) for row in dset})
        for delta in dset:
            _update_min_energy(energy,mat,ub,lb,delta)

def precision_from_delta(delta,clev):
    delta=np.array(delta)
    sq_delta=1./(delta*delta)
    P=np.diag(sq_delta)
    return(-2*np.log(clev)*P)

def scat_pix_detect(data,threshold,noise,upper=None,lower=None,full_output=False):
    """ Obtain an homogeneous representation using the scattered pixels over a threshold.

    This function generates an homogeneous representation by using only those pixels above the threshold. 
    Each pixel generates several identical values depending on the intensity of each pixel (i.e., floor(intensity/noise)). 

    :param data: n-dimensional array containing the data to be processed.
    :type data: ndarray
    :param threshold: the theshold to consider a pixel relevant to be included in the representation.
    :type threshold: float
    :param noise : noise level to be subtracted from the intensities 
    :type noise: float
    :returns: An astropy table with all the ::math:`\\mu` and the metadata.
    :rtype: Table

    """
# COMMENT Removed
#This can also be understood as very small Gaussians ::math:`G(x) = a \\exp(-0.5 (\mu - x)^\\top P (\mu - x))` 
#    where ::math:`a=\sigma` correspond to the noise parameter,  the center ::math:`\\mu` is the pixel position (pos) and ::math:`P` is a diagonal matrix of
#    the form ::math:`-2\\log(\delta) \\cdot I` (::math:`\delta` is a small number

    #Restrict data to what the corresponding slab
    data=data[_slab(data,upper,lower)]

    ff=np.where(data>threshold)
    if isinstance(data,np.ma.MaskedArray):
        inten=data[ff].filled()
    else: 
        inten=data[ff]
    if full_output:
        residual=np.nan_to_num(data)
        synthetic=np.zeros(data.shape)
    ntimes=(inten/noise).astype(int)
    center=np.transpose(ff).astype(float)
    positions=[]
    for cen,tim in zip(center,ntimes):
        #print cen,tim,noise
        mylst=[cen.astype(int)]*tim
        positions.extend(mylst)
        if full_output:
            residual[tuple(cen.astype(int))]-=tim*noise
            synthetic[tuple(cen.astype(int))]=tim*noise
    positions=np.array(positions)
    #rep=Table([positions],names=['center'])
    if full_output:
        return positions,synthetic,residual
    return positions

def scat_kernel_detect(data,kernel,threshold=None,delta=None,noise=None,upper=None,lower=None,full_output=False,sym=False,verbose = False):
    residual = np.nan_to_num(data)
    log.info("Residual copy built");
    energy = residual.copy()
    log.info("Energy copy built");
    if full_output:
        synthetic = np.zeros(residual.shape)
        log.info("Synthetic matrix built");
        elist = []
    if sym != False:
        #(ev, ef) = _eighth_mould(P, delta)
        (ev,ef) = sym
        # Horrible hack to support 2D images
        lbz = (0,0,0)
        if len(data.shape) == 2:
            lbz = (0,0)
        log.info("Populating Energy Matrix for the first time (it might take a while)");
        _update_energies_sym(residual, energy, ev, ef, lb=lbz, ub=residual.shape)
    else:
        log.error("Non symetrical kernels not supported yet")
        return
    positions = []
    niter = 0
    delta = np.array(delta)
    while True:
        niter += 1
        idx = np.unravel_index(energy.argmax(), energy.shape)
        max_ener = energy[idx]
        if verbose and niter % 1000 == 0:
            log.info("Iteration: " + str(niter))
            log.info("Maximum energy E = " + str(max_ener) + " SNR = " + str(max_ener / noise))
        if max_ener < noise:
            if verbose:
                log.info("Criterion Met: Energy < Noise Level ")
            break
        if (max_ener < threshold):
            if verbose:
                log.info("Criterion Met: SNR=" + str(max_ener / noise) + "<" + str(threshold / noise))
            break
        ub = idx + delta + 1
        lb = idx - delta
        _add(residual, -noise * kernel, lb, ub)
        if full_output:
            _add(synthetic, noise * kernel, lb, ub)
            elist.append(max_ener)
        _update_energies_sym(residual, energy, ev, ef, lb, ub)
        positions.append(idx)
    positions = np.array(positions)
    if full_output:
        return positions, synthetic, residual, energy, elist
    return positions



# TODO: Candidate for deprecation
def bubble_detect(data,meta=None,noise=None,threshold=None,delta=None,gamma=0.1,full_output=False,verbose=False):
    if delta is None:
        if meta is None:
            delta=[1,1,1]
        else:
            spa=np.ceil((np.abs(meta['BMIN']/meta['CDELT1']) - 1)/2.0)
            delta=[1,spa,spa]
    if noise is None:
        noise=rms(data)
    if threshold is None:
        threshold=snr_estimation(data,mask=mask,noise=noise)*noise
    if verbose:
        print(threshold,noise,delta)
    P=precision_from_delta(delta,gamma)
    mould=create_mould(P,delta)
    #equant=mould.sum()*noise
    residual=np.nan_to_num(data)
    energy=residual.copy()
    if full_output:
        synthetic=np.zeros(residual.shape)
        elist=[]
    (ev,ef)= eighth_mould(P, delta)
    _update_energies_sym(residual,energy,ev,ef,lb=(0,0,0),ub=residual.shape)
    positions=[]
    niter=0
    delta=np.array(delta)
    while True:
        niter+=1
        idx      = np.unravel_index(energy.argmax(), energy.shape)
        max_ener = energy[idx]
        if verbose and niter%1000==0:
            log.info("Iteration: "+str(niter))
            log.info("Maximum energy E = "+str(max_ener)+" SNR = "+str(max_ener/noise))
        if max_ener < noise:
            if verbose:
                log.info("Criterion Met: Energy < Noise Level ")
            break
        if (max_ener < threshold):
            if verbose:
                log.info("Criterion Met: SNR="+str(max_ener/noise)+"<"+str(threshold/noise))
            break
        ub=idx + delta + 1
        lb=idx - delta
        _add(residual,-noise*mould,lb,ub)
        if full_output:
            _add(synthetic,noise*mould,lb,ub)
            elist.append(max_ener)
        _update_energies_sym(residual,energy,ev,ef,lb,ub)
        positions.append(idx)
    positions=np.array(positions)
    rep=Table([positions],names=['center'])
    if full_output:
        return rep,synthetic,residual,energy,elist
    return rep


    # DO THIS FOR BUBBLE
def synthesize_bubbles(syn,pos,mould,nlevel,delta):
    for idx in pos:
        ub=idx + delta + 1
        lb=idx - delta
        _add(syn,nlevel*mould,lb,ub)
    return syn


