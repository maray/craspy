
import numpy as np
from astropy.table import Table
from astropy import log

from .algorithm import Algorithm
from .data import Data
from .flux import standarize
from .homogen import synthesize_bubbles, _precision_from_delta, scat_pix_detect

def _vertical_flux_decomposition(rep,delta,noise,kernel,n_partitions,shape):
    n_rep=len(rep)/n_partitions
    img_list=[]
    vmax=0
    for i in range(n_partitions):
        synNew=np.zeros(shape)
        ini=n_rep*i
        end=n_rep*(i+1) - 1
        p_rep=rep[ini:end]
        synthesize_bubbles(synNew,p_rep,kernel,noise,delta)
        img=synNew.sum(axis=(0))
        vmax=max(vmax,img.max())
        img_list.append(img)
    return img_list,vmax



class HRep(Algorithm):


    def toVFD(table, n_partitions, shape):
        rep, delta, noise, kernel = HRep.toTuple(table)
        return _vertical_flux_decomposition(rep, delta, noise, kernel, n_partitions, shape)

    def toImage(table, template):
        rep, delta, noise, kernel = HRep.toTuple(table)
        synNew = np.zeros(template.data.shape)
        synthesize_bubbles(synNew, rep, kernel, noise, delta)
        scale = table.meta['SCALE']
        shift = table.meta['SHIFT']
        return Data(synNew * scale - shift, wcs=template.wcs, unit=template.unit, meta=template.meta)

    def toTuple(table):
        rep = np.array([table[c] for c in table.columns])
        rep = rep.T
        delta = np.array([table.meta['DELTAX'], table.meta['DELTAY'], table.meta['DELTAZ']])
        noise = table.meta['NOISE']
        P = precision_from_delta(delta, 0.1)
        kernel = create_mould(P, delta)
        return rep, delta, noise, kernel

    def default_params(self):
        if 'KERNEL' not in self.config:
            self.config['KERNEL'] = 'PIXEL'
        if 'DELTA' not in self.config:
            self.config['DELTA'] = None
        if 'RMS' not in self.config:
            self.config['RMS'] = None
        if 'SNR' not in self.config:
            self.config['SNR'] = None
        if 'STANDAR' not in self.config:
            self.config['STANDAR'] = True
        if 'VERBOSE' not in self.config:
            self.config['VERBOSE'] = False
        if 'GAMMA' not in self.config:
            self.config['GAMMA'] = 0.1

    def run(self, cube):
        """
            Run the Homogenous Representation algorithm a Data Object.

            Parameters
            ----------
            cube : the cube to represent

            Returns
            -------
            result : ???
        """
        delta = self.config['DELTA']
        noise = self.config['RMS']
        snr = self.config['SNR']
        standar = self.config['STANDAR']
        verbose = self.config['VERBOSE']
        gamma = self.config['GAMMA']

        scale = 1.0
        shift = 0.0

        if standar:
            (cube, scale, shift) = standarize(cube)

        if snr is None:
            snr = snr_estimation(cube.data, mask=cube.mask, noise=noise)

        if delta is None:
            if cube.meta is None:
                delta = [1, 1, 1]
            else:
                spa = np.ceil((np.abs(cube.meta['BMIN'] / cube.meta['CDELT1']) - 1) / 2.0)
                delta = [1, spa, spa]
        if noise is None:
            noise = flux._rms(cube.data)

        if self.config['KERNEL'] == 'PIXEL':
            positions,synthetic,residual=scat_pix_detect(cube.data,threshold=snr*noise,full_output=True)

        if self.config['KERNEL'] == 'METABUBBLE':

            # if verbose:
            #     log.info(snr, noise, delta)
            P = homogen._precision_from_delta(delta, gamma)
            kernel = flux._create_mould(P, delta)
            sym = flux._eighth_mould(P, delta)
            positions, synthetic, residual, energy, elist = homogen.scat_kernel_detect(cube.data,delta=delta,kernel=kernel,threshold=snr*noise,noise=noise,full_output=True,sym=sym,verbose=verbose)
        positions = np.array(positions)
        # Pack metadata
        metapack = dict()
        metapack['DELTAX']=delta[0]
        metapack['DELTAY'] = delta[1]
        metapack['DELTAZ'] = delta[2]
        metapack['SCALE'] = scale
        metapack['SHIFT'] = shift
        metapack['KERNEL'] = self.config['KERNEL']
        metapack['RMS'] = noise
        metapack['SNR'] = snr
        metapack['GAMMA'] = gamma
        rep = Table(positions, names=['x','y','z'],meta=metapack)

        return rep, Data(synthetic,meta=cube.meta,mask=cube.mask,unit=cube.unit,wcs=cube.wcs),Data(residual,meta=cube.meta,mask=cube.mask,unit=cube.unit,wcs=cube.wcs)