"""
This module is for modeling the mapping from pixel to world coordinates in the 
spectrograph slits. 

Note this may not be necessary for WCSs that are linear with the sky/pixel offsets (like NIRSpec)
"""
__all__ = ['WCSW2P', 'ChebyshevW2P']

from abc import ABC, abstractmethod

import numpy as np

from astropy import units as u

import jax
import jax.numpy as jnp


class W2P(ABC):
    """
    A class that maps from a specific slit-centric view of world coordinates to pixel coordinates.
    """
    @abstractmethod
    def evaluate_wcs_values(self, ra, dec, wave):
        """
        ra, dec should be in degrees, wave should be in microns. 
        
        Returns xslit, yslit in pixels
        """
        pass

class WCSW2P(W2P):
    def __init__(self, wcs):
        self.wcs = wcs

    def evaluate_wcs_values(self, ra, dec, wave):
        if not isinstance(ra, np.ndarray):
            ra = np.array([ra])
        if not isinstance(dec, np.ndarray):
            dec = np.array([dec])
        if not isinstance(wave, np.ndarray):
            wave = np.array([wave])
        return self.wcs.world_to_pixel_values(ra, dec, wave)

class ChebyshevW2P(W2P):
    """
    Approximates the WCS using chebyshev polynomials, sped up via jax/orthax.

    The algorithmic tests for this are in the ``fast_wcs.ipynb`` notebook.
    """
    def __init__(self, chebcoeffs, domain_infos, degs, matmul_precision='highest'):
        self.chebcoeffs = chebcoeffs
        self.domain_infos = domain_infos
        self.degs = degs
        self.matmul_precision = matmul_precision

    @classmethod
    def derive_from_wcs(cls, wcs, degs, datashape, matmul_precision='highest', coeff_dtype=jnp.float32):
        sky, wave = wcs.pixel_to_world(*np.indices(datashape)[::-1])
        domain_infos = [cls.get_unitdomaininfo(v) for v in (sky.ra, sky.dec, wave)]

        dpts = [np.polynomial.chebyshev.chebpts1(d+1) for d in degs]
        ds = np.array(np.meshgrid(*dpts)).reshape(3, -1)
        wpts = np.array([cls.from_cheb_domain(d, di) for d,di in zip(ds, domain_infos)])
        xs, ys = wcs.world_to_pixel_values(*wpts)

        msk = np.isfinite(xs) & np.isfinite(ys) & np.all(np.isfinite(wpts), axis=0)
        
        vandermat = np.polynomial.chebyshev.chebvander3d(*ds[:, msk], degs)
        solns, _, _, _ = np.linalg.lstsq(vandermat, np.array([xs[msk], ys[msk]]).T)

        return cls(chebcoeffs=jnp.array(solns, dtype=coeff_dtype), 
                   domain_infos=domain_infos,
                   degs=degs, matmul_precision=matmul_precision)

    @staticmethod
    def get_unitdomaininfo(v):
        if isinstance(v, u.Quantity):
            v = v.value

        mi = np.nanmin(v)
        mx = np.nanmax(v)

        offset = mi
        scale = mx - mi

        return scale, offset

    @staticmethod
    def to_cheb_domain(v, domaininfo):
        scale, offset = domaininfo
        ud = (v - offset)/scale # [0, 1]
        return ud*2 - 1

    @staticmethod
    def from_cheb_domain(d, domaininfo):
        scale, offset = domaininfo
        ud = (d + 1) / 2
        return ud*scale + offset

    def evaluate_wcs_values(self, ra, dec, wave):
        rad = self.to_cheb_domain(ra, self.domain_infos[0])
        decd = self.to_cheb_domain(dec, self.domain_infos[1])
        waved = self.to_cheb_domain(wave, self.domain_infos[2])
        with jax.default_matmul_precision(self.matmul_precision):
            return _cheb_approx(rad, decd, waved, self.degs, self.chebcoeffs)

@jax.jit(static_argnums=3)
def _cheb_approx(x, y, z, degs, coeffs):
    # TODO: replace with lazy loading from py 3.15
    import orthax

    vandermat = orthax.chebyshev.chebvander3d(x, y, z, degs)
    return (vandermat @ coeffs).T
