"""
This module is for building a full forward model, including likelihood
evaluation.
"""
__all__ = ['NIRSpecSlitModel']

import math

from astropy import units as u
from astropy import constants as cnst

from nirspectroperf.spec_model import PhoenixModelsDirect
from nirspectroperf.psf import NIRSpecPSF
from nirspectroperf.wcs import WCSW2P
from nirspectroperf.build2d import scatter_flux

import numba
from numba import cuda

class NIRSpecSlitModel:
    param_names = 'raoff, decoff, v, teff, logg, feh'.split(', ')
    param_units = [u.arcsec, u.arcsec, u.km/u.s, u.K, u.dimensionless_unscaled, u.dimensionless_unscaled]

    def __init__(self, spec_model, psf_model, w2p, slit_data_template, ra0_deg, dec0_deg):
        """
        slit_data_template is a 2d array where the nans and dtype will be used to determine the output array shape
        """
        self.spec_model = spec_model
        self.psf_model = psf_model
        self.w2p = w2p
        self.slit_data_template = slit_data_template
        self._ra0_deg = ra0_deg
        self._dec0_deg = dec0_deg


    @classmethod
    def make_from_template(cls, observation_template_dm, slit):

        spec_model = PhoenixModelsDirect()
        psf_model = NIRSpecPSF(observation_template_dm, slit)
        slit_data_template = slit.data
        w2p = WCSW2P(slit.meta.wcs)

        ra0_deg = slit.source_ra
        dec0_deg = slit.source_dec

        return cls(spec_model, psf_model, w2p, slit_data_template, ra0_deg, dec0_deg)

    @property
    def ra0(self):
        return self._ra0_deg * u.deg
    @property
    def dec0(self):
        return self._dec0_deg * u.deg

    _ARCSEC_TO_DEG = u.arcsec.to(u.deg)
    _CKMS = cnst.c.to_value(param_units[2])
    def make_model(self, model_params, slit_data_template):
            raoff, decoff, v, teff, logg, feh = model_params
    
            spec = self.spec_model.make_spectrum(teff, logg, feh)
            
            ra = self._ra0_deg + raoff*self._ARCSEC_TO_DEG
            dec = self._dec0_deg + decoff*self._ARCSEC_TO_DEG
            wave = self.spec_model.wave_microns*(1 + v/self._CKMS)
    
            xslit, yslit = self.w2p.evaluate_wcs_values(ra, dec, wave)
    
            psf_flux, psf_x, psf_y = self.psf_model.evaluate_psf(xslit, yslit, spec.spectral_axis.mean())
    
            return scatter_flux(xslit, yslit, spec.flux.value, 
                            psf_x, psf_y, psf_flux, 
                            self.psf_model.oversampling, 
                            slit_data_template, 
                            sum_flux=True, 
                            ret_host_array=False)
            
    def loglike(self, model_params, slit_data, slit_data_unc):
        """
        slit_data and slit_data_unc should be cuda device arrays
        """
        modelflux = self.make_model(model_params, slit_data)
        return gpu_sum(loglike_norm_elementwise(modelflux, slit_data, slit_data_unc))


_NORM_FACTOR = 0.5*math.log(2*math.pi)
@numba.vectorize(['float32(float32, float32, float32)'], target='cuda')
def loglike_norm_elementwise(model, data, unc):
    return _NORM_FACTOR - unc - 0.5*((model-data)/unc)**2

@cuda.reduce
def gpu_sum(a, b):
    return a + b