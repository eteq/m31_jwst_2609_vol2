"""
This module is for building a full forward model, including likelihood
evaluation.
"""
__all__ = ['NIRSpecSlitStarModel']

import math

import numpy as np

from astropy import units as u
from astropy import constants as cnst

from nirspectroperf.spec_model import PhoenixModelsResampled
from nirspectroperf.psf import NIRSpecPSF
from nirspectroperf.wcs import WCSW2P
from nirspectroperf.build2d import scatter_flux

import cupy as cp

class NIRSpecSlitStarModel:
    """
    Forward model of a nirspec slit 
    """
    param_names = 'F0, x_slit_offset, y_slit_offset, v, teff, logg, feh'.split(', ')
    param_units = [u.arcsec, u.arcsec, u.km/u.s, u.K, u.dimensionless_unscaled, u.dimensionless_unscaled]

    def __init__(self, spec_model, psf_model, w2p, slit_data_template, 
                 ra0_deg, dec0_deg, x_slit0, y_slit0):
        """
        slit_data_template is a 2d array where the nans and dtype will be used to determine the output array shape
        """
        self.spec_model = spec_model
        self.psf_model = psf_model
        self.w2p = w2p
        self.slit_data_template = slit_data_template


        # these are where the source nominally is when the x/y offsets are 0, in ra/dec and slitx/slity
        self._ra0_deg = ra0_deg
        self._dec0_deg = dec0_deg
        self.x_slit0 = x_slit0
        self.y_slit0 = y_slit0


    @classmethod
    def make_from_template(cls, observation_template_dm, slit, nspec=4096):
        allwls = slit.meta.wcs.pixel_to_world(*np.indices(slit.data.shape)[::-1])[1]
        minwl = np.nanmin(allwls)
        maxwl = np.nanmax(allwls)
        spec_model = PhoenixModelsResampled(np.linspace(minwl, maxwl, nspec))

        psf_model = NIRSpecPSF(observation_template_dm, slit)
        slit_data_template = slit.data
        w2p = WCSW2P(slit.meta.wcs)


        ra0_deg = slit.source_ra
        dec0_deg = slit.source_dec
        source_x0 = slit.source_xpos
        source_y0 = slit.source_ypos

        return cls(spec_model, psf_model, w2p, slit_data_template, 
                   ra0_deg, dec0_deg, source_x0, source_y0)

    @property
    def ra0(self):
        return self._ra0_deg * u.deg
    @property
    def dec0(self):
        return self._dec0_deg * u.deg

    _ARCSEC_TO_DEG = u.arcsec.to(u.deg)
    _CKMS = cnst.c.to_value(param_units[2])
    def make_model(self, model_params, slit_data_template):
            F0, x_slit_offset, y_slit_offset, v, teff, logg, feh = model_params
    
            spec = self.spec_model.make_spectrum(teff, logg, feh)

            raoff = decoff = 0 # might change in the future as a separate parameter?
            ra = self._ra0_deg + raoff*self._ARCSEC_TO_DEG
            dec = self._dec0_deg + decoff*self._ARCSEC_TO_DEG
            wave = self.spec_model.wave_microns*(1 + v/self._CKMS)
    
            xspec, yspec = self.w2p.evaluate_wcs_values(ra, dec, wave)
            xslit = self.x_slit0 + x_slit_offset
            yslit = self.y_slit0 + y_slit_offset
    
            psf_y, psf_x, psf_flux = self.psf_model.evaluate_psf(xslit, yslit, spec.spectral_axis.mean())
    
            return scatter_flux(xspec, yspec, F0*spec.flux.value, 
                                   psf_x, psf_y, psf_flux, 
                                   self.psf_model.oversampling, 
                                   slit_data_template, 
                                   sum_flux=True, 
                                   ret_host_array=False)

    _NORM_FACTOR = 0.5*math.log(2*math.pi)
    def loglike(self, model_params, slit_data, slit_data_unc):
        """
        slit_data and slit_data_unc should be cuda device arrays
        """
        modelflux = self.make_model(model_params, slit_data)

        modelfluxc = cp.asarray(modelflux)
        data = cp.asarray(slit_data)
        unc = cp.asarray(slit_data_unc)
        loglikes = self._NORM_FACTOR - unc - 0.5*((modelfluxc - data)/unc)**2
        return cp.sum(loglikes)