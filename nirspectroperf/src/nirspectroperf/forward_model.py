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
                 ra0_deg=None, dec0_deg=None, x_slit0=None, y_slit0=None):
        """
        slit_data_template is a 2d array where the nans and dtype will be used 
        to determine the output array shape or something that has a .data 
        attribute that behaves like that

        if slit_data_template is an actual slit object, ra0/dec0/xslit/yslit
        will be inferred from it
        """
        self.spec_model = spec_model
        self.psf_model = psf_model
        self.w2p = w2p

        if ra0_deg is None:
            if hasattr(slit_data_template, 'source_ra'):
                ra0_deg = slit_data_template.source_ra
            else:
                raise ValueError("ra0_deg must be provided or slit_data_template must have source_ra")
        if dec0_deg is None:
            if hasattr(slit_data_template, 'source_dec'):
                dec0_deg = slit_data_template.source_dec
            else:
                raise ValueError("dec0_deg must be provided or slit_data_template must have source_dec")
        if x_slit0 is None:
            if hasattr(slit_data_template, 'source_xpos'):
                x_slit0 = slit_data_template.source_xpos
            else:
                raise ValueError("x_slit0 must be provided or slit_data_template must have source_xpos")
        if y_slit0 is None:
            if hasattr(slit_data_template, 'source_ypos'):
                y_slit0 = slit_data_template.source_ypos
            else:
                raise ValueError("y_slit0 must be provided or slit_data_template must have source_ypos")

        # these are where the source nominally is when the x/y offsets are 0, in ra/dec and slitx/slity
        try:
            self._ra0_deg = ra0_deg.to_value(u.deg)
        except AttributeError:
            self._ra0_deg = ra0_deg
        try:
            self._dec0_deg = dec0_deg.to_value(u.deg)
        except AttributeError:
            self._dec0_deg = dec0_deg

        #TODO: same as the above, but I can't recall if at this point x_slit0 are arcsec or slit coords...
        self.x_slit0 = x_slit0
        self.y_slit0 = y_slit0


        if hasattr(slit_data_template, 'shape'):
            self.slit_data_template = slit_data_template
        elif hasattr(slit_data_template, 'data'):
            self.slit_data_template = slit_data_template.data
        else:
            raise ValueError("slit_data_template must have a 'shape' or 'data' attribute")


    @classmethod
    def make_from_template(cls, observation_template_dm, slit, nspec=4096):
        allwls = slit.meta.wcs.pixel_to_world(*np.indices(slit.data.shape)[::-1])[1]
        minwl = np.nanmin(allwls)
        maxwl = np.nanmax(allwls)
        spec_model = PhoenixModelsResampled(np.linspace(minwl, maxwl, nspec))

        psf_model = NIRSpecPSF(observation_template_dm, slit)
        w2p = WCSW2P(slit.meta.wcs)

        # since we are providing a slit as the data template, it is inferred from the attributes of the slit object

        return cls(spec_model, psf_model, w2p, slit)

    @property
    def ra0(self):
        return self._ra0_deg * u.deg
    @property
    def dec0(self):
        return self._dec0_deg * u.deg

    _ARCSEC_TO_DEG = u.arcsec.to(u.deg)
    _CKMS = cnst.c.to_value(param_units[2])
    def make_model(self, model_params):
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
                                   self.slit_data_template, 
                                   sum_flux=True, 
                                   ret_host_array=False)

    _NORM_FACTOR = 0.5*math.log(2*math.pi)
    def loglike(self, model_params, slit_data, slit_data_unc):
        """
        slit_data and slit_data_unc should be cuda device arrays
        """

        #TODO: be more consistent about the template - right now the hidden assumption here is that the nans match the slit_data.
        modelflux = self.make_model(model_params)
        msk = cp.isfinite(modelflux)

        modelfluxc = cp.asarray(modelflux)
        data = cp.asarray(slit_data)
        unc = cp.asarray(slit_data_unc)
        loglikes = self._NORM_FACTOR - unc - 0.5*((modelfluxc - data)/unc)**2
        return cp.sum(loglikes[cp.isfinite(loglikes)])