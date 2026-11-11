"""
This module is for building a full forward model, including likelihood
evaluation.
"""
__all__ = ['NIRSpecSlitStarModel']

import math
from functools import cached_property

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
    param_units = [u.nJy/u.ct, u.arcsec, u.arcsec, u.km/u.s, u.K, u.dimensionless_unscaled, u.dimensionless_unscaled]
    assert len(param_names) == len(param_units)
    param_name_to_unit = dict(zip(param_names, param_units))

    def __init__(self, spec_model, psf_model, w2p, slit_data_template, 
                 ra0_deg=None, dec0_deg=None, x_slit0=None, y_slit0=None):
        """
        slit_data_template is a 2d array where the nans and dtype will be used 
        to determine the output array shape or something that has a .data 
        attribute that behaves like that

        if slit_data_template is an actual slit object, ra0/dec0/xslit/yslit
        will be inferred from it

        x_slit0/y_slit0 will be interpreted as slit fractions if they are floats, 
        or as angular quantities if they have units.
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

        try:
            self._x_slit0_asec = x_slit0.to_value(u.arcsec)
        except AttributeError:
            # assume its a slit fraction
            self._x_slit0_asec = x_slit0*NIRSpecPSF.msa_slit_width.value
        try:
            self._y_slit0_asec = y_slit0.to_value(u.arcsec)
        except AttributeError:
            # assume its already a slit fraction
            self._y_slit0_asec = y_slit0*NIRSpecPSF.msa_slit_height.value


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
    
    @property
    def x_slit0(self):
        return self._x_slit0_asec * u.arcsec
    @property
    def y_slit0(self):
        return self._y_slit0_asec * u.arcsec

    def slit_offsets_frac(self, x_slit_offset=0, y_slit_offset=0):
        """
        where a slit lands in the slit, expressed as a fraction of the slit width and height

        x/y_slit_offset interpretated the same as evaluate_psf in the psf_model (i.e., arcsec if not a Quantity)
        """
        xa, ya = self.psf_model._xyoffconv(x_slit_offset, y_slit_offset)
        xfrac = (xa + self._x_slit0_asec)/self.psf_model.msa_slit_width.to_value(u.arcsec)
        yfrac = (ya + self._y_slit0_asec)/self.psf_model.msa_slit_height.to_value(u.arcsec)
        return xfrac, yfrac

    _ARCSEC_TO_DEG = u.arcsec.to(u.deg)
    _CKMS = cnst.c.to_value(param_name_to_unit['v'])
    def make_model(self, model_params):
            F0, x_slit_offset, y_slit_offset, v, teff, logg, feh = model_params
    
            spec = self.spec_model.make_spectrum(teff, logg, feh)
            fnudnu = spec.flux.value*self.spec_model.dnu_hz

            raoff = decoff = 0 # might change in the future as a separate parameter?
            ra = self._ra0_deg + raoff*self._ARCSEC_TO_DEG
            dec = self._dec0_deg + decoff*self._ARCSEC_TO_DEG
            wave = self.spec_model.wave_microns*(1 + v/self._CKMS)

            xspec, yspec = self.w2p.evaluate_wcs_values(ra, dec, wave)
            xslit = self._x_slit0_asec + x_slit_offset
            yslit = self._y_slit0_asec + y_slit_offset
    
            psf_x, psf_y, psf_flux = self.psf_model.evaluate_psf(xslit, yslit, spec.spectral_axis.mean())

            pxperasec = self.psf_model.detector_pixelscale.to_value(u.pixel/u.arcsec)
            psf_x = psf_x + pxperasec * x_slit_offset
            psf_y = psf_y + pxperasec * y_slit_offset

            return scatter_flux(xspec, yspec, F0*fnudnu, 
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