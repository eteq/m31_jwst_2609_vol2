"""
This module is for modeling the PSF of the spectrograph. The main goal is to
do so quickly enough to be part of the inner inference loop.

Note that one needs to set STPSF_PATH envar before using anything that 
depends on stpsf
"""

__all__ = ['NIRSpecPSF']

from abc import ABC, abstractmethod

import numpy as np

from astropy import units as u
from astropy.time import Time


class PSFBase(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def evaluate_psf(self, xslit, yslit, wavelength):
        """
        should return psf_x, psf_y, psf_flux where psf_x/y are in native pixel coordinates
        """
        pass

class NIRSpecPSF(PSFBase):
    msa_slit_width = 0.2 * u.arcsec
    msa_slit_height = 0.46 * u.arcsec

    def __init__(self, 
                    observation_template_dm, 
                    slit, 
                    oversampling=4, 
                    fov=48):
        import stpsf
        
        self.nirspec_psf = nrs = stpsf.NIRSpec()
        # at the time of this writing, the detector position variance is probably not right, or at leat not important
        #nrs.detector = observation_template_dm.meta.instrument.detector
        nrs.filter = observation_template_dm.meta.instrument.filter
        nrs.disperser = observation_template_dm.meta.instrument.grating
        xdet, ydet = slit.meta.wcs.transform('detector', 'sca', *np.indices(slit.data.shape)[::-1])
        nrs.detector_position = (np.mean(xdet), np.mean(ydet))
        nrs.load_wss_opd_by_date(Time(observation_template_dm.meta.exposure.mid_time, format='mjd'))
        self.oversampling = oversampling
        self.fov_pixels = self.fov_arcsec = None
        if u.arcsec.is_equivalent(fov):
            self.fov_arcsec = fov.to_value(u.arcsec)
        elif u.pixel.is_equivalent(fov):
            self.fov_pixels = fov.to_value(u.pixel)
        else:
            self.fov_pixels = fov

        nrs.image_mask = 'Single MSA open shutter'

    def evaluate_psf(self, xslit, yslit, wavelength, whichext='OVERDIST'):
        """
        xslit/yslit can be quantity or fraction of slit width/height, 0 is centered in the slit
        wavelength must be a quantity
        """
        nrs = self.nirspec_psf
        
        if hasattr(xslit, 'unit'):
            x0 = xslit.to_value(u.arcsec)
        else:
            x0 = self.msa_slit_width.to_value(u.arcsec) * xslit

        if hasattr(yslit, 'unit'):
            y0 = yslit.to_value(u.arcsec)
        else:
            y0 = self.msa_slit_height.to_value(u.arcsec) * yslit

        nrs.options.update({'source_offset_x': x0, 'source_offset_y': y0})
        psffits = nrs.calc_psf(monochromatic=wavelength.to_value(u.meter), 
                               oversample=self.oversampling, 
                               fov_arcsec=self.fov_arcsec,
                               fov_pixels=self.fov_pixels,
                               display=False)
        
        psf_flux = psffits[whichext].data

        #TODO: fix!
        psfcen_x = psfcen_y = 0

        ys, xs = np.indices(psf_flux.shape)
        psf_x = (xs - (xs.shape[1]-1)/2 - psfcen_x)/self.oversampling
        psf_y = (ys - (ys.shape[0]-1)/2 - psfcen_y)/self.oversampling

        return psf_x, psf_y, psf_flux
