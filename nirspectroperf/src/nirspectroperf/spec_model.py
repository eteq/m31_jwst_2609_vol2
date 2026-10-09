"""
This module is for generating 1d spectral models. The main goal is to
do so quickly enough to be part of the inner inference loop.
"""
import os
import re
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np

from astropy import units as u
from astropy.io import fits

from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler

PHOENIX_MODEL_PATH = Path(os.environ.get('PHOENIX_MODEL_PATH', '../../../phoenix/fullgrid'))


class SpectrumModelBase(ABC):
    @property
    @abstractmethod 
    def wave_microns(self):
        """
        Returns a 1d array of wavelengths in microns
        """
        pass

    @abstractmethod
    def make_raw_spectrum(self, teff, logg, feh):
        """
        Returns flux for the spectrum in Jy
        """
        pass

    def make_spectrum(self, teff, logg, feh):
        """
        Returns a specutils Spectrum1D object with units of microns and Jy
        """
        raw_spectrum = self.make_raw_spectrum(teff, logg, feh)
        return Spectrum(spectral_axis=self.wave_microns << u.micron, 
                        flux=raw_spectrum << u.count)

    def resample_spectrum_to_wave(self, newsax, teff, log, feh, cache=True):
        FluxConservingResampler = resampler()
        spectrum = self.make_spectrum(teff, log, feh)
        meandisp = np.mean(np.diff(newsax))  # as a heuristic check for whether newsax is the same
        
        if cache:
            if not hasattr(self, '_resample_cache'):
                self._resample_cache = {}
            key = (teff, log, feh, meandisp)
            if key not in self._resample_cache:
                self._resample_cache[key] = respec
            return self._resample_cache[key]

        respec = FluxConservingResampler(spectrum, newsax)
        if cache:
            self._resample_cache[(teff, log, feh, meandisp)] = respec
        return respec


class PhoenixModelsDirect(SpectrumModelBase):
    param_names = 'raoff, decoff, z, teff, logg, feh'.split(', ')
    _phoenix_wave_microns = None
    _phoenix_paths_by_tefflgz = None

    @staticmethod
    def _load_wave():
        wavefn = [path for path in PHOENIX_MODEL_PATH.glob('*.fits') if 'WAVE' in path.name]
        with fits.open(wavefn[0]) as f:
            wavehdu = f[0]
            ph_wave = wavehdu.data << u.Unit(wavehdu.header['UNIT'])
            PhoenixModelsDirect._phoenix_wave_microns = ph_wave.to_value(u.micron)

    @staticmethod
    def _load_model_paths():
        phoenix_model_paths = [path for path in PHOENIX_MODEL_PATH.glob('*.fits') if 'WAVE' not in path.name]
        teffgz = [re.match(r'lte(\d*)-(.{4})(.{4}).PHOENIX-ACES.*', fn.name).groups() for fn in phoenix_model_paths]
        dct = {(float(tgz[0]), float(tgz[1]), float(tgz[2])):
                p for p, tgz in zip(phoenix_model_paths, teffgz)}
        PhoenixModelsDirect._phoenix_paths_by_tefflgz = dct

    def __init__(self):
        if PhoenixModelsDirect._phoenix_wave_microns is None:
            PhoenixModelsDirect._load_wave()
        if PhoenixModelsDirect._phoenix_paths_by_tefflgz is None:
            PhoenixModelsDirect._load_model_paths()

    @property
    def wave_microns(self):
        return PhoenixModelsDirect._phoenix_wave_microns

    def make_raw_spectrum(self, teff, logg, feh):
        for tgz, path in PhoenixModelsDirect._phoenix_paths_by_tefflgz.items():
            if (teff, logg, feh) == tgz:
                with fits.open(path) as f:
                    fluxhdu = f[0]
                    ph_flux = fluxhdu.data << u.Unit(fluxhdu.header['BUNIT'])
                    return ph_flux.to_value(u.Jy, u.spectral_density(self.wave_microns << u.micron))
        raise ValueError("No matching Phoenix model found")

    def get_available_teff_logg_feh(self):
        return list(PhoenixModelsDirect._phoenix_paths_by_tefflgz.keys())

    