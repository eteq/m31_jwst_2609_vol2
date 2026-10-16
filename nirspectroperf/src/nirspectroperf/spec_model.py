"""
This module is for generating 1d spectral models. The main goal is to
do so quickly enough to be part of the inner inference loop.
"""
__all__ = ['PhoenixModelsDirect', 'PhoenixModelsResampled',
           'StellarParametersRescaler', 'BlackbodyResMLPModel',
           'supress_multiple_slash_warning']

import os
import re
import time
import pickle
import warnings
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np

from astropy import units as u
from astropy.io import fits
from astropy import constants as cnst

from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler

import jax
from jax import numpy as jnp
import equinox as eqx
import optax


DEFAULT_PHOENIX_MODEL_PATH = Path(os.environ.get('PHOENIX_MODEL_PATH', '../phoenix/fullgrid'))


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
        meandisp = np.mean(np.diff(newsax))  # as a heuristic check for whether newsax is the same
        resampler = FluxConservingResampler()
        
        if cache:
            if not hasattr(self, '_resample_cache'):
                self._resample_cache = {}
            key = (teff, log, feh, meandisp)
            if key not in self._resample_cache:
                self._resample_cache[key] = resampler(self.make_spectrum(teff, log, feh), newsax)
            return self._resample_cache[key]
        else:
            respec = resampler(self.make_spectrum(teff, log, feh), newsax)
            if cache:
                self._resample_cache[(teff, log, feh, meandisp)] = respec
            return respec


class PhoenixModelsDirect(SpectrumModelBase):
    _phoenix_wave_microns = None
    _phoenix_paths_by_tefflgz = None

    @staticmethod
    def _load_wave(modelpath=DEFAULT_PHOENIX_MODEL_PATH):
        wavefn = [path for path in modelpath.glob('*.fits') if 'WAVE' in path.name]
        with fits.open(wavefn[0]) as f:
            wavehdu = f[0]
            ph_wave = wavehdu.data << u.Unit(wavehdu.header['UNIT'])
            PhoenixModelsDirect._phoenix_wave_microns = ph_wave.to_value(u.micron)

    @staticmethod
    def _load_model_paths(modelpath=DEFAULT_PHOENIX_MODEL_PATH):
        phoenix_model_paths = [path for path in modelpath.glob('*.fits') if 'WAVE' not in path.name]
        teffgz = [re.match(r'lte(\d*)-(.{4})(.{4}).PHOENIX-ACES.*', fn.name).groups() for fn in phoenix_model_paths]
        dct = {(float(tgz[0]), float(tgz[1]), float(tgz[2])):
                p for p, tgz in zip(phoenix_model_paths, teffgz)}
        PhoenixModelsDirect._phoenix_paths_by_tefflgz = dct

    def __init__(self, modelpath=DEFAULT_PHOENIX_MODEL_PATH):
        if PhoenixModelsDirect._phoenix_wave_microns is None:
            PhoenixModelsDirect._load_wave(modelpath)
        if PhoenixModelsDirect._phoenix_paths_by_tefflgz is None:
            PhoenixModelsDirect._load_model_paths(modelpath)

    @property
    def wave_microns(self):
        return PhoenixModelsDirect._phoenix_wave_microns

    def make_raw_spectrum(self, teff, logg, feh):
        key = (teff, logg, feh)
        path = PhoenixModelsDirect._phoenix_paths_by_tefflgz.get(key, None)
        if path is None:
            raise ValueError("No matching Phoenix model found")
        else:
            with fits.open(path) as f:
                fluxhdu = f[0]

                if 'BUNIT' in fluxhdu.header:
                    fluxunit = u.Unit(fluxhdu.header['BUNIT'])
                else:
                    warnings.warn(f"file {path} missing BUNIT, assuiming erg/s/cm^2/cm")
                    fluxunit = u.Unit('erg/s/cm^2/cm')

                return (fluxhdu.data << fluxunit).to_value(u.Jy, u.spectral_density(self.wave_microns << u.micron))

    def get_available_teff_logg_feh(self):
        return list(PhoenixModelsDirect._phoenix_paths_by_tefflgz.keys())

class PhoenixModelsResampled(SpectrumModelBase):
    def __init__(self, new_wave, direct_model=None, use_cache=True):
        super().__init__()

        if isinstance(new_wave, u.Quantity):
            self.new_wave_microns = new_wave.to_value(u.micron)
        else:
            self.new_wave_microns = new_wave

        if direct_model is None:
            direct_model = PhoenixModelsDirect()
        self.direct_model = direct_model

        self.use_cache = use_cache

    @classmethod
    def make_from_target_spacing(cls, target_spacing=0.5*u.angstrom, minwl=0.96690142*u.micron, maxwl=1.38101388*u.micron, **kwargs):
        direct_model = kwargs.get('direct_model', None)
        if direct_model is None:
            direct_model = PhoenixModelsDirect()

        if isinstance(target_spacing, u.Quantity):
            target_spacing_microns = target_spacing.to_value(u.micron)
        else:
            target_spacing_microns = target_spacing

        if isinstance(minwl, u.Quantity):
            minmicrons = minwl.to_value(u.micron)
        else:
            minmicrons = minwl
        if isinstance(maxwl, u.Quantity):
            maxmicrons = maxwl.to_value(u.micron)
        else:
            maxmicrons = maxwl
        assert minmicrons < maxmicrons, 'min must be less than max!'

        ph_wave = direct_model.wave_microns

        msk = (minmicrons < ph_wave) & (ph_wave < maxmicrons)
        resample_factor = int(round(target_spacing_microns/np.max(np.diff(ph_wave[msk]))))
        resample_msk = (minmicrons < ph_wave) & (ph_wave < (maxmicrons+target_spacing_microns))
        cutoff_len = int(np.sum(resample_msk)/resample_factor)*resample_factor

        ph_wave[resample_msk][:cutoff_len].reshape(-1, resample_factor)
        diced = ph_wave[resample_msk][:cutoff_len].reshape(-1, resample_factor)
        new_wave = diced.mean(axis=1) 

        return cls(new_wave, **kwargs)

    def make_all_spectra(self, pool=None, pbar=True):
        """
        Note this always remakes them whether they are cached or not.  
        use_cache only determines if the remade spectra are cached at the end
        """
        params = self.get_available_teff_logg_feh()

        if pbar:
            if pbar is True:
                from tqdm.auto import tqdm
                pbar = tqdm
        else:
            pbar = iter

        use_cache = self.use_cache
        try:
            self.use_cache = False

            def f(paramset):
                return self.make_spectrum(*paramset)


            if pool is None:
                specs = [f(elem) for elem in pbar(params)]
            else:
                specs = list(pbar(pool.imap(f, params), total=len(params)))

            meandispl = [np.mean(np.diff(specs[0].spectral_axis))]
            if use_cache:
                assert len(params) == len(specs)
                self.direct_model._resample_cache = {tuple(list(param) + meandispl): spec for param, spec in zip(params, specs)}

            return params, specs
        finally:
            self.use_cache = use_cache

    @property
    def wave_microns(self):
        return self.new_wave_microns

    def get_available_teff_logg_feh(self):
        return self.direct_model.get_available_teff_logg_feh()

    def make_raw_spectrum(self, teff, logg, feh):
        return self.make_spectrum(teff, logg, feh).flux.value

    def make_spectrum(self, teff, logg, feh):
        return self.direct_model.resample_spectrum_to_wave(self.new_wave_microns << u.micron, teff, logg, feh, self.use_cache)

    def save_cache(self, cachepath):
        if isinstance(cachepath, str):
            cachepath = Path(cachepath)

        # this is not great since it depends on the internals of Spectrum staying the same, but for a quick cache its not the end of the world
        fluxdct = {k:v.flux.value for k,v in self.direct_model._resample_cache.items()}

        with cachepath.open('wb') as f:
            pickle.dump(fluxdct, f, -1)

    def reload_cache(self, cachepath):
        if isinstance(cachepath, str):
            cachepath = Path(cachepath)

        with cachepath.open('rb') as f:
            fluxdct = pickle.load(f)

        assert isinstance(fluxdct, dict)

        if not hasattr(self.direct_model, '_resample_cache'):
            self.direct_model._resample_cache = {}

        # this is not great since it depends on the internals of Spectrum staying the same, but for a quick cache its not the end of the world
        for k, v in fluxdct.items():
            self.direct_model._resample_cache[k] = Spectrum(spectral_axis=self.wave_microns << u.micron, 
                                                            flux=v << u.count)


PLANCK_EXP_CONSTANT = (cnst.h *cnst.c/(cnst.k_B)).to_value(u.K * u.micron)

class StellarParametersRescaler:
    def __init__(self, stellar_parameter_arrs):
        spmx = jnp.max(stellar_parameter_arrs, axis=0)
        self.stellar_parameters_min = jnp.min(stellar_parameter_arrs, axis=0)
        self.stellar_parameters_range = spmx - self.stellar_parameters_min
        
    def rescale_stellar_parameters(self, sparr):
        """
        sparr is expected to have trailing dimension matching the number of stellar parameters (3 in this case)
        """
        return (sparr - self.stellar_parameters_min) / self.stellar_parameters_range

    def unscale_stellar_parameters(self, sparr_scaled):
        """
        sparr_scaled is expected to have trailing dimension matching the number of stellar parameters (3 in this case)
        """
        return sparr_scaled * self.stellar_parameters_range + self.stellar_parameters_min


class BlackbodyResMLPModel(eqx.Module):
    layers: list
    temp_index: int
    stellar_parameters_rescaler: StellarParametersRescaler

    def __init__(self, rkey, 
                       in_size=3,
                       out_size=8945, 
                       temp_index=0, 
                       width_size=2048, 
                       depth=3, 
                       stellar_parameters_rescaler=None,
                       activation=jax.nn.gelu):

        if rkey is None:
            rkey = jax.random.key(time.time_ns())
        keys = jax.random.split(rkey, depth+1)

        self.temp_index = temp_index

        self.layers = [eqx.nn.Linear(in_size, width_size, key=keys[0]), activation]
        for i in range(1, depth):
            self.layers.append(eqx.nn.Linear(width_size, width_size, key=keys[i]))
            self.layers.append(activation)
        self.layers.append(eqx.nn.Linear(width_size, out_size, key=keys[-1]))

        if stellar_parameters_rescaler is None:
            self.stellar_parameters_rescaler = lambda x:x
        else:
            self.stellar_parameters_rescaler = stellar_parameters_rescaler

    @classmethod
    def make_from_templates(cls, stellar_parameters, normalized_stellar_models, rkey, **kwargs):
        kwargs['rkey'] = rkey
        kwargs['in_size'] = stellar_parameters.shape[-1]
        kwargs['out_size'] = normalized_stellar_models.shape[-1]
        kwargs['stellar_parameters_rescaler'] = StellarParametersRescaler(stellar_parameters)

        return cls(**kwargs)


    def __call__(self, x):
        tK = self.stellar_parameters_rescaler.unscale_stellar_parameters(x)[self.temp_index]
        
        for layer in self.layers:
            x = layer(x)

        return x + self.bb_flamb_jax(tK)

    @staticmethod
    def bb_flamb_jax(wave_micron, tK):
        planck_exponent = PLANCK_EXP_CONSTANT / wave_micron
        unscaled =  wave_micron**-5. / (jnp.exp(planck_exponent / tK) - 1.0)
        # a clever scaling might be possible using the wien displacement law,
        # but to match the data, just normalize the same way as the data (divide by the median and half)
        return unscaled/jnp.median(unscaled) * 0.5


    @staticmethod
    def bb_fnu_jax(wave_micron, tK):
        planck_exponent = PLANCK_EXP_CONSTANT / wave_micron
        unscaled =  wave_micron**-3. / (jnp.exp(planck_exponent / tK) - 1.0)
        # a clever scaling might be possible using the wien displacement law,
        # but to match the data, just normalize the same way as the data (divide by the median and half)
        return unscaled/jnp.median(unscaled) * 0.5

    @staticmethod
    def mean_loss(model, xsp, y, delta=1.0, lambmax=2, lambmin=1, s=.3, lthresh=-4):
        huber_term, grad_term, _, gscaling = BlackbodyResMLPModel.loss_pieces(model, xsp, y, delta, lambmax, lambmin, s, lthresh)
        return huber_term + gscaling * grad_term


    @staticmethod
    def loss_pieces(model, xsp, y, delta=1.0, lambmax=2, lambmin=1., s=.3, lthresh=-4):
        pred_y = jax.vmap(model)(xsp)

        # the factor of 2 is to make the loss more directly comparable to MSE loss
        huber_term = 2 * jnp.mean(optax.losses.huber_loss(pred_y, y, delta=delta))
        
        grad_term = jnp.mean((jnp.diff(pred_y, axis=1) - jnp.diff(y, axis=1))**2)

        w = jax.nn.sigmoid((lthresh-jnp.log10(huber_term))/s)

        gscaling = (w*(lambmax - lambmin) + lambmin)
        
        return huber_term, grad_term, w, gscaling

    @classmethod
    def train(cls, stellar_parameters, normalized_stellar_models, batch_size, nepochs, validation_split_fraction, random_key=None, pbar=True):
        tqdm = None
        if pbar is not None:
            from tqdm.auto import tqdm

        # necessary if you're fiddling with some of the epoch/cycles/etc
        jax.clear_caches()

        if random_key is None:
            random_key = jax.random.key(time.time_ns())

        rkmod, rkepoch, rkvsplit = jax.random.split(random_key, 3)

        model = cls.make_from_templates(stellar_parameters=stellar_parameters, 
                                        normalized_stellar_models=normalized_stellar_models, 
                                        rkey=rkmod)
        stellar_parameters_rescaled = model.stellar_parameters_rescaler.rescale_stellar_parameters(stellar_parameters)

        nvalid = int(normalized_stellar_models.shape[0] * validation_split_fraction)
        valididx = jax.random.choice(rkvsplit, normalized_stellar_models.shape[0], (nvalid,))


        validation_normalized_stellar_models = normalized_stellar_models[valididx]
        validation_stellar_parameters_rescaled = stellar_parameters_rescaled[valididx]
        train_normalized_stellar_models = jnp.delete(normalized_stellar_models, valididx, axis=0)
        train_stellar_parameters_rescaled = jnp.delete(stellar_parameters_rescaled, valididx, axis=0)

        nperepoch = int(np.ceil(train_normalized_stellar_models.shape[0] / batch_size))
        ntot = nepochs*nperepoch

        ncycles = 4
        single_schedule = optax.schedules.warmup_cosine_decay_schedule(
            init_value=1e-4,
            peak_value=3e-3,
            warmup_steps=int(ntot*0.1/ncycles),
            decay_steps=int(ntot/ncycles),
            end_value=1e-4
        )
        lr_schedule = optax.schedules.join_schedules([single_schedule]*ncycles, ((np.arange(ncycles-1)+1)*ntot/ncycles).astype(int))

        optim = optax.inject_hyperparams(optax.adamw)(learning_rate=lr_schedule)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        mean_loss = eqx.filter_jit(cls.mean_loss)
        loss_pieces = eqx.filter_jit(cls.loss_pieces)


        @eqx.filter_jit
        def make_step(model, opt_state, xsp, y):
            loss_value, grads = eqx.filter_value_and_grad(mean_loss)(model, xsp, y)
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value

        if tqdm is not None:
            t = tqdm(total=ntot)

        epoch_boundaries = [0]
        vlosses = [mean_loss(model, validation_stellar_parameters_rescaled, validation_normalized_stellar_models)]
        tlosses = []
        lps = []
        lrs = []
        for ei in range(nepochs):
            rki, rkepoch = jax.random.split(rkepoch)
            permorder = jax.random.permutation(rki, jnp.arange(len(train_normalized_stellar_models)))

            for bi in range(nperepoch):
                speci = permorder[bi*batch_size:(bi+1)*batch_size]
                sflux = train_normalized_stellar_models[speci]
                sparams = train_stellar_parameters_rescaled[speci]

                model, opt_state, loss_value = make_step(model, opt_state, sparams, sflux)

                if tqdm is not None:
                    t.update()
                    t.set_description(f'Epoch {ei+1}/{nepochs}, Logloss: {jnp.log10(loss_value):.4f}, Logvloss: {jnp.log10(vlosses[-1]):.4f}, LR: {opt_state.hyperparams["learning_rate"]:.2e}')
                tlosses.append(loss_value)
                lrs.append(opt_state.hyperparams['learning_rate'])

                lps.append(loss_pieces(model, sparams, sflux))

            vlosses.append(mean_loss(model, validation_stellar_parameters_rescaled, validation_normalized_stellar_models))
            epoch_boundaries.append(len(tlosses))

        train_info = {'epoch_boundaries': epoch_boundaries, 'training_loss':tlosses, 'validation_loss':vlosses, 'learning_rates':lrs}

        return model, train_info


def supress_multiple_slash_warning():
    """
    Supresses an annoying warning that is from the slightly incorrect way the phoenix fits files are provided
    """
    warnings.filterwarnings('ignore', ".*'erg/s/cm\\^2/cm' contains multiple slashes.*")