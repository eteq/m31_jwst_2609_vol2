"""
This module is for modeling the PSF of the spectrograph. The main goal is to
do so quickly enough to be part of the inner inference loop.

Note that one needs to set STPSF_PATH envar before using anything that 
depends on stpsf
"""

__all__ = ['NIRSpecPSF']

import time
from abc import ABC, abstractmethod

import numpy as np

from astropy import units as u
from astropy.time import Time

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

class PSFBase(ABC):
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

        self.last_header = None

    def evaluate_psf(self, xslitoff, yslitoff, wavelength, whichext='OVERDIST'):
        """
        xslit/yslit can be quantity or fraction of slit width/height, 0 is centered in the slit
        wavelength must be a quantity
        """
        nrs = self.nirspec_psf
        
        if hasattr(xslitoff, 'unit'):
            x0 = xslitoff.to_value(u.arcsec)
        else:
            x0 = self.msa_slitoff_width.to_value(u.arcsec) * xslitoff

        if hasattr(yslitoff, 'unit'):
            y0 = yslitoff.to_value(u.arcsec)
        else:
            y0 = self.msa_slit_height.to_value(u.arcsec) * yslitoff

        nrs.options.update({'source_offset_x': x0, 'source_offset_y': y0})
        psffits = nrs.calc_psf(monochromatic=wavelength.to_value(u.meter), 
                               oversample=self.oversampling, 
                               fov_arcsec=self.fov_arcsec,
                               fov_pixels=self.fov_pixels,
                               display=False)
        
        psf_flux = psffits[whichext].data
        self.last_header = psffits[whichext].header

        psfcen_x = x0
        psfcen_y = y0

        ys, xs = np.indices(psf_flux.shape)
        psf_x = (xs - (xs.shape[1]-1)/2 - psfcen_x)/self.oversampling
        psf_y = (ys - (ys.shape[0]-1)/2 - psfcen_y)/self.oversampling

        return psf_x, psf_y, psf_flux

    def make_psf_points(self, n, wavelength, whichext='OVERDIST', slit_fraction=1, pbar=True):
        psfxoffs = (np.random.rand(n)-0.5)*self.msa_slit_width*slit_fraction
        psfyoffs = (np.random.rand(n)-0.5)*self.msa_slit_height*slit_fraction

        if pbar:
            if pbar is True:
                from tqdm.auto import tqdm as pbar
        else:
            pbar = lambda x: x

        psfress = [self.evaluate_psf(x, y, wavelength, whichext)
                for x, y in pbar(list(zip(psfxoffs, psfyoffs)))]
        psfs = [p[2] for p in psfress]
        xpx, ypx = psfress[0][:2]
        
        return np.array(psfs), np.array(xpx), np.array(ypx), np.array(psfxoffs), np.array(psfyoffs)

    def make_psf_grid(self, n, wavelength, whichext='OVERDIST', slit_fraction=1, pbar=True):
        nperax = int(n**0.5)
        if nperax != n**0.5:
            raise ValueError("n must be a perfect square")

        halfw = 0.5*self.msa_slit_width
        psfxs = np.linspace(-halfw * slit_fraction, halfw * slit_fraction, nperax)
        halfh = 0.5*self.msa_slit_height
        psfys = np.linspace(-halfh * slit_fraction, halfh * slit_fraction, nperax)

        if pbar:
            if pbar is True:
                from tqdm.auto import tqdm as pbar
        else:
            pbar = lambda x: x

        psfress = []
        xs = []
        ys = []
        t = pbar(total=n)
        for x in psfxs:
            for y in psfys:
                psfress.append(self.evaluate_psf(x, y, wavelength, whichext))
                xs.append(x)
                ys.append(y)
                t.update()

        psfs = [p[2] for p in psfress]
        xpx, ypx = psfress[0][:2]

        return np.array(psfs), np.array(xpx), np.array(ypx), np.array(xs), np.array(ys)


class GaborLayer(eqx.Module):
    w0: float
    s0: float
    lin1: eqx.nn.Linear
    #lin2: eqx.nn.Linear
    
    def __init__(self, nin, nout, w0=20, s0=10, rkey=None,trainable0s=True):
        if rkey is None:
            rkey = jax.random.PRNGKey(0)

        rk1, = jax.random.split(rkey, 1)
        self.lin1 = eqx.nn.Linear(nin, nout, key=rk1)

        if trainable0s:
            self.w0 = jnp.array(w0, dtype=jnp.float32)
            self.s0 = jnp.array(s0, dtype=jnp.float32)
        else:
            self.w0 = w0
            self.s0 = s0

    def __call__(self, x):
        x1 = self.lin1(x)

        return self.nonlinearity(x1)
    
class CosGaborLayer(GaborLayer):
    def nonlinearity(self, x1):
        return jnp.cos(self.w0 * x1) * jnp.exp(-(x1 * self.s0)**2)

class ComplexGaborLayer(GaborLayer):
    def nonlinearity(self, x1):
        t1 = x1 * (self.w0 * 1j)

        z = x1 * self.s0

        # these seem like they should be the same but the last one seems to handle corner cases better
        #t2 = jnp.abs(z)**2
        #t2 = z * z.conj()
        t2 = jnp.real(z)**2 + jnp.imag(z)**2

        return jnp.exp(t1-t2)


class PSFWIRE(eqx.Module):
    layers: list

    def __init__(self, n_in_coo,
                       hidden_size=256, 
                       num_hidden_layers=2, 
                       final_activation=jax.nn.sigmoid,
                       rkey=None,
                       complex=True,
                       w0=20, s0=10,
                       trainable_activations=False):
        if rkey is None:
            rkey = jax.random.PRNGKey(0)

        rkeys = jax.random.split(rkey, num_hidden_layers+2)

        if complex:
            layers = [ComplexGaborLayer(n_in_coo, hidden_size, rkey=rkeys[0],
                                    w0=w0, s0=s0, trainable0s=trainable_activations)]
            for i in range(num_hidden_layers):
                layers.append(ComplexGaborLayer(hidden_size, hidden_size, rkey=rkeys[i+1],
                                            w0=w0, s0=s0, trainable0s=trainable_activations))

            layers.append(eqx.nn.Linear(hidden_size, 1, key=rkeys[-1]))
            layers.append(jnp.real)
        else:
            layers = [CosGaborLayer(n_in_coo, hidden_size, rkey=rkeys[0],
                                    w0=w0, s0=s0, trainable0s=trainable_activations)]
            for i in range(num_hidden_layers):
                layers.append(CosGaborLayer(hidden_size, hidden_size, rkey=rkeys[i+1],
                                            w0=w0, s0=s0, trainable0s=trainable_activations))

            layers.append(eqx.nn.Linear(hidden_size, 1, key=rkeys[-1]))
        layers.append(final_activation)

        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()

class CoordinateNormalizer:
    def __init__(self, coord):
        if coord is None:
            self.min = None
            self.ptp = None
        else:
            self.min = jnp.min(coord, axis=-1)
            self.ptp = jnp.ptp(coord, axis=-1)

    def normalize(self, coord):
        return ((coord.T - self.min)/self.ptp).T * 2 - 1
    
    def denormalize(self, norm_coord):
        return ((norm_coord.T + 1)/2 * self.ptp + self.min).T

    def split(self, idxs1, idxs2):
        min1 = self.min[np.array(idxs1)]
        min2 = self.min[np.array(idxs2)]
        ptp1 = self.ptp[np.array(idxs1)]
        ptp2 = self.ptp[np.array(idxs2)]

        n1 = CoordinateNormalizer(None)
        n1.min = min1
        n1.ptp = ptp1

        n2 = CoordinateNormalizer(None)
        n2.min = min2
        n2.ptp = ptp2

        return n1, n2


class NIRSpecPSFINREmulator(PSFBase):
    def __init__(self, trained_model, coordinate_normalizer, xygrid, normalize_flux=True):
        self.trained_model = trained_model

        self.xygrid = xygrid

        npx, noff = coordinate_normalizer.split([0, 1], [2, 3])
        self.coordinate_normalizer = npx
        self.offset_normalizer = noff

        if len(coordinate_normalizer.min) == 5:
            _, nwave = coordinate_normalizer.split([0, 1, 2 ,3], [4,])
            self.wavelength_normalizer = nwave

        self._vmapped_trained_model = jax.vmap(self.trained_model)

        # here we pre-prepare an input array.  The first two elements of the first dimension are
        # pre-populated with the pixel coordinates since those don't change. The latter two 
        # elements are populated at evaluation time
        uxygrid = jnp.array(self.coordinate_normalizer.normalize(np.array(self.xygrid)))
        self._ingrid = jnp.concatenate([uxygrid, jnp.zeros((2, *uxygrid.shape[1:]))], axis=0)

        self.normalize_flux = normalize_flux

        self._jitted_eval = None

    def evaluate_psf(self, xslitoff, yslitoff, wavelength):
        if self._jitted_eval is None:
            self._jitted_eval = jax.jit(self._do_eval)

        return self._jitted_eval(xslitoff, yslitoff, wavelength)

    def _do_eval(self, xslitoff, yslitoff, wavelength):
        # we currently ignore wavelength since the variation is basically nil

        xyoffs = jnp.array([xslitoff, yslitoff])
        uxo, uyo = self.offset_normalizer.normalize(xyoffs)

        inarr = self._ingrid
        inarr = inarr.at[2].set(uxo)
        inarr = inarr.at[3].set(uyo)

        psf = self._vmapped_trained_model(inarr.reshape(4, -1).T).reshape(inarr.shape[1:])

        if self.normalize_flux:
            psf = psf / psf.sum()
        return psf
        
    @classmethod
    def train(cls, psfarrs, psfxoffs, psfyoffs, batch_size, nepochs, 
              ncycles, minlr, peaklr, warmup_fraction,
              rkey, xypx=None, 
              loss_meantologweight='progressive', loss_eps=0,
              pbar=True, **inrkwargs):
        if 'n_in_coo' in inrkwargs and inrkwargs['n_in_coo'] != 4:
            raise ValueError("n_in_coo should not be specified in inrkwargs")
        inrkwargs['n_in_coo'] = 4

        inrkwargs.setdefault('hidden_size', 128)
        inrkwargs.setdefault('num_hidden_layers', 2)
        inrkwargs.setdefault('s0', 10)
        inrkwargs.setdefault('w0', 20)
        
        model = PSFWIRE(**inrkwargs)

        psfarrspall = psfarrs.reshape(psfarrs.shape[0], -1)

        if xypx is None:
            xpx, ypx = jnp.indices(psfarrs.shape[1:])
        else:
            xpx, ypx = xypx

        xpxall = jnp.broadcast_to(xpx.ravel(), psfarrspall.shape)
        ypxall = jnp.broadcast_to(ypx.ravel(), psfarrspall.shape)

        psfxall = jnp.broadcast_to(psfxoffs[..., None], psfarrspall.shape)
        psfyall = jnp.broadcast_to(psfyoffs[..., None], psfarrspall.shape)

        xs = jnp.stack((xpxall.ravel(), ypxall.ravel(), psfxall.ravel(), psfyall.ravel()), axis=-1).astype(float)
        xnorm = CoordinateNormalizer(xs.T)
        
        xs = xnorm.normalize(xs.T).T
        ys = jnp.array(psfarrspall.ravel()/psfarrspall.max())

        def mloss(model, x, y, wl=0.5, eps=0):
            yp = jax.vmap(model)(x)
            yt = y

            lterm = jnp.log((yp/(yt + eps) + yt/(yp + eps))/2)/17.7
            mseterm = (yt-yp)**2
            losses = wl * lterm + (1 - wl) * mseterm

            return np.mean(losses)

        @eqx.filter_jit
        def make_step(model, opt_state, x, y, opt_updater, wl, eps):
            loss, grads = eqx.filter_value_and_grad(mloss)(model, x, y, wl=wl, eps=eps)
            updates, opt_state = opt_updater(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        ntot = nepochs
        single_schedule = optax.schedules.warmup_cosine_decay_schedule(
                init_value=minlr,
                peak_value=peaklr,
                warmup_steps=int(ntot*warmup_fraction/ncycles),
                decay_steps=int(ntot/ncycles),
                end_value=minlr
            )
        lr_schedule = optax.schedules.join_schedules([single_schedule]*ncycles, ((np.arange(ncycles-1)+1)*ntot/ncycles).astype(int))
        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule)

        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        losses = jnp.ones((nepochs,)) * jnp.nan
        if pbar:
            if pbar is True:
                from tqdm.auto import tqdm
                pbar = tqdm
            else:
                try: 
                    f = float(pbar)
                except:
                    pass # assume it failed and continue
                pbar = lambda x: PrintFracPbar(x, printfrac=f)
        else:
            pbar = lambda x:x

        pb = pbar(range(nepochs))
        lrs = []
        for i in pb:
            rkey, subkey = jax.random.split(rkey)
            batch_idx = jax.random.choice(subkey, ys.size, shape=(batch_size,), replace=False)#, p=sampleprobs)
            xi = xs[batch_idx]
            yi = ys[batch_idx]
            if loss_meantologweight == 'progressive':
                fraccycle = (i / (nepochs-1) * ncycles )% 1
                wls = (np.ones(batch_size)*(1-fraccycle)) 
                model, opt_state, loss_value = make_step(model, opt_state, xi, yi, optimizer.update, 
                                                        wl=wls, eps=loss_eps)
            else:
                model, opt_state, loss_value = make_step(model, opt_state, xi, yi, optimizer.update, 
                                                        wl=loss_meantologweight, eps=loss_eps)

            losses = losses.at[i].set(loss_value)
            lrs.append(opt_state.hyperparams['learning_rate'])
            if hasattr(pb, 'set_postfix'):
                pb.set_postfix({"loss": losses[i]})

        train_info = {
            'losses': losses,
            'learning_rates': lrs
        }
        return cls(model, xnorm, (xpx, ypx)), train_info

class PrintFracPbar:
    def __init__(self, wrapped, printfrac=.1, total=None):
        self.wrapped = wrapped
        self.printfrac = printfrac

        self.current = None
        self.lenwrapped = total
        self.starttime = None

    def __iter__(self):
        if self.current is None:
            lenwrapped = len(self.wrapped)
            new = self.__class__(iter(self.wrapped), self.printfrac, total=lenwrapped)
            new.current = 0
            return new
        else:
            return self

    def __next__(self):
        if self.starttime is None:
            self.starttime = time.time()
            
        nextelem = next(self.wrapped) #raises stop when done
        et = time.time()

        if self.current % int(self.lenwrapped * self.printfrac) == 0:
            if self.current != 0:
                dt = et - self.starttime
                currfrac = self.current / self.lenwrapped
                remaining_time = (dt/currfrac) * (1-currfrac)
                print(f'On iter {self.current} of {self.lenwrapped} ({currfrac:.3%}). Time elapsed:{dt:.5} seconds. Estimated remaining time:{remaining_time:.5} seconds.')

        self.current += 1
        return nextelem