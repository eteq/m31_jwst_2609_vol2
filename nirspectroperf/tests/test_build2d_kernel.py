"""
Tests for the lowl-level kernel elements of nirspectroperf.build2d

These need either a CUDA GPU or numba's CUDA simulator.  To run on a machine
without a GPU, set ``NUMBA_ENABLE_CUDASIM=1`` (slow, but the problem sizes
here are kept small).
"""
import math

import numpy as np
import pytest
from numba import cuda

from nirspectroperf.build2d import scatter_flux_kernel, compute_influence

pytestmark = pytest.mark.skipif(not cuda.is_available(),
                                reason='needs a CUDA GPU or NUMBA_ENABLE_CUDASIM=1')


def _make_psf(oversampling, npsf, xshift, yshift, sigma=0.7):
    """
    A Gaussian PSF on an oversampled grid, with psf_x/psf_y offset from the
    centered grid by (xshift, yshift) detector pixels - mimicking what
    `SlitPSFModel.evaluate_psf` and the forward model's slit-offset shift do.
    Indexed [x, y].
    """
    xs, ys = np.indices((npsf, npsf))
    psf_x = (xs - (npsf - 1)/2)/oversampling + xshift
    psf_y = (ys - (npsf - 1)/2)/oversampling + yshift
    # make the flux asymmetric in x/y so transposition errors would show up
    psf_flux = np.exp(-0.5*((psf_x/sigma)**2 + (psf_y/(1.5*sigma))**2))
    psf_flux /= psf_flux.sum()
    return psf_x, psf_y, psf_flux


def _overlap(d, oversampling):
    """1d overlap fraction of a subpixel at offset ``d`` with a detector pixel."""
    return np.clip(-oversampling*np.abs(d) + (1 + oversampling)/2, 0, 1)


def _brute_force(xspec, yspec, fluxspec, psf_x, psf_y, psf_flux, oversampling, shape):
    """Reference: sum over *every* PSF subpixel for every spectrum point and pixel."""
    ny, nx = shape
    out = np.zeros(shape)
    for xsi, ysi, fsi in zip(xspec, yspec, fluxspec):
        # [x, i, j] and [y, i, j]
        wx = _overlap(xsi - np.arange(nx)[:, None, None] + psf_x[None], oversampling)
        wy = _overlap(ysi - np.arange(ny)[:, None, None] + psf_y[None], oversampling)
        out += fsi*np.einsum('xij,yij,ij->yx', wx, wy, psf_flux)
    return out


def _run_kernel(xspec, yspec, fluxspec, psf_x, psf_y, psf_flux, oversampling, shape):
    """
    Runs `scatter_flux_kernel` the same way `scatter_flux` does, but without
    needing cupy (so it also works under the CUDA simulator).
    """
    template = np.zeros(shape, dtype=np.float32)
    maxpsf = max(np.max(np.abs(psf_x)), np.max(np.abs(psf_y))) + 0.5 + 0.5/oversampling
    startidxc, nspecc = compute_influence(xspec, yspec, float(maxpsf), template)

    nthreads = math.ceil(nspecc.copy_to_host().max()/32)*32
    outslitall = cuda.to_device(np.zeros(shape + (nthreads,), dtype=np.float32))

    f32 = lambda a: cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    scatter_flux_kernel[shape[::-1], nthreads](f32(xspec), f32(yspec), f32(fluxspec),
                                               f32(psf_x), f32(psf_y), f32(psf_flux),
                                               startidxc, nspecc,
                                               outslitall, oversampling)
    return outslitall.copy_to_host().sum(axis=-1)


@pytest.mark.parametrize('xshift,yshift', [(0, 0),
                                           (0.3, -0.2),
                                           (0.6, 0.45),
                                           (-0.6, -0.7),
                                           (1.1, -1.3)])
def test_scatter_flux_matches_brute_force(xshift, yshift):
    """
    The kernel only loops over a window of PSF subpixels near each output
    pixel.  Check that the window never cuts off subpixels that overlap the
    pixel, including when the PSF grid is shifted relative to the source (as
    happens for a source offset in the slit).
    """
    oversampling = 4
    shape = (10, 20)  # [y, x]

    psf_x, psf_y, psf_flux = _make_psf(oversampling, 20, xshift, yshift)

    # a slightly tilted trace, sampled at non-integer positions
    nspec = 45
    xspec = np.linspace(3.13, 16.71, nspec)
    yspec = 4.37 + 0.05*(xspec - xspec[0])
    fluxspec = 1 + 0.5*np.sin(xspec)

    expected = _brute_force(xspec, yspec, fluxspec, psf_x, psf_y, psf_flux,
                            oversampling, shape)
    result = _run_kernel(xspec, yspec, fluxspec, psf_x, psf_y, psf_flux,
                         oversampling, shape)

    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6*expected.max())
    # all the flux should land on the detector for this setup
    np.testing.assert_allclose(result.sum(), fluxspec.sum(), rtol=1e-4)
