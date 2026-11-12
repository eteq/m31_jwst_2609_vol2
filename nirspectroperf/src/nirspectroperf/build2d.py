"""
This module is for constructing the 2d slit images.  This should me the most
computationally expensive part of the forward model, and is therefore
heavily optimized via cuda/numba.
"""

__all__ = ['scatter_flux', 'compute_influence']

import math

import numpy as np
from numba import cuda, float32

@cuda.jit
def scatter_flux_kernel(xspec, yspec, fluxspec,
                        psf_x, psf_y, psf_flux, 
                        startidx, nspec, 
                        outslitall,
                        oversampling):
    """
    This kernel computes the 2d flux by placing the PSF at a series of positions and spreading the flux out according to that psf.

    ``xspec`` and ``yspec`` are the x and y coordinates of the spectrum points.
    ``fluxspec`` is the flux of the spectrum points.
    ``psf_x``, ``psf_y``, are the x and y offset of the psf pixels from the PSF center, in *detector* coordinates (i.e. not-oversampled)
    ``psf_x``, ``psf_y``, and ``psf_flux`` are all indexed [x, y] (axis 0 is x, axis 1 is y).
    ``psf_flux`` is the flux of the psf pixels. The usual assumption is that this sums to 1 (or to whatever fraction of the total flux is contained in the PSF array).
    ``startidx`` and ``nspec`` define the starting index and number of spectrum points influencing each pixel.
    ``outslitall`` is the output 2D slit image array, indexed [y, x, spectrum point]
    like the detector images (unlike the PSF arrays).
    ``oversampling`` is the oversampling factor for the PSF.
    """
    
    pixel_slope = float32(-oversampling)
    pixel_intercept = float32((1+oversampling)/2)

    x = cuda.blockIdx.x
    y = cuda.blockIdx.y
    tidx = cuda.threadIdx.x

    if math.isnan(outslitall[y, x, tidx]):
        return

    if tidx >= nspec[y, x]:
        return
    
    specidx = tidx + startidx[y, x]
    
    xsi = xspec[specidx]
    ysi = yspec[specidx]
    fsi = fluxspec[specidx]

    flux_contrib = float32(0)
    xo = float32(xsi) - float32(x)
    yo = float32(ysi) - float32(y)

    # here we compute the lowest and highest pixel that can be influenced by the PSF.
    # we also clamp the pixel range to be within the PSF array bounds just in case
    npsfx = psf_x.shape[0]
    xlower = int(math.floor((x - xsi - 1/2)*oversampling - 1) + npsfx/2)
    xlower = max(0, xlower)
    xupper = int(math.ceil((x - xsi + 1/2)*oversampling + 2) + npsfx/2)
    xupper = min(npsfx, xupper)
    xrng = xupper - xlower

    npsfy = psf_y.shape[1]
    ylower = int(math.floor((y - ysi - 1/2)*oversampling - 1) + npsfy/2)
    ylower = max(0, ylower)
    yupper = int(math.ceil((y - ysi + 1/2)*oversampling + 2) + npsfy/2)
    yupper = min(npsfy, yupper)
    yrng = yupper - ylower

    for iraw in range(xrng):
        for jraw in range(yrng):
            j = ylower + jraw
            i = xlower + (iraw + tidx) % xrng

            
            xpi = xo + psf_x[i, j]
            ypi = yo + psf_y[i, j]

            ox = pixel_slope * abs(xpi) + pixel_intercept
            oy = pixel_slope * abs(ypi) + pixel_intercept

            if ox <= 0 or oy <= 0:
                continue

            ox = max(float32(0), min(float32(1), ox))
            oy = max(float32(0), min(float32(1), oy))


            flux_contrib += ox*oy*fsi*psf_flux[i, j]
    outslitall[y, x, tidx] = flux_contrib


@cuda.jit
def compute_influence_kernel(xspec, yspec, startidx, nspec, maxpsf):
    x = cuda.blockIdx.x
    y = cuda.threadIdx.x
    maxpsfc = float32(maxpsf)

    startidxi = -1
    count = 0
    for i, (xs, ys) in enumerate(zip(xspec, yspec)):
        if abs(float32(x) - xs) <= maxpsfc:
            if abs(float32(y) - ys) <= maxpsfc:
                count += 1
                if startidxi == -1:
                    startidxi = i
    
    startidx[y, x] = startidxi
    nspec[y, x] = count


def compute_influence(xspec, yspec, maxpsf, slittemplate):
    """
    Computes where the spectrum influences the slit image, assuming a maximum 
    PSF size of maxpsf. 
    
    Returns two arrays, `startidx` and `nspec`, which are the starting index of the
    spectrum that influences each pixel, and the number of spectrum points
    that influence each pixel, respectively. Both are device arays.
    """
    kernelspec_influence = (slittemplate.shape[1], slittemplate.shape[0])

    xspecc = to_cuda_array_if_needed(xspec)
    yspecc = to_cuda_array_if_needed(yspec)

    startidxc = cuda.device_array(slittemplate.shape, dtype=np.int32)
    nspecc = cuda.device_array(slittemplate.shape, dtype=np.int32)

    compute_influence_kernel[kernelspec_influence](xspecc, yspecc, startidxc, nspecc, maxpsf)

    return startidxc, nspecc

def to_cuda_array_if_needed(arr, dtype=np.float32):
    if cuda.is_cuda_array(arr):
        return cuda.as_cuda_array(arr)
    else:
        return cuda.to_device(arr.astype(dtype))


def scatter_flux(xspec, yspec, spectrum_fnudnu, 
                 psf_x, psf_y, psf_flux, 
                 oversampling, 
                 slittemplate, 
                 sum_flux=True, 
                 ret_host_array=False,
                 outslitall=None):
    """
    This function takes host-side arrays, and converts them into a form the GPU can use.
    """
    import cupy as cp
    xspecc = to_cuda_array_if_needed(xspec)
    yspecc = to_cuda_array_if_needed(yspec)
    fluxspecc = to_cuda_array_if_needed(spectrum_fnudnu)

    psf_xc = to_cuda_array_if_needed(psf_x)
    psf_yc = to_cuda_array_if_needed(psf_y)
    psf_fluxc = to_cuda_array_if_needed(psf_flux)

    maxpsf = max(np.max(psf_x), np.max(psf_y)).astype(np.float32).ravel()[0]
    startidxc, nspecc = compute_influence(xspecc, yspecc, float(maxpsf), slittemplate)

    longest_stretch = int(cp.array(nspecc, copy=False).max())
    longest_stretch32 = math.ceil(longest_stretch / 32) * 32   
    assert longest_stretch32 <= 1024, f'Only 1024 threads per block, but longest stretch is {longest_stretch}'
    kernelspec = (slittemplate.shape[::-1], longest_stretch32)

    slitarr = np.zeros_like(slittemplate)
    slitarr[np.isnan(slittemplate)] = np.nan
    if outslitall is None:
        outslitall = cuda.to_device(np.repeat(np.expand_dims(slitarr, axis=-1), kernelspec[-1], axis=-1).astype(np.float32))


    scatter_flux_kernel[kernelspec](xspecc, yspecc, fluxspecc, 
                            psf_xc, psf_yc, psf_fluxc,
                            startidxc, nspecc, 
                            outslitall, oversampling)
    
    if sum_flux:
        out = cuda.as_cuda_array(cp.sum(cp.array(outslitall, copy=False), axis=-1))
    else:
        out = outslitall


    if ret_host_array:
        return out.copy_to_host()
    else:
        return out