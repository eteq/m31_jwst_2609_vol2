__all__ = ['scatter_flux', 'compute_influence']

import math

import numpy as np
from numba import cuda, float32
import cupy as cp

@cuda.jit
def scatter_flux_kernel(xspec, yspec, fluxspec,
                        psf_x, psf_y, psf_flux, 
                        startidx, nspec, 
                        outslitall,
                        oversampling):
    
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

    xlower = int(math.floor((x - xsi - 1/2)*oversampling - 1) + psf_x.shape[0]/2)
    xupper = int(math.ceil((x - xsi + 1/2)*oversampling + 2) + psf_x.shape[0]/2)
    xrng = xupper - xlower
    ylower = int(math.floor((y - ysi - 1/2)*oversampling - 1) + psf_x.shape[1]/2)
    yupper = int(math.ceil((y - ysi + 1/2)*oversampling + 2) + psf_x.shape[1]/2)
    yrng = yupper - ylower

    for iraw in range(xrng):
        for jraw in range(yrng):
            #i = xlower + iraw
            j = ylower + jraw
            i = xlower + (iraw + tidx) % xrng
            #j = ylower + (jraw + tidx) % yrng

            
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

    xspecc = to_cuda_array_if_needed(xspec.astype(np.float32))
    yspecc = to_cuda_array_if_needed(yspec.astype(np.float32))

    startidxc = cuda.device_array(slittemplate.shape, dtype=np.int32)
    nspecc = cuda.device_array(slittemplate.shape, dtype=np.int32)

    compute_influence[kernelspec_influence](xspecc, yspecc, startidxc, nspecc, maxpsf)

    return startidxc, nspecc


def to_cuda_array_if_needed(arr, dtype=np.float32):
    if cuda.is_cuda_array(arr):
        return arr
    else:
        return cuda.to_device(arr.astype(dtype))


def scatter_flux(xspec, yspec, spectrum_fnu, psf_x, psf_y, psf_flux, oversampling, slittemplate, sum_flux=True, ret_host_array=False):
    """
    This function takes host-side arrays, and converts them into a form the GPU can use.
    """
    xspecc = to_cuda_array_if_needed(xspec)
    yspecc = to_cuda_array_if_needed(yspec)
    fluxspecc = to_cuda_array_if_needed(spectrum_fnu)

    psf_xc = to_cuda_array_if_needed(psf_x)
    psf_yc = to_cuda_array_if_needed(psf_y)
    psf_fluxc = to_cuda_array_if_needed(psf_flux)

    maxpsf = max(np.max(psf_x), np.max(psf_y)).astype(np.float32).ravel()
    startidxc, nspecc = compute_influence(xspecc, yspecc, maxpsf)

    longest_stretch = int(cp.array(nspecc, copy=False).max())
    longest_stretch32 = math.ceil(longest_stretch / 32) * 32   
    kernelspec = (slittemplate.shape[::-1], longest_stretch32)

    slitarr = np.zeros_like(slittemplate)
    slitarr[np.isnan(slittemplate)] = np.nan
    outslitall = cuda.to_device(np.repeat(np.expand_dims(slitarr, axis=-1), kernelspec[-1], axis=-1))


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