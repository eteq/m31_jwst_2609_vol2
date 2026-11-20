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

    # Here we compute the range of PSF subpixel indices that can overlap this
    # output pixel.  A subpixel overlaps if its offset is within the pixel
    # half-width plus the subpixel half-width.  psf_x/psf_y are assumed to be
    # linear in index with spacing 1/oversampling, but may be arbitrarily
    # shifted (e.g. by the source offset in the slit), so the grid origin is
    # read from the arrays rather than assumed to be centered.  

    halfwidth_both = float32(0.5) + float32(0.5) / float32(oversampling)

    x0 = psf_x[0, 0]
    xlower = int(math.floor((x - xsi - halfwidth_both - x0)*oversampling))
    xlower = max(0, xlower)
    xupper = int(math.ceil((x - xsi + halfwidth_both - x0)*oversampling)) + 1
    xupper = min(psf_x.shape[0], xupper)
    xrng = xupper - xlower

    y0 = psf_y[0, 0]
    ylower = int(math.floor((y - ysi - halfwidth_both - y0)*oversampling))
    ylower = max(0, ylower)
    yupper = int(math.ceil((y - ysi + halfwidth_both - y0)*oversampling)) + 1
    yupper = min(psf_y.shape[1], yupper)
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
def compute_influence_kernel(xspec, yspec, startidx, nspec, maxpsf, noncontig):
    x = cuda.blockIdx.x
    y = cuda.threadIdx.x
    maxpsfc = float32(maxpsf)

    startidxi = -1
    lastidxi = -1
    count = 0
    for i, (xs, ys) in enumerate(zip(xspec, yspec)):
        if abs(float32(x) - xs) <= maxpsfc:
            if abs(float32(y) - ys) <= maxpsfc:
                count += 1
                if startidxi == -1:
                    startidxi = i
                lastidxi = i

    startidx[y, x] = startidxi
    nspec[y, x] = count

    # scatter_flux_kernel assumes the influencing spectrum points are contiguous
    # (startidx .. startidx+nspec-1).  Flag any pixel where that's not true.
    # This could end up conflicting across threads, but  because writes only happen
    # to set it to 1, there isn't a conflict
    if count > 0 and lastidxi - startidxi + 1 != count:
        noncontig[0] = 1


def compute_influence(xspec, yspec, maxpsf, slittemplate, check_contiguous=True):
    """
    Computes where the spectrum influences the slit image, assuming a maximum
    PSF size of maxpsf.

    Returns two arrays, `startidx` and `nspec`, which are the starting index of the
    spectrum that influences each pixel, and the number of spectrum points
    that influence each pixel, respectively. Both are device arrays.

    If ``check_contiguous`` is True, raises a ValueError if the spectrum points
    influencing any pixel are not a contiguous run of indices (which
    `scatter_flux_kernel` assumes).  This requires a small device-to-host copy
    and synchronization, so it can be turned off for performance if needed.
    """
    kernelspec_influence = (slittemplate.shape[1], slittemplate.shape[0])

    xspecc = to_cuda_array_if_needed(xspec)
    yspecc = to_cuda_array_if_needed(yspec)

    startidxc = cuda.device_array(slittemplate.shape, dtype=np.int32)
    nspecc = cuda.device_array(slittemplate.shape, dtype=np.int32)
    noncontigc = cuda.to_device(np.zeros(1, dtype=np.int32))

    compute_influence_kernel[kernelspec_influence](xspecc, yspecc, startidxc, nspecc, maxpsf, noncontigc)

    if check_contiguous and noncontigc.copy_to_host()[0]:
        raise ValueError("The spectrum points influencing at least one pixel are "
                         "not contiguous in index, which scatter_flux assumes. "
                         "This probably means the trace isn't monotonic")

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
                 outslitall=None,
                 check_contiguous=True):
    """
    This function takes host-side arrays, and converts them into a form the GPU can use.

    ``check_contiguous`` is passed to `compute_influence`; set it to False to
    skip the (small) cost of verifying the contiguity assumption.
    """
    import cupy as cp
    xspecc = to_cuda_array_if_needed(xspec)
    yspecc = to_cuda_array_if_needed(yspec)
    fluxspecc = to_cuda_array_if_needed(spectrum_fnudnu)

    psf_xc = to_cuda_array_if_needed(psf_x)
    psf_yc = to_cuda_array_if_needed(psf_y)
    psf_fluxc = to_cuda_array_if_needed(psf_flux)

    # A spectrum point influences a pixel if any PSF subpixel overlaps it.  That
    # can happen out to the largest offset in the PSF on either side, plus half
    # a detector pixel (the pixel's own half-width) plus half a subpixel (the
    # subpixel's half-width).
    maxpsf = max(float(np.max(np.abs(psf_x))), float(np.max(np.abs(psf_y)))) + 0.5 + 0.5/oversampling
    startidxc, nspecc = compute_influence(xspecc, yspecc, float(maxpsf), slittemplate,
                                          check_contiguous=check_contiguous)

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