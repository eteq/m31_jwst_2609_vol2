from pathlib import Path
import numpy as np

import nirspectroperf
from jwst import datamodels
from matplotlib import pyplot as plt

def test_wcs():
    n6791fits = list(Path('../ngc6791_cals').glob('*.fits'))
    dm0 = datamodels.open(n6791fits[0])
    slit = dm0.slits[49]
    swcs = slit.meta.wcs

    cw2p = nirspectroperf.ChebyshevW2P.derive_from_wcs(swcs, (2, 2, 7), slit.data.shape)

    sky_allpx, wave_allpx = swcs.pixel_to_world(*np.indices(slit.data.shape)[::-1])
    msk = np.isfinite(sky_allpx.ra.value) & np.isfinite(sky_allpx.dec.value) & np.isfinite(wave_allpx.value)
    masked_vals = [d.value[msk] for d in (sky_allpx.ra, sky_allpx.dec, wave_allpx)]

    xs_true, ys_true = swcs.world_to_pixel_values(*masked_vals)

    xs_approx, ys_approx = cw2p.evaluate_wcs_values(*masked_vals)

    dx = xs_true - xs_approx
    dy = ys_true - ys_approx

    print(dx.mean(), dx.std(), dx.min(), dx.max())
    print(dy.mean(), dy.std(), dy.min(), dy.max())

    assert abs(dx.mean()) < 1e-4
    assert abs(dy.mean()) < 1e-4

    assert dx.std() < 1e-3
    assert dy.std() < 1e-3

    assert max(abs(dx.min()), abs(dx.max())) < 1e-3
    assert max(abs(dy.min()), abs(dy.max())) < 1e-3