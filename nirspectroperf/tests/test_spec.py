import pytest

import numpy as np

from astropy import units as u

from nirspectroperf.spec_model import PhoenixModelsDirect, PhoenixModelsResampled
from specutils import Spectrum

def test_direct_spectrum():

    mods = PhoenixModelsDirect()

    spec = mods.make_spectrum(5800, 4.5, 0.0)

    assert isinstance(spec, Spectrum)

    with pytest.raises(ValueError):
        mods.make_spectrum(5800, 4.5, 0.01)

    assert len(mods.get_available_teff_logg_feh()[0]) == 3

def test_resampled():
    from pathlib import Path
    print(Path('.').absolute())
    newsax = np.linspace(1, 1.3, 1000) << u.micron
    mods = PhoenixModelsResampled(newsax)
    spec = mods.make_spectrum(5800, 4.5, 0.0)

    assert isinstance(spec, Spectrum)
    assert spec.flux.size == 1000
    assert np.sum(spec.flux) > 0