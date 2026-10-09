import pytest

import numpy as np

from astropy import units as u

from pathlib import Path
from nirspectroperf.psf import NIRSpecPSF
from jwst import datamodels


def test_nirspec_psf():
    n6791fits = list(Path('../ngc6791_cals').glob('*.fits'))
    dm0 = datamodels.open(n6791fits[0])
    slit = dm0.slits[49]
    
    nrs_psf = NIRSpecPSF(observation_template_dm=dm0, 
                            slit=slit, 
                            oversampling=4, 
                            fov=50)

    px, py, psf = nrs_psf.evaluate_psf(0.0, 0.0, 1.1 * u.micron)
    assert isinstance(psf, np.ndarray)
    assert psf.shape == (50*4, 50*4)

    assert isinstance(px, np.ndarray)
    assert isinstance(py, np.ndarray)
    assert psf.shape == px.shape == py.shape