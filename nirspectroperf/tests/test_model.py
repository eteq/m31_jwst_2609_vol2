from pathlib import Path
import numpy as np

import nirspectroperf
from jwst import datamodels
from matplotlib import pyplot as plt

def test_model_spectra():
    n6791fits = list(Path('../ngc6791_cals').glob('*.fits'))
    dm0 = datamodels.open(n6791fits[0])
    slit = dm0.slits[49]

    model = nirspectroperf.NIRSpecSlitModel.make_from_template(dm0, slit)
    params = [1, 0, 0, 0, 5800, 4.5, 0.0]
    mod = model.make_model(params, slit.data)

    assert mod.shape == slit.data.shape

    modarr = mod.copy_to_host()
    assert np.any(np.isnan(modarr))
    assert not np.all(np.isnan(modarr))

    # data content sanity check
    # confirm the data have a lot of variation along one axis, but only a bit along the other
    qs0 = np.nanquantile(np.nansum(modarr, axis=0), [0.1, 0.5, 0.9])
    assert qs0.std()/qs0.mean() < 0.1

    qs1 = np.nanquantile(np.nansum(modarr, axis=1), [0.1, 0.5, 0.9])
    assert 1 < qs1.std()/qs1.mean() < 2