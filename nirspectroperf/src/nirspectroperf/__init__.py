from .build2d import *
from .forward_model import *
from .psf import *
from .spec_model import *
from .wcs import *

__version__ = '0.0.1'

def get_template_notebooks():
    """
    Get the template notebooks available in the package.

    Returns:
        dict: A dictionary mapping notebook names to their paths/resource objects.
    """
    from importlib import resources

    nbdir = resources.files() / 'template_notebooks'
    assert nbdir.is_dir(), f"Template notebooks directory not found: {nbdir}"

    nbs = [item for item in nbdir.iterdir() 
                if item.is_file() 
                if item.name.endswith('.ipynb')]

    return {nb.name[:-6]: nb for nb in nbs}