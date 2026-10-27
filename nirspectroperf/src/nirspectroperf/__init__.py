from .build2d import *
from .forward_model import *
from .psf import *
from .spec_model import *
from .wcs import *

__version__ = '0.0.1'

def get_template_notebooks() -> dict:
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


def populate_template_notebook(template_nb_name: str, 
                               output_name: str="{template_nb_name}.ipynb", 
                               overwrite: bool=False, 
                               **parameters) -> str:
    """
    Populate a template notebook with the given parameters and save it as a new notebook.

    Parameters
    ----------
        template_nb_name (str): The name of the template notebook to use.
        output_name (str, optional): The name of the output notebook. Defaults to "{template_nb_name}.ipynb".
        overwrite (bool, optional): Whether to overwrite the output notebook if it already exists. Defaults to False.
        **parameters: Additional parameters to pass to the notebook.

    Returns
    -------
        str: The path to the populated notebook.
    """
    import papermill
    import os

    template_nb_path = get_template_notebooks()[template_nb_name]

    if '{' in output_name and '}' in output_name:
        output_name = output_name.format(template_nb_name=template_nb_name)

    if not overwrite and os.path.exists(output_name):
        raise FileExistsError(f"Output notebook already exists: {output_name}")

    papermill.execute_notebook(template_nb_path, output_name, prepare_only=True, parameters=parameters)

    return output_name
