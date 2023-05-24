import numpy as np
import tifffile
import yaml
from pathlib import Path
from .parameters import Param
from .parameters_indices import ParamIndices

def save_mask(mask, filename):
   
    tifffile.imsave(filename, mask.astype('uint8'))

def load_mask(filename):
    
    mask = tifffile.imread(filename)
    return mask

def load_params_yml(params, file_name='Parameters.yml'):
    
    if not params.project_path.joinpath(file_name).exists():
        raise FileNotFoundError(f"Project {params.project_path} does not exist")

    with open(params.project_path.joinpath(file_name)) as file:
        documents = yaml.full_load(file)
    for k in documents.keys():
        setattr(params, k, documents[k])

    return params

def load_project_params(folder):
    """Load project parameters from yaml file in a given folder."""

    folder = Path(folder)
    params = Param(project_path=folder)
    params = load_params_yml(params)
    
    return params

def load_index_params(folder):
    """Load index parameters from yaml file in a given folder."""

    folder = Path(folder)
    params = ParamIndices(project_path=folder)
    params = load_params_yml(params, file_name='Parameters_indices.yml')
    
    return params


def get_mask_path(export_folder):

    export_folder = Path(export_folder)
    return export_folder.joinpath('mask.tif')



