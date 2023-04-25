import numpy as np
import tifffile
import yaml

def save_mask(mask, filename):
   
    tifffile.imsave(filename, mask.astype('uint8'))

def load_mask(filename):
    
    mask = tifffile.imread(filename)
    return mask

def load_params_yml(params):
    
    if not params.project_path.joinpath('Parameters.yml').exists():
        raise FileNotFoundError(f"Project {params.project_path} does not exist")

    with open(params.project_path.joinpath('Parameters.yml')) as file:
        documents = yaml.full_load(file)
    for k in documents.keys():
        setattr(params, k, documents[k])

    return params

def get_mask_path(export_folder):

    return export_folder.joinpath('mask.tif')



