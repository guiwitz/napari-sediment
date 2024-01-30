import numpy as np
import tifffile
import yaml
from pathlib import Path

import zarr
from .parameters.parameters import Param
from .parameters.parameters_endmembers import ParamEndMember
from .parameters.parameters_plots import Paramplot

def save_mask(mask, filename):
   
    tifffile.imsave(filename, mask.astype('uint8'))

def load_mask(filename):
    
    mask = tifffile.imread(filename)
    return mask

def save_image_to_zarr(image, zarr_path):
    """Create a zarr file and stores image in it.
    
    Parameters
    ----------
    image : array
        Image to save. Dims are (bands, rows, cols) or (rows, cols).
    zarr_path : str
        Path to save zarr to.
    """

    if image.ndim == 2:
        chunks = (image.shape[0], image.shape[1])
    elif image.ndim == 3:
        chunks = (1, image.shape[1], image.shape[2])

    im_zarr = zarr.open(zarr_path, mode='w', shape=image.shape,
               chunks=chunks, dtype=image.dtype)
    im_zarr[:] = image


def load_params_yml(params, file_name='Parameters.yml'):
    
    if not Path(params.project_path).joinpath(file_name).exists():
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

def load_endmember_params(folder):
    """Load index parameters from yaml file in a given folder."""

    folder = Path(folder)
    params = ParamEndMember(project_path=folder)
    params = load_params_yml(params, file_name='Parameters_indices.yml')
    
    return params

def load_plots_params(file_path):
    """Load plot parameters from yaml file in a given folder."""

    file_path = Path(file_path)
    params_plots = Paramplot()
    with open(file_path) as file:
        documents = yaml.full_load(file)
    for k in documents.keys():
        setattr(params_plots, k, documents[k])
    
    return params_plots

def get_mask_path(export_folder):

    export_folder = Path(export_folder)
    return export_folder.joinpath('mask.tif')

def get_data_background_path(current_folder, background_text='_WR_'):

    current_folder = Path(current_folder)
    wr_folders = list(current_folder.glob(f'*{background_text}*'))
    wr_folders = [x for x in wr_folders if x.is_dir()]
    other_folders = list(current_folder.glob('*'))
    other_folders = [x for x in other_folders if x.is_dir()]
    if len(wr_folders) == 0:
        raise Exception('No white reference folder found')
    if len(wr_folders) > 1:
        raise Exception('More than one white reference folder found')
    
    wr_folder = wr_folders[0]
    wr_beginning = wr_folder.name.split('WR')[0]
    acquistion_folder = None
    for of in other_folders:
        if wr_folder.name != of.name:
            if wr_beginning in of.name:
                acquistion_folder = of
    if acquistion_folder is None:
        raise Exception('No matching acquisition folder found')

    white_file_path = list(wr_folder.joinpath('capture').glob('WHITE*.hdr'))[0]
    dark_for_white_file_path = list(wr_folder.joinpath('capture').glob('DARK*.hdr'))[0]
    dark_for_im_file_path = list(acquistion_folder.joinpath('capture').glob('DARK*.hdr'))[0]
    imhdr_path = list(acquistion_folder.joinpath('capture').glob(wr_beginning+'*.hdr'))[0]
    
    return acquistion_folder, wr_folder, white_file_path, dark_for_white_file_path, dark_for_im_file_path, imhdr_path
    


