import numpy as np
import tifffile
import yaml
from pathlib import Path
import os

import zarr
from ..data_structures.parameters import Param
from ..data_structures.parameters_endmembers import ParamEndMember
from ..data_structures.parameters_plots import Paramplot

def save_mask(mask, filename):

    if not filename.parent.exists():
        os.makedirs(filename.parent, exist_ok=True)
   
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

    main_folder = current_folder.parent
    current_folder = Path(current_folder)

    wr_folders = list(main_folder.glob(f'*{background_text}*'))
    wr_folders = [x for x in wr_folders if x.is_dir()]
    wr_folder_beginnings = [wr.name.split(background_text)[0] for wr in wr_folders]

    wr_folder = None
    for ind, wf in enumerate(wr_folder_beginnings):
        if wf in current_folder.name:
            wr_folder = wr_folders[ind]
            wr_folder_beginning = wr_folder_beginnings[ind]

    if wr_folder is None:
        raise Exception('No white reference folder found')

    white_file_path = list(wr_folder.joinpath('capture').glob('WHITE*.hdr'))[0]
    dark_for_white_file_path = list(wr_folder.joinpath('capture').glob('DARK*.hdr'))[0]
    dark_for_im_file_path = list(current_folder.joinpath('capture').glob('DARK*.hdr'))[0]
    imhdr_path = list(current_folder.joinpath('capture').glob(wr_folder_beginning+'*.hdr'))[0]
    
    return current_folder, wr_folder, white_file_path, dark_for_white_file_path, dark_for_im_file_path, imhdr_path
    


