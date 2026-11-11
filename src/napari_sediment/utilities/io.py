import numpy as np
import tifffile
import yaml
from pathlib import Path
import os
from warnings import warn

import zarr
from ..data_structures.parameters import Param
from ..data_structures.parameters_endmembers import ParamEndMember
from ..data_structures.parameters_plots import Paramplot
from ..data_structures.imchannels import ImChannels

def save_mask(mask, filename):

    if not filename.parent.exists():
        os.makedirs(filename.parent, exist_ok=True)
   
    tifffile.imwrite(filename, mask.astype('uint8'))

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

    if not Path(file_path).exists():
        warn(f"File {file_path} does not exist")
        return None
    
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

def crop_mask_to_bounds(mask, mask_origin, row_bounds, col_bounds):
    """Extract the region [row_bounds, col_bounds] (in full image coordinates)
    from a mask whose top-left corner is located at mask_origin (row, col).
    Parts of the region not covered by the mask are set to 0."""

    # plain ints: bounds may be unsigned numpy ints, which would wrap around when subtracted
    row_bounds = [int(x) for x in row_bounds]
    col_bounds = [int(x) for x in col_bounds]
    mask_origin = [int(x) for x in mask_origin]
    out_shape = (row_bounds[1] - row_bounds[0], col_bounds[1] - col_bounds[0])
    out = np.zeros(out_shape, dtype=mask.dtype)
    r0 = row_bounds[0] - mask_origin[0]
    c0 = col_bounds[0] - mask_origin[1]
    rs, re = max(r0, 0), min(r0 + out_shape[0], mask.shape[0])
    cs, ce = max(c0, 0), min(c0 + out_shape[1], mask.shape[1])
    if re > rs and ce > cs:
        out[rs-r0:re-r0, cs-c0:ce-c0] = mask[rs:re, cs:ce]
    return out

def mask_to_main_roi(mask, row_bounds, col_bounds):
    """Return a mask with the shape of the main roi defined by row_bounds and
    col_bounds. Masks are saved with the main roi shape, but older versions
    could save masks covering the full image, which are cropped here."""

    row_bounds = [int(x) for x in row_bounds]
    col_bounds = [int(x) for x in col_bounds]
    roi_shape = (row_bounds[1] - row_bounds[0], col_bounds[1] - col_bounds[0])
    if mask.shape == roi_shape:
        return mask
    if mask.shape[0] >= row_bounds[1] and mask.shape[1] >= col_bounds[1]:
        return crop_mask_to_bounds(mask, (0, 0), row_bounds, col_bounds)
    warn(f'Mask shape {mask.shape} does not match main roi shape {roi_shape}.')
    return crop_mask_to_bounds(mask, (row_bounds[0], col_bounds[0]), row_bounds, col_bounds)

def get_mask_roi(main_folder, main_roi_index=0):
    """Get mask from a given folder and roi index, with the shape of the main roi."""

    main_folder = Path(main_folder)
    params = load_project_params(main_folder)
    mask_path = get_mask_path(main_folder.joinpath(f'roi_{main_roi_index}'))
    row_bounds, col_bounds = params.get_formatted_col_row_bounds(main_roi_index)
    if mask_path.is_file():
        mask = mask_to_main_roi(load_mask(mask_path), row_bounds, col_bounds)
    else:
        mask = np.zeros((row_bounds[1]-row_bounds[0], col_bounds[1]-col_bounds[0]), dtype=np.uint8)
    
    return mask

def get_rgb_roi(main_folder, main_roi_index=0):
    """Get rgb image from a given folder and roi index."""

    params = load_project_params(main_folder)
    row_bounds, col_bounds = params.get_formatted_col_row_bounds(main_roi_index)
    myimage = ImChannels(imhdr_path=main_folder.joinpath('corrected.zarr'))
    rgb_cube = np.array(myimage.get_image_cube_bands(
        bands=params.rgb, roi=[row_bounds[0], row_bounds[1], col_bounds[0], col_bounds[1]]))
    
    return rgb_cube, myimage
   

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
    
def get_im_main_roi(export_folder, bands, mainroi_index=None):
    """Get bands from a given main roi from an export folder, i.e.
     containing a corrected.zarr file.

    Parameters
    ----------
    export_folder : str
        Path to export folder.
    bands : list of int
        List of bands to get.
    mainroi_index : list of int
        Index of main roi to get. Default is None.
        If None, get full image.

    Returns
    -------
    new_cube : array
        Image cube containing the bands in dimensions (bands, rows, cols).
     
     """
    
    params = load_project_params(folder=export_folder)
    mainrois, subrois, measurementroi = params.get_formatted_rois()

    roi_bounds = None
    if mainroi_index is not None:
        roi_bounds = [mainrois[mainroi_index][:,0].min(),mainrois[mainroi_index][:,0].max(),
                       mainrois[mainroi_index][:,1].min(), mainrois[mainroi_index][:,1].max()]


    if not export_folder.joinpath('corrected.zarr').exists():
        raise FileNotFoundError(f"File {export_folder.joinpath('corrected.zarr')} does not exist")
    
    imagechannels = ImChannels(export_folder.joinpath('corrected.zarr'))
    
    new_cube = imagechannels.get_image_cube(channels=bands,
                roi=roi_bounds)

    return new_cube
    
