import numpy as np
import tifffile
import yaml
import zarr
from dask.distributed import Client
from spectral import open_image
from tqdm import tqdm
from .sediproc import white_dark_correct

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

def correct_channel_to_zarr(im_path, white_path, dark_path, im_zarr, c):
    """White dark correction and save to zarr
    
    Parameters
    ----------
    im_path : str
        Path to image to be corrected
    white_path : str
        Path to white image
    dark_path : str
        Path to dark image
    im_zarr : zarr
        Zarr to save corrected image to
    c : int
        Channel to correct
    
    Returns
    -------
    None
    
    """
    
    im_reg = open_image(im_path)
    white = open_image(white_path)
    dark = open_image(dark_path)

    img_load = im_reg.read_band(c)
    img_white_load = white.read_band(c)
    img_dark_load = dark.read_band(c)
    corrected = white_dark_correct(
        data=img_load[np.newaxis,:,:],
        white_data=img_white_load[:,:,np.newaxis], 
        dark_data=img_dark_load[:,:,np.newaxis]
    )[0]
    
    im_zarr[c, :,:] = corrected

    return None

def save_to_zarr(imhdr_path, white_file_path, dark_file_path, zarr_path):

    img = open_image(imhdr_path)

    samples = img.ncols
    lines = img.nrows
    bands = img.nbands

    z1 = zarr.open(zarr_path, mode='w', shape=(bands, lines,samples),
               chunks=(1, lines, samples), dtype='f8')

    client = Client()
    
    process = []
    for c in range(bands):
        process.append(client.submit(correct_channel_to_zarr, imhdr_path, white_file_path, dark_file_path, z1, c))
    
    for k in tqdm(range(len(process)), "frame segmentation"):
        future = process[k]
        out = future.result()
        future.cancel()
        del future

    client.close()

