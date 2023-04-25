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

def correct_single_channel(im_path, white_path, dark_path, im_zarr, zarr_ind, band):
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
    band : int
        Channel to correct
    zarr_ind: int
        Index of zarr to save corrected image to
    
    Returns
    -------
    None
    
    """
    
    im_reg = open_image(im_path)
    white = open_image(white_path)
    dark = open_image(dark_path)

    img_load = im_reg.read_band(band)
    img_white_load = white.read_band(band)
    img_dark_load = dark.read_band(band)
    corrected = white_dark_correct(
        data=img_load[np.newaxis,:,:],
        white_data=img_white_load[:,:,np.newaxis], 
        dark_data=img_dark_load[:,:,np.newaxis]
    )[0]
    
    im_zarr[zarr_ind, :,:] = corrected

    return None

def correct_save_to_zarr(imhdr_path, white_file_path, dark_file_path, zarr_path, band_indices=None):

    img = open_image(imhdr_path)

    samples = img.ncols
    lines = img.nrows
    if band_indices is None:
        bands = img.nbands
        band_indices = np.arange(bands)
    else:
        bands = len(band_indices)

    z1 = zarr.open(zarr_path, mode='w', shape=(bands, lines,samples),
               chunks=(1, lines, samples), dtype='f8')

    client = Client()
    
    process = []
    for ind, c in enumerate(band_indices):
        process.append(client.submit(
            correct_single_channel,
            imhdr_path, white_file_path, dark_file_path, z1, ind, c))
    
    for k in tqdm(range(len(process)), "correcting and saving to zarr"):
        future = process[k]
        out = future.result()
        future.cancel()
        del future

    z1.attrs['metadata'] = {
        'wavelength': list(np.array(img.metadata['wavelength'])[band_indices])}

    client.close()

