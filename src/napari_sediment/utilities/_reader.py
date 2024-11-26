"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
from spectral import open_image
import zarr
from pathlib import Path
import dask.array as da

def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """

    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".hdr") and not path.endswith(".zarr"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path for a hdr or zarr file and return a layer data tuple. If
    'wavelegnth' is in the metadata, the data will be interpreted as hyperspectral
    data. If 'index_name' is in the metadata, the data will be interpreted as an
    index map. Otherwise, the data will be interpreted as an image.

    Parameters
    ----------
    path : str
        Path to file

    Returns
    -------
    layer_data : list of tuple
        A list of LayerData tuples with the data, metadata, and layer type.

    """
    
    # load data and metadata
    array, metadata = read_spectral(path)
    
    if 'wavelength' in metadata:
        add_kwargs = {'name': metadata['wavelength'], 'channel_axis': 2}
        array = np.moveaxis(array, 2, 0)
    elif 'index_name' in metadata:
        add_kwargs = {'name': metadata['index_name']}

    layer_type = "image"
    return [(array, {}, layer_type)]

def read_spectral(path, bands=None, row_bounds=None, col_bounds=None):
    """Read spectral data from an hdr or zarr file.

    Parameters
    ----------
    path: str
        path to hdr or zarr file
    bands: list of int
        list of bands indices to read
    row_bounds: tuple of int
        (row_start, row_end)
    col_bounds: tuple of int
        (col_start, col_end)
    
    Returns
    -------
    data: ndarray
        spectral data
    metadata: dict
        metadata with keys 'wavelength' (list of str), 'centers' (list of float)
    """

    path = Path(path)
    if path.suffix == '.hdr':
        img = open_image(path)

        metadata = img.metadata
        metadata['centers'] = img.bands.centers

        if bands is None:
            bands = np.arange(0, len(metadata['wavelength']))

        if (row_bounds is None) and (col_bounds is None):
            data = img.read_bands(bands)
        else:
            if row_bounds is None:
                row_bounds = (0, img.nrows)
            if col_bounds is None:
                col_bounds = (0, img.ncols)

            data = img.read_subregion(row_bounds=row_bounds, col_bounds=col_bounds, bands=bands)

    elif path.suffix == '.zarr':
        zarr_image = zarr.open(path, mode='r')

        if 'metadata' in zarr_image.attrs:
            metadata = zarr_image.attrs['metadata']

            if 'index_name' in zarr_image.attrs['metadata']:
                data = np.array(zarr_image)
            
            elif 'wavelength' in zarr_image.attrs['metadata']:
                if bands is None:
                    bands = np.arange(zarr_image.shape[0])
                else :
                    bands = np.array(bands)
                
                if row_bounds is None:
                    row_bounds = (0, zarr_image.shape[1])
                if col_bounds is None:
                    col_bounds = (0, zarr_image.shape[2])

                zarr_image = da.from_zarr(path)
                data = zarr_image[bands, row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]]
                    
                data = np.moveaxis(data, 0, 2)
            else:
                data = np.array(zarr_image)
        else:
            data = np.array(zarr_image)
            metadata = {}
        
    return data, metadata

def get_rgb_index(metadata=None, path=None, red=640, green=545, blue=460):
    
    if metadata is None:
        if path is None:
            raise ValueError('Either metadata or path must be provided')
        else:
            img = open_image(path)
            metadata = img.metadata

    rgb = [red, green, blue]
    rgb_ch = [np.argmin(np.abs(np.array(metadata['wavelength']).astype(float) - x)) for x in rgb]
    rgb_wl = [metadata['wavelength'][x] for x in rgb_ch]

    return rgb_ch, rgb_wl

def read_hyper_zarr(zarr_path):

    hyperzarr = zarr.open(zarr_path, mode='r')
    return hyperzarr
