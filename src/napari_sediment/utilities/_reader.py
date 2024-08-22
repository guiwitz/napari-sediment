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
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    #paths = [path] if isinstance(path, str) else path
    # load all files into array
    array, metadata = read_spectral(path)# for _path in paths]
    # stack arrays into single array
    #data = np.squeeze(np.stack(arrays))

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {'name': metadata['wavelength'], 'channel_axis': 2}

    layer_type = "image"  # optional, default is "image"
    array = np.moveaxis(array, 2, 0)
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
        zarr_image = read_hyper_zarr(path)
        metadata = zarr_image.attrs['metadata']
        if bands is None:
            bands = np.arange(zarr_image.shape[0])
        else :
            bands = np.array(bands)
        
        if row_bounds is None:
            row_bounds = (0, zarr_image.shape[1])
        if col_bounds is None:
            col_bounds = (0, zarr_image.shape[2])

        #data = zarr_image.get_orthogonal_selection(
        #    (bands, slice(row_bounds[0], row_bounds[1]), slice(col_bounds[0],col_bounds[1])))
        import dask.array as da
        zarr_image = da.from_zarr(path)
        data = zarr_image[bands, row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]]
            
        data = np.moveaxis(data, 0, 2)
        
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
