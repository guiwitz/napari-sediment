import numpy as np
from spectral import open_image
from sklearn.covariance import EllipticEnvelope
import skimage
import zarr
from dask.distributed import Client
from spectral import open_image
from tqdm import tqdm
import pystripe

from ._reader import read_spectral

def compute_average_in_roi(file_path, channel_indices, roi, white_path=None):
    """Compute average reflectance in a region of interest (ROI).

    Parameters
    ----------
    file_path : str
        Path to file.
    channel_indices : list of int
        List of channel indices to load.
    roi : tuple
        Tuple of (row_bounds, col_bounds) where row_bounds and col_bounds are
        tuples of (min, max).
    bands : list of int, optional
        List of bands to include in average. If None, all bands are included.
    white_path : str, optional
        Path to white reference image. If None, no white reference is applied.

    Returns
    -------
    average : float
        Average reflectance in ROI.
    """
    
    if white_path is not None:
        img_white = open_image(white_path)
        white_data = img_white.read_subregion(row_bounds=[0, img_white.nrows], col_bounds=roi[1], bands=channel_indices)
        white_av = white_data.mean(axis=0)
        white_max = white_av.max()          

    data_av = np.zeros((len(channel_indices), roi[0][1]-roi[0][0]))
    for ind, ch in enumerate(channel_indices):

        data, _ = read_spectral(
            file_path,
            bands=[ch],
            row_bounds=(roi[0][0], roi[0][1]),
            col_bounds=(roi[1][0], roi[1][1]),
            )
        
        if white_path is not None:
            data = white_max * (data / white_av[:,ind])
        data_av[ind] = data.mean(axis=1)[:,0].copy()

    return data_av

def get_rgb_channels(wavelengths, rgb=[640, 545, 460]):
    """Get indices of channels closest to RGB wavelengths.

    Parameters
    ----------
    wavelengths : list of float
        List of wavelengths.
    rgb : list of float, optional
        List of RGB wavelengths. Default is [640, 545, 460].

    Returns
    -------
    rgb_ch : list of int
        List of indices of channels closest to RGB wavelengths.
    """

    rgb_ch = [np.argmin(np.abs(np.array(wavelengths).astype(float) - x)) for x in rgb]
    return rgb_ch

def white_dark_correct(data, white_data, dark_data):
    """White and dark reference correction.

    Parameters
    ----------
    data : array
        Data to correct. Dims are (bands, rows, cols).
    white_data : array
        White reference data. Dims are (rows, cols, bands)
    dark_data : array
        Dark reference data. Dims are (rows, cols, bands)
    
    Returns
    -------
    im_corr : array
        Corrected data. Dims are (bands, rows, cols).
    """
    
    white_av = white_data.mean(axis=0)
    dark_av = dark_data.mean(axis=0)

    data = np.moveaxis(data,0,1)
    white_av = np.moveaxis(white_av,0,1)
    dark_av = np.moveaxis(dark_av,0,1)

    
    im_corr = (data - dark_av) / (white_av - dark_av)
    im_corr = np.moveaxis(im_corr, 1,0)
    im_corr[im_corr < 0] = 0
    im_corr = (im_corr * 2**12).astype(np.uint16)

    return im_corr

def phasor(image_stack, harmonic=1):
    """Compute phasor components from image stack.

    Parameters
    ----------
    image_stack : array
        Image stack. Dims are (bands, rows, cols).
    harmonic : int, optional
        Harmonic to use. Default is 1.
    
    Returns
    -------
    g : array
        G component. Dims are (rows, cols).
    s : array
        S component. Dims are (rows, cols).
    md : array
        Magnitude of the phasor. Dims are (rows, cols).
    ph : array
        Phase of the phasor. Dims are (rows, cols).
    """

    data = np.fft.fft(image_stack, axis=0)
    dc = data[0].real
    # change the zeros to the img average
    dc = np.where(dc != 0, dc, int(np.mean(dc)))
    g = data[harmonic].real
    g /= -dc
    s = data[harmonic].imag
    s /= -dc

    md = np.sqrt(g ** 2 + s ** 2)
    ph = np.angle(data[harmonic], deg=True)
    
    return g, s, md, ph

def fit_1dgaussian_without_outliers(data):
    """Fit a gaussian to data discarding outliers.
    
    Parameters
    ----------
    data : array
        Data to fit.
    
    Returns
    -------
    mean_val : float
        Mean value of data.
    std_val : float
        Standard deviation of data.

    """

    tofilter = np.ravel(data)
    cov = EllipticEnvelope(random_state=0).fit(tofilter[:, np.newaxis])
    std_val = np.sqrt(cov.covariance_)[0,0]
    mean_val = cov.location_[0]

    return mean_val, std_val


def remove_top_bottom(data, std_fact=3, split_min=20):
    """Remove bands around an image where intensity is too high or too low.
    
    Parameters
    ----------
    data : array
        Data to process. Dims are (rows, cols).
    std_fact : float, optional
        Number of standard deviations to use as threshold. Default is 3.
    split_min : int, optional
        Minimum number of consecutive indices to keep region. Default is 20.
    
    """
    # compute projection
    proj = data.mean(axis=1)

    med_val, std_val = fit_1dgaussian_without_outliers(data)

    # keep only points within reasonable range
    sel = (proj < med_val + std_fact * std_val) & (proj > med_val - std_fact * std_val)

    # create indices for projection
    xval = np.arange(len(proj))

    # keep only previously selected indices and check which are consecutive
    # the goal here is to remove indices on within noisy edges. We want to 
    # keep only indices in longer regions belonging to good regions
    steps = np.diff(xval[sel])
    # split series of consecutive indices into groups
    splits = np.split(xval[sel], np.where(steps != 1)[0]+1)
    # keep only splits with more than 10 indices
    long_stretch = [s for s in splits if len(s) > split_min]
    # recover first and last row to keep
    first_index = long_stretch[0][0]
    last_index = long_stretch[-1][-1]

    return first_index, last_index

def remove_left_right(data):
    """Mask vertical image edges where large variations of intensity
    occur either because of background or bad sample structure.

    Parameters
    ----------
    data : array
        Data to process. Dims are (rows, cols).
    
    Returns
    -------
    left_index : int
        Index of left edge.
    right_index : int
        Index of right edge.

    """

    slope = np.diff(np.mean(skimage.filters.gaussian(data,sigma=5), axis=0))
    std_val = np.std(slope[len(slope)//3 : 2*len(slope)//3])
    med_val = np.mean(slope[len(slope)//3 : 2*len(slope)//3])
    std_fact= 5
    sel = (slope < med_val + std_fact * std_val) & (slope > med_val - std_fact * std_val)
    xval = np.arange(len(slope))
    steps = np.diff(xval[sel])
    # split series of consecutive indices into groups
    splits = np.split(xval[sel], np.where(steps != 1)[0]+1)
    # keep only splits with more than 10 indices
    long_stretch = [len(s) for s in splits]
    sel_split = splits[np.argmax(long_stretch)]
    first_index = sel_split[0]
    last_index = sel_split[-1]
    return first_index, last_index

def correct_single_channel(
        im_path, white_path, dark_path, im_zarr,
        zarr_ind, band, white_correction=True, destripe=True,
        ):
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
    white_correction : bool, optional
        Whether to perform white correction. Default is True.
    destripe : bool, optional
        Whether to perform destriping. Default is True.
    
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
    
    corrected = img_load.copy()
    if white_correction:
        corrected = white_dark_correct(
            data=img_load[np.newaxis,:,:],
            white_data=img_white_load[:,:,np.newaxis], 
            dark_data=img_dark_load[:,:,np.newaxis]
        )[0]
    if destripe:
        corrected = pystripe.filter_streaks(corrected.T, sigma=[128, 256], level=7, wavelet='db2').T
    
    im_zarr[zarr_ind, :,:] = corrected

    return None

def correct_save_to_zarr(imhdr_path, white_file_path, dark_file_path,
                         zarr_path, band_indices=None, white_correction=True, destripe=True):

    img = open_image(imhdr_path)

    samples = img.ncols
    lines = img.nrows
    if band_indices is None:
        bands = img.nbands
        band_indices = np.arange(bands)
    else:
        bands = len(band_indices)

    z1 = zarr.open(zarr_path, mode='w', shape=(bands, lines,samples),
               chunks=(1, lines, samples), dtype='u2')#'f8')

    client = Client()
    
    process = []
    for ind, c in enumerate(band_indices):
        process.append(client.submit(
            correct_single_channel,
            imhdr_path, white_file_path, dark_file_path, z1, ind, c, True, True))
    
    for k in tqdm(range(len(process)), "correcting and saving to zarr"):
        future = process[k]
        out = future.result()
        future.cancel()
        del future

    z1.attrs['metadata'] = {
        'wavelength': list(np.array(img.metadata['wavelength'])[band_indices])}

    client.close()
