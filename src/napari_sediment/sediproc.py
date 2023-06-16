import numpy as np
from spectral import open_image
from spectral.algorithms import calc_stats
from ._reader import read_spectral
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import skimage
import zarr
from dask.distributed import Client
from tqdm import tqdm
#import pystripe


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

def load_white_dark(white_file_path, dark_for_im_file_path,
                    dark_for_white_file_path=None, channel_indices=None,
                    col_bounds=None, clean_white=False):
    """Load white and dark reference images. In case a separate white reference is used
    (not the one acquired at the same time as the image), the corresponding dark reference
    should be used to correct it. Optionally corrects the white reference by removing rows
    outside of the expected noise range.

    Parameters
    ----------
    white_file_path : str
        Path to white reference image.
    dark_for_im_file_path : str
        Path to dark reference image for image.
    dark_for_white_file_path : str, optional
        Path to dark reference image for white reference. If None, no
        dark reference for white reference is returned.
    channel_indices : list of int, optional
        List of channel indices to load. If None, all channels are loaded.
    col_bounds : tuple, optional
        Tuple of (min, max) column indices to load. If None, all columns are loaded.
    clean_white : bool, optional
        If True, remove rows outside of the expected noise range from the white reference.

    Returns
    -------
    im_white : array
        White reference image. Dims are (rows, cols, bands).
    im_dark : array
        Dark reference image for image. Dims are (rows, cols, bands).
    im_dark_for_white : array
        Dark reference image for white reference. Dims are (rows, cols, bands).
        
    """

    im_white, _ = read_spectral(path=white_file_path, bands=channel_indices, col_bounds=col_bounds)
    im_dark, _ = read_spectral(path=dark_for_im_file_path, bands=channel_indices, col_bounds=col_bounds)
    im_dark_for_white=None
    if dark_for_white_file_path is not None:
        im_dark_for_white, _ = read_spectral(path=dark_for_white_file_path, bands=channel_indices, col_bounds=col_bounds)
    
    if clean_white:
        im_white = clean_white_ref(im_white)

    return im_white, im_dark, im_dark_for_white


def white_dark_correct(data, white_data, dark_for_im_data, dark_for_white_data=None):
    """White and dark reference correction.

    Parameters
    ----------
    data : array
        Data to correct. Dims are (bands, rows, cols).
    white_data : array
        White reference data. Dims are (rows, cols, bands)
    dark_for_im_data : array
        Dark reference data for image. Dims are (rows, cols, bands)
    dark_for_white_data: array
        Dark reference data for white ref. Dims are (rows, cols, bands)
    
    Returns
    -------
    im_corr : array
        Corrected data. Dims are (bands, rows, cols).
    """
    
    data_to_process = [white_data, dark_for_im_data, dark_for_white_data]
    for ind, d in enumerate(data_to_process):
        if d is not None:
            d_av = d.mean(axis=0)
            d_av = np.moveaxis(d_av, 0, 1)
            data_to_process[ind] = d_av
    data = np.moveaxis(data, 0, 1)
    white_av, dark_for_im_av, dark_for_white_av = data_to_process
    if dark_for_white_av is None:
        im_corr = (data - dark_for_im_av) / (white_av - dark_for_im_av)
    else:
        im_corr = (data - dark_for_im_av) / (white_av - dark_for_white_av)
    
    im_corr = np.moveaxis(im_corr, 1,0)
    im_corr[im_corr < 0] = 0
    im_corr = (im_corr * 2**12).astype(np.uint16)

    return im_corr

def clean_white_ref(white_image):
    """Remove noise rows from white ref. Return clean white ref"""

    white_mean = white_image.mean(axis=2)
    # compute mean over columns
    col_means = np.nanmean(white_mean, axis=0)
    col_sdevs = np.nanstd(white_mean, axis=0)

    # create noise threshold
    sdevssub = col_means - (3 * col_sdevs)

    # check which rows are above threshold
    submean = white_mean-sdevssub
    # find rows with no negative number i.e. good samples
    rowpos = np.sum((submean < 0), axis=1) ==0

    # keep good rows
    white_sel = white_image[rowpos,:]

    return white_sel

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
        im_path, white_path, dark_for_im_path, dark_for_white_path, im_zarr,
        zarr_ind, band, background_correction=True, destripe=False,
        ):
    """White dark correction and save to zarr
    
    Parameters
    ----------
    im_path : str
        Path to image to be corrected
    white_path : str
        Path to white image
    dark_for_im_path : str
        Path to dark image for image
    dark_for_white_path : str
        Path to dark image for white ref
    im_zarr : zarr
        Zarr to save corrected image to
    band : int
        Channel to correct
    zarr_ind: int
        Index of zarr to save corrected image to
    background_correction : bool, optional
        Whether to perform white correction. Default is True.
    destripe : bool, optional
        Whether to perform destriping. Default is True.
    
    Returns
    -------
    None
    
    """
    
    im_reg = open_image(im_path)
    white = open_image(white_path)
    dark = open_image(dark_for_im_path)
    dark_white = open_image(dark_for_white_path)

    img_load = im_reg.read_band(band)
    img_white_load = white.read_band(band)
    img_dark_load = dark.read_band(band)
    img_dark_white_load = dark_white.read_band(band)
    
    corrected = img_load.copy()
    if background_correction:
        corrected = white_dark_correct(
            data=img_load[np.newaxis,:,:],
            white_data=img_white_load[:,:,np.newaxis], 
            dark_for_im_data=img_dark_load[:,:,np.newaxis],
            dark_for_white_data=img_dark_white_load[:,:,np.newaxis],
        )[0]
    if destripe:
        import pystripe
        corrected = pystripe.filter_streaks(corrected.T, sigma=[128, 256], level=7, wavelet='db2').T
    
    im_zarr[zarr_ind, :,:] = corrected

    return None

def correct_save_to_zarr(imhdr_path, white_file_path, dark_for_im_file_path,
                         dark_for_white_file_path , zarr_path, band_indices=None,
                         background_correction=True, destripe=True):

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
            imhdr_path, white_file_path,
            dark_for_im_file_path, dark_for_white_file_path,
            z1, ind, c, True, True))
    
    for k in tqdm(range(len(process)), "correcting and saving to zarr"):
        future = process[k]
        out = future.result()
        future.cancel()
        del future

    z1.attrs['metadata'] = {
        'wavelength': list(np.array(img.metadata['wavelength'])[band_indices]),
        'centers': list(np.array(img.bands.centers)[band_indices])
        }

    client.close()


def spectral_clustering(pixel_vectors, dbscan_eps=0.5):
    """Perform spectral clustering on pixel vectors
    
    Parameters
    ----------
    pixel_vectors : array
        Array of pixel vectors. Dims are (n_pixels, n_bands).
    dbscan_eps : float, optional
        Epsilon parameter for DBSCAN. Default is 0.5.
    
    Returns
    -------
    labels : array
        Cluster labels for each pixel.
    """

    pixel_vectors = pixel_vectors.astype(np.float32)
    X = StandardScaler().fit_transform(pixel_vectors)
    dbscan = DBSCAN(eps=dbscan_eps)

    # cluster the three first components
    dbscan.fit(X=X[:,0:3])

    labels = dbscan.labels_

    return labels

def find_index_of_band(band_list, band_value):
    """Find index of band in list of bands"""
    
    band_list = np.array(band_list)
    band_index = np.argmin(np.abs(band_list-band_value))
    
    return band_index

def custom_ppi(X, niters=1000, threshold=0, centered=False):

    if not centered:
        stats = calc_stats(X)
        X = X - stats.mean

    shape = X.shape
    X = X.reshape(-1, X.shape[-1])
    nbands = X.shape[-1]

    counts = np.zeros(X.shape[0], dtype=np.uint32)
    total_ppi = [0]
    for i in range(niters):
        r = np.random.rand(nbands) - 0.5
        r /= np.sqrt(np.sum(r * r))
        s = X.dot(r)
        imin = np.argmin(s)
        imax = np.argmax(s)

        updating = True
        if threshold == 0:
            # Only the two extreme pixels are incremented
            counts[imin] += 1
            counts[imax] += 1
        else:
            # All pixels within threshold distance from the two extremes
            counts[s >= (s[imax] - threshold)] += 1
            counts[s <= (s[imin] + threshold)] += 1
        total_ppi.append(np.sum(counts > 0))
    
    return counts.reshape(shape[:2]), total_ppi
