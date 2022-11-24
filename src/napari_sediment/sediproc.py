import numpy as np
from spectral import open_image

from ._reader import read_spectral

def compute_average_in_roi(file_path, channel_indices, roi, white_path=None):
    """Compute average reflectance in a region of interest (ROI).

    Parameters
    ----------
    file_path : str
        Path to file.
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
        #white_data = img_white.read_bands(self.channel_indices)
        white_av = white_data.mean(axis=0)
        white_max = white_av.max()          

    data_av = np.zeros((len(channel_indices), roi[0][1]-roi[0][0]))
    for ind, ch in enumerate(channel_indices):
        #print(f'channel: {ch}')

        #print('compute')
        data, _ = read_spectral(file_path, bands=[ch],
            row_bounds=(roi[0][0], roi[0][1]),
            col_bounds=(roi[1][0], roi[1][1]),
            )
        #print('done')
        #print('white correct')
        if white_path is not None:
            data = white_max * (data / white_av[:,ind])
        data_av[ind] = data.mean(axis=1)[:,0].copy()
        #print('done')

    return data_av