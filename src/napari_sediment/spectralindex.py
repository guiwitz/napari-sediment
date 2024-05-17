from dataclasses import dataclass, field
import numpy as np
from scipy.signal import savgol_filter
import dask.array as da

from .sediproc import find_index_of_band

@dataclass
class SpectralIndex:
    """
    Class for keeping track of processing parameters.
    
    Parameters
    ---------
    index_name: str
        name of index
    index_type: str
        one of 'Ratio', 'RABD', 'RABA'
    left_band: int
        left band to compute index
    right_band: int
        right band to compute index
    trough_band: int
        trough band to compute index
    numerator_band: int
        numerator band to compute ratio index
    denominator_band: int
        denominator band to compute ratio index
    left_band_default: int
        default left band to compute index
    right_band_default: int
        default right band to compute index
    trough_band_default: int
        default trough band to compute index
    numerator_band_default: int
        default numerator band to compute ratio index
    denominator_band_default: int
        default denominator band to compute ratio index
    index_map: np.ndarray
        index map
    index_projection: np.ndarray
        index projection
    
    """

    index_name: str = None
    index_type: str = None
    left_band: int = None
    right_band: int = None
    middle_band: int = None
    left_band_default: int = None
    right_band_default: int = None
    middle_band_default: int = None
    index_map: np.ndarray = None
    index_proj: np.ndarray = None
    
    
    def __post_init__(self):
        """Use defaults for bands."""

        self.left_band = self.left_band_default
        self.right_band = self.right_band_default
        self.middle_band = self.middle_band_default


def compute_index_RABD(left, trough, right, row_bounds, col_bounds, imagechannels):
    """Compute the index RABD.
    
    Parameters
    ----------
    left: float
        left band
    trough: float
        trough band
    right: float
        right band
    row_bounds: tuple of int
        (row_start, row_end)
    col_bounds: tuple of int
        (col_start, col_end)
    imagechannels: ImageChannels
        image channels object

    Returns
    -------
    RABD: float
        RABD index
    """

    ltr = [left, trough, right]
    # find indices from the end-members plot (in case not all bands were used
    # This is not necessary as bands will not be skipped in the middle of the spectrum
    #ltr_endmember_indices = find_index_of_band(self.endmember_bands, ltr)
    # find band indices in the complete dataset
    ltr_stack_indices = find_index_of_band(imagechannels.centers,ltr)

    # number of bands between edges and trough
    #X_left = ltr_endmember_indices[1]-ltr_endmember_indices[0]
    #X_right = ltr_endmember_indices[2]-ltr_endmember_indices[1]
    X_left = ltr_stack_indices[1]-ltr_stack_indices[0]
    X_right = ltr_stack_indices[2]-ltr_stack_indices[1]

    # load the correct bands
    roi = np.concatenate([row_bounds, col_bounds])
    ltr_cube = imagechannels.get_image_cube(
        channels=ltr_stack_indices, roi=roi)
    ltr_cube = ltr_cube.astype(np.float32)+0.0000001

    # compute indices
    RABD = ((ltr_cube[0] * X_right + ltr_cube[2] * X_left) / (X_left + X_right)) / ltr_cube[1] 
    RABD = np.asarray(RABD, np.float32)
    return RABD

def compute_index_RABA(left, right, row_bounds, col_bounds, imagechannels):
    """Compute the index RABA.
    
    Parameters
    ----------
    left: float
        left band
    right: float
        right band
    row_bounds: tuple of int
        (row_start, row_end)
    col_bounds: tuple of int
        (col_start, col_end)
    imagechannels: ImageChannels
        image channels object

    Returns
    -------
    RABA: float
        RABA index
    """

    ltr = [left, right]
    # find band indices in the complete dataset
    ltr_stack_indices = [find_index_of_band(imagechannels.centers, x) for x in ltr]
    # main roi
    roi = np.concatenate([row_bounds, col_bounds])
    # number of bands between edges and trough
    R0_RN_cube = imagechannels.get_image_cube(channels=ltr_stack_indices, roi=roi)
    R0_RN_cube = R0_RN_cube.astype(np.float32)
    num_bands = ltr_stack_indices[1] - ltr_stack_indices[0]
    line = (R0_RN_cube[1] - R0_RN_cube[0])/num_bands
    RABA_array = None
    for i in range(num_bands):
        Ri = imagechannels.get_image_cube(channels=[ltr_stack_indices[0]+i], roi=roi)
        Ri = Ri.astype(np.float32) + 0.0000001
        if RABA_array is None:
            RABA_array = ((R0_RN_cube[0] + i*line) / Ri[0] ) - 1
        else:
            RABA_array += ((R0_RN_cube[0] + i*line) / Ri[0] ) - 1
    RABA_array = np.asarray(RABA_array, np.float32)
    return RABA_array
    
def compute_index_ratio(left, right, row_bounds, col_bounds, imagechannels):
    """Compute the index ratio.
        
    Parameters
    ----------
    left: float
        left band
    right: float
        right band
    row_bounds: tuple of int
        (row_start, row_end)
    col_bounds: tuple of int
        (col_start, col_end)
    imagechannels: ImageChannels
        image channels object

    Returns
    -------
    ratio: float
        ratio index
    """
    ltr = [left, right]
    # find band indices in the complete dataset
    ltr_stack_indices = [find_index_of_band(imagechannels.centers, x) for x in ltr]
    # main roi
    roi = np.concatenate([row_bounds, col_bounds])
    numerator_denominator = imagechannels.get_image_cube(channels=ltr_stack_indices, roi=roi)
    numerator_denominator = numerator_denominator.astype(np.float32)
    ratio = numerator_denominator[0] / (numerator_denominator[1] + 0.0000001)
    ratio = np.asarray(ratio, np.float32)
    return ratio

def clean_index_map(index_map):

    index_map = index_map.copy()
    index_map[index_map == np.inf] = 0
    percentiles = np.percentile(index_map, [1, 99])
    index_map = np.clip(index_map, percentiles[0], percentiles[1])
    if isinstance(index_map, da.Array):
        index_map = index_map.compute()

    return index_map

def compute_index_projection(index_image, mask, colmin, colmax, smooth_window=None):
    """Compute the projection of the index map.
    
    Parameters
    ----------
    index_map: np.ndarray
        index map
    mask: np.ndarray
        mask
    colmin: int
        minimum column
    colmax: int
        maximum column
    smooth_window: int
        window size for smoothing the projection

    Returns
    -------
    projection: np.ndarray
        projection of the index map
    """
    index_image[mask==1] = np.nan
    proj = np.nanmean(index_image[:,colmin:colmax],axis=1)

    if smooth_window is not None:
        proj = savgol_filter(proj, window_length=smooth_window, polyorder=3)


    return proj

        