from pathlib import Path
from warnings import warn
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import dask.array as da
import yaml
import dask.array as da
import zarr

from .sediproc import find_index_of_band
from ..data_structures.spectralindex import SpectralIndex


def built_in_indices():
    """Return a dictionary of built-in SpectralIndex objects.
    In the future, this could be replaced by a mechanism to load indices from a file.
    """
    
    index_def = {
            'RABD510': [470, 510, 530],
            'RABD660670': [590, 665, 730],
        }
    index_collection = {}
    for key, value in index_def.items():
        index_collection[key] = SpectralIndex(index_name=key,
                            index_type='RABD',
                            left_band_default=value[0],
                            middle_band_default=value[1],
                            right_band_default=value[2]
                            )
        
    index_def = {
        'RABD510norm': [470, 510, 530],
        'RABD660670norm': [590, 665, 730],
    }
    for key, value in index_def.items():
        index_collection[key] = SpectralIndex(index_name=key,
                            index_type='RABDnorm',
                            left_band_default=value[0],
                            middle_band_default=value[1],
                            right_band_default=value[2]
                            )
        
    index_def = {
        'RABA410560': [410, 560],
    }
    for key, value in index_def.items():
        index_collection[key] = SpectralIndex(index_name=key,
                            index_type='RABA',
                            left_band_default=value[0],
                            right_band_default=value[1]
                            )
        
    index_def = {
        'R590R690': [590, 690],
        'R660R670': [660, 670]
    }
    for key, value in index_def.items():
        index_collection[key] = SpectralIndex(index_name=key,
                            index_type='Ratio',
                            left_band_default=value[0],
                            right_band_default=value[1]
                            )
        
    index_collection['RMean'] = SpectralIndex(index_name='RMean',
                            index_type='RMean',
                            left_band_default=300,
                            right_band_default=900
                            )
    
    return index_collection


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

def compute_index_RMean(left, right, row_bounds, col_bounds, imagechannels):
    """Compute the index RMean.
    
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
    RMean: float
        RMean index
    """
    if left is None:
        ltr = [imagechannels.centers[0], imagechannels.centers[-1]]
    else:
        ltr = [left, right]
    # find band indices in the complete dataset
    ltr_stack_indices = [find_index_of_band(imagechannels.centers, x) for x in ltr]
    # main roi
    roi = np.concatenate([row_bounds, col_bounds])
    # number of bands between edges and trough
    R0_RN_cube = imagechannels.get_image_cube(channels=np.arange(ltr_stack_indices[0], ltr_stack_indices[1]+1), roi=roi)
    if R0_RN_cube.dtype == np.uint16:
        # if data saved as integer, normalize here
        R0_RN_cube = R0_RN_cube / 4096
    R0_RN_cube = R0_RN_cube.astype(np.float32)
    RMean = np.mean(R0_RN_cube, axis=0)
    RMean = np.asarray(RMean, np.float32)

    return RMean

def compute_index_RABDnorm(left, trough, right, row_bounds, col_bounds, imagechannels):

    rabd = compute_index_RABD(left, trough, right, row_bounds, col_bounds, imagechannels)
    rmean = compute_index_RMean(None, None, row_bounds, col_bounds, imagechannels)
    rabd_norm = rabd / rmean
    return rabd_norm

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

def create_index(index_name, index_type, boundaries):
    """Create a new index SpectralIndex object.
    
    Parameters
    ----------
    index_name: str
        name of the index
    index_type: str
        type of the index, one of RABD, RABA, Ratio, RMean, RABDnorm
    boundaries: list
        list of bands
    
    Returns
    -------
    new_index: SpectralIndex
        new index object

    """
    
    if index_type in ['RABD', 'RABDnorm']:
        new_index = SpectralIndex(index_name=index_name,
                            index_type=index_type,
                            left_band_default=boundaries[0],
                            middle_band_default=boundaries[1],
                            right_band_default=boundaries[2],
                            )
        
    elif index_type in ['RABA', 'Ratio', 'RMean']:
        new_index = SpectralIndex(index_name=index_name,
                            index_type=index_type,
                            left_band_default=boundaries[0],
                            right_band_default=boundaries[1],
                            )
    else:
        raise ValueError('Index type not recognized.')
        
    return new_index

def export_index_series(index_series, file_path):
    """Export the index series to a yml file.
    
    Parameters
    ----------
    index_series: dict
        dictionary of SpectralIndex objects
    file_path: str or Path
        file where to save the index series
    
    """

    index_series = [x.dict_spectral_index() for key, x in index_series.items()]
    index_series = {'index_definition': index_series}
    file_path = Path(file_path)
    if file_path.suffix !='.yml':
        file_path = file_path.with_suffix('.yml')
    with open(file_path, "w") as file:
        yaml.dump(index_series, file)

def compute_index(spectral_index, row_bounds, col_bounds, imagechannels):
    """Given a spectral indef definition and image, compute an index map.
    
    Parameters
    ----------
    spectral_index: SpectralIndex
        index object
    row_bounds: tuple of int
        (row_start, row_end)
    col_bounds: tuple of int
        (col_start, col_end)
    imagechannels: ImageChannels
        image channels object
    
    Returns
    -------
    computed_index: 2D np.ndarray
        computed index map
    
    """

    funs3 = {'RABDnorm': compute_index_RABDnorm, 'RABD': compute_index_RABD}
    
    funs2 = {'RABA': compute_index_RABA,
            'Ratio': compute_index_ratio, 'RMean': compute_index_RMean}
    
    if spectral_index.index_type in ['RABD', 'RABDnorm']:
        computed_index = funs3[spectral_index.index_type](
            left=spectral_index.left_band,
            trough=spectral_index.middle_band,
            right=spectral_index.right_band,
            row_bounds=row_bounds,
            col_bounds=col_bounds,
            imagechannels=imagechannels)
    elif spectral_index.index_type in ['RABA', 'RMean', 'Ratio']:
        computed_index = funs2[spectral_index.index_type](
            left=spectral_index.left_band,
            right=spectral_index.right_band,
            row_bounds=row_bounds,
            col_bounds=col_bounds,
            imagechannels=imagechannels)
    else:
        print(f'unknown index type: {spectral_index.index_type}')
        return None
    
    return computed_index

def compute_and_clean_index(spectral_index, row_bounds, col_bounds, imagechannels):
    """Given a spectral indef definition and image, compute a clean version
    of an index map, with inf values replaced with 0 and clipping of the
    intensity within 1-99 percentiles.
    
    Parameters
    ----------
    spectral_index: SpectralIndex
        index object
    row_bounds: tuple of int
        (row_start, row_end)
    col_bounds: tuple of int
        (col_start, col_end)
    imagechannels: ImageChannels
        image channels object
    
    Returns
    -------
    computed_index: 2D np.ndarray
        computed index map
    
    """

    computed_index = compute_index(
        spectral_index=spectral_index,
        row_bounds=row_bounds, col_bounds=col_bounds,
        imagechannels=imagechannels)
    computed_index = clean_index_map(computed_index)

    return computed_index

def clean_index_map(index_map):
    """Given an index map, clean it up by replacing inf values with 0
    and clipping the intensity within 1-99 percentiles.
    
    Parameters
    ----------
    index_map: np.ndarray
        index map
    
    Returns
    -------
    index_map: 2D np.ndarray
        cleaned index map
    
    """

    index_map = index_map.copy()
    index_map[index_map == np.inf] = 0
    percentiles = np.percentile(index_map, [1, 99])
    index_map = np.clip(index_map, percentiles[0], percentiles[1])
    if isinstance(index_map, da.Array):
        index_map = index_map.compute()

    return index_map

def load_index_series(index_file):
    """Load the index series from a yml file.
    
    Parameters
    ----------
    index_file: str or Path
        file where the index series is saved
    
    Returns
    -------
    index_collection: dict
        dictionary of SpectralIndex objects

    """
    
    index_collection = {}
    with open(index_file) as file:
        index_series = yaml.full_load(file)
    for index_element in index_series['index_definition']:
        index_collection[index_element['index_name']] = SpectralIndex(**index_element)

    return index_collection

def compute_normalized_index_params(project_list, index_params_file, export_folder):
    """Given a folder with multiple experiments, load all projections and compute
    the overall min and max values for each index. Then export a new index yml file
    with updated plotting ranges.

    Parameters
    ----------
    project_list: list of Path
        list of paths to export folders
    index_params_file: Path
        path to the index parameters file used to list the required indices
    export_folder: Path
        path to the folder where to save the updated index yml file

    """

    indices = load_index_series(index_params_file)

    # gather all projections
    all_proj = []
    for ex in project_list:
        roi_folders = list(ex.glob('roi*'))
        roi_folders = [x.name for x in roi_folders if x.is_dir()]
            
        for roi_ind in range(len(roi_folders)):
            proj_path = ex.joinpath(f'roi_{roi_ind}').joinpath('index_plots').joinpath('index_projection.csv')
            if not proj_path.is_file():
                continue
            all_proj.append(pd.read_csv(proj_path))
    all_proj = pd.concat(all_proj, axis=0)

    # compute min max of index projections
    min_vals = all_proj.min()
    max_vals = all_proj.max()

    # update indices with new min max
    for k, ind in indices.items():
        ind.index_map_range = [min_vals[k].item(), max_vals[k].item()]

    export_index_series(index_series=indices, file_path=export_folder.joinpath('normalized_index_settings.yml'))

def save_index_zarr(project_folder, main_roi_index, index_name, index_map, overwrite=False):
    """Save an index map to a zarr file.

    Parameters
    ----------
    project_folder: Path
        path to the project folder
    main_roi_index: int
        index of the main roi
    index_name: str
        name of the index
    index_map: np.ndarray
        index map
    overwrite: bool
        whether to overwrite the file if it already exists

    Returns
    -------

    """
    
    folder_name = project_folder.joinpath(f'roi_{main_roi_index}').joinpath('index_maps')
    if not folder_name.is_dir():
        folder_name.mkdir()
    file_name = folder_name.joinpath(index_name + '.zarr')

    if file_name.is_dir() and not overwrite:
        warn(f'File {file_name} already exists')
        return None
    
    dtype = 'f4'
    
    z1 = zarr.open(file_name, mode='w', shape=index_map.shape,
                   chunks=index_map.shape, dtype=dtype)

    z1.attrs['metadata'] = {
        'index_name': index_name,
        }
    
    z1[:,:] = index_map

def load_index_zarr(project_folder, main_roi_index, index_name):
    """Load an index map from a zarr file.

    Parameters
    ----------
    project_folder: Path
        path to the project folder
    main_roi_index: int
        index of the main roi
    index_name: str
        name of the index
    
    Returns
    -------
    index_map: np.ndarray
        index map
    
    """
    
    file_name = project_folder.joinpath(f'roi_{main_roi_index}', 'index_maps', index_name + '.zarr')
    if not file_name.is_dir():
        warn(f'File {file_name} does not exist')
        return None
    zarr_image = da.from_zarr(file_name)
    
    return np.array(zarr_image)

def load_projection(project_folder, main_roi_index, index_name):
    """Load an index projection from a csv file.

    Parameters
    ----------
    project_folder: Path
        path to the project folder
    main_roi_index: int
        index of the main roi
    index_name: str
        name of the index

    Returns
    -------
    proj: np.ndarray
        index projection
        
    """
    
    file_name = project_folder.joinpath(f'roi_{main_roi_index}', 'index_plots', 'index_projection.csv')
    if not file_name.is_file():
        warn(f'File {file_name} does not exist')
        return None
    proj = pd.read_csv(file_name)
    
    return proj[index_name].values