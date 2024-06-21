from pathlib import Path
from .io import get_data_background_path
from .sediproc import correct_save_to_zarr
from .imchannels import ImChannels
from .parameters.parameters import Param

def batch_preprocessing(folder_to_analyze, export_folder, background_text='_WR_',
                        min_max_band=None, background_correction=True, destripe=True, use_dask=True, chunk_size=1000):

    export_folder = Path(export_folder)
    _, _, white_file_path, dark_for_white_file_path, dark_for_im_file_path, imhdr_path = get_data_background_path(folder_to_analyze, background_text=background_text)
    folder_to_analyze_name = folder_to_analyze.name
    export_folder = export_folder.joinpath(folder_to_analyze_name)

    if not export_folder.is_dir():
        export_folder.mkdir()

    param = Param(
        project_path=export_folder,
        file_path=imhdr_path,
        white_path=white_file_path,
        dark_for_im_path=dark_for_im_file_path,
        dark_for_white_path=dark_for_white_file_path,
        main_roi=[],
        rois=[])
    
    correct_save_to_zarr(
        imhdr_path=imhdr_path,
        white_file_path=white_file_path,
        dark_for_im_file_path=dark_for_im_file_path,
        dark_for_white_file_path=dark_for_white_file_path,
        zarr_path=export_folder.joinpath('corrected.zarr'),
        band_indices=None,
        min_max_bands=min_max_band,
        background_correction=background_correction,
        destripe=destripe,
        use_dask=use_dask,
        chunk_size=chunk_size
        )
    imchannels = ImChannels(export_folder.joinpath('corrected.zarr'))
    param.main_roi = [[
        0, 0,
        imchannels.nrows, 0,
        imchannels.nrows, imchannels.ncols,
        0, imchannels.ncols
        ]]
    param.save_parameters()