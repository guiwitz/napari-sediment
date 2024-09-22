import numpy as np
import os

from napari_sediment.utilities.sediproc import convert_bil_raw_to_zarr
from napari_sediment.utilities._reader import read_spectral
from spectral import open_image
from pathlib import Path
from dask.array.core import Array

data_folder = Path(os.path.expanduser("~")).joinpath('Sediment_synthetic')
data_folder = data_folder.joinpath('sediment_data', 'Demo', 'capture/')

def test_convert():
    
    convert_bil_raw_to_zarr(
        hdr_path=data_folder.joinpath('Demo.hdr'),
        export_folder=data_folder)

    assert data_folder.joinpath('Demo.zarr').is_dir()
    assert data_folder.joinpath('Demo.zarr').joinpath('0.0.0').exists()

def test_reader_hdr():
    data, metadata = read_spectral(data_folder.joinpath('Demo.hdr'), bands=[1,2])
    assert data.shape == (130,120,2), "Wrong data dimensions"

def test_reader_zarr():
    data, metadata = read_spectral(data_folder.joinpath('Demo.zarr'), bands=[1,2])

    assert data.shape == (130,120,2), "Wrong data dimensions"

    assert isinstance(data, Array), f"Data type should be dask array but is {type(data)}"