from pathlib import Path
import numpy as np
from ..widgets.sediment_widget import SedimentWidget
from napari_sediment.data import synthetic
import os
from pathlib import Path
import pytest
from spectral.io.envi import save_image


def create_data(random_seed=42):

    if os.path.exists(f'src/napari_sediment/data/synthetic/Synthetic{random_seed}/Synthetic{random_seed}_123/capture/Synthetic{random_seed}_123.hdr'):
        pass
    
    channels = 80
    im_test, dark_ref, dark_for_white_ref, white_ref = synthetic.generate_synthetic_dataset(
    image_mean=1000, image_std=5, min_val=300, max_val=400, height=130, width=120, 
    ref_height=20, channels=channels, white_ref_added_signal=2000, pattern_weight=10, pattern_width=10, random_seed=random_seed)

    im_test = synthetic.add_signal_to_image(im_test=im_test, widths=[15, 30], ch_positions = [40, 40],
                                            row_boundaries=[[10,20], [60,70]], col_boundaries=[[10,110],[10,110]], amplitudes=[-400, -400], channels=80)

    im_test = synthetic.add_ellipse_to_image(im_test, 100, 37, 10, 20, -600)

    main_path = Path(f'src/napari_sediment/data/synthetic/Synthetic{random_seed}')
    os.makedirs(main_path.joinpath(f'Synthetic{random_seed}_123/capture').as_posix(), exist_ok=True)
    os.makedirs(main_path.joinpath(f'Synthetic{random_seed}_WR_123/capture').as_posix(), exist_ok=True)

    metadata = {'wavelength': [str(x) for x in np.linspace(300, 900, channels)], 'interleave': 'bil'}
    save_image(
        hdr_file=main_path.joinpath(f'Synthetic{random_seed}_123/capture/Synthetic{random_seed}_123.hdr'),
        image=im_test, ext='raw', force=True, metadata=metadata, interleave='bil')
    
    save_image(
        hdr_file=main_path.joinpath(f'Synthetic{random_seed}_123/capture/DARKREF_Synthetic{random_seed}_123.hdr'),
        image=dark_ref, ext='raw', force=True, metadata=metadata, interleave='bil')
    
    save_image(
        hdr_file=main_path.joinpath(f'Synthetic{random_seed}_WR_123/capture/DARKREF_Synthetic{random_seed}_123.hdr'),
        image=dark_for_white_ref, ext='raw', force=True, metadata=metadata, interleave='bil')
    save_image(
        hdr_file=main_path.joinpath(f'Synthetic{random_seed}_WR_123/capture/WHITEREF_Synthetic{random_seed}_123.hdr'),
        image=white_ref, ext='raw', force=True, metadata=metadata, interleave='bil')
         

def test_select_file(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    
    create_data(random_seed=1)
    create_data(random_seed=2)
    viewer = make_napari_viewer()
    self = SedimentWidget(viewer)

    imhdr_path = Path('src/napari_sediment/data/synthetic/Synthetic1/Synthetic1_123/capture/Synthetic1_123.hdr')
    self.set_paths(imhdr_path)
    self._on_select_file()
    
    assert 'red' in viewer.layers
    assert 'green' in viewer.layers
    assert 'blue' in viewer.layers
    assert 'imcube' in viewer.layers
    assert len(self.imagechannels.channel_names) == 80, f"Expected 80 channels got {len(self.imagechannels.channel_names)}"