import napari
import os
from spectral.io.envi import save_image
from pathlib import Path
import numpy as np
import pooch
from magicgui import magic_factory

from ..data import synthetic
from ..widgets.sediment_widget import SedimentWidget


def create_synthetic():
    
    main_path = Path(os.path.expanduser("~")).joinpath('Sediment_synthetic')

    imhdr_path = create_data(random_seed=1, main_path=main_path)

    viewer = napari.current_viewer()
    widget = SedimentWidget(viewer)
    viewer.window.add_dock_widget(widget)

    widget.set_paths(imhdr_path)
    widget._on_select_file()

def create_cake():
    
    file_path = pooch.retrieve(
        url="doi:10.5281/zenodo.13925618/small_cake.zip",
        known_hash="md5:ed6a8a288edfbe91b92879adba695c1f",
        processor=pooch.Unzip(),
        fname = 'small_cake_download.zip',
        path=Path(os.path.expanduser("~"))
        )
    
    imhdr_path = Path(os.path.expanduser("~")).joinpath('small_cake_download.zip.unzip', 'small_cake_240212', 'capture', 'small_cake.hdr')
    viewer = napari.current_viewer()
    widget = SedimentWidget(viewer)
    viewer.window.add_dock_widget(widget)

    widget.set_paths(imhdr_path)
    widget._on_select_file()


@magic_factory(
        demo_name={'choices': ['synthetic', 'cake']},
        call_button="Load data",
        )
def demo_data(demo_name: str = 'synthetic'):
    if demo_name == 'synthetic':
        create_synthetic()
    elif demo_name == 'cake':
        create_cake()

    return


def create_data(random_seed=42, main_path=None):

    if main_path is None:
        main_path = Path(f'src/napari_sediment/data/synthetic/Synthetic{random_seed}')
    else:
        main_path = Path(main_path).joinpath(f'Synthetic{random_seed}')

    if os.path.exists(main_path.joinpath('Synthetic{random_seed}_123','capture', 'Synthetic{random_seed}_123.hdr')):
        pass
    
    channels = 80
    im_test, dark_ref, dark_for_white_ref, white_ref = synthetic.generate_synthetic_dataset(
    image_mean=1000, image_std=5, min_val=300, max_val=400, height=130, width=120, 
    ref_height=20, channels=channels, white_ref_added_signal=2000, pattern_weight=10, pattern_width=10, random_seed=random_seed)

    im_test = synthetic.add_signal_to_image(im_test=im_test, widths=[15, 30], ch_positions = [40, 40],
                                            row_boundaries=[[10,20], [60,70]], col_boundaries=[[10,110],[10,110]], amplitudes=[-400, -400], channels=80)

    im_test = synthetic.add_ellipse_to_image(im_test, 100, 37, 10, 20, -600)

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
    
    return main_path.joinpath(f'Synthetic{random_seed}_123/capture/Synthetic{random_seed}_123.hdr')

