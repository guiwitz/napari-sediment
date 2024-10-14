from pathlib import Path
from napari_sediment.widgets.sediment_widget import SedimentWidget
from napari_sediment.data.data_contribution import create_data

from pathlib import Path

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