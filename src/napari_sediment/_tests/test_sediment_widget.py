from pathlib import Path
from napari_sediment.sediment_widget import SedimentWidget


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_q_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    self = SedimentWidget(viewer)

    imhdr_path = Path('/Users/gw18g940/Desktop/Test_data/Zahajska/synthetic/Synthetic1/Synthetic1_123/capture/Synthetic1_123.hdr')
    self.set_paths(imhdr_path)
    self._on_select_file()
    
    assert 'red' in viewer.layers
    assert 'green' in viewer.layers
    assert 'blue' in viewer.layers
    assert 'mask' in viewer.layers
    assert len(self.imagechannels.channel_names) == 80, f"Expected 80 channels got {len(self.imagechannels.channel_names)}"