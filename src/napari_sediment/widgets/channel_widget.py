from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget,
                            QLabel, QFileDialog, QListWidget, QAbstractItemView,
                            QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,)
from qtpy.QtCore import Qt
import numpy as np
from napari.utils import progress

class ChannelWidget(QListWidget):
    """Widget to handle channel selection and display. Works only i parent widget
    has:
    - an attribute called imagechannels, which is an instance of ImageChannels. For
    example with the SedimentWidget widget.
    - an attribute called row_bounds and col_bounds, which are the current crop
    bounds."""

    def __init__(self, viewer, imagechannels=None, translate=False):
        super().__init__()

        self.viewer = viewer
        self.imagechannels = imagechannels
        self.translate = translate

        self.channel_indices = None
        
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        #self.itemClicked.connect(self._on_change_channel_selection)


    def _on_change_channel_selection(self, row_bounds=None, col_bounds=None):
        """Load images upon of change in channel selection.
        Considers crop bounds.
        """
        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Updating channel selection")
            # get selected channels
            selected_channels = [item.text() for item in self.selectedItems()]
            new_channel_indices = [self.imagechannels.channel_names.index(channel) for channel in selected_channels]
            new_channel_indices = np.sort(new_channel_indices)

            if (row_bounds is None) or (col_bounds is None):
                roi = None
            else:
                roi = np.concatenate([row_bounds, col_bounds])

            new_cube = self.imagechannels.get_image_cube(
                channels=new_channel_indices,
                roi=roi)

            self.channel_indices = new_channel_indices
            self.bands = self.imagechannels.centers[np.array(self.channel_indices).astype(int)]
            
            layer_name = 'imcube'
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = new_cube
                self.viewer.layers[layer_name].refresh()
            else:
                self.viewer.add_image(
                    new_cube,
                    name=layer_name,
                    rgb=False,
                )
            if self.translate:
                self.viewer.layers[layer_name].translate = (0, row_bounds[0], col_bounds[0])

            # put mask as top layer
            if 'mask' in self.viewer.layers:
                mask_ind = [x.name for x in self.viewer.layers].index('mask')
                self.viewer.layers.move(mask_ind, len(self.viewer.layers))
        self.viewer.window._status_bar._toggle_activity_dock(False)
        

    def get_selected_channel_bands(self):
        
        bands = self.imagechannels.centers[np.array(self.channel_indices).astype(int)]
        return bands

    def _update_channel_list(self, imagechannels=None):
        """Update channel list"""

        if imagechannels is not None:
            self.imagechannels = imagechannels

        # clear existing items
        self.clear()

        # add new items
        for channel in self.imagechannels.channel_names:
            self.addItem(channel)
