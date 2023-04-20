from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget,
                            QLabel, QFileDialog, QListWidget, QAbstractItemView,
                            QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,)
from qtpy.QtCore import Qt
import numpy as np

class ChannelWidget(QListWidget):
    """Widget to handle channel selection and display. Works only if parent widget
    has an attribute called imagechannels, which is an instance of ImageChannels. For
    example with the SedimentWidget widget."""

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.itemClicked.connect(self._on_change_channel_selection)

    def _on_change_channel_selection(self):
        """Load images upon of change in channel selection.
        Considers crop bounds.
        """

        # get selected channels
        selected_channels = [item.text() for item in self.selectedItems()]
        new_channel_indices = [self.parent.imagechannels.channel_names.index(channel) for channel in selected_channels]

        roi = np.concatenate([self.parent.row_bounds, self.parent.col_bounds])

        self.parent.imagechannels.read_channels(
            channels=new_channel_indices, 
            roi=roi)

        new_cube = self.parent.imagechannels.get_image_cube(
            channels=new_channel_indices,
            roi=roi)

        self.parent.channel_indices = new_channel_indices

        if 'imcube' in self.parent.viewer.layers:
            self.parent.viewer.layers['imcube'].data = new_cube
            #self.viewer.layers['imcube'].translate = (0, self.parent.row_bounds[0], self.parent.col_bounds[0])
        else:
            self.parent.viewer.add_image(
                new_cube,
                name='imcube',
                rgb=False, 
                #translate=(0, self.row_bounds[0], self.col_bounds[0])
                )
            
    def _update_channel_list(self):
        """Update channel list"""

        # clear existing items
        self.clear()

        # add new items
        for channel in self.parent.imagechannels.channel_names:
            self.addItem(channel)
