from qtpy.QtWidgets import (QComboBox, QPushButton, QWidget,
                            QLabel, QFileDialog, QListWidget, QAbstractItemView,
                            QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,)
from qtpy.QtCore import Qt
import numpy as np
import dask.array as da
from napari_guitils.gui_structures import VHGroup, TabSet
from superqt import QDoubleRangeSlider

from ..utilities.utils import update_contrast_on_layer

class RGBWidget(QWidget):
    """Widget to handle channel selection and display. Works only i parent widget
    has:
    - an attribute called imagechannels, which is an instance of ImageChannels. For
    example with the SedimentWidget widget.
    - an attribute called row_bounds and col_bounds, which are the current crop
    bounds."""

    def __init__(self, viewer, imagechannels=None, translate=True):
        super().__init__()

        self.viewer = viewer
        self.imagechannels = imagechannels
        self.rgb = [640, 545, 460]
        self.row_bounds = None
        self.col_bounds = None
        self.translate = translate

        self.rgbmain_group = VHGroup('RGB', orientation='G')
        #self.tabs.add_named_tab('Main', self.rgbmain_group.gbox)

        self.rgb_bands_group = VHGroup('Select bands to display as RGB', orientation='G')
        self.rgbmain_group.glayout.addWidget(self.rgb_bands_group.gbox, 0, 0, 1, 2)

        self.btn_default_rgb = QPushButton('Default RGB')
        self.btn_default_rgb.setToolTip("Set default RGB channels")
        self.rgb_bands_group.glayout.addWidget(self.btn_default_rgb, 0, 0, 1, 6)
        self.btn_RGB = QPushButton('Load RGB')
        self.btn_RGB.setToolTip("Load RGB channels")
        self.spin_rchannel = QSpinBox()
        self.spin_rchannel.setRange(0, 1000)
        self.spin_rchannel.setValue(640)
        self.spin_gchannel = QSpinBox()
        self.spin_gchannel.setRange(0, 1000)
        self.spin_gchannel.setValue(545)
        self.spin_bchannel = QSpinBox()
        self.spin_bchannel.setRange(0, 1000)
        self.spin_bchannel.setValue(460)

        self.rgb_bands_group.glayout.addWidget(QLabel('R'), 1, 0, 1, 1)
        self.rgb_bands_group.glayout.addWidget(self.spin_rchannel, 1, 1, 1, 1)
        self.rgb_bands_group.glayout.addWidget(QLabel('G'), 1, 2, 1, 1)
        self.rgb_bands_group.glayout.addWidget(self.spin_gchannel, 1, 3, 1, 1)
        self.rgb_bands_group.glayout.addWidget(QLabel('B'), 1, 4, 1, 1)
        self.rgb_bands_group.glayout.addWidget(self.spin_bchannel, 1, 5, 1, 1)
        self.rgb_bands_group.glayout.addWidget(self.btn_RGB, 2, 0, 1, 6)

        self.rgb_layer_group = VHGroup('Select layer to display as RGB', orientation='G')
        #self.rgbmain_group.glayout.addWidget(self.rgb_layer_group.gbox, 1, 0, 1, 2)

        self.combo_layer_to_rgb = QComboBox()
        self.rgb_layer_group.glayout.addWidget(QLabel('Layer to display'), 0, 0, 1, 1)
        self.rgb_layer_group.glayout.addWidget(self.combo_layer_to_rgb, 0, 1, 1, 1)
        self.btn_dislpay_as_rgb = QPushButton('Display layer as RGB')
        self.rgb_layer_group.glayout.addWidget(self.btn_dislpay_as_rgb, 1, 0, 2, 2)

        self.slider_contrast = QDoubleRangeSlider(Qt.Horizontal)
        self.slider_contrast.setRange(0, 1)
        self.slider_contrast.setSingleStep(0.01)
        self.slider_contrast.setSliderPosition([0, 1])
        self.rgbmain_group.glayout.addWidget(QLabel("RGB Contrast"), 1, 0, 1, 1)
        self.rgbmain_group.glayout.addWidget(self.slider_contrast, 1, 1, 1, 1)

        self.add_connections()

    def add_connections(self):

        self.btn_RGB.clicked.connect(self._on_click_RGB)
        self.spin_rchannel.valueChanged.connect(self._on_change_rgb)
        self.spin_gchannel.valueChanged.connect(self._on_change_rgb)
        self.spin_bchannel.valueChanged.connect(self._on_change_rgb)
        self.btn_dislpay_as_rgb.clicked.connect(self.display_imcube_indices_as_rgb)
        self.btn_default_rgb.clicked.connect(self._set_rgb_default)
        self.slider_contrast.valueChanged.connect(self._on_change_contrast)

        self.viewer.layers.events.inserted.connect(self._update_combo_layers)
        self.viewer.layers.events.removed.connect(self._update_combo_layers)


    def _on_change_rgb(self, event=None):

        self.rgb = [self.spin_rchannel.value(), self.spin_gchannel.value(), self.spin_bchannel.value()]
    
    def set_rgb(self, rgb):
            
        self.spin_rchannel.setValue(rgb[0])
        self.spin_gchannel.setValue(rgb[1])
        self.spin_bchannel.setValue(rgb[2])

    def _set_rgb_default(self):

        self.spin_rchannel.setValue(640)
        self.spin_gchannel.setValue(545)
        self.spin_bchannel.setValue(460)

    def _on_click_RGB(self, event=None):
        """Load RGB image. Band indices are in self.rgb which are set by the spin boxes"""

        roi = None
        if (self.row_bounds is not None) and (self.col_bounds is not None):
            roi = np.concatenate([self.row_bounds, self.col_bounds])
        self.rgb_ch, self.rgb_names = self.imagechannels.get_indices_of_bands(self.rgb)
        rgb_cube = self.imagechannels.get_image_cube(self.rgb_ch, roi=roi)
        self.add_rgb_cube_to_viewer(rgb_cube)

    def get_current_rgb_cube(self):

        rgb_cube = np.array([self.viewer.layers[c].data for c in ['red', 'green', 'blue']])
        return rgb_cube
    
    def _on_change_contrast(self, event=None):
        """Update contrast limits of RGB channels"""
        
        rgb = ['red', 'green', 'blue']
        for c in rgb:
            if isinstance(self.viewer.layers[c].data, da.Array):
                contrast_limits = np.percentile(self.viewer.layers[c].data.compute(), (2,98))
            else:
                contrast_limits = np.percentile(self.viewer.layers[c].data, (2,98))
            contrast_range = contrast_limits[1] - contrast_limits[0]
            newlimits = contrast_limits.copy()
            newlimits[0] = contrast_limits[0] + self.slider_contrast.value()[0] * contrast_range
            newlimits[1] = contrast_limits[0] + self.slider_contrast.value()[1] * contrast_range
            self.viewer.layers[c].contrast_limits = newlimits


    def _update_combo_layers(self):

        admit_layers = ['imcube', 'imcube_corrected', 'imcube_destripe']
        self.combo_layer_to_rgb.clear()
        for a in admit_layers:
            if a in self.viewer.layers:
                self.combo_layer_to_rgb.addItem(a)

    def load_and_display_rgb_bands(self, roi=None):

        self.rgb_ch, self.rgb_names = self.imagechannels.get_indices_of_bands(self.rgb)
        rgb_cube = self.imagechannels.get_image_cube(self.rgb_ch, roi=roi)
        
        self.add_rgb_cube_to_viewer(rgb_cube)

    def display_imcube_indices_as_rgb(self, event=None, channels=None):

        if channels is None:
            channels = [0, 1, 2]

        layer_name = self.combo_layer_to_rgb.currentText()
        rgb_cube = np.array([self.viewer.layers[layer_name].data[ind] for ind in channels])

        self.add_rgb_cube_to_viewer(rgb_cube)

    def add_rgb_cube_to_viewer(self, rgb_cube):
        
        cmaps = ['red', 'green', 'blue']
        for ind, cmap in enumerate(cmaps):
            if cmap not in self.viewer.layers:
                self.viewer.add_image(
                    rgb_cube[ind],
                    name=cmap,
                    colormap=cmap,
                    blending='additive')
            else:
                self.viewer.layers[cmap].data = rgb_cube[ind]
                if self.translate:
                    self.viewer.layers[cmap].translate = (self.row_bounds[0], self.col_bounds[0])
            
            update_contrast_on_layer(self.viewer.layers[cmap])