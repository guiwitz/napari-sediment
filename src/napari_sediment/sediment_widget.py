"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from pathlib import Path
from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget, QComboBox,
                            QTabWidget, QGroupBox, QHBoxLayout, QGridLayout,
                            QLabel, QFileDialog, QListWidget, QAbstractItemView,
                            QCheckBox)
from qtpy.QtCore import Qt

import napari
import numpy as np
import pystripe
from spectral import open_image

from napari_matplotlib.base import NapariMPLWidget

from .folder_list_widget import FolderList
from ._reader import read_spectral
from .sediproc import compute_average_in_roi

if TYPE_CHECKING:
    import napari


class SedimentWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_image_name = None
        self.channel_indices = None
        self.metadata = None
        self.image_indices = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # loading tab
        self.loading = QWidget()
        self._loading_layout = QVBoxLayout()
        self.loading.setLayout(self._loading_layout)
        self.tabs.addTab(self.loading, 'Loading')

        self.files_group = VHGroup('File selection', orientation='G')
        self._loading_layout.addWidget(self.files_group.gbox)

        self.files_group.glayout.addWidget(QLabel("List of images"), 0, 0, 1, 2)

        self.file_list = FolderList(napari_viewer)
        self.files_group.glayout.addWidget(self.file_list, 1, 0, 1, 2)

        self.files_group.glayout.addWidget(QLabel("White images"), 2, 0, 1, 2)
        self.file_list_white = FolderList(napari_viewer)
        self.files_group.glayout.addWidget(self.file_list_white, 3, 0, 1, 2)

        self.files_group.glayout.addWidget(QLabel("Dark images"), 4, 0, 1, 2)
        self.file_list_dark = FolderList(napari_viewer)
        self.files_group.glayout.addWidget(self.file_list_dark, 5, 0, 1, 2)

        self.folder_group = VHGroup('Folder selection')
        self._loading_layout.addWidget(self.folder_group.gbox)

        self.btn_select_file_folder = QPushButton("Select data folder")
        self.folder_group.glayout.addWidget(self.btn_select_file_folder)

        self.btn_select_white_folder = QPushButton("Select white folder")
        self.folder_group.glayout.addWidget(self.btn_select_white_folder)

        self.btn_select_dark_folder = QPushButton("Select dark folder")
        self.folder_group.glayout.addWidget(self.btn_select_dark_folder)

        self.check_same_folder = QCheckBox("Same folder for all")
        self.check_same_folder.setChecked(True)
        self.folder_group.glayout.addWidget(self.check_same_folder)

        # channels tab
        self.channels_tab = QWidget()
        self._channels_tab_layout = QVBoxLayout()
        self.channels_tab.setLayout(self._channels_tab_layout)
        self.tabs.addTab(self.channels_tab, 'Channels')

        self.channel_selection_group = VHGroup('Select', orientation='G')
        self._channels_tab_layout.addWidget(self.channel_selection_group.gbox)
        
        self.channel_selection_group.glayout.addWidget(QLabel('Channels to load'), 0, 0, 1, 2)
        self.qlist_channels = QListWidget()
        self.qlist_channels.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.channel_selection_group.glayout.addWidget(self.qlist_channels, 1,0,1,2)
        
        self.channel_selection_group.glayout.addWidget(QLabel('Channels to analyze'), 2, 0, 1, 2)
        self.qlist_channels_analyze = QListWidget()
        self.qlist_channels_analyze.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.channel_selection_group.glayout.addWidget(self.qlist_channels_analyze, 3,0,1,2)

        self.rgb_group = VHGroup('Global', orientation='G')
        self._channels_tab_layout.addWidget(self.rgb_group.gbox)

        self.btn_RGB = QPushButton('Load RGB')
        self.rgb_group.glayout.addWidget(self.btn_RGB, 0, 0, 1, 2)

        self.btn_all_for_analysis = QPushButton('Select all for analysis')
        self.rgb_group.glayout.addWidget(self.btn_all_for_analysis, 1, 0, 1, 2)

        # processing tab
        self.processing = QWidget()
        self._processing_layout = QVBoxLayout()
        self.processing.setLayout(self._processing_layout)
        self.tabs.addTab(self.processing, 'Processing')

        self.btn_destripe = QPushButton("Destripe")
        self._processing_layout.addWidget(self.btn_destripe)
        self.check_destripe_all = QCheckBox("Destripe all")
        self._processing_layout.addWidget(self.check_destripe_all)
        self.btn_white_correct = QPushButton("White correct")
        self._processing_layout.addWidget(self.btn_white_correct)
        self.btn_compute_index = QPushButton("Index")
        self._processing_layout.addWidget(self.btn_compute_index)

        # Plot tab
        self.plot_tab = QWidget()
        self._plot_tab_layout = QVBoxLayout()
        self.plot_tab.setLayout(self._plot_tab_layout)
        self.tabs.addTab(self.plot_tab, 'Plots')

        self.scan_plot = SpectralPlotter(napari_viewer=self.viewer)
        self._plot_tab_layout.addWidget(self.scan_plot)

        self.add_connections()

    def add_connections(self):
        """Add callbacks"""

        self.btn_select_file_folder.clicked.connect(self._on_click_select_file_folder)
        self.file_list.currentItemChanged.connect(self._on_select_file)
        self.qlist_channels.itemClicked.connect(self._on_change_channel_selection)
        self.btn_destripe.clicked.connect(self._on_click_destripe)
        self.btn_white_correct.clicked.connect(self._on_click_white_correct)
        self.btn_RGB.clicked.connect(self._on_click_RGB)
        self.btn_compute_index.clicked.connect(self._compute_index)

        self.viewer.mouse_move_callbacks.append(self._shift_move_callback)


    def _on_click_select_file_folder(self):
        """Interactively select folder to analyze"""

        file_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        self.file_list.update_from_path(file_folder)
        self.reset_channels = True

        if self.check_same_folder.isChecked():
            self.file_list_white.update_from_path(file_folder)
            self.file_list_dark.update_from_path(file_folder)

    def _on_select_file(self):
        
        success = self.open_file()
        if not success:
            return False

    def _update_channel_list(self):
        """Update channel list"""

        # clear existing items
        self.qlist_channels.clear()
        self.qlist_channels_analyze.clear()

        # add new items
        for channel in self.metadata['wavelength']:
            self.qlist_channels.addItem(channel)
            self.qlist_channels_analyze.addItem(channel)

    def _on_change_channel_selection(self):
        """Change channel selection"""

        # get selected channels
        selected_channels = [item.text() for item in self.qlist_channels.selectedItems()]

        # get channel indices
        new_channel_indices = [self.metadata['wavelength'].index(channel) for channel in selected_channels]
        if self.channel_indices is not None:
            to_remove = list(set(self.channel_indices) - set(new_channel_indices))
            to_add = list(set(new_channel_indices) - set(self.channel_indices))
        else:
            to_remove = []
            to_add = new_channel_indices

        if len(to_remove) > 0:
            for r in to_remove:
                self.viewer.layers.remove(self.metadata['wavelength'][r])
        if len(to_add) > 0:
            # load selected channels
            data, _ = read_spectral(self.file_list.folder_path.joinpath(self.file_list.currentItem().text()), to_add)        
            self.viewer.add_image(data, channel_axis=2, name=np.array(self.metadata['wavelength'])[to_add])#, colormap='gray', blending='additive')
        self.channel_indices = new_channel_indices

    def open_file(self):
        """Open file in napari"""

        self.channel_indices = None
        self.image_indices = None
        # clear existing layers.
        while len(self.viewer.layers) > 0:
            self.viewer.layers.clear()
        
        # if file list is empty stop here
        if self.file_list.currentItem() is None:
            return False
        
        # open image
        image_name = self.file_list.currentItem().text()
        image_path = self.file_list.folder_path.joinpath(image_name)
        
        # reset acquisition index if new image is selected
        if image_name != self.current_image_name:
            self.current_image_name = image_name
            data, self.metadata = read_spectral(image_path, [0])

        #self.viewer.add_image(data, channel_axis=2, name=self.metadata['wavelength'])
        self._update_channel_list()
        #self.qlist_channels.item(0).setSelected(True)
        #self.channel_indices = [0]

        self._on_click_RGB()

        self._add_roi()

    def _add_roi(self):
         self.roi_layer = self.viewer.add_shapes(
            ndim = 2,
            name='rois', edge_color='red', face_color=np.array([0,0,0,0]), edge_width=10)
        
    def _compute_index(self):
        """Compute index"""

        # get selected channels
        selected_channels = [item.text() for item in self.qlist_channels_analyze.selectedItems()]

        # get channel indices
        channel_indices = [self.metadata['wavelength'].index(channel) for channel in selected_channels]

        # get selected roi
        selected_roi = self.viewer.layers['rois'].data[0].astype(int)

        file_path = self.file_list.get_selected_filepath()

        white_path = self.file_list_white.get_selected_filepath()

        data_av = compute_average_in_roi(
            file_path=file_path,
            channel_indices=channel_indices, 
            roi=((selected_roi[0][0], selected_roi[2][0]),(selected_roi[0][1], selected_roi[1][1])),
            white_path=white_path,
        )
        self.image_indices = data_av

    def _on_click_destripe(self):
        
        if self.check_destripe_all.isChecked():
            ch_names = self._get_all_channel_names()
            layer_list = [self.viewer.layers[x] for x in ch_names]
        else:
            layer_list = [self.viewer.layers[self.viewer.layers.selection.active.name]]
        for lay in layer_list:
            data = lay.data
            data_destripe = pystripe.filter_streaks(data.T, sigma=[128, 256], level=7, wavelet='db2')
            lay.data = data_destripe.T

    def _on_click_white_correct(self):

        img_white = open_image(self.file_list_white.get_selected_filepath())
        #white_data = img_white.read_subregion(row_bounds=(0,1500), col_bounds=(0,600), bands=widget.channel_indices)
        white_data = img_white.read_bands(self.channel_indices)
        white_av = white_data.mean(axis=0)

        for ind, ch_ind in enumerate(self.channel_indices):

            ch_name = self._get_channel_name_from_index(ch_ind)
            lay = self.viewer.layers[ch_name]

            lay.data = white_av.max() * (lay.data / white_av[:,ind])
            lay.contrast_limits_range = (lay.data.min(), lay.data.max())
            lay.contrast_limits = np.percentile(lay.data, (2,98))


    def _on_click_RGB(self):

        rgb = [640, 545, 460]
        rgb_ch = [np.argmin(np.abs(np.array(self.metadata['wavelength']).astype(float) - x)) for x in rgb]
        [self.qlist_channels.item(x).setSelected(True) for x in rgb_ch]
        self._on_change_channel_selection()
        self.viewer.layers[self.metadata['wavelength'][rgb_ch[0]]].colormap = 'red'
        self.viewer.layers[self.metadata['wavelength'][rgb_ch[1]]].colormap = 'green'
        self.viewer.layers[self.metadata['wavelength'][rgb_ch[2]]].colormap = 'blue'

    def _get_channel_name_from_index(self, index):
        return self.metadata['wavelength'][index]

    def _get_all_channel_names(self):

        ch_names = [self._get_channel_name_from_index(ch_ind) for ch_ind in self.channel_indices]
        return ch_names
    
    def _shift_move_callback(self, viewer, event):
        """Receiver for napari.viewer.mouse_move_callbacks, checks for 'Shift' event modifier.
        If event contains 'Shift' and layer attribute contains napari layers the cursor position is written to the
        cursor_pos attribute and the _draw method is called afterwards.
        """

        if 'Shift' in event.modifiers and self.viewer.layers:
            self.cursor_pos = np.rint(self.viewer.cursor.position).astype(int)
            
            '''if (self.cursor_pos[0] < self.viewer.layers[0].data.shape[0]) and (self.cursor_pos[1] < self.viewer.layers[0].data.shape[1]):
                spectral_pixel = np.array([lay.data[self.cursor_pos[0], self.cursor_pos[1]] for lay in self.viewer.layers if isinstance(lay, napari.layers.image.image.Image)])

                self.scan_plot.axes.clear()
                self.scan_plot.axes.plot(spectral_pixel)
                
                self.scan_plot.canvas.figure.canvas.draw()'''
            
            selected_roi = self.viewer.layers['rois'].data[0].astype(int)
            roi=((selected_roi[0][0], selected_roi[2][0]),(selected_roi[0][1], selected_roi[1][1]))

            if (self.cursor_pos[0] < roi[0][1]) and (self.cursor_pos[0] > roi[0][0]) and (self.cursor_pos[1] < roi[1][1]) and (self.cursor_pos[1] > roi[1][0]):
                if self.image_indices is not None:
                    spectral_pixel = self.image_indices[:, self.cursor_pos[0]-roi[0][0]]

                    self.scan_plot.axes.clear()
                    self.scan_plot.axes.plot(spectral_pixel)
                    
                    self.scan_plot.canvas.figure.canvas.draw()


class VHGroup():
    """Group box with specific layout.

    Parameters
    ----------
    name: str
        Name of the group box
    orientation: str
        'V' for vertical, 'H' for horizontal, 'G' for grid
    """

    def __init__(self, name, orientation='V'):
        self.gbox = QGroupBox(name)
        if orientation=='V':
            self.glayout = QVBoxLayout()
        elif orientation=='H':
            self.glayout = QHBoxLayout()
        elif orientation=='G':
            self.glayout = QGridLayout()
        else:
            raise Exception(f"Unknown orientation {orientation}") 

        self.gbox.setLayout(self.glayout)

class SpectralPlotter(NapariMPLWidget):
    """Subclass of napari_matplotlib NapariMPLWidget for voxel position based time series plotting.
    This widget contains a matplotlib figure canvas for plot visualisation and the matplotlib toolbar for easy option
    controls. The widget is not meant for direct docking to the napari viewer.
    Plot visualisation is triggered by moving the mouse cursor over the voxels of an image layer while holding the shift
    key. The first dimension is handled as time. This widget needs a napari viewer instance and a LayerSelector instance
    to work properly.
    Attributes:
        axes : matplotlib.axes.Axes
        selector : napari_time_series_plotter.LayerSelector
        cursor_pos : tuple of current mouse cursor position in the napari viewer
    """
    def __init__(self, napari_viewer, options=None):
        super().__init__(napari_viewer)
        self.axes = self.canvas.figure.subplots()
        self.cursor_pos = np.array([])
       

    def clear(self):
        """
        Clear the canvas.
        """
        self.axes.clear()