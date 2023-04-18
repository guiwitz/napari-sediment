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
                            QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox, QSlider,)
from qtpy.QtCore import Qt
from superqt import QDoubleRangeSlider

import numpy as np
import pystripe
from spectral import open_image
from skimage.measure import points_in_poly

from napari_matplotlib.base import NapariMPLWidget

from napari_guitils.gui_widgets import FolderList
from napari_guitils.gui_structures import VHGroup, create_tabs, TabSet
from ._reader import read_spectral
from .sediproc import (compute_average_in_roi, white_dark_correct,
                       phasor, remove_top_bottom, remove_left_right,
                       fit_1dgaussian_without_outliers)
from .imchannels import ImChannels
from .io import save_mask, load_mask

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
        self.imhdr_path = None
        self.row_bounds = None
        self.col_bounds = None
        self.imagechannels = None
        self.rgb = [640, 545, 460]
        self.rgb_ch = None
        self.rgb_names = None
        self.viewer2 = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['Main', 'Processing', 'Mask', 'Plotting','Options']
        self.tabs = TabSet(self.tab_names)

        self.main_layout.addWidget(self.tabs)

        # loading tab
        self.files_group = VHGroup('File selection', orientation='G')
        self.tabs.add_named_tab('Main', self.files_group.gbox)

        self.btn_select_imhdr_file = QPushButton("Select imhdr file")
        self.imhdr_path_display = QLineEdit("No path")
        self.files_group.glayout.addWidget(self.btn_select_imhdr_file, 0, 0, 1, 1)
        self.files_group.glayout.addWidget(self.imhdr_path_display, 0, 1, 1, 1)

        self.btn_select_export_folder = QPushButton("Select export folder")
        self.export_path_display = QLineEdit("No path")
        self.files_group.glayout.addWidget(self.btn_select_export_folder, 1, 0, 1, 1)
        self.files_group.glayout.addWidget(self.export_path_display, 1, 1, 1, 1)

        # channel selection
        self.main_group = VHGroup('Select', orientation='G')
        self.tabs.add_named_tab('Main', self.main_group.gbox)

        self.main_group.glayout.addWidget(QLabel('Channels to load'), 0, 0, 1, 2)
        self.qlist_channels = QListWidget()
        self.qlist_channels.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.main_group.glayout.addWidget(self.qlist_channels, 1,0,1,2)
        
        self.main_group.glayout.addWidget(QLabel('Channels to analyze'), 2, 0, 1, 2)
        self.qlist_channels_analyze = QListWidget()
        self.qlist_channels_analyze.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.main_group.glayout.addWidget(self.qlist_channels_analyze, 3,0,1,2)

        self.btn_RGB = QPushButton('Load RGB')
        self.main_group.glayout.addWidget(self.btn_RGB, 4, 0, 1, 2)

        self.btn_select_all = QPushButton('Select all')
        self.main_group.glayout.addWidget(self.btn_select_all, 5, 0, 1, 2)

        self.btn_new_view = QPushButton('New view')
        self.main_group.glayout.addWidget(self.btn_new_view, 6, 0, 1, 2)
        self.btn_new_view.clicked.connect(self.new_view)

        # Plot tab
        self.scan_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.tabs.add_named_tab('Plotting', self.scan_plot)

        self._create_options_tab()
        self._create_processing_tab()
        self._create_mask_tab()
        self.add_connections()

    def new_view(self):
        import napari
        self.viewer2 = napari.Viewer()

    def _create_processing_tab(self):
        
        self.tabs.widget(self.tab_names.index('Processing')).layout().setAlignment(Qt.AlignTop)
        # processing tab
        self.process_group = VHGroup('Process Hypercube', orientation='G')
        self.tabs.add_named_tab('Processing', self.process_group.gbox)

        self.btn_destripe = QPushButton("Destripe")
        self.process_group.glayout.addWidget(self.btn_destripe)
        self.btn_white_correct = QPushButton("White correct")
        self.process_group.glayout.addWidget(self.btn_white_correct)
        self.btn_compute_index = QPushButton("Index")
        self.process_group.glayout.addWidget(self.btn_compute_index)

        self.processRGB_group = VHGroup('Process RGB', orientation='G')
        self.tabs.add_named_tab('Processing', self.processRGB_group.gbox)
        self.btn_RGBwhite_correct = QPushButton("RGB White correct")
        self.processRGB_group.glayout.addWidget(self.btn_RGBwhite_correct)

    def _create_mask_tab(self):
            
        self.tabs.widget(self.tab_names.index('Mask')).layout().setAlignment(Qt.AlignTop)
        
        self.mask_export_group = VHGroup('Mask export', orientation='G')
        self.mask_creation_group = VHGroup('Mask processing', orientation='G')
        self.tabs.add_named_tab('Mask', self.mask_export_group.gbox)
        self.tabs.add_named_tab('Mask', self.mask_creation_group.gbox)

        self.btn_save_mask = QPushButton("Save mask")
        self.mask_export_group.glayout.addWidget(self.btn_save_mask)
        self.btn_load_mask = QPushButton("Load mask")
        self.mask_export_group.glayout.addWidget(self.btn_load_mask)

        # add horizontal slider for thresholding
        self.btn_border_mask = QPushButton("Border mask")
        self.mask_creation_group.glayout.addWidget(self.btn_border_mask, 0, 0, 1, 2)
        self.spin_mask_threshold = QDoubleRangeSlider(Qt.Horizontal)#QDoubleSpinBox()
        self.spin_mask_threshold.setRange(0, 1)
        self.spin_mask_threshold.setSingleStep(0.01)
        self.spin_mask_threshold.setSliderPosition([0, 1])
        self.mask_creation_group.glayout.addWidget(QLabel("Threshold value"), 1, 0, 1, 1)
        self.mask_creation_group.glayout.addWidget(self.spin_mask_threshold, 1, 1, 1, 1)
        self.btn_update_mask = QPushButton("Manual Threshold mask")
        self.mask_creation_group.glayout.addWidget(self.btn_update_mask, 2, 0, 1, 2)
        self.btn_automated_mask = QPushButton("Automated mask")
        self.mask_creation_group.glayout.addWidget(self.btn_automated_mask, 3, 0, 1, 1)
        self.spin_automated_mask_width = QDoubleSpinBox()
        self.spin_automated_mask_width.setRange(0.1, 10)
        self.spin_automated_mask_width.setSingleStep(0.1)
        self.mask_creation_group.glayout.addWidget(self.spin_automated_mask_width, 3, 1, 1, 1)
        self.btn_compute_phaseor = QPushButton("Compute Phasor")
        self.mask_creation_group.glayout.addWidget(self.btn_compute_phaseor, 4, 0, 1, 2)
        self.btn_select_by_phasor = QPushButton("Phasor mask")
        self.mask_creation_group.glayout.addWidget(self.btn_select_by_phasor, 5, 0, 1, 2)
        self.btn_combine_masks = QPushButton("Combine masks")
        self.mask_creation_group.glayout.addWidget(self.btn_combine_masks, 6, 0, 1, 2)
        

    def _create_options_tab(self):
        
        self.background_group = VHGroup('Background selection')
        self.crop_group = VHGroup('Crop selection', orientation='G')

        self.tabs.add_named_tab('Options', self.background_group.gbox)
        self.tabs.add_named_tab('Options', self.crop_group.gbox)

        self.btn_select_white_file = QPushButton("Select white folder")
        self.background_group.glayout.addWidget(self.btn_select_white_file)

        self.btn_select_dark_file = QPushButton("Select dark folder")
        self.background_group.glayout.addWidget(self.btn_select_dark_file)

        crop_bounds_name = ['Min row', 'Max row', 'Min col', 'Max col']
        self.crop_bounds = {x: QSpinBox() for x in crop_bounds_name}
        for ind, c in enumerate(crop_bounds_name):
            self.crop_group.glayout.addWidget(QLabel(c), ind, 0, 1, 1)
            self.crop_group.glayout.addWidget(self.crop_bounds[c], ind, 1, 1, 1)

        self.check_use_crop = QCheckBox("Use crop")
        self.btn_refresh_crop = QPushButton("Refresh crop")
        self.crop_group.glayout.addWidget(self.check_use_crop, ind+1, 0, 1, 1)
        self.crop_group.glayout.addWidget(self.btn_refresh_crop, ind+1, 1, 1, 1)


    def add_connections(self):
        """Add callbacks"""

        self.btn_select_export_folder.clicked.connect(self._on_click_select_file_folder)
        self.btn_select_imhdr_file.clicked.connect(self._on_click_select_imhdr)
        self.qlist_channels.itemClicked.connect(self._on_change_channel_selection)
        self.btn_destripe.clicked.connect(self._on_click_destripe)
        self.btn_white_correct.clicked.connect(self._on_click_white_correct)
        self.btn_RGBwhite_correct.clicked.connect(self._on_click_white_correct)
        self.btn_RGB.clicked.connect(self._on_click_RGB)
        self.btn_select_all.clicked.connect(self._on_click_select_all)
        self.btn_compute_index.clicked.connect(self._compute_index)
        self.check_use_crop.stateChanged.connect(self._on_click_use_crop)
        self.btn_refresh_crop.clicked.connect(self._on_click_use_crop)
        self.btn_border_mask.clicked.connect(self._on_click_remove_borders)
        self.btn_update_mask.clicked.connect(self._on_click_update_mask)
        self.btn_automated_mask.clicked.connect(self._on_click_automated_threshold)
        self.btn_compute_phaseor.clicked.connect(self._on_click_compute_phasor)
        self.btn_select_by_phasor.clicked.connect(self._on_click_select_by_phasor)
        self.btn_combine_masks.clicked.connect(self._on_click_combine_masks)
        self.btn_save_mask.clicked.connect(self._on_click_save_mask)
        self.btn_load_mask.clicked.connect(self._on_click_load_mask)
        self.btn_select_white_file.clicked.connect(self._on_click_select_white_file)
        self.btn_select_dark_file.clicked.connect(self._on_click_select_dark_file)

        self.viewer.mouse_move_callbacks.append(self._shift_move_callback)


    def _on_click_select_file_folder(self):
        """Interactively select folder to analyze"""

        self.export_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))

    def _on_select_file(self):
        
        success = self.open_file()
        if not success:
            return False
        
    def _on_click_select_imhdr(self):
        """Interactively select hdr file"""

        imhdr_path = QFileDialog.getOpenFileName(self, "Select file")[0]
        self.set_paths(imhdr_path)
        self._on_select_file()

    def _on_click_select_white_file(self):
        """Interactively select white reference"""
        
        self.white_file_path = Path(QFileDialog.getOpenFileName(self, "Select White Ref")[0])

    def _on_click_select_dark_file(self):
        """Interactively select white reference"""
        
        self.dark_file_path = Path(QFileDialog.getOpenFileName(self, "Select Dark Ref")[0])

    def set_paths(self, imhdr_path):
        """Update image and white/dark image paths"""
        
        self.imhdr_path = Path(imhdr_path)
        self.imhdr_path_display.setText(self.imhdr_path.as_posix())

        # define default B/W files
        self.white_file_path = self.imhdr_path.parent.joinpath('WHITEREF_' + self.imhdr_path.name)
        self.dark_file_path = self.imhdr_path.parent.joinpath('DARKREF_' + self.imhdr_path.name)
        

    def _update_channel_list(self):
        """Update channel list"""

        # clear existing items
        self.qlist_channels.clear()
        self.qlist_channels_analyze.clear()

        # add new items
        for channel in self.imagechannels.channel_names:
            self.qlist_channels.addItem(channel)
            self.qlist_channels_analyze.addItem(channel)

    def _on_change_channel_selection(self):
        """Load images upon of change in channel selection.
        Considers crop bounds.
        """

        # get selected channels
        selected_channels = [item.text() for item in self.qlist_channels.selectedItems()]
        new_channel_indices = [self.imagechannels.channel_names.index(channel) for channel in selected_channels]

        if self.check_use_crop.isChecked():
            roi = np.concatenate([self.row_bounds, self.col_bounds])
        else:
            roi = None

        self.imagechannels.read_channels(
            channels=new_channel_indices, 
            roi=roi)

        new_cube = self.imagechannels.get_image_cube(
            channels=new_channel_indices,
            roi=roi)

        self.channel_indices = new_channel_indices

        if 'imcube' in self.viewer.layers:
            self.viewer.layers['imcube'].data = new_cube
            self.viewer.layers['imcube'].translate = (0, self.row_bounds[0], self.col_bounds[0])
        else:
            self.viewer.add_image(
                new_cube,
                name='imcube',
                rgb=False, 
                translate=(0, self.row_bounds[0], self.col_bounds[0]))#, colormap='gray', blending='additive')


    def open_file(self):
        """Open file in napari"""

        self.channel_indices = None
        self.image_indices = None
        # clear existing layers.
        while len(self.viewer.layers) > 0:
            self.viewer.layers.clear()
        
        # if file list is empty stop here
        if self.imhdr_path is None:
            return False
        
        # open image
        image_name = self.imhdr_path.name
        
        # reset acquisition index if new image is selected
        if image_name != self.current_image_name:
            self.current_image_name = image_name
            self.imagechannels = ImChannels(self.imhdr_path)

        self.crop_bounds['Min row'].setMaximum(self.imagechannels.nrows-1)
        self.crop_bounds['Max row'].setMaximum(self.imagechannels.nrows)
        self.crop_bounds['Min col'].setMaximum(self.imagechannels.ncols-1)
        self.crop_bounds['Max col'].setMaximum(self.imagechannels.ncols)
        self.crop_bounds['Max row'].setValue(self.imagechannels.nrows)
        self.crop_bounds['Max col'].setValue(self.imagechannels.ncols)

        self.row_bounds = [0, self.imagechannels.nrows]
        self.col_bounds = [0, self.imagechannels.ncols]
        
        self._update_channel_list()

        self._on_click_RGB()

        self._add_roi()
        self._add_mask()

    def _on_click_use_crop(self):
        """Update crop bounds. Reload cropped image if crop is checked."""
        
        if self.check_use_crop.isChecked():
            self.row_bounds = [
                self.crop_bounds['Min row'].value(), self.crop_bounds['Max row'].value()]
            self.col_bounds = [
                self.crop_bounds['Min col'].value(), self.crop_bounds['Max col'].value()]
        else:
            self.row_bounds = [0, self.imagechannels.nrows]
            self.col_bounds = [0, self.imagechannels.ncols]
        
        self._on_change_channel_selection()


    def _add_roi(self):
         self.roi_layer = self.viewer.add_shapes(
             ndim = 2,
             name='rois', edge_color='red', face_color=np.array([0,0,0,0]), edge_width=10)
         
    def _add_mask(self):
        self.mask_layer = self.viewer.add_labels(
            np.zeros((self.imagechannels.nrows,self.imagechannels.ncols), dtype=np.uint8),
            name='mask')
        
    def _compute_index(self):
        """Compute index"""

        # get selected channels
        selected_channels = [item.text() for item in self.qlist_channels_analyze.selectedItems()]

        # get channel indices
        channel_indices = [self.metadata['wavelength'].index(channel) for channel in selected_channels]

        # get selected roi
        selected_roi = self.viewer.layers['rois'].data[0].astype(int)

        white_path = self.white_file_path

        data_av = compute_average_in_roi(
            file_path=self.imhdr_path,
            channel_indices=channel_indices, 
            roi=((selected_roi[0][0], selected_roi[2][0]),(selected_roi[0][1], selected_roi[1][1])),
            white_path=white_path,
        )
        self.image_indices = data_av

    def _on_click_destripe(self):
        """Destripe image"""
        
        data_destripe = self.viewer.layers['imcube'].data.copy()
        for d in range(data_destripe.shape[0]):
            data_destripe[d] = pystripe.filter_streaks(data_destripe[d].T, sigma=[128, 256], level=7, wavelet='db2').T

        if 'imcube_destripe' in self.viewer.layers:
            self.viewer.layers['imcube_destripe'].data = data_destripe
        else:
            self.viewer.add_image(data_destripe, name='imcube_destripe', rgb=False)


    def _on_click_white_correct(self, event):
        """White correct image"""

        img_white = open_image(self.white_file_path)
        img_dark = open_image(self.dark_file_path)
        
        if self.sender().text() == 'White correct':

            white_data = img_white.read_bands(self.channel_indices)
            dark_data = img_dark.read_bands(self.channel_indices)
            if self.check_use_crop.isChecked():
                white_data = white_data[:,self.col_bounds[0]:self.col_bounds[1],:]
                dark_data = dark_data[:,self.col_bounds[0]:self.col_bounds[1],:]
            #data = np.moveaxis(self.viewer.layers['imcube'].data.copy(), 0,1)

            im_corr = white_dark_correct(self.viewer.layers['imcube'].data, white_data, dark_data)

            if 'imcube_corrected' in self.viewer.layers:
                self.viewer.layers['imcube_corrected'].data = im_corr
            else:
                self.viewer.add_image(im_corr, name='imcube_corrected', rgb=False)
                self.viewer.layers['imcube_corrected'].translate = (0, self.row_bounds[0], self.col_bounds[0])

        elif self.sender().text() == 'RGB White correct':
            for ind in self.rgb_ch:
                ch_name = self.imagechannels.channel_names[ind]
                if ch_name in self.viewer.layers:
                    data = self.viewer.layers[ch_name].data[np.newaxis, :,:]
                    white_data = img_white.read_bands([ind])
                    dark_data = img_dark.read_bands([ind])

                    im_corr = white_dark_correct(data, white_data, dark_data)
                    self.viewer.layers[ch_name].data = im_corr[0,:,:]
                    self.viewer.layers[ch_name].contrast_limits_range = (self.viewer.layers[ch_name].data.min(), self.viewer.layers[ch_name].data.max())
                    self.viewer.layers[ch_name].contrast_limits = np.percentile(self.viewer.layers[ch_name].data, (2,98))
            
            im = np.mean(np.stack([self.viewer.layers[x].data for x in self.rgb_names], axis=0), axis=0)
            self.spin_mask_threshold.setRange(im.min(), im.max())
            self.spin_mask_threshold.setSliderPosition([im.min(), im.max()])

    def _on_click_remove_borders(self):
        """Remove borders from image"""
        
        im = np.mean(np.stack([self.viewer.layers[x].data for x in self.rgb_names], axis=0), axis=0)
        first_row, last_row = remove_top_bottom(im)
        first_col, last_col = remove_left_right(im)
        if 'border-mask' in self.viewer.layers:
            self.viewer.layers['border-mask'].data[:] = 0
        else:
            self.viewer.add_labels(np.zeros_like(im, dtype=np.uint8), name='border-mask', opacity=0.5)
        self.viewer.layers['border-mask'].data[0:first_row,:] = 1
        self.viewer.layers['border-mask'].data[last_row::,:] = 1
        self.viewer.layers['border-mask'].data[:, 0:first_col] = 1
        self.viewer.layers['border-mask'].data[:, last_col::] = 1
        self.viewer.layers['border-mask'].refresh()

    def _on_click_automated_threshold(self):
        """Automatically set threshold for mask based on mean RGB pixel intensity"""

        im = np.mean(np.stack([self.viewer.layers[x].data for x in self.rgb_names], axis=0), axis=0)
        if 'border-mask' in self.viewer.layers:
            pix_selected = im[self.viewer.layers['border-mask'].data == 0]
        else:
            pix_selected = np.ravel(im)
        med_val, std_val = fit_1dgaussian_without_outliers(data=pix_selected[::5])
        self.spin_mask_threshold.setRange(im.min(), im.max())
        fact = self.spin_automated_mask_width.value()
        self.spin_mask_threshold.setSliderPosition([med_val - fact*std_val,med_val + fact*std_val])
        self._on_click_update_mask()
    
    def _on_click_update_mask(self):
        """Update mask based on current threshold"""
        
        data = np.mean(np.stack([self.viewer.layers[x].data for x in self.rgb_names], axis=0), axis=0)
        mask = ((data < self.spin_mask_threshold.value()[0]) | (data > self.spin_mask_threshold.value()[1])).astype(np.uint8)
        self.viewer.layers['mask'].data = mask
        self.viewer.layers['mask'].refresh()

    def _on_click_compute_phasor(self):
        """Compute phasor from image. Opens a new viewer with 2D histogram of 
        g, s values.
        """
            
        data, _ = read_spectral(
            self.imhdr_path,
            bands=np.arange(0, len(self.imagechannels.channel_names),10),
            row_bounds=self.row_bounds,
            col_bounds=self.col_bounds
        )
        self.g, self.s, _, _ = phasor(np.moveaxis(data,2,0), harmonic=2)
        out,_,_ = np.histogram2d(np.ravel(self.g), np.ravel(self.s), bins=[50,50])
        #phasor_points = np.stack([np.ravel(g), np.ravel(s)]).T
        if self.viewer2 is None:
            self.new_view()
        #self.viewer2.add_points(phasor_points)
        self.viewer2.add_image(out, name='phasor', rgb=False)
        self.viewer2.add_shapes(name='select_phasor')

    def _on_click_select_by_phasor(self):
        """Select good pixels based on phasor values. Uses a polygon to select pixels"""

        poly_coord = self.viewer2.layers['select_phasor'].data[0]
        poly_coord = poly_coord / self.viewer2.layers['phasor'].data.shape

        poly_coord = (
            poly_coord * np.array(
            [self.g.max()-self.g.min(),
             self.s.max()-self.s.min()])
             ) + np.array([self.g.min(), self.s.min()])
        
        g_s_points = np.stack([self.g.ravel(), self.s.ravel()]).T
        in_out = points_in_poly(g_s_points, poly_coord)
        in_out_image = np.reshape(in_out, self.g.shape)
        if 'phasor-mask' in self.viewer.layers:
            self.viewer.layers['phasor-mask'].data = in_out_image
        else:
            self.viewer.add_labels(in_out_image, name='phasor-mask')

    def _on_click_combine_masks(self):
        """Combine masks from border removel, phasor and thresholding"""

        mask_complete = self.viewer.layers['mask'].data.copy()
        if 'phasor-mask' in self.viewer.layers:
            mask_complete = mask_complete + self.viewer.layers['phasor-mask'].data
        if 'border-mask' in self.viewer.layers:
            mask_complete = mask_complete + self.viewer.layers['border-mask'].data
        
        if 'complete-mask' in self.viewer.layers:
            self.viewer.layers['complete-mask'].data = mask_complete
        else:
            self.viewer.add_labels(mask_complete, name='complete-mask')

    def _on_click_save_mask(self):
        """Save mask to file"""
            
        if 'complete-mask' in self.viewer.layers:
            mask = self.viewer.layers['complete-mask'].data
        else:
            mask = self.viewer.layers['mask'].data

        save_mask(mask, self.get_mask_path())

    def _on_click_load_mask(self):
        """Load mask from file"""
        
        mask = load_mask(self.get_mask_path())
        if 'mask' in self.viewer.layers:
            self.viewer.layers['mask'].data = mask
        else:
            self.viewer.add_labels(mask, name='mask')

    def get_mask_path(self):

        return self.export_folder.joinpath('mask.tif')


    def _on_click_RGB(self):
        """Load RGB image"""

        self.rgb_ch = [np.argmin(np.abs(np.array(self.imagechannels.channel_names).astype(float) - x)) for x in self.rgb]
        self.rgb_names = [self.imagechannels.channel_names[x] for x in self.rgb_ch]

        [self.qlist_channels.item(x).setSelected(True) for x in self.rgb_ch]
        self._on_change_channel_selection()

        cmap = ['red', 'green', 'blue']
        for c, cmap in zip(self.rgb_ch, cmap):
            self.viewer.add_image(
                self.imagechannels.channel_array[c],
                name=self.imagechannels.channel_names[c],
                colormap=cmap,
                blending='additive')

        im = np.sum(np.stack([self.viewer.layers[x].data for x in self.rgb_names], axis=0), axis=0)  
        self.spin_mask_threshold.setRange(im.min(), im.max())
        self.spin_mask_threshold.setValue([im.min(), im.max()])

    def _on_click_select_all(self):
        self.qlist_channels.selectAll()
        self._on_change_channel_selection()


    def _get_channel_name_from_index(self, index):
        
        if self.imagechannels is None:
            return None
        return self.imagechannels.channel_names[index]

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
            
            #self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.row_bounds[1]-self.row_bounds[0]-1)
            #self.cursor_pos[2] = np.clip(self.cursor_pos[2], 0, self.col_bounds[1]-self.col_bounds[0]-1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], self.row_bounds[0],self.row_bounds[1]-1)
            self.cursor_pos[2] = np.clip(self.cursor_pos[2], self.col_bounds[0],self.col_bounds[1]-1)
            spectral_pixel = self.viewer.layers['imcube'].data[
                :, self.cursor_pos[1]-self.row_bounds[0], self.cursor_pos[2]-self.col_bounds[0]
            ]

            self.scan_plot.axes.clear()
            self.scan_plot.axes.plot(spectral_pixel)
            
            self.scan_plot.canvas.figure.canvas.draw()


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