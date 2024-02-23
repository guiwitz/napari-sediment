"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from pathlib import Path
import warnings
from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget,
                            QLabel, QFileDialog, QComboBox,
                            QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,
                            QScrollArea)
from qtpy.QtCore import Qt
from superqt import QDoubleRangeSlider, QDoubleSlider
from napari.qt import get_current_stylesheet
from napari.utils import progress

import numpy as np
#import pystripe
from skimage.measure import points_in_poly
import skimage
from scipy.ndimage import binary_fill_holes
from spectral.algorithms import remove_continuum
from scipy.signal import savgol_filter

from napari_guitils.gui_structures import VHGroup, TabSet
from ._reader import read_spectral
from .sediproc import (white_dark_correct, load_white_dark,
                       phasor, remove_top_bottom, remove_left_right,
                       fit_1dgaussian_without_outliers, correct_save_to_zarr,
                       find_index_of_band, savgol_destripe)
from .imchannels import ImChannels
from .io import save_mask, load_mask, get_mask_path, load_project_params
from .parameters.parameters import Param
from .spectralplot import SpectralPlotter
from .widgets.channel_widget import ChannelWidget
from .images import save_rgb_tiff_image
from .widgets.rgb_widget import RGBWidget
from .utils import update_contrast_on_layer
from .batch_preproc import BatchPreprocWidget

import napari


class SedimentWidget(QWidget):
    
    def __init__(self, napari_viewer):
        super().__init__()
        
        self.viewer = napari_viewer
        self.params = Param()
        self.current_image_name = None
        self.metadata = None
        self.imhdr_path = None
        self.row_bounds = None
        self.col_bounds = None
        self.imagechannels = None
        self.viewer2 = None
        #self.pixclass = None
        self.export_folder = None
        self.mainroi_min_col = None
        self.mainroi_max_col = None
        self.mainroi_min_row = None
        self.mainroi_max_row = None
        self.spectral_pixel = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['&Main', 'Pro&cessing', 'Mas&k', '&ROI', '&Export-Import', 'P&lotting']#,'&Options']
        self.tabs = TabSet(self.tab_names)

        self.main_layout.addWidget(self.tabs)

        self._create_main_tab()
        self._create_options_tab()
        self._create_processing_tab()
        self._create_mask_tab()
        self._create_roi_tab()
        self._create_export_tab()
        self._create_plot_tab()

        self.add_connections()

    def new_view(self):
        import napari
        self.viewer2 = napari.Viewer()

    def _create_main_tab(self):

        # file selection
        self.files_group = VHGroup('Files and folders', orientation='G')
        self.tabs.add_named_tab('&Main', self.files_group.gbox)

        self.btn_select_imhdr_file = QPushButton("Select hdr file")
        self.btn_select_imhdr_file.setToolTip("Select a file with .hdr extension")
        self.imhdr_path_display = QLineEdit("No path")
        self.files_group.glayout.addWidget(self.btn_select_imhdr_file, 0, 0, 1, 1)
        self.files_group.glayout.addWidget(self.imhdr_path_display, 0, 1, 1, 1)

        self.btn_select_export_folder = QPushButton("Set export folder")
        self.btn_select_export_folder.setToolTip(
            "Select a folder where to save the results and intermeditate files")
        self.export_path_display = QLineEdit("No path")
        self.files_group.glayout.addWidget(self.btn_select_export_folder, 1, 0, 1, 1)
        self.files_group.glayout.addWidget(self.export_path_display, 1, 1, 1, 1)

        # metadata
        self.metadata_group = VHGroup('Metadata', orientation='G')
        self.tabs.add_named_tab('&Main', self.metadata_group.gbox)

        self.metadata_location = QLineEdit("No location")
        self.metadata_location.setToolTip("Indicate the location of data acquisition")
        self.metadata_group.glayout.addWidget(QLabel('Location'), 0, 0, 1, 1)
        self.metadata_group.glayout.addWidget(self.metadata_location, 0, 1, 1, 1)
        self.spinbox_metadata_scale = QDoubleSpinBox()
        self.spinbox_metadata_scale.setToolTip("Indicate conversion factor from pixel to mm")
        self.spinbox_metadata_scale.setRange(1, 1000)
        self.spinbox_metadata_scale.setSingleStep(0.1)
        self.spinbox_metadata_scale.setValue(1)
        self.metadata_group.glayout.addWidget(QLabel('Scale'), 1, 0, 1, 1)
        self.metadata_group.glayout.addWidget(self.spinbox_metadata_scale, 1, 1, 1, 1)

        # channel selection
        self.main_group = VHGroup('Bands', orientation='G')
        self.tabs.add_named_tab('&Main', self.main_group.gbox)

        self.main_group.glayout.addWidget(QLabel('Bands to load'), 0, 0, 1, 2)
        self.qlist_channels = ChannelWidget(self.viewer, translate=True)
        self.qlist_channels.itemClicked.connect(self._on_change_select_bands)
        self.qlist_channels.setToolTip(
            "Select one or more (hold shift) bands to load. Loaded bands are displayed in the imcube layer.")
        self.main_group.glayout.addWidget(self.qlist_channels, 1,0,1,2)

        self.btn_select_all = QPushButton('Select all')
        self.btn_select_all.setEnabled(False)
        self.main_group.glayout.addWidget(self.btn_select_all, 2, 0, 1, 2)

        self.check_sync_bands_rgb = QCheckBox("Sync bands with RGB")
        self.check_sync_bands_rgb.setToolTip("Display same bands in RGB as in imcube")
        self.check_sync_bands_rgb.setChecked(True)
        self.main_group.glayout.addWidget(self.check_sync_bands_rgb, 3, 0, 1, 2)
        self.qlist_channels.setEnabled(False)

        self.rgb_widget = RGBWidget(viewer=self.viewer)
        self.tabs.add_named_tab('&Main', self.rgb_widget.rgbmain_group.gbox)
        self.rgb_widget.btn_RGB.clicked.connect(self._on_click_sync_RGB)

    def _create_processing_tab(self):
        
        self.tabs.widget(self.tab_names.index('Pro&cessing')).layout().setAlignment(Qt.AlignTop)

        self.background_group = VHGroup('Background correction', orientation='G')
        self.tabs.add_named_tab('Pro&cessing', self.background_group.gbox)

        self.btn_select_dark_file = QPushButton("Manual selection")
        self.qtext_select_dark_file = QLineEdit()
        self.qtext_select_dark_file.setText('No path')
        self.background_group.glayout.addWidget(QLabel('Dark ref'), 0, 0, 1, 1)
        self.background_group.glayout.addWidget(self.qtext_select_dark_file, 0, 1, 1, 1)
        self.background_group.glayout.addWidget(self.btn_select_dark_file, 0, 2, 1, 1)

        self.btn_select_white_file = QPushButton("Manual selection")
        self.qtext_select_white_file = QLineEdit()
        self.qtext_select_white_file.setText('No path')
        self.background_group.glayout.addWidget(QLabel('White ref'), 1, 0, 1, 1)
        self.background_group.glayout.addWidget(self.qtext_select_white_file, 1, 1, 1, 1)
        self.background_group.glayout.addWidget(self.btn_select_white_file, 1, 2, 1, 1)

        self.btn_select_dark_for_white_file = QPushButton("Manual selection")
        self.qtext_select_dark_for_white_file = QLineEdit()
        self.qtext_select_dark_for_white_file.setText('No path')
        self.background_group.glayout.addWidget(QLabel('Dark ref for White'), 2, 0, 1, 1)
        self.background_group.glayout.addWidget(self.qtext_select_dark_for_white_file, 2, 1, 1, 1)
        self.background_group.glayout.addWidget(self.btn_select_dark_for_white_file, 2, 2, 1, 1)
        self.combo_layer_background = QComboBox()
        self.background_group.glayout.addWidget(QLabel('Layer'), 3, 0, 1, 1)
        self.background_group.glayout.addWidget(self.combo_layer_background, 3, 1, 1, 2)
        self.btn_background_correct = QPushButton("Correct")
        self.background_group.glayout.addWidget(self.btn_background_correct, 4, 0, 1, 3)


        self.destripe_group = VHGroup('Destripe', orientation='G')
        self.tabs.add_named_tab('Pro&cessing', self.destripe_group.gbox)
        self.combo_layer_destripe = QComboBox()
        self.destripe_group.glayout.addWidget(QLabel('Layer'), 0, 0, 1, 1)
        self.destripe_group.glayout.addWidget(self.combo_layer_destripe, 0, 1, 1, 1)
        self.qspin_destripe_width = QSpinBox()
        self.qspin_destripe_width.setRange(1, 1000)
        self.qspin_destripe_width.setValue(100)
        self.destripe_group.glayout.addWidget(QLabel('Savgol Width'), 1, 0, 1, 1)
        self.destripe_group.glayout.addWidget(self.qspin_destripe_width, 1, 1, 1, 1)
        self.btn_destripe = QPushButton("Destripe")
        self.destripe_group.glayout.addWidget(self.btn_destripe, 2, 0, 1, 2)


        self.batch_group = VHGroup('Correct full dataset', orientation='G')
        self.tabs.add_named_tab('Pro&cessing', self.batch_group.gbox)
        
        self.batch_group.glayout.addWidget(QLabel("Crop bands"), 0,0,1,1)
        self.slider_batch_wavelengths = QDoubleRangeSlider(Qt.Horizontal)
        self.slider_batch_wavelengths.setRange(0, 1000)
        self.slider_batch_wavelengths.setSingleStep(1)
        self.slider_batch_wavelengths.setSliderPosition([0, 1000])
        self.batch_group.glayout.addWidget(self.slider_batch_wavelengths, 0,2,1,1)
        
        self.spin_batch_wavelengths_min = QDoubleSpinBox()
        self.spin_batch_wavelengths_min.setRange(0, 1000)
        self.spin_batch_wavelengths_min.setSingleStep(1)
        self.batch_group.glayout.addWidget(self.spin_batch_wavelengths_min, 0, 1, 1, 1)
        self.spin_batch_wavelengths_max = QDoubleSpinBox()
        self.spin_batch_wavelengths_max.setRange(0, 1000)
        self.spin_batch_wavelengths_max.setSingleStep(1)
        self.batch_group.glayout.addWidget(self.spin_batch_wavelengths_max, 0, 3, 1, 1)

        self.check_batch_white = QCheckBox("White correct")
        self.check_batch_destripe = QCheckBox("Destripe")
        self.check_batch_white.setChecked(True)
        self.check_batch_destripe.setChecked(True)
        self.batch_group.glayout.addWidget(self.check_batch_white, 1, 0, 1, 1)
        self.batch_group.glayout.addWidget(self.check_batch_destripe, 2, 0, 1, 1)

        self.spin_chunk_size = QSpinBox()
        self.spin_chunk_size.setRange(1, 10000)
        self.spin_chunk_size.setValue(500)
        self.batch_group.glayout.addWidget(QLabel("Chunk size"), 3, 0, 1, 1)
        self.batch_group.glayout.addWidget(self.spin_chunk_size, 3, 1, 1, 1)
        
        self.btn_batch_correct = QPushButton("Correct and save data")
        self.batch_group.glayout.addWidget(self.btn_batch_correct, 4, 0, 1, 4)

        self.multiexp_group = VHGroup('Correct multiple datasets', orientation='G')
        self.tabs.add_named_tab('Pro&cessing', self.multiexp_group.gbox)
        self.btn_show_multiexp_batch = QPushButton("Process in batch")
        self.multiexp_group.glayout.addWidget(self.btn_show_multiexp_batch, 0, 0, 1, 1)
        self.btn_show_multiexp_batch.clicked.connect(self._on_click_multiexp_batch)
        self.multiexp_batch = None

        self.check_use_dask = QCheckBox("Use dask")
        self.check_use_dask.setChecked(True)
        self.check_use_dask.setToolTip("Use dask to parallelize computation")
        self.tabs.add_named_tab('Pro&cessing', self.check_use_dask)

    def _on_click_multiexp_batch(self):

        if self.multiexp_batch is None:
            self.multiexp_batch = BatchPreprocWidget(
                self.viewer,
                background_correct=self.check_batch_white.isChecked(),
                destripe=self.check_batch_destripe.isChecked(),
                savgol_window=self.qspin_destripe_width.value(),
                min_band=self.slider_batch_wavelengths.value()[0],
                max_band=self.slider_batch_wavelengths.value()[1],
                chunk_size=self.spin_chunk_size.value(),
            )
            self.multiexp_batch.setStyleSheet(get_current_stylesheet())

        self.multiexp_batch.show()


    def _create_mask_tab(self):
            
        self.tabs.widget(self.tab_names.index('Mas&k')).layout().setAlignment(Qt.AlignTop)
        
        self.mask_layersel_group = VHGroup('1. Select layer to use', orientation='G')
        self.tabs.add_named_tab('Mas&k', self.mask_layersel_group.gbox)
        self.combo_layer_mask = QComboBox()
        self.mask_layersel_group.glayout.addWidget(self.combo_layer_mask)

        self.mask_generation_group = VHGroup('2. Create one or more masks', orientation='G')
        self.tabs.add_named_tab('Mas&k', self.mask_generation_group.gbox)

        self.mask_assemble_group = VHGroup('3. Assemble masks', orientation='G')
        self.tabs.add_named_tab('Mas&k', self.mask_assemble_group.gbox)
        
        self.mask_group_border = VHGroup('Border mask', orientation='G')
        self.mask_group_border.gbox.setToolTip("Detect background regions on the borders and remove them")
        self.mask_group_manual = VHGroup('Manual Threshold', orientation='G')
        self.mask_group_manual.gbox.setToolTip("Manually set a threshold on intensity of average imcube")

        self.mask_group_auto = VHGroup('Auto Threshold', orientation='G')
        self.mask_group_auto.gbox.setToolTip("Fit intensity distribution with a Gaussian and set threshold at a given width")

        self.mask_group_ml = VHGroup('Pixel Classifier', orientation='G')
        self.mask_group_ml.gbox.setToolTip("Use a pixel classifier to generate a mask")
        
        self.mask_tabs = TabSet(['Border', 'Manual', 'Auto', 'ML'])
        self.mask_generation_group.glayout.addWidget(self.mask_tabs)
        self.mask_tabs.add_named_tab('Border', self.mask_group_border.gbox)
        self.mask_tabs.add_named_tab('Manual', self.mask_group_manual.gbox)
        self.mask_tabs.add_named_tab('Auto', self.mask_group_auto.gbox)
        #self.mask_tabs.add_named_tab('ML', self.mask_group_ml.gbox)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.mask_group_ml.gbox)
        self.mask_tabs.add_named_tab('ML', scroll)

        for g in [self.mask_group_border, self.mask_group_manual, self.mask_group_auto, self.mask_group_ml]:
            g.glayout.setAlignment(Qt.AlignTop)
        
        # border
        
        self.btn_border_mask = QPushButton("Generate mask")
        self.mask_group_border.glayout.addWidget(self.btn_border_mask, 0, 0, 1, 2)

        # manual
        self.slider_mask_threshold = QDoubleRangeSlider(Qt.Horizontal)
        self.slider_mask_threshold.setRange(0, 1)
        self.slider_mask_threshold.setSingleStep(0.01)
        self.slider_mask_threshold.setSliderPosition([0, 1])
        self.mask_group_manual.glayout.addWidget(QLabel("Min/Max Threshold"), 0, 0, 1, 1)
        self.mask_group_manual.glayout.addWidget(self.slider_mask_threshold, 0, 1, 1, 1)
        self.btn_update_mask = QPushButton("Generate mask")
        self.mask_group_manual.glayout.addWidget(self.btn_update_mask, 1, 0, 1, 2)
        
        # auto
        self.btn_automated_mask = QPushButton("Generate mask")
        self.mask_group_auto.glayout.addWidget(self.btn_automated_mask, 0, 0, 1, 1)
        self.spin_automated_mask_width = QDoubleSpinBox()
        self.spin_automated_mask_width.setToolTip("Set threshold at a given width of the intensity distribution")
        self.spin_automated_mask_width.setRange(0.1, 10)
        self.spin_automated_mask_width.setSingleStep(0.1)
        self.mask_group_auto.glayout.addWidget(QLabel('Intensity distribution Width'), 1, 0, 1, 1)
        self.mask_group_auto.glayout.addWidget(self.spin_automated_mask_width, 1, 1, 1, 1)

        # phasor
        #self.btn_compute_phasor = QPushButton("Compute Phasor")
        #self.mask_group_phasor.glayout.addWidget(self.btn_compute_phasor, 0, 0, 1, 2)
        #self.btn_select_by_phasor = QPushButton("Phasor mask")
        #self.mask_group_phasor.glayout.addWidget(self.btn_select_by_phasor, 1, 0, 1, 2)

        # ml
        from .classifier import ConvPaintSpectralWidget
        self.mlwidget = ConvPaintSpectralWidget(self.viewer)
        self.mask_group_ml.glayout.addWidget(self.mlwidget)
        
        # combine
        self.btn_combine_masks = QPushButton("Combine masks")
        self.mask_assemble_group.glayout.addWidget(self.btn_combine_masks, 0, 0, 1, 2)
        self.btn_clean_mask = QPushButton("Clean mask")
        self.mask_assemble_group.glayout.addWidget(self.btn_clean_mask, 1, 0, 1, 2)
        
    def _create_roi_tab(self):

        self.tabs.widget(self.tab_names.index('&ROI')).layout().setAlignment(Qt.AlignTop)

        self.roi_group = VHGroup('Main ROI', orientation='G')
        self.tabs.add_named_tab('&ROI', self.roi_group.gbox)
        self.btn_add_main_roi = QPushButton("Add main ROI")
        self.btn_add_main_roi.setToolTip("Maximal &ROI only removing fully masked border")
        self.roi_group.glayout.addWidget(self.btn_add_main_roi, 0, 0, 1, 2)

        self.subroi_group = VHGroup('Sub-ROI', orientation='G')
        self.tabs.add_named_tab('&ROI', self.subroi_group.gbox)
        #self.btn_add_sub_roi = QPushButton("Add analysis &ROI")
        #self.roi_group.glayout.addWidget(self.btn_add_sub_roi, 1, 0, 1, 2)
        self.subroi_group.glayout.addWidget(QLabel(
            'Set desired sub-&ROI width and double-click in viewer to place them'), 0, 0, 1, 2)
        self.spin_roi_width = QSpinBox()
        self.spin_roi_width.setRange(1, 1000)
        self.spin_roi_width.setValue(20)
        self.subroi_group.glayout.addWidget(QLabel('Sub-ROI width'), 1, 0, 1, 1)
        self.subroi_group.glayout.addWidget(self.spin_roi_width, 1, 1, 1, 1)

    def _create_export_tab(self):

        self.tabs.widget(self.tab_names.index('&Export-Import')).layout().setAlignment(Qt.AlignTop)

        self.mask_group_project = VHGroup('Project', orientation='G')
        self.tabs.add_named_tab('&Export-Import', self.mask_group_project.gbox)
        self.btn_export = QPushButton("Export Project")
        self.btn_export.setToolTip(
            "Export all info necessary for next steps and to reload the project")
        self.mask_group_project.glayout.addWidget(self.btn_export)
        self.btn_import = QPushButton("Import Project")
        self.mask_group_project.glayout.addWidget(self.btn_import)
        self.check_load_corrected = QCheckBox("Load corrected data if available")
        self.check_load_corrected.setToolTip("Load corrected data instead of raw")
        self.check_load_corrected.setChecked(True)
        self.mask_group_project.glayout.addWidget(self.check_load_corrected)

        # io
        self.mask_group_export = VHGroup('Mask', orientation='G')
        self.tabs.add_named_tab('&Export-Import', self.mask_group_export.gbox)
        self.btn_save_mask = QPushButton("Save mask")
        self.btn_save_mask.setToolTip("Save only mask as tiff")
        self.mask_group_export.glayout.addWidget(self.btn_save_mask)
        self.btn_load_mask = QPushButton("Load mask")
        self.mask_group_export.glayout.addWidget(self.btn_load_mask)
        
        self.mask_group_capture = VHGroup('Other exports', orientation='G')
        self.tabs.add_named_tab('&Export-Import', self.mask_group_capture.gbox)
        self.btn_snapshot = QPushButton("Snapshot")
        self.btn_snapshot.setToolTip("Save snapshot of current viewer")
        self.mask_group_capture.glayout.addWidget(self.btn_snapshot, 0, 0, 1, 2)
        self.lineedit_rgb_tiff = QLineEdit()
        self.lineedit_rgb_tiff.setText('rgb.tiff')
        self.mask_group_capture.glayout.addWidget(self.lineedit_rgb_tiff, 1, 0, 1, 1)
        self.btn_save_rgb_tiff = QPushButton("Save RGB tiff")
        self.btn_save_rgb_tiff.setToolTip("Save current RGB layer as high-res tiff")
        self.mask_group_capture.glayout.addWidget(self.btn_save_rgb_tiff, 1, 1, 1, 1)

    def _create_options_tab(self):
        
        self.crop_group = VHGroup('Crop selection', orientation='G')

        #self.tabs.add_named_tab('&Options', self.crop_group.gbox)

        self.check_use_external_ref = QCheckBox("Use external reference")
        self.check_use_external_ref.setChecked(True)

        crop_bounds_name = ['Min row', 'Max row', 'Min col', 'Max col']
        self.crop_bounds = {x: QSpinBox() for x in crop_bounds_name}
        for ind, c in enumerate(crop_bounds_name):
            self.crop_group.glayout.addWidget(QLabel(c), ind, 0, 1, 1)
            self.crop_group.glayout.addWidget(self.crop_bounds[c], ind, 1, 1, 1)

        self.check_use_crop = QCheckBox("Use crop")
        self.btn_refresh_crop = QPushButton("Refresh crop")
        self.crop_group.glayout.addWidget(self.check_use_crop, ind+1, 0, 1, 1)
        self.crop_group.glayout.addWidget(self.btn_refresh_crop, ind+1, 1, 1, 1)

    def _create_plot_tab(self):

        # Plot tab
        self.scan_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.scan_plot.axes.set_xlabel('Wavelength (nm)', color='white')
        self.scan_plot.axes.set_ylabel('Intensity', color='white')
        self.tabs.add_named_tab('P&lotting', self.scan_plot)

        self.check_remove_continuum = QCheckBox("Remove continuum")
        self.check_remove_continuum.setChecked(True)
        self.tabs.add_named_tab('P&lotting', self.check_remove_continuum)

        self.slider_spectrum_savgol = QDoubleSlider(Qt.Horizontal)
        self.slider_spectrum_savgol.setRange(1, 100)
        self.slider_spectrum_savgol.setSingleStep(1)
        self.slider_spectrum_savgol.setSliderPosition(5)
        self.tabs.add_named_tab('P&lotting', self.slider_spectrum_savgol)
        


    def add_connections(self):
        """Add callbacks"""

        self.btn_select_export_folder.clicked.connect(self._on_click_select_export_folder)
        self.btn_select_imhdr_file.clicked.connect(self._on_click_select_imhdr)
        self.btn_select_white_file.clicked.connect(self._on_click_select_white_file)
        self.btn_select_dark_file.clicked.connect(self._on_click_select_dark_file)
        self.btn_select_dark_for_white_file.clicked.connect(self._on_click_select_dark_for_white_file)
        self.btn_destripe.clicked.connect(self._on_click_destripe)
        self.btn_background_correct.clicked.connect(self._on_click_background_correct)
        self.rgb_widget.btn_RGB.clicked.connect(self._update_threshold_limits)
        self.btn_select_all.clicked.connect(self._on_click_select_all)
        self.check_sync_bands_rgb.stateChanged.connect(self._on_click_sync_RGB)
        self.rgb_widget.btn_dislpay_as_rgb.clicked.connect(self._update_threshold_limits)
        self.check_use_crop.stateChanged.connect(self._on_click_use_crop)
        self.btn_refresh_crop.clicked.connect(self._on_click_use_crop)
        self.btn_batch_correct.clicked.connect(self._on_click_batch_correct)
        self.slider_batch_wavelengths.valueChanged.connect(self._on_change_batch_wavelengths)
        self.spin_batch_wavelengths_min.valueChanged.connect(self._on_change_spin_batch_wavelengths)
        self.spin_batch_wavelengths_max.valueChanged.connect(self._on_change_spin_batch_wavelengths)
        
        # mask
        self.btn_border_mask.clicked.connect(self._on_click_remove_borders)
        self.btn_update_mask.clicked.connect(self._on_click_intensity_threshold)
        self.btn_automated_mask.clicked.connect(self._on_click_automated_threshold)
        #self.btn_compute_phasor.clicked.connect(self._on_click_compute_phasor)
        #self.btn_select_by_phasor.clicked.connect(self._on_click_select_by_phasor)
        self.btn_combine_masks.clicked.connect(self._on_click_combine_masks)
        self.btn_clean_mask.clicked.connect(self._on_click_clean_mask)
        self.combo_layer_mask.currentIndexChanged.connect(self._on_select_layer_for_mask)

        # &ROI
        self.btn_add_main_roi.clicked.connect(self._on_click_add_main_roi)

        # capture
        self.btn_save_mask.clicked.connect(self._on_click_save_mask)
        self.btn_load_mask.clicked.connect(self._on_click_load_mask)
        self.btn_snapshot.clicked.connect(self._on_click_snapshot)
        self.btn_export.clicked.connect(self.export_project)
        self.btn_import.clicked.connect(self.import_project)
        self.btn_save_rgb_tiff.clicked.connect(self._on_click_save_rgb_tiff)
        
        # mouse
        self.viewer.mouse_move_callbacks.append(self._shift_move_callback)
        self.viewer.mouse_double_click_callbacks.append(self._add_analysis_roi)
        self.slider_spectrum_savgol.valueChanged.connect(self.update_spectral_plot)
        self.check_remove_continuum.stateChanged.connect(self.update_spectral_plot)

        # layer callbacks
        self.viewer.layers.events.inserted.connect(self._update_combo_layers_destripe)
        self.viewer.layers.events.removed.connect(self._update_combo_layers_destripe)
        self.viewer.layers.events.inserted.connect(self._update_combo_layers_background)
        self.viewer.layers.events.removed.connect(self._update_combo_layers_background)

    def _on_click_select_export_folder(self):
        """Interactively select folder to analyze"""

        self.export_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        self.export_path_display.setText(self.export_folder.as_posix())

    def _on_select_file(self):
        
        success = self.open_file()
        if not success:
            return False
        
    def _on_click_select_imhdr(self):
        """Interactively select hdr file"""

        imhdr_path = Path(QFileDialog.getOpenFileName(self, "Select file")[0])
        if imhdr_path.parent.suffix == '.zarr':
            imhdr_path = imhdr_path.parent
        self.set_paths(imhdr_path)
        self._on_select_file()

    def _on_click_select_white_file(self):
        """Interactively select white reference"""
        
        self.white_file_path = Path(QFileDialog.getOpenFileName(self, "Select White Ref")[0])
        self.qtext_select_white_file.setText(self.white_file_path.as_posix())

    def _on_click_select_dark_file(self):
        """Interactively select white reference"""
        
        self.dark_for_white_file_path = Path(QFileDialog.getOpenFileName(self, "Select Dark Ref ofr white")[0])
        self.qtext_select_dark_file.setText(self.dark_for_white_file_path.as_posix())

    def _on_click_select_dark_for_white_file(self):
        """Interactively select white reference"""
        
        self.dark_for_im_file_path = Path(QFileDialog.getOpenFileName(self, "Select Dark Ref for image")[0])
        self.qtext_select_dark_for_white_file.setText(self.dark_for_im_file_path.as_posix())

    def set_paths(self, imhdr_path):
        """Update image and white/dark image paths"""

        self.white_file_path = None
        self.dark_for_white_file_path = None
        self.dark_for_im_file_path = None
        
        self.imhdr_path = Path(imhdr_path)
        self.imhdr_path_display.setText(self.imhdr_path.as_posix())

        if self.check_use_external_ref.isChecked():
            try:
                refpath = None
                wr_files = list(self.imhdr_path.parent.parent.parent.glob('*_WR*'))
                for wr in wr_files:
                    wr_first_part = wr.name.split('WR')[0]
                    if wr_first_part in self.imhdr_path.name:
                        refpath = wr
                if refpath is None:
                    raise Exception('No matching white reference folder found')
        

                #name_parts = self.imhdr_path.name.split('_')
                #refpath = list(self.imhdr_path.parent.parent.parent.glob('*'+name_parts[1]+'_WR*'))[0]
                
                self.white_file_path = list(refpath.joinpath('capture').glob('WHITE*.hdr'))[0]
                self.dark_for_white_file_path = list(refpath.joinpath('capture').glob('DARK*.hdr'))[0]

                self.qtext_select_white_file.setText(self.white_file_path.as_posix())
                self.qtext_select_dark_file.setText(self.dark_for_white_file_path.as_posix())
            except:
                warnings.warn('Low exposure White and dark reference files not found. Please select manually.')
            try:
                self.dark_for_im_file_path = list(self.imhdr_path.parent.glob('DARK*.hdr'))[0]
                self.qtext_select_dark_for_white_file.setText(self.dark_for_im_file_path.as_posix())
            except:
                warnings.warn('No Dark Ref found for image')

        else:
            self.dark_for_white_file_path = None
            self.dark_for_im_file_path = list(self.imhdr_path.parent.glob('DARK*.hdr'))[0]
            self.white_file_path = list(self.imhdr_path.parent.glob('WHITE*.hdr'))[0]


    def open_file(self):
        """Open file in napari"""

        # clear existing layers.
        while len(self.viewer.layers) > 0:
            self.viewer.layers.clear()
        
        # if file list is empty stop here
        if self.imhdr_path is None:
            return False
        
        # open image
        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Opening image")
            image_name = self.imhdr_path.name
            
            # reset acquisition index if new image is selected
            #if image_name != self.current_image_name:
            self.current_image_name = image_name
            #self.imagechannels = ImChannels(self.imhdr_path)
            if (self.check_load_corrected.isChecked()) and (self.export_folder is not None):
                if not self.export_folder.joinpath('corrected.zarr').exists():
                    warnings.warn('Corrected image not found. Loading raw image instead.')
                    self.imagechannels = ImChannels(self.imhdr_path)
                else:
                    self.imagechannels = ImChannels(self.export_folder.joinpath('corrected.zarr'))
            else:
                self.imagechannels = ImChannels(self.imhdr_path)

            self.crop_bounds['Min row'].setMaximum(self.imagechannels.nrows-1)
            self.crop_bounds['Max row'].setMaximum(self.imagechannels.nrows)
            self.crop_bounds['Min col'].setMaximum(self.imagechannels.ncols-1)
            self.crop_bounds['Max col'].setMaximum(self.imagechannels.ncols)
            self.crop_bounds['Max row'].setValue(self.imagechannels.nrows)
            self.crop_bounds['Max col'].setValue(self.imagechannels.ncols)

            self.row_bounds = [0, self.imagechannels.nrows]
            self.col_bounds = [0, self.imagechannels.ncols]
            
            self.qlist_channels._update_channel_list(imagechannels=self.imagechannels)

            self.rgb_widget.imagechannels = self.imagechannels
            self.rgb_widget._on_click_RGB()
            # add imcube from RGB
            [self.qlist_channels.item(x).setSelected(True) for x in self.rgb_widget.rgb_ch]
            self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)
            self._update_threshold_limits()

            #self._add_roi_layer()
            self._add_mask()
            self._update_range_wavelength()
            
        self.viewer.window._status_bar._toggle_activity_dock(False)

    def _on_click_sync_RGB(self, event=None):
        """Select same channels for imcube as loaded for RGB"""
        
        if not self.check_sync_bands_rgb.isChecked():
            self.qlist_channels.setEnabled(True)
            self.btn_select_all.setEnabled(True)
        else:
            self.qlist_channels.setEnabled(False)
            self.btn_select_all.setEnabled(False)
            self.qlist_channels.clearSelection()
            [self.qlist_channels.item(x).setSelected(True) for x in self.rgb_widget.rgb_ch]
            self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)
            self._update_threshold_limits()

    def _update_range_wavelength(self):
        """Update range of wavelength slider"""
        
        wavelengths = np.array(self.imagechannels.channel_names).astype(float)
        self.slider_batch_wavelengths.setRange(np.round(wavelengths[0]), np.round(wavelengths[-1]))
        self.slider_batch_wavelengths.setSliderPosition([np.round(wavelengths[0]), np.round(wavelengths[-1])])

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
        
        layers = ['manual-mask', 'clean-mask', 'intensity-mask',
                  'complete-mask', 'border-mask', 
                  'ml-mask', 'main-roi', 'rois']
        for l in layers:
            if l in self.viewer.layers:
                if isinstance(self.viewer.layers[l], napari.layers.labels.labels.Labels):
                    self.viewer.layers[l].data = self.viewer.layers[l].data[self.row_bounds[0]:self.row_bounds[1], self.col_bounds[0]:self.col_bounds[1]]
                self.translate_layer(l)
        
        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)


    def _add_roi_layer(self):
        """Add &ROI layers to napari viewer"""

        edge_width = np.min([10, self.viewer.layers['imcube'].data.shape[1]//100])
        if 'main-roi' not in self.viewer.layers:
            self.roi_layer = self.viewer.add_shapes(
                ndim = 2,
                name='main-roi', edge_color='blue', face_color=np.array([0,0,0,0]), edge_width=edge_width)
        
        if 'rois' not in self.viewer.layers:
            self.roi_layer = self.viewer.add_shapes(
                ndim = 2,
                name='rois', edge_color='red', face_color=np.array([0,0,0,0]), edge_width=edge_width)
         
    def _on_click_add_main_roi(self):

        self._add_roi_layer()

        if 'clean-mask' in self.viewer.layers:
            mask = self.viewer.layers['clean-mask'].data.copy()
        else:
            if 'complete-mask' not in self.viewer.layers:
                self._on_click_combine_masks()
            mask = self.viewer.layers['complete-mask'].data.copy() 

        bounds_col = np.where(mask.min(axis=0)==0)
        bounds_row = np.where(mask.min(axis=1)==0)

        self.set_main_roi_bounds(min_col=bounds_col[0][0], max_col=bounds_col[0][-1], min_row=bounds_row[0][0], max_row=bounds_row[0][-1])

        new_roi = [
            [self.mainroi_min_row,self.mainroi_min_col],
            [self.mainroi_max_row,self.mainroi_min_col],
            [self.mainroi_max_row,self.mainroi_max_col],
            [self.mainroi_min_row,self.mainroi_max_col]]
        self.viewer.layers['main-roi'].data = []
        self.viewer.layers['main-roi'].add_rectangles(new_roi, edge_color='b')

    def set_main_roi_bounds(self, min_col, max_col, min_row, max_row):
            
        self.mainroi_min_col = min_col
        self.mainroi_max_col = max_col
        self.mainroi_min_row = min_row
        self.mainroi_max_row = max_row

    def _add_analysis_roi(self, viewer=None, event=None, cursor_pos=None):
        """Add roi to layer"""

        if cursor_pos is None:
            cursor_pos = np.rint(self.viewer.cursor.position).astype(int)
            
        if self.mainroi_min_row is None:
            min_row = 0
            max_row = self.imagechannels.nrows
        else:
            min_row = self.mainroi_min_row
            max_row = self.mainroi_max_row
        new_roi = [
            [min_row, cursor_pos[2]-self.spin_roi_width.value()//2],
            [max_row,cursor_pos[2]-self.spin_roi_width.value()//2],
            [max_row,cursor_pos[2]+self.spin_roi_width.value()//2],
            [min_row,cursor_pos[2]+self.spin_roi_width.value()//2]]
        
        if not 'rois' in self.viewer.layers:
            self._add_roi_layer()
        self.viewer.layers['rois'].add_rectangles(new_roi, edge_color='r')


    def _add_mask(self):
        self.mask_layer = self.viewer.add_labels(
            np.zeros((self.imagechannels.nrows,self.imagechannels.ncols), dtype=np.uint8),
            name='manual-mask')

    def _on_click_destripe(self):
        """Destripe image"""
        
        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Destriping image")

        selected_layer = self.combo_layer_destripe.currentText()
        if (selected_layer == 'None') or (selected_layer == 'imcube'):
            data_destripe = self.viewer.layers['imcube'].data.copy()
        elif selected_layer == 'imcube_corrected':
            data_destripe = self.viewer.layers['imcube_corrected'].data.copy()
        elif selected_layer == 'RGB':
            data_destripe = np.stack([self.viewer.layers[x].data for x in ['red', 'green', 'blue']], axis=0)
        
        for d in range(data_destripe.shape[0]):
            #data_destripe[d] = pystripe.filter_streaks(data_destripe[d].T, sigma=[128, 256], level=7, wavelet='db2').T
            width = self.qspin_destripe_width.value()
            data_destripe[d] = savgol_destripe(data_destripe[d], width=width, order=2)

        if (selected_layer == 'RGB') | (self.check_sync_bands_rgb.isChecked()):
            for ind, x in enumerate(['red', 'green', 'blue']):
                self.viewer.layers[x].data = data_destripe[ind]
        
        if (selected_layer == 'None') or (selected_layer == 'imcube') | (selected_layer == 'imcube_corrected') | (self.check_sync_bands_rgb.isChecked()):
            if 'imcube_destripe' in self.viewer.layers:
                self.viewer.layers['imcube_destripe'].data = data_destripe
            else:
                self.viewer.add_image(data_destripe, name='imcube_destripe', rgb=False)
        self.viewer.window._status_bar._toggle_activity_dock(False)


    def _on_click_background_correct(self, event=None):
        """White correct image"""
        
        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("White correcting image")

            selected_layer = self.combo_layer_background.currentText()
            if selected_layer == 'imcube':
                channel_indices = self.qlist_channels.channel_indices
            elif selected_layer == 'RGB':
                channel_indices = np.sort(self.rgb_widget.rgb_ch)

            col_bounds = (self.col_bounds if self.check_use_crop.isChecked() else None)
            white_data, dark_data, dark_for_white_data = load_white_dark(
                white_file_path=self.white_file_path,
                dark_for_im_file_path=self.dark_for_im_file_path,
                dark_for_white_file_path=self.dark_for_white_file_path,
                channel_indices=channel_indices,
                col_bounds=col_bounds,
                clean_white=True
                )

            if (selected_layer == 'imcube') | (self.check_sync_bands_rgb.isChecked()):
                im_corr = white_dark_correct(
                    self.viewer.layers['imcube'].data, white_data, dark_data, dark_for_white_data)
                
                if 'imcube_corrected' in self.viewer.layers:
                    self.viewer.layers['imcube_corrected'].data = im_corr
                else:
                    self.viewer.add_image(im_corr, name='imcube_corrected', rgb=False)
                    self.viewer.layers['imcube_corrected'].translate = (0, self.row_bounds[0], self.col_bounds[0])

            if (selected_layer == 'RGB') | (self.check_sync_bands_rgb.isChecked()):
                sorted_rgb_indices = np.argsort(self.rgb_widget.rgb_ch)
                rgb_sorted = np.asarray(['red', 'green', 'blue'])[sorted_rgb_indices]
                rgb_sorted = [str(x) for x in rgb_sorted]

                im_corr = white_dark_correct(
                    np.stack([self.viewer.layers[x].data for x in rgb_sorted], axis=0), 
                    white_data, dark_data, dark_for_white_data)
                
                for ind, c in enumerate(rgb_sorted):
                    self.viewer.layers[c].data = im_corr[ind]
                    update_contrast_on_layer(self.viewer.layers[c])
                    self.viewer.layers[c].refresh()

        self.viewer.window._status_bar._toggle_activity_dock(False)

    def _update_combo_layers_destripe(self):
        
        admit_layers = ['imcube', 'imcube_corrected']
        self.combo_layer_destripe.clear()
        self.combo_layer_mask.clear()
        self.combo_layer_destripe.addItem('RGB')
        for a in admit_layers:
            if a in self.viewer.layers:
                self.combo_layer_destripe.addItem(a)
                self.combo_layer_mask.addItem(a)

    def _update_combo_layers_background(self):
        
        admit_layers = ['imcube']
        self.combo_layer_background.clear()
        self.combo_layer_background.addItem('RGB')
        for a in admit_layers:
            if a in self.viewer.layers:
                self.combo_layer_background.addItem(a)      
        

    def _on_change_batch_wavelengths(self, event):

        self.spin_batch_wavelengths_min.valueChanged.disconnect(self._on_change_spin_batch_wavelengths)
        self.spin_batch_wavelengths_max.valueChanged.disconnect(self._on_change_spin_batch_wavelengths)
        self.spin_batch_wavelengths_max.setMinimum(self.slider_batch_wavelengths.minimum())
        self.spin_batch_wavelengths_max.setMaximum(self.slider_batch_wavelengths.maximum())
        self.spin_batch_wavelengths_min.setMinimum(self.slider_batch_wavelengths.minimum())
        self.spin_batch_wavelengths_min.setMaximum(self.slider_batch_wavelengths.maximum())
        self.spin_batch_wavelengths_max.setValue(self.slider_batch_wavelengths.value()[1])
        self.spin_batch_wavelengths_min.setValue(self.slider_batch_wavelengths.value()[0])
        self.spin_batch_wavelengths_min.valueChanged.connect(self._on_change_spin_batch_wavelengths)
        self.spin_batch_wavelengths_max.valueChanged.connect(self._on_change_spin_batch_wavelengths)

    def _on_change_spin_batch_wavelengths(self, event):

        self.slider_batch_wavelengths.valueChanged.disconnect(self._on_change_batch_wavelengths)
        self.slider_batch_wavelengths.setSliderPosition([self.spin_batch_wavelengths_min.value(), self.spin_batch_wavelengths_max.value()])
        self.slider_batch_wavelengths.valueChanged.connect(self._on_change_batch_wavelengths)

    def _on_click_batch_correct(self):

        if self.export_folder is None:
            self._on_click_select_export_folder()

        min_max_band = [self.slider_batch_wavelengths.value()[0], self.slider_batch_wavelengths.value()[1]]
        
        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Preprocessing full image")
        
            correct_save_to_zarr(
                imhdr_path=self.imhdr_path,
                white_file_path=self.white_file_path,
                dark_for_im_file_path=self.dark_for_im_file_path,
                dark_for_white_file_path=self.dark_for_white_file_path,
                zarr_path=self.export_folder.joinpath('corrected.zarr'),
                band_indices=None,
                min_max_bands=min_max_band,
                background_correction=self.check_batch_white.isChecked(),
                destripe=self.check_batch_destripe.isChecked(),
                use_dask=self.check_use_dask.isChecked(),
                chunk_size=self.spin_chunk_size.value()
                )
            
            # reload corrected image as zarr
            self.open_file()
            
        self.viewer.window._status_bar._toggle_activity_dock(False)

    def get_summary_image_for_mask(self):
        """Get summary image"""

        selected_layer = self.combo_layer_mask.currentText()
        im = np.mean(self.viewer.layers[selected_layer].data, axis=0)
        return im
    
    def _on_select_layer_for_mask(self):
        
        selected_layer = self.combo_layer_mask.currentText()
        if selected_layer in self.viewer.layers:
            im = np.mean(self.viewer.layers[selected_layer].data, axis=0)
            self.slider_mask_threshold.setRange(im.min(), im.max())
            self.slider_mask_threshold.setSliderPosition([im.min(), im.max()])


    def translate_layer(self, mask_name):
        """Translate mask"""

        self.viewer.layers[mask_name].translate = (self.row_bounds[0], self.col_bounds[0])


    def _on_click_remove_borders(self):
        """Remove borders from image"""
        
        im = self.get_summary_image_for_mask()

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
        self.translate_layer('border-mask')
        self.viewer.layers['border-mask'].refresh()

    def _update_threshold_limits(self):
        
        im = self.get_summary_image_for_mask()
        self.slider_mask_threshold.setRange(im.min(), im.max())
        self.slider_mask_threshold.setSliderPosition([im.min(), im.max()])

    def _on_click_automated_threshold(self):
        """Automatically set threshold for mask based on mean RGB pixel intensity"""

        im = self.get_summary_image_for_mask()
        if 'border-mask' in self.viewer.layers:
            pix_selected = im[self.viewer.layers['border-mask'].data == 0]
        else:
            pix_selected = np.ravel(im)
        med_val, std_val = fit_1dgaussian_without_outliers(data=pix_selected[::5])
        #self.slider_mask_threshold.setRange(im.min(), im.max())
        fact = self.spin_automated_mask_width.value()
        self.slider_mask_threshold.setSliderPosition(
            [
                np.max([med_val - fact*std_val, self.slider_mask_threshold.minimum()]),
                np.min([med_val + fact*std_val, self.slider_mask_threshold.maximum()])
             ]
        ),
        #self._on_click_update_mask()
        self._on_click_intensity_threshold()


    def _on_click_intensity_threshold(self, event=None):
        """Create mask based on intensity threshold"""

        data = self.get_summary_image_for_mask()
        mask = ((data < self.slider_mask_threshold.value()[0]) | (data > self.slider_mask_threshold.value()[1])).astype(np.uint8)
        self.update_mask(mask, 'intensity-mask')
    
    '''def _on_click_update_mask(self):
        """Update mask based on current threshold"""
        
        data = self.get_summary_image_for_mask()
        mask = ((data < self.slider_mask_threshold.value()[0]) | (data > self.slider_mask_threshold.value()[1])).astype(np.uint8)
        self.update_mask(mask)
        self.translate_layer('mask')
        self.viewer.layers['mask'].refresh()'''

    def update_mask(self, mask, name='mask'):

        if name in self.viewer.layers:
            self.viewer.layers[name].data = mask
        else:
            self.viewer.add_labels(mask, name=name)

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

        mask_complete = np.zeros((self.imagechannels.nrows,self.imagechannels.ncols), dtype=np.uint8)
        if 'manual-mask' in self.viewer.layers:
            mask_complete = mask_complete + self.viewer.layers['manual-mask'].data
        if 'intensity-mask' in self.viewer.layers:
            mask_complete = mask_complete + self.viewer.layers['intensity-mask'].data
        if 'phasor-mask' in self.viewer.layers:
            mask_complete = mask_complete + self.viewer.layers['phasor-mask'].data
        if 'border-mask' in self.viewer.layers:
            mask_complete = mask_complete + self.viewer.layers['border-mask'].data
        if 'ml-mask' in self.viewer.layers:
            mask_complete = mask_complete + (self.viewer.layers['ml-mask'].data == 1)
        
        mask_complete = np.asarray((mask_complete > 0), np.uint8)

        if 'complete-mask' in self.viewer.layers:
            self.viewer.layers['complete-mask'].data = mask_complete
        else:
            self.viewer.add_labels(mask_complete, name='complete-mask')
        self.translate_layer('complete-mask')

    def _on_click_clean_mask(self):
        
        if 'complete-mask' not in self.viewer.layers:
            self._on_click_combine_masks()
        mask = self.viewer.layers['complete-mask'].data == 0
        mask_lab = skimage.morphology.label(mask)
        mask_prop = skimage.measure.regionprops_table(mask_lab, properties=('label', 'area'))
        final_mask = mask_lab == mask_prop['label'][np.argmax(mask_prop['area'])]
        mask_filled = binary_fill_holes(final_mask)
        mask_filled = (mask_filled == 0).astype(np.uint8)
        self.viewer.add_labels(mask_filled, name='clean-mask')
        self.translate_layer('clean-mask')

    def _on_click_save_mask(self):
        """Save mask to file"""

        if self.export_folder is None: 
            self._on_click_select_export_folder()

        if 'clean-mask' in self.viewer.layers:
            mask = self.viewer.layers['clean-mask'].data
        elif 'complete-mask' in self.viewer.layers:
            mask = self.viewer.layers['complete-mask'].data
        #elif 'mask' in self.viewer.layers:
        #    mask = self.viewer.layers['mask'].data
        else:
            mask = np.zeros((self.imagechannels.nrows,self.imagechannels.ncols), dtype=np.uint8)
            warnings.warn('No mask found. Uinsg empty mask.')

        save_mask(mask, get_mask_path(self.export_folder))

    def _on_click_load_mask(self):
        """Load mask from file"""
        
        mask_path = get_mask_path(self.export_folder)
        if mask_path.exists():
            mask = load_mask(mask_path)
            self.update_mask(mask)
        else:
            warnings.warn('No mask found')

    def _on_click_snapshot(self):
        """Save snapshot of viewer"""

        if self.export_folder is None: 
            self._on_click_select_export_folder()

        self.viewer.screenshot(str(self.export_folder.joinpath('snapshot.png')))

    def _on_click_save_rgb_tiff(self):
        """Save RGB image to tiff file"""

        rgb = ['red', 'green', 'blue']
        image_list = [self.viewer.layers[c].data for c in rgb]
        contrast_list = [self.viewer.layers[c].contrast_limits for c in rgb]
        save_rgb_tiff_image(image_list, contrast_list, self.export_folder.joinpath(self.lineedit_rgb_tiff.text()))

    def _on_change_select_bands(self, event=None):

        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)

    def _on_click_select_all(self):
        self.qlist_channels.selectAll()
        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)


    def _get_channel_name_from_index(self, index):
        
        if self.imagechannels is None:
            return None
        return self.imagechannels.channel_names[index]
    
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
            self.spectral_pixel = self.viewer.layers['imcube'].data[
                :, self.cursor_pos[1]-self.row_bounds[0], self.cursor_pos[2]-self.col_bounds[0]
            ]
            self.update_spectral_plot()

    def update_spectral_plot(self, event=None):
            
        if self.spectral_pixel is None:
            return

        self.scan_plot.axes.clear()
        self.scan_plot.axes.set_xlabel('Wavelength (nm)', color='white')
        self.scan_plot.axes.set_ylabel('Intensity', color='white')

        spectral_pixel = np.array(self.spectral_pixel, dtype=np.float64)
        
        if self.check_remove_continuum.isChecked(): 
            spectral_pixel = remove_continuum(spectral_pixel, self.qlist_channels.bands)

        filter_window = int(self.slider_spectrum_savgol.value())
        if filter_window > 3:
            spectral_pixel = savgol_filter(spectral_pixel, window_length=filter_window, polyorder=3)

        self.scan_plot.axes.plot(self.qlist_channels.bands, spectral_pixel)
        
        self.scan_plot.canvas.figure.canvas.draw()

    def save_params(self):
        """Save parameters"""
        
        if self.export_folder is None:
            self._on_click_select_export_folder()

        if 'main-roi' not in self.viewer.layers:
            mainroi = []
        else:
            mainroi = [list(x.flatten()) for x in self.viewer.layers['main-roi'].data]
            mainroi = [[x.item() for x in y] for y in mainroi]

        if 'rois' not in self.viewer.layers:
            rois = []
        else:
            rois = [list(x.flatten()) for x in self.viewer.layers['rois'].data]
            rois = [[x.item() for x in y] for y in rois]

        self.params.project_path = self.export_folder
        self.params.file_path = self.imhdr_path
        self.params.white_path = self.white_file_path
        self.params.dark_for_im_path = self.dark_for_im_file_path
        self.params.dark_for_white_path = self.dark_for_white_file_path
        self.params.main_roi = mainroi
        self.params.rois = rois
        self.params.location = self.metadata_location.text()
        self.params.scale = self.spinbox_metadata_scale.value()
        self.params.rgb = self.rgb_widget.rgb

        self.params.save_parameters()

    def export_project(self):
        """Export data"""

        if self.export_folder is None:
            self._on_click_select_export_folder()

        self.save_params()

        self._on_click_save_mask()

    def import_project(self):
        
        if self.export_folder is None:
            self._on_click_select_export_folder()

        self.params = load_project_params(folder=self.export_folder)

        # files
        self.imhdr_path = Path(self.params.file_path)
        self.white_file_path = Path(self.params.white_path)
        self.dark_for_im_file_path = Path(self.params.dark_for_im_path)
        self.dark_for_white_file_path = Path(self.params.dark_for_white_path)

        # set defaults
        self.rgb_widget.set_rgb(self.params.rgb)

        # load data
        self._on_select_file()
        self._on_click_load_mask()

        # metadata
        self.metadata_location.setText(self.params.location)
        self.spinbox_metadata_scale.setValue(self.params.scale)

        # rois
        self._add_roi_layer()
        mainroi = [np.array(x).reshape(4,2) for x in self.params.main_roi]
        if mainroi:
            mainroi[0] = mainroi[0].astype(int)
            self.viewer.layers['main-roi'].add_rectangles(mainroi, edge_color='b')
            self.set_main_roi_bounds(
            min_col=mainroi[0][:,1].min(),
            max_col=mainroi[0][:,1].max(),
            min_row=mainroi[0][:,0].min(),
            max_row=mainroi[0][:,0].max()
        )
        rois = [np.array(x).reshape(4,2) for x in self.params.rois]
        if rois:
            self.viewer.layers['rois'].add_rectangles(rois, edge_color='r')