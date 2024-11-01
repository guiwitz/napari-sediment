"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from pathlib import Path
import warnings
import os
import napari
from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget,
                            QLabel, QFileDialog, QComboBox,
                            QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,
                            QScrollArea, QGridLayout, QVBoxLayout)
from qtpy.QtCore import Qt
from superqt import QDoubleRangeSlider, QLabeledDoubleRangeSlider, QDoubleSlider
from magicgui.widgets import FileEdit
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
from ..utilities._reader import read_spectral
from ..utilities.sediproc import (white_dark_correct, load_white_dark,
                       phasor, remove_top_bottom, remove_left_right,
                       fit_1dgaussian_without_outliers, correct_save_to_zarr,
                       savgol_destripe)
from ..data_structures.imchannels import ImChannels
from ..utilities.io import save_mask, load_mask, load_project_params
from ..data_structures.parameters import Param
from ..widget_utilities.spectralplotter import SpectralPlotter
from ..widget_utilities.classifier import ConvPaintSpectralWidget
from ..widget_utilities.channel_widget import ChannelWidget
from ..utilities.images import save_rgb_tiff_image
from ..widget_utilities.rgb_widget import RGBWidget
from ..utilities.utils import update_contrast_on_layer
from .batch_preproc_widget import BatchPreprocWidget


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
        self.spectral_pixel = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['&Main', 'Pro&cessing', '&ROI',
                          'Mas&k', 'I&O', 'P&lotting','Meta&data']
        self.tabs = TabSet(self.tab_names,
                           tab_layouts=[QVBoxLayout(), QVBoxLayout(), QVBoxLayout(), QVBoxLayout(),
                                        QVBoxLayout(), QGridLayout(), QGridLayout()])

        self.main_layout.addWidget(self.tabs)

        self._create_main_tab()
        self._create_options_tab()
        self._create_processing_tab()
        self._create_mask_tab()
        self._create_roi_tab()
        self._create_export_tab()
        self._create_plot_tab()
        self._create_metadata_tab()

        self.add_connections()

    def new_view(self):
        #import napari
        self.viewer2 = napari.Viewer()

    def _create_main_tab(self):
        """
        Generates the "Main" tab and its elements.
        """

        # Elements "Files and folders"
        self.files_group = VHGroup('Files and folders', orientation='G')
        self.files_group.gbox.setToolTip("Either select data and Project export location or select Project location and import project.")
        self.tabs.add_named_tab('&Main', self.files_group.gbox)

        ### Elements "Select hdr file" ###
        self.btn_select_imhdr_file = QPushButton("Select hdr file")
        self.btn_select_imhdr_file.setToolTip("Select a file with .hdr extension")
        self.imhdr_path_display = QLineEdit("No path")
        self.files_group.glayout.addWidget(self.btn_select_imhdr_file, 0, 0, 1, 1)
        self.files_group.glayout.addWidget(self.imhdr_path_display, 0, 1, 1, 1)

        ### Elements "Set Project folder" ###
        self.btn_select_export_folder = QPushButton("Set Project folder")
        self.btn_select_export_folder.setToolTip(
            "Select a folder where to save the results and intermeditate files")
        self.export_path_display = QLineEdit("No path")
        self.files_group.glayout.addWidget(self.btn_select_export_folder, 1, 0, 1, 1)
        self.files_group.glayout.addWidget(self.export_path_display, 1, 1, 1, 1)

        ### Button "Import Project" ###
        self.btn_import = QPushButton("Import Project")
        self.files_group.glayout.addWidget(self.btn_import, 2, 0, 1, 1)

        ### Checkbox "Load corrected data if available" ###
        self.check_load_corrected = QCheckBox("Load corrected data if available")
        self.check_load_corrected.setToolTip("Load corrected data instead of raw")
        self.check_load_corrected.setChecked(True)
        self.files_group.glayout.addWidget(self.check_load_corrected, 2, 1, 1, 1)

        ### Button "Export Project" ###
        self.btn_export = QPushButton("Export Project")
        self.btn_export.setToolTip(
            "Export all info necessary for next steps and to reload the project")
        self.files_group.glayout.addWidget(self.btn_export, 3, 0, 1, 1)

        # Elements "Bands"
        self.main_group = VHGroup('Bands', orientation='G')
        self.tabs.add_named_tab('&Main', self.main_group.gbox)

        ### Channel list as channel widget ###
        self.main_group.glayout.addWidget(QLabel('Bands to load'), 0, 0, 1, 2)
        self.qlist_channels = ChannelWidget(self.viewer, translate=True)
        self.qlist_channels.itemClicked.connect(self._on_change_select_bands)
        self.qlist_channels.setToolTip(
            "Select one or more (hold shift) bands to load. Loaded bands are displayed in the imcube layer.")
        self.main_group.glayout.addWidget(self.qlist_channels, 1,0,1,2)

        ### Button "Select all" ###
        self.btn_select_all = QPushButton('Select all')
        self.btn_select_all.setEnabled(False)
        self.main_group.glayout.addWidget(self.btn_select_all, 2, 0, 1, 2)

        ### Checkbox "Sync bands with RGB" ###
        self.check_sync_bands_rgb = QCheckBox("Sync bands with RGB")
        self.check_sync_bands_rgb.setToolTip("Display same bands in RGB as in imcube")
        self.check_sync_bands_rgb.setChecked(True)
        self.main_group.glayout.addWidget(self.check_sync_bands_rgb, 3, 0, 1, 2)
        self.qlist_channels.setEnabled(False)

        # RGB widget
        self.rgb_widget = RGBWidget(viewer=self.viewer)
        self.tabs.add_named_tab('&Main', self.rgb_widget.rgbmain_group.gbox)
        self.rgb_widget.btn_RGB.clicked.connect(self._on_click_sync_RGB)

    def _create_processing_tab(self):
        """
        Generates the "Processing" tab and its elements.
        """
        
        self.tabs.widget(self.tab_names.index('Pro&cessing')).layout().setAlignment(Qt.AlignTop)

        # Group "Background correction"
        self.background_group = VHGroup('Background correction', orientation='G')
        self.tabs.add_named_tab('Pro&cessing', self.background_group.gbox)

        # ### Elements "Background Keyword"
        self.textbox_background_keyword = QLineEdit('_WR_')
        self.textbox_background_keyword.setToolTip("Keyword to identify background files")
        self.background_group.glayout.addWidget(QLabel('Background keyword'), 0, 0, 1, 1)
        self.background_group.glayout.addWidget(self.textbox_background_keyword, 0, 1, 1, 1)

        ### Elements "Dark ref" ###
        self.btn_select_dark_file = QPushButton("Manual selection")
        self.qtext_select_dark_file = QLineEdit()
        self.qtext_select_dark_file.setText('No path')
        self.background_group.glayout.addWidget(QLabel('Dark ref'), 1, 0, 1, 1)
        self.background_group.glayout.addWidget(self.qtext_select_dark_file, 1, 1, 1, 1)
        self.background_group.glayout.addWidget(self.btn_select_dark_file, 1, 2, 1, 1)

        ### Elements "White ref" ###
        self.btn_select_white_file = QPushButton("Manual selection")
        self.qtext_select_white_file = QLineEdit()
        self.qtext_select_white_file.setText('No path')
        self.background_group.glayout.addWidget(QLabel('White ref'), 2, 0, 1, 1)
        self.background_group.glayout.addWidget(self.qtext_select_white_file, 2, 1, 1, 1)
        self.background_group.glayout.addWidget(self.btn_select_white_file, 2, 2, 1, 1)

        ### Elements "Dark ref for image" ###
        self.btn_select_dark_for_im_file = QPushButton("Manual selection")
        self.qtext_select_dark_for_white_file = QLineEdit()
        self.qtext_select_dark_for_white_file.setText('No path')
        self.background_group.glayout.addWidget(QLabel('Dark ref for image'), 3, 0, 1, 1)
        self.background_group.glayout.addWidget(self.qtext_select_dark_for_white_file, 3, 1, 1, 1)
        self.background_group.glayout.addWidget(self.btn_select_dark_for_im_file, 3, 2, 1, 1)

        ### Combo box "Layer" ###
        self.combo_layer_background = QComboBox()
        self.background_group.glayout.addWidget(QLabel('Layer'), 4, 0, 1, 1)
        self.background_group.glayout.addWidget(self.combo_layer_background, 4, 1, 1, 2)

        ### Button "Correct" ###
        self.btn_background_correct = QPushButton("Correct")
        self.background_group.glayout.addWidget(self.btn_background_correct, 5, 0, 1, 3)

        # Group "Destripe"
        self.destripe_group = VHGroup('Destripe', orientation='G')
        self.tabs.add_named_tab('Pro&cessing', self.destripe_group.gbox)

        ### Combo box "Layer" ###
        self.combo_layer_destripe = QComboBox()
        self.destripe_group.glayout.addWidget(QLabel('Layer'), 0, 0, 1, 1)
        self.destripe_group.glayout.addWidget(self.combo_layer_destripe, 0, 1, 1, 1)
        self.qspin_destripe_width = QSpinBox()
        self.qspin_destripe_width.setRange(1, 1000)
        self.qspin_destripe_width.setValue(100)

        ### Spin box "Savgol Width" ###
        self.destripe_group.glayout.addWidget(QLabel('Savgol Width'), 1, 0, 1, 1)
        self.destripe_group.glayout.addWidget(self.qspin_destripe_width, 1, 1, 1, 1)

        ### Button "Destripe ###
        self.btn_destripe = QPushButton("Destripe")
        self.btn_destripe.setToolTip("Apply Savitzky-Golay filter to remove stripes")
        self.destripe_group.glayout.addWidget(self.btn_destripe, 2, 0, 1, 2)

        # Group "Correct full dataset"
        self.batch_group = VHGroup('Correct full dataset', orientation='G')
        self.tabs.add_named_tab('Pro&cessing', self.batch_group.gbox)
        
        ### Elements "Crop bands" ###
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

        ### Checkboxes "White correct" and "Destripe" ###
        self.check_batch_white = QCheckBox("White correct")
        self.check_batch_destripe = QCheckBox("Destripe")
        self.check_batch_white.setChecked(True)
        self.check_batch_destripe.setChecked(False)
        self.batch_group.glayout.addWidget(self.check_batch_white, 1, 0, 1, 1)
        self.batch_group.glayout.addWidget(self.check_batch_destripe, 2, 0, 1, 1)

        ### Spinbox "Chunk size" ###
        self.spin_chunk_size = QSpinBox()
        self.spin_chunk_size.setRange(1, 10000)
        self.spin_chunk_size.setValue(500)
        self.batch_group.glayout.addWidget(QLabel("Chunk size"), 3, 0, 1, 1)
        self.batch_group.glayout.addWidget(self.spin_chunk_size, 3, 1, 1, 1)

        ### Checkbox "Convert to integer" ###
        self.check_save_as_float = QCheckBox("Save as floats")
        self.check_save_as_float.setChecked(True)
        self.check_save_as_float.setToolTip("Save data as floats. Otherwise convert to integers after multiplication by 4096.")
        self.batch_group.glayout.addWidget(self.check_save_as_float, 4, 0, 1, 4)
        
        ### Button "Correct and save data" ###
        self.btn_batch_correct = QPushButton("Correct and save data")
        self.batch_group.glayout.addWidget(self.btn_batch_correct, 5, 0, 1, 4)

        # Group "Correct multiple data sets"
        self.multiexp_group = VHGroup('Correct multiple datasets', orientation='G')
        self.tabs.add_named_tab('Pro&cessing', self.multiexp_group.gbox)

        ### Button "Process in batch" ###
        self.btn_show_multiexp_batch = QPushButton("Process in batch")
        self.multiexp_group.glayout.addWidget(self.btn_show_multiexp_batch, 0, 0, 1, 1)
        self.multiexp_batch = None

        # Checkbox "Use dask"
        self.check_use_dask = QCheckBox("Use dask")
        self.check_use_dask.setChecked(False)
        self.check_use_dask.setToolTip("Use dask to parallelize computation")
        self.tabs.add_named_tab('Pro&cessing', self.check_use_dask)
  
    def _create_roi_tab(self):
        """
        Generates the "ROI" tab and its elements.
        """

        self.tabs.widget(self.tab_names.index('&ROI')).layout().setAlignment(Qt.AlignTop)

        # Group "Add Main ROI manually"
        self.roi_group = VHGroup('Add Main ROI manually', orientation='G')
        self.tabs.add_named_tab('&ROI', self.roi_group.gbox)

        ### Button "Add main ROI" ###
        self.btn_add_main_roi = QPushButton("Add main ROI")
        self.btn_add_main_roi.setToolTip("Maximal &ROI only removing fully masked border")
        self.roi_group.glayout.addWidget(self.btn_add_main_roi, 0, 0, 1, 2)

        ### Spinbox "Main ROI width" ###
        self.spin_main_roi_width = QSpinBox()
        self.spin_main_roi_width.setRange(1, 1000)
        self.spin_main_roi_width.setValue(20)
        self.roi_group.glayout.addWidget(QLabel('Main ROI width'), 1, 0, 1, 1)
        self.roi_group.glayout.addWidget(self.spin_main_roi_width, 1, 1, 1, 1)
        
        # Group "Crop and select"
        self.roicrop_group = VHGroup('Crop and select', orientation='G')
        self.tabs.add_named_tab('&ROI', self.roicrop_group.gbox)

        ### Button "Crop with main" ###
        self.btn_main_crop = QPushButton("Crop with main")
        self.btn_main_crop.setToolTip("Crop image with main ROI")

        ### Button "Reset crop" ###
        self.btn_main_crop_reset = QPushButton("Reset crop")
        self.btn_main_crop_reset.setToolTip("Reset crop to full image")
        self.roicrop_group.glayout.addWidget(self.btn_main_crop, 0, 0, 1, 1)
        self.roicrop_group.glayout.addWidget(self.btn_main_crop_reset, 0, 1, 1, 1)

        ### Spinbox "Selected ROI" ###
        self.spin_selected_roi = QSpinBox()
        self.spin_selected_roi.setRange(0, 0)
        self.spin_selected_roi.setValue(0)
        self.roicrop_group.glayout.addWidget(QLabel('Selected main ROI'), 1, 0, 1, 1)
        self.roicrop_group.glayout.addWidget(self.spin_selected_roi, 1, 1, 1, 1)

        # Group "Import Main ROI"
        self.roiimport_group = VHGroup('Import Main ROI', orientation='G')
        self.tabs.add_named_tab('&ROI', self.roiimport_group.gbox)

        ### File select "Import ROI" ###
        self.file_import_roi = FileEdit()
        self.file_import_roi.native.setToolTip("Choose Parameters.yml file")
        self.roiimport_group.glayout.addWidget(QLabel('Choose Parameter.yml file'), 0, 0, 1, 1)
        self.roiimport_group.glayout.addWidget(self.file_import_roi.native, 0, 1, 1, 1)
        self.btn_import_roi = QPushButton("Import main ROI from file")
        self.roiimport_group.glayout.addWidget(self.btn_import_roi, 1, 0, 1, 2)

        # Group "Sub-ROI"
        self.subroi_group = VHGroup('Sub-ROI', orientation='G')
        self.tabs.add_named_tab('&ROI', self.subroi_group.gbox)
        self.subroi_group.glayout.addWidget(QLabel(
            'Set desired sub-&ROI width and double-click in viewer to place them'), 0, 0, 1, 2)
        
        ### Spinbox "Sub-ROI width" ###
        self.spin_roi_width = QSpinBox()
        self.spin_roi_width.setRange(1, 1000)
        self.spin_roi_width.setValue(20)
        self.subroi_group.glayout.addWidget(QLabel('Sub-ROI width'), 1, 0, 1, 1)
        self.subroi_group.glayout.addWidget(self.spin_roi_width, 1, 1, 1, 1)

    def _create_mask_tab(self):
        """
        Generates the "Mask" tab and its elements.
        """
            
        self.tabs.widget(self.tab_names.index('Mas&k')).layout().setAlignment(Qt.AlignTop)
        
        # Group "Select layer to use"
        self.mask_layersel_group = VHGroup('1. Select layer to use', orientation='G')
        self.tabs.add_named_tab('Mas&k', self.mask_layersel_group.gbox)
        self.combo_layer_mask = QComboBox()
        self.mask_layersel_group.glayout.addWidget(self.combo_layer_mask)

        # Group "Create one or more masks"
        self.mask_generation_group = VHGroup('2. Create one or more masks', orientation='G')
        self.tabs.add_named_tab('Mas&k', self.mask_generation_group.gbox)

        ### "Create one or more masks" element Subtabs ###
        self.mask_tabs = TabSet(['Drawing', 'Border', 'Man. thresh.', 'Auto thresh.', 'ML'])
        self.mask_generation_group.glayout.addWidget(self.mask_tabs)
        
        ### Subtab "Drawing" ###
        ##### Group "Manual drawing" #####
        self.mask_group_draw = VHGroup('Manual drawing', orientation='G')
        self.mask_group_draw.gbox.setToolTip("Draw a mask manually")
        self.mask_tabs.add_named_tab('Drawing', self.mask_group_draw.gbox)
        ##### Button "Add manual mask" #####
        self.btn_add_draw_mask = QPushButton("Add manual mask")
        self.mask_group_draw.glayout.addWidget(self.btn_add_draw_mask, 0, 0, 1, 2)

        ### Subtab "Border" ###
        ##### Group "Border mask" #####
        self.mask_group_border = VHGroup('Border mask', orientation='G')
        self.mask_group_border.gbox.setToolTip("Detect background regions on the borders and remove them")
        self.mask_tabs.add_named_tab('Border', self.mask_group_border.gbox)
        ##### Button "Generate mask" #####
        self.btn_border_mask = QPushButton("Generate mask")
        self.mask_group_border.glayout.addWidget(self.btn_border_mask, 0, 0, 1, 2)

        ### Subtab "Man. thresh." ###
        ##### Group "Manual Threshold" #####
        self.mask_group_manual = VHGroup('Manual Threshold', orientation='G')
        self.mask_group_manual.gbox.setToolTip("Manually set a threshold on intensity of average imcube")
        self.mask_tabs.add_named_tab('Man. thresh.', self.mask_group_manual.gbox)
        ##### "Min/Max" Slider and label #####
        self.slider_mask_threshold = QLabeledDoubleRangeSlider(Qt.Horizontal)
        self.slider_mask_threshold.setRange(0, 1)
        self.slider_mask_threshold.setSingleStep(0.01)
        self.slider_mask_threshold.setSliderPosition([0, 1])
        self.mask_group_manual.glayout.addWidget(QLabel("Min/Max Threshold"), 0, 0, 1, 1)
        self.mask_group_manual.glayout.addWidget(self.slider_mask_threshold, 0, 1, 1, 1)
        ##### Button "Generate mask" #####
        self.btn_update_mask = QPushButton("Generate mask")
        self.mask_group_manual.glayout.addWidget(self.btn_update_mask, 1, 0, 1, 2)

        ### Subtab "Auto thresh" ##
        ##### Group "Auto Threshold" #####
        self.mask_group_auto = VHGroup('Auto Threshold', orientation='G')
        self.mask_group_auto.gbox.setToolTip("Assume a Gaussian pixel intensity distribution (mu, sigma) and set a threshold at mu +/- sigma*factor")
        self.mask_tabs.add_named_tab('Auto thresh.', self.mask_group_auto.gbox)
        ##### Button "Generate mask" #####
        self.btn_automated_mask = QPushButton("Generate mask")
        self.mask_group_auto.glayout.addWidget(self.btn_automated_mask, 0, 0, 1, 2)
        ##### Elements "Distribution width factor" #####
        self.spin_automated_mask_width = QDoubleSpinBox()
        self.spin_automated_mask_width.setToolTip("Assuming a Gaussian pixel intensity distribution (mu, sigma), set a threshold at mu +/- sigma*factor.")
        self.spin_automated_mask_width.setRange(0.1, 10)
        self.spin_automated_mask_width.setSingleStep(0.1)
        #self.mask_group_auto.glayout.addWidget(QLabel('Pixel distribution width factor'), 1, 0, 1, 1)
        auto_threshold_label = QLabel('<a href=\"https://guiwitz.github.io/napari-sediment/Details_sediment.html#auto-threshold\">Distribution width factor</a>')
        auto_threshold_label.setTextFormat(Qt.RichText)
        auto_threshold_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        auto_threshold_label.setOpenExternalLinks(True)
        self.mask_group_auto.glayout.addWidget(auto_threshold_label, 1, 0, 1, 1)
        self.mask_group_auto.glayout.addWidget(self.spin_automated_mask_width, 1, 1, 1, 1)

        ### Subtab "ML" ###
        ##### "Pixel Classifier" element #####
        self.mask_group_ml = VHGroup('Pixel Classifier', orientation='G')
        self.mask_group_ml.gbox.setToolTip("Use a pixel classifier to generate a mask")
        #self.mask_tabs.add_named_tab('ML', self.mask_group_ml.gbox)
        ##### ConvPaintSpectralWidget #####
        #from ..widget_utilities.classifier import ConvPaintSpectralWidget
        self.mlwidget = ConvPaintSpectralWidget(self.viewer)
        self.mask_group_ml.glayout.addWidget(self.mlwidget)
        ##### Subtab "ML" scroller ##### 
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.mask_group_ml.gbox)
        self.mask_tabs.add_named_tab('ML', scroll)

        # Align subtabs
        for g in [self.mask_group_border, self.mask_group_manual, self.mask_group_auto, self.mask_group_ml]:
            g.glayout.setAlignment(Qt.AlignTop)
        
        # phasor
        #self.btn_compute_phasor = QPushButton("Compute Phasor")
        #self.mask_group_phasor.glayout.addWidget(self.btn_compute_phasor, 0, 0, 1, 2)
        #self.btn_select_by_phasor = QPushButton("Phasor mask")
        #self.mask_group_phasor.glayout.addWidget(self.btn_select_by_phasor, 1, 0, 1, 2)

        # Group "Assemble masks"
        self.mask_assemble_group = VHGroup('3. Assemble masks', orientation='G')
        self.tabs.add_named_tab('Mas&k', self.mask_assemble_group.gbox)
        
        ### Button "Combine masks" ###
        self.btn_combine_masks = QPushButton("Combine masks")
        self.mask_assemble_group.glayout.addWidget(self.btn_combine_masks, 0, 0, 1, 2)

        ### Button "Clean mask" ###
        self.btn_clean_mask = QPushButton("Clean mask")
        self.mask_assemble_group.glayout.addWidget(self.btn_clean_mask, 1, 0, 1, 2)
 
    def _create_export_tab(self):
        """
        Generates the "IO" tab and its elements.
        """

        self.tabs.widget(self.tab_names.index('I&O')).layout().setAlignment(Qt.AlignTop)

        # Group "Mask"
        self.mask_group_export = VHGroup('Mask', orientation='G')
        self.tabs.add_named_tab('I&O', self.mask_group_export.gbox)

        ### Button "Save mask" ###
        self.btn_save_mask = QPushButton("Save mask")
        self.btn_save_mask.setToolTip("Save only mask as tiff")
        self.mask_group_export.glayout.addWidget(self.btn_save_mask)

        ### Button "Load mask" ###
        self.btn_load_mask = QPushButton("Load mask")
        self.mask_group_export.glayout.addWidget(self.btn_load_mask)
        
        # Group "Other exports"
        self.mask_group_capture = VHGroup('Other exports', orientation='G')
        self.tabs.add_named_tab('I&O', self.mask_group_capture.gbox)

        ### Button "Snapshot" ###
        self.btn_snapshot = QPushButton("Snapshot")
        self.btn_snapshot.setToolTip("Save snapshot of current viewer")
        self.mask_group_capture.glayout.addWidget(self.btn_snapshot, 0, 0, 1, 2)

        ### Elements "rgb.tiff" ###
        self.lineedit_rgb_tiff = QLineEdit()
        self.lineedit_rgb_tiff.setText('rgb.tiff')
        self.mask_group_capture.glayout.addWidget(self.lineedit_rgb_tiff, 1, 0, 1, 1)
        self.btn_save_rgb_tiff = QPushButton("Save RGB tiff")
        self.btn_save_rgb_tiff.setToolTip("Save current RGB layer as high-res tiff")
        self.mask_group_capture.glayout.addWidget(self.btn_save_rgb_tiff, 1, 1, 1, 1)

    def _create_options_tab(self):
        """
        "Options" tab not implemented so far
        """

        self.crop_group = VHGroup('Crop selection', orientation='G')

        self.check_use_external_ref = QCheckBox("Use external reference")
        self.check_use_external_ref.setChecked(True)

    def _create_plot_tab(self):
        """
        Generates the "Plotting" tab and its elements.
        """

        # SpectralPlotter
        self.scan_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.scan_plot.axes.set_xlabel('Wavelength (nm)', color='black')
        self.scan_plot.axes.set_ylabel('Intensity', color='black')
        self.tabs.add_named_tab('P&lotting', self.scan_plot, (0,0,1,2))

        # Checkbox "Remove continuum"
        self.check_remove_continuum = QCheckBox("Remove continuum")
        self.check_remove_continuum.setChecked(True)
        self.tabs.add_named_tab('P&lotting', self.check_remove_continuum, (1,0,1,2))

        # Slider "Smoothing window size"
        self.slider_spectrum_savgol = QDoubleSlider(Qt.Horizontal)
        self.slider_spectrum_savgol.setRange(1, 100)
        self.slider_spectrum_savgol.setSingleStep(1)
        self.slider_spectrum_savgol.setSliderPosition(5)
        self.tabs.add_named_tab('P&lotting', QLabel('Smoothing window size'), (2,0,1,1))
        self.tabs.add_named_tab('P&lotting', self.slider_spectrum_savgol, (2,1,1,1))

    def _create_metadata_tab(self):
        """
        Generates the "Metadata" tab and its elements.
        """

        # Group "Metadata"
        self.metadata_group = VHGroup('Metadata', orientation='G')
        self.tabs.add_named_tab('Meta&data', self.metadata_group.gbox)
        self.tabs.widget(self.tab_names.index('Meta&data')).layout().setAlignment(Qt.AlignTop)

        ### Elements "Location" ###
        self.metadata_location = QLineEdit("No location")
        self.metadata_location.setToolTip("Indicate the location of data acquisition")
        self.metadata_group.glayout.addWidget(QLabel('Location'), 0, 0, 1, 1)
        self.metadata_group.glayout.addWidget(self.metadata_location, 0, 1, 1, 1)

        ### Spinbox "Scale" ###
        self.spinbox_metadata_scale = QDoubleSpinBox()
        self.spinbox_metadata_scale.setToolTip("Indicate conversion factor from pixel to mm")
        self.spinbox_metadata_scale.setDecimals(4)
        self.spinbox_metadata_scale.setRange(0.0, 1000)
        self.spinbox_metadata_scale.setSingleStep(0.0001)
        self.spinbox_metadata_scale.setValue(1)
        self.metadata_group.glayout.addWidget(QLabel('Pixel Size'), 1, 0, 1, 1)
        self.metadata_group.glayout.addWidget(self.spinbox_metadata_scale, 1, 1, 1, 1)

        ### Spinbox "Unit" ###
        self.metadata_scale_unit = QLineEdit("mm")
        self.metadata_scale_unit.setToolTip("Indicate the unit of the scale")
        self.metadata_group.glayout.addWidget(QLabel('Unit'), 2, 0, 1, 1)
        self.metadata_group.glayout.addWidget(self.metadata_scale_unit, 2, 1, 1, 1)

        # Group "Interactive scale"
        self.interactive_scale_group = VHGroup('Interactive scale', orientation='G')
        self.tabs.add_named_tab('Meta&data', self.interactive_scale_group.gbox)
        
        ### Button "Add scale layer" ###
        self.btn_add_scale_layer = QPushButton("Add scale layer")
        self.interactive_scale_group.glayout.addWidget(self.btn_add_scale_layer, 0, 0, 1, 2)
        
        ### Spinbox "Scale size in units" ###
        self.spinbox_scale_size_units = QDoubleSpinBox()
        self.spinbox_scale_size_units.setRange(1, 100000)
        self.spinbox_scale_size_units.setValue(100)
        self.spinbox_scale_size_units.setSingleStep(1)
        self.interactive_scale_group.glayout.addWidget(QLabel('Scale size in units'), 1, 0, 1, 1)
        self.interactive_scale_group.glayout.addWidget(self.spinbox_scale_size_units, 1, 1, 1, 1)
        
        ### Button "Compute pixel size" ###
        self.btn_compute_pixel_size = QPushButton("Compute pixel size with scale")
        self.btn_compute_pixel_size.setEnabled(False)
        self.interactive_scale_group.glayout.addWidget(self.btn_compute_pixel_size, 2, 0, 1, 2)
        self.btn_compute_pixel_size_roi = QPushButton("Compute pixel size with main-roi")
        self.interactive_scale_group.glayout.addWidget(self.btn_compute_pixel_size_roi, 3, 0, 1, 2)

    def add_connections(self):
        """
        Connects GUI elements to functions to be executed when GUI elements are activated 
        """
        
        # Elements of the "Main" tab
        self.btn_select_export_folder.clicked.connect(self._on_click_select_export_folder)
        self.btn_select_imhdr_file.clicked.connect(self._on_click_select_imhdr)
        self.rgb_widget.btn_RGB.clicked.connect(self._update_threshold_limits)
        self.btn_select_all.clicked.connect(self._on_click_select_all)
        self.check_sync_bands_rgb.stateChanged.connect(self._on_click_sync_RGB)
        self.rgb_widget.btn_dislpay_as_rgb.clicked.connect(self._update_threshold_limits)

        # Elements of the "Processing" tab
        self.btn_select_white_file.clicked.connect(self._on_click_select_white_file)
        self.btn_select_dark_file.clicked.connect(self._on_click_select_dark_file)
        self.btn_select_dark_for_im_file.clicked.connect(self._on_click_select_dark_for_im_file)
        self.btn_destripe.clicked.connect(self._on_click_destripe)
        self.btn_background_correct.clicked.connect(self._on_click_background_correct)
        self.btn_batch_correct.clicked.connect(self._on_click_batch_correct)
        self.slider_batch_wavelengths.valueChanged.connect(self._on_change_batch_wavelengths)
        self.spin_batch_wavelengths_min.valueChanged.connect(self._on_change_spin_batch_wavelengths)
        self.spin_batch_wavelengths_max.valueChanged.connect(self._on_change_spin_batch_wavelengths)
        self.btn_show_multiexp_batch.clicked.connect(self._on_click_multiexp_batch)
        
        # Elements of the "ROI" tab
        self.btn_add_main_roi.clicked.connect(self._on_click_add_main_roi)
        self.btn_main_crop.clicked.connect(self._on_crop_with_main)
        self.btn_main_crop_reset.clicked.connect(self._on_reset_crop)
        self.btn_import_roi.clicked.connect(self._on_click_import_roi)
        
        # Elements of the "Mask" tab
        self.btn_add_draw_mask.clicked.connect(self._add_manual_mask)
        self.btn_border_mask.clicked.connect(self._on_click_remove_borders)
        self.btn_update_mask.clicked.connect(self._on_click_intensity_threshold)
        self.btn_automated_mask.clicked.connect(self._on_click_automated_threshold)
        self.btn_combine_masks.clicked.connect(self._on_click_combine_masks)
        self.btn_clean_mask.clicked.connect(self._on_click_clean_mask)
        self.combo_layer_mask.currentIndexChanged.connect(self._on_select_layer_for_mask)

        # Elements of the "IO" tab
        self.btn_save_mask.clicked.connect(self._on_click_save_mask)
        self.btn_load_mask.clicked.connect(self._on_click_load_mask)
        self.btn_snapshot.clicked.connect(self._on_click_snapshot)
        self.btn_export.clicked.connect(self.export_project)
        self.btn_import.clicked.connect(self.import_project)
        self.btn_save_rgb_tiff.clicked.connect(self._on_click_save_rgb_tiff)

        # Elements of the "Plotting" tab
        self.slider_spectrum_savgol.valueChanged.connect(self.update_spectral_plot)
        self.check_remove_continuum.stateChanged.connect(self.update_spectral_plot)

        # Elements of the "Metadata" tab
        self.btn_add_scale_layer.clicked.connect(self._on_click_add_scale_layer)
        self.btn_compute_pixel_size.clicked.connect(self._on_click_compute_pixel_size)
        self.btn_compute_pixel_size_roi.clicked.connect(self._on_click_compute_pixel_size_roi)
        
        # Viewer callbacks for mouse behaviour
        self.viewer.mouse_move_callbacks.append(self._shift_move_callback)
        self.viewer.mouse_double_click_callbacks.append(self._add_analysis_roi)

        # Viewer callbacks for layer behaviour
        self.viewer.layers.events.inserted.connect(self._update_combo_layers_destripe)
        self.viewer.layers.events.removed.connect(self._update_combo_layers_destripe)
        self.viewer.layers.events.inserted.connect(self._update_combo_layers_background)
        self.viewer.layers.events.removed.connect(self._update_combo_layers_background)
        self.viewer.layers.events.inserted.connect(self.translate_layer_on_add)


    # Functions for "Main" tab elements
    def _on_click_select_imhdr(self):
        """
        Interactively select hdr file
        Called: "Main" tab, button "Select hdr file"
        """
        imhdr_path = QFileDialog.getOpenFileName(self, "Select file")[0]
        if imhdr_path == '':
            return
        imhdr_path = Path(imhdr_path)
        if imhdr_path.parent.suffix == '.zarr':
            imhdr_path = imhdr_path.parent
        self.set_paths(imhdr_path)
        self._on_select_file()
        self._on_click_add_main_roi()

    def _on_click_select_export_folder(self):
        """
        Interactively select folder to analyze
        Called: "Main" tab, button "Set Project folder"
        """
        return_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if return_dir == '':
            return
        self.export_folder = Path(str(return_dir))
        self.export_path_display.setText(self.export_folder.as_posix())
      
    def import_project(self):
        """
        Import a project
        Called: "Main" tab, button "Import Project"
        """
        if self.export_folder is None:
            self._on_click_select_export_folder()
        #main_roi_folders = self.export_folder.glob('main_roi_*')

        self.params = load_project_params(folder=self.export_folder)#.joinpath(f'main_roi_{self.spin_selected_roi.value()}'))

        # files
        self.imhdr_path = Path(self.params.file_path)

        # check if background references have already been set
        if self.params.white_path is not None:
            self.white_file_path = Path(self.params.white_path)
            self.dark_for_im_file_path = Path(self.params.dark_for_im_path)
            self.dark_for_white_file_path = Path(self.params.dark_for_white_path)
        else:
            self.set_paths(self.imhdr_path)

        # set defaults
        self.rgb_widget.set_rgb(self.params.rgb)

        # load data
        self._on_select_file()

        # metadata
        self.metadata_location.setText(self.params.location)
        self.spinbox_metadata_scale.setValue(self.params.scale)
        self.metadata_scale_unit.setText(self.params.scale_units)

        # rois
        self._add_roi_layer()
        mainroi = [np.array(x).reshape(4,2) for x in self.params.main_roi]
        if mainroi:
            mainroi[0] = mainroi[0].astype(int)
            self.viewer.layers['main-roi'].add_rectangles(mainroi, edge_color='b')
        self.spin_selected_roi.setRange(0, len(mainroi)-1)

        rois = [[np.array(x).reshape(4,2) for x in y] for y in self.params.rois]
        self.roi_list = {ind: r for ind, r in enumerate(rois)}
        for ind, roi in enumerate(rois):
            self.spin_selected_roi.setValue(ind)
            self._add_roi_layer()
            self.viewer.layers[f'rois_{ind}'].add_rectangles(roi, edge_color='r')

        self.spin_selected_roi.setValue(0)

        # crop if needed
        self._on_crop_with_main()

        # load masks
        self._on_click_load_mask()

    def export_project(self):
        """
        Export data
        Called: "Main" tab, button "Export Project"
        """
        if self.export_folder is None:
            self._on_click_select_export_folder()

        # create roi folders
        for i in range(len(self.viewer.layers['main-roi'].data)):
            roi_folder = self.export_folder.joinpath(f'roi_{i}')
            if not roi_folder.is_dir():
                roi_folder.mkdir(exist_ok=True)

        self.save_params()

        self._on_click_save_mask()
    
    def _on_change_select_bands(self, event=None):
        """
        Select individual RGB channel
        Called: "Main" tab, channel in channel widget
        """
        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)
    
    def _on_click_select_all(self):
        """
        Select all RGB channels
        Called: "Main" tab, button "Select all"
        """
        self.qlist_channels.selectAll()
        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)

    def _on_click_sync_RGB(self, event=None):
        """
        Select same channels for imcube as loaded for RGB
        Called: "Main" tab, checkbox "Sync bands with RGB"
        """
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

    def _update_threshold_limits(self):
        """
        Load selected RGB channels
        Called: "Main" tab, button "Load RGB", see "rgb_widget.py"
        """
        im = self.get_summary_image_for_mask()
        if 'border-mask' in self.viewer.layers:
            im = im[self.viewer.layers['border-mask'].data == 0]
        self.slider_mask_threshold.setRange(im.min(), im.max())
        self.slider_mask_threshold.setSliderPosition([im.min(), im.max()])


    # Functions for "Processing" tab elements
    def _on_click_select_dark_file(self):
        """
        Interactively select dark reference
        Called: "Processing" tab, button "Manual selection" for "Dark ref"
        """
        return_path = QFileDialog.getOpenFileName(self, "Select Dark Ref for white")[0]
        if return_path == '':
            return
        self.dark_for_white_file_path = Path(return_path)
        self.qtext_select_dark_file.setText(self.dark_for_white_file_path.as_posix())

    def _on_click_select_white_file(self):
        """
        Interactively select white reference
        Called: "Processing" tab, button "Manual selection" for "White ref"
        """
        return_path = QFileDialog.getOpenFileName(self, "Select White Ref")[0]
        if self.white_file_path == '':
            return
        self.white_file_path = Path(return_path)
        self.qtext_select_white_file.setText(self.white_file_path.as_posix())

    def _on_click_select_dark_for_im_file(self):
        """
        Interactively select dark reference for image
        Called: "Processing" tab, button "Manual selection" for "Dark ref for image"
        """
        return_path = QFileDialog.getOpenFileName(self, "Select Dark Ref for image")[0]
        if return_path == '':
            return
        self.dark_for_im_file_path = Path(return_path)
        self.qtext_select_dark_for_white_file.setText(self.dark_for_im_file_path.as_posix())
    
    def _on_click_background_correct(self, event=None):
        """
        White correct image
        Called: "Processing" tab, button "Correct"
        """
        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("White correcting image")

            selected_layer = self.combo_layer_background.currentText()
            if selected_layer == 'imcube':
                channel_indices = self.qlist_channels.channel_indices
            elif selected_layer == 'RGB':
                channel_indices = np.sort(self.rgb_widget.rgb_ch)

            #col_bounds = (self.col_bounds if self.check_main_crop.isChecked() else None)
            col_bounds = self.col_bounds
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
                    self.viewer.layers['imcube_corrected'].visible = False
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

    def _on_click_destripe(self):
        """
        Destripe image
        Called: "Processing" tab, button "Destripe"
        """
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
                update_contrast_on_layer(self.viewer.layers[x])
                self.viewer.layers[x].refresh()
        
        if (selected_layer == 'None') or (selected_layer == 'imcube') | (selected_layer == 'imcube_corrected') | (self.check_sync_bands_rgb.isChecked()):
            if 'imcube_destripe' in self.viewer.layers:
                self.viewer.layers['imcube_destripe'].data = data_destripe
            else:
                self.viewer.add_image(data_destripe, name='imcube_destripe', rgb=False)
                self.viewer.layers['imcube_destripe'].visible = False
        self.viewer.window._status_bar._toggle_activity_dock(False)

    def _on_change_batch_wavelengths(self, event):
        """
        Called: "Processing" tab, slider "Crop bands"
        """
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
        """
        Called: "Processing" tab, slider "Crop bands"
        """

        self.slider_batch_wavelengths.valueChanged.disconnect(self._on_change_batch_wavelengths)
        self.slider_batch_wavelengths.setSliderPosition([self.spin_batch_wavelengths_min.value(), self.spin_batch_wavelengths_max.value()])
        self.slider_batch_wavelengths.valueChanged.connect(self._on_change_batch_wavelengths)

    def _on_click_batch_correct(self):
        """
        Applies and saves data correction
        Called: "Processing" tab, button "Correct and save data"
        """
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
                chunk_size=self.spin_chunk_size.value(),
                use_float=self.check_save_as_float.isChecked(),
                )
            self.save_params()
            
            # reload corrected image as zarr
            self.open_file()
            
        self.viewer.window._status_bar._toggle_activity_dock(False)

    def _on_click_multiexp_batch(self):
        """
        Instantiates a BatchPreprocWidget object from "batch_preproc.py"
        Called: "Processing" tab, button "Process in batch"
        """
        
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


    # Functions for "ROI" tab elements
    def _on_click_add_main_roi(self, event=None):
        """
        Add main ROI
        Called: "ROI" tab, button "Add main ROI"
        """
        self._add_roi_layer()

        col_min = self.col_bounds[0]
        col_max = self.col_bounds[1]
        col_width = self.spin_main_roi_width.value()

        col_middle = (col_max+col_min) // 2
        col_left = col_middle - col_width // 2
        col_right = col_middle + col_width - (col_width // 2)

        new_roi = [
            [self.row_bounds[0],col_left],
            [self.row_bounds[1],col_left],
            [self.row_bounds[1],col_right],
            [self.row_bounds[0],col_right]]
        #self.viewer.layers['main-roi'].data = []
        self.viewer.layers['main-roi'].add_rectangles(new_roi, edge_color='b')
        self.viewer.layers.selection.active = self.viewer.layers['main-roi']

    def _on_crop_with_main(self, event=None):
        """
        Generate sub-ROI
        Called: "ROI" tab, button "Crop with main"
        """
        #if self.check_main_crop.isChecked():
        if 'main-roi' in self.viewer.layers:
            main_roi = self.viewer.layers['main-roi'].data[self.spin_selected_roi.value()]
            self.row_bounds = np.array([main_roi[:,0].min(), main_roi[:,0].max()], dtype=np.uint16)
            self.col_bounds = np.array([main_roi[:,1].min(), main_roi[:,1].max()], dtype=np.uint16)
        else:
            self.row_bounds = [0, self.imagechannels.nrows]
            self.col_bounds = [0, self.imagechannels.ncols]

        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)
        self.rgb_widget.row_bounds = self.row_bounds
        self.rgb_widget.col_bounds = self.col_bounds
        self.rgb_widget._on_click_RGB()
        self.remove_masks()
        self._on_click_load_mask()

    def _on_reset_crop(self, event=None):
        """
        Called: "ROI" tab, button "Reset crop
        """    
        self.row_bounds = [0, self.imagechannels.nrows]
        self.col_bounds = [0, self.imagechannels.ncols]

        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)
        self.rgb_widget.row_bounds = self.row_bounds
        self.rgb_widget.col_bounds = self.col_bounds
        self.rgb_widget._on_click_RGB()

    def _on_click_import_roi(self, event=None):
        """
        Import ROI
        Called: "ROI" tab, button "Import ROI"
        """
        
        roi_param = load_project_params(self.file_import_roi.value.parent)
        roi = [np.array(x).reshape(4,2) for x in roi_param.main_roi]
        self.viewer.layers['main-roi'].data = roi


    # Functions for "Mask" tab elements
    def _on_select_layer_for_mask(self):
        """
        Select layer to use
        Called: "Mask" tab, combobox at label "Select layer to use"
        """
        selected_layer = self.combo_layer_mask.currentText()
        if selected_layer in self.viewer.layers:
            im = np.mean(self.viewer.layers[selected_layer].data, axis=0)
            if 'border-mask' in self.viewer.layers:
                im = im[self.viewer.layers['border-mask'].data == 0]
            self.slider_mask_threshold.setRange(im.min(), im.max())
            self.slider_mask_threshold.setSliderPosition([im.min(), im.max()])
    
    def _add_manual_mask(self):
        """
        Add manual mask
        Called: "Mask" tab, subtab "Drawing", button "Add manual mask"
        """
        self.mask_layer = self.viewer.add_labels(
            np.zeros((self.row_bounds[1]-self.row_bounds[0],
                      self.col_bounds[1]-self.col_bounds[0]), dtype=np.uint8),
            name='manual-mask')

    def _on_click_remove_borders(self):
        """
        Remove borders from image
        Called: "Mask" tab, "Border" subtab, button "Generate mask"
        """
        im = self.get_summary_image_for_mask()

        first_row, last_row = remove_top_bottom(im)
        first_col, last_col = remove_left_right(im)
        if 'border-mask' in self.viewer.layers:
            self.viewer.layers['border-mask'].data[:] = 0
        else:
            mask = np.asarray(np.zeros(im.shape, dtype=np.uint8))
            self.viewer.add_labels(mask, name='border-mask', opacity=0.5)
        self.viewer.layers['border-mask'].data[0:first_row,:] = 1
        self.viewer.layers['border-mask'].data[last_row::,:] = 1
        self.viewer.layers['border-mask'].data[:, 0:first_col] = 1
        self.viewer.layers['border-mask'].data[:, last_col::] = 1
        self.viewer.layers['border-mask'].refresh()
        # update threshold limits to exclude borders
        self._update_threshold_limits()

    def _on_click_intensity_threshold(self, event=None):
        """
        Create mask based on intensity threshold
        Called: "Mask" tab, "Man. thresh." subtab, button "Generate mask"
        """
        data = self.get_summary_image_for_mask()
        min_th = self.slider_mask_threshold.value()[0]
        max_th = self.slider_mask_threshold.value()[1]
        mask = ((data < self.slider_mask_threshold.value()[0]) | (data > self.slider_mask_threshold.value()[1])).astype(np.uint8)
        mask = np.asarray(mask)
        self.update_mask(mask, 'intensity-mask')

        self.slider_mask_threshold.setSliderPosition([min_th, max_th])

    def _on_click_automated_threshold(self):
        """
        Automatically set threshold for mask based on mean RGB pixel intensity
        Called: "Mask" tab, "Auto thresh." subtab, button "Generate mask"
        """
        im = np.asarray(self.get_summary_image_for_mask())
        if 'border-mask' in self.viewer.layers:
            pix_selected = im[self.viewer.layers['border-mask'].data == 0]
        else:
            pix_selected = np.ravel(im)
        med_val, std_val = fit_1dgaussian_without_outliers(data=pix_selected[::5])
        fact = self.spin_automated_mask_width.value()
        self.slider_mask_threshold.setSliderPosition(
            [
                np.max([med_val - fact*std_val, self.slider_mask_threshold.minimum()]),
                np.min([med_val + fact*std_val, self.slider_mask_threshold.maximum()])
             ]
        ),
        self._on_click_intensity_threshold()

    def _on_click_combine_masks(self):
        """
        Combine masks from border removel, phasor and thresholding
        Called: "Mask" tab, button "Combine masks"
        """
        mask_complete = np.zeros((self.row_bounds[1]-self.row_bounds[0],
                                  self.col_bounds[1]-self.col_bounds[0]), dtype=np.uint8)
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

    def _on_click_clean_mask(self):
        """
        Called: "Mask" tab, button "Clean mask"
        """
        if 'complete-mask' not in self.viewer.layers:
            self._on_click_combine_masks()
        mask = self.viewer.layers['complete-mask'].data == 0
        mask_lab = skimage.morphology.label(mask)
        mask_prop = skimage.measure.regionprops_table(mask_lab, properties=('label', 'area'))
        final_mask = mask_lab == mask_prop['label'][np.argmax(mask_prop['area'])]
        mask_filled = binary_fill_holes(final_mask)
        mask_filled = (mask_filled == 0).astype(np.uint8)
        self.viewer.add_labels(mask_filled, name='clean-mask')
    

    # Functions for "IO" tab elements
    def _on_click_save_mask(self):
        """
        Save mask to file
        Called: "IO" tab, button "Save mask" 
        """

        mask_list = ['manual-mask','intensity-mask','border-mask','ml-mask']
        mask_present = any([m in self.viewer.layers for m in mask_list])

        if self.export_folder is None: 
            self._on_click_select_export_folder()

        mask = None
        if 'clean-mask' in self.viewer.layers:
            mask = self.viewer.layers['clean-mask'].data
        elif 'complete-mask' in self.viewer.layers:
            mask = self.viewer.layers['complete-mask'].data
        elif mask_present:
            self._on_click_combine_masks()
            mask = self.viewer.layers['complete-mask'].data
        
        if mask is not None:
            save_mask(mask, Path(self.export_folder).joinpath(f'roi_{self.spin_selected_roi.value()}').joinpath('mask.tif'))

        
    def _on_click_load_mask(self):
        """
        Load mask from file
        Called: "IO" tab, button "Load mask"
        """
        if self.export_folder is None:
            return
        mask_path = Path(self.export_folder).joinpath(f'roi_{self.spin_selected_roi.value()}').joinpath('mask.tif')
        if mask_path.exists():
            mask = load_mask(mask_path)
            self.update_mask(mask, 'complete-mask')
        else:
            warnings.warn('No mask found')

    def _on_click_snapshot(self):
        """
        Save snapshot of viewer
        Called: "IO" tab, button "Snapshot"
        """
        if self.export_folder is None: 
            self._on_click_select_export_folder()

        self.viewer.screenshot(str(self.export_folder.joinpath('snapshot.png')))

    def _on_click_save_rgb_tiff(self):
        """
        Save RGB image to tiff file
        Called: "IO" tab, button "Save RGB tiff"
        """
        rgb = ['red', 'green', 'blue']
        image_list = [self.viewer.layers[c].data for c in rgb]
        contrast_list = [self.viewer.layers[c].contrast_limits for c in rgb]
        save_rgb_tiff_image(image_list, contrast_list, self.export_folder.joinpath(self.lineedit_rgb_tiff.text()))


    # Functions for "Plotting" tab elements
    def update_spectral_plot(self, event=None):
        """
        Called: "Plotting" tab, slider at label "Smoothing window size"
        """   
        if self.spectral_pixel is None:
            return

        self.scan_plot.axes.clear()
        self.scan_plot.axes.set_xlabel('Wavelength (nm)', color='black')
        self.scan_plot.axes.set_ylabel('Intensity', color='black')

        spectral_pixel = np.array(self.spectral_pixel, dtype=np.float64)
        
        if self.check_remove_continuum.isChecked(): 
            spectral_pixel = remove_continuum(spectral_pixel, self.qlist_channels.bands)

        filter_window = int(self.slider_spectrum_savgol.value())
        if filter_window > 3:
            if filter_window > len(spectral_pixel):
                warnings.warn(f'No smoothing applied. Filter window size, currently {filter_window},\n'
                              f'is larger than the number of bands, {len(spectral_pixel)}.\n'
                              f'Please select a smaller window size or add more bands.')
            else:
                spectral_pixel = savgol_filter(spectral_pixel, window_length=filter_window, polyorder=3)

        self.scan_plot.axes.plot(self.qlist_channels.bands, spectral_pixel)
        
        self.scan_plot.canvas.figure.canvas.draw()

    # Functions for "Metadata" tab elements
    def _on_click_add_scale_layer(self):
        """
        Add interactive scale layer
        Called: "Metadata" tab, button "Add scale layer"
        """
        if 'scale' not in self.viewer.layers:
            image_widht = self.col_bounds[1] - self.col_bounds[0]
            image_height = self.row_bounds[1] - self.row_bounds[0]
            line = np.array([[self.row_bounds[0] + image_height//3, self.col_bounds[0]],
                             [self.row_bounds[0] + 2*image_height//3, self.col_bounds[0]]])
            self.viewer.add_shapes(
                data=line,
                shape_type='line',
                edge_color='g',
                edge_width=int(image_widht/10),
                name='scale',
                ndim=2,
            )
        self.btn_compute_pixel_size.setEnabled(True)

    def _on_click_compute_pixel_size(self, event=None):
        """
        Compute pixel size with interactive scale
        Called: "Metadata" tab, button "Compute pixel size from scale"
        """
        if 'scale' not in self.viewer.layers:
            warnings.warn('No scale layer found. Please add scale layer first.')
            return

        scale = self.viewer.layers['scale'].data[0]
        scale_size_px = np.sqrt(np.sum((scale[0] - scale[1])**2))
        scale_size_units = self.spinbox_scale_size_units.value()
        pixel_size = scale_size_units / scale_size_px
        self.spinbox_metadata_scale.setValue(pixel_size)

    def _on_click_compute_pixel_size_roi(self, event=None):
        """
        Compute pixel size using the main roi
        Called: "Metadata" tab, button "Compute pixel size from ROI"
        """
        if 'main-roi' not in self.viewer.layers:
            warnings.warn('No main roi found. Please add main roi first.')
            return
        elif len(self.viewer.layers['main-roi'].data) == 0:
            warnings.warn('Main roi is empty. Please draw a roi first.')
            return

        current_main_roi = self.viewer.layers['main-roi'].data[self.spin_selected_roi.value()]
        current_main_roi_height = np.abs(current_main_roi[0,0] - current_main_roi[1,0])
        scale_size_px = current_main_roi_height
        scale_size_units = self.spinbox_scale_size_units.value()
        pixel_size = scale_size_units / scale_size_px
        self.spinbox_metadata_scale.setValue(pixel_size)


    # Helper Functions
    ### Helper functions used in tab element functions ###
    def _on_select_file(self):
        """
        Helper function used in: 
        "_on_click_select_imhdr" ("Main" tab), 
        "import_project" ("Main" tab)
        """
        success = self.open_file()
        if not success:
            return False

    def set_paths(self, imhdr_path):
        """
        Update image and white/dark image paths
        Helper function used in: 
        "_on_click_select_imhdr" ("Main" tab), 
        "import_project" ("Main" tab)
        """
        self.white_file_path = None
        self.dark_for_white_file_path = None
        self.dark_for_im_file_path = None
        
        self.imhdr_path = Path(imhdr_path)
        self.imhdr_path_display.setText(self.imhdr_path.as_posix())

        keyword = self.textbox_background_keyword.text()

        if self.check_use_external_ref.isChecked():
            try:
                refpath = None
                wr_files = list(self.imhdr_path.parent.parent.parent.glob(f'*{keyword}*'))
                for wr in wr_files:
                    wr_first_part = wr.name.split(keyword)[0]
                    if wr_first_part in self.imhdr_path.name:
                        refpath = wr
                if refpath is None:
                    raise Exception('No matching white reference folder found')
                        
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

    def save_params(self):
        """
        Save parameters
        Helper function used in:
        "_on_click_batch_correct" ("Main" tab), 
        "export_project" ("Processing" tab)
        """
        if self.export_folder is None:
            self._on_click_select_export_folder()

        full_roi = [[self.row_bounds[0], self.col_bounds[0],
                          self.row_bounds[1], self.col_bounds[0],
                          self.row_bounds[1], self.col_bounds[1],
                          self.row_bounds[0], self.col_bounds[1]]]

        if 'main-roi' not in self.viewer.layers:
            mainroi = full_roi
        else:
            mainroi = [list(x.flatten()) for x in self.viewer.layers['main-roi'].data]
            mainroi = [[x.item() for x in y] for y in mainroi]

        rois = []
        for i in range(len(mainroi)):
            if f'rois_{i}' in self.viewer.layers:
                if len(self.viewer.layers[f'rois_{i}'].data) == 0:
                    rois.append([np.array(mainroi[i])])
                else:
                    rois.append(self.viewer.layers[f'rois_{i}'].data)
            else:
                rois.append([np.array(mainroi[i])])
        rois = [[list(y.flatten()) for y in x] for x in rois]
        rois = [[[z.item() for z in x] for x in y] for y in rois]

        self.params.project_path = self.export_folder
        self.params.file_path = self.imhdr_path
        self.params.white_path = self.white_file_path
        self.params.dark_for_im_path = self.dark_for_im_file_path
        self.params.dark_for_white_path = self.dark_for_white_file_path
        self.params.location = self.metadata_location.text()
        self.params.scale = self.spinbox_metadata_scale.value()
        self.params.scale_units = self.metadata_scale_unit.text()
        self.params.rgb = self.rgb_widget.rgb

        self.params.main_roi = mainroi
        self.params.rois = rois
        self.params.save_parameters()

    def get_summary_image_for_mask(self):
        """
        Get summary image
        Helper function used in: 
        "_update_threshold_limits" ("Main" tab), 
        "_on_click_remove_borders" ("Mask" tab), 
        "_on_click_automated_threshold" ("Mask" tab), 
        "_on_click_intensity_threshold" ("Mask" tab)
        """
        selected_layer = self.combo_layer_mask.currentText()
        im = np.mean(self.viewer.layers[selected_layer].data, axis=0)
        return im

    def _add_roi_layer(self):
        """
        Add &ROI layers to napari viewer
        Helper function used in:
        "import_project" ("Main" tab),
        "_on_click_add_main_roi" ("ROI" tab), 
        "_add_analysis_roi" (Viewer callbacks for mouse behaviour)
        """
        edge_width = np.min([10, self.imagechannels.ncols//100])
        if edge_width < 1:
            edge_width = 1
        if 'main-roi' not in self.viewer.layers:
            roi_layer = self.viewer.add_shapes(
                ndim = 2,
                name='main-roi', edge_color='blue', face_color=np.array([0,0,0,0]), edge_width=edge_width)
            roi_layer.mouse_drag_callbacks.append(self._roi_to_int_on_mouse_release)

            roi_layer.events.data.connect(self._update_roi_spinbox)

        if f'rois_{self.spin_selected_roi.value()}' not in self.viewer.layers:
            roi_layer = self.viewer.add_shapes(
                ndim = 2,
                name=f'rois_{self.spin_selected_roi.value()}', edge_color='red', face_color=np.array([0,0,0,0]), edge_width=edge_width)

            roi_layer.mouse_drag_callbacks.append(self._roi_to_int_on_mouse_release)

    def remove_masks(self):
        """
        Remove all masks
        Helper function used in "_on_crop_with_main" ("ROI" tab)
        """

        mask_names = ['ml-mask', 'border-mask', 'intensity-mask',
                        'complete-mask', 'clean-mask', 'manual-mask',
                      'annotations', 'segmentation', 'mask']
        for m in mask_names:
            if m in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[m])

    def update_mask(self, mask, name='mask'):
        """
        Helper function used in: 
        "_on_click_intensity_threshold" ("Mask" tab), 
        "_on_click_load_mask" ("IO" tab)
        """
        if name in self.viewer.layers:
            self.viewer.layers[name].data = mask
        else:
            self.viewer.add_labels(mask, name=name)

    def open_file(self):
        """
        Open file in napari
        Helper function used in:
        "_on_click_batch_correct" ("Processing" tab),
        "_on_select_file" (Helper function)
        """
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
            zarr_converted = None
            if self.export_folder is not None:
                zarr_converted_test = self.export_folder.joinpath(self.imhdr_path.stem+'.zarr')
                if zarr_converted_test.exists():
                    zarr_converted = zarr_converted_test
            zarr_converted_local = self.imhdr_path.with_suffix('.zarr')
            if (self.check_load_corrected.isChecked()) and (self.export_folder is not None):
                if not self.export_folder.joinpath('corrected.zarr').exists():
                    warnings.warn('Corrected image not found. Loading raw image instead.')
                    if zarr_converted is not None:
                        self.imagechannels = ImChannels(zarr_converted)
                    elif zarr_converted_local.exists():
                        self.imagechannels = ImChannels(zarr_converted_local)
                    else:
                        self.imagechannels = ImChannels(self.imhdr_path)
                else:
                    self.imagechannels = ImChannels(self.export_folder.joinpath('corrected.zarr'))
            elif zarr_converted is not None:
                self.imagechannels = ImChannels(zarr_converted)
            elif zarr_converted_local.exists():
                self.imagechannels = ImChannels(zarr_converted_local)   
            else:
                self.imagechannels = ImChannels(self.imhdr_path)

            self.row_bounds = [0, self.imagechannels.nrows]
            self.col_bounds = [0, self.imagechannels.ncols]
            
            self.qlist_channels._update_channel_list(imagechannels=self.imagechannels)

            self.rgb_widget.imagechannels = self.imagechannels
            self.rgb_widget._on_click_RGB()
            self.rgb_widget.row_bounds = self.row_bounds
            self.rgb_widget.col_bounds = self.col_bounds
            # add imcube from RGB
            [self.qlist_channels.item(x).setSelected(True) for x in self.rgb_widget.rgb_ch]
            self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)
            self._update_threshold_limits()
            self._update_range_wavelength()
            self.viewer.layers['imcube'].visible = False
            self.mlwidget.select_layer_widget.native.setCurrentText('imcube')

            # adjust main roi size
            self.spin_main_roi_width.setRange(1, self.col_bounds[1]-self.col_bounds[0])
            self.spin_main_roi_width.setValue(self.col_bounds[1]-self.col_bounds[0])
            
        self.viewer.window._status_bar._toggle_activity_dock(False)
        return True

    
    ### Helper functions used in other helper functions ###
    def _roi_to_int_on_mouse_release(self, layer, event):
        """
        Round roi coordinates to integer on mouse release
        Helper function used in "_add_roi_layer" (Helper function)
        """
        
        yield
        while event.type == 'mouse_move':
            yield
        if event.type == 'mouse_release':
            layer.data = [np.around(x) for x in layer.data]
    
    def _update_roi_spinbox(self, event):
        """
        Helper function used in "_add_roi_layer" (Helper function)
        """
        self.spin_selected_roi.setRange(0, len(self.viewer.layers['main-roi'].data)-1)

    def _update_range_wavelength(self):
        """
        Update range of wavelength slider
        Helper function used in "open_file" (Helper function)
        """
        wavelengths = np.array(self.imagechannels.channel_names).astype(float)
        self.slider_batch_wavelengths.setRange(np.round(wavelengths[0]), np.round(wavelengths[-1]))
        self.slider_batch_wavelengths.setSliderPosition([np.round(wavelengths[0]), np.round(wavelengths[-1])])


    ### Helper functions used in functions with no obvious implementation ###
    def translate_layer(self, mask_name):
        """
        Translate mask
        Helper function used in "crop_masks" (Functions with no obvious implementation)
        """
        self.viewer.layers[mask_name].translate = (self.row_bounds[0], self.col_bounds[0])


    # Viewer callbacks for mouse behaviour
    def _add_analysis_roi(self, viewer=None, event=None, cursor_pos=None):
        """Add roi to layer"""

        if cursor_pos is None:
            cursor_pos = np.rint(self.viewer.cursor.position).astype(int)
            
        if self.row_bounds is None:
            min_row = 0
            max_row = self.imagechannels.nrows
        else:
            min_row = self.row_bounds[0]
            max_row = self.row_bounds[1]
        new_roi = [
            [min_row, cursor_pos[2]-self.spin_roi_width.value()//2],
            [max_row,cursor_pos[2]-self.spin_roi_width.value()//2],
            [max_row,cursor_pos[2]+self.spin_roi_width.value()//2],
            [min_row,cursor_pos[2]+self.spin_roi_width.value()//2]]
        
        layer_name = f'rois_{self.spin_selected_roi.value()}'
        if not layer_name in self.viewer.layers:
            self._add_roi_layer()
        self.viewer.layers[layer_name].add_rectangles(new_roi, edge_color='r')

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


    # Viewer callbacks for layer behaviour
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

    def translate_layer_on_add(self, mask_layer):
        """Translate mask"""

        mask_names = ['ml-mask', 'border-mask', 'intensity-mask',
                        'complete-mask', 'clean-mask', 'manual-mask',
                      'annotations', 'segmentation', 'mask']
        if mask_layer.value.name in mask_names:
            mask_layer.value.translate = (self.row_bounds[0], self.col_bounds[0])


    # Functions with no obvious implementation
    def _get_channel_name_from_index(self, index):
        """
        Returns channel name from index
        """
        if self.imagechannels is None:
            return None
        return self.imagechannels.channel_names[index]

    def _on_click_compute_phasor(self):
        """
        Compute phasor from image. Opens a new viewer with 2D histogram of 
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

    def crop_masks(self):
        """
        Crop masks
        """
        layers = ['manual-mask', 'clean-mask', 'intensity-mask',
                  'complete-mask', 'border-mask', 
                  'ml-mask', 'main-roi', 'rois']
        for l in layers:
            if l in self.viewer.layers:
                if isinstance(self.viewer.layers[l], napari.layers.labels.labels.Labels):
                    self.viewer.layers[l].data = self.viewer.layers[l].data[self.row_bounds[0]:self.row_bounds[1], self.col_bounds[0]:self.col_bounds[1]]
                self.translate_layer(l)