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
                            QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,)
from qtpy.QtCore import Qt
from superqt import QDoubleRangeSlider

import numpy as np
#import pystripe
from skimage.measure import points_in_poly
import skimage
from scipy.ndimage import binary_fill_holes

from napari_guitils.gui_structures import VHGroup, TabSet
from ._reader import read_spectral
from .sediproc import (white_dark_correct, load_white_dark,
                       phasor, remove_top_bottom, remove_left_right,
                       fit_1dgaussian_without_outliers, correct_save_to_zarr,
                       find_index_of_band)
from .imchannels import ImChannels
from .io import save_mask, load_mask, get_mask_path, load_project_params
from .parameters import Param
from .spectralplot import SpectralPlotter
from .channel_widget import ChannelWidget
from .widgets.mlwidget import MLWidget
from .images import save_rgb_tiff_image

import napari


class SedimentWidget(QWidget):
    
    def __init__(self, napari_viewer):
        super().__init__()
        
        self.viewer = napari_viewer
        self.params = Param()
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
        #self.pixclass = None
        self.export_folder = None
        self.mainroi_min_col = None
        self.mainroi_max_col = None
        self.mainroi_min_row = None
        self.mainroi_max_row = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['Main', 'Processing', 'Mask', 'ROI', 'Export', 'Plotting','Options']
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

        # metadata
        self.metadata_group = VHGroup('Metadata', orientation='G')
        self.tabs.add_named_tab('Main', self.metadata_group.gbox)

        self.metadata_location = QLineEdit("No location")
        self.metadata_group.glayout.addWidget(QLabel('Location'), 0, 0, 1, 1)
        self.metadata_group.glayout.addWidget(self.metadata_location, 0, 1, 1, 1)
        self.spinbox_metadata_scale = QDoubleSpinBox()
        self.spinbox_metadata_scale.setRange(1, 1000)
        self.spinbox_metadata_scale.setSingleStep(0.1)
        self.spinbox_metadata_scale.setValue(1)
        self.metadata_group.glayout.addWidget(QLabel('Scale'), 1, 0, 1, 1)
        self.metadata_group.glayout.addWidget(self.spinbox_metadata_scale, 1, 1, 1, 1)

        # channel selection
        self.main_group = VHGroup('Select', orientation='G')
        self.tabs.add_named_tab('Main', self.main_group.gbox)

        self.main_group.glayout.addWidget(QLabel('Channels to load'), 0, 0, 1, 2)
        self.qlist_channels = ChannelWidget(self)
        self.main_group.glayout.addWidget(self.qlist_channels, 1,0,1,2)

        # loading selection
        self.btn_RGB = QPushButton('Load RGB')
        self.main_group.glayout.addWidget(self.btn_RGB, 3, 0, 1, 2)

        self.btn_select_all = QPushButton('Select all')
        self.main_group.glayout.addWidget(self.btn_select_all, 4, 0, 1, 2)

        self.btn_dislpay_as_rgb = QPushButton('Display as RGB')
        self.main_group.glayout.addWidget(self.btn_dislpay_as_rgb, 5, 0, 1, 1)
        self.combo_layer_to_rgb = QComboBox()
        self.main_group.glayout.addWidget(self.combo_layer_to_rgb, 5, 1, 1, 1)

        #self.btn_new_view = QPushButton('New view')
        #self.main_group.glayout.addWidget(self.btn_new_view, 6, 0, 1, 2)
        #self.btn_new_view.clicked.connect(self.new_view)
        self.slider_contrast = QDoubleRangeSlider(Qt.Horizontal)
        self.slider_contrast.setRange(0, 1)
        self.slider_contrast.setSingleStep(0.01)
        self.slider_contrast.setSliderPosition([0, 1])
        self.main_group.glayout.addWidget(QLabel("RGB Contrast"), 6, 0, 1, 1)
        self.main_group.glayout.addWidget(self.slider_contrast, 6, 1, 1, 1)



    def _create_processing_tab(self):
        
        self.tabs.widget(self.tab_names.index('Processing')).layout().setAlignment(Qt.AlignTop)

        self.process_group = VHGroup('Process Hypercube', orientation='G')
        self.tabs.add_named_tab('Processing', self.process_group.gbox)

        self.btn_background_correct = QPushButton("Background correct")
        self.process_group.glayout.addWidget(self.btn_background_correct)

        self.destripe_group = VHGroup('Destripe', orientation='G')
        self.tabs.add_named_tab('Processing', self.destripe_group.gbox)
        self.combo_layer_destripe = QComboBox()
        #self.combo_layer_destripe.addItems(['RGB', 'imcube'])
        self.destripe_group.glayout.addWidget(self.combo_layer_destripe, 0, 0, 1, 1)
        self.btn_destripe = QPushButton("Destripe")
        self.destripe_group.glayout.addWidget(self.btn_destripe)


        self.batch_group = VHGroup('Batch', orientation='G')
        self.tabs.add_named_tab('Processing', self.batch_group.gbox)
        self.btn_batch_correct = QPushButton("Save corrected images")
        self.batch_group.glayout.addWidget(self.btn_batch_correct, 0, 0, 1, 3)
        self.slider_batch_wavelengths = QDoubleRangeSlider(Qt.Horizontal)
        self.slider_batch_wavelengths.setRange(0, 1000)
        self.slider_batch_wavelengths.setSingleStep(1)
        self.slider_batch_wavelengths.setSliderPosition([0, 1000])
        self.batch_group.glayout.addWidget(self.slider_batch_wavelengths, 1,1,1,1)
        
        self.spin_batch_wavelengths_min = QDoubleSpinBox()
        self.spin_batch_wavelengths_min.setRange(0, 1000)
        self.spin_batch_wavelengths_min.setSingleStep(1)
        self.batch_group.glayout.addWidget(self.spin_batch_wavelengths_min, 1, 0, 1, 1)
        self.spin_batch_wavelengths_max = QDoubleSpinBox()
        self.spin_batch_wavelengths_max.setRange(0, 1000)
        self.spin_batch_wavelengths_max.setSingleStep(1)
        self.batch_group.glayout.addWidget(self.spin_batch_wavelengths_max, 1, 2, 1, 1)

        self.check_batch_white = QCheckBox("White correct")
        self.check_batch_destripe = QCheckBox("Destripe")
        self.check_batch_white.setChecked(True)
        self.check_batch_destripe.setChecked(True)
        self.batch_group.glayout.addWidget(self.check_batch_white, 2, 0, 1, 1)
        self.batch_group.glayout.addWidget(self.check_batch_destripe, 2, 1, 1, 1)


    def _create_mask_tab(self):
            
        self.tabs.widget(self.tab_names.index('Mask')).layout().setAlignment(Qt.AlignTop)
        
        
        self.mask_group_border = VHGroup('Mask processing', orientation='G')
        self.mask_group_manual = VHGroup('Manual Threshold', orientation='G')
        self.mask_group_auto = VHGroup('Auto Threshold', orientation='G')
        self.mask_group_phasor = VHGroup('Phasor Threshold', orientation='G')
        self.mask_group_ml = VHGroup('Pixel Classifier', orientation='G')
        self.mask_group_combine = VHGroup('Combine', orientation='G')
        self.tabs.add_named_tab('Mask', self.mask_group_border.gbox)
        self.tabs.add_named_tab('Mask', self.mask_group_manual.gbox)
        self.tabs.add_named_tab('Mask', self.mask_group_auto.gbox)
        self.tabs.add_named_tab('Mask', self.mask_group_phasor.gbox)
        self.tabs.add_named_tab('Mask', self.mask_group_ml.gbox)
        self.tabs.add_named_tab('Mask', self.mask_group_combine.gbox)

        # border
        self.btn_border_mask = QPushButton("Border mask")
        self.mask_group_border.glayout.addWidget(self.btn_border_mask, 0, 0, 1, 2)

        # manual
        self.slider_mask_threshold = QDoubleRangeSlider(Qt.Horizontal)
        self.slider_mask_threshold.setRange(0, 1)
        self.slider_mask_threshold.setSingleStep(0.01)
        self.slider_mask_threshold.setSliderPosition([0, 1])
        self.mask_group_manual.glayout.addWidget(QLabel("Threshold value"), 0, 0, 1, 1)
        self.mask_group_manual.glayout.addWidget(self.slider_mask_threshold, 0, 1, 1, 1)
        self.btn_update_mask = QPushButton("Manual Threshold mask")
        self.mask_group_manual.glayout.addWidget(self.btn_update_mask, 1, 0, 1, 2)
        
        # auto
        self.btn_automated_mask = QPushButton("Automated mask")
        self.mask_group_auto.glayout.addWidget(self.btn_automated_mask, 0, 0, 1, 1)
        self.spin_automated_mask_width = QDoubleSpinBox()
        self.spin_automated_mask_width.setRange(0.1, 10)
        self.spin_automated_mask_width.setSingleStep(0.1)
        self.mask_group_auto.glayout.addWidget(QLabel('Distr. Width'), 1, 0, 1, 1)
        self.mask_group_auto.glayout.addWidget(self.spin_automated_mask_width, 1, 1, 1, 1)

        # phasor
        self.btn_compute_phasor = QPushButton("Compute Phasor")
        self.mask_group_phasor.glayout.addWidget(self.btn_compute_phasor, 0, 0, 1, 2)
        self.btn_select_by_phasor = QPushButton("Phasor mask")
        self.mask_group_phasor.glayout.addWidget(self.btn_select_by_phasor, 1, 0, 1, 2)

        # ml
        self.mlwidget = MLWidget(self, self.viewer)
        self.mask_group_ml.glayout.addWidget(self.mlwidget)
        
        # combine
        self.btn_combine_masks = QPushButton("Combine masks")
        self.mask_group_combine.glayout.addWidget(self.btn_combine_masks, 0, 0, 1, 2)
        self.btn_clean_mask = QPushButton("Clean mask")
        self.mask_group_combine.glayout.addWidget(self.btn_clean_mask, 1, 0, 1, 2)
        
    def _create_roi_tab(self):

        self.tabs.widget(self.tab_names.index('ROI')).layout().setAlignment(Qt.AlignTop)

        self.roi_group = VHGroup('ROI definition', orientation='G')
        self.tabs.add_named_tab('ROI', self.roi_group.gbox)
        self.btn_add_main_roi = QPushButton("Add main ROI")
        self.roi_group.glayout.addWidget(self.btn_add_main_roi, 0, 0, 1, 2)
        self.btn_add_sub_roi = QPushButton("Add analysis ROI")
        self.roi_group.glayout.addWidget(self.btn_add_sub_roi, 1, 0, 1, 2)
        self.spin_roi_width = QSpinBox()
        self.spin_roi_width.setRange(1, 1000)
        self.spin_roi_width.setValue(20)
        self.roi_group.glayout.addWidget(QLabel('ROI width'), 2, 0, 1, 1)
        self.roi_group.glayout.addWidget(self.spin_roi_width, 2, 1, 1, 1)

    def _create_export_tab(self):

        self.tabs.widget(self.tab_names.index('Export')).layout().setAlignment(Qt.AlignTop)

        # io
        self.mask_group_export = VHGroup('Mask export', orientation='G')
        self.tabs.add_named_tab('Export', self.mask_group_export.gbox)
        self.btn_save_mask = QPushButton("Save mask")
        self.mask_group_export.glayout.addWidget(self.btn_save_mask)
        self.btn_load_mask = QPushButton("Load mask")
        self.mask_group_export.glayout.addWidget(self.btn_load_mask)
        
        self.mask_group_capture = VHGroup('Captures', orientation='G')
        self.tabs.add_named_tab('Export', self.mask_group_capture.gbox)
        self.btn_snapshot = QPushButton("Snapshot")
        self.mask_group_capture.glayout.addWidget(self.btn_snapshot)
        self.lineedit_rgb_tiff = QLineEdit()
        self.lineedit_rgb_tiff.setText('rgb.tiff')
        self.mask_group_capture.glayout.addWidget(self.lineedit_rgb_tiff)
        self.btn_save_rgb_tiff = QPushButton("Save RGB tiff")
        self.mask_group_capture.glayout.addWidget(self.btn_save_rgb_tiff)

        self.mask_group_project = VHGroup('Project', orientation='G')
        self.tabs.add_named_tab('Export', self.mask_group_project.gbox)
        self.btn_export = QPushButton("Export")
        self.mask_group_project.glayout.addWidget(self.btn_export)
        self.btn_import = QPushButton("Import")
        self.mask_group_project.glayout.addWidget(self.btn_import)
        

    def _create_options_tab(self):
        
        self.background_group = VHGroup('Background selection', orientation='G')
        self.crop_group = VHGroup('Crop selection', orientation='G')

        self.tabs.add_named_tab('Options', self.background_group.gbox)
        self.tabs.add_named_tab('Options', self.crop_group.gbox)

        self.check_use_external_ref = QCheckBox("Use external reference")
        self.check_use_external_ref.setChecked(True)

        self.btn_select_white_file = QPushButton("Select white ref")
        self.qtext_select_white_file = QLineEdit()
        self.background_group.glayout.addWidget(self.btn_select_white_file, 0, 0, 1, 1)
        self.background_group.glayout.addWidget(self.qtext_select_white_file, 0, 1, 1, 1)

        self.btn_select_dark_file = QPushButton("Select dark ref")
        self.qtext_select_dark_file = QLineEdit()
        self.background_group.glayout.addWidget(self.btn_select_dark_file, 1, 0, 1, 1)
        self.background_group.glayout.addWidget(self.qtext_select_dark_file, 1, 1, 1, 1)

        self.btn_select_dark_for_white_file = QPushButton("Select dark ref for white ref")
        self.qtext_select_dark_for_white_file = QLineEdit()
        self.background_group.glayout.addWidget(self.btn_select_dark_for_white_file, 2, 0, 1, 1)
        self.background_group.glayout.addWidget(self.qtext_select_dark_for_white_file, 2, 1, 1, 1)

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
        self.tabs.add_named_tab('Plotting', self.scan_plot)


    def add_connections(self):
        """Add callbacks"""

        self.btn_select_export_folder.clicked.connect(self._on_click_select_export_folder)
        self.btn_select_imhdr_file.clicked.connect(self._on_click_select_imhdr)
        self.btn_select_white_file.clicked.connect(self._on_click_select_white_file)
        self.btn_select_dark_file.clicked.connect(self._on_click_select_dark_file)
        self.btn_select_dark_for_white_file.clicked.connect(self._on_click_select_dark_for_white_file)
        self.btn_destripe.clicked.connect(self._on_click_destripe)
        self.btn_background_correct.clicked.connect(self._on_click_background_correct)
        self.btn_RGB.clicked.connect(self._on_click_RGB)
        self.btn_select_all.clicked.connect(self._on_click_select_all)
        self.btn_dislpay_as_rgb.clicked.connect(self.display_as_rgb)
        self.check_use_crop.stateChanged.connect(self._on_click_use_crop)
        self.btn_refresh_crop.clicked.connect(self._on_click_use_crop)
        self.btn_batch_correct.clicked.connect(self._on_click_batch_correct)
        self.slider_batch_wavelengths.valueChanged.connect(self._on_change_batch_wavelengths)
        self.slider_contrast.valueChanged.connect(self._on_change_contrast)
        
        # mask
        self.btn_border_mask.clicked.connect(self._on_click_remove_borders)
        self.btn_update_mask.clicked.connect(self._on_click_update_mask)
        self.btn_automated_mask.clicked.connect(self._on_click_automated_threshold)
        self.btn_compute_phasor.clicked.connect(self._on_click_compute_phasor)
        self.btn_select_by_phasor.clicked.connect(self._on_click_select_by_phasor)
        self.btn_combine_masks.clicked.connect(self._on_click_combine_masks)
        self.btn_clean_mask.clicked.connect(self._on_click_clean_mask)

        # ROI
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

        # layer callbacks
        self.viewer.layers.events.inserted.connect(self._update_combo_layers)
        self.viewer.layers.events.removed.connect(self._update_combo_layers)

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
        
        self.qlist_channels._update_channel_list()

        self._on_click_RGB()

        #self._add_roi_layer()
        self._add_mask()
        self._update_range_wavelength()

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
        
        layers = ['mask', 'clean-mask', 
                  'complete-mask', 'border-mask', 
                  'ml-mask', 'main-roi', 'rois']
        for l in layers:
            if l in self.viewer.layers:
                if isinstance(self.viewer.layers[l], napari.layers.labels.labels.Labels):
                    self.viewer.layers[l].data = self.viewer.layers[l].data[self.row_bounds[0]:self.row_bounds[1], self.col_bounds[0]:self.col_bounds[1]]
                self.translate_layer(l)
        
        self.qlist_channels._on_change_channel_selection()


    def _add_roi_layer(self):
        """Add ROI layers to napari viewer"""

        if 'main-roi' not in self.viewer.layers:
            self.roi_layer = self.viewer.add_shapes(
                ndim = 2,
                name='main-roi', edge_color='blue', face_color=np.array([0,0,0,0]), edge_width=10)
        
        if 'rois' not in self.viewer.layers:
            self.roi_layer = self.viewer.add_shapes(
                ndim = 2,
                name='rois', edge_color='red', face_color=np.array([0,0,0,0]), edge_width=10)
         
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
        self.viewer.layers['main-roi'].add_rectangles(new_roi, edge_color='b', edge_width=10)

    def set_main_roi_bounds(self, min_col, max_col, min_row, max_row):
            
        self.mainroi_min_col = min_col
        self.mainroi_max_col = max_col
        self.mainroi_min_row = min_row
        self.mainroi_max_row = max_row

    def _add_analysis_roi(self, viewer, event):
        """Add roi to layer"""

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
        self.viewer.layers['rois'].add_rectangles(new_roi, edge_color='r', edge_width=10)


    def _add_mask(self):
        self.mask_layer = self.viewer.add_labels(
            np.zeros((self.imagechannels.nrows,self.imagechannels.ncols), dtype=np.uint8),
            name='mask')

    def _on_click_destripe(self):
        """Destripe image"""
        
        selected_layer = self.combo_layer_destripe.currentText()
        if (selected_layer == 'None') or (selected_layer == 'imcube'):
            data_destripe = self.viewer.layers['imcube'].data.copy()
        elif selected_layer == 'imcube_corrected':
            data_destripe = self.viewer.layers['imcube_corrected'].data.copy()
        elif selected_layer == 'RGB':
            data_destripe = np.stack([self.viewer.layers[x].data for x in self.rgb_names], axis=0)
        
        for d in range(data_destripe.shape[0]):
            data_destripe[d] = pystripe.filter_streaks(data_destripe[d].T, sigma=[128, 256], level=7, wavelet='db2').T

        if selected_layer == 'RGB':
            for ind, x in enumerate(self.rgb_names):
                self.viewer.layers[x].data = data_destripe[ind]
        else:
            if 'imcube_destripe' in self.viewer.layers:
                self.viewer.layers['imcube_destripe'].data = data_destripe
            else:
                self.viewer.add_image(data_destripe, name='imcube_destripe', rgb=False)


    def _on_click_background_correct(self, event):
        """White correct image"""
                
            
        col_bounds = (self.col_bounds if self.check_use_crop.isChecked() else None)
        white_data, dark_data, dark_for_white_data = load_white_dark(
            white_file_path=self.white_file_path,
            dark_for_im_file_path=self.dark_for_im_file_path,
            dark_for_white_file_path=self.dark_for_white_file_path,
            channel_indices=self.channel_indices,
            col_bounds=col_bounds,
            clean_white=True
            )

        im_corr = white_dark_correct(
            self.viewer.layers['imcube'].data, white_data, dark_data, dark_for_white_data)

        if 'imcube_corrected' in self.viewer.layers:
            self.viewer.layers['imcube_corrected'].data = im_corr
        else:
            self.viewer.add_image(im_corr, name='imcube_corrected', rgb=False)
            self.viewer.layers['imcube_corrected'].translate = (0, self.row_bounds[0], self.col_bounds[0])


    def _update_combo_layers(self):
        
        admit_layers = ['imcube', 'imcube_corrected', 'RBB']
        self.combo_layer_destripe.clear()
        for a in admit_layers:
            if a in self.viewer.layers:
                self.combo_layer_destripe.addItem(a)

        admit_layers = ['imcube', 'imcube_corrected', 'imcube_destripe']
        self.combo_layer_to_rgb.clear()
        for a in admit_layers:
            if a in self.viewer.layers:
                self.combo_layer_to_rgb.addItem(a)        
        

    def _on_change_batch_wavelengths(self, event):

        self.spin_batch_wavelengths_max.setMinimum(self.slider_batch_wavelengths.minimum())
        self.spin_batch_wavelengths_max.setMaximum(self.slider_batch_wavelengths.maximum())
        self.spin_batch_wavelengths_min.setMinimum(self.slider_batch_wavelengths.minimum())
        self.spin_batch_wavelengths_min.setMaximum(self.slider_batch_wavelengths.maximum())
        self.spin_batch_wavelengths_max.setValue(self.slider_batch_wavelengths.value()[1])
        self.spin_batch_wavelengths_min.setValue(self.slider_batch_wavelengths.value()[0])

    def _on_click_batch_correct(self):

        if self.export_folder is None:
            self._on_click_select_export_folder()

        min_band = np.argmin(np.abs(np.array(self.imagechannels.channel_names).astype(float) - self.slider_batch_wavelengths.value()[0]))
        max_band = np.argmin(np.abs(np.array(self.imagechannels.channel_names).astype(float) - self.slider_batch_wavelengths.value()[1]))
        bands_to_correct = np.arange(min_band, max_band+1)
        correct_save_to_zarr(
            imhdr_path=self.imhdr_path,
            white_file_path=self.white_file_path,
            dark_for_im_file_path=self.dark_for_im_file_path,
            dark_for_white_file_path=self.dark_for_white_file_path,
            zarr_path=self.export_folder.joinpath('corrected.zarr'),
            band_indices=bands_to_correct,
            background_correction=self.check_batch_white.isChecked(),
            destripe=self.check_batch_destripe.isChecked())


    def get_summary_image(self):
        """Get summary image"""

        #im = np.mean(np.stack([self.viewer.layers[x].data for x in self.rgb_names], axis=0), axis=0)
        #return im[self.row_bounds[0]:self.row_bounds[1], self.col_bounds[0]:self.col_bounds[1]]
        im = np.mean(self.viewer.layers['imcube'].data, axis=0)
        return im

    def translate_layer(self, mask_name):
        """Translate mask"""

        self.viewer.layers[mask_name].translate = (self.row_bounds[0], self.col_bounds[0])


    def _on_click_remove_borders(self):
        """Remove borders from image"""
        
        im = self.get_summary_image()

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
        
        im = self.get_summary_image()
        self.slider_mask_threshold.setRange(im.min(), im.max())
        self.slider_mask_threshold.setSliderPosition([im.min(), im.max()])

    def _on_click_automated_threshold(self):
        """Automatically set threshold for mask based on mean RGB pixel intensity"""

        im = self.get_summary_image()
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
        self._on_click_update_mask()
    
    def _on_click_update_mask(self):
        """Update mask based on current threshold"""
        
        data = self.get_summary_image()
        mask = ((data < self.slider_mask_threshold.value()[0]) | (data > self.slider_mask_threshold.value()[1])).astype(np.uint8)
        self.update_mask(mask)
        self.translate_layer('mask')
        self.viewer.layers['mask'].refresh()

    def update_mask(self, mask):

        if 'mask' in self.viewer.layers:
            self.viewer.layers['mask'].data = mask
        else:
            self.viewer.add_labels(mask, name='mask')

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
        if 'ml-mask' in self.viewer.layers:
            mask_complete = mask_complete + (self.viewer.layers['ml-mask'].data == 1)
        
        mask_complete = (mask_complete>0).astype(np.uint8)

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
        else:
            mask = self.viewer.layers['mask'].data

        save_mask(mask, get_mask_path(self.export_folder))

    def _on_click_load_mask(self):
        """Load mask from file"""
        
        mask = load_mask(get_mask_path(self.export_folder))
        self.update_mask(mask)

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
        

    def _on_click_RGB(self):
        """Load RGB image"""

        #self.rgb_ch = [np.argmin(np.abs(np.array(self.imagechannels.channel_names).astype(float) - x)) for x in self.rgb]
        self.rgb_ch = [find_index_of_band(self.imagechannels.centers, x) for x in self.rgb]
        self.rgb_names = [self.imagechannels.channel_names[x] for x in self.rgb_ch]

        [self.qlist_channels.item(x).setSelected(True) for x in self.rgb_ch]
        self.qlist_channels._on_change_channel_selection()

        self.display_as_rgb()

        self._update_threshold_limits()

    def display_as_rgb(self):

        cmaps = ['red', 'green', 'blue']
        layer_name = self.combo_layer_to_rgb.currentText()
        for ind, cmap in enumerate(cmaps):
            if cmap not in self.viewer.layers:
                self.viewer.add_image(
                    self.viewer.layers[layer_name].data[ind],
                    name=cmap,
                    colormap=cmap,
                    blending='additive')
            else:
                self.viewer.layers[cmap].data = self.viewer.layers[layer_name].data[ind]
            
            self.viewer.layers[cmap].contrast_limits_range = (self.viewer.layers[cmap].data.min(), self.viewer.layers[cmap].data.max())
            self.viewer.layers[cmap].contrast_limits = np.percentile(self.viewer.layers[cmap].data, (2,98))
            
        self._update_threshold_limits()

    def _on_change_contrast(self, event=None):
        """Update contrast limits of RGB channels"""
        
        rgb = ['red', 'green', 'blue']
        for c in rgb:
            contrast_limits = np.percentile(self.viewer.layers[c].data, (2,98))
            contrast_range = contrast_limits[1] - contrast_limits[0]
            newlimits = contrast_limits.copy()
            newlimits[0] = contrast_limits[0] + self.slider_contrast.value()[0] * contrast_range
            newlimits[1] = contrast_limits[0] + self.slider_contrast.value()[1] * contrast_range
            self.viewer.layers[c].contrast_limits = newlimits

    def _on_click_select_all(self):
        self.qlist_channels.selectAll()
        self.qlist_channels._on_change_channel_selection()


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

        self._on_select_file()
        self._on_click_load_mask()

        # metadata
        self.metadata_location.setText(self.params.location)
        self.spinbox_metadata_scale.setValue(self.params.scale)

        # rois
        mainroi = [np.array(x).reshape(4,2) for x in self.params.main_roi]
        mainroi[0] = mainroi[0].astype(int)
        rois = [np.array(x).reshape(4,2) for x in self.params.rois]
        self.viewer.layers['main-roi'].add_rectangles(mainroi, edge_color='b', edge_width=10)
        self.viewer.layers['rois'].add_rectangles(rois, edge_color='r', edge_width=10)

        self.set_main_roi_bounds(
            min_col=mainroi[0][:,1].min(),
            max_col=mainroi[0][:,1].max(),
            min_row=mainroi[0][:,0].min(),
            max_row=mainroi[0][:,0].max()
        )