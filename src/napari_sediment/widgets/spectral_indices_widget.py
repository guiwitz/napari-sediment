from pathlib import Path
from dataclasses import asdict
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import warnings
import yaml
from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget,
                            QLabel, QFileDialog, QSpinBox,
                            QComboBox, QLineEdit, QSizePolicy,
                            QGridLayout, QCheckBox, QDoubleSpinBox,
                            QColorDialog, QScrollArea, QSizePolicy)
from qtpy.QtCore import Qt, QRect
from qtpy.QtGui import QPixmap, QColor, QPainter
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider
from magicgui.widgets import FileEdit
from cmap import Colormap
from napari.utils import progress
import pandas as pd
#from napari_matplotlib.base import NapariMPLWidget
from napari_guitils.gui_structures import TabSet, VHGroup

from ..data_structures.parameters import Param
from ..data_structures.spectralindex import SpectralIndex
from ..data_structures.parameters_plots import Paramplot
from ..data_structures.imchannels import ImChannels
from ..widget_utilities.spectralplotter import SpectralPlotter
from ..widget_utilities.channel_widget import ChannelWidget
from ..widget_utilities.rgb_widget import RGBWidget
from ..widget_utilities.folder_list_widget import FolderListWidget
from ..utilities.io import (load_project_params, load_plots_params,
                            load_mask, get_mask_path)
from ..utilities.spectralindex_compute import (compute_index_projection,
                            clean_index_map, create_index, export_index_series,
                            compute_and_clean_index, load_index_zarr, load_projection,
                            save_index_zarr, compute_normalized_index_params, built_in_indices)
from ..utilities.spectralplot import (batch_create_plots, save_tif_cmap,
                                      plot_spectral_profile, plot_multi_spectral_profile)
from ..utilities.utils import wavelength_to_rgb
from ..utilities.morecolormaps import get_cmap_catalogue
get_cmap_catalogue()

class SpectralIndexWidget(QWidget):
    """
    Widget for the SpectralIndices.
    
    Parameters
    ----------
    napari_viewer: napari.Viewer

    Attributes
    ----------
    viewer: napari.Viewer
        napari viewer
    params: Param
        parameters for data
    params_plots: Paramplot
        parameters for plots
    em_boundary_lines: list of matplotlib.lines.Line2D
        lines for the em plot
    end_members: array
        each column hold values of an end member, last column is the bands
    endmember_bands: array
        bands of the end members (same as last column of end_members)
    index_file: str
        path to the index file
    main_layout: QVBoxLayout
        main layout of the widget
    tab_names: list of str
        names of the tabs
    tabs: TabSet
        tab widget
    
    
    
    """
    
    def __init__(self, napari_viewer):
        super().__init__()
        
        self.viewer = napari_viewer
        #self.viewer2 = None

        self.create_index_list()

        self.export_folder = None

        self.params = Param()
        self.params_plots = Paramplot()
        self.params_multiplots = Paramplot()

        self.em_boundary_lines = None
        self.end_members = None
        self.endmember_bands = None
        self.index_file = None
        self.current_plot_type = 'single'

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ["&Main", "&Index Definition", "Index C&ompute", "P&lots", "Batch"]#, "Plotslive"]
        self.tabs = TabSet(self.tab_names, tab_layouts=[None, QGridLayout(), None, QGridLayout(), None])

        self.main_layout.addWidget(self.tabs)


        # "Main" Tab Elements
        ### Elements "Files and folders" ###
        self.files_group = VHGroup('Files and Folders', orientation='G')
        self.tabs.add_named_tab('&Main', self.files_group.gbox)
        self.btn_select_export_folder = QPushButton("Set Project folder")
        self.export_path_display = QLineEdit("No path")
        self.files_group.glayout.addWidget(self.btn_select_export_folder, 0, 0, 1, 1)
        self.files_group.glayout.addWidget(self.export_path_display, 0, 1, 1, 1)
        self.btn_load_project = QPushButton("Import Project")
        self.files_group.glayout.addWidget(self.btn_load_project, 1, 0, 1, 1)
        self.spin_selected_roi = QSpinBox()
        self.spin_selected_roi.setRange(0, 0)
        self.spin_selected_roi.setValue(0)
        self.files_group.glayout.addWidget(QLabel('Selected ROI'), 2, 0, 1, 1)
        self.files_group.glayout.addWidget(self.spin_selected_roi, 2, 1, 1, 1)
        
        ### Elements "Bands" ###
        self.band_group = VHGroup('Bands', orientation='G')
        self.tabs.add_named_tab('&Main', self.band_group.gbox)
        self.band_group.glayout.addWidget(QLabel('Bands to load'), 0, 0, 1, 2)
        self.qlist_channels = ChannelWidget(self.viewer)
        self.band_group.glayout.addWidget(self.qlist_channels, 1,0,1,2)
        self.qlist_channels.itemClicked.connect(self._on_change_select_bands)

        ### RGB Widget ###
        self.rgbwidget = RGBWidget(viewer=self.viewer, translate=False)
        self.tabs.add_named_tab('&Main', self.rgbwidget.rgbmain_group.gbox)


        # "Index Definition" Tab 
        ### Elements "Index Definition" ###
        self._create_indices_tab()
        tab_rows = self.tabs.widget(1).layout().rowCount()

        ### SpectralPlotter ###
        self.em_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.tabs.add_named_tab('&Index Definition', self.em_plot, grid_pos=(tab_rows, 0, 1, 3))

        ### Index Definition Elements for SpectralPlotter ###
        self.em_boundaries_range = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.em_boundaries_range.setValue((0, 0, 0))
        self.em_boundaries_range2 = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.em_boundaries_range2.setValue((0, 0))
        self.tabs.add_named_tab('&Index Definition', QLabel('RABD/RABDnorm'), grid_pos=(tab_rows+1, 0, 1, 1))
        self.tabs.add_named_tab('&Index Definition', self.em_boundaries_range, grid_pos=(tab_rows+1, 1, 1, 2))
        self.tabs.add_named_tab('&Index Definition', QLabel('RABA/Ratio/RMean'), grid_pos=(tab_rows+2, 0, 1, 1))
        self.tabs.add_named_tab('&Index Definition', self.em_boundaries_range2, grid_pos=(tab_rows+2, 1, 1, 2))
        self.btn_create_index = QPushButton("New index")
        self.tabs.add_named_tab('&Index Definition', self.btn_create_index, grid_pos=(tab_rows+3, 0, 1, 1))
        self.combobox_index_type = QComboBox()
        self.combobox_index_type.addItems(['RABD', 'RABDnorm', 'RABA', 'Ratio', 'RMean'])
        self.tabs.add_named_tab('&Index Definition', self.combobox_index_type, grid_pos=(tab_rows+3, 1, 1, 1))
        self.qtext_new_index_name = QLineEdit()
        self.tabs.add_named_tab('&Index Definition', self.qtext_new_index_name, grid_pos=(tab_rows+3, 2, 1, 2))
        self.btn_update_index = QPushButton("Update current index")
        self.tabs.add_named_tab('&Index Definition', self.btn_update_index, grid_pos=(tab_rows+4, 0, 1, 1))
        self.btn_save_endmembers_plot = QPushButton("Save endmembers plot")
        self.tabs.add_named_tab('&Index Definition', self.btn_save_endmembers_plot, grid_pos=(tab_rows+5, 0, 1, 3))


        # "Index Compute" Tab
        ### Elements "Index Selection" ###
        self.index_pick_group = VHGroup('Index Selection', orientation='G')
        self.index_pick_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Index C&ompute', self.index_pick_group.gbox)
        self._create_index_io_pick()

        ### I/O index settings ###
        self.index_io_group = VHGroup('Index I/O', orientation='G')
        self.index_io_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Index C&ompute', self.index_io_group.gbox)
        self.btn_export_index_settings = QPushButton("Export index settings")
        self.index_io_group.glayout.addWidget(self.btn_export_index_settings, 0, 0, 1, 2)
        self.btn_import_index_settings = QPushButton("Import index settings")
        self.index_io_group.glayout.addWidget(self.btn_import_index_settings, 1, 0, 1, 2)
        self.index_file_display = QLineEdit("No file selected")
        self.index_io_group.glayout.addWidget(self.index_file_display, 2, 0, 1, 2)

        ### Elements "Projection" ###
        self.index_options_group = VHGroup('Projection', orientation='G')
        self.index_options_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Index C&ompute', self.index_options_group.gbox)
        self.check_smooth_projection = QCheckBox("Smooth projection")
        self.index_options_group.glayout.addWidget(self.check_smooth_projection, 0, 0, 1, 2)
        self.slider_index_savgol = QLabeledDoubleSlider(Qt.Horizontal)
        self.slider_index_savgol.setRange(5, 100)
        self.slider_index_savgol.setSingleStep(1)
        self.slider_index_savgol.setSliderPosition(5)
        self.index_options_group.glayout.addWidget(QLabel('Smoothing window size'), 1, 0, 1, 1)
        self.index_options_group.glayout.addWidget(self.slider_index_savgol, 1, 1, 1, 1)
        self.spin_roi_width = QSpinBox()
        self.spin_roi_width.setRange(1, 1000)
        self.spin_roi_width.setValue(20)
        self.index_options_group.glayout.addWidget(QLabel('Projection roi width'), 2, 0, 1, 1)
        self.index_options_group.glayout.addWidget(self.spin_roi_width, 2, 1, 1, 1)
        self.btn_save_roi = QPushButton("Save Projection ROI")
        self.index_options_group.glayout.addWidget(self.btn_save_roi, 3, 0, 1, 2)

        ### Elements "Compute and export" ###
        self.index_compute_group = VHGroup('Compute and export', orientation='G')
        self.index_compute_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Index C&ompute', self.index_compute_group.gbox)
        self.btn_add_index_maps_to_viewer = QPushButton("Add index map(s) to Viewer")
        self.index_compute_group.glayout.addWidget(self.btn_add_index_maps_to_viewer, 1, 0, 1, 2)

        self.btn_export_index_tiff = QPushButton("Export index map(s) to tiff")
        self.index_compute_group.glayout.addWidget(self.btn_export_index_tiff, 2, 0, 1, 2)
        self.btn_export_indices_csv = QPushButton("Export index projections to csv")
        self.index_compute_group.glayout.addWidget(self.btn_export_indices_csv, 3, 0, 1, 2)
        self.btn_export_indices_zarr = QPushButton("Export index map(s) to zarr")
        self.index_compute_group.glayout.addWidget(self.btn_export_indices_zarr, 4, 0, 1, 2)
        self.check_force_recompute = QCheckBox("Force recompute")
        self.index_compute_group.glayout.addWidget(self.check_force_recompute, 5, 0, 1, 2)
        self.check_force_recompute.setChecked(True)
        self.check_force_recompute.setToolTip("Force recompute of index maps. If only adjusting plot options can be unchecked.")


        # "Plots" Tab

        self.tabs.widget(3).layout().setAlignment(Qt.AlignTop)

        ### Create figure
        self.scale = 1.0
        self.pix_width = None
        self.pix_height = None

        self.pixlabel = QLabel()
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.pixlabel)
        self.scrollArea.setWidgetResizable(True)

        self.index_plot_live = SpectralPlotter(napari_viewer=self.viewer)
        self.index_plot_live.canvas.figure.set_layout_engine('none')

        ### Elements "Live plot" ###
        self.live_plot_group = VHGroup('Live plot', orientation='G')
        self.live_plot_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('P&lots', self.live_plot_group.gbox)

        self.btn_create_index_plot = QPushButton("Create index plot")
        self.btn_create_multi_index_plot = QPushButton("Create multi-index plot")
        self.live_plot_group.glayout.addWidget(self.btn_create_index_plot, 1, 0, 1, 2)
        self.live_plot_group.glayout.addWidget(self.btn_create_multi_index_plot, 1, 2, 1, 2)

        # Plot formatting
        self.spin_title_font = QDoubleSpinBox()
        self.spin_label_font = QDoubleSpinBox()
        for sbox in [self.spin_label_font, self.spin_title_font]:
            sbox.setRange(0, 100)
            sbox.setValue(12)
            sbox.setSingleStep(1)
        self.live_plot_group.glayout.addWidget(QLabel('Title Font'), 2, 0, 1, 1)
        self.live_plot_group.glayout.addWidget(self.spin_title_font, 2, 1, 1, 1)
        self.live_plot_group.glayout.addWidget(QLabel('Label Font'), 2, 2, 1, 1)
        self.live_plot_group.glayout.addWidget(self.spin_label_font, 2, 3, 1, 1)
        self.qcolor_plotline = QColorDialog()
        self.btn_qcolor_plotline = QPushButton("Select plot line color")
        self.live_plot_group.glayout.addWidget(self.btn_qcolor_plotline, 3, 0, 1, 2)
        self.qcolor_plotline.setCurrentColor(Qt.blue)
        self.spin_plot_thickness = QDoubleSpinBox()
        self.spin_plot_thickness.setRange(1, 10)
        self.spin_plot_thickness.setValue(1)
        self.spin_plot_thickness.setSingleStep(0.1)
        self.live_plot_group.glayout.addWidget(QLabel('Plot line thickness'), 3, 2, 1, 1)
        self.live_plot_group.glayout.addWidget(self.spin_plot_thickness, 3, 3, 1, 1)

        self.button_zoom_in = QPushButton('Zoom IN', self)
        self.button_zoom_in.clicked.connect(self.on_zoom_in)
        self.button_zoom_out = QPushButton('Zoom OUT', self) 
        self.button_zoom_out.clicked.connect(self.on_zoom_out)
        
        self.spin_preview_dpi = QSpinBox()
        self.spin_preview_dpi.setRange(10, 1000)
        self.spin_preview_dpi.setValue(100)
        self.spin_preview_dpi.setSingleStep(1)

        self.spin_final_dpi = QSpinBox()
        self.spin_final_dpi.setRange(100, 1000)
        self.spin_final_dpi.setValue(300)
        self.spin_final_dpi.setSingleStep(1)

        self.live_plot_group.glayout.addWidget(self.button_zoom_in, 4, 0, 1, 1)
        self.live_plot_group.glayout.addWidget(self.button_zoom_out, 4, 1, 1, 1)
        self.live_plot_group.glayout.addWidget(QLabel('Preview DPI'), 5, 0, 1, 1)
        self.live_plot_group.glayout.addWidget(self.spin_preview_dpi, 5, 1, 1, 1)
        self.live_plot_group.glayout.addWidget(QLabel('Final DPI'), 6, 0, 1, 1)
        self.live_plot_group.glayout.addWidget(self.spin_final_dpi, 6, 1, 1, 1)

        ### Elements "Live plot" ###
        self.metadata_group = VHGroup('Metadata', orientation='G')
        self.metadata_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('P&lots', self.metadata_group.gbox)

        self.metadata_location = QLineEdit("No location")
        self.metadata_location.setToolTip("Indicate the location of data acquisition")
        self.metadata_group.glayout.addWidget(QLabel('Location'), 0, 0, 1, 1)
        self.metadata_group.glayout.addWidget(self.metadata_location, 0, 1, 1, 3)
        self.spinbox_metadata_scale = QDoubleSpinBox()
        self.spinbox_metadata_scale.setToolTip("Indicate conversion factor from pixel to mm")
        self.spinbox_metadata_scale.setDecimals(4)
        self.spinbox_metadata_scale.setRange(0, 1000)
        self.spinbox_metadata_scale.setSingleStep(0.0001)
        self.spinbox_metadata_scale.setValue(1)
        self.scale_name = QLineEdit("No location")
        self.metadata_group.glayout.addWidget(QLabel('Scale'), 1, 0, 1, 1)
        self.metadata_group.glayout.addWidget(self.spinbox_metadata_scale, 1, 1, 1, 1)

        ### Elements "Save plot" ###
        self.save_plot_group = VHGroup('Save plot', orientation='G')
        self.save_plot_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('P&lots', self.save_plot_group.gbox)

        self.btn_save_plot_params = QPushButton("Save plot parameters")
        self.save_plot_group.glayout.addWidget(self.btn_save_plot_params, 0, 0, 1, 2)
        self.btn_load_plot_params = QPushButton("Load plot parameters")
        self.save_plot_group.glayout.addWidget(self.btn_load_plot_params, 1, 0, 1, 2)

        self.btn_save_all_plot = QPushButton("Save index maps and index plots")
        self.save_plot_group.glayout.addWidget(self.btn_save_all_plot, 2, 0, 1, 2)
        self.check_index_all_rois = QCheckBox("Process all ROIs")
        self.check_index_all_rois.setToolTip("Process all ROIs with the same settings")
        self.check_index_all_rois.setChecked(False)
        self.save_plot_group.glayout.addWidget(self.check_index_all_rois, 3, 0, 1, 1)
        self.check_normalize_single_export = QCheckBox("Normalize index plots")
        self.check_normalize_single_export.setChecked(False)
        self.check_normalize_single_export.setEnabled(False)
        self.check_normalize_single_export.setToolTip("Normalize index plots across ROIs")
        self.save_plot_group.glayout.addWidget(self.check_normalize_single_export, 3, 1, 1, 1)

        # "Batch" Tab
        self.tabs.widget(4).layout().setAlignment(Qt.AlignTop)
        self.btn_select_main_folder = QPushButton("Select main folder")
        self.tabs.add_named_tab('Batch', self.btn_select_main_folder)
        self.main_path_display = QLineEdit("No path")
        self.tabs.add_named_tab('Batch',self.main_path_display)
        self.tabs.add_named_tab('Batch', QLabel('Available folders'))
        self.file_list = FolderListWidget(napari_viewer)
        self.tabs.add_named_tab('Batch',self.file_list)
        self.file_list.setFixedHeight(100)
        self.file_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.batch_plot_params_file = FileEdit()
        self.tabs.add_named_tab('Batch', QLabel('Plot parameters file'))
        self.tabs.add_named_tab('Batch', self.batch_plot_params_file.native)
        self.batch_index_params_file = FileEdit()
        self.tabs.add_named_tab('Batch', QLabel('Index settings file'))
        self.tabs.add_named_tab('Batch', self.batch_index_params_file.native)
        self.btn_batch_create_plots = QPushButton("Create plots")
        self.tabs.add_named_tab('Batch', self.btn_batch_create_plots)
        self.check_normalize = QCheckBox("Normalize")
        self.tabs.add_named_tab('Batch', self.check_normalize)
        
        self._connect_spin_bounds()
        self.add_connections()

        self.var_init()

    def _create_indices_tab(self):
        """ Generates the elements of the "Index Definition" subelement in the "Index Definition" tab. """

        self.current_index_type = 'RABD'

        self.indices_group = VHGroup('&Index Definition', orientation='G')
        self.tabs.add_named_tab('&Index Definition', self.indices_group.gbox, [1, 0, 1, 3])

        self.qcom_indices = QComboBox()
        self.qcom_indices.addItems([value.index_name for key, value in self.index_collection.items()])
        self.spin_index_left, self.spin_index_right, self.spin_index_middle= [self._spin_boxes() for _ in range(3)]
        self.indices_group.glayout.addWidget(self.qcom_indices, 0, 0, 1, 1)
        self.indices_group.glayout.addWidget(self.spin_index_left, 1, 0, 1, 1)
        self.indices_group.glayout.addWidget(self.spin_index_middle, 1, 1, 1, 1)
        self.indices_group.glayout.addWidget(self.spin_index_right, 1, 2, 1, 1)
        

    def _spin_boxes(self, minval=0, maxval=1000):
        """
        Create a spin box with a range of minval to maxval.
        Helper function used in "_create_indices_tab"
        """
        spin = QSpinBox()
        spin.setRange(minval, maxval)
        return spin
          
    def create_index_list(self):
        """
        Create a collection of spectral indices as index_collection attribute.
        Index collection used in the following functions:
        - compute_selected_indices_map_and_proj: 
          computes index map and projection for index_name and completes the index_collection attributes.
        - create_index_io_pick: 
          creates tick boxes for picking indices to export
        - create_single_index_plot: 
          creates a single index plot. The plot can be displayed live but is not saved to file
        - create_multi_index_plot: 
          creates a multi-index plot. The plot can be displayed live but is not saved to file
        - create_and_save_all_single_index_plot: 
          creates and saves all single index plots for the selected indices and saves to file
        - create_projection_table: 
          creates the projection table for the index_name
        - _on_click_new_index: 
          adds new custom index
        - _on_click_update_index: 
          updates the current index
        - _on_compute_index_maps: 
          computes the index and adds it to napari
        - _on_add_index_map_to_viewer: 
          computes the index and adds it to napari
        - _on_change_index_map_rendering: 
          update the contrast limits of the index layers
        - _on_change_index_index
        - _on_export_index_projection
        - _on_click_export_index_tiff: exports index maps to tiff
        - _on_click_export_index_settings: exports index setttings
        - _on_click_import_index_settings: load index settings from file
        - var_init
        """

        self.index_collection = built_in_indices()
        
        
    def add_connections(self):
        """Add callbacks"""

        self.btn_select_export_folder.clicked.connect(self._on_click_select_export_folder)
        self.btn_load_project.clicked.connect(self.import_project)
        self.spin_selected_roi.valueChanged.connect(self.load_data)
        self.btn_save_roi.clicked.connect(self._on_click_save_roi)
        self.em_boundaries_range.valueChanged.connect(self._on_change_em_boundaries)
        self.em_boundaries_range2.valueChanged.connect(self._on_change_em_boundaries)
        self.btn_add_index_maps_to_viewer.clicked.connect(self._on_add_index_map_to_viewer)
        self.btn_save_endmembers_plot.clicked.connect(self.save_endmembers_plot)
        self.btn_create_index.clicked.connect(self._on_click_new_index)
        self.btn_update_index.clicked.connect(self._on_click_update_index)
        self.qcom_indices.activated.connect(self._on_change_index_index)
        self.btn_export_index_tiff.clicked.connect(self._on_click_export_index_tiff)
        self.btn_export_index_settings.clicked.connect(self._on_click_export_index_settings)
        self.btn_import_index_settings.clicked.connect(self._on_click_import_index_settings)
        self.btn_create_index_plot.clicked.connect(self._on_click_create_single_index_liveplot)
        self.btn_create_multi_index_plot.clicked.connect(self._on_click_create_multi_index_liveplot)
        self.btn_export_indices_csv.clicked.connect(self._on_export_index_projection)
        self.btn_export_indices_zarr.clicked.connect(self._on_click_export_index_zarr)

        self.connect_plot_formatting()
        self.btn_qcolor_plotline.clicked.connect(self._on_click_open_plotline_color_dialog)
        self.btn_save_all_plot.clicked.connect(self._on_click_create_and_save_all_plots)
        self.check_index_all_rois.stateChanged.connect(self._on_change_index_all_rois)
        self.btn_save_plot_params.clicked.connect(self._on_click_save_plot_parameters)
        self.btn_load_plot_params.clicked.connect(self._on_click_load_plot_parameters)

        self.btn_select_main_folder.clicked.connect(self._on_click_select_main_batch_folder)
        self.file_list.currentTextChanged.connect(self._on_change_filelist)
        self.btn_batch_create_plots.clicked.connect(self._on_click_batch_create_plots)

        self.viewer.mouse_double_click_callbacks.append(self._add_analysis_roi)
        self.viewer.mouse_double_click_callbacks.append(self.pick_pixel)

    def _connect_spin_bounds(self):

        self.spin_index_left.valueChanged.connect(self._on_change_spin_bounds)
        self.spin_index_middle.valueChanged.connect(self._on_change_spin_bounds)
        self.spin_index_right.valueChanged.connect(self._on_change_spin_bounds)

    def _disconnect_spin_bounds(self):
            
        self.spin_index_left.valueChanged.disconnect(self._on_change_spin_bounds)
        self.spin_index_middle.valueChanged.disconnect(self._on_change_spin_bounds)
        self.spin_index_right.valueChanged.disconnect(self._on_change_spin_bounds)

    def _on_change_spin_bounds(self, event=None):
        """
        Called: "Index Definition" tab, index spinboxes left, middle and right
        """

        if self.current_index_type in ['RABD', 'RABDnorm']:
            self.em_boundaries_range.setValue(
                (self.spin_index_left.value(), self.spin_index_middle.value(),
                self.spin_index_right.value()))
        else:
            self.em_boundaries_range2.setValue(
                (self.spin_index_left.value(), self.spin_index_right.value()))
    
    def _on_click_select_export_folder(self, event=None, export_folder=None):
        """
        Interactively select folder to analyze
        Called: "Main" tab, button "Set Project Folder"
        """

        if export_folder is None:
            return_path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            if return_path == '':
                return
            self.export_folder = Path(return_path)
        else:
            self.export_folder = Path(export_folder)
        self.export_path_display.setText(self.export_folder.as_posix())

    def _on_click_select_index_file(self):
        """Interactively select folder to analyze"""

        self.index_file = Path(str(QFileDialog.getOpenFileName(self, "Select Index file")[0]))
        self.index_file_display.setText(self.index_file.as_posix())

    def import_project(self):
        """
        Import pre-processed project: corrected roi and mask
        Called: "Main" tab, button "Import Project"
        """
        
        if self.export_folder is None:
            self._on_click_select_export_folder()

        export_path_roi = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')

        self.params = load_project_params(folder=self.export_folder)
        self.metadata_location.setText(self.params.location)
        self.spinbox_metadata_scale.setValue(self.params.scale)

        self.params_plots = load_plots_params(self.export_folder.joinpath('params_plots.yml'))
        if self.params_plots is None:
            self.params_plots = Paramplot(
                red_contrast_limits=None, green_contrast_limits=None, blue_contrast_limits=None)
        self.imhdr_path = Path(self.params.file_path)

        self.mainroi, _, self.measurement_roi = self.params.get_formatted_rois()

        self.spin_selected_roi.setRange(0, len(self.mainroi)-1)
        self.spin_selected_roi.setValue(0)
        
        self.load_data()

        
    def load_data(self, event=None):
        """
        Called: "Main" tab, qspinbox "Selected ROI"
        """
        
        to_remove = [l.name for l in self.viewer.layers if l.name not in ['imcube', 'red', 'green', 'blue']]
        for r in to_remove:
            self.viewer.layers.remove(r)

        self.var_init()

        roi_ind = self.spin_selected_roi.value()

        self.row_bounds, self.col_bounds = self.params.get_formatted_col_row_bounds(roi_ind)
        
        self.cursor_pos = np.array([np.mean(self.row_bounds), np.mean(self.col_bounds)])
        self.cursor_pos = self.cursor_pos.astype(int)

        export_path_roi = self.export_folder.joinpath(f'roi_{roi_ind}')

        self.imagechannels = ImChannels(self.export_folder.joinpath('corrected.zarr'))
        self.qlist_channels._update_channel_list(imagechannels=self.imagechannels)
        self.rgbwidget.imagechannels = self.imagechannels

        self.get_RGB()
        self.rgbwidget.load_and_display_rgb_bands(roi=np.concatenate([self.row_bounds, self.col_bounds]))
        self.rgbwidget._update_rgb_contrast(contrast_limits=
                                            [self.params_plots.red_contrast_limits,
                                             self.params_plots.green_contrast_limits,
                                             self.params_plots.blue_contrast_limits])

        self._on_click_load_mask()

        self.add_measurement_roi_layer()
        self.viewer.layers['rois'].data = [self.measurement_roi[roi_ind]]

        self.update_emplot_data()
        
        self.plot_endmembers()

        self._on_change_index_index()

        self._update_save_plot_parameters()
        self.current_plot_type = 'multi'
        self._update_save_plot_parameters()
        self.current_plot_type = 'single'

    def var_init(self):
        
        self.end_members = None
        self.endmember_bands = None
        self.index_file = None
        self.current_plot_type = 'single'
        self.em_plot.axes.clear()

        for key in self.index_collection.keys():
            self.index_collection[key].index_map = None
            self.index_collection[key].index_proj = None

    def _add_analysis_roi(self, viewer=None, event=None, roi_xpos=None):
        """Add roi to layer"""
        
        if 'Shift' not in event.modifiers:
            min_row = 0
            max_row = self.row_bounds[1] - self.row_bounds[0]
            if roi_xpos is None:
                cursor_pos = np.rint(self.viewer.cursor.position).astype(int)
                
                new_roi = [
                    [min_row, cursor_pos[2]-self.spin_roi_width.value()//2],
                    [max_row,cursor_pos[2]-self.spin_roi_width.value()//2],
                    [max_row,cursor_pos[2]+self.spin_roi_width.value()//2],
                    [min_row,cursor_pos[2]+self.spin_roi_width.value()//2]]
            
            else:
                new_roi = [
                    [min_row, roi_xpos-self.spin_roi_width.value()//2],
                    [max_row,roi_xpos-self.spin_roi_width.value()//2],
                    [max_row,roi_xpos+self.spin_roi_width.value()//2],
                    [min_row,roi_xpos+self.spin_roi_width.value()//2]]

            self.add_measurement_roi_layer()
            
            self.viewer.layers['rois'].data = [new_roi]
            self.viewer.layers['rois'].refresh()

    def add_measurement_roi_layer(self):
        """Add layer for measurement roi used to compute projections"""

        edge_width = np.min([10, self.viewer.layers['imcube'].data.shape[1]//100])
        if edge_width < 1:
            edge_width = 1
        
        if 'rois' not in self.viewer.layers:
            self.viewer.add_shapes(
                ndim = 2,
                name='rois', edge_color='red', face_color=np.array([0,0,0,0]), edge_width=edge_width)

    def pick_pixel(self, viewer, event, cursor_pos=None):
        """Pick pixel and plot spectral profile"""

        if 'Shift' in event.modifiers:
            if cursor_pos is None:
                self.cursor_pos = np.rint(viewer.cursor.position).astype(int)[-2::]
                self.update_emplot_data()
                self.plot_endmembers()

    def get_RGB(self):
        
        rgb_ch, rgb_names = self.imagechannels.get_indices_of_bands(self.rgbwidget.rgb)
        [self.qlist_channels.item(x).setSelected(True) for x in rgb_ch]
        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)

    def _on_change_select_bands(self, event=None):
        """
        Called: "Main" tab, channel in "Bands to load"
        """

        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)

    def _on_click_load_mask(self):
        """Load mask from file"""
        
        export_path_roi = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')
        mask_path = get_mask_path(export_path_roi)

        if mask_path.is_file():
            mask = load_mask(mask_path)
        else:
            mask = np.zeros(
                shape=(self.row_bounds[1]-self.row_bounds[0], self.col_bounds[1]-self.col_bounds[0]),
                dtype=np.uint8)

        if 'mask' in self.viewer.layers:
            self.viewer.layers['mask'].data = mask
        else:
            self.viewer.add_labels(mask, name='mask')

    def update_emplot_data(self):
        """Import endmembers or compute them and update the em plot boundaries."""

        export_path_roi = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')
        if export_path_roi.joinpath('end_members.csv').exists():
            self.end_members = pd.read_csv(export_path_roi.joinpath('end_members.csv')).values
            self.endmember_bands = self.end_members[:,-1]
            self.end_members = self.end_members[:,:-1]
        else:
            self.endmember_bands = self.imagechannels.centers
            # this loads an image into self.end_members, instead of actual
            # end-members. Leads to errors in other places e.g when changing 
            # to ROI without end-members. Maybe leftover from
            # previous implementation? Commenting out for the moment.
            '''self.end_members = self.imagechannels.get_image_cube(
                channels=np.arange(len(self.endmember_bands)),
                roi=[
                    self.cursor_pos[0]+self.row_bounds[0], self.cursor_pos[0]+self.row_bounds[0]+1,
                    self.cursor_pos[1]+self.col_bounds[0], self.cursor_pos[1]+self.col_bounds[0]+1]
                    ).compute().ravel()
            '''
        
        self.em_boundaries_range.setRange(min=self.endmember_bands[0],max=self.endmember_bands[-1])
        self.em_boundaries_range.setValue(
            (self.endmember_bands[0], (self.endmember_bands[-1]+self.endmember_bands[0])/2, self.endmember_bands[-1]))
        self.em_boundaries_range2.setRange(min=self.endmember_bands[0],max=self.endmember_bands[-1])
        self.em_boundaries_range2.setValue(
            (self.endmember_bands[0], self.endmember_bands[-1]))

    def plot_endmembers(self, event=None):
        """Cluster the pure pixels and plot the endmembers as average of clusters."""

        self.em_plot.axes.clear()
        # only plot if endmembers are available
        if self.end_members is not None:
            
            self.em_plot.axes.plot(self.endmember_bands, self.end_members)

            out = wavelength_to_rgb(self.endmember_bands.min(), self.endmember_bands.max(), 100)
            ax_histx = self.em_plot.axes.inset_axes([0.0,-0.5, 1.0, 1], sharex=self.em_plot.axes)
            ax_histx.imshow(out, extent=(self.endmember_bands.min(),self.endmember_bands.max(), 0,10))
            ax_histx.set_axis_off()

            self.em_plot.axes.set_xlabel('Wavelength', color='black')
            self.em_plot.axes.set_ylabel('Continuum removed', color='black')
            self.em_plot.axes.xaxis.label.set_color('black')
            self.em_plot.axes.yaxis.label.set_color('black')
            self.em_plot.axes.tick_params(axis='both', colors='black')
            self.em_plot.canvas.figure.patch.set_facecolor('white')
            self.em_plot.canvas.figure.canvas.draw()

    def save_endmembers_plot(self):
        """
        Save the endmembers plot to file
        Called: "Index Definition" tab, button "Save endmembers plot"
        """

        export_folder = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')

        if self.em_boundary_lines is not None:
            num_lines = len(self.em_boundary_lines)
            for i in range(num_lines):
                self.em_boundary_lines[i].set_color([0,0,0,0])
        self.em_plot.canvas.figure.canvas.draw()
        self.em_plot.canvas.figure.savefig(export_folder.joinpath('endmembers.png'), dpi=300)
        if self.em_boundary_lines is not None:
            for i in range(num_lines):
                self.em_boundary_lines[i].set_color([1,0,0])
        self.em_plot.canvas.figure.canvas.draw()

    def _on_change_em_boundaries(self, event=None):
        """
        Update the em plot when the em boundaries are changed.
        Called: "Index Definition" tab, sliders "RABD" and "RABA/Ratio"
        """
        
        #self._disconnect_spin_bounds()
        # update from interactive limit change
        if type(event) == tuple:
            if self.current_index_type == 'RABD':
                current_triplet = np.array(self.em_boundaries_range.value(), dtype=np.uint16)
                self.spin_index_left.setValue(current_triplet[0])
                self.spin_index_middle.setValue(current_triplet[1])
                self.spin_index_right.setValue(current_triplet[2])
            else:
                current_triplet = np.array(self.em_boundaries_range2.value(), dtype=np.uint16)
                self.spin_index_left.setValue(current_triplet[0])
                self.spin_index_right.setValue(current_triplet[1])
            
        # update from spinbox change
        else:
            if self.current_index_type == 'RABD':
                current_triplet = [self.spin_index_left.value(), self.spin_index_middle.value(), self.spin_index_right.value()]
                current_triplet = [float(x) for x in current_triplet]
                self.em_boundaries_range.setValue(current_triplet)
            else:
                current_triplet = [self.spin_index_left.value(), self.spin_index_right.value()]
                current_triplet = [float(x) for x in current_triplet]
                self.em_boundaries_range2.setValue(current_triplet)

        if self.em_boundary_lines is not None:
            num_lines = len(self.em_boundary_lines)
            for i in range(num_lines):
                self.em_boundary_lines.pop(0).remove()

        if self.end_members is not None:
            ymin = self.end_members.min()
            ymax = self.end_members.max()
            if self.current_index_type == 'RABD':
                x_toplot = current_triplet
                ymin_toplot = 3*[ymin]
                ymax_toplot = 3*[ymax]
            else:
                x_toplot = current_triplet
                ymin_toplot = 2*[ymin]
                ymax_toplot = 2*[ymax]
            
            try:
                self.em_boundary_lines = self.em_plot.axes.plot(
                    [x_toplot, x_toplot],
                    [ymin_toplot, ymax_toplot],
                    'r--'
            )
            except:
                pass
            self.em_plot.canvas.figure.canvas.draw()
        
        #self._connect_spin_bounds()

    def _update_save_plot_parameters(self):

        if self.current_plot_type == 'single':
            current_param = self.params_plots
        else:
            current_param = self.params_multiplots

        current_param.plot_thickness = self.spin_plot_thickness.value()
        current_param.title_font = self.spin_title_font.value()
        current_param.label_font = self.spin_label_font.value()
        
        current_param.red_contrast_limits = np.array(self.viewer.layers['red'].contrast_limits).tolist()
        current_param.green_contrast_limits = np.array(self.viewer.layers['green'].contrast_limits).tolist()
        current_param.blue_contrast_limits = np.array(self.viewer.layers['blue'].contrast_limits).tolist()
        current_param.rgb_bands = self.rgbwidget.rgb

    def _on_click_save_plot_parameters(self, event=None, file_path=None):
        """
        Called: "Plots" tab, button "Save plot parameters"
        """
            
        if file_path is None:
            file_path = Path(str(QFileDialog.getSaveFileName(self, "Select plot parameters file")[0]))
        self._update_save_plot_parameters()
        self.params_plots.save_parameters(file_path)

    def _on_click_load_plot_parameters(self, event=None, file_path=None):
        """
        Called: "Plots" tab, button "Load plot parameters"
        """
        
        try:
            self.disconnect_plot_formatting()
        except:
            pass
        if file_path is None:
            file_path = Path(str(QFileDialog.getOpenFileName(self, "Select plot parameters file")[0]))
        self.params_plots = load_plots_params(file_path=file_path)
        self.set_plot_interface(params=self.params_plots)
        self.rgbwidget.load_and_display_rgb_bands(roi=np.concatenate([self.row_bounds, self.col_bounds]))
        self.set_plot_interface(params=self.params_plots)
        self.connect_plot_formatting()
        self.update_single_or_multi_index_plot()
        

    def set_plot_interface(self, params):
        self.spin_plot_thickness.setValue(params.plot_thickness)
        self.spin_title_font.setValue(params.title_font)
        self.spin_label_font.setValue(params.label_font)
        self.qcolor_plotline.setCurrentColor(QColor(*[int(x*255) for x in params.color_plotline]))
        self.viewer.layers['red'].contrast_limits = params.red_contrast_limits
        self.viewer.layers['green'].contrast_limits = params.green_contrast_limits
        self.viewer.layers['blue'].contrast_limits = params.blue_contrast_limits
        self.rgbwidget.rgb = params.rgb_bands
        

    def disconnect_plot_formatting(self):
        """Disconnect plot editing widgets while loading parameters to avoid overwriting
        the loaded parameters."""
        
        try:
            self.spin_plot_thickness.valueChanged.disconnect(self.update_single_or_multi_index_plot)
            self.spin_title_font.valueChanged.disconnect(self.update_single_or_multi_index_plot)
            self.spin_label_font.valueChanged.disconnect(self.update_single_or_multi_index_plot)
            self.qcolor_plotline.currentColorChanged.disconnect(self.update_line_color)
        except:
            pass

    def connect_plot_formatting(self):
        """Reconnect plot editing widgets after loading parameters."""

        self.spin_plot_thickness.valueChanged.connect(self.update_single_or_multi_index_plot)
        self.spin_title_font.valueChanged.connect(self.update_single_or_multi_index_plot)
        self.spin_label_font.valueChanged.connect(self.update_single_or_multi_index_plot)
        self.qcolor_plotline.currentColorChanged.connect(self.update_line_color)

    def index_map_and_proj(self, index_name):
        
        colmin, colmax = self.get_roi_bounds()

        toplot = self.viewer.layers[index_name].data
        toplot = clean_index_map(toplot)

        proj = compute_index_projection(
            toplot, self.viewer.layers['mask'].data,
            colmin=colmin, colmax=colmax,
            smooth_window=self.get_smoothing_window())
        
        return toplot, proj
    
    def get_smoothing_window(self):
        if self.check_smooth_projection.isChecked():
            return int(self.slider_index_savgol.value())
        else:
            return None
    
    def _on_click_create_single_index_liveplot(self, event=None):
        """
        Called: "Plots" tab, button "Create index plot"
        """
        self.current_plot_type = 'single'
        self.disconnect_plot_formatting()
        self.set_plot_interface(params=self.params_plots)
        self.create_single_index_plot(event=event)
        self.connect_plot_formatting()

    def _on_click_create_multi_index_liveplot(self, event=None):
        """
        Called: "Plots" tab, button "Create mult-index plot"
        """
        self.current_plot_type = 'multi'
        self.disconnect_plot_formatting()
        self.set_plot_interface(params=self.params_multiplots)
        self.create_multi_index_plot(event=event)
        self.connect_plot_formatting()

    def _on_click_create_and_save_all_plots(self, event=None):
        """
        Called: "Index Compute" tab, button "Create and Save all index plots"
        """
        self.current_plot_type = 'single'
        self.disconnect_plot_formatting()
        self.set_plot_interface(params=self.params_plots)
        self.create_and_save_all_single_index_plot(event=event)
        self.create_and_save_multi_index_plot(event=event, force_recompute=False)
        self._on_click_export_index_tiff()
        self._on_click_save_plot_parameters(file_path=self.export_folder.joinpath('plot_settings.yml'))

        self.connect_plot_formatting()
        self._on_click_export_index_settings()

    def _on_change_index_all_rois(self, event=None):
        """
        Called: "Index Compute" tab, checkbox "Process all ROIs"
        """

        # Disable the "Normalize" checkbox if not multiple rois are computed
        if self.check_index_all_rois.isChecked():
            self.check_normalize_single_export.setEnabled(True)
        else:
            self.check_normalize_single_export.setEnabled(False)
    
    def update_single_or_multi_index_plot(self, event=None):
        if self.current_plot_type == 'single':
            self.create_single_index_plot(event=event)
        else:
            self.create_multi_index_plot(event=event)

    def update_line_color(self, event=None):
        """Override the line color for the plot."""

        if self.current_plot_type == 'single':
            current_param = self.params_plots
        else:
            current_param = self.params_multiplots

        current_param.color_plotline = [self.qcolor_plotline.currentColor().getRgb()[x]/255 for x in range(3)]
        

    def _on_click_batch_create_plots(self, event=None):
        """
        Create all plots for all projects in the main folder, given
        index and plotting settings
        Called: "Batch" tab, button "Create plots"
        """

        export_folder = self.file_list.folder_path
        exported_projects = list(export_folder.iterdir())
        exported_projects = [e for e in exported_projects if e.is_dir()]

        batch_create_plots(
            project_list=exported_projects,
            index_params_file=self.batch_index_params_file.value,
            plot_params_file=self.batch_plot_params_file.value
            )
        
        if self.check_normalize.isChecked():
            compute_normalized_index_params(
                project_list=exported_projects,
                index_params_file=self.batch_index_params_file.value,
                export_folder=export_folder
            )
            batch_create_plots(
                project_list=exported_projects,
                index_params_file=export_folder.joinpath('normalized_index_settings.yml'),
                plot_params_file=self.batch_plot_params_file.value,
                normalize=True,
                load_data=True
            )
            

    def get_roi_bounds(self):

        if 'rois' not in self.viewer.layers:
            return self.col_bounds[0], self.col_bounds[1]
        elif len(self.viewer.layers['rois'].data) == 0:
            return self.col_bounds[0], self.col_bounds[1]
        else:
            colmin = int(self.viewer.layers['rois'].data[0][:,1].min())
            colmax = int(self.viewer.layers['rois'].data[0][:,1].max())

        return colmin, colmax
    
    def compute_selected_indices_map_and_proj(self, index_names, force_recompute=False):
        """Compute index map and projection for index_name
        and complete the index_collection attributes."""

        colmin, colmax = self.get_roi_bounds()
        for index_name in index_names:
            # if not force_recompute, check if index map is already computed
            # or stored in zarr file
            if not force_recompute:
                if self.index_collection[index_name].index_map is None:
                    self.index_collection[index_name].index_map = load_index_zarr(
                        project_folder=self.export_folder,
                        main_roi_index=self.spin_selected_roi.value(),
                        index_name=self.index_collection[index_name].index_name)
                    self.index_collection[index_name].index_proj = load_projection(
                        project_folder=self.export_folder,
                        main_roi_index=self.spin_selected_roi.value(),
                        index_name=self.index_collection[index_name].index_name)
            
            # if force_recompute or index map is not stored, compute it
            if (self.index_collection[index_name].index_map is None) or force_recompute:
                computed_index = compute_and_clean_index(
                    spectral_index=self.index_collection[index_name],
                    row_bounds=self.row_bounds, col_bounds=self.col_bounds,
                    imagechannels=self.imagechannels)
                proj = compute_index_projection(
                    computed_index, self.viewer.layers['mask'].data,
                    colmin=colmin, colmax=colmax,
                    smooth_window=self.get_smoothing_window())
                self.index_collection[index_name].index_map = computed_index
                self.index_collection[index_name].index_proj = proj

    def create_single_index_plot(self, event=None, force_recompute=None):
        """Create a single index plot. The plot can be displayed live but is not
        saved to file."""

        if force_recompute is None:
            force_recompute = self.check_force_recompute.isChecked()

        self._update_save_plot_parameters()
        self.params.location = self.metadata_location.text()
        self.params.scale = self.spinbox_metadata_scale.value()

        # get rgb image and index image to plot
        rgb_image = self.get_rgb_array()

        # get selected indices
        index_series = [x for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
        if len(index_series) == 0:
            warnings.warn('No index selected') 
            return
        elif len(index_series) > 2:
            raise ValueError('Only one or two indices can be selected for single index plot')

        # get mask
        mask = self.viewer.layers['mask'].data

        # compute maps if necessary
        self.compute_selected_indices_map_and_proj([ind.index_name for ind in index_series],
                                                   force_recompute=force_recompute)

        roi = None
        if 'rois' in self.viewer.layers:
            roi=self.viewer.layers['rois'].data[0]

        format_dict = asdict(self.params_plots)
        _, self.ax1, self.ax2, self.ax3 = plot_spectral_profile(
            rgb_image=rgb_image, mask=mask, index_obj=[self.index_collection[ind.index_name] for ind in index_series],
            format_dict=format_dict, scale=self.params.scale, scale_unit=self.params.scale_units,
            location=self.params.location, fig=self.index_plot_live.canvas.figure, 
            roi=roi)

        # save temporary low-res figure for display in napari
        self.index_plot_live.canvas.figure.savefig(
            self.export_folder.joinpath('temp.png'),
            dpi=self.spin_preview_dpi.value())#, bbox_inches="tight")

        # update napari preview
        if self.pix_width is None:
            self.pix_width = self.pixlabel.size().width()
            self.pix_height = self.pixlabel.size().height()
        self.pixmap = QPixmap(self.export_folder.joinpath('temp.png').as_posix())
        self.pixlabel.setPixmap(self.pixmap)
        self.scrollArea.show()

    def create_multi_index_plot(self, event=None, show_plot=True, force_recompute=None):
        """Create a multi-index plot. The plot can be displayed live but is not
        saved to file."""
        
        if force_recompute is None:
            force_recompute = self.check_force_recompute.isChecked()

        self._update_save_plot_parameters()
        self.params.location = self.metadata_location.text()
        self.params.scale = self.spinbox_metadata_scale.value()
        # get rgb image and index image to plot
        rgb_image = [self.viewer.layers[c].data for c in ['red', 'green', 'blue']]
        if isinstance(rgb_image[0], da.Array):
            rgb_image = [x.compute() for x in rgb_image]
        
        index_series = [x for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
        self.compute_selected_indices_map_and_proj([x.index_name for x in index_series], force_recompute=force_recompute)
        
        roi = None
        if 'rois' in self.viewer.layers:
            roi=self.viewer.layers['rois'].data[0]

        format_dict = asdict(self.params_multiplots)
        plot_multi_spectral_profile(
            rgb_image=rgb_image, mask=self.viewer.layers['mask'].data,
            index_objs=index_series, 
            format_dict=format_dict,
            scale=self.params.scale,
            scale_unit=self.params.scale_units,
            location=self.params.location, 
            fig=self.index_plot_live.canvas.figure,
            roi=roi)
        
        if show_plot:
            # save temporary low-res figure for display in napari
            self.index_plot_live.canvas.figure.savefig(
                self.export_folder.joinpath('temp.png'),
                dpi=self.spin_preview_dpi.value())#, bbox_inches="tight")
            
            # update napari preview
            if self.pix_width is None:
                self.pix_width = self.pixlabel.size().width()
                self.pix_height = self.pixlabel.size().height()
            self.pixmap = QPixmap(self.export_folder.joinpath('temp.png').as_posix())
            #self.pixlabel.setPixmap(self.pixmap.scaled(self.pix_width, self.pix_height, Qt.KeepAspectRatio))
            self.pixlabel.setPixmap(self.pixmap)
            self.scrollArea.show()

    def create_and_save_all_single_index_plot(self, event=None):
        """Create and save all single index plots for the selected indices and
        save to file. If normalize is checked, also create and save normalized
        index plots."""

        roi_to_process = None
        if not self.check_index_all_rois.isChecked():
            roi_to_process = [self.spin_selected_roi.value()]

        self._update_save_plot_parameters()
        self._on_click_save_plot_parameters(file_path=self.export_folder.joinpath('plot_settings.yml'))
        self.params.location = self.metadata_location.text()
        self.params.scale = self.spinbox_metadata_scale.value()
        self.params.save_parameters()
        self._on_click_export_index_settings()

        batch_create_plots(
                project_list=[self.export_folder],
                index_params_file=self.export_folder.joinpath('index_settings.yml'),
                plot_params_file=self.export_folder.joinpath('plot_settings.yml'),
                normalize=False, load_data=False, roi_to_process=roi_to_process
            )

        if self.check_normalize_single_export.isChecked():
            compute_normalized_index_params(
                project_list=[self.export_folder],
                index_params_file=self.export_folder.joinpath('index_settings.yml'),
                export_folder=self.export_folder)
                
            batch_create_plots(
                project_list=[self.export_folder],
                index_params_file=self.export_folder.joinpath('normalized_index_settings.yml'),
                plot_params_file=self.export_folder.joinpath('plot_settings.yml'),
                normalize=True,
                load_data=True,
                roi_to_process=roi_to_process
            )
            

    def create_and_save_multi_index_plot(self, event=None, force_recompute=None):
        """Create and save multi index plot for the selected indices and
        save to file."""

        if force_recompute is None:
            force_recompute = self.check_force_recompute.isChecked()

        self.create_multi_index_plot(event=None, show_plot=False, force_recompute=force_recompute)
        self.index_plot_live.canvas.figure.savefig(
                self.plot_folder().joinpath(f'multi_index_plot.png'),
                dpi=self.spin_final_dpi.value())

    def on_close_callback(self):
        print('Viewer closed')

    def _on_resize_pixlabel(self, event=None):

        self.pixlabel.setPixmap(self.pixmap.scaled(self.pixlabel.size().width(), self.pixlabel.size().height(), Qt.KeepAspectRatio))


    def _on_click_reset_figure_size(self, event=None):
        """Reset figure size to default"""

        self.index_plot_live.canvas.set_size_inches(self.fig_size)
        self.index_plot_live.canvas.figure.canvas.draw()
        self.index_plot_live.canvas.figure.canvas.flush_events()

        vsize = self.viewer.window.geometry()
        self.viewer.window.resize(vsize[2]-10,vsize[3]-10)
        self.viewer.window.resize(vsize[2],vsize[3])

    def on_zoom_in(self, event):
        """
        Called: "Plots" tab, button "Zoom IN"
        """
        self.scale *= 2
        self.resize_image()

    def on_zoom_out(self, event):
        """
        Called: "Plots" tab, button "Zoom OUT"
        """
        self.scale /= 2
        self.resize_image()

    def resize_image(self):
        size = self.pixmap.size()

        scaled_pixmap = self.pixmap.scaled(self.scale * size)

        self.pixlabel.setPixmap(scaled_pixmap)

    def create_projection_table(self, index_names):
        """Create the projection table for the index_name"""

        proj_pd = pd.DataFrame({'depth': np.arange(0,len(self.index_collection[index_names[0]].index_proj))})

        for i in index_names:
            proj = self.index_collection[i].index_proj   
            proj_pd[i] = proj

        return proj_pd
    
    def _on_click_open_plotline_color_dialog(self, event=None):
        """
        Show label color dialog
        Called: "Plots" tab, button "Select plot line color"
        """
        
        self.qcolor_plotline.show()

    def _on_click_new_index(self, event):
        """
        Add new custom index
        Called: "Index Definition" tab, button "New index"
        """

        name = self.qtext_new_index_name.text()
        self.current_index_type = self.combobox_index_type.currentText()

        if self.combobox_index_type.currentText() in ('RABD', 'RABDnorm'):
            current_bands = np.array(self.em_boundaries_range.value(), dtype=np.uint16)
        else:
            current_bands = np.array(self.em_boundaries_range2.value(), dtype=np.uint16)
        self.index_collection[name] = create_index(
            index_name=name, index_type=self.combobox_index_type.currentText(), 
            boundaries=current_bands)
        
        if name not in [self.qcom_indices.itemText(i) for i in range(self.qcom_indices.count())]:
            self.qcom_indices.addItem(name)

            ## add box to pick
            num_boxes = len(self.index_collection)
            self.index_pick_group.glayout.addWidget(QLabel(name), num_boxes, 0, 1, 1)
            newbox = QCheckBox()
            self.index_pick_boxes[name] = newbox
            self.index_pick_group.glayout.addWidget(newbox, num_boxes, 1, 1, 1)

        self.qcom_indices.setCurrentText(name)
        

    def _on_click_update_index(self, event):
        """
        Update the current index.
        Called: "Index Definition" tab, button "Update current index"
        """

        name = self.qcom_indices.currentText()
        
        if self.current_index_type == 'RABD':
            current_bands = np.array(self.em_boundaries_range.value(), dtype=np.uint16)
            self.index_collection[name].left_band = current_bands[0]
            self.index_collection[name].right_band = current_bands[2]
            self.index_collection[name].middle_band = current_bands[1]
        else:
            current_bands = np.array(self.em_boundaries_range2.value(), dtype=np.uint16)
            self.index_collection[name].left_band = current_bands[0]
            self.index_collection[name].right_band = current_bands[1]
            self.index_collection[name].middle_band = None

    def _on_add_index_map_to_viewer(self, event=None, force_recompute=None):
        """
        Compute the index and add to napari.
        Called: "Index Compute" tab, button "Add index map(s) to Viewer"
        """

        if force_recompute is None:
            force_recompute = self.check_force_recompute.isChecked()

        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Computing index")

            index_series = [x for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
            
            self.compute_selected_indices_map_and_proj(
                index_names=[x.index_name for x in index_series],
                force_recompute=force_recompute)
            
            for i in index_series:
                computed_index = self.index_collection[i.index_name].index_map
                if i.index_name in self.viewer.layers:
                    self.viewer.layers[i.index_name].data = computed_index
                    self.viewer.layers[i.index_name].refresh()
                else:
                    colormap = Colormap(self.index_collection[i.index_name].colormap).identifier
                    contrast_limits = self.index_collection[i.index_name].index_map_range
                    layer = self.viewer.add_image(
                        data=computed_index, name=i.index_name, colormap=colormap,
                        blending='additive', contrast_limits=contrast_limits)
                    layer.events.contrast_limits.connect(self._on_change_index_map_rendering)
                    layer.events.colormap.connect(self._on_change_index_map_rendering)
        self.viewer.window._status_bar._toggle_activity_dock(False)

    def _on_change_index_map_rendering(self, event=None):
        """Update the contrast limits of the index layers."""

        if event is None:
            self.viewer.layers.selection.active
        else:
            layer = self.viewer.layers[event.index]
        if layer.name in self.index_collection:
            self.index_collection[layer.name].index_map_range = layer.contrast_limits
            self.index_collection[layer.name].colormap = layer.colormap._display_name

    
    def _on_change_index_index(self, event=None):
        """
        Called: "Index Definition" tab, slider "RABD" and comboboxes in subelement "Index Definition" 
        """

        current_index = self.index_collection[self.qcom_indices.currentText()]
        self.current_index_type = current_index.index_type
        self.spin_index_left.setValue(current_index.left_band)
        self.spin_index_right.setValue(current_index.right_band)
        if self.current_index_type == 'RABD':
            self.spin_index_middle.setValue(current_index.middle_band)

        if self.current_index_type == 'RABD':
            self.spin_index_middle.setVisible(True)
        else:
            self.spin_index_middle.setVisible(False)

        self._on_change_em_boundaries()

    def _create_index_io_pick(self):
        """
        Create tick boxes for picking indices to export.
        Called: "Index Compute" tab, "Index Selection" element, checkboxes
        """

        self.index_pick_boxes = {}
        for ind, key_val in enumerate(self.index_collection.items()):
            
            self.index_pick_group.glayout.addWidget(QLabel(key_val[0]), ind, 0, 1, 1)
            newbox = QCheckBox()
            self.index_pick_boxes[key_val[0]] = newbox
            self.index_pick_group.glayout.addWidget(newbox, ind, 1, 1, 1)
    
    def _on_click_save_plot(self, event=None, export_file=None):
        """
        Called: "Plots" tab, button "Save plot"
        """
        
        export_folder = self.plot_folder()
        if export_file is None:
            export_file = export_folder.joinpath(self.qcom_indices.currentText()+'_index_plot.png')
        self.index_plot_live.canvas.figure.savefig(
            fname=export_file, dpi=self.spin_final_dpi.value())#, bbox_inches="tight")
        self._on_export_index_projection()

    def _on_export_index_projection(self, event=None, force_recompute=False):
        """
        Called: "Index Compute" tab, button "Export index projections to csv"
        """

        export_folder = self.plot_folder()
        index_names = [x.index_name for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
        self.compute_selected_indices_map_and_proj(index_names, force_recompute=force_recompute)
                
        proj_pd = self.create_projection_table(index_names)
        proj_pd[f'depth [{self.params.scale_units}]'] = proj_pd['depth'] * self.params.scale
        proj_pd.to_csv(export_folder.joinpath('index_projection.csv'), index=False)

    def _on_click_export_index_zarr(self, event=None, force_recompute=False):
        """
        Export index maps to zarr
        Called: "Index Compute" tab, button "Export index map(s) to zarr"
        """
        
        index_series = [x for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
        self.compute_selected_indices_map_and_proj([x.index_name for x in index_series], force_recompute=False)
        
        for index_item in index_series:
            save_index_zarr(
                project_folder=self.export_folder, 
                main_roi_index=self.spin_selected_roi.value(),
                index_name=index_item.index_name,
                index_map=index_item.index_map,
                overwrite=True)

    def _on_click_export_index_tiff(self, event=None, force_recompute=False):
        """
        Export index maps to tiff
        Called: "Index Compute" tab, button "Export index map(s) to tiff"
        """
        
        export_folder = self.plot_folder()
        index_series = [x for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
        self.compute_selected_indices_map_and_proj([x.index_name for x in index_series], force_recompute=force_recompute)
        
        for index_item in index_series:
            index_map = index_item.index_map#self.viewer.layers[key].data
            contrast = index_item.index_map_range#self.viewer.layers[key].contrast_limits
            napari_cmap = index_item.colormap#self.viewer.layers[key].colormap
            export_path = export_folder.joinpath(f'{index_item.index_name}_index_map.tif')
            save_tif_cmap(image=index_map, image_path=export_path,
                            napari_cmap=napari_cmap, contrast=contrast)

    def _on_click_export_index_settings(self, event=None, file_path=None):
        """
        Export index settings
        Called: "Index Compute" tab, button "Export index settings"
        """

        index_series = {key: x for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()}
        for key, x in index_series.items():
            if self.check_smooth_projection.isChecked():
                x.smooth_proj_window = self.get_smoothing_window()
            else:
                x.smooth_proj_window = None
        file_path = self.export_folder.joinpath('index_settings.yml')
        export_index_series(index_series, file_path)
        

    def _on_click_import_index_settings(self, event=None):
        """
        Load index settings from file.
        Called: "Index Compute" tab, button "Import index settings"
        """
        
        if self.index_file is None:
            self._on_click_select_index_file()
        # clear existing state
        self.qcom_indices.clear()
        self.index_pick_boxes = {}
        self.index_collection = {}

        for i in reversed(range(self.index_pick_group.glayout.count())): 
            self.index_pick_group.glayout.itemAt(i).widget().setParent(None)

        with open(self.index_file) as file:
            index_series = yaml.full_load(file)
        for index_element in index_series['index_definition']:
            self.index_collection[index_element['index_name']] = SpectralIndex(**index_element)
            self.qcom_indices.addItem(index_element['index_name'])
            self.index_pick_boxes[index_element['index_name']] = QCheckBox()
            self.index_pick_group.glayout.addWidget(QLabel(index_element['index_name']), self.qcom_indices.count(), 0, 1, 1)
            self.index_pick_group.glayout.addWidget(self.index_pick_boxes[index_element['index_name']], self.qcom_indices.count(), 1, 1, 1)
        self.qcom_indices.setCurrentText(index_element['index_name'])
    
        self._on_change_index_index()

    def _on_click_save_roi(self, event=None):
        """
        Called: "Index Compute" tab, element "Projection", button "Save Projection ROI"
        """

        if 'rois' in self.viewer.layers:
            
            measurement_roi = list(self.viewer.layers['rois'].data[0].astype(int).flatten())
            measurement_roi = [x.item() for x in measurement_roi]
            if len(self.params.measurement_roi) == 0:
                self.params.measurement_roi = [[] for _ in range(len(self.params.main_roi))]
            self.params.measurement_roi[self.spin_selected_roi.value()] = measurement_roi
        
        self.params.save_parameters()

    def _on_click_select_main_batch_folder(self, event=None, main_folder=None):
        """
        Called: "Batch" tab, button "Select main folder"
        """
        
        if main_folder is None:
            main_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        else:
            main_folder = Path(main_folder)
        self.main_path_display.setText(main_folder.as_posix())
        self.file_list.update_from_path(main_folder)

    def _on_change_filelist(self):
        """
        Called: "Batch" tab, file list "Available folders"
        """
        
        main_folder = Path(self.file_list.folder_path)
        if self.file_list.currentItem() is None:
            return
        current_folder = main_folder.joinpath(self.file_list.currentItem().text())

    ### Helper functions
    def get_rgb_array(self):

        rgb_image = [self.viewer.layers[c].data for c in ['red', 'green', 'blue']]
        if isinstance(rgb_image[0], da.Array):
            rgb_image = [x.compute() for x in rgb_image]
        return rgb_image
    
    def plot_folder(self):
        """If necessary create folder for plots and return path."""

        export_folder = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')
        export_folder = export_folder.joinpath('index_plots')
        export_folder.mkdir(exist_ok=True)
        return export_folder
    
    def index_maps_zarr_folder(self):
        """If necessary create folder for index maps and return path."""

        export_folder = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')
        export_folder = export_folder.joinpath('index_maps')
        export_folder.mkdir(exist_ok=True)
        return export_folder


class ScaledPixmapLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setScaledContents(True)

    def paintEvent(self, event):
        if self.pixmap():
            pm = self.pixmap()
            originalRatio = pm.width() / pm.height()
            currentRatio = self.width() / self.height()
            if originalRatio != currentRatio:
                qp = QPainter(self)
                pm = self.pixmap().scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                rect = QRect(0, 0, pm.width(), pm.height())
                rect.moveCenter(self.rect().center())
                qp.drawPixmap(rect, pm)
                return
        super().paintEvent(event)

        


    