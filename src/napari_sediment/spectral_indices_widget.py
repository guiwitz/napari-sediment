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
                            QColorDialog, QScrollArea)
from qtpy.QtCore import Qt, QRect
from qtpy.QtGui import QPixmap, QColor, QPainter
from superqt import QLabeledDoubleRangeSlider, QDoubleSlider

from napari.utils import progress
import pandas as pd
from microfilm import colorify
from cmap import Colormap
from matplotlib_scalebar.scalebar import ScaleBar
from napari_matplotlib.base import NapariMPLWidget

from .parameters.parameters import Param
from .parameters.parameters_endmembers import ParamEndMember
from .io import load_project_params, load_endmember_params, load_plots_params
from .imchannels import ImChannels
from .sediproc import find_index_of_band
from .spectralplot import SpectralPlotter, plot_spectral_profile, plot_multi_spectral_profile
from .widgets.channel_widget import ChannelWidget
from .widgets.rgb_widget import RGBWidget
from .parameters.parameters_plots import Paramplot
from .spectralindex import (SpectralIndex, compute_index_RABD, compute_index_RABA,
                            compute_index_ratio, compute_index_projection,
                            clean_index_map, save_tif_cmap)
from .io import load_mask, get_mask_path
from .utils import wavelength_to_rgb


from napari_guitils.gui_structures import TabSet, VHGroup


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
    params_indices: ParamEndMember
        parameters for end members
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

        self.params = Param()
        self.params_indices = ParamEndMember()
        self.params_plots = Paramplot()
        self.params_multiplots = Paramplot()

        self.em_boundary_lines = None
        self.end_members = None
        self.endmember_bands = None
        self.index_file = None
        self.current_plot_type = 'single'

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ["&Main", "&Index Definition", "Index C&ompute", "P&lots"]#, "Plotslive"]
        self.tabs = TabSet(self.tab_names, tab_layouts=[None, QGridLayout(), None, QGridLayout()])

        self.main_layout.addWidget(self.tabs)

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
        
        self.band_group = VHGroup('Bands', orientation='G')
        self.tabs.add_named_tab('&Main', self.band_group.gbox)
        self.band_group.glayout.addWidget(QLabel('Bands to load'), 0, 0, 1, 2)
        self.qlist_channels = ChannelWidget(self.viewer)
        self.band_group.glayout.addWidget(self.qlist_channels, 1,0,1,2)
        self.qlist_channels.itemClicked.connect(self._on_change_select_bands)

        self.rgbwidget = RGBWidget(viewer=self.viewer, translate=False)
        self.tabs.add_named_tab('&Main', self.rgbwidget.rgbmain_group.gbox)

        # indices tab
        self._create_indices_tab()
        tab_rows = self.tabs.widget(1).layout().rowCount()
        self.em_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.tabs.add_named_tab('&Index Definition', self.em_plot, grid_pos=(tab_rows, 0, 1, 3))
        self.em_boundaries_range = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.em_boundaries_range.setValue((0, 0, 0))
        self.em_boundaries_range2 = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.em_boundaries_range2.setValue((0, 0))
        self.tabs.add_named_tab('&Index Definition', QLabel('RABD'), grid_pos=(tab_rows+1, 0, 1, 1))
        self.tabs.add_named_tab('&Index Definition', self.em_boundaries_range, grid_pos=(tab_rows+1, 1, 1, 2))
        self.tabs.add_named_tab('&Index Definition', QLabel('RABA/Ratio'), grid_pos=(tab_rows+2, 0, 1, 1))
        self.tabs.add_named_tab('&Index Definition', self.em_boundaries_range2, grid_pos=(tab_rows+2, 1, 1, 2))
        self.btn_create_index = QPushButton("New index")
        self.tabs.add_named_tab('&Index Definition', self.btn_create_index, grid_pos=(tab_rows+3, 0, 1, 1))
        self.qtext_new_index_name = QLineEdit()
        self.tabs.add_named_tab('&Index Definition', self.qtext_new_index_name, grid_pos=(tab_rows+3, 1, 1, 2))
        self.btn_update_index = QPushButton("Update current index")
        self.tabs.add_named_tab('&Index Definition', self.btn_update_index, grid_pos=(tab_rows+4, 0, 1, 1))

        self.btn_save_endmembers_plot = QPushButton("Save endmembers plot")
        self.tabs.add_named_tab('&Index Definition', self.btn_save_endmembers_plot, grid_pos=(tab_rows+5, 0, 1, 3))

        # Index C&ompute tab
        self.index_pick_group = VHGroup('Index Selection', orientation='G')
        self.index_pick_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Index C&ompute', self.index_pick_group.gbox)
        self._create_index_io_pick()

        self.index_options_group = VHGroup('Projection', orientation='G')
        self.index_options_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Index C&ompute', self.index_options_group.gbox)
        self.check_smooth_projection = QCheckBox("Smooth projection")
        self.index_options_group.glayout.addWidget(self.check_smooth_projection, 0, 0, 1, 2)
        self.slider_index_savgol = QDoubleSlider(Qt.Horizontal)
        self.slider_index_savgol.setRange(1, 100)
        self.slider_index_savgol.setSingleStep(1)
        self.slider_index_savgol.setSliderPosition(5)
        self.index_options_group.glayout.addWidget(QLabel('Smoothing window size'), 1, 0, 1, 1)
        self.index_options_group.glayout.addWidget(self.slider_index_savgol, 1, 1, 1, 1)
        self.spin_roi_width = QSpinBox()
        self.spin_roi_width.setRange(1, 1000)
        self.spin_roi_width.setValue(20)
        self.index_options_group.glayout.addWidget(QLabel('Projection roi width'), 2, 0, 1, 1)
        self.index_options_group.glayout.addWidget(self.spin_roi_width, 2, 1, 1, 1)

        self.index_compute_group = VHGroup('Compute and export', orientation='G')
        self.index_compute_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Index C&ompute', self.index_compute_group.gbox)
        self.btn_compute_index_maps = QPushButton("(Re-)Compute index map(s)")
        self.index_compute_group.glayout.addWidget(self.btn_compute_index_maps)
        self.btn_add_index_maps_to_viewer = QPushButton("Add index map(s) to Viewer")
        self.index_compute_group.glayout.addWidget(self.btn_add_index_maps_to_viewer)

        self.btn_export_index_tiff = QPushButton("Export index map(s) to tiff")
        self.index_compute_group.glayout.addWidget(self.btn_export_index_tiff)
        self.btn_export_indices_csv = QPushButton("Export index projections to csv")
        self.index_compute_group.glayout.addWidget(self.btn_export_indices_csv)
        self.btn_export_index_settings = QPushButton("Export index settings")
        self.index_compute_group.glayout.addWidget(self.btn_export_index_settings)
        self.btn_import_index_settings = QPushButton("Import index settings")
        self.index_compute_group.glayout.addWidget(self.btn_import_index_settings)
        self.index_file_display = QLineEdit("No file selected")
        self.index_compute_group.glayout.addWidget(self.index_file_display)


        #self.index_plot = SpectralPlotter(napari_viewer=self.viewer)
        #self.tabs.add_named_tab('P&lots', self.index_plot)

        self.pixlabel = QLabel()
        #self.pixlabel = ScaledPixmapLabel()#QLabel()
        '''self.pixlabel.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        self.pixlabel.setScaledContents(True)
        self.pixlabel.resizeEvent = self._on_resize_pixlabel'''

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
        self.spin_final_dpi.setValue(100)
        self.spin_final_dpi.setSingleStep(1)

        self.scale = 1.0
        self.pix_width = None
        self.pix_height = None

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.pixlabel)

        #self.index_plot_live = SpectralPlotter(napari_viewer=self.viewer)
        self.index_plot_live = NapariMPLWidget(napari_viewer=self.viewer)
        self.index_plot_live.figure.set_layout_engine('none')
        #self.tabs.add_named_tab('Plotslive', self.index_plot_live)

        #self.tabs.add_named_tab('P&lots', self.scrollArea, grid_pos=(0, 0, 1, 2))
        self.scrollArea.setWidgetResizable(True)
        #self.scrollArea.setFixedHeight(200)

        self.tabs.add_named_tab('P&lots', self.button_zoom_in, grid_pos=(14, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.button_zoom_out, grid_pos=(14, 1, 1, 1))
        self.tabs.add_named_tab('P&lots', QLabel('Preview DPI'), grid_pos=(15, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_preview_dpi, grid_pos=(15, 1, 1, 1))
        self.tabs.add_named_tab('P&lots', QLabel('Final DPI'), grid_pos=(16, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_final_dpi, grid_pos=(16, 1, 1, 1))
        

        self.btn_create_index_plot = QPushButton("Create index plot")
        self.btn_create_multi_index_plot = QPushButton("Create multi-index plot")
        self.tabs.add_named_tab('P&lots', self.btn_create_index_plot, grid_pos=(1, 0, 1, 2))
        self.tabs.add_named_tab('P&lots', self.btn_create_multi_index_plot, grid_pos=(1, 2, 1, 2))
        self.spin_left_right_margin_fraction = QDoubleSpinBox()
        self.spin_left_right_margin_fraction.setRange(0, 100)
        self.spin_left_right_margin_fraction.setValue(0.4)
        self.spin_left_right_margin_fraction.setSingleStep(0.1)
        self.tabs.add_named_tab('P&lots', QLabel('L/R Margin'), grid_pos=(2, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_left_right_margin_fraction, grid_pos=(2, 1, 1, 1))
        self.spin_bottom_top_margin_fraction = QDoubleSpinBox()
        self.spin_bottom_top_margin_fraction.setRange(0, 100)
        self.spin_bottom_top_margin_fraction.setValue(0.1)
        self.spin_bottom_top_margin_fraction.setSingleStep(0.01)
        self.tabs.add_named_tab('P&lots', QLabel('B/T Margin'), grid_pos=(2, 2, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_bottom_top_margin_fraction, grid_pos=(2, 3, 1, 1))
        
        self.spin_figure_size_factor = QDoubleSpinBox()
        self.spin_figure_size_factor.setRange(1, 100)
        self.spin_figure_size_factor.setValue(1)
        self.spin_figure_size_factor.setSingleStep(1)
        self.tabs.add_named_tab('P&lots', QLabel('Figure size factor'), grid_pos=(3, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_figure_size_factor, grid_pos=(3, 1, 1, 1))

        self.spin_plot_image_w_fraction = QDoubleSpinBox()
        self.spin_plot_image_w_fraction.setRange(0, 100)
        self.spin_plot_image_w_fraction.setValue(0.25)
        self.spin_plot_image_w_fraction.setSingleStep(0.1)
        self.tabs.add_named_tab('P&lots', QLabel('Plot width fraction'), grid_pos=(3, 2, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_plot_image_w_fraction, grid_pos=(3, 3, 1, 1))
        self.spin_title_font_factor = QDoubleSpinBox()
        self.spin_label_font_factor = QDoubleSpinBox()
        for sbox in [self.spin_label_font_factor, self.spin_title_font_factor]:
            sbox.setRange(0, 100)
            sbox.setValue(12)
            sbox.setSingleStep(1)
        self.tabs.add_named_tab('P&lots', QLabel('Title Font'), grid_pos=(4, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_title_font_factor, grid_pos=(4, 1, 1, 1))
        self.tabs.add_named_tab('P&lots', QLabel('Label Font'), grid_pos=(4, 2, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_label_font_factor, grid_pos=(4, 3, 1, 1))
        self.qcolor_plotline = QColorDialog()
        self.btn_qcolor_plotline = QPushButton("Select plot line color")
        self.tabs.add_named_tab('P&lots', self.btn_qcolor_plotline, grid_pos=(5, 0, 1, 2))
        self.qcolor_plotline.setCurrentColor(Qt.blue)
        self.spin_plot_thickness = QDoubleSpinBox()
        self.spin_plot_thickness.setRange(1, 10)
        self.spin_plot_thickness.setValue(1)
        self.spin_plot_thickness.setSingleStep(0.1)
        self.tabs.add_named_tab('P&lots', QLabel('Plot line thickness'), grid_pos=(5, 2, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_plot_thickness, grid_pos=(5, 3, 1, 1))
        #self.btn_reset_figure_size = QPushButton("Reset figure size")
        #self.tabs.add_named_tab('P&lots', self.btn_reset_figure_size, grid_pos=(9, 0, 1, 2))

        self.metadata_location = QLineEdit("No location")
        self.metadata_location.setToolTip("Indicate the location of data acquisition")
        self.tabs.add_named_tab('P&lots', QLabel('Location'), grid_pos=(6, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.metadata_location, grid_pos=(6, 1, 1, 3))
        self.spinbox_metadata_scale = QDoubleSpinBox()
        self.spinbox_metadata_scale.setToolTip("Indicate conversion factor from pixel to mm")
        self.spinbox_metadata_scale.setRange(0, 1000)
        self.spinbox_metadata_scale.setSingleStep(0.001)
        self.spinbox_metadata_scale.setValue(1)
        self.tabs.add_named_tab('P&lots', QLabel('Scale'), grid_pos=(7, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spinbox_metadata_scale, grid_pos=(7, 1, 1, 1))

        self.btn_save_plot = QPushButton("Save plot")
        self.tabs.add_named_tab('P&lots', self.btn_save_plot, grid_pos=(8, 0, 1, 2))
        self.btn_save_plot_params = QPushButton("Save plot parameters")
        self.tabs.add_named_tab('P&lots', self.btn_save_plot_params, grid_pos=(9, 0, 1, 2))
        self.btn_load_plot_params = QPushButton("Load plot parameters")
        self.tabs.add_named_tab('P&lots', self.btn_load_plot_params, grid_pos=(10, 0, 1, 2))

        
        self._connect_spin_bounds()
        self.add_connections()

    def _create_indices_tab(self):

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
        """Create a spin box with a range of minval to maxval"""
        
        spin = QSpinBox()
        spin.setRange(minval, maxval)
        return spin
          
    def create_index_list(self):

        index_def = {
            'RABD510': [470, 510, 530],
            'RABD660670': [590, 665, 730],
        }
        self.index_collection = {}
        for key, value in index_def.items():
            self.index_collection[key] = SpectralIndex(index_name=key,
                              index_type='RABD',
                              left_band_default=value[0],
                              middle_band_default=value[1],
                              right_band_default=value[2]
                              )
            
        index_def = {
            'RABA410560': [410, 560],
        }
        for key, value in index_def.items():
            self.index_collection[key] = SpectralIndex(index_name=key,
                              index_type='RABA',
                              left_band_default=value[0],
                              right_band_default=value[1]
                              )
            
        index_def = {
            'R590R690': [590, 690],
            'R660R670': [660, 670]
        }
        for key, value in index_def.items():
            self.index_collection[key] = SpectralIndex(index_name=key,
                              index_type='Ratio',
                              left_band_default=value[0],
                              right_band_default=value[1]
            )

    def add_connections(self):
        """Add callbacks"""

        self.btn_select_export_folder.clicked.connect(self._on_click_select_export_folder)
        self.btn_load_project.clicked.connect(self.import_project)
        self.spin_selected_roi.valueChanged.connect(self.load_data)
        self.em_boundaries_range.valueChanged.connect(self._on_change_em_boundaries)
        self.em_boundaries_range2.valueChanged.connect(self._on_change_em_boundaries)
        self.btn_compute_index_maps.clicked.connect(self._on_compute_index_maps)
        self.btn_add_index_maps_to_viewer.clicked.connect(self._on_add_index_map_to_viewer)
        self.btn_save_endmembers_plot.clicked.connect(self.save_endmembers_plot)
        self.btn_create_index.clicked.connect(self._on_click_new_index)
        self.btn_update_index.clicked.connect(self._on_click_update_index)
        self.qcom_indices.activated.connect(self._on_change_index_index)
        self.btn_export_index_tiff.clicked.connect(self._on_click_export_index_tiff)
        self.btn_export_index_settings.clicked.connect(self._on_click_export_index_settings)
        self.btn_import_index_settings.clicked.connect(self._on_click_import_index_settings)
        self.btn_create_index_plot.clicked.connect(self._on_click_create_single_index_plot)
        self.btn_create_multi_index_plot.clicked.connect(self._on_click_create_multi_index_plot)
        self.btn_export_indices_csv.clicked.connect(self._on_export_index_projection)

        self.connect_plot_formatting()
        self.btn_qcolor_plotline.clicked.connect(self._on_click_open_plotline_color_dialog)
        self.btn_save_plot.clicked.connect(self._on_click_save_plot)
        #self.btn_reset_figure_size.clicked.connect(self._on_click_reset_figure_size)
        self.btn_save_plot_params.clicked.connect(self._on_click_save_plot_parameters)
        self.btn_load_plot_params.clicked.connect(self._on_click_load_plot_parameters)

        self.viewer.mouse_double_click_callbacks.append(self._add_analysis_roi)

    def _connect_spin_bounds(self):

        self.spin_index_left.valueChanged.connect(self._on_change_spin_bounds)
        self.spin_index_middle.valueChanged.connect(self._on_change_spin_bounds)
        self.spin_index_right.valueChanged.connect(self._on_change_spin_bounds)

    def _disconnect_spin_bounds(self):
            
        self.spin_index_left.valueChanged.disconnect(self._on_change_spin_bounds)
        self.spin_index_middle.valueChanged.disconnect(self._on_change_spin_bounds)
        self.spin_index_right.valueChanged.disconnect(self._on_change_spin_bounds)

    def _on_change_spin_bounds(self, event=None):

        if self.current_index_type == 'RABD':
            self.em_boundaries_range.setValue(
                (self.spin_index_left.value(), self.spin_index_middle.value(),
                self.spin_index_right.value()))
        else:
            self.em_boundaries_range2.setValue(
                (self.spin_index_left.value(), self.spin_index_right.value()))
    
    def _on_click_select_export_folder(self, event=None, export_folder=None):
        """Interactively select folder to analyze"""

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
        """Import pre-processed project: corrected roi and mask"""
        
        if self.export_folder is None:
            self._on_click_select_export_folder()

        export_path_roi = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')

        self.params = load_project_params(folder=self.export_folder)
        self.metadata_location.setText(self.params.location)
        self.spinbox_metadata_scale.setValue(self.params.scale)
        self.params_indices = load_endmember_params(folder=export_path_roi)

        self.imhdr_path = Path(self.params.file_path)

        self.mainroi = np.array([np.array(x).reshape(4,2) for x in self.params.main_roi]).astype(int)
        
        self.spin_selected_roi.setRange(0, len(self.mainroi)-1)
        self.spin_selected_roi.setValue(0)
        
        self.load_data()
        
    def load_data(self, event=None):

        to_remove = [l.name for l in self.viewer.layers if l.name not in ['imcube', 'red', 'green', 'blue']]
        for r in to_remove:
            self.viewer.layers.remove(r)

        self.var_init()

        self.row_bounds = [
            self.mainroi[self.spin_selected_roi.value()][:,0].min(),
            self.mainroi[self.spin_selected_roi.value()][:,0].max()]
        self.col_bounds = [
            self.mainroi[self.spin_selected_roi.value()][:,1].min(),
            self.mainroi[self.spin_selected_roi.value()][:,1].max()]

        export_path_roi = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')

        self.imagechannels = ImChannels(self.export_folder.joinpath('corrected.zarr'))
        self.qlist_channels._update_channel_list(imagechannels=self.imagechannels)
        self.rgbwidget.imagechannels = self.imagechannels

        self.get_RGB()
        self.rgbwidget.load_and_display_rgb_bands(roi=np.concatenate([self.row_bounds, self.col_bounds]))

        self._on_click_load_mask()

        self.end_members = pd.read_csv(export_path_roi.joinpath('end_members.csv')).values
        self.endmember_bands = self.end_members[:,-1]
        self.end_members = self.end_members[:,:-1]

        self.em_boundaries_range.setRange(min=self.endmember_bands[0],max=self.endmember_bands[-1])
        self.em_boundaries_range.setValue(
            (self.endmember_bands[0], (self.endmember_bands[-1]+self.endmember_bands[0])/2, self.endmember_bands[-1]))
        self.em_boundaries_range2.setRange(min=self.endmember_bands[0],max=self.endmember_bands[-1])
        self.em_boundaries_range2.setValue(
            (self.endmember_bands[0], self.endmember_bands[-1]))
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
        
        edge_width = np.min([10, self.viewer.layers['imcube'].data.shape[1]//100])
        if edge_width < 1:
            edge_width = 1
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

        
        if 'rois' not in self.viewer.layers:
            self.viewer.add_shapes(
                ndim = 2,
                name='rois', edge_color='red', face_color=np.array([0,0,0,0]), edge_width=edge_width)
         
        self.viewer.layers['rois'].add_rectangles(new_roi, edge_color='red', edge_width=edge_width)


    def get_RGB(self):
        
        rgb_ch, rgb_names = self.imagechannels.get_indices_of_bands(self.rgbwidget.rgb)
        [self.qlist_channels.item(x).setSelected(True) for x in rgb_ch]
        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)

    def _on_change_select_bands(self, event=None):

        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)

    def _on_click_load_mask(self):
        """Load mask from file"""
        
        export_path_roi = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')
        mask = load_mask(get_mask_path(export_path_roi))#[self.row_bounds[0]:self.row_bounds[1], self.col_bounds[0]:self.col_bounds[1]]
        if 'mask' in self.viewer.layers:
            self.viewer.layers['mask'].data = mask
        else:
            self.viewer.add_labels(mask, name='mask')

    def plot_endmembers(self, event=None):
        """Cluster the pure pixels and plot the endmembers as average of clusters."""

        self.em_plot.axes.clear()
        self.em_plot.axes.plot(self.endmember_bands, self.end_members)

        out = wavelength_to_rgb(self.endmember_bands.min(), self.endmember_bands.max(), 100)
        ax_histx = self.em_plot.axes.inset_axes([0.0,-0.5, 1.0, 1], sharex=self.em_plot.axes)
        ax_histx.imshow(out, extent=(self.endmember_bands.min(),self.endmember_bands.max(), 0,10))
        ax_histx.set_axis_off()

        self.em_plot.axes.set_xlabel('Wavelength', color='white')
        self.em_plot.axes.set_ylabel('Continuum removed', color='white')
        self.em_plot.axes.xaxis.label.set_color('black')
        self.em_plot.axes.yaxis.label.set_color('black')
        self.em_plot.axes.tick_params(axis='both', colors='black')
        self.em_plot.figure.patch.set_facecolor('white')
        self.em_plot.figure.canvas.draw()

    def save_endmembers_plot(self):
        """Save the endmembers plot to file"""

        export_folder = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')

        if self.em_boundary_lines is not None:
            num_lines = len(self.em_boundary_lines)
            for i in range(num_lines):
                self.em_boundary_lines[i].set_color([0,0,0,0])
        self.em_plot.figure.canvas.draw()
        self.em_plot.figure.savefig(export_folder.joinpath('endmembers.png'), dpi=300)
        if self.em_boundary_lines is not None:
            for i in range(num_lines):
                self.em_boundary_lines[i].set_color([1,0,0])
        self.em_plot.figure.canvas.draw()

    def _on_change_em_boundaries(self, event=None):
        """Update the em plot when the em boundaries are changed."""
        
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
            self.em_plot.figure.canvas.draw()
        
        #self._connect_spin_bounds()

    def _update_save_plot_parameters(self):

        if self.current_plot_type == 'single':
            current_param = self.params_plots
        else:
            current_param = self.params_multiplots

        current_param.color_plotline = [self.qcolor_plotline.currentColor().getRgb()[x]/255 for x in range(3)]
        current_param.plot_thickness = self.spin_plot_thickness.value()
        current_param.title_font_factor = self.spin_title_font_factor.value()
        current_param.label_font_factor = self.spin_label_font_factor.value()
        current_param.left_right_margin_fraction = self.spin_left_right_margin_fraction.value()
        current_param.bottom_top_margin_fraction = self.spin_bottom_top_margin_fraction.value()
        current_param.plot_image_w_fraction = self.spin_plot_image_w_fraction.value()
        current_param.figure_size_factor = self.spin_figure_size_factor.value()
        for key in self.index_collection:
            if key in self.viewer.layers:
                current_param.index_colormap[key] = self.viewer.layers[key].colormap.name
        current_param.red_contrast_limits = np.array(self.viewer.layers['red'].contrast_limits).tolist()
        current_param.green_contrast_limits = np.array(self.viewer.layers['green'].contrast_limits).tolist()
        current_param.blue_contrast_limits = np.array(self.viewer.layers['blue'].contrast_limits).tolist()
        current_param.rgb_bands = self.rgbwidget.rgb

    def _on_click_save_plot_parameters(self, event=None, file_path=None):
            
        if file_path is None:
            file_path = Path(str(QFileDialog.getSaveFileName(self, "Select plot parameters file")[0]))
        self._update_save_plot_parameters()
        self.params_plots.save_parameters(file_path)

    def _on_click_load_plot_parameters(self, event=None, file_path=None):
        
        try:
            self.disconnect_plot_formatting()
        except:
            pass
        if file_path is None:
            file_path = Path(str(QFileDialog.getOpenFileName(self, "Select plot parameters file")[0]))
        self.params_plots = load_plots_params(file_path=file_path)
        self.set_plot_interface(params=self.params_plots)
        self.rgbwidget._on_click_RGB()
        self.connect_plot_formatting()
        self.create_index_plot()

    def set_plot_interface(self, params):
        self.spin_plot_thickness.setValue(params.plot_thickness)
        self.spin_title_font_factor.setValue(params.title_font_factor)
        self.spin_label_font_factor.setValue(params.label_font_factor)
        self.spin_left_right_margin_fraction.setValue(params.left_right_margin_fraction)
        self.spin_bottom_top_margin_fraction.setValue(params.bottom_top_margin_fraction)
        self.spin_plot_image_w_fraction.setValue(params.plot_image_w_fraction)
        self.spin_figure_size_factor.setValue(params.figure_size_factor)
        self.qcolor_plotline.setCurrentColor(QColor(*[int(x*255) for x in params.color_plotline]))
        for key in params.index_colormap:
            if key in self.viewer.layers:
                self.viewer.layers[key].colormap = params.index_colormap[key]
        self.viewer.layers['red'].contrast_limits = params.red_contrast_limits
        self.viewer.layers['green'].contrast_limits = params.green_contrast_limits
        self.viewer.layers['blue'].contrast_limits = params.blue_contrast_limits
        self.rgbwidget.rgb = params.rgb_bands
        

    def disconnect_plot_formatting(self):
        """Disconnect plot editing widgets while loading parameters to avoid overwriting
        the loaded parameters."""
        
        self.spin_plot_thickness.valueChanged.disconnect(self.create_index_plot)
        self.spin_title_font_factor.valueChanged.disconnect(self.create_index_plot)
        self.spin_label_font_factor.valueChanged.disconnect(self.create_index_plot)
        self.spin_left_right_margin_fraction.valueChanged.disconnect(self.create_index_plot)
        self.spin_bottom_top_margin_fraction.valueChanged.disconnect(self.create_index_plot)
        self.spin_plot_image_w_fraction.valueChanged.disconnect(self.create_index_plot)
        self.spin_figure_size_factor.valueChanged.disconnect(self.create_index_plot)
        self.qcolor_plotline.currentColorChanged.disconnect(self.create_index_plot)

    def connect_plot_formatting(self):
        """Reconnect plot editing widgets after loading parameters."""

        self.spin_plot_thickness.valueChanged.connect(self.create_index_plot)
        self.spin_title_font_factor.valueChanged.connect(self.create_index_plot)
        self.spin_label_font_factor.valueChanged.connect(self.create_index_plot)
        self.spin_left_right_margin_fraction.valueChanged.connect(self.create_index_plot)
        self.spin_bottom_top_margin_fraction.valueChanged.connect(self.create_index_plot)
        self.spin_plot_image_w_fraction.valueChanged.connect(self.create_index_plot)
        self.spin_figure_size_factor.valueChanged.connect(self.create_index_plot)
        self.qcolor_plotline.currentColorChanged.connect(self.create_index_plot)

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
    
    def _on_click_create_single_index_plot(self, event=None):
        self.current_plot_type = 'single'
        self.disconnect_plot_formatting()
        self.set_plot_interface(params=self.params_plots)
        self.create_single_index_plot(event=event)
        self.connect_plot_formatting()

    def _on_click_create_multi_index_plot(self, event=None):
        self.current_plot_type = 'multi'
        self.disconnect_plot_formatting()
        self.set_plot_interface(params=self.params_multiplots)
        self.create_multi_index_plot(event=event)
        self.connect_plot_formatting()
    
    def create_index_plot(self, event=None):
        if self.current_plot_type == 'single':
            self.create_single_index_plot(event=event)
        else:
            self.create_multi_index_plot(event=event)

    def get_roi_bounds(self):

        if 'rois' not in self.viewer.layers:
            return self.col_bounds[0], self.col_bounds[1]
        elif len(self.viewer.layers['rois'].data) == 0:
            return self.col_bounds[0], self.col_bounds[1]
        else:
            colmin = int(self.viewer.layers['rois'].data[0][:,1].min())
            colmax = int(self.viewer.layers['rois'].data[0][:,1].max())

        return colmin, colmax

    def create_single_index_plot(self, event=None):
        """Create the index plot."""

        self._update_save_plot_parameters()
        self.params.location = self.metadata_location.text()
        self.params.scale = self.spinbox_metadata_scale.value()

        # get rgb image and index image to plot
        rgb_image = [self.viewer.layers[c].data for c in ['red', 'green', 'blue']]
        if isinstance(rgb_image[0], da.Array):
            rgb_image = [x.compute() for x in rgb_image]

        index_series = [x for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
        if len(index_series) == 0:
            warnings.warn('No index selected') 
            return
        elif len(index_series) > 1:
            warnings.warn('Multiple indices selected, only the first one will be plotted')

        mask = self.viewer.layers['mask'].data
        colmin, colmax = self.get_roi_bounds()

        if self.index_collection[index_series[0].index_name].index_map is None:
            computed_index = self.compute_index(self.index_collection[index_series[0].index_name])
            computed_index = clean_index_map(computed_index)
            proj = compute_index_projection(
                computed_index, mask,
                colmin=colmin, colmax=colmax,
                smooth_window=self.get_smoothing_window())
            self.index_collection[index_series[0].index_name].index_map = computed_index
            self.index_collection[index_series[0].index_name].index_proj = proj
        else:
            computed_index = self.index_collection[index_series[0].index_name].index_map
            proj = self.index_collection[index_series[0].index_name].index_proj   

        roi = None
        if 'rois' in self.viewer.layers:
            roi=self.viewer.layers['rois'].data[0]

        format_dict = asdict(self.params_plots)
        _, self.ax1, self.ax2, self.ax3 = plot_spectral_profile(
            rgb_image=rgb_image, mask=mask, index_obj=self.index_collection[index_series[0].index_name],
            format_dict=format_dict, scale=self.params.scale,
            location=self.params.location, fig=self.index_plot_live.figure, 
            roi=roi)

        # save temporary low-res figure for display in napari
        self.index_plot_live.figure.savefig(
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
        '''if self.pixlabel.size().height() < self.pixlabel.size().width():
            self.pixlabel.setPixmap(self.pixmap.scaledToWidth(self.pixlabel.size().width()))
        else:
            self.pixlabel.setPixmap(self.pixmap.scaledToHeight(self.pixlabel.size().height()))'''

        #self.index_plot_live.figure.canvas.draw()
        #self.index_plot_live.figure.canvas.flush_events()

        #vsize = self.viewer.window.geometry()
        #self.viewer.window.resize(vsize[2]-10,vsize[3]-10)
        #self.viewer.window.resize(vsize[2],vsize[3])

    def create_multi_index_plot(self, event=None):
        
        self._update_save_plot_parameters()
        self.params.location = self.metadata_location.text()
        self.params.scale = self.spinbox_metadata_scale.value()
        # get rgb image and index image to plot
        rgb_image = [self.viewer.layers[c].data for c in ['red', 'green', 'blue']]
        if isinstance(rgb_image[0], da.Array):
            rgb_image = [x.compute() for x in rgb_image]

        colmin, colmax = self.get_roi_bounds()
        
        index_series = [x for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
        for i in index_series:
            if self.index_collection[i.index_name].index_map is None:
                computed_index = self.compute_index(self.index_collection[i.index_name])
                computed_index = clean_index_map(computed_index)
                proj = compute_index_projection(
                    computed_index, self.viewer.layers['mask'].data,
                    colmin=colmin, colmax=colmax,
                    smooth_window=self.get_smoothing_window())
                self.index_collection[i.index_name].index_map = computed_index
                self.index_collection[i.index_name].index_proj = proj
            else:
                computed_index = self.index_collection[i.index_name].index_map
                proj = self.index_collection[i.index_name].index_proj   
        
        roi = None
        if 'rois' in self.viewer.layers:
            roi=self.viewer.layers['rois'].data[0]

        format_dict = asdict(self.params_multiplots)
        plot_multi_spectral_profile(
            rgb_image=rgb_image, mask=self.viewer.layers['mask'].data,
            index_objs=index_series, 
            format_dict=format_dict, scale=self.params.scale,
            fig=self.index_plot_live.figure,
            roi=roi)
        
        # save temporary low-res figure for display in napari
        self.index_plot_live.figure.savefig(
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


    def on_close_callback(self):
        print('Viewer closed')

    def _on_resize_pixlabel(self, event=None):

        self.pixlabel.setPixmap(self.pixmap.scaled(self.pixlabel.size().width(), self.pixlabel.size().height(), Qt.KeepAspectRatio))


    def _on_click_reset_figure_size(self, event=None):
        """Reset figure size to default"""

        self.index_plot_live.figure.set_size_inches(self.fig_size)
        self.index_plot_live.figure.canvas.draw()
        self.index_plot_live.figure.canvas.flush_events()

        vsize = self.viewer.window.geometry()
        self.viewer.window.resize(vsize[2]-10,vsize[3]-10)
        self.viewer.window.resize(vsize[2],vsize[3])

    def on_zoom_in(self, event):
        self.scale *= 2
        self.resize_image()

    def on_zoom_out(self, event):
        self.scale /= 2
        self.resize_image()

    def resize_image(self):
        size = self.pixmap.size()

        scaled_pixmap = self.pixmap.scaled(self.scale * size)

        self.pixlabel.setPixmap(scaled_pixmap)

    def _on_click_save_plot(self, event=None, export_file=None):
        
        export_folder = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')
        if export_file is None:
            export_file = export_folder.joinpath(self.qcom_indices.currentText()+'_index_plot.png')
        self.index_plot_live.figure.savefig(
            fname=export_file, dpi=self.spin_final_dpi.value())#, bbox_inches="tight")
        self._on_export_index_projection()

    def _on_export_index_projection(self, event=None):

        colmin, colmax = self.get_roi_bounds()
        export_folder = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')
        index_series = [x for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
        proj_pd = None
        for i in index_series:
            if self.index_collection[i.index_name].index_map is None:
                computed_index = self.compute_index(self.index_collection[i.index_name])
                computed_index = clean_index_map(computed_index)
                proj = compute_index_projection(
                    computed_index,
                    self.viewer.layers['mask'].data,
                    colmin=colmin, colmax=colmax,
                    smooth_window=self.get_smoothing_window())
                self.index_collection[i.index_name].index_map = computed_index
                self.index_collection[i.index_name].index_proj = proj
            else:
                computed_index = self.index_collection[i.index_name].index_map
                proj = self.index_collection[i.index_name].index_proj
            
            if proj_pd is None:
                proj_pd = pd.DataFrame({'depth': np.arange(0,len(proj))})
            proj_pd[i.index_name] = proj

        proj_pd.to_csv(export_folder.joinpath('index_projection.csv'), index=False)

    def _on_click_open_plotline_color_dialog(self, event=None):
        """Show label color dialog"""
        
        self.qcolor_plotline.show()

    def _on_adjust_font_size(self, event=None):
        ## Not used. If used in future, scaling needs to be fixed
        im_h = self.viewer.layers['imcube'].data.shape[-2]
        font_factor = self.spin_title_font_factor.value()
        for label in (self.ax1.get_yticklabels() + 
                      self.ax3.get_yticklabels() + 
                      self.ax3.get_xticklabels()):
            label.set_fontsize(int(font_factor*im_h))

        self.index_plot_live.figure.suptitle(self.qcom_indices.currentText() + '\n' + self.params.location,
                     fontsize=int(font_factor*im_h))

        self.index_plot_live.canvas.draw()

    def _on_click_new_index(self, event):
        """Add new custom index"""

        name = self.qtext_new_index_name.text()
        if self.current_index_type == 'RABD':
            current_bands = np.array(self.em_boundaries_range.value(), dtype=np.uint16)
            self.index_collection[name] = SpectralIndex(index_name=name,
                              index_type='RABD',
                              left_band_default=current_bands[0],
                              middle_band_default=current_bands[1],
                              right_band_default=current_bands[2]
                              )
            
        elif self.current_index_type == 'RABA':
            current_bands = np.array(self.em_boundaries_range2.value(), dtype=np.uint16)
            self.index_collection[name] = SpectralIndex(index_name=name,
                              index_type='RABA',
                              left_band_default=current_bands[0],
                              right_band_default=current_bands[1],
                              )
        elif self.current_index_type == 'ratio':
            current_bands = np.array(self.em_boundaries_range2.value(), dtype=np.uint16)
            self.index_collection[name] = SpectralIndex(index_name=name,
                              index_type='ratio',
                              left_band_default=current_bands[0],
                              right_band_default=current_bands[1],
                              )
        
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
        """Update the current index."""

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

    def compute_index(self, spectral_index):
        """Compute the index and add to napari."""

        if spectral_index.index_type == 'RABD':
            computed_index = compute_index_RABD(
                left=spectral_index.left_band,
                trough=spectral_index.middle_band,
                right=spectral_index.right_band,
                row_bounds=self.row_bounds,
                col_bounds=self.col_bounds,
                imagechannels=self.imagechannels)
        elif spectral_index.index_type == 'RABA':
            computed_index = compute_index_RABA(
                left=spectral_index.left_band,
                right=spectral_index.right_band,
                row_bounds=self.row_bounds,
                col_bounds=self.col_bounds,
                imagechannels=self.imagechannels)
        elif spectral_index.index_type == 'Ratio':
            computed_index = compute_index_ratio(
                left=spectral_index.left_band,
                right=spectral_index.right_band,
                row_bounds=self.row_bounds,
                col_bounds=self.col_bounds,
                imagechannels=self.imagechannels)
        else:
            print(f'unknown index type: {spectral_index.index_type}')
            return None
        return computed_index
    
    def _on_compute_index_maps(self, event):
        """Compute the index and add to napari."""

        colmin, colmax = self.get_roi_bounds()
        index_series = [x for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
        for i in index_series:
            computed_index = self.compute_index(self.index_collection[i.index_name])
            computed_index = clean_index_map(computed_index)
            proj = compute_index_projection(
                computed_index,
                self.viewer.layers['mask'].data,
                colmin=colmin, colmax=colmax,
                smooth_window=self.get_smoothing_window())
            self.index_collection[i.index_name].index_map = computed_index
            self.index_collection[i.index_name].index_proj = proj
        self._on_add_index_map_to_viewer()

    def _on_add_index_map_to_viewer(self, event=None):
        """Compute the index and add to napari."""

        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Computing index")

            colmin, colmax = self.get_roi_bounds()
            index_series = [x for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
            for i in index_series:
                if self.index_collection[i.index_name].index_map is None:
                    computed_index = self.compute_index(self.index_collection[i.index_name])
                    computed_index = clean_index_map(computed_index)
                    proj = compute_index_projection(
                        computed_index,
                        self.viewer.layers['mask'].data,
                        colmin=colmin, colmax=colmax,
                        smooth_window=self.get_smoothing_window())
                    self.index_collection[i.index_name].index_map = computed_index
                    self.index_collection[i.index_name].index_proj = proj
                else:
                    computed_index = self.index_collection[i.index_name].index_map
                    proj = self.index_collection[i.index_name].index_proj
                
                if i.index_name in self.viewer.layers:
                    #self.viewer.layers.remove(i.index_name)
                    self.viewer.layers[i.index_name].data = computed_index
                    self.viewer.layers[i.index_name].refresh()
                else:
                    colormap = self.index_collection[i.index_name].colormap
                    contrast_limits = self.index_collection[i.index_name].index_map_range
                    layer = self.viewer.add_image(
                        data=computed_index, name=i.index_name, colormap=colormap,
                        blending='additive', contrast_limits=contrast_limits)
                    layer.events.contrast_limits.connect(self._on_change_index_map_rendering)
                    layer.events.colormap.connect(self._on_change_index_map_rendering)
        self.viewer.window._status_bar._toggle_activity_dock(False)

    def _on_change_index_map_rendering(self, event=None):
        """Update the contrast limits of the index layers."""

        layer = self.viewer.layers.selection.active
        if layer.name in self.index_collection:
            self.index_collection[layer.name].index_map_range = layer.contrast_limits
            self.index_collection[layer.name].colormap = layer.colormap.name

    
    def _on_change_index_index(self, event=None):

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
        """Create tick boxes for picking indices to export."""

        self.index_pick_boxes = {}
        for ind, key_val in enumerate(self.index_collection.items()):
            
            self.index_pick_group.glayout.addWidget(QLabel(key_val[0]), ind, 0, 1, 1)
            newbox = QCheckBox()
            self.index_pick_boxes[key_val[0]] = newbox
            self.index_pick_group.glayout.addWidget(newbox, ind, 1, 1, 1)

    def _on_click_export_index_tiff(self, event=None):
        """Export index maps to tiff"""
        
        export_folder = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')
        
        for key, index in self.index_collection.items():
            if key in self.viewer.layers:
                index_map = self.viewer.layers[key].data
                contrast = self.viewer.layers[key].contrast_limits
                napari_cmap = self.viewer.layers[key].colormap
                export_path = export_folder.joinpath(f'{key}_index_map.tif')
                save_tif_cmap(image=index_map, image_path=export_path,
                              napari_cmap=napari_cmap, contrast=contrast)

    def _on_click_export_index_settings(self, event=None, file_path=None):
        """Export index setttings"""

        '''index_series = [pd.Series(asdict(x)) for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
        index_table = pd.DataFrame(index_series)
        index_table.drop(columns=['index_map', 'index_proj'], inplace=True)
        if file_path is None:
            file_path = Path(str(QFileDialog.getSaveFileName(self, "Select index settings file")[0]))
        if file_path.suffix != '.csv':
            file_path = file_path.with_suffix('.csv')
        index_table.to_csv(file_path, index=False)'''

        index_series = [x.dict_spectral_index() for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
        index_series = {'index_definition': index_series}
        if file_path is None:
            file_path = Path(str(QFileDialog.getSaveFileName(self, "Select index settings file")[0]))
        if file_path.suffix != '.yml':
                file_path = file_path.with_suffix('.yml')
        with open(file_path, "w") as file:
            yaml.dump(index_series, file)

    def _on_click_import_index_settings(self, event=None):
        """Load index settings from file."""
        
        if self.index_file is None:
            self._on_click_select_index_file()
        # clear existing state
        self.qcom_indices.clear()
        self.index_pick_boxes = {}
        self.index_collection = {}

        for i in reversed(range(self.index_pick_group.glayout.count())): 
            self.index_pick_group.glayout.itemAt(i).widget().setParent(None)

        # import table, populate combobox, export tick boxes and index_collection
        #index_table = pd.read_csv(self.export_folder.joinpath('index_settings.csv'))

        with open(self.index_file) as file:
            index_series = yaml.full_load(file)
        for index_element in index_series['index_definition']:
            self.index_collection[index_element['index_name']] = SpectralIndex(**index_element)
            self.qcom_indices.addItem(index_element['index_name'])
            self.index_pick_boxes[index_element['index_name']] = QCheckBox()
            self.index_pick_group.glayout.addWidget(QLabel(index_element['index_name']), self.qcom_indices.count(), 0, 1, 1)
            self.index_pick_group.glayout.addWidget(self.index_pick_boxes[index_element['index_name']], self.qcom_indices.count(), 1, 1, 1)
        self.qcom_indices.setCurrentText(index_element['index_name'])
        
        '''index_table = pd.read_csv(self.index_file)
        index_table = index_table.replace(np.nan, None)
        for _, index_row in index_table.iterrows():
            row_dict = index_row.to_dict()
            if row_dict['middle_band'] is not None:
                row_dict['middle_band'] = int(row_dict['middle_band'])
                row_dict['middle_band_default'] = int(row_dict['middle_band_default'])
            self.index_collection[index_row.index_name] = SpectralIndex(**row_dict)
            self.index_collection[index_row.index_name].middle_bands = index_row.index_type
            self.qcom_indices.addItem(index_row.index_name)
            self.index_pick_boxes[index_row.index_name] = QCheckBox()
            self.index_pick_group.glayout.addWidget(QLabel(index_row.index_name), self.qcom_indices.count(), 0, 1, 1)
            self.index_pick_group.glayout.addWidget(self.index_pick_boxes[index_row.index_name], self.qcom_indices.count(), 1, 1, 1)
        self.qcom_indices.setCurrentText(index_row.index_name)'''
        self._on_change_index_index()

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

        


    