from pathlib import Path
from dataclasses import asdict
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget,
                            QLabel, QFileDialog, QSpinBox,
                            QComboBox, QLineEdit, QSizePolicy,
                            QGridLayout, QCheckBox, QDoubleSpinBox,
                            QColorDialog, QScrollArea)
from qtpy.QtCore import Qt, QRect
from qtpy.QtGui import QPixmap, QColor, QPainter
from superqt import QLabeledDoubleRangeSlider
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
from .spectralplot import SpectralPlotter, plot_spectral_profile
from .widgets.channel_widget import ChannelWidget
from .widgets.rgb_widget import RGBWidget
from .parameters.parameters_plots import Paramplot
from .spectralindex import (SpectralIndex, compute_index_RABD, compute_index_RABA,
                            compute_index_ratio)
from .io import load_mask, get_mask_path


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

        self.create_index_list()

        self.params = Param()
        self.params_indices = ParamEndMember()
        self.params_plots = Paramplot()

        self.em_boundary_lines = None
        self.end_members = None
        self.endmember_bands = None
        self.index_file = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ["&Main", "&Indices", "I&O", "&ROI", "P&lots"]#, "Plotslive"]
        self.tabs = TabSet(self.tab_names, tab_layouts=[None, QGridLayout(), None, None, QGridLayout(), None])

        self.main_layout.addWidget(self.tabs)

        self.btn_select_export_folder = QPushButton("Select project folder")
        self.export_path_display = QLineEdit("No path")
        self.tabs.add_named_tab('&Main', self.btn_select_export_folder)
        self.tabs.add_named_tab('&Main', self.export_path_display)
        self.btn_load_project = QPushButton("Load project")
        self.tabs.add_named_tab('&Main', self.btn_load_project)
        self.qlist_channels = ChannelWidget(self.viewer)
        self.qlist_channels.itemClicked.connect(self._on_change_select_bands)
        self.tabs.add_named_tab('&Main', self.qlist_channels)

        self.rgbwidget = RGBWidget(viewer=self.viewer)
        self.tabs.add_named_tab('&Main', self.rgbwidget.rgbmain_group.gbox)

        # indices tab
        self._create_indices_tab()
        tab_rows = self.tabs.widget(1).layout().rowCount()
        self.em_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.tabs.add_named_tab('&Indices', self.em_plot, grid_pos=(tab_rows, 0, 1, 3))
        self.em_boundaries_range = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.em_boundaries_range.setValue((0, 0, 0))
        self.tabs.add_named_tab('&Indices', self.em_boundaries_range, grid_pos=(tab_rows+1, 0, 1, 3))
        self.btn_create_index = QPushButton("New index")
        self.tabs.add_named_tab('&Indices', self.btn_create_index, grid_pos=(tab_rows+2, 0, 1, 1))
        self.qtext_new_index_name = QLineEdit()
        self.tabs.add_named_tab('&Indices', self.qtext_new_index_name, grid_pos=(tab_rows+2, 1, 1, 2))
        self.btn_update_index = QPushButton("Update current index")
        self.tabs.add_named_tab('&Indices', self.btn_update_index, grid_pos=(tab_rows+3, 0, 1, 1))

        self.btn_compute_RABD = QPushButton("Compute Index")
        self.tabs.add_named_tab('&Indices', self.btn_compute_RABD, grid_pos=(tab_rows+4, 0, 1, 3))

        # I&O tab
        self.index_pick_group = VHGroup('&Indices', orientation='G')
        self.index_pick_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('I&O', self.index_pick_group.gbox)
        self._create_index_io_pick()
        self.btn_export_index_settings = QPushButton("Export index settings")
        self.tabs.add_named_tab('I&O', self.btn_export_index_settings)
        self.btn_import_index_settings = QPushButton("Import index settings")
        self.tabs.add_named_tab('I&O', self.btn_import_index_settings)
        self.index_file_display = QLineEdit("No file selected")
        self.tabs.add_named_tab('I&O', self.index_file_display)

        self.spin_roi_width = QSpinBox()
        self.spin_roi_width.setRange(1, 1000)
        self.spin_roi_width.setValue(20)
        self.tabs.add_named_tab('&ROI', self.spin_roi_width)

        #self.index_plot = SpectralPlotter(napari_viewer=self.viewer)
        #self.tabs.add_named_tab('P&lots', self.index_plot)

        self.pixlabel = ScaledPixmapLabel()#QLabel()
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
        self.tabs.add_named_tab('P&lots', self.btn_create_index_plot, grid_pos=(1, 0, 1, 2))
        self.spin_left_right_margin_fraction = QDoubleSpinBox()
        self.spin_left_right_margin_fraction.setRange(0, 100)
        self.spin_left_right_margin_fraction.setValue(0.1)
        self.spin_left_right_margin_fraction.setSingleStep(0.1)
        self.tabs.add_named_tab('P&lots', QLabel('L/R Margin fraction'), grid_pos=(2, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_left_right_margin_fraction, grid_pos=(2, 1, 1, 1))
        self.spin_bottom_top_margin_fraction = QDoubleSpinBox()
        self.spin_bottom_top_margin_fraction.setRange(0, 100)
        self.spin_bottom_top_margin_fraction.setValue(0.05)
        self.spin_bottom_top_margin_fraction.setSingleStep(0.01)
        self.tabs.add_named_tab('P&lots', QLabel('B/T Margin fraction'), grid_pos=(3, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_bottom_top_margin_fraction, grid_pos=(3, 1, 1, 1))
        self.spin_plot_image_w_fraction = QDoubleSpinBox()
        self.spin_plot_image_w_fraction.setRange(0, 100)
        self.spin_plot_image_w_fraction.setValue(0.25)
        self.spin_plot_image_w_fraction.setSingleStep(0.1)
        self.tabs.add_named_tab('P&lots', QLabel('Plot width fraction'), grid_pos=(4, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_plot_image_w_fraction, grid_pos=(4, 1, 1, 1))
        self.spin_title_font_factor = QDoubleSpinBox()
        self.spin_label_font_factor = QDoubleSpinBox()
        for sbox in [self.spin_label_font_factor, self.spin_title_font_factor]:
            self.spin_title_font_factor.setRange(0, 1)
            self.spin_title_font_factor.setValue(0.01)
            self.spin_title_font_factor.setSingleStep(0.01)
        self.tabs.add_named_tab('P&lots', QLabel('Title Font factor'), grid_pos=(5, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_title_font_factor, grid_pos=(5, 1, 1, 1))
        self.tabs.add_named_tab('P&lots', QLabel('Label Font factor'), grid_pos=(6, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_label_font_factor, grid_pos=(6, 1, 1, 1))
        self.qcolor_plotline = QColorDialog()
        self.btn_qcolor_plotline = QPushButton("Select plot line color")
        self.tabs.add_named_tab('P&lots', self.btn_qcolor_plotline, grid_pos=(7, 0, 1, 2))
        self.qcolor_plotline.setCurrentColor(Qt.blue)
        self.spin_plot_thickness = QDoubleSpinBox()
        self.spin_plot_thickness.setRange(1, 10)
        self.spin_plot_thickness.setValue(1)
        self.spin_plot_thickness.setSingleStep(0.1)
        self.tabs.add_named_tab('P&lots', QLabel('Plot line thickness'), grid_pos=(8, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_plot_thickness, grid_pos=(8, 1, 1, 1))
        #self.btn_reset_figure_size = QPushButton("Reset figure size")
        #self.tabs.add_named_tab('P&lots', self.btn_reset_figure_size, grid_pos=(9, 0, 1, 2))
        self.spin_figure_size_factor = QDoubleSpinBox()
        self.spin_figure_size_factor.setRange(1, 100)
        self.spin_figure_size_factor.setValue(10)
        self.spin_figure_size_factor.setSingleStep(1)
        self.tabs.add_named_tab('P&lots', QLabel('Figure size factor'), grid_pos=(9, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_figure_size_factor, grid_pos=(9, 1, 1, 1))
        self.spin_scale_font_size = QSpinBox()
        self.spin_scale_font_size.setRange(1, 100)
        self.spin_scale_font_size.setValue(1)
        self.spin_scale_font_size.setSingleStep(1)
        self.tabs.add_named_tab('P&lots', QLabel('Scale font size'), grid_pos=(10, 0, 1, 1))
        self.tabs.add_named_tab('P&lots', self.spin_scale_font_size, grid_pos=(10, 1, 1, 1))
        self.btn_save_plot = QPushButton("Save plot")
        self.tabs.add_named_tab('P&lots', self.btn_save_plot, grid_pos=(11, 0, 1, 2))

        self.btn_save_plot_params = QPushButton("Save plot parameters")
        self.tabs.add_named_tab('P&lots', self.btn_save_plot_params, grid_pos=(12, 0, 1, 2))
        self.btn_load_plot_params = QPushButton("Load plot parameters")
        self.tabs.add_named_tab('P&lots', self.btn_load_plot_params, grid_pos=(13, 0, 1, 2))

        
        self._connect_spin_bounds()
        self.add_connections()

    def _create_indices_tab(self):

        self.current_index_type = 'RABD'

        self.indices_group = VHGroup('&Indices', orientation='G')
        self.tabs.add_named_tab('&Indices', self.indices_group.gbox, [1, 0, 1, 3])

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
        self.em_boundaries_range.valueChanged.connect(self._on_change_em_boundaries)
        self.btn_compute_RABD.clicked.connect(self._on_click_compute_index)
        self.btn_create_index.clicked.connect(self._on_click_new_index)
        self.btn_update_index.clicked.connect(self._on_click_update_index)
        self.qcom_indices.activated.connect(self._on_change_index_index)
        self.btn_export_index_settings.clicked.connect(self._on_click_export_index_settings)
        self.btn_import_index_settings.clicked.connect(self._on_click_import_index_settings)
        self.btn_create_index_plot.clicked.connect(self.create_index_plot)
        
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

        self.em_boundaries_range.setValue(
            (self.spin_index_left.value(), self.spin_index_middle.value(),
              self.spin_index_right.value()))
    
    def _on_click_select_export_folder(self, event=None, export_folder=None):
        """Interactively select folder to analyze"""

        if export_folder is None:
            self.export_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
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

        self.params = load_project_params(folder=self.export_folder)
        self.params_indices = load_endmember_params(folder=self.export_folder)

        self.imhdr_path = Path(self.params.file_path)

        self.mainroi = np.array([np.array(x).reshape(4,2) for x in self.params.main_roi]).astype(int)
        self.row_bounds = [self.mainroi[0][:,0].min(), self.mainroi[0][:,0].max()]
        self.col_bounds = [self.mainroi[0][:,1].min(), self.mainroi[0][:,1].max()]
        
        self.imagechannels = ImChannels(self.export_folder.joinpath('corrected.zarr'))
        self.qlist_channels._update_channel_list(imagechannels=self.imagechannels)
        self.rgbwidget.imagechannels = self.imagechannels

        self.get_RGB()
        self.rgbwidget.load_and_display_rgb_bands(roi=np.concatenate([self.row_bounds, self.col_bounds]))

        self._on_click_load_mask()

        self.end_members = pd.read_csv(self.export_folder.joinpath('end_members.csv')).values
        self.endmember_bands = self.end_members[:,-1]
        self.end_members = self.end_members[:,:-1]

        self.em_boundaries_range.setRange(min=self.endmember_bands[0],max=self.endmember_bands[-1])
        self.em_boundaries_range.setValue(
            (self.endmember_bands[0], (self.endmember_bands[-1]+self.endmember_bands[0])/2, self.endmember_bands[-1]))
        
        self.plot_endmembers()
        self._on_change_index_index()

    def _add_analysis_roi(self, viewer=None, event=None, roi_xpos=None):
        """Add roi to layer"""
        
        edge_width = np.min([10, self.viewer.layers['imcube'].data.shape[1]//100])
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
         
        self.viewer.layers['rois'].add_rectangles(new_roi, edge_color='r')


    def get_RGB(self):
        
        rgb_ch, rgb_names = self.imagechannels.get_indices_of_bands(self.rgbwidget.rgb)
        [self.qlist_channels.item(x).setSelected(True) for x in rgb_ch]
        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)

    def _on_change_select_bands(self, event=None):

        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)

    def _on_click_load_mask(self):
        """Load mask from file"""
        
        mask = load_mask(get_mask_path(self.export_folder))#[self.row_bounds[0]:self.row_bounds[1], self.col_bounds[0]:self.col_bounds[1]]
        if 'mask' in self.viewer.layers:
            self.viewer.layers['mask'].data = mask
        else:
            self.viewer.add_labels(mask, name='mask')

    def plot_endmembers(self, event=None):
        """Cluster the pure pixels and plot the endmembers as average of clusters."""

        self.em_plot.axes.clear()
        self.em_plot.axes.plot(self.endmember_bands, self.end_members)
        self.em_plot.figure.canvas.draw()

    def _on_change_em_boundaries(self, event=None):
        """Update the em plot when the em boundaries are changed."""
        
        #self._disconnect_spin_bounds()
        # update from interactive limit change
        if type(event) == tuple:
            current_triplet = np.array(self.em_boundaries_range.value(), dtype=np.uint16)
            if len(current_triplet) == 3:
                self.spin_index_left.setValue(current_triplet[0])
                self.spin_index_middle.setValue(current_triplet[1])
                self.spin_index_right.setValue(current_triplet[2])
            elif len(current_triplet) == 2:
                self.spin_index_left.setValue(current_triplet[0])
                self.spin_index_right.setValue(current_triplet[1])
            
        # update from spinbox change
        else:
            if self.current_index_type == 'RABD':
                current_triplet = [self.spin_index_left.value(), self.spin_index_middle.value(), self.spin_index_right.value()]
            else:
                current_triplet = [self.spin_index_left.value(), self.spin_index_right.value()]
            current_triplet = [float(x) for x in current_triplet]

            self.em_boundaries_range.setValue(current_triplet)

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
            self.em_boundary_lines = self.em_plot.axes.plot(
                [x_toplot, x_toplot],
                [
                    ymin_toplot, ymax_toplot
                ], 'r--'
            )
            self.em_plot.figure.canvas.draw()
        
        #self._connect_spin_bounds()

    def _update_save_plot_parameters(self):

        self.params_plots.color_plotline = [self.qcolor_plotline.currentColor().getRgb()[x]/255 for x in range(3)]
        self.params_plots.plot_thickness = self.spin_plot_thickness.value()
        self.params_plots.title_font_factor = self.spin_title_font_factor.value()
        self.params_plots.label_font_factor = self.spin_label_font_factor.value()
        self.params_plots.scale_font_size = self.spin_scale_font_size.value()
        self.params_plots.left_right_margin_fraction = self.spin_left_right_margin_fraction.value()
        self.params_plots.bottom_top_margin_fraction = self.spin_bottom_top_margin_fraction.value()
        self.params_plots.plot_image_w_fraction = self.spin_plot_image_w_fraction.value()
        self.params_plots.figure_size_factor = self.spin_figure_size_factor.value()
        for key in self.index_collection:
            if key in self.viewer.layers:
                self.params_plots.index_colormap[key] = self.viewer.layers[key].colormap.name
        self.params_plots.red_conrast_limits = np.array(self.viewer.layers['red'].contrast_limits).tolist()
        self.params_plots.green_conrast_limits = np.array(self.viewer.layers['green'].contrast_limits).tolist()
        self.params_plots.blue_conrast_limits = np.array(self.viewer.layers['blue'].contrast_limits).tolist()
        self.params_plots.rgb_bands = self.rgbwidget.rgb

    def _on_click_save_plot_parameters(self, event=None, file_path=None):
            
        if file_path is None:
            file_path = Path(str(QFileDialog.getSaveFileName(self, "Select plot parameters file")[0]))
        self._update_save_plot_parameters()
        self.params_plots.save_parameters(file_path)

    def _on_click_load_plot_parameters(self, event=None, file_path=None):
        
        self.disconnect_plot_formatting()
        if file_path is None:
            file_path = Path(str(QFileDialog.getOpenFileName(self, "Select plot parameters file")[0]))
        self.params_plots = load_plots_params(file_path=file_path)

        self.spin_plot_thickness.setValue(self.params_plots.plot_thickness)
        self.spin_title_font_factor.setValue(self.params_plots.title_font_factor)
        self.spin_label_font_factor.setValue(self.params_plots.label_font_factor)
        self.spin_scale_font_size.setValue(self.params_plots.scale_font_size)
        self.spin_left_right_margin_fraction.setValue(self.params_plots.left_right_margin_fraction)
        self.spin_bottom_top_margin_fraction.setValue(self.params_plots.bottom_top_margin_fraction)
        self.spin_plot_image_w_fraction.setValue(self.params_plots.plot_image_w_fraction)
        self.spin_figure_size_factor.setValue(self.params_plots.figure_size_factor)
        self.qcolor_plotline.setCurrentColor(QColor(*[int(x*255) for x in self.params_plots.color_plotline]))
        for key in self.params_plots.index_colormap:
            if key in self.viewer.layers:
                self.viewer.layers[key].colormap = self.params_plots.index_colormap[key]
        self.viewer.layers['red'].contrast_limits = self.params_plots.red_conrast_limits
        self.viewer.layers['green'].contrast_limits = self.params_plots.green_conrast_limits
        self.viewer.layers['blue'].contrast_limits = self.params_plots.blue_conrast_limits
        self.rgbwidget.rgb = self.params_plots.rgb_bands
        
        self.rgbwidget._on_click_RGB()
        self.connect_plot_formatting()
        self.create_index_plot()

    def disconnect_plot_formatting(self):
        """Disconnect plot editing widgets while loading parameters to avoid overwriting
        the loaded parameters."""
        
        self.spin_plot_thickness.valueChanged.disconnect(self.create_index_plot)
        self.spin_title_font_factor.valueChanged.disconnect(self.create_index_plot)
        self.spin_label_font_factor.valueChanged.disconnect(self.create_index_plot)
        self.spin_scale_font_size.valueChanged.disconnect(self.create_index_plot)
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
        self.spin_scale_font_size.valueChanged.connect(self.create_index_plot)
        self.spin_left_right_margin_fraction.valueChanged.connect(self.create_index_plot)
        self.spin_bottom_top_margin_fraction.valueChanged.connect(self.create_index_plot)
        self.spin_plot_image_w_fraction.valueChanged.connect(self.create_index_plot)
        self.spin_figure_size_factor.valueChanged.connect(self.create_index_plot)
        self.qcolor_plotline.currentColorChanged.connect(self.create_index_plot)

    def create_index_plot(self, event=None):
        """Create the index plot."""

        self._update_save_plot_parameters()

        # get rgb image and index image to plot
        rgb_image = [self.viewer.layers[c].data for c in ['red', 'green', 'blue']]
        if self.qcom_indices.currentText() not in self.viewer.layers:
            self._on_click_compute_index(event=None)
        toplot = self.viewer.layers[self.qcom_indices.currentText()].data
        toplot[toplot == np.inf] = 0
        percentiles = np.percentile(toplot, [1, 99])
        toplot = np.clip(toplot, percentiles[0], percentiles[1])
        mask = self.viewer.layers['mask'].data

        if isinstance(rgb_image[0], da.Array):
            rgb_image = [x.compute() for x in rgb_image]
        if isinstance(toplot, da.Array):
            toplot = toplot.compute()

        format_dict = asdict(self.params_plots)
        _, self.ax1, self.ax2, self.ax3 = plot_spectral_profile(
            rgb_image=rgb_image, mask=mask, index_image=toplot, index_name=self.qcom_indices.currentText(),
                                format_dict=format_dict, scale=self.params.scale,
                                location=self.params.location, fig=self.index_plot_live.figure, 
                                roi=self.viewer.layers['rois'].data[0])

        # save temporary low-res figure for display in napari
        self.index_plot_live.figure.savefig(
            self.export_folder.joinpath('temp.png'),
            dpi=self.spin_preview_dpi.value())#, bbox_inches="tight")

        # update napari preview
        if self.pix_width is None:
            self.pix_width = self.pixlabel.size().width()
            self.pix_height = self.pixlabel.size().height()
        self.pixmap = QPixmap(self.export_folder.joinpath('temp.png').as_posix())
        #self.pixlabel.setPixmap(self.pixmap.scaled(self.pixlabel.size().width(), self.pixlabel.size().height(), Qt.KeepAspectRatio))
        self.pixlabel.setPixmap(self.pixmap.scaled(self.pix_width, self.pix_height, Qt.KeepAspectRatio))
        #self.pixlabel.show()
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
        
        if export_file is None:
            export_file = self.export_folder.joinpath(self.qcom_indices.currentText()+'_index_plot.png')
        self.index_plot_live.figure.savefig(
            fname=export_file, dpi=self.spin_final_dpi.value())#, bbox_inches="tight")

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

        current_bands = np.array(self.em_boundaries_range.value(), dtype=np.uint16)
        name = self.qtext_new_index_name.text()
        if self.current_index_type == 'RABD':
            self.index_collection[name] = SpectralIndex(index_name=name,
                              index_type='RABD',
                              left_band_default=current_bands[0],
                              middle_band_default=current_bands[1],
                              right_band_default=current_bands[2]
                              )
            
        elif self.current_index_type == 'RABA':
            self.index_collection[name] = SpectralIndex(index_name=name,
                              index_type='RABA',
                              left_band_default=current_bands[0],
                              right_band_default=current_bands[1],
                              )
        elif self.current_index_type == 'ratio':
            self.index_collection[name] = SpectralIndex(index_name=name,
                              index_type='ratio',
                              left_band_default=current_bands[0],
                              right_band_default=current_bands[1],
                              )
        
        self.qcom_indices.addItem(name)
        self.qcom_indices.setCurrentText(name)

        ## add box to pick
        num_boxes = len(self.index_collection)
        self.index_pick_group.glayout.addWidget(QLabel(name), num_boxes, 0, 1, 1)
        newbox = QCheckBox()
        self.index_pick_boxes[name] = newbox
        self.index_pick_group.glayout.addWidget(newbox, num_boxes, 1, 1, 1)
        

    def _on_click_update_index(self, event):
        """Update the current index."""

        current_bands = np.array(self.em_boundaries_range.value(), dtype=np.uint16)
        name = self.qcom_indices.currentText()
        
        if self.current_index_type == 'RABD':
            self.index_collection[name].left_band = current_bands[0]
            self.index_collection[name].right_band = current_bands[2]
            self.index_collection[name].middle_band = current_bands[1]
        else:
            self.index_collection[name].left_band = current_bands[0]
            self.index_collection[name].right_band = current_bands[1]
            self.index_collection[name].middle_band = None            
            

    def _on_click_compute_index(self, event):
        """Compute the index and add to napari."""

        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Computing index")

            if self.current_index_type == 'RABD':
                rabd_indices = compute_index_RABD(
                    left=self.spin_index_left.value(),
                    trough=self.spin_index_middle.value(),
                    right=self.spin_index_right.value(),
                    row_bounds=self.row_bounds,
                    col_bounds=self.col_bounds,
                    imagechannels=self.imagechannels)
                self.viewer.add_image(rabd_indices, name=self.qcom_indices.currentText(), colormap='viridis', blending='additive')
            elif self.current_index_type == 'RABA':
                raba_indices = compute_index_RABA(
                    left=self.spin_index_left.value(),
                    right=self.spin_index_right.value(),
                    row_bounds=self.row_bounds,
                    col_bounds=self.col_bounds,
                    imagechannels=self.imagechannels)
                self.viewer.add_image(raba_indices, name=self.qcom_indices.currentText(), colormap='viridis', blending='additive')
            elif self.current_index_type == 'Ratio':
                ratio_indices = compute_index_ratio(
                    left=self.spin_index_left.value(),
                    right=self.spin_index_right.value(),
                    row_bounds=self.row_bounds,
                    col_bounds=self.col_bounds,
                    imagechannels=self.imagechannels)
                self.viewer.add_image(ratio_indices, name=self.qcom_indices.currentText(), colormap='viridis', blending='additive')
            else:
                print(f'unknown index type: {self.current_index_type}')
        self.viewer.window._status_bar._toggle_activity_dock(False)
    
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


    def _on_click_export_index_settings(self, event=None, file_path=None):
        """Export index setttings"""

        index_series = [pd.Series(asdict(x)) for key, x in self.index_collection.items() if self.index_pick_boxes[key].isChecked()]
        index_table = pd.DataFrame(index_series)
        if file_path is None:
            file_path = Path(str(QFileDialog.getSaveFileName(self, "Select index settings file")[0]))
        if file_path.suffix != '.csv':
            file_path = file_path.with_suffix('.csv')
        index_table.to_csv(file_path, index=False)

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
        index_table = pd.read_csv(self.index_file)
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
        self.qcom_indices.setCurrentText(index_row.index_name)
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

        


    