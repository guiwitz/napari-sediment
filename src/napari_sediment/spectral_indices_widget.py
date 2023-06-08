from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget,
                            QLabel, QFileDialog, QSpinBox,
                            QComboBox, QLineEdit, QSizePolicy)
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from superqt import QLabeledDoubleRangeSlider
from spectral import open_image
from spectral.algorithms import calc_stats, mnf, noise_from_diffs, remove_continuum
from spectral.algorithms import ppi
import pandas as pd
from microfilm import colorify

from .parameters import Param
from .parameters_endmembers import ParamEndMember
from .io import load_project_params, load_endmember_params
from .imchannels import ImChannels
from .sediproc import find_index_of_band
from ._reader import read_spectral
from .spectralplot import SpectralPlotter
from .channel_widget import ChannelWidget
from .io import load_mask, get_mask_path
from napari_guitils.gui_structures import TabSet, VHGroup


class SpectralIndexWidget(QWidget):
    """Widget for the SpectralIndices."""
    
    def __init__(self, napari_viewer):
        super().__init__()
        
        self.viewer = napari_viewer

        self.rgb = [640, 545, 460]

        self.index_triplets = []

        self.params = Param()
        self.params_indices = ParamEndMember()
        self.ppi_boundary_lines = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['Main', "ROI", "Plots"]
        self.tabs = TabSet(self.tab_names)

        self.main_layout.addWidget(self.tabs)

        self.btn_select_export_folder = QPushButton("Select project folder")
        self.export_path_display = QLineEdit("No path")
        self.tabs.add_named_tab('Main', self.btn_select_export_folder)
        self.tabs.add_named_tab('Main', self.export_path_display)
        self.btn_load_project = QPushButton("Load project")
        self.tabs.add_named_tab('Main', self.btn_load_project)
        self.qlist_channels = ChannelWidget(self)
        self.tabs.add_named_tab('Main', self.qlist_channels)

        self.ppi_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.tabs.add_named_tab('Main', self.ppi_plot)
        self.ppi_boundaries_range = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.ppi_boundaries_range.setValue((0, 0, 0))
        self.tabs.add_named_tab('Main', self.ppi_boundaries_range)
        self.btn_add_triplets = QPushButton("Add triplets")
        self.tabs.add_named_tab('Main', self.btn_add_triplets)

        self.btn_compute_RABD = QPushButton("Compute RABD")
        self.tabs.add_named_tab('Main', self.btn_compute_RABD)
        self.qcom_triplets = QComboBox()
        self.tabs.add_named_tab('Main', self.qcom_triplets)

        self.spin_roi_width = QSpinBox()
        self.spin_roi_width.setRange(1, 1000)
        self.spin_roi_width.setValue(20)
        self.tabs.add_named_tab('ROI', self.spin_roi_width)

        #self.index_plot = SpectralPlotter(napari_viewer=self.viewer)
        #self.tabs.add_named_tab('Plots', self.index_plot)

        self.pixlabel = QLabel()
        self.pixlabel.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding)
        #self.pixlabel.resizeEvent = self._on_resize_preview
        self.tabs.add_named_tab('Plots', self.pixlabel)

        self.add_connections()

    def add_connections(self):
        """Add callbacks"""

        self.btn_select_export_folder.clicked.connect(self._on_click_select_export_folder)
        self.btn_load_project.clicked.connect(self.import_project)
        self.ppi_boundaries_range.valueChanged.connect(self._on_change_ppi_boundaries)
        self.btn_compute_RABD.clicked.connect(self._on_click_compute_RABD)
        self.btn_add_triplets.clicked.connect(self._on_click_add_triplets)
        self.qcom_triplets.currentIndexChanged.connect(self._on_change_ppi_boundaries)
        
        self.viewer.mouse_double_click_callbacks.append(self._add_analysis_roi)

    def _on_click_select_export_folder(self):
        """Interactively select folder to analyze"""

        self.export_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        self.export_path_display.setText(self.export_folder.as_posix())

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
        self.qlist_channels._update_channel_list()

        self.get_RGB()

        self.end_members = pd.read_csv(self.export_folder.joinpath('end_members.csv')).values
        self.endmember_bands = self.end_members[:,-1]
        self.end_members = self.end_members[:,:-1]

        self.ppi_boundaries_range.setRange(min=self.endmember_bands[0],max=self.endmember_bands[-1])
        self.ppi_boundaries_range.setValue(
            (self.endmember_bands[0], (self.endmember_bands[-1]+self.endmember_bands[0])/2, self.endmember_bands[-1]))
        
        self.plot_ppi()

    def _add_analysis_roi(self, viewer, event):
        """Add roi to layer"""
        
        cursor_pos = np.rint(self.viewer.cursor.position).astype(int)
        min_row = 0
        max_row = self.row_bounds[1] - self.row_bounds[0]

        new_roi = [
            [min_row, cursor_pos[2]-self.spin_roi_width.value()//2],
            [max_row,cursor_pos[2]-self.spin_roi_width.value()//2],
            [max_row,cursor_pos[2]+self.spin_roi_width.value()//2],
            [min_row,cursor_pos[2]+self.spin_roi_width.value()//2]]
        
        if 'rois' not in self.viewer.layers:
            self.viewer.add_shapes(
                ndim = 2,
                name='rois', edge_color='red', face_color=np.array([0,0,0,0]), edge_width=10)
         
        self.viewer.layers['rois'].add_rectangles(new_roi, edge_color='r', edge_width=10)


    def get_RGB(self):
        
        self.rgb_ch = [np.argmin(np.abs(np.array(self.imagechannels.channel_names).astype(float) - x)) for x in self.rgb]
        self.rgb_names = [self.imagechannels.channel_names[x] for x in self.rgb_ch]
        [self.qlist_channels.item(x).setSelected(True) for x in self.rgb_ch]
        self.qlist_channels._on_change_channel_selection()

    def plot_ppi(self, event=None):
        """Cluster the pure pixels and plot the endmembers as average of clusters."""

        self.ppi_plot.axes.clear()
        self.ppi_plot.axes.plot(self.endmember_bands, self.end_members)
        self.ppi_plot.canvas.figure.canvas.draw()

    def _on_change_ppi_boundaries(self, event):
        """Update the PPI plot when the PPI boundaries are changed."""

        if type(event) == tuple:
            current_triplet = list(self.ppi_boundaries_range.value())
        else:
            current_triplet = self.qcom_triplets.currentText().split('-')
            current_triplet = [float(x) for x in current_triplet]
            self.ppi_boundaries_range.setValue(current_triplet)

        if self.ppi_boundary_lines is not None:
                num_lines = len(self.ppi_boundary_lines)
                for i in range(num_lines):
                    self.ppi_boundary_lines.pop(0).remove()

        if self.end_members is not None:
            ymin = self.end_members.min()
            ymax = self.end_members.max()
            self.ppi_boundary_lines = self.ppi_plot.axes.plot(
                [
                    [current_triplet[0], current_triplet[1], current_triplet[2]], [current_triplet[0], current_triplet[1], current_triplet[2]]
                ],
                [
                    [ymin, ymin, ymin], [ymax, ymax, ymax]
                ], 'r--'
            )
            self.ppi_plot.canvas.figure.canvas.draw()

    def create_index_plot(self):
        """Create the index plot."""


        toplot = self.viewer.layers['RABD'].data
        toplot[toplot == np.inf] = 0

        rgb_to_plot = self.viewer.layers['imcube'].data.copy()
        rgb_to_plot, _, _, _ = colorify.multichannel_to_rgb(
            rgb_to_plot,
            cmaps=['pure_red', 'pure_green', 'pure_blue'], 
            rescale_type='limits', limits=[0,4000], proj_type='sum')

        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(2,8))

        ax[0].imshow(rgb_to_plot, aspect='auto')
        ax[1].imshow(self.viewer.layers['RABD'].data, vmin=0, vmax=2, aspect='auto')
        #ax[2].imshow(np.ones((toplot.shape[0],toplot.shape[1],3)))
        if 'rois' in self.viewer.layers:
            roi = self.viewer.layers['rois'].data[0]
            colmin = int(roi[0,1])
            colmax = int(roi[3,1])
            proj = self.viewer.layers['RABD'].data[:,colmin:colmax].mean(axis=1)
            ax[2].imshow(rgb_to_plot, alpha=0.0,aspect='auto')
            ax[2].plot(1000*proj, np.arange(len(proj)))
            #ax[2].invert_yaxis()
            
        ax[0].set_xticks([])
        ax[1].set_xticks([])


        #ax[2].set_aspect(100*ax[0].get_data_ratio())
        fig.subplots_adjust(wspace=0)
        ax[0].set_ylabel('depth')
        fig.savefig(self.export_folder.joinpath('index_plot.png'))

        self.pixmap = QPixmap(self.export_folder.joinpath('index_plot.png').as_posix())
        self.pixlabel.setPixmap(self.pixmap.scaledToHeight(self.pixlabel.size().height()))

    def _on_click_add_triplets(self, event):
        #self.index_triplets.append(list(self.ppi_boundaries_range.value()))
        self.qcom_triplets.addItem(
            f'{self.ppi_boundaries_range.value()[0]}-{self.ppi_boundaries_range.value()[1]}-{self.ppi_boundaries_range.value()[2]}')


    def _on_click_compute_RABD(self, event):
        
        rabd_indices = self.compute_interactive_index_RABD(
            left=self.ppi_boundaries_range.value()[0],
            trough=self.ppi_boundaries_range.value()[1],
            right=self.ppi_boundaries_range.value()[2]
        )
        self.viewer.add_image(rabd_indices, name='RABD', colormap='viridis', blending='additive')

    def compute_interactive_index_RABD(self, left, trough, right):
        """Compute the index RAB."""

        ltr = [left, trough, right]
        # find indices from the end-members plot (in case not all bands were used
        ltr_endmember_indices = [find_index_of_band(self.endmember_bands, x) for x in ltr]
        # find band indices in the complete dataset
        ltr_stack_indices = [find_index_of_band(self.imagechannels.centers, x) for x in ltr]

        # number of bands between edges and trough
        X_left = ltr_endmember_indices[1]-ltr_endmember_indices[0]
        X_right = ltr_endmember_indices[2]-ltr_endmember_indices[1]

        # load the correct bands
        roi = np.concatenate([self.row_bounds, self.col_bounds])
        ltr_cube = self.imagechannels.get_image_cube(
            channels=ltr_stack_indices, roi=roi)
        ltr_cube = ltr_cube.astype(np.float32)

        # compute indices
        RABD = ((ltr_cube[0] * X_right + ltr_cube[2] * X_left) / (X_left + X_right)) / ltr_cube[1] 

        return RABD

    