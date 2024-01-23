from dataclasses import asdict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QGridLayout, QLineEdit,
                            QFileDialog)
from qtpy.QtCore import Qt
from napari_guitils.gui_structures import TabSet
from microfilm.microplot import microshow

from .spectralindex import (SpectralIndex, compute_index_RABD, compute_index_RABA,
                            compute_index_ratio)
from .spectralplot import plot_spectral_profile  
from .imchannels import ImChannels
from .io import load_project_params, load_plots_params
from .widgets.rgb_widget import RGBWidget


class BatchWidget(QWidget):
    """
    Widget for the SpectralIndices.
    
    Parameters
    ----------
    napari_viewer: napari.Viewer

    Attributes
    ----------
    viewer: napari.Viewer
        napari viewer
    
    
    """
    
    def __init__(self, napari_viewer):
        super().__init__()
        
        self.viewer = napari_viewer
        self.index_file = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ["&Main", "&Options"]
        self.tabs = TabSet(self.tab_names, tab_layouts=[None, QGridLayout()])

        self.tabs.widget(0).layout().setAlignment(Qt.AlignTop)
        self.tabs.widget(1).layout().setAlignment(Qt.AlignTop)

        self.main_layout.addWidget(self.tabs)

        self.btn_select_data_folder = QPushButton("Select data folder")
        self.data_path_display = QLineEdit("No path")
        self.tabs.add_named_tab('&Main', self.btn_select_data_folder)
        self.tabs.add_named_tab('&Main', self.data_path_display)

        self.btn_select_export_folder = QPushButton("Select export folder")
        self.export_path_display = QLineEdit("No path")
        self.tabs.add_named_tab('&Main', self.btn_select_export_folder)
        self.tabs.add_named_tab('&Main', self.export_path_display)

        self.btn_import_index_settings = QPushButton("Import index settings")
        self.tabs.add_named_tab('&Main', self.btn_import_index_settings)
        self.index_file_display = QLineEdit("No file selected")
        self.tabs.add_named_tab('&Main', self.index_file_display)

        self.btn_import_plot_settings = QPushButton("Import plot settings")
        self.tabs.add_named_tab('&Main', self.btn_import_plot_settings)
        self.plot_file_display = QLineEdit("No file selected")
        self.tabs.add_named_tab('&Main', self.plot_file_display)

        self.add_connections()


    def add_connections(self):
        """Add callbacks"""

        self.btn_select_export_folder.clicked.connect(self._on_click_select_export_folder)
        self.btn_select_data_folder.clicked.connect(self._on_click_select_data_folder)
        self.btn_import_index_settings.clicked.connect(self._on_click_import_index_settings)
        self.btn_import_plot_settings.clicked.connect(self._on_click_import_plot_settings)

    def _on_click_select_index_file(self, event=None, index_file=None):
        """Interactively select folder to analyze"""

        if index_file is None:
            self.index_file = Path(str(QFileDialog.getOpenFileName(self, "Select Index file")[0]))
        else:
            self.index_file = Path(index_file)
        self.index_file_display.setText(self.index_file.as_posix())


    def _on_click_select_export_folder(self, event=None, export_folder=None):
        """Interactively select folder to analyze"""

        if export_folder is None:
            self.export_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        else:
            self.export_folder = Path(export_folder)
        self.export_path_display.setText(self.export_folder.as_posix())

    def _on_click_select_data_folder(self, event=None, data_folder=None):
        """Interactively select folder to analyze"""

        if data_folder is None:
            self.data_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        else:
            self.data_folder = Path(data_folder)
        self.data_path_display.setText(self.data_folder.as_posix())

    def _on_click_import_index_settings(self):

        if self.index_file is None:
            self._on_click_select_index_file()

        self.index_collection = {}

        # import table, populate combobox, export tick boxes and index_collection
        index_table = pd.read_csv(self.index_file)
        index_table = index_table.replace(np.nan, None)
        for _, index_row in index_table.iterrows():
            row_dict = index_row.to_dict()
            if row_dict['middle_band'] is not None:
                row_dict['middle_band'] = int(row_dict['middle_band'])
                row_dict['middle_band_default'] = int(row_dict['middle_band_default'])
            self.index_collection[index_row.index_name] = SpectralIndex(**row_dict)
            self.index_collection[index_row.index_name].middle_bands = index_row.index_type

    def _on_click_import_plot_settings(self, event=None, file_path=None):
        
        if file_path is None:
            file_path = Path(str(QFileDialog.getOpenFileName(self, "Select plot parameters file")[0]))
        else:
            file_path = Path(file_path)
        self.plot_file_display.setText(file_path.as_posix())
        self.params_plots = load_plots_params(file_path=file_path)


    def batch_process(self):
        
        fig, ax = plt.subplots()

        for f in self.data_folder.iterdir():
            if f.name[0] != '.':
                imagechannels = ImChannels(f.joinpath('corrected.zarr'))

                self.params = load_project_params(folder=f)

                self.mainroi = np.array([np.array(x).reshape(4,2) for x in self.params.main_roi]).astype(int)
                self.row_bounds = [self.mainroi[0][:,0].min(), self.mainroi[0][:,0].max()]
                self.col_bounds = [self.mainroi[0][:,1].min(), self.mainroi[0][:,1].max()]

                for spectral_name, spectral_index in self.index_collection.items():
        
                    if spectral_index.index_type == 'RABD':
                        index_data = compute_index_RABD(
                            left=spectral_index.left_band,
                            trough=spectral_index.middle_band,
                            right=spectral_index.right_band,
                            row_bounds=self.row_bounds,
                            col_bounds=self.col_bounds,
                            imagechannels=imagechannels)
                        self.viewer.add_image(index_data, name=f'{f}_{spectral_name}', colormap='viridis')
                    elif spectral_index.index_type == 'RABA':
                        index_data = compute_index_RABA(
                            left=spectral_index.left_band,
                            right=spectral_index.right_band,
                            row_bounds=self.row_bounds,
                            col_bounds=self.col_bounds,
                            imagechannels=imagechannels)
                    elif spectral_index.index_type == 'Ratio':
                        index_data = compute_index_ratio(
                            left=spectral_index.left_band,
                            right=spectral_index.right_band,
                            row_bounds=self.row_bounds,
                            col_bounds=self.col_bounds,
                            imagechannels=imagechannels)
                
                    self.rgb_ch, self.rgb_names = imagechannels.get_indices_of_bands(self.params_plots.rgb_bands)
                    self.rgb_cube = np.asarray(imagechannels.get_image_cube(self.rgb_ch))
                    index_data = np.asarray(index_data)

                    self.export_rgb_image_scaled(f'{f.name}_{spectral_name}')

                    roi = np.array([np.array(x).reshape(4,2) for x in self.params.rois]).astype(int)

                    format_dict = asdict(self.params_plots)
                    fig, ax1, ax2, ax3 = plot_spectral_profile(
                        rgb_image=[x for x in self.rgb_cube], index_image=index_data, index_name=spectral_name,
                                            format_dict=format_dict, scale=self.params.scale,
                                            location=self.params.location, fig=fig, 
                                            roi=roi[0])
                    
                    # save temporary low-res figure for display in napari
                    fig.savefig(
                        self.export_folder.joinpath(f'{f.name}_{spectral_name}.png'),
                        dpi=100)#, bbox_inches="tight")
                    
    def export_rgb_image_scaled(self, image_name):
        
        data = np.asarray(self.rgb_cube)
        limits = [[np.percentile(x, 2), np.percentile(x,98)] for x in data]
        microim = microshow(data, cmaps=['pure_red', 'pure_green', 'pure_blue'], limits=limits)
        microim.savefig(self.export_folder.joinpath(f'{image_name}_rgb.png'),dpi=300)
        
        im_sqrt = np.sqrt(data)
        limits = [[np.percentile(x, 2), np.percentile(x,98)] for x in im_sqrt]
        microim = microshow(im_sqrt, cmaps=['pure_red', 'pure_green', 'pure_blue'], limits=limits)
        microim.savefig(self.export_folder.joinpath(f'{image_name}_rgb_qrt.png'),dpi=300)

        im_log = np.log(data)
        limits = [[np.percentile(x, 2), np.percentile(x,98)] for x in im_log]
        microim = microshow(im_log, cmaps=['pure_red', 'pure_green', 'pure_blue'], limits=limits)
        microim.savefig(self.export_folder.joinpath(f'{image_name}_rgb_log.png'),dpi=300)
