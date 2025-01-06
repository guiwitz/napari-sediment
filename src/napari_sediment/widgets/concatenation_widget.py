from pathlib import Path
from warnings import warn
import numpy as np
import pandas as pd
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QGridLayout,
                            QCheckBox, QSpinBox, QLabel, QSizePolicy,
                            QTableWidget, QTableWidgetItem, QDoubleSpinBox,
                            QListWidget, QAbstractItemView, QRadioButton, QButtonGroup)
from qtpy.QtCore import Qt
from cmap import Colormap
import tifffile
from magicgui.widgets import FileEdit
from napari_guitils.gui_structures import TabSet
from napari_guitils.gui_structures import VHGroup

from ..data_structures.imchannels import ImChannels
from ..widget_utilities.folder_list_widget import FolderListWidget
from ..utilities.io import get_im_main_roi
from ..utilities.io import load_project_params, load_plots_params
from ..utilities.spectralindex_compute import (load_index_zarr, load_projection,
                                               load_index_series, save_zarr)
from ..utilities.spectralplot import create_rgb_image, save_tif_cmap
from ..utilities.morecolormaps import get_cmap_catalogue
get_cmap_catalogue()


class ConcatenationWidget(QWidget):
    """
    Widget to interactively assemble data.
    
    Parameters
    ----------
    napari_viewer: napari.Viewer
        napari viewer

    Attributes
    ----------
    viewer: napari.Viewer
        napari viewer
    
    
    """
    
    def __init__(self, napari_viewer):
        super().__init__()
        
        self.viewer = napari_viewer
        self.file_list = None
        self.indices = None
        self.params_plots = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ["Data", "Concatenation", "Export"]
        self.tabs = TabSet(self.tab_names, tab_layouts=[QGridLayout(), QGridLayout(), QGridLayout()])

        self.tabs.widget(0).layout().setAlignment(Qt.AlignTop)
        self.tabs.widget(1).layout().setAlignment(Qt.AlignTop)
        self.tabs.widget(2).layout().setAlignment(Qt.AlignTop)

        self.main_layout.addWidget(self.tabs)

        ## Data selection tab
        self.data_selection_group = VHGroup('Select data', orientation='G')

        self.data_selection_group.glayout.setAlignment(Qt.AlignTop)

        self.tabs.add_named_tab('Data', self.data_selection_group.gbox, (0, 0, 1, 2))

        self.data_selection_group.glayout.addWidget(QLabel('Select main folder'))
        self.main_path_display = FileEdit('d')
        self.data_selection_group.glayout.addWidget(self.main_path_display.native)
        self.data_selection_group.glayout.addWidget(QLabel('Available folders'))
        self.file_list = FolderListWidget(
            viewer=napari_viewer, only_folders=True, ignore=['assembled'])
        self.file_list.setMaximumHeight(100)
        self.file_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.data_selection_group.glayout.addWidget(self.file_list)

        self.btn_load_all = QPushButton("Load all")
        self.btn_load_all.setToolTip("Load all data from the selected folders")
        self.data_selection_group.glayout.addWidget(self.btn_load_all)

        self.band_group = VHGroup('Bands / Maps', orientation='G')
        self.band_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Data', self.band_group.gbox, (1, 0, 1, 2))

        self.bands_to_load_list = QListWidget()
        self.bands_to_load_list.setToolTip("Select bands to load")
        self.bands_to_load_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.band_group.glayout.addWidget(self.bands_to_load_list, 0, 0, 1, 2)

        self.button_group_channels = QButtonGroup()
        self.radio_bands = QRadioButton('Bands')
        self.radio_bands.setChecked(True)
        self.radio_maps = QRadioButton('Maps')
        self.radio_rgb = QRadioButton('RGB')
        self.channel_buttons = [self.radio_bands, self.radio_maps, self.radio_rgb]
        self.button_group_channels.addButton(self.radio_bands, id=1)
        self.button_group_channels.addButton(self.radio_maps, id=2)
        self.button_group_channels.addButton(self.radio_rgb, id=3)
        self.band_group.glayout.addWidget(self.radio_bands, 1, 0, 1, 1)
        self.band_group.glayout.addWidget(self.radio_maps, 2, 0, 1, 1)
        self.band_group.glayout.addWidget(self.radio_rgb, 3, 0, 1, 1)

        self.btn_update_channels = QPushButton("Update loaded channels")
        self.btn_update_channels.setToolTip("Overwrite loaded channels with new selection")
        self.band_group.glayout.addWidget(self.btn_update_channels, 4, 0, 1, 2)

        self.render_group = VHGroup('Render', orientation='G')
        self.render_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Data', self.render_group.gbox, (2, 0, 1, 2))

        self.min_quantile = QDoubleSpinBox()
        self.min_quantile.setToolTip("Minimum quantile for contrast limits")
        self.min_quantile.setDecimals(2)
        self.min_quantile.setRange(0, 50)
        self.min_quantile.setSingleStep(0.1)
        self.min_quantile.setValue(2)
        self.render_group.glayout.addWidget(QLabel('Min quantile'), 0, 0, 1, 1)
        self.render_group.glayout.addWidget(self.min_quantile, 0, 1, 1, 1)

        self.max_quantile = QDoubleSpinBox()
        self.max_quantile.setToolTip("Maximum quantile for contrast limits")
        self.max_quantile.setDecimals(2)
        self.max_quantile.setRange(50, 100)
        self.max_quantile.setSingleStep(0.1)
        self.max_quantile.setValue(98)
        self.render_group.glayout.addWidget(QLabel('Max quantile'), 1, 0, 1, 1)
        self.render_group.glayout.addWidget(self.max_quantile, 1, 1, 1, 1)

        self.render_group.glayout.addWidget(QLabel('Select index file'))
        self.index_yml_path = FileEdit('r')
        self.render_group.glayout.addWidget(self.index_yml_path.native)

        self.params_plots_path = FileEdit('r')
        self.render_group.glayout.addWidget(QLabel('Select plots params file'))
        self.render_group.glayout.addWidget(self.params_plots_path.native)

        ## Concatenation tab
        self.data_order_group = VHGroup('Data order', orientation='G')
        self.data_order_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Concatenation', self.data_order_group.gbox, (0, 0, 1, 2))
        self._create_data_order_widget()
        self.data_order_group.glayout.addWidget(self.table_widget, 0, 0, 1, 2)

        self.spinbox_metadata_scale = QDoubleSpinBox()
        self.spinbox_metadata_scale.setToolTip("Units per pixel")
        self.spinbox_metadata_scale.setDecimals(4)
        self.spinbox_metadata_scale.setRange(0.0, 1000)
        self.spinbox_metadata_scale.setSingleStep(0.0001)
        self.spinbox_metadata_scale.setValue(1)
        self.data_order_group.glayout.addWidget(QLabel('Pixel Size'), 1, 0, 1, 1)
        self.data_order_group.glayout.addWidget(self.spinbox_metadata_scale, 1, 1, 1, 1)

        self.btn_apply_shifts = QPushButton("Apply vertical shifts")
        self.btn_apply_shifts.setToolTip("Apply shifts in the table to the data")
        self.data_order_group.glayout.addWidget(self.btn_apply_shifts, 2, 0, 1, 2)

        self.proj_group = VHGroup('Projection', orientation='G')
        self.proj_group.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Concatenation', self.proj_group.gbox, (3, 0, 1, 2))
        self.spinbox_proj_factor = QDoubleSpinBox()
        self.spinbox_proj_factor.setToolTip("Multiplicative factor for projection")
        self.spinbox_proj_factor.setDecimals(1)
        self.spinbox_proj_factor.setRange(0.1, 10000)
        self.spinbox_proj_factor.setSingleStep(1)
        self.spinbox_proj_factor.setValue(100)
        self.proj_group.glayout.addWidget(QLabel('Projection factor'), 0, 0, 1, 1)
        self.proj_group.glayout.addWidget(self.spinbox_proj_factor, 0, 1, 1, 1)
        self.spinbox_proj_shift = QSpinBox()
        self.spinbox_proj_shift.setToolTip("Horizontal shift in pixels for projection")
        self.spinbox_proj_shift.setRange(-1000, 1000)
        self.spinbox_proj_shift.setSingleStep(1)
        self.spinbox_proj_shift.setValue(0)
        self.proj_group.glayout.addWidget(QLabel('Projection shift'), 1, 0, 1, 1)
        self.proj_group.glayout.addWidget(self.spinbox_proj_shift, 1, 1, 1, 1)

        self.btn_assemble = QPushButton("Assemble")
        self.btn_assemble.setToolTip("Assemble the data and add it to the viewer")
        self.tabs.add_named_tab('Concatenation', self.btn_assemble, (4, 0, 1, 2))

        self.btn_export_shifts = QPushButton("Export shifts")
        self.tabs.add_named_tab('Export', self.btn_export_shifts, (0, 0, 1, 2))
        self.btn_import_shifts = QPushButton("Import shifts")
        self.tabs.add_named_tab('Export', self.btn_import_shifts, (1, 0, 1, 2))

        self.export_path_display = FileEdit('d')
        self.tabs.add_named_tab('Export', QLabel('Select export folder'), (2, 0, 1, 1))
        self.tabs.add_named_tab('Export', self.export_path_display.native, (2, 1, 1, 1))
        
        self.add_connections()

    def _create_data_order_widget(self):
        """
        Create a table with the list of data and corresponding shifts.
        The rows are draggable in order to change the order of the data.
        """

        if self.file_list.count() == 0:
            self.table_widget = DragDropTable(1, 4)
            self.table_widget.setHorizontalHeaderLabels(
                ["Dataset", "Shift in unit", "Top", "Bottom"])

            self.table_widget.setDragDropMode(QTableWidget.InternalMove)  # Enable drag-and-drop
            self.table_widget.setSelectionBehavior(QTableWidget.SelectRows)  # Select entire rows
            self.table_widget.setDragEnabled(True)
            self.table_widget.setDropIndicatorShown(True)
            return

        self.table_widget.setRowCount(0)  # Clear existing rows
        for row in range(self.file_list.count()):
            row_idx = self.table_widget.rowCount()
            self.table_widget.insertRow(row_idx) 

            file = self.file_list.item(row).text()
            self.table_widget.setItem(row, 0, QTableWidgetItem(file))
            self.table_widget.setItem(row, 1, QTableWidgetItem(str(0)))
            self.table_widget.setCellWidget(row, 2, QCheckBox())
            self.table_widget.setCellWidget(row, 3, QCheckBox())
            self.table_widget.cellWidget(row,2).setChecked(True)

        self.table_widget.cellWidget(row,3).setChecked(True)



    def add_connections(self):
        """Add callbacks"""

        self.main_path_display.changed.connect(self._on_click_select_main_folder)
        self.btn_load_all.clicked.connect(self._on_click_load_all)
        self.file_list.model().rowsInserted.connect(self._create_data_order_widget)
        self.btn_update_channels.clicked.connect(self.update_loaded_channels)
        self.btn_apply_shifts.clicked.connect(self._on_click_apply_shifts)
        self.spinbox_proj_factor.valueChanged.connect(self._on_update_projection)
        self.spinbox_proj_shift.valueChanged.connect(self._on_update_projection)
        self.btn_export_shifts.clicked.connect(self._on_click_export_shifts)
        self.btn_import_shifts.clicked.connect(self._on_click_load_shifts)

        self.btn_assemble.clicked.connect(self._on_click_assemble)

        self.radio_bands.toggled.connect(self._update_channel_list)
        self.radio_maps.toggled.connect(self._update_channel_list)
        self.radio_rgb.toggled.connect(self._update_channel_list)

        self.min_quantile.valueChanged.connect(self._on_update_contrast_limits)
        self.max_quantile.valueChanged.connect(self._on_update_contrast_limits)
        self.index_yml_path.changed.connect(self._on_load_index_yml)
        self.params_plots_path.changed.connect(self._on_load_params_plots_yml)

    def _on_click_select_main_folder(self, event=None, main_folder=None):
        """Update the file list when a new main folder is selected
        Connected to main_path_display widget"""
        
        if main_folder is not None:
            self.file_list.update_from_path(Path(main_folder))
            self.main_path_display.value = main_folder
        else:
            self.file_list.update_from_path(self.main_path_display.value)

    def _on_load_index_yml(self, event=None):

        if self.index_yml_path.value == Path('.'):
            self.indices = None
        else:
            self.indices = load_index_series(self.index_yml_path.value)

    def _on_load_params_plots_yml(self, event=None):
        
        if self.params_plots_path.value == Path('.'):
            self.params_plots = None
        else:
            self.params_plots = load_plots_params(self.params_plots_path.value)

    def _on_click_load_all(self, event=None, init_load=True, indices=None):
        """Import all data from the selected folders
        Connected to btn_load_all widget"""

        main_folder = Path(self.file_list.folder_path)

        if init_load:
            # load indices
            if self.index_yml_path.value == Path('.'):
                if main_folder.joinpath(self.get_file_list()[0], 'index_settings.yml').exists():
                    self.indices = load_index_series(main_folder.joinpath(self.get_file_list()[0], 'index_settings.yml'))
                    self.index_yml_path.value = main_folder.joinpath(self.get_file_list()[0], 'index_settings.yml')
                #else:
                #    raise ValueError('Select an index yml file')
            else:
                self.indices = load_index_series(self.index_yml_path.value)
            
            # load plots params
            if self.params_plots_path.value == Path('.'):
                if main_folder.joinpath(self.get_file_list()[0], 'params_plots.yml').exists():
                    self.params_plots = load_plots_params(main_folder.joinpath(self.get_file_list()[0], 'params_plots.yml'))
                    self.params_plots_path.value = main_folder.joinpath(self.get_file_list()[0], 'params_plots.yml')
                #else:
                #    raise ValueError('Select a plots params yml file')
            else:
                self.params_plots = load_plots_params(self.params_plots_path.value)

            # load params, set scale and channels
            params = load_project_params(main_folder.joinpath(self.get_file_list()[0]))
            self.spinbox_metadata_scale.setValue(params.scale)

            self._update_channel_list()

        # if no indices are provided, load the first band for raw data 
        # or the first map for maps
        if indices is None:
            bands = [0]
            map_ind = 0
        else:
            bands = indices
            map_ind = indices[0]

        # load the data and projections. For raw data, the projections are the median
        # of the data along the y-axis. For maps, the pre-computed projection is loaded
        self.proj = []
        
        for f in self.get_file_list():
            
            cmap = 'gray'
            max_width = 0
            current_folder = main_folder.joinpath(f)

            # load index map
            if self.radio_maps.isChecked():

                index_name = self.bands_to_load_list.item(map_ind).text()
                cube = load_index_zarr(project_folder=current_folder, main_roi_index=0, index_name=index_name)
                cube = cube[np.newaxis, ...]
                self.proj.append(load_projection(project_folder=current_folder, main_roi_index=0, index_name=index_name))
                
                if self.indices is not None:
                    cmap = Colormap(self.indices[index_name].colormap).identifier

            # load raw data  
            elif self.radio_bands.isChecked():

                cube = get_im_main_roi(
                    export_folder=current_folder, bands=bands, mainroi_index=0)
                
                self.proj.append(np.median(np.array(cube[0]), axis=1))

            elif self.radio_rgb.isChecked():
                 
                cube = self.get_rbg_cube(current_folder)
                self.proj.append(np.median(np.array(np.mean(cube,axis=0)), axis=1))

            # compute the maximum width of the data to place the projections
            max_width = max(max_width, cube.shape[2])

            # add the data to the viewer. Connect layers to the interactive translation
            if self.radio_bands.isChecked() or self.radio_maps.isChecked():
                if f in self.viewer.layers:
                    self.viewer.layers[f].data = [cube, cube[:,::2,::2]]
                    self.viewer.layers[f].colormap = cmap
                    self.viewer.layers[f].refresh()
                    self.viewer.layers[f].reset_contrast_limits()
                else:
                    self.viewer.add_image([cube, cube[:,::2,::2]], name=f, colormap=cmap)
                    self.viewer.layers[f].events.affine.connect(self._on_interactive_translate)
                
                if self.radio_maps.isChecked():
                    self.viewer.layers[f].contrast_limits = self.indices[index_name].index_map_range
                    
                self.viewer.layers[f].refresh()

            else:
                '''if self.params_plots.red_contrast_limits is None:
                    self.params_plots.red_contrast_limits = np.percentile(cube.ravel(), (2,98))
                    self.params_plots.green_contrast_limits = np.percentile(cube.ravel(), (2,98))
                    self.params_plots.blue_contrast_limits = np.percentile(cube.ravel(), (2,98))'''

                rgb_to_plot = create_rgb_image(
                    cube, red_contrast_limits=self.params_plots.red_contrast_limits, 
                    green_contrast_limits=self.params_plots.green_contrast_limits,
                    blue_contrast_limits=self.params_plots.blue_contrast_limits)
                
                if f in self.viewer.layers:
                    if self.viewer.layers[f].rgb:
                        self.viewer.layers[f].data = [rgb_to_plot, rgb_to_plot[::2,::2,:], rgb_to_plot[::4,::4,:]]
                        self.viewer.layers[f].refresh()
                    else:
                        self.viewer.layers.remove(f)
                
                if f not in self.viewer.layers:
                    self.viewer.add_image([
                        rgb_to_plot, rgb_to_plot[::2,::2, :], rgb_to_plot[::4,::4, :]],
                        name=f, rgb=True)
                    self.viewer.layers[f].events.affine.connect(self._on_interactive_translate)
            
        if self.radio_bands.isChecked():
            self._on_update_contrast_limits()

        self._on_click_apply_shifts()
        
        # set the range of the projection shift spinbox and add the projections to the viewer
        self.spinbox_proj_shift.setRange(-2 * max_width, 2 * max_width)
        self.spinbox_proj_shift.setValue(max_width)
        self._on_update_projection()

    def get_rbg_cube(self, folder):

        imagechannels = ImChannels(self.file_list.folder_path.joinpath(self.get_file_list()[0], 'corrected.zarr'))
        rgb_ch, rgb_names = imagechannels.get_indices_of_bands(self.params_plots.rgb_bands)
        cube = get_im_main_roi(
            export_folder=folder, bands=rgb_ch, mainroi_index=0)
        return cube

    def _on_update_contrast_limits(self, event=None):
        """Update the contrast limits of the layers"""

        all_cubes = np.array(np.concatenate([self.viewer.layers[f].data[0].ravel() for f in self.get_file_list()]))
        min_val = np.nanpercentile(all_cubes.ravel(), self.min_quantile.value())
        max_val = np.nanpercentile(all_cubes.ravel(), self.max_quantile.value())
        for f in self.get_file_list():
            self.viewer.layers[f].contrast_limits = (min_val, max_val)
            if self.radio_maps.isChecked():
                index_name = self.bands_to_load_list.selectedItems()[0].text()
                self.indices[index_name].index_map_range = self.viewer.layers[f].contrast_limits

    def get_file_list(self):

        file_list = []
        for i in range(self.file_list.count()):
            f = self.file_list.item(i).text()
            file_list.append(f)
        return file_list

    def _on_update_projection(self, event=None):
        """Add the projections to the viewer.
        self.spinbox_proj_factor.value() is a multiplicative factor for the projection to make it visible
        self.spinbox_proj_shift.value() is a horizontal shift in pixels for the projection"""
        
        if 'projection' not in self.viewer.layers:
            self.viewer.add_shapes(name='projection')
            edge_width = 0.5
        else:
            edge_width = self.viewer.layers['projection'].edge_width[0]

        proj_plot = []
        mean_proj = np.nanmean(np.concatenate(self.proj))
        for i, f in enumerate(self.get_file_list()):
            
            p = self.proj[i].copy()
            p_range = np.arange(0,len(p))
            p_range = p_range[~np.isnan(p)]
            p = p[~np.isnan(p)]
            
            if self.viewer.layers[f].rgb:
                translate = self.viewer.layers[f].affine.translate[0]
            else:
                translate = self.viewer.layers[f].affine.translate[1]
            proj_plot.append(np.array([p_range + translate, 
                                       self.spinbox_proj_factor.value() * (p - mean_proj) + self.spinbox_proj_shift.value()]).T)

        if len(self.viewer.layers['projection'].data) == 0: 
            self.viewer.layers['projection'].add(proj_plot, shape_type='path', edge_color='red', edge_width=edge_width)
        
        else:
            self.viewer.layers['projection'].data = proj_plot
        self.viewer.layers['projection'].refresh()
        

    def update_loaded_channels(self, event=None):
        """Update the loaded channels with the new selection"""
        
        selected_indexes = self.bands_to_load_list.selectedIndexes()
        indices = [index.row() for index in selected_indexes]
        if indices == []:
            indices = None
        self._on_click_load_all(event=None, init_load=False, indices=indices)

    def _update_channel_list(self):
        """Update channel list. Bands or computed zarr maps"""
        
        if self.radio_maps.isChecked():
            if self.indices is None:
                self._on_load_index_yml()
            if self.indices is None:
                warn('Select an index yml file')
                self.radio_bands.setChecked(True)
            else:
                self.clear_state()
                maps = list(self.indices.keys())
                for map in maps:
                    self.bands_to_load_list.addItem(map)
        
        elif self.radio_bands.isChecked():
            imagechannels = ImChannels(self.file_list.folder_path.joinpath(self.get_file_list()[0], 'corrected.zarr'))
            self.clear_state()
            # add new items
            for channel in imagechannels.channel_names:
                self.bands_to_load_list.addItem(channel)

        elif self.radio_rgb.isChecked():
            if self.params_plots is None:
                self._on_load_params_plots_yml()
            if self.params_plots is None:
                warn('Select a plots params yml file')
                self.radio_bands.setChecked(True)
            else:
                self.clear_state()

        '''elif self.radio_rgb.isChecked():

            if self.params_plots_path.value == Path('.'):
                raise ValueError('Select a plots params yml file')
            else:
                self.params_plots = load_plots_params(self.params_plots_path.value)'''

    def clear_state(self):

        # clear existing items
        self.bands_to_load_list.clear()
        for f in self.get_file_list():
            if f in self.viewer.layers:
                self.viewer.layers.remove(f)

    def _on_click_apply_shifts(self, event=None):
        """Apply the shifts in the table to the data.
        Uses the affine.translate property of the layers to shift the data"""

        total_shift = 0
        for i in range(self.table_widget.rowCount()):
            
            f = self.table_widget.item(i, 0).text()
            shift = float(self.table_widget.item(i, 1).text())
            total_shift += shift / self.spinbox_metadata_scale.value()
            
            if self.viewer.layers[f].rgb:
                self.viewer.layers[f].affine.translate = (total_shift, 0)
            else:
                self.viewer.layers[f].affine.translate = (0, total_shift, 0)
            self.viewer.layers[f].refresh()

        # move the projections
        self._on_update_projection()

    def _on_interactive_translate(self, event=None):
        """Update the table with the interactive translation of the layers"""
        
        previous_shift = 0
        for i in range(1, self.table_widget.rowCount()):

            f = self.table_widget.item(i, 0).text()

            if self.viewer.layers[f].rgb:
                translation = self.viewer.layers[f].affine.translate[0]
            else:
                translation = self.viewer.layers[f].affine.translate[1]
            shift = translation - previous_shift
            previous_shift = translation
            
            shift = shift * self.spinbox_metadata_scale.value()
            self.table_widget.item(i, 1).setText(str(shift)) 

        self._on_update_projection()

    def _on_click_export_shifts(self, event=None):
        """Export the shifts to a csv file"""

        if self.export_path_display.value == Path('.'):
            export_folder = Path(self.file_list.folder_path)
        else:
            export_folder = Path(self.export_path_display.value)

        shifts = []
        for i in range(self.table_widget.rowCount()):
            f = self.table_widget.item(i, 0).text()
            shift = float(self.table_widget.item(i, 1).text())
            shifts.append([f, shift])

        shifts_pd = pd.DataFrame(shifts, columns=['file', 'shift'])
        shifts_pd.to_csv(export_folder.joinpath('shifts.csv'), index=False)

    def _on_click_load_shifts(self, event=None):
        """Load the shifts from a csv file"""

        if self.export_path_display.value == Path('.'):
            export_folder = Path(self.file_list.folder_path)
        else:
            export_folder = Path(self.export_path_display.value)

        shifts_pd = pd.read_csv(export_folder.joinpath('shifts.csv'))
        for i, row in shifts_pd.iterrows():
            f = row['file']
            shift = row['shift']
            self.table_widget.setItem(i, 0, QTableWidgetItem(f))
            self.table_widget.setItem(i, 1, QTableWidgetItem(str(shift)))

    def _on_click_assemble(self, event=None):
        """Assemble the data and add it to the viewer. For sequential layers #1 and #2,
        for an overlap of the last N pixels of layer #1 and the first N pixels of layer #2,
        the overlap is taken from the layer #2."""

        '''if self.indices is None:
            raise ValueError('Load the index yml')
        if self.params_plots is None:
            if self.params_plots_path.value != Path('.'):
                self.params_plots = load_plots_params(self.params_plots_path.value)
            else:
                raise ValueError('Load the plots params yml')'''
        if self.export_path_display.value == Path('.'):
            raise ValueError('Select an export folder')
        
        main_folder = Path(self.file_list.folder_path)

        all_shifts = []
        for i in range(self.table_widget.rowCount()):
            all_shifts.append(float(self.table_widget.item(i, 1).text()))
        
        index_list = list(self.indices.keys())
        for b in range(len(index_list) + 1):
            all_cubes = []
            all_projections = []
            for i in range(self.table_widget.rowCount()):
                
                f = self.table_widget.item(i, 0).text()
                current_folder = main_folder.joinpath(f)

                if b < len(index_list):
                    index_name = index_list[b]
                    cube = load_index_zarr(project_folder=current_folder, main_roi_index=0, index_name=index_name)
                    all_cubes.append(cube[np.newaxis, ...])
                    all_projections.append(load_projection(project_folder=current_folder, main_roi_index=0, index_name=index_name))
                else:
                    all_cubes.append(self.get_rbg_cube(current_folder))
                    all_projections.append(np.median(np.array(np.mean(all_cubes[-1],axis=0)), axis=1))
            
            cols = np.array([a.shape[2] for a in all_cubes])
            max_cols = cols.max()
            to_pad = max_cols - cols
            for i in range(len(all_cubes)):
                all_cubes[i] = np.pad(all_cubes[i], ((0,0), (0,0), (0, to_pad[i])))
                #if i < len(all_cubes) - 1:

                shift_pix = int(all_shifts[i] / self.spinbox_metadata_scale.value())

                if i==0:
                    print(f'{i}: here1')
                    top_lim = 0
                else:
                    if (self.table_widget.cellWidget(i, 2).isChecked()):
                        print(f'{i}: here2')
                        top_lim = 0
                    else:
                        print(f'{i}: here3')
                        top_lim = all_cubes[i-1].shape[1] - shift_pix
                
                if i==len(all_cubes)-1:
                    print(f'{i}: here4')
                    bottom_lim = all_cubes[i].shape[1]
                else:
                    shift_pix_next = int(all_shifts[i+1] / self.spinbox_metadata_scale.value())
                    if (self.table_widget.cellWidget(i, 3).isChecked()):
                        print(f'{i}: here5')
                        bottom_lim = all_cubes[i].shape[1]
                    else:
                        print(f'{i}: here6')
                        if not self.table_widget.cellWidget(i+1, 2).isChecked():
                            raise ValueError('Either bottom of the current layer or top of the next layer must be selected')
                        bottom_lim = shift_pix_next
                print(f'top_lim: {top_lim}, bottom_lim: {bottom_lim}')
                all_cubes[i] = all_cubes[i][:, top_lim:bottom_lim, :]
                all_projections[i] = all_projections[i][top_lim:bottom_lim]
            full_map = np.concatenate(all_cubes, axis=1)
            full_proj = np.concatenate(all_projections, axis=0)

            if b == len(self.indices.keys()):
                rgb_map = create_rgb_image(
                    full_map,
                    red_contrast_limits=self.params_plots.red_contrast_limits, 
                    green_contrast_limits=self.params_plots.green_contrast_limits,
                    blue_contrast_limits=self.params_plots.blue_contrast_limits)
                
                tifffile.imwrite(Path(self.export_path_display.value).joinpath(f'concat_rgb.tiff'), 
                                 (255 * rgb_map).astype(np.uint8))
                
                if f'concat_RGB' in self.viewer.layers:
                    self.viewer.layers[f'concat_RGB'].data = [rgb_map, rgb_map[::2,::2,:], rgb_map[::4,::4,:]]
                else:
                    self.viewer.add_image([rgb_map, rgb_map[::2,::2,:], rgb_map[::4,::4,:]], name='concat_RGB', rgb=True)
                self.viewer.layers[f'concat_RGB'].refresh()

            else:
                full_map = full_map[0]
                z1 = save_zarr(full_map, Path(self.export_path_display.value).joinpath(f'concat_{index_name}.zarr'))
                save_tif_cmap(image=full_map,
                              image_path=Path(self.export_path_display.value).joinpath(f'concat_{index_name}.tiff'),
                              napari_cmap=self.indices[index_name].colormap,
                              contrast=self.indices[index_name].index_map_range, overwrite=False)

                cmap = Colormap(self.indices[index_name].colormap).identifier

                if f'{index_name}_full' in self.viewer.layers:
                    self.viewer.layers[f'{index_name}_full'].data = full_map
                else:
                    self.viewer.add_image(full_map, name=f'{index_name}_full', colormap=cmap)
                self.viewer.layers[f'{index_name}_full'].refresh()

                if f'{index_name}_full_proj' in self.viewer.layers:
                    self.viewer.layers[f'{index_name}_full_proj'].data = [full_proj]
                else:
                    self.viewer.add_shapes(name=f'{index_name}_full_proj')
                    self.viewer.layers[f'{index_name}_full_proj'].add(
                        np.array([np.arange(0,len(full_proj)), full_proj]).T,
                        shape_type='path', edge_color='red', edge_width=0.5)
                    self.viewer.layers[f'{index_name}_full_proj'].refresh()


class DragDropTable(QTableWidget):
    """Table widget with option to move rows by dragging and dropping"""

    def __init__(self, rows, columns):
        super().__init__(rows, columns)
        self.setDragDropMode(QTableWidget.InternalMove)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)

    def dropEvent(self, event):
        # Get source and target row indexes
        source_row = self.currentRow()
        target_index = self.indexAt(event.pos())
        if not target_index.isValid():
            return
        
        target_row = target_index.row()

        if source_row == target_row:
            return  # No need to move if source and target are the same
        
        # Save the row data from the source row
        source_row_data = []
        for col in range(self.columnCount()):
            if isinstance(self.cellWidget(source_row, col), QCheckBox):
                source_row_data.append(self.cellWidget(source_row, col).isChecked())
            else:
                source_row_data.append(self.item(source_row, col).text())

        # Remove the source row
        self.removeRow(source_row)
        
        # Adjust target row index if the source row was above the target
        if source_row < target_row:
            target_row -= 1

        # Insert the saved row data at the target row
        self.insertRow(target_row)
        for col, data in enumerate(source_row_data):
            if isinstance(data, str):
                self.setItem(target_row, col, QTableWidgetItem(data))
            elif isinstance(data, bool):
                self.setCellWidget(target_row, col, QCheckBox())
                self.cellWidget(target_row, col).setChecked(data)

        event.accept()



    

