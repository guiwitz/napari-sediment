from pathlib import Path
import warnings
import numpy as np
from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget,
                            QLabel, QFileDialog, QSlider,
                            QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,)
from qtpy.QtCore import Qt
from superqt import QDoubleSlider

from napari.utils import progress, DirectLabelColormap
from superqt import QLabeledDoubleRangeSlider
from spectral.algorithms import calc_stats, mnf, noise_from_diffs, remove_continuum
from scipy.signal import savgol_filter
import zarr
import pandas as pd

from ..data_structures.parameters import Param
from ..data_structures.parameters_endmembers import ParamEndMember
from ..utilities.io import load_project_params, load_endmember_params, save_image_to_zarr
from ..data_structures.imchannels import ImChannels
from ..widget_utilities.spectralplotter import SpectralPlotter
from ..widget_utilities.channel_widget import ChannelWidget
from ..utilities.io import load_mask, get_mask_path
from ..widget_utilities.rgb_widget import RGBWidget
from ..utilities.utils import wavelength_to_rgb
from ..utilities.hyperanalysis import (compute_vertical_correlations, compute_end_members,
                            reduce_with_mnf, export_dim_reduction_data)
from ..utilities.sediproc import custom_ppi
from napari_guitils.gui_structures import TabSet, VHGroup


class HyperAnalysisWidget(QWidget):
    """Widget for the hyperanalysis plugin."""
    
    def __init__(self, napari_viewer):
        super().__init__()
        
        self.viewer = napari_viewer
        self.params = Param()
        self.export_folder = None

        self.var_init()

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['Main', 'Reduction', 'PPI','End Members', 'Plotting']
        self.tabs = TabSet(self.tab_names)

        self.main_layout.addWidget(self.tabs)

        # loading tab
        self.files_group = VHGroup('Files and Folders', orientation='G')
        self.tabs.add_named_tab('Main', self.files_group.gbox)
        self.btn_select_export_folder = QPushButton("Set Project folder")
        self.export_path_display = QLineEdit("No path")
        self.files_group.glayout.addWidget(self.btn_select_export_folder, 1, 0, 1, 1)
        self.files_group.glayout.addWidget(self.export_path_display, 1, 1, 1, 1)
        self.btn_load_project = QPushButton("Import project")
        self.files_group.glayout.addWidget(self.btn_load_project, 2, 0, 1, 1)
        self.check_load_corrected = QCheckBox("Load corrected data")
        self.check_load_corrected.setChecked(True)
        self.files_group.glayout.addWidget(self.check_load_corrected, 2, 1, 1, 1)
        self.btn_save_index_project = QPushButton("Export Project")
        self.files_group.glayout.addWidget(self.btn_save_index_project, 3, 0, 1, 1)
        self.spin_selected_roi = QSpinBox()
        self.spin_selected_roi.setRange(0, 0)
        self.spin_selected_roi.setValue(0)
        self.spin_selected_roi_current = 0
        self.files_group.glayout.addWidget(QLabel('Selected ROI'), 4, 0, 1, 1)
        self.files_group.glayout.addWidget(self.spin_selected_roi, 4, 1, 1, 1)

        # channel selection
        self.main_group = VHGroup('Bands', orientation='G')
        self.tabs.add_named_tab('Main', self.main_group.gbox)

        self.main_group.glayout.addWidget(QLabel('Bands to load'), 0, 0, 1, 2)
        self.qlist_channels = ChannelWidget(self.viewer)
        self.qlist_channels.itemClicked.connect(self._on_change_select_bands)
        self.main_group.glayout.addWidget(self.qlist_channels, 1,0,1,2)
        self.btn_select_all = QPushButton("Select all")
        self.main_group.glayout.addWidget(self.btn_select_all, 2, 0, 1, 2)

        self.rgbwidget = RGBWidget(viewer=self.viewer, translate=False)
        self.tabs.add_named_tab('Main', self.rgbwidget.rgbmain_group.gbox)

        #self.process_group_io = VHGroup('IO', orientation='G')
        #self.tabs.add_named_tab('Main', self.process_group_io.gbox)
        #self.btn_save_index_project = QPushButton("Export Project")
        #self.process_group_io.glayout.addWidget(self.btn_save_index_project)
        #self.btn_load_index_project = QPushButton("Load index project")
        #self.process_group_io.glayout.addWidget(self.btn_load_index_project)

         # Plot tab
        self.scan_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.tabs.add_named_tab('Plotting', self.scan_plot)

        self._add_processing_tab()
        self._add_ppi_tab()
        self._add_endmember_tab()
        self.add_connections()

        # mouse
        self.viewer.mouse_move_callbacks.append(self._shift_move_callback)

    def var_init(self):

        self.params_endmembers = ParamEndMember()
        self.eigen_line = None
        self.corr_line = None
        self.corr_limit_line = None
        self.ppi_boundary_lines = None
        self.selected_bands = None
        self.end_members = None
        self.end_members_raw = None
        self.eigenvals = None
        self.mnfr = None
        self.spectral_pixel = None
        self.pure_members = None
        self.all_coef = []

    def _add_processing_tab(self):

        self.tabs.widget(self.tab_names.index('Reduction')).layout().setAlignment(Qt.AlignTop)
        # processing tab
        self.process_group_mnfr = VHGroup('Spectral reduction', orientation='G')
        self.tabs.add_named_tab('Reduction', self.process_group_mnfr.gbox)

        self.btn_mnfr = QPushButton("Compute MNF")
        self.process_group_mnfr.glayout.addWidget(self.btn_mnfr, 0, 0, 1, 2)

        self.reduce_on_eigen_group = VHGroup('Reduce on eigenvalues', orientation='G')
        self.process_group_mnfr.glayout.addWidget(self.reduce_on_eigen_group.gbox)
        self.btn_reduce_mnfr = QPushButton("Reduce on eigenvalues")
        self.reduce_on_eigen_group.glayout.addWidget(self.btn_reduce_mnfr, 0, 0, 1, 2)
        self.spin_eigen_threshold = QDoubleSpinBox()
        self.spin_eigen_threshold.setRange(0, 10)
        self.spin_eigen_threshold.setSingleStep(0.01)
        self.spin_eigen_threshold.setValue(1)
        self.reduce_on_eigen_group.glayout.addWidget(QLabel('Eigenvalue threshold'), 1, 0, 1, 1)
        self.reduce_on_eigen_group.glayout.addWidget(self.spin_eigen_threshold, 1, 1, 1, 1)
        
        self.reduce_on_correlation_group = VHGroup('Reduce on correlation', orientation='G')
        self.process_group_mnfr.glayout.addWidget(self.reduce_on_correlation_group.gbox)
        self.btn_reduce_correlation = QPushButton("Reduce on correlation")
        self.reduce_on_correlation_group.glayout.addWidget(self.btn_reduce_correlation, 0, 0, 1, 2)
        self.spin_correlation_threshold = QDoubleSpinBox()
        self.spin_correlation_threshold.setRange(0, 1)
        self.spin_correlation_threshold.setSingleStep(0.01)
        self.spin_correlation_threshold.setValue(0.0)
        self.reduce_on_correlation_group.glayout.addWidget(QLabel('Correlation threshold'), 1, 0, 1, 1)
        self.reduce_on_correlation_group.glayout.addWidget(self.spin_correlation_threshold, 1, 1, 1, 1)

        # eigen tab
        self.eigen_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.tabs.add_named_tab('Reduction', self.eigen_plot)
        self.corr_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.tabs.add_named_tab('Reduction', self.corr_plot)
        self.slider_corr_limit = QSlider(Qt.Horizontal)
        self.slider_corr_limit.setRange(0, 1000)
        self.slider_corr_limit.setValue(1000)
        self.tabs.add_named_tab('Reduction', self.slider_corr_limit)
        

    def _add_ppi_tab(self):

        self.process_group_ppi = VHGroup('PPI', orientation='G')
        self.tabs.add_named_tab('PPI', self.process_group_ppi.gbox)

        self.tabs.widget(self.tab_names.index('PPI')).layout().setAlignment(Qt.AlignTop)


        self.ppi_threshold = QSpinBox()
        self.ppi_threshold.setRange(0, 100)
        self.ppi_threshold.setSingleStep(1)
        self.ppi_threshold.setValue(10)
        self.process_group_ppi.glayout.addWidget(QLabel('Threshold PPI counts'), 0, 0, 1, 1)
        self.process_group_ppi.glayout.addWidget(self.ppi_threshold, 0, 1, 1, 1)


        self.ppi_proj_threshold = QDoubleSpinBox()
        self.ppi_proj_threshold.setRange(0, 1)
        self.ppi_proj_threshold.setSingleStep(0.1)
        self.ppi_proj_threshold.setValue(0)
        self.process_group_ppi.glayout.addWidget(QLabel('Threshold PPI projection'), 1, 0, 1, 1)
        self.process_group_ppi.glayout.addWidget(self.ppi_proj_threshold, 1, 1, 1, 1)

        self.ppi_iterations = QSpinBox()
        self.ppi_iterations.setRange(0, 100000)
        self.ppi_iterations.setSingleStep(1)
        self.ppi_iterations.setValue(5000)
        self.process_group_ppi.glayout.addWidget(QLabel('Iterations'), 2, 0, 1, 1)
        self.process_group_ppi.glayout.addWidget(self.ppi_iterations, 2, 1, 1, 1)
        
        self.btn_ppi = QPushButton("PPI")
        self.process_group_ppi.glayout.addWidget(self.btn_ppi, 3, 0, 1, 2)

    def _add_endmember_tab(self):

        self.process_group_endmember = VHGroup('End-members', orientation='G')
        self.tabs.add_named_tab('End Members', self.process_group_endmember.gbox)

        self.btn_update_endmembers = QPushButton("Compute end-members")
        self.process_group_endmember.glayout.addWidget(self.btn_update_endmembers, 0, 0, 1, 2)
        self.qspin_endm_eps = QDoubleSpinBox()
        self.qspin_endm_eps.setRange(0, 10)
        self.qspin_endm_eps.setSingleStep(0.1)
        self.qspin_endm_eps.setValue(0.5)
        self.process_group_endmember.glayout.addWidget(QLabel('DBSCAN eps'), 1, 0, 1, 1)
        self.process_group_endmember.glayout.addWidget(self.qspin_endm_eps, 1, 1, 1, 1)

        self.ppi_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.tabs.add_named_tab('End Members', self.ppi_plot)

        self.plot_options_group = VHGroup('Options', orientation='G')
        self.tabs.add_named_tab('End Members', self.plot_options_group.gbox)

        self.check_remove_continuum = QCheckBox("Remove continuum")
        self.check_remove_continuum.setChecked(True)
        self.plot_options_group.glayout.addWidget(self.check_remove_continuum, 0, 0, 1, 2)

        self.slider_spectrum_savgol = QDoubleSlider(Qt.Horizontal)
        self.slider_spectrum_savgol.setRange(1, 100)
        self.slider_spectrum_savgol.setSingleStep(1)
        self.slider_spectrum_savgol.setSliderPosition(5)
        self.plot_options_group.glayout.addWidget(QLabel('Savitzky-Golay filter window'), 1, 0, 1, 1)
        self.tabs.add_named_tab('End Members', self.slider_spectrum_savgol)
        self.plot_options_group.glayout.addWidget(self.slider_spectrum_savgol, 1, 1, 1, 1)


    def add_connections(self):
        """Add callbacks"""

        self.btn_select_export_folder.clicked.connect(self._on_click_select_export_folder)
        self.btn_load_project.clicked.connect(self.import_project)
        self.btn_select_all.clicked.connect(self._on_click_select_all)
        self.spin_selected_roi.valueChanged.connect(self.load_data)
        self.btn_mnfr.clicked.connect(self._on_click_mnfr)
        self.btn_reduce_mnfr.clicked.connect(self._on_click_reduce_mnfr_on_eigen)
        self.btn_reduce_correlation.clicked.connect(self._on_click_reduce_mnfr_on_correlation)
        self.btn_ppi.clicked.connect(self._on_click_ppi)
        self.btn_save_index_project.clicked.connect(self.save_index_project)
        #self.btn_load_index_project.clicked.connect(self.import_index_project)
        self.slider_corr_limit.valueChanged.connect(self._on_change_corr_limit)
        self.btn_update_endmembers.clicked.connect(self._on_click_update_endmembers)
        self.check_remove_continuum.stateChanged.connect(self.update_endmembers)
        self.slider_spectrum_savgol.valueChanged.connect(self.update_endmembers)

        cid = self.eigen_plot.canvas.mpl_connect('button_release_event', self._on_interactive_eigen_threshold)
        cid2 = self.corr_plot.canvas.mpl_connect('button_release_event', self._on_interactive_corr_threshold)

    def _on_click_select_export_folder(self, event=None, alternate_path=None):
        """Interactively select folder to analyze"""

        if alternate_path is not None:
            self.export_folder = Path(alternate_path)
        else:
            return_path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            if return_path == '':
                return
            self.export_folder = Path(return_path)

        self.export_path_display.setText(self.export_folder.as_posix())

    def import_project(self):
        """Import pre-processed project: corrected roi and mask"""
        
        if self.export_folder is None:
            self._on_click_select_export_folder()

        self.params = load_project_params(folder=self.export_folder)

        self.imhdr_path = Path(self.params.file_path)
        #self.white_file_path = Path(self.params.white_path)
        #self.dark_for_im_path = Path(self.params.dark_for_im_path)
        #self.dark_for_white_path = Path(self.params.dark_for_white_path)

        self.mainroi = np.array([np.array(x).reshape(4,2) for x in self.params.main_roi]).astype(int)
        self.rois = [np.array([np.array(x).reshape(4,2) for x in roi]).astype(int) for roi in self.params.rois]
        
        self.spin_selected_roi.setRange(0, len(self.mainroi)-1)
        self.load_data()


    def load_data(self, new_val=None):
        """Load data for selected ROI"""

        roi_folder = self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}')
        if not roi_folder.exists():
            warnings.warn('No data for selected ROI')
            self.spin_selected_roi.valueChanged.disconnect(self.load_data)
            self.spin_selected_roi.setValue(self.spin_selected_roi_current)
            self.spin_selected_roi.valueChanged.connect(self.load_data)
            return
        
        self.spin_selected_roi_current = self.spin_selected_roi.value()
        self.corr_plot.axes.clear()
        self.eigen_plot.axes.clear()
        self.ppi_plot.axes.clear()
        
        to_remove = [l.name for l in self.viewer.layers if l.name not in ['imcube', 'red', 'green', 'blue']]
        for r in to_remove:
            self.viewer.layers.remove(r)
        self.var_init()
        
        curremt_sub_roi = 0# to be used to select the correct sub-roi in future
        self.row_bounds = [
            self.rois[self.spin_selected_roi.value()][curremt_sub_roi,:,0].min(),
            self.rois[self.spin_selected_roi.value()][curremt_sub_roi,:,0].max()]
        self.col_bounds = [
            self.rois[self.spin_selected_roi.value()][curremt_sub_roi,:,1].min(),
            self.rois[self.spin_selected_roi.value()][curremt_sub_roi,:,1].max()]
        
        self._on_click_load_mask()

        if self.check_load_corrected.isChecked():
            self.imagechannels = ImChannels(self.export_folder.joinpath('corrected.zarr'))
        else:
            self.imagechannels = ImChannels(self.imhdr_path)
        self.qlist_channels._update_channel_list(imagechannels=self.imagechannels)
        self.rgbwidget.imagechannels = self.imagechannels
        self.rgbwidget.rgb = self.params.rgb
        self.rgbwidget.row_bounds = self.row_bounds
        self.rgbwidget.col_bounds = self.col_bounds
        self.rgbwidget._on_click_RGB()

        self._on_click_select_all()
        if self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}').joinpath('Parameters_indices.yml').exists():
            self.import_index_project()


    def import_index_project(self):
        """Import pre-processed project (corrected roi and mask) as well as denoised
        and/or reduced stacks"""

        #if self.export_folder is None:
        #    self._on_click_select_export_folder()
        export_path = Path(self.export_folder).joinpath(f'roi_{self.spin_selected_roi.value()}')

        # import main project
        #self.import_project()
        
        # load index parameters
        self.params_endmembers = load_endmember_params(folder=export_path)

        # update selected channels
        for i in range(self.params_endmembers.min_max_channel[0], self.params_endmembers.min_max_channel[1]+1):
            self.qlist_channels.item(i).setSelected(True)
        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)

        # load stacks
        self.load_stacks()
        self.load_plots()

        # set UI setting using index parameters
        if self.params_endmembers.eigen_threshold is not None:
            self.spin_eigen_threshold.setValue(self.params_endmembers.eigen_threshold)
        if self.params_endmembers.correlation_threshold is not None:
            self.spin_correlation_threshold.setValue(self.params_endmembers.correlation_threshold)
        self.ppi_iterations.setValue(self.params_endmembers.ppi_iterations)
        self.ppi_threshold.setValue(self.params_endmembers.ppi_threshold)
        self.slider_corr_limit.setRange(0,len(self.all_coef))
        if self.params_endmembers.corr_limit is not None:
            self.slider_corr_limit.setValue(self.params_endmembers.corr_limit)

        # add plots
        self.plot_eigenvals()
        self.plot_correlation()

        if self.end_members is not None:
            self.plot_endmembers()
            self.map_colors_endmembers_ppi()
        else:
            if 'pure' in self.viewer.layers:
                self.compute_end_members()
                self.plot_endmembers()
                self.map_colors_endmembers_ppi()


    def save_index_project(self):
        """Save parameters and stacks related to denoising/reduction"""

        export_path = Path(self.export_folder).joinpath(f'roi_{self.spin_selected_roi.value()}')
        self.params_endmembers.project_path = export_path.as_posix()
        self.params_endmembers.min_max_channel = [
            int(self.qlist_channels.channel_indices[0]),
            int(self.qlist_channels.channel_indices[-1])]
        self.params_endmembers.eigen_threshold = float(self.spin_eigen_threshold.value())
        self.params_endmembers.correlation_threshold = float(self.spin_correlation_threshold.value())
        self.params_endmembers.ppi_iterations = self.ppi_iterations.value()
        self.params_endmembers.ppi_threshold = self.ppi_threshold.value()
        self.params_endmembers.corr_limit = self.slider_corr_limit.value()

        self.params_endmembers.save_parameters()
        self.save_stacks()
        self.save_plots()

    def save_stacks(self):
        """Save denoised and reduced staks to zarr"""

        export_path = Path(self.export_folder).joinpath(f'roi_{self.spin_selected_roi.value()}')

        layer_names = ['mnf', 'denoised', 'pure', 'pure_members']
        for lname in layer_names:
            if lname in self.viewer.layers:
                save_image_to_zarr(
                    image=self.viewer.layers[lname].data,
                    zarr_path=export_path.joinpath(f'{lname}.zarr')
                )
    
    def load_stacks(self):
        """Load denoised and reduced staks from zarr"""

        export_path = Path(self.export_folder).joinpath(f'roi_{self.spin_selected_roi.value()}') 
        for name in ['mnf', 'denoised']:
            if export_path.joinpath(f'{name}.zarr').is_dir():
                im = np.array(zarr.open_array(export_path.joinpath(f'{name}.zarr')))
                self.image_mnfr = np.moveaxis(im, 0, 2)
                self.viewer.add_image(im, name=name)
        
        if export_path.joinpath('pure.zarr').is_dir():
            im = np.array(zarr.open_array(export_path.joinpath('pure.zarr')))
            self.viewer.add_labels(im, name='pure')

        if export_path.joinpath('pure_members.zarr').is_dir():
            im = np.array(zarr.open_array(export_path.joinpath('pure_members.zarr')))
            self.viewer.add_labels(im, name='pure_members')
    
    def save_plots(self):
        """Save plots to csv"""

        export_path = Path(self.export_folder).joinpath(f'roi_{self.spin_selected_roi.value()}') 

        export_dim_reduction_data(export_path,
                                  eigenvals=self.eigenvals,
                                  all_coef=self.all_coef,
                                  end_members=self.end_members,
                                  bands_used=self.qlist_channels.bands)
        
    def load_plots(self):
        """Load csv files"""
        
        export_path = Path(self.export_folder).joinpath(f'roi_{self.spin_selected_roi.value()}') 
        if export_path.joinpath('eigenvalues.csv').is_file():
            self.eigenvals = pd.read_csv(export_path.joinpath('eigenvalues.csv')).eigenvalues.values
            self.plot_eigenvals()
        if export_path.joinpath('correlation.csv').is_file():
            self.all_coef = pd.read_csv(export_path.joinpath('correlation.csv')).correlation.values
            self.plot_correlation()
        if export_path.joinpath('end_members.csv').is_file():
            self.end_members = pd.read_csv(export_path.joinpath('end_members.csv')).values
            self.end_members = self.end_members[:,:-1]
            self.plot_endmembers()
    

    def _on_click_load_mask(self):
        """Load mask from file"""
        
        main_roi_row_min = self.mainroi[self.spin_selected_roi.value()][:,0].min()
        main_roi_col_min = self.mainroi[self.spin_selected_roi.value()][:,1].min()

        mask_path = get_mask_path(self.export_folder.joinpath(f'roi_{self.spin_selected_roi.value()}'))
        if mask_path.is_file():
            mask = load_mask(mask_path)
            mask = mask[self.row_bounds[0]-main_roi_row_min:self.row_bounds[1]-main_roi_row_min,
                        self.col_bounds[0]-main_roi_col_min:self.col_bounds[1]-main_roi_col_min]
        else:
            mask = np.zeros(
                shape=(self.row_bounds[1]-self.row_bounds[0], self.col_bounds[1]-self.col_bounds[0]),
                dtype=np.uint8)

        if 'mask' in self.viewer.layers:
            self.viewer.layers['mask'].data = mask
        else:
            self.viewer.add_labels(mask, name='mask')

    def _on_change_select_bands(self, event=None):

        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)

    def _on_click_select_all(self, event=None):
        self.qlist_channels.selectAll()
        self.qlist_channels._on_change_channel_selection(self.row_bounds, self.col_bounds)

    def _on_click_mnfr(self):
        """Compute MNF transform and compute vertical correlation. Keep all bands."""
        
        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Computing MNF")
            data = np.asarray(np.moveaxis(self.viewer.layers['imcube'].data,0,2), np.float32)
            signal = calc_stats(
                image=data,
                mask=self.viewer.layers['mask'].data,
                index=0)
            noise = noise_from_diffs(data)
            self.mnfr = mnf(signal, noise)
            self.eigenvals = self.mnfr.napc.eigenvalues

            self._compute_mnfr_bands()
            self._compute_vert_correlation()
            self.plot_eigenvals()
        self.viewer.window._status_bar._toggle_activity_dock(False)

    def plot_eigenvals(self):
        """Display eigenvalues from MNFR transform"""

        self.eigen_plot.axes.clear()
        self.eigen_plot.axes.plot(self.eigenvals)
        self.eigen_plot.axes.set_title('Eigenvalues', fontdict={'color':'black'})
            
        self.eigen_plot.canvas.figure.canvas.draw()

    def _compute_mnfr_bands(self):
        """Extract actual MNFR bands and plot them"""

        data = np.asarray(np.moveaxis(self.viewer.layers['imcube'].data,0,2), np.float32)
        self.image_mnfr = self.mnfr.reduce(data, num=data.shape[2])#, num=last_index)

        if 'mnf' in self.viewer.layers:
            self.viewer.layers['mnf'].data = np.moveaxis(self.image_mnfr, 2, 0)
        else:
            self.viewer.add_image(np.moveaxis(self.image_mnfr, 2, 0), name='mnf', rgb=False)


    def _on_click_reduce_mnfr_on_eigen(self):
        """Select bands based on eigenvalues. As a general rule, keep 
        bands with eigenvalues > 1.0."""
        
        last_index = np.arange(0,len(self.eigenvals))[self.eigenvals > self.spin_eigen_threshold.value()][-1]
        self.selected_bands = self.image_mnfr[:,:, 0:last_index].copy()
        if 'denoised' in self.viewer.layers:
            self.viewer.layers['denoised'].data = np.moveaxis(self.selected_bands, 2, 0)
        else:
            self.viewer.add_image(np.moveaxis(self.selected_bands, 2, 0), name='denoised', rgb=False)


    def _compute_vert_correlation(self):
        """Compute correlation between lines within each band."""

        self.all_coef = compute_vertical_correlations(self.image_mnfr)
        self.slider_corr_limit.setRange(0, len(self.all_coef))
        self.slider_corr_limit.setValue(len(self.all_coef))

        self.plot_correlation()

    def plot_correlation(self):
        """Plot correlation between lines within each band."""

        self.corr_plot.axes.clear()
        self.corr_plot.axes.plot(self.all_coef)#, linewidth=0.1, markersize=0.5)
        self.corr_plot.axes.set_title('Line correlation', fontdict={'color':'black'})
        self.corr_plot.canvas.figure.canvas.draw()
        

    def _on_click_reduce_mnfr_on_correlation(self):
        """Select bands based on correlation between lines. As a general rule, keep
        bands with correlated signal >0 meaning there are structures in the image."""

        if 'mnf' not in self.viewer.layers:
            raise ValueError('Must compute MNF first.')
        
        self.selected_bands = reduce_with_mnf(im_mnf=self.image_mnfr,
                                              corr_coefficients=self.all_coef,
                                              corr_threshold=self.spin_correlation_threshold.value(),
                                              max_index=self.slider_corr_limit.value())

        if 'denoised' in self.viewer.layers:
            self.viewer.layers['denoised'].data = np.moveaxis(self.selected_bands, 2, 0)
        else:
            self.viewer.add_image(np.moveaxis(self.selected_bands, 2, 0), name='denoised', rgb=False)
        self.viewer.layers['denoised'].refresh()

    def _on_click_ppi(self):
        """Find pure pixels using PPI algorithm."""

        if self.selected_bands is None:
            raise ValueError('Must reduce bands first')

        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Pure pixel detection")
            # create image where masked pixels are set to 0. This avoids detecting masked pixels as pure pixels.
            im_masked = np.moveaxis(self.selected_bands.copy(),2,0)
            im_masked[:, self.viewer.layers['mask'].data == 1] = 0
            im_masked = np.moveaxis(im_masked,0,2)

            #pure = ppi(im_masked, niters=self.ppi_iterations.value(), display=0)
            #from ..utilities.sediproc import custom_ppi
            pure, self.total_ppi_series = custom_ppi(
                im_masked,
                niters=self.ppi_iterations.value(),
                threshold=self.ppi_proj_threshold.value(),
            )
            if 'pure' in self.viewer.layers:
                self.viewer.layers['pure'].data = pure
            else:
                self.viewer.add_labels(pure, name='pure')
            
            self._on_click_update_endmembers()
        self.viewer.window._status_bar._toggle_activity_dock(False)

    def _on_click_update_endmembers(self, event=None):
        """Update end-members based on threshold."""

        self.compute_end_members()
        self.plot_endmembers()
        self.map_colors_endmembers_ppi()

    def compute_end_members(self):
        """"Cluster the pure pixels and compute average end-members."""
        
        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Compute end-members")
        
            pure = np.asarray(self.viewer.layers['pure'].data)
            imcube_data = np.asarray(self.viewer.layers['imcube'].data)
            im_cube_denoised = self.viewer.layers['denoised'].data

            self.end_members_raw, self.end_members_labels = compute_end_members(
                pure=pure, 
                im_cube=imcube_data,
                im_cube_denoised=im_cube_denoised, 
                ppi_threshold=self.ppi_threshold.value(),
                dbscan_eps=self.qspin_endm_eps.value())
            
            self.update_endmembers()
            self.pure_members = np.zeros_like(pure)
            self.pure_members[pure > self.ppi_threshold.value()] = self.end_members_labels+1
            if 'pure_members' in self.viewer.layers:
                self.viewer.layers['pure_members'].data = self.pure_members
            else:
                self.viewer.add_labels(self.pure_members, name='pure_members')
            self.viewer.layers['pure_members'].refresh()
            
        self.viewer.window._status_bar._toggle_activity_dock(False)

    def map_colors_endmembers_ppi(self):

        lines = self.ppi_plot.axes.get_lines()
        line_colors = [[0,0,0]] + [line.get_color() for line in lines]
        color_dict = {ind: line_colors[ind] for ind in range(0, self.viewer.layers['pure_members'].data.max()+1)}
        color_dict[None] = [1,0,0]
        self.viewer.layers['pure_members'].colormap = DirectLabelColormap(color_dict=color_dict)

    def update_endmembers(self, event=None):

        if self.end_members_raw is None:
            return
        
        endmember = self.end_members_raw
        if self.check_remove_continuum.isChecked(): 
            endmember = remove_continuum(self.end_members_raw.T, self.qlist_channels.bands)
            endmember = endmember.T

        filter_window = int(self.slider_spectrum_savgol.value())
        if filter_window > 3:
            endmember = savgol_filter(endmember, window_length=filter_window, polyorder=3, axis=0)

        self.end_members = endmember

        self.plot_endmembers()

    
    def plot_endmembers(self, event=None):
        """Cluster the pure pixels and plot the endmembers as average of clusters."""

        self.ppi_plot.axes.clear()
        self.ppi_plot.axes.plot(self.qlist_channels.bands, self.end_members)

        out = wavelength_to_rgb(self.qlist_channels.bands.min(), self.qlist_channels.bands.max(), 100)
        ax_histx = self.ppi_plot.axes.inset_axes([0.0,-0.5, 1.0, 1], sharex=self.ppi_plot.axes)
        ax_histx.imshow(out, extent=(self.qlist_channels.bands.min(),self.qlist_channels.bands.max(), 0,10))
        ax_histx.set_axis_off()

        self.ppi_plot.axes.set_xlabel('Wavelength', color='black')
        self.ppi_plot.axes.set_ylabel('Continuum removed', color='black')
        self.ppi_plot.axes.tick_params(axis='both', colors='black')
        self.ppi_plot.canvas.figure.patch.set_facecolor('white')
        self.ppi_plot.canvas.figure.canvas.draw()



    def _shift_move_callback(self, viewer, event):
        """Receiver for napari.viewer.mouse_move_callbacks, checks for 'Shift' event modifier.
        If event contains 'Shift' and layer attribute contains napari layers the cursor position is written to the
        cursor_pos attribute and the _draw method is called afterwards.
        """

        nrows = self.viewer.layers['imcube'].data.shape[1]
        ncols = self.viewer.layers['imcube'].data.shape[2]
        if 'Shift' in event.modifiers and self.viewer.layers:
            self.cursor_pos = np.rint(self.viewer.cursor.position).astype(int)
            
            #self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.row_bounds[1]-self.row_bounds[0]-1)
            #self.cursor_pos[2] = np.clip(self.cursor_pos[2], 0, self.col_bounds[1]-self.col_bounds[0]-1)
            #self.cursor_pos[1] = np.clip(self.cursor_pos[1], self.row_bounds[0],self.row_bounds[1]-1)
            #self.cursor_pos[2] = np.clip(self.cursor_pos[2], self.col_bounds[0],self.col_bounds[1]-1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0,nrows-1)
            self.cursor_pos[2] = np.clip(self.cursor_pos[2], 0,ncols-1)
            self.spectral_pixel = self.viewer.layers['imcube'].data[
                #:, self.cursor_pos[1]-self.row_bounds[0], self.cursor_pos[2]-self.col_bounds[0]
                :, self.cursor_pos[1], self.cursor_pos[2]
            ]
            self.update_spectral_plot()

    def update_spectral_plot(self, event=None):
            
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
            spectral_pixel = savgol_filter(spectral_pixel, window_length=filter_window, polyorder=3)

        self.scan_plot.axes.plot(self.qlist_channels.bands, spectral_pixel)
        
        self.scan_plot.canvas.figure.canvas.draw()

    def _on_interactive_eigen_threshold(self, event):
        """Select eigenvalue threshold by clicking on the eigenvalue plot."""
        
        if event.inaxes:
            self.spin_eigen_threshold.setValue(event.ydata)

            selected = self.eigenvals > event.ydata
            if len(selected) > 0:
                last_index = np.arange(0,len(self.eigenvals))[self.eigenvals > event.ydata][-1]
            else:
                raise ValueError(f'No bands with eigenvals > {event.ydata}')

            if self.eigen_line is not None:
                num_lines = len(self.eigen_line)
                for i in range(num_lines):
                    self.eigen_line.pop(0).remove()
            self.eigen_line = self.eigen_plot.axes.plot(
                [[0, last_index], [len(self.eigenvals), last_index]], 
                [[event.ydata, self.eigenvals.min()],[event.ydata,self.eigenvals.max()]],'r')

            self.eigen_plot.canvas.figure.canvas.draw()


    def _on_interactive_corr_threshold(self, event):
        """Select correlation threshold by clicking on the correlation plot."""
        
        if event.inaxes:
            self.spin_correlation_threshold.setValue(event.ydata)

            acceptable_range = np.arange(self.slider_corr_limit.value())
            accepted_corr = self.all_coef[acceptable_range]
            accepted_indices = acceptable_range[accepted_corr > self.spin_correlation_threshold.value()]
            if len(accepted_indices) > 0:
                last_index = acceptable_range[accepted_corr > self.spin_correlation_threshold.value()][-1]
            else:
                raise ValueError(f'No bands with correlation > {self.spin_correlation_threshold.value()}')
        

            if self.corr_line is not None:
                num_lines = len(self.corr_line)
                for i in range(num_lines):
                    self.corr_line.pop(0).remove()
            self.corr_line = self.corr_plot.axes.plot(
                [[0, last_index], [len(self.all_coef), last_index]],
                [[event.ydata, self.all_coef.min()], [event.ydata, self.all_coef.max()]], 'r')
                 
            
            self.corr_plot.canvas.figure.canvas.draw()

    def _on_change_corr_limit(self, event):
        """Set a limit to the last band to consider for when using the correlation threshold."""

        if self.corr_limit_line is not None:
            self.corr_limit_line.pop(0).remove()
        
        self.corr_limit_line = self.corr_plot.axes.plot(
            self.slider_corr_limit.value()*np.ones(2), [self.all_coef.min(), self.all_coef.max()], 'b')
        self.corr_plot.canvas.figure.canvas.draw()
