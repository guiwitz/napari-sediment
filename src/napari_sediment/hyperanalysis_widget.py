from pathlib import Path
import numpy as np
from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget,
                            QLabel, QFileDialog, QListWidget, QAbstractItemView,
                            QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,)
from qtpy.QtCore import Qt
from spectral import open_image
from spectral.algorithms import calc_stats, mnf, noise_from_diffs, remove_continuum
from spectral.algorithms import ppi


from .parameters import Param
from .io import load_params_yml
from .imchannels import ImChannels
from .sediproc import white_dark_correct
from ._reader import read_spectral
from .spectralplot import SpectralPlotter, SelectRange
from .channel_widget import ChannelWidget
from .io import load_mask, get_mask_path
from napari_guitils.gui_structures import TabSet, VHGroup


class HyperAnalysisWidget(QWidget):
    """Widget for the hyperanalysis plugin."""
    
    def __init__(self, napari_viewer):
        super().__init__()
        
        self.viewer = napari_viewer
        self.params = Param()

        self.export_folder = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['Main', 'Processing', 'Eigenvalues','PPI', 'Plotting']
        self.tabs = TabSet(self.tab_names)

        self.main_layout.addWidget(self.tabs)

        # loading tab
        self.files_group = VHGroup('Select', orientation='G')
        self.tabs.add_named_tab('Main', self.files_group.gbox)
        self.btn_select_export_folder = QPushButton("Select project folder")
        self.export_path_display = QLineEdit("No path")
        self.files_group.glayout.addWidget(self.btn_select_export_folder, 1, 0, 1, 1)
        self.files_group.glayout.addWidget(self.export_path_display, 1, 1, 1, 1)
        self.btn_load_project = QPushButton("Load project")
        self.files_group.glayout.addWidget(self.btn_load_project, 2, 0, 1, 2)
        self.check_load_corrected = QCheckBox("Load corrected data")
        self.check_load_corrected.setChecked(True)

        # channel selection
        self.main_group = VHGroup('Select', orientation='G')
        self.tabs.add_named_tab('Main', self.main_group.gbox)

        self.main_group.glayout.addWidget(QLabel('Channels to load'), 0, 0, 1, 2)
        self.qlist_channels = ChannelWidget(self)
        self.main_group.glayout.addWidget(self.qlist_channels, 1,0,1,2)
        self.btn_select_all = QPushButton("Select all")
        self.main_group.glayout.addWidget(self.btn_select_all, 2, 0, 1, 1)

         # Plot tab
        self.scan_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.tabs.add_named_tab('Plotting', self.scan_plot)

        # eigen tab
        self.eigen_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.eigen_sel = SelectRange(parent=None, ax=self.eigen_plot.axes)
        self.tabs.add_named_tab('Eigenvalues', self.eigen_plot)
        self.coef_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.tabs.add_named_tab('Eigenvalues', self.coef_plot)

        self.ppi_plot = SpectralPlotter(napari_viewer=self.viewer)
        self.tabs.add_named_tab('PPI', self.ppi_plot)


        self._add_processing_tab()
        self.add_connections()

        # mouse
        self.viewer.mouse_move_callbacks.append(self._shift_move_callback)

    def _add_processing_tab(self):

        self.tabs.widget(self.tab_names.index('Processing')).layout().setAlignment(Qt.AlignTop)
        # processing tab
        self.process_group = VHGroup('Process Hypercube', orientation='G')
        self.tabs.add_named_tab('Processing', self.process_group.gbox)
        self.process_group_mnfr = VHGroup('MNFR', orientation='G')
        self.tabs.add_named_tab('Processing', self.process_group_mnfr.gbox)
        self.process_group_ppi = VHGroup('PPI', orientation='G')
        self.tabs.add_named_tab('Processing', self.process_group_ppi.gbox)

        self.btn_destripe = QPushButton("Destripe")
        self.process_group.glayout.addWidget(self.btn_destripe)
        self.btn_white_correct = QPushButton("White correct")
        self.process_group.glayout.addWidget(self.btn_white_correct)
        self.btn_mnfr = QPushButton("MNFR")
        self.process_group_mnfr.glayout.addWidget(self.btn_mnfr)
        self.btn_reduce_mnfr = QPushButton("Reduce MNFR")
        self.process_group_mnfr.glayout.addWidget(self.btn_reduce_mnfr)
        self.spin_mnfr_threshold = QDoubleSpinBox()
        self.spin_mnfr_threshold.setRange(0, 1)
        self.spin_mnfr_threshold.setSingleStep(0.01)
        self.spin_mnfr_threshold.setValue(0.99)
        self.process_group_mnfr.glayout.addWidget(self.spin_mnfr_threshold)

        
        self.ppi_threshold = QSpinBox()
        self.ppi_threshold.setRange(0, 100)
        self.ppi_threshold.setSingleStep(1)
        self.ppi_threshold.setValue(10)
        self.process_group_ppi.glayout.addWidget(QLabel('Threshold PPI counts'), 0, 0, 1, 1)
        self.process_group_ppi.glayout.addWidget(self.ppi_threshold, 0, 1, 1, 1)
        self.ppi_iterations = QSpinBox()
        self.ppi_iterations.setRange(0, 10000)
        self.ppi_iterations.setSingleStep(1)
        self.ppi_iterations.setValue(5000)
        self.process_group_ppi.glayout.addWidget(QLabel('Iterations'), 2, 0, 1, 1)
        self.process_group_ppi.glayout.addWidget(self.ppi_iterations, 2, 1, 1, 1)
        self.btn_ppi = QPushButton("PPI")
        self.process_group_ppi.glayout.addWidget(self.btn_ppi, 3, 0, 1, 2)



    def add_connections(self):
        """Add callbacks"""

        self.btn_select_export_folder.clicked.connect(self._on_click_select_export_folder)
        self.btn_load_project.clicked.connect(self.import_project)
        self.btn_select_all.clicked.connect(self._on_click_select_all)
        self.btn_white_correct.clicked.connect(self._on_click_white_correct)
        self.btn_mnfr.clicked.connect(self._on_click_mnfr)
        self.btn_reduce_mnfr.clicked.connect(self._on_click_reduce_mnfr)
        self.btn_ppi.clicked.connect(self._on_click_ppi)

    def _on_click_select_export_folder(self):
        """Interactively select folder to analyze"""

        self.export_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        self.export_path_display.setText(self.export_folder.as_posix())

    def load_params(self):
        
        self.params = Param(project_path=self.export_folder)
        self.params = load_params_yml(self.params)

    def import_project(self):
        
        if self.export_folder is None:
            self._on_click_select_export_folder()

        self.load_params()

        self.imhdr_path = Path(self.params.file_path)
        self.white_file_path = Path(self.params.white_path)
        self.dark_file_path = Path(self.params.dark_path)

        self.mainroi = np.array([np.array(x).reshape(4,2) for x in self.params.main_roi]).astype(int)
        self.rois = np.array([np.array(x).reshape(4,2) for x in self.params.rois]).astype(int)
        self.row_bounds = [self.rois[0][:,0].min(), self.rois[0][:,0].max()]
        self.col_bounds = [self.rois[0][:,1].min(), self.rois[0][:,1].max()]

        self._on_click_load_mask()

        if self.check_load_corrected.isChecked():
            self.imagechannels = ImChannels(self.export_folder.joinpath('corrected.zarr'))
        else:
            self.imagechannels = ImChannels(self.imhdr_path)
        self.qlist_channels._update_channel_list()


    def _on_click_load_mask(self):
        """Load mask from file"""
        
        mask = load_mask(get_mask_path(self.export_folder))[self.row_bounds[0]:self.row_bounds[1], self.col_bounds[0]:self.col_bounds[1]]
        if 'mask' in self.viewer.layers:
            self.viewer.layers['mask'].data = mask
        else:
            self.viewer.add_labels(mask, name='mask')

    def _on_click_select_all(self):
        self.qlist_channels.selectAll()
        self.qlist_channels._on_change_channel_selection()

    def _on_click_white_correct(self, event):
        """White correct image"""

        img_white = open_image(self.white_file_path)
        img_dark = open_image(self.dark_file_path)

        white_data, _ = read_spectral(self.white_file_path, self.channel_indices, [0, img_white.nrows], self.col_bounds)
        dark_data, _ = read_spectral(self.dark_file_path, self.channel_indices, [0, img_dark.nrows], self.col_bounds)
            
        im_corr = white_dark_correct(self.viewer.layers['imcube'].data, white_data, dark_data)

        if 'imcube_corrected' in self.viewer.layers:
            self.viewer.layers['imcube_corrected'].data = im_corr
        else:
            self.viewer.add_image(im_corr, name='imcube_corrected', rgb=False)

    def _on_click_mnfr(self):

        signal = calc_stats(
            image=np.moveaxis(self.viewer.layers['imcube'].data,0,2),
            mask=self.viewer.layers['mask'].data,
            index=0)
        noise = noise_from_diffs(np.moveaxis(self.viewer.layers['imcube'].data,0,2))
        self.mnfr = mnf(signal, noise)
        self.eigenvals = self.mnfr.napc.eigenvalues

        self.eigen_plot.axes.clear()
        self.eigen_plot.axes.plot(self.mnfr.napc.eigenvalues)
            
        self.eigen_plot.canvas.figure.canvas.draw()

    def _on_click_reduce_mnfr(self):
        
        last_index = np.arange(0,len(self.eigenvals))[self.eigenvals > self.spin_mnfr_threshold.value()][-1]
        denoised = self.mnfr.reduce(np.moveaxis(self.viewer.layers['imcube'].data,0,2), num=last_index)
        
        all_coef = []
        for i in range(denoised.shape[2]):
            
            im = denoised[1::,:,i]
            im_shift = denoised[0:-1,:,i]
            
            all_coef.append(np.corrcoef(im.flatten(), im_shift.flatten())[0,1])

        self.all_coef = all_coef
        self.coef_plot.axes.clear()
        self.coef_plot.axes.plot(all_coef)#, linewidth=0.1, markersize=0.5)
        self.coef_plot.canvas.figure.canvas.draw()
        
        large_coeff = np.arange(len(all_coef))[np.array(all_coef) > 0.88]
        selected_bands = denoised[:,:, large_coeff].copy()
        self.selected_bands = selected_bands
        if 'denoised' in self.viewer.layers:
            self.viewer.layers['denoised'].data = np.moveaxis(selected_bands, 2, 0)
        else:
            self.viewer.add_image(np.moveaxis(selected_bands, 2, 0), name='denoised', rgb=False)
        self.viewer.layers['denoised'].refresh()

    def _on_click_ppi(self):

        self.pure = ppi(self.selected_bands, niters=self.ppi_iterations.value(), display=0)
        if 'pure' in self.viewer.layers:
            self.viewer.layers['pure'].data = self.pure
        else:
            self.viewer.add_labels(self.pure, name='pure')
        
        vects = self.viewer.layers['imcube'].data[:,self.pure > self.ppi_threshold.value()]

        im = open_image(self.imhdr_path)
        band_centers = np.array(im.bands.centers)[self.channel_indices]
        out = remove_continuum(spectra=vects.T, bands=band_centers)
        self.ppi_plot.axes.clear()
        self.ppi_plot.axes.plot(band_centers, out.T)#, linewidth=0.1, markersize=0.5)
        self.ppi_plot.canvas.figure.canvas.draw()


    def _shift_move_callback(self, viewer, event):
        """Receiver for napari.viewer.mouse_move_callbacks, checks for 'Shift' event modifier.
        If event contains 'Shift' and layer attribute contains napari layers the cursor position is written to the
        cursor_pos attribute and the _draw method is called afterwards.
        """

        if 'Shift' in event.modifiers and 'imcube' in self.viewer.layers:
            self.cursor_pos = np.rint(self.viewer.cursor.position).astype(int)
            
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], self.row_bounds[0],self.row_bounds[1]-1)
            self.cursor_pos[2] = np.clip(self.cursor_pos[2], self.col_bounds[0],self.col_bounds[1]-1)
            spectral_pixel = self.viewer.layers['imcube'].data[
                :, self.cursor_pos[1]-self.row_bounds[0], self.cursor_pos[2]-self.col_bounds[0]
            ]

            self.scan_plot.axes.clear()
            self.scan_plot.axes.plot(spectral_pixel)
            
            self.scan_plot.canvas.figure.canvas.draw()