from pathlib import Path
import matplotlib.pyplot as plt
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QGridLayout, QLineEdit,
                            QFileDialog, QCheckBox, QSpinBox, QLabel)
from qtpy.QtCore import Qt
from napari_guitils.gui_structures import TabSet
from napari_guitils.gui_structures import VHGroup

from .imchannels import ImChannels
from .folder_list_widget import FolderListWidget
from .sediproc import correct_save_to_zarr
from .io import get_data_background_path
from .widgets.channel_widget import ChannelWidget
from .parameters.parameters import Param

class BatchPreprocWidget(QWidget):
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
    
    def __init__(self, napari_viewer, 
                 destripe=False, background_correct=True, savgol_window=None, min_band=None, max_band=None):
        super().__init__()
        
        self.viewer = napari_viewer
        self.index_file = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ["&Preprocessing", "Paths"]
        self.tabs = TabSet(self.tab_names, tab_layouts=[None, QGridLayout()])

        self.tabs.widget(0).layout().setAlignment(Qt.AlignTop)
        self.tabs.widget(1).layout().setAlignment(Qt.AlignTop)

        self.main_layout.addWidget(self.tabs)

        self.data_selection_group = VHGroup('Select data', orientation='G')
        self.export_group = VHGroup('Export location', orientation='G')
        self.tabs.add_named_tab('&Preprocessing', self.data_selection_group.gbox)
        self.tabs.add_named_tab('&Preprocessing', self.export_group.gbox)

        self.btn_select_main_folder = QPushButton("Select main folder")
        self.data_selection_group.glayout.addWidget(self.btn_select_main_folder)
        self.main_path_display = QLineEdit("No path")
        self.data_selection_group.glayout.addWidget(self.main_path_display)
        self.data_selection_group.glayout.addWidget(QLabel('Available folders'))
        self.file_list = FolderListWidget(napari_viewer)
        self.data_selection_group.glayout.addWidget(self.file_list)
        self.file_list.setMaximumHeight(100)

        self.data_selection_group.glayout.addWidget(QLabel('Bands to display'))
        self.qlist_channels = ChannelWidget(self.viewer, translate=False)
        self.qlist_channels.itemClicked.connect(self._on_change_select_bands)
        self.data_selection_group.glayout.addWidget(self.qlist_channels)

        self.btn_select_preproc_export_folder = QPushButton("Select export folder")
        self.preproc_export_path_display = QLineEdit("No path")
        self.export_group.glayout.addWidget(self.btn_select_preproc_export_folder)
        self.export_group.glayout.addWidget(self.preproc_export_path_display)

        self.options_group = VHGroup('Options', orientation='G')
        self.tabs.add_named_tab('&Preprocessing', self.options_group.gbox)
        self.check_do_background_correction = QCheckBox("Background correction")
        self.check_do_background_correction.setChecked(background_correct)
        self.options_group.glayout.addWidget(self.check_do_background_correction, 0, 0, 1, 1)
        
        self.check_do_destripe = QCheckBox("Destripe")
        self.check_do_destripe.setChecked(destripe)
        self.options_group.glayout.addWidget(self.check_do_destripe, 1, 0, 1, 1)
        self.qspin_destripe_width = QSpinBox()
        self.qspin_destripe_width.setRange(1, 1000)
        if savgol_window is not None:
            self.qspin_destripe_width.setValue(savgol_window)
        else:
            self.qspin_destripe_width.setValue(100)
        self.options_group.glayout.addWidget(QLabel('Savgol Width'), 2, 0, 1, 1)
        self.options_group.glayout.addWidget(self.qspin_destripe_width, 2, 1, 1, 1)
        self.qspin_min_band = QSpinBox()
        self.qspin_min_band.setRange(0, 1000)
        if min_band is not None:
            self.qspin_min_band.setValue(min_band)
        else:
            self.qspin_min_band.setValue(0)

        self.check_do_min_max = QCheckBox("Crop bands")
        self.check_do_min_max.setChecked(False)
        self.options_group.glayout.addWidget(self.check_do_min_max, 3, 0, 1, 1)

        self.options_group.glayout.addWidget(QLabel('Min band'), 4, 0, 1, 1)
        self.options_group.glayout.addWidget(self.qspin_min_band, 4, 1, 1, 1)
        self.qspin_max_band = QSpinBox()
        self.qspin_max_band.setRange(0, 1000)
        if max_band is not None:
            self.qspin_max_band.setValue(max_band)
        else:
            self.qspin_max_band.setValue(1000)
        self.options_group.glayout.addWidget(QLabel('Max band'), 5, 0, 1, 1)
        self.options_group.glayout.addWidget(self.qspin_max_band, 5, 1, 1, 1)
        self.qspin_max_band.setEnabled(False)
        self.qspin_min_band.setEnabled(False)

        self.check_use_dask = QCheckBox("Use dask")
        self.check_use_dask.setChecked(True)
        self.check_use_dask.setToolTip("Use dask to parallelize computation")
        self.tabs.add_named_tab('&Preprocessing', self.check_use_dask)

        self.btn_preproc_folder = QPushButton("Preprocess")
        self.tabs.add_named_tab('&Preprocessing', self.btn_preproc_folder)

        self.selected_data_folder = QLineEdit("No path")
        self.selected_reference_folder = QLineEdit("No path")
        self.imhdr_path_display = QLineEdit("No file selected")
        self.white_file_path_display = QLineEdit("No file selected")
        self.dark_for_white_file_path_display = QLineEdit("No file selected")
        self.dark_for_im_file_path_display = QLineEdit("No file selected")
        self.tabs.add_named_tab('Paths', QLabel('Data folder'))
        self.tabs.add_named_tab('Paths', self.selected_data_folder)
        self.tabs.add_named_tab('Paths', QLabel('Reference folder'))
        self.tabs.add_named_tab('Paths', self.selected_reference_folder)
        self.tabs.add_named_tab('Paths', QLabel('hdr file'))
        self.tabs.add_named_tab('Paths', self.imhdr_path_display)
        self.tabs.add_named_tab('Paths', QLabel('White ref'))
        self.tabs.add_named_tab('Paths', self.white_file_path_display)
        self.tabs.add_named_tab('Paths', QLabel('Dark for white ref'))
        self.tabs.add_named_tab('Paths', self.dark_for_white_file_path_display)
        self.tabs.add_named_tab('Paths', QLabel('Darf for image ref'))
        self.tabs.add_named_tab('Paths', self.dark_for_im_file_path_display)

        self.add_connections()


    def add_connections(self):
        """Add callbacks"""

        self.btn_select_main_folder.clicked.connect(self._on_click_select_main_folder)
        self.btn_select_preproc_export_folder.clicked.connect(self._on_click_select_preproc_export_folder)
        self.btn_preproc_folder.clicked.connect(self._on_click_batch_correct)
        self.file_list.currentTextChanged.connect(self._on_change_filelist)
        self.check_do_min_max.stateChanged.connect(self._on_change_min_max)

    def _on_change_min_max(self, event=None):
        if self.check_do_min_max.isChecked():
            self.qspin_max_band.setEnabled(True)
            self.qspin_min_band.setEnabled(True)
        else:
            self.qspin_max_band.setEnabled(False)
            self.qspin_min_band.setEnabled(False)

    def _on_change_select_bands(self, event=None):

        self.qlist_channels._on_change_channel_selection()

    def _on_change_filelist(self):
        
        main_folder = Path(self.file_list.folder_path)
        if self.file_list.currentItem() is None:
            return
        current_folder = main_folder.joinpath(self.file_list.currentItem().text())

        background_text = '_WR_'
        acquistion_folder, wr_folder, white_file_path, dark_file_path, dark_for_im_file_path, imhdr_path = get_data_background_path(current_folder, background_text=background_text)
        wr_beginning = wr_folder.name.split(background_text)[0]

        self.selected_data_folder.setText(acquistion_folder.as_posix())
        self.selected_reference_folder.setText(wr_folder.as_posix())

        self.white_file_path = white_file_path
        self.dark_for_white_file_path = dark_file_path
        self.dark_for_im_file_path = dark_for_im_file_path
        self.imhdr_path = imhdr_path

        self.white_file_path_display.setText(self.white_file_path.as_posix())
        self.dark_for_white_file_path_display.setText(self.dark_for_white_file_path.as_posix())
        self.dark_for_im_file_path_display.setText(self.dark_for_im_file_path.as_posix())
        self.imhdr_path_display.setText(self.imhdr_path.as_posix())

        # clear existing layers.
        while len(self.viewer.layers) > 0:
            self.viewer.layers.clear()
        
        # if file list is empty stop here
        if self.imhdr_path is None:
            return False
        
        # open image
        self.imagechannels = ImChannels(self.imhdr_path)
        self.qlist_channels._update_channel_list(imagechannels=self.imagechannels)
        


    def _on_click_select_main_folder(self, event=None, main_folder=None):
        
        if main_folder is None:
            main_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        else:
            main_folder = Path(main_folder)
        self.main_path_display.setText(main_folder.as_posix())
        self.file_list.update_from_path(main_folder)

    def _on_click_select_preproc_export_folder(self, event=None, preproc_export_path=None):
        """Interactively select folder to analyze"""

        if preproc_export_path is None:
            self.preproc_export_path = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        else:
            self.preproc_export_path = Path(preproc_export_path)
        self.preproc_export_path_display.setText(self.preproc_export_path.as_posix())


    def _on_click_select_data_folder(self, event=None, data_folder=None):
        """Interactively select folder to analyze"""

        if data_folder is None:
            self.data_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        else:
            self.data_folder = Path(data_folder)
        self.data_path_display.setText(self.data_folder.as_posix())

    def _on_click_batch_correct(self, event=None):

        background_text = '_WR_'

        if self.preproc_export_path is None:
            self._on_click_select_preproc_export_folder()

        main_folder = Path(self.file_list.folder_path)
        for c in range(self.file_list.count()):
            f = self.file_list.item(c).text()
            current_folder = main_folder.joinpath(f)

            acquistion_folder, wr_folder, white_file_path, dark_for_white_file_path, dark_for_im_file_path, imhdr_path = get_data_background_path(current_folder, background_text=background_text)
            wr_beginning = wr_folder.name.split(background_text)[0]
            export_folder = self.preproc_export_path.joinpath(f)#wr_beginning)

            if not export_folder.is_dir():
                export_folder.mkdir()

            min_max_band = None
            if self.check_do_min_max.isChecked():
                min_band = self.qspin_min_band.value()
                max_band = self.qspin_max_band.value()
                min_max_band = [min_band, max_band]

            param = Param(
                project_path=export_folder,
                file_path=imhdr_path,
                white_path=white_file_path,
                dark_for_im_path=dark_for_im_file_path,
                dark_for_white_path=dark_for_white_file_path,
                main_roi=[],
                rois=[])
            
            correct_save_to_zarr(
                imhdr_path=imhdr_path,
                white_file_path=white_file_path,
                dark_for_im_file_path=dark_for_im_file_path,
                dark_for_white_file_path=dark_for_white_file_path,
                zarr_path=export_folder.joinpath('corrected.zarr'),
                band_indices=None,
                min_max_bands=min_max_band,
                background_correction=self.check_do_background_correction.isChecked(),
                destripe=self.check_do_destripe.isChecked(),
                use_dask=self.check_use_dask.isChecked(),
                )
            imchannels = ImChannels(export_folder.joinpath('corrected.zarr'))
            param.main_roi = [[
                0, 0,
                imchannels.nrows, 0,
                imchannels.nrows, imchannels.ncols,
                0, imchannels.ncols
                ]]
            param.save_parameters()
