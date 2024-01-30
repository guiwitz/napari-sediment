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

        self.tab_names = ["&Preprocessing", "Options"]
        self.tabs = TabSet(self.tab_names, tab_layouts=[None, QGridLayout()])

        self.tabs.widget(0).layout().setAlignment(Qt.AlignTop)
        self.tabs.widget(1).layout().setAlignment(Qt.AlignTop)

        self.main_layout.addWidget(self.tabs)

        self.btn_select_main_folder = QPushButton("Select main folder")
        self.tabs.add_named_tab('&Preprocessing', self.btn_select_main_folder)
        self.main_path_display = QLineEdit("No path")
        self.tabs.add_named_tab('&Preprocessing', self.main_path_display)

        self.file_list = FolderListWidget(napari_viewer)
        self.tabs.add_named_tab('&Preprocessing', self.file_list)
        self.file_list.setMaximumHeight(100)

        self.selected_data_folder = QLineEdit("No path")
        self.selected_reference_folder = QLineEdit("No path")
        self.imhdr_path_display = QLineEdit("No file selected")
        self.white_file_path_display = QLineEdit("No file selected")
        self.dark_for_white_file_path_display = QLineEdit("No file selected")
        self.dark_for_im_file_path_display = QLineEdit("No file selected")
        self.tabs.add_named_tab('&Preprocessing', self.selected_data_folder)
        self.tabs.add_named_tab('&Preprocessing', self.selected_reference_folder)
        self.tabs.add_named_tab('&Preprocessing', self.imhdr_path_display)
        self.tabs.add_named_tab('&Preprocessing', self.white_file_path_display)
        self.tabs.add_named_tab('&Preprocessing', self.dark_for_white_file_path_display)
        self.tabs.add_named_tab('&Preprocessing', self.dark_for_im_file_path_display)

        self.btn_select_preproc_export_folder = QPushButton("Select export folder")
        self.preproc_export_path_display = QLineEdit("No path")
        self.tabs.add_named_tab('&Preprocessing', self.btn_select_preproc_export_folder)
        self.tabs.add_named_tab('&Preprocessing', self.preproc_export_path_display)
        self.btn_preproc_folder = QPushButton("Preprocess")
        self.tabs.add_named_tab('&Preprocessing', self.btn_preproc_folder)

        self.qlist_channels = ChannelWidget(self.viewer, translate=False)
        self.qlist_channels.itemClicked.connect(self._on_change_select_bands)
        self.tabs.add_named_tab('&Preprocessing', self.qlist_channels)

        self.options_group = VHGroup('Options', orientation='G')
        self.tabs.add_named_tab('&Preprocessing', self.options_group.gbox)
        self.check_do_destripe = QCheckBox("Destripe")
        self.check_do_destripe.setChecked(destripe)
        self.options_group.glayout.addWidget(self.check_do_destripe, 0, 0, 1, 1)
        self.check_do_background_correction = QCheckBox("Background correction")
        self.check_do_background_correction.setChecked(background_correct)
        self.options_group.glayout.addWidget(self.check_do_background_correction, 1, 0, 1, 1)
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
        self.options_group.glayout.addWidget(QLabel('Min band'), 3, 0, 1, 1)
        self.options_group.glayout.addWidget(self.qspin_min_band, 3, 1, 1, 1)
        self.qspin_max_band = QSpinBox()
        self.qspin_max_band.setRange(0, 1000)
        if max_band is not None:
            self.qspin_max_band.setValue(max_band)
        else:
            self.qspin_max_band.setValue(1000)
        self.options_group.glayout.addWidget(QLabel('Max band'), 4, 0, 1, 1)
        self.options_group.glayout.addWidget(self.qspin_max_band, 4, 1, 1, 1)

        self.add_connections()


    def add_connections(self):
        """Add callbacks"""

        self.btn_select_main_folder.clicked.connect(self._on_click_select_main_folder)
        self.btn_select_preproc_export_folder.clicked.connect(self._on_click_select_preproc_export_folder)
        self.btn_preproc_folder.clicked.connect(self._on_click_batch_correct)
        self.file_list.currentTextChanged.connect(self._on_change_filelist)

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

        if not self.preproc_export_path.joinpath(wr_beginning).is_dir():
            self.preproc_export_path.joinpath(wr_beginning).mkdir()

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
            main_folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        else:
            main_folder = Path(main_folder)
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

            '''min_band = np.argmin(np.abs(np.array(self.imagechannels.channel_names).astype(float) - self.slider_batch_wavelengths.value()[0]))
            max_band = np.argmin(np.abs(np.array(self.imagechannels.channel_names).astype(float) - self.slider_batch_wavelengths.value()[1]))
            bands_to_correct = np.arange(min_band, max_band+1)'''
            
            correct_save_to_zarr(
                imhdr_path=imhdr_path,
                white_file_path=white_file_path,
                dark_for_im_file_path=dark_for_im_file_path,
                dark_for_white_file_path=dark_for_white_file_path,
                zarr_path=self.preproc_export_path.joinpath(wr_beginning).joinpath('corrected.zarr'),
                band_indices=None,
                background_correction=True,#self.check_batch_white.isChecked(),
                destripe=False,#self.check_batch_destripe.isChecked(),
                use_dask=True
                )