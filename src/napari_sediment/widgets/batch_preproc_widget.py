from pathlib import Path
import matplotlib.pyplot as plt
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QGridLayout, QLineEdit,
                            QFileDialog, QCheckBox, QSpinBox, QLabel)
from qtpy.QtCore import Qt
from magicgui.widgets import FileEdit
from napari.utils import progress
from napari_guitils.gui_structures import TabSet
from napari_guitils.gui_structures import VHGroup

from ..data_structures.imchannels import ImChannels
from ..widget_utilities.folder_list_widget import FolderListWidget
from ..utilities.sediproc import correct_save_to_zarr
from ..utilities.io import get_data_background_path
from ..widget_utilities.channel_widget import ChannelWidget
from ..utilities.batch_preproc import batch_preprocessing


class BatchPreprocWidget(QWidget):
    """
    Widget for the SpectralIndices.
    
    Parameters
    ----------
    napari_viewer: napari.Viewer
        napari viewer
    destripe: bool
        If True, apply destriping
    background_correct: bool
        If True, apply background correction
    savgol_window: int
        Width of the savgol filter
    min_band: int
        Minimum band to crop
    max_band: int
        Maximum band to crop
    chunk_size: int
        Chunk size for zarr saving

    Attributes
    ----------
    viewer: napari.Viewer
        napari viewer
    
    
    """
    
    def __init__(self, napari_viewer, 
                 destripe=False, background_correct=True, savgol_window=None,
                 min_band=None, max_band=None, chunk_size=500):
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

        # Pre-processing tab
        self.data_selection_group = VHGroup('Select data', orientation='G')
        self.export_group = VHGroup('Export location', orientation='G')
        self.tabs.add_named_tab('&Preprocessing', self.data_selection_group.gbox)
        self.tabs.add_named_tab('&Preprocessing', self.export_group.gbox)

        self.data_selection_group.glayout.addWidget(QLabel('Select main folder'))
        self.main_path_display = FileEdit('d')
        self.data_selection_group.glayout.addWidget(self.main_path_display.native)
        self.data_selection_group.glayout.addWidget(QLabel('Available folders'))
        self.file_list = FolderListWidget(napari_viewer)
        self.data_selection_group.glayout.addWidget(self.file_list)
        self.file_list.setMaximumHeight(100)

        self.data_selection_group.glayout.addWidget(QLabel('Bands to display'))
        self.qlist_channels = ChannelWidget(self.viewer, translate=False)
        self.qlist_channels.itemClicked.connect(self._on_change_select_bands)
        self.data_selection_group.glayout.addWidget(self.qlist_channels)

        self.preproc_export_path_display = FileEdit('d')
        self.export_group.glayout.addWidget(QLabel('Select export folder'))
        self.export_group.glayout.addWidget(self.preproc_export_path_display.native)

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

        self.spin_chunksize = QSpinBox()
        self.spin_chunksize.setRange(1, 10000)
        self.spin_chunksize.setValue(chunk_size)
        self.options_group.glayout.addWidget(QLabel('Chunk size'), 6, 0, 1, 1)
        self.options_group.glayout.addWidget(self.spin_chunksize, 6, 1, 1, 1)

        self.check_use_dask = QCheckBox("Use dask")
        self.check_use_dask.setChecked(True)
        self.check_use_dask.setToolTip("Use dask to parallelize computation")
        self.tabs.add_named_tab('&Preprocessing', self.check_use_dask)

        self.check_save_as_float = QCheckBox("Save as floats")
        self.check_save_as_float.setChecked(True)
        self.check_save_as_float.setToolTip("Save data as floats. Otherwise convert to integers after multiplication by 4096.")
        self.tabs.add_named_tab('&Preprocessing', self.check_save_as_float)

        self.btn_preproc_folder = QPushButton("Preprocess")
        self.tabs.add_named_tab('&Preprocessing', self.btn_preproc_folder)

        # Paths tab
        self.textbox_background_keyword = QLineEdit('_WR_')
        self.textbox_background_keyword.setToolTip("Keyword to identify background files")
        self.tabs.add_named_tab('Paths', QLabel('Background keyword'), (0, 0, 1, 1))
        self.tabs.add_named_tab('Paths', self.textbox_background_keyword, (0, 1, 1, 1))

        self.selected_data_folder = FileEdit('r')
        self.selected_reference_folder = FileEdit('r')
        self.imhdr_path_display = FileEdit('r')
        self.white_file_path_display = FileEdit('r')
        self.dark_for_white_file_path_display = FileEdit('r')
        self.dark_for_im_file_path_display = FileEdit('r')
        self.tabs.add_named_tab('Paths', QLabel('Data folder'))
        self.tabs.add_named_tab('Paths', self.selected_data_folder.native)
        self.tabs.add_named_tab('Paths', QLabel('Reference folder'))
        self.tabs.add_named_tab('Paths', self.selected_reference_folder.native)
        self.tabs.add_named_tab('Paths', QLabel('hdr file'))
        self.tabs.add_named_tab('Paths', self.imhdr_path_display.native)
        self.tabs.add_named_tab('Paths', QLabel('White ref'))
        self.tabs.add_named_tab('Paths', self.white_file_path_display.native)
        self.tabs.add_named_tab('Paths', QLabel('Dark for white ref'))
        self.tabs.add_named_tab('Paths', self.dark_for_white_file_path_display.native)
        self.tabs.add_named_tab('Paths', QLabel('Darf for image ref'))
        self.tabs.add_named_tab('Paths', self.dark_for_im_file_path_display.native)

        self.add_connections()


    def add_connections(self):
        """Add callbacks"""

        self.main_path_display.changed.connect(self._on_click_select_main_folder)
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

        background_text = self.textbox_background_keyword.text()
        acquistion_folder, wr_folder, white_file_path, dark_file_path, dark_for_im_file_path, imhdr_path = get_data_background_path(current_folder, background_text=background_text)
        wr_beginning = wr_folder.name.split(background_text)[0]

        self.selected_data_folder.value = acquistion_folder.as_posix()
        self.selected_reference_folder.value = wr_folder.as_posix()

        self.white_file_path = white_file_path
        self.dark_for_white_file_path = dark_file_path
        self.dark_for_im_file_path = dark_for_im_file_path
        self.imhdr_path = imhdr_path

        self.white_file_path_display.value = self.white_file_path.as_posix()
        self.dark_for_white_file_path_display.value = self.dark_for_white_file_path.as_posix()
        self.dark_for_im_file_path_display.value = self.dark_for_im_file_path.as_posix()
        self.imhdr_path_display.value = self.imhdr_path.as_posix()

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
        
        if main_folder is not None:
            self.file_list.update_from_path(Path(main_folder))
            self.main_path_display.value = main_folder
        else:
            self.file_list.update_from_path(self.main_path_display.value)

    def _on_click_select_data_folder(self, event=None, data_folder=None):
        """Interactively select folder to analyze"""

        if data_folder is None:
            self.data_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        else:
            self.data_folder = Path(data_folder)
        self.data_path_display.setText(self.data_folder.as_posix())

    def _on_click_batch_correct(self, event=None):

        background_text = self.textbox_background_keyword.text()

        if self.preproc_export_path_display.value == Path('.'):
            self.preproc_export_path_display._on_choose_clicked()

        main_folder = Path(self.file_list.folder_path)

        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(range(self.file_list.count())) as pbr:
            pbr.set_description("Batch processing folder")
            for c in pbr:
                f = self.file_list.item(c).text()
                current_folder = main_folder.joinpath(f)

                min_max_band = None
                if self.check_do_min_max.isChecked():
                    min_band = self.qspin_min_band.value()
                    max_band = self.qspin_max_band.value()
                    min_max_band = [min_band, max_band]

                batch_preprocessing(
                    folder_to_analyze=current_folder,
                    export_folder=self.preproc_export_path_display.value,
                    background_text=background_text,
                    min_max_band=min_max_band,
                    background_correction=self.check_do_background_correction.isChecked(),
                    destripe=self.check_do_destripe.isChecked(),
                    use_dask=self.check_use_dask.isChecked(),
                    chunk_size=self.spin_chunksize.value(),
                    use_float=self.check_save_as_float.isChecked()
                )
        self.viewer.window._status_bar._toggle_activity_dock(False)
