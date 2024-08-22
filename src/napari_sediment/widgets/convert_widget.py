from pathlib import Path
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QPushButton,
                            QLineEdit, QFileDialog)
from qtpy.QtCore import Qt

from napari_guitils.gui_structures import TabSet, VHGroup
from ..data_structures.parameters import Param
from ..utilities.sediproc import convert_bil_raw_to_zarr

class ConvertWidget(QWidget):
    
    def __init__(self, napari_viewer):
        super().__init__()
        
        self.viewer = napari_viewer
        self.params = Param()

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['raw2zarr']
        self.tabs = TabSet(self.tab_names, tab_layouts=[QGridLayout()])
        self.tabs.widget(0).layout().setAlignment(Qt.AlignTop)

        self.files_group = VHGroup('Files and folders', orientation='G')
        self.tabs.add_named_tab('raw2zarr', self.files_group.gbox)
        self.main_layout.addWidget(self.tabs)

        self.btn_select_imhdr_file = QPushButton("Select hdr file")
        self.btn_select_imhdr_file.setToolTip("Select a file with .hdr extension")
        self.imhdr_path_display = QLineEdit("No path")
        self.files_group.glayout.addWidget(self.btn_select_imhdr_file, 0, 0, 1, 1)
        self.files_group.glayout.addWidget(self.imhdr_path_display, 0, 1, 1, 1)

        self.btn_select_export_folder = QPushButton("Set export folder")
        self.btn_select_export_folder.setToolTip(
            "Select a folder where to save the results and intermeditate files")
        self.export_path_display = QLineEdit("No path")
        self.files_group.glayout.addWidget(self.btn_select_export_folder, 1, 0, 1, 1)
        self.files_group.glayout.addWidget(self.export_path_display, 1, 1, 1, 1)

        self.btn_convert = QPushButton("Convert")
        self.btn_convert.setToolTip("Convert .hdr to .zarr")
        self.files_group.glayout.addWidget(self.btn_convert, 2, 0, 1, 2)

        self.add_connections()


    def add_connections(self):

        self.btn_select_export_folder.clicked.connect(self._on_click_select_export_folder)
        self.btn_select_imhdr_file.clicked.connect(self._on_click_select_imhdr)
        self.btn_convert.clicked.connect(self._on_click_convert)

    def _on_click_select_export_folder(self, event=None, export_folder=None):
        """Interactively select folder to analyze"""

        if export_folder is not None:
            self.export_folder = Path(export_folder)
        else:
            self.export_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        self.export_path_display.setText(self.export_folder.as_posix())

    def _on_click_select_imhdr(self, event=None, imhdr_path=None):
        """Interactively select hdr file"""

        if imhdr_path is not None:
            imhdr_path = Path(imhdr_path)
        else:
            imhdr_path = Path(QFileDialog.getOpenFileName(self, "Select file")[0])
        
        if imhdr_path.parent.suffix == '.zarr':
            imhdr_path = imhdr_path.parent
        
        self.imhdr_path = Path(imhdr_path)
        self.imhdr_path_display.setText(self.imhdr_path.as_posix())

    def _on_click_convert(self, event=None):
        """Convert .hdr to .zarr"""
        
        convert_bil_raw_to_zarr(self.imhdr_path, self.export_folder)
        self.save_params()

    def save_params(self):

        self.params.project_path = self.export_folder
        self.params.file_path = self.imhdr_path
        self.params.main_roi = []
        self.params.rois = []
        self.params.save_parameters()

