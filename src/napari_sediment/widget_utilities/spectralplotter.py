#from napari_matplotlib.base import NapariMPLWidget
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from qtpy.QtWidgets import QWidget, QVBoxLayout

class SpectralPlotter(QWidget):
    """Subclass of napari_matplotlib NapariMPLWidget for voxel position based time series plotting.
    This widget contains a matplotlib figure canvas for plot visualisation and the matplotlib toolbar for easy option
    controls. The widget is not meant for direct docking to the napari viewer.
    Plot visualisation is triggered by moving the mouse cursor over the voxels of an image layer while holding the shift
    key. The first dimension is handled as time. This widget needs a napari viewer instance and a LayerSelector instance
    to work properly.
    Attributes:
        axes : matplotlib.axes.Axes
        selector : napari_time_series_plotter.LayerSelector
        cursor_pos : tuple of current mouse cursor position in the napari viewer
    """
    def __init__(self, napari_viewer, options=None, tight_layout=True):
        super().__init__()

        self.canvas = FigureCanvasQTAgg()
        self.canvas.figure.set_tight_layout(tight_layout)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        self.axes = self.canvas.figure.subplots()
        self.cursor_pos = np.array([])
        self.axes.tick_params(colors='black')
       

        self.setLayout(QVBoxLayout())
        #self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)
        self.layout().addWidget(self.toolbar)