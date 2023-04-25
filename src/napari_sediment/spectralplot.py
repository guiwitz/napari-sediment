from napari_matplotlib.base import NapariMPLWidget
from matplotlib.widgets import SpanSelector
import numpy as np

class SpectralPlotter(NapariMPLWidget):
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
    def __init__(self, napari_viewer, options=None):
        super().__init__(napari_viewer)
        self.axes = self.canvas.figure.subplots()
        self.cursor_pos = np.array([])
        self.axes.tick_params(colors='white')
       

    def clear(self):
        """
        Clear the canvas.
        """
        #self.axes.clear()
        pass

class SelectRange:
    
    def __init__(self, parent, ax):
        
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.parent = parent
        self.myline1 = None
        self.myline2 = None
        
        self.span = SpanSelector(ax, onselect=self.onselect, direction='horizontal')#, button=1)
        

    def onselect(self, min_pos, max_pos):
        
        if self.myline1 is not None:
            self.myline1.pop(0).remove()
        if self.myline2 is not None:
            self.myline2.pop(0).remove()
        min_max = [self.ax.lines[0].get_data()[1].min(),
                   self.ax.lines[0].get_data()[1].max()]
        self.myline1 = self.ax.plot([min_pos, min_pos], min_max)
        self.myline2 = self.ax.plot([max_pos, max_pos], min_max)
        self.min_pos = min_pos
        self.max_pos = max_pos
        
    def disconnect(self):
        self.span.disconnect_events()
        self.canvas.draw_idle()