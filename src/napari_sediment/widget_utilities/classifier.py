import numpy as np

from napari_convpaint.conv_paint import ConvPaintWidget

class ConvPaintSpectralWidget(ConvPaintWidget):
    """Widget for training a classifier on a spectral image stack. Adapted from 
    base class ConvPaintWidget in napari_convpaint.conv_paint.py. UI is simplified
    and some defaults like multi-channel image and RGB input are set. Also adds
    an ml-mask layer to the viewer."""
    
    def __init__(self, viewer, parent=None, project=False, third_party=True):
        super().__init__(viewer, parent, project, third_party)

        self.viewer = viewer
        self.check_use_project.hide()
        self.prediction_all_btn.hide()
        self.update_model_on_project_btn.hide()
        self.tabs.setTabVisible(1, False)

        self.prediction_btn.clicked.connect(self.update_ml_mask)
        self.update_model_btn.clicked.connect(self.predict)
        self.update_model_btn.clicked.connect(self.update_ml_mask)
        self.update_model_btn.setText( 'Train + Segment')
        self.check_tile_image.setChecked(True)

        
        self.add_connections_local()

    def add_connections_local(self):
        
        self.select_layer_widget.changed.connect(self.default_to_multi_channel)

    def default_to_multi_channel(self, event=None):

        if self.select_layer_widget.value.ndim == 3:
            
            self.radio_multi_channel.setChecked(True)
        
    def update_ml_mask(self, event):
        """Add ml-mask layer to viewer when segmenting"""
        
        if 'ml-mask' in self.viewer.layers:
            self.viewer.layers['ml-mask'].data = (self.viewer.layers['segmentation'].data == 1).astype(np.uint8)
        else:
            self.viewer.add_labels((self.viewer.layers['segmentation'].data==1).astype(np.uint8), name='ml-mask')
