from torchvision import models
from torch import nn
from collections import OrderedDict
import torch
import numpy as np
import skimage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from napari_convpaint.conv_paint import ConvPaintWidget

class ConvPaintSpectralWidget(ConvPaintWidget):
    """Widget for training a classifier on a spectral image stack. Adapted from 
    base class ConvPaintWidget in napari_convpaint.conv_paint.py. UI is simplified
    and some defaults like multi-channel image and RGB input are set. Also adds
    an ml-mask layer to the viewer."""
    
    def __init__(self, viewer):
        super().__init__(viewer)

        self.viewer = viewer
        self.check_use_project.hide()
        self.prediction_all_btn.hide()
        self.update_model_on_project_btn.hide()
        self.tabs.setTabVisible(1, False)

        self.prediction_btn.clicked.connect(self.update_ml_mask)

    def update_ml_mask(self, event):
        """Add ml-mask layer to viewer when segmenting"""
        
        if 'ml-mask' in self.viewer.layers:
            self.viewer.layers['ml-mask'].data = (self.viewer.layers['segmentation'].data == 1).astype(np.uint8)
        else:
            self.viewer.add_labels((self.viewer.layers['segmentation'].data==1).astype(np.uint8), name='ml-mask')
