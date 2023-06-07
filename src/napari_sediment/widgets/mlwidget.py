from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget,
                            QLabel, QFileDialog, QListWidget, QAbstractItemView,
                            QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,QGridLayout)
from qtpy.QtCore import Qt
import numpy as np
import skimage

from ..classifier import Classifier

class MLWidget(QWidget):
    """Widget for ml pixel classification. Parent should have :
    - an attribute called rgb_names, specifying the channels to use."""

    def __init__(self, parent, napari_viewer):
        super().__init__(parent)

        self.parent = parent
        self.viewer = napari_viewer
        self.pixclass = None

        self.main_layout = QGridLayout()
        self.setLayout(self.main_layout)
        
        self.btn_add_annotation_layer = QPushButton("Add annotation layer")
        self.main_layout.addWidget(self.btn_add_annotation_layer, 0, 0, 1, 2)
        self.check_smoothing = QCheckBox('Gaussian smoothing')
        self.check_smoothing.setChecked(False)
        self.main_layout.addWidget(self.check_smoothing, 1, 0, 1, 1)
        self.spin_gaussian_smoothing = QDoubleSpinBox()
        self.spin_gaussian_smoothing.setRange(0.1, 10)
        self.spin_gaussian_smoothing.setSingleStep(0.1)
        self.spin_gaussian_smoothing.setValue(3)
        self.main_layout.addWidget(self.spin_gaussian_smoothing, 1, 1, 1, 1)
        self.btn_reset_mlmodel = QPushButton("Reset/Initialize ML model")
        self.main_layout.addWidget(self.btn_reset_mlmodel, 2, 0, 1, 2)
        self.btn_ml_mask = QPushButton("Pixel Classifier mask")
        self.main_layout.addWidget(self.btn_ml_mask, 3, 0, 1, 2)

        self.add_connections()

    def add_connections(self):

        self.btn_add_annotation_layer.clicked.connect(self._on_click_add_annotation_layer)
        self.btn_reset_mlmodel.clicked.connect(self._on_initialize_model)
        self.btn_ml_mask.clicked.connect(self._on_click_ml_mask)

    def _on_click_add_annotation_layer(self):
        """Add annotation layer to viewer"""

        if 'annotations' in self.viewer.layers:
            print('Annotations layer already exists')
            return
        self.viewer.add_labels(np.zeros_like(self.viewer.layers['mask'].data), name='annotations', opacity=0.5)

    def _on_initialize_model(self):

        if 'annotations' not in self.viewer.layers:
            raise ValueError('No annotation layer found')
        
        reduce_fact = 4
        #data = np.stack([self.viewer.layers[x].data for x in self.parent.rgb_names], axis=0)
        data = self.viewer.layers['imcube'].data
        if len(data) !=3:
            raise ValueError('Only three channel images are supported')
        annotations = self.viewer.layers['annotations'].data[::reduce_fact,::reduce_fact]
        if self.check_smoothing.isChecked():
            data = skimage.filters.gaussian(data, sigma=self.spin_gaussian_smoothing.value(), preserve_range=True)[:, ::4, ::4]
        else:
            data = data[:, ::reduce_fact, ::reduce_fact]
        self.pixclass = Classifier(data=data, annotations=annotations)
        #self.pixclass.load_model(model_type='resnet50')
        self.pixclass.load_model(model_type='vgg16')
        
        self.pixclass.compute_multiscale_features()

    def _on_click_ml_mask(self):

        if 'annotations' not in self.viewer.layers:
            raise ValueError('No annotation layer found')
        
        if self.pixclass is None:
            self._on_initialize_model()

        reduce_fact = 4
        annotations = self.viewer.layers['annotations'].data[::reduce_fact,::reduce_fact]
        self.pixclass.annotations = annotations
        self.pixclass.extract_annotated_features()
        self.pixclass.train_model()
        pred = self.pixclass.predict()
        pred = (pred == 1).astype(np.uint8)
        predict_upscale = skimage.transform.resize(
            pred, self.viewer.layers['annotations'].data.shape, order=0)
        if 'ml-mask' in self.viewer.layers:
            self.viewer.layers['ml-mask'].data = predict_upscale
        else:
            self.viewer.add_labels((predict_upscale==1).astype(np.uint8), name='ml-mask')