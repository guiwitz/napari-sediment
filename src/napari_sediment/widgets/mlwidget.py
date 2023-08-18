from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget,
                            QLabel, QFileDialog, QListWidget, QAbstractItemView,
                            QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,QGridLayout)
from qtpy.QtCore import Qt
import numpy as np
import skimage

from ..classifier import Classifier
from napari_convpaint.conv_paint_utils import Hookmodel
from napari_convpaint.conv_paint_utils import (get_features_current_layers,
                                               train_classifier, predict_image)

class MLWidget(QWidget):
    """Widget for ml pixel classification. Parent should have :
    - an attribute called rgb_names, specifying the channels to use."""

    def __init__(self, parent, napari_viewer):
        super().__init__(parent)

        self.parent = parent
        self.viewer = napari_viewer
        self.pixclass = None
        self.model = None
        self.data = None

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
        self.spin_downscale = QSpinBox()
        self.spin_downscale.setRange(1, 10)
        self.spin_downscale.setSingleStep(1)
        self.spin_downscale.setValue(4)
        self.main_layout.addWidget(QLabel('Downscale factor'), 2, 0, 1, 1)
        self.main_layout.addWidget(self.spin_downscale, 2, 1, 1, 1)
        self.btn_reset_mlmodel = QPushButton("(Re-)train pixel classifier")
        self.main_layout.addWidget(self.btn_reset_mlmodel, 3, 0, 1, 2)
        self.btn_ml_mask = QPushButton("Generate mask")
        self.main_layout.addWidget(self.btn_ml_mask, 4, 0, 1, 2)

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

    def _on_initialize_model(self, event=None):

        if 'annotations' not in self.viewer.layers:
            raise ValueError('No annotation layer found')
        
        reduce_fact = self.spin_downscale.value()
        data = self.viewer.layers['imcube'].data
        if len(data) !=3:
            raise ValueError('Only three channel images are supported')
        annotations = self.viewer.layers['annotations'].data[::reduce_fact,::reduce_fact]
        if self.check_smoothing.isChecked():
            data = skimage.filters.gaussian(data, sigma=self.spin_gaussian_smoothing.value(), preserve_range=True)[:, ::4, ::4]
        else:
            data = data[:, ::reduce_fact, ::reduce_fact]
        self.data = data.mean(axis=0)
        
        if self.model is None:
            self.model = Hookmodel(model_name='vgg16')
            self.model.register_hooks(selected_layers=[list(self.model.module_dict.keys())[0]])

        features, targets = get_features_current_layers(
            model=self.model, image=self.data, annotations=annotations, scalings=[1,2], order=1, use_min_features=False)
        self.random_forest = train_classifier(features, targets)


    def _on_click_ml_mask(self):

        if 'annotations' not in self.viewer.layers:
            raise ValueError('No annotation layer found')
        
        if self.model is None:
            self._on_initialize_model()
        
        pred = predict_image(self.data, self.model, self.random_forest, scalings=[1,2], order=1, use_min_features=False)
        pred = (pred == 1).astype(np.uint8)
        predict_upscale = skimage.transform.resize(
            pred, self.viewer.layers['annotations'].data.shape, order=0)
        if 'ml-mask' in self.viewer.layers:
            self.viewer.layers['ml-mask'].data = predict_upscale
        else:
            self.viewer.add_labels((predict_upscale==1).astype(np.uint8), name='ml-mask')