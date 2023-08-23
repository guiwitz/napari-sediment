from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget,
                            QLabel, QFileDialog, QListWidget, QAbstractItemView,
                            QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,QGridLayout)
from qtpy.QtCore import Qt
import numpy as np
import skimage
from pathlib import Path

from ..classifier import Classifier
from napari_convpaint.conv_paint_utils import Hookmodel, Classifier
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
        self.classifier = None
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
        self.btn_save_model = QPushButton("Save model")
        self.main_layout.addWidget(self.btn_save_model, 5, 0, 1, 2)
        self.btn_load_model = QPushButton("Load model")
        self.main_layout.addWidget(self.btn_load_model, 6, 0, 1, 2)

        self.add_connections()

    def add_connections(self):

        self.btn_add_annotation_layer.clicked.connect(self._on_click_add_annotation_layer)
        self.btn_reset_mlmodel.clicked.connect(self._on_initialize_model)
        self.btn_ml_mask.clicked.connect(self._on_click_ml_mask)
        self.btn_save_model.clicked.connect(self._on_click_save_model)
        self.btn_load_model.clicked.connect(self._on_click_load_model)


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
        self.data = self.get_data()
        annotations = self.viewer.layers['annotations'].data[::reduce_fact,::reduce_fact]
        
        if self.classifier is None:
            self.classifier = Classifier()

        features, targets = get_features_current_layers(
            model=self.classifier.model, image=self.data, annotations=annotations,
            scalings=self.classifier.param.scalings,
            order=self.classifier.param.order, use_min_features=self.classifier.param.use_min_features)
        self.classifier.random_forest = train_classifier(features, targets)

    def get_data(self):

        reduce_fact = self.spin_downscale.value()
        #data = self.viewer.layers['imcube'].data
        data = self.parent.rgb_widget.get_current_rgb_cube()

        if len(data) !=3:
            raise ValueError('Only three channel images are supported')
        
        if self.check_smoothing.isChecked():
            data = skimage.filters.gaussian(data, sigma=self.spin_gaussian_smoothing.value(), preserve_range=True)[:, ::4, ::4]
        else:
            data = data[:, ::reduce_fact, ::reduce_fact]
        return data.mean(axis=0)

    def _on_click_ml_mask(self):

        #if 'annotations' not in self.viewer.layers:
        #    raise ValueError('No annotation layer found')
        
        if self.classifier is None:
            self._on_initialize_model()
        
        if self.data is None:
            self.data = self.get_data()

        pred = predict_image(
            self.data, self.classifier.model,
            self.classifier.random_forest,
            self.classifier.param.scalings,
            self.classifier.param.order, self.classifier.param.use_min_features)
        
        pred = (pred == 1).astype(np.uint8)
        predict_upscale = skimage.transform.resize(
            pred, self.viewer.layers['mask'].data.shape, order=0)
        if 'ml-mask' in self.viewer.layers:
            self.viewer.layers['ml-mask'].data = predict_upscale
        else:
            self.viewer.add_labels((predict_upscale==1).astype(np.uint8), name='ml-mask')

    def _on_click_save_model(self, event=None, save_path=None):
            
        if save_path is None:
            dialog = QFileDialog()
            save_path, _ = dialog.getSaveFileName(self, "Save model", None, "JOBLIB (*.joblib)")
            save_path = Path(save_path)

        self.classifier.save_classifier(save_path)

    def _on_click_load_model(self, event=None, load_path=None):

        self.classifier = Classifier()
    
        if load_path is None:
            dialog = QFileDialog()
            load_path, _ = dialog.getOpenFileName(self, "Load model", None, "JOBLIB (*.joblib)")
            load_path = Path(load_path)

        self.classifier.load_model(load_path)
