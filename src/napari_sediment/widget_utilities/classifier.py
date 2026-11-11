import numpy as np
from qtpy.QtCore import QTimer

from napari_convpaint.convpaint_widget import ConvpaintWidget

class ConvPaintSpectralWidget(ConvpaintWidget):
    """Widget for training a classifier on a spectral image stack. Adapted from 
    base class ConvPaintWidget in napari_convpaint.conv_paint.py. UI is simplified
    and some defaults like multi-channel image and RGB input are set. Also adds
    an ml-mask layer to the viewer."""
    
    def __init__(self, viewer, parent=None, third_party=True):
        super().__init__(viewer, parent=parent, third_party=third_party)

        self._spectral_init_done = False

        # Convpaint adds annotations on image selection even in third-party mode;
        # here layers are only added when the ML tab is opened. Must be set before
        # convpaint's deferred init, which already selects the image.
        self.auto_add_layers = False
        if hasattr(self, 'check_auto_add_layers'):
            self.check_auto_add_layers.setChecked(False)

        # Convpaint only refreshes its layer dropdowns once initialized (i.e. when
        # first shown), and napari only refreshes top-level dock widgets. Since
        # this widget sits in a sub-tab, keep the dropdowns in sync from the start.
        self.viewer.layers.events.inserted.connect(self._reset_layer_choices)
        self.viewer.layers.events.removed.connect(self._reset_layer_choices)

    def _reset_layer_choices(self, event=None):
        """Refresh layer dropdowns while convpaint has not yet been initialized."""

        if not self._spectral_init_done:
            self.image_layer_selection_widget.reset_choices()
            self.annotations_layer_selection_widget.reset_choices()

    def ensure_init(self):
        """Initialize synchronously. Convpaint's showEvent only schedules the
        initialization (QTimer), so its own ensure_init() can return before the
        model exists, e.g. when called from a tab-change handler."""

        self._post_init_done = True
        self._late_init()

    def _late_init(self):
        """Convpaint defers model creation, connections and defaults until the
        widget is first shown, so spectral-specific defaults are applied here.
        Runs only once, whether triggered by showEvent or ensure_init."""

        if self._spectral_init_done:
            return
        self._spectral_init_done = True

        self.image_layer_selection_widget.reset_choices()
        self.annotations_layer_selection_widget.reset_choices()
        super()._late_init()

        self.fe_scaling_factors.setCurrentText('[1,2]')
        self._on_set_fe_model()
        # Set after the FE as setting the FE can reset tiling to its defaults
        self.check_tile_image.setChecked(True)

        self.add_connections_local()
        self.default_to_multi_channel()

        # Replace annotations created with a shape that doesn't fit multi-channel data
        img = self.image_layer_selection_widget.value
        annot = self.annotations_layer_selection_widget.value
        if (img is not None and annot is not None
                and not self._approve_annotations_layer_shape(annot, img)):
            self._on_add_annot_layer(force_add=True)

    def _delayed_on_select_layer(self, event=None):
        """Same as the base class, but with timers owned by the widget: Qt stops
        them when the widget is destroyed, whereas the base class's unparented
        QTimer.singleShot can fire on deleted widgets (e.g. on viewer close)."""

        if not hasattr(self, '_select_layer_timer'):
            self._select_layer_timer = QTimer(self)
            self._select_layer_timer.setSingleShot(True)
            self._select_layer_timer.timeout.connect(self._on_select_layer)
            self._block_select_timer = QTimer(self)
            self._block_select_timer.setSingleShot(True)
            self._block_select_timer.timeout.connect(
                lambda: setattr(self, '_block_layer_select', True))
        self._block_layer_select = False
        self._select_layer_timer.start(100)
        self._block_select_timer.start(200)

    def add_connections_local(self):

        # Connected after the base class handlers so that segmentation is up to date
        self.train_classifier_btn.clicked.connect(self.update_ml_mask)
        self.segment_btn.clicked.connect(self.update_ml_mask)
        self.image_layer_selection_widget.changed.connect(self.default_to_multi_channel)

    def default_to_multi_channel(self, event=None):

        img = self.image_layer_selection_widget.value
        if img is not None and img.ndim == 3 and not getattr(img, 'rgb', False):
            self.radio_multi_channel.setChecked(True)

    def on_image_cropped(self):
        """Update after the data of the image layer were replaced by a crop with a
        different extent (e.g. other main ROI). Annotations memorized for training refer
        to the previous crop and are discarded. The trained classifier is kept."""

        if not self._spectral_init_done:
            return
        self._reset_train_features()
        self._on_select_layer()

    def add_annotation_layer(self):
        """Add an empty annotations layer for the selected image."""

        self.ensure_init()
        self._on_add_annot_layer(force_add=True)

    def update_ml_mask(self, event=None):
        """Add ml-mask layer to viewer when segmenting"""

        if self.seg_tag not in self.viewer.layers:
            return
        seg_layer = self.viewer.layers[self.seg_tag]
        mask = (seg_layer.data == 1).astype(np.uint8)
        if 'ml-mask' in self.viewer.layers:
            self.viewer.layers['ml-mask'].data = mask
        else:
            self.viewer.add_labels(
                mask, name='ml-mask', scale=seg_layer.scale, translate=seg_layer.translate)
