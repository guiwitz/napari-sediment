import numpy as np


def update_contrast_on_layer(napari_layer):

    data = np.asarray(napari_layer.data)
          
    napari_layer.contrast_limits_range = (data.min(), data.max())
    napari_layer.contrast_limits = np.percentile(data, (2,98))