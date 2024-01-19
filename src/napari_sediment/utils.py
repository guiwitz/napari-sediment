import dask.array as da
import numpy as np


def update_contrast_on_layer(napari_layer):

    if isinstance(napari_layer.data, da.Array):
        napari_layer.contrast_limits_range = (napari_layer.data.min().compute(), napari_layer.data.max().compute())
        napari_layer.contrast_limits = np.percentile(napari_layer.data.compute(), (2,98))
    else:       
        napari_layer.contrast_limits_range = (napari_layer.data.min(), napari_layer.data.max())
        napari_layer.contrast_limits = np.percentile(napari_layer.data, (2,98))