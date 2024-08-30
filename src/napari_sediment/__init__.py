
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .utilities._reader import napari_get_reader
from .widgets.sediment_widget import SedimentWidget

__all__ = (
    "napari_get_reader",
    "SedimentWidget",
)
