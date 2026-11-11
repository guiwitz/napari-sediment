"""Keep track of which plugin widget currently "owns" the viewer content.

Several widgets of the plugin can be docked in the same viewer and use layers
with the same names ('imcube', 'red', 'mask', ...) and viewer-wide callbacks
(mouse, layer events). The widget that last loaded data is the active one:
callbacks of other widgets are ignored, and these widgets are disabled until
they load data again."""

import weakref

from qtpy.QtWidgets import QLabel

_active_widgets = {}


def activate_widget(widget, viewer, clear_layers=True):
    """Make widget the active widget of viewer. The previously active widget
    is deactivated, and optionally all layers are removed."""

    previous = _get_active(viewer)
    _active_widgets[id(viewer)] = (weakref.ref(viewer), weakref.ref(widget))

    if previous is not None and previous is not widget:
        try:
            if hasattr(previous, 'on_deactivated'):
                previous.on_deactivated()
            set_widget_inactive(previous, True)
        except RuntimeError:
            # widget was closed and its Qt objects deleted
            pass
    set_widget_inactive(widget, False)

    if clear_layers:
        while len(viewer.layers) > 0:
            viewer.layers.clear()


def is_active_widget(widget, viewer):
    """Check whether widget is the active widget of viewer. Before any widget
    loaded data, all widgets are considered active."""

    active = _get_active(viewer)
    if active is None:
        return True
    return active is widget


def _get_active(viewer):
    """Return the active widget of viewer, or None."""

    entry = _active_widgets.get(id(viewer))
    if entry is None:
        return None
    viewer_ref, widget_ref = entry
    # ids can be reused once a viewer is deleted
    if viewer_ref() is not viewer:
        del _active_widgets[id(viewer)]
        return None
    return widget_ref()


def set_widget_inactive(widget, inactive):
    """Show a message and disable all tabs except the first one (used to open
    or import data) of an inactive widget."""

    if not hasattr(widget, 'tabs') or not hasattr(widget, 'main_layout'):
        return
    if not hasattr(widget, '_inactive_label'):
        widget._inactive_label = QLabel(
            'Data in the viewer were loaded by another widget. Open or import '
            'data to use this widget again.')
        widget._inactive_label.setWordWrap(True)
        widget._inactive_label.setStyleSheet('color: orange; font-weight: bold;')
        widget.main_layout.insertWidget(0, widget._inactive_label)
    widget._inactive_label.setVisible(inactive)
    for i in range(1, widget.tabs.count()):
        widget.tabs.setTabEnabled(i, not inactive)
    if inactive:
        widget.tabs.setCurrentIndex(0)
