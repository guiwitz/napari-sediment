from napari_matplotlib.base import NapariMPLWidget
from matplotlib.widgets import SpanSelector
import numpy as np
from cmap import Colormap
from napari.utils import colormaps
from microfilm import colorify
from matplotlib_scalebar.scalebar import ScaleBar
from .spectralindex import compute_index_projection

def plot_spectral_profile(rgb_image, mask, index_obj, format_dict, scale=1,
                          location="", fig=None, roi=None, left_margin=0,
                            right_margin=0, bottom_margin=0, top_margin=0,
                          repeat=True):

    index_name = index_obj.index_name
    index_image = index_obj.index_map
    proj = index_obj.index_proj
    index_contrast_limits = index_obj.index_map_range
    index_colormap = index_obj.colormap

    im_w = index_image.shape[1]
    im_h = index_image.shape[0]

    title_font = format_dict['title_font']
    label_font = format_dict['label_font']
    color_plotline = format_dict['color_plotline']
    plot_thickness = format_dict['plot_thickness']
    red_contrast_limits = format_dict['red_contrast_limits']
    green_contrast_limits = format_dict['green_contrast_limits']
    blue_contrast_limits = format_dict['blue_contrast_limits']

    # get colormap
    newmap = Colormap(colormaps.ALL_COLORMAPS[index_colormap].colors)
    mpl_map = newmap.to_matplotlib()

    rgb_to_plot = create_rgb_image(rgb_image, red_contrast_limits, green_contrast_limits, blue_contrast_limits)
    rgb_to_plot[mask == 1, :] = 0

    if im_h / im_w > 2:
        a4_size = np.array([11.69, 8.27])
    else:
        a4_size = np.array([8.27, 11.69])
    a4_margins = a4_size - np.array([bottom_margin + top_margin, left_margin + right_margin])

    pixel_in_inches = a4_margins[0] / im_h
    im_height_inches = a4_margins[0]
    im_width_inches = im_w * pixel_in_inches
    plot_width_inches = a4_margins[1] - 2 * im_width_inches 

    # The figure and axes are set explicitly to make sure that the axes fill the figure
    # This is achieved using the add_axes method instead of subplots
    fig_size = [a4_size[1], a4_size[0]]
    fig.clear()
    fig.set_size_inches(fig_size)
    fig.set_facecolor('white')
    
    ax1 = fig.add_axes(rect=(left_margin/a4_size[1], bottom_margin/a4_size[0], im_width_inches/a4_size[1], im_height_inches/a4_size[0]))
    ax2 = fig.add_axes(rect=(im_width_inches/a4_size[1]+left_margin/a4_size[1], bottom_margin/a4_size[0], im_width_inches/a4_size[1], im_height_inches/a4_size[0]))
    ax3 = fig.add_axes(rect=((2*im_width_inches+left_margin)/a4_size[1], bottom_margin/a4_size[0], plot_width_inches/a4_size[1], im_height_inches/a4_size[0]))

    ax1.imshow(rgb_to_plot, aspect='auto')
    if index_contrast_limits is None:
        non_nan = index_image[~np.isnan(index_image)]
        vmin = np.percentile(non_nan, 0.1)
        vmax = np.percentile(non_nan, 99.9)
    else:
        vmin = index_contrast_limits[0]
        vmax = index_contrast_limits[1]
    index_image[mask==1] = np.nan
    ax2.imshow(index_image, aspect='auto', interpolation='none', cmap=mpl_map, vmin=vmin, vmax=vmax)

    if roi is not None:
        roi = roi.copy()
        roi[1,0] -=0.5
        roi[2,0] -=0.5
        roi[0,0] -=0.4
        roi[3,0] -=0.4
        roi = np.concatenate([roi, roi[[0]]])
        ax2.plot(roi[:,1], roi[:,0], 'r')
    
    ax3.plot(proj, np.arange(len(proj)), color=np.array(color_plotline), linewidth=plot_thickness)

    ax3.set_ylim(0, len(proj))
    ax3.yaxis.tick_right()
    ax3.invert_yaxis()
    
    # set y axis scale
    for ax in [ax1, ax3]:
        tickpos = np.array([x.get_position()[1] for x in  ax.get_yticklabels()])[1:-1]
        newlabels = scale * np.array(tickpos)
        ax.set_yticks(ticks=tickpos, labels = newlabels)

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    for label in (ax1.get_yticklabels() + ax3.get_yticklabels() + ax3.get_xticklabels()):
        label.set_fontsize(label_font)
    
    ax2.set_ylim(im_h - 0.5, -0.5)
    ax1.set_ylabel('depth [mm]', fontsize=label_font)
    ax3.set_ylabel('depth [mm]', fontsize=label_font)
    ax3.set_xlabel('Index value', fontsize=label_font)
    ax3.yaxis.set_label_position('right')
    suptitle = fig.suptitle(index_name + '\n' + location,
                    fontsize=title_font)
    
    # check the size of titles labels and tickmarks, adjust margins accordingly,
    # and repeat the plot

    # adjust left margin
    renderer = fig.canvas.get_renderer()
    text = ax1.yaxis.label
    label_width = get_text_width(text, renderer, fig)
    y_tick_widths = [get_text_width(label, renderer, fig) for label in ax1.get_yticklabels()]
    max_y_tick_width = max(y_tick_widths)
    left_margin = 1.0 * (3 * label_width + max_y_tick_width) * a4_size[1]

    # adjust right margin
    text = ax3.yaxis.label
    label_width = get_text_width(text, renderer, fig)
    y_tick_widths = [get_text_width(label, renderer, fig) for label in ax3.get_yticklabels()]
    max_y_tick_width = max(y_tick_widths)
    right_margin = 1.0 * (3 * label_width + max_y_tick_width) * a4_size[1]

    # adjust bottom margin
    text = ax3.xaxis.label
    label_height = get_text_height(text, renderer, fig)
    x_tick_heights = [get_text_height(label, renderer, fig) for label in ax3.get_xticklabels()]
    max_x_tick_height = max(x_tick_heights)
    bottom_margin = 1.0 * (3 * label_height + max_x_tick_height) * a4_size[0]

    # adjust top margin
    bbox = suptitle.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    title_height = bbox.ymax - bbox.ymin
    top_margin = 2 * title_height * a4_size[0]

    if repeat:
        plot_spectral_profile(rgb_image, mask, index_obj, format_dict, scale=scale,
                          location=location, fig=fig, roi=roi, left_margin=left_margin,
                          right_margin=right_margin, bottom_margin=bottom_margin,
                          top_margin=top_margin,
                          repeat=False)


    return fig, ax1, ax2, ax3

def get_text_width(text, renderer, fig):
    bbox = text.get_window_extent(renderer)
    # Convert from display to figure coordinates
    bbox = bbox.transformed(fig.transFigure.inverted())
    return bbox.xmax - bbox.xmin

def get_text_height(text, renderer, fig):
    bbox = text.get_window_extent(renderer)
    # Convert from display to figure coordinates
    bbox = bbox.transformed(fig.transFigure.inverted())
    return bbox.ymax - bbox.ymin


def plot_multi_spectral_profile(rgb_image, mask, index_objs, format_dict, scale=1,
                                location="", fig=None, roi=None):

    left_right_margin_fraction = format_dict['left_right_margin_fraction']
    bottom_top_margin_fraction = format_dict['bottom_top_margin_fraction']
    #plot_image_w_fraction = format_dict['plot_image_w_fraction']
    title_font_factor = format_dict['title_font_factor']
    label_font_factor = format_dict['label_font_factor']
    color_plotline = format_dict['color_plotline']
    plot_thickness = format_dict['plot_thickness']
    figure_size_factor = format_dict['figure_size_factor']
    #scale_font_size = format_dict['scale_font_size']
    #index_colormap = format_dict['index_colormap']
    red_contrast_limits = format_dict['red_contrast_limits']
    green_contrast_limits = format_dict['green_contrast_limits']
    blue_contrast_limits = format_dict['blue_contrast_limits']
    
    rgb_to_plot = create_rgb_image(rgb_image, red_contrast_limits, green_contrast_limits, blue_contrast_limits)
    rgb_to_plot[mask==1,:] = 0

    im_h = rgb_image[0].shape[0]
    im_w = rgb_image[0].shape[1]

    #width = width/height
    #height = 1

    width_tot = (len(index_objs)+1) * im_w

    to_add_top = bottom_top_margin_fraction*im_h
    to_add_bottom = bottom_top_margin_fraction*im_h
    to_add_left = left_right_margin_fraction * width_tot
    to_add_right = left_right_margin_fraction * width_tot

    width_tot_margin = width_tot + to_add_left + to_add_right
    height_tot_margin = im_h + to_add_top + to_add_bottom

    left_margin = to_add_left / width_tot_margin
    bottom_margin = to_add_bottom / height_tot_margin

    plot_width = im_w / width_tot_margin
    
    a4_size = np.array([11.69, 8.27])
    size_ratio = a4_size / np.array([width_tot_margin, height_tot_margin])
    min_ratio = size_ratio.min()
    fig_size = figure_size_factor * min_ratio * np.array([width_tot_margin, height_tot_margin])
    
    fig.clear()
    fig.set_size_inches(fig_size)
    fig.set_facecolor('white')
    halfplot = len(index_objs) // 2
    axes = []
    shift = 0
    for i in range(len(index_objs)):
        if i == halfplot:
            shift = 1
        proj = index_objs[i].index_proj
        index_name = index_objs[i].index_name
        axes.append(fig.add_axes(rect=(left_margin+((i+shift)*plot_width),bottom_margin, plot_width, im_h / height_tot_margin)))
        axes[-1].plot(proj, np.arange(len(proj)), color=np.array(color_plotline), linewidth=plot_thickness)
        axes[-1].set_ylim(0, len(proj))
        if (i!=0) and (i!=len(proj)-1):
            axes[-1].yaxis.set_visible(False)
        if i == len(proj)-1:
            axes[-1].yaxis.tick_right()
            axes[-1].yaxis.set_label_position('right')
        axes[-1].invert_yaxis()
        axes[-1].set_title(index_name, fontsize=title_font_factor)
    
    axes.append(fig.add_axes(rect=(left_margin+(halfplot*plot_width),bottom_margin, plot_width, im_h / height_tot_margin)))
    axes[-1].imshow(rgb_to_plot)
    axes[-1].yaxis.set_visible(False)
    axes[-1].xaxis.set_visible(False)
    if roi is not None:
        roi = roi.copy()
        roi[1,0] -=0.5
        roi[2,0] -=0.5
        roi[0,0] -=0.4
        roi[3,0] -=0.4
        roi = np.concatenate([roi, roi[[0]]])
        axes[-1].plot(roi[:,1], roi[:,0], 'r')
    axes[-1].set_ylim(im_h-0.5, -0.5)
    #axes[-1].invert_yaxis()

    for ax in axes:
        for label in (ax.get_yticklabels() + ax.get_yticklabels() + ax.get_xticklabels()):
            label.set_fontsize(label_font_factor)

    axes_to_scale = [axes[0]]
    if len(proj) > 1:
        axes_to_scale.append(axes[-2])
    for ax in axes_to_scale:
        ax.set_ylabel('depth [mm]', fontsize=label_font_factor)
        tickpos = np.array([x.get_position()[1] for x in  ax.get_yticklabels()])[1:-1]
        newlabels = scale * np.array(tickpos)
        print(newlabels)
        ax.set_yticks(ticks=tickpos, labels = newlabels)

    fig.suptitle('Spectral indices' + '\n' + location,
                    fontsize=title_font_factor)
    return fig

def create_rgb_image(rgb_image, red_contrast_limits, green_contrast_limits, blue_contrast_limits):
    
    rgb_to_plot = rgb_image.copy()
    rgb_to_plot, _, _, _ = colorify.multichannel_to_rgb(
        rgb_to_plot,
        cmaps=['pure_red', 'pure_green', 'pure_blue'], 
        rescale_type='limits', 
        limits=[red_contrast_limits, green_contrast_limits, blue_contrast_limits],
        proj_type='sum')
    return rgb_to_plot

class SpectralPlotter(NapariMPLWidget):
    """Subclass of napari_matplotlib NapariMPLWidget for voxel position based time series plotting.
    This widget contains a matplotlib figure canvas for plot visualisation and the matplotlib toolbar for easy option
    controls. The widget is not meant for direct docking to the napari viewer.
    Plot visualisation is triggered by moving the mouse cursor over the voxels of an image layer while holding the shift
    key. The first dimension is handled as time. This widget needs a napari viewer instance and a LayerSelector instance
    to work properly.
    Attributes:
        axes : matplotlib.axes.Axes
        selector : napari_time_series_plotter.LayerSelector
        cursor_pos : tuple of current mouse cursor position in the napari viewer
    """
    def __init__(self, napari_viewer, options=None):
        super().__init__(napari_viewer)
        self.axes = self.canvas.figure.subplots()
        self.cursor_pos = np.array([])
        self.axes.tick_params(colors='white')
       

    def clear(self):
        """
        Clear the canvas.
        """
        #self.axes.clear()
        pass

class SelectRange:
    
    def __init__(self, parent, ax, single=False):
        
        self.ax = ax
        self.single = single
        self.canvas = ax.figure.canvas
        self.parent = parent
        self.myline1 = None
        self.myline2 = None
        self.min_pos = None
        self.max_pos = None
        
        self.span = SpanSelector(ax, onselect=self.onselect, direction='horizontal',
                                 interactive=True, props=dict(facecolor='blue', alpha=0.5))#, button=1)
        

    def onselect(self, min_pos, max_pos):
        
        if self.myline1 is not None:
            self.myline1.pop(0).remove()
        if self.myline2 is not None:
            self.myline2.pop(0).remove()
        
        min_max = [self.ax.lines[0].get_data()[1].min(),
                   self.ax.lines[0].get_data()[1].max()]

        self.myline2 = self.ax.plot([max_pos, max_pos], min_max, 'r')
        if not self.single:
            self.myline1 = self.ax.plot([min_pos, min_pos], min_max, 'r')
            self.min_pos = min_pos
        
        self.max_pos = max_pos
        
    def disconnect(self):
        self.span.disconnect_events()
        self.canvas.draw_idle()