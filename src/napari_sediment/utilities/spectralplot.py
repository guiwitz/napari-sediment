import os
from dataclasses import asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmap import Colormap
from napari.utils import colormaps
from microfilm import colorify
import tifffile
import cmap
from matplotlib import cm
import matplotlib as mpl
from ..utilities.io import get_mask_roi, get_rgb_roi, load_plots_params, load_project_params
from ..utilities.spectralindex_compute import (load_index_series, save_index_zarr, load_index_zarr, load_projection,
                                               compute_and_clean_index, compute_index_projection,
                                               compute_overlay_RGB)
from ..data_structures.parameters_plots import Paramplot



def plot_spectral_profile(rgb_image, mask, index_obj, format_dict, scale=1,
                          scale_unit='mm', location="", fig=None, roi=None, left_margin=0,
                          right_margin=0, bottom_margin=0, top_margin=0,
                          repeat=True, color_proj_by_index=True, add_colorbar=True):
    """Given data inputs of RGB image, mask and index, create a figure with RGB image,
    index map and index projection. Typically called by plot_experiment_roi_index."""
    if isinstance(index_obj, list):
        if len(index_obj) > 2:
            raise ValueError('Only two indices can be plotted at the same time')
    else:
        index_obj = [index_obj]

    if len(index_obj) > 1:
        add_colorbar = False
    
    '''for i in range(len(index_obj)):
        index_name = index_obj[i].index_name
        index_image = index_obj[i].index_map
        proj = index_obj[i].index_proj
        index_contrast_limits = index_obj[i].index_map_range
        index_colormap = index_obj[i].colormap'''

    im_w = index_obj[0].index_map.shape[1]
    im_h = index_obj[0].index_map.shape[0]

    title_font = format_dict['title_font']
    label_font = format_dict['label_font']
    color_plotline = format_dict['color_plotline']
    plot_thickness = format_dict['plot_thickness']
    red_contrast_limits = format_dict['red_contrast_limits']
    green_contrast_limits = format_dict['green_contrast_limits']
    blue_contrast_limits = format_dict['blue_contrast_limits']

    # create RGB overlay image
    index_image, mlp_colormaps = compute_overlay_RGB(index_obj)

    if color_proj_by_index:
        color_plotline = [colormaps.ALL_COLORMAPS[x.colormap].colors[-1,:] for x in index_obj]

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
    if plot_width_inches < 2:
        im_width_inches_new = (a4_margins[1] - 2) / 2
        ratio = im_width_inches_new / im_width_inches
        im_height_inches = im_height_inches * ratio
        im_width_inches = im_width_inches_new
        plot_width_inches = 2

    # The figure and axes are set explicitly to make sure that the axes fill the figure
    # This is achieved using the add_axes method instead of subplots
    fig_size = [a4_size[1], a4_size[0]]
    if fig is None:
        fig = plt.figure()
    fig.clear()
    fig.set_size_inches(fig_size)
    fig.set_facecolor('white')
    
    ax1 = fig.add_axes(rect=(left_margin/a4_size[1], bottom_margin/a4_size[0], im_width_inches/a4_size[1], im_height_inches/a4_size[0]))
    ax2 = fig.add_axes(rect=(im_width_inches/a4_size[1]+left_margin/a4_size[1], bottom_margin/a4_size[0], im_width_inches/a4_size[1], im_height_inches/a4_size[0]))
    ax3_left= (2 * im_width_inches+left_margin)/a4_size[1]
    ax3 = fig.add_axes(rect=(ax3_left, bottom_margin/a4_size[0], plot_width_inches/a4_size[1], im_height_inches/a4_size[0]))
    if add_colorbar:
        #ax2b = fig.add_axes(rect=((2*im_width_inches+left_margin+0.1)/a4_size[1], bottom_margin/a4_size[0],0.8*cbar_frac * plot_width_inches/a4_size[1], 0.5*im_height_inches/a4_size[0]))
        ax2b = fig.add_axes(rect=(1.1 * im_width_inches/a4_size[1] + left_margin/a4_size[1], bottom_margin/a4_size[0] - 0.05 * im_height_inches/a4_size[0], 0.8 * im_width_inches/a4_size[1], 0.04 * im_height_inches/a4_size[0])) 
        norm = mpl.colors.Normalize(vmin=index_obj[0].index_map_range[0], vmax=index_obj[0].index_map_range[1])
        colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=mlp_colormaps[0]), cax=ax2b, orientation='horizontal')

    ax1.imshow(rgb_to_plot, aspect='auto')
    '''if index_contrast_limits is None:
        non_nan = index_image[~np.isnan(index_image)]
        vmin = np.percentile(non_nan, 0.1)
        vmax = np.percentile(non_nan, 99.9)
    else:
        vmin = index_contrast_limits[0]
        vmax = index_contrast_limits[1]'''
    index_image[mask==1,:] = np.nan
    ax2.imshow(index_image, aspect='auto', interpolation='none')

    if roi is not None:
        roi_array = np.array(roi)
        roi_array[1,0] -=0.5
        roi_array[2,0] -=0.5
        roi_array[0,0] -=0.5
        roi_array[3,0] -=0.5
        roi_array = np.concatenate([roi_array, roi_array[[0]]])
        ax2.plot(roi_array[:,1], roi_array[:,0], 'r')
    
    proj = [x.index_proj for x in index_obj]
    mean_proj = [np.nanmean(x) for x in proj]
    for ind, p in enumerate(proj):
        ax3.plot(p, np.arange(len(p)), color=np.array(color_plotline[ind]), linewidth=plot_thickness)
        ax3.plot(np.ones_like(p) * mean_proj[ind], np.arange(len(p)), color=color_plotline[ind], linestyle='--')

    ax3.set_ylim(0, len(proj[0]))
    ax3.yaxis.tick_right()
    ax3.invert_yaxis()
    
    # set y axis scale
    for ax in [ax1]:
        tickpos = np.array([x.get_position()[1] for x in  ax.get_yticklabels()])[1:-1]
        new_labels = scale * np.array(tickpos)
        tickdist = (new_labels[-1] - new_labels[0]) / 10
        order_of_mag = 10 ** int(np.floor(np.log10(tickdist)))
        tickdist = order_of_mag * (tickdist // order_of_mag)
        new_labels = np.arange(0, new_labels[-1] + 5 * tickdist, tickdist)
        new_labels = new_labels[new_labels <= im_h * scale]
        new_tickpos = new_labels / scale
        ax.set_yticks(ticks=new_tickpos, labels=new_labels)
    ax3.set_yticks(ticks=new_tickpos, labels=new_labels)

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    for label in (ax1.get_yticklabels() + ax3.get_yticklabels() + ax3.get_xticklabels()):
        label.set_fontsize(label_font)
    
    ax2.set_ylim(im_h - 0.5, -0.5)
    ax1.set_ylabel(f'depth [{scale_unit}]', fontsize=label_font)
    ax3.set_ylabel(f'depth [{scale_unit}]', fontsize=label_font)
    ax3.set_xlabel('Index value', fontsize=label_font)
    ax3.yaxis.set_label_position('right')
    full_name = ", ".join([x.index_name for x in index_obj])
    suptitle = fig.suptitle(full_name + '\n' + location,
                    fontsize=title_font)
    
    # redraw to make sure, measures are ok (see https://stackoverflow.com/questions/69252219/uncorrect-bbox-coordinates-with-get-window-extent-method-and-transform-option)
    fig.canvas.draw()
    # add colorbar
    lowest_y = get_text_miny(ax3.xaxis.label, fig.canvas.get_renderer(), fig)
    if add_colorbar:
        texts = ax2b.get_xticklabels()
        cbar_miny = get_text_miny(texts[0], fig.canvas.get_renderer(), fig)
        lowest_y = min(lowest_y, cbar_miny)

    mean_text = [f'Mean {index_obj[i].index_name}: {mean_proj[i]:.2f}' for i in range(len(index_obj))]
    mean_text = '\n'.join(mean_text)
    description_text = fig.text(ax3_left, lowest_y - 0.1, mean_text, ha='left', fontsize=label_font)
    
    # check the size of titles labels and tickmarks, adjust margins accordingly,
    # and repeat the plot
    # redraw to make sure, measures are ok (see https://stackoverflow.com/questions/69252219/uncorrect-bbox-coordinates-with-get-window-extent-method-and-transform-option)
    fig.canvas.draw()
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
    bottom_margin = -get_text_miny(description_text, renderer, fig) * a4_size[0]

    # adjust top margin
    bbox = suptitle.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    title_height = bbox.ymax - bbox.ymin
    top_margin = 2 * title_height * a4_size[0]

    if repeat:
        # remake plot with adjusted margins adapted to content 
        plot_spectral_profile(rgb_image, mask, index_obj, format_dict, scale=scale,
                          scale_unit=scale_unit, location=location, fig=fig, roi=roi,
                          left_margin=left_margin, right_margin=right_margin,
                          bottom_margin=bottom_margin, top_margin=top_margin,
                          repeat=False, color_proj_by_index=color_proj_by_index, add_colorbar=add_colorbar)


    return fig, ax1, ax2, ax3

def plot_spectral_profile_original(rgb_image, mask, index_obj, format_dict, scale=1,
                          scale_unit='mm', location="", fig=None, roi=None, left_margin=0,
                          right_margin=0, bottom_margin=0, top_margin=0,
                          repeat=True, color_proj_by_index=True):
    """Given data inputs of RGB image, mask and index, create a figure with RGB image,
    index map and index projection. Typically called by plot_experiment_roi_index."""
    
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

    if color_proj_by_index:
        color_plotline = colormaps.ALL_COLORMAPS[index_colormap].colors[-1,:]

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
    if plot_width_inches < 2:
        im_width_inches_new = (a4_margins[1] - 2) / 2
        ratio = im_width_inches_new / im_width_inches
        im_height_inches = im_height_inches * ratio
        im_width_inches = im_width_inches_new
        plot_width_inches = 2

    # The figure and axes are set explicitly to make sure that the axes fill the figure
    # This is achieved using the add_axes method instead of subplots
    fig_size = [a4_size[1], a4_size[0]]
    if fig is None:
        fig = plt.figure()
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
        roi_array = np.array(roi)
        roi_array[1,0] -=0.5
        roi_array[2,0] -=0.5
        roi_array[0,0] -=0.5
        roi_array[3,0] -=0.5
        roi_array = np.concatenate([roi_array, roi_array[[0]]])
        ax2.plot(roi_array[:,1], roi_array[:,0], 'r')
    
    ax3.plot(proj, np.arange(len(proj)), color=np.array(color_plotline), linewidth=plot_thickness)
    ax3.plot(np.ones_like(proj) * np.nanmean(proj), np.arange(len(proj)), color='black', linestyle='--')

    ax3.set_ylim(0, len(proj))
    ax3.yaxis.tick_right()
    ax3.invert_yaxis()
    
    # set y axis scale
    for ax in [ax1]:
        tickpos = np.array([x.get_position()[1] for x in  ax.get_yticklabels()])[1:-1]
        new_labels = scale * np.array(tickpos)
        tickdist = (new_labels[-1] - new_labels[0]) / 10
        order_of_mag = 10 ** int(np.floor(np.log10(tickdist)))
        tickdist = order_of_mag * (tickdist // order_of_mag)
        new_labels = np.arange(0, new_labels[-1] + 5 * tickdist, tickdist)
        new_labels = new_labels[new_labels <= im_h * scale]
        new_tickpos = new_labels / scale
        ax.set_yticks(ticks=new_tickpos, labels=new_labels)
    ax3.set_yticks(ticks=new_tickpos, labels=new_labels)

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    for label in (ax1.get_yticklabels() + ax3.get_yticklabels() + ax3.get_xticklabels()):
        label.set_fontsize(label_font)
    
    ax2.set_ylim(im_h - 0.5, -0.5)
    ax1.set_ylabel(f'depth [{scale_unit}]', fontsize=label_font)
    ax3.set_ylabel(f'depth [{scale_unit}]', fontsize=label_font)
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
        # remake plot with adjusted margins adapted to content 
        plot_spectral_profile(rgb_image, mask, index_obj, format_dict, scale=scale,
                          scale_unit=scale_unit, location=location, fig=fig, roi=roi,
                          left_margin=left_margin, right_margin=right_margin,
                          bottom_margin=bottom_margin, top_margin=top_margin,
                          repeat=False, color_proj_by_index=color_proj_by_index)


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

def get_text_miny(text, renderer, fig):
    bbox = text.get_window_extent(renderer)
    # Convert from display to figure coordinates
    bbox = bbox.transformed(fig.transFigure.inverted())
    return bbox.ymin


def plot_multi_spectral_profile(rgb_image, mask, index_objs, format_dict, scale=1,
                                scale_unit='mm', location="", fig=None, roi=None,
                                left_margin=0, right_margin=0, bottom_margin=0,
                                top_margin=0, repeat=True, color_proj_by_index=True):
    
    """
    Given data inputs of RGB image, mask and indices, create a figure of
    multi-spectral profiles. Typically called by plot_experiment_multispectral_roi.
    """

    title_font = format_dict['title_font']
    label_font = format_dict['label_font']
    color_plotline = format_dict['color_plotline']
    plot_thickness = format_dict['plot_thickness']
    red_contrast_limits = format_dict['red_contrast_limits']
    green_contrast_limits = format_dict['green_contrast_limits']
    blue_contrast_limits = format_dict['blue_contrast_limits']
    
    rgb_to_plot = create_rgb_image(rgb_image, red_contrast_limits, green_contrast_limits, blue_contrast_limits)
    rgb_to_plot[mask==1,:] = 0

    #a4_size = np.array([11.69, 8.27])
    a4_size = np.array([8.27, 11.69])
    a4_margins = a4_size - np.array([bottom_margin + top_margin, left_margin + right_margin])

    im_h = rgb_image[0].shape[0]
    im_w = rgb_image[0].shape[1]

    pixel_in_inches = a4_margins[0] / im_h
    im_height_inches = a4_margins[0]
    im_width_inches = im_w * pixel_in_inches
    plot_width_inches = a4_margins[1] / (len(index_objs) + 1)
    if plot_width_inches > 2:
        plot_width_inches = 2
    
    im_with_for_plot = plot_width_inches
    if im_with_for_plot < im_width_inches:
        #ratio = im_width_inches / plot_width_inches
        #im_height_inches = im_height_inches / ratio
        #plot_width_inches = plot_width_inches / ratio
        im_with_for_plot = im_width_inches

    width_tot = len(index_objs) * plot_width_inches + im_with_for_plot
    if width_tot > a4_margins[1]:
        ratio = width_tot / a4_margins[1]
        im_height_inches = im_height_inches / ratio
        plot_width_inches = plot_width_inches / ratio
        im_with_for_plot = im_with_for_plot / ratio
    
    fig_size = [a4_size[1], a4_size[0]]
    if fig is None:
        fig = plt.figure()
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

        # get line color from colormap
        if color_proj_by_index:
            current_color = colormaps.ALL_COLORMAPS[index_objs[i].colormap].colors[-1,:]
        else:
            current_color = np.array(color_plotline)

        axes.append(fig.add_axes(rect=(
            (left_margin + (i * plot_width_inches + shift * im_with_for_plot)) / a4_size[1],
            bottom_margin / a4_size[0], plot_width_inches / a4_size[1],
            im_height_inches / a4_size[0])))
        axes[-1].plot(proj, np.arange(len(proj)), color=current_color, linewidth=plot_thickness)
        axes[-1].plot(np.ones_like(proj) * np.nanmean(proj), np.arange(len(proj)), color='black', linestyle='--')
        axes[-1].set_ylim(0, len(proj))
        if (i!=0) and (i!=len(proj)-1):
            axes[-1].yaxis.set_visible(False)
        if i == len(index_objs)-1:
            axes[-1].yaxis.tick_right()
            axes[-1].yaxis.set_label_position('right')
        axes[-1].invert_yaxis()
        plot_title = axes[-1].set_title(index_name, fontsize=title_font)
    
    axes.append(fig.add_axes(rect=(
        (left_margin + halfplot * plot_width_inches) / a4_size[1], 
        bottom_margin / a4_size[0], im_with_for_plot / a4_size[1],
        im_height_inches / a4_size[0])))
    
    axes[-1].imshow(rgb_to_plot)
    axes[-1].yaxis.set_visible(True)
    axes[-1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[-1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    # trying and failing to get the rectangle to be the right size
    '''
    rect_width = 2*im_w  # Image width + extra padding
    rect_height = im_h#im_height_inches  # Image height + extra padding
    rect_x = 0  # Start slightly outside the left edge of the image
    rect_y = 0  # Start slightly outside the bottom edge of the image

    # Create the rectangle
    import matplotlib.patches as patches
    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=2, edgecolor='k', facecolor='none')
    # Add the rectangle to the axes
    axes[-1].add_patch(rect)
    '''

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
            label.set_fontsize(label_font)

    axes_to_scale = [axes[0]]
    if len(proj) > 1:
        axes[-2].yaxis.set_visible(True)
        axes[-2].yaxis.tick_right()
        axes[-2].yaxis.set_label_position('right')
        axes_to_scale.append(axes[-2])

    for ax in axes_to_scale:
        ax.set_ylabel(f'depth [{scale_unit}]', fontsize=label_font)
        tickpos = np.array([x.get_position()[1] for x in  ax.get_yticklabels()])[1:-1]
        newlabels = scale * np.array(tickpos)
        ax.set_yticks(ticks=tickpos, labels = newlabels)
        ax.tick_params(axis='both', labelsize=10)  # Set x-axis tick labels size

    for ax in axes:
        ax.tick_params(axis='x', labelsize=10)  # Set x-axis tick labels size


    suptitle = fig.suptitle('Spectral indices' + '\n' + location,
                    fontsize=title_font, y=0.95)
    
    # adjust left margin
    renderer = fig.canvas.get_renderer()
    text = axes[0].yaxis.label
    label_width = get_text_width(text, renderer, fig)
    y_tick_widths = [get_text_width(label, renderer, fig) for label in axes[0].get_yticklabels()]
    max_y_tick_width = max(y_tick_widths)
    left_margin = 1.0 * (3 * label_width + max_y_tick_width) * a4_size[1]

    # adjust right margin
    text = axes[-2].yaxis.label
    label_width = get_text_width(text, renderer, fig)
    y_tick_widths = [get_text_width(label, renderer, fig) for label in axes[-2].get_yticklabels()]
    max_y_tick_width = max(y_tick_widths)
    right_margin = 1.0 * (3 * label_width + max_y_tick_width) * a4_size[1]

    # adjust bottom margin
    text = axes[0].xaxis.label
    label_height = get_text_height(text, renderer, fig)
    x_tick_heights = [get_text_height(label, renderer, fig) for label in axes[0].get_xticklabels()]
    max_x_tick_height = max(x_tick_heights)
    bottom_margin = 3.0 * max_x_tick_height * a4_size[0]

    # adjust top margin
    bbox = plot_title.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    title_height = bbox.ymax - bbox.ymin
    top_margin = 1 + 2 * title_height * a4_size[0]

    if repeat:
        plot_multi_spectral_profile(rgb_image, mask, index_objs, format_dict, scale=scale,
                                scale_unit=scale_unit, location=location, fig=fig,
                                roi=roi, left_margin=left_margin,
                                right_margin=right_margin, bottom_margin=bottom_margin,
                                top_margin=top_margin, repeat=False, color_proj_by_index=color_proj_by_index)

    return fig

def create_rgb_image(rgb_image, red_contrast_limits, green_contrast_limits, blue_contrast_limits):
    """Create an MxNx3 RGB image with adjusted contrast limits."""

    rgb_to_plot = rgb_image.copy()
    rgb_to_plot, _, _, _ = colorify.multichannel_to_rgb(
        rgb_to_plot,
        cmaps=['pure_red', 'pure_green', 'pure_blue'], 
        rescale_type='limits', 
        limits=[red_contrast_limits, green_contrast_limits, blue_contrast_limits],
        proj_type='sum')
    return rgb_to_plot

def plot_experiment_roi_index(export_folder, index, params_plots=None, main_roi_index=0,
                    mask=None, rgb_cube=None, myimage=None, add_colorbar=True, load_data=False):
    """Plot spectral index for a given experiment and region of interest. Data
    such as the mask and rgb image are loaded from the export folder, but can also
    be provided as arguments to avoid repeated loading of the same data. This function
    is a wrapper around the plot_spectral_profile function providing the necessary data
    based on the export folder.
    
    Parameters
    ----------
    export_folder : str
        path to the export folder containing params file
    index : object
        index object from data_structures.spectralindex
    params_plots : object
        plot parameters object from data_structures.parameters_plots
    main_roi_index : int
        index of the main roi to plot
    mask : np.ndarray
        mask for the roi
    rgb_cube : np.ndarray
        rgb image for the roi
    myimage : object
        ImChannels object with image data
    add_colorbar : bool
        whether to add a colorbar to the index plot
    load_data : bool
        whether to load pre-computed zarr index map and projection files

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure with the plot
    
    """

    params = load_project_params(export_folder)
    if params_plots is None:
        params_plots = Paramplot()
    format_dict = asdict(params_plots)
    
    row_bounds, col_bounds = params.get_formatted_col_row_bounds(main_roi_index)
    measurement_rois = params.get_formatted_measurement_roi()
    measurement_roi = measurement_rois[main_roi_index]
    colmin_proj = measurement_roi[:,1].min()
    colmax_proj = measurement_roi[:,1].max()

    # get mask
    if mask is None:
        mask = get_mask_roi(main_folder=export_folder, main_roi_index=main_roi_index)

    # get RGB image
    if (rgb_cube is None) or (myimage is None): 
        rgb_cube, myimage = get_rgb_roi(main_folder=export_folder, main_roi_index=main_roi_index)
    
    # adjust rgb contrast
    if params_plots.red_contrast_limits == []:
        min_contrast = np.percentile(rgb_cube, 1, axis=[1,2])
        max_contrast = np.percentile(rgb_cube, 99, axis=[1,2])
        params_plots.red_contrast_limits = [min_contrast[0], max_contrast[0]]
        params_plots.green_contrast_limits = [min_contrast[1], max_contrast[1]]
        params_plots.blue_contrast_limits = [min_contrast[2], max_contrast[2]]

    # compute index map and projection
    if isinstance(index, list):
        if len(index) > 2:
            raise ValueError('Only one or two indices can be plotted at a time')
    else:
        index = [index]
    
    for ind in index:
        if load_data:
            ind.index_map = load_index_zarr(
                project_folder=export_folder, main_roi_index=main_roi_index, index_name=ind.index_name)
            ind.index_proj = load_projection(project_folder=export_folder, main_roi_index=main_roi_index, index_name=ind.index_name)
        
        # if data is not loaded or inexistent, compute it
        if ind.index_map is None:
            ind.index_map = compute_and_clean_index(
                    spectral_index=ind, row_bounds=row_bounds,
                    col_bounds=col_bounds, imagechannels=myimage)
        if ind.index_proj is None:         
            ind.index_proj = compute_index_projection(
                    ind.index_map, mask,
                    colmin=colmin_proj, colmax=colmax_proj,
                    smooth_window=ind.smooth_proj_window)
    
    fig, ax1, ax2, ax3 = plot_spectral_profile(
        rgb_image=rgb_cube, mask=mask, index_obj=index,
        format_dict=format_dict, scale=params.scale, scale_unit=params.scale_units,
        location=params.location, fig=None, 
        roi=measurement_roi, repeat=True, add_colorbar=add_colorbar)
    
    return fig, mask, rgb_cube, myimage

def plot_experiment_multispectral_roi(export_folder, indices, params_plots=None, main_roi_index=0,
                    mask=None, rgb_cube=None, myimage=None, recompute=True):
    """Plot multispectral index projections for a given experiment and region of interest. Data
    such as the mask and rgb image are loaded from the export folder, but can also
    be provided as arguments to avoid repeated loading of the same data. This function
    is a wrapper around the plot_multi_spectral_profile function providing the necessary data
    based on the export folder.
    
    Parameters
    ----------
    export_folder : str
        path to the export folder containing params file
    indices : list
        list of index objects from data_structures.spectralindex
    params_plots : object
        plot parameters object from data_structures.parameters_plots
    main_roi_index : int
        index of the main roi to plot
    mask : np.ndarray
        mask for the roi
    rgb_cube : np.ndarray
        rgb image for the roi
    myimage : object
        ImChannels object with image data
    recompute : bool
        whether to recompute the index maps and projections
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        figure with the plot
    
    """

    params = load_project_params(export_folder)
    if params_plots is None:
        params_plots = Paramplot()
    format_dict = asdict(params_plots)

    row_bounds, col_bounds = params.get_formatted_col_row_bounds(main_roi_index)
    measurement_rois = params.get_formatted_measurement_roi()
    measurement_roi = measurement_rois[main_roi_index]
    colmin_proj = measurement_roi[:,1].min()
    colmax_proj = measurement_roi[:,1].max()

    # get mask
    if mask is None:
        mask = get_mask_roi(main_folder=export_folder, main_roi_index=main_roi_index)

    # get RGB image
    if (rgb_cube is None) or (myimage is None): 
        rgb_cube, myimage = get_rgb_roi(main_folder=export_folder, main_roi_index=main_roi_index)

    measurement_rois = params.get_formatted_measurement_roi()
    measurement_roi = measurement_rois[main_roi_index]

    if recompute is True:
        # compute index map and projection
        for k in indices.keys():
            indices[k].index_map = compute_and_clean_index(
                    spectral_index=indices[k], row_bounds=row_bounds,
                    col_bounds=col_bounds, imagechannels=myimage)
                    
            indices[k].index_proj = compute_index_projection(
                        indices[k].index_map, mask,
                        colmin=colmin_proj, colmax=colmax_proj,
                        smooth_window=indices[k].smooth_proj_window)

    fig = plot_multi_spectral_profile(
        rgb_image=rgb_cube, mask=mask,
        index_objs=[indices[k] for k in indices.keys()], 
        format_dict=format_dict, scale=params.scale, scale_unit=params.scale_units,
        fig=None, roi=measurement_roi, repeat=True)
    
    return fig

def batch_create_plots(project_list, index_params_file, plot_params_file,
                       normalize=False, load_data=False, roi_to_process=None):
    """Create index plots for a list of projects. Calls plot_experiment_roi_index and 
    plot_experiment_multispectral_roi for each project and roi.
    
    Parameters
    ----------
    project_list: list of Path
        list of project folders (containing Parameters.yml)
    index_params_file: Path
        path to index parameters file
    plot_params_file: Path
        path to plot parameters file
    normalize: bool
        whether to save plots in normalized folder
    load_data: bool
        whether to load pre-computed zarr index map and projection files
    roi_to_process: int
        index of the roi to process. If None, all rois are processed.

    """

    params_plots = load_plots_params(plot_params_file)
    fig, ax = plt.subplots()
    dpi = 300

    for ex in project_list:

        params = load_project_params(folder=ex)
        if roi_to_process is not None:
            roi_folders = [f'roi_{roi_to_process}']
        else:
            roi_folders = list(ex.glob('roi*'))
            roi_folders = [x.name for x in roi_folders if x.is_dir()]
            
            if len(roi_folders) == 0:
                os.makedirs(ex.joinpath('roi_0'))
                roi_folders = ['roi_0']

        for roi_ind in range(len(roi_folders)):
            
            # index are loaded here to reset them for each roi
            indices = load_index_series(index_params_file)
            
            roi_folder = ex.joinpath(f'roi_{roi_ind}')
            if normalize:
                roi_plot_folder = roi_folder.joinpath('index_plots_normalized')
                if not roi_plot_folder.exists():
                    roi_plot_folder.mkdir()
            else:
                roi_plot_folder = roi_folder.joinpath('index_plots')
                if not roi_plot_folder.exists():
                    roi_plot_folder.mkdir()

            mask, myimage, rgb_cube = None, None, None
            proj_pd = None
            for k in indices.keys():
                
                fig, mask, rgb_cube, myimage = plot_experiment_roi_index(
                    export_folder=ex, index=indices[k], params_plots=params_plots,
                    main_roi_index=roi_ind, mask=mask, rgb_cube=rgb_cube, myimage=myimage, load_data=load_data)

                fig.savefig(
                    roi_plot_folder.joinpath(f'{indices[k].index_name}_index_plot.png'), dpi=dpi)
                
                # tif and zarr maps
                # only overwrite if data are not loaded (even if they exist). If load_data is true but the data do not exist,
                # then the data are saved.
                overwrite = True
                if load_data:
                    overwrite = False
                index_map = indices[k].index_map
                contrast = indices[k].index_map_range
                napari_cmap = indices[k].colormap
                export_path = roi_plot_folder.joinpath(f'{indices[k].index_name}_index_map.tif')
                save_tif_cmap(image=index_map, image_path=export_path,
                                napari_cmap=napari_cmap, contrast=contrast, overwrite=overwrite)
                save_index_zarr(
                    project_folder=ex, main_roi_index=roi_ind, index_name=indices[k].index_name,
                    index_map=indices[k].index_map, overwrite=overwrite)
                
                # export projection to csv
                if proj_pd is None:
                    proj_pd = pd.DataFrame({'depth': np.arange(0, len(indices[k].index_proj))})
                proj_pd[indices[k].index_name] = indices[k].index_proj
            
            proj_pd[f'depth [{params.scale_units}]'] = proj_pd['depth'] * params.scale
            if overwrite or not roi_plot_folder.joinpath('index_projection.csv').is_file():
                proj_pd.to_csv(roi_plot_folder.joinpath('index_projection.csv'), index=False)

            # pair plots
            index_keys = list(indices.keys())
            for k in range(len(index_keys)):
                for j in range(k+1, len(index_keys)):
                    fig, mask, rgb_cube, myimage = plot_experiment_roi_index(
                    export_folder=ex, index=[indices[index_keys[k]], indices[index_keys[j]]], params_plots=params_plots,
                    main_roi_index=roi_ind, mask=mask, rgb_cube=rgb_cube, myimage=myimage, load_data=False)

                    fig.savefig(
                            roi_plot_folder.joinpath(f'{indices[index_keys[k]].index_name}_{indices[index_keys[j]].index_name}_index_plot.png'), dpi=dpi)
                    index_image, _ = compute_overlay_RGB(index_obj=[indices[index_keys[k]], indices[index_keys[j]]])
                    index_image[np.isnan(indices[index_keys[k]].index_map),:] = 1
                    tifffile.imwrite(
                        roi_plot_folder.joinpath(f'{indices[index_keys[k]].index_name}_{indices[index_keys[j]].index_name}_index_map.tif'),
                        data=(255*index_image[:, :, :3]).astype(np.uint8))
                    
                    
            # create multi index plot
            fig = plot_experiment_multispectral_roi(export_folder=ex, indices=indices,
                                              params_plots=params_plots, main_roi_index=roi_ind,
                    mask=mask, rgb_cube=rgb_cube, myimage=myimage, recompute=False)

            fig.savefig(roi_plot_folder.joinpath('multi_index_plot'), dpi=dpi)
            plt.close(fig)

def save_tif_cmap(image, image_path, napari_cmap, contrast, overwrite=False):
    """Save image as tiff with colormap using specified contrast. The
    saved image is only for visualization purposes, as the values are
    rescaled and transformed to RGB.

    Parameters
    ----------
    image: np.ndarray
        image to save
    image_path: str
        path to save image
    napari_cmap: napari Colormap
        napari colormap or str
    contrast: tuple of float
        contrast
    overwrite: bool
        whether to overwrite existing file

    """
    
    if image_path.exists() and not overwrite:
        return
    
    if isinstance(napari_cmap, str):
        current_cmap = cmap.Colormap(napari_cmap).to_matplotlib()
    else:
        current_cmap = cmap.Colormap(napari_cmap.colors).to_matplotlib()

    if contrast is None:
        contrast = (np.nanmin(image), np.nanmax(image))
    norm_image = np.clip(image, a_min=contrast[0], a_max=contrast[1])
    norm_image = (norm_image - np.nanmin(norm_image)) / (np.nanmax(norm_image) - np.nanmin(norm_image))
    
    colored_image = current_cmap(norm_image)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

    tifffile.imwrite(image_path, colored_image)
    

