import numpy as np
from cmap import Colormap
from napari.utils import colormaps
from microfilm import colorify

def plot_spectral_profile(rgb_image, mask, index_obj, format_dict, scale=1,
                          scale_unit='mm', location="", fig=None, roi=None, left_margin=0,
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

    if color_plotline == []:
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
        plot_spectral_profile(rgb_image, mask, index_obj, format_dict, scale=scale,
                          scale_unit=scale_unit, location=location, fig=fig, roi=roi,
                          left_margin=left_margin, right_margin=right_margin,
                          bottom_margin=bottom_margin, top_margin=top_margin,
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
                                scale_unit='mm', location="", fig=None, roi=None,
                                left_margin=0, right_margin=0, bottom_margin=0,
                                top_margin=0, repeat=True):

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
        if color_plotline == []:
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
                                top_margin=top_margin, repeat=False)

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
