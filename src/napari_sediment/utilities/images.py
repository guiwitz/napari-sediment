import numpy as np
import tifffile

def save_rgb_tiff_image(image_list, contrast_list, path_to_save):

    for ind, image_lims in enumerate(zip(image_list, contrast_list)):
        
        image = image_lims[0]
        lims = image_lims[1]
        
        image = image.astype(float)
        
        image = (image - lims[0]) / (lims[1] - lims[0])
        image[image < 0] = 0
        image[image > 1] = 1
        image = (image *255).astype(np.uint8)
        image_list[ind] = image
    
    image_stack = np.stack(image_list, axis=0)

    tifffile.imwrite(path_to_save, image_stack)