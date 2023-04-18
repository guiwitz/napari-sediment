import tifffile

def save_mask(mask, filename):
   
    tifffile.imsave(filename, mask.astype('uint8'))

def load_mask(filename):
    
    mask = tifffile.imread(filename)
    return mask