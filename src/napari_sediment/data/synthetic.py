from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np

from spectral.io.envi import save_image

from skimage.draw import ellipse


def generate_image(min_val, max_val, height, width, channels, pattern_weight,
                  pattern_width=10, random_seed=None):
    """
    Generates an image with dimensions (height, width, channels). Pixel values
    are drawn from a uniform distribution between min_val and max_val. In addition
    a vertical pattern (vertical stripes) is added to the image by drawing for each
    column a value from a normal distribution with mean 0 and standard deviation 10
    which is added with a weight pattern_weight to the pixel values. The output image
    has dimensions (channels, height, width). In order to create the same pattern
    on different images, the random seed can be set.

    Parameters
    ----------
    min_val : int
        minimum value for the image random values
    max_val : int
        maximum value for the image random values
    height : int
        height of the image
    width : int
        width of the image
    channels : int
        number of channels of the image
    pattern_weight : float
        weight of added pattern
    pattern_width : int
        width (sigma) of the pattern values
    random_seed : int
        random seed for reproducibility
    
    Returns
    -------
    im_ref : array
        generated image
    
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    pattern = np.random.normal(0, pattern_width, width)#
    pattern[pattern<0] = 0
    pattern = pattern.astype(np.uint16)
    np.random.seed()
    im_ref = np.random.randint(min_val, max_val, (height, width, channels), dtype=np.uint16)
    im_ref = np.moveaxis(np.moveaxis(im_ref,1,2) + pattern_weight*pattern, 2,1)
    
    return im_ref

def generate_synthetic_dataset(image_mean, image_std, min_val, max_val, height, width, 
                               ref_height, channels, white_ref_added_signal, pattern_weight, 
                               pattern_width=10, random_seed=42, data_name='Synthetic',
                               save_path=None):
    """
    Generate a synthetic dataset for testing purposes.

    Parameters
    ----------
    image_mean : int
        mean value for the image gaussian noise
    image_std : int
        standard deviation for the image gaussian noise
    min_val : int
        minimum value for the background random values
    max_val : int
        maximum value for the background random values
    height : int
        height of the image
    width : int
        width of the image
    ref_height : int
        height of the background images
    channels : int
        number of channels of the image
    white_ref_added_signal : float
        constant value added to the white reference
    pattern_weight : float
        weight of added pattern to background
    pattern_width : int
        width (sigma) of the pattern values
    random_seed : int
        random seed for reproducibility
    main_path : str or Path
        path where the images are saved, by default images are not saved
    data_name : str
        name of the dataset, by default 'Synthetic'
    
    Returns
    -------
    im_test : array
        test image
    dark_ref : array
        dark reference image
    dark_for_white_ref : array
        dark reference image for white reference
    white_ref : array
        white reference image

    """

    dark_ref = generate_image(min_val=min_val, max_val=max_val,
                             height=ref_height, width=width,
                             channels=channels, pattern_weight=pattern_weight,
                             pattern_width=pattern_width, random_seed=random_seed)
    dark_for_white_ref = generate_image(min_val=min_val, max_val=max_val,
                                       height=ref_height, width=width,
                                       channels=channels, pattern_weight=pattern_weight,
                                       pattern_width=pattern_width, random_seed=random_seed)

    white_ref = generate_image(min_val=min_val, max_val=max_val,
                              height=ref_height, width=width,
                              channels=channels, pattern_weight=pattern_weight,
                              pattern_width=pattern_width, random_seed=random_seed)
    white_ref = white_ref + white_ref_added_signal
    
    im_test = generate_image(min_val=min_val, max_val=max_val,
                            height=height, width=width,
                            channels=channels, pattern_weight=pattern_weight,
                            pattern_width=pattern_width, random_seed=random_seed)
    
    im_test = im_test + np.random.normal(image_mean, image_std, im_test.shape).astype(np.uint16)

    

    return im_test, dark_ref, dark_for_white_ref, white_ref

def save_test_dataset(data_name, save_path, **kwargs):
    """
    Save a synthetic dataset for testing purposes.

    Parameters
    ----------
    data_name : str
        name of the dataset
    save_path : str or Path
        path where the images are saved
    **kwargs : dict
        parameters for generate_synthetic_dataset
    
    Returns
    -------
    
    """

    im_test, dark_ref, dark_for_white_ref, white_ref = generate_synthetic_dataset(**kwargs)
    
    im_test = add_signal_to_image(im_test=im_test, widths=[15, 30], ch_positions = [40, 40],
                                        row_boundaries=[[10,20], [60,70]], col_boundaries=[[10,110],[10,110]], amplitudes=[-400, -400], channels=80)

    im_test = add_ellipse_to_image(im_test, 100, 37, 10, 20, -600)

    main_path = Path(save_path)
    
    os.makedirs(main_path.joinpath(f'{data_name}/capture').as_posix(), exist_ok=True)
    os.makedirs(main_path.joinpath(f'{data_name}_WR_/capture').as_posix(), exist_ok=True)

    metadata = {'wavelength': [str(x) for x in np.linspace(300, 900, kwargs['channels'])], 'interleave': 'bil'}
    save_image(
        hdr_file=main_path.joinpath(f'{data_name}/capture/{data_name}.hdr'),
        image=im_test, ext='raw', force=True, metadata=metadata, interleave='bil')
    
    save_image(
        hdr_file=main_path.joinpath(f'{data_name}/capture/DARKREF_{data_name}.hdr'),
        image=dark_ref, ext='raw', force=True, metadata=metadata, interleave='bil')
    
    save_image(
        hdr_file=main_path.joinpath(f'{data_name}_WR_/capture/DARKREF_{data_name}.hdr'),
        image=dark_for_white_ref, ext='raw', force=True, metadata=metadata, interleave='bil')
    save_image(
        hdr_file=main_path.joinpath(f'{data_name}_WR_/capture/WHITEREF_{data_name}.hdr'),
        image=white_ref, ext='raw', force=True, metadata=metadata, interleave='bil')
    

def mat1(pos, mu=10, A=10, im_ch=100):
    """Create a 1D gaussian pattern with a given position within a signal of
    length im_ch."""

    ransig = np.random.normal(loc=mu, scale=1)
    if ransig <=0:
        ransig=0.001
    
    return A*np.exp(-(np.arange(0,im_ch)-pos)**2 /ransig)# + 0.01*np.random.randint(0,10,(im_ch),dtype=np.uint16)

def add_signal_to_image(im_test, widths, ch_positions, row_boundaries, 
                        col_boundaries, amplitudes, channels):
    """Given a test image, e.g. generated with generate_synthetic_dataset, add
    some specific signal to it at chosen spatial and spectral locations.
    
    Parameters
    ----------
    im_test : array
        test image to which the signal is added
    widths : list of int
        widths of the signals
    ch_positions : list of int
        positions of the signals in the channels
    row_boundaries : list of list int
        boundaries for the rows where the signal is added
    col_boundaries : list of list int
        boundaries for the columns where the signal is added
    amplitudes : list of int
        amplitudes of the signals
    channels : int
        number of channels of the image
    
    Returns
    -------
    im_test : array
        test image with the added signals
    """

    for width, ch_position, amplitude, rows, cols in zip(widths, ch_positions, amplitudes, row_boundaries, col_boundaries):
        for i in range(rows[0],rows[1]):
            for j in range(cols[0],cols[1]):
                im_test[i, j, :] = im_test[i, j, :] + mat1(pos=ch_position, mu=width, A=amplitude, im_ch=channels)
    
    return im_test

def add_ellipse_to_image(im_test, r, c, r_radius, c_radius, amplitude):
    """Given a test image, e.g. generated with generate_synthetic_dataset, add
    an ellipse to it at a specific spatial location.

    Parameters
    ----------
    im_test : array
        test image to which the signal is added
    r : int
        row position of the ellipse
    c : int
        column position of the ellipse
    r_radius : int
        radius of the ellipse in the row direction
    c_radius : int
        radius of the ellipse in the column direction
    amplitude : float
        amplitude of the signal
    
    Returns
    -------
    im_test : array
        test image with the added ellipse
    """
    rr, cc = ellipse(r=r, c=c, r_radius=r_radius, c_radius=c_radius, shape=im_test.shape)
    im_test[rr, cc, :] = im_test[rr, cc, :] + np.asarray(amplitude)
    im_test[im_test<0] = 0
    im_test = im_test.astype(np.uint16)
    
    return im_test

            
    