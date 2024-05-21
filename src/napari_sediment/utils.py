import numpy as np
import colour


def update_contrast_on_layer(napari_layer):

    data = np.asarray(napari_layer.data)
          
    napari_layer.contrast_limits_range = (data.min(), data.max())
    napari_layer.contrast_limits = np.percentile(data, (2,98))

def wavelength_to_rgb(min_wavelength, max_wavelength, width):

    min_wavelength = int(min_wavelength)
    max_wavelength = int(max_wavelength)
    all_cols = []
    for val in range(min_wavelength,max_wavelength):
        
        if (val < 361) or (val > 770):
            rgb = [0, 0, 0]
        else:
            sigma = 2
            mu = val
            x = np.arange(mu-10, mu+10)
            spectrum = (1/(sigma * (2 * np.pi)**0.5)) * np.exp(-0.5*((x-mu) / sigma)**2)
            spectrum = {x[i]: spectrum[i] for i in range(len(x))}
            sd = colour.SpectralDistribution(spectrum)
            XYZ = colour.sd_to_XYZ(sd)
            rgb = colour.XYZ_to_sRGB(XYZ)
            rgb = np.clip(rgb, 0, 1)

        all_cols.append(rgb)
    
    all_cols = np.stack(all_cols)
    all_cols = np.ones((width, all_cols.shape[0], all_cols.shape[1])) * all_cols

    return all_cols