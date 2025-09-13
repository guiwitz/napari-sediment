import requests
import re
from pathlib import Path
from warnings import warn
from napari.utils.colormaps import Colormap as NpColormap
import numpy as np
from napari.utils.colormaps.colormap_utils import (ensure_colormap)
from cmap._catalog import Catalog
from cmap import Colormap
from natsort import natsorted

def get_all_neurocyto_links():
    """Get all links to the neurocyto colormaps."""

    # URL of the webpage
    url = 'https://sites.imagej.net/NeuroCyto-LUTs/luts/'

    # Send a GET request
    response = requests.get(url)

    # Check if the request was successful
    all_links = []
    if response.status_code == 200:
        # Extract all href links using a regex
        links = re.findall(r'href="([^"]+)"', response.text)
        
        # Filter out parent directory link and print full URLs
        for link in links:
            if link != '../':  # Exclude parent directory link
                full_url = url + link
                all_links.append(full_url)
    else:
        warn(f"Failed to get links from {url}")
        return []

    all_links = all_links[5:]

    return all_links

def get_neurocyto_colormaps():
    """Add all neurocyto colormaps to the napari colormaps.
    DO NOT USE FOR THE MOMENT. THE LARGE NUMBER OF REQUEST WILL
    BLOCK ACCESS TO THE SITE. ALSO NOT ALL LUTS ARE FORMATTED THE
    SAME WAY. FOR EXAMPLE SOME HAVE HEADERS SOME NOT, SOME ARE
    HAVE TO BE DOWNLOADED DIRECTLY SOME NOT."""
    
    NapariColormaps = {}
    all_links = get_all_neurocyto_links()
    for lut_path in all_links:
        try:
            lut = np.loadtxt(lut_path, delimiter="\t", skiprows=1)
            NapariColormaps[Path(lut_path).stem] = ensure_colormap(NpColormap(lut[:, 1:4] / 255, name=Path(lut_path).stem, display_name=Path(lut_path).stem))
        except:
            warn(f"Failed to load colormap from {lut_path}")

def get_cmap_catalogue():
    """Get all colormaps from the cmap package and add them to the napari colormaps.
    Using the ensure_colormap function from napari adds the colormaps to the napari colormaps.
    Note that colormap names have not be handled carefully. Two formats appear:
    - chrislet:bop_blue This name appears in the menu of the colormaps. This name cannot
    be used e.g. to add an image using the add_image function. However, this is the name
    to be used with cmap.Colormap. The name can also be recovered via the _display_name attribute of
    a napari Colormap object.
    - chrislet_bop_blue This is the name that can be used with add_image. This name cannot be
    used to get the colormap from cmap.Colormap. The name can also be recovered via the name attribute of
    a napari Colormap object and of a napari layer.
    """
    
    catalog = Catalog()
    cmap_names = catalog.unique_keys(prefer_short_names=False, categories='sequential', normalized_names=True)
    NapariColormaps = {}
    for cn in natsorted(cmap_names):
        if not Colormap(cn).interpolation == 'nearest':
            NapariColormaps[cn] = ensure_colormap(Colormap(cn).to_napari())