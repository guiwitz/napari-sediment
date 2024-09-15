import numpy as np
from dataclasses import dataclass, field
from ..utilities._reader import read_spectral
from ..utilities.sediproc import find_index_of_band


@dataclass
class ImChannels:
    """
    Class for handling partial import of HDR images.
    
    Paramters
    ---------
    imhdr_path: str
        path where the project is saved
    channels: list of str
        list of available channels
    rois: list of arrays
        current roi of each image, None means full image
    channel_array: list of arrays
        current array of each channel. Can contain different rois
    metadata: dict
        metadata of the image
    nrows: int
        number of rows in the image
    ncols: int
        number of columns in the image
    centers: array of float
        band centers of the channels
    
    """
    imhdr_path: str = None
    channels: list[str] = None
    rois: list[list] = None
    channel_array: list[list] = None
    metadata: dict = field(default_factory=dict)
    nrows: int = None
    ncols: int = None
    centers: np.ndarray = None

    def __post_init__(self):
    
        data, metadata = read_spectral(
                path=self.imhdr_path,
                bands=[0],
                row_bounds=None,
                col_bounds=None,
            )
        self.channel_names = metadata['wavelength']
        self.rois = [None] * len(self.channel_names)
        self.channel_array = [None] * len(self.channel_names)
        self.channel_array[0] = data[:,:,0]
        self.metadata = metadata
        self.nrows = data.shape[0]
        self.ncols = data.shape[1]
        self.centers = np.array(metadata['centers'])

    def read_channels(self, channels=None, roi=None):
        """
        Get channels from the image.
        
        Parameters
        ----------
        channels: list of int
            indices of channel to get
        roi: array
            [row_start, row_end, col_start, col_end], None means full image
        
        """

        if channels is None:
            raise ValueError('channels must be provided')
        
        channels_full_image = []
        channels_partial_image = []
        for channel in channels:
            if roi is None:
                if self.rois[channel] is None:
                    if self.channel_array[channel] is None:
                        channels_full_image.append(channel)
                else:
                    channels_full_image.append(channel)
            else:
                # if a new roi is provided, reload the channel even if full frame is already loaded
                if self.rois[channel] is None:
                    #if self.channel_array[channel] is None:
                    channels_partial_image.append(channel)
                else:
                    if not np.array_equal(roi, self.rois[channel]):
                        channels_partial_image.append(channel)
                
        if len(channels_full_image) > 0:
            data, _ = read_spectral(
                path=self.imhdr_path,
                bands=channels_full_image,
                row_bounds=None,
                col_bounds=None,
            )
            for ind, c in enumerate(channels_full_image):
                self.channel_array[c] = data[:,:,ind]
                self.rois[c] = None
        
        if len(channels_partial_image) > 0:
            data, _ = read_spectral(
                path=self.imhdr_path,
                bands=channels_partial_image,
                row_bounds=[roi[0], roi[1]],
                col_bounds=[roi[2], roi[3]],
            )
            for ind, c in enumerate(channels_partial_image):
                self.channel_array[c] = data[:,:,ind]
                self.rois[c] = roi

    def get_image_cube(self, channels=None, roi=None):
        """
        Get image stack containing the selected channels indices.
        
        Parameters
        ----------
        channels: list of int
            indices of channel to get
        roi: array
            [row_start, row_end, col_start, col_end], None means full image

        Returns
        -------
        data: array
            array of shape (n_channels, n_rows, n_cols)
        
        """

        if channels is None:
            raise ValueError('channels must be provided')
        
        # make sure data is loaded
        self.read_channels(channels, roi)

        # get data
        if roi is None:
            data = np.stack([self.channel_array[c] for c in channels], axis=0)
        else:
            full = [0, self.nrows, 0, self.ncols]
            data = []
            for ind, r in enumerate(roi):
                if r is None:
                    roi[ind] = full[ind]
            '''data = np.zeros(
                shape=(len(channels), roi[1]-roi[0], roi[3]-roi[2]),
                dtype=self.channel_array[channels[0]].dtype)
            for ind, c in enumerate(channels):
                if self.rois[c] is None:
                    data[ind,:,:] = self.channel_array[c][roi[0]:roi[1], roi[2]:roi[3]]
                else:
                    data[ind,:,:] = self.channel_array[c]'''
            for ind, c in enumerate(channels):
                if self.rois[c] is None:
                    data.append(self.channel_array[c][roi[0]:roi[1], roi[2]:roi[3]])
                else:
                    data.append(self.channel_array[c])
            data = np.stack(data, axis=0)

        return data
    
    def get_indices_of_bands(self, bands):
        """
        Given the bands centers of the dataset and a set of band values to recover
        find the indices of the closest bands in the dataset. E.g if the dataset
        has bands [450, 500, 550, 600] and bands = [460, 550], the function will
        return [0, 2]. Those bands indices can then be used e.g by get_image_cube

        Parameters
        ----------
        bands: list of float
            list of band values for which to find the index of the closest
            bands in the dataset

        Returns
        -------
        bands_indices: list of int
            list of indices of the bands in the dataset closest to the desired bands
        bands_names: list of str
            list of band names corresponding to bands_indices
        """

        bands_indices = find_index_of_band(self.centers, bands)
        #bands_indices = [np.argmin(np.abs(np.array(self.channel_names).astype(float) - x)) for x in bands]
        bands_names = [self.channel_names[x] for x in bands_indices]

        return bands_indices, bands_names
