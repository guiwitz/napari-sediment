import numpy as np
from dataclasses import dataclass, field
from ._reader import read_spectral


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
    
    """
    imhdr_path: str = None
    channels: list[str] = None
    rois: list[list] = None
    channel_array: list[list] = None
    metadata: dict = field(default_factory=dict)
    nrows: int = None
    ncols: int = None

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
        print(f'datashape: {data.shape}')
        self.nrows = data.shape[0]
        self.ncols = data.shape[1]

    def read_channels(self, channels=None, roi=None):
        """
        Get channels from the image.
        
        Parameters
        ----------
        channels: list of int
            list of channels to get
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
                if self.rois[channel] is None:
                    if self.channel_array[channel] is None:
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
        Get channels from the image.
        
        Parameters
        ----------
        channels: list of int
            list of channels to get
        roi: array
            [row_start, row_end, col_start, col_end], None means full image
        
        """

        if channels is None:
            raise ValueError('channels must be provided')
        
        # make sure data is loaded
        self.read_channels(channels, roi)

        # get data
        if roi is None:
            data = np.stack([self.channel_array[c] for c in channels], axis=0)
        else:
            data = np.zeros(
                shape=(len(channels), roi[1]-roi[0], roi[3]-roi[2]),
                dtype=self.channel_array[channels[0]].dtype)
            for ind, c in enumerate(channels):
                if self.rois[c] is None:
                    data[ind,:,:] = self.channel_array[c][roi[0]:roi[1], roi[2]:roi[3]]
                else:
                    data[ind,:,:] = self.channel_array[c]

        return data
