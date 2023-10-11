from dataclasses import dataclass, field
import numpy as np

from .sediproc import find_index_of_band

@dataclass
class SpectralIndex:
    """
    Class for keeping track of processing parameters.
    
    Parameters
    ---------
    index_name: str
        name of index
    index_type: str
        one of 'Ratio', 'RABD', 'RABA'
    left_band: int
        left band to compute index
    right_band: int
        right band to compute index
    trough_band: int
        trough band to compute index
    numerator_band: int
        numerator band to compute ratio index
    denominator_band: int
        denominator band to compute ratio index
    left_band_default: int
        default left band to compute index
    right_band_default: int
        default right band to compute index
    trough_band_default: int
        default trough band to compute index
    numerator_band_default: int
        default numerator band to compute ratio index
    denominator_band_default: int
        default denominator band to compute ratio index
    
    """

    index_name: str = None
    index_type: str = None
    left_band: int = None
    right_band: int = None
    middle_band: int = None
    left_band_default: int = None
    right_band_default: int = None
    middle_band_default: int = None
    
    
    def __post_init__(self):
        """Use defaults for bands."""

        self.left_band = self.left_band_default
        self.right_band = self.right_band_default
        self.middle_band = self.middle_band_default


def compute_index_RABD(left, trough, right, row_bounds, col_bounds, imagechannels):
        """Compute the index RAB.
        
        Parameters
        ----------
        left: float
            left band
        trough: float
            trough band
        right: float

        Returns
        -------
        RABD: float
            RABD index
        """

        ltr = [left, trough, right]
        # find indices from the end-members plot (in case not all bands were used
        # This is not necessary as bands will not be skipped in the middle of the spectrum
        #ltr_endmember_indices = find_index_of_band(self.endmember_bands, ltr)
        # find band indices in the complete dataset
        ltr_stack_indices = find_index_of_band(imagechannels.centers,ltr)

        # number of bands between edges and trough
        #X_left = ltr_endmember_indices[1]-ltr_endmember_indices[0]
        #X_right = ltr_endmember_indices[2]-ltr_endmember_indices[1]
        X_left = ltr_stack_indices[1]-ltr_stack_indices[0]
        X_right = ltr_stack_indices[2]-ltr_stack_indices[1]

        # load the correct bands
        roi = np.concatenate([row_bounds, col_bounds])
        ltr_cube = imagechannels.get_image_cube(
            channels=ltr_stack_indices, roi=roi)
        ltr_cube = ltr_cube.astype(np.float32)

        # compute indices
        RABD = ((ltr_cube[0] * X_right + ltr_cube[2] * X_left) / (X_left + X_right)) / ltr_cube[1] 

        return RABD
        