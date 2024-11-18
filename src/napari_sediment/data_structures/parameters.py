from dataclasses import dataclass, field
import dataclasses
from pathlib import Path
import yaml
import numpy as np

@dataclass
class Param:
    """
    Class for keeping track of processing parameters.
    
    Paramters
    ---------
    project_path: str
        path where the project is saved
    file_paths: list[str]
        list of paths of files belonging to the project
    dark_for_im_path: str
        path of dark image for the files
    dark_for_white_path: str
        path of dark image for the white image
    main_roi: array
        main roi 
    rois: dict of arrays
        flat list of rois
    measurement_roi: array
        roi for measurement. Note that these coordinates are relative
        to the main roi and not the image.
    scale: float
        scale of the image in scale_units/px
    scale_units: str
        units of the scale
    location: str
        location of the sample
    rgb: list
        list of rgb bands
    
    """
    project_path: str = None
    file_path: str = None
    white_path: str = None
    dark_for_im_path: str = None
    dark_for_white_path: str = None
    main_roi: list = field(default_factory=list)
    rois: list = field(default_factory=list)
    measurement_roi: list = field(default_factory=list)
    scale: float = 1
    scale_units: str = 'mm'
    location: str = ''
    rgb: list = field(default_factory=list)

    def __post_init__(self):
        self.rgb = [640, 545, 460]
    
    def save_parameters(self, alternate_path=None):
        """Save parameters as yml file.
        
        Parameters
        ----------
        alternate_path : str or Path, optional
            place where to save the parameters file.
        
        """

        if alternate_path is not None:
            save_path = Path(alternate_path).joinpath("Parameters.yml")
        else:
            save_path = Path(self.project_path).joinpath("Parameters.yml")
    
        with open(save_path, "w") as file:
            dict_to_save = dataclasses.asdict(self)
            for path_name in ['project_path', 'file_path', 'white_path', 'dark_for_im_path', 'dark_for_white_path']:
                if dict_to_save[path_name] is not None:
                    if not isinstance(dict_to_save[path_name], str):
                        dict_to_save[path_name] = dict_to_save[path_name].as_posix()
            
            yaml.dump(dict_to_save, file)

    def format_measurement_roi(self):
        """Make sure that the measurement roi is formatted correctly
        and that empy rois are replaced with defaults, i.e. original sub-rois."""

        mainroi = self.get_formatted_main_roi()
        subrois = self.get_formatted_subrois()
           
        if self.measurement_roi == []:
            self.measurement_roi = [[] for x in range(len(mainroi))]
        
        for i in range(len(mainroi)):
            if self.measurement_roi[i] == []:
                row_bounds = [
                        subrois[i][0][:,0].min() - mainroi[i][:,0].min(),
                        subrois[i][0][:,0].max() - mainroi[i][:,0].min()]
                col_bounds = [
                        subrois[i][0][:,1].min() - mainroi[i][:,1].min(),
                        subrois[i][0][:,1].max() - mainroi[i][:,1].min()]
                
                roi_square = np.array([
                    row_bounds[0], col_bounds[0],
                    row_bounds[1], col_bounds[0],
                    row_bounds[1], col_bounds[1],
                    row_bounds[0], col_bounds[1]
                    ])
                self.measurement_roi[i] = list(roi_square)
                self.measurement_roi[i] = [x.item() for x in self.measurement_roi[i]] 

    def get_formatted_main_roi(self):
        """Get mainroi as formatted array."""
        
        return np.array([np.array(x).reshape(4,2) for x in self.main_roi]).astype(int)
    
    def get_formatted_measurement_roi(self):
        """Get mainroi as formatted array."""
        
        self.format_measurement_roi()
        return np.array([np.array(x).reshape(4,2) for x in self.measurement_roi]).astype(int)
    
    def get_formatted_subrois(self):
        """Get rois as formatted arrays."""
        
        return [[np.array(x).reshape(4,2) for x in y] for y in self.rois]
    
    def get_formatted_col_row_bounds(self, mainroi_index):
        """Get column and row bounds of the mainroi.
        
        Parameters
        ----------
        mainroi_index : int
            index of the mainroi to get bounds for.
        
        Returns
        -------
        row_bounds : list
            row bounds [rmin, rmax]
        col_bounds : list
            column bounds [cmin, cmax]
        
        """

        mainroi = self.get_formatted_main_roi()
        row_bounds = [
                        mainroi[mainroi_index][:,0].min(),
                        mainroi[mainroi_index][:,0].max()]
        col_bounds = [
                    mainroi[mainroi_index][:,1].min(),
                    mainroi[mainroi_index][:,1].max()]
        
        return row_bounds, col_bounds


    def get_formatted_rois(self):
        """Get rois as formatted arrays."""
        
        mainroi = self.get_formatted_main_roi()
        subrois = self.get_formatted_subrois()
        measurement_roi = self.get_formatted_measurement_roi()

        

        return mainroi, subrois, measurement_roi