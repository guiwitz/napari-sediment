from dataclasses import dataclass, field
import dataclasses
from pathlib import Path
import yaml

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
        roi for measurement
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