from dataclasses import dataclass, field
import dataclasses
from pathlib import Path
import yaml

@dataclass
class Paramplot:
    """
    Class for keeping track of processing parameters.
    
    Paramters
    ----------
    title_font: float
        font size for the title
    label_font: float
        font size for the labels
    color_plotline: list
        triplet of rgb values for the plot line
    plot_thickness: float
        thickness of the plot line
    red_contrast_limits: list
        contrast limits for red channel
    green_contrast_limits: list
        contrast limits for green channel
    blue_contrast_limits: list
        contrast limits for blue channel
    rgb_bands: list
        list of rgb bands
    
    """

    title_font: float = 12
    label_font: float = 12
    color_plotline: list = field(default_factory=list)
    plot_thickness: float = 1
    red_contrast_limits: list = field(default_factory=list)
    green_contrast_limits: list = field(default_factory=list)
    blue_contrast_limits: list = field(default_factory=list)
    rgb_bands: list = field(default_factory=list)

    def __post_init__(self):
        self.color_plotline = [0, 0, 0]
    
    def save_parameters(self, save_path):
        """Save parameters as yml file.
        
        Parameters
        ----------
        file_path : str or Path, optional
            place where to save the parameters file.
        
        """

        save_path = Path(save_path)
        if save_path.suffix != '.yml':
            save_path = save_path.with_suffix('.yml')

    
        with open(save_path, "w") as file:
            dict_to_save = dataclasses.asdict(self)
            '''for path_name in ['project_path', 'file_path', 'white_path', 'dark_for_im_path', 'dark_for_white_path']:
                if dict_to_save[path_name] is not None:
                    if not isinstance(dict_to_save[path_name], str):
                        dict_to_save[path_name] = dict_to_save[path_name].as_posix()'''
            
            yaml.dump(dict_to_save, file)