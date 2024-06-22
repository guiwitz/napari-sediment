from dataclasses import dataclass, field
import dataclasses
from pathlib import Path
import yaml

@dataclass
class ParamEndMember:
    """
    Class for keeping track of processing parameters.
    
    Parameters
    ---------
    project_path: str
        path where the project is saved
    min_max_channel : list of int
        [min, max] channel to be used for processing
    eigen_threshold: float
        threshold for eigenvalue
    orrelation_threshold: float
        threshold for correlation
    ppi_threshold: float
        threshold for ppi
    ppi_iterations: int
        number of iterations for ppi
    corr_limit: int
        last index to take into account for correlation
        (removing large correlation in bad bands)

    
    """
    project_path: str = None
    min_max_channel: list = field(default_factory=list)
    eigen_threshold: float = None
    correlation_threshold: float = None
    ppi_threshold: float = None
    ppi_iterations: int = None
    corr_limit = int = None


    def save_parameters(self, alternate_path=None):
        """Save parameters as yml file.
        
        Parameters
        ----------
        alternate_path : str or Path, optional
            place where to save the parameters file.
        
        """

        if alternate_path is not None:
            save_path = Path(alternate_path).joinpath("Parameters_indices.yml")
        else:
            save_path = Path(self.project_path).joinpath("Parameters_indices.yml")
    
        with open(save_path, "w") as file:
            dict_to_save = dataclasses.asdict(self)
            for path_name in ['project_path']:
                if dict_to_save[path_name] is not None:
                    if not isinstance(dict_to_save[path_name], str):
                        dict_to_save[path_name] = dict_to_save[path_name].as_posix()
            
            yaml.dump(dict_to_save, file)