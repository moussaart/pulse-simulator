import os
import sys
import importlib.util
import inspect
from typing import Dict, Type, List
from .base_algorithm import BaseLocalizationAlgorithm

class AlgorithmLoader:
    """
    Responsible for discovery and loading of custom localization algorithms
    from the user_algorithms directory.
    """
    
    def __init__(self, algorithms_dir: str):
        self.algorithms_dir = algorithms_dir
        self.loaded_algorithms: Dict[str, Type[BaseLocalizationAlgorithm]] = {}

    def discover_algorithms(self) -> Dict[str, Type[BaseLocalizationAlgorithm]]:
        """
        Scans the algorithms directory and returns a dictionary of valid algorithms.
        Returns:
            Dict mapping algorithm names to algorithm classes
        """
        self.loaded_algorithms = {}
        
        if not os.path.exists(self.algorithms_dir):
            os.makedirs(self.algorithms_dir, exist_ok=True)
            return {}

        # Add the parent directory of algorithms_dir to sys.path to allow imports
        # This might be needed if user algorithms import from each other or relative imports
        # For now, we'll just load them by path
        
        for filename in os.listdir(self.algorithms_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                file_path = os.path.join(self.algorithms_dir, filename)
                self._load_algorithm_from_file(file_path)
                
        return self.loaded_algorithms

    def _load_algorithm_from_file(self, file_path: str):
        """
        Loads a single python file and checks for BaseLocalizationAlgorithm subclasses.
        """
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # Scan module for subclasses of BaseLocalizationAlgorithm
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseLocalizationAlgorithm) and 
                        obj is not BaseLocalizationAlgorithm):
                        
                        try:
                            # Instantiate to get the name property, or checking if it's an abstract class?
                            # For now just check if it's not abstract
                            if not inspect.isabstract(obj):
                                algorithm_instance = obj() # Instantiate to get name
                                algo_name = algorithm_instance.name
                                self.loaded_algorithms[algo_name] = obj
                                print(f"Loaded custom algorithm: {algo_name} from {module_name}")
                        except Exception as e:
                            print(f"Error inspecting class {name} in {file_path}: {e}")

        except Exception as e:
            print(f"Failed to load algorithm from {file_path}: {e}")
