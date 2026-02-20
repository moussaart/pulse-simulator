from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import numpy as np

@dataclass
class AlgorithmInput:
    """
    Data structure containing all necessary inputs for a localization algorithm.
    """
    measurements: List[float]
    anchors: List[Any]  # List of Anchor objects
    tag: Any            # Tag object
    dt: float           # Time step
    state: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    initialized: bool = False
    imu_data_on: bool = False
    control_input: Optional[np.ndarray] = None  # [ax, ay]
    is_los: Optional[List[bool]] = None         # LOS/NLOS status for each anchor
    
    # Additional optional parameters that might be passed
    params: Optional[dict] = None

@dataclass
class AlgorithmOutput:
    """
    Data structure for the output of a localization algorithm.
    """
    position: Tuple[float, float]
    state: np.ndarray
    covariance: np.ndarray
    initialized: bool
    extra_data: Optional[dict] = None  # For debug or visualization (e.g., innovation history)

class BaseLocalizationAlgorithm(ABC):
    """
    Abstract base class that all custom localization algorithms must inherit from.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the display name of the algorithm."""
        pass
        
    @abstractmethod
    def initialize(self) -> None:
        """
        Called when the algorithm is selected or reset.
        Should reset internal state.
        """
        pass
        
    @abstractmethod
    def update(self, input_data: AlgorithmInput) -> AlgorithmOutput:
        """
        Calculate the new position based on inputs.
        
        Args:
            input_data: AlgorithmInput object containing measurements, state, etc.
            
        Returns:
            AlgorithmOutput object containing new position, state, coverage, etc.
        """
        pass
