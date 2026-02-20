"""
Filter Data Adapter
Extracts filter state and covariance data from localization algorithms.
"""
import numpy as np
from typing import Dict, Optional, Callable, Any, Tuple

from src.api.collectors.data_collector import FilterOutput


class FilterDataAdapter:
    """
    Adapter for extracting filter state data from localization algorithms.
    Supports extensibility through a filter registry for custom filters.
    """
    
    def __init__(self):
        # Registry for custom filter capture functions
        self._filter_registry: Dict[str, Callable] = {}
        
        # Register default filters
        self._register_default_filters()
        
    def _register_default_filters(self):
        """Register capture functions for built-in filters"""
        self.register_filter("Extended Kalman Filter", self._capture_ekf_state)
        self.register_filter("Unscented Kalman Filter", self._capture_ukf_state)
        self.register_filter("Adaptive Extended Kalman Filter", self._capture_aekf_state)
        self.register_filter("NLOS-Aware AEKF", self._capture_nlos_aware_state)
        self.register_filter("Improved Adaptive EKF", self._capture_iaekf_state)
        self.register_filter("IMU assisted NLOS-Aware AEKF", self._capture_imu_aekf_state)
        self.register_filter("Trilateration", self._capture_trilateration_state)
        
    def register_filter(self, name: str, capture_func: Callable) -> None:
        """
        Register a custom filter capture function.
        
        Args:
            name: Name of the filter (must match algorithm name in simulator)
            capture_func: Function that takes (estimated_pos, error, state, P, R, Q, **kwargs)
                         and returns a FilterOutput
        """
        self._filter_registry[name] = capture_func
        
    def capture_state(self,
                      algorithm_name: str,
                      estimated_pos: Tuple[float, float],
                      error: float,
                      state: np.ndarray = None,
                      P: np.ndarray = None,
                      R: np.ndarray = None,
                      Q: np.ndarray = None,
                      **kwargs) -> FilterOutput:
        """
        Capture filter state for any registered filter.
        
        Args:
            algorithm_name: Name of the algorithm
            estimated_pos: Estimated position (x, y)
            error: Estimation error
            state: State vector
            P: State covariance matrix
            R: Measurement noise covariance
            Q: Process noise covariance
            **kwargs: Additional filter-specific parameters
            
        Returns:
            FilterOutput with captured state
        """
        # Find matching capture function
        for filter_name, capture_func in self._filter_registry.items():
            if filter_name in algorithm_name:
                return capture_func(
                    algorithm_name, estimated_pos, error, 
                    state, P, R, Q, **kwargs
                )
        
        # Default capture for unknown filters
        return self._capture_generic_state(
            algorithm_name, estimated_pos, error, 
            state, P, R, Q, **kwargs
        )
    
    def _capture_generic_state(self,
                               algorithm_name: str,
                               estimated_pos: Tuple[float, float],
                               error: float,
                               state: np.ndarray = None,
                               P: np.ndarray = None,
                               R: np.ndarray = None,
                               Q: np.ndarray = None,
                               **kwargs) -> FilterOutput:
        """Generic state capture for any filter"""
        return FilterOutput(
            filter_name=algorithm_name,
            estimated_position=estimated_pos,
            estimation_error=error,
            state_vector=self._safe_copy(state),
            state_covariance=self._safe_copy(P),
            measurement_covariance=self._safe_copy(R),
            process_noise_covariance=self._safe_copy(Q),
            filter_params=kwargs
        )
    
    def _capture_ekf_state(self,
                           algorithm_name: str,
                           estimated_pos: Tuple[float, float],
                           error: float,
                           state: np.ndarray = None,
                           P: np.ndarray = None,
                           R: np.ndarray = None,
                           Q: np.ndarray = None,
                           **kwargs) -> FilterOutput:
        """Capture EKF state"""
        return FilterOutput(
            filter_name=algorithm_name,
            estimated_position=estimated_pos,
            estimation_error=error,
            state_vector=self._safe_copy(state),
            state_covariance=self._safe_copy(P),
            measurement_covariance=self._safe_copy(R),
            process_noise_covariance=self._safe_copy(Q),
            filter_params={'filter_type': 'EKF'}
        )
    
    def _capture_ukf_state(self,
                           algorithm_name: str,
                           estimated_pos: Tuple[float, float],
                           error: float,
                           state: np.ndarray = None,
                           P: np.ndarray = None,
                           R: np.ndarray = None,
                           Q: np.ndarray = None,
                           **kwargs) -> FilterOutput:
        """Capture UKF state"""
        params = {
            'filter_type': 'UKF',
            'alpha': kwargs.get('alpha', 0.3),
            'beta': kwargs.get('beta', 2.0),
            'kappa': kwargs.get('kappa', -1)
        }
        return FilterOutput(
            filter_name=algorithm_name,
            estimated_position=estimated_pos,
            estimation_error=error,
            state_vector=self._safe_copy(state),
            state_covariance=self._safe_copy(P),
            measurement_covariance=self._safe_copy(R),
            process_noise_covariance=self._safe_copy(Q),
            filter_params=params
        )
    
    def _capture_aekf_state(self,
                            algorithm_name: str,
                            estimated_pos: Tuple[float, float],
                            error: float,
                            state: np.ndarray = None,
                            P: np.ndarray = None,
                            R: np.ndarray = None,
                            Q: np.ndarray = None,
                            **kwargs) -> FilterOutput:
        """Capture Adaptive EKF state"""
        return FilterOutput(
            filter_name=algorithm_name,
            estimated_position=estimated_pos,
            estimation_error=error,
            state_vector=self._safe_copy(state),
            state_covariance=self._safe_copy(P),
            measurement_covariance=self._safe_copy(R),
            process_noise_covariance=self._safe_copy(Q),
            innovation=self._safe_copy(kwargs.get('innovation')),
            filter_params={'filter_type': 'AEKF', 'is_adaptive': True}
        )
    
    def _capture_nlos_aware_state(self,
                                   algorithm_name: str,
                                   estimated_pos: Tuple[float, float],
                                   error: float,
                                   state: np.ndarray = None,
                                   P: np.ndarray = None,
                                   R: np.ndarray = None,
                                   Q: np.ndarray = None,
                                   **kwargs) -> FilterOutput:
        """Capture NLOS-Aware AEKF state"""
        params = {
            'filter_type': 'NLOS_AWARE_AEKF',
            'is_nlos_aware': True,
            'alpha': kwargs.get('alpha', 0.3),
            'beta': kwargs.get('beta', 2.0),
            'nlos_factor': kwargs.get('nlos_factor', 10.0)
        }
        return FilterOutput(
            filter_name=algorithm_name,
            estimated_position=estimated_pos,
            estimation_error=error,
            state_vector=self._safe_copy(state),
            state_covariance=self._safe_copy(P),
            measurement_covariance=self._safe_copy(R),
            process_noise_covariance=self._safe_copy(Q),
            innovation=self._safe_copy(kwargs.get('innovation')),
            filter_params=params
        )
    
    def _capture_iaekf_state(self,
                             algorithm_name: str,
                             estimated_pos: Tuple[float, float],
                             error: float,
                             state: np.ndarray = None,
                             P: np.ndarray = None,
                             R: np.ndarray = None,
                             Q: np.ndarray = None,
                             **kwargs) -> FilterOutput:
        """Capture Improved Adaptive EKF state"""
        params = {
            'filter_type': 'IAEKF',
            'mu': kwargs.get('mu', 0.95),
            'alpha': kwargs.get('alpha', 0.3),
            'xi': kwargs.get('xi', 20),
            'lambda_min': kwargs.get('lambda_min', 0.1),
            'lambda_max': kwargs.get('lambda_max', 3.0),
            'tau': kwargs.get('tau', 0.95)
        }
        return FilterOutput(
            filter_name=algorithm_name,
            estimated_position=estimated_pos,
            estimation_error=error,
            state_vector=self._safe_copy(state),
            state_covariance=self._safe_copy(P),
            measurement_covariance=self._safe_copy(R),
            process_noise_covariance=self._safe_copy(Q),
            innovation=self._safe_copy(kwargs.get('innovation_history')),
            filter_params=params
        )
    
    def _capture_imu_aekf_state(self,
                                algorithm_name: str,
                                estimated_pos: Tuple[float, float],
                                error: float,
                                state: np.ndarray = None,
                                P: np.ndarray = None,
                                R: np.ndarray = None,
                                Q: np.ndarray = None,
                                **kwargs) -> FilterOutput:
        """Capture IMU-assisted NLOS-Aware AEKF state"""
        params = {
            'filter_type': 'IMU_NLOS_AWARE_AEKF',
            'is_imu_assisted': True,
            'is_nlos_aware': True,
            'zupt_threshold': kwargs.get('zupt_threshold', 0.05)
        }
        return FilterOutput(
            filter_name=algorithm_name,
            estimated_position=estimated_pos,
            estimation_error=error,
            state_vector=self._safe_copy(state),
            state_covariance=self._safe_copy(P),
            measurement_covariance=self._safe_copy(R),
            process_noise_covariance=self._safe_copy(Q),
            filter_params=params
        )
    
    def _capture_trilateration_state(self,
                                     algorithm_name: str,
                                     estimated_pos: Tuple[float, float],
                                     error: float,
                                     state: np.ndarray = None,
                                     P: np.ndarray = None,
                                     R: np.ndarray = None,
                                     Q: np.ndarray = None,
                                     **kwargs) -> FilterOutput:
        """Capture Trilateration state (no covariance)"""
        return FilterOutput(
            filter_name=algorithm_name,
            estimated_position=estimated_pos,
            estimation_error=error,
            filter_params={'filter_type': 'Trilateration', 'is_geometric': True}
        )
    
    @staticmethod
    def _safe_copy(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Safely copy a numpy array"""
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            return arr.copy()
        return np.array(arr)
    
    def get_registered_filters(self) -> list:
        """Get list of registered filter names"""
        return list(self._filter_registry.keys())
