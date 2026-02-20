"""
Parallel Computing Utilities for UWB Localization Simulation

Provides thread pool and process pool executors for parallel execution of:
- Distance measurements
- Algorithm computations
- Sigma point transformations
- LOS condition checks
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Tuple, Callable, Any, Dict, Optional
from dataclasses import dataclass
import multiprocessing
from functools import partial
import os
import logging

from src.core.parallel.gpu_backend import gpu_manager, get_array_module, to_gpu, to_cpu, to_gpu_batch

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel execution"""
    max_workers: int = None  # None means use cpu_count()
    use_threads: bool = True  # True for ThreadPool, False for ProcessPool
    chunk_size: int = 1  # For batching small tasks
    
    def __post_init__(self):
        if self.max_workers is None:
            # Use at most 4 workers for simulation to avoid overhead
            self.max_workers = min(4, multiprocessing.cpu_count())


class ParallelExecutor:
    """
    Manages parallel execution of tasks using thread or process pools.
    
    Uses ThreadPoolExecutor for I/O-bound tasks and tasks that need
    to share memory (like updating shared state).
    Uses ProcessPoolExecutor for CPU-bound tasks that don't share state.
    """
    
    def __init__(self, config: ParallelConfig = None):
        self.config = config or ParallelConfig()
        self._thread_pool = None
        self._process_pool = None
    
    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        """Lazy initialization of thread pool"""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        return self._thread_pool
    
    @property
    def process_pool(self) -> ProcessPoolExecutor:
        """Lazy initialization of process pool"""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)
        return self._process_pool
    
    def map_threads(self, func: Callable, items: List[Any], *args, **kwargs) -> List[Any]:
        """
        Execute function on items in parallel using threads.
        Good for I/O-bound tasks or tasks that share memory.
        
        Args:
            func: Function to execute
            items: List of items to process
            *args: Additional arguments passed to func
            **kwargs: Additional keyword arguments passed to func
            
        Returns:
            List of results in the same order as items
        """
        if len(items) <= 1:
            # Don't use parallelism for single items
            return [func(item, *args, **kwargs) for item in items]
        
        # Create partial function with fixed args/kwargs
        if args or kwargs:
            partial_func = partial(func, *args, **kwargs)
        else:
            partial_func = func
        
        # Submit all tasks
        futures = {self.thread_pool.submit(partial_func, item): i 
                   for i, item in enumerate(items)}
        
        # Collect results in order
        results = [None] * len(items)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error in parallel execution: {e}")
                results[idx] = None
        
        return results
    
    def map_processes(self, func: Callable, items: List[Any]) -> List[Any]:
        """
        Execute function on items in parallel using processes.
        Good for CPU-bound tasks that don't share state.
        
        Note: func and items must be picklable.
        
        Args:
            func: Function to execute (must be picklable)
            items: List of items to process (must be picklable)
            
        Returns:
            List of results in the same order as items
        """
        if len(items) <= 1:
            return [func(item) for item in items]
        
        try:
            results = list(self.process_pool.map(func, items, 
                                                  chunksize=self.config.chunk_size))
            return results
        except Exception as e:
            print(f"Process pool error, falling back to sequential: {e}")
            return [func(item) for item in items]
    
    def execute_concurrent(self, tasks: List[Tuple[Callable, tuple, dict]]) -> List[Any]:
        """
        Execute multiple different tasks concurrently.
        
        Args:
            tasks: List of (function, args, kwargs) tuples
            
        Returns:
            List of results in the same order as tasks
        """
        if len(tasks) <= 1:
            func, args, kwargs = tasks[0]
            return [func(*args, **kwargs)]
        
        futures = {}
        for i, (func, args, kwargs) in enumerate(tasks):
            future = self.thread_pool.submit(func, *args, **kwargs)
            futures[future] = i
        
        results = [None] * len(tasks)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error in concurrent execution: {e}")
                results[idx] = None
        
        return results
    
    def shutdown(self):
        """Shutdown all pools"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None
        if self._process_pool:
            self._process_pool.shutdown(wait=False)
            self._process_pool = None


# Global executor instance
_global_executor = None

def get_executor() -> ParallelExecutor:
    """Get or create the global parallel executor"""
    global _global_executor
    if _global_executor is None:
        _global_executor = ParallelExecutor()
    return _global_executor


def parallel_distance_measurements(
    tag,
    anchors: List,
    channel_conditions,
    simulation_time: float,
    mode: str = "SS-TWR"
) -> List[Tuple[float, List, bool]]:
    """
    Measure distances to multiple anchors in parallel.
    
    Args:
        tag: Tag object
        anchors: List of Anchor objects
        channel_conditions: ChannelConditions object
        simulation_time: Current simulation time
        mode: TWR mode ("SS-TWR" or "DS-TWR")
        
    Returns:
        List of (distance, messages, is_los) tuples for each anchor
    """
    def measure_single_anchor(anchor):
        """Measure distance to a single anchor"""
        try:
            # Update LOS condition for this anchor
            channel_conditions.update_los_condition(anchor.position, tag.position)
            
            # Get measurement
            distance, messages = tag.measure_distance_with_logs(
                anchor, channel_conditions, simulation_time, mode)
            
            # Check LOS condition
            is_los = channel_conditions.check_los_to_anchor(anchor.position, tag.position)
            
            return (distance, messages, is_los)
        except Exception as e:
            print(f"Error measuring distance to {anchor.id}: {e}")
            return (float('inf'), [], True)
    
    if len(anchors) <= 2:
        # Sequential for small numbers
        return [measure_single_anchor(anchor) for anchor in anchors]
    
    executor = get_executor()
    return executor.map_threads(measure_single_anchor, anchors)


def parallel_los_checks(
    anchors: List,
    tag_position,
    channel_conditions
) -> List[bool]:
    """
    Check LOS conditions for multiple anchors in parallel.
    
    Args:
        anchors: List of Anchor objects
        tag_position: Tag position
        channel_conditions: ChannelConditions object
        
    Returns:
        List of boolean LOS conditions (True=LOS, False=NLOS)
    """
    def check_single_los(anchor):
        """Check LOS for a single anchor"""
        return channel_conditions.check_los_to_anchor(anchor.position, tag_position)
    
    if len(anchors) <= 2:
        return [check_single_los(anchor) for anchor in anchors]
    
    executor = get_executor()
    return executor.map_threads(check_single_los, anchors)


def parallel_sigma_points(
    sigmas: np.ndarray,
    transform_func: Callable,
    *func_args
) -> np.ndarray:
    """
    Transform sigma points in parallel for UKF.
    
    Args:
        sigmas: Array of sigma points (2n+1 x n)
        transform_func: Function to apply to each sigma point
        *func_args: Additional arguments for transform_func
        
    Returns:
        Transformed sigma points array
    """
    n_sigmas = sigmas.shape[0]
    
    if n_sigmas <= 3:
        # Sequential for small number of sigma points
        return np.array([transform_func(s, *func_args) for s in sigmas])
    
    def transform_single(sigma):
        return transform_func(sigma, *func_args)
    
    executor = get_executor()
    results = executor.map_threads(transform_single, list(sigmas))
    return np.array(results)


def parallel_algorithm_execution(
    algorithm_configs: List[Dict],
    measurements: List[float],
    tag,
    anchors: List,
    dt: float,
    is_los: List[int] = None,
    imu_data_on: bool = False,
    u: np.ndarray = None
) -> List[Dict]:
    """
    Execute multiple localization algorithms in parallel.
    
    Args:
        algorithm_configs: List of algorithm configuration dicts
        measurements: Distance measurements
        tag: Tag object
        anchors: List of anchors
        dt: Time step
        is_los: LOS conditions
        imu_data_on: Whether IMU data is enabled
        u: IMU control input
        
    Returns:
        List of updated algorithm configs with results
    """
    from src.core.localization import LocalizationAlgorthimes
    
    def execute_single_algorithm(config):
        """Execute a single algorithm"""
        try:
            config = config.copy()  # Don't modify original
            name = config['name']
            
            if name == 'EKF':
                result = LocalizationAlgorthimes.extended_kalman_filter(
                    measurements=measurements,
                    tag=tag,
                    anchors=anchors,
                    ekf_state=config['state'],
                    ekf_P=config['P'],
                    ekf_initialized=config['initialized'],
                    dt=dt,
                    Q=config.get('Q'),
                    R=config.get('R'),
                    imu_data_on=imu_data_on,
                    u=u
                )
                position, config['state'], config['P'], config['initialized'] = result
                
            elif name == 'UKF':
                result = LocalizationAlgorthimes.unscented_kalman_filter(
                    measurements=measurements,
                    tag=tag,
                    anchors=anchors,
                    ukf_state=config['state'],
                    ukf_P=config['P'],
                    ukf_initialized=config['initialized'],
                    dt=dt,
                    Q=config.get('Q'),
                    R=config.get('R'),
                    imu_data_on=imu_data_on,
                    u=u
                )
                position, config['state'], config['P'], config['initialized'] = result
                
            elif name == 'AEKF':
                result = LocalizationAlgorthimes.adaptive_extended_kalman_filter(
                    measurements=measurements,
                    tag=tag,
                    anchors=anchors,
                    aekf_state=config['state'],
                    aekf_P=config['P'],
                    aekf_initialized=config['initialized'],
                    dt=dt,
                    Q=config.get('Q'),
                    R=config.get('R'),
                    imu_data_on=imu_data_on,
                    u=u
                )
                position, config['state'], config['P'], config['initialized'], config['Q'], config['R'] = result
                
            elif name == 'IMPROVED_AEKF':
                result = LocalizationAlgorthimes.improved_adaptive_ekf(
                    measurements=measurements,
                    tag=tag,
                    anchors=anchors,
                    aekf_state=config['state'],
                    aekf_P=config['P'],
                    aekf_initialized=config['initialized'],
                    dt=dt,
                    Q=config.get('Q'),
                    R=config.get('R'),
                    prev_R=config.get('prev_R'),
                    innovation_history=config.get('innovation_history')
                )
                (position, config['innovation_history'], config['state'], config['P'], 
                 config['initialized'], config['Q'], config['R']) = result
                config['prev_R'] = config['R']
                
            elif name in ('LOS_AEKF', 'LOS_AEKF_EC'):
                # Determine is_los to use
                is_los_used = is_los if is_los is not None else [0] * len(measurements)
                
                result = LocalizationAlgorthimes.Nlos_aware_aekf(
                    measurements=measurements,
                    tag=tag,
                    anchors=anchors,
                    aekf_state=config['state'],
                    aekf_P=config['P'],
                    aekf_initialized=config['initialized'],
                    is_los=is_los_used,
                    alpha=config.get('alpha', 0.3),
                    beta=config.get('beta', 2.0),
                    nlos_factor=config.get('nlos_factor', 10.0),
                    dt=dt,
                    Q=config.get('Q'),
                    R=config.get('R'),
                    imu_data_on=imu_data_on,
                    u=u
                )
                position, config['state'], config['P'], config['initialized'], config['Q'], config['R'] = result
            
            elif name == 'IMU_ONLY':
                if hasattr(tag, 'imu_data') and len(tag.imu_data.acc_x) > 0:
                    imu_measurements = [float(tag.imu_data.acc_x[-1]), float(tag.imu_data.acc_y[-1])]
                else:
                    imu_measurements = [0.0, 0.0]
                    
                result = LocalizationAlgorthimes.imu_only_filter(
                    tag=tag,
                    measurements=imu_measurements,
                    state=config['state'],
                    P=config['P'],
                    initialized=config['initialized'],
                    dt=dt
                )
                position, config['state'], config['P'], config['initialized'] = result
                
            elif name == 'IMU_NLOS_AEKF':
                is_los_used = is_los if is_los is not None else [0] * len(measurements)
                
                result = LocalizationAlgorthimes.IMU_assisted_Nlos_aware_aekf(
                    measurements=measurements,
                    tag=tag,
                    anchors=anchors,
                    state=config['state'],
                    P=config['P'],
                    initialized=config['initialized'],
                    is_los=is_los_used,
                    alpha=config.get('alpha', 0.3),
                    beta=config.get('beta', 2.0),
                    nlos_factor=config.get('nlos_factor', 10.0),
                    dt=dt,
                    zupt_threshold=0.05,
                    R=config.get('R')
                )
                position, config['state'], config['P'], config['initialized'], config['R'] = result
            
            else:
                # Fallback to trilateration
                position = LocalizationAlgorthimes.trilateration(measurements, anchors)
            
            config['position'] = position
            config['error'] = np.sqrt((position[0] - tag.position.x)**2 + 
                                      (position[1] - tag.position.y)**2)
            return config
            
        except Exception as e:
            print(f"Error executing algorithm {config.get('name', 'unknown')}: {e}")
            config['position'] = (0, 0)
            config['error'] = float('inf')
            return config
    
    if len(algorithm_configs) <= 2:
        return [execute_single_algorithm(config) for config in algorithm_configs]
    
    executor = get_executor()
    return executor.map_threads(execute_single_algorithm, algorithm_configs)


class ParallelUKF:
    """
    Parallelized Unscented Kalman Filter implementation.
    
    Parallelizes the sigma point transformations which are the most
    computationally expensive part of the UKF.
    """
    
    def __init__(self, n_states: int = 4, alpha: float = 0.3, beta: float = 2.0, kappa: float = -1):
        self.n = n_states
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._executor = get_executor()
    
    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points from state and covariance"""
        n = len(x)
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        # Ensure positive definiteness
        try:
            U = np.linalg.cholesky((n + lambda_) * P)
        except np.linalg.LinAlgError:
            U = np.linalg.cholesky((n + lambda_) * (P + np.eye(n) * 1e-6))
        
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for k in range(n):
            sigmas[k + 1] = x + U[k]
            sigmas[n + k + 1] = x - U[k]
        
        return sigmas
    
    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance weights"""
        n = self.n
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        Wm = np.full(2 * n + 1, 1. / (2 * (n + lambda_)))
        Wc = np.full(2 * n + 1, 1. / (2 * (n + lambda_)))
        Wm[0] = lambda_ / (n + lambda_)
        Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        
        return Wm, Wc
    
    def transform_sigma_points_parallel(
        self,
        sigmas: np.ndarray,
        transform_func: Callable,
        *args
    ) -> np.ndarray:
        """Transform sigma points in parallel"""
        return parallel_sigma_points(sigmas, transform_func, *args)
    
    def unscented_transform(
        self,
        sigmas: np.ndarray,
        Wm: np.ndarray,
        Wc: np.ndarray,
        noise_cov: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform unscented transform to get mean and covariance.
        
        Args:
            sigmas: Transformed sigma points
            Wm: Mean weights
            Wc: Covariance weights
            noise_cov: Additional noise covariance to add
            
        Returns:
            (mean, covariance) tuple
        """
        # Weighted mean
        x = np.dot(Wm, sigmas)
        
        # Weighted covariance — VECTORIZED via einsum (no Python loop)
        # Y = sigmas - x has shape (2n+1, n_out)
        Y = sigmas - x  # (2n+1, n_out)
        # Weighted outer products: P = Σ Wc[k] * y_k @ y_k^T
        # Using einsum: 'ki,kj->ij' with weights broadcast
        P = np.einsum('k,ki,kj->ij', Wc, Y, Y)
        
        if noise_cov is not None:
            P += noise_cov
        
        return x, P


# Vectorized operations for numpy arrays
def vectorized_distance_calc(
    tag_pos: Tuple[float, float],
    anchor_positions: np.ndarray
) -> np.ndarray:
    """
    Calculate distances from tag to all anchors using vectorized operations.
    Uses GPU when available and array size warrants it.
    
    Args:
        tag_pos: (x, y) tuple of tag position
        anchor_positions: Nx2 array of anchor positions
        
    Returns:
        Array of distances (NumPy)
    """
    xp = get_array_module()
    use_gpu = gpu_manager.should_use_gpu(anchor_positions.size)
    
    if use_gpu:
        pos = to_gpu(np.array(tag_pos))
        anchors = to_gpu(anchor_positions)
        diff = anchors - pos
        result = xp.sqrt(xp.sum(diff**2, axis=1))
        return to_cpu(result)
    
    diff = anchor_positions - np.array(tag_pos)
    return np.sqrt(np.sum(diff**2, axis=1))


def vectorized_jacobian(
    state: np.ndarray,
    anchor_positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate measurement Jacobian and predicted measurements vectorized.
    Uses GPU when available.
    
    Args:
        state: State vector [x, y, vx, vy]
        anchor_positions: Nx2 array of anchor positions
        
    Returns:
        (H, h) tuple - Jacobian matrix and predicted measurements
    """
    xp = get_array_module()
    use_gpu = gpu_manager.should_use_gpu(anchor_positions.size)
    
    if use_gpu:
        s = to_gpu(state)
        ap = to_gpu(anchor_positions)
        
        dx = s[0] - ap[:, 0]
        dy = s[1] - ap[:, 1]
        d = xp.sqrt(dx**2 + dy**2)
        d = xp.maximum(d, 1e-6)
        
        n_anchors = anchor_positions.shape[0]
        H = xp.zeros((n_anchors, 4))
        H[:, 0] = dx / d
        H[:, 1] = dy / d
        
        return to_cpu(H), to_cpu(d)
    
    n_anchors = anchor_positions.shape[0]
    dx = state[0] - anchor_positions[:, 0]
    dy = state[1] - anchor_positions[:, 1]
    d = np.sqrt(dx**2 + dy**2)
    d = np.maximum(d, 1e-6)
    
    H = np.zeros((n_anchors, 4))
    H[:, 0] = dx / d
    H[:, 1] = dy / d
    
    return H, d

