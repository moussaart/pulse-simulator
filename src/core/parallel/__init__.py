"""
Parallel Computing Module for UWB Localization Simulation

This module provides parallel execution utilities for computationally intensive
tasks in the localization simulation, including:
- Distance measurements across multiple anchors
- Multiple algorithm comparisons
- Sigma point transformations in UKF
- Anchor visualization updates
- GPU/CUDA acceleration when NVIDIA GPU is available
"""

from src.core.parallel.parallel_utils import (
    ParallelExecutor,
    parallel_distance_measurements,
    parallel_algorithm_execution,
    parallel_sigma_points,
    parallel_los_checks
)

from src.core.parallel.gpu_backend import (
    GPUManager,
    gpu_manager,
    is_gpu_available,
    get_array_module,
    to_gpu,
    to_cpu,
    to_gpu_batch,
    to_cpu_batch,
    gpu_accelerated,
)

from src.core.parallel.cuda_kernels import (
    batch_measure_distances,
    vectorized_cir_pulse_superposition,
    batch_path_loss,
    batch_toa_detection,
    batch_ranging_errors,
    batch_channel_statistics,
)

__all__ = [
    'ParallelExecutor',
    'parallel_distance_measurements',
    'parallel_algorithm_execution',
    'parallel_sigma_points',
    'parallel_los_checks',
    # GPU acceleration
    'GPUManager',
    'gpu_manager',
    'is_gpu_available',
    'get_array_module',
    'to_gpu',
    'to_cpu',
    'to_gpu_batch',
    'to_cpu_batch',
    'gpu_accelerated',
    # CUDA kernels
    'batch_measure_distances',
    'vectorized_cir_pulse_superposition',
    'batch_path_loss',
    'batch_toa_detection',
    'batch_ranging_errors',
    'batch_channel_statistics',
]
