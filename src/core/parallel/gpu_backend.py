"""
GPU Backend Module for UWB Localization Simulation

Provides automatic NVIDIA GPU detection via CuPy and transparent fallback
to NumPy when no CUDA-capable GPU is available.

Features:
- CUDA stream pool for concurrent kernel execution
- GPU memory pool warm-up and pre-allocation
- Batch array transfer helpers (to_gpu_batch / to_cpu_batch)
- Force-GPU mode for maximum acceleration
- @gpu_accelerated decorator for automatic GPU/CPU dispatch

Usage:
    from src.core.parallel.gpu_backend import gpu_manager, get_array_module

    xp = get_array_module()  # Returns cupy if GPU available, else numpy
    arr = xp.array([1, 2, 3])  # Works on GPU or CPU transparently
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import logging
import os
import sys
import functools
import time as _time

logger = logging.getLogger(__name__)


def _setup_cuda_dll_path():
    """Add CUDA Toolkit DLL directories to the search path on Windows."""
    if sys.platform != 'win32':
        return
    
    cuda_path = os.environ.get('CUDA_PATH', '')
    if not cuda_path:
        return
    
    # CUDA 13.x puts DLLs in bin/x64, older versions use bin directly
    dll_dirs = [
        os.path.join(cuda_path, 'bin', 'x64'),
        os.path.join(cuda_path, 'bin'),
    ]
    
    for dll_dir in dll_dirs:
        if os.path.isdir(dll_dir):
            # Python 3.8+ on Windows requires explicit DLL directory registration
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(dll_dir)
                except OSError:
                    pass
            # Also add to PATH as fallback
            if dll_dir not in os.environ.get('PATH', ''):
                os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

_setup_cuda_dll_path()


@dataclass
class GPUConfig:
    """Configuration for GPU acceleration"""
    enabled: bool = True          # Whether to attempt GPU usage
    device_id: int = 0            # CUDA device index
    memory_limit: Optional[int] = None  # GPU memory limit in bytes (None = no limit)
    batch_size: int = 1024        # Default batch size for GPU operations
    min_array_size: int = 100     # Minimum array size to benefit from GPU transfer
    force_gpu: bool = False       # Force GPU for all operations regardless of array size
    n_streams: int = 4            # Number of CUDA streams for concurrent execution
    warmup_bytes: int = 64 * 1024 * 1024  # 64 MB warm-up allocation


@dataclass
class GPUInfo:
    """Information about the detected GPU"""
    available: bool = False
    device_name: str = "None"
    compute_capability: str = "N/A"
    total_memory_mb: float = 0.0
    cuda_version: str = "N/A"
    cupy_version: str = "N/A"


class GPUManager:
    """
    Singleton manager for GPU resources.
    
    Handles GPU detection, array module selection, data transfer,
    CUDA stream management and memory pool warm-up.
    Provides a unified interface that works on both GPU (CuPy) and CPU (NumPy).
    """
    
    _instance: Optional['GPUManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config = GPUConfig()
        self._gpu_info = GPUInfo()
        self._cupy = None
        self._gpu_available = False
        self._streams: List[Any] = []
        self._stream_index = 0
        self._perf_stats: Dict[str, float] = {}
        self._detect_gpu()
    
    def _detect_gpu(self):
        """Detect NVIDIA GPU and CuPy availability"""
        if not self._config.enabled:
            logger.info("GPU acceleration disabled by configuration")
            return
        
        try:
            import cupy as cp
            
            # Test that CUDA actually works
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                logger.info("CuPy installed but no CUDA devices found")
                return
            
            # Select device
            device_id = min(self._config.device_id, device_count - 1)
            cp.cuda.Device(device_id).use()
            
            # Get device info
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            
            device_name = props.get('name', b'Unknown')
            if isinstance(device_name, bytes):
                device_name = device_name.decode()
            
            self._gpu_info = GPUInfo(
                available=True,
                device_name=str(device_name),
                compute_capability=f"{props.get('major', '?')}.{props.get('minor', '?')}",
                total_memory_mb=props.get('totalGlobalMem', 0) / (1024 * 1024),
                cuda_version=str(cp.cuda.runtime.runtimeGetVersion()),
                cupy_version=cp.__version__
            )
            
            # Quick smoke test — allocate and compute on GPU
            test = cp.array([1.0, 2.0, 3.0])
            _ = cp.sum(test)
            
            # Set memory limit if configured
            if self._config.memory_limit is not None:
                pool = cp.get_default_memory_pool()
                pool.set_limit(size=self._config.memory_limit)
            
            self._cupy = cp
            self._gpu_available = True
            
            # Create CUDA stream pool for concurrent execution
            self._create_stream_pool()
            
            # Warm up GPU memory pool
            self._warmup_memory_pool()
            
            logger.info(f"GPU Detected: {self._gpu_info.device_name} "
                       f"({self._gpu_info.total_memory_mb:.0f} MB, "
                       f"SM {self._gpu_info.compute_capability}) "
                       f"| {self._config.n_streams} CUDA streams")
            
        except ImportError:
            logger.info("CuPy not installed - running on CPU (pip install cupy-cuda12x)")
        except Exception as e:
            logger.warning(f"GPU detection failed: {e} - falling back to CPU")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CUDA Stream Pool
    # ─────────────────────────────────────────────────────────────────────────
    
    def _create_stream_pool(self):
        """Create a pool of CUDA streams for concurrent kernel execution."""
        if not self._gpu_available or self._cupy is None:
            return
        try:
            cp = self._cupy
            self._streams = [cp.cuda.Stream(non_blocking=True) 
                           for _ in range(self._config.n_streams)]
            logger.debug(f"Created {len(self._streams)} CUDA streams")
        except Exception as e:
            logger.warning(f"Failed to create CUDA stream pool: {e}")
            self._streams = []
    
    def get_stream(self) -> Any:
        """
        Get the next CUDA stream from the round-robin pool.
        Returns None if GPU is not available.
        """
        if not self._streams:
            return None
        stream = self._streams[self._stream_index % len(self._streams)]
        self._stream_index += 1
        return stream
    
    def _warmup_memory_pool(self):
        """
        Pre-allocate GPU memory to avoid allocation stalls during simulation.
        Allocates and immediately frees a block to warm up the memory pool.
        """
        if not self._gpu_available or self._cupy is None:
            return
        try:
            cp = self._cupy
            warmup_size = self._config.warmup_bytes
            # Allocate a temporary array to warm up the pool
            temp = cp.empty(warmup_size // 8, dtype=cp.float64)
            del temp
            # Don't free all blocks — keep them in the pool for reuse
            logger.debug(f"GPU memory pool warmed up ({warmup_size // (1024*1024)} MB)")
        except Exception as e:
            logger.warning(f"GPU memory warm-up failed: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def available(self) -> bool:
        """Whether a CUDA GPU is available and working"""
        return self._gpu_available
    
    @property
    def info(self) -> GPUInfo:
        """GPU information"""
        return self._gpu_info
    
    @property
    def config(self) -> GPUConfig:
        """GPU configuration"""
        return self._config
    
    @property
    def cupy(self):
        """Direct access to cupy module, or None if not available"""
        return self._cupy
    
    def configure(self, **kwargs):
        """Update GPU configuration. Takes same kwargs as GPUConfig fields."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        # Re-detect if 'enabled' changed
        if 'enabled' in kwargs:
            self._gpu_available = False
            self._cupy = None
            self._streams = []
            if kwargs['enabled']:
                self._detect_gpu()
        # Handle force_gpu toggling
        if 'force_gpu' in kwargs and kwargs['force_gpu']:
            self._config.min_array_size = 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # Array Module & Data Transfer
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_array_module(self) -> Any:
        """
        Get the array module to use (CuPy for GPU, NumPy for CPU).
        
        Returns:
            cupy if GPU is available, numpy otherwise
        """
        if self._gpu_available and self._cupy is not None:
            return self._cupy
        return np
    
    def to_gpu(self, array: np.ndarray) -> Any:
        """
        Transfer a NumPy array to GPU memory.
        Returns GPU array if available, otherwise returns the input unchanged.
        """
        if self._gpu_available and self._cupy is not None:
            if isinstance(array, np.ndarray):
                return self._cupy.asarray(array)
        return array
    
    def to_cpu(self, array) -> np.ndarray:
        """
        Transfer an array to CPU memory (NumPy).
        Works with both CuPy and NumPy arrays.
        """
        if self._gpu_available and self._cupy is not None:
            if hasattr(array, 'get'):
                return array.get()
        return np.asarray(array)
    
    def to_gpu_batch(self, *arrays: np.ndarray) -> Tuple:
        """
        Transfer multiple NumPy arrays to GPU in one call.
        
        Args:
            *arrays: Variable number of NumPy arrays
            
        Returns:
            Tuple of GPU arrays (or original arrays if GPU not available)
        """
        if self._gpu_available and self._cupy is not None:
            return tuple(self._cupy.asarray(a) if isinstance(a, np.ndarray) else a 
                        for a in arrays)
        return arrays
    
    def to_cpu_batch(self, *arrays) -> Tuple[np.ndarray, ...]:
        """
        Transfer multiple GPU arrays to CPU in one call.
        
        Args:
            *arrays: Variable number of GPU/CPU arrays
            
        Returns:
            Tuple of NumPy arrays
        """
        return tuple(self.to_cpu(a) for a in arrays)
    
    def should_use_gpu(self, array_size: int) -> bool:
        """
        Determine if GPU should be used for the given data size.
        In force_gpu mode, always returns True if GPU is available.
        """
        if self._config.force_gpu and self._gpu_available:
            return True
        return (self._gpu_available and 
                array_size >= self._config.min_array_size)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Synchronization & Memory
    # ─────────────────────────────────────────────────────────────────────────
    
    def synchronize(self):
        """Wait for all GPU operations to complete (all streams)."""
        if self._gpu_available and self._cupy is not None:
            # Synchronize all streams
            for stream in self._streams:
                stream.synchronize()
            # Also sync the default stream
            self._cupy.cuda.Stream.null.synchronize()
    
    def synchronize_stream(self, stream=None):
        """Synchronize a specific stream, or null stream if None."""
        if not self._gpu_available or self._cupy is None:
            return
        if stream is not None:
            stream.synchronize()
        else:
            self._cupy.cuda.Stream.null.synchronize()
    
    def memory_info(self) -> Dict[str, float]:
        """Get GPU memory usage in MB"""
        if not self._gpu_available or self._cupy is None:
            return {"used_mb": 0, "total_mb": 0, "free_mb": 0}
        
        pool = self._cupy.get_default_memory_pool()
        free, total = self._cupy.cuda.runtime.memGetInfo()
        return {
            "used_mb": pool.used_bytes() / (1024 * 1024),
            "total_mb": total / (1024 * 1024),
            "free_mb": free / (1024 * 1024),
            "pool_total_mb": pool.total_bytes() / (1024 * 1024)
        }
    
    def clear_memory(self):
        """Free cached GPU memory"""
        if self._gpu_available and self._cupy is not None:
            pool = self._cupy.get_default_memory_pool()
            pool.free_all_blocks()
            pinned_pool = self._cupy.get_default_pinned_memory_pool()
            pinned_pool.free_all_blocks()
    
    def get_status_string(self) -> str:
        """Get a human-readable status string for display in UI"""
        if self._gpu_available:
            mem = self.memory_info()
            mode = "Force-GPU" if self._config.force_gpu else "Auto"
            return (f"GPU: {self._gpu_info.device_name} | "
                    f"Memory: {mem['used_mb']:.0f}/{mem['total_mb']:.0f} MB (Alloc: {mem['pool_total_mb']:.0f} MB) | "
                    f"Streams: {len(self._streams)} | Mode: {mode} | "
                    f"CuPy {self._gpu_info.cupy_version}")
        else:
            return "GPU: Not available (CPU mode)"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Performance Tracking
    # ─────────────────────────────────────────────────────────────────────────
    
    def track_time(self, label: str, elapsed: float):
        """Accumulate timing stats for a labeled operation."""
        if label not in self._perf_stats:
            self._perf_stats[label] = 0.0
        self._perf_stats[label] += elapsed
    
    def get_perf_stats(self) -> Dict[str, float]:
        """Get accumulated performance stats."""
        return dict(self._perf_stats)
    
    def reset_perf_stats(self):
        """Reset performance tracking."""
        self._perf_stats.clear()


# --- Module-level convenience functions ---

# Global singleton instance
gpu_manager = GPUManager()


def get_array_module():
    """Get the array module (cupy or numpy) based on GPU availability"""
    return gpu_manager.get_array_module()


def to_gpu(array: np.ndarray):
    """Transfer array to GPU if available"""
    return gpu_manager.to_gpu(array)


def to_cpu(array) -> np.ndarray:
    """Transfer array to CPU"""
    return gpu_manager.to_cpu(array)


def to_gpu_batch(*arrays: np.ndarray):
    """Transfer multiple arrays to GPU if available"""
    return gpu_manager.to_gpu_batch(*arrays)


def to_cpu_batch(*arrays):
    """Transfer multiple arrays to CPU"""
    return gpu_manager.to_cpu_batch(*arrays)


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available"""
    return gpu_manager.available


def gpu_accelerated(func):
    """
    Decorator that automatically handles GPU/CPU dispatch.
    
    Wraps a function so that:
    1. NumPy inputs are transferred to GPU if available
    2. The function runs with the appropriate array module
    3. Results are transferred back to CPU
    4. Timing is tracked for performance analysis
    
    The decorated function receives an extra `xp` keyword argument
    with the active array module.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = _time.perf_counter()
        xp = gpu_manager.get_array_module()
        kwargs['xp'] = xp
        result = func(*args, **kwargs)
        elapsed = _time.perf_counter() - t0
        gpu_manager.track_time(func.__name__, elapsed)
        return result
    return wrapper
