"""
CUDA-Accelerated Computational Kernels for UWB Simulation

GPU-optimized versions of the computationally intensive operations in the
UWB channel model. All functions work transparently on both GPU (CuPy) and
CPU (NumPy) through the xp (array module) pattern.

Main acceleration targets:
- CIR generation: vectorized Gaussian pulse superposition (batched for all anchors)
- Path loss: batch computation for multiple anchors
- ToA detection: parallel threshold crossing (padded batch)
- Ranging errors: GPU-accelerated random number generation
- Channel statistics: batch PDP, RMS delay spread, kurtosis
- Fused CuPy ElementwiseKernel for pulse+phase+amplitude
"""

import numpy as np
from typing import Tuple, List, Optional
from src.core.parallel.gpu_backend import (
    get_array_module, to_cpu, to_gpu, to_gpu_batch, to_cpu_batch, gpu_manager
)
import logging
import time as _time

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 0. Fused CUDA Kernel (CuPy ElementwiseKernel)
# ─────────────────────────────────────────────────────────────────────────────

_fused_pulse_kernel = None

def _get_fused_pulse_kernel():
    """
    Lazily create a CuPy ElementwiseKernel that fuses the Gaussian pulse
    computation with complex amplitude multiplication.
    
    This avoids multiple GPU memory passes for exp, multiply, etc.
    """
    global _fused_pulse_kernel
    if _fused_pulse_kernel is not None:
        return _fused_pulse_kernel
    
    cp = gpu_manager.cupy
    if cp is None:
        return None
    
    try:
        _fused_pulse_kernel = cp.ElementwiseKernel(
            'float64 dt, float64 sigma, float64 amp, float64 phase',
            'complex128 result',
            '''
            double gauss = exp(-0.5 * (dt / sigma) * (dt / sigma));
            double re = amp * cos(phase) * gauss;
            double im = amp * sin(phase) * gauss;
            result = thrust::complex<double>(re, im);
            ''',
            'fused_pulse_kernel'
        )
        logger.debug("Created fused CUDA pulse kernel")
    except Exception as e:
        logger.warning(f"Failed to create fused kernel: {e}")
        _fused_pulse_kernel = None
    
    return _fused_pulse_kernel


# ─────────────────────────────────────────────────────────────────────────────
# I. Vectorized CIR Generation
# ─────────────────────────────────────────────────────────────────────────────

def vectorized_cir_pulse_superposition(
    t_vector: np.ndarray,
    path_delays: np.ndarray,
    path_amplitudes: np.ndarray,
    path_phases: np.ndarray,
    pulse_width: float,
    return_on_device: bool = False
) -> np.ndarray:
    """
    Vectorized Gaussian pulse superposition for CIR generation.
    
    h(t) = Σ_k  α_k * exp(j·φ_k) * exp(-0.5 * ((t - τ_k) / σ)²)
    
    Uses fused CUDA kernel when available for reduced memory traffic.
    
    Args:
        t_vector: Time samples array (N_t,)
        path_delays: Delay for each multipath component (N_paths,)
        path_amplitudes: Amplitude for each component (N_paths,)
        path_phases: Phase for each component (N_paths,)
        pulse_width: Gaussian pulse width (1/bandwidth)
        return_on_device: If True, returns CuPy array (if GPU used). If False, converts to CPU.
    
    Returns:
        Complex impulse response h(t) of shape (N_t,)
    """
    xp = get_array_module()
    
    n_paths = len(path_delays)
    n_time = len(t_vector)
    
    # Transfer to GPU if beneficial
    # If return_on_device is requested, we MUST use GPU if available
    use_gpu = gpu_manager.should_use_gpu(n_paths * n_time) or (return_on_device and gpu_manager.available)
    
    if use_gpu:
        t_vec, delays, amps, phases = to_gpu_batch(
            t_vector, path_delays, path_amplitudes, path_phases)
    else:
        t_vec = xp.asarray(t_vector)
        delays = xp.asarray(path_delays)
        amps = xp.asarray(path_amplitudes)
        phases = xp.asarray(path_phases)
    
    sigma = pulse_width / 2.355  # FWHM to sigma
    
    # Try fused kernel first (single pass over GPU memory)
    fused_kernel = _get_fused_pulse_kernel() if use_gpu else None
    
    if fused_kernel is not None:
        try:
            # Broadcasting: (N_t, 1) - (1, N_paths) → (N_t, N_paths)
            dt = t_vec[:, None] - delays[None, :]
            amps_bc = xp.broadcast_to(amps[None, :], dt.shape)
            phases_bc = xp.broadcast_to(phases[None, :], dt.shape)
            
            # Fused kernel: Gaussian * complex amplitude in one pass
            pulses_complex = fused_kernel(dt, sigma, amps_bc, phases_bc)
            
            # Sum over all paths
            h_t = xp.sum(pulses_complex, axis=1)
            return h_t if return_on_device else to_cpu(h_t)
        except Exception:
            pass  # Fallback to standard path
    
    # Standard vectorized path (broadcasting)
    dt = t_vec[:, None] - delays[None, :]           # (N_t, N_paths)
    pulses = xp.exp(-0.5 * (dt / sigma) ** 2)       # Gaussian pulses
    complex_amps = amps * xp.exp(1j * phases)        # (N_paths,)
    h_t = xp.dot(pulses, complex_amps)               # (N_t,)
    
    return h_t if return_on_device else to_cpu(h_t)


# ─────────────────────────────────────────────────────────────────────────────
# II. Batch Path Loss Computation
# ─────────────────────────────────────────────────────────────────────────────

def batch_path_loss(
    distances: np.ndarray,
    frequency: float,
    center_frequency: float,
    path_loss_exponent: float,
    reference_distance: float,
    reference_loss_db: float,
    shadow_fading_std: float,
    frequency_decay_factor: float,
    is_los_array: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """
    Compute path loss for multiple anchor-tag pairs simultaneously.
    
    PL(f, d) = PL_0 + 20·κ·log10(f/f_c) + 10·n·log10(d/d_0) + X_σ + NLOS_const
    """
    xp = get_array_module()
    
    use_gpu = gpu_manager.should_use_gpu(len(distances))
    
    if use_gpu:
        d, is_los = to_gpu_batch(
            np.maximum(distances, 0.001).astype(np.float64),
            is_los_array.astype(np.float64)
        )
    else:
        d = xp.maximum(xp.asarray(distances, dtype=np.float64), 0.001)
        is_los = xp.asarray(is_los_array, dtype=np.float64)
    
    # All computations vectorized
    kappa = frequency_decay_factor
    freq_term = 20.0 * kappa * xp.log10(frequency / center_frequency)
    dist_term = 10.0 * path_loss_exponent * xp.log10(d / reference_distance)
    shadowing = xp.random.normal(0, shadow_fading_std, size=len(d))
    nlos_const = (1.0 - is_los) * 10.0  # 10 dB extra for NLOS
    ref_loss = -reference_loss_db  # reference_loss_db is gain (e.g. -43dB), negate for loss (+43dB)
    
    total_loss_db = ref_loss + freq_term + dist_term + shadowing + nlos_const
    
    breakdown = {
        "ref_loss": float(ref_loss),
        "freq_loss": float(freq_term),
        "dist_loss": to_cpu(dist_term),
        "shadowing": to_cpu(shadowing),
        "nlos_loss": to_cpu(nlos_const),
        "total_loss": to_cpu(total_loss_db)
    }
    
    return to_cpu(total_loss_db), breakdown


# ─────────────────────────────────────────────────────────────────────────────
# III. Batch ToA Detection (Padded)
# ─────────────────────────────────────────────────────────────────────────────

def batch_toa_detection(
    time_vectors: List[np.ndarray],
    h_t_list: List[np.ndarray],
    snr_linears: np.ndarray,
    threshold_factor: float = 0.1
) -> List[Tuple[float, bool]]:
    """
    Detect Time of Arrival from multiple CIRs.
    
    Uses padded-batch GPU approach: pads all CIRs to equal length,
    processes all at once on GPU, then extracts per-anchor results.
    """
    xp = get_array_module()
    n_cirs = len(time_vectors)
    
    if n_cirs == 0:
        return []
    
    # Pad all CIRs to the same length for batch GPU processing
    max_len = max(len(t) for t in time_vectors)
    use_gpu = gpu_manager.should_use_gpu(n_cirs * max_len)
    
    if use_gpu and n_cirs > 1:
        # Padded batch approach
        t_batch = np.zeros((n_cirs, max_len), dtype=np.float64)
        h_batch = np.zeros((n_cirs, max_len), dtype=np.complex128)
        lengths = np.zeros(n_cirs, dtype=np.int64)
        
        for i, (t_vec, h_t) in enumerate(zip(time_vectors, h_t_list)):
            # If t_vec/h_t are already on GPU, we need to bring them back to CPU for packing?
            # Or we can pack them on GPU? Packing on GPU is harder with padding.
            # Best approach for now: optimized "ping" to CPU for packing, then batch "pong" to GPU.
            # OR better: if they are GPU arrays, use cp.zeros and copy into them?
            # Complicated due to variable lengths.
            # Let's check type.
            
            # If inputs are already GPU arrays (from return_on_device=True)
            if hasattr(t_vec, 'device'): # CuPy array
                # Use to_cpu for packing (costly but necessary for ragged batching unless we rewrite to avoid padding)
                # NOTE: For maximum performance we should pre-allocate max size on GPU, but length varies randomly.
                t_vec = to_cpu(t_vec)
                h_t = to_cpu(h_t)
            
            L = len(t_vec)
            t_batch[i, :L] = t_vec
             # Ensure h_t is complex
            h_batch[i, :L] = h_t
            lengths[i] = L
        
        # Transfer to GPU
        t_g, h_g = to_gpu_batch(t_batch, h_batch)
        
        # Compute PDP for all CIRs at once
        pdp = xp.abs(h_g) ** 2
        peak_power = xp.max(pdp, axis=1, keepdims=True)  # (n_cirs, 1)
        effective_threshold = threshold_factor * peak_power  # (n_cirs, 1)
        
        # Mask: above threshold
        above = pdp > effective_threshold  # (n_cirs, max_len)
        
        # Extract results per CIR
        results = []
        above_cpu = to_cpu(above)
        t_cpu = to_cpu(t_g)
        pdp_cpu = to_cpu(pdp)
        
        for i in range(n_cirs):
            L = int(lengths[i])
            indices = np.where(above_cpu[i, :L])[0]
            
            if len(indices) == 0:
                idx = int(np.argmax(pdp_cpu[i, :L]))
                results.append((float(t_cpu[i, idx]), False))
            else:
                first_idx = int(indices[0])
                results.append((float(t_cpu[i, first_idx]), True))
        
        return results
    
    # Sequential fallback
    results = []
    for t_vec, h_t in zip(time_vectors, h_t_list):
        if use_gpu:
            t_g = to_gpu(t_vec)
            h_g = to_gpu(h_t)
        else:
            t_g = xp.asarray(t_vec)
            h_g = xp.asarray(h_t)
        
        pdp = xp.abs(h_g) ** 2
        peak_power = xp.max(pdp)
        effective_threshold = threshold_factor * peak_power
        
        indices = xp.where(pdp > effective_threshold)[0]
        
        if len(indices) == 0:
            idx = int(xp.argmax(pdp))
            results.append((float(to_cpu(t_g[idx])), False))
        else:
            first_idx = int(indices[0])
            results.append((float(to_cpu(t_g[first_idx])), True))
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# IV. Batch Ranging Error Generation
# ─────────────────────────────────────────────────────────────────────────────

def batch_ranging_errors(
    n_samples: int,
    sigma_crlb_array: np.ndarray,
    sigma_jitter: float,
    is_los_array: np.ndarray,
    noise_factors: np.ndarray,
    error_biases: np.ndarray,
    rms_delay_spreads: np.ndarray,
    c: float = 299792458.0,
    noise_model: str = "gaussian"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ranging errors for N measurements simultaneously on GPU.
    
    All random number generation is batched for GPU efficiency.
    """
    xp = get_array_module()
    use_gpu = gpu_manager.should_use_gpu(n_samples)
    
    if use_gpu:
        sigma_crlb, is_los, nf, eb, rds = to_gpu_batch(
            sigma_crlb_array.astype(np.float64),
            is_los_array.astype(np.float64),
            noise_factors.astype(np.float64),
            error_biases.astype(np.float64),
            rms_delay_spreads.astype(np.float64)
        )
    else:
        sigma_crlb = xp.asarray(sigma_crlb_array, dtype=np.float64)
        is_los = xp.asarray(is_los_array, dtype=np.float64)
        nf = xp.asarray(noise_factors, dtype=np.float64)
        eb = xp.asarray(error_biases, dtype=np.float64)
        rds = xp.asarray(rms_delay_spreads, dtype=np.float64)
    
    # Total std = sqrt(sigma_crlb² + sigma_jitter²)
    total_std = xp.sqrt(sigma_crlb ** 2 + sigma_jitter ** 2)
    
    # Generate noise samples (all at once on GPU)
    model_lower = noise_model.lower()
    if model_lower == "gaussian":
        noise = xp.random.normal(0, 1, size=n_samples) * total_std
    elif "non-centralized" in model_lower:
        # Non-centralized Gaussian: mean shifted by 0.5*std
        noise = xp.random.normal(0, 1, size=n_samples) * total_std + total_std * 0.5
    elif model_lower == "laplace":
        b = total_std / xp.sqrt(2.0)
        noise = xp.random.laplace(0, 1, size=n_samples) * b
    elif model_lower == "uniform":
        w = total_std * xp.sqrt(3.0)
        noise = xp.random.uniform(-1, 1, size=n_samples) * w
    elif "student" in model_lower:
        df = 4
        scale = total_std / xp.sqrt(df / (df - 2))
        noise = xp.random.standard_normal(size=n_samples) * scale
    elif "mixed" in model_lower:
        base = xp.random.normal(0, 1, size=n_samples) * total_std
        heavy = xp.random.normal(0, 1, size=n_samples) * total_std * 3
        mask = xp.random.random(size=n_samples) < 0.9
        noise = xp.where(mask, base, heavy)
    else:
        noise = xp.random.normal(0, 1, size=n_samples) * total_std
    
    # Apply NLOS noise multiplier
    nlos_mask = (1.0 - is_los)
    noise = noise * (is_los + nlos_mask * nf)
    
    # NLOS bias: exponential random bias scaled by delay spread
    rho = 0.1
    exp_random = xp.random.exponential(1.0, size=n_samples)
    bias = nlos_mask * (eb + c * rds * rho * exp_random)
    
    errors = noise + bias
    
    return to_cpu(errors), to_cpu(total_std)


# ─────────────────────────────────────────────────────────────────────────────
# V. Batch Channel Statistics
# ─────────────────────────────────────────────────────────────────────────────

def batch_channel_statistics(
    time_vectors: List[np.ndarray],
    h_t_list: List[np.ndarray]
) -> List[dict]:
    """
    Compute channel statistics for multiple CIRs at once on GPU.
    
    For each CIR, computes:
    - PDP (Power Delay Profile)
    - Mean Excess Delay
    - RMS Delay Spread
    - Kurtosis of amplitude distribution
    
    Uses padded batch processing for GPU efficiency.
    
    Args:
        time_vectors: List of time vectors, one per CIR
        h_t_list: List of complex impulse responses
    
    Returns:
        List of dicts with statistics for each CIR
    """
    xp = get_array_module()
    n_cirs = len(time_vectors)
    
    if n_cirs == 0:
        return []
    
    max_len = max(len(t) for t in time_vectors)
    use_gpu = gpu_manager.should_use_gpu(n_cirs * max_len)
    
    if use_gpu and n_cirs > 1:
        # Pad and batch
        t_batch = np.zeros((n_cirs, max_len), dtype=np.float64)
        h_batch = np.zeros((n_cirs, max_len), dtype=np.complex128)
        mask_batch = np.zeros((n_cirs, max_len), dtype=np.float64)
        lengths = []
        
        for i, (t_vec, h_t) in enumerate(zip(time_vectors, h_t_list)):
            # Handle potential GPU arrays
            if hasattr(t_vec, 'device'):
                t_vec = to_cpu(t_vec)
                h_t = to_cpu(h_t)
                
            L = len(t_vec)
            t_batch[i, :L] = t_vec
            h_batch[i, :L] = h_t
            mask_batch[i, :L] = 1.0
            lengths.append(L)
        
        t_g, h_g, mask_g = to_gpu_batch(t_batch, h_batch, mask_batch)
        
        # PDP for all CIRs
        pdp = xp.abs(h_g) ** 2 * mask_g
        total_energy = xp.sum(pdp, axis=1, keepdims=True)  # (n_cirs, 1)
        
        # Avoid division by zero
        total_energy_safe = xp.maximum(total_energy, 1e-30)
        pdp_norm = pdp / total_energy_safe
        
        # Mean excess delay: sum(t * pdp_norm) - t[0]
        t_start = t_g[:, 0:1]  # (n_cirs, 1)
        weighted_t = xp.sum(t_g * pdp_norm, axis=1)  # (n_cirs,)
        mean_excess_delay = weighted_t - t_g[:, 0]
        
        # RMS delay spread
        second_moment = xp.sum((t_g ** 2) * pdp_norm, axis=1)
        rms_delay_spread = xp.sqrt(xp.maximum(second_moment - weighted_t ** 2, 0.0))
        
        # Kurtosis of magnitude
        mag = xp.abs(h_g) * mask_g
        # Per-CIR statistics using masked operations
        n_valid = xp.sum(mask_g, axis=1, keepdims=True)
        n_valid_safe = xp.maximum(n_valid, 1.0)
        mag_mean = xp.sum(mag, axis=1, keepdims=True) / n_valid_safe
        mag_centered = (mag - mag_mean) * mask_g
        mag_var = xp.sum(mag_centered ** 2, axis=1, keepdims=True) / n_valid_safe
        mag_std = xp.sqrt(xp.maximum(mag_var, 1e-30))
        mag_normalized = mag_centered / mag_std
        kurtosis = xp.sum((mag_normalized ** 4) * mask_g, axis=1) / xp.squeeze(n_valid_safe)
        
        # Transfer back
        mean_delay_cpu = to_cpu(mean_excess_delay)
        rms_cpu = to_cpu(rms_delay_spread)
        kurtosis_cpu = to_cpu(kurtosis)
        total_e_cpu = to_cpu(xp.squeeze(total_energy))
        
        results = []
        for i in range(n_cirs):
            e = float(total_e_cpu[i]) if n_cirs > 1 else float(total_e_cpu)
            results.append({
                'mean_excess_delay': float(mean_delay_cpu[i]) if e > 0 else 0.0,
                'rms_delay_spread': float(rms_cpu[i]) if e > 0 else 0.0,
                'kurtosis': float(kurtosis_cpu[i]) if e > 0 else 0.0
            })
        return results
    
    # Sequential fallback
    results = []
    for t_vec, h_t in zip(time_vectors, h_t_list):
        pdp = np.abs(h_t) ** 2
        total_energy = np.sum(pdp)
        
        if total_energy > 0:
            pdp_norm = pdp / total_energy
            mean_excess_delay = np.sum(t_vec * pdp_norm) - t_vec[0]
            second_moment = np.sum((t_vec ** 2) * pdp_norm)
            rms_delay_spread = np.sqrt(max(second_moment - (np.sum(t_vec * pdp_norm)) ** 2, 0.0))
            
            mag = np.abs(h_t)
            mag_mean = np.mean(mag)
            mag_std = np.std(mag)
            kurtosis = float(np.mean(((mag - mag_mean) / mag_std) ** 4)) if mag_std > 0 else 0.0
        else:
            mean_excess_delay = rms_delay_spread = kurtosis = 0.0
        
        results.append({
            'mean_excess_delay': float(mean_excess_delay),
            'rms_delay_spread': float(rms_delay_spread),
            'kurtosis': float(kurtosis)
        })
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# VI. Batch Distance Measurement (Full Pipeline) — THE BIG ONE
# ─────────────────────────────────────────────────────────────────────────────

def batch_measure_distances(
    channel_model,
    true_distances: np.ndarray,
    is_los_array: np.ndarray,
    anchor_positions: Optional[List] = None
) -> List:
    """
    Perform complete UWB ranging simulation for multiple anchors simultaneously.
    
    Full GPU-accelerated pipeline:
    1. Batch path loss computation
    2. Vectorized link budget
    3. Per-anchor CIR generation (GPU-vectorized internally)
    4. Batch ToA detection (padded GPU)
    5. Batch ranging error generation
    6. Batch channel statistics computation
    
    The per-anchor CIR generation still requires individual calls due to
    random multipath structure, but all post-CIR processing is batched.
    
    Args:
        channel_model: UWBChannelModel instance
        true_distances: Array of true distances to each anchor (N,)
        is_los_array: LOS condition for each anchor (N,)
        anchor_positions: List of anchor Positions (Optional)
    
    Returns:
        List of RangingResult objects (one per anchor)
    """
    from src.core.uwb.uwb_types import RangingResult
    
    t0_perf = _time.perf_counter()
    
    n_anchors = len(true_distances)
    xp = get_array_module()
    c = channel_model.c
    
    # --- 1. Batch Path Loss (all anchors at once) ---
    path_losses, pl_breakdowns = batch_path_loss(
        distances=true_distances,
        frequency=channel_model.uwb_params.center_frequency,
        center_frequency=channel_model.uwb_params.center_frequency,
        path_loss_exponent=channel_model.current_path_loss_params.path_loss_exponent,
        reference_distance=channel_model.current_path_loss_params.reference_distance,
        reference_loss_db=channel_model.current_path_loss_params.reference_loss_db,
        shadow_fading_std=channel_model.current_path_loss_params.shadow_fading_std,
        frequency_decay_factor=channel_model.current_path_loss_params.frequency_decay_factor,
        is_los_array=is_los_array
    )
    
    # --- 2. Vectorized Link Budget (all anchors at once) ---
    tx_power_total_dbm = channel_model.uwb_params.tx_power_dbm
    if tx_power_total_dbm < -30:
        bw_mhz = channel_model.uwb_params.bandwidth / 1e6
        tx_power_total_dbm += 10 * np.log10(bw_mhz)
    
    rx_power_dbm = (tx_power_total_dbm +
                    channel_model.uwb_params.tx_antenna_gain_dbi +
                    channel_model.uwb_params.rx_antenna_gain_dbi -
                    path_losses)
    
    rx_power_watts = 10 ** ((rx_power_dbm - 30) / 10)
    noise_power_watts = channel_model.calculate_thermal_noise()
    snr_linear = rx_power_watts / noise_power_watts
    snr_db = 10 * np.log10(np.maximum(snr_linear, 1e-10))
    
    # --- 3. Per-anchor CIR generation (GPU-vectorized internally) ---
    # Each CIR has random multipath structure, so we generate individually
    # but collect for batch post-processing
    time_vectors = []
    h_t_list = []
    t0_trues = []
    B = channel_model.uwb_params.bandwidth
    
    # Pre-calculate workload size to decide if we should keep data on GPU
    # Approx: N_anchors * (Delay / dt) -> N_anchors * N_samples
    # Simple heuristic: if we have anchors, it's worth it for the full pipeline
    keep_on_gpu = gpu_manager.available and n_anchors > 0
    
    for i in range(n_anchors):
        dist = float(true_distances[i])
        is_los = bool(is_los_array[i])
        anchor_pos = anchor_positions[i] if anchor_positions is not None else None

        # Force return_on_device=True if we are keeping on GPU
        # We need to update channel_model.generate_unified_cir to accept this arg
        if hasattr(channel_model, 'generate_unified_cir'):
            try:
                # Type check to see if method accepts return_on_device
                # For now, we assume it does based on our update plan
                t_vec, h_t, t0_true = channel_model.generate_unified_cir(
                    dist, is_los, return_on_device=keep_on_gpu, anchor_pos=anchor_pos)
            except TypeError:
                # Fallback if method signature hasn't updated yet
                 t_vec, h_t, t0_true = channel_model.generate_unified_cir(dist, is_los, anchor_pos=anchor_pos)
                 if keep_on_gpu and not hasattr(h_t, 'device'):
                     t_vec = to_gpu(t_vec)
                     h_t = to_gpu(h_t)
        else:
            # Fallback
            t_vec, h_t, t0_true = channel_model.generate_unified_cir(dist, is_los)
            
        time_vectors.append(t_vec)
        h_t_list.append(h_t)
        t0_trues.append(t0_true)
    
    # --- 4. Batch ToA Detection (padded GPU) ---
    # We pass the list of (likely) GPU arrays directly
    # batch_toa_detection handles lists of GPU arrays correctly
    toa_results = batch_toa_detection(
        time_vectors=time_vectors,
        h_t_list=h_t_list,
        snr_linears=snr_linear,
        threshold_factor=channel_model.detection_threshold_factor
    )
    
    # --- 5. Batch Ranging Errors (all anchors at once) ---
    snr_safe = np.maximum(snr_linear, 0.1)
    sigma_crlb_array = c / (2 * np.pi * B * np.sqrt(2 * snr_safe))
    sigma_jitter = channel_model.uwb_params.fixed_noise_std
    
    noise_factors = np.array([channel_model.current_noise_factor] * n_anchors)
    error_biases = np.array([channel_model.current_error_bias] * n_anchors)
    rms_delay_spreads = np.array([channel_model.current_rms_delay_spread] * n_anchors)
    
    errors, total_stds = batch_ranging_errors(
        n_samples=n_anchors,
        sigma_crlb_array=sigma_crlb_array,
        sigma_jitter=sigma_jitter,
        is_los_array=is_los_array,
        noise_factors=noise_factors,
        error_biases=error_biases,
        rms_delay_spreads=rms_delay_spreads,
        c=c,
        noise_model=channel_model.noise_model
    )
    
    # --- 6. Batch Channel Statistics (GPU) ---
    # Also handles GPU arrays in list
    stats_list = batch_channel_statistics(time_vectors, h_t_list)
    
    # --- 7. Assemble Results ---
    # Threshold offset calibration (constant, compute once)
    pulse_width = 1.0 / B
    sigma_pulse = pulse_width / 2.355
    th = channel_model.detection_threshold_factor
    if 0 < th < 1:
        time_offset = sigma_pulse * np.sqrt(-1 * np.log(th))
        dist_offset = time_offset * c
    else:
        dist_offset = 0.0
    
    results = []
    for i in range(n_anchors):
        dist = float(true_distances[i])
        is_los = bool(is_los_array[i])
        toa_est_raw, detected_flag = toa_results[i]
        t0_true = t0_trues[i]
        t_vec = time_vectors[i]
        h_t = h_t_list[i]
        
        mp_bias_observed = (toa_est_raw - t0_true) * c
        noise_err_m = float(errors[i])
        sigma_std = float(total_stds[i])
        
        measured_dist = (toa_est_raw * c) + noise_err_m + dist_offset
        total_error = measured_dist - dist
        
        # Filter unrealistic spikes
        if is_los and abs(total_error) > 0.6:
            total_error = 0.6 if total_error > 0 else -0.6
            measured_dist = dist + total_error
        elif not is_los and abs(total_error) > 0.8:
            total_error = 0.8 if total_error > 0 else -0.8
            measured_dist = dist + total_error
        
        stats = stats_list[i]
        
        # Helper to safely get scalar from potential GPU array or numpy array
        def get_scalar(val):
            return float(val) if np.isscalar(val) else float(to_cpu(val))

        # We probably have GPU arrays for t_vec and h_t now
        # RangingResult expects NumPy arrays usually, but let's check what we need to do.
        # Ideally we convert them to CPU here for storage/display
        t_vec_cpu = to_cpu(t_vec)
        h_t_cpu = to_cpu(h_t)
        
        results.append(RangingResult(
            measured_distance=measured_dist,
            true_distance=dist,
            toa_estimate=measured_dist / c,
            toa_true=dist / c,
            noise_error=noise_err_m,
            multipath_bias=mp_bias_observed,
            nlos_error=0.0,
            total_error=total_error,
            snr_db=float(snr_db[i]),
            snr_linear=float(snr_linear[i]),
            path_loss_db=float(path_losses[i]),
            received_power_dbm=float(rx_power_dbm[i]),
            is_los=is_los,
            first_path_detected=detected_flag,
            measurement_std=sigma_std,
            cir_time_vector=t_vec_cpu,
            cir_amplitude=np.abs(h_t_cpu),
            cir_first_path_index=np.argmax(t_vec_cpu >= toa_est_raw) if detected_flag else np.argmax(np.abs(h_t_cpu)),
            cir_complex=h_t_cpu,
            rms_delay_spread=stats['rms_delay_spread'],
            mean_excess_delay=stats['mean_excess_delay'],
            kurtosis=stats['kurtosis'],
            path_loss_breakdown={k: (float(v) if np.isscalar(v) else float(v[i])) 
                                  for k, v in pl_breakdowns.items()}
        ))
    
    elapsed = _time.perf_counter() - t0_perf
    gpu_manager.track_time('batch_measure_distances', elapsed)
    
    return results
