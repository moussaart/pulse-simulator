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
- Fused CuPy RawKernel & ElementwiseKernel for pulse+phase+amplitude+ToA+stats
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
# 0. Fused CUDA Kernels (CuPy RawKernel & ElementwiseKernel)
# ─────────────────────────────────────────────────────────────────────────────

_fused_pulse_kernel = None
_batch_cir_toa_stats_kernel = None


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


def _get_batch_cir_toa_stats_kernel():
    """
    Lazily create a single-launch fused CUDA RawKernel that computes:
    1. Gaussian pulse superposition & Power Delay Profile (PDP) for all anchors
       using shared-memory path caching and sorted-delay pruning
    2. Peak power and peak index per anchor via shared-memory reduction
    3. First threshold crossing ToA index (with peak fallback) via shared-memory reduction
    4. Optional channel statistics (mean excess delay, RMS delay spread, kurtosis)
    All in one CUDA kernel launch with a single H2D upload and a single D2H download.
    """
    global _batch_cir_toa_stats_kernel
    if _batch_cir_toa_stats_kernel is not None:
        return _batch_cir_toa_stats_kernel

    cp = gpu_manager.cupy
    if cp is None:
        return None

    try:
        _batch_cir_toa_stats_kernel = cp.RawKernel(r'''
extern "C" __global__
void batch_cir_toa_stats(
    const double* __restrict__ path_data,     // [3, N_cirs, stride]: col 0 = (t_start, n_paths, n_samp), cols 1..n_paths = (delays, amp_re, amp_im)
    const double dt,
    const double inv_sigma,
    const double threshold_factor,
    double* __restrict__ pdp_out,             // [N_cirs, max_samples] (only used if write_complex || max_samples > 4096)
    double* __restrict__ h_re_out,            // [N_cirs, max_samples] (only used if write_complex)
    double* __restrict__ h_im_out,            // [N_cirs, max_samples] (only used if write_complex)
    double* __restrict__ summary_out,         // [N_cirs, 6]: toa_est, detected, first_idx, mean_delay, rms_spread, kurtosis
    const int n_cirs,
    const int stride_p,                       // max_paths + 1
    const int max_samples,
    const int write_complex,
    const int compute_stats
) {
    int cir_idx = blockIdx.x;
    if (cir_idx >= n_cirs) return;

    int tid = threadIdx.x;
    int plane_stride = n_cirs * stride_p;
    int base_p = cir_idx * stride_p;

    double t_start = path_data[base_p];
    int n_paths = (int)path_data[plane_stride + base_p];
    int n_samp = (int)path_data[2 * plane_stride + base_p];

    const double* __restrict__ g_delays = path_data + base_p + 1;
    const double* __restrict__ g_amp_re = path_data + plane_stride + base_p + 1;
    const double* __restrict__ g_amp_im = path_data + 2 * plane_stride + base_p + 1;

    // Cache up to 256 paths in shared memory for fast broadcast across threads
    __shared__ double s_delays[256];
    __shared__ double s_amp_re[256];
    __shared__ double s_amp_im[256];
    __shared__ double s_val[256];
    __shared__ int s_idx[256];
    __shared__ double s_stats[5];

    if (n_paths <= 256) {
        if (tid < n_paths) {
            s_delays[tid] = g_delays[tid];
            s_amp_re[tid] = g_amp_re[tid];
            s_amp_im[tid] = g_amp_im[tid];
        }
        __syncthreads();
    }

    const double* __restrict__ delays = (n_paths <= 256) ? s_delays : g_delays;
    const double* __restrict__ amp_re = (n_paths <= 256) ? s_amp_re : g_amp_re;
    const double* __restrict__ amp_im = (n_paths <= 256) ? s_amp_im : g_amp_im;

    int row_offset = cir_idx * max_samples;
    int use_reg_pdp = (!write_complex && max_samples <= 4096) ? 1 : 0;
    double reg_pdp[16];

    // --- PASS 1: Compute CIR, PDP, and local peak ---
    double local_max_pdp = -1.0;
    int local_peak_idx = 0;

    double local_e = 0.0;
    double local_wt = 0.0;
    double local_wt2 = 0.0;
    double local_mag = 0.0;
    double local_mag2 = 0.0;

    int step_idx = 0;
    for (int s = tid; s < max_samples; s += blockDim.x, ++step_idx) {
        if (s < n_samp) {
            double t = t_start + s * dt;
            double sum_re = 0.0;
            double sum_im = 0.0;
            for (int k = 0; k < n_paths; ++k) {
                double diff = (t - delays[k]) * inv_sigma;
                // delays is sorted ascending:
                // if diff < -8.5 => delays[k] > t + 8.5*sigma, all later k are even further away
                if (diff < -8.5) break;
                if (diff > 8.5) continue;
                double gauss = exp(-0.5 * diff * diff);
                sum_re += gauss * amp_re[k];
                sum_im += gauss * amp_im[k];
            }
            double p = sum_re * sum_re + sum_im * sum_im;
            if (use_reg_pdp) {
                reg_pdp[step_idx] = p;
            } else {
                int out_idx = row_offset + s;
                pdp_out[out_idx] = p;
                if (write_complex) {
                    h_re_out[out_idx] = sum_re;
                    h_im_out[out_idx] = sum_im;
                }
            }
            if (p > local_max_pdp) {
                local_max_pdp = p;
                local_peak_idx = s;
            }
            if (compute_stats) {
                local_e += p;
                local_wt += t * p;
                local_wt2 += (t * t) * p;
                double m = sqrt(p);
                local_mag += m;
                local_mag2 += p;
            }
        } else if (!use_reg_pdp) {
            int out_idx = row_offset + s;
            pdp_out[out_idx] = 0.0;
            if (write_complex) {
                h_re_out[out_idx] = 0.0;
                h_im_out[out_idx] = 0.0;
            }
        }
    }

    s_val[tid] = local_max_pdp;
    s_idx[tid] = local_peak_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            double v_other = s_val[tid + stride];
            int i_other = s_idx[tid + stride];
            double v_cur = s_val[tid];
            int i_cur = s_idx[tid];
            if (v_other > v_cur || (v_other == v_cur && i_other < i_cur)) {
                s_val[tid] = v_other;
                s_idx[tid] = i_other;
            }
        }
        __syncthreads();
    }

    double peak_power = s_val[0];
    int peak_idx = s_idx[0];
    double eff_thresh = threshold_factor * peak_power;

    // --- PASS 2: Find first threshold crossing index ---
    int local_first_idx = n_samp + 1;
    step_idx = 0;
    for (int s = tid; s < n_samp; s += blockDim.x, ++step_idx) {
        double p = use_reg_pdp ? reg_pdp[step_idx] : pdp_out[row_offset + s];
        if (p > eff_thresh) {
            local_first_idx = s;
            break;
        }
    }

    s_idx[tid] = local_first_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_idx[tid + stride] < s_idx[tid]) {
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    int first_crossing = s_idx[0];
    int detected = (first_crossing < n_samp) ? 1 : 0;
    int chosen_idx = detected ? first_crossing : peak_idx;

    double mean_delay = 0.0;
    double rms_spread = 0.0;
    double kurtosis = 0.0;

    if (compute_stats && n_samp > 0) {
        s_val[tid] = local_e;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_val[tid] += s_val[tid + stride];
            __syncthreads();
        }
        if (tid == 0) s_stats[0] = s_val[0];
        __syncthreads();

        s_val[tid] = local_wt;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_val[tid] += s_val[tid + stride];
            __syncthreads();
        }
        if (tid == 0) s_stats[1] = s_val[0];
        __syncthreads();

        s_val[tid] = local_wt2;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_val[tid] += s_val[tid + stride];
            __syncthreads();
        }
        if (tid == 0) s_stats[2] = s_val[0];
        __syncthreads();

        s_val[tid] = local_mag;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_val[tid] += s_val[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            s_stats[3] = s_val[0] / (double)n_samp;
        }
        __syncthreads();

        s_val[tid] = local_mag2;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_val[tid] += s_val[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            double m_mean = s_stats[3];
            double m_var = (s_val[0] / (double)n_samp) - (m_mean * m_mean);
            s_stats[4] = (m_var > 1e-30) ? sqrt(m_var) : 0.0;
        }
        __syncthreads();

        double m_mean = s_stats[3];
        double m_std = s_stats[4];
        double local_kurt = 0.0;
        if (m_std > 0.0) {
            double inv_std = 1.0 / m_std;
            step_idx = 0;
            for (int s = tid; s < n_samp; s += blockDim.x, ++step_idx) {
                double p = use_reg_pdp ? reg_pdp[step_idx] : pdp_out[row_offset + s];
                double z = (sqrt(p) - m_mean) * inv_std;
                double z2 = z * z;
                local_kurt += z2 * z2;
            }
        }
        s_val[tid] = local_kurt;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_val[tid] += s_val[tid + stride];
            __syncthreads();
        }

        if (tid == 0 && s_stats[0] > 0.0) {
            double tot_e = s_stats[0];
            double wt_norm = s_stats[1] / tot_e;
            double wt2_norm = s_stats[2] / tot_e;
            mean_delay = wt_norm - t_start;
            double var_t = wt2_norm - wt_norm * wt_norm;
            rms_spread = (var_t > 0.0) ? sqrt(var_t) : 0.0;
            kurtosis = (m_std > 0.0) ? (s_val[0] / (double)n_samp) : 0.0;
        }
    }

    if (tid == 0) {
        int sum_base = cir_idx * 6;
        summary_out[sum_base + 0] = t_start + chosen_idx * dt;
        summary_out[sum_base + 1] = (double)detected;
        summary_out[sum_base + 2] = (double)chosen_idx;
        summary_out[sum_base + 3] = mean_delay;
        summary_out[sum_base + 4] = rms_spread;
        summary_out[sum_base + 5] = kurtosis;
    }
}
''', 'batch_cir_toa_stats')
        logger.debug("Created fused batch CIR+ToA+Stats CUDA kernel")
    except Exception as e:
        logger.warning(f"Failed to create batch CIR+ToA+Stats kernel: {e}")
        _batch_cir_toa_stats_kernel = None

    return _batch_cir_toa_stats_kernel


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
    """
    n_paths = len(path_delays)
    n_time = len(t_vector)
    
    use_gpu = gpu_manager.should_use_gpu(n_paths * n_time) or (return_on_device and gpu_manager.available)
    xp = get_array_module(use_gpu=use_gpu)
    
    if use_gpu:
        t_vec, delays, amps, phases = to_gpu_batch(
            t_vector, path_delays, path_amplitudes, path_phases)
    else:
        t_vec = np.asarray(t_vector)
        delays = np.asarray(path_delays)
        amps = np.asarray(path_amplitudes)
        phases = np.asarray(path_phases)
    
    sigma = pulse_width / 2.355  # FWHM to sigma
    
    fused_kernel = _get_fused_pulse_kernel() if use_gpu else None
    
    if fused_kernel is not None:
        try:
            dt = t_vec[:, None] - delays[None, :]
            amps_bc = xp.broadcast_to(amps[None, :], dt.shape)
            phases_bc = xp.broadcast_to(phases[None, :], dt.shape)
            
            pulses_complex = fused_kernel(dt, sigma, amps_bc, phases_bc)
            h_t = xp.sum(pulses_complex, axis=1)
            return h_t if return_on_device else to_cpu(h_t)
        except Exception:
            pass
    
    dt = t_vec[:, None] - delays[None, :]
    pulses = xp.exp(-0.5 * (dt / sigma) ** 2)
    complex_amps = amps * xp.exp(1j * phases)
    h_t = xp.dot(pulses, complex_amps)
    
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
    use_gpu = gpu_manager.should_use_gpu(len(distances))
    xp = get_array_module(use_gpu=use_gpu)
    
    if use_gpu:
        d, is_los = to_gpu_batch(
            np.maximum(distances, 0.001).astype(np.float64),
            is_los_array.astype(np.float64)
        )
    else:
        d = np.maximum(np.asarray(distances, dtype=np.float64), 0.001)
        is_los = np.asarray(is_los_array, dtype=np.float64)
    
    kappa = frequency_decay_factor
    freq_term = 20.0 * kappa * float(np.log10(frequency / center_frequency))
    dist_term = 10.0 * path_loss_exponent * xp.log10(d / reference_distance)
    shadowing = xp.random.normal(0, shadow_fading_std, size=len(d))
    nlos_const = (1.0 - is_los) * 10.0
    ref_loss = -reference_loss_db
    
    total_loss_db = ref_loss + freq_term + dist_term + shadowing + nlos_const
    total_loss_cpu = to_cpu(total_loss_db)
    
    breakdown = {
        "ref_loss": float(ref_loss),
        "freq_loss": float(freq_term),
        "dist_loss": to_cpu(dist_term),
        "shadowing": to_cpu(shadowing),
        "nlos_loss": to_cpu(nlos_const),
        "total_loss": total_loss_cpu
    }
    
    return total_loss_cpu, breakdown


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
    """
    n_cirs = len(time_vectors)
    if n_cirs == 0:
        return []
    
    results = []
    for t_vec, h_t in zip(time_vectors, h_t_list):
        t_c = to_cpu(t_vec)
        h_c = to_cpu(h_t)
        pdp = np.abs(h_c) ** 2
        peak_power = np.max(pdp)
        effective_threshold = threshold_factor * peak_power
        
        indices = np.where(pdp > effective_threshold)[0]
        if len(indices) == 0:
            idx = int(np.argmax(pdp))
            results.append((float(t_c[idx]), False))
        else:
            first_idx = int(indices[0])
            results.append((float(t_c[first_idx]), True))
    
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
    Generate ranging errors for N measurements simultaneously.
    """
    use_gpu = gpu_manager.should_use_gpu(n_samples)
    xp = get_array_module(use_gpu=use_gpu)
    
    if use_gpu:
        sigma_crlb, is_los, nf, eb, rds = to_gpu_batch(
            sigma_crlb_array.astype(np.float64),
            is_los_array.astype(np.float64),
            noise_factors.astype(np.float64),
            error_biases.astype(np.float64),
            rms_delay_spreads.astype(np.float64)
        )
    else:
        sigma_crlb = np.asarray(sigma_crlb_array, dtype=np.float64)
        is_los = np.asarray(is_los_array, dtype=np.float64)
        nf = np.asarray(noise_factors, dtype=np.float64)
        eb = np.asarray(error_biases, dtype=np.float64)
        rds = np.asarray(rms_delay_spreads, dtype=np.float64)
    
    total_std = xp.sqrt(sigma_crlb ** 2 + sigma_jitter ** 2)
    
    model_lower = noise_model.lower()
    if model_lower == "gaussian":
        noise = xp.random.normal(0, 1, size=n_samples) * total_std
    elif "non-centralized" in model_lower:
        noise = xp.random.normal(0, 1, size=n_samples) * total_std + total_std * 0.5
    elif model_lower == "laplace":
        b = total_std / float(np.sqrt(2.0))
        noise = xp.random.laplace(0, 1, size=n_samples) * b
    elif model_lower == "uniform":
        w = total_std * float(np.sqrt(3.0))
        noise = xp.random.uniform(-1, 1, size=n_samples) * w
    elif "student" in model_lower:
        df = 4
        scale = total_std / float(np.sqrt(df / (df - 2)))
        noise = xp.random.standard_normal(size=n_samples) * scale
    elif "mixed" in model_lower:
        base = xp.random.normal(0, 1, size=n_samples) * total_std
        heavy = xp.random.normal(0, 1, size=n_samples) * total_std * 3
        mask = xp.random.random(size=n_samples) < 0.9
        noise = xp.where(mask, base, heavy)
    else:
        noise = xp.random.normal(0, 1, size=n_samples) * total_std
    
    nlos_mask = (1.0 - is_los)
    noise = noise * (is_los + nlos_mask * nf)
    
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
    Compute channel statistics for multiple CIRs.
    """
    n_cirs = len(time_vectors)
    if n_cirs == 0:
        return []
    
    results = []
    for t_vec, h_t in zip(time_vectors, h_t_list):
        t_c = to_cpu(t_vec)
        h_c = to_cpu(h_t)
        pdp = np.abs(h_c) ** 2
        total_energy = np.sum(pdp)
        
        if total_energy > 0:
            pdp_norm = pdp / total_energy
            weighted_t = np.sum(t_c * pdp_norm)
            mean_excess_delay = weighted_t - t_c[0]
            second_moment = np.sum((t_c ** 2) * pdp_norm)
            rms_delay_spread = np.sqrt(max(second_moment - weighted_t ** 2, 0.0))
            
            mag = np.sqrt(pdp)
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
# VI. Batch Distance Measurement (Full Pipeline) — SINGLE-LAUNCH FUSED GPU KERNEL
# ─────────────────────────────────────────────────────────────────────────────

_gpu_dummy_buf = None
_gpu_summary_buf = None
_gpu_summary_cap = 0


def batch_measure_distances(
    channel_model,
    true_distances: np.ndarray,
    is_los_array: np.ndarray,
    anchor_positions: Optional[List] = None,
    compute_stats: bool = True,
    include_cir: bool = True
) -> List:
    """
    Perform complete UWB ranging simulation for multiple anchors simultaneously.
    
    Full GPU-accelerated pipeline:
    1. Batch path loss computation
    2. Vectorized link budget
    3. Single-launch fused CUDA kernel for CIR, PDP, Peak, ToA & Channel Stats
    4. Batch ranging error generation
    5. Optional CIR waveform download
    """
    global _gpu_dummy_buf, _gpu_summary_buf, _gpu_summary_cap
    from src.core.uwb.uwb_types import RangingResult
    
    n_anchors = len(true_distances)
    if n_anchors == 0:
        return []

    c = channel_model.c
    B = channel_model.uwb_params.bandwidth
    
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
        bw_mhz = B / 1e6
        tx_power_total_dbm += 10 * np.log10(bw_mhz)
    
    rx_power_dbm = (tx_power_total_dbm +
                    channel_model.uwb_params.tx_antenna_gain_dbi +
                    channel_model.uwb_params.rx_antenna_gain_dbi -
                    path_losses)
    
    rx_power_watts = 10 ** ((rx_power_dbm - 30) / 10)
    noise_power_watts = channel_model.calculate_thermal_noise()
    snr_linear = rx_power_watts / noise_power_watts
    snr_db = 10 * np.log10(np.maximum(snr_linear, 1e-10))
    
    # --- 3. Multipath Parameter Generation & Fused Single-Launch GPU Kernel ---
    pulse_width = 1.0 / B
    sigma_pulse = pulse_width / 2.355
    inv_sigma = 1.0 / sigma_pulse
    th = channel_model.detection_threshold_factor

    fused_cir_kernel = _get_batch_cir_toa_stats_kernel() if gpu_manager.available else None

    if fused_cir_kernel is not None and hasattr(channel_model, 'generate_sv_multipath_params'):
        cp = gpu_manager.cupy
        mp_list = []
        max_paths = 1
        max_samples = 1
        dt_val = 1.0 / (16.0 * B)

        for i in range(n_anchors):
            dist = float(true_distances[i])
            is_los = bool(is_los_array[i])
            anchor_pos = anchor_positions[i] if anchor_positions is not None else None
            delays, amps, phases, t0_true, t_start, dt_val, n_samp = (
                channel_model.generate_sv_multipath_params(
                    distance=dist,
                    is_los=is_los,
                    anchor_pos=anchor_pos,
                    avg_loss_db=float(path_losses[i]),
                )
            )
            n_p = len(delays)
            if n_p > max_paths:
                max_paths = n_p
            if n_samp > max_samples:
                max_samples = n_samp
            mp_list.append((delays, amps, phases, t0_true, t_start, n_p, n_samp))

        # Pack metadata + multipath parameters into ONE contiguous array [3, n_anchors, max_paths + 1]
        stride_p = max_paths + 1
        path_data_cpu = np.zeros((3, n_anchors, stride_p), dtype=np.float64)
        t0_trues = np.empty(n_anchors, dtype=np.float64)

        for i, (delays, amps, phases, t0_true, t_start, n_p, n_samp) in enumerate(mp_list):
            path_data_cpu[0, i, 0] = t_start
            path_data_cpu[1, i, 0] = float(n_p)
            path_data_cpu[2, i, 0] = float(n_samp)
            if n_p > 0:
                path_data_cpu[0, i, 1:1 + n_p] = delays
                path_data_cpu[1, i, 1:1 + n_p] = amps * np.cos(phases)
                path_data_cpu[2, i, 1:1 + n_p] = amps * np.sin(phases)
            t0_trues[i] = t0_true

        # Single H2D transfer for all anchors' multipath data and metadata
        path_data_g = cp.asarray(path_data_cpu)

        if _gpu_summary_buf is None or _gpu_summary_cap < n_anchors:
            _gpu_summary_buf = cp.empty((max(n_anchors, 16), 6), dtype=cp.float64)
            _gpu_summary_cap = max(n_anchors, 16)
        summary_g = _gpu_summary_buf[:n_anchors]

        write_complex = 1 if include_cir else 0
        if include_cir:
            pdp_g = cp.empty((n_anchors, max_samples), dtype=cp.float64)
            h_re_g = cp.empty((n_anchors, max_samples), dtype=cp.float64)
            h_im_g = cp.empty((n_anchors, max_samples), dtype=cp.float64)
        elif max_samples > 4096:
            pdp_g = cp.empty((n_anchors, max_samples), dtype=cp.float64)
            h_re_g = pdp_g
            h_im_g = pdp_g
        else:
            if _gpu_dummy_buf is None:
                _gpu_dummy_buf = cp.empty(1, dtype=cp.float64)
            pdp_g = _gpu_dummy_buf
            h_re_g = _gpu_dummy_buf
            h_im_g = _gpu_dummy_buf

        # Launch 1 block of 256 threads per anchor
        fused_cir_kernel(
            (n_anchors,),
            (256,),
            (
                path_data_g,
                np.float64(dt_val),
                np.float64(inv_sigma),
                np.float64(th),
                pdp_g,
                h_re_g,
                h_im_g,
                summary_g,
                np.int32(n_anchors),
                np.int32(stride_p),
                np.int32(max_samples),
                np.int32(write_complex),
                np.int32(1 if compute_stats else 0),
            )
        )

        # --- 4. Batch Ranging Errors (runs concurrently on CPU while GPU kernel executes) ---
        snr_safe = np.maximum(snr_linear, 0.1)
        sigma_crlb_array = c / (2 * np.pi * B * np.sqrt(2 * snr_safe))
        sigma_jitter = channel_model.uwb_params.fixed_noise_std
        noise_factors = np.full(n_anchors, channel_model.current_noise_factor, dtype=np.float64)
        error_biases = np.full(n_anchors, channel_model.current_error_bias, dtype=np.float64)
        rms_delay_spreads = np.full(n_anchors, channel_model.current_rms_delay_spread, dtype=np.float64)

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

        # Single D2H synchronization for all scalar outputs
        summary_cpu = summary_g.get()

        if include_cir:
            h_re_cpu = h_re_g.get()
            h_im_cpu = h_im_g.get()
            pdp_cpu = pdp_g.get()

        # --- 5. Assemble Results ---
        if 0 < th < 1:
            time_offset = sigma_pulse * np.sqrt(-1 * np.log(th))
            dist_offset = time_offset * c
        else:
            dist_offset = 0.0

        results = []
        for i in range(n_anchors):
            dist = float(true_distances[i])
            is_los = bool(is_los_array[i])
            toa_est_raw = float(summary_cpu[i, 0])
            detected_flag = bool(summary_cpu[i, 1] > 0.5)
            first_idx = int(summary_cpu[i, 2])
            t0_true = float(t0_trues[i])

            mp_bias_observed = (toa_est_raw - t0_true) * c
            noise_err_m = float(errors[i])
            sigma_std = float(total_stds[i])

            measured_dist = (toa_est_raw * c) + noise_err_m + dist_offset
            total_error = measured_dist - dist

            # Filter unrealistic spikes
            if is_los and abs(total_error) > 0.2:
                total_error = 0.2 if total_error > 0 else -0.2
                measured_dist = dist + total_error
            elif not is_los and abs(total_error) > 0.4:
                total_error = 0.4 if total_error > 0 else -0.4
                measured_dist = dist + total_error

            mean_delay = float(summary_cpu[i, 3]) if compute_stats else 0.0
            rms_spread = float(summary_cpu[i, 4]) if compute_stats else 0.0
            kurt_val = float(summary_cpu[i, 5]) if compute_stats else 0.0

            if include_cir:
                n_samp = int(path_data_cpu[2, i, 0])
                t_start = float(path_data_cpu[0, i, 0])
                t_vec_cpu = np.arange(n_samp, dtype=np.float64) * dt_val + t_start
                h_t_cpu = h_re_cpu[i, :n_samp] + 1j * h_im_cpu[i, :n_samp]
                cir_amp_cpu = np.sqrt(pdp_cpu[i, :n_samp])
            else:
                t_vec_cpu = None
                h_t_cpu = None
                cir_amp_cpu = None

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
                cir_amplitude=cir_amp_cpu,
                cir_first_path_index=first_idx,
                cir_complex=h_t_cpu,
                rms_delay_spread=rms_spread,
                mean_excess_delay=mean_delay,
                kurtosis=kurt_val,
                path_loss_breakdown={
                    "ref_loss": pl_breakdowns["ref_loss"],
                    "freq_loss": pl_breakdowns["freq_loss"],
                    "dist_loss": float(pl_breakdowns["dist_loss"][i]),
                    "shadowing": float(pl_breakdowns["shadowing"][i]),
                    "nlos_loss": float(pl_breakdowns["nlos_loss"][i]),
                    "total_loss": float(pl_breakdowns["total_loss"][i])
                }
            ))

        return results

    # --- Fallback Per-anchor CIR generation ---
    time_vectors = []
    h_t_list = []
    t0_trues = []
    keep_on_gpu = gpu_manager.available and n_anchors > 0
    
    for i in range(n_anchors):
        dist = float(true_distances[i])
        is_los = bool(is_los_array[i])
        anchor_pos = anchor_positions[i] if anchor_positions is not None else None

        if hasattr(channel_model, 'generate_unified_cir'):
            try:
                t_vec, h_t, t0_true = channel_model.generate_unified_cir(
                    dist, is_los, return_on_device=keep_on_gpu, anchor_pos=anchor_pos,
                    avg_loss_db=float(path_losses[i]))
            except TypeError:
                t_vec, h_t, t0_true = channel_model.generate_unified_cir(dist, is_los, anchor_pos=anchor_pos)
                if keep_on_gpu and not hasattr(h_t, '__cuda_array_interface__'):
                    t_vec = to_gpu(t_vec)
                    h_t = to_gpu(h_t)
        else:
            t_vec, h_t, t0_true = channel_model.generate_unified_cir(dist, is_los)
            
        time_vectors.append(t_vec)
        h_t_list.append(h_t)
        t0_trues.append(t0_true)
    
    toa_results = batch_toa_detection(
        time_vectors=time_vectors,
        h_t_list=h_t_list,
        snr_linears=snr_linear,
        threshold_factor=channel_model.detection_threshold_factor
    )
    
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
    
    if compute_stats:
        stats_list = batch_channel_statistics(time_vectors, h_t_list)
    else:
        stats_list = [{'mean_excess_delay': 0.0, 'rms_delay_spread': 0.0, 'kurtosis': 0.0}] * n_anchors
    
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

        t_vec_cpu = to_cpu(t_vec) if include_cir or not np.isfinite(toa_est_raw) else None
        h_t_cpu = to_cpu(h_t) if include_cir or not np.isfinite(toa_est_raw) else None

        if not np.isfinite(toa_est_raw) and t_vec_cpu is not None and h_t_cpu is not None:
            peak_idx = int(np.argmax(np.abs(h_t_cpu)))
            toa_est_raw = float(t_vec_cpu[peak_idx])
        
        mp_bias_observed = (toa_est_raw - t0_true) * c
        noise_err_m = float(errors[i])
        sigma_std = float(total_stds[i])
        
        measured_dist = (toa_est_raw * c) + noise_err_m + dist_offset
        total_error = measured_dist - dist
        
        if is_los and abs(total_error) > 0.2:
            total_error = 0.2 if total_error > 0 else -0.2
            measured_dist = dist + total_error
        elif not is_los and abs(total_error) > 0.4:
            total_error = 0.4 if total_error > 0 else -0.4
            measured_dist = dist + total_error
        
        stats = stats_list[i]
        
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
            cir_time_vector=t_vec_cpu if include_cir else None,
            cir_amplitude=np.abs(h_t_cpu) if include_cir and h_t_cpu is not None else None,
            cir_first_path_index=(
                int(np.argmax(t_vec_cpu >= toa_est_raw)) if detected_flag and t_vec_cpu is not None
                else (int(np.argmax(np.abs(h_t_cpu))) if h_t_cpu is not None else 0)
            ),
            cir_complex=h_t_cpu if include_cir else None,
            rms_delay_spread=stats['rms_delay_spread'],
            mean_excess_delay=stats['mean_excess_delay'],
            kurtosis=stats['kurtosis'],
            path_loss_breakdown={
                "ref_loss": pl_breakdowns["ref_loss"],
                "freq_loss": pl_breakdowns["freq_loss"],
                "dist_loss": float(pl_breakdowns["dist_loss"][i]),
                "shadowing": float(pl_breakdowns["shadowing"][i]),
                "nlos_loss": float(pl_breakdowns["nlos_loss"][i]),
                "total_loss": float(pl_breakdowns["total_loss"][i])
            }
        ))
    
    return results
