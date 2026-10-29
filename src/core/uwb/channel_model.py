from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from src.core.uwb.uwb_devices import Position
import time
import logging
from .uwb_types import (
    UWBParameters, PathLossParams, SVModelParams, RangingResult,
    CM1_LOS_0_4M, CM2_NLOS_0_4M, CM3_NLOS_4_10M, CM4_EXTREME_NLOS
)
from .Nlos_zones import NLOSZone, PolygonNLOSZone, MovingNLOSZone
from src.core.parallel.gpu_backend import gpu_manager, get_array_module, to_gpu, to_cpu
from src.core.parallel.cuda_kernels import vectorized_cir_pulse_superposition
from src.core.parallel.geometry_kernels import batch_los_check_gpu

logger = logging.getLogger(__name__)


"""
IEEE 802.15.3a UWB Channel Model Implementation

This module implements a comprehensive mathematical modeling of UWB propagation for distance estimation,
unifying continuous frequency response, geometric path loss, log-normal shadowing, and multipath delay effects.

Key Features:
- Unified UWB Channel Impulse Response (CIR)
- Frequency-Dependent Amplitude Model
- Frequency-Dependent Path Loss & Log-Normal Shadowing
- Ranging and Distance Estimation Model
- Multipath Bias & NLOS Blockage Modeling
"""

class UWBChannelModel:
    """IEEE 802.15.3a compliant UWB channel model with S-V multipath and frequency dependence."""
    
    def __init__(self):
        # System parameters
        self.uwb_params = UWBParameters()
        self.temperature_k = 290  # Kelvin
        self.k_boltzmann = 1.38e-23
        self.c = 299792458  # Speed of light
        
        # S-V model parameters (IEEE 802.15.3a)
        self.los_sv_params = CM1_LOS_0_4M
        self.nlos_sv_params = CM2_NLOS_0_4M
        self.current_sv_params = self.los_sv_params
        
        # Maintain some legacy attributes for compatibility if needed
        self.cluster_decay = self.los_sv_params.cluster_decay
        self.ray_decay = self.los_sv_params.ray_decay
        
        # Path loss parameters
        self.los_path_loss_params = PathLossParams(
            path_loss_exponent=self.los_sv_params.path_loss_exponent,
            shadow_fading_std=self.los_sv_params.shadow_fading_std,
            frequency_decay_factor=1.0  # Kappa for frequency dependence
        )
        self.current_path_loss_params = self.los_path_loss_params
        
        # Channel state
        self.is_los = True
        self.nlos_zones = []
        self.moving_nlos_zones = []
        
        # NLOS error parameters (updated dynamically based on zones)
        self.current_noise_factor = 1.0
        self.current_error_bias = 0.0
        self.current_rms_delay_spread = 5e-9  # Default 5ns for LOS
        
        # Detection threshold
        self.detection_threshold_factor = 0.1
        
        # Cached thermal noise (invalidated when params change)
        self._cached_thermal_noise = None
        self._cached_noise_params_hash = None
        
        # Environmental state for cluster generation
        self._current_anchor_pos = None
        
        # Channel frequency map (channel number -> {center_freq, bandwidth})
        self.channel_map = {
            1: {"center_freq": 3494.4e6, "bandwidth": 499.2e6},
            2: {"center_freq": 3993.6e6, "bandwidth": 499.2e6},
            3: {"center_freq": 4492.8e6, "bandwidth": 499.2e6},
            4: {"center_freq": 3993.6e6, "bandwidth": 1331.2e6}, # Channel 4 is wide
            5: {"center_freq": 6489.6e6, "bandwidth": 499.2e6},
            9: {"center_freq": 7987.2e6, "bandwidth": 499.2e6},
            10: {"center_freq": 8486.4e6, "bandwidth": 499.2e6},
        }

    # Alias for backward compatibility if needed
    ChannelConditions = "UWBChannelModel" 

    def update_uwb_parameters(self, uwb_params: UWBParameters):
        """Update UWB system parameters."""
        self.uwb_params = uwb_params
        self._cached_thermal_noise = None  # Invalidate cache

    def update_path_loss_parameters(self, path_loss_params: PathLossParams):
        """Update path loss parameters for both LOS baseline and current active params."""
        self.los_path_loss_params = path_loss_params
        # Also update current active params so changes take effect immediately
        self.current_path_loss_params = path_loss_params

    def set_uwb_channel(self, channel: int):
        """Set UWB channel center frequency and bandwidth."""
        if channel in self.channel_map:
            params = self.channel_map[channel]
            self.uwb_params.center_frequency = params["center_freq"]
            self.uwb_params.bandwidth = params["bandwidth"]
        else:
            raise ValueError(f"Invalid UWB channel {channel}")

    def calculate_thermal_noise(self) -> float:
        """Calculate thermal noise power in Watts: P_n = k * T * B * F. (cached)"""
        params_hash = (self.uwb_params.noise_figure_db, self.uwb_params.bandwidth, self.temperature_k)
        if self._cached_thermal_noise is not None and self._cached_noise_params_hash == params_hash:
            return self._cached_thermal_noise
        
        noise_dimless = 10**(self.uwb_params.noise_figure_db/10)
        result = (self.k_boltzmann * self.temperature_k * 
                self.uwb_params.bandwidth * noise_dimless)
        self._cached_thermal_noise = result
        self._cached_noise_params_hash = params_hash
        return result

    # -------------------------------------------------------------------------
    # I. Unified UWB Channel Impulse Response (CIR) & Prop Loss
    # -------------------------------------------------------------------------

    def calculate_path_loss_and_shadowing(self, distance: float, frequency: float, is_los: bool) -> Tuple[float, dict]:
        """
        Calculate the total path loss L(f, d) including frequency dependence and shadowing.
        
        PL(f, d) = PL_0 + 20*kappa*log10(f/f_c) + 10*n*log10(d/d_0) + X_sigma
        
        Returns:
            Tuple[float, dict]: Total Path Loss in dB and a dictionary of components.
        """
        params = self.current_path_loss_params
        f_c = self.uwb_params.center_frequency
        d_0 = params.reference_distance
        # Avoid log(0)
        d = max(distance, 0.001)
        
        # 1. Frequency Dependence
        # kappa is typically 1.0 for free space, varies indoors
        kappa = params.frequency_decay_factor
        freq_term = 20 * kappa * np.log10(frequency / f_c)
        
        # 2. Distance Dependence (Geometric Path Loss)
        # n is the path loss exponent
        n = params.path_loss_exponent
        dist_term = 10 * n * np.log10(d / d_0)
        
        # 3. Log-Normal Shadowing
        # X_sigma ~ N(0, sigma^2)
        shadowing = np.random.normal(0, params.shadow_fading_std)
        
        # 4. NLOS extra attenuation (if applicable, simple constant often used, 
        # but here it's implicitly handled by 'n' and 'sigma' from the zone params)
        # We can add an explicit bias if modeled that way, but usually zone params cover it.
        # Adding explicit constant for significant blockages:
        nlos_const = 0.0 if is_los else 10.0 # 10dB additional loss for NLOS as a baseline
        
        # Reference loss at d_0 (usually ~ -43 dB, so we flip sign for 'Loss')
        # PL_0 is given as -43dB (gain), so Loss_0 is +43dB
        ref_loss = -params.reference_loss_db
        
        total_loss_db = ref_loss + freq_term + dist_term + shadowing + nlos_const
        
        breakdown = {
            "ref_loss": ref_loss,
            "freq_loss": freq_term,
            "dist_loss": dist_term,
            "shadowing": shadowing,
            "nlos_loss": nlos_const,
            "total_loss": total_loss_db
        }
        
        return total_loss_db, breakdown

    def generate_sv_multipath_params(
        self,
        distance: float,
        is_los: bool,
        anchor_pos: Optional[Position] = None,
        avg_loss_db: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, int]:
        """
        Generate S-V multipath component delays, amplitudes, phases, and time-grid metadata
        without materializing the full time vector on CPU.

        Returns:
            Tuple of (path_delays, path_amplitudes, path_phases, t0, t_start, dt, n_samples)
        """
        params = self.current_sv_params
        Lambda = params.cluster_arrival_rate
        lambda_ray = params.ray_arrival_rate
        Gamma = params.cluster_decay
        gamma = params.ray_decay
        
        t0 = distance / self.c  # Direct path delay
        
        # 1. Generate Cluster Arrival Times (Poisson Process) — VECTORIZED
        base_lambda_clusters = 5.0
        
        # Global contribution: more obstacles in the room -> more background scattering
        total_obstacles = len(self.nlos_zones) + len(self.moving_nlos_zones)
        global_boost = total_obstacles * 0.4
        
        # Local contribution: obstacles near the anchor -> more early reflections/clusters
        local_boost = 0.0
        if total_obstacles > 0:
            pos_for_local = anchor_pos or self._current_anchor_pos
            if pos_for_local:
                nearby_count = self._count_nearby_obstacles(pos_for_local, radius=4.0)
                local_boost = nearby_count * 1.5
            
        mean_clusters = base_lambda_clusters + global_boost + local_boost
        
        # Also increase the cluster arrival rate (Lambda) when many obstacles exist
        if total_obstacles > 0 or local_boost > 0:
            env_factor = 1.0 + 0.05 * (total_obstacles + local_boost)
            Lambda *= env_factor
            Gamma *= np.sqrt(env_factor) 

        n_clusters = max(1, np.random.poisson(mean_clusters))
        if n_clusters > 1:
            cluster_intervals = np.random.exponential(1/Lambda, size=n_clusters - 1)
            T = np.empty(n_clusters)
            T[0] = t0
            np.cumsum(cluster_intervals, out=T[1:])
            T[1:] += t0
        else:
            T = np.array([t0])
        
        # 2. Generate All Ray Parameters — VECTORIZED
        n_rays_per_cluster = np.maximum(1, np.random.poisson(5, size=n_clusters))
        total_rays = int(np.sum(n_rays_per_cluster))
        
        # Pre-calc reference amplitude (use caller-provided value to avoid duplicate computation)
        if avg_loss_db is None:
            avg_loss_db, _ = self.calculate_path_loss_and_shadowing(
                distance, self.uwb_params.center_frequency, is_los)
        avg_amp = 10 ** (-avg_loss_db / 20)
        
        # Generate all ray inter-arrival intervals at once
        all_ray_intervals = np.random.exponential(1/lambda_ray, size=total_rays)
        all_fading = np.random.rayleigh(scale=1.0, size=total_rays)
        all_phases = np.random.uniform(0, 2*np.pi, size=total_rays)
        
        # Expand cluster onset times to per-ray arrays via np.repeat
        T_per_ray = np.repeat(T, n_rays_per_cluster)
        
        # Cluster decay scale per ray
        cluster_scales = np.exp(-(T_per_ray - t0) / Gamma)
        
        # Per-cluster cumulative ray delays
        all_ray_intervals_copy = all_ray_intervals.copy()
        cluster_starts = np.zeros(n_clusters, dtype=np.intp)
        np.cumsum(n_rays_per_cluster[:-1], out=cluster_starts[1:])
        all_ray_intervals_copy[cluster_starts] = 0.0
        
        global_cumsum = np.cumsum(all_ray_intervals_copy)
        cluster_baselines = np.repeat(
            np.concatenate(([0.0], global_cumsum[cluster_starts[1:] - 1])) if n_clusters > 1
            else np.array([0.0]),
            n_rays_per_cluster
        )
        tau_rays = global_cumsum - cluster_baselines
        
        # Absolute path delays
        path_delays = T_per_ray + tau_rays
        
        # Ray amplitude decay within each cluster
        ray_scales = np.exp(-tau_rays / gamma)
        power_scales = cluster_scales * ray_scales
        path_amplitudes = avg_amp * np.sqrt(power_scales) * all_fading
        path_phases = all_phases
        
        # Sort by delay
        sort_idx = np.argsort(path_delays)
        path_delays = path_delays[sort_idx]
        path_amplitudes = path_amplitudes[sort_idx]
        path_phases = path_phases[sort_idx]
        
        # Time grid parameters
        oversample_factor = 16
        dt = 1.0 / (oversample_factor * self.uwb_params.bandwidth)
        if len(path_delays) > 0:
            duration = float(path_delays[-1] - path_delays[0] + 50e-9)
            t_start = float(path_delays[0] - 5e-9)
        else:
            duration = 100e-9
            t_start = float(t0 - 5e-9)
        n_samples = max(1, int(np.ceil(duration / dt)))
        
        return path_delays, path_amplitudes, path_phases, float(t0), t_start, float(dt), n_samples

    def generate_unified_cir(self, distance: float, is_los: bool, return_on_device: bool = False, anchor_pos: Optional[Position] = None, avg_loss_db: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate the Unified UWB Channel Impulse Response h(t, f).
        
        Combines S-V multipath structure with frequency-dependent amplitude.
        Uses vectorized NumPy operations for multipath parameter generation
        (replaces nested Python for-loops) and GPU-accelerated pulse superposition.
        
        h(t) = Sum_k Sum_l alpha_{k,l} * exp(j*phi_{k,l}) * delta(t - T_k - tau_{k,l})
        
        Returns:
            times: Time vector
            h_t: Complex impulse response
            first_path_delay: Delay of the first arriving path
        """
        path_delays, path_amplitudes, path_phases, t0, t_start, dt, n_samples = (
            self.generate_sv_multipath_params(
                distance=distance,
                is_los=is_los,
                anchor_pos=anchor_pos,
                avg_loss_db=avg_loss_db,
            )
        )
        
        t_vector = np.arange(n_samples, dtype=np.float64) * dt + t_start
        
        # Pulse width
        pulse_width = 1 / self.uwb_params.bandwidth
        
        # GPU-accelerated vectorized pulse superposition
        h_t = vectorized_cir_pulse_superposition(
            t_vector=t_vector,
            path_delays=path_delays,
            path_amplitudes=path_amplitudes,
            path_phases=path_phases,
            pulse_width=pulse_width,
            return_on_device=return_on_device
        )
        
        if return_on_device and gpu_manager.available:
            t_vector = to_gpu(t_vector)
            
        return t_vector, h_t, t0
    
    def generate_unified_cir_batch(self, distances: np.ndarray, is_los_array: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate CIRs for multiple distances simultaneously.
        
        Each CIR has random multipath structure so generation is per-anchor,
        but this method provides a convenient batch interface.
        
        Args:
            distances: Array of distances (N,)
            is_los_array: Boolean LOS conditions (N,)
        
        Returns:
            List of (time_vector, h_t, t0_true) tuples
        """
        results = []
        for dist, is_los in zip(distances, is_los_array):
            results.append(self.generate_unified_cir(float(dist), bool(is_los)))
        return results

    # -------------------------------------------------------------------------
    # II. Ranging & Distance Estimation
    # -------------------------------------------------------------------------

    def detect_toa(self, time_vector: np.ndarray, h_t: np.ndarray, snr_linear: float) -> Tuple[float, bool]:
        """
        Detect Time of Arrival (ToA) from CIR.
        Uses GPU array module for accelerated PDP computation.
        
        Method:
        1. Calculate Power Delay Profile (PDP) = |h(t)|^2
        2. Set dynamic threshold relative to peak
        3. Find first crossing.
        """
        use_gpu = gpu_manager.should_use_gpu(len(time_vector))
        xp = get_array_module(use_gpu=use_gpu)
        
        if use_gpu:
            t_g, h_g = to_gpu(time_vector), to_gpu(h_t)
        else:
            t_g, h_g = np.asarray(time_vector), np.asarray(h_t)
        
        pdp = xp.abs(h_g)**2
        peak_power = xp.max(pdp)
        effective_threshold = self.detection_threshold_factor * peak_power
        
        indices = xp.where(pdp > effective_threshold)[0]
        
        if len(indices) == 0:
             idx = int(xp.argmax(pdp))
             return float(to_cpu(t_g[idx])), False
        
        first_idx = int(indices[0])
        estimated_toa = float(to_cpu(t_g[first_idx]))
        return estimated_toa, True

    def calculate_ranging_errors(self, true_distance: float, snr_linear: float, is_los: bool) -> float:
        """
        Calculate stochastic ranging errors (Noise + NLOS Bias).
        
        Logic:
        - Noise Error (Jitter): ~ CRLB
        - NLOS Bias: Extra delay distributions
        """
        # 1. Noise Error (CRLB based)
        # sigma_noise = c / (2*pi*B * sqrt(2*SNR))
        B = self.uwb_params.bandwidth
        # Clamp SNR to avoid div zero
        snr_safe = max(snr_linear, 0.1)
        sigma_crlb = self.c / (2 * np.pi * B * np.sqrt(2 * snr_safe))
        
        # Combine CRLB noise with hardware implementation noise
        # so the noise model operates on a meaningful magnitude
        total_std = np.sqrt(sigma_crlb**2 + self.uwb_params.fixed_noise_std**2)
        
        noise_error = self._generate_noise(total_std, self.noise_model)
        if not is_los:
             # NLOS can have higher variance/noise
             noise_error *= self.current_noise_factor
        
        # 2. Multipath/NLOS Bias
        # If NLOS, we add a positive bias due to obstruction/diffraction
        bias = 0.0
        if not is_los:
            # Explicit zone bias
            bias += self.current_error_bias
            # Statistical bias (RMS delay spread related)
            rho = 0.1 
            bias += self.c * self.current_rms_delay_spread * rho * np.random.exponential(1.0)
            
        return noise_error + bias, total_std

    def measure_distance_detailed(self, true_distance: float, is_los: bool = True, anchor_pos: Optional[Position] = None, compute_stats: bool = True) -> RangingResult:
        """
        Perform complete UWB ranging simulation.
        
        Steps:
        1. Physics: Path Loss -> Received Power -> SNR
        2. Channel: Generate Multipath CIR
        3. Detection: Estimate ToA
        4. Error Modeling: Apply stochastic errors
        """
        
        # --- 1. Link Budget ---
        # Calculate Average Signal Power at Receiver
        path_loss_db, pl_breakdown = self.calculate_path_loss_and_shadowing(
            distance=true_distance, 
            frequency=self.uwb_params.center_frequency, 
            is_los=is_los
        )
        
        # Adjust Tx Power: If value is small (<-30dBm), assume it represents PSD (-41.3 dBm/MHz)
        # and convert to Total Power.
        tx_power_total_dbm = self.uwb_params.tx_power_dbm
        if tx_power_total_dbm < -30:
            bw_mhz = self.uwb_params.bandwidth / 1e6
            tx_power_total_dbm += 10 * np.log10(bw_mhz)
        
        rx_power_dbm = (tx_power_total_dbm + 
                        self.uwb_params.tx_antenna_gain_dbi + 
                        self.uwb_params.rx_antenna_gain_dbi - 
                        path_loss_db) # Note: path_loss_db is positive loss
        
        rx_power_watts = 10**((rx_power_dbm - 30)/10)
        noise_power_watts = self.calculate_thermal_noise()
        snr_linear = rx_power_watts / noise_power_watts
        snr_db = 10 * np.log10(snr_linear)
        
        # --- 2. Channel Impulse Response ---
        # Ensure intermediate arrays stay on the GPU if GPU acceleration is active
        t_vec, h_t, t0_true = self.generate_unified_cir(
            true_distance, is_los, return_on_device=gpu_manager.available, anchor_pos=anchor_pos, avg_loss_db=path_loss_db
        )
        
        # --- 3. Detection ---
        toa_est_raw, detected_flag = self.detect_toa(t_vec, h_t, snr_linear)
        
        mp_bias_observed = (toa_est_raw - t0_true) * self.c
        
        # --- 4. Add Receiver Noise Errors ---
        noise_err_m, sigma_std = self.calculate_ranging_errors(true_distance, snr_linear, is_los)
        
        # Final Distance
        pulse_width = 1.0 / self.uwb_params.bandwidth
        sigma = pulse_width / 2.355
        th = self.detection_threshold_factor
        if th < 1.0 and th > 0.0:
            time_offset = sigma * np.sqrt(-1 * np.log(th))
            dist_offset = time_offset * self.c
        else:
            dist_offset = 0.0
            
        measured_dist = (toa_est_raw * self.c) + noise_err_m + dist_offset
        
        # Breakdown
        total_error = measured_dist - true_distance
        
        # Filter unrealistic spikes
        if is_los and abs(total_error) > 0.6:
            total_error = 0.6 if total_error > 0 else -0.6
            measured_dist = true_distance + total_error
        elif not is_los and abs(total_error) > 0.8:
            total_error = 0.8 if total_error > 0 else -0.8
            measured_dist = true_distance + total_error
        
        # --- 5. Channel Statistics (GPU-accelerated, only when requested) ---
        mean_excess_delay = 0.0
        rms_delay_spread = 0.0
        kurtosis = 0.0
        
        if compute_stats:
            use_gpu = gpu_manager.should_use_gpu(len(h_t))
            xp = get_array_module(use_gpu=use_gpu)
            
            if use_gpu:
                t_g, h_g = to_gpu(t_vec), to_gpu(h_t)
            else:
                t_g, h_g = np.asarray(t_vec), np.asarray(h_t)
            
            pdp = xp.abs(h_g)**2
            total_energy = xp.sum(pdp)
            
            if float(to_cpu(total_energy)) > 0:
                pdp_norm = pdp / total_energy
                weighted_t = xp.sum(t_g * pdp_norm)
                mean_excess_delay = float(to_cpu(weighted_t - t_g[0]))
                second_moment = xp.sum((t_g**2) * pdp_norm)
                rms_delay_spread = float(to_cpu(xp.sqrt(xp.maximum(second_moment - weighted_t**2, 0.0))))
                
                mag = xp.abs(h_g)
                mag_mean = xp.mean(mag)
                mag_std = xp.std(mag)
                if float(to_cpu(mag_std)) > 0:
                     kurtosis = float(to_cpu(xp.mean(((mag - mag_mean) / mag_std) ** 4)))
                else:
                     kurtosis = 0.0

        # Transfer GPU arrays to CPU arrays right before constructing the final RangingResult object
        t_vec_cpu = to_cpu(t_vec)
        h_t_cpu = to_cpu(h_t)

        return RangingResult(
            measured_distance=measured_dist,
            true_distance=true_distance,
            toa_estimate=measured_dist/self.c,
            toa_true=true_distance/self.c,
            noise_error=noise_err_m,
            multipath_bias=mp_bias_observed,
            nlos_error=0.0,
            total_error=total_error,
            snr_db=snr_db,
            snr_linear=snr_linear,
            path_loss_db=path_loss_db,
            received_power_dbm=rx_power_dbm,
            is_los=is_los,
            first_path_detected=detected_flag,
            measurement_std=sigma_std,
            cir_time_vector=t_vec_cpu,
            cir_amplitude=np.abs(h_t_cpu),
            cir_first_path_index=np.argmax(t_vec_cpu >= toa_est_raw) if detected_flag else np.argmax(np.abs(h_t_cpu)),
            cir_complex=h_t_cpu,
            rms_delay_spread=rms_delay_spread,
            mean_excess_delay=mean_excess_delay,
            kurtosis=kurtosis,
            path_loss_breakdown=pl_breakdown
        )
        
    def measure_distance(self, true_distance: float, is_los: bool = True, anchor_pos: Optional[Position] = None, compute_stats: bool = True) -> Tuple[float, float]:
        """Simple interface for backward compatibility."""
        res = self.measure_distance_detailed(true_distance, is_los, anchor_pos=anchor_pos, compute_stats=compute_stats)
        return res.measured_distance, res.measurement_std

    def measure_distance_batch(
        self,
        true_distances: np.ndarray,
        is_los_array: np.ndarray,
        anchor_positions: Optional[List[Position]] = None,
        compute_stats: bool = True,
        include_cir: bool = True
    ) -> List[RangingResult]:
        """
        Batch UWB ranging simulation for multiple anchors.
        Uses GPU acceleration when available for path loss and CIR generation.
        
        Args:
            true_distances: Array of true distances (N,)
            is_los_array: Boolean LOS conditions (N,)
            anchor_positions: Optional list of anchor Positions
            compute_stats: Whether to compute channel statistics (kurtosis, delay spread)
            include_cir: Whether to transfer full CIR waveforms to CPU
        
        Returns:
            List of RangingResult objects
        """
        from src.core.parallel.cuda_kernels import batch_measure_distances
        return batch_measure_distances(
            self,
            true_distances,
            is_los_array,
            anchor_positions=anchor_positions,
            compute_stats=compute_stats,
            include_cir=include_cir
        )

    # -------------------------------------------------------------------------
    # III. Environment & Update Logic
    # -------------------------------------------------------------------------
    
    def _count_nearby_obstacles(self, pos: Position, radius: float = 3.0) -> int:
        """Count obstacles within a certain radius of a given position."""
        count = 0
        for zone in self.nlos_zones + self.moving_nlos_zones:
            # Get center point of zone
            if isinstance(zone, NLOSZone):
                cx, cy = (zone.x1 + zone.x2) / 2, (zone.y1 + zone.y2) / 2
            elif isinstance(zone, MovingNLOSZone):
                cx, cy = zone.current_pos
            elif isinstance(zone, PolygonNLOSZone):
                pts = np.array(zone.points)
                cx, cy = np.mean(pts, axis=0)
            else:
                continue
            
            dist = np.sqrt((pos.x - cx)**2 + (pos.y - cy)**2)
            if dist <= radius:
                count += 1
        return count

    def update_los_condition(self, anchor_pos: Position, tag_pos: Position):
        """
        Check Line of Sight and update channel parameters (Zones).
        """
        current_time = time.time()
        # Update moving zones
        for zone in self.moving_nlos_zones:
            zone.update_position(current_time)
            
        # Default: LOS
        self.is_los = True
        self.current_path_loss_params = self.los_path_loss_params
        self.current_sv_params = self.los_sv_params
        self.current_noise_factor = 1.0
        self.current_error_bias = 0.0
        
        # Check Intersections
        active_zone = None
        
        # Static Zones
        for zone in self.nlos_zones:
            if self._line_intersects_zone(anchor_pos, tag_pos, zone):
                active_zone = zone
                break
                
        # Moving Zones (override static if hit)
        for zone in self.moving_nlos_zones:
            if self._line_intersects_zone(anchor_pos, tag_pos, zone):
                active_zone = zone
                break
        
        # Save anchor position for environment-aware cluster generation
        self._current_anchor_pos = anchor_pos

        if active_zone:
            self.is_los = False
            # Update Params from Zone
            if hasattr(active_zone, 'path_loss_params'):
                self.current_path_loss_params = active_zone.path_loss_params
            else:
                self.current_path_loss_params.path_loss_exponent = 3.5 # Generically higher
            
            if hasattr(active_zone, 'noise_factor'):
                self.current_noise_factor = active_zone.noise_factor
            
            if hasattr(active_zone, 'error_bias'):
                self.current_error_bias = active_zone.error_bias
                
            # Update S-V params if zone specifies CM2/CM3 etc.
            # Simplified: Use CM2 if NLOS default
            self.current_sv_params = self.nlos_sv_params

    # -------------------------------------------------------------------------
    # Helpers: Geometry & Noise Simplification
    # -------------------------------------------------------------------------

    def _line_intersects_zone(self, tx: Position, rx: Position, zone) -> bool:
        """Helper to dispatch intersection checks."""
        # Check specific zone types
        if isinstance(zone, NLOSZone):
            return self._line_intersects_rect(tx.x, tx.y, rx.x, rx.y, zone.x1, zone.y1, zone.x2, zone.y2)
        elif isinstance(zone, MovingNLOSZone) or isinstance(zone, PolygonNLOSZone):
            points = zone.get_points() if hasattr(zone, 'get_points') else zone.points
            return self._line_intersects_poly(tx.x, tx.y, rx.x, rx.y, points)
        return False

    def _line_intersects_rect(self, x1, y1, x2, y2, rx1, ry1, rx2, ry2):
        lines = [((rx1,ry1), (rx2,ry1)), ((rx2,ry1), (rx2,ry2)), ((rx2,ry2), (rx1,ry2)), ((rx1,ry2), (rx1,ry1))]
        for p1, p2 in lines:
            if self._segment_intersect((x1,y1), (x2,y2), p1, p2): return True
        return False

    def _line_intersects_poly(self, x1, y1, x2, y2, points):
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i+1)%len(points)]
            if self._segment_intersect((x1,y1), (x2,y2), p1, p2): return True
        return False

    def _segment_intersect(self, A, B, C, D):
        def ccw(p1, p2, p3):
            return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    # Simplified Noise Interface (Cleaned up from previous version)
    def _generate_noise(self, base_noise: float, model: str = "gaussian") -> float:
        """Generate noise sample (Simplified)."""
        model = model.lower()
        if model == "gaussian":
            return np.random.normal(0, base_noise)
        elif "non-centralized" in model:
             return np.random.normal(base_noise * 0.5, base_noise)
        elif model == "uniform":
             w = base_noise * np.sqrt(3)
             return np.random.uniform(-w, w)
        elif model == "laplace":
             b = base_noise / np.sqrt(2)
             return np.random.laplace(0, b)
        elif "mixed" in model:
             if np.random.random() < 0.9:
                 return np.random.normal(0, base_noise)
             else:
                 return np.random.normal(0, base_noise * 3)
        elif "student" in model:
             df = 4
             scale = base_noise / np.sqrt(df / (df - 2))
             return np.random.standard_t(df) * scale
        return np.random.normal(0, base_noise)
        
    def set_noise_model(self, model: str):
        """Set the statistical noise model."""
        self._noise_model = model.lower()

    @property
    def noise_model(self) -> str:
        return getattr(self, '_noise_model', "gaussian")

    def add_nlos_zone(self, x1, y1, x2, y2):
        self.nlos_zones.append(NLOSZone(x1, y1, x2, y2))
        
    def add_moving_nlos_zone(self, *args, **kwargs):
        self.moving_nlos_zones.append(MovingNLOSZone(*args, **kwargs))

    def check_los_condition(self, tx_pos: Position, rx_pos: Position) -> bool:
        """Check if path between transmitter and receiver is LOS"""
        for zone in self.nlos_zones:
            if self._line_intersects_zone(tx_pos, rx_pos, zone):
                return False

        for zone in self.moving_nlos_zones:
            if self._line_intersects_zone(tx_pos, rx_pos, zone):
                return False
                
        return True

    def check_los_to_anchor(self, anchor_pos: Position, tag_pos: Position) -> bool:
        """Alias for check_los_condition for backward compatibility"""
        return self.check_los_condition(anchor_pos, tag_pos)

    def batch_update_los_conditions(
        self, anchor_positions_array: np.ndarray, tag_pos: Position
    ) -> np.ndarray:
        """
        GPU-accelerated batch LOS check for all anchors simultaneously.

        Uses a CUDA kernel (via geometry_kernels.py) to test all
        anchor-tag line segments against all zone edges in parallel.

        Args:
            anchor_positions_array: (N_anchors, 2) array of [x, y]
            tag_pos: Tag Position object

        Returns:
            Boolean NumPy array (N_anchors,): True = LOS, False = NLOS
        """
        import time as _time
        if self.moving_nlos_zones:
            current_time = _time.time()
            for zone in self.moving_nlos_zones:
                zone.update_position(current_time)

        return batch_los_check_gpu(
            anchor_positions=anchor_positions_array,
            tag_position=(tag_pos.x, tag_pos.y),
            nlos_zones=self.nlos_zones,
            moving_nlos_zones=self.moving_nlos_zones
        )

    # -------------------------------------------------------------------------
    # IV. Backward Compatibility & Helper Methods
    # -------------------------------------------------------------------------

    def get_received_signal_quality(self, distance: float) -> Tuple[float, float]:
        """
        Estimate Signal Quality (0-1) and SNR (linear) for a given distance.
        Used by channel_adapter.py
        """
        path_loss_db, _ = self.calculate_path_loss_and_shadowing(
            distance, 
            self.uwb_params.center_frequency, 
            self.is_los
        )
        
        tx_power_total_dbm = self.uwb_params.tx_power_dbm
        if tx_power_total_dbm < -30:
            bw_mhz = self.uwb_params.bandwidth / 1e6
            tx_power_total_dbm += 10 * np.log10(bw_mhz)
            
        rx_power_dbm = (tx_power_total_dbm + 
                        self.uwb_params.tx_antenna_gain_dbi + 
                        self.uwb_params.rx_antenna_gain_dbi - 
                        path_loss_db)
        
        rx_power_watts = 10**((rx_power_dbm - 30)/10)
        noise_power_watts = self.calculate_thermal_noise()
        snr_linear = rx_power_watts / noise_power_watts
        
        snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -100
        quality = np.clip((snr_db + 10) / 30.0, 0.0, 1.0)
        
        return quality, snr_linear

    def calculate_path_loss(self, distance: float, is_los: bool) -> float:
        """
        Calculate linear path loss (amplitude attenuation).
        Used by channel_adapter.py
        """
        pl_db, _ = self.calculate_path_loss_and_shadowing(
            distance, 
            self.uwb_params.center_frequency, 
            is_los
        )
        return 10 ** (-pl_db / 20.0)

    @property
    def n_paths(self) -> int:
        return 10 # Dummy value for compatibility

# Alias for backward compatibility
ChannelConditions = UWBChannelModel
