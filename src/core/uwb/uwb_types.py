from dataclasses import dataclass

from typing import List, Tuple, Optional
import numpy as np

@dataclass
class UWBParameters:
    """UWB system parameters following report specifications"""
    # Center frequency (Hz), typical UWB: 3.1-10.6 GHz
    center_frequency: float = 6.5e9
    # Bandwidth (Hz), typical UWB: ≥500 MHz per report
    bandwidth: float = 500e6
    # Transmit power (dBm), following FCC regulations
    tx_power_dbm: float = -41.3
    # Antenna gains (dBi)
    tx_antenna_gain_dbi: float = 0.0
    rx_antenna_gain_dbi: float = 0.0
    # System noise figure (dB)
    noise_figure_db: float = 6.0
    # Reference path loss at 1m (dB) - from report
    reference_loss_db: float = -43.0
    # Timing resolution (ps)
    timing_jitter: float = 10e-12  # 10 ps
    # Fixed hardware implementation noise std (m) - floor error
    fixed_noise_std: float = 0.1  # 100mm default per user request
    
    @property
    def wavelength(self) -> float:
        """Calculate wavelength in meters"""
        return 299792458 / self.center_frequency
    
    @property
    def time_resolution(self) -> float:
        """Calculate time resolution based on bandwidth"""
        return 1 / self.bandwidth
    
    @property
    def range_resolution(self) -> float:
        """Calculate range resolution based on bandwidth"""
        return 299792458 / (2 * self.bandwidth)

"""
Path Loss Model Parameters

Implements the path loss model from report:
PL(d) = PL_0 + 10n*log10(d/d_0) + X_σ where:
- n: Path loss exponent (1.8 LOS, 3.2 NLOS)
- d_0: Reference distance (1m)
- PL_0: Reference path loss (-43 dB)
- X_σ: Shadow fading (σ = 3dB LOS, 6dB NLOS)
"""
@dataclass
class PathLossParams:
    """Path loss parameters based on the UWB channel model report"""
    # Path loss exponent (n=1.8 for LOS, n=3.2 for NLOS per report)
    path_loss_exponent: float = 2.0
    # Reference distance (m)
    reference_distance: float = 1.0
    # Reference path loss (dB)
    reference_loss_db: float = -43.0
    # Shadow fading std (3dB for LOS, 6dB for NLOS per report)
    shadow_fading_std: float = 2.0
    # Frequency decay factor (κ): 1.0 for free space, varies for indoor
    frequency_decay_factor: float = 1.0


@dataclass
class SVModelParams:
    """
    Saleh-Valenzuela model parameters per IEEE 802.15.3a.
    
    The S-V model describes multipath arrivals in clusters, where:
    - Clusters arrive according to Poisson process with rate Λ
    - Rays within clusters arrive with rate λ
    - Power decays exponentially with constants Γ (cluster) and γ (ray)
    """
    # Cluster arrival rate Λ (1/s) - Poisson rate for cluster arrivals
    cluster_arrival_rate: float = 0.0233e9  # ~23.3 MHz for CM1
    # Ray arrival rate λ (1/s) - Poisson rate for ray arrivals within clusters
    ray_arrival_rate: float = 2.5e9  # 2.5 GHz for CM1
    # Cluster power decay constant Γ (seconds)
    cluster_decay: float = 7.1e-9  # 7.1 ns for CM1
    # Ray power decay constant γ (seconds)
    ray_decay: float = 4.3e-9  # 4.3 ns for CM1
    # Path loss exponent n
    path_loss_exponent: float = 1.8
    # Shadow fading standard deviation σ_s (dB)
    shadow_fading_std: float = 3.0
    # RMS delay spread τ_rms (seconds) - for NLOS error calculation
    rms_delay_spread: float = 15e-9  # 15 ns typical indoor


@dataclass 
class EnvironmentConfig:
    """
    Environment configuration for IEEE 802.15.3a channel models (CM1-CM4).
    
    - CM1: LOS, 0-4m residential
    - CM2: NLOS, 0-4m residential  
    - CM3: NLOS, 4-10m indoor
    - CM4: Extreme NLOS, through walls
    """
    name: str
    los_params: SVModelParams
    nlos_params: SVModelParams
    max_distance: float = 10.0


# IEEE 802.15.3a Pre-defined Channel Model Configurations
CM1_LOS_0_4M = SVModelParams(
    cluster_arrival_rate=0.0233e9,
    ray_arrival_rate=2.5e9,
    cluster_decay=7.1e-9,
    ray_decay=4.3e-9,
    path_loss_exponent=1.8,
    shadow_fading_std=3.0,
    rms_delay_spread=5e-9
)

CM2_NLOS_0_4M = SVModelParams(
    cluster_arrival_rate=0.4e9,
    ray_arrival_rate=0.5e9,
    cluster_decay=5.5e-9,
    ray_decay=6.7e-9,
    path_loss_exponent=3.5,
    shadow_fading_std=3.0,
    rms_delay_spread=10e-9
)

CM3_NLOS_4_10M = SVModelParams(
    cluster_arrival_rate=0.0667e9,
    ray_arrival_rate=2.1e9,
    cluster_decay=14.0e-9,
    ray_decay=7.9e-9,
    path_loss_exponent=4.5,
    shadow_fading_std=3.0,
    rms_delay_spread=20e-9
)

CM4_EXTREME_NLOS = SVModelParams(
    cluster_arrival_rate=0.0667e9,
    ray_arrival_rate=2.1e9,
    cluster_decay=24.0e-9,
    ray_decay=12.0e-9,
    path_loss_exponent=6.0,
    shadow_fading_std=3.0,
    rms_delay_spread=25e-9
)


@dataclass
class RangingResult:
    """
    Complete ranging measurement result with all error components.
    
    Provides detailed breakdown of ranging errors for analysis and
    algorithm development.
    """
    # Primary outputs
    measured_distance: float  # Final estimated distance (m)
    true_distance: float      # Actual geometric distance (m)
    
    # Time domain
    toa_estimate: float       # Estimated Time of Arrival (s)
    toa_true: float           # True Time of Arrival (s)
    
    # Error components (all in meters)
    noise_error: float        # CRLB-based noise error
    multipath_bias: float     # Bias from multipath interference
    nlos_error: float         # Additional error from NLOS blockage
    total_error: float        # measured_distance - true_distance
    
    # Channel metrics
    snr_db: float             # Signal-to-Noise Ratio (dB)
    snr_linear: float         # SNR in linear scale
    path_loss_db: float       # Total path loss (dB)
    received_power_dbm: float # Received signal power (dBm)
    
    # Condition flags
    is_los: bool              # True if Line-of-Sight exists
    first_path_detected: bool # True if direct path was detected
    
    # Standard deviation of measurement
    measurement_std: float    # Expected std of measurement (m)
    
    # Optional CIR Data (for visualization)
    cir_time_vector: Optional[np.ndarray] = None
    cir_amplitude: Optional[np.ndarray] = None
    cir_first_path_index: Optional[int] = None
    
    # Advanced Channel Metrics
    cir_complex: Optional[np.ndarray] = None  # Complex CIR
    rms_delay_spread: Optional[float] = None  # RMS delay spread (s)
    mean_excess_delay: Optional[float] = None # Mean excess delay (s)
    kurtosis: Optional[float] = None          # Kurtosis of magnitude profile
    path_loss_breakdown: Optional[dict] = None # Detailed PL components
