"""
Channel Data Adapter
Extracts channel/EM data from ChannelConditions.
"""
import numpy as np
from typing import TYPE_CHECKING

from src.api.collectors.data_collector import ChannelLinkData

if TYPE_CHECKING:
    from src.core.uwb.channel_model import ChannelConditions
    from src.core.uwb.uwb_devices import Anchor, Tag


class ChannelDataAdapter:
    """
    Adapter for extracting channel and electromagnetic data from ChannelConditions.
    Designed for AI training data collection.
    """
    
    def extract_link_data(self, 
                          channel_conditions: 'ChannelConditions',
                          anchor: 'Anchor', 
                          tag: 'Tag') -> ChannelLinkData:
        """
        Extract all channel data for a single tag-anchor link.
        
        Args:
            channel_conditions: ChannelConditions object
            anchor: Anchor object
            tag: Tag object
            
        Returns:
            ChannelLinkData with all channel parameters
        """
        # Calculate true distance
        distance = anchor.position.distance_to(tag.position)
        
        # Check LOS condition
        is_los = channel_conditions.check_los_to_anchor(anchor.position, tag.position)
        
        # Get signal quality and SNR
        signal_quality, snr_linear = channel_conditions.get_received_signal_quality(distance)
        snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf
        
        # Calculate path loss
        path_loss_linear = channel_conditions.calculate_path_loss(distance, is_los)
        path_loss_db = -20 * np.log10(path_loss_linear) if path_loss_linear > 0 else np.inf
        
        # Get noise information
        noise_std = self._estimate_noise_std(channel_conditions, distance, snr_linear)
        
        # Get channel model parameters
        uwb_params = channel_conditions.uwb_params
        path_loss_params = channel_conditions.current_path_loss_params
        
        return ChannelLinkData(
            anchor_id=anchor.id,
            is_los=is_los,
            snr_db=float(snr_db),
            snr_linear=float(snr_linear),
            path_loss_db=float(path_loss_db),
            noise_std=float(noise_std),
            signal_quality=float(signal_quality),
            channel_model="saleh_valenzuela",
            center_frequency_hz=uwb_params.center_frequency,
            bandwidth_hz=uwb_params.bandwidth,
            path_loss_exponent=path_loss_params.path_loss_exponent,
            shadow_fading_std=path_loss_params.shadow_fading_std,
            noise_model=channel_conditions.noise_model
        )
    
    def _estimate_noise_std(self, 
                            channel_conditions: 'ChannelConditions',
                            distance: float,
                            snr_linear: float) -> float:
        """
        Estimate the noise standard deviation for a measurement.
        
        Uses the Cramer-Rao Lower Bound (CRLB) formula from the channel model.
        """
        c = channel_conditions.c
        bandwidth = channel_conditions.uwb_params.bandwidth
        
        # Calculate CRLB-based noise
        if snr_linear > 0:
            crlb = (c**2) / (8 * np.pi**2 * bandwidth**2 * snr_linear)
            base_noise = np.sqrt(crlb)
        else:
            base_noise = 1.0  # Default fallback
        
        # Apply signal quality degradation
        signal_quality, _ = channel_conditions.get_received_signal_quality(distance)
        base_noise *= (1 + (1 - signal_quality) * 2)
        
        # Apply NLOS factor if applicable
        if not channel_conditions.is_los:
            base_noise *= channel_conditions.current_noise_factor
            
        return base_noise
    
    def get_snr(self, 
                channel_conditions: 'ChannelConditions', 
                distance: float) -> tuple:
        """
        Get SNR for a given distance.
        
        Returns:
            Tuple of (snr_db, snr_linear)
        """
        _, snr_linear = channel_conditions.get_received_signal_quality(distance)
        snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf
        return float(snr_db), float(snr_linear)
    
    def get_path_loss(self,
                      channel_conditions: 'ChannelConditions',
                      distance: float,
                      is_los: bool) -> float:
        """
        Get path loss in dB for a given distance.
        """
        path_loss_linear = channel_conditions.calculate_path_loss(distance, is_los)
        path_loss_db = -20 * np.log10(path_loss_linear) if path_loss_linear > 0 else np.inf
        return float(path_loss_db)
    
    def get_channel_model_params(self, 
                                  channel_conditions: 'ChannelConditions') -> dict:
        """
        Get all channel model parameters.
        
        Returns:
            Dictionary with all channel model parameters
        """
        uwb_params = channel_conditions.uwb_params
        path_loss_params = channel_conditions.current_path_loss_params
        
        return {
            # UWB parameters
            'center_frequency_hz': uwb_params.center_frequency,
            'bandwidth_hz': uwb_params.bandwidth,
            'tx_power_dbm': uwb_params.tx_power_dbm,
            'tx_antenna_gain_dbi': uwb_params.tx_antenna_gain_dbi,
            'rx_antenna_gain_dbi': uwb_params.rx_antenna_gain_dbi,
            'noise_figure_db': uwb_params.noise_figure_db,
            
            # Path loss parameters
            'path_loss_exponent': path_loss_params.path_loss_exponent,
            'reference_distance': path_loss_params.reference_distance,
            'reference_loss_db': path_loss_params.reference_loss_db,
            'shadow_fading_std': path_loss_params.shadow_fading_std,
            
            # S-V model parameters
            'cluster_decay': channel_conditions.cluster_decay,
            'ray_decay': channel_conditions.ray_decay,
            'n_paths': channel_conditions.n_paths,
            
            # Current state
            'is_los': channel_conditions.is_los,
            'noise_model': channel_conditions.noise_model,
            'noise_factor': channel_conditions.current_noise_factor,
            'error_bias': channel_conditions.current_error_bias
        }
