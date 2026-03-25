"""
AI Training API for UWB Indoor Positioning Simulator

This module provides APIs for AI/ML training workflows
within the UWB simulator.

Main Entry Points:
    AITrainingAPI     - Scoped GET/SET facade for AI training configuration
    TrainingDataAPI   - Data collection and export for offline training

Example Usage:
    from src.api import AITrainingAPI

    api = AITrainingAPI(sim_context)
    print(api.get_algorithm_info())
    api.set_input_mode("uwb")
"""

from src.api.training_api import TrainingDataAPI
from src.api.ai_training_facade import AITrainingAPI
from src.api.collectors import (
    DataCollector,
    DataSample,
    ChannelLinkData,
    FilterOutput,
    DataBuffer
)
from src.api.adapters import (
    ChannelDataAdapter,
    FilterDataAdapter,
    GeometryDataAdapter,
    EnergyDataAdapter,
)
from src.api.export import DataExporter

__all__ = [
    # Main APIs
    'AITrainingAPI',
    'TrainingDataAPI',
    
    # Data structures
    'DataCollector',
    'DataSample',
    'ChannelLinkData',
    'FilterOutput',
    'DataBuffer',
    
    # Adapters
    'ChannelDataAdapter',
    'FilterDataAdapter',
    'GeometryDataAdapter',
    'EnergyDataAdapter',
    
    # Export
    'DataExporter'
]

