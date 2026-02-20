"""
AI Training API for UWB Indoor Positioning Simulator

This module provides a simple, direct API for collecting training data
from the UWB simulator for AI/ML applications.

Main Entry Point:
    TrainingDataAPI - Simple facade for all data collection operations

Example Usage:
    from src.api import TrainingDataAPI
    
    api = TrainingDataAPI()
    api.select_data(channel=True, filter_outputs=True, ground_truth=True)
    api.enable_collection()
    
    # ... run simulation ...
    
    api.export_to_file("training_data.npz")
"""

from src.api.training_api import TrainingDataAPI
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
    GeometryDataAdapter
)
from src.api.export import DataExporter

__all__ = [
    # Main API
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
    
    # Export
    'DataExporter'
]
