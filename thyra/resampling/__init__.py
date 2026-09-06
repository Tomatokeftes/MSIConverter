"""Mass axis resampling module for MSI data.

This module provides functionality for resampling mass spectrometry
imaging data to common mass axes, enabling consistent analysis across
pixels and datasets.
"""

from .common_axis import CommonAxisBuilder
from .constants import BinaryDataType, ImzMLAccessions, SpectrumType, Thresholds
from .data_characteristics import DataCharacteristics
from .decision_tree import ResamplingDecisionTree
from .instrument_detectors import (
    CentroidImzMLDetector,
    DefaultDetector,
    FTICRDetector,
    InstrumentDetector,
    InstrumentDetectorChain,
    OrbitrapDetector,
    RapiflexDetector,
    TimsTOFDetector,
)
from .mobility_grid import (
    MOBILITY_CHANNELS,
    MobilityGrid,
    build_mobility_grid,
    linear_channel,
)
from .strategies import NearestNeighborStrategy, TICPreservingStrategy
from .types import AxisType, MassAxis, ResamplingConfig, ResamplingMethod

__all__ = [
    # Decision tree and strategies
    "ResamplingDecisionTree",
    "CommonAxisBuilder",
    "NearestNeighborStrategy",
    "TICPreservingStrategy",
    # Types
    "ResamplingMethod",
    "AxisType",
    "MassAxis",
    "ResamplingConfig",
    # Constants
    "ImzMLAccessions",
    "SpectrumType",
    "BinaryDataType",
    "Thresholds",
    # Data characteristics
    "DataCharacteristics",
    # Instrument detectors (Strategy pattern)
    "InstrumentDetector",
    "InstrumentDetectorChain",
    "CentroidImzMLDetector",
    "RapiflexDetector",
    "TimsTOFDetector",
    "FTICRDetector",
    "OrbitrapDetector",
    "DefaultDetector",
    # Ion mobility grid
    "MOBILITY_CHANNELS",
    "MobilityGrid",
    "build_mobility_grid",
    "linear_channel",
]
