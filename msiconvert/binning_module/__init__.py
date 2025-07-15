# msiconvert/binning_module/__init__.py
"""
Modular binning system for mass spectrometry data.

This module provides non-linear binning strategies for different Time-of-Flight
(TOF) instrument types, with a focus on maintaining consistent bin resolution
at a reference m/z value.

Example usage:
    from msiconvert.binning_module import BinningRequest, BinningService, StrategyFactory
    
    # Create request with bin size specification
    request = BinningRequest(
        min_mz=100.0,
        max_mz=2000.0,
        model_type='linear',
        bin_size_mu=5.0,  # 5 milli-Daltons
        reference_mz=1000.0
    )
    
    # Create service with appropriate strategy
    strategy = StrategyFactory.create_strategy(request.model_type)
    service = BinningService(strategy)
    
    # Generate bin edges
    result = service.generate_binned_axis(request)
"""

from .application.models import BinningRequest
from .application.factory import StrategyFactory
from .domain.models import BinningResult
from .services.binning_service import BinningService
from .infrastructure.config import BinningConfig
from .exceptions import (
    BinningError,
    InvalidParametersError,
    BinningLimitExceededError,
    UnknownStrategyError
)

__all__ = [
    # Main classes
    'BinningRequest',
    'BinningResult',
    'BinningService',
    'StrategyFactory',
    'BinningConfig',
    
    # Exceptions
    'BinningError',
    'InvalidParametersError',
    'BinningLimitExceededError',
    'UnknownStrategyError',
]

__version__ = '1.0.0'