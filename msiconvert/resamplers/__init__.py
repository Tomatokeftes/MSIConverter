# msiconvert/resampling_module/__init__.py
"""
Modular resampling system for mass spectrometry data.

This module provides non-linear resampling strategies for different Time-of-Flight
(TOF) instrument types, with a focus on maintaining consistent bin resolution
at a reference m/z value.

Example usage:
    from msiconvert.resampling_module import ResamplingRequest, ResamplingService, StrategyFactory
    
    # Create request with bin size specification
    request = ResamplingRequest(
        min_mz=100.0,
        max_mz=2000.0,
        model_type='linear',
        bin_size_mu=5.0,  # 5 milli-u
        reference_mz=1000.0
    )
    
    # Create service with appropriate strategy
    strategy = StrategyFactory.create_strategy(request.model_type)
    service = ResamplingService(strategy)
    
    # Generate bin edges
    result = service.generate_resampled_axis(request)
"""

from .application.models import ResamplingRequest
from .application.factory import StrategyFactory
from .domain.models import ResamplingResult
from .services.resampling_service import ResamplingService
from .infrastructure.config import ResamplingConfig
from .exceptions import (
    ResamplingError,
    InvalidParametersError,
    ResamplingLimitExceededError,
    UnknownStrategyError
)

__all__ = [
    # Main classes
    'ResamplingRequest',
    'ResamplingResult',
    'ResamplingService',
    'StrategyFactory',
    'ResamplingConfig',
    
    # Exceptions
    'ResamplingError',
    'InvalidParametersError',
    'ResamplingLimitExceededError',
    'UnknownStrategyError',
]

__version__ = '1.0.0'