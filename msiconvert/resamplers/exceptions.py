# msiconvert/resamplers/exceptions.py
"""Custom exceptions for the resampling module."""


class ResamplingError(Exception):
    """Base exception for resampling module."""
    pass


class InvalidParametersError(ResamplingError):
    """Invalid input parameters."""
    pass


class ResamplingLimitExceededError(ResamplingError):
    """Exceeded system limits."""
    pass


class UnknownStrategyError(ResamplingError):
    """Unknown instrument type."""
    pass