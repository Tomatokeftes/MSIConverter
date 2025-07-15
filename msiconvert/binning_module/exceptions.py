# msiconvert/binning_module/exceptions.py
"""Custom exceptions for the binning module."""


class BinningError(Exception):
    """Base exception for binning module."""
    pass


class InvalidParametersError(BinningError):
    """Invalid input parameters."""
    pass


class BinningLimitExceededError(BinningError):
    """Exceeded system limits."""
    pass


class UnknownStrategyError(BinningError):
    """Unknown instrument type."""
    pass