# msiconvert/processing/__init__.py

"""
Processing utilities for MSI data conversion.

This package contains modules for handling various data processing tasks
during MSI data conversion, including interpolation, chunking, and streaming.
"""

from .interpolation import InterpolationResult, SpectralInterpolator

__all__ = ["SpectralInterpolator", "InterpolationResult"]
