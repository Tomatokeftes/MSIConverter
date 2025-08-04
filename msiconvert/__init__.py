"""
MSIConverter - Convert Mass Spectrometry Imaging data to SpatialData/Zarr format.

This package provides tools for converting MSI data from various formats (ImzML, Bruker)
into the modern SpatialData/Zarr format with automatic pixel size detection.
"""

# Suppress known warnings from dependencies BEFORE any imports that trigger them
import warnings
import os

# Set environment variable to configure Dask before any imports
# This prevents the legacy DataFrame deprecation warning
os.environ.setdefault("DASK_DATAFRAME__QUERY_PLANNING", "True")

# Configure comprehensive warning suppression before importing dependencies
warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
warnings.filterwarnings("ignore", category=FutureWarning, module="spatialdata")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numba")
# xarray_schema uses deprecated pkg_resources API - suppressed until they update
warnings.filterwarnings("ignore", category=UserWarning, module="xarray_schema")
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=UserWarning
)
# SpatialData uses deprecated functools.partial behavior - suppressed until they update
warnings.filterwarnings(
    "ignore", 
    message="functools.partial will be a method descriptor in future Python versions",
    category=FutureWarning
)
# Numba deprecation warning about nopython parameter
warnings.filterwarnings(
    "ignore",
    message="nopython is set for njit and is ignored",
    category=RuntimeWarning
)

# Import readers and converters to trigger registration
from . import converters  # This triggers converter registrations  # noqa: F401
from . import readers  # This triggers reader registrations  # noqa: F401
from .convert import convert_msi

__version__ = "1.8.3"

# Import key components - avoid wildcard imports
try:
    from .converters.spatialdata.converter import SpatialDataConverter
except ImportError:
    # SpatialData dependencies not available
    SpatialDataConverter = None

# Expose main API
__all__ = [
    "__version__",
    "convert_msi",
    "SpatialDataConverter",
]
