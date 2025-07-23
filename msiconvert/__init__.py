"""
MSIConverter - Convert Mass Spectrometry Imaging data to SpatialData/Zarr format.

This package provides tools for converting MSI data from various formats (ImzML, Bruker)
into the modern SpatialData/Zarr format with automatic pixel size detection.
"""

__version__ = "1.8.2"

# Configure Dask and suppress warnings early
import warnings
import os

# Set Dask configuration via environment variable (takes effect before import)
os.environ.setdefault("DASK_DATAFRAME__QUERY_PLANNING", "True")
# Disable Dask logging spam
os.environ.setdefault("DASK_LOGGING__DISTRIBUTED", "error")
os.environ.setdefault("DASK_DISTRIBUTED__ADMIN__LOG_LEVEL", "error")
os.environ.setdefault("DASK_DISTRIBUTED__WORKER__LOG_LEVEL", "error")

# Suppress all the annoying warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=".*dask.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*xarray_schema.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*paramiko.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*distributed.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*cryptography.*")

# Silence distributed computing logs
import logging
# Disable duplicate INFO logging from distributed packages
logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)
logging.getLogger("distributed.worker").setLevel(logging.ERROR)
logging.getLogger("distributed.nanny").setLevel(logging.ERROR)
logging.getLogger("distributed.core").setLevel(logging.ERROR)
logging.getLogger("distributed.http").setLevel(logging.ERROR)

# Import readers to trigger format detector registration
from . import readers  # This triggers the format detector registrations
from .convert import convert_msi

# Import key components - avoid wildcard imports
try:
    from .converters.spatialdata_converter import SpatialDataConverter
except ImportError:
    # SpatialData dependencies not available
    SpatialDataConverter = None

# Expose main API
__all__ = [
    "__version__",
    "convert_msi",
    "SpatialDataConverter",
]
