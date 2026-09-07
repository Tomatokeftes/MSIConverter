"""MSIConverter - Convert Mass Spectrometry Imaging data to SpatialData/Zarr format.

This package provides tools for converting MSI data from various formats
(ImzML, Bruker) into the modern SpatialData/Zarr format with automatic
pixel size detection.
"""

# Suppress known warnings from dependencies
import warnings
from typing import Any

# Configure Dask to use new query planning (silences legacy DataFrame warning)
try:
    import dask

    dask.config.set({"dataframe.query-planning": True})
except ImportError:
    pass

# Import readers and converters to trigger registration
from . import converters  # This triggers converter registrations  # noqa: F401
from . import readers  # This triggers reader registrations  # noqa: F401
from .convert import convert_msi
from .preview import MsiPreview, preview_msi  # noqa: F401

# Suppress remaining dependency warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
warnings.filterwarnings("ignore", category=FutureWarning, module="spatialdata")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numba")
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="The legacy Dask DataFrame implementation is deprecated",
    category=FutureWarning,
)

__version__ = "3.18.0"

# Import key components - avoid wildcard imports
try:
    from .converters.spatialdata.converter import SpatialDataConverter
except ImportError:
    # SpatialData dependencies not available
    SpatialDataConverter: Any = None  # type: ignore[no-redef]

# Expose main API
__all__ = [
    "__version__",
    "convert_msi",
    "preview_msi",
    "MsiPreview",
    "SpatialDataConverter",
]
