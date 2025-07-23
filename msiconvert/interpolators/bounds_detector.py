"""
Bounds detection service for interpolation planning
"""

from typing import TYPE_CHECKING
from ..metadata.base_extractor import BaseMetadataExtractor
from ..metadata.bruker_extractor import BrukerMetadataExtractor
from ..metadata.imzml_extractor import ImzMLMetadataExtractor
from . import BoundsInfo
from pathlib import Path

if TYPE_CHECKING:
    from ..core.base_reader import BaseMSIReader

def detect_bounds_from_reader(reader: 'BaseMSIReader') -> BoundsInfo:
    """
    Detect dataset bounds from reader metadata without scanning all spectra.
    
    Args:
        reader: MSI reader instance
        
    Returns:
        BoundsInfo with detected bounds
        
    Raises:
        ValueError: If bounds cannot be determined from metadata
    """
    import logging
    logging.info("[DEBUG] detect_bounds_from_reader started")
    
    # Try to get bounds from reader if it has the new methods
    if hasattr(reader, 'get_mass_bounds') and hasattr(reader, 'get_spatial_bounds'):
        try:
            logging.info("[DEBUG] Getting mass bounds from reader...")
            min_mz, max_mz = reader.get_mass_bounds()
            logging.info(f"[DEBUG] Mass bounds: {min_mz} - {max_mz}")
            
            logging.info("[DEBUG] Getting spatial bounds from reader...")
            spatial_bounds = reader.get_spatial_bounds()
            logging.info(f"[DEBUG] Spatial bounds: {spatial_bounds}")
            
            # Get dimensions to calculate n_spectra
            logging.info("[DEBUG] Getting dimensions from reader...")
            dimensions = reader.get_dimensions()
            n_spectra = dimensions[0] * dimensions[1] * dimensions[2]
            logging.info(f"[DEBUG] Dimensions: {dimensions}, n_spectra: {n_spectra}")
            
            return BoundsInfo(
                min_mz=min_mz,
                max_mz=max_mz,
                min_x=spatial_bounds['min_x'],
                max_x=spatial_bounds['max_x'],
                min_y=spatial_bounds['min_y'],
                max_y=spatial_bounds['max_y'],
                n_spectra=n_spectra
            )
        except Exception as e:
            # Fall back to metadata extraction
            logging.info(f"[DEBUG] Failed to get bounds from reader: {e}")
            pass
    
    # Fall back to direct metadata extraction
    logging.info("[DEBUG] Falling back to detect_bounds_from_path")
    return detect_bounds_from_path(reader.data_path)

def detect_bounds_from_path(data_path: Path) -> BoundsInfo:
    """
    Detect bounds directly from data path using appropriate metadata extractor.
    
    Args:
        data_path: Path to MSI data
        
    Returns:
        BoundsInfo with detected bounds
        
    Raises:
        ValueError: If data format is not supported or bounds cannot be determined
    """
    data_path = Path(data_path)
    
    # Determine format and create appropriate extractor
    if data_path.is_dir() and any(f.suffix in ['.tsf', '.tdf'] for f in data_path.glob('analysis.*')):
        # Bruker format
        extractor = BrukerMetadataExtractor(data_path)
    elif data_path.suffix.lower() == '.imzml':
        # ImzML format
        extractor = ImzMLMetadataExtractor(data_path)
    else:
        raise ValueError(f"Unsupported data format or path: {data_path}")
    
    # Extract bounds
    min_mz, max_mz = extractor.extract_mass_bounds()
    spatial_bounds = extractor.extract_spatial_bounds()
    
    # Calculate n_spectra from spatial bounds
    n_spectra = ((spatial_bounds['max_x'] - spatial_bounds['min_x'] + 1) * 
                 (spatial_bounds['max_y'] - spatial_bounds['min_y'] + 1))
    
    return BoundsInfo(
        min_mz=min_mz,
        max_mz=max_mz,
        min_x=spatial_bounds['min_x'],
        max_x=spatial_bounds['max_x'],
        min_y=spatial_bounds['min_y'],
        max_y=spatial_bounds['max_y'],
        n_spectra=n_spectra
    )

def validate_bounds(bounds: BoundsInfo) -> bool:
    """
    Validate that bounds are reasonable for interpolation.
    
    Args:
        bounds: BoundsInfo to validate
        
    Returns:
        True if bounds are valid, False otherwise
    """
    # Check mass range
    if bounds.min_mz <= 0 or bounds.max_mz <= bounds.min_mz:
        return False
        
    # Check spatial bounds
    if (bounds.min_x > bounds.max_x or 
        bounds.min_y > bounds.max_y or
        bounds.n_spectra <= 0):
        return False
        
    # Check reasonable ranges
    if bounds.max_mz - bounds.min_mz < 1.0:  # Less than 1 Da range
        return False
        
    return True