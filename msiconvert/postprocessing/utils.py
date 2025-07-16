# msiconvert/postprocessing/utils.py
"""Utility functions for post-processing operations."""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def estimate_chunk_size(
    n_pixels: int, 
    n_mz: int, 
    available_memory_gb: float = 4.0,
    safety_factor: float = 0.5
) -> int:
    """
    Estimate optimal chunk size based on available memory.
    
    Parameters
    ----------
    n_pixels : int
        Total number of pixels
    n_mz : int
        Number of m/z values
    available_memory_gb : float
        Available memory in GB (default: 4.0)
    safety_factor : float
        Safety factor to avoid using all memory (default: 0.5)
        
    Returns
    -------
    int
        Recommended chunk size (number of pixels)
    """
    # Estimate memory per pixel (assuming float32 storage)
    bytes_per_pixel = n_mz * 4  # 4 bytes per float32
    
    # Convert available memory to bytes
    available_bytes = available_memory_gb * 1e9 * safety_factor
    
    # Calculate chunk size
    chunk_size = int(available_bytes / bytes_per_pixel)
    
    # Ensure reasonable bounds
    chunk_size = max(100, min(chunk_size, 10000))
    
    logger.info(f"Estimated chunk size: {chunk_size} pixels")
    return chunk_size


def interpolate_spectrum_safe(
    old_mz: np.ndarray,
    old_intensities: np.ndarray,
    new_mz: np.ndarray,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Safely interpolate a spectrum to new m/z values.
    
    Handles edge cases like empty spectra, out-of-bounds values, etc.
    
    Parameters
    ----------
    old_mz : np.ndarray
        Original m/z values
    old_intensities : np.ndarray
        Original intensity values
    new_mz : np.ndarray
        New m/z values to interpolate to
    fill_value : float
        Value to use for out-of-bounds m/z values
        
    Returns
    -------
    np.ndarray
        Interpolated intensities
    """
    if len(old_intensities) == 0:
        return np.full(len(new_mz), fill_value)
    
    if len(old_intensities) == 1:
        # Single point - can't interpolate
        if np.any(np.isclose(new_mz, old_mz[0])):
            result = np.full(len(new_mz), fill_value)
            close_idx = np.argmin(np.abs(new_mz - old_mz[0]))
            result[close_idx] = old_intensities[0]
            return result
        else:
            return np.full(len(new_mz), fill_value)
    
    # Use numpy's interp with bounds handling
    return np.interp(new_mz, old_mz, old_intensities, left=fill_value, right=fill_value)


def validate_zarr_structure(store) -> Tuple[bool, Optional[str]]:
    """
    Validate that a zarr store has the expected structure for MSI data.
    
    Parameters
    ----------
    store : zarr.Group
        Opened zarr store
        
    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)
    """
    try:
        if 'tables' not in store:
            return False, "No 'tables' group found in zarr store"
        
        tables = store['tables']
        msi_tables = [key for key in tables.keys() if "average" not in key.lower()]
        
        if not msi_tables:
            return False, "No MSI data tables found"
        
        # Check first table structure
        first_table = tables[msi_tables[0]]
        
        required_groups = ['X', 'var', 'obs']
        for group in required_groups:
            if group not in first_table:
                return False, f"Missing required group '{group}' in table"
        
        # Check for m/z values
        var_group = first_table['var']
        has_mz = False
        
        if 'mz' in var_group:
            has_mz = True
        elif '_index' in var_group:
            # Try to parse index as m/z values
            try:
                index_values = var_group['_index'][0:10]  # Sample first 10
                float(index_values[0])  # Try to convert to float
                has_mz = True
            except:
                pass
        
        if not has_mz:
            return False, "No m/z values found in var group"
        
        return True, None
        
    except Exception as e:
        return False, f"Error validating structure: {str(e)}"


def get_memory_usage_gb() -> float:
    """Get current memory usage in GB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1e9