# msiconvert/core/resampled_reader.py
"""Wrapper for readers that applies resampling to the data."""

import numpy as np
from typing import Dict, Any, Tuple, Generator, Optional
from numpy.typing import NDArray
import logging

from .base_reader import BaseMSIReader
from ..resamplers import ResamplingRequest, ResamplingService, StrategyFactory

logger = logging.getLogger(__name__)


class ResampledReader(BaseMSIReader):
    """
    Wrapper that applies resampling to any MSI reader.
    
    This wrapper intercepts the common mass axis and spectra,
    applying resampling and interpolation to reduce data size.
    """
    
    def __init__(self, reader: BaseMSIReader, resampling_params: Dict[str, Any]):
        """
        Initialize resampled reader wrapper.
        
        Parameters
        ----------
        reader : BaseMSIReader
            The underlying reader to wrap
        resampling_params : Dict[str, Any]
            Resampling parameters including mode, size, etc.
        """
        self.reader = reader
        self.resampling_params = resampling_params
        self._resampled_mass_axis: Optional[NDArray[np.float64]] = None
        self._resampling_result = None
        self._bin_centers: Optional[NDArray[np.float64]] = None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata including resampling information."""
        metadata = self.reader.get_metadata()
        
        # Add resampling information
        metadata['resampling'] = {
            'enabled': True,
            'mode': self.resampling_params['mode'],
            'parameters': self.resampling_params,
            'timestamp': str(np.datetime64('now'))
        }
        
        if self._resampling_result:
            metadata['resampling']['final_num_bins'] = self._resampling_result.final_num_bins
            metadata['resampling']['achieved_width_mda'] = (
                self._resampling_result.achieved_width_at_ref_mz_da * 1000
            )
        
        return metadata
    
    def get_dimensions(self) -> Tuple[int, int, int]:
        """Return dimensions from underlying reader."""
        return self.reader.get_dimensions()
    
    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """
        Return resampled common mass axis.
        
        This creates bin centers from the bin edges for the new mass axis.
        """
        if self._resampled_mass_axis is None:
            # Get original mass axis to determine range
            original_axis = self.reader.get_common_mass_axis()
            
            # Determine m/z range
            min_mz = float(np.min(original_axis))
            max_mz = float(np.max(original_axis))

            logger.info(f"Creating resampled mass axis for range [{min_mz:.2f}, {max_mz:.2f}]")
            
            # Create resampling request
            request = ResamplingRequest(
                min_mz=min_mz,
                max_mz=max_mz,
                model_type=self.resampling_params['mode'],
                num_bins=self.resampling_params.get('num_bins'),
                bin_size_mu=self.resampling_params.get('bin_size_mu'),
                reference_mz=self.resampling_params.get('reference_mz', 1000.0)
            )
            
            # Generate bins
            strategy = StrategyFactory.create_strategy(request.model_type)
            service = ResamplingService(strategy)
            self._resampling_result = service.generate_resampled_axis(request)
            
            # Calculate bin centers as new mass axis
            edges = self._resampling_result.bin_edges
            self._bin_centers = (edges[:-1] + edges[1:]) / 2.0
            self._resampled_mass_axis = self._bin_centers
            
            logger.info(
                f"Created resampled mass axis with {len(self._resampled_mass_axis)} bins "
                f"(reduced from {len(original_axis)} points)"
            )
        
        return self._resampled_mass_axis
    
    def iter_spectra(
        self, 
        batch_size: Optional[int] = None
    ) -> Generator[Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]], None, None]:
        """
        Iterate through spectra, applying resampling via interpolation.
        
        For each spectrum, interpolates intensities to bin centers.
        """
        # Ensure resampled mass axis is created
        resampled_mzs = self.get_common_mass_axis()
        bin_edges = self._resampling_result.bin_edges
        
        # Iterate through original spectra
        for coords, mzs, intensities in self.reader.iter_spectra(batch_size):
            if len(mzs) == 0:
                yield coords, resampled_mzs, np.zeros_like(resampled_mzs)
                continue
            
            # Interpolate intensities to bin centers
            # Using linear interpolation with extrapolation as 0
            resampled_intensities = np.interp(
                resampled_mzs,  # New x values (bin centers)
                mzs,         # Original x values
                intensities, # Original y values
                left=0.0,    # Value for extrapolation below
                right=0.0    # Value for extrapolation above
            )
            
            yield coords, resampled_mzs, resampled_intensities
    
    def close(self) -> None:
        """Close underlying reader."""
        self.reader.close()