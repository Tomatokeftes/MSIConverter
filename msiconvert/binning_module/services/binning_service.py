# msiconvert/binning_module/services/binning_service.py
"""Service layer for orchestrating the binning process."""

import logging
from typing import Optional

from ..domain.strategies import BinningStrategy
from ..domain.models import BinningResult
from ..application.models import BinningRequest
from ..infrastructure.config import BinningConfig
from ..exceptions import BinningLimitExceededError, InvalidParametersError


logger = logging.getLogger(__name__)


class BinningService:
    """
    Orchestrate the binning process using appropriate strategy.
    
    This service handles the business logic of:
    - Parameter validation against configuration limits
    - Coordinating strategy execution
    - Building the final result
    """
    
    def __init__(self, strategy: BinningStrategy, config: Optional[BinningConfig] = None):
        """
        Initialize the binning service.
        
        Parameters
        ----------
        strategy : BinningStrategy
            Strategy instance to use for binning
        config : Optional[BinningConfig]
            Configuration instance (uses defaults if not provided)
        """
        self.strategy = strategy
        self.config = config or BinningConfig()
    
    def generate_binned_axis(self, request: BinningRequest) -> BinningResult:
        """
        Generate binned axis based on request parameters.
        
        Parameters
        ----------
        request : BinningRequest
            Validated request parameters
            
        Returns
        -------
        BinningResult
            Result containing bin edges and metadata
            
        Raises
        ------
        BinningLimitExceededError
            If calculated bins exceed maximum allowed
        InvalidParametersError
            If parameters result in invalid binning
        """
        logger.info(
            f"Generating binned axis for {request.model_type} model "
            f"with m/z range [{request.min_mz}, {request.max_mz}]"
        )
        
        # Validate m/z range against config limits
        self._validate_mz_range(request.min_mz, request.max_mz)
        
        # Calculate number of bins based on provided parameters
        if request.num_bins is not None:
            num_bins = request.num_bins
            # Calculate corresponding bin width at reference m/z
            bin_width_da = self.strategy.calculate_target_width(
                num_bins=num_bins,
                reference_mz=request.reference_mz,
                min_mz=request.min_mz,
                max_mz=request.max_mz
            )
            logger.info(
                f"Using specified {num_bins} bins, "
                f"resulting in {bin_width_da*1000:.3f} mDa width at m/z {request.reference_mz}"
            )
        else:
            # Convert milli-Daltons to Daltons
            target_width_da = request.bin_size_mu / 1000.0
            
            # Calculate required number of bins
            num_bins = self.strategy.calculate_num_bins(
                target_width_da=target_width_da,
                reference_mz=request.reference_mz,
                min_mz=request.min_mz,
                max_mz=request.max_mz
            )
            bin_width_da = target_width_da
            logger.info(
                f"Targeting {request.bin_size_mu} mDa width at m/z {request.reference_mz}, "
                f"requires {num_bins} bins"
            )
        
        # Validate against maximum bins limit
        if num_bins > self.config.MAX_ALLOWED_BINS:
            raise BinningLimitExceededError(
                f"Calculated bins ({num_bins}) exceeds maximum allowed "
                f"({self.config.MAX_ALLOWED_BINS}). Consider using a larger bin size."
            )
        
        # Generate bin edges
        bin_edges = self.strategy.generate_bin_edges(
            min_mz=request.min_mz,
            max_mz=request.max_mz,
            num_bins=num_bins
        )
        
        # Validate bin edges
        self._validate_bin_edges(bin_edges, request.min_mz, request.max_mz)
        
        # Calculate actual achieved width at reference m/z
        achieved_width = self._calculate_achieved_width(
            bin_edges=bin_edges,
            reference_mz=request.reference_mz
        )
        
        # Log warning if achieved width differs significantly from target
        if request.bin_size_mu is not None:
            relative_error = abs(achieved_width * 1000 - request.bin_size_mu) / request.bin_size_mu
            if relative_error > self.config.BIN_WIDTH_TOLERANCE:
                logger.warning(
                    f"Achieved bin width ({achieved_width*1000:.3f} mDa) differs from "
                    f"target ({request.bin_size_mu} mDa) by {relative_error*100:.1f}%"
                )
        
        # Build and return result
        result = BinningResult(
            bin_edges=bin_edges,
            final_num_bins=num_bins,
            achieved_width_at_ref_mz_da=achieved_width,
            parameters_used=request
        )
        
        logger.info(
            f"Successfully generated {result.final_num_bins} bins "
            f"with {result.achieved_width_at_ref_mz_da*1000:.3f} mDa width "
            f"at reference m/z {request.reference_mz}"
        )
        
        return result
    
    def _validate_mz_range(self, min_mz: float, max_mz: float) -> None:
        """
        Validate m/z range against configuration limits.
        
        Parameters
        ----------
        min_mz : float
            Minimum m/z value
        max_mz : float
            Maximum m/z value
            
        Raises
        ------
        InvalidParametersError
            If range is outside configured limits
        """
        if min_mz < self.config.MIN_MZ_VALUE:
            raise InvalidParametersError(
                f"min_mz ({min_mz}) is below minimum allowed value "
                f"({self.config.MIN_MZ_VALUE})"
            )
        
        if max_mz > self.config.MAX_MZ_VALUE:
            raise InvalidParametersError(
                f"max_mz ({max_mz}) exceeds maximum allowed value "
                f"({self.config.MAX_MZ_VALUE})"
            )
    
    def _validate_bin_edges(self, bin_edges, min_mz: float, max_mz: float) -> None:
        """
        Validate generated bin edges.
        
        Parameters
        ----------
        bin_edges : NDArray[np.float64]
            Array of bin edges
        min_mz : float
            Expected minimum m/z
        max_mz : float
            Expected maximum m/z
            
        Raises
        ------
        InvalidParametersError
            If bin edges are invalid
        """
        import numpy as np
        
        if len(bin_edges) < 2:
            raise InvalidParametersError("Insufficient bin edges generated")
        
        if not np.all(np.diff(bin_edges) > 0):
            raise InvalidParametersError("Bin edges are not monotonically increasing")
        
        # Allow small tolerance for floating point comparison
        tolerance = 1e-10
        if abs(bin_edges[0] - min_mz) > tolerance:
            raise InvalidParametersError(
                f"First bin edge ({bin_edges[0]}) does not match min_mz ({min_mz})"
            )
        
        if abs(bin_edges[-1] - max_mz) > tolerance:
            raise InvalidParametersError(
                f"Last bin edge ({bin_edges[-1]}) does not match max_mz ({max_mz})"
            )
    
    def _calculate_achieved_width(self, bin_edges, reference_mz: float) -> float:
        """
        Calculate the actual bin width achieved at the reference m/z.
        
        Parameters
        ----------
        bin_edges : NDArray[np.float64]
            Array of bin edges
        reference_mz : float
            Reference m/z value
            
        Returns
        -------
        float
            Bin width in Daltons at reference m/z
        """
        import numpy as np
        
        # Find the bin containing the reference m/z
        bin_index = np.searchsorted(bin_edges, reference_mz, side='right') - 1
        
        # Ensure we're within valid range
        if bin_index < 0:
            bin_index = 0
        elif bin_index >= len(bin_edges) - 1:
            bin_index = len(bin_edges) - 2
        
        # Calculate width of the bin containing reference m/z
        width = bin_edges[bin_index + 1] - bin_edges[bin_index]
        
        return width