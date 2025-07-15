# msiconvert/binning_module/domain/models.py
"""Domain models for binning results."""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..application.models import BinningRequest


@dataclass(frozen=True)
class BinningResult:
    """
    Immutable value object containing binning operation results.
    
    Attributes
    ----------
    bin_edges : NDArray[np.float64]
        Array of bin edge values in m/z (length = num_bins + 1)
    final_num_bins : int
        Total number of bins created
    achieved_width_at_ref_mz_da : float
        Actual bin width achieved at reference m/z (in Daltons)
    parameters_used : BinningRequest
        Original request parameters for traceability
    """
    bin_edges: NDArray[np.float64]
    final_num_bins: int
    achieved_width_at_ref_mz_da: float
    parameters_used: 'BinningRequest'
    
    def __post_init__(self):
        """Validate result consistency."""
        if len(self.bin_edges) != self.final_num_bins + 1:
            raise ValueError(
                f"Inconsistent result: {len(self.bin_edges)} edges "
                f"for {self.final_num_bins} bins"
            )
        
        if not np.all(np.diff(self.bin_edges) > 0):
            raise ValueError("Bin edges must be monotonically increasing")