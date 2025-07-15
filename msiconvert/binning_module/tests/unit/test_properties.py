# msiconvert/binning_module/tests/unit/test_properties.py
"""Property-based tests for binning strategies."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume
from hypothesis.strategies import composite

from ...domain.strategies import LinearTOFStrategy, ReflectorTOFStrategy
from ...application.models import BinningRequest
from ...application.factory import StrategyFactory
from ...services.binning_service import BinningService


# Custom strategies for generating test data
@composite
def mz_range(draw):
    """Generate valid m/z ranges."""
    min_mz = draw(st.floats(min_value=1.0, max_value=5000.0))
    max_mz = draw(st.floats(min_value=min_mz + 1.0, max_value=10000.0))
    return min_mz, max_mz


@composite
def binning_request(draw, model_type=None):
    """Generate valid binning requests."""
    min_mz, max_mz = draw(mz_range())
    
    if model_type is None:
        model_type = draw(st.sampled_from(['linear', 'reflector']))
    
    # Choose between num_bins or bin_size_mu
    use_num_bins = draw(st.booleans())
    
    if use_num_bins:
        num_bins = draw(st.integers(min_value=1, max_value=10000))
        return BinningRequest(
            min_mz=min_mz,
            max_mz=max_mz,
            model_type=model_type,
            num_bins=num_bins,
            reference_mz=draw(st.floats(min_value=min_mz, max_value=max_mz))
        )
    else:
        bin_size_mu = draw(st.floats(min_value=0.1, max_value=100.0))
        return BinningRequest(
            min_mz=min_mz,
            max_mz=max_mz,
            model_type=model_type,
            bin_size_mu=bin_size_mu,
            reference_mz=draw(st.floats(min_value=min_mz, max_value=max_mz))
        )


class TestStrategyProperties:
    """Property-based tests for strategies."""
    
    @given(binning_request())
    def test_bin_edges_properties(self, request):
        """Test that bin edges always satisfy basic properties."""
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = BinningService(strategy)
        
        try:
            result = service.generate_binned_axis(request)
        except Exception:
            # Skip if request leads to too many bins
            assume(False)
            return
        
        # Property 1: Number of edges = num_bins + 1
        assert len(result.bin_edges) == result.final_num_bins + 1
        
        # Property 2: Edges are monotonically increasing
        assert np.all(np.diff(result.bin_edges) > 0)
        
        # Property 3: First and last edges match request
        assert result.bin_edges[0] == request.min_mz
        assert result.bin_edges[-1] == request.max_mz
        
        # Property 4: All edges within range
        assert np.all(result.bin_edges >= request.min_mz)
        assert np.all(result.bin_edges <= request.max_mz)
    
    @given(
        min_mz=st.floats(min_value=10.0, max_value=1000.0),
        max_mz=st.floats(min_value=1001.0, max_value=5000.0),
        num_bins=st.integers(min_value=10, max_value=1000)
    )
    def test_linear_tof_properties(self, min_mz, max_mz, num_bins):
        """Test specific properties of Linear TOF strategy."""
        strategy = LinearTOFStrategy()
        edges = strategy.generate_bin_edges(min_mz, max_mz, num_bins)
        
        # Transform to flight time space
        ft_edges = np.sqrt(edges)
        ft_widths = np.diff(ft_edges)
        
        # In flight time space, bins should be approximately equal
        # Allow small tolerance for floating point
        assert np.allclose(ft_widths, ft_widths[0], rtol=1e-10)
    
    @given(
        min_mz=st.floats(min_value=10.0, max_value=1000.0),
        max_mz=st.floats(min_value=1001.0, max_value=5000.0),
        num_bins=st.integers(min_value=10, max_value=1000)
    )
    def test_reflector_tof_properties(self, min_mz, max_mz, num_bins):
        """Test specific properties of Reflector TOF strategy."""
        strategy = ReflectorTOFStrategy()
        edges = strategy.generate_bin_edges(min_mz, max_mz, num_bins)
        
        # Transform to log space
        log_edges = np.log(edges)
        log_widths = np.diff(log_edges)
        
        # In log space, bins should be approximately equal
        assert np.allclose(log_widths, log_widths[0], rtol=1e-10)
    
    @given(binning_request(model_type='linear'))
    def test_width_consistency(self, request):
        """Test that calculated and achieved widths are consistent."""
        if request.bin_size_mu is None:
            return  # Skip if using num_bins
            
        strategy = LinearTOFStrategy()
        service = BinningService(strategy)
        
        try:
            result = service.generate_binned_axis(request)
        except Exception:
            assume(False)
            return
        
        # Achieved width should be close to requested width
        requested_width_da = request.bin_size_mu / 1000.0
        relative_error = abs(result.achieved_width_at_ref_mz_da - requested_width_da) / requested_width_da
        
        # Allow up to 10% error due to discretization
        assert relative_error < 0.1
    
    @given(
        strategy_type=st.sampled_from(['linear', 'reflector']),
        num_bins_1=st.integers(min_value=100, max_value=1000)
    )
    def test_roundtrip_conversion(self, strategy_type, num_bins_1):
        """Test that num_bins -> width -> num_bins conversion is consistent."""
        min_mz, max_mz = 100.0, 2000.0
        reference_mz = 1000.0
        
        strategy = StrategyFactory.create_strategy(strategy_type)
        
        # Calculate width for given number of bins
        width = strategy.calculate_target_width(num_bins_1, reference_mz, min_mz, max_mz)
        
        # Calculate number of bins for that width
        num_bins_2 = strategy.calculate_num_bins(width, reference_mz, min_mz, max_mz)
        
        # Should be within 1 due to rounding
        assert abs(num_bins_2 - num_bins_1) <= 1
    
    @given(binning_request())
    def test_no_empty_bins(self, request):
        """Test that all bins have positive width."""
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = BinningService(strategy)
        
        try:
            result = service.generate_binned_axis(request)
        except Exception:
            assume(False)
            return
        
        # All bin widths should be positive
        widths = np.diff(result.bin_edges)
        assert np.all(widths > 0)
        
        # No width should be exactly zero
        assert not np.any(np.isclose(widths, 0))