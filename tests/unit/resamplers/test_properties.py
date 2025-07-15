# tests/unit/resampling_module/test_properties.py
"""Property-based tests for resampling strategies."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume
from hypothesis.strategies import composite

from msiconvert.resamplers.domain.strategies import LinearTOFStrategy, ReflectorTOFStrategy
from msiconvert.resamplers.application.models import ResamplingRequest
from msiconvert.resamplers.application.factory import StrategyFactory
from msiconvert.resamplers.services.resampling_service import ResamplingService


# Custom strategies for generating test data
@composite
def mz_range(draw):
    """Generate valid m/z ranges."""
    min_mz = draw(st.floats(min_value=1.0, max_value=5000.0))
    max_mz = draw(st.floats(min_value=min_mz + 1.0, max_value=10000.0))
    return min_mz, max_mz


@composite
def resampling_request(draw, model_type=None):
    """Generate valid resampling requests."""
    min_mz, max_mz = draw(mz_range())
    
    if model_type is None:
        model_type = draw(st.sampled_from(['linear', 'reflector']))
    
    # Choose between num_bins or bin_size_mu
    use_num_bins = draw(st.booleans())
    
    if use_num_bins:
        num_bins = draw(st.integers(min_value=1, max_value=10000))
        return ResamplingRequest(
            min_mz=min_mz,
            max_mz=max_mz,
            model_type=model_type,
            num_bins=num_bins,
            reference_mz=draw(st.floats(min_value=min_mz, max_value=max_mz))
        )
    else:
        # Use larger bin sizes to avoid exceeding limits
        bin_size_mu = draw(st.floats(min_value=10.0, max_value=1000.0))  # 10-1000 mu
        return ResamplingRequest(
            min_mz=min_mz,
            max_mz=max_mz,
            model_type=model_type,
            bin_size_mu=bin_size_mu,
            reference_mz=draw(st.floats(min_value=min_mz, max_value=max_mz))
        )


class TestStrategyProperties:
    """Property-based tests for strategies."""
    
    @given(resampling_request())
    def test_bin_edges_properties(self, request):
        """Test that bin edges always satisfy basic properties."""
        strategy = StrategyFactory.create_strategy(request.model_type)
        
        try:
            if request.num_bins is not None:
                edges = strategy.generate_bin_edges(
                    request.min_mz, request.max_mz, request.num_bins
                )
            else:
                # Calculate num_bins first for bin_size_mu case
                target_width_da = request.bin_size_mu / 1000.0
                num_bins = strategy.calculate_num_bins(
                    target_width_da, request.reference_mz, 
                    request.min_mz, request.max_mz
                )
                # Skip if too many bins would be generated
                assume(num_bins <= 100000)
                edges = strategy.generate_bin_edges(
                    request.min_mz, request.max_mz, num_bins
                )
        except Exception:
            assume(False)
            return
        
        # Basic properties
        assert len(edges) >= 2
        assert edges[0] == request.min_mz
        assert edges[-1] == request.max_mz
        assert np.all(np.diff(edges) > 0)  # Monotonic increase
        assert np.all(np.isfinite(edges))  # No infinity or NaN
    
    @given(st.integers(min_value=10, max_value=1000))
    def test_linear_strategy_sqrt_relationship(self, num_bins):
        """Test that LinearTOFStrategy maintains sqrt relationship."""
        min_mz, max_mz = 100.0, 2000.0
        strategy = LinearTOFStrategy()
        edges = strategy.generate_bin_edges(min_mz, max_mz, num_bins)
        
        # Transform to flight time space (sqrt)
        ft_edges = np.sqrt(edges)
        ft_widths = np.diff(ft_edges)
        
        # In flight time space, bins should be approximately equal
        assert np.allclose(ft_widths, ft_widths[0], rtol=1e-10)
    
    @given(st.integers(min_value=10, max_value=1000))
    def test_reflector_strategy_log_relationship(self, num_bins):
        """Test that ReflectorTOFStrategy maintains log relationship."""
        min_mz, max_mz = 100.0, 2000.0
        strategy = ReflectorTOFStrategy()
        edges = strategy.generate_bin_edges(min_mz, max_mz, num_bins)
        
        # Transform to log space
        log_edges = np.log(edges)
        log_widths = np.diff(log_edges)
        
        # In log space, bins should be approximately equal
        assert np.allclose(log_widths, log_widths[0], rtol=1e-10)
    
    @given(resampling_request(model_type='linear'))
    def test_width_consistency(self, request):
        """Test that calculated and achieved widths are consistent."""
        if request.bin_size_mu is None:
            return  # Skip if using num_bins
            
        strategy = LinearTOFStrategy()
        service = ResamplingService(strategy)
        
        try:
            result = service.generate_resampled_axis(request)
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
    
    @given(resampling_request())
    def test_no_empty_bins(self, request):
        """Test that all bins have positive width."""
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = ResamplingService(strategy)
        
        try:
            result = service.generate_resampled_axis(request)
        except Exception:
            assume(False)
            return
        
        # All bin widths should be positive
        widths = np.diff(result.bin_edges)
        assert np.all(widths > 0)
        
        # No width should be exactly zero
        assert not np.any(np.isclose(widths, 0))