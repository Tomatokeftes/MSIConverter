# tests/unit/resampling_module/test_resampling_service.py
"""Unit tests for ResamplingService."""

import pytest
import numpy as np

from msiconvert.resamplers.services.resampling_service import ResamplingService
from msiconvert.resamplers.domain.strategies import LinearTOFStrategy, ReflectorTOFStrategy
from msiconvert.resamplers.application.models import ResamplingRequest
from msiconvert.resamplers.infrastructure.config import ResamplingConfig
from msiconvert.resamplers.exceptions import ResamplingLimitExceededError, InvalidParametersError


class TestResamplingService:
    """Test ResamplingService orchestration."""
    
    @pytest.fixture
    def linear_service(self):
        """Create service with linear strategy."""
        strategy = LinearTOFStrategy()
        config = ResamplingConfig()
        return ResamplingService(strategy, config)
    
    @pytest.fixture
    def reflector_service(self):
        """Create service with reflector strategy."""
        strategy = ReflectorTOFStrategy()
        config = ResamplingConfig()
        return ResamplingService(strategy, config)
    
    def test_generate_with_num_bins(self, linear_service):
        """Test generation with specified number of bins."""
        request = ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='linear',
            num_bins=1000
        )
        
        result = linear_service.generate_resampled_axis(request)
        
        assert result.final_num_bins == 1000
        assert len(result.bin_edges) == 1001
        assert result.bin_edges[0] == 100.0
        assert result.bin_edges[-1] == 2000.0
        assert result.achieved_width_at_ref_mz_da > 0
        assert result.parameters_used == request
    
    def test_generate_with_bin_size(self, reflector_service):
        """Test generation with specified bin size."""
        request = ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='reflector',
            bin_size_mu=50.0,  # 50 mu - larger to avoid exceeding limits
            reference_mz=1000.0
        )
        
        result = reflector_service.generate_resampled_axis(request)
        
        assert result.final_num_bins > 0
        assert len(result.bin_edges) == result.final_num_bins + 1
        assert result.bin_edges[0] == 100.0
        assert result.bin_edges[-1] == 2000.0
        
        # Check that achieved width is close to target (allow 10% tolerance)
        assert abs(result.achieved_width_at_ref_mz_da * 1000 - 50.0) < 10.0
    
    def test_max_bins_limit(self, linear_service):
        """Test that service enforces maximum bins limit."""
        # Create config with low limit
        config = ResamplingConfig(MAX_ALLOWED_BINS=100)
        service = ResamplingService(LinearTOFStrategy(), config)
        
        request = ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='linear',
            num_bins=200  # Exceeds limit
        )
        
        with pytest.raises(ResamplingLimitExceededError, match="exceeds maximum allowed"):
            service.generate_resampled_axis(request)
    
    def test_mz_range_validation(self, linear_service):
        """Test m/z range validation against config."""
        # Create config with restricted range
        config = ResamplingConfig(MIN_MZ_VALUE=50.0, MAX_MZ_VALUE=1000.0)
        service = ResamplingService(LinearTOFStrategy(), config)
        
        # Test below minimum
        request = ResamplingRequest(
            min_mz=10.0,  # Below MIN_MZ_VALUE
            max_mz=500.0,
            model_type='linear',
            num_bins=100
        )
        
        with pytest.raises(InvalidParametersError, match="below minimum allowed"):
            service.generate_resampled_axis(request)
        
        # Test above maximum
        request = ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,  # Above MAX_MZ_VALUE
            model_type='linear',
            num_bins=100
        )
        
        with pytest.raises(InvalidParametersError, match="exceeds maximum allowed"):
            service.generate_resampled_axis(request)
    
    def test_achieved_width_calculation(self, linear_service):
        """Test calculation of achieved width at reference m/z."""
        request = ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='linear',
            num_bins=1000,
            reference_mz=1000.0
        )
        
        result = linear_service.generate_resampled_axis(request)
        
        # Find bin containing reference m/z
        bin_idx = np.searchsorted(result.bin_edges, 1000.0, side='right') - 1
        expected_width = result.bin_edges[bin_idx + 1] - result.bin_edges[bin_idx]
        
        assert abs(result.achieved_width_at_ref_mz_da - expected_width) < 1e-10
    
    def test_edge_validation(self, linear_service):
        """Test that service validates generated edges."""
        # Mock a strategy that generates invalid edges
        class BadStrategy:
            def calculate_num_bins(self, *args, **kwargs):
                return 10
            
            def calculate_target_width(self, *args, **kwargs):
                return 0.1
            
            def generate_bin_edges(self, min_mz, max_mz, num_bins):
                # Return non-monotonic edges
                return np.array([100.0, 200.0, 150.0, 300.0])
        
        service = ResamplingService(BadStrategy(), ResamplingConfig())
        
        request = ResamplingRequest(
            min_mz=100.0,
            max_mz=300.0,
            model_type='linear',
            num_bins=3
        )
        
        with pytest.raises(InvalidParametersError, match="not monotonically increasing"):
            service.generate_resampled_axis(request)
    
    def test_different_reference_mz(self, linear_service):
        """Test resampling with different reference m/z values."""
        # Low reference m/z
        request1 = ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='linear',
            bin_size_mu=100.0,  # Larger bin size to avoid exceeding limits
            reference_mz=200.0
        )
        
        result1 = linear_service.generate_resampled_axis(request1)
        
        # High reference m/z
        request2 = ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='linear',
            bin_size_mu=100.0,  # Larger bin size to avoid exceeding limits
            reference_mz=1800.0
        )
        
        result2 = linear_service.generate_resampled_axis(request2)
        
        # Different reference m/z should result in different number of bins
        assert result1.final_num_bins != result2.final_num_bins