# tests/unit/resampling_module/test_models.py
"""Unit tests for data models."""

import pytest
import numpy as np

from msiconvert.resamplers.application.models import ResamplingRequest
from msiconvert.resamplers.domain.models import ResamplingResult
from msiconvert.resamplers.exceptions import InvalidParametersError


class TestResamplingRequest:
    """Test ResamplingRequest validation."""
    
    def test_valid_request_with_num_bins(self):
        """Test creating valid request with num_bins."""
        request = ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='linear',
            num_bins=1000
        )
        
        assert request.min_mz == 100.0
        assert request.max_mz == 2000.0
        assert request.model_type == 'linear'
        assert request.num_bins == 1000
        assert request.bin_size_mu is None
        assert request.reference_mz == 1000.0  # Default
    
    def test_valid_request_with_bin_size(self):
        """Test creating valid request with bin_size_mu."""
        request = ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='reflector',
            bin_size_mu=5.0,
            reference_mz=1500.0
        )
        
        assert request.min_mz == 100.0
        assert request.max_mz == 2000.0
        assert request.model_type == 'reflector'
        assert request.num_bins is None
        assert request.bin_size_mu == 5.0
        assert request.reference_mz == 1500.0
    
    def test_invalid_mz_range(self):
        """Test invalid m/z range validation."""
        with pytest.raises(InvalidParametersError, match="min_mz must be less than max_mz"):
            ResamplingRequest(
                min_mz=2000.0,
                max_mz=100.0,  # Invalid: min > max
                model_type='linear',
                num_bins=100
            )
    
    def test_invalid_model_type(self):
        """Test invalid model type validation."""
        with pytest.raises(InvalidParametersError, match="Invalid model_type"):
            ResamplingRequest(
                min_mz=100.0,
                max_mz=2000.0,
                model_type='unknown',
                num_bins=100
            )
    
    def test_missing_size_parameters(self):
        """Test that either num_bins or bin_size_mu must be provided."""
        with pytest.raises(InvalidParametersError, match="Either num_bins or bin_size_mu"):
            ResamplingRequest(
                min_mz=100.0,
                max_mz=2000.0,
                model_type='linear'
            )
    
    def test_both_size_parameters(self):
        """Test that only one size parameter can be provided."""
        with pytest.raises(InvalidParametersError, match="Only one of num_bins or bin_size_mu"):
            ResamplingRequest(
                min_mz=100.0,
                max_mz=2000.0,
                model_type='linear',
                num_bins=100,
                bin_size_mu=5.0
            )
    
    def test_invalid_num_bins(self):
        """Test num_bins validation."""
        with pytest.raises(InvalidParametersError, match="num_bins must be positive"):
            ResamplingRequest(
                min_mz=100.0,
                max_mz=2000.0,
                model_type='linear',
                num_bins=0
            )
    
    def test_invalid_bin_size(self):
        """Test bin_size_mu validation."""
        with pytest.raises(InvalidParametersError, match="bin_size_mu must be positive"):
            ResamplingRequest(
                min_mz=100.0,
                max_mz=2000.0,
                model_type='linear',
                bin_size_mu=-5.0
            )
    
    def test_invalid_reference_mz(self):
        """Test reference_mz validation."""
        with pytest.raises(InvalidParametersError, match="reference_mz must be positive"):
            ResamplingRequest(
                min_mz=100.0,
                max_mz=2000.0,
                model_type='linear',
                num_bins=100,
                reference_mz=-1000.0
            )


class TestResamplingResult:
    """Test ResamplingResult validation."""

    @pytest.fixture
    def sample_request(self):
        """Create sample request for testing."""
        return ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='linear',
            num_bins=100
        )
    
    def test_valid_result(self, sample_request):
        """Test creating valid result."""
        bin_edges = np.linspace(100.0, 2000.0, 101)  # 100 bins
        
        result = ResamplingResult(
            bin_edges=bin_edges,
            final_num_bins=100,
            achieved_width_at_ref_mz_da=19.0,
            parameters_used=sample_request
        )
        
        assert len(result.bin_edges) == 101
        assert result.final_num_bins == 100
        assert result.achieved_width_at_ref_mz_da == 19.0
        assert result.parameters_used == sample_request
    
    def test_result_properties(self, sample_request):
        """Test computed properties of result."""
        bin_edges = np.array([100.0, 150.0, 220.0, 310.0, 420.0])  # 4 bins
        
        result = ResamplingResult(
            bin_edges=bin_edges,
            final_num_bins=4,
            achieved_width_at_ref_mz_da=50.0,
            parameters_used=sample_request
        )
        
        # Test manual calculation of bin widths (since no property exists)
        expected_widths = np.array([50.0, 70.0, 90.0, 110.0])
        actual_widths = np.diff(result.bin_edges)
        np.testing.assert_array_equal(actual_widths, expected_widths)
        
        # Test manual calculation of bin centers (since no property exists)
        expected_centers = np.array([125.0, 185.0, 265.0, 365.0])
        actual_centers = (result.bin_edges[:-1] + result.bin_edges[1:]) / 2
        np.testing.assert_array_equal(actual_centers, expected_centers)
    
    def test_invalid_bin_edges(self, sample_request):
        """Test validation of bin edges."""
        # The ResamplingResult model validates monotonic edges in __post_init__
        # So this test should verify that non-monotonic edges raise an error
        bin_edges = np.array([100.0, 200.0, 150.0, 300.0])  # Non-monotonic
        
        with pytest.raises(ValueError, match="Bin edges must be monotonically increasing"):
            ResamplingResult(
                bin_edges=bin_edges,
                final_num_bins=3,
                achieved_width_at_ref_mz_da=50.0,
                parameters_used=sample_request
            )
    
    def test_result_consistency(self, sample_request):
        """Test that bin_edges length is consistent with final_num_bins."""
        bin_edges = np.linspace(100.0, 2000.0, 51)  # 50 bins
        
        result = ResamplingResult(
            bin_edges=bin_edges,
            final_num_bins=50,
            achieved_width_at_ref_mz_da=38.0,
            parameters_used=sample_request
        )
        
        assert len(result.bin_edges) == result.final_num_bins + 1
        
        # Test manual calculation of bin widths and centers
        bin_widths = np.diff(result.bin_edges)
        bin_centers = (result.bin_edges[:-1] + result.bin_edges[1:]) / 2
        assert len(bin_widths) == result.final_num_bins
        assert len(bin_centers) == result.final_num_bins