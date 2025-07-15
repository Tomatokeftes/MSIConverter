# msiconvert/binning_module/tests/unit/test_models.py
"""Unit tests for domain and application models."""

import pytest
import numpy as np

from ...domain.models import BinningResult
from ...application.models import BinningRequest
from ...exceptions import InvalidParametersError


class TestBinningRequest:
    """Test BinningRequest validation."""
    
    def test_valid_request_with_num_bins(self):
        """Test creating valid request with num_bins."""
        request = BinningRequest(
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
        assert request.reference_mz == 1000.0
    
    def test_valid_request_with_bin_size(self):
        """Test creating valid request with bin_size_mu."""
        request = BinningRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='reflector',
            bin_size_mu=5.0,
            reference_mz=500.0
        )
        
        assert request.bin_size_mu == 5.0
        assert request.num_bins is None
        assert request.reference_mz == 500.0
    
    def test_model_type_normalization(self):
        """Test that model type is normalized to lowercase."""
        request = BinningRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='LINEAR',
            num_bins=100
        )
        
        assert request.model_type == 'linear'
    
    def test_invalid_mz_range(self):
        """Test validation of m/z range."""
        # min_mz > max_mz
        with pytest.raises(InvalidParametersError, match="min_mz must be less than max_mz"):
            BinningRequest(
                min_mz=2000.0,
                max_mz=100.0,
                model_type='linear',
                num_bins=100
            )
        
        # Negative min_mz
        with pytest.raises(InvalidParametersError, match="min_mz must be positive"):
            BinningRequest(
                min_mz=-100.0,
                max_mz=2000.0,
                model_type='linear',
                num_bins=100
            )
        
        # Zero max_mz
        with pytest.raises(InvalidParametersError, match="max_mz must be positive"):
            BinningRequest(
                min_mz=100.0,
                max_mz=0.0,
                model_type='linear',
                num_bins=100
            )
    
    def test_invalid_model_type(self):
        """Test invalid model type."""
        with pytest.raises(InvalidParametersError, match="Invalid model_type"):
            BinningRequest(
                min_mz=100.0,
                max_mz=2000.0,
                model_type='unknown',
                num_bins=100
            )
    
    def test_missing_size_parameters(self):
        """Test that either num_bins or bin_size_mu must be provided."""
        with pytest.raises(InvalidParametersError, match="Either num_bins or bin_size_mu"):
            BinningRequest(
                min_mz=100.0,
                max_mz=2000.0,
                model_type='linear'
            )
    
    def test_both_size_parameters(self):
        """Test that only one size parameter can be provided."""
        with pytest.raises(InvalidParametersError, match="Only one of num_bins or bin_size_mu"):
            BinningRequest(
                min_mz=100.0,
                max_mz=2000.0,
                model_type='linear',
                num_bins=100,
                bin_size_mu=5.0
            )
    
    def test_invalid_num_bins(self):
        """Test num_bins validation."""
        with pytest.raises(InvalidParametersError, match="num_bins must be positive"):
            BinningRequest(
                min_mz=100.0,
                max_mz=2000.0,
                model_type='linear',
                num_bins=0
            )
    
    def test_invalid_bin_size(self):
        """Test bin_size_mu validation."""
        with pytest.raises(InvalidParametersError, match="bin_size_mu must be positive"):
            BinningRequest(
                min_mz=100.0,
                max_mz=2000.0,
                model_type='linear',
                bin_size_mu=-5.0
            )
    
    def test_invalid_reference_mz(self):
        """Test reference_mz validation."""
        with pytest.raises(InvalidParametersError, match="reference_mz must be positive"):
            BinningRequest(
                min_mz=100.0,
                max_mz=2000.0,
                model_type='linear',
                num_bins=100,
                reference_mz=-1000.0
            )


class TestBinningResult:
    """Test BinningResult validation."""
    
    @pytest.fixture
    def sample_request(self):
        """Create sample request for testing."""
        return BinningRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='linear',
            num_bins=100
        )
    
    def test_valid_result(self, sample_request):
        """Test creating valid result."""
        edges = np.linspace(100.0, 2000.0, 101)
        
        result = BinningResult(
            bin_edges=edges,
            final_num_bins=100,
            achieved_width_at_ref_mz_da=0.019,
            parameters_used=sample_request
        )
        
        assert len(result.bin_edges) == 101
        assert result.final_num_bins == 100
        assert result.achieved_width_at_ref_mz_da == 0.019
        assert result.parameters_used == sample_request
    
    def test_inconsistent_edges_and_bins(self, sample_request):
        """Test validation of edge count vs bin count."""
        edges = np.linspace(100.0, 2000.0, 100)  # Wrong number of edges
        
        with pytest.raises(ValueError, match="Inconsistent result"):
            BinningResult(
                bin_edges=edges,
                final_num_bins=100,
                achieved_width_at_ref_mz_da=0.019,
                parameters_used=sample_request
            )
    
    def test_non_monotonic_edges(self, sample_request):
        """Test validation of monotonic edges."""
        edges = np.array([100.0, 200.0, 150.0, 300.0])  # Non-monotonic
        
        with pytest.raises(ValueError, match="monotonically increasing"):
            BinningResult(
                bin_edges=edges,
                final_num_bins=3,
                achieved_width_at_ref_mz_da=0.1,
                parameters_used=sample_request
            )
    
    def test_immutability(self, sample_request):
        """Test that result is immutable."""
        edges = np.linspace(100.0, 2000.0, 101)
        
        result = BinningResult(
            bin_edges=edges,
            final_num_bins=100,
            achieved_width_at_ref_mz_da=0.019,
            parameters_used=sample_request
        )
        
        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            result.final_num_bins = 200