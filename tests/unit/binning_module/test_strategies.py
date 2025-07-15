# tests/unit/binning_module/test_strategies.py
"""Unit tests for binning strategies."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from msiconvert.binning_module.domain.strategies import LinearTOFStrategy, ReflectorTOFStrategy


class TestLinearTOFStrategy:
    """Test LinearTOFStrategy implementation."""
    
    @pytest.fixture
    def strategy(self):
        """Create LinearTOFStrategy instance."""
        return LinearTOFStrategy()
    
    def test_calculate_num_bins(self, strategy):
        """Test number of bins calculation."""
        # Test basic calculation with larger bin width
        num_bins = strategy.calculate_num_bins(
            target_width_da=0.1,  # 100 mDa - larger to get reasonable number
            reference_mz=1000.0,
            min_mz=100.0,
            max_mz=2000.0
        )
        
        assert isinstance(num_bins, int)
        assert num_bins > 0
        assert num_bins < 100000  # Reasonable upper bound for larger bin size
        
    def test_calculate_target_width(self, strategy):
        """Test target width calculation."""
        width = strategy.calculate_target_width(
            num_bins=1000,
            reference_mz=1000.0,
            min_mz=100.0,
            max_mz=2000.0
        )
        
        assert isinstance(width, float)
        assert width > 0
        
    def test_generate_bin_edges(self, strategy):
        """Test bin edge generation."""
        num_bins = 100
        min_mz = 100.0
        max_mz = 2000.0
        
        edges = strategy.generate_bin_edges(min_mz, max_mz, num_bins)
        
        # Check basic properties
        assert len(edges) == num_bins + 1
        assert edges[0] == min_mz
        assert edges[-1] == max_mz
        
        # Check monotonic increase
        assert np.all(np.diff(edges) > 0)
        
        # Check non-linearity (bin widths should increase)
        widths = np.diff(edges)
        assert widths[-1] > widths[0]
    
    def test_roundtrip_consistency(self, strategy):
        """Test that num_bins -> width -> num_bins is consistent."""
        original_num_bins = 500
        reference_mz = 1000.0
        min_mz = 100.0
        max_mz = 2000.0
        
        # Calculate width for given bins
        width = strategy.calculate_target_width(
            original_num_bins, reference_mz, min_mz, max_mz
        )
        
        # Calculate bins for that width
        calculated_bins = strategy.calculate_num_bins(
            width, reference_mz, min_mz, max_mz
        )
        
        # Should be close (within rounding)
        assert abs(calculated_bins - original_num_bins) <= 1


class TestReflectorTOFStrategy:
    """Test ReflectorTOFStrategy implementation."""
    
    @pytest.fixture
    def strategy(self):
        """Create ReflectorTOFStrategy instance."""
        return ReflectorTOFStrategy()
    
    def test_calculate_num_bins(self, strategy):
        """Test number of bins calculation."""
        num_bins = strategy.calculate_num_bins(
            target_width_da=0.1,  # 100 mDa - larger to get reasonable number
            reference_mz=1000.0,
            min_mz=100.0,
            max_mz=2000.0
        )
        
        assert isinstance(num_bins, int)
        assert num_bins > 0
        assert num_bins < 100000  # Reasonable upper bound for larger bin size
        
    def test_calculate_target_width(self, strategy):
        """Test target width calculation."""
        width = strategy.calculate_target_width(
            num_bins=1000,
            reference_mz=1000.0,
            min_mz=100.0,
            max_mz=2000.0
        )
        
        assert isinstance(width, float)
        assert width > 0
        
    def test_generate_bin_edges(self, strategy):
        """Test bin edge generation."""
        num_bins = 100
        min_mz = 100.0
        max_mz = 2000.0
        
        edges = strategy.generate_bin_edges(min_mz, max_mz, num_bins)
        
        # Check basic properties
        assert len(edges) == num_bins + 1
        assert edges[0] == min_mz
        assert edges[-1] == max_mz
        
        # Check monotonic increase
        assert np.all(np.diff(edges) > 0)
        
        # Check non-linearity
        widths = np.diff(edges)
        assert widths[-1] > widths[0]
        
    def test_logarithmic_spacing(self, strategy):
        """Test that bins are logarithmically spaced."""
        edges = strategy.generate_bin_edges(100.0, 1000.0, 9)
        
        # In log space, bins should be equally spaced
        log_edges = np.log(edges)
        log_widths = np.diff(log_edges)
        
        # Check that log widths are approximately equal
        assert_array_almost_equal(log_widths, log_widths[0], decimal=10)
    
    def test_edge_cases(self, strategy):
        """Test edge cases."""
        # Single bin
        edges = strategy.generate_bin_edges(100.0, 200.0, 1)
        assert len(edges) == 2
        assert edges[0] == 100.0
        assert edges[1] == 200.0
        
        # Very small range
        edges = strategy.generate_bin_edges(999.9, 1000.1, 10)
        assert len(edges) == 11
        assert np.all(np.diff(edges) > 0)
    
    def test_roundtrip_consistency(self, strategy):
        """Test that num_bins -> width -> num_bins is consistent."""
        original_num_bins = 500
        reference_mz = 1000.0
        min_mz = 100.0
        max_mz = 2000.0
        
        # Calculate width for given bins
        width = strategy.calculate_target_width(
            original_num_bins, reference_mz, min_mz, max_mz
        )
        
        # Calculate bins for that width
        calculated_bins = strategy.calculate_num_bins(
            width, reference_mz, min_mz, max_mz
        )
        
        # Should be close (within rounding)
        assert abs(calculated_bins - original_num_bins) <= 1