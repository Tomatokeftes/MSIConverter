# msiconvert/binning_module/tests/integration/test_full_workflow.py
"""Integration tests for the complete binning workflow."""

import pytest
import numpy as np
from typing import List

from ...application.models import BinningRequest
from ...application.factory import StrategyFactory
from ...services.binning_service import BinningService
from ...infrastructure.config import BinningConfig


class TestFullWorkflow:
    """Test complete binning workflows."""
    
    def test_linear_tof_workflow(self):
        """Test complete workflow for linear TOF instrument."""
        # Create request
        request = BinningRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='linear',
            bin_size_mu=5.0,  # 5 milli-Daltons
            reference_mz=1000.0
        )
        
        # Create service with appropriate strategy
        config = BinningConfig()
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = BinningService(strategy, config)
        
        # Generate bin edges
        result = service.generate_binned_axis(request)
        
        # Validate result
        assert result.final_num_bins > 0
        assert len(result.bin_edges) == result.final_num_bins + 1
        assert result.bin_edges[0] == request.min_mz
        assert result.bin_edges[-1] == request.max_mz
        assert np.all(np.diff(result.bin_edges) > 0)
        
        # Check non-linearity
        widths = np.diff(result.bin_edges)
        assert widths[-1] > widths[0]  # Bins get wider at higher m/z
        
        # Check achieved width is close to target
        assert abs(result.achieved_width_at_ref_mz_da * 1000 - 5.0) < 0.5
    
    def test_reflector_tof_workflow(self):
        """Test complete workflow for reflector TOF instrument."""
        # Create request with number of bins
        request = BinningRequest(
            min_mz=50.0,
            max_mz=3000.0,
            model_type='reflector',
            num_bins=5000
        )
        
        # Create service
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = BinningService(strategy)
        
        # Generate bin edges
        result = service.generate_binned_axis(request)
        
        # Validate result
        assert result.final_num_bins == 5000
        assert len(result.bin_edges) == 5001
        assert result.achieved_width_at_ref_mz_da > 0
        
        # Check logarithmic spacing
        log_edges = np.log(result.bin_edges)
        log_widths = np.diff(log_edges)
        # Log widths should be approximately constant
        assert np.std(log_widths) < 1e-10
    
    def test_workflow_with_different_configs(self):
        """Test workflow with custom configuration."""
        # Create custom config
        config = BinningConfig(
            DEFAULT_REFERENCE_MZ=500.0,
            MAX_ALLOWED_BINS=1000,
            MIN_MZ_VALUE=10.0,
            MAX_MZ_VALUE=5000.0
        )
        
        # Create request
        request = BinningRequest(
            min_mz=10.0,
            max_mz=5000.0,
            model_type='linear',
            num_bins=900  # Within limit
        )
        
        # Create service
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = BinningService(strategy, config)
        
        # Should work fine
        result = service.generate_binned_axis(request)
        assert result.final_num_bins == 900
    
    def test_edge_cases(self):
        """Test edge cases in the workflow."""
        # Very small m/z range
        request = BinningRequest(
            min_mz=999.0,
            max_mz=1001.0,
            model_type='linear',
            bin_size_mu=1.0  # 1 mDa
        )
        
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = BinningService(strategy)
        result = service.generate_binned_axis(request)
        
        # Should handle small ranges gracefully
        assert result.final_num_bins >= 2
        assert result.bin_edges[0] == 999.0
        assert result.bin_edges[-1] == 1001.0
        
        # Single bin case
        request = BinningRequest(
            min_mz=100.0,
            max_mz=200.0,
            model_type='reflector',
            num_bins=1
        )
        
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = BinningService(strategy)
        result = service.generate_binned_axis(request)
        
        assert result.final_num_bins == 1
        assert len(result.bin_edges) == 2
        assert result.bin_edges[0] == 100.0
        assert result.bin_edges[-1] == 200.0
    
    def test_performance_with_large_bins(self):
        """Test performance with large number of bins."""
        request = BinningRequest(
            min_mz=1.0,
            max_mz=10000.0,
            model_type='linear',
            num_bins=100000  # Large but within default limit
        )
        
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = BinningService(strategy)
        
        # Should complete without issues
        result = service.generate_binned_axis(request)
        
        assert result.final_num_bins == 100000
        assert len(result.bin_edges) == 100001
        
        # Verify edges are properly spaced
        assert np.all(np.diff(result.bin_edges) > 0)
    
    def test_consistency_across_strategies(self):
        """Test that strategies produce consistent results."""
        # Both strategies should produce similar number of bins for similar parameters
        base_params = {
            'min_mz': 100.0,
            'max_mz': 2000.0,
            'bin_size_mu': 5.0,
            'reference_mz': 1000.0
        }
        
        # Linear TOF
        linear_request = BinningRequest(**base_params, model_type='linear')
        linear_strategy = StrategyFactory.create_strategy('linear')
        linear_service = BinningService(linear_strategy)
        linear_result = linear_service.generate_binned_axis(linear_request)
        
        # Reflector TOF
        reflector_request = BinningRequest(**base_params, model_type='reflector')
        reflector_strategy = StrategyFactory.create_strategy('reflector')
        reflector_service = BinningService(reflector_strategy)
        reflector_result = reflector_service.generate_binned_axis(reflector_request)
        
        # Number of bins should be in same order of magnitude
        ratio = linear_result.final_num_bins / reflector_result.final_num_bins
        assert 0.1 < ratio < 10.0  # Within an order of magnitude
        
        # Both should achieve similar width at reference
        assert abs(linear_result.achieved_width_at_ref_mz_da - 
                   reflector_result.achieved_width_at_ref_mz_da) < 0.01