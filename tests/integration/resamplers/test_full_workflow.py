# tests/integration/resampling_module/test_full_workflow.py
"""Integration tests for the complete resampling workflow."""

import pytest
import numpy as np
from typing import List

from msiconvert.resamplers.application.models import ResamplingRequest
from msiconvert.resamplers.application.factory import StrategyFactory
from msiconvert.resamplers.services.resampling_service import ResamplingService
from msiconvert.resamplers.infrastructure.config import ResamplingConfig


class TestFullWorkflow:
    """Test complete resampling workflows."""
    
    def test_linear_tof_workflow(self):
        """Test complete workflow for linear TOF instrument."""
        # Create request with larger bin size to avoid limits
        request = ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='linear',
            bin_size_mu=50.0,  # 50 milli-u
            reference_mz=1000.0
        )
        
        # Create service with appropriate strategy
        config = ResamplingConfig()
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = ResamplingService(strategy, config)
        
        # Generate bin edges
        result = service.generate_resampled_axis(request)
        
        # Validate result
        assert result.final_num_bins > 0
        assert len(result.bin_edges) == result.final_num_bins + 1
        assert result.bin_edges[0] == request.min_mz
        assert result.bin_edges[-1] == request.max_mz
        assert np.all(np.diff(result.bin_edges) > 0)
        
        # Check non-linearity
        widths = np.diff(result.bin_edges)
        assert widths[-1] > widths[0]  # Bins get wider at higher m/z
        
        # Check achieved width is close to target (allow 10% tolerance)
        assert abs(result.achieved_width_at_ref_mz_da * 1000 - 50.0) < 10.0
    
    def test_reflector_tof_workflow(self):
        """Test complete workflow for reflector TOF instrument."""
        # Create request with number of bins
        request = ResamplingRequest(
            min_mz=50.0,
            max_mz=3000.0,
            model_type='reflector',
            num_bins=5000
        )
        
        # Create service
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = ResamplingService(strategy)
        
        # Generate bin edges
        result = service.generate_resampled_axis(request)
        
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
        # Create custom config with higher limits
        custom_config = ResamplingConfig(
            MAX_ALLOWED_BINS=200000,
            BIN_WIDTH_TOLERANCE=0.05
        )
        
        request = ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='linear',
            bin_size_mu=20.0,  # Smaller bin size possible with higher limit
            reference_mz=1000.0
        )
        
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = ResamplingService(strategy, custom_config)
        
        result = service.generate_resampled_axis(request)
        
        assert result.final_num_bins > 0
        assert result.final_num_bins <= custom_config.MAX_ALLOWED_BINS
        
        # Check achieved width is close to target
        assert abs(result.achieved_width_at_ref_mz_da * 1000 - 20.0) < 5.0
    
    def test_strategy_comparison(self):
        """Test comparing different strategies with same parameters."""
        # Both strategies should produce similar number of bins for similar parameters
        base_params = {
            'min_mz': 100.0,
            'max_mz': 2000.0,
            'bin_size_mu': 50.0,  # Larger bin size
            'reference_mz': 1000.0
        }
        
        # Linear TOF
        linear_request = ResamplingRequest(**base_params, model_type='linear')
        linear_strategy = StrategyFactory.create_strategy('linear')
        linear_service = ResamplingService(linear_strategy)
        linear_result = linear_service.generate_resampled_axis(linear_request)
        
        # Reflector TOF
        reflector_request = ResamplingRequest(**base_params, model_type='reflector')
        reflector_strategy = StrategyFactory.create_strategy('reflector')
        reflector_service = ResamplingService(reflector_strategy)
        reflector_result = reflector_service.generate_resampled_axis(reflector_request)
        
        # Number of bins should be in same order of magnitude
        ratio = linear_result.final_num_bins / reflector_result.final_num_bins
        assert 0.1 < ratio < 10.0  # Within an order of magnitude
        
        # Both should achieve similar width at reference (within 20% tolerance)
        assert abs(linear_result.achieved_width_at_ref_mz_da - 
                   reflector_result.achieved_width_at_ref_mz_da) < 0.02
    
    def test_error_handling_workflow(self):
        """Test error handling in complete workflow."""
        # Test with parameters that would exceed limits
        config = ResamplingConfig(MAX_ALLOWED_BINS=1000)  # Low limit
        
        request = ResamplingRequest(
            min_mz=100.0,
            max_mz=2000.0,
            model_type='linear',
            bin_size_mu=1.0,  # Very small bin size to exceed limit
            reference_mz=1000.0
        )
        
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = ResamplingService(strategy, config)
        
        # Should raise ResamplingLimitExceededError
        from msiconvert.resamplers.exceptions import ResamplingLimitExceededError
        with pytest.raises(ResamplingLimitExceededError):
            service.generate_resampled_axis(request)
    
    def test_edge_case_workflows(self):
        """Test edge case scenarios in workflows."""
        # Test with very small m/z range
        request = ResamplingRequest(
            min_mz=999.0,
            max_mz=1001.0,
            model_type='reflector',
            num_bins=10
        )
        
        strategy = StrategyFactory.create_strategy(request.model_type)
        service = ResamplingService(strategy)
        result = service.generate_resampled_axis(request)
        
        assert result.final_num_bins == 10
        assert len(result.bin_edges) == 11
        assert np.all(np.diff(result.bin_edges) > 0)
        
        # Test with single bin
        request_single = ResamplingRequest(
            min_mz=100.0,
            max_mz=200.0,
            model_type='linear',
            num_bins=1
        )
        
        result_single = service.generate_resampled_axis(request_single)
        assert result_single.final_num_bins == 1
        assert len(result_single.bin_edges) == 2
        assert result_single.bin_edges[0] == 100.0
        assert result_single.bin_edges[1] == 200.0