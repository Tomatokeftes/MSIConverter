#!/usr/bin/env python3
"""
Test suite for Phase 3 converter integration
"""

import sys
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_interpolation_config():
    """Test interpolation configuration class"""
    print("=== Testing Interpolation Configuration ===")
    
    try:
        from msiconvert.config import InterpolationConfig
        
        # Test default configuration
        config = InterpolationConfig()
        print(f"[OK] Default config: enabled={config.enabled}, method={config.method}")
        assert not config.enabled, "Default should be disabled"
        # Default config should have None bins when disabled
        assert config.interpolation_bins is None, "Default disabled config should have None bins"
        
        # Test enabled configuration with bins
        config_bins = InterpolationConfig(
            enabled=True,
            interpolation_bins=50000,
            method="pchip"
        )
        print(f"[OK] Bins config: {config_bins.interpolation_bins} bins")
        
        # Test enabled configuration with width
        config_width = InterpolationConfig(
            enabled=True,
            interpolation_width=0.01,
            interpolation_width_mz=400.0,
            method="adaptive"
        )
        print(f"[OK] Width config: {config_width.interpolation_width} Da at {config_width.interpolation_width_mz} m/z")
        
        # Test validation - should fail with both bins and width
        try:
            invalid_config = InterpolationConfig(
                enabled=True,
                interpolation_bins=50000,
                interpolation_width=0.01
            )
            print("[FAIL] Should have failed with both parameters")
            return False
        except ValueError as e:
            print(f"[OK] Correctly rejected invalid config: {e}")
        
        # Test configuration summary
        summary = config_bins.get_summary()
        print(f"[OK] Config summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Interpolation config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_base_converter_interpolation():
    """Test base converter interpolation methods"""
    print("\n=== Testing Base Converter Interpolation ===")
    
    try:
        from msiconvert.core.base_converter import BaseMSIConverter
        from msiconvert.config import InterpolationConfig
        
        # Create mock reader with interpolation support
        mock_reader = Mock()
        mock_reader.get_mass_bounds.return_value = (100.0, 1000.0)
        mock_reader.get_spatial_bounds.return_value = {
            'min_x': 0, 'max_x': 99, 'min_y': 0, 'max_y': 99
        }
        mock_reader.get_metadata.return_value = {
            'instrument': {'type': 'tof', 'resolution_at_400': 10000}
        }
        mock_reader.get_dimensions.return_value = (100, 100, 1)
        mock_reader.get_common_mass_axis.return_value = np.linspace(100, 1000, 10000)
        
        # Create test converter class
        class TestConverter(BaseMSIConverter):
            def _create_data_structures(self):
                return {"test": True}
            def _save_output(self, data_structures):
                return True
        
        converter = TestConverter(mock_reader, Path("/tmp/test"), dataset_id="test")
        
        # Test without interpolation config
        should_interpolate = converter._should_interpolate()
        print(f"[OK] Should interpolate without config: {should_interpolate}")
        assert not should_interpolate, "Should not interpolate without config"
        
        # Test with interpolation config
        converter.interpolation_config = InterpolationConfig(enabled=True, interpolation_bins=5000)
        should_interpolate = converter._should_interpolate()
        print(f"[OK] Should interpolate with config: {should_interpolate}")
        assert should_interpolate, "Should interpolate with enabled config"
        
        # Test setup interpolation
        with patch('msiconvert.interpolators.bounds_detector.detect_bounds_from_reader') as mock_bounds:
            mock_bounds.return_value = Mock(min_mz=100.0, max_mz=1000.0, n_spectra=10000)
            
            interp_config = converter._setup_interpolation()
            print(f"[OK] Setup interpolation config: {type(interp_config).__name__}")
            assert interp_config is not None, "Should create interpolation config"
            assert len(interp_config.target_mass_axis) > 0, "Should have target mass axis"
        
        # Test optimal workers calculation
        optimal_workers = converter._calculate_optimal_workers()
        print(f"[OK] Optimal workers: {optimal_workers}")
        assert optimal_workers >= 4, "Should have at least 4 workers"
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Base converter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reader_interpolation_methods():
    """Test reader interpolation methods"""
    print("\n=== Testing Reader Interpolation Methods ===")
    
    try:
        from msiconvert.core.base_reader import BaseMSIReader
        
        class TestReader(BaseMSIReader):
            def get_metadata(self):
                return {"test": True}
            def get_dimensions(self):
                return (100, 100, 1)
            def get_common_mass_axis(self):
                return np.linspace(100, 1000, 5000)
            def iter_spectra(self, batch_size=None):
                for i in range(5):
                    coords = (i % 10, i // 10, 0)
                    mz = np.array([100.0 + i, 200.0 + i, 300.0 + i])
                    intensity = np.array([1000.0, 500.0, 750.0])
                    yield coords, mz, intensity
            def close(self):
                pass
                
        reader = TestReader(Path("/tmp/test"))
        
        # Test mass bounds (should use fallback)
        mass_bounds = reader.get_mass_bounds()
        print(f"[OK] Mass bounds: {mass_bounds}")
        assert len(mass_bounds) == 2, "Should return min and max"
        
        # Test spatial bounds
        spatial_bounds = reader.get_spatial_bounds()
        print(f"[OK] Spatial bounds: {spatial_bounds}")
        required_keys = {'min_x', 'max_x', 'min_y', 'max_y'}
        assert required_keys.issubset(spatial_bounds.keys()), "Should have all required keys"
        
        # Test memory usage estimation
        memory_usage = reader.get_estimated_memory_usage()
        print(f"[OK] Memory usage estimate: {memory_usage}")
        assert 'total_spectra' in memory_usage, "Should include total spectra"
        
        # Test buffered iteration
        from msiconvert.interpolators.buffer_pool import SpectrumBufferPool
        buffer_pool = SpectrumBufferPool(n_buffers=10, buffer_size=1000)
        
        count = 0
        for buffer in reader.iter_spectra_buffered(buffer_pool):
            count += 1
            assert hasattr(buffer, 'coords'), "Buffer should have coords"
            assert hasattr(buffer, 'actual_size'), "Buffer should have actual_size"
            buffer_pool.return_buffer(buffer)
            
        print(f"[OK] Buffered iteration: {count} spectra processed")
        assert count == 5, "Should process all spectra"
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Reader interpolation methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spatialdata_converter_integration():
    """Test SpatialData converter interpolation integration"""
    print("\n=== Testing SpatialData Converter Integration ===")
    
    try:
        # Skip if SpatialData not available
        try:
            from msiconvert.converters.spatialdata_converter import SpatialDataConverter
        except ImportError as e:
            print(f"[SKIP] SpatialData not available: {e}")
            return True
            
        from msiconvert.config import InterpolationConfig
        
        # Create mock reader
        mock_reader = Mock()
        mock_reader.get_dimensions.return_value = (10, 10, 1)
        mock_reader.get_common_mass_axis.return_value = np.linspace(100, 1000, 1000)
        mock_reader.get_metadata.return_value = {"source": "test"}
        mock_reader.get_pixel_size.return_value = (50.0, 50.0)
        mock_reader.close = Mock()
        
        try:
            converter = SpatialDataConverter(
                reader=mock_reader,
                output_path=Path("/tmp/test.zarr"),
                dataset_id="test"
            )
        except ImportError as e:
            print(f"[SKIP] SpatialData converter not available during instantiation: {e}")
            return True
        
        # Test interpolation config integration
        interp_config = InterpolationConfig(
            enabled=True,
            method="pchip", 
            interpolation_bins=500
        )
        
        # Test that convert_with_interpolation method exists
        assert hasattr(converter, 'convert_with_interpolation'), "Should have interpolation conversion method"
        print("[OK] SpatialData converter has interpolation method")
        
        # Test interpolation writer creation
        test_data_structures = {
            "mode": "3d_volume",
            "sparse_data": Mock(),
            "total_intensity": np.zeros(1000),
            "pixel_count": 0,
            "tic_values": np.zeros((10, 10, 1))
        }
        
        writer = converter._create_interpolation_writer(test_data_structures)
        print("[OK] Created interpolation writer")
        
        # Test writer function
        test_coords = (5, 5, 0)
        test_mass_axis = np.linspace(100, 1000, 1000)
        test_intensities = np.random.rand(1000).astype(np.float32) * 100
        
        # Mock the _get_pixel_index method
        converter._get_pixel_index = Mock(return_value=55)
        converter._dimensions = (10, 10, 1)
        converter._non_empty_pixel_count = 0
        
        writer(test_coords, test_mass_axis, test_intensities)
        print("[OK] Writer function executed successfully")
        
        # Test size reduction calculation
        test_stats = {
            'config': {
                'original_mass_points': 10000,
                'target_bins': 1000
            }
        }
        
        size_reduction = converter._calculate_size_reduction(test_stats)
        print(f"[OK] Size reduction calculation: {size_reduction}")
        assert 'reduction_factor' in size_reduction, "Should calculate reduction factor"
        
        return True
        
    except Exception as e:
        print(f"[FAIL] SpatialData converter integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_mock_integration():
    """Test end-to-end integration with mocked components"""
    print("\n=== Testing End-to-End Mock Integration ===")
    
    try:
        from msiconvert.config import InterpolationConfig
        from msiconvert.interpolators.streaming_engine import InterpolationConfig as StreamingConfig
        from msiconvert.interpolators.physics_models import TOFPhysics
        
        # Test configuration flow
        user_config = InterpolationConfig(
            enabled=True,
            method="pchip",
            interpolation_bins=1000,
            max_workers=8,
            validate_quality=True
        )
        
        print(f"[OK] User config created: {user_config.get_summary()}")
        
        # Test physics model creation
        physics = TOFPhysics(resolution_at_400=10000)
        mass_axis = physics.create_optimal_mass_axis(100, 1000, target_bins=1000)
        print(f"[OK] Physics model created mass axis: {len(mass_axis)} points")
        
        # Test streaming config creation
        streaming_config = StreamingConfig(
            method=user_config.method,
            target_mass_axis=mass_axis,
            n_workers=user_config.max_workers // 2,  # Conservative
            validate_quality=user_config.validate_quality
        )
        
        print(f"[OK] Streaming config: {streaming_config.n_workers} workers, "
              f"{len(streaming_config.target_mass_axis)} target bins")
        
        # Test configuration compatibility
        assert streaming_config.method == user_config.method, "Methods should match"
        assert streaming_config.validate_quality == user_config.validate_quality, "Quality validation should match"
        
        print("[OK] Configuration flow validated")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] End-to-end integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling in integration scenarios"""
    print("\n=== Testing Error Handling ===")
    
    try:
        from msiconvert.core.base_converter import BaseMSIConverter
        from msiconvert.config import InterpolationConfig
        
        # Create converter with reader missing interpolation methods
        mock_reader = Mock()
        mock_reader.get_dimensions.return_value = (10, 10, 1)
        mock_reader.get_common_mass_axis.return_value = np.linspace(100, 1000, 1000)
        
        # Remove interpolation methods to test fallback
        if hasattr(mock_reader, 'get_mass_bounds'):
            delattr(mock_reader, 'get_mass_bounds')
        if hasattr(mock_reader, 'get_spatial_bounds'):
            delattr(mock_reader, 'get_spatial_bounds')
        
        class TestConverter(BaseMSIConverter):
            def _create_data_structures(self):
                return {"test": True}
            def _save_output(self, data_structures):
                return True
        
        converter = TestConverter(mock_reader, Path("/tmp/test"))
        converter.interpolation_config = InterpolationConfig(enabled=True)
        
        # Test that interpolation is disabled when methods are missing
        should_interpolate = converter._should_interpolate()
        print(f"[OK] Should interpolate with missing methods: {should_interpolate}")
        assert not should_interpolate, "Should not interpolate with missing reader methods"
        
        # Test invalid configuration handling
        try:
            invalid_config = InterpolationConfig(
                enabled=True,
                method="invalid_method"
            )
            print("[FAIL] Should have failed with invalid method")
            return False
        except ValueError as e:
            print(f"[OK] Correctly rejected invalid method: {e}")
        
        # Test worker count validation
        try:
            invalid_workers = InterpolationConfig(
                enabled=True,
                min_workers=10,
                max_workers=5  # min > max
            )
            print("[FAIL] Should have failed with invalid worker counts")
            return False
        except ValueError as e:
            print(f"[OK] Correctly rejected invalid worker counts: {e}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 3 integration tests"""
    print("Phase 3 Converter Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_interpolation_config,
        test_base_converter_interpolation,
        test_reader_interpolation_methods,
        test_spatialdata_converter_integration,
        test_end_to_end_mock_integration,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Integration Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("[SUCCESS] All Phase 3 integration tests passed!")
        return True
    else:
        print("[ERROR] Some integration tests failed!")
        return False

if __name__ == "__main__":
    success = main()