#!/usr/bin/env python3
"""
Integration test for Phases 4-5 as specified in the review document.

Tests the complete interpolation pipeline with quality monitoring
and converter integration, matching the review document's test structure.
"""

import sys
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_end_to_end_interpolation_pipeline():
    """Test complete interpolation pipeline as specified in review document"""
    print("=== Testing End-to-End Interpolation Pipeline ===")
    
    try:
        from msiconvert.interpolators.streaming_engine import (
            StreamingInterpolationEngine, InterpolationConfig
        )
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor
        from msiconvert.interpolators.physics_models import TOFPhysics
        
        # Setup interpolation config (matching review document)
        physics_model = TOFPhysics(resolution_at_400=10000)
        target_mass_axis = physics_model.create_optimal_mass_axis(100, 1000, target_bins=1000)
        
        config = InterpolationConfig(
            method="pchip",
            target_mass_axis=target_mass_axis,
            n_workers=4,
            buffer_size=100,
            validate_quality=True
        )
        
        print(f"[OK] Configuration created: method={config.method}, "
              f"bins={len(config.target_mass_axis)}, workers={config.n_workers}")
        
        # Create streaming engine
        engine = StreamingInterpolationEngine(config)
        
        # Create mock reader with test data
        mock_reader = Mock()
        mock_reader.get_dimensions.return_value = (10, 10, 1)
        mock_reader.get_metadata.return_value = {"test": True}
        
        # Mock spectral data
        def mock_iter_spectra_buffered(buffer_pool):
            for i in range(5):  # 5 test spectra
                buffer = buffer_pool.get_buffer()
                buffer.coords = (i % 3, i // 3, 0)
                
                # Create test spectrum
                mz = np.array([150.0 + i*50, 250.0 + i*50, 350.0 + i*50])
                intensity = np.array([1000.0, 800.0, 600.0]) * (1.0 + i*0.1)
                
                buffer.fill(mz, intensity.astype(np.float32))
                yield buffer
                
        mock_reader.iter_spectra_buffered = mock_iter_spectra_buffered
        
        # Create output writer
        output_data = []
        def mock_writer(coords, mass_axis, intensities):
            output_data.append({
                'coords': coords,
                'n_points': len(intensities),
                'tic': np.sum(intensities)
            })
            
        # Process dataset
        stats = engine.process_dataset(
            reader=mock_reader,
            output_writer=mock_writer
        )
        
        print(f"[OK] Pipeline processed {stats['spectra_written']} spectra")
        print(f"[OK] Quality summary: {stats.get('quality_summary', {})}")
        
        # Verify results
        assert stats['spectra_written'] == 5
        assert len(output_data) == 5
        assert 'quality_summary' in stats
        
        # Check that all spectra were written with correct mass axis length
        for spectrum in output_data:
            assert spectrum['n_points'] == len(target_mass_axis)
            assert spectrum['tic'] > 0  # Non-zero intensity
        
        return True
        
    except Exception as e:
        print(f"[FAIL] End-to-end pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_converter_integration():
    """Test BaseMSIConverter interpolation integration"""
    print("\\n=== Testing Converter Integration ===")
    
    try:
        from msiconvert.core.base_converter import BaseMSIConverter
        from msiconvert.config import InterpolationConfig
        from msiconvert.interpolators.bounds_detector import BoundsInfo
        
        # Create test converter class
        class TestConverter(BaseMSIConverter):
            def _create_data_structures(self):
                return {"test_data": []}
            
            def _save_output(self, data_structures):
                return True
                
            def _create_interpolation_writer(self, data_structures):
                def writer(coords, mass_axis, intensities):
                    data_structures["test_data"].append({
                        'coords': coords,
                        'mass_axis_length': len(mass_axis),
                        'intensities_length': len(intensities)
                    })
                return writer
                
            def _finalize_interpolated_output(self, data_structures, stats):
                pass
        
        # Create mock reader with interpolation support
        mock_reader = Mock()
        mock_reader.get_dimensions.return_value = (5, 5, 1)
        mock_reader.get_metadata.return_value = {
            'instrument': {'type': 'tof', 'resolution_at_400': 10000}
        }
        mock_reader.get_mass_bounds.return_value = (100.0, 1000.0)
        mock_reader.get_spatial_bounds.return_value = {
            'min_x': 0, 'max_x': 4, 'min_y': 0, 'max_y': 4
        }
        mock_reader.get_estimated_memory_usage.return_value = {
            'total_spectra': 25, 'avg_spectrum_points': 1000
        }
        
        # Mock buffered iteration
        def mock_iter_buffered(buffer_pool):
            for i in range(3):  # 3 test spectra
                buffer = buffer_pool.get_buffer()
                buffer.coords = (i, 0, 0)
                mz = np.array([200.0, 400.0, 600.0])
                intensity = np.array([1000.0, 800.0, 600.0])
                buffer.fill(mz, intensity.astype(np.float32))
                yield buffer
                
        mock_reader.iter_spectra_buffered = mock_iter_buffered
        
        # Create converter
        converter = TestConverter(mock_reader, Path("/tmp/test_output"))
        
        # Test should_interpolate
        assert not converter._should_interpolate()  # No config set
        print("[OK] _should_interpolate returns False without config")
        
        # Set interpolation config
        converter.interpolation_config = InterpolationConfig(
            enabled=True,
            method="pchip",
            interpolation_bins=500
        )
        
        # Mock bounds detection
        with patch('msiconvert.interpolators.bounds_detector.detect_bounds_from_reader') as mock_bounds:
            mock_bounds.return_value = BoundsInfo(
                min_mz=100.0, max_mz=1000.0, 
                min_x=0, max_x=4, min_y=0, max_y=4,
                n_spectra=25
            )
            
            assert converter._should_interpolate()  # Should work with config
            print("[OK] _should_interpolate returns True with proper config")
            
            # Test setup interpolation
            interp_config = converter._setup_interpolation()
            assert interp_config is not None
            assert interp_config.method == "pchip"
            assert len(interp_config.target_mass_axis) > 0
            print(f"[OK] Setup interpolation: {len(interp_config.target_mass_axis)} bins")
            
            # Test interpolation conversion
            result = converter._convert_with_interpolation()
            print(f"[OK] Interpolation conversion result: {result}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Converter integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spatialdata_converter_interpolation():
    """Test SpatialData converter interpolation methods"""
    print("\\n=== Testing SpatialData Converter Interpolation ===")
    
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
        mock_reader.get_dimensions.return_value = (3, 3, 1)
        mock_reader.get_common_mass_axis.return_value = np.linspace(100, 1000, 5000)
        mock_reader.get_metadata.return_value = {"source": "test"}
        mock_reader.get_pixel_size.return_value = (50.0, 50.0)
        mock_reader.close = Mock()
        
        # Create converter
        converter = SpatialDataConverter(
            reader=mock_reader,
            output_path=Path("/tmp/test.zarr"),
            dataset_id="test"
        )
        
        # Test interpolation config
        interp_config = InterpolationConfig(
            enabled=True,
            method="pchip",
            interpolation_bins=1000
        )
        
        # Test that convert_with_interpolation method exists
        assert hasattr(converter, 'convert_with_interpolation')
        print("[OK] SpatialData converter has interpolation method")
        
        # Create data structures
        converter._dimensions = (3, 3, 1)
        converter._common_mass_axis = np.linspace(100, 1000, 1000)
        data_structures = converter._create_data_structures()
        
        # Test interpolation writer creation
        writer = converter._create_interpolation_writer(data_structures)
        print("[OK] Created interpolation writer")
        
        # Test writer function
        test_coords = (1, 1, 0)
        test_mass_axis = np.linspace(100, 1000, 1000)
        test_intensities = np.random.rand(1000).astype(np.float32) * 100
        
        # Mock pixel index calculation
        converter._get_pixel_index = Mock(return_value=4)
        converter._non_empty_pixel_count = 0
        
        writer(test_coords, test_mass_axis, test_intensities)
        print("[OK] Writer function executed successfully")
        
        # Verify data was written
        assert converter._non_empty_pixel_count > 0
        
        return True
        
    except Exception as e:
        print(f"[FAIL] SpatialData converter interpolation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_integration_with_streaming():
    """Test quality monitoring integration with streaming engine"""
    print("\\n=== Testing Quality Integration with Streaming ===")
    
    try:
        from msiconvert.interpolators.streaming_engine import StreamingInterpolationEngine
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor
        from msiconvert.interpolators.physics_models import TOFPhysics
        
        # Create config with quality validation enabled
        physics_model = TOFPhysics(resolution_at_400=10000)
        target_mass_axis = physics_model.create_optimal_mass_axis(100, 1000, target_bins=500)
        
        config = Mock()
        config.method = "pchip"
        config.target_mass_axis = target_mass_axis
        config.n_workers = 2
        config.buffer_size = 50
        config.validate_quality = True
        config.max_memory_gb = 1.0
        config.use_streaming = True
        config.adaptive_workers = False
        
        # Create engine
        engine = StreamingInterpolationEngine(config)
        
        # Verify quality monitor was created
        assert hasattr(engine, 'quality_monitor')
        assert engine.quality_monitor is not None
        print("[OK] Quality monitor integrated with streaming engine")
        
        # Create simple mock reader
        mock_reader = Mock()
        
        def mock_buffered_iter(buffer_pool):
            # Generate 3 test spectra
            for i in range(3):
                buffer = buffer_pool.get_buffer()
                buffer.coords = (i, 0, 0)
                
                # Create test spectrum
                mz = np.array([200.0, 400.0, 600.0])
                intensity = np.array([1000.0, 500.0, 750.0])
                buffer.fill(mz, intensity.astype(np.float32))
                
                yield buffer
                
        mock_reader.iter_spectra_buffered = mock_buffered_iter
        
        # Process with quality monitoring
        results = []
        def test_writer(coords, mass_axis, intensities):
            results.append({
                'coords': coords,
                'n_points': len(intensities),
                'tic': np.sum(intensities)
            })
            
        stats = engine.process_dataset(
            reader=mock_reader,
            output_writer=test_writer
        )
        
        print(f"[OK] Processed {stats['spectra_written']} spectra with quality monitoring")
        
        # Check quality statistics
        if 'quality_summary' in stats:
            quality = stats['quality_summary']
            print(f"[OK] Quality summary: {quality.get('total_samples', 0)} samples monitored")
            
        assert stats['spectra_written'] == 3
        assert len(results) == 3
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Quality integration test failed: {e}")
        import traceback
        traceback.print_exc() 
        return False

def main():
    """Run all Phase 4-5 integration tests"""
    print("Phase 4-5 Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_end_to_end_interpolation_pipeline,
        test_converter_integration,
        test_spatialdata_converter_interpolation,
        test_quality_integration_with_streaming
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\\n" + "=" * 50)
    print("Integration Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("[SUCCESS] All Phase 4-5 integration tests passed!")
        return True
    else:
        print("[ERROR] Some integration tests failed!")
        return False

if __name__ == "__main__":
    success = main()