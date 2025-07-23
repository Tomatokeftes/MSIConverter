#!/usr/bin/env python3
"""
End-to-end integration tests for interpolation functionality.
Tests the complete interpolation pipeline with real data.
"""

import tempfile
import subprocess
import sys
from pathlib import Path
import numpy as np

def test_bruker_interpolation_with_bins():
    """Test Bruker interpolation using bin-based specification"""
    # Check if test data exists
    data_path = Path("data/20231109_PEA_NEDC.d")
    if not data_path.exists():
        print("SKIPPED:("Bruker test data not available")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_bruker_bins.zarr"
        
        # Test direct interpolation engine
        from msiconvert.readers.bruker.bruker_reader import BrukerReader
        from msiconvert.interpolators.streaming_engine import StreamingInterpolationEngine, InterpolationConfig
        from msiconvert.interpolators.physics_models import TOFPhysics
        
        # Initialize reader
        reader = BrukerReader(data_path)
        
        # Get mass bounds
        min_mz, max_mz = reader.get_mass_bounds()
        
        # Create physics model and target mass axis
        physics = TOFPhysics(resolution_at_400=10000)
        target_mass_axis = physics.create_optimal_mass_axis(min_mz, max_mz, target_bins=5000)
        
        # Create interpolation config
        config = InterpolationConfig(
            method="pchip",
            target_mass_axis=target_mass_axis,
            n_workers=2,
            buffer_size=50,
            validate_quality=True
        )
        
        # Create streaming engine
        engine = StreamingInterpolationEngine(config)
        
        # Collect results
        results = []
        def collect_output(coords, mass_axis, intensities):
            results.append({
                'coords': coords,
                'n_points': len(intensities),
                'tic': float(np.sum(intensities)),
                'max_intensity': float(np.max(intensities))
            })
        
        # Process limited dataset (first 10 spectra for speed)
        class LimitedReader:
            def __init__(self, original_reader, limit=10):
                self.original_reader = original_reader
                self.limit = limit
                self.data_path = original_reader.data_path
                
            def iter_spectra_buffered(self, buffer_pool):
                count = 0
                for buffer in self.original_reader.iter_spectra_buffered(buffer_pool):
                    if count >= self.limit:
                        break
                    yield buffer
                    count += 1
                    
            def __getattr__(self, name):
                return getattr(self.original_reader, name)
        
        limited_reader = LimitedReader(reader, limit=10)
        
        # Process dataset
        engine.process_dataset(
            reader=limited_reader,
            output_writer=collect_output
        )
        
        # Get performance stats
        stats = engine.get_performance_stats()
        
        # Assertions
        assert stats['spectra_written'] == 10, f"Expected 10 spectra, got {stats['spectra_written']}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert all(r['n_points'] == 5000 for r in results), "All spectra should have 5000 points"
        assert all(r['tic'] > 0 for r in results), "All spectra should have positive TIC"
        assert stats['overall_throughput_per_sec'] > 0, "Should have positive throughput"
        
        reader.close()

def test_imzml_interpolation_with_width():
    """Test ImzML interpolation using width-based specification"""
    # Check if test data exists
    data_path = Path("data/pea.imzML")
    if not data_path.exists():
        print("SKIPPED:("ImzML test data not available")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_imzml_width.zarr"
        
        # Test direct interpolation engine
        from msiconvert.readers.imzml_reader import ImzMLReader
        from msiconvert.interpolators.streaming_engine import StreamingInterpolationEngine, InterpolationConfig
        from msiconvert.interpolators.physics_models import OrbitrapPhysics
        
        # Initialize reader
        reader = ImzMLReader(data_path)
        
        # Estimate mass range from a few spectra
        mass_values = []
        count = 0
        for coords, mzs, intensities in reader.iter_spectra():
            if len(mzs) > 0:
                mass_values.extend([mzs[0], mzs[-1]])
                count += 1
                if count >= 5:
                    break
        
        if mass_values:
            min_mz, max_mz = min(mass_values), max(mass_values)
        else:
            min_mz, max_mz = 100, 1000
        
        # Create physics model for Orbitrap
        physics = OrbitrapPhysics(resolution_at_400=60000)
        target_mass_axis = physics.create_optimal_mass_axis(
            min_mz, max_mz, width_at_mz=(0.1, 400.0)
        )
        
        # Create interpolation config
        config = InterpolationConfig(
            method="pchip",
            target_mass_axis=target_mass_axis,
            n_workers=2,
            buffer_size=30,
            validate_quality=True
        )
        
        # Create streaming engine
        engine = StreamingInterpolationEngine(config)
        
        # Collect results
        results = []
        def collect_output(coords, mass_axis, intensities):
            results.append({
                'coords': coords,
                'n_points': len(intensities),
                'tic': float(np.sum(intensities)),
                'max_intensity': float(np.max(intensities)) if len(intensities) > 0 else 0.0
            })
        
        # Process limited dataset (first 5 spectra for speed)
        class LimitedImzMLReader:
            def __init__(self, original_reader, limit=5):
                self.original_reader = original_reader
                self.limit = limit
                self.data_path = original_reader.data_path
                
            def iter_spectra_buffered(self, buffer_pool):
                count = 0
                for coords, mzs, intensities in self.original_reader.iter_spectra():
                    if count >= self.limit:
                        break
                    
                    buffer = buffer_pool.get_buffer()
                    buffer.coords = coords
                    buffer.fill(mzs, intensities.astype(np.float32))
                    
                    yield buffer
                    count += 1
                    
            def __getattr__(self, name):
                return getattr(self.original_reader, name)
        
        limited_reader = LimitedImzMLReader(reader, limit=5)
        
        # Process dataset
        engine.process_dataset(
            reader=limited_reader,
            output_writer=collect_output
        )
        
        # Get performance stats
        stats = engine.get_performance_stats()
        
        # Assertions
        assert stats['spectra_written'] == 5, f"Expected 5 spectra, got {stats['spectra_written']}"
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        assert all(r['n_points'] == len(target_mass_axis) for r in results), f"All spectra should have {len(target_mass_axis)} points"
        assert all(r['tic'] >= 0 for r in results), "All spectra should have non-negative TIC"
        assert stats['overall_throughput_per_sec'] > 0, "Should have positive throughput"
        
        reader.close()

def test_quality_metrics_validation():
    """Test that quality metrics are properly calculated and within expected ranges"""
    # Check if test data exists
    data_path = Path("data/20231109_PEA_NEDC.d")
    if not data_path.exists():
        print("SKIPPED:("Bruker test data not available")
    
    from msiconvert.readers.bruker.bruker_reader import BrukerReader
    from msiconvert.interpolators.streaming_engine import StreamingInterpolationEngine, InterpolationConfig
    from msiconvert.interpolators.physics_models import TOFPhysics
    
    # Initialize reader
    reader = BrukerReader(data_path)
    
    # Get mass bounds
    min_mz, max_mz = reader.get_mass_bounds()
    
    # Create physics model and target mass axis
    physics = TOFPhysics(resolution_at_400=10000)
    target_mass_axis = physics.create_optimal_mass_axis(min_mz, max_mz, target_bins=2000)
    
    # Create interpolation config with quality validation
    config = InterpolationConfig(
        method="pchip",
        target_mass_axis=target_mass_axis,
        n_workers=1,  # Single worker for deterministic results
        buffer_size=20,
        validate_quality=True
    )
    
    # Create streaming engine
    engine = StreamingInterpolationEngine(config)
    
    # Collect results
    quality_results = []
    def collect_output(coords, mass_axis, intensities):
        quality_results.append({
            'coords': coords,
            'tic': float(np.sum(intensities)),
            'n_peaks': np.sum(intensities > np.max(intensities) * 0.01)  # Count peaks > 1% of max
        })
    
    # Process limited dataset
    class LimitedReader:
        def __init__(self, original_reader, limit=5):
            self.original_reader = original_reader
            self.limit = limit
            self.data_path = original_reader.data_path
            
        def iter_spectra_buffered(self, buffer_pool):
            count = 0
            for buffer in self.original_reader.iter_spectra_buffered(buffer_pool):
                if count >= self.limit:
                    break
                yield buffer
                count += 1
                
        def __getattr__(self, name):
            return getattr(self.original_reader, name)
    
    limited_reader = LimitedReader(reader, limit=5)
    
    # Process dataset
    engine.process_dataset(
        reader=limited_reader,
        output_writer=collect_output
    )
    
    # Get performance stats with quality metrics
    stats = engine.get_performance_stats()
    
    # Assertions for quality metrics
    if 'quality_summary' in stats and stats['quality_summary']:
        quality = stats['quality_summary']
        
        # TIC ratio should be reasonable (not too far from 1.0)
        if 'avg_tic_ratio' in quality:
            tic_ratio = quality['avg_tic_ratio']
            assert 0.1 <= tic_ratio <= 10.0, f"TIC ratio {tic_ratio} seems unreasonable"
        
        # Should have processed some quality samples
        if 'total_samples' in quality:
            assert quality['total_samples'] > 0, "Should have quality samples"
    
    # Basic functionality assertions
    assert stats['spectra_written'] == 5, f"Expected 5 spectra, got {stats['spectra_written']}"
    assert len(quality_results) == 5, f"Expected 5 results, got {len(quality_results)}"
    assert all(r['tic'] >= 0 for r in quality_results), "All TIC values should be non-negative"
    
    reader.close()

def test_interpolation_memory_efficiency():
    """Test that interpolation doesn't consume excessive memory"""
    # Check if test data exists
    data_path = Path("data/20231109_PEA_NEDC.d")
    if not data_path.exists():
        print("SKIPPED:("Bruker test data not available")
    
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    from msiconvert.readers.bruker.bruker_reader import BrukerReader
    from msiconvert.interpolators.streaming_engine import StreamingInterpolationEngine, InterpolationConfig
    from msiconvert.interpolators.physics_models import TOFPhysics
    
    # Initialize reader
    reader = BrukerReader(data_path)
    
    # Get mass bounds
    min_mz, max_mz = reader.get_mass_bounds()
    
    # Create physics model and target mass axis
    physics = TOFPhysics(resolution_at_400=10000)
    target_mass_axis = physics.create_optimal_mass_axis(min_mz, max_mz, target_bins=1000)
    
    # Create interpolation config with memory limits
    config = InterpolationConfig(
        method="pchip",
        target_mass_axis=target_mass_axis,
        n_workers=2,
        buffer_size=20,
        max_memory_gb=1.0,  # Limit to 1GB
        validate_quality=False  # Disable to reduce memory overhead
    )
    
    # Create streaming engine
    engine = StreamingInterpolationEngine(config)
    
    # Simple output collector
    result_count = {'count': 0}
    def collect_output(coords, mass_axis, intensities):
        result_count['count'] += 1
    
    # Process limited dataset
    class LimitedReader:
        def __init__(self, original_reader, limit=20):
            self.original_reader = original_reader
            self.limit = limit
            self.data_path = original_reader.data_path
            
        def iter_spectra_buffered(self, buffer_pool):
            count = 0
            for buffer in self.original_reader.iter_spectra_buffered(buffer_pool):
                if count >= self.limit:
                    break
                yield buffer
                count += 1
                
        def __getattr__(self, name):
            return getattr(self.original_reader, name)
    
    limited_reader = LimitedReader(reader, limit=20)
    
    # Process dataset
    engine.process_dataset(
        reader=limited_reader,
        output_writer=collect_output
    )
    
    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Get performance stats
    stats = engine.get_performance_stats()
    
    # Assertions
    assert result_count['count'] == 20, f"Expected 20 spectra, got {result_count['count']}"
    assert memory_increase < 500, f"Memory increase {memory_increase:.1f}MB seems excessive"
    assert stats['overall_throughput_per_sec'] > 0, "Should have positive throughput"
    
    # Check buffer pool stats
    if 'buffer_pool_stats' in stats:
        buffer_stats = stats['buffer_pool_stats']
        assert buffer_stats['emergency_allocations'] < 50, "Too many emergency allocations"
    
    reader.close()

if __name__ == "__main__":
    # Run tests directly for development
    import sys
    
    print("Running interpolation end-to-end integration tests...")
    
    try:
        test_bruker_interpolation_with_bins()
        print("✓ Bruker interpolation with bins test passed")
    except Exception as e:
        print(f"✗ Bruker interpolation with bins test failed: {e}")
    
    try:
        test_imzml_interpolation_with_width()
        print("✓ ImzML interpolation with width test passed")
    except Exception as e:
        print(f"✗ ImzML interpolation with width test failed: {e}")
    
    try:
        test_quality_metrics_validation()
        print("✓ Quality metrics validation test passed")
    except Exception as e:
        print(f"✗ Quality metrics validation test failed: {e}")
    
    try:
        test_interpolation_memory_efficiency()
        print("✓ Memory efficiency test passed")
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
    
    print("Integration tests completed!")