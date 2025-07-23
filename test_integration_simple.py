#!/usr/bin/env python3
"""
Simple integration test for interpolation functionality
"""

import sys
from pathlib import Path
import numpy as np

def test_bruker_direct_interpolation():
    """Test Bruker interpolation directly"""
    print("Testing Bruker interpolation with direct engine...")
    
    data_path = Path("data/20231109_PEA_NEDC.d")
    if not data_path.exists():
        print("SKIPPED: Bruker test data not available")
        return True
    
    try:
        from msiconvert.readers.bruker.bruker_reader import BrukerReader
        from msiconvert.interpolators.streaming_engine import StreamingInterpolationEngine, InterpolationConfig
        from msiconvert.interpolators.physics_models import TOFPhysics
        
        # Initialize reader
        reader = BrukerReader(data_path)
        print(f"Loaded Bruker data: {reader.n_spectra} spectra")
        
        # Get mass bounds
        min_mz, max_mz = reader.get_mass_bounds()
        print(f"Mass range: {min_mz:.1f} - {max_mz:.1f} m/z")
        
        # Create physics model and target mass axis
        physics = TOFPhysics(resolution_at_400=10000)
        target_mass_axis = physics.create_optimal_mass_axis(min_mz, max_mz, target_bins=2000)
        print(f"Target mass axis: {len(target_mass_axis)} bins")
        
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
                'max_intensity': float(np.max(intensities))
            })
        
        # Process limited dataset (first 5 spectra for speed)
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
        
        # Get performance stats
        stats = engine.get_performance_stats()
        
        # Validate results
        print(f"Processed: {stats['spectra_written']} spectra")
        print(f"Throughput: {stats['overall_throughput_per_sec']:.1f} spectra/sec")
        print(f"Results collected: {len(results)}")
        
        # Basic assertions
        assert stats['spectra_written'] == 5, f"Expected 5 spectra, got {stats['spectra_written']}"
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        
        # Check that all results have the same number of points (should be target_mass_axis length)
        expected_points = len(target_mass_axis)
        actual_points = [r['n_points'] for r in results]
        print(f"Expected points: {expected_points}, Actual points: {actual_points}")
        assert all(r['n_points'] == expected_points for r in results), f"All spectra should have {expected_points} points, got {actual_points}"
        
        assert all(r['tic'] > 0 for r in results), "All spectra should have positive TIC"
        assert stats['overall_throughput_per_sec'] > 0, "Should have positive throughput"
        
        reader.close()
        print("[SUCCESS] Bruker interpolation test passed")
        return True
        
    except Exception as e:
        print(f"[FAILED] Bruker interpolation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imzml_direct_interpolation():
    """Test ImzML interpolation directly"""
    print("\\nTesting ImzML interpolation with direct engine...")
    
    data_path = Path("data/pea.imzML")
    if not data_path.exists():
        print("SKIPPED: ImzML test data not available")
        return True
    
    try:
        from msiconvert.readers.imzml_reader import ImzMLReader
        from msiconvert.interpolators.streaming_engine import StreamingInterpolationEngine, InterpolationConfig
        from msiconvert.interpolators.physics_models import OrbitrapPhysics
        
        # Initialize reader
        reader = ImzMLReader(data_path)
        dimensions = reader.get_dimensions()
        print(f"Loaded ImzML data: {dimensions[0]} x {dimensions[1]} x {dimensions[2]}")
        
        # Estimate mass range from a few spectra
        mass_values = []
        count = 0
        for coords, mzs, intensities in reader.iter_spectra():
            if len(mzs) > 0:
                mass_values.extend([mzs[0], mzs[-1]])
                count += 1
                if count >= 3:
                    break
        
        if mass_values:
            min_mz, max_mz = min(mass_values), max(mass_values)
        else:
            min_mz, max_mz = 100, 1000
        
        print(f"Mass range: {min_mz:.1f} - {max_mz:.1f} m/z")
        
        # Create physics model for Orbitrap
        physics = OrbitrapPhysics(resolution_at_400=60000)
        target_mass_axis = physics.create_optimal_mass_axis(min_mz, max_mz, target_bins=1000)
        print(f"Target mass axis: {len(target_mass_axis)} bins")
        
        # Create interpolation config
        config = InterpolationConfig(
            method="pchip",
            target_mass_axis=target_mass_axis,
            n_workers=1,
            buffer_size=20,
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
        
        # Process limited dataset (first 3 spectra for speed)
        class LimitedImzMLReader:
            def __init__(self, original_reader, limit=3):
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
        
        limited_reader = LimitedImzMLReader(reader, limit=3)
        
        # Process dataset
        engine.process_dataset(
            reader=limited_reader,
            output_writer=collect_output
        )
        
        # Get performance stats
        stats = engine.get_performance_stats()
        
        # Validate results
        print(f"Processed: {stats['spectra_written']} spectra")
        print(f"Throughput: {stats['overall_throughput_per_sec']:.1f} spectra/sec")
        print(f"Results collected: {len(results)}")
        
        # Basic assertions
        assert stats['spectra_written'] == 3, f"Expected 3 spectra, got {stats['spectra_written']}"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        
        # Check that all results have the same number of points (should be target_mass_axis length)
        expected_points = len(target_mass_axis)
        actual_points = [r['n_points'] for r in results]
        print(f"Expected points: {expected_points}, Actual points: {actual_points}")
        assert all(r['n_points'] == expected_points for r in results), f"All spectra should have {expected_points} points, got {actual_points}"
        
        assert all(r['tic'] >= 0 for r in results), "All spectra should have non-negative TIC"
        assert stats['overall_throughput_per_sec'] > 0, "Should have positive throughput"
        
        reader.close()
        print("[SUCCESS] ImzML interpolation test passed")
        return True
        
    except Exception as e:
        print(f"[FAILED] ImzML interpolation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run integration tests"""
    print("Running interpolation integration tests...")
    print("=" * 60)
    
    results = []
    results.append(test_bruker_direct_interpolation())
    results.append(test_imzml_direct_interpolation())
    
    print("\\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\\n[SUCCESS] All integration tests passed!")
        return 0
    else:
        print("\\n[FAILED] Some integration tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())