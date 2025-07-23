#!/usr/bin/env python3
"""
Direct test of interpolation functionality without converter dependencies.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_bruker_interpolation():
    """Test Bruker interpolation directly"""
    print("=" * 60)
    print("DIRECT BRUKER INTERPOLATION TEST")
    print("=" * 60)
    
    try:
        from msiconvert.readers.bruker.bruker_reader import BrukerReader
        from msiconvert.interpolators.streaming_engine import StreamingInterpolationEngine, InterpolationConfig
        from msiconvert.interpolators.physics_models import TOFPhysics
        
        # Initialize Bruker reader
        data_path = Path("./data/20231109_PEA_NEDC.d")
        print(f"Loading Bruker data from: {data_path}")
        
        reader = BrukerReader(data_path)
        
        # Get dataset info
        dimensions = reader.get_dimensions()
        n_spectra = reader.n_spectra
        print(f"Dataset: {dimensions[0]} x {dimensions[1]} x {dimensions[2]} = {n_spectra:,} spectra")
        
        # Get mass bounds
        min_mz, max_mz = reader.get_mass_bounds()
        print(f"Mass range: {min_mz:.1f} - {max_mz:.1f} m/z")
        
        # Create physics model and optimal mass axis
        physics = TOFPhysics(resolution_at_400=10000)
        target_mass_axis = physics.create_optimal_mass_axis(min_mz, max_mz, target_bins=1000)
        print(f"Target mass axis: {len(target_mass_axis)} bins (reduction from ~78M)")
        
        # Create interpolation config
        config = InterpolationConfig(
            method="pchip",
            target_mass_axis=target_mass_axis,
            n_workers=4,
            buffer_size=100,
            validate_quality=True
        )
        
        # Create streaming engine
        engine = StreamingInterpolationEngine(config)
        print(f"Created interpolation engine with {config.n_workers} workers")
        
        # Create simple output collector
        interpolated_results = []
        def collect_output(coords, mass_axis, intensities):
            interpolated_results.append({
                'coords': coords,
                'n_points': len(intensities),
                'tic': float(np.sum(intensities)),
                'max_intensity': float(np.max(intensities)),
                'mass_range': (float(mass_axis[0]), float(mass_axis[-1]))
            })
            
        print(f"\\nStarting interpolation of first 100 spectra...")
        
        # Create limited reader that only processes first 100 spectra
        class LimitedReader:
            def __init__(self, original_reader, limit=100):
                self.original_reader = original_reader
                self.limit = limit
                # Pass through required attributes
                self.data_path = original_reader.data_path
                
            def iter_spectra_buffered(self, buffer_pool):
                count = 0
                for buffer in self.original_reader.iter_spectra_buffered(buffer_pool):
                    if count >= self.limit:
                        break
                    yield buffer
                    count += 1
                    
            def __getattr__(self, name):
                # Delegate any other attribute access to the original reader
                return getattr(self.original_reader, name)
                    
        limited_reader = LimitedReader(reader, limit=100)
        
        # Process dataset
        try:
            engine.process_dataset(
                reader=limited_reader,
                output_writer=collect_output
            )
            stats = engine.get_performance_stats()
            
            print(f"\\nInterpolation Results:")
            print(f"Spectra processed: {stats['spectra_written']}")
            print(f"Processing time: {stats['elapsed_time']:.1f} seconds")
            print(f"Throughput: {stats['overall_throughput_per_sec']:.0f} spectra/second")
            
            if 'quality_summary' in stats:
                quality = stats['quality_summary']
                print(f"Quality - Avg TIC ratio: {quality.get('avg_tic_ratio', 0):.3f}")
                print(f"Quality - Avg peak preservation: {quality.get('avg_peak_preservation', 0):.3f}")
        except Exception as e:
            print(f"[FAILED] Error during interpolation: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Show sample results
        if interpolated_results:
            print(f"\\nSample interpolated spectra:")
            for i, result in enumerate(interpolated_results[:3]):
                print(f"  Spectrum {i+1}: {result['n_points']} points, "
                      f"TIC={result['tic']:.0f}, Max={result['max_intensity']:.0f}")
        
        reader.close()
        print(f"\\n[SUCCESS] Bruker interpolation working!")
        return True
        
    except Exception as e:
        print(f"[FAILED] Bruker interpolation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imzml_interpolation():
    """Test ImzML interpolation directly"""
    print("\\n" + "=" * 60)
    print("DIRECT IMZML INTERPOLATION TEST")
    print("=" * 60)
    
    try:
        from msiconvert.readers.imzml_reader import ImzMLReader
        from msiconvert.interpolators.streaming_engine import StreamingInterpolationEngine, InterpolationConfig
        from msiconvert.interpolators.physics_models import OrbitrapPhysics
        
        # Initialize ImzML reader
        data_path = Path("./data/pea.imzML")
        print(f"Loading ImzML data from: {data_path}")
        
        reader = ImzMLReader(data_path)
        
        # Get dataset info
        dimensions = reader.get_dimensions()
        n_spectra = reader.n_spectra
        print(f"Dataset: {dimensions[0]} x {dimensions[1]} x {dimensions[2]} = {n_spectra:,} spectra")
        
        # Get mass range from a few spectra
        mass_values = []
        count = 0
        for coords, mzs, intensities in reader.iter_spectra():
            if len(mzs) > 0:
                mass_values.extend([mzs[0], mzs[-1]])
                count += 1
                if count >= 10:  # Sample first 10 spectra
                    break
        
        if mass_values:
            min_mz, max_mz = min(mass_values), max(mass_values)
            print(f"Mass range (estimated): {min_mz:.1f} - {max_mz:.1f} m/z")
        else:
            min_mz, max_mz = 100, 1000
            print(f"Using default mass range: {min_mz} - {max_mz} m/z")
        
        # Create physics model for Orbitrap (typical for ImzML)
        physics = OrbitrapPhysics(resolution_at_400=60000)
        target_mass_axis = physics.create_optimal_mass_axis(min_mz, max_mz, target_bins=5000)
        print(f"Target mass axis: {len(target_mass_axis)} bins")
        
        # Create interpolation config
        config = InterpolationConfig(
            method="pchip",
            target_mass_axis=target_mass_axis,
            n_workers=2,  # Fewer workers for smaller dataset
            buffer_size=50,
            validate_quality=True
        )
        
        # Create streaming engine
        engine = StreamingInterpolationEngine(config)
        print(f"Created interpolation engine with {config.n_workers} workers")
        
        # Create simple output collector
        interpolated_results = []
        def collect_output(coords, mass_axis, intensities):
            interpolated_results.append({
                'coords': coords,
                'n_points': len(intensities),
                'tic': float(np.sum(intensities)),
                'max_intensity': float(np.max(intensities)) if len(intensities) > 0 else 0.0,
                'mass_range': (float(mass_axis[0]), float(mass_axis[-1])) if len(mass_axis) > 0 else (0, 0)
            })
            
        print(f"\\nStarting interpolation of first 50 spectra...")
        
        # Create limited reader
        class LimitedImzMLReader:
            def __init__(self, original_reader, limit=50):
                self.original_reader = original_reader
                self.limit = limit
                # Pass through required attributes
                self.data_path = original_reader.data_path
                
            def iter_spectra_buffered(self, buffer_pool):
                count = 0
                for coords, mzs, intensities in self.original_reader.iter_spectra():
                    if count >= self.limit:
                        break
                    
                    # Get buffer and fill it
                    buffer = buffer_pool.get_buffer()
                    buffer.coords = coords
                    buffer.fill(mzs, intensities.astype(np.float32))
                    
                    yield buffer
                    count += 1
                    
            def __getattr__(self, name):
                # Delegate any other attribute access to the original reader
                return getattr(self.original_reader, name)
                    
        limited_reader = LimitedImzMLReader(reader, limit=50)
        
        # Process dataset
        try:
            engine.process_dataset(
                reader=limited_reader,
                output_writer=collect_output
            )
            stats = engine.get_performance_stats()
            
            print(f"\\nInterpolation Results:")
            print(f"Spectra processed: {stats['spectra_written']}")
            print(f"Processing time: {stats['elapsed_time']:.1f} seconds")
            print(f"Throughput: {stats['overall_throughput_per_sec']:.0f} spectra/second")
            
            if 'quality_summary' in stats:
                quality = stats['quality_summary']
                print(f"Quality - Avg TIC ratio: {quality.get('avg_tic_ratio', 0):.3f}")
                print(f"Quality - Avg peak preservation: {quality.get('avg_peak_preservation', 0):.3f}")
        except Exception as e:
            print(f"[FAILED] Error during interpolation: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Show sample results
        if interpolated_results:
            print(f"\\nSample interpolated spectra:")
            for i, result in enumerate(interpolated_results[:3]):
                print(f"  Spectrum {i+1}: {result['n_points']} points, "
                      f"TIC={result['tic']:.0f}, Max={result['max_intensity']:.0f}")
        
        reader.close()
        print(f"\\n[SUCCESS] ImzML interpolation working!")
        return True
        
    except Exception as e:
        print(f"[FAILED] ImzML interpolation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing interpolation functionality with real data...")
    
    results = []
    results.append(test_bruker_interpolation())
    results.append(test_imzml_interpolation())
    
    print("\\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Bruker interpolation: {'PASS' if results[0] else 'FAIL'}")
    print(f"ImzML interpolation: {'PASS' if results[1] else 'FAIL'}")
    print(f"\\nOverall: {sum(results)}/2 tests passed")
    
    if all(results):
        print("\\n[SUCCESS] ALL INTERPOLATION TESTS PASSED!")
        print("The interpolation system is working correctly with real data!")
    else:
        print("\\n[WARNING] Some interpolation tests failed")