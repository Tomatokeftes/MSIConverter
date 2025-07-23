#!/usr/bin/env python3
"""
Test suite for Phase 2 memory management and buffer pool
"""

import sys
import numpy as np
from pathlib import Path
import threading
import time

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_spectrum_buffer_pool():
    """Test spectrum buffer pool functionality"""
    print("=== Testing Spectrum Buffer Pool ===")
    
    try:
        from msiconvert.interpolators.buffer_pool import SpectrumBufferPool
        
        # Initialize buffer pool
        pool = SpectrumBufferPool(n_buffers=10, buffer_size=1000)
        print(f"[OK] Buffer pool initialized with 10 buffers")
        
        # Test getting and returning buffers
        buffers = []
        for i in range(5):
            buffer = pool.get_buffer()
            buffers.append(buffer)
            print(f"[OK] Got buffer {buffer.buffer_id}")
            
        # Test buffer usage
        test_mz = np.array([100.0, 200.0, 300.0])
        test_intensity = np.array([10.0, 20.0, 30.0])
        
        buffer = buffers[0]
        buffer.coords = (10, 20, 0)
        buffer.fill(test_mz, test_intensity)
        
        retrieved_mz, retrieved_intensity = buffer.get_data()
        assert np.array_equal(retrieved_mz, test_mz), "Retrieved m/z should match"
        assert np.array_equal(retrieved_intensity, test_intensity), "Retrieved intensity should match"
        print(f"[OK] Buffer fill and retrieve works correctly")
        
        # Return buffers
        for buffer in buffers:
            pool.return_buffer(buffer)
        
        # Check stats
        stats = pool.get_stats()
        print(f"[OK] Pool stats: {stats['free_buffers']}/{stats['total_buffers']} free, "
              f"{stats['memory_usage_mb']:.1f} MB")
        
        # Test pool exhaustion (emergency allocation)
        all_buffers = []
        for i in range(15):  # More than pool size
            buffer = pool.get_buffer()
            all_buffers.append(buffer)
            
        stats_after = pool.get_stats()
        print(f"[OK] Emergency allocations: {stats_after['emergency_allocations']}")
        
        # Return all buffers
        for buffer in all_buffers:
            pool.return_buffer(buffer)
            
        return True
        
    except Exception as e:
        print(f"[FAIL] Buffer pool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_buffer_pool_threading():
    """Test buffer pool thread safety"""
    print("\n=== Testing Buffer Pool Thread Safety ===")
    
    try:
        from msiconvert.interpolators.buffer_pool import SpectrumBufferPool
        
        pool = SpectrumBufferPool(n_buffers=20, buffer_size=1000)
        results = []
        errors = []
        
        def worker_thread(thread_id):
            """Worker thread that gets and returns buffers"""
            try:
                local_buffers = []
                
                # Get some buffers
                for i in range(5):
                    buffer = pool.get_buffer()
                    local_buffers.append(buffer)
                    
                    # Fill with test data
                    test_mz = np.array([float(thread_id * 100 + i)])
                    test_intensity = np.array([float(thread_id * 10 + i)])
                    buffer.fill(test_mz, test_intensity)
                    
                # Small delay to simulate work
                time.sleep(0.01)
                
                # Return buffers
                for buffer in local_buffers:
                    pool.return_buffer(buffer)
                    
                results.append(f"Thread {thread_id} completed successfully")
                
            except Exception as e:
                errors.append(f"Thread {thread_id} failed: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        print(f"[OK] Threading test: {len(results)} threads completed, {len(errors)} errors")
        
        if errors:
            for error in errors:
                print(f"    Error: {error}")
                
        stats = pool.get_stats()
        print(f"[OK] Final stats: {stats['free_buffers']}/{stats['total_buffers']} free")
        
        return len(errors) == 0
        
    except Exception as e:
        print(f"[FAIL] Threading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adaptive_memory_manager():
    """Test adaptive memory manager"""
    print("\n=== Testing Adaptive Memory Manager ===")
    
    try:
        from msiconvert.interpolators.buffer_pool import AdaptiveMemoryManager
        
        # Initialize memory manager
        manager = AdaptiveMemoryManager(target_memory_gb=4.0)
        print(f"[OK] Memory manager initialized")
        
        # Test memory statistics
        stats = manager.get_memory_stats()
        print(f"[OK] Memory stats: {stats['available_memory_gb']:.1f} GB available, "
              f"{stats['process_memory_gb']:.1f} GB process")
        print(f"    PSUtil available: {stats['psutil_available']}")
        
        # Test optimal buffer calculation
        optimal_buffers = manager.calculate_optimal_buffer_count(
            spectrum_size=100000,
            n_workers=4
        )
        print(f"[OK] Optimal buffer count for 100k spectrum, 4 workers: {optimal_buffers}")
        
        # Test memory pressure detection
        pressure = manager.check_memory_pressure()
        print(f"[OK] Memory pressure check: {pressure}")
        
        # Test with different spectrum sizes
        sizes = [10000, 100000, 300000]
        for size in sizes:
            buffers = manager.calculate_optimal_buffer_count(size, 8)
            print(f"[OK] Size {size:>6}: {buffers:>3} buffers recommended")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Memory manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interpolated_spectrum():
    """Test InterpolatedSpectrum container"""
    print("\n=== Testing InterpolatedSpectrum Container ===")
    
    try:
        from msiconvert.interpolators.buffer_pool import InterpolatedSpectrum
        
        # Create test data
        coords = (10, 20, 0)
        intensities = np.array([100.0, 50.0, 75.0], dtype=np.float32)
        interpolation_time = 0.05
        quality_metrics = {"tic_ratio": 0.98, "peak_preservation": 0.95}
        
        # Create InterpolatedSpectrum
        spectrum = InterpolatedSpectrum(
            coords=coords,
            intensities=intensities,
            interpolation_time=interpolation_time,
            quality_metrics=quality_metrics
        )
        
        print(f"[OK] InterpolatedSpectrum created: {spectrum}")
        print(f"    Coordinates: {spectrum.coords}")
        print(f"    Intensities shape: {spectrum.intensities.shape}")
        print(f"    Quality metrics: {spectrum.quality_metrics}")
        
        # Test without quality metrics
        spectrum_simple = InterpolatedSpectrum(
            coords=(5, 10, 0),
            intensities=np.array([25.0, 30.0])
        )
        
        print(f"[OK] Simple spectrum: {spectrum_simple}")
        print(f"    Default quality metrics: {spectrum_simple.quality_metrics}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] InterpolatedSpectrum test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test memory efficiency of buffer operations"""
    print("\n=== Testing Memory Efficiency ===")
    
    try:
        from msiconvert.interpolators.buffer_pool import SpectrumBufferPool
        import tracemalloc
        
        # Start memory tracing
        tracemalloc.start()
        
        # Create buffer pool
        pool = SpectrumBufferPool(n_buffers=50, buffer_size=100000)
        
        # Simulate intensive buffer usage
        for iteration in range(100):
            # Get buffer
            buffer = pool.get_buffer()
            
            # Fill with random data
            mz_data = np.random.rand(1000) * 1000
            intensity_data = np.random.rand(1000).astype(np.float32) * 10000
            buffer.fill(mz_data, intensity_data)
            
            # Retrieve data (simulating processing)
            retrieved_mz, retrieved_intensity = buffer.get_data()
            
            # Return buffer
            pool.return_buffer(buffer)
            
        # Check memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"[OK] Memory efficiency test completed")
        print(f"    Current memory: {current / 1e6:.1f} MB")
        print(f"    Peak memory: {peak / 1e6:.1f} MB")
        
        # Check pool stats
        final_stats = pool.get_stats()
        print(f"    Buffer pool: {final_stats['free_buffers']}/{final_stats['total_buffers']} free")
        print(f"    Emergency allocations: {final_stats['emergency_allocations']}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Memory efficiency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 2 memory management tests"""
    print("Phase 2 Memory Management Test Suite")
    print("=" * 50)
    
    tests = [
        test_spectrum_buffer_pool,
        test_buffer_pool_threading,
        test_adaptive_memory_manager,
        test_interpolated_spectrum,
        test_memory_efficiency
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Memory Management Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("[SUCCESS] All Phase 2 memory tests passed!")
        return True
    else:
        print("[ERROR] Some memory tests failed!")
        return False

if __name__ == "__main__":
    success = main()