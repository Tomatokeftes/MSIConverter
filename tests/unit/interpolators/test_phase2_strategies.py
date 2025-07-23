#!/usr/bin/env python3
"""
Test suite for Phase 2 interpolation strategies
"""

import sys
import numpy as np
from pathlib import Path
import time

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_pchip_strategy():
    """Test PCHIP interpolation strategy"""
    print("=== Testing PCHIP Strategy ===")
    
    try:
        from msiconvert.interpolators.strategies import PchipInterpolationStrategy
        
        # Initialize strategy
        strategy = PchipInterpolationStrategy(validate_quality=True)
        print(f"[OK] Strategy initialized: {strategy.name}")
        print(f"[OK] Description: {strategy.description}")
        
        # Test basic interpolation
        mz_old = np.array([100.0, 150.0, 200.0, 250.0])
        intensity_old = np.array([1000.0, 500.0, 800.0, 200.0], dtype=np.float32)
        mz_new = np.linspace(100, 250, 100)
        
        intensity_new = strategy.interpolate_spectrum(mz_old, intensity_old, mz_new)
        print(f"[OK] Basic interpolation: {len(intensity_new)} points generated")
        
        # Verify non-negative values (PCHIP guarantee)
        assert np.all(intensity_new >= 0), "PCHIP should guarantee non-negative values"
        print("[OK] Non-negative values verified")
        
        # Test with validation
        intensity_new, metrics = strategy.interpolate_with_validation(mz_old, intensity_old, mz_new)
        print(f"[OK] With validation: TIC ratio={metrics.tic_ratio:.3f}, "
              f"peaks preserved={metrics.peak_preservation:.3f}")
        
        # Test edge cases
        empty_mz = np.array([])
        empty_intensity = np.array([], dtype=np.float32)
        result_empty = strategy.interpolate_spectrum(empty_mz, empty_intensity, mz_new)
        assert np.all(result_empty == 0), "Empty input should return zeros"
        print("[OK] Empty input handling")
        
        # Single point test
        single_mz = np.array([150.0])
        single_intensity = np.array([1000.0], dtype=np.float32)
        result_single = strategy.interpolate_spectrum(single_mz, single_intensity, mz_new)
        print(f"[OK] Single point handling: {np.sum(result_single > 0)} non-zero points")
        
        # Test performance info
        perf_info = strategy.get_performance_info()
        print(f"[OK] Performance info: {perf_info['monotonic']}, prevents overshooting: {perf_info['prevents_overshooting']}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] PCHIP strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adaptive_strategy():
    """Test adaptive interpolation strategy"""
    print("\n=== Testing Adaptive Strategy ===")
    
    try:
        from msiconvert.interpolators.strategies import AdaptiveInterpolationStrategy
        
        # Test different performance priorities
        for priority in ["speed", "quality", "balanced"]:
            strategy = AdaptiveInterpolationStrategy(performance_priority=priority)
            print(f"[OK] Strategy initialized with {priority} priority")
            
            # Test with sparse data (should use linear)
            sparse_mz = np.array([100.0, 200.0])
            sparse_intensity = np.array([1000.0, 500.0], dtype=np.float32)
            mz_new = np.linspace(100, 200, 50)
            
            result = strategy.interpolate_spectrum(sparse_mz, sparse_intensity, mz_new)
            selection_info = strategy.get_strategy_selection_info(sparse_mz, sparse_intensity, mz_new)
            print(f"[OK] Sparse data -> {selection_info['selected_strategy']} strategy")
            
            # Test with dense data (should use PCHIP for quality/balanced)
            dense_mz = np.linspace(100, 200, 100)
            dense_intensity = np.random.rand(100).astype(np.float32) * 1000
            
            result = strategy.interpolate_spectrum(dense_mz, dense_intensity, mz_new)
            selection_info = strategy.get_strategy_selection_info(dense_mz, dense_intensity, mz_new)
            print(f"[OK] Dense data -> {selection_info['selected_strategy']} strategy")
            print(f"    Reasoning: {selection_info['reasoning']}")
        
        # Test performance info
        perf_info = strategy.get_performance_info()
        print(f"[OK] Adaptive strategy info: {perf_info['adapts_to_data']}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Adaptive strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_factory():
    """Test strategy factory function"""
    print("\n=== Testing Strategy Factory ===")
    
    try:
        from msiconvert.interpolators.strategies import create_interpolation_strategy, get_strategy_info
        
        # Test creating strategies
        pchip = create_interpolation_strategy("pchip")
        print(f"[OK] Created PCHIP strategy: {pchip.name}")
        
        adaptive = create_interpolation_strategy("adaptive", performance_priority="speed")
        print(f"[OK] Created adaptive strategy: {adaptive.name}")
        
        # Test invalid strategy
        try:
            invalid = create_interpolation_strategy("invalid_strategy")
            print("[FAIL] Should have failed with invalid strategy")
            return False
        except ValueError as e:
            print(f"[OK] Correctly rejected invalid strategy: {e}")
        
        # Test strategy info
        info = get_strategy_info()
        print(f"[OK] Strategy info retrieved for {len(info)} strategies")
        for name, strategy_info in info.items():
            print(f"    {name}: {strategy_info.get('description', 'No description')[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Strategy factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_metrics():
    """Test quality metrics functionality"""
    print("\n=== Testing Quality Metrics ===")
    
    try:
        from msiconvert.interpolators.strategies.base_strategy import QualityMetrics
        
        # Create test metrics
        metrics = QualityMetrics(
            tic_ratio=0.98,
            peak_preservation=0.95,
            n_peaks_original=10,
            n_peaks_interpolated=9,
            interpolation_time=0.05
        )
        
        print(f"[OK] QualityMetrics created: TIC={metrics.tic_ratio}, "
              f"peaks preserved={metrics.peak_preservation}")
        
        # Test metrics validation
        from msiconvert.interpolators.strategies import PchipInterpolationStrategy
        strategy = PchipInterpolationStrategy(validate_quality=True)
        
        # Create test data
        mz_old = np.array([100, 120, 140, 160, 180, 200])
        intensity_old = np.array([100, 50, 80, 30, 60, 40], dtype=np.float32)
        mz_new = np.linspace(100, 200, 50)
        intensity_new = np.linspace(100, 40, 50).astype(np.float32)
        
        validation_result = strategy.validate_interpolation(
            mz_old, intensity_old, mz_new, intensity_new, 0.05
        )
        
        print(f"[OK] Validation completed: TIC ratio={validation_result.tic_ratio:.3f}")
        print(f"    Original peaks: {validation_result.n_peaks_original}")
        print(f"    Interpolated peaks: {validation_result.n_peaks_interpolated}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Quality metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interpolation_performance():
    """Test interpolation performance characteristics"""
    print("\n=== Testing Interpolation Performance ===")
    
    try:
        from msiconvert.interpolators.strategies import PchipInterpolationStrategy
        
        strategy = PchipInterpolationStrategy(validate_quality=False)
        
        # Test different spectrum sizes
        sizes = [100, 1000, 10000]
        target_size = 90000  # Typical target from review document
        
        for size in sizes:
            # Create test data
            mz_old = np.linspace(100, 1000, size)
            intensity_old = np.random.rand(size).astype(np.float32) * 1000
            mz_new = np.linspace(100, 1000, target_size)
            
            # Time the interpolation
            start_time = time.time()
            result = strategy.interpolate_spectrum(mz_old, intensity_old, mz_new)
            elapsed_time = time.time() - start_time
            
            throughput = 1.0 / elapsed_time if elapsed_time > 0 else float('inf')
            
            print(f"[OK] Size {size:>5} -> {target_size:>6}: {elapsed_time:.4f}s, "
                  f"{throughput:.1f} spectra/sec")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 2 strategy tests"""
    print("Phase 2 Interpolation Strategies Test Suite")
    print("=" * 50)
    
    tests = [
        test_pchip_strategy,
        test_adaptive_strategy,
        test_strategy_factory,
        test_quality_metrics,
        test_interpolation_performance
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Strategy Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("[SUCCESS] All Phase 2 strategy tests passed!")
        return True
    else:
        print("[ERROR] Some strategy tests failed!")
        return False

if __name__ == "__main__":
    success = main()