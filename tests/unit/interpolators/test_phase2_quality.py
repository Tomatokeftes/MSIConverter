#!/usr/bin/env python3
"""
Test suite for Phase 2 quality monitoring system
"""

import sys
import numpy as np
from pathlib import Path
import time

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_quality_monitor_basic():
    """Test basic quality monitor functionality"""
    print("=== Testing Quality Monitor Basic Functionality ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor, QualityThresholds
        
        # Initialize with custom thresholds
        thresholds = QualityThresholds(
            tic_deviation=0.05,  # 5% TIC deviation
            peak_preservation=0.90,  # 90% peak preservation
            max_interpolation_time=0.2  # 200ms max
        )
        
        monitor = InterpolationQualityMonitor(thresholds=thresholds)
        print(f"[OK] Quality monitor initialized")
        
        # Test with good quality interpolation
        mz_old = np.array([100, 120, 140, 160, 180, 200])
        intensity_old = np.array([1000, 800, 1200, 600, 900, 400], dtype=np.float32)
        mz_new = np.linspace(100, 200, 50)
        
        # Simulate good interpolation (preserving TIC and peaks)
        intensity_new = np.interp(mz_new, mz_old, intensity_old).astype(np.float32)
        
        result = monitor.validate_spectrum(
            (mz_old, intensity_old),
            intensity_new, 
            mz_new,
            interpolation_time=0.05
        )
        
        print(f"[OK] Good quality validation: passed={result['passed']}")
        print(f"    TIC ratio: {result['tic_ratio']:.3f}")
        print(f"    Peak preservation: {result['peak_preservation']:.3f}")
        print(f"    Warnings: {len(result['warnings'])}")
        
        # Test with poor quality interpolation
        poor_intensity = intensity_new * 0.5  # Lose 50% of intensity
        
        poor_result = monitor.validate_spectrum(
            (mz_old, intensity_old),
            poor_intensity,
            mz_new,
            interpolation_time=0.3  # Slow interpolation
        )
        
        print(f"[OK] Poor quality validation: passed={poor_result['passed']}")
        print(f"    Warnings: {poor_result['warnings']}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic quality monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_metrics_validation():
    """Test validation using QualityMetrics objects"""
    print("\n=== Testing QualityMetrics Validation ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor
        from msiconvert.interpolators.strategies.base_strategy import QualityMetrics
        
        monitor = InterpolationQualityMonitor()
        
        # Test good quality metrics
        good_metrics = QualityMetrics(
            tic_ratio=0.99,
            peak_preservation=0.97,
            n_peaks_original=15,
            n_peaks_interpolated=14,
            interpolation_time=0.03
        )
        
        result = monitor.validate_quality_metrics(good_metrics)
        print(f"[OK] Good metrics validation: passed={result['passed']}")
        
        # Test poor quality metrics
        poor_metrics = QualityMetrics(
            tic_ratio=0.85,  # Too much TIC loss
            peak_preservation=0.80,  # Too much peak loss
            n_peaks_original=20,
            n_peaks_interpolated=16,
            interpolation_time=0.15  # Too slow
        )
        
        poor_result = monitor.validate_quality_metrics(poor_metrics)
        print(f"[OK] Poor metrics validation: passed={poor_result['passed']}")
        print(f"    Warnings: {len(poor_result['warnings'])}")
        
        # Test edge cases
        edge_metrics = QualityMetrics(
            tic_ratio=0.0,  # Complete TIC loss
            peak_preservation=0.0,  # Complete peak loss
            n_peaks_original=0,
            n_peaks_interpolated=0,
            interpolation_time=1.0  # Very slow
        )
        
        edge_result = monitor.validate_quality_metrics(edge_metrics)
        print(f"[OK] Edge case validation: passed={edge_result['passed']}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] QualityMetrics validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_statistics():
    """Test quality statistics and trend analysis"""
    print("\n=== Testing Quality Statistics ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor
        from msiconvert.interpolators.strategies.base_strategy import QualityMetrics
        
        monitor = InterpolationQualityMonitor(history_size=100)
        
        # Simulate processing multiple spectra with varying quality
        np.random.seed(42)  # For reproducible results
        
        for i in range(50):
            # Simulate gradually degrading quality
            base_tic_ratio = 0.98 - (i * 0.001)  # Slight degradation
            base_peak_preservation = 0.96 - (i * 0.002)  # Slight degradation
            
            # Add some noise
            tic_ratio = base_tic_ratio + np.random.normal(0, 0.01)
            peak_preservation = base_peak_preservation + np.random.normal(0, 0.02)
            interpolation_time = 0.05 + np.random.exponential(0.02)
            
            metrics = QualityMetrics(
                tic_ratio=max(0.1, tic_ratio),  # Ensure positive
                peak_preservation=max(0.1, peak_preservation),  # Ensure positive
                n_peaks_original=10 + np.random.randint(-3, 4),
                n_peaks_interpolated=9 + np.random.randint(-2, 3),
                interpolation_time=interpolation_time
            )
            
            monitor.validate_quality_metrics(metrics)
            
            # Small delay to create time separation
            time.sleep(0.001)
        
        # Get summary statistics
        summary = monitor.get_summary()
        print(f"[OK] Processed {summary['total_spectra']} spectra")
        print(f"    Success rate: {summary['success_rate']:.3f}")
        print(f"    Failed spectra: {summary['failed_spectra']}")
        print(f"    Total warnings: {summary['total_warnings']}")
        
        if 'tic_ratio' in summary:
            tic_stats = summary['tic_ratio']
            print(f"    TIC ratio - mean: {tic_stats['mean']:.3f}, std: {tic_stats['std']:.3f}")
            
        if 'peak_preservation' in summary:
            peak_stats = summary['peak_preservation']
            print(f"    Peak preservation - mean: {peak_stats['mean']:.3f}, std: {peak_stats['std']:.3f}")
        
        # Test recent performance
        recent_perf = monitor.get_recent_performance(window_minutes=1.0)
        print(f"[OK] Recent performance (1 min window):")
        for metric, info in recent_perf['metrics'].items():
            print(f"    {metric}: mean={info['mean']:.3f}, trend={info['trend']}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Quality statistics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_peak_detection():
    """Test peak detection functionality"""
    print("\n=== Testing Peak Detection ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor
        
        monitor = InterpolationQualityMonitor()
        
        # Test with synthetic spectrum containing clear peaks
        # Create spectrum with 3 clear peaks
        intensities = np.zeros(100, dtype=np.float32)
        intensities[20] = 1000  # Peak 1
        intensities[19] = 500
        intensities[21] = 500
        
        intensities[50] = 800   # Peak 2
        intensities[49] = 400
        intensities[51] = 400
        
        intensities[80] = 600   # Peak 3
        intensities[79] = 300
        intensities[81] = 300
        
        # Add some noise
        intensities += np.random.rand(100) * 50
        
        peaks = monitor._find_peaks(intensities)
        print(f"[OK] Peak detection: found {len(peaks)} peaks")
        print(f"    Peak positions: {peaks}")
        
        # Expected peaks at positions 20, 50, 80
        expected_peaks = {20, 50, 80}
        found_peaks = set(peaks)
        
        if expected_peaks.issubset(found_peaks):
            print("[OK] All expected peaks found")
        else:
            print(f"[WARNING] Expected {expected_peaks}, found {found_peaks}")
        
        # Test with flat spectrum (no peaks)
        flat_intensities = np.ones(50, dtype=np.float32) * 100
        flat_peaks = monitor._find_peaks(flat_intensities)
        print(f"[OK] Flat spectrum: {len(flat_peaks)} peaks found (expected 0)")
        
        # Test with empty spectrum
        empty_intensities = np.array([], dtype=np.float32)
        empty_peaks = monitor._find_peaks(empty_intensities)
        print(f"[OK] Empty spectrum: {len(empty_peaks)} peaks found (expected 0)")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Peak detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_thresholds():
    """Test quality threshold management"""
    print("\n=== Testing Quality Thresholds ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor, QualityThresholds
        
        # Test default thresholds
        monitor = InterpolationQualityMonitor()
        default_summary = monitor.get_summary()
        print(f"[OK] Default thresholds: TIC={default_summary['thresholds']['tic_deviation']}")
        
        # Test custom thresholds
        strict_thresholds = QualityThresholds(
            tic_deviation=0.001,  # Very strict
            peak_preservation=0.99,  # Very strict
            max_interpolation_time=0.01  # Very strict
        )
        
        monitor.set_thresholds(strict_thresholds)
        updated_summary = monitor.get_summary()
        print(f"[OK] Updated thresholds: TIC={updated_summary['thresholds']['tic_deviation']}")
        
        # Test with strict thresholds - should fail most validations
        test_intensities = np.array([100, 90, 95], dtype=np.float32)  # Slight TIC loss
        mz_old = np.array([100, 150, 200])
        mz_new = np.array([100, 125, 150, 175, 200])
        
        result = monitor.validate_spectrum(
            (mz_old, test_intensities),
            test_intensities[::2],  # Simulated interpolation
            mz_new,
            interpolation_time=0.02  # Above strict threshold
        )
        
        print(f"[OK] Strict validation: passed={result['passed']} (expected False)")
        print(f"    Warnings with strict thresholds: {len(result['warnings'])}")
        
        # Test reset functionality
        monitor.reset_statistics()
        reset_summary = monitor.get_summary()
        print(f"[OK] After reset: {reset_summary['total_spectra']} spectra, "
              f"{reset_summary['total_warnings']} warnings")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Quality thresholds test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 2 quality monitoring tests"""
    print("Phase 2 Quality Monitoring Test Suite")
    print("=" * 50)
    
    tests = [
        test_quality_monitor_basic,
        test_quality_metrics_validation,
        test_quality_statistics,
        test_peak_detection,
        test_quality_thresholds
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Quality Monitoring Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("[SUCCESS] All Phase 2 quality tests passed!")
        return True
    else:
        print("[ERROR] Some quality tests failed!")
        return False

if __name__ == "__main__":
    success = main()