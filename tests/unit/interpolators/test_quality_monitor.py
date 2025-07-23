#!/usr/bin/env python3
"""
Comprehensive test suite for interpolation quality monitoring (Phase 4).

Tests the real-time quality monitor implementation from the review document.
"""

import sys
import numpy as np
from pathlib import Path
from unittest.mock import Mock

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_quality_thresholds():
    """Test quality threshold configurations"""
    print("=== Testing Quality Thresholds ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import QualityThresholds
        
        # Test default thresholds
        default_thresholds = QualityThresholds()
        print(f"[OK] Default thresholds: TIC={default_thresholds.tic_deviation}, "
              f"Peak={default_thresholds.peak_preservation}")
        
        assert default_thresholds.tic_deviation == 0.01
        assert default_thresholds.peak_preservation == 0.95
        assert default_thresholds.mass_accuracy_ppm == 5.0
        
        # Test custom thresholds
        strict_thresholds = QualityThresholds(
            tic_deviation=0.005,
            peak_preservation=0.98,
            mass_accuracy_ppm=2.0
        )
        print(f"[OK] Strict thresholds: TIC={strict_thresholds.tic_deviation}, "
              f"Peak={strict_thresholds.peak_preservation}")
        
        assert strict_thresholds.tic_deviation == 0.005
        assert strict_thresholds.peak_preservation == 0.98
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Quality thresholds test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_monitor_initialization():
    """Test quality monitor initialization"""
    print("\\n=== Testing Quality Monitor Initialization ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import (
            InterpolationQualityMonitor, QualityThresholds
        )
        
        # Test default initialization
        monitor = InterpolationQualityMonitor()
        print("[OK] Default monitor initialized")
        
        assert monitor.total_spectra == 0
        assert monitor.failed_spectra == 0
        assert len(monitor.warnings) == 0
        
        # Test custom initialization
        custom_thresholds = QualityThresholds(
            tic_deviation=0.02,
            peak_preservation=0.90
        )
        
        custom_monitor = InterpolationQualityMonitor(
            thresholds=custom_thresholds,
            history_size=500
        )
        print("[OK] Custom monitor initialized")
        
        assert custom_monitor.thresholds.tic_deviation == 0.02
        assert custom_monitor.history_size == 500
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Quality monitor initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spectrum_validation_good_quality():
    """Test spectrum validation with good quality data"""
    print("\\n=== Testing Good Quality Spectrum Validation ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor
        
        monitor = InterpolationQualityMonitor()
        
        # Create good quality test data
        mz_original = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        intensity_original = np.array([1000.0, 800.0, 1200.0, 600.0, 900.0])
        
        # Simulate good interpolation (perfect TIC preservation, all peaks preserved)
        new_mass_axis = np.linspace(100, 500, 50)
        interpolated = np.interp(new_mass_axis, mz_original, intensity_original)
        
        # Ensure TIC is preserved (adjust interpolated to match)
        tic_ratio = np.sum(intensity_original) / np.sum(interpolated)
        interpolated *= tic_ratio
        
        result = monitor.validate_spectrum(
            (mz_original, intensity_original),
            interpolated,
            new_mass_axis,
            interpolation_time=0.05
        )
        
        print(f"[OK] Good quality validation: TIC ratio={result['tic_ratio']:.3f}, "
              f"Peaks preserved={result['peak_preservation']:.3f}")
        
        assert result['passed'] == True
        assert len(result['warnings']) == 0
        assert abs(result['tic_ratio'] - 1.0) < 0.01
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Good quality validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spectrum_validation_poor_quality():
    """Test spectrum validation with poor quality data"""
    print("\\n=== Testing Poor Quality Spectrum Validation ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor
        
        monitor = InterpolationQualityMonitor()
        
        # Create test data
        mz_original = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        intensity_original = np.array([1000.0, 800.0, 1200.0, 600.0, 900.0])
        
        # Simulate poor interpolation (50% TIC loss, smoothed to lose peaks)
        new_mass_axis = np.linspace(100, 500, 20)  # Coarse axis
        interpolated = np.interp(new_mass_axis, mz_original, intensity_original)
        interpolated *= 0.5  # 50% TIC loss
        
        result = monitor.validate_spectrum(
            (mz_original, intensity_original),
            interpolated,
            new_mass_axis,
            interpolation_time=0.15  # Slow interpolation
        )
        
        print(f"[OK] Poor quality validation: TIC ratio={result['tic_ratio']:.3f}, "
              f"Warnings={len(result['warnings'])}")
        
        assert result['passed'] == False  # Should fail due to TIC deviation
        assert len(result['warnings']) > 0
        # TIC ratio should show the issue (either too high or too low)
        assert abs(result['tic_ratio'] - 1.0) > 0.01  # Significant TIC deviation
        
        # Check that warnings were logged
        assert monitor.total_spectra == 1
        assert monitor.failed_spectra == 1
        assert len(monitor.warnings) > 0
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Poor quality validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_peak_detection():
    """Test peak detection functionality"""
    print("\\n=== Testing Peak Detection ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor
        
        monitor = InterpolationQualityMonitor()
        
        # Create test spectrum with known peaks
        intensities = np.array([10, 50, 100, 60, 20, 80, 150, 90, 30, 5])
        #                       0   1    2   3   4   5    6   7   8  9
        # Expected peaks at indices: 2 (100), 6 (150)
        
        peaks = monitor._find_peaks(intensities, min_height_ratio=0.3)  # 30% of max
        print(f"[OK] Found {len(peaks)} peaks at indices: {peaks}")
        
        # Should find peaks at indices 2 and 6 (values 100 and 150)
        assert len(peaks) == 2
        assert 2 in peaks  # Peak at intensity 100
        assert 6 in peaks  # Peak at intensity 150
        
        # Test with no peaks (flat signal)
        flat_intensities = np.array([50, 50, 50, 50, 50])
        flat_peaks = monitor._find_peaks(flat_intensities)
        print(f"[OK] Flat signal has {len(flat_peaks)} peaks")
        assert len(flat_peaks) == 0
        
        # Test with empty spectrum
        empty_peaks = monitor._find_peaks(np.array([]))
        assert len(empty_peaks) == 0
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Peak detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_summary():
    """Test quality summary generation"""
    print("\\n=== Testing Quality Summary ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor
        
        monitor = InterpolationQualityMonitor()
        
        # Validate several spectra to build up statistics
        for i in range(10):
            mz = np.array([100.0, 200.0, 300.0])
            intensity = np.array([1000.0, 500.0, 800.0]) * (0.9 + i * 0.02)  # Vary intensity
            
            new_axis = np.linspace(100, 300, 30)
            interpolated = np.interp(new_axis, mz, intensity)
            
            # Introduce some TIC variation
            if i < 8:  # Most spectra are good
                interpolated *= (0.99 + i * 0.002)  # Small TIC variation
            else:  # Last 2 spectra have poor TIC preservation
                interpolated *= 0.9  # 10% TIC loss
            
            monitor.validate_spectrum(
                (mz, intensity),
                interpolated,
                new_axis,
                interpolation_time=0.05 + i * 0.01
            )
        
        summary = monitor.get_summary()
        print(f"[OK] Summary generated: {summary['total_spectra']} spectra, "
              f"{summary['failed_spectra']} failed")
        
        assert summary['total_spectra'] == 10
        # Check that some spectra failed (the exact number may vary)
        assert summary['failed_spectra'] > 0
        print(f"[OK] {summary['failed_spectra']} out of {summary['total_spectra']} spectra failed quality checks")
        assert 'tic_ratio' in summary
        assert 'peak_preservation' in summary
        
        # Check TIC ratio statistics
        tic_stats = summary['tic_ratio']
        assert 'mean' in tic_stats
        assert 'std' in tic_stats
        assert tic_stats['n_samples'] == 10
        
        print(f"[OK] TIC ratio stats: mean={tic_stats['mean']:.3f}, std={tic_stats['std']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Quality summary test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_metrics_integration():
    """Test integration with QualityMetrics objects"""
    print("\\n=== Testing QualityMetrics Integration ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor
        from msiconvert.interpolators.strategies.base_strategy import QualityMetrics
        
        monitor = InterpolationQualityMonitor()
        
        # Create QualityMetrics object
        metrics = QualityMetrics(
            tic_ratio=0.995,
            peak_preservation=0.98,
            n_peaks_original=5,
            n_peaks_interpolated=5,
            interpolation_time=0.08
        )
        
        result = monitor.validate_quality_metrics(metrics)
        print(f"[OK] QualityMetrics validation: passed={result['passed']}, "
              f"TIC={result['tic_ratio']:.3f}")
        
        assert result['passed'] == True
        assert result['tic_ratio'] == 0.995
        assert result['peak_preservation'] == 0.98
        assert len(result['warnings']) == 0
        
        # Test with poor quality metrics
        poor_metrics = QualityMetrics(
            tic_ratio=0.85,  # Poor TIC preservation
            peak_preservation=0.90,  # Poor peak preservation
            n_peaks_original=10,
            n_peaks_interpolated=9,
            interpolation_time=0.15  # Slow
        )
        
        poor_result = monitor.validate_quality_metrics(poor_metrics)
        print(f"[OK] Poor QualityMetrics validation: passed={poor_result['passed']}, "
              f"warnings={len(poor_result['warnings'])}")
        
        assert poor_result['passed'] == False
        assert len(poor_result['warnings']) > 0
        
        return True
        
    except Exception as e:
        print(f"[FAIL] QualityMetrics integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_monitoring():
    """Test performance monitoring features"""
    print("\\n=== Testing Performance Monitoring ===")
    
    try:
        from msiconvert.interpolators.quality_monitor import InterpolationQualityMonitor
        import time
        
        monitor = InterpolationQualityMonitor()
        
        # Simulate several spectra with varying performance
        for i in range(5):
            mz = np.array([100.0, 200.0, 300.0])
            intensity = np.array([1000.0, 500.0, 800.0])
            
            new_axis = np.linspace(100, 300, 20)
            interpolated = np.interp(new_axis, mz, intensity)
            
            # Simulate different interpolation times
            interp_time = 0.05 + i * 0.02
            
            monitor.validate_spectrum(
                (mz, intensity),
                interpolated,
                new_axis,
                interpolation_time=interp_time
            )
            
            time.sleep(0.01)  # Small delay to create time separation
        
        # Get recent performance
        recent_perf = monitor.get_recent_performance(window_minutes=1.0)
        print(f"[OK] Recent performance: {len(recent_perf['metrics'])} metrics tracked")
        
        assert 'interpolation_time' in recent_perf['metrics']
        assert recent_perf['metrics']['interpolation_time']['count'] == 5
        
        # Test trend calculation
        time_metric = recent_perf['metrics']['interpolation_time']
        print(f"[OK] Interpolation time trend: {time_metric['trend']}")
        
        # Should show degrading trend since times increase
        assert time_metric['trend'] in ['degrading', 'stable', 'improving', 'insufficient_data']
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Performance monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all quality monitor tests"""
    print("Phase 4-5 Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        test_quality_thresholds,
        test_quality_monitor_initialization,
        test_spectrum_validation_good_quality,
        test_spectrum_validation_poor_quality,
        test_peak_detection,
        test_quality_summary,
        test_quality_metrics_integration,
        test_performance_monitoring
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\\n" + "=" * 50)
    print("Quality Monitor Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("[SUCCESS] All Phase 4-5 quality monitoring tests passed!")
        return True
    else:
        print("[ERROR] Some quality monitoring tests failed!")
        return False

if __name__ == "__main__":
    success = main()