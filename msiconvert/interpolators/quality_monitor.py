"""
Real-time quality monitoring for interpolation operations.
"""

from collections import defaultdict, deque
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import logging
from dataclasses import dataclass
from .strategies.base_strategy import QualityMetrics

@dataclass
class QualityThresholds:
    """Quality thresholds for interpolation validation"""
    tic_deviation: float = 0.01  # 1% TIC deviation allowed
    peak_preservation: float = 0.95  # 95% peaks must be preserved
    mass_accuracy_ppm: float = 5.0  # 5 ppm mass accuracy
    max_interpolation_time: float = 0.1  # 100ms max per spectrum
    
class InterpolationQualityMonitor:
    """Monitor interpolation quality in real-time"""
    
    def __init__(self, 
                 thresholds: Optional[QualityThresholds] = None,
                 history_size: int = 1000):
        """
        Initialize quality monitor.
        
        Args:
            thresholds: Quality thresholds, uses defaults if None
            history_size: Number of recent samples to keep in memory
        """
        self.thresholds = thresholds or QualityThresholds()
        self.history_size = history_size
        
        # Metrics storage
        self.metrics = defaultdict(deque)
        self.warnings = deque(maxlen=100)  # Keep last 100 warnings
        self.error_counts = defaultdict(int)
        
        # Summary statistics
        self.total_spectra = 0
        self.failed_spectra = 0
        self.start_time = time.time()
        
        logging.info(f"Quality monitor initialized with thresholds: "
                    f"TIC deviation={self.thresholds.tic_deviation}, "
                    f"peak preservation={self.thresholds.peak_preservation}")
        
    def validate_spectrum(self, 
                         original: Tuple[np.ndarray, np.ndarray],
                         interpolated: np.ndarray,
                         new_mass_axis: np.ndarray,
                         interpolation_time: float = 0.0) -> Dict[str, float]:
        """
        Validate single spectrum interpolation.
        
        Args:
            original: Tuple of (mz_old, intensity_old)
            interpolated: Interpolated intensity values
            new_mass_axis: New mass axis
            interpolation_time: Time taken for interpolation
            
        Returns:
            Dictionary with validation metrics
        """
        mz_old, intensity_old = original
        
        # Initialize result
        result = {
            'tic_ratio': 1.0,
            'peak_preservation': 1.0,
            'n_peaks_original': 0,
            'n_peaks_interpolated': 0,
            'interpolation_time': interpolation_time,
            'warnings': [],
            'passed': True
        }
        
        self.total_spectra += 1
        
        try:
            # Total Ion Current preservation
            tic_old = np.sum(intensity_old) if len(intensity_old) > 0 else 0.0
            tic_new = np.sum(interpolated) if len(interpolated) > 0 else 0.0
            
            if tic_old > 0:
                tic_ratio = tic_new / tic_old
                result['tic_ratio'] = tic_ratio
                
                tic_deviation = abs(1 - tic_ratio)
                if tic_deviation > self.thresholds.tic_deviation:
                    warning = f"TIC deviation: {tic_ratio:.3f} (expected ~1.0, threshold={self.thresholds.tic_deviation})"
                    result['warnings'].append(warning)
                    self.warnings.append(warning)
                    self.error_counts['tic_deviation'] += 1
                    result['passed'] = False
            
            # Peak preservation analysis
            peaks_old = self._find_peaks(intensity_old)
            peaks_new = self._find_peaks(interpolated)
            
            result['n_peaks_original'] = len(peaks_old)
            result['n_peaks_interpolated'] = len(peaks_new)
            
            if len(peaks_old) > 0:
                peak_preservation = len(peaks_new) / len(peaks_old)
                result['peak_preservation'] = peak_preservation
                
                if peak_preservation < self.thresholds.peak_preservation:
                    warning = (f"Poor peak preservation: {peak_preservation:.3f} "
                             f"(threshold={self.thresholds.peak_preservation})")
                    result['warnings'].append(warning)
                    self.warnings.append(warning)
                    self.error_counts['peak_preservation'] += 1
                    result['passed'] = False
            
            # Performance check
            if interpolation_time > self.thresholds.max_interpolation_time:
                warning = (f"Slow interpolation: {interpolation_time:.3f}s "
                         f"(threshold={self.thresholds.max_interpolation_time}s)")
                result['warnings'].append(warning)
                self.warnings.append(warning)
                self.error_counts['slow_interpolation'] += 1
                
            # Check for negative values (should not happen with PCHIP)
            if np.any(interpolated < 0):
                warning = f"Negative intensities detected: {np.sum(interpolated < 0)} points"
                result['warnings'].append(warning)
                self.warnings.append(warning)
                self.error_counts['negative_intensities'] += 1
                result['passed'] = False
                
            # Check for NaN or Inf values
            if not np.all(np.isfinite(interpolated)):
                warning = "Non-finite values detected in interpolated spectrum"
                result['warnings'].append(warning)
                self.warnings.append(warning)
                self.error_counts['non_finite_values'] += 1
                result['passed'] = False
            
        except Exception as e:
            warning = f"Quality validation failed: {e}"
            result['warnings'].append(warning)
            self.warnings.append(warning)
            self.error_counts['validation_errors'] += 1
            result['passed'] = False
            logging.error(f"Quality validation error: {e}")
        
        # Store metrics for analysis
        self._store_metrics(result)
        
        if not result['passed']:
            self.failed_spectra += 1
            
        return result
        
    def validate_quality_metrics(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """
        Validate QualityMetrics object.
        
        Args:
            metrics: QualityMetrics object from interpolation strategy
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'tic_ratio': metrics.tic_ratio,
            'peak_preservation': metrics.peak_preservation,
            'n_peaks_original': metrics.n_peaks_original,
            'n_peaks_interpolated': metrics.n_peaks_interpolated,
            'interpolation_time': metrics.interpolation_time,
            'warnings': [],
            'passed': True
        }
        
        # TIC validation
        tic_deviation = abs(1 - metrics.tic_ratio)
        if tic_deviation > self.thresholds.tic_deviation:
            warning = f"TIC deviation: {metrics.tic_ratio:.3f}"
            result['warnings'].append(warning)
            self.warnings.append(warning)
            result['passed'] = False
            
        # Peak preservation validation
        if metrics.peak_preservation < self.thresholds.peak_preservation:
            warning = f"Poor peak preservation: {metrics.peak_preservation:.3f}"
            result['warnings'].append(warning)
            self.warnings.append(warning)
            result['passed'] = False
            
        # Performance validation
        if metrics.interpolation_time > self.thresholds.max_interpolation_time:
            warning = f"Slow interpolation: {metrics.interpolation_time:.3f}s"
            result['warnings'].append(warning)
            self.warnings.append(warning)
            
        self._store_metrics(result)
        self.total_spectra += 1
        
        if not result['passed']:
            self.failed_spectra += 1
            
        return result
        
    def _find_peaks(self, 
                   intensities: np.ndarray, 
                   min_height_ratio: float = 0.01) -> np.ndarray:
        """
        Simple peak detection for quality assessment.
        
        Args:
            intensities: Intensity array
            min_height_ratio: Minimum peak height as fraction of maximum
            
        Returns:
            Array of peak indices
        """
        if len(intensities) <= 2:
            return np.array([], dtype=np.int32)
            
        max_intensity = np.max(intensities)
        if max_intensity == 0:
            return np.array([], dtype=np.int32)
            
        min_height = max_intensity * min_height_ratio
        peaks = []
        
        # Find local maxima above threshold
        for i in range(1, len(intensities) - 1):
            if (intensities[i] > intensities[i-1] and 
                intensities[i] > intensities[i+1] and
                intensities[i] > min_height):
                peaks.append(i)
                
        return np.array(peaks, dtype=np.int32)
        
    def _store_metrics(self, result: Dict[str, Any]):
        """
        Store metrics in rolling history.
        
        Args:
            result: Validation result dictionary
        """
        # Store with timestamp
        timestamp = time.time()
        
        for key in ['tic_ratio', 'peak_preservation', 'interpolation_time']:
            if key in result:
                self.metrics[key].append((timestamp, result[key]))
                
                # Maintain history size limit
                if len(self.metrics[key]) > self.history_size:
                    self.metrics[key].popleft()
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive quality summary.
        
        Returns:
            Dictionary with quality summary statistics
        """
        summary = {
            'total_spectra': self.total_spectra,
            'failed_spectra': self.failed_spectra,
            'success_rate': 1.0 - (self.failed_spectra / self.total_spectra) if self.total_spectra > 0 else 1.0,
            'runtime_minutes': (time.time() - self.start_time) / 60.0,
            'error_counts': dict(self.error_counts),
            'recent_warnings': list(self.warnings)[-10:],  # Last 10 warnings
            'total_warnings': len(self.warnings),
            'thresholds': {
                'tic_deviation': self.thresholds.tic_deviation,
                'peak_preservation': self.thresholds.peak_preservation,
                'max_interpolation_time': self.thresholds.max_interpolation_time
            }
        }
        
        # Calculate statistics for each metric
        for metric_name, values in self.metrics.items():
            if values:
                metric_values = [v[1] for v in values]  # Extract values (ignore timestamps)
                summary[metric_name] = {
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values),
                    'median': np.median(metric_values),
                    'n_samples': len(metric_values)
                }
                
        return summary
        
    def get_recent_performance(self, window_minutes: float = 5.0) -> Dict[str, Any]:
        """
        Get performance statistics for recent time window.
        
        Args:
            window_minutes: Time window in minutes for recent statistics
            
        Returns:
            Dictionary with recent performance metrics
        """
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = {}
        
        for metric_name, values in self.metrics.items():
            recent_values = [v[1] for v in values if v[0] >= cutoff_time]
            
            if recent_values:
                recent_metrics[metric_name] = {
                    'mean': np.mean(recent_values),
                    'count': len(recent_values),
                    'trend': self._calculate_trend(values, cutoff_time)
                }
                
        return {
            'window_minutes': window_minutes,
            'metrics': recent_metrics,
            'recent_warnings': [w for w in self.warnings if w][-5:]  # Last 5 warnings
        }
        
    def _calculate_trend(self, 
                        values: deque, 
                        cutoff_time: float) -> str:
        """
        Calculate trend for a metric (improving/degrading/stable).
        
        Args:
            values: Deque of (timestamp, value) tuples
            cutoff_time: Cutoff time for recent data
            
        Returns:
            Trend description
        """
        recent_values = [(t, v) for t, v in values if t >= cutoff_time]
        
        if len(recent_values) < 3:
            return "insufficient_data"
            
        # Split into two halves and compare means
        mid_point = len(recent_values) // 2
        first_half = [v[1] for v in recent_values[:mid_point]]
        second_half = [v[1] for v in recent_values[mid_point:]]
        
        if len(first_half) == 0 or len(second_half) == 0:
            return "insufficient_data"
            
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)
        
        # Calculate relative change
        if first_mean != 0:
            relative_change = (second_mean - first_mean) / abs(first_mean)
        else:
            relative_change = 0
            
        if abs(relative_change) < 0.05:  # Less than 5% change
            return "stable"
        elif relative_change > 0:
            return "improving"
        else:
            return "degrading"
            
    def reset_statistics(self):
        """Reset all statistics and start fresh"""
        self.metrics.clear()
        self.warnings.clear()
        self.error_counts.clear()
        self.total_spectra = 0
        self.failed_spectra = 0
        self.start_time = time.time()
        logging.info("Quality monitor statistics reset")
        
    def set_thresholds(self, thresholds: QualityThresholds):
        """
        Update quality thresholds.
        
        Args:
            thresholds: New quality thresholds
        """
        self.thresholds = thresholds
        logging.info(f"Quality thresholds updated: {thresholds}")
        
    def export_metrics(self, filepath: str):
        """
        Export metrics to file for analysis.
        
        Args:
            filepath: Path to export file
        """
        try:
            import json
            
            export_data = {
                'summary': self.get_summary(),
                'metrics_history': {
                    name: list(values) for name, values in self.metrics.items()
                },
                'export_timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            logging.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logging.error(f"Failed to export metrics: {e}")