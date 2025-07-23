"""
Quality report generation for interpolation results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np


class InterpolationQualityReport:
    """Generate comprehensive quality reports for interpolation results."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize quality report generator.
        
        Args:
            config: Interpolation configuration used
        """
        self.config = config
        self.stats = {}
        self.warnings = []
        
    def add_stats(self, stats: Dict[str, Any]):
        """Add performance and quality statistics."""
        self.stats.update(stats)
        
    def add_warning(self, warning: str):
        """Add a quality warning."""
        self.warnings.append(warning)
        
    def generate_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Args:
            output_path: Optional path to save report as JSON
            
        Returns:
            Dictionary containing the complete report
        """
        report = {
            "metadata": {
                "generation_time": datetime.now().isoformat(),
                "interpolation_method": self.config.get("method", "unknown"),
                "physics_model": self.config.get("physics_model", "unknown"),
                "target_bins": self.config.get("interpolation_bins"),
                "target_width": self.config.get("interpolation_width"),
            },
            "performance": self._generate_performance_section(),
            "quality": self._generate_quality_section(),
            "memory": self._generate_memory_section(),
            "warnings": self.warnings,
            "summary": self._generate_summary()
        }
        
        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logging.info(f"Quality report saved to: {output_path}")
            
        return report
        
    def _generate_performance_section(self) -> Dict[str, Any]:
        """Generate performance metrics section."""
        return {
            "throughput_spectra_per_sec": self.stats.get("overall_throughput_per_sec", 0),
            "total_processing_time_sec": self.stats.get("elapsed_time", 0),
            "spectra_processed": self.stats.get("spectra_written", 0),
            "interpolation_efficiency": self._calculate_efficiency(),
            "worker_utilization": self._calculate_worker_utilization()
        }
        
    def _generate_quality_section(self) -> Dict[str, Any]:
        """Generate quality metrics section."""
        quality_summary = self.stats.get("quality_summary", {})
        
        return {
            "tic_preservation": {
                "average_ratio": quality_summary.get("avg_tic_ratio", 1.0),
                "acceptable": abs(quality_summary.get("avg_tic_ratio", 1.0) - 1.0) < 0.01
            },
            "peak_preservation": {
                "average_ratio": quality_summary.get("avg_peak_preservation", 1.0),
                "acceptable": quality_summary.get("avg_peak_preservation", 1.0) > 0.95
            },
            "data_integrity": self._assess_data_integrity(),
            "interpolation_accuracy": self._assess_interpolation_accuracy()
        }
        
    def _generate_memory_section(self) -> Dict[str, Any]:
        """Generate memory usage section."""
        memory_stats = self.stats.get("memory_stats", {})
        
        return {
            "peak_usage_gb": memory_stats.get("peak_memory_gb", 0),
            "buffer_efficiency": memory_stats.get("buffer_hit_rate", 0),
            "memory_pressure_events": memory_stats.get("pressure_events", 0),
            "gc_collections": memory_stats.get("gc_collections", 0)
        }
        
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall summary and recommendations."""
        performance_rating = self._rate_performance()
        quality_rating = self._rate_quality()
        
        return {
            "overall_rating": min(performance_rating, quality_rating),
            "performance_rating": performance_rating,
            "quality_rating": quality_rating,
            "recommendations": self._generate_recommendations(),
            "pass_criteria": {
                "tic_preservation": abs(self.stats.get("quality_summary", {}).get("avg_tic_ratio", 1.0) - 1.0) < 0.01,
                "peak_preservation": self.stats.get("quality_summary", {}).get("avg_peak_preservation", 1.0) > 0.95,
                "no_critical_warnings": len([w for w in self.warnings if "critical" in w.lower()]) == 0
            }
        }
        
    def _calculate_efficiency(self) -> float:
        """Calculate interpolation efficiency score."""
        throughput = self.stats.get("overall_throughput_per_sec", 0)
        expected_throughput = 200  # Expected ~200 spectra/sec
        
        return min(throughput / expected_throughput, 1.0) if expected_throughput > 0 else 0.0
        
    def _calculate_worker_utilization(self) -> float:
        """Calculate worker utilization efficiency."""
        config_workers = self.config.get("max_workers", 4)
        actual_workers = self.stats.get("config", {}).get("n_workers", config_workers)
        
        return actual_workers / config_workers if config_workers > 0 else 1.0
        
    def _assess_data_integrity(self) -> str:
        """Assess overall data integrity."""
        tic_ratio = self.stats.get("quality_summary", {}).get("avg_tic_ratio", 1.0)
        peak_preservation = self.stats.get("quality_summary", {}).get("avg_peak_preservation", 1.0)
        
        if abs(tic_ratio - 1.0) < 0.005 and peak_preservation > 0.98:
            return "excellent"
        elif abs(tic_ratio - 1.0) < 0.01 and peak_preservation > 0.95:
            return "good"
        elif abs(tic_ratio - 1.0) < 0.02 and peak_preservation > 0.90:
            return "acceptable"
        else:
            return "poor"
            
    def _assess_interpolation_accuracy(self) -> str:
        """Assess interpolation accuracy based on method and results."""
        method = self.config.get("method", "unknown")
        
        if method == "pchip":
            return "high"  # PCHIP is monotonic and preserves peaks
        elif method == "linear":
            return "medium"  # Linear is simple but may lose peaks
        else:
            return "unknown"
            
    def _rate_performance(self) -> str:
        """Rate overall performance."""
        throughput = self.stats.get("overall_throughput_per_sec", 0)
        
        if throughput > 300:
            return "excellent"
        elif throughput > 200:
            return "good"
        elif throughput > 100:
            return "acceptable"
        else:
            return "poor"
            
    def _rate_quality(self) -> str:
        """Rate overall quality."""
        integrity = self._assess_data_integrity()
        critical_warnings = len([w for w in self.warnings if "critical" in w.lower()])
        
        if integrity == "excellent" and critical_warnings == 0:
            return "excellent"
        elif integrity in ["excellent", "good"] and critical_warnings == 0:
            return "good"
        elif integrity == "acceptable" and critical_warnings <= 1:
            return "acceptable"
        else:
            return "poor"
            
    def _generate_recommendations(self) -> list:
        """Generate recommendations for improvement."""
        recommendations = []
        
        throughput = self.stats.get("overall_throughput_per_sec", 0)
        if throughput < 100:
            recommendations.append("Consider increasing worker count or optimizing chunk size")
            
        tic_ratio = self.stats.get("quality_summary", {}).get("avg_tic_ratio", 1.0)
        if abs(tic_ratio - 1.0) > 0.02:
            recommendations.append("TIC preservation is suboptimal - check interpolation method")
            
        peak_preservation = self.stats.get("quality_summary", {}).get("avg_peak_preservation", 1.0)
        if peak_preservation < 0.90:
            recommendations.append("Peak preservation is low - consider using PCHIP method")
            
        if len(self.warnings) > 10:
            recommendations.append("High number of warnings - review interpolation parameters")
            
        return recommendations
        
    def print_summary(self):
        """Print a concise summary to console."""
        print("\n" + "="*60)
        print("INTERPOLATION QUALITY REPORT SUMMARY")
        print("="*60)
        
        performance = self._rate_performance()
        quality = self._rate_quality()
        
        print(f"Performance Rating: {performance.upper()}")
        print(f"Quality Rating: {quality.upper()}")
        print(f"Throughput: {self.stats.get('overall_throughput_per_sec', 0):.1f} spectra/sec")
        
        quality_summary = self.stats.get("quality_summary", {})
        if quality_summary:
            print(f"TIC Preservation: {quality_summary.get('avg_tic_ratio', 1.0):.3f}")
            print(f"Peak Preservation: {quality_summary.get('avg_peak_preservation', 1.0):.3f}")
            
        if self.warnings:
            print(f"Warnings: {len(self.warnings)}")
            
        recommendations = self._generate_recommendations()
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
                
        print("="*60)