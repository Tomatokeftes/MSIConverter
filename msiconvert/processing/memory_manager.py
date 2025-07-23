"""
Adaptive memory management for interpolation processing.
"""

import logging
import gc
from typing import Dict, Optional
import numpy as np

# Import psutil conditionally
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
    logging.warning("psutil not available, memory management will use fallback values")


class AdaptiveMemoryManager:
    """
    Monitors and manages memory usage during interpolation processing.
    
    Provides adaptive scaling of workers and buffer sizes based on available memory
    and processing requirements.
    """
    
    def __init__(self, target_memory_gb: float = 8.0, safety_factor: float = 0.8):
        """
        Initialize memory manager.
        
        Args:
            target_memory_gb: Target memory usage limit in GB
            safety_factor: Safety factor for memory calculations (0.0-1.0)
        """
        self.target_memory = target_memory_gb * 1e9 * safety_factor
        self.safety_factor = safety_factor
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        
        # Memory monitoring
        self.peak_memory = 0.0
        self.memory_samples = []
        
        logging.info(f"Memory manager initialized with {target_memory_gb:.1f}GB target, "
                    f"{safety_factor:.1f} safety factor")
        
    def get_available_memory(self) -> float:
        """
        Get available system memory in bytes.
        
        Returns:
            Available memory in bytes
        """
        if not PSUTIL_AVAILABLE:
            return 8e9  # Fallback to 8GB when psutil unavailable
            
        try:
            mem_info = psutil.virtual_memory()
            return mem_info.available
        except Exception as e:
            logging.error(f"Error getting available memory: {e}")
            return 8e9  # Fallback to 8GB
        
    def get_process_memory(self) -> float:
        """
        Get current process memory usage in bytes.
        
        Returns:
            Process memory usage in bytes
        """
        if not PSUTIL_AVAILABLE or self.process is None:
            # Fallback: estimate based on target memory usage
            estimated_memory = self.target_memory * 0.5  # Assume 50% of target
            if estimated_memory > self.peak_memory:
                self.peak_memory = estimated_memory
            return estimated_memory
            
        try:
            memory_info = self.process.memory_info()
            current_memory = memory_info.rss
            
            # Track peak memory
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
                
            return current_memory
        except Exception as e:
            logging.error(f"Error getting process memory: {e}")
            return 0.0
            
    def calculate_optimal_workers(self, 
                                 spectrum_size: int,
                                 interpolation_memory_mb: float = 4.6) -> int:
        """
        Calculate optimal number of workers based on available memory.
        
        Args:
            spectrum_size: Expected size of interpolated spectra
            interpolation_memory_mb: Memory per interpolation operation in MB
            
        Returns:
            Optimal number of workers
        """
        # Get available memory
        available_memory = self.get_available_memory()
        process_memory = self.get_process_memory()
        
        # Reserve memory for base operations
        base_memory_gb = 2.0  # Base memory for reader, converters, etc.
        available_for_workers = min(
            self.target_memory - base_memory_gb * 1e9,
            available_memory * 0.6  # Don't use more than 60% of available
        )
        
        # Memory per worker (interpolation + buffers)
        interpolation_memory = interpolation_memory_mb * 1e6
        buffer_memory = spectrum_size * (8 + 4) * 2  # mz + intensity, original + interpolated
        memory_per_worker = interpolation_memory + buffer_memory
        
        # Calculate optimal workers
        optimal_workers = max(1, int(available_for_workers / memory_per_worker))
        
        # Apply reasonable bounds
        cpu_count = psutil.cpu_count() if PSUTIL_AVAILABLE else 4  # Fallback to 4 cores
        optimal_workers = min(optimal_workers, cpu_count * 2)  # Max 2x CPU cores
        optimal_workers = max(optimal_workers, 1)  # Minimum 1 worker
        
        logging.info(f"Memory calculation: {available_for_workers/1e9:.1f}GB available, "
                    f"{memory_per_worker/1e6:.1f}MB per worker, "
                    f"optimal workers: {optimal_workers}")
        
        return optimal_workers
        
    def calculate_optimal_chunk_size(self, 
                                    total_spectra: int,
                                    spectrum_size: int) -> int:
        """
        Calculate optimal chunk size for processing.
        
        Args:
            total_spectra: Total number of spectra to process
            spectrum_size: Size of each spectrum after interpolation
            
        Returns:
            Optimal chunk size
        """
        # Target chunk memory (aim for ~100MB chunks)
        target_chunk_memory = 100e6
        
        # Memory per spectrum (original + interpolated)
        memory_per_spectrum = spectrum_size * (8 + 4) * 2
        
        # Calculate chunk size
        chunk_size = max(1, int(target_chunk_memory / memory_per_spectrum))
        
        # Apply reasonable bounds
        chunk_size = min(chunk_size, total_spectra)  # Don't exceed total
        chunk_size = max(chunk_size, 10)  # Minimum 10 spectra per chunk
        chunk_size = min(chunk_size, 10000)  # Maximum 10k spectra per chunk
        
        logging.info(f"Calculated chunk size: {chunk_size} spectra "
                    f"({chunk_size * memory_per_spectrum / 1e6:.1f}MB per chunk)")
        
        return chunk_size
        
    def check_memory_pressure(self, threshold_factor: float = 0.9) -> bool:
        """
        Check if system is under memory pressure.
        
        Args:
            threshold_factor: Threshold as fraction of target memory
            
        Returns:
            True if under memory pressure
        """
        current_memory = self.get_process_memory()
        available_memory = self.get_available_memory()
        total_memory = psutil.virtual_memory().total if PSUTIL_AVAILABLE else 16e9  # Fallback to 16GB
        
        # Check if we're approaching limits
        process_pressure = current_memory > (self.target_memory * threshold_factor)
        system_pressure = available_memory < (total_memory * 0.15)  # Less than 15% available
        
        if process_pressure or system_pressure:
            logging.warning(f"Memory pressure detected: "
                          f"Process: {current_memory/1e9:.2f}GB/{self.target_memory/1e9:.1f}GB, "
                          f"Available: {available_memory/1e9:.2f}GB")
            return True
            
        return False
        
    def optimize_memory(self) -> None:
        """
        Perform memory optimization (garbage collection, etc.)
        """
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Log memory state
            current_memory = self.get_process_memory()
            logging.info(f"Memory optimization: collected {collected} objects, "
                        f"current memory: {current_memory/1e9:.2f}GB")
            
        except Exception as e:
            logging.error(f"Error during memory optimization: {e}")
            
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        try:
            current_memory = self.get_process_memory()
            available_memory = self.get_available_memory()
            total_memory = psutil.virtual_memory().total if PSUTIL_AVAILABLE else 16e9  # Fallback to 16GB
            
            return {
                'current_memory_gb': current_memory / 1e9,
                'peak_memory_gb': self.peak_memory / 1e9,
                'target_memory_gb': self.target_memory / 1e9,
                'available_memory_gb': available_memory / 1e9,
                'total_memory_gb': total_memory / 1e9,
                'memory_utilization': current_memory / total_memory,
                'under_pressure': self.check_memory_pressure()
            }
        except Exception as e:
            logging.error(f"Error getting memory stats: {e}")
            return {
                'current_memory_gb': 0.0,
                'peak_memory_gb': 0.0,
                'target_memory_gb': self.target_memory / 1e9,
                'available_memory_gb': 0.0,
                'total_memory_gb': 0.0,
                'memory_utilization': 0.0,
                'under_pressure': False
            }
            
    def suggest_worker_adjustment(self, current_workers: int) -> Optional[int]:
        """
        Suggest worker count adjustment based on memory pressure.
        
        Args:
            current_workers: Current number of workers
            
        Returns:
            Suggested worker count, or None if no change needed
        """
        if self.check_memory_pressure(threshold_factor=0.85):
            # Under pressure - reduce workers
            suggested = max(1, int(current_workers * 0.75))
            if suggested != current_workers:
                logging.info(f"Memory pressure detected, suggesting {suggested} workers "
                           f"(down from {current_workers})")
                return suggested
                
        elif not self.check_memory_pressure(threshold_factor=0.6):
            # Memory available - could increase workers
            available_memory = self.get_available_memory()
            if available_memory > 2e9:  # More than 2GB available
                cpu_count = psutil.cpu_count() if PSUTIL_AVAILABLE else 4  # Fallback to 4 cores
                suggested = min(current_workers + 1, cpu_count * 2)
                if suggested != current_workers:
                    logging.info(f"Memory available, suggesting {suggested} workers "
                               f"(up from {current_workers})")
                    return suggested
                    
        return None