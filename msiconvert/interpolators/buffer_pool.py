"""
Buffer pool implementation for zero-copy spectrum operations.
"""

from typing import List, Optional
import numpy as np
from threading import Lock
import logging
from . import SpectrumBuffer

class SpectrumBufferPool:
    """Thread-safe buffer pool for zero-copy operations"""
    
    def __init__(self, n_buffers: int = 100, buffer_size: int = 100000):
        """
        Initialize buffer pool.
        
        Args:
            n_buffers: Number of buffers to pre-allocate
            buffer_size: Size of each buffer (number of data points)
        """
        self.buffer_size = buffer_size
        self.lock = Lock()
        
        # Pre-allocate all buffers
        self.free_buffers: List[SpectrumBuffer] = []
        for i in range(n_buffers):
            buffer = SpectrumBuffer(
                buffer_id=i,
                mz_buffer=np.empty(buffer_size, dtype=np.float64),
                intensity_buffer=np.empty(buffer_size, dtype=np.float32)
            )
            self.free_buffers.append(buffer)
            
        self.allocated_count = 0
        self.total_buffers = n_buffers
        self.emergency_allocations = 0
        
        logging.info(f"Initialized buffer pool with {n_buffers} buffers of size {buffer_size}")
        
    def get_buffer(self) -> SpectrumBuffer:
        """
        Get a buffer from the pool.
        
        Returns:
            Available spectrum buffer
        """
        with self.lock:
            if not self.free_buffers:
                # Emergency allocation if pool is exhausted
                buffer = SpectrumBuffer(
                    buffer_id=1000 + self.allocated_count,
                    mz_buffer=np.empty(self.buffer_size, dtype=np.float64),
                    intensity_buffer=np.empty(self.buffer_size, dtype=np.float32)
                )
                self.allocated_count += 1
                self.emergency_allocations += 1
                
                if self.emergency_allocations % 10 == 1:  # Log every 10th emergency allocation
                    logging.warning(
                        f"Buffer pool exhausted, emergency allocation #{self.emergency_allocations}. "
                        f"Consider increasing pool size."
                    )
                
                return buffer
                
            return self.free_buffers.pop()
            
    def return_buffer(self, buffer: SpectrumBuffer):
        """
        Return a buffer to the pool.
        
        Args:
            buffer: Buffer to return to pool
        """
        with self.lock:
            # Reset buffer state
            buffer.actual_size = 0
            buffer.coords = (0, 0, 0)
            
            # Only return pre-allocated buffers to pool
            if buffer.buffer_id < self.total_buffers:
                self.free_buffers.append(buffer)
            # Emergency allocations are just discarded (GC will handle them)
            
    def get_stats(self) -> dict:
        """
        Get buffer pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        with self.lock:
            return {
                "total_buffers": self.total_buffers,
                "free_buffers": len(self.free_buffers),
                "allocated_buffers": self.total_buffers - len(self.free_buffers),
                "emergency_allocations": self.emergency_allocations,
                "buffer_size": self.buffer_size,
                "memory_usage_mb": (self.total_buffers + self.emergency_allocations) * 
                                 self.buffer_size * (8 + 4) / 1e6  # 8 bytes for float64 + 4 for float32
            }
            
    def resize_pool(self, new_size: int):
        """
        Resize the buffer pool (adds buffers, doesn't remove existing ones).
        
        Args:
            new_size: New total number of buffers
        """
        with self.lock:
            if new_size <= self.total_buffers:
                logging.warning(f"Cannot shrink buffer pool from {self.total_buffers} to {new_size}")
                return
                
            additional_buffers = new_size - self.total_buffers
            
            for i in range(additional_buffers):
                buffer = SpectrumBuffer(
                    buffer_id=self.total_buffers + i,
                    mz_buffer=np.empty(self.buffer_size, dtype=np.float64),
                    intensity_buffer=np.empty(self.buffer_size, dtype=np.float32)
                )
                self.free_buffers.append(buffer)
                
            self.total_buffers = new_size
            logging.info(f"Expanded buffer pool to {new_size} buffers")

class InterpolatedSpectrum:
    """Container for interpolated spectrum results"""
    
    def __init__(self, 
                 coords: tuple,
                 intensities: np.ndarray,
                 interpolation_time: float = 0.0,
                 quality_metrics: Optional[dict] = None):
        """
        Initialize interpolated spectrum container.
        
        Args:
            coords: Spectrum coordinates (x, y, z)
            intensities: Interpolated intensity values
            interpolation_time: Time taken for interpolation
            quality_metrics: Optional quality metrics
        """
        self.coords = coords
        self.intensities = intensities
        self.interpolation_time = interpolation_time
        self.quality_metrics = quality_metrics or {}
        
    def __repr__(self):
        return (f"InterpolatedSpectrum(coords={self.coords}, "
                f"n_points={len(self.intensities)}, "
                f"time={self.interpolation_time:.4f}s)")

class AdaptiveMemoryManager:
    """Monitors and adapts to memory pressure"""
    
    def __init__(self, target_memory_gb: float = 8.0, safety_factor: float = 0.8):
        """
        Initialize memory manager.
        
        Args:
            target_memory_gb: Target memory usage in GB
            safety_factor: Safety factor for memory usage (0.8 = use 80% of target)
        """
        self.target_memory = target_memory_gb * 1e9 * safety_factor
        
        try:
            import psutil
            self.process = psutil.Process()
            self.psutil_available = True
        except ImportError:
            logging.warning("psutil not available, memory monitoring disabled")
            self.psutil_available = False
            self.process = None
        
    def get_available_memory(self) -> float:
        """
        Get available memory in bytes.
        
        Returns:
            Available memory in bytes, or estimate if psutil unavailable
        """
        if not self.psutil_available:
            return self.target_memory  # Return target as estimate
            
        try:
            import psutil
            mem_info = psutil.virtual_memory()
            return mem_info.available
        except Exception:
            return self.target_memory
        
    def get_process_memory(self) -> float:
        """
        Get current process memory usage in bytes.
        
        Returns:
            Process memory usage in bytes
        """
        if not self.psutil_available or self.process is None:
            return 0.0
            
        try:
            return self.process.memory_info().rss
        except Exception:
            return 0.0
        
    def calculate_optimal_buffer_count(self, 
                                     spectrum_size: int, 
                                     n_workers: int) -> int:
        """
        Calculate optimal number of buffers based on available memory.
        
        Args:
            spectrum_size: Expected spectrum size (number of data points)
            n_workers: Number of worker threads
            
        Returns:
            Optimal buffer count
        """
        # Memory per worker for interpolation (estimated 4.6MB at 300k bins)
        interpolation_memory_per_worker = 4.6e6
        interpolation_memory = n_workers * interpolation_memory_per_worker
        
        # Get available memory
        available = self.get_available_memory()
        process_memory = self.get_process_memory()
        
        # Calculate memory available for buffers
        buffer_memory = min(
            self.target_memory - interpolation_memory,
            available * 0.5  # Don't use more than 50% of available
        )
        
        # Each buffer needs space for original + interpolated data
        # mz (float64) + intensity (float32) = 8 + 4 = 12 bytes per point
        # Factor of 2 for safety margin
        memory_per_buffer = spectrum_size * 12 * 2
        
        if memory_per_buffer <= 0:
            return n_workers * 2  # Minimum fallback
            
        optimal_count = int(buffer_memory / memory_per_buffer)
        
        # Ensure reasonable bounds
        min_buffers = n_workers * 2
        max_buffers = n_workers * 10
        
        return max(min_buffers, min(optimal_count, max_buffers))
        
    def check_memory_pressure(self) -> bool:
        """
        Check if system is under memory pressure.
        
        Returns:
            True if under memory pressure, False otherwise
        """
        if not self.psutil_available:
            return False
            
        try:
            available = self.get_available_memory()
            process_mem = self.get_process_memory()
            
            # Under pressure if process uses >80% of target or <20% system memory available
            return (process_mem > self.target_memory or 
                    available < self.get_total_memory() * 0.2)
        except Exception:
            return False
    
    def get_total_memory(self) -> float:
        """
        Get total system memory.
        
        Returns:
            Total system memory in bytes
        """
        if not self.psutil_available:
            return self.target_memory / 0.8  # Estimate based on target
            
        try:
            import psutil
            return psutil.virtual_memory().total
        except Exception:
            return self.target_memory / 0.8
            
    def get_memory_stats(self) -> dict:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            "target_memory_gb": self.target_memory / 1e9,
            "available_memory_gb": self.get_available_memory() / 1e9,
            "process_memory_gb": self.get_process_memory() / 1e9,
            "total_memory_gb": self.get_total_memory() / 1e9,
            "memory_pressure": self.check_memory_pressure(),
            "psutil_available": self.psutil_available
        }