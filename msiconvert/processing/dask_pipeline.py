"""
Dask-based interpolation pipeline for memory-efficient processing.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

# Import Dask conditionally
try:
    import dask
    from dask.distributed import Client, Future, as_completed, progress
    from dask import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    # Create dummy classes for type hints when Dask is not available
    Client = None
    Future = None
    as_completed = None
    progress = None
    delayed = None
    logging.warning("Dask not available, falling back to threading-based processing")

from ..interpolators.streaming_engine import StreamingInterpolationConfig
from ..interpolators.strategies import create_interpolation_strategy
from ..interpolators import SpectrumBuffer
from ..interpolators.buffer_pool import SpectrumBufferPool
from .memory_manager import AdaptiveMemoryManager


class DaskInterpolationPipeline:
    """
    Dask-based interpolation pipeline for memory-efficient processing.
    
    This replaces the basic threading approach with Dask for:
    - Better memory management
    - Adaptive worker scaling  
    - Progress reporting
    - Fault tolerance
    """
    
    def __init__(self, config: StreamingInterpolationConfig, client: Optional[Any] = None):
        """
        Initialize Dask pipeline.
        
        Args:
            config: Interpolation configuration
            client: Optional Dask client (will create one if not provided)
        """
        self.config = config
        self.client = client
        self.memory_manager = AdaptiveMemoryManager(
            target_memory_gb=config.max_memory_gb,
            safety_factor=0.8
        )
        
        # Performance tracking
        self.stats = {
            'spectra_read': 0,
            'spectra_interpolated': 0,
            'spectra_written': 0,
            'total_time': 0.0,
            'interpolation_time': 0.0,
            'errors': 0,
            'memory_peak_gb': 0.0
        }
        
        # Setup Dask client if not provided
        if DASK_AVAILABLE and self.client is None:
            self._setup_dask_client()
        elif not DASK_AVAILABLE:
            logging.warning("Dask not available, performance will be suboptimal")
            
    def _setup_dask_client(self) -> None:
        """Setup local Dask client with optimized settings"""
        try:
            # Calculate optimal workers based on memory
            optimal_workers = self.memory_manager.calculate_optimal_workers(
                spectrum_size=len(self.config.target_mass_axis) if self.config.target_mass_axis is not None else 100000,
                interpolation_memory_mb=4.6  # Memory per interpolation
            )
            
            # Limit workers to config maximum
            n_workers = min(optimal_workers, self.config.n_workers or 4)
            
            # Calculate memory per worker with safety buffer
            # Workers need ~350MB for interpolation based on error logs showing 278MB usage
            base_memory_gb = self.config.max_memory_gb / n_workers
            # Ensure minimum 400MB per worker with 25% safety buffer
            min_memory_gb = 0.4  # 400MB minimum
            memory_per_worker_gb = max(base_memory_gb, min_memory_gb)
            memory_per_worker = f"{memory_per_worker_gb:.1f}GB"
            
            self.client = Client(
                n_workers=min(n_workers, 4),  # LIMIT workers to max 4 for now
                threads_per_worker=1,  # Single-threaded workers for memory efficiency
                memory_limit=memory_per_worker,
                dashboard_address=None,  # Disable dashboard for production
                silence_logs=logging.ERROR,  # ONLY show errors
                processes=False,  # Use threads for simpler management
                # Disable all the logging spam
                asynchronous=False
            )
            
            logging.info(f"Dask client created with {n_workers} workers, "
                        f"{memory_per_worker} memory per worker "
                        f"(increased from calculated {base_memory_gb:.1f}GB to ensure adequate memory)")
                        
        except Exception as e:
            logging.error(f"Failed to setup Dask client: {e}")
            self.client = None
            
    def process_dataset(self, 
                       reader,
                       output_writer: Callable,
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process dataset with Dask-based interpolation.
        
        Args:
            reader: MSI data reader
            output_writer: Function to write interpolated spectra
            progress_callback: Optional progress callback
            
        Returns:
            Processing statistics
        """
        logging.info("[DEBUG] DaskInterpolationPipeline.process_dataset started")
        start_time = time.time()
        
        try:
            if DASK_AVAILABLE and self.client:
                logging.info("[DEBUG] Using Dask for processing")
                return self._process_with_dask(reader, output_writer, progress_callback)
            else:
                logging.info("[DEBUG] Falling back to streaming engine")
                return self._process_with_threading(reader, output_writer, progress_callback)
                
        finally:
            self.stats['total_time'] = time.time() - start_time
            
    def _process_with_dask(self, reader, output_writer, progress_callback) -> Dict[str, Any]:
        """Process using Dask distributed computing"""
        logging.info("Starting Dask-based interpolation processing")
        
        # Get dataset info
        dimensions = reader.get_dimensions()
        total_spectra = dimensions[0] * dimensions[1] * dimensions[2]
        
        # Calculate optimal chunk size for memory efficiency
        base_chunk_size = self.memory_manager.calculate_optimal_chunk_size(
            total_spectra=total_spectra,
            spectrum_size=len(self.config.target_mass_axis) if self.config.target_mass_axis is not None else 100000
        )
        
        # Use SMALLER chunks for better streaming to disk - aim for ~50-100 spectra per chunk
        # Calculate current number of workers
        current_workers = len(self.client.scheduler_info()['workers']) if hasattr(self.client, 'scheduler_info') else 4
        chunk_size = min(base_chunk_size, max(50, total_spectra // (current_workers * 4)))
        
        logging.info(f"Processing {total_spectra} spectra in chunks of {chunk_size}")
        
        # Create chunks of spectra
        chunks = self._create_spectrum_chunks(reader, chunk_size)
        
        # Submit interpolation tasks to Dask
        futures = []
        for chunk_id, chunk_data in enumerate(chunks):
            future = self.client.submit(
                self._interpolate_chunk,
                chunk_data,
                self.config.target_mass_axis,
                self.config.method,
                key=f'interpolate-chunk-{chunk_id}'
            )
            futures.append(future)
            
        # Process results as they complete with IMMEDIATE streaming to disk
        completed_spectra = 0
        
        # Add a proper progress bar for interpolation
        
        try:
            # Use tqdm progress bar only if no custom progress callback is provided
            if progress_callback is None:
                # Enhanced progress bar with rate and ETA
                progress_context = tqdm(
                    total=total_spectra, 
                    desc="Interpolating spectra", 
                    unit="spectrum",
                    unit_scale=True,
                    miniters=1,
                    mininterval=0.5,  # Update every 0.5 seconds
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                )
            else:
                # Use a dummy context manager when custom callback is provided
                progress_context = self._null_context()
            
            with progress_context as pbar:
                for future in as_completed(futures):
                    try:
                        # IMMEDIATE streaming - process each spectrum as it completes
                        chunk_results = future.result()
                        chunk_size = len(chunk_results)
                        
                        # Stream each spectrum immediately to disk (no accumulation)
                        for coords, intensities in chunk_results:
                            output_writer(coords, self.config.target_mass_axis, intensities)
                            self.stats['spectra_written'] += 1
                            completed_spectra += 1
                        
                        # Update progress tracking
                        if progress_callback is None and hasattr(pbar, 'update'):
                            # Update tqdm progress bar for entire chunk at once (more efficient)
                            pbar.update(chunk_size)
                        elif progress_callback is not None:
                            # Update custom progress callback
                            progress_callback(completed_spectra, total_spectra)
                        
                        # Clear chunk results immediately to free memory
                        del chunk_results
                            
                    except Exception as e:
                        # Reduce logging verbosity - only log every 10th error
                        if self.stats['errors'] % 10 == 0:
                            logging.error(f"Chunk processing failed: {e}")
                        self.stats['errors'] += 1
                        
        except KeyboardInterrupt:
            logging.info("Processing interrupted, cancelling remaining tasks")
            for future in futures:
                future.cancel()
            raise
            
        # Update final stats
        self.stats['spectra_interpolated'] = completed_spectra
        
        return self.stats
        
    def _process_with_threading(self, reader, output_writer, progress_callback) -> Dict[str, Any]:
        """Fallback to threading-based processing when Dask unavailable"""
        logging.warning("Using threading fallback - performance may be suboptimal")
        
        # Import the existing streaming engine
        from ..interpolators.streaming_engine import StreamingInterpolationEngine
        
        # Create threading-based config
        thread_config = StreamingInterpolationConfig(
            method=self.config.method,
            target_mass_axis=self.config.target_mass_axis,
            n_workers=min(self.config.n_workers or 4, 8),  # Limit workers for threading
            buffer_size=self.config.buffer_size,
            validate_quality=self.config.validate_quality
        )
        
        # Use existing streaming engine
        engine = StreamingInterpolationEngine(thread_config)
        engine.process_dataset(reader, output_writer, progress_callback)
        
        # Return compatible stats
        return engine.get_performance_stats()
        
    def _create_spectrum_chunks(self, reader, chunk_size: int) -> List[List[Tuple]]:
        """Create chunks of spectrum data for processing"""
        chunks = []
        current_chunk = []
        
        # Create buffer pool for reading
        buffer_pool = SpectrumBufferPool(
            n_buffers=min(chunk_size * 2, 1000),
            buffer_size=100000  # Estimate for original spectrum size
        )
        
        try:
            for spectrum_buffer in reader.iter_spectra_buffered(buffer_pool):
                # Extract data from buffer
                mz_data, intensity_data = spectrum_buffer.get_data()
                
                # Store spectrum data (copy to avoid buffer reuse issues)
                spectrum_data = (
                    spectrum_buffer.coords,
                    mz_data.copy(),
                    intensity_data.copy()
                )
                current_chunk.append(spectrum_data)
                self.stats['spectra_read'] += 1
                
                # Return buffer to pool
                buffer_pool.return_buffer(spectrum_buffer)
                
                # Create new chunk when full
                if len(current_chunk) >= chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = []
                    
        except Exception as e:
            logging.error(f"Error reading spectra: {e}")
            self.stats['errors'] += 1
            
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        logging.info(f"Created {len(chunks)} chunks for processing")
        return chunks
        
    @staticmethod
    def _interpolate_chunk(chunk_data: List[Tuple], 
                          target_mass_axis: NDArray[np.float64],
                          method: str) -> List[Tuple]:
        """
        Static method for Dask worker to interpolate a chunk of spectra.
        
        Args:
            chunk_data: List of (coords, mz_array, intensity_array) tuples
            target_mass_axis: Target mass axis for interpolation
            method: Interpolation method name
            
        Returns:
            List of (coords, interpolated_intensities) tuples
        """
        import gc
        
        # Create interpolator for this worker
        interpolator = create_interpolation_strategy(method, validate_quality=False)
        
        results = []
        for coords, mz_old, intensity_old in chunk_data:
            try:
                # Perform interpolation
                intensity_new = interpolator.interpolate_spectrum(
                    mz_old, intensity_old, target_mass_axis
                )
                
                results.append((coords, intensity_new))
                
                # Clear input arrays to free memory immediately
                del mz_old, intensity_old
                
            except Exception as e:
                logging.error(f"Failed to interpolate spectrum at {coords}: {e}")
                # Return zeros on failure to maintain data integrity
                results.append((coords, np.zeros_like(target_mass_axis)))
                
        # Clear input chunk data and force garbage collection
        del chunk_data
        gc.collect()
        
        return results
        
    def _null_context(self):
        """Null context manager for when Dask progress is unavailable"""
        class NullContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NullContext()
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if self.stats['total_time'] > 0:
            stats['overall_throughput_per_sec'] = self.stats['spectra_written'] / self.stats['total_time']
        else:
            stats['overall_throughput_per_sec'] = 0.0
            
        if self.stats['interpolation_time'] > 0:
            stats['interpolation_throughput_per_sec'] = self.stats['spectra_interpolated'] / self.stats['interpolation_time']
        else:
            stats['interpolation_throughput_per_sec'] = 0.0
            
        # Add memory statistics
        if hasattr(self, 'memory_manager'):
            stats['memory_usage'] = self.memory_manager.get_memory_stats()
            
        return stats
        
    def shutdown(self) -> None:
        """Clean shutdown of Dask resources"""
        if self.client and DASK_AVAILABLE:
            try:
                self.client.close()
                logging.info("Dask client closed successfully")
            except Exception as e:
                logging.error(f"Error closing Dask client: {e}")