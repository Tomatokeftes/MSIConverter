"""
High-performance streaming interpolation engine for MSI data.
"""

from queue import Queue, Empty
from threading import Thread, Event
from typing import Optional, Dict, Any, Callable, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass
import time
import logging
from collections import deque

from .buffer_pool import SpectrumBufferPool, InterpolatedSpectrum, AdaptiveMemoryManager
from .strategies import create_interpolation_strategy, InterpolationStrategy
from . import BoundsInfo

if TYPE_CHECKING:
    from ..core.base_reader import BaseMSIReader

@dataclass
class InterpolationConfig:
    """Configuration for interpolation pipeline"""
    method: str = "pchip"
    target_mass_axis: Optional[np.ndarray] = None
    n_workers: int = 4
    buffer_size: int = 1000
    max_memory_gb: float = 8.0
    use_streaming: bool = True
    adaptive_workers: bool = True
    validate_quality: bool = True
    queue_timeout: float = 1.0  # Timeout for queue operations

class StreamingInterpolationEngine:
    """High-performance streaming interpolation engine"""
    
    def __init__(self, config: InterpolationConfig):
        """
        Initialize streaming interpolation engine.
        
        Args:
            config: Interpolation configuration
        """
        self.config = config
        
        # Queues for producer-consumer pattern
        self.input_queue = Queue(maxsize=config.buffer_size * 2)
        self.output_queue = Queue(maxsize=config.buffer_size)
        
        # Thread management
        self.workers = []
        self.producer_thread = None
        self.writer_thread = None
        self.shutdown_event = Event()
        
        # Buffer pool and memory management
        estimated_spectrum_size = len(config.target_mass_axis) if config.target_mass_axis is not None else 100000
        self.memory_manager = AdaptiveMemoryManager(
            target_memory_gb=config.max_memory_gb
        )
        
        optimal_buffers = self.memory_manager.calculate_optimal_buffer_count(
            estimated_spectrum_size, config.n_workers
        )
        
        self.buffer_pool = SpectrumBufferPool(
            n_buffers=optimal_buffers,
            buffer_size=estimated_spectrum_size
        )
        
        # Performance monitoring
        self.stats = {
            'spectra_read': 0,
            'spectra_interpolated': 0,
            'spectra_written': 0,
            'start_time': 0,
            'end_time': 0,
            'throughput_history': deque(maxlen=100),
            'errors': 0,
            'warnings': 0
        }
        
        # Quality monitoring
        self.quality_metrics = []
        
        logging.info(f"Initialized streaming engine with {config.n_workers} workers, "
                    f"{optimal_buffers} buffers")
        
    def process_dataset(self, 
                       reader: 'BaseMSIReader',
                       output_writer: Callable,
                       progress_callback: Optional[Callable] = None):
        """
        Main entry point for processing a dataset.
        
        Args:
            reader: MSI reader instance
            output_writer: Function to write interpolated spectra
            progress_callback: Optional progress callback function
        """
        try:
            self.stats['start_time'] = time.time()
            
            # Step 1: Get bounds and validate target mass axis
            bounds = self._extract_bounds(reader)
            if self.config.target_mass_axis is None:
                self.config.target_mass_axis = self._create_optimal_mass_axis(bounds)
                
            logging.info(f"Processing {bounds.n_spectra} spectra with "
                        f"{len(self.config.target_mass_axis)} target bins")
            
            # Step 2: Start producer thread
            self.producer_thread = Thread(
                target=self._producer_task,
                args=(reader, bounds.n_spectra),
                name="InterpolationProducer"
            )
            self.producer_thread.start()
            
            # Step 3: Start worker threads
            for i in range(self.config.n_workers):
                worker = Thread(
                    target=self._worker_task,
                    args=(i,),
                    name=f"InterpolationWorker-{i}"
                )
                worker.start()
                self.workers.append(worker)
                
            # Step 4: Start writer thread
            self.writer_thread = Thread(
                target=self._writer_task,
                args=(output_writer, bounds.n_spectra, progress_callback),
                name="InterpolationWriter"
            )
            self.writer_thread.start()
            
            # Step 5: Monitor and adapt if enabled
            if self.config.adaptive_workers:
                self._monitor_and_adapt()
                
            # Wait for completion
            self._wait_for_completion()
            
        except Exception as e:
            logging.error(f"Dataset processing failed: {e}")
            self.shutdown_event.set()
            raise
        finally:
            self.stats['end_time'] = time.time()
            
    def _producer_task(self, reader: 'BaseMSIReader', total_spectra: int):
        """
        Producer thread - reads spectra from reader.
        
        Args:
            reader: MSI reader instance
            total_spectra: Total number of spectra to read
        """
        try:
            logging.info("Producer thread started")
            
            if hasattr(reader, 'iter_spectra_buffered'):
                # Use buffered iteration if available
                for spectrum_buffer in reader.iter_spectra_buffered(self.buffer_pool):
                    if self.shutdown_event.is_set():
                        break
                        
                    self.input_queue.put(spectrum_buffer, timeout=self.config.queue_timeout)
                    self.stats['spectra_read'] += 1
                    
            else:
                # Fallback to regular iteration
                for spectrum_data in reader.iter_spectra():
                    if self.shutdown_event.is_set():
                        break
                        
                    # Get buffer and fill it
                    buffer = self.buffer_pool.get_buffer()
                    coords, mz_values, intensities = spectrum_data
                    buffer.coords = coords
                    buffer.fill(mz_values, intensities)
                    
                    self.input_queue.put(buffer, timeout=self.config.queue_timeout)
                    self.stats['spectra_read'] += 1
                    
        except Exception as e:
            logging.error(f"Producer thread failed: {e}")
            self.stats['errors'] += 1
        finally:
            # Send sentinel values to workers
            for _ in range(self.config.n_workers):
                try:
                    self.input_queue.put(None, timeout=self.config.queue_timeout)
                except:
                    pass
            logging.info(f"Producer thread finished, read {self.stats['spectra_read']} spectra")
                
    def _worker_task(self, worker_id: int):
        """
        Worker thread - performs interpolation.
        
        Args:
            worker_id: Worker thread identifier
        """
        try:
            # Create interpolation strategy for this worker
            interpolator = create_interpolation_strategy(
                self.config.method,
                validate_quality=self.config.validate_quality
            )
            
            logging.info(f"Worker {worker_id} started with {interpolator.name} strategy")
            
            while not self.shutdown_event.is_set():
                try:
                    # Get spectrum from queue
                    spectrum_buffer = self.input_queue.get(timeout=self.config.queue_timeout)
                    
                    if spectrum_buffer is None:  # Sentinel value
                        break
                        
                    # Extract data from buffer
                    mz_old, intensity_old = spectrum_buffer.get_data()
                    
                    # Perform interpolation
                    start_time = time.time()
                    
                    if hasattr(interpolator, 'interpolate_with_validation') and self.config.validate_quality:
                        intensity_new, quality_metrics = interpolator.interpolate_with_validation(
                            mz_old, intensity_old, self.config.target_mass_axis
                        )
                        # Store quality metrics for analysis
                        self.quality_metrics.append(quality_metrics)
                    else:
                        intensity_new = interpolator.interpolate_spectrum(
                            mz_old, intensity_old, self.config.target_mass_axis
                        )
                        quality_metrics = None
                    
                    interpolation_time = time.time() - start_time
                    
                    # Create result
                    result = InterpolatedSpectrum(
                        coords=spectrum_buffer.coords,
                        intensities=intensity_new,
                        interpolation_time=interpolation_time,
                        quality_metrics=quality_metrics.__dict__ if quality_metrics else None
                    )
                    
                    # Return buffer to pool
                    self.buffer_pool.return_buffer(spectrum_buffer)
                    
                    # Queue result for writing
                    self.output_queue.put(result, timeout=self.config.queue_timeout)
                    self.stats['spectra_interpolated'] += 1
                    
                except Empty:
                    continue  # Timeout, check shutdown event
                except Exception as e:
                    logging.error(f"Worker {worker_id} error: {e}")
                    self.stats['errors'] += 1
                    
        except Exception as e:
            logging.error(f"Worker {worker_id} failed: {e}")
            self.stats['errors'] += 1
        finally:
            # Send sentinel to writer
            try:
                self.output_queue.put(None, timeout=self.config.queue_timeout)
            except:
                pass
            logging.info(f"Worker {worker_id} finished, interpolated {self.stats['spectra_interpolated']} spectra")
        
    def _writer_task(self, 
                    output_writer: Callable, 
                    total_spectra: int, 
                    progress_callback: Optional[Callable]):
        """
        Writer thread - writes interpolated spectra.
        
        Args:
            output_writer: Function to write interpolated spectra
            total_spectra: Total number of spectra expected
            progress_callback: Optional progress callback
        """
        try:
            sentinels_received = 0
            logging.info("Writer thread started")
            
            while sentinels_received < self.config.n_workers:
                try:
                    result = self.output_queue.get(timeout=self.config.queue_timeout)
                    
                    if result is None:  # Sentinel from worker
                        sentinels_received += 1
                        continue
                        
                    # Write interpolated spectrum
                    output_writer(
                        result.coords, 
                        self.config.target_mass_axis, 
                        result.intensities
                    )
                    
                    self.stats['spectra_written'] += 1
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(self.stats['spectra_written'], total_spectra)
                        
                    # Update throughput tracking
                    current_time = time.time()
                    if len(self.stats['throughput_history']) == 0:
                        self.stats['throughput_history'].append((current_time, self.stats['spectra_written']))
                    elif current_time - self.stats['throughput_history'][-1][0] > 5.0:  # Every 5 seconds
                        self.stats['throughput_history'].append((current_time, self.stats['spectra_written']))
                        
                except Empty:
                    continue  # Timeout, check for more results
                except Exception as e:
                    logging.error(f"Writer thread error: {e}")
                    self.stats['errors'] += 1
                    
        except Exception as e:
            logging.error(f"Writer thread failed: {e}")
            self.stats['errors'] += 1
        finally:
            logging.info(f"Writer thread finished, wrote {self.stats['spectra_written']} spectra")
            
    def _monitor_and_adapt(self):
        """Monitor performance and adapt worker count if needed"""
        # This is a placeholder for adaptive worker scaling
        # Full implementation would monitor throughput and adjust workers
        pass
        
    def _wait_for_completion(self):
        """Wait for all threads to complete"""
        logging.info("Waiting for processing completion...")
        
        # Wait for producer
        if self.producer_thread:
            self.producer_thread.join()
            
        # Wait for workers
        for worker in self.workers:
            worker.join()
            
        # Wait for writer
        if self.writer_thread:
            self.writer_thread.join()
            
        logging.info("All threads completed")
        
    def _extract_bounds(self, reader: 'BaseMSIReader') -> BoundsInfo:
        """Extract bounds from reader"""
        # Import here to avoid circular imports
        from .bounds_detector import detect_bounds_from_reader
        return detect_bounds_from_reader(reader)
        
    def _create_optimal_mass_axis(self, bounds: BoundsInfo) -> np.ndarray:
        """Create optimal mass axis using physics model"""
        # Import here to avoid circular imports
        from .physics_models import create_physics_model
        
        # Use TOF physics as default if no instrument info available
        physics_model = create_physics_model("tof")
        
        # Create axis with reasonable number of bins (90k default from review)
        target_bins = 90000
        return physics_model.create_optimal_mass_axis(
            bounds.min_mz, bounds.max_mz, target_bins=target_bins
        )
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        elapsed_time = (self.stats['end_time'] or time.time()) - self.stats['start_time']
        
        # Calculate throughput
        overall_throughput = self.stats['spectra_written'] / elapsed_time if elapsed_time > 0 else 0.0
        
        # Calculate recent throughput from history
        recent_throughput = 0.0
        if len(self.stats['throughput_history']) >= 2:
            recent_time, recent_count = self.stats['throughput_history'][-1]
            prev_time, prev_count = self.stats['throughput_history'][-2]
            time_diff = recent_time - prev_time
            count_diff = recent_count - prev_count
            recent_throughput = count_diff / time_diff if time_diff > 0 else 0.0
        
        # Quality metrics summary
        quality_summary = {}
        if self.quality_metrics:
            tic_ratios = [m.tic_ratio for m in self.quality_metrics]
            peak_preservations = [m.peak_preservation for m in self.quality_metrics]
            interpolation_times = [m.interpolation_time for m in self.quality_metrics]
            
            quality_summary = {
                'avg_tic_ratio': np.mean(tic_ratios),
                'avg_peak_preservation': np.mean(peak_preservations),
                'avg_interpolation_time': np.mean(interpolation_times),
                'total_samples': len(self.quality_metrics)
            }
        
        return {
            'spectra_read': self.stats['spectra_read'],
            'spectra_interpolated': self.stats['spectra_interpolated'],
            'spectra_written': self.stats['spectra_written'],
            'elapsed_time': elapsed_time,
            'overall_throughput_per_sec': overall_throughput,
            'recent_throughput_per_sec': recent_throughput,
            'errors': self.stats['errors'],
            'warnings': self.stats['warnings'],
            'buffer_pool_stats': self.buffer_pool.get_stats(),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'quality_summary': quality_summary,
            'config': {
                'method': self.config.method,
                'n_workers': self.config.n_workers,
                'target_bins': len(self.config.target_mass_axis) if self.config.target_mass_axis is not None else 0
            }
        }
        
    def shutdown(self):
        """Gracefully shutdown the interpolation engine"""
        logging.info("Shutting down interpolation engine...")
        self.shutdown_event.set()
        
        # Clear queues to unblock threads
        try:
            while not self.input_queue.empty():
                self.input_queue.get_nowait()
        except:
            pass
            
        try:
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
        except:
            pass