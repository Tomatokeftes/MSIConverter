# MSI Interpolation Module - Ultra-Detailed Implementation Guide for Claude Sonnet

## 🎯 Overview
This guide provides step-by-step, copy-paste ready implementation instructions for adding intelligent interpolation to MSIConverter. The system will handle terabyte-scale datasets while reducing file sizes by 50-90% using physics-based mass axis resampling.

**Key Performance Facts:**
- Reading: 17,531 spectra/sec (ImzML), 2,705 spectra/sec (Bruker)
- Interpolation: 427 spectra/sec (90k bins) → 211 spectra/sec (300k bins)
- **Interpolation is 41-83x slower than reading - it's THE bottleneck**

## 📦 Data Types and Interfaces

### Core Data Types
```python
from typing import TypedDict, NamedTuple, Protocol
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

# Spectrum data type used throughout the system
class SpectrumData(NamedTuple):
    """Standard spectrum representation"""
    coords: Tuple[int, int, int]  # (x, y, z) coordinates
    mz_values: NDArray[np.float64]  # m/z values
    intensities: NDArray[np.float32]  # intensity values

# Bounds information from metadata
@dataclass
class BoundsInfo:
    """Dataset bounds from metadata extraction"""
    min_mz: float
    max_mz: float
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    n_spectra: int
    
# Buffer for zero-copy operations
@dataclass
class SpectrumBuffer:
    """Reusable buffer for spectrum data"""
    buffer_id: int
    mz_buffer: NDArray[np.float64]  # Pre-allocated
    intensity_buffer: NDArray[np.float32]  # Pre-allocated
    actual_size: int = 0  # Actual data size
    
    def fill(self, mz: NDArray, intensity: NDArray) -> None:
        """Fill buffer with data"""
        self.actual_size = len(mz)
        self.mz_buffer[:self.actual_size] = mz
        self.intensity_buffer[:self.actual_size] = intensity
        
    def get_data(self) -> Tuple[NDArray, NDArray]:
        """Get actual data from buffer"""
        return (self.mz_buffer[:self.actual_size], 
                self.intensity_buffer[:self.actual_size])
```

### Reader Interface Extensions
```python
# These methods MUST be added to BaseMSIReader
class BaseMSIReader(ABC):
    # Existing methods...
    
    @abstractmethod
    def get_mass_bounds(self) -> Tuple[float, float]:
        """Get min/max m/z WITHOUT scanning all spectra"""
        pass
        
    @abstractmethod
    def get_spatial_bounds(self) -> Dict[str, int]:
        """Get spatial bounds WITHOUT scanning all spectra"""
        pass
        
    @abstractmethod
    def get_estimated_memory_usage(self) -> Dict[str, float]:
        """Estimate memory requirements"""
        pass
        
    @abstractmethod
    def iter_spectra_buffered(self, buffer_pool: BufferPool) -> Iterator[SpectrumBuffer]:
        """Iterate using pre-allocated buffers"""
        pass
```

## 🏗️ Phase 1: Metadata Infrastructure (Week 1)

### 1.1 Directory Structure
```bash
# Create these directories first
msiconvert/metadata/
├── __init__.py
├── base_extractor.py
├── bruker_extractor.py
├── imzml_extractor.py
└── metadata_models.py

msiconvert/interpolators/
├── __init__.py
├── bounds_detector.py
├── physics_models.py
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── pchip_strategy.py
│   └── adaptive_strategy.py
└── streaming_engine.py
```

### 1.2 Metadata Models (metadata_models.py)
```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class InstrumentMetadata(BaseModel):
    """Instrument-specific metadata"""
    instrument_type: str  # "TOF", "Orbitrap", "FTICR"
    model: Optional[str] = None
    resolution_at_400: Optional[float] = None
    
class AcquisitionMetadata(BaseModel):
    """Acquisition parameters"""
    mass_range_lower: float = Field(alias="MzAcqRangeLower")
    mass_range_upper: float = Field(alias="MzAcqRangeUpper")
    pixel_size_x: float = Field(alias="SpotSizeX")
    pixel_size_y: float = Field(alias="SpotSizeY")
    
class SpatialMetadata(BaseModel):
    """Spatial bounds from metadata"""
    min_x: int = Field(alias="ImagingAreaMinXIndexPos")
    max_x: int = Field(alias="ImagingAreaMaxXIndexPos")
    min_y: int = Field(alias="ImagingAreaMinYIndexPos")
    max_y: int = Field(alias="ImagingAreaMaxYIndexPos")
    
class DatasetMetadata(BaseModel):
    """Complete dataset metadata"""
    instrument: InstrumentMetadata
    acquisition: AcquisitionMetadata
    spatial: SpatialMetadata
    raw_metadata: Dict[str, Any] = {}
```

### 1.3 Bruker Metadata Extractor (bruker_extractor.py)
```python
import sqlite3
from pathlib import Path
from typing import Dict, Any, Tuple
from .base_extractor import BaseMetadataExtractor
from .metadata_models import DatasetMetadata, InstrumentMetadata, AcquisitionMetadata, SpatialMetadata

class BrukerMetadataExtractor(BaseMetadataExtractor):
    """Extract metadata from Bruker .d folders WITHOUT reading spectra"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.db_path = data_path / "analysis.tsf"  # or .tdf
        if not self.db_path.exists():
            self.db_path = data_path / "analysis.tdf"
            
    def extract_mass_bounds(self) -> Tuple[float, float]:
        """Get mass bounds from GlobalMetadata table"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Query for mass bounds
            query = """
            SELECT Value FROM GlobalMetadata 
            WHERE Key IN ('MzAcqRangeLower', 'MzAcqRangeUpper')
            ORDER BY Key
            """
            results = cursor.execute(query).fetchall()
            
            if len(results) != 2:
                raise ValueError("Mass bounds not found in metadata")
                
            min_mz = float(results[0][0])
            max_mz = float(results[1][0])
            
            return min_mz, max_mz
            
    def extract_spatial_bounds(self) -> Dict[str, int]:
        """Get spatial bounds from GlobalMetadata table"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Query for spatial bounds
            keys = [
                'ImagingAreaMinXIndexPos',
                'ImagingAreaMaxXIndexPos', 
                'ImagingAreaMinYIndexPos',
                'ImagingAreaMaxYIndexPos'
            ]
            
            query = f"""
            SELECT Key, Value FROM GlobalMetadata 
            WHERE Key IN ({','.join(['?' for _ in keys])})
            """
            
            results = cursor.execute(query, keys).fetchall()
            bounds = {row[0]: int(row[1]) for row in results}
            
            return {
                'min_x': bounds['ImagingAreaMinXIndexPos'],
                'max_x': bounds['ImagingAreaMaxXIndexPos'],
                'min_y': bounds['ImagingAreaMinYIndexPos'],
                'max_y': bounds['ImagingAreaMaxYIndexPos']
            }
            
    def extract_instrument_info(self) -> Dict[str, Any]:
        """Get instrument type from metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get instrument model
            query = "SELECT Value FROM GlobalMetadata WHERE Key = 'InstrumentName'"
            result = cursor.execute(query).fetchone()
            
            instrument_name = result[0] if result else "Unknown"
            
            # Determine type from name
            instrument_type = "TOF"  # default
            if "orbitrap" in instrument_name.lower():
                instrument_type = "Orbitrap"
            elif "fticr" in instrument_name.lower():
                instrument_type = "FTICR"
                
            return {
                "type": instrument_type,
                "model": instrument_name
            }
```

### 1.4 ImzML Metadata Extractor (imzml_extractor.py)
```python
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Tuple
from .base_extractor import BaseMetadataExtractor

class ImzMLMetadataExtractor(BaseMetadataExtractor):
    """Extract metadata from imzML header WITHOUT parsing all spectra"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        
    def extract_from_header(self) -> Dict[str, Any]:
        """Parse imzML header efficiently"""
        # Parse only the header section
        tree = ET.parse(self.data_path)
        root = tree.getroot()
        
        # Extract namespace
        ns = {'mzml': root.tag.split('}')[0].strip('{')}
        
        metadata = {}
        
        # Get scan settings
        scan_settings = root.find('.//mzml:scanSettingsList/mzml:scanSettings', ns)
        if scan_settings:
            # Extract pixel size
            for param in scan_settings.findall('.//mzml:cvParam', ns):
                if param.get('name') == 'pixel size x':
                    metadata['pixel_size_x'] = float(param.get('value'))
                elif param.get('name') == 'pixel size y':
                    metadata['pixel_size_y'] = float(param.get('value'))
                    
        # Get mass range from first spectrum (quick peek)
        first_spectrum = root.find('.//mzml:spectrum[1]', ns)
        if first_spectrum:
            for param in first_spectrum.findall('.//mzml:cvParam', ns):
                if param.get('name') == 'lowest observed m/z':
                    metadata['min_mz'] = float(param.get('value'))
                elif param.get('name') == 'highest observed m/z':
                    metadata['max_mz'] = float(param.get('value'))
                    
        return metadata
```

## 🏗️ Phase 2: Physics Models & Interpolation Core (Week 2)

### 2.1 Physics Models (physics_models.py)
```python
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple

class InstrumentPhysics(ABC):
    """Base class for instrument-specific physics models"""
    
    @abstractmethod
    def calculate_resolution_at_mz(self, mz: float) -> float:
        """Calculate mass resolution at given m/z"""
        pass
        
    @abstractmethod
    def calculate_width_at_mz(self, mz: float, width_at_reference: float, reference_mz: float) -> float:
        """Calculate bin width at given m/z based on width at reference m/z"""
        pass
        
    def create_optimal_mass_axis(self, 
                                min_mz: float, 
                                max_mz: float,
                                target_bins: Optional[int] = None,
                                width_at_mz: Optional[Tuple[float, float]] = None) -> NDArray[np.float64]:
        """
        Create physics-based optimal mass axis (like SCiLS).
        
        Args:
            min_mz: Minimum m/z value
            max_mz: Maximum m/z value  
            target_bins: Target number of bins (option 1)
            width_at_mz: Tuple of (width, reference_mz) for width at specific m/z (option 2)
            
        Must specify either target_bins OR width_at_mz, not both.
        """
        if (target_bins is None) == (width_at_mz is None):
            raise ValueError("Must specify either target_bins OR width_at_mz, not both or neither")
            
        if width_at_mz is not None:
            # Option 2: Use specified width at reference m/z (like SCiLS)
            width_at_ref, ref_mz = width_at_mz
            return self._create_axis_from_width(min_mz, max_mz, width_at_ref, ref_mz)
        else:
            # Option 1: Use target number of bins (like SCiLS)
            return self._create_axis_from_bins(min_mz, max_mz, target_bins)
            
    def _create_axis_from_width(self, min_mz: float, max_mz: float, 
                               width_at_ref: float, ref_mz: float) -> NDArray[np.float64]:
        """Create axis using specified width at reference m/z"""
        axis = [min_mz]
        current_mz = min_mz
        
        while current_mz < max_mz:
            # Calculate width at current m/z based on physics
            width = self.calculate_width_at_mz(current_mz, width_at_ref, ref_mz)
            current_mz += width
            
            if current_mz <= max_mz:
                axis.append(current_mz)
                
        return np.array(axis)
        
    def _create_axis_from_bins(self, min_mz: float, max_mz: float, 
                              target_bins: int) -> NDArray[np.float64]:
        """Create axis with approximately target number of bins"""
        # First, estimate average width needed
        avg_width_estimate = (max_mz - min_mz) / target_bins
        
        # Use this to estimate width at reference m/z (e.g., 400)
        ref_mz = 400.0
        if min_mz <= ref_mz <= max_mz:
            width_at_ref = avg_width_estimate
        else:
            # Use middle of range as reference
            ref_mz = (min_mz + max_mz) / 2
            width_at_ref = avg_width_estimate
            
        # Create initial axis
        axis = self._create_axis_from_width(min_mz, max_mz, width_at_ref, ref_mz)
        
        # Adjust if we're far from target
        actual_bins = len(axis)
        if abs(actual_bins - target_bins) > target_bins * 0.1:  # >10% off
            # Scale the reference width
            scaling_factor = actual_bins / target_bins
            adjusted_width = width_at_ref / scaling_factor
            axis = self._create_axis_from_width(min_mz, max_mz, adjusted_width, ref_mz)
            
        return axis

class TOFPhysics(InstrumentPhysics):
    """Time-of-Flight physics model"""
    
    def __init__(self, resolution_at_400: float = 10000):
        self.resolution_at_400 = resolution_at_400
        
    def calculate_resolution_at_mz(self, mz: float) -> float:
        """R ∝ sqrt(m/z) for linear TOF"""
        return self.resolution_at_400 * np.sqrt(mz / 400.0)
        
    def calculate_width_at_mz(self, mz: float, width_at_reference: float, reference_mz: float) -> float:
        """
        Width scales with sqrt(m/z) for TOF.
        width(mz) = width_ref * sqrt(mz / mz_ref)
        """
        return width_at_reference * np.sqrt(mz / reference_mz)

class OrbitrapPhysics(InstrumentPhysics):
    """Orbitrap physics model"""
    
    def __init__(self, resolution_at_400: float = 120000):
        self.resolution_at_400 = resolution_at_400
        
    def calculate_resolution_at_mz(self, mz: float) -> float:
        """R = R_400 * sqrt(400/m)"""
        return self.resolution_at_400 * np.sqrt(400.0 / mz)
        
    def calculate_width_at_mz(self, mz: float, width_at_reference: float, reference_mz: float) -> float:
        """
        Width scales with sqrt(m/z) for Orbitrap (inverse of resolution).
        width(mz) = width_ref * sqrt(mz / mz_ref)
        """
        return width_at_reference * np.sqrt(mz / reference_mz)

# Usage examples:
physics = TOFPhysics()

# Option 1: Specify number of bins (like SCiLS)
mass_axis_1 = physics.create_optimal_mass_axis(100, 1000, target_bins=90000)

# Option 2: Specify width at reference m/z (like SCiLS)
# E.g., 0.01 Da width at m/z 400
mass_axis_2 = physics.create_optimal_mass_axis(100, 1000, width_at_mz=(0.01, 400))
```

### 2.2 PCHIP Interpolation Strategy (pchip_strategy.py)
```python
import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import Tuple, Optional
from numpy.typing import NDArray
import numba

class PchipInterpolationStrategy:
    """PCHIP interpolation with optimizations"""
    
    def __init__(self, use_caching: bool = True, use_simd: bool = True):
        self.use_caching = use_caching
        self.use_simd = use_simd
        self._coefficient_cache = {}  # LRU cache for coefficients
        
    def interpolate_spectrum(self, 
                           mz_old: NDArray[np.float64],
                           intensity_old: NDArray[np.float32],
                           mz_new: NDArray[np.float64]) -> NDArray[np.float32]:
        """Interpolate spectrum to new mass axis"""
        
        # Handle edge cases
        if len(mz_old) == 0:
            return np.zeros(len(mz_new), dtype=np.float32)
            
        if len(mz_old) == 1:
            # Single peak - place at nearest new m/z
            idx = np.searchsorted(mz_new, mz_old[0])
            result = np.zeros(len(mz_new), dtype=np.float32)
            if 0 <= idx < len(mz_new):
                result[idx] = intensity_old[0]
            return result
            
        # Check cache if enabled
        if self.use_caching:
            cache_key = self._get_cache_key(mz_old, mz_new)
            if cache_key in self._coefficient_cache:
                coeffs = self._coefficient_cache[cache_key]
                return self._apply_coefficients(coeffs, intensity_old)
                
        # Create interpolator
        interpolator = PchipInterpolator(mz_old, intensity_old, 
                                       extrapolate=False)
        
        # Interpolate
        intensity_new = interpolator(mz_new)
        
        # Handle extrapolation (set to 0)
        intensity_new = np.nan_to_num(intensity_new, nan=0.0)
        
        # Ensure non-negative (PCHIP guarantees this, but be safe)
        intensity_new = np.maximum(intensity_new, 0.0)
        
        return intensity_new.astype(np.float32)
        
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def _interpolate_batch_simd(self, 
                               mz_old_list: List[NDArray],
                               intensity_old_list: List[NDArray],
                               mz_new: NDArray) -> List[NDArray]:
        """SIMD-optimized batch interpolation"""
        # Numba-optimized parallel interpolation
        pass
```

### 2.3 Streaming Interpolation Engine (streaming_engine.py)
```python
from queue import Queue
from threading import Thread
from typing import Optional, Dict, Any, Callable
import numpy as np
from dataclasses import dataclass
import time

@dataclass
class InterpolationConfig:
    """Configuration for interpolation pipeline"""
    method: str = "pchip"
    target_mass_axis: Optional[NDArray[np.float64]] = None
    n_workers: int = 4
    buffer_size: int = 1000
    max_memory_gb: float = 8.0
    use_streaming: bool = True
    adaptive_workers: bool = True
    validate_quality: bool = True
    
class StreamingInterpolationEngine:
    """High-performance streaming interpolation engine"""
    
    def __init__(self, config: InterpolationConfig):
        self.config = config
        self.input_queue = Queue(maxsize=config.buffer_size * 2)
        self.output_queue = Queue(maxsize=config.buffer_size)
        self.buffer_pool = SpectrumBufferPool(
            n_buffers=config.n_workers * 3,
            buffer_size=len(config.target_mass_axis) if config.target_mass_axis else 100000
        )
        self.workers = []
        self.producer_thread = None
        self.writer_thread = None
        
        # Performance monitoring
        self.stats = {
            'spectra_read': 0,
            'spectra_interpolated': 0,
            'spectra_written': 0,
            'throughput_history': deque(maxlen=100)
        }
        
    def process_dataset(self, 
                       reader: BaseMSIReader,
                       output_writer: Callable,
                       progress_callback: Optional[Callable] = None):
        """Main entry point for processing"""
        
        # Step 1: Get bounds and create optimal mass axis
        bounds = self._extract_bounds(reader)
        if self.config.target_mass_axis is None:
            self.config.target_mass_axis = self._create_optimal_mass_axis(bounds)
            
        # Step 2: Start producer thread
        self.producer_thread = Thread(
            target=self._producer_task,
            args=(reader, bounds.n_spectra)
        )
        self.producer_thread.start()
        
        # Step 3: Start worker threads
        for i in range(self.config.n_workers):
            worker = Thread(
                target=self._worker_task,
                args=(i,)
            )
            worker.start()
            self.workers.append(worker)
            
        # Step 4: Start writer thread
        self.writer_thread = Thread(
            target=self._writer_task,
            args=(output_writer, bounds.n_spectra, progress_callback)
        )
        self.writer_thread.start()
        
        # Step 5: Monitor and adapt
        if self.config.adaptive_workers:
            self._monitor_and_adapt()
            
        # Wait for completion
        self._wait_for_completion()
        
    def _producer_task(self, reader: BaseMSIReader, total_spectra: int):
        """Producer thread - reads spectra"""
        try:
            for spectrum_buffer in reader.iter_spectra_buffered(self.buffer_pool):
                self.input_queue.put(spectrum_buffer)
                self.stats['spectra_read'] += 1
                
        finally:
            # Send sentinel values
            for _ in range(self.config.n_workers):
                self.input_queue.put(None)
                
    def _worker_task(self, worker_id: int):
        """Worker thread - performs interpolation"""
        interpolator = self._create_interpolator()
        
        while True:
            # Get spectrum from queue
            spectrum_buffer = self.input_queue.get()
            if spectrum_buffer is None:  # Sentinel
                break
                
            # Extract data
            mz_old, intensity_old = spectrum_buffer.get_data()
            
            # Interpolate
            start_time = time.time()
            intensity_new = interpolator.interpolate_spectrum(
                mz_old, intensity_old, self.config.target_mass_axis
            )
            interpolation_time = time.time() - start_time
            
            # Create result
            result = InterpolatedSpectrum(
                coords=spectrum_buffer.coords,
                intensities=intensity_new,
                interpolation_time=interpolation_time
            )
            
            # Return buffer to pool
            self.buffer_pool.return_buffer(spectrum_buffer)
            
            # Queue for writing
            self.output_queue.put(result)
            self.stats['spectra_interpolated'] += 1
            
        # Send sentinel
        self.output_queue.put(None)
        
    def _writer_task(self, output_writer: Callable, total_spectra: int, 
                    progress_callback: Optional[Callable]):
        """Writer thread - writes interpolated spectra"""
        sentinels_received = 0
        
        with tqdm(total=total_spectra, desc="Interpolating") as pbar:
            while sentinels_received < self.config.n_workers:
                result = self.output_queue.get()
                
                if result is None:
                    sentinels_received += 1
                    continue
                    
                # Write spectrum
                output_writer(result.coords, self.config.target_mass_axis, 
                            result.intensities)
                
                self.stats['spectra_written'] += 1
                pbar.update(1)
                
                if progress_callback:
                    progress_callback(self.stats['spectra_written'], total_spectra)
```

## 🏗️ Phase 3: Memory Management (Week 3)

### 3.1 Buffer Pool Implementation
```python
from typing import List, Optional
import numpy as np
from threading import Lock

class SpectrumBufferPool:
    """Thread-safe buffer pool for zero-copy operations"""
    
    def __init__(self, n_buffers: int = 100, buffer_size: int = 100000):
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
        
    def get_buffer(self) -> SpectrumBuffer:
        """Get a buffer from the pool"""
        with self.lock:
            if not self.free_buffers:
                # Emergency allocation
                buffer = SpectrumBuffer(
                    buffer_id=1000 + self.allocated_count,
                    mz_buffer=np.empty(self.buffer_size, dtype=np.float64),
                    intensity_buffer=np.empty(self.buffer_size, dtype=np.float32)
                )
                self.allocated_count += 1
                return buffer
                
            return self.free_buffers.pop()
            
    def return_buffer(self, buffer: SpectrumBuffer):
        """Return a buffer to the pool"""
        with self.lock:
            buffer.actual_size = 0  # Reset
            self.free_buffers.append(buffer)
```

### 3.2 Adaptive Memory Manager
```python
import psutil
from typing import Dict
import gc

class AdaptiveMemoryManager:
    """Monitors and adapts to memory pressure"""
    
    def __init__(self, target_memory_gb: float = 8.0, safety_factor: float = 0.8):
        self.target_memory = target_memory_gb * 1e9 * safety_factor
        self.process = psutil.Process()
        
    def get_available_memory(self) -> float:
        """Get available memory in bytes"""
        mem_info = psutil.virtual_memory()
        return mem_info.available
        
    def get_process_memory(self) -> float:
        """Get current process memory usage"""
        return self.process.memory_info().rss
        
    def calculate_optimal_buffer_count(self, spectrum_size: int, 
                                     n_workers: int) -> int:
        """Calculate optimal number of buffers"""
        # Memory per worker for interpolation (4.6MB at 300k bins)
        interpolation_memory = n_workers * 4.6e6
        
        # Available for buffers
        available = self.get_available_memory()
        process_memory = self.get_process_memory()
        
        buffer_memory = min(
            self.target_memory - interpolation_memory,
            available * 0.5  # Don't use more than 50% of available
        )
        
        # Each buffer needs space for original + interpolated
        memory_per_buffer = spectrum_size * (8 + 4) * 2  # mz + intensity, x2
        
        optimal_count = int(buffer_memory / memory_per_buffer)
        
        # Ensure reasonable bounds
        return max(n_workers * 2, min(optimal_count, n_workers * 10))
        
    def check_memory_pressure(self) -> bool:
        """Check if under memory pressure"""
        available = self.get_available_memory()
        process_mem = self.get_process_memory()
        
        # Under pressure if process uses >80% of target or <20% available
        return (process_mem > self.target_memory or 
                available < psutil.virtual_memory().total * 0.2)
```

## 🏗️ Phase 4: Quality Validation

### 4.1 Real-time Quality Monitor
```python
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple

class InterpolationQualityMonitor:
    """Monitor interpolation quality in real-time"""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            'tic_deviation': 0.01,  # 1% TIC deviation allowed
            'peak_preservation': 0.95,  # 95% peaks must be preserved
            'mass_accuracy_ppm': 5.0  # 5 ppm mass accuracy
        }
        
        self.metrics = defaultdict(list)
        self.warnings = []
        
    def validate_spectrum(self, 
                         original: Tuple[NDArray, NDArray],
                         interpolated: NDArray,
                         new_mass_axis: NDArray) -> Dict[str, float]:
        """Validate single spectrum interpolation"""
        mz_old, intensity_old = original
        
        # Total Ion Current preservation
        tic_old = np.sum(intensity_old)
        tic_new = np.sum(interpolated)
        
        if tic_old > 0:
            tic_ratio = tic_new / tic_old
            if abs(1 - tic_ratio) > self.thresholds['tic_deviation']:
                self.warnings.append(
                    f"TIC deviation: {tic_ratio:.3f} (expected ~1.0)"
                )
        else:
            tic_ratio = 1.0
            
        # Peak preservation
        peaks_old = self._find_peaks(intensity_old)
        peaks_new = self._find_peaks(interpolated)
        
        if len(peaks_old) > 0:
            peak_preservation = len(peaks_new) / len(peaks_old)
        else:
            peak_preservation = 1.0
            
        # Store metrics
        self.metrics['tic_ratios'].append(tic_ratio)
        self.metrics['peak_preservation'].append(peak_preservation)
        
        return {
            'tic_ratio': tic_ratio,
            'peak_preservation': peak_preservation,
            'n_peaks_original': len(peaks_old),
            'n_peaks_interpolated': len(peaks_new)
        }
        
    def _find_peaks(self, intensities: NDArray, 
                   min_height: Optional[float] = None) -> NDArray:
        """Simple peak detection"""
        if min_height is None:
            min_height = np.max(intensities) * 0.01  # 1% of max
            
        # Find local maxima
        peaks = []
        for i in range(1, len(intensities) - 1):
            if (intensities[i] > intensities[i-1] and 
                intensities[i] > intensities[i+1] and
                intensities[i] > min_height):
                peaks.append(i)
                
        return np.array(peaks)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get quality summary"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n_samples': len(values)
                }
                
        summary['warnings'] = self.warnings[-10:]  # Last 10 warnings
        summary['total_warnings'] = len(self.warnings)
        
        return summary
```

## 🏗️ Phase 5: Integration Points

### 5.1 Update Base Converter (base_converter.py)
```python
# Add these methods to BaseMSIConverter
class BaseMSIConverter(ABC):
    # Existing methods...
    
    def _should_interpolate(self) -> bool:
        """Decide if interpolation should be applied"""
        # Check configuration
        if not self.config.enable_interpolation:
            return False
            
        # Check if reader supports required methods
        required_methods = ['get_mass_bounds', 'get_spatial_bounds', 
                          'iter_spectra_buffered']
        for method in required_methods:
            if not hasattr(self.reader, method):
                logging.warning(f"Reader missing {method}, skipping interpolation")
                return False
                
        return True
        
    def _setup_interpolation(self) -> InterpolationConfig:
        """Setup interpolation configuration"""
        # Get bounds from reader
        min_mz, max_mz = self.reader.get_mass_bounds()
        bounds_info = BoundsInfo(
            min_mz=min_mz,
            max_mz=max_mz,
            **self.reader.get_spatial_bounds(),
            n_spectra=self.reader.get_dimensions()[0] * 
                     self.reader.get_dimensions()[1] * 
                     self.reader.get_dimensions()[2]
        )
        
        # Create physics model
        instrument_info = self.reader.get_metadata().get('instrument', {})
        physics_model = self._create_physics_model(instrument_info)
        
        # Create optimal mass axis using SCiLS-like options
        if self.config.interpolation_width:
            # Option 2: Use width at reference m/z
            width_at_mz = (self.config.interpolation_width, 
                          self.config.interpolation_width_mz)
            target_mass_axis = physics_model.create_optimal_mass_axis(
                min_mz, max_mz, width_at_mz=width_at_mz
            )
        else:
            # Option 1: Use number of bins
            target_bins = self.config.interpolation_bins or 90000
            target_mass_axis = physics_model.create_optimal_mass_axis(
                min_mz, max_mz, target_bins=target_bins
            )
        
        # Create config
        return InterpolationConfig(
            method=self.config.interpolation_method,
            target_mass_axis=target_mass_axis,
            n_workers=self._calculate_optimal_workers(),
            buffer_size=1000,
            max_memory_gb=self.config.max_memory_gb,
            use_streaming=True,
            adaptive_workers=self.config.adaptive_workers,
            validate_quality=self.config.validate_quality
        )
        
    def _convert_with_interpolation(self):
        """New conversion path with interpolation"""
        # Setup interpolation
        interp_config = self._setup_interpolation()
        
        # Create streaming engine
        engine = StreamingInterpolationEngine(interp_config)
        
        # Create output writer specific to format
        output_writer = self._create_interpolation_writer(
            interp_config.target_mass_axis
        )
        
        # Process dataset
        engine.process_dataset(
            reader=self.reader,
            output_writer=output_writer,
            progress_callback=self.progress_callback
        )
        
        # Finalize output
        self._finalize_interpolated_output(engine.stats)
```

### 5.2 Update SpatialData Converter (spatialdata_converter.py)
```python
# Add these methods to SpatialDataConverter
class SpatialDataConverter(BaseMSIConverter):
    # Existing methods...
    
    def _create_interpolation_writer(self, target_mass_axis: NDArray[np.float64]):
        """Create writer function for interpolated data"""
        # Store target mass axis
        self._interpolated_mass_axis = target_mass_axis
        
        # Pre-allocate output structures
        self._setup_interpolated_structures(target_mass_axis)
        
        def write_interpolated_spectrum(coords: Tuple[int, int, int],
                                      mass_axis: NDArray[np.float64],
                                      intensities: NDArray[np.float32]):
            """Write single interpolated spectrum"""
            # Add to sparse matrix
            x, y, z = coords
            
            if self.slice_by_slice:
                slice_id = f"{self.dataset_id}_z{z}"
                if slice_id in self._slice_data:
                    pixel_idx = y * self._dimensions[0] + x
                    self._slice_data[slice_id]['sparse_data'][pixel_idx, :] = intensities
            else:
                pixel_idx = self._get_pixel_index(x, y, z)
                self._sparse_data[pixel_idx, :] = intensities
                
        return write_interpolated_spectrum
        
    def _finalize_interpolated_output(self, stats: Dict[str, Any]):
        """Finalize output with interpolation metadata"""
        # Add interpolation metadata
        interp_metadata = {
            'interpolation_method': self.config.interpolation_method,
            'original_mass_points': len(self._original_mass_axis),
            'interpolated_mass_points': len(self._interpolated_mass_axis),
            'size_reduction_factor': len(self._original_mass_axis) / 
                                   len(self._interpolated_mass_axis),
            'interpolation_stats': stats
        }
        
        # Update table metadata
        for table in self.spatial_data.tables.values():
            table.uns['interpolation'] = interp_metadata
```

### 5.3 Update Reader Classes
```python
# Add to bruker_reader.py
class BrukerReader(BaseMSIReader):
    # Existing methods...
    
    def get_mass_bounds(self) -> Tuple[float, float]:
        """Get mass bounds from metadata WITHOUT scanning spectra"""
        if not hasattr(self, '_mass_bounds'):
            extractor = BrukerMetadataExtractor(self.data_path)
            self._mass_bounds = extractor.extract_mass_bounds()
        return self._mass_bounds
        
    def get_spatial_bounds(self) -> Dict[str, int]:
        """Get spatial bounds from metadata"""
        if not hasattr(self, '_spatial_bounds'):
            extractor = BrukerMetadataExtractor(self.data_path)
            self._spatial_bounds = extractor.extract_spatial_bounds()
        return self._spatial_bounds
        
    def iter_spectra_buffered(self, buffer_pool: SpectrumBufferPool) -> Iterator[SpectrumBuffer]:
        """Iterate using pre-allocated buffers"""
        # Use existing iteration logic but with buffers
        for coords, mzs, intensities in self.iter_spectra():
            # Get buffer from pool
            buffer = buffer_pool.get_buffer()
            
            # Fill buffer
            buffer.coords = coords
            buffer.fill(mzs, intensities)
            
            yield buffer

# Similar updates for ImzMLReader
```

## 🏗️ Phase 6: CLI Integration

### 6.1 Update CLI Arguments (__main__.py)
```python
# Add to argument parser
interpolation_group = parser.add_argument_group('Interpolation options')

interpolation_group.add_argument(
    '--enable-interpolation',
    action='store_true',
    help='Enable intelligent mass axis interpolation (reduces file size 50-90%%)'
)

interpolation_group.add_argument(
    '--interpolation-method',
    choices=['pchip', 'linear', 'adaptive'],
    default='pchip',
    help='Interpolation method (default: pchip - monotonic, no overshooting)'
)

# SCiLS-like bin specification options
interpolation_group.add_argument(
    '--interpolation-bins',
    type=int,
    help='Number of bins for interpolated mass axis (option 1, like SCiLS)'
)

interpolation_group.add_argument(
    '--interpolation-width',
    type=float,
    help='Bin width in Da for interpolation (option 2, like SCiLS)'
)

interpolation_group.add_argument(
    '--interpolation-width-mz',
    type=float,
    default=400.0,
    help='Reference m/z for bin width specification (default: 400)'
)

interpolation_group.add_argument(
    '--physics-model',
    choices=['auto', 'tof', 'orbitrap', 'fticr'],
    default='auto',
    help='Instrument physics model for optimal spacing (default: auto-detect)'
)

interpolation_group.add_argument(
    '--interpolation-quality-report',
    action='store_true',
    help='Generate quality report after interpolation'
)

interpolation_group.add_argument(
    '--adaptive-workers',
    action='store_true',
    default=True,
    help='Adaptively scale worker count based on performance'
)

interpolation_group.add_argument(
    '--max-interpolation-workers',
    type=int,
    default=80,
    help='Maximum number of interpolation workers (default: 80)'
)

# In the main processing logic:
if args.interpolation_bins and args.interpolation_width:
    parser.error("Cannot specify both --interpolation-bins and --interpolation-width")
```

### 6.2 Update Configuration (config.py)
```python
# Add to existing config.py
@dataclass
class InterpolationConfig:
    """Interpolation-specific configuration"""
    enabled: bool = False
    method: str = "pchip"
    
    # SCiLS-like bin specification
    interpolation_bins: Optional[int] = None
    interpolation_width: Optional[float] = None
    interpolation_width_mz: float = 400.0
    
    preserve_peaks: bool = True
    use_physics_model: bool = True
    physics_model: str = "auto"
    validate_quality: bool = True
    max_memory_gb: float = 8.0
    adaptive_workers: bool = True
    max_workers: int = 80
    min_workers: int = 4
    
    # Quality thresholds
    tic_deviation_threshold: float = 0.01
    peak_preservation_threshold: float = 0.95
    
    # Performance settings
    use_simd: bool = True
    cache_coefficients: bool = True
    buffer_pool_size: int = 1000

# Update main Config class
@dataclass
class Config:
    # Existing fields...
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
```

## 📋 Testing Implementation

### 7.1 Unit Tests Structure
```bash
tests/unit/interpolators/
├── test_physics_models.py
├── test_interpolation_strategies.py
├── test_streaming_engine.py
├── test_buffer_pool.py
└── test_quality_monitor.py
```

### 7.2 Example Unit Test (test_physics_models.py)
```python
import pytest
import numpy as np
from msiconvert.interpolators.physics_models import TOFPhysics, OrbitrapPhysics

class TestTOFPhysics:
    def test_resolution_calculation(self):
        """Test TOF resolution calculation"""
        physics = TOFPhysics(resolution_at_400=10000)
        
        # Test at reference point
        assert physics.calculate_resolution_at_mz(400) == 10000
        
        # Test scaling with sqrt(m/z)
        assert physics.calculate_resolution_at_mz(1600) == pytest.approx(20000)
        assert physics.calculate_resolution_at_mz(100) == pytest.approx(5000)
        
    def test_optimal_mass_axis(self):
        """Test optimal mass axis generation"""
        physics = TOFPhysics()
        axis = physics.create_optimal_mass_axis(100, 1000, 1000)
        
        # Check bounds
        assert axis[0] == pytest.approx(100)
        assert axis[-1] <= 1000
        
        # Check non-linear spacing
        spacings = np.diff(axis)
        assert spacings[-1] > spacings[0]  # Spacing increases with m/z
        
    def test_minimum_spacing(self):
        """Test minimum spacing calculation"""
        physics = TOFPhysics(resolution_at_400=10000)
        
        # At m/z 400, R=10000, spacing = 400/(2*10000) = 0.02
        spacing = physics.calculate_minimum_spacing(400)
        assert spacing == pytest.approx(0.02)
```

### 7.3 Integration Test Example
```python
import pytest
from pathlib import Path
import numpy as np
from msiconvert.readers.bruker import BrukerReader
from msiconvert.interpolators.streaming_engine import StreamingInterpolationEngine

@pytest.mark.integration
class TestInterpolationPipeline:
    def test_end_to_end_interpolation(self, sample_bruker_data):
        """Test complete interpolation pipeline"""
        # Setup reader
        reader = BrukerReader(sample_bruker_data)
        
        # Setup interpolation config
        config = InterpolationConfig(
            method="pchip",
            n_workers=4,
            buffer_size=100,
            validate_quality=True
        )
        
        # Create engine
        engine = StreamingInterpolationEngine(config)
        
        # Process small test dataset
        results = []
        def capture_results(coords, mass_axis, intensities):
            results.append((coords, intensities))
            
        engine.process_dataset(reader, capture_results)
        
        # Validate results
        assert len(results) > 0
        
        # Check quality metrics
        summary = engine.quality_monitor.get_summary()
        assert summary['tic_ratios']['mean'] == pytest.approx(1.0, rel=0.01)
        assert summary['peak_preservation']['mean'] > 0.95
```

## 🚨 Critical Implementation Notes

### Memory Management
1. **NEVER** load full data into memory - use bounds only
2. **ALWAYS** use buffer pools for spectrum data
3. **MONITOR** memory usage continuously
4. **SCALE** workers based on available memory

### Performance Optimization
1. **PCHIP is the bottleneck** - optimize aggressively
2. **Cache coefficients** when possible
3. **Use SIMD** via numba for batch operations

### Error Handling
```python
# Always wrap interpolation in try-except
try:
    interpolated = interpolator.interpolate_spectrum(mz_old, intensity_old, mz_new)
except Exception as e:
    logging.error(f"Interpolation failed for spectrum {coords}: {e}")
    # Return original or zeros
    interpolated = np.zeros_like(mz_new)
```

### Quality Validation
1. **Check TIC preservation** - should be within 1%
2. **Monitor peak loss** - should preserve >95%
3. **Validate no negative values** - PCHIP guarantees this
4. **Log warnings** for quality issues

## 📊 Expected Performance

With this implementation:
- **Memory usage**: <16GB for 1TB datasets
- **Processing speed**: 200-400 spectra/sec (with 80 workers)
- **File size reduction**: 50-90%
- **Quality preservation**: >95% of peaks retained