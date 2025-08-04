# Dask-Based Streaming MSI Converter Architecture Plan

## Executive Summary

This document outlines a comprehensive plan to implement true out-of-core, memory-efficient MSI data processing using Dask, with support for streaming interpolation and incremental Zarr writing. The goal is to handle datasets of any size (100+ GB) with constant memory usage.

## Current State Analysis

### What We Have (Partially Working)
- ✅ Memory estimation logic (`_estimate_memory_requirements()`)
- ✅ Chunking decision logic (`_should_use_chunked_processing()`)
- ✅ Basic chunked data structures in converters
- ✅ Integration with existing SpatialData converter architecture

### Critical Issues with Current Approach
- ❌ **Still accumulates all data in memory** (lists just delay the problem)
- ❌ **No interpolation** (removed for simplicity)
- ❌ **No true Zarr streaming** (creates full sparse matrix at end)
- ❌ **Inefficient for large datasets** (O(n) memory usage)

### Recommendation: **Selective Revert + Dask Integration**
- **Keep**: Memory estimation, chunking detection, converter structure
- **Replace**: Data accumulation logic with Dask-based streaming
- **Add**: Interpolation back with Dask lazy evaluation
- **Add**: True incremental Zarr writing

## Architecture Overview

### Core Principles
1. **Constant Memory Usage**: Memory usage independent of dataset size
2. **Lazy Evaluation**: Build computation graph, execute optimally
3. **Streaming I/O**: Read chunks → Process → Write → Discard
4. **Backwards Compatibility**: Small datasets use fast path, large datasets use streaming
5. **Maintainable**: Clean abstractions, testable components

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MSI Reader    │───▶│  Dask Processor │───▶│  Zarr Writer    │
│  (Chunked)      │    │   (Streaming)   │    │ (Incremental)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Spectrum Chunks │    │ Interpolation   │    │ SpatialData     │
│   (Memory)      │    │   (Lazy)        │    │  Structure      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Dask Integration Strategy

### Phase 1: Dask Task Graph Setup
```python
class DaskStreamingConverter(BaseSpatialDataConverter):
    def __init__(self, *args, use_dask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dask = use_dask or self._should_use_dask_processing()
        
    def _should_use_dask_processing(self) -> bool:
        """Decide whether to use Dask based on dataset size"""
        return self._should_use_chunked_processing()  # Reuse existing logic
```

### Phase 2: Simplified Dask Streaming Pipeline
```python
def _create_dask_pipeline(self):
    """Create Dask computation graph for streaming interpolation and writing"""
    
    # Prerequisites: Mass axis already generated (fast, non-Dask)
    assert self.common_mass_axis is not None
    
    # Step 1: Create pixel chunks (lazy)
    pixel_chunks = self._create_pixel_chunk_tasks()
    
    # Step 2: Process chunks with interpolation to known mass axis (lazy)
    processed_chunks = pixel_chunks.map(self._process_chunk_with_interpolation)
    
    # Step 3: Write to Zarr incrementally (lazy)
    written_chunks = processed_chunks.map(self._write_chunk_to_zarr)
    
    # Step 4: Finalize SpatialData structure (lazy)
    final_result = written_chunks.reduce(self._finalize_spatialdata)
    
    return final_result
```

### Phase 3: Memory-Bounded Execution
```python
def convert(self):
    if self.use_dask:
        return self._convert_with_dask()
    else:
        return super().convert()  # Use existing fast path
        
def _convert_with_dask(self):
    with dask.config.set({'array.chunk-size': '128MB'}):
        pipeline = self._create_dask_pipeline()
        return pipeline.compute()
```

## Interpolation Strategy

### Mass Axis Generation (Pre-Dask, Fast)
- **Insight**: Most datasets have metadata, those without are typically small
- **Solution**: Metadata-first approach with simple fallback scan

### Efficient Mass Axis Generation with Non-Linear Binning
```python
def _generate_common_mass_axis(self):
    """Generate mass axis efficiently with appropriate binning strategy"""
    
    # Get essential metadata (should contain mass range)
    metadata = self.reader.get_essential_metadata()
    
    # Extract mass range from essential metadata
    if hasattr(metadata, 'mass_range') and metadata.mass_range:
        min_mz, max_mz = metadata.mass_range
        logging.info(f"Mass range from metadata: {min_mz:.2f} - {max_mz:.2f}")
    else:
        # Fallback: Quick scan for datasets without mass range in metadata
        logging.warning("No mass range in metadata, scanning dataset...")
        min_mz, max_mz = self._scan_for_mass_range()
    
    # Use manual analyzer type (TOF default) - will be automatic later
    analyzer_type = getattr(self, 'analyzer_type', 'tof')
    logging.info(f"Using analyzer type: {analyzer_type}")
    
    return self._create_mass_axis(min_mz, max_mz, analyzer_type)

def _create_mass_axis(self, min_mz: float, max_mz: float, analyzer_type: str = "unknown") -> np.ndarray:
    """Create mass axis with analyzer-specific binning strategy"""
    
    if analyzer_type in ["tof", "time-of-flight"]:
        return self._create_tof_mass_axis(min_mz, max_mz)
    elif analyzer_type in ["orbitrap", "ft-icr", "fticr"]:
        return self._create_fourier_mass_axis(min_mz, max_mz)
    elif analyzer_type in ["quadrupole", "triple-quad", "qtof"]:
        return self._create_quadrupole_mass_axis(min_mz, max_mz)
    else:
        # Default: linear binning with configurable resolution
        logging.warning(f"Unknown analyzer type '{analyzer_type}', using default linear binning")
        return self._create_linear_mass_axis(min_mz, max_mz)

def _create_tof_mass_axis(self, min_mz: float, max_mz: float) -> np.ndarray:
    """Create mass axis for TOF analyzer with variable bin width"""
    # Simple approach: bin width at reference mass, scales with √m/z
    bin_width_at_ref = getattr(self, 'tof_bin_width_da', 0.01)  # 0.01 Da at reference
    reference_mz = getattr(self, 'tof_reference_mz', 500.0)     # Reference m/z
    
    # Alternative: specify number of bins
    if hasattr(self, 'tof_n_bins'):
        return self._create_axis_from_n_bins(min_mz, max_mz, self.tof_n_bins, 'tof')
    
    # Generate variable-width bins (bin width ∝ √m/z)
    bin_edges = []
    current_mz = min_mz
    
    while current_mz <= max_mz:
        bin_edges.append(current_mz)
        # Scale bin width: width ∝ √(current_mz / reference_mz)
        scaling_factor = np.sqrt(current_mz / reference_mz)
        bin_width = bin_width_at_ref * scaling_factor
        current_mz += bin_width
    
    bin_edges.append(max_mz)  # Ensure we cover the full range
    
    # Convert edges to centroids
    bin_edges = np.array(bin_edges)
    centroids = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    logging.info(f"TOF mass axis: {len(centroids)} bins, {bin_width_at_ref} Da at {reference_mz} m/z")
    return centroids

def _create_fourier_mass_axis(self, min_mz: float, max_mz: float) -> np.ndarray:
    """Create mass axis for Orbitrap/FT-ICR with constant bin width"""
    # Simple approach: constant bin width
    bin_width = getattr(self, 'fourier_bin_width_da', 0.001)  # 1 mDa default
    
    # Alternative: specify number of bins
    if hasattr(self, 'fourier_n_bins'):
        return self._create_axis_from_n_bins(min_mz, max_mz, self.fourier_n_bins, 'linear')
    
    # Generate linear bins with constant width
    n_bins = int((max_mz - min_mz) / bin_width) + 1
    bin_edges = np.linspace(min_mz, max_mz, n_bins + 1)
    
    # Convert edges to centroids
    centroids = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    logging.info(f"Fourier mass axis: {len(centroids)} bins, {bin_width} Da constant width")
    return centroids

def _create_quadrupole_mass_axis(self, min_mz: float, max_mz: float) -> np.ndarray:
    """Create mass axis for quadrupole with constant step size"""
    # Simple approach: constant step size
    step_size = getattr(self, 'quadrupole_step_size_da', 0.1)  # 0.1 Da default
    
    # Alternative: specify number of bins
    if hasattr(self, 'quadrupole_n_bins'):
        return self._create_axis_from_n_bins(min_mz, max_mz, self.quadrupole_n_bins, 'linear')
    
    # Generate linear bins
    n_bins = int((max_mz - min_mz) / step_size) + 1
    bin_edges = np.linspace(min_mz, max_mz, n_bins + 1)
    
    # Convert edges to centroids
    centroids = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    logging.info(f"Quadrupole mass axis: {len(centroids)} bins, {step_size} Da steps")
    return centroids

def _create_linear_mass_axis(self, min_mz: float, max_mz: float) -> np.ndarray:
    """Create linear mass axis (fallback/default)"""
    n_bins = getattr(self, 'default_n_bins', 2000)
    return self._create_axis_from_n_bins(min_mz, max_mz, n_bins, 'linear')

def _create_axis_from_n_bins(self, min_mz: float, max_mz: float, n_bins: int, spacing_type: str) -> np.ndarray:
    """Create mass axis from specified number of bins"""
    if spacing_type == 'linear':
        bin_edges = np.linspace(min_mz, max_mz, n_bins + 1)
        centroids = (bin_edges[:-1] + bin_edges[1:]) / 2
        logging.info(f"Linear mass axis: {len(centroids)} bins")
        return centroids
    elif spacing_type == 'tof':
        # Variable spacing for TOF with specified number of bins
        # Use logarithmic-like distribution
        # This is more complex - for now, fall back to linear
        bin_edges = np.linspace(min_mz, max_mz, n_bins + 1)
        centroids = (bin_edges[:-1] + bin_edges[1:]) / 2
        logging.info(f"TOF mass axis: {len(centroids)} bins (linear approximation)")
        return centroids
    else:
        raise ValueError(f"Unknown spacing type: {spacing_type}")

def _detect_mass_analyzer_type(self, metadata=None) -> str:
    """Detect mass analyzer type from metadata or heuristics"""
    # For now, this method is placeholder - analyzer type is set manually
    # Will be enhanced when metadata structure is updated
    
    # Future implementation will check:
    # - metadata.instrument_info.mass_analyzer
    # - ImzML cvParam values for analyzer type
    # - Bruker instrument configuration
    # - File format heuristics
    
    logging.debug("Analyzer detection not yet implemented - using manual setting")
    return "tof"  # Default fallback

def _scan_for_mass_range(self):
    """Simple full scan for datasets without metadata (typically small)"""
    min_mz, max_mz = float('inf'), float('-inf')
    
    # Direct iteration - fast enough for small datasets
    for coords, mzs, intensities in self.reader.iter_spectra():
        if len(mzs) > 0:
            min_mz = min(min_mz, np.min(mzs))
            max_mz = max(max_mz, np.max(mzs))
    
    logging.info(f"Mass range from scan: {min_mz:.2f} - {max_mz:.2f}")
    return min_mz, max_mz
```

### Mass Axis Strategy Benefits
- ✅ **Metadata-first**: Instant for most datasets (ImzML with cvParams, Bruker with MaldiFrameInfo)
- ✅ **Analyzer-specific binning**: Optimal resolution for each mass analyzer type
- ✅ **Physics-aware**: Accounts for resolution characteristics (TOF: ∝√m/z, Orbitrap: ∝m/z, etc.)
- ✅ **Simple fallback**: Direct scan for small datasets without overhead
- ✅ **No Dask complexity**: Mass axis generation stays simple and fast
- ✅ **Foundation for streaming**: Mass axis ready before Dask processing begins

### Simplified Non-Linear Binning Strategies

#### 1. **TOF (Time-of-Flight) Analyzers**
- **Approach**: Bin width at reference mass, scales with √m/z
- **Parameters**: 
  - `tof_bin_width_da = 0.01` (bin width at reference mass)
  - `tof_reference_mz = 500.0` (reference m/z)
  - OR `tof_n_bins = 2000` (total number of bins)
- **Implementation**: Variable bin width ∝ √(m/z), return centroids
- **Example**: 0.01 Da at 500 m/z → 0.014 Da at 1000 m/z

#### 2. **Orbitrap/FT-ICR Analyzers** 
- **Approach**: Constant bin width (high resolution)
- **Parameters**: 
  - `fourier_bin_width_da = 0.001` (constant 1 mDa steps)
  - OR `fourier_n_bins = 5000` (total number of bins)
- **Implementation**: Linear spacing with fine resolution, return centroids
- **Example**: 0.001 Da steps throughout entire mass range

#### 3. **Quadrupole Analyzers**
- **Approach**: Constant step size (lower resolution)
- **Parameters**: 
  - `quadrupole_step_size_da = 0.1` (0.1 Da steps)
  - OR `quadrupole_n_bins = 1000` (total number of bins)
- **Implementation**: Linear spacing with appropriate resolution, return centroids
- **Example**: 0.1 Da steps (good for unit resolution instruments)

#### 4. **Unknown/Mixed Analyzers**
- **Approach**: Linear binning with configurable number of bins
- **Parameters**: 
  - `default_n_bins = 2000` (total number of bins)
- **Implementation**: `np.linspace()` for bin edges, return centroids
- **Example**: 2000 equally-spaced bins across mass range

### **Key Implementation Details:**

#### **Centroids vs Edges**
- **Bin edges**: `[100.0, 100.05, 100.10, ...]` (boundaries)
- **Bin centroids**: `[100.025, 100.075, 100.125, ...]` (representative m/z)
- **Choice**: **Use centroids** - more meaningful for interpolation and analysis

#### **Configuration Options**
```python
# Manual analyzer type specification (current implementation)
# Option 1: Specify bin width/step size
converter = SpatialDataConverter(
    reader=reader,
    analyzer_type='tof',              # Manual: 'tof', 'orbitrap', 'quadrupole'
    tof_bin_width_da=0.005,           # Finer binning for TOF
    tof_reference_mz=500.0            # Reference mass for scaling
)

converter = SpatialDataConverter(
    reader=reader,
    analyzer_type='orbitrap',         # Manual analyzer type
    fourier_bin_width_da=0.0005       # Ultra-high res for Orbitrap
)

converter = SpatialDataConverter(
    reader=reader,
    analyzer_type='quadrupole',       # Manual analyzer type
    quadrupole_step_size_da=0.05      # Finer steps for quad
)

# Option 2: Specify number of bins
converter = SpatialDataConverter(
    reader=reader,
    analyzer_type='tof',              # Manual analyzer type
    tof_n_bins=3000                   # More bins for TOF
)

# Default: TOF analyzer (most common in MSI)
converter = SpatialDataConverter(
    reader=reader
    # analyzer_type defaults to 'tof'
    # Uses default TOF binning parameters
)
```

### Dask Streaming Interpolation (Second Pass)
```python
def _create_dask_interpolation_pipeline(self):
    """Create Dask pipeline for streaming interpolation with known mass axis"""
    
    # Mass axis already available from fast generation above
    assert self.common_mass_axis is not None
    
    # Create pixel chunks (lazy)
    pixel_chunks = self._create_pixel_chunk_tasks()
    
    # Process chunks with interpolation (lazy)
    processed_chunks = pixel_chunks.map(self._process_chunk_with_interpolation)
    
    # Write to Zarr incrementally (lazy)
    written_chunks = processed_chunks.map(self._write_chunk_to_zarr)
    
    # Execute pipeline with memory management
    return written_chunks.compute()

@dask.delayed
def _process_chunk_with_interpolation(self, pixel_chunk):
    """Process chunk of pixels with interpolation to known mass axis"""
    interpolated_data = []
    
    for coords, mzs, intensities in pixel_chunk:
        if len(mzs) > 0:
            # Interpolate to pre-generated common mass axis
            interpolated = np.interp(
                self.common_mass_axis, mzs, intensities, 
                left=0.0, right=0.0
            )
            # Convert to sparse representation immediately (memory efficient)
            sparse_indices, sparse_values = self._to_sparse(interpolated)
            interpolated_data.append((coords, sparse_indices, sparse_values))
    
    return interpolated_data  # Small sparse representation, not full spectra
```

## Memory Management

### Zarr Incremental Writing Strategy
```python
class IncrementalZarrWriter:
    def __init__(self, output_path, total_pixels, n_masses):
        self.zarr_root = zarr.open(output_path, mode='w')
        self._setup_zarr_arrays(total_pixels, n_masses)
        self.write_lock = threading.Lock()
        
    def _setup_zarr_arrays(self, total_pixels, n_masses):
        """Create resizable Zarr arrays for sparse matrix"""
        # Sparse matrix in COO format for incremental building
        self.zarr_root.create_dataset(
            'sparse_data', shape=(0,), maxshape=(None,), 
            dtype='f4', chunks=10000
        )
        self.zarr_root.create_dataset(
            'sparse_rows', shape=(0,), maxshape=(None,), 
            dtype='i4', chunks=10000
        )
        self.zarr_root.create_dataset(
            'sparse_cols', shape=(0,), maxshape=(None,), 
            dtype='i4', chunks=10000
        )
        
    @dask.delayed
    def write_chunk(self, chunk_data):
        """Thread-safe chunk writing"""
        with self.write_lock:
            current_size = len(self.zarr_root['sparse_data'])
            chunk_size = len(chunk_data['values'])
            
            # Resize arrays
            new_size = current_size + chunk_size
            self.zarr_root['sparse_data'].resize(new_size)
            self.zarr_root['sparse_rows'].resize(new_size)
            self.zarr_root['sparse_cols'].resize(new_size)
            
            # Write chunk data
            self.zarr_root['sparse_data'][current_size:new_size] = chunk_data['values']
            self.zarr_root['sparse_rows'][current_size:new_size] = chunk_data['rows']
            self.zarr_root['sparse_cols'][current_size:new_size] = chunk_data['cols']
            
        return chunk_size  # Return metadata only
```

### Memory Usage Guarantees
- **Target**: < 2GB memory usage regardless of dataset size
- **Mechanism**: Dask automatic memory management + explicit chunk cleanup
- **Monitoring**: Memory usage tracking and warnings

## Implementation Phases

### Phase 0: Efficient Mass Axis Generation with Manual Analyzer Type (Week 1) **[RECOMMENDED START]**
**Why start here**: Self-contained, testable, maintains accuracy, builds foundation

- [ ] Implement metadata-first mass axis generation using essential metadata
- [ ] Add manual analyzer type specification (default: 'tof')
- [ ] Implement analyzer-specific binning strategies:
  - [ ] TOF: Variable bin width ∝ √m/z (bin width at reference mass)
  - [ ] Orbitrap/FT-ICR: Constant bin width (high resolution)
  - [ ] Quadrupole: Constant step size (moderate resolution)
  - [ ] Unknown: Configurable linear fallback
- [ ] Support both bin width and n_bins configuration approaches
- [ ] Return bin centroids as representative m/z values
- [ ] Add fallback mass range scan for datasets without metadata  
- [ ] Placeholder for future automatic analyzer detection
- [ ] Validate accuracy and appropriateness for each analyzer type
- [ ] Benchmark performance vs current implementation
- [ ] Integration test: Drop-in replacement with enhanced physics-aware binning

**Success Criteria**: 
- ✅ Manual analyzer type specification with TOF as default
- ✅ Analyzer-appropriate mass axis for each instrument type
- ✅ Works with existing converters unchanged (enhanced capability)
- ✅ Instant mass axis generation using essential metadata
- ✅ Bin centroids returned for meaningful interpolation
- ✅ Both bin width and bin count configuration options
- ✅ Fast fallback scan for datasets without mass range in metadata
- ✅ Configurable binning parameters for fine-tuning

### Phase 1: Dask Foundation (Week 2)
- [ ] Implement `DaskStreamingConverter` base class
- [ ] Add Dask dependency and configuration
- [ ] Implement memory-based vs Dask routing logic
- [ ] Create pixel chunk generation with Dask delayed
- [ ] Integrate Phase 0 mass axis generation
- [ ] Basic end-to-end test with small dataset

### Phase 2: Dask Streaming Integration (Week 2)  
- [ ] Implement Dask streaming interpolation with known mass axis
- [ ] Add chunked pixel processing with Dask delayed
- [ ] Optimize interpolation performance for chunks
- [ ] Test interpolation accuracy vs original implementation
- [ ] Memory usage validation for streaming pipeline

### Phase 3: Zarr Streaming (Week 3)
- [ ] Implement `IncrementalZarrWriter` class
- [ ] Thread-safe incremental sparse matrix writing
- [ ] Integrate with SpatialData structure creation
- [ ] Handle coordinate and metadata writing
- [ ] Test with medium datasets (1-10GB)

### Phase 4: Optimization & Integration (Week 4)
- [ ] Performance tuning (chunk sizes, parallelism)
- [ ] Memory pressure monitoring and adaptation
- [ ] Integration with existing converter factory
- [ ] Comprehensive error handling and recovery
- [ ] Large dataset testing (50-100GB)

### Phase 5: Production Ready (Week 5)
- [ ] Documentation and examples
- [ ] Benchmark against original converters
- [ ] Memory usage validation across dataset sizes
- [ ] CI/CD integration
- [ ] User-facing configuration options

## Risk Mitigation

### Technical Risks
1. **Dask Overhead**: May be slower for small datasets
   - *Mitigation*: Dual-path approach (Dask vs standard)
   
2. **Zarr Write Performance**: Incremental writing may be slow
   - *Mitigation*: Batch writes, optimize chunk sizes
   
3. **Memory Leaks**: Dask tasks may not release memory properly
   - *Mitigation*: Explicit cleanup, memory monitoring
   
4. **SpatialData Compatibility**: Zarr-backed arrays may not work seamlessly
   - *Mitigation*: Extensive testing, fallback to in-memory for final assembly

### Complexity Risks
1. **Too Many Abstractions**: Over-engineering the solution
   - *Mitigation*: Start simple, add complexity incrementally
   
2. **Debugging Difficulty**: Dask task graphs are hard to debug
   - *Mitigation*: Comprehensive logging, task visualization tools

## Testing Strategy

### Unit Tests
- Mock readers for controlled data generation
- Memory usage validation tests
- Interpolation accuracy tests
- Zarr writing correctness tests

### Integration Tests
- Small datasets (< 1GB): Verify correctness vs original
- Medium datasets (1-10GB): Memory usage validation
- Large datasets (10-100GB): End-to-end conversion tests
- Stress tests: Memory pressure, error recovery

### Performance Benchmarks
- Conversion speed vs dataset size
- Memory usage vs dataset size
- Comparison with original converters
- Scalability analysis

## Configuration & User Interface

### Configuration Options
```python
# Automatic mode (recommended)
converter = SpatialDataConverter(
    reader=reader,
    output_path=output_path,
    auto_streaming=True,  # Use Dask for large datasets
    target_memory_gb=8.0
)

# Manual control
converter = SpatialDataConverter(
    reader=reader,
    output_path=output_path,
    use_dask=True,
    dask_chunk_size=1000,
    dask_memory_limit='2GB'
)
```

### CLI Integration
```bash
# Automatic streaming
msiconvert data.imzml output.zarr --auto-streaming

# Manual streaming control
msiconvert data.imzml output.zarr --use-dask --memory-limit 4GB --chunk-size 500
```

## Success Metrics

### Functional Requirements
- [ ] Convert 100GB+ datasets with <4GB memory usage
- [ ] Maintain interpolation accuracy (< 0.1% difference)
- [ ] Backwards compatibility with existing SpatialData workflows
- [ ] Performance: < 2x slower than original for small datasets

### Non-Functional Requirements
- [ ] Maintainable codebase with clear abstractions
- [ ] Comprehensive error handling and recovery
- [ ] Good user experience (progress bars, memory monitoring)
- [ ] Extensive documentation and examples

## Conclusion

This architecture provides a path to true out-of-core MSI processing while maintaining compatibility with existing workflows. The phased approach allows for incremental development and validation, reducing risk while building towards a production-ready streaming converter.

## **RECOMMENDED STARTING POINT: Phase 0 - Efficient Mass Axis Generation**

### Why This Is The Best Place To Start:

1. **Self-Contained**: Can be implemented and tested independently
2. **Maintains Functionality**: Drop-in replacement for existing mass axis generation
3. **Immediate Value**: Instant mass axis for datasets with metadata
4. **Foundation Building**: Required before Dask streaming can begin
5. **Low Risk**: If it fails, existing code is unchanged
6. **Testable**: Easy to validate accuracy against current implementation

### Simplified Implementation Strategy:
```python
# Add to BaseSpatialDataConverter
def _generate_common_mass_axis(self):
    # Get mass range from essential metadata (instant)
    metadata = self.reader.get_essential_metadata()
    if hasattr(metadata, 'mass_range') and metadata.mass_range:
        min_mz, max_mz = metadata.mass_range
    else:
        # Fallback: quick scan for datasets without metadata
        min_mz, max_mz = self._scan_for_mass_range()
    
    # Use manual analyzer type (TOF default) - automatic detection comes later
    analyzer_type = getattr(self, 'analyzer_type', 'tof')
    return self._create_mass_axis(min_mz, max_mz, analyzer_type)
```

### Success Metrics for Phase 0:
- [ ] Produces analyzer-appropriate mass axis for each instrument type
- [ ] Instant generation using mass range from essential metadata
- [ ] Proper TOF binning (variable width ∝ √m/z) vs Orbitrap binning (constant width)
- [ ] Returns bin centroids as representative m/z values
- [ ] Supports both bin width and n_bins configuration approaches
- [ ] Fast fallback scan for datasets without mass range in metadata
- [ ] Works with all existing readers (ImzML, Bruker)
- [ ] Configurable binning parameters (width or count)
- [ ] Comprehensive test coverage

### Benefits of This Enhanced Approach:
- ✅ **Intuitive parameters** - bin width at reference mass or total number of bins
- ✅ **Physics-aware binning** optimized for each analyzer type
- ✅ **Centroids-based** - more meaningful m/z values for interpolation
- ✅ **No Dask complexity** for mass axis generation
- ✅ **Instant results** using essential metadata mass range
- ✅ **Flexible configuration** - width-based or count-based approaches
- ✅ **Proper scaling** with mass for TOF instruments
- ✅ **High-resolution constant binning** for Fourier-transform instruments
- ✅ **Simple fallback** for unknown analyzer types
- ✅ **Clean separation** between mass axis generation and streaming processing
- ✅ **Ready for Dask** streaming in subsequent phases

**Next Steps**: Implement Phase 0, validate accuracy and performance, then proceed to Phase 1 Dask streaming foundation.