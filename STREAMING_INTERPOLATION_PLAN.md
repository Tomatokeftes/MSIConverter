# Streaming Interpolation Architecture Plan

## Overview
Implementation of memory-efficient, parallel interpolation for massive MSI datasets (millions of pixels, 100+ GB) using non-linear mass axis binning and streaming processing.

## Core Architecture

### 1. Mass Axis Generation
```python
def generate_common_mass_axis(min_mz, max_mz, bin_strategy):
    """Generate non-linear mass axis with higher resolution at lower masses"""
    # Strategy A: Desired bin width at specific mass
    # Strategy B: Fixed number of bins distributed non-linearly
    # Uses equivalent mass calculation formula for TOF/Orbitrap/FTICR
```

**Priority Order for min/max m/z detection:**
1. **Metadata extraction** (fast, when available)
2. **Full iteration** (fallback, expensive)

### 2. Streaming Interpolation Pipeline

```
Input Dataset → Mass Axis → Chunk Processing → Interpolation → Temporary Storage → SpatialData Assembly
```

#### Core Components:

**A. Chunk-Based Processing**
- Process datasets in configurable chunks (e.g., 1000-10000 pixels)
- Memory-aware chunk sizing based on available RAM
- Progressive disk writing to avoid memory exhaustion

**B. Parallel Interpolation Engine**
```python
class StreamingInterpolator:
    def __init__(self, common_mass_axis, n_workers=None, chunk_size=None):
        self.mass_axis = common_mass_axis
        self.workers = n_workers or cpu_count()
        self.chunk_size = chunk_size or self._estimate_chunk_size()

    def interpolate_chunk(self, pixel_chunk):
        """Interpolate a chunk of pixels in parallel"""
        # Each worker gets a subset of pixels
        # Returns interpolated spectra for the chunk
```

**C. Memory Management Strategies**

1. **Temporary Storage Approach** (Primary)
   - Write interpolated chunks to HDF5/Zarr temporary files
   - Use memory mapping for efficient access
   - Clean up temporary files after SpatialData assembly

2. **Incremental SpatialData Writing** (Preferred if possible)
   - Investigate Zarr's append capabilities
   - Write directly to final SpatialData structure
   - Avoid intermediate storage completely

### 3. Implementation Phases

#### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Mass axis generation algorithms
- [ ] Metadata extraction for min/max m/z
- [ ] Basic interpolation framework
- [ ] Memory estimation utilities

#### Phase 2: Parallel Processing (Week 2-3)
- [ ] Multiprocessing interpolation engine
- [ ] Chunk-based pixel iteration
- [ ] Progress tracking and monitoring
- [ ] Error handling for corrupted pixels

#### Phase 3: Memory Optimization (Week 3-4)
- [ ] Temporary storage implementation (HDF5/Zarr)
- [ ] Memory profiling and optimization
- [ ] Adaptive chunk sizing
- [ ] Memory pressure monitoring

#### Phase 4: SpatialData Integration (Week 4-5)
- [ ] Investigate Zarr incremental writing
- [ ] SpatialData assembly from chunks
- [ ] Metadata preservation during streaming
- [ ] Final optimization and testing

#### Phase 5: Benchmarking & Validation (Week 5-6)
- [ ] Performance testing with large datasets
- [ ] Memory usage validation
- [ ] Accuracy verification against non-streaming approach
- [ ] Documentation and examples

## Technical Specifications

### Interpolation Methods
- **Linear interpolation** (default, fastest)
- **Cubic spline** (higher accuracy, slower)
- **Configurable method selection** based on dataset size/accuracy needs

### Memory Constraints
- **Target memory usage**: < 8GB for any dataset size
- **Chunk size adaptation**: Based on available RAM and pixel complexity
- **Garbage collection**: Explicit cleanup between chunks

### Performance Targets
- **Throughput**: > 1000 pixels/second on modern hardware
- **Memory efficiency**: Constant memory usage regardless of dataset size
- **Scalability**: Linear performance scaling with CPU cores

## Key Implementation Details

### 1. Mass Axis Calculation
```python
def equivalent_mass_binning(min_mz, max_mz, resolution_at_mz, target_mz):
    """
    Create non-linear mass axis based on equivalent mass resolution
    Higher resolution (smaller bins) at lower masses
    """
    # Implementation depends on instrument type (TOF/Orbitrap/FTICR)
```

### 2. Adaptive Chunking
```python
def estimate_chunk_size(available_memory, spectrum_length, pixel_count):
    """Estimate optimal chunk size based on system resources"""
    # Consider interpolation memory overhead (~3x spectrum size)
    # Account for multiprocessing memory duplication
    # Leave buffer for system operations
```

### 3. Zarr Incremental Writing Investigation
```python
# Key questions to research:
# - Can we append to Zarr arrays incrementally?
# - How does SpatialData handle partial array initialization?
# - What's the performance impact of incremental vs batch writing?
```

## Risk Mitigation

### Memory Overflow
- **Solution**: Aggressive chunking + temporary storage fallback
- **Monitoring**: Real-time memory usage tracking

### Interpolation Accuracy
- **Solution**: Configurable interpolation methods
- **Validation**: Comparison tests against reference implementations

### Performance Bottlenecks
- **Solution**: Profiling-guided optimization
- **Fallbacks**: Adaptive algorithm selection based on dataset characteristics

### Data Corruption
- **Solution**: Robust error handling per pixel
- **Recovery**: Skip corrupted pixels with detailed logging

## Success Metrics
- [ ] Process 1M+ pixel datasets without memory issues
- [ ] Complete 100GB dataset processing in <2 hours

## Integration Points

### Existing Codebase
- **Registry system**: Register new streaming converters
- **Reader interface**: Extend for chunk-based iteration
- **Progress tracking**: Integration with existing tqdm usage

### User Interface
- **CLI flags**: `--streaming`, `--chunk-size`, `--interpolation-method`
- **Python API**: Backward compatible with automatic streaming detection
- **Configuration**: Memory limits and performance tuning options
