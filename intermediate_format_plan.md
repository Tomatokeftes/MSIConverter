# Intermediate Data Format Plan for MSI Interpolation

## Problem Statement

The Bruker TSF SDK inherently causes 100% CPU usage during spectrum reading, even with minimal operations. This CPU bottleneck conflicts with our goal of reserving computational resources for the actual interpolation mathematics. We need a two-stage pipeline to separate expensive TSF reading from efficient interpolation processing.

## Proposed Solution: Intermediate Zarr Format

### Architecture Overview

```
Stage 1: TSF → Intermediate Zarr    (One-time 100% CPU cost)
Stage 2: Zarr → Interpolated Grid   (CPU-efficient, dask-optimized)
```

## Core Design Decisions

### 1. Metadata Handling
**Decision**: Write metadata directly to SpatialData output, not to intermediate format.
**Rationale**: 
- Metadata is small and read once
- No benefit to storing in intermediate format
- Reduces intermediate format complexity
- Essential metadata already cached in memory during Stage 1

### 2. Variable Spectrum Length Challenge

**The Problem**: Each frame has different `NumPeaks` (observed range: 1,500-3,600+)
- Cannot use regular 2D arrays without significant padding waste
- Need storage format that handles variable-length efficiently

**Proposed Solution**: Ragged Array Approach with Length Indexing

```python
# Zarr structure:
dataset.zarr/
├── mz_values/           # 1D concatenated array of all m/z values
├── intensities/         # 1D concatenated array of all intensities  
├── spectrum_offsets/    # 1D array: start index for each spectrum
├── spectrum_lengths/    # 1D array: number of peaks per spectrum
└── coordinates/         # 2D array: [frame_id, x, y, z] per spectrum
```

**Access Pattern**:
```python
# To get spectrum N:
start_idx = spectrum_offsets[N]
length = spectrum_lengths[N]
mz_values = zarr_mz[start_idx:start_idx + length]
intensities = zarr_intensities[start_idx:start_idx + length]
coords = zarr_coords[N]
```

### 3. Chunking Strategy

**Primary Goal**: Optimize for interpolation access patterns (reading multiple spectra for gridding)

**Recommended Approach**: Spectrum-Based Chunking
- **Chunk size**: 1,000-2,000 spectra per chunk
- **Rationale**: 
  - Interpolation typically processes batches of spectra
  - Sequential frame reading is common
  - Reduces number of chunk reads
  - Good balance between memory usage and I/O efficiency

**Alternative Considered**: Spatial Chunking (32x32 pixel tiles)
- **Rejected because**: 
  - More complex indexing for sequential frame access
  - Irregular geometry makes chunk boundaries arbitrary
  - Interpolation often needs non-local spatial context

### 4. Data Types and Compression

```python
# Optimized data types:
mz_values: float32        # Sufficient precision for m/z
intensities: float32      # Sufficient precision for intensities
spectrum_offsets: uint64  # Support large datasets (>4B total peaks)
spectrum_lengths: uint16  # Max 65K peaks per spectrum (observed max ~3.6K)
coordinates: int16        # Pixel coordinates (max 32K pixels per dimension)
```

**Compression**: 
- **Zarr compressor**: `blosc` with `lz4` (fast decompression)
- **Compression level**: 5 (balance between speed and size)

## Implementation Plan

### Stage 1: TSF to Zarr Converter

```python
class TSFToZarrConverter:
    def __init__(self, tsf_path, zarr_path, chunk_size=1500):
        self.tsf_path = tsf_path
        self.zarr_path = zarr_path
        self.chunk_size = chunk_size  # spectra per chunk
    
    def convert(self):
        # 1. Initialize Bruker reader (accept 100% CPU cost)
        reader = BrukerReader(self.tsf_path)
        
        # 2. Get total spectrum count for array sizing
        total_spectra = reader.get_frame_count()
        
        # 3. Estimate total peaks (from NumPeaks cache)
        estimated_total_peaks = self._estimate_total_peaks(reader)
        
        # 4. Create Zarr arrays
        zarr_root = zarr.open(self.zarr_path, mode='w')
        
        # Pre-allocate with estimated sizes
        mz_array = zarr_root.create_dataset(
            'mz_values', 
            shape=(estimated_total_peaks,), 
            dtype=np.float32,
            chunks=(self.chunk_size * 2000,),  # ~2K peaks per spectrum avg
            compressor=zarr.Blosc(cname='lz4', clevel=5)
        )
        
        # ... similar for other arrays
        
        # 5. Stream conversion (handle 100% CPU during this phase)
        self._stream_convert(reader, zarr_arrays)
        
        # 6. Finalize and optimize
        self._optimize_chunks(zarr_arrays)
```

### Stage 2: Zarr to Interpolation Interface

```python
class ZarrSpectrumReader:
    def __init__(self, zarr_path):
        self.zarr_root = zarr.open(zarr_path, mode='r')
        
    def iter_spectra_batch(self, batch_size=1000):
        """Efficient batch iteration for dask processing."""
        for batch_start in range(0, len(self.spectrum_lengths), batch_size):
            batch_end = min(batch_start + batch_size, len(self.spectrum_lengths))
            
            # Read batch metadata
            batch_offsets = self.spectrum_offsets[batch_start:batch_end]
            batch_lengths = self.spectrum_lengths[batch_start:batch_end]
            batch_coords = self.coordinates[batch_start:batch_end]
            
            # Read spectral data for entire batch
            min_offset = batch_offsets[0]
            max_offset = batch_offsets[-1] + batch_lengths[-1]
            
            batch_mz = self.mz_values[min_offset:max_offset]
            batch_intensities = self.intensities[min_offset:max_offset]
            
            # Yield individual spectra from batch
            for i, (offset, length, coords) in enumerate(zip(batch_offsets, batch_lengths, batch_coords)):
                local_start = offset - min_offset
                yield (
                    coords,
                    batch_mz[local_start:local_start + length],
                    batch_intensities[local_start:local_start + length]
                )
```

## Performance Analysis

### Expected Benefits

1. **CPU Usage Reduction**: 
   - Stage 1: Accept 100% CPU for one-time conversion
   - Stage 2: ~10-20% CPU for zarr reading vs 100% for TSF reading
   - Net result: 80-90% CPU savings for repeated processing

2. **I/O Efficiency**:
   - Sequential chunk reads vs random frame access
   - Compressed storage reduces disk bandwidth
   - Dask-optimized chunking for parallel processing

3. **Memory Efficiency**:
   - Stream processing without loading full dataset
   - Configurable chunk sizes for memory budget
   - No SDK memory overhead

### Estimated Storage Requirements

```
Original TSF file: ~2GB (analysis.tsf)
Intermediate Zarr: ~3-4GB (uncompressed raw data + indexing)
With compression: ~1.5-2GB (30-40% compression typical)
```

### Performance Projections

```
TSF Reading:     1,200 spectra/sec @ 100% CPU
Zarr Reading:    5,000-10,000 spectra/sec @ 15% CPU
Net speedup:     4-8x effective throughput per CPU cycle
```

## Limitations and Trade-offs

### 1. Storage Overhead
- **Cost**: Additional 1.5-2GB intermediate storage
- **Benefit**: 4-8x processing efficiency for interpolation workflows

### 2. Two-Stage Pipeline
- **Cost**: Must run conversion step before interpolation
- **Benefit**: Conversion is one-time cost, interpolation becomes highly efficient

### 3. Complexity
- **Cost**: Additional codebase complexity for format conversion
- **Benefit**: Clean separation of concerns, better testability

### 4. Memory Requirements
- **Cost**: Need sufficient RAM for chunk processing (1-2GB recommended)
- **Benefit**: Configurable chunk sizes accommodate different hardware

## Integration with Existing Workflow

### Modified Conversion Pipeline

```python
def convert_msi_with_intermediate(
    input_path,
    output_path,
    intermediate_path=None,
    use_existing_intermediate=True
):
    # Use/create intermediate format
    if intermediate_path is None:
        intermediate_path = input_path.with_suffix('.zarr')
    
    if not intermediate_path.exists() or not use_existing_intermediate:
        # Stage 1: TSF → Zarr (100% CPU acceptable)
        converter = TSFToZarrConverter(input_path, intermediate_path)
        converter.convert()
    
    # Stage 2: Zarr → SpatialData (CPU efficient)
    zarr_reader = ZarrSpectrumReader(intermediate_path)
    spatial_converter = SpatialDataConverter(zarr_reader, output_path)
    spatial_converter.convert()
```

### Dask Integration

```python
# Efficient dask processing
def create_dask_interpolation_workflow(zarr_path, interpolation_params):
    zarr_reader = ZarrSpectrumReader(zarr_path)
    
    # Create dask delayed tasks from zarr chunks
    chunk_tasks = []
    for chunk_start in range(0, zarr_reader.total_spectra, chunk_size):
        chunk_task = dask.delayed(interpolate_chunk)(
            zarr_reader, chunk_start, chunk_start + chunk_size, interpolation_params
        )
        chunk_tasks.append(chunk_task)
    
    # Execute with optimal parallelization
    return dask.compute(*chunk_tasks)
```

## Conclusion

This intermediate format approach provides a clean architectural solution to the Bruker SDK CPU bottleneck. By accepting the one-time cost of 100% CPU during TSF reading, we can achieve 4-8x effective throughput for interpolation workflows while maintaining full data fidelity and enabling efficient dask parallelization.

The ragged array design efficiently handles variable spectrum lengths without padding waste, and the chunking strategy optimizes for sequential access patterns common in interpolation algorithms.