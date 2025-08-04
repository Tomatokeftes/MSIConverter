# Dask Processing Optimization Plan

**Date**: August 3, 2025  
**Status**: Analysis Complete, Ready for Implementation  
**Current Performance**: Standard (11s) vs Dask (45s+) - Need to optimize Dask

---

## 🎯 **Objective**
Optimize Dask processing to be **faster than standard processing** while maintaining memory efficiency and parallelization benefits for large datasets.

**Target**: Get Dask processing under 10 seconds for test dataset (currently 45s+)

---

## 📊 **Current Performance Analysis**

### Test Dataset (pea.imzML):
- **Dimensions**: 131×133×1 = 17,423 pixels
- **Actual spectra**: 12,737 non-empty pixels  
- **Mass axis**: 13,398 bins (0.1 Da TOF binning)
- **Memory estimate**: 0.17GB

### Performance Comparison:
| Processing Mode | Time | Status |
|----------------|------|--------|
| **Standard (`--no-dask`)** | ~11 seconds | ✅ Working, fast |
| **Dask (default)** | 45+ seconds | ❌ Too slow |
| **Target Dask** | <10 seconds | 🎯 Goal |

---

## 🔍 **Root Cause Analysis: Why Dask is Slow**

### Current Dask Pipeline Flow:
```
1. Create 13 chunks (1000 pixels each)
2. For each chunk:
   → Task 1: Read pixels individually (1000× get_spectrum calls)
   → Task 2: Interpolate each pixel individually  
   → Task 3: Combine results
3. Execute all tasks with .compute()
4. Process final results
```

### **Critical Bottlenecks Identified:**/

#### 2. **Double Task Overhead** 🚨
```python
# Two-stage processing creates unnecessary overhead:W
Stage 1: Read pixels → List[Dict] (raw m/z data)
Stage 2: Interpolate → List[InterpolationResult] (processed data)
```

**Problems:**
- Double task scheduling overhead
- Memory duplication (raw + processed data)
- Serialization/deserialization between Dask tasks

#### 3. **Poor Memory Patterns** ⚠️
- Creates intermediate dictionaries for every pixel
- Stores both raw m/z arrays AND interpolated results simultaneously
- No memory reuse between chunks

#### 4. **Limited Parallelism** ⚠️
- Only 13 chunks = limited parallel utilization
- File reading is serial within each task
- Reader may not be thread-safe

---

## 🎯 **Optimization Strategy**

### **Key Insight**: Work with file format's natural access patterns, not against them!

**Reader already provides optimized sequential access:**
```python
# EFFICIENT: Sequential file reading (what standard processing uses)
for coords, mzs, intensities in reader.iter_spectra():  # Optimized iteration
    # Process as they come in file order
```

### **Core Optimizations:**

#### 1. **Sequential Batch Processing** (High Impact)
Instead of coordinate-based chunking, use reader's natural iteration:

```python
# Current approach:
chunks = [pixel_indices_0_to_1000, pixel_indices_1000_to_2000, ...]

# Optimized approach:
spectrum_batches = [
    list(reader.iter_spectra(batch_size=1000)),  # Sequential file access
    # Process batches in parallel
]
```

#### 2. **Single-Stage Tasks** (High Impact)
Combine read + interpolate in one step:

```python
# Instead of:
@delayed
def read_pixels(): return [raw_dicts...]
@delayed  
def interpolate_pixels(raw_dicts): return [results...]

# Do:
@delayed
def process_spectrum_batch(spectrum_batch):
    results = []
    for coords, mzs, intensities in spectrum_batch:
        result = interpolator.interpolate_spectrum(mzs, intensities, coords, pixel_idx)
        results.append(result)
    return results
```

#### 3. **Optimized Chunk Sizing** (Medium Impact)
- More smaller chunks for better parallelism (e.g., 500 pixels/chunk = 25 chunks)
- Better CPU utilization across cores
- More responsive progress tracking

#### 4. **Memory Optimization** (Low-Medium Impact)
- Eliminate intermediate dictionaries
- Stream results directly to sparse matrix
- Reduce memory copying

---

## 🛠️ **Implementation Plan**

### **Phase 1: Quick Wins** (Target: 2-3 hours)
1. **Single-stage processing**: Combine read+interpolate tasks
2. **Sequential iteration**: Use `reader.iter_spectra()` instead of `get_spectrum()`
3. **Test performance**: Should see immediate 2-3x improvement

### **Phase 2: Parallelization** (Target: 1-2 hours)  
1. **Batch-based chunking**: Split `iter_spectra()` into parallel batches
2. **Optimize chunk size**: Find sweet spot for parallelism vs overhead
3. **Progress tracking**: Ensure progress bars work with new approach

### **Phase 3: Memory Optimization** (Target: 1 hour)
1. **Eliminate dictionaries**: Direct InterpolationResult creation
2. **Stream processing**: Direct sparse matrix updates
3. **Memory profiling**: Verify improvements

---

## 📁 **Key Files to Modify**

### Primary Files:
- `msiconvert/converters/spatialdata/base_spatialdata_converter.py`
  - `_create_dask_pipeline()` - Switch to sequential batching
  - `_create_pixel_chunk_task()` - Use iter_spectra() instead
  - `_process_dask_chunk()` - Single-stage processing

### Secondary Files:
- `msiconvert/processing/interpolation.py`
  - `create_dask_interpolation_task()` - Adapt for new approach
- Test files to verify performance improvements

---

## 🧪 **Testing Strategy**

### Performance Benchmarks:
```bash
# Current baseline
time python -m msiconvert test_data/pea.imzML test_standard.zarr --pixel-size 25 --no-dask
# Target: ~11 seconds

# Optimized Dask (after implementation)  
time python -m msiconvert test_data/pea.imzML test_dask.zarr --pixel-size 25
# Target: <10 seconds
```

### Validation Tests:
1. **Correctness**: Compare outputs (standard vs optimized Dask)
2. **Memory usage**: Monitor peak memory consumption
3. **Larger datasets**: Test with bigger files to verify scaling
4. **Different formats**: Test ImzML and Bruker formats

---

## 💡 **Expected Outcomes**

### **Immediate Benefits:**
- **3-5x speed improvement** from eliminating random file access
- **2x memory reduction** from single-stage processing
- **Better progress tracking** with smaller chunks

### **Long-term Benefits:**
- **True parallel scaling** for larger datasets
- **Memory efficiency** for datasets that don't fit in RAM
- **Consistent performance** across different file sizes

---

## 🚨 **Key Decisions Made**

### **Architecture Decisions:**
1. ✅ **Default to Dask processing** (implemented)
2. ✅ **Keep `--no-dask` option** for small datasets
3. ✅ **Fixed negative pixel handling** (zero out negatives)
4. ✅ **Fixed mass axis mapping** (nearest neighbor for binned axes)
5. ✅ **Reduced default mass axis size** (300k → 10k bins)

### **Performance Decisions:**
1. 🎯 **Use sequential file access** (to implement)
2. 🎯 **Single-stage Dask tasks** (to implement)  
3. 🎯 **Batch-based parallelization** (to implement)

---

## 📋 **Next Session TODO**

### **Ready to Start:**
1. [ ] Implement sequential batch processing in `_create_dask_pipeline()`
2. [ ] Modify `_create_pixel_chunk_task()` to use `iter_spectra()`
3. [ ] Combine read+interpolate in single task
4. [ ] Test performance with pea.imzML dataset
5. [ ] Compare output correctness vs standard processing

### **Code Locations:**
- **Main file**: `base_spatialdata_converter.py:701-787`
- **Test command**: `python -m msiconvert test_data/pea.imzML test_output.zarr --pixel-size 25`
- **Baseline**: Standard processing ~11s, Dask processing 45s+

---

## 🎯 **Success Criteria**

- [ ] Dask processing completes in <10 seconds (vs current 45s+)
- [ ] Output matches standard processing (correctness)  
- [ ] Memory usage remains reasonable
- [ ] Larger datasets show parallel scaling benefits
- [ ] All existing tests continue to pass

---

## 📝 **Notes**

- **Cannot vectorize interpolation**: Each pixel has different m/z ranges, peak counts, positions
- **Reader optimization is key**: `iter_spectra()` is already optimized for sequential access
- **File format awareness**: ImzML and Bruker have different optimal access patterns
- **Memory vs Speed tradeoff**: Dask should provide both better memory management AND speed

**Bottom line**: Current Dask implementation works against the file format's strengths. Fix this, and we should see dramatic improvements.