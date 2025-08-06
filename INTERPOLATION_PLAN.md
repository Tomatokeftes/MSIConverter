# MSI Interpolation Implementation Plan

## Overview

This document outlines the step-by-step implementation of spectral interpolation capabilities for MSIConverter, focusing on PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation as the primary method. The implementation will be done incrementally, starting with basic interpolation and progressively adding advanced features like Dask integration, streaming, and memory optimization.

## Phase 1: Mass Axis Generation (Foundation)

### 1.1 Mass Axis Generator Architecture
- Create `msiconvert/interpolators/mass_axis/` directory structure
- Implement `BaseMassAxisGenerator` abstract class
- Define mass axis configuration parameters (bins, range, spacing type)
- Create factory pattern for mass axis generation strategies

### 1.2 Linear Mass Axis Implementation
- Implement `LinearMassAxisGenerator` class
- Use `EssentialMetadata.mass_range` to determine min/max bounds
- Generate linearly spaced mass axis with configurable number of bins
- Handle edge cases (invalid ranges, insufficient data points)
- Validate generated axis against input data characteristics

### 1.3 Physics-Based Mass Axis (Advanced Step)
- Research and implement mass analyzer-specific equations:
  - **TOF (Time-of-Flight)**: Non-linear relationship between m/z and time
  - **Orbitrap**: Frequency-based mass determination
  - **FTICR**: Cyclotron frequency relationships
  - **Quadrupole**: Linear scanning patterns
- Create analyzer-specific generators (`TOFMassAxisGenerator`, `OrbitrapMassAxisGenerator`, etc.)
- Use instrument metadata to automatically select appropriate generator
- Implement validation against known instrument calibration standards

## Phase 2: Core Interpolation Infrastructure

### 2.1 Base Interpolator Architecture
- Create base `msiconvert/interpolators/` structure (separate from mass_axis)
- Implement `BaseInterpolator` abstract class with common interface
- Define interpolation configuration and parameters
- Create factory pattern for interpolator selection
- Integrate with mass axis generators

### 2.2 PCHIP Interpolator Implementation
- Implement `PchipInterpolator` class inheriting from `BaseInterpolator`
- Use `scipy.interpolate.PchipInterpolator` as backend
- Accept target mass axis from mass axis generators
- Handle edge cases (single points, duplicate m/z values, NaN handling)
- Implement spectrum-to-axis interpolation logic

### 2.3 Integration Points
- Add interpolation options to `convert_msi()` function parameters
- Modify converter base classes to support interpolated data flow
- Update CLI to include interpolation flags and parameters
- Ensure backward compatibility (interpolation optional)

## Phase 3: SpatialData Integration

### 3.1 Converter Modifications
- Update `SpatialDataConverter` to handle interpolated spectra
- Modify spectrum processing pipeline to include interpolation step
- Ensure coordinate system preservation during interpolation
- Handle metadata updates (original vs interpolated mass ranges)

### 3.2 Data Flow Architecture
```
Raw Spectrum → Mass Axis Generation → Interpolation → SpatialData Formatting → Zarr Storage
```

### 3.3 Memory Management
- Implement spectrum batching to prevent memory overflow
- Add configurable batch sizes based on available memory
- Include progress tracking for long interpolation processes

## Phase 4: Configuration and Parameters

### 4.1 Interpolation Parameters
- **Target mass axis**: User-defined or auto-generated
- **Interpolation bins**: Number of points in target axis
- **Mass range**: Custom range or auto-detected from data
- **Axis type**: Linear, logarithmic, or adaptive spacing
- **Edge handling**: Zero-padding, extrapolation, or truncation

### 4.2 Configuration Integration
- Add interpolation section to configuration system
- Support both programmatic and CLI parameter passing
- Include validation for parameter combinations
- Provide sensible defaults for common use cases

## Phase 5: Quality Assurance and Validation

### 5.1 Testing Strategy
- Unit tests for interpolator classes and edge cases
- Integration tests with ImzML and Bruker readers
- Performance benchmarks with different dataset sizes
- Accuracy validation against known interpolation results

### 5.2 Error Handling
- Graceful degradation when interpolation fails
- Clear error messages for invalid parameters
- Fallback to original data when appropriate
- Memory limit detection and warnings

## Phase 6: Advanced Features (Future Phases)

### 6.1 Multiple Interpolation Methods
- Linear interpolation as lightweight alternative
- Spline interpolation for smooth curves
- Custom interpolation methods for specific instrument types

### 6.2 Adaptive Interpolation
- Dynamic mass axis generation based on spectral density
- Region-specific interpolation parameters
- Automatic parameter optimization

### 6.3 Performance Optimization
- Vectorized interpolation operations
- Memory-mapped intermediate storage
- Parallel processing for independent spectra

## Phase 7: Dask Integration (Future)

### 7.1 Distributed Processing
- Dask-based parallel interpolation
- Chunk-based processing for large datasets
- Memory-efficient streaming workflows

### 7.2 Scalability Features
- Out-of-core processing for datasets larger than RAM
- Distributed computing cluster support
- Progress monitoring and job management

## Implementation Priority

### Immediate (Phase 1-3)
1. **Mass Axis Generation** (START HERE): Create `LinearMassAxisGenerator` using `EssentialMetadata.mass_range`
2. **Core Interpolation**: Implement `PchipInterpolator` that accepts target mass axis
3. **SpatialData Integration**: Update converter pipeline to support interpolated workflows
4. **CLI Support**: Add interpolation parameters (`--interpolate`, `--interpolation-bins`, etc.)

### Short-term (Phase 4-5)
1. **Configuration System**: Interpolation parameters and validation
2. **Testing Suite**: Unit and integration tests for mass axis and interpolation
3. **Error Handling**: Graceful degradation and user-friendly error messages
4. **Documentation**: Usage examples and API documentation

### Medium-term (Phase 6)
1. **Physics-Based Mass Axis**: Implement analyzer-specific equations
2. **Advanced Interpolation**: Multiple interpolation methods and adaptive features
3. **Performance Optimization**: Vectorization and memory efficiency improvements

### Long-term (Phase 7)
1. **Dask Integration**: Distributed processing and parallel interpolation
2. **Streaming**: Out-of-core processing for massive datasets
3. **Distributed Computing**: Cluster support and job management

## Codebase Integration

### Proposed Directory Structure
```
msiconvert/
├── interpolators/
│   ├── __init__.py
│   ├── base_interpolator.py          # BaseInterpolator abstract class
│   ├── pchip_interpolator.py         # PCHIP implementation
│   └── mass_axis/
│       ├── __init__.py
│       ├── base_generator.py         # BaseMassAxisGenerator abstract class
│       ├── linear_generator.py       # LinearMassAxisGenerator
│       └── physics_generators.py     # TOF, Orbitrap, FTICR generators (future)
```

### Integration Points
- **Essential Metadata Access**: Mass axis generators will use `EssentialMetadata.mass_range` from existing metadata system
- **Converter Integration**: Modify `SpatialDataConverter` to optionally include interpolation step
- **CLI Integration**: Extend `convert_msi()` parameters and CLI argument parser
- **Configuration System**: Add interpolation section to existing config framework

### Why This Location?
1. **Specialized Functionality**: Interpolation is a distinct scientific computation warranting its own module
2. **Mass Axis Coupling**: Mass axis generation is primarily used by interpolation, so co-location makes sense
3. **Future Extensibility**: Easy to add new interpolation methods and mass axis strategies
4. **Testing Isolation**: Enables focused unit testing of interpolation logic
5. **Optional Feature**: Keeps interpolation separate from core conversion pipeline

## Key Design Decisions

### Architecture Principles
- **Modular Design**: Interpolators as pluggable components
- **Backward Compatibility**: All interpolation features optional
- **Memory Efficiency**: Process data in batches, avoid loading entire datasets
- **Extensibility**: Easy to add new interpolation methods
- **Configuration-Driven**: Support both programmatic and declarative configuration

### Technical Choices
- **PCHIP as Primary Method**: Proven performance in spectral analysis
- **SciPy Backend**: Leverage well-tested scientific computing libraries
- **Zarr-Compatible Output**: Maintain compatibility with existing storage format
- **Progressive Implementation**: Start simple, add complexity incrementally

### Integration Strategy
- **Minimal Core Changes**: Avoid major refactoring of existing code
- **Optional by Default**: Existing workflows unaffected
- **CLI-First Approach**: Make interpolation easily accessible to users
- **Test-Driven Development**: Comprehensive testing at each phase

## Success Criteria

### Phase 1 Success Metrics (Mass Axis Generation)
- [ ] Linear mass axis generator creates valid axes from EssentialMetadata
- [ ] Generated mass axes cover full data range with specified number of bins
- [ ] Edge cases handled gracefully (invalid ranges, zero data points)
- [ ] Unit tests achieve >95% coverage for mass axis generation

### Phase 2-3 Success Metrics (Core Interpolation)
- [ ] PCHIP interpolator correctly processes sample spectra
- [ ] Integration tests pass with ImzML and Bruker data
- [ ] CLI accepts interpolation parameters
- [ ] Memory usage remains reasonable for test datasets

### Overall Project Success
- [ ] 10x+ improvement in mass axis uniformity
- [ ] Negligible impact on processing time for small datasets
- [ ] Graceful scaling to large datasets (>1GB)
- [ ] Comprehensive test coverage (>90%)
- [ ] Clear documentation and usage examples

## Risk Mitigation

### Technical Risks
- **Memory Overflow**: Implement batching and progress monitoring
- **Performance Degradation**: Profile and optimize critical paths
- **Interpolation Artifacts**: Extensive validation against known data
- **Integration Complexity**: Maintain clean separation of concerns

### Project Risks
- **Scope Creep**: Implement phases sequentially, validate each step
- **Compatibility Issues**: Thorough testing with existing workflows
- **User Adoption**: Clear documentation and migration guides
- **Maintenance Burden**: Simple, well-documented code architecture
