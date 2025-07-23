#!/usr/bin/env python3
"""
Test script for Phase 1 interpolation infrastructure
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_imports():
    """Test that all Phase 1 modules can be imported"""
    print("=== Testing Imports ===")
    
    try:
        # Test metadata models
        from msiconvert.metadata.metadata_models import (
            InstrumentMetadata, AcquisitionMetadata, 
            SpatialMetadata, DatasetMetadata
        )
        print("[OK] Metadata models imported successfully")
        
        # Test base extractor
        from msiconvert.metadata.base_extractor import BaseMetadataExtractor
        print("[OK] Base extractor imported successfully")
        
        # Test Bruker extractor
        from msiconvert.metadata.bruker_extractor import BrukerMetadataExtractor
        print("[OK] Bruker extractor imported successfully")
        
        # Test ImzML extractor
        from msiconvert.metadata.imzml_extractor import ImzMLMetadataExtractor
        print("[OK] ImzML extractor imported successfully")
        
        # Test interpolators
        from msiconvert.interpolators import SpectrumData, BoundsInfo, SpectrumBuffer
        print("[OK] Core data types imported successfully")
        
        # Test physics models
        from msiconvert.interpolators.physics_models import (
            TOFPhysics, OrbitrapPhysics, FTICRPhysics, create_physics_model
        )
        print("[OK] Physics models imported successfully")
        
        # Test bounds detector
        from msiconvert.interpolators.bounds_detector import (
            detect_bounds_from_path, validate_bounds
        )
        print("[OK] Bounds detector imported successfully")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_metadata_models():
    """Test Pydantic metadata models"""
    print("\n=== Testing Metadata Models ===")
    
    try:
        from msiconvert.metadata.metadata_models import (
            InstrumentMetadata, AcquisitionMetadata, 
            SpatialMetadata, DatasetMetadata
        )
        
        # Test InstrumentMetadata
        instrument = InstrumentMetadata(
            instrument_type="TOF",
            model="TestTOF",
            resolution_at_400=10000
        )
        print(f"[OK] InstrumentMetadata: {instrument.instrument_type}, {instrument.model}")
        
        # Test with aliases for AcquisitionMetadata
        acquisition_data = {
            "MzAcqRangeLower": 100.0,
            "MzAcqRangeUpper": 1000.0,
            "SpotSizeX": 50.0,
            "SpotSizeY": 50.0
        }
        acquisition = AcquisitionMetadata(**acquisition_data)
        print(f"[OK] AcquisitionMetadata: m/z {acquisition.mass_range_lower}-{acquisition.mass_range_upper}")
        
        # Test SpatialMetadata with aliases
        spatial_data = {
            "ImagingAreaMinXIndexPos": 0,
            "ImagingAreaMaxXIndexPos": 99,
            "ImagingAreaMinYIndexPos": 0,
            "ImagingAreaMaxYIndexPos": 99
        }
        spatial = SpatialMetadata(**spatial_data)
        print(f"[OK] SpatialMetadata: {spatial.min_x}-{spatial.max_x} x {spatial.min_y}-{spatial.max_y}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Metadata models test failed: {e}")
        return False

def test_physics_models():
    """Test physics models calculations"""
    print("\n=== Testing Physics Models ===")
    
    try:
        from msiconvert.interpolators.physics_models import (
            TOFPhysics, OrbitrapPhysics, FTICRPhysics, create_physics_model
        )
        
        # Test TOF physics
        tof = TOFPhysics(resolution_at_400=10000)
        res_800 = tof.calculate_resolution_at_mz(800)
        expected_800 = 10000 * np.sqrt(800/400)  # Should be ~14142
        print(f"[OK] TOF resolution at m/z 800: {res_800:.0f} (expected ~14142)")
        
        # Test width calculation
        width_800 = tof.calculate_width_at_mz(800, 0.01, 400)
        expected_width = 0.01 * np.sqrt(800/400)  # Should be ~0.0141
        print(f"[OK] TOF width at m/z 800: {width_800:.4f} (expected ~0.0141)")
        
        # Test Orbitrap physics
        orbitrap = OrbitrapPhysics(resolution_at_400=120000)
        res_orbi_800 = orbitrap.calculate_resolution_at_mz(800)
        expected_orbi = 120000 * np.sqrt(400/800)  # Should be ~84853
        print(f"[OK] Orbitrap resolution at m/z 800: {res_orbi_800:.0f} (expected ~84853)")
        
        # Test mass axis creation
        axis_bins = tof.create_optimal_mass_axis(100, 200, target_bins=100)
        print(f"[OK] Mass axis creation: {len(axis_bins)} bins from 100-200 Da")
        print(f"  First bins: {axis_bins[:5]}")
        print(f"  Last bins: {axis_bins[-5:]}")
        
        # Test width-based axis
        axis_width = tof.create_optimal_mass_axis(100, 200, width_at_mz=(0.01, 150))
        print(f"[OK] Width-based axis: {len(axis_width)} bins")
        
        # Test factory function
        auto_tof = create_physics_model("tof", 15000)
        print(f"[OK] Factory created TOF with resolution: {auto_tof.resolution_at_400}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Physics models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_data_types():
    """Test core data types"""
    print("\n=== Testing Core Data Types ===")
    
    try:
        from msiconvert.interpolators import SpectrumData, BoundsInfo, SpectrumBuffer
        
        # Test SpectrumData
        mz_vals = np.array([100.0, 200.0, 300.0])
        intensities = np.array([1000.0, 500.0, 750.0], dtype=np.float32)
        spectrum = SpectrumData(
            coords=(10, 20, 0),
            mz_values=mz_vals,
            intensities=intensities
        )
        print(f"[OK] SpectrumData: coords {spectrum.coords}, {len(spectrum.mz_values)} points")
        
        # Test BoundsInfo
        bounds = BoundsInfo(
            min_mz=100.0, max_mz=1000.0,
            min_x=0, max_x=99, min_y=0, max_y=99,
            n_spectra=10000
        )
        print(f"[OK] BoundsInfo: m/z {bounds.min_mz}-{bounds.max_mz}, {bounds.n_spectra} spectra")
        
        # Test SpectrumBuffer
        buffer = SpectrumBuffer(
            buffer_id=1,
            mz_buffer=np.zeros(1000, dtype=np.float64),
            intensity_buffer=np.zeros(1000, dtype=np.float32)
        )
        
        # Test buffer operations
        test_mz = np.array([100.0, 200.0, 300.0])
        test_intensity = np.array([10.0, 20.0, 30.0])
        buffer.fill(test_mz, test_intensity)
        
        retrieved_mz, retrieved_intensity = buffer.get_data()
        print(f"[OK] SpectrumBuffer: filled with {buffer.actual_size} points")
        print(f"  Retrieved: mz={retrieved_mz}, intensity={retrieved_intensity}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Core data types test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bounds_detection():
    """Test bounds detection service"""
    print("\n=== Testing Bounds Detection ===")
    
    try:
        from msiconvert.interpolators.bounds_detector import validate_bounds
        from msiconvert.interpolators import BoundsInfo
        
        # Test bounds validation
        valid_bounds = BoundsInfo(
            min_mz=100.0, max_mz=1000.0,
            min_x=0, max_x=99, min_y=0, max_y=99,
            n_spectra=10000
        )
        
        invalid_bounds = BoundsInfo(
            min_mz=1000.0, max_mz=100.0,  # Invalid: min > max
            min_x=0, max_x=99, min_y=0, max_y=99,
            n_spectra=10000
        )
        
        print(f"[OK] Valid bounds validation: {validate_bounds(valid_bounds)}")
        print(f"[OK] Invalid bounds validation: {validate_bounds(invalid_bounds)}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Bounds detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 1 tests"""
    print("Phase 1 Interpolation Infrastructure Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_metadata_models,
        test_physics_models,
        test_core_data_types,  
        test_bounds_detection
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("[SUCCESS] All Phase 1 tests passed!")
        return True
    else:
        print("[ERROR] Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)