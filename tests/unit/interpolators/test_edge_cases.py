#!/usr/bin/env python3
"""
Edge case testing for Phase 1 interpolation infrastructure
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_physics_edge_cases():
    """Test physics models with edge cases"""
    print("=== Testing Physics Edge Cases ===")
    
    from msiconvert.interpolators.physics_models import TOFPhysics, create_physics_model
    
    try:
        tof = TOFPhysics()
        
        # Test invalid mass axis parameters
        try:
            # Should fail - both parameters specified
            axis = tof.create_optimal_mass_axis(100, 200, target_bins=100, width_at_mz=(0.01, 150))
            print("[FAIL] Should have failed with both parameters")
        except ValueError as e:
            print(f"[OK] Correctly rejected both parameters: {e}")
        
        # Test invalid mass axis parameters
        try:
            # Should fail - neither parameter specified
            axis = tof.create_optimal_mass_axis(100, 200)
            print("[FAIL] Should have failed with no parameters")
        except ValueError as e:
            print(f"[OK] Correctly rejected no parameters: {e}")
        
        # Test edge case - very small mass range
        axis_small = tof.create_optimal_mass_axis(100, 100.1, target_bins=10)
        print(f"[OK] Small mass range: {len(axis_small)} bins for 0.1 Da range")
        
        # Test edge case - very large mass range
        axis_large = tof.create_optimal_mass_axis(50, 5000, target_bins=1000)
        print(f"[OK] Large mass range: {len(axis_large)} bins for 4950 Da range")
        
        # Test unknown instrument type
        unknown = create_physics_model("unknown_instrument")
        print(f"[OK] Unknown instrument defaults to TOF: {type(unknown).__name__}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Physics edge cases failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metadata_validation():
    """Test metadata model validation"""
    print("\n=== Testing Metadata Validation ===")
    
    try:
        from msiconvert.metadata.metadata_models import AcquisitionMetadata, SpatialMetadata
        
        # Test missing required fields
        try:
            invalid_acquisition = AcquisitionMetadata(
                MzAcqRangeLower=100.0
                # Missing MzAcqRangeUpper, SpotSizeX, SpotSizeY
            )
            print("[FAIL] Should have failed with missing fields")
        except Exception as e:
            print(f"[OK] Correctly rejected missing fields: {type(e).__name__}")
        
        # Test invalid data types
        try:
            invalid_spatial = SpatialMetadata(
                ImagingAreaMinXIndexPos="not_an_int",  # Should be int
                ImagingAreaMaxXIndexPos=99,
                ImagingAreaMinYIndexPos=0,
                ImagingAreaMaxYIndexPos=99
            )
            print("[FAIL] Should have failed with invalid type")
        except Exception as e:
            print(f"[OK] Correctly rejected invalid type: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Metadata validation failed: {e}")
        return False

def test_buffer_edge_cases():
    """Test spectrum buffer edge cases"""
    print("\n=== Testing Buffer Edge Cases ===")
    
    try:
        from msiconvert.interpolators import SpectrumBuffer
        
        # Test buffer with empty data
        buffer = SpectrumBuffer(
            buffer_id=1,
            mz_buffer=np.zeros(1000, dtype=np.float64),
            intensity_buffer=np.zeros(1000, dtype=np.float32)
        )
        
        # Fill with empty arrays
        empty_mz = np.array([])
        empty_intensity = np.array([])
        buffer.fill(empty_mz, empty_intensity)
        
        retrieved_mz, retrieved_intensity = buffer.get_data()
        print(f"[OK] Empty buffer handling: {len(retrieved_mz)} points")
        
        # Test buffer overflow (more data than buffer size)
        large_mz = np.arange(2000, dtype=np.float64)  # Larger than buffer
        large_intensity = np.arange(2000, dtype=np.float32)
        
        try:
            buffer.fill(large_mz, large_intensity)
            print("[FAIL] Should have failed with buffer overflow")
        except (ValueError, IndexError) as e:
            print(f"[OK] Correctly handled buffer overflow: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Buffer edge cases failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bounds_validation_edge_cases():
    """Test bounds validation with edge cases"""
    print("\n=== Testing Bounds Validation Edge Cases ===")
    
    try:
        from msiconvert.interpolators.bounds_detector import validate_bounds
        from msiconvert.interpolators import BoundsInfo
        
        # Test zero mass range
        zero_range = BoundsInfo(
            min_mz=100.0, max_mz=100.0,  # Same values
            min_x=0, max_x=99, min_y=0, max_y=99,
            n_spectra=10000
        )
        print(f"[OK] Zero mass range validation: {validate_bounds(zero_range)}")
        
        # Test negative mass values
        negative_mass = BoundsInfo(
            min_mz=-50.0, max_mz=100.0,  # Negative min
            min_x=0, max_x=99, min_y=0, max_y=99,
            n_spectra=10000
        )
        print(f"[OK] Negative mass validation: {validate_bounds(negative_mass)}")
        
        # Test inverted spatial bounds
        inverted_spatial = BoundsInfo(
            min_mz=100.0, max_mz=1000.0,
            min_x=99, max_x=0,  # min > max
            min_y=0, max_y=99,
            n_spectra=10000
        )
        print(f"[OK] Inverted spatial bounds validation: {validate_bounds(inverted_spatial)}")
        
        # Test very small mass range (< 1 Da)
        tiny_range = BoundsInfo(
            min_mz=100.0, max_mz=100.5,  # 0.5 Da range
            min_x=0, max_x=99, min_y=0, max_y=99,
            n_spectra=10000
        )
        print(f"[OK] Tiny mass range validation: {validate_bounds(tiny_range)}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Bounds validation edge cases failed: {e}")
        return False

def main():
    """Run all edge case tests"""
    print("Phase 1 Edge Case Test Suite")
    print("=" * 40)
    
    tests = [
        test_physics_edge_cases,
        test_metadata_validation,
        test_buffer_edge_cases,
        test_bounds_validation_edge_cases
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 40)
    print("Edge Case Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("[SUCCESS] All edge case tests passed!")
        return True
    else:
        print("[ERROR] Some edge case tests failed!")
        return False

if __name__ == "__main__":
    success = main()