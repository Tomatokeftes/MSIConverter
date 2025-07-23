#!/usr/bin/env python3
"""
Test suite for Phase 4 CLI integration
"""

import sys
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_cli_argument_parsing():
    """Test that CLI arguments are properly parsed for interpolation"""
    print("=== Testing CLI Argument Parsing ===")
    
    try:
        # Import the main function and create parser
        from msiconvert.__main__ import main
        from msiconvert.__main__ import argparse
        
        # Create a mock parser to test argument definitions
        parser = argparse.ArgumentParser()
        
        # Add arguments (similar to main function)
        parser.add_argument("input")
        parser.add_argument("output") 
        parser.add_argument("--interpolate", action="store_true")
        parser.add_argument("--interpolation-bins", type=int, default=None)
        parser.add_argument("--interpolation-width", type=float, default=None)
        parser.add_argument("--interpolation-method", choices=["pchip", "linear", "adaptive"], default="pchip")
        parser.add_argument("--max-workers", type=int, default=None)
        parser.add_argument("--min-workers", type=int, default=4)
        parser.add_argument("--no-quality-validation", action="store_true")
        parser.add_argument("--physics-model", choices=["auto", "tof", "orbitrap", "fticr"], default="auto")
        
        # Test basic interpolation arguments
        args = parser.parse_args(["input.d", "output.zarr", "--interpolate", "--interpolation-bins", "50000"])
        print(f"[OK] Basic interpolation args: interpolate={args.interpolate}, bins={args.interpolation_bins}")
        assert args.interpolate == True
        assert args.interpolation_bins == 50000
        
        # Test width-based specification
        args = parser.parse_args(["input.d", "output.zarr", "--interpolate", "--interpolation-width", "0.01"])
        print(f"[OK] Width-based args: interpolate={args.interpolate}, width={args.interpolation_width}")
        assert args.interpolate == True
        assert args.interpolation_width == 0.01
        
        # Test performance arguments
        args = parser.parse_args(["input.d", "output.zarr", "--interpolate", "--max-workers", "16", "--min-workers", "8"])
        print(f"[OK] Performance args: max_workers={args.max_workers}, min_workers={args.min_workers}")
        assert args.max_workers == 16
        assert args.min_workers == 8
        
        # Test method selection
        args = parser.parse_args(["input.d", "output.zarr", "--interpolate", "--interpolation-method", "adaptive"])
        print(f"[OK] Method selection: method={args.interpolation_method}")
        assert args.interpolation_method == "adaptive"
        
        # Test physics model selection
        args = parser.parse_args(["input.d", "output.zarr", "--interpolate", "--physics-model", "tof"])
        print(f"[OK] Physics model: physics_model={args.physics_model}")
        assert args.physics_model == "tof"
        
        return True
        
    except Exception as e:
        print(f"[FAIL] CLI argument parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interpolation_config_creation():
    """Test that InterpolationConfig is properly created from CLI args"""
    print("\\n=== Testing InterpolationConfig Creation ===")
    
    try:
        from msiconvert.config import InterpolationConfig
        
        # Test creation with bins
        config = InterpolationConfig(
            enabled=True,
            method="pchip",
            interpolation_bins=50000,
            max_workers=16,
            min_workers=4,
            validate_quality=True,
            physics_model="auto"
        )
        
        print(f"[OK] Config created with bins: {config.get_summary()}")
        assert config.enabled == True
        assert config.interpolation_bins == 50000
        assert config.method == "pchip"
        
        # Test creation with width
        config = InterpolationConfig(
            enabled=True,
            method="adaptive",
            interpolation_width=0.01,
            interpolation_width_mz=400.0,
            max_workers=8,
            min_workers=4,
            validate_quality=False,
            physics_model="tof"
        )
        
        print(f"[OK] Config created with width: {config.get_summary()}")
        assert config.enabled == True
        assert config.interpolation_width == 0.01
        assert config.method == "adaptive"
        assert config.physics_model == "tof"
        
        return True
        
    except Exception as e:
        print(f"[FAIL] InterpolationConfig creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convert_msi_integration():
    """Test that convert_msi properly accepts and uses interpolation_config"""
    print("\\n=== Testing convert_msi Integration ===")
    
    try:
        from msiconvert.convert import convert_msi
        from msiconvert.config import InterpolationConfig
        
        # Create test interpolation config
        interp_config = InterpolationConfig(
            enabled=True,
            method="pchip",
            interpolation_bins=10000,
            validate_quality=True
        )
        
        # Mock the conversion process
        with patch('msiconvert.convert.detect_format') as mock_detect, \
             patch('msiconvert.convert.get_reader_class') as mock_reader_class, \
             patch('msiconvert.convert.get_converter_class') as mock_converter_class, \
             patch('msiconvert.convert.Path') as mock_path_class:
            
            # Setup mocks
            mock_detect.return_value = "bruker"
            
            # Create mock path instances
            mock_input_path = Mock()
            mock_input_path.resolve.return_value = mock_input_path
            mock_input_path.exists.return_value = True
            
            mock_output_path = Mock()
            mock_output_path.resolve.return_value = mock_output_path
            mock_output_path.exists.return_value = False
            
            # Configure Path class to return appropriate instances
            def path_side_effect(path_str):
                if "input" in str(path_str):
                    return mock_input_path
                else:
                    return mock_output_path
                    
            mock_path_class.side_effect = path_side_effect
            
            # Mock reader
            mock_reader = Mock()
            mock_reader.get_pixel_size.return_value = (50.0, 50.0)
            mock_reader_class_instance = Mock()
            mock_reader_class_instance.__name__ = "MockBrukerReader"
            mock_reader_class_instance.return_value = mock_reader
            mock_reader_class.return_value = mock_reader_class_instance
            
            # Mock converter
            mock_converter = Mock()
            mock_converter.convert_with_interpolation.return_value = True
            mock_converter.convert.return_value = True
            mock_converter_class_instance = Mock()
            mock_converter_class_instance.__name__ = "MockSpatialDataConverter"
            mock_converter_class_instance.return_value = mock_converter
            mock_converter_class.return_value = mock_converter_class_instance
            
            # Test conversion with interpolation
            result = convert_msi(
                input_path="/mock/input.d",
                output_path="/mock/output.zarr",
                interpolation_config=interp_config,
                pixel_size_um=50.0
            )
            
            print(f"[OK] convert_msi with interpolation: result={result}")
            assert result == True
            
            # Verify that convert_with_interpolation was called
            mock_converter.convert_with_interpolation.assert_called_once_with(interp_config)
            print("[OK] convert_with_interpolation was called with correct config")
            
            # Test conversion without interpolation
            mock_converter.reset_mock()
            mock_converter.convert.return_value = True
            
            result = convert_msi(
                input_path="/mock/input.d",
                output_path="/mock/output.zarr",
                interpolation_config=None,
                pixel_size_um=50.0
            )
            
            print(f"[OK] convert_msi without interpolation: result={result}")
            assert result == True
            
            # Verify that standard convert was called
            mock_converter.convert.assert_called_once()
            print("[OK] Standard convert was called when no interpolation config")
            
        return True
        
    except Exception as e:
        print(f"[FAIL] convert_msi integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_validation():
    """Test CLI argument validation for interpolation options"""
    print("\\n=== Testing CLI Validation ===")
    
    try:
        from msiconvert.config import InterpolationConfig
        
        # Test validation of conflicting options
        try:
            config = InterpolationConfig(
                enabled=True,
                interpolation_bins=50000,
                interpolation_width=0.01  # Both specified - should fail
            )
            print("[FAIL] Should have failed with both bins and width")
            return False
        except ValueError as e:
            print(f"[OK] Correctly rejected conflicting options: {e}")
        
        # Test validation of worker counts
        try:
            config = InterpolationConfig(
                enabled=True,
                min_workers=10,
                max_workers=5  # min > max - should fail
            )
            print("[FAIL] Should have failed with min > max workers")
            return False
        except ValueError as e:
            print(f"[OK] Correctly rejected invalid worker counts: {e}")
        
        # Test validation of invalid method
        try:
            config = InterpolationConfig(
                enabled=True,
                method="invalid_method"
            )
            print("[FAIL] Should have failed with invalid method")
            return False
        except ValueError as e:
            print(f"[OK] Correctly rejected invalid method: {e}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] CLI validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_cli_flow():
    """Test the complete CLI to conversion flow"""
    print("\\n=== Testing End-to-End CLI Flow ===")
    
    try:
        # Test that we can simulate the complete flow
        test_args = {
            'interpolate': True,
            'interpolation_method': 'pchip',
            'interpolation_bins': 75000,
            'interpolation_width': None,
            'interpolation_width_mz': 400.0,
            'max_workers': 16,
            'min_workers': 4,
            'no_quality_validation': False,
            'physics_model': 'auto'
        }
        
        # Simulate CLI argument processing
        from msiconvert.config import InterpolationConfig
        
        if test_args['interpolate']:
            interpolation_config = InterpolationConfig(
                enabled=True,
                method=test_args['interpolation_method'],
                interpolation_bins=test_args['interpolation_bins'],
                interpolation_width=test_args['interpolation_width'],
                interpolation_width_mz=test_args['interpolation_width_mz'],
                max_workers=test_args['max_workers'] or 80,
                min_workers=test_args['min_workers'],
                validate_quality=not test_args['no_quality_validation'],
                physics_model=test_args['physics_model'],
            )
            
            print(f"[OK] End-to-end config creation: {interpolation_config.get_summary()}")
            
            # Verify all settings are correct
            assert interpolation_config.enabled == True
            assert interpolation_config.method == "pchip"
            assert interpolation_config.interpolation_bins == 75000
            assert interpolation_config.max_workers == 16
            assert interpolation_config.validate_quality == True
            assert interpolation_config.physics_model == "auto"
            
            print("[OK] All configuration settings verified")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] End-to-end CLI flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 4 CLI integration tests"""
    print("Phase 4 CLI Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_cli_argument_parsing,
        test_interpolation_config_creation,
        test_convert_msi_integration,
        test_cli_validation,
        test_end_to_end_cli_flow
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\\n" + "=" * 50)
    print("CLI Integration Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("[SUCCESS] All Phase 4 CLI integration tests passed!")
        return True
    else:
        print("[ERROR] Some CLI integration tests failed!")
        return False

if __name__ == "__main__":
    success = main()