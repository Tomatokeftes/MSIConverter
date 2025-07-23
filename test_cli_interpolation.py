#!/usr/bin/env python3
"""
Test interpolation through the CLI interface
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def run_cli_test(input_file, output_file, interpolation_bins=None, interpolation_width=None):
    """Run msiconvert with interpolation enabled"""
    cmd = [
        sys.executable, "-m", "msiconvert",
        str(input_file),
        str(output_file),
        "--format", "spatialdata",
        "--pixel-size", "50",
        "--interpolate"
    ]
    
    if interpolation_bins:
        cmd.extend(["--interpolation-bins", str(interpolation_bins)])
    elif interpolation_width:
        cmd.extend(["--interpolation-width", str(interpolation_width)])
        
    cmd.extend(["--log-level", "INFO"])
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
            
        print(f"\nExecution time: {elapsed:.1f} seconds")
        print(f"Return code: {result.returncode}")
        
        # Check if output file was created
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"Output file created: {output_file} ({size_mb:.1f} MB)")
            return True
        else:
            print("ERROR: Output file not created")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        # Clean up output file
        if output_file.exists():
            import shutil
            if output_file.is_dir():
                shutil.rmtree(output_file)
            else:
                output_file.unlink()

def main():
    print("Testing MSIConvert CLI with interpolation enabled")
    print("=" * 60)
    
    # Test 1: Bruker data with bin-based interpolation
    print("\nTest 1: Bruker data with 10000 bins")
    print("-" * 60)
    success1 = run_cli_test(
        Path("data/20231109_PEA_NEDC.d"),
        Path("test_output_bruker_10k.zarr"),
        interpolation_bins=10000
    )
    
    # Test 2: ImzML data with width-based interpolation
    print("\n\nTest 2: ImzML data with 0.1 Da width")
    print("-" * 60)
    success2 = run_cli_test(
        Path("data/pea.imzML"),
        Path("test_output_imzml_width.zarr"),
        interpolation_width=0.1
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Bruker with bins): {'PASS' if success1 else 'FAIL'}")
    print(f"Test 2 (ImzML with width): {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("\n[SUCCESS] All CLI interpolation tests passed!")
        return 0
    else:
        print("\n[FAILED] Some CLI interpolation tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())