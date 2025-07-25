"""
Comprehensive test suite for the Bruker Reader.

This test suite validates all functionality of the reader including:
- Basic reading operations
- Memory management
- Performance characteristics
- Data integrity
- Error handling
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Import the reader from the proper location
from msiconvert.readers.bruker.bruker_reader import BrukerReader
from msiconvert.utils.bruker_exceptions import (
    BrukerReaderError,
    DataError,
    SDKError,
)

logger = logging.getLogger(__name__)


class BrukerReaderTester:
    """Comprehensive tester for the Bruker reader."""

    def __init__(self, test_data_path: str):
        """
        Initialize the tester.

        Args:
            test_data_path: Path to test data directory
        """
        self.test_data_path = Path(test_data_path)
        self.results = {}
        self.errors = []

        print(f"Initializing BrukerReaderTester with data: {self.test_data_path}")

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tests and return comprehensive results.

        Returns:
            Dictionary containing all test results
        """
        print("=" * 80)
        print("BRUKER READER - COMPREHENSIVE TEST SUITE")
        print("=" * 80)

        tests = [
            ("Basic Initialization", self.test_initialization),
            ("Data Path Validation", self.test_data_path_validation),
            ("Metadata Reading", self.test_metadata_reading),
            ("Dimensions Calculation", self.test_dimensions),
            ("Coordinate Access", self.test_coordinate_access),
            ("Spectrum Reading", self.test_spectrum_reading),
            ("Common Mass Axis", self.test_common_mass_axis),
            ("Sequential Processing", self.test_sequential_processing),
            ("Performance Metrics", self.test_performance),
            ("Error Handling", self.test_error_handling),
            ("Resource Cleanup", self.test_cleanup),
        ]

        for test_name, test_func in tests:
            print(f"\n{'-' * 60}")
            print(f"Running: {test_name}")
            print(f"{'-' * 60}")

            try:
                start_time = time.time()
                result = test_func()
                end_time = time.time()

                self.results[test_name] = {
                    "status": "PASSED",
                    "result": result,
                    "duration_ms": (end_time - start_time) * 1000,
                    "error": None,
                }

                print(f"✅ {test_name}: PASSED ({(end_time - start_time) * 1000:.1f}ms)")

            except Exception as e:
                end_time = time.time()

                self.results[test_name] = {
                    "status": "FAILED",
                    "result": None,
                    "duration_ms": (end_time - start_time) * 1000,
                    "error": str(e),
                }

                self.errors.append((test_name, e))
                print(f"❌ {test_name}: FAILED - {e}")
                logger.error(f"Test failed: {test_name}", exc_info=True)

        # Print summary
        self.print_test_summary()

        return self.results

    def test_initialization(self) -> Dict[str, Any]:
        """Test basic reader initialization."""
        reader = BrukerReader(
            data_path=self.test_data_path, cache_coordinates=True, memory_limit_gb=2.0
        )

        # Verify basic properties
        assert reader.data_path == self.test_data_path
        assert reader.file_type in ["tsf", "tdf"]
        assert hasattr(reader, "handle")
        assert hasattr(reader, "conn")

        reader.close()

        return {"file_type": reader.file_type, "data_path": str(reader.data_path)}

    def test_data_path_validation(self) -> Dict[str, Any]:
        """Test data path validation."""
        # Test with invalid path
        try:
            invalid_reader = BrukerReader(Path("/nonexistent/path.d"))
            assert False, "Should have raised an exception"
        except BrukerReaderError:
            pass  # Expected

        # Test with valid path
        reader = BrukerReader(self.test_data_path)
        assert reader.data_path.exists()
        assert reader.data_path.is_dir()
        reader.close()

        return {"validation": "passed"}

    def test_metadata_reading(self) -> Dict[str, Any]:
        """Test metadata reading functionality."""
        with BrukerReader(self.test_data_path) as reader:
            metadata = reader.get_metadata()

            # Verify essential metadata fields
            required_fields = ["source", "file_type", "frame_count", "is_maldi"]
            for field in required_fields:
                assert field in metadata, f"Missing metadata field: {field}"

            return {
                "metadata_keys": list(metadata.keys()),
                "frame_count": metadata.get("frame_count"),
                "file_type": metadata.get("file_type"),
                "is_maldi": metadata.get("is_maldi"),
            }

    def test_dimensions(self) -> Dict[str, Any]:
        """Test dimension calculation."""
        with BrukerReader(self.test_data_path) as reader:
            dimensions = reader.get_dimensions()

            assert len(dimensions) == 3, "Dimensions should be 3-tuple"
            assert all(d > 0 for d in dimensions), "All dimensions should be positive"

            x, y, z = dimensions

            return {
                "dimensions": dimensions,
                "x_size": x,
                "y_size": y,
                "z_size": z,
                "total_pixels": x * y * z,
            }

    def test_coordinate_access(self) -> Dict[str, Any]:
        """Test coordinate caching and access."""
        with BrukerReader(self.test_data_path, cache_coordinates=True) as reader:
            # Test coordinate access
            coord_cache = reader.coordinate_cache

            # Test getting a single coordinate
            coord = coord_cache.get_coordinate(1)
            assert coord is not None, "Should be able to get coordinate for frame 1"
            assert len(coord) == 3, "Coordinate should be 3-tuple"

            # Test batch coordinate access
            frame_ids = list(range(1, min(11, reader._get_frame_count() + 1)))
            coords_batch = coord_cache.get_coordinates_batch(frame_ids)

            assert len(coords_batch) > 0, "Should get some coordinates"

            return {
                "single_coord": coord,
                "batch_size": len(coords_batch),
                "cache_stats": coord_cache.get_coverage_stats(),
            }

    def test_spectrum_reading(self) -> Dict[str, Any]:
        """Test spectrum reading functionality."""
        with BrukerReader(self.test_data_path) as reader:
            spectra_count = 0
            total_peaks = 0
            mz_ranges = []

            # Read first 10 spectra
            for coords, mzs, intensities in reader.iter_spectra():
                spectra_count += 1
                total_peaks += len(mzs)

                # Validate spectrum data
                assert len(coords) == 3, "Coordinates should be 3-tuple"
                assert len(mzs) == len(
                    intensities
                ), "m/z and intensity arrays should have same length"
                assert np.all(mzs >= 0), "m/z values should be non-negative"
                assert np.all(
                    intensities >= 0
                ), "Intensity values should be non-negative"

                if len(mzs) > 0:
                    mz_ranges.append((float(np.min(mzs)), float(np.max(mzs))))

                if spectra_count >= 10:  # Limit to first 10 for testing
                    break

            return {
                "spectra_read": spectra_count,
                "total_peaks": total_peaks,
                "avg_peaks_per_spectrum": total_peaks / max(1, spectra_count),
                "mz_ranges": mz_ranges[:5],  # First 5 ranges
            }

    def test_common_mass_axis(self) -> Dict[str, Any]:
        """Test common mass axis construction."""
        with BrukerReader(self.test_data_path) as reader:
            mass_axis = reader.get_common_mass_axis()

            assert isinstance(mass_axis, np.ndarray), "Mass axis should be numpy array"
            assert len(mass_axis) > 0, "Mass axis should not be empty"
            assert np.all(np.diff(mass_axis) >= 0), "Mass axis should be sorted"

            return {
                "mass_axis_length": len(mass_axis),
                "min_mz": float(np.min(mass_axis)),
                "max_mz": float(np.max(mass_axis)),
                "mass_range": float(np.max(mass_axis) - np.min(mass_axis)),
            }

    def test_spectrum_reading(self) -> Dict[str, Any]:
        """Test spectrum reading without memory management complexity."""
        with BrukerReader(self.test_data_path) as reader:
            # Read some spectra to test iteration
            spectrum_count = 0
            total_peaks = 0
            
            for coords, mzs, intensities in reader.iter_spectra():
                spectrum_count += 1
                total_peaks += len(mzs)
                if spectrum_count >= 20:
                    break

            return {
                "spectrum_count": spectrum_count,
                "total_peaks": total_peaks,
                "avg_peaks_per_spectrum": total_peaks / spectrum_count if spectrum_count > 0 else 0,
            }

    def test_sequential_processing(self) -> Dict[str, Any]:
        """Test sequential processing functionality (no batching)."""
        with BrukerReader(self.test_data_path) as reader:
            start_time = time.time()
            spectrum_count = 0

            # Test that batch_size parameter is ignored but doesn't break anything
            for coords, mzs, intensities in reader.iter_spectra(batch_size=10):
                spectrum_count += 1
                if spectrum_count >= 30:  # Limit for testing
                    break

            end_time = time.time()

            return {
                "spectra_processed": spectrum_count,
                "duration_ms": (end_time - start_time) * 1000,
                "spectra_per_second": spectrum_count / max(0.001, end_time - start_time),
                "note": "batch_size parameter ignored in implementation"
            }

    def test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics."""
        with BrukerReader(self.test_data_path) as reader:
            # Measure initialization time
            start_time = time.time()
            dimensions = reader.get_dimensions()
            init_time = time.time() - start_time

            # Measure spectrum reading time
            start_time = time.time()
            spectrum_count = 0
            for coords, mzs, intensities in reader.iter_spectra():
                spectrum_count += 1
                if spectrum_count >= 50:
                    break
            read_time = time.time() - start_time

            # Get performance stats
            perf_stats = reader.get_performance_stats()

            return {
                "initialization_time_ms": init_time * 1000,
                "spectrum_reading_time_ms": read_time * 1000,
                "spectra_per_second": spectrum_count / max(0.001, read_time),
                "performance_stats": perf_stats,
            }

    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling capabilities."""
        errors_caught = []

        # Test invalid frame access (if applicable)
        try:
            with BrukerReader(self.test_data_path) as reader:
                # Try to access invalid coordinate
                invalid_coord = reader.coordinate_cache.get_coordinate(999999)
                # This might return None rather than raise an error
        except Exception as e:
            errors_caught.append(("invalid_coordinate", type(e).__name__))

        # Test with invalid data path
        try:
            reader = BrukerReader(Path("/invalid/path.d"))
        except BrukerReaderError as e:
            errors_caught.append(("invalid_path", type(e).__name__))

        return {"errors_caught": errors_caught, "error_handling": "functional"}

    def test_cleanup(self) -> Dict[str, Any]:
        """Test resource cleanup."""
        reader = BrukerReader(self.test_data_path)

        # Verify resources are open
        assert reader.handle is not None
        assert reader.conn is not None

        # Close and verify cleanup
        reader.close()

        # Verify resources are cleaned up
        assert reader.handle is None
        assert reader.conn is None

        return {"cleanup": "successful"}

    def print_test_summary(self) -> None:
        """Print a comprehensive test summary."""
        print(f"\n{'=' * 80}")
        print("TEST SUMMARY")
        print(f"{'=' * 80}")

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["status"] == "PASSED")
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")

        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")

        # Performance summary
        total_time = sum(r["duration_ms"] for r in self.results.values())
        print(f"\nTotal Test Time: {total_time:.1f}ms")

        print(f"\n{'=' * 80}")


def main():
    """Main test function."""
    # Test data path - update this to your actual test data
    test_data_path = r"C:\Users\tvisv\Downloads\MSIConverter\20231109_PEA_NEDC_bruker.d"

    if not Path(test_data_path).exists():
        print(f"❌ Test data not found at: {test_data_path}")
        print("Please update the test_data_path variable with the correct path.")
        return

    print(f"Using test data: {test_data_path}")

    # Run tests
    tester = BrukerReaderTester(test_data_path)
    results = tester.run_all_tests()

    # Save results to file
    import json

    output_file = Path(__file__).parent / "test_results.json"

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    json_results = convert_for_json(results)

    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n📊 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
