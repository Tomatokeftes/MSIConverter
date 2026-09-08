"""
Unit tests for Bruker calibration metadata reading functionality.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from thyra.readers.bruker.timstof.timstof_reader import BrukerReader


class TestCalibrationMetadataReading:
    """Test calibration metadata reading from calibration.sqlite."""

    def test_read_calibration_metadata_basic(self):
        """Test reading basic calibration metadata."""
        # Create a temporary calibration database
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test.d"
            data_path.mkdir()
            cal_db = data_path / "calibration.sqlite"

            # Create a minimal calibration database
            conn = sqlite3.connect(cal_db)
            cursor = conn.cursor()

            # Create CalibrationState table
            cursor.execute(
                """
                CREATE TABLE CalibrationState (
                    Id INTEGER PRIMARY KEY,
                    Key TEXT,
                    DateTime TEXT,
                    Source TEXT
                )
            """
            )

            # Insert a single calibration state
            cursor.execute(
                """
                INSERT INTO CalibrationState (Id, Key, DateTime, Source)
                VALUES (1, 'test-uuid-123', '2025-01-01T12:00:00.000+00:00', 'timsTOF')
            """
            )

            # Create CalibrationInfo table (empty for basic test)
            cursor.execute(
                """
                CREATE TABLE CalibrationInfo (
                    Id INTEGER PRIMARY KEY,
                    CalibrationState INTEGER,
                    KeyName TEXT,
                    Value TEXT
                )
            """
            )

            conn.commit()
            conn.close()

            # Create a mock BrukerReader with minimal setup
            with (
                patch.object(BrukerReader, "_validate_data_path"),
                patch.object(BrukerReader, "_detect_file_type"),
                patch.object(BrukerReader, "_setup_components"),
                patch.object(BrukerReader, "_initialize_sdk"),
                patch.object(BrukerReader, "_initialize_database"),
                patch.object(BrukerReader, "_preload_frame_num_peaks"),
            ):

                reader = BrukerReader.__new__(BrukerReader)
                reader.data_path = data_path
                reader.use_recalibrated_state = True  # Set required attribute

                # Call the method directly
                metadata = reader._read_calibration_metadata()

                # Verify metadata
                assert metadata is not None
                assert metadata["calibration_id"] == 1
                assert metadata["calibration_uuid"] == "test-uuid-123"
                assert (
                    metadata["calibration_datetime"] == "2025-01-01T12:00:00.000+00:00"
                )
                assert metadata["calibration_source"] == "timsTOF"
                assert metadata["num_calibration_versions"] == 1
                assert metadata["recalibrated"] is False
                assert metadata["original_calibration_datetime"] is None
                assert metadata["calibration_file_size"] > 0

    def test_read_calibration_metadata_recalibrated(self):
        """Test reading calibration metadata from recalibrated dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test.d"
            data_path.mkdir()
            cal_db = data_path / "calibration.sqlite"

            conn = sqlite3.connect(cal_db)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE CalibrationState (
                    Id INTEGER PRIMARY KEY,
                    Key TEXT,
                    DateTime TEXT,
                    Source TEXT
                )
            """
            )

            # Insert multiple calibration states (original + recalibrations)
            cursor.execute(
                """
                INSERT INTO CalibrationState (Id, Key, DateTime, Source) VALUES
                (1, 'original-uuid', '2025-01-01T10:00:00.000+00:00', 'timsTOF'),
                (2, 'recal-uuid-1', '2025-02-01T14:00:00.000+00:00', 'DataAnalysis'),
                (3, 'recal-uuid-2', '2025-03-01T16:00:00.000+00:00', 'DataAnalysis')
            """
            )

            cursor.execute(
                """
                CREATE TABLE CalibrationInfo (
                    Id INTEGER PRIMARY KEY,
                    CalibrationState INTEGER,
                    KeyName TEXT,
                    Value TEXT
                )
            """
            )

            # Add software version for active state
            cursor.execute(
                """
                INSERT INTO CalibrationInfo (CalibrationState, KeyName, Value)
                VALUES (3, 'CalibrationSoftwareVersion', '6.1')
            """
            )

            cursor.execute(
                """
                INSERT INTO CalibrationInfo (CalibrationState, KeyName, Value)
                VALUES (3, 'CalibrationUser', 'demo_user')
            """
            )

            conn.commit()
            conn.close()

            with (
                patch.object(BrukerReader, "_validate_data_path"),
                patch.object(BrukerReader, "_detect_file_type"),
                patch.object(BrukerReader, "_setup_components"),
                patch.object(BrukerReader, "_initialize_sdk"),
                patch.object(BrukerReader, "_initialize_database"),
                patch.object(BrukerReader, "_preload_frame_num_peaks"),
            ):

                reader = BrukerReader.__new__(BrukerReader)
                reader.data_path = data_path
                reader.use_recalibrated_state = True

                metadata = reader._read_calibration_metadata()

                # Verify it picked the active (highest ID) state
                assert metadata is not None
                assert metadata["calibration_id"] == 3
                assert metadata["calibration_uuid"] == "recal-uuid-2"
                assert metadata["calibration_source"] == "DataAnalysis"
                assert metadata["num_calibration_versions"] == 3
                assert metadata["recalibrated"] is True
                assert (
                    metadata["original_calibration_datetime"]
                    == "2025-01-01T10:00:00.000+00:00"
                )
                assert metadata["calibration_software_version"] == "6.1"
                assert metadata["calibration_user"] == "demo_user"

    def test_read_calibration_metadata_missing_file(self):
        """Test handling of missing calibration.sqlite file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test.d"
            data_path.mkdir()
            # Don't create calibration.sqlite

            with (
                patch.object(BrukerReader, "_validate_data_path"),
                patch.object(BrukerReader, "_detect_file_type"),
                patch.object(BrukerReader, "_setup_components"),
                patch.object(BrukerReader, "_initialize_sdk"),
                patch.object(BrukerReader, "_initialize_database"),
                patch.object(BrukerReader, "_preload_frame_num_peaks"),
            ):

                reader = BrukerReader.__new__(BrukerReader)
                reader.data_path = data_path

                metadata = reader._read_calibration_metadata()

                # Should return None gracefully
                assert metadata is None

    def test_read_calibration_metadata_corrupted_db(self):
        """Test handling of corrupted calibration database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test.d"
            data_path.mkdir()
            cal_db = data_path / "calibration.sqlite"

            # Create corrupted database (missing required tables)
            conn = sqlite3.connect(cal_db)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE DummyTable (id INTEGER)")
            conn.commit()
            conn.close()

            try:
                with (
                    patch.object(BrukerReader, "_validate_data_path"),
                    patch.object(BrukerReader, "_detect_file_type"),
                    patch.object(BrukerReader, "_setup_components"),
                    patch.object(BrukerReader, "_initialize_sdk"),
                    patch.object(BrukerReader, "_initialize_database"),
                    patch.object(BrukerReader, "_preload_frame_num_peaks"),
                ):

                    reader = BrukerReader.__new__(BrukerReader)
                    reader.data_path = data_path
                    reader.use_recalibrated_state = True

                    metadata = reader._read_calibration_metadata()

                    # Should return None on error
                    assert metadata is None
            finally:
                # Ensure the database connection is closed before cleanup
                import gc

                gc.collect()  # Force garbage collection to close any open connections


class TestCalibrationMetadataIntegration:
    """Test integration of calibration metadata with metadata extractor."""

    def test_calibration_metadata_passed_to_extractor(self):
        """Test that calibration metadata is passed to BrukerMetadataExtractor."""
        from thyra.metadata.extractors.bruker_extractor import BrukerMetadataExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test.d"
            data_path.mkdir()

            # Mock calibration metadata
            cal_metadata = {
                "calibration_id": 1,
                "calibration_uuid": "test-uuid",
                "calibration_datetime": "2025-01-01T12:00:00.000+00:00",
                "calibration_source": "timsTOF",
                "num_calibration_versions": 1,
                "recalibrated": False,
            }

            # Create mock connection
            mock_conn = MagicMock()

            # Create extractor with calibration metadata
            extractor = BrukerMetadataExtractor(
                mock_conn, data_path, calibration_metadata=cal_metadata
            )

            # Verify it stored the metadata
            assert extractor.calibration_metadata == cal_metadata

    def test_calibration_metadata_in_format_specific(self):
        """Test that calibration metadata appears in format_specific metadata."""
        from thyra.metadata.extractors.bruker_extractor import BrukerMetadataExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test.d"
            data_path.mkdir()

            # Create dummy analysis file
            analysis_file = data_path / "analysis.tsf"
            analysis_file.touch()

            cal_metadata = {
                "calibration_id": 1,
                "calibration_uuid": "test-uuid",
                "recalibrated": True,
            }

            mock_conn = MagicMock()
            # Mock the cursor to prevent errors in _is_maldi_dataset
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (0,)  # No MALDI data
            mock_conn.cursor.return_value = mock_cursor

            extractor = BrukerMetadataExtractor(
                mock_conn, data_path, calibration_metadata=cal_metadata
            )

            # Get format-specific metadata
            format_specific = extractor._extract_bruker_specific()

            # Verify calibration metadata is included
            assert "calibration" in format_specific
            assert format_specific["calibration"] == cal_metadata


class TestDefaultCalibrationBehavior:
    """Test default calibration state behavior."""

    def test_use_recalibrated_state_default_true(self):
        """Test that use_recalibrated_state defaults to True."""
        import inspect

        from thyra.readers.bruker.timstof.timstof_reader import BrukerReader

        # Get the __init__ signature
        sig = inspect.signature(BrukerReader.__init__)
        param = sig.parameters["use_recalibrated_state"]

        # Verify default is True
        assert param.default is True

    def test_use_recalibrated_state_can_be_overridden(self):
        """Test that use_recalibrated_state can be set to False."""

        def mock_detect(self):
            self.file_type = "tsf"

        with (
            patch.object(BrukerReader, "_validate_data_path"),
            patch.object(BrukerReader, "_detect_file_type", mock_detect),
            patch.object(BrukerReader, "_read_calibration_metadata", return_value=None),
            patch.object(BrukerReader, "_setup_components"),
            patch.object(BrukerReader, "_initialize_sdk"),
            patch.object(BrukerReader, "_initialize_database"),
            patch.object(BrukerReader, "_detect_regions", return_value=[]),
            patch.object(BrukerReader, "_select_region", return_value=(None, None)),
            patch.object(BrukerReader, "_parse_mis_alignment", return_value={}),
            patch.object(BrukerReader, "_build_positions_from_db", return_value=[]),
            patch.object(BrukerReader, "_build_header_alignment", return_value={}),
            patch.object(BrukerReader, "_preload_frame_num_peaks", return_value={}),
        ):

            reader = BrukerReader(Path("/fake/path.d"), use_recalibrated_state=False)

            assert reader.use_recalibrated_state is False


@pytest.mark.skipif(
    not Path("test_data/260225_SN_L10.d").exists(), reason="Test data not available"
)
class TestRealCalibrationData:
    """Tests using real calibration data (if available)."""

    def test_read_real_calibration_metadata(self):
        """Test reading calibration metadata from real dataset."""
        data_path = Path("test_data/260225_SN_L10.d")

        # Just test the metadata reading function directly
        with (
            patch.object(BrukerReader, "_validate_data_path"),
            patch.object(BrukerReader, "_detect_file_type"),
            patch.object(BrukerReader, "_setup_components"),
            patch.object(BrukerReader, "_initialize_sdk"),
            patch.object(BrukerReader, "_initialize_database"),
            patch.object(BrukerReader, "_preload_frame_num_peaks"),
        ):

            reader = BrukerReader.__new__(BrukerReader)
            reader.data_path = data_path
            reader.use_recalibrated_state = True

            metadata = reader._read_calibration_metadata()

            # Verify we got metadata
            assert metadata is not None
            assert "calibration_id" in metadata
            assert "calibration_uuid" in metadata
            assert "calibration_datetime" in metadata
            assert "calibration_source" in metadata
            assert "num_calibration_versions" in metadata
            assert "recalibrated" in metadata

            # This dataset is known to have 3 calibration states
            assert metadata["num_calibration_versions"] == 3
            assert metadata["recalibrated"] is True
            assert metadata["calibration_id"] == 3  # Active state
