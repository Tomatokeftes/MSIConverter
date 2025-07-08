import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from msiconvert.metadata.extractor import MetadataExtractor

class TestMetadataExtractor:
    def test_extract_imzml_metadata(self):
        mock_reader = MagicMock()
        mock_reader.get_metadata.return_value = {
            "acquisition_date": "2023-01-01",
            "instrument_type": "TOF",
            "frame_count": 100
        }
        mock_reader.analysis_directory = Path("/test/imzml/data")
        mock_reader.file_type = "imzml"

        extractor = MetadataExtractor(mock_reader)
        metadata = extractor.extract()

        assert metadata == {
            "acquisition_date": "2023-01-01",
            "instrument_type": "TOF",
            "frame_count": 100
        }
        mock_reader.get_metadata.assert_called_once()

    def test_extract_bruker_metadata(self):
        mock_reader = MagicMock()
        mock_reader.get_metadata.return_value = {
            "bruker_software_version": "1.0",
            "data_format": "TSF",
            "frame_count": 50
        }
        mock_reader.analysis_directory = Path("/test/bruker/data")
        mock_reader.file_type = "bruker"

        extractor = MetadataExtractor(mock_reader)
        metadata = extractor.extract()

        assert metadata == {
            "bruker_software_version": "1.0",
            "data_format": "TSF",
            "frame_count": 50
        }
        mock_reader.get_metadata.assert_called_once()

    def test_extract_empty_metadata(self):
        mock_reader = MagicMock()
        mock_reader.get_metadata.return_value = {}
        mock_reader.analysis_directory = Path("/test/empty/data")
        mock_reader.file_type = "unknown"

        extractor = MetadataExtractor(mock_reader)
        metadata = extractor.extract()

        assert metadata == {}
        mock_reader.get_metadata.assert_called_once()
