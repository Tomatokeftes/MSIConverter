"""
Tests for the ontology checking tool.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Import the main function directly to avoid import issues
from msiconvert.tools.check_ontology import main


class TestCheckOntology:
    """Tests for the ontology checking tool."""

    def test_main_file_check(self, tmp_path, capsys):
        """Verify the CLI output for a single file check with no unknown terms."""
        # Create a dummy imzML file
        dummy_imzml = tmp_path / "test.imzML"
        dummy_imzml.write_text("""<mzML><cvList><cv id="MS" fullName="MS" version="1.0" uri="http://example.com/ms.obo"/></cvList><run><spectrumList count="0"/></run></mzML>""")

        # Mock at the correct import location (inside main function)
        with patch('msiconvert.metadata.validator.ImzMLOntologyValidator') as mock_validator_class, \
             patch('msiconvert.metadata.ontology.cache.ONTOLOGY') as mock_ontology:
            
            # Setup mocks
            mock_validator_instance = mock_validator_class.return_value
            mock_validator_instance.validate_file.return_value = {
                'summary': 'Test summary for file\n\nNo unknown terms encountered.',
                'unknown_terms': []
            }
            mock_ontology.report_unknown_terms.return_value = "No unknown terms encountered."

            # Set up command line arguments
            original_argv = sys.argv.copy()
            try:
                sys.argv = ["check_ontology", str(dummy_imzml)]

                # Run main
                main()

                # Assertions
                mock_validator_class.assert_called_once()
                mock_validator_instance.validate_file.assert_called_once_with(dummy_imzml)
                
                captured = capsys.readouterr()
                assert "Test summary for file" in captured.out
                assert "No unknown terms encountered." in captured.out
                
            finally:
                sys.argv = original_argv

    def test_main_directory_check(self, tmp_path, capsys):
        """Verify the CLI output for a directory check."""
        # Create a dummy directory and files
        dummy_dir = tmp_path / "test_dir"
        dummy_dir.mkdir()
        (dummy_dir / "file1.imzML").write_text("""<mzML></mzML>""")
        (dummy_dir / "file2.imzML").write_text("""<mzML></mzML>""")

        # Mock at the correct import location
        with patch('msiconvert.metadata.validator.ImzMLOntologyValidator') as mock_validator_class, \
             patch('msiconvert.metadata.ontology.cache.ONTOLOGY') as mock_ontology:
            
            # Setup mocks
            mock_validator_instance = mock_validator_class.return_value
            mock_validator_instance.validate_directory.return_value = {
                'files_checked': 2,
                'all_unknown_terms': {"term1", "term2"},
                'per_file_results': {}
            }
            mock_ontology.report_unknown_terms.return_value = "Found 2 unknown terms"

            # Set up command line arguments
            original_argv = sys.argv.copy()
            try:
                sys.argv = ["check_ontology", str(dummy_dir)]

                # Run main
                main()

                # Assertions
                mock_validator_class.assert_called_once()
                mock_validator_instance.validate_directory.assert_called_once_with(dummy_dir)
                
                captured = capsys.readouterr()
                assert "Checked 2 files" in captured.out
                assert "Found 2 unique unknown terms" in captured.out
                assert "term1" in captured.out
                assert "term2" in captured.out
                
            finally:
                sys.argv = original_argv

    def test_main_output_to_json(self, tmp_path, capsys):
        """Verify that results are correctly saved to a JSON file."""
        # Create test files
        output_json = tmp_path / "output.json"
        dummy_imzml = tmp_path / "test.imzML"
        dummy_imzml.write_text("""<mzML></mzML>""")

        # Mock at the correct import location
        with patch('msiconvert.metadata.validator.ImzMLOntologyValidator') as mock_validator_class, \
             patch('msiconvert.metadata.ontology.cache.ONTOLOGY') as mock_ontology:
            
            # Setup mocks
            mock_validator_instance = mock_validator_class.return_value
            validation_result = {
                'summary': 'Test summary for JSON',
                'unknown_terms': ["json_term"]
            }
            mock_validator_instance.validate_file.return_value = validation_result
            mock_ontology.report_unknown_terms.return_value = "No unknown terms encountered."

            # Set up command line arguments
            original_argv = sys.argv.copy()
            try:
                sys.argv = ["check_ontology", str(dummy_imzml), "--output", str(output_json)]

                # Run main
                main()

                # Assertions
                captured = capsys.readouterr()
                assert f"Results saved to {output_json}" in captured.out
                
                assert output_json.exists()
                with open(output_json, 'r') as f:
                    content = json.load(f)
                assert content == validation_result
                
            finally:
                sys.argv = original_argv

    def test_main_verbose_output(self, tmp_path, capsys):
        """Verify the verbose output includes logging setup."""
        # Create a dummy imzML file
        dummy_imzml = tmp_path / "test.imzML"
        dummy_imzml.write_text("""<mzML></mzML>""")

        # Mock at the correct import location
        with patch('msiconvert.metadata.validator.ImzMLOntologyValidator') as mock_validator_class, \
             patch('msiconvert.metadata.ontology.cache.ONTOLOGY') as mock_ontology, \
             patch('logging.basicConfig') as mock_logging:
            
            # Setup mocks
            mock_validator_instance = mock_validator_class.return_value
            mock_validator_instance.validate_file.return_value = {
                'summary': 'Verbose summary',
                'unknown_terms': []
            }
            mock_ontology.report_unknown_terms.return_value = "No unknown terms encountered."

            # Set up command line arguments
            original_argv = sys.argv.copy()
            try:
                sys.argv = ["check_ontology", str(dummy_imzml), "--verbose"]

                # Run main
                main()

                # Assertions - check that logging was configured
                mock_logging.assert_called_once()
                args, kwargs = mock_logging.call_args
                assert kwargs.get('level') == __import__('logging').INFO
                
                captured = capsys.readouterr()
                assert "Verbose summary" in captured.out
                
            finally:
                sys.argv = original_argv

    def test_main_help(self, capsys):
        """Test help output."""
        # Set up command line arguments
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["check_ontology", "--help"]

            # Run main with exit handling
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            # Check exit code
            assert exc_info.value.code == 0
            
            # Check help content
            captured = capsys.readouterr()
            assert "Check ontology terms in imzML files" in captured.out
            assert "--output" in captured.out
            assert "--verbose" in captured.out
            
        finally:
            sys.argv = original_argv