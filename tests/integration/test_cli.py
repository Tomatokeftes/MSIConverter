"""
Integration tests for the command-line interface.
"""
import pytest
import sys
import subprocess
from pathlib import Path
import shutil

from msiconvert.__main__ import main


class TestCommandLineInterface:
    """Test the command-line interface."""
    
    def test_cli_help(self, capsys):
        """Test the help output."""
        # Set up command line arguments
        sys.argv = ["msiconvert", "--help"]
        
        # Run main with exit handling
        with pytest.raises(SystemExit) as e:
            main()
        
        # Check exit code
        assert e.value.code == 0
        
        # Get captured output
        captured = capsys.readouterr()
        
        # Check help content
        assert "Convert MSI data to SpatialData format" in captured.out
        assert "--format" in captured.out
        assert "--dataset-id" in captured.out
        assert "--pixel-size" in captured.out
        assert "--handle-3d" in captured.out
        assert "--optimize-chunks" in captured.out
    
    def test_cli_convert(self, create_minimal_imzml, temp_dir, monkeypatch):
        """Test basic CLI conversion."""
        # Get test data
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "cli_output.h5ad"
        
        # Use monkeypatch to simulate command line arguments
        sys.argv = [
            "msiconvert",
            str(imzml_path),
            str(output_path),
            "--format", "spatialdata",
            "--dataset-id", "cli_test",
            "--pixel-size", "3.5"
        ]
        
        # Run main with exit handling
        try:
            main()
            code = 0
        except SystemExit as e:
            code = e.code
        
        # Check results
        assert code == 0
        assert output_path.exists()
    
    def test_cli_missing_args(self, capsys):
        """Test CLI behavior with missing arguments."""
        # Set up command line arguments with missing output
        sys.argv = ["msiconvert", "input.imzML"]
        
        # Run main with exit handling
        with pytest.raises(SystemExit):
            main()
        
        # Get captured output
        captured = capsys.readouterr()
        
        # Check error content
        assert "error" in captured.err.lower()
    
    def test_cli_invalid_format(self, create_minimal_imzml, temp_dir):
        """Test CLI behavior with invalid format."""
        # Get test data
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "invalid_format.h5ad"
        
        # Set up command line arguments with invalid format
        sys.argv = [
            "msiconvert",
            str(imzml_path),
            str(output_path),
            "--format", "invalid_format"
        ]
        
        # Run main with exit handling
        with pytest.raises(SystemExit):
            main()
        
        # Check output file does not exist
        assert not output_path.exists()
    
    def test_cli_handle_3d(self, create_minimal_imzml, temp_dir, monkeypatch):
        """Test CLI with 3D handling option."""
        # Get test data
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "cli_3d_output.h5ad"
        
        # Mock the convert_msi function to capture args
        handle_3d_value = None
        original_convert_msi = __import__("msiconvert").convert.convert_msi
        
        def mock_convert_msi(input_path, output_path, **kwargs):
            nonlocal handle_3d_value
            handle_3d_value = kwargs.get("handle_3d", False)
            return True
            
        monkeypatch.setattr("msiconvert.__main__.convert_msi", mock_convert_msi)
        
        # Set up command line arguments with 3D handling
        sys.argv = [
            "msiconvert",
            str(imzml_path),
            str(output_path),
            "--handle-3d"
        ]
        
        # Run main
        try:
            main()
            code = 0
        except SystemExit as e:
            code = e.code
        
        # Check results
        assert code == 0
        assert handle_3d_value is True
    
    
    
    def test_cli_log_level(self, create_minimal_imzml, temp_dir, monkeypatch):
        """Test CLI with different log levels."""
        # Get test data
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "cli_log_test.h5ad"
        
        # Mock the logging.basicConfig function
        configured_level = None
        
        def mock_basic_config(**kwargs):
            nonlocal configured_level
            configured_level = kwargs.get("level")
            
        monkeypatch.setattr("logging.basicConfig", mock_basic_config)
        
        # Mock convert_msi to always return True
        monkeypatch.setattr("msiconvert.__main__.convert_msi", lambda *args, **kwargs: True)
        
        # Set up command line arguments with debug log level
        sys.argv = [
            "msiconvert",
            str(imzml_path),
            str(output_path),
            "--log-level", "DEBUG"
        ]
        
        # Run main
        try:
            main()
            code = 0
        except SystemExit as e:
            code = e.code
        
        # Check results
        assert code == 0
        assert configured_level == __import__("logging").DEBUG