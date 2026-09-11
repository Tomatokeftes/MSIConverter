"""
Integration tests for the command-line interface.
"""

import sys

import pytest

from thyra.__main__ import main


class TestCommandLineInterface:
    """Test the command-line interface."""

    def test_cli_help(self, capsys):
        """Test the help output."""
        # Set up command line arguments
        sys.argv = ["thyra", "--help"]

        # Run main with exit handling
        with pytest.raises(SystemExit) as e:
            main()

        # Check exit code
        assert e.value.code == 0

        # Get captured output
        captured = capsys.readouterr()

        # Check help content
        assert "Convert MSI data to SpatialData format" in captured.out
        assert "--dataset-id" in captured.out
        assert "--pixel-size" in captured.out
        assert "--handle-3d" in captured.out

        # --optimize-chunks is a deprecated no-op: still accepted so existing
        # scripts keep running, but no longer advertised.
        assert "--optimize-chunks" not in captured.out

        # --format has one legal value, equal to its default, so no invocation
        # can change what it selects: still accepted so existing scripts keep
        # running, but no longer advertised.
        assert "--format" not in captured.out

    def test_cli_convert(self, create_minimal_imzml, temp_dir, monkeypatch):
        """Test basic CLI conversion.

        The ``--format spatialdata`` below is deliberate, not leftover: it is
        the "still accepted" half of the pair whose other half is
        ``test_cli_help``'s "no longer advertised". Do not drop it.
        """
        # Get test data
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "cli_output.h5ad"

        # Use monkeypatch to simulate command line arguments
        sys.argv = [
            "thyra",
            str(imzml_path),
            str(output_path),
            "--format",
            "spatialdata",
            "--dataset-id",
            "cli_test",
            "--pixel-size",
            "3.5",
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
        sys.argv = ["thyra", "input.imzML"]

        # Run main with exit handling
        with pytest.raises(SystemExit):
            main()

        # Get captured output
        captured = capsys.readouterr()

        # Check error content
        assert "error" in captured.err.lower()

    def test_cli_rejects_a_format_it_does_not_write(
        self, create_minimal_imzml, temp_dir
    ):
        """An unknown ``--format`` value fails before any data is read.

        ``--format`` is hidden, but its ``click.Choice`` is what keeps an
        unknown value a usage error (exit 2) rather than something accepted
        and quietly ignored. This is the only test guarding that Choice, so
        it stays even though the flag itself no longer selects anything.
        """
        # Get test data
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "invalid_format.h5ad"

        # Set up command line arguments with invalid format
        sys.argv = [
            "thyra",
            str(imzml_path),
            str(output_path),
            "--format",
            "invalid_format",
        ]

        # Run main with exit handling
        with pytest.raises(SystemExit) as e:
            main()

        # Click's own usage error, raised before the reader opens anything
        assert e.value.code == 2

        # Check output file does not exist
        assert not output_path.exists()

    def test_cli_handle_3d(self, create_minimal_imzml, temp_dir, monkeypatch):
        """Test CLI with 3D handling option."""
        # Get test data
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "cli_3d_output.h5ad"

        # Mock the convert_msi function to capture args
        handle_3d_value = None

        def mock_convert_msi(input_path, output_path, **kwargs):
            nonlocal handle_3d_value
            handle_3d_value = kwargs.get("handle_3d", False)
            return True

        monkeypatch.setattr("thyra.__main__.convert_msi", mock_convert_msi)

        # Set up command line arguments with 3D handling
        sys.argv = [
            "thyra",
            str(imzml_path),
            str(output_path),
            "--handle-3d",
            "--pixel-size",
            "1.0",
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

        # Mock the setup_logging function
        configured_level = None

        def mock_setup_logging(log_level=None, log_file=None):
            nonlocal configured_level
            configured_level = log_level

        monkeypatch.setattr("thyra.__main__.setup_logging", mock_setup_logging)

        # Mock convert_msi to always return True
        monkeypatch.setattr("thyra.__main__.convert_msi", lambda *args, **kwargs: True)

        # Set up command line arguments with debug log level
        sys.argv = [
            "thyra",
            str(imzml_path),
            str(output_path),
            "--log-level",
            "DEBUG",
            "--pixel-size",
            "1.0",
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
