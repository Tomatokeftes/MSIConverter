"""
Tests for the simplified format registry system.
"""

import pytest

import thyra
from thyra.core.base_converter import BaseMSIConverter
from thyra.core.base_reader import BaseMSIReader
from thyra.core.registry import (
    _registry,
    detect_format,
    get_converter_class,
    get_reader_class,
    register_converter,
    register_reader,
)
from thyra.errors import ConversionRefused


class TestRegistry:
    """Test the registry functionality."""

    def setup_method(self):
        """Set up each test by clearing the registries."""
        # Store original registry values to restore later
        with _registry._lock:
            self.original_readers = _registry._readers.copy()
            self.original_converters = _registry._converters.copy()

            # Clear registries for testing
            _registry._readers.clear()
            _registry._converters.clear()

    def teardown_method(self):
        """Restore original registry values after each test."""
        with _registry._lock:
            _registry._readers.clear()
            _registry._readers.update(self.original_readers)

            _registry._converters.clear()
            _registry._converters.update(self.original_converters)

    def test_register_reader(self):
        """Test registering a reader class."""

        # Create a test reader class
        class TestReader(BaseMSIReader):
            def get_metadata(self):
                pass

            def get_dimensions(self):
                pass

            def get_common_mass_axis(self):
                pass

            def iter_spectra(self, batch_size=None):
                pass

            def close(self):
                pass

        # Register it
        register_reader("test_format")(TestReader)

        # Check if it was properly registered
        with _registry._lock:
            assert "test_format" in _registry._readers
            assert _registry._readers["test_format"] == TestReader

        # Test getting the reader class
        retrieved_class = get_reader_class("test_format")
        assert retrieved_class == TestReader

    def test_register_converter(self):
        """Test registering a converter class."""

        # Create a test converter class
        class TestConverter(BaseMSIConverter):
            def _create_data_structures(self):
                pass

            def _save_output(self, data_structures):
                pass

        # Register it
        register_converter("test_format")(TestConverter)

        # Check if it was properly registered
        with _registry._lock:
            assert "test_format" in _registry._converters
            assert _registry._converters["test_format"] == TestConverter

        # Test getting the converter class
        retrieved_class = get_converter_class("test_format")
        assert retrieved_class == TestConverter

    def test_detect_format_imzml(self, tmp_path):
        """Test ImzML format detection via extension."""
        # Create test files
        imzml_file = tmp_path / "test.imzml"
        ibd_file = tmp_path / "test.ibd"
        imzml_file.touch()
        ibd_file.touch()

        assert detect_format(imzml_file) == "imzml"

    def test_detect_format_bruker(self, tmp_path):
        """Test Bruker format detection via extension."""
        # Create test directory
        bruker_dir = tmp_path / "test.d"
        bruker_dir.mkdir()
        (bruker_dir / "analysis.tsf").touch()

        assert detect_format(bruker_dir) == "bruker"

    def test_detect_format_bruker_tdf(self, tmp_path):
        """Test Bruker format detection with .tdf file."""
        # Create test directory
        bruker_dir = tmp_path / "test.d"
        bruker_dir.mkdir()
        (bruker_dir / "analysis.tdf").touch()

        assert detect_format(bruker_dir) == "bruker"

    def test_unsupported_extension(self, tmp_path):
        """Test error for unsupported extension."""
        unknown_file = tmp_path / "test.xyz"
        unknown_file.touch()

        with pytest.raises(ValueError, match="Unsupported format"):
            detect_format(unknown_file)

    def test_shimadzu_imdx_recognised_but_in_development(self, tmp_path):
        """Shimadzu .imdx files get a guidance error, not 'unsupported'."""
        imdx_file = tmp_path / "test.imdx"
        imdx_file.touch()

        with pytest.raises(ValueError, match="in development"):
            detect_format(imdx_file)

    def test_shimadzu_kbd_recognised_but_in_development(self, tmp_path):
        """Shimadzu .kbd files get a guidance error, not 'unsupported'."""
        kbd_file = tmp_path / "test.kbd"
        kbd_file.touch()

        with pytest.raises(ValueError, match="export the dataset as imzML"):
            detect_format(kbd_file)

    def test_missing_ibd_file(self, tmp_path):
        """Test error for ImzML without .ibd file."""
        imzml_file = tmp_path / "test.imzml"
        imzml_file.touch()

        with pytest.raises(ValueError, match="requires corresponding .ibd file"):
            detect_format(imzml_file)

    def test_bruker_missing_analysis_files(self, tmp_path):
        """Test error for Bruker .d directory without analysis files."""
        bruker_dir = tmp_path / "test.d"
        bruker_dir.mkdir()

        with pytest.raises(ValueError, match="missing analysis files"):
            detect_format(bruker_dir)

    def test_bruker_not_directory(self, tmp_path):
        """Test error for .d file instead of directory."""
        fake_bruker = tmp_path / "test.d"
        fake_bruker.touch()  # Create as file, not directory

        with pytest.raises(ValueError, match="requires .d directory"):
            detect_format(fake_bruker)

    def test_nonexistent_path(self, tmp_path):
        """Test error for non-existent path."""
        nonexistent = tmp_path / "nonexistent.imzml"

        with pytest.raises(ValueError, match="Input path does not exist"):
            detect_format(nonexistent)

    def test_detect_format_waters(self, tmp_path):
        """Test Waters format detection via .raw directory with _FUNC*.DAT."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()
        (raw_dir / "_FUNC001.DAT").write_bytes(b"\x00" * 16)

        assert detect_format(raw_dir) == "waters"

    def test_raw_file_without_phi_magic_is_rejected(self, tmp_path):
        """A .raw file is a PHI candidate, so the error names both vendors.

        Waters .raw is a directory and PHI .raw is a file, so a plain file
        that lacks the PHI SOFH magic belongs to neither.
        """
        fake_raw = tmp_path / "test.raw"
        fake_raw.touch()

        with pytest.raises(ValueError, match="Unrecognised .raw file"):
            detect_format(fake_raw)

    def test_detect_format_phi(self, tmp_path):
        """Test PHI format detection via .raw file with SOFH magic."""
        phi_raw = tmp_path / "test.raw"
        phi_raw.write_bytes(b"SOFH\r\nImagePixels: 4\r\nEOFH\r\n")

        assert detect_format(phi_raw) == "phi"

    def test_waters_missing_func_files(self, tmp_path):
        """Test error for .raw directory without _FUNC*.DAT files."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        with pytest.raises(ValueError, match="_FUNC.*DAT"):
            detect_format(raw_dir)

    def test_detect_format_waters_generic_directory(self, tmp_path):
        """Test Waters detection from a generic directory (no .raw extension)."""
        some_dir = tmp_path / "my_data"
        some_dir.mkdir()
        (some_dir / "_FUNC001.DAT").write_bytes(b"\x00" * 16)

        assert detect_format(some_dir) == "waters"

    def test_get_nonexistent_reader(self):
        """A reader miss is a refusal, not a bare ValueError.

        Asserted as ``ConversionRefused`` rather than ``ValueError``
        because the type is what ``convert_msi`` dispatches on: the
        refusal handler prints the message once and keeps the traceback
        for DEBUG, the generic handler prints a traceback at ERROR. A
        ``ValueError`` assertion passes under either, so it could not tell
        the two apart -- and the registry is the one place every caller of
        a format name goes through, which is why the convention belongs
        here rather than in each caller's error mapping.
        """
        with pytest.raises(ConversionRefused, match="No reader for format"):
            get_reader_class("nonexistent_format")

    def test_get_nonexistent_converter(self):
        """A converter miss is a refusal too, for the same reason.

        ``thyra/convert.py`` used to route this lookup through a wrapper,
        ``_resolve_converter_class``, that caught the registry's error and
        re-raised ``ConversionRefused`` for any format name containing
        "spatialdata". The wrapper is gone and ``_create_converter`` calls
        ``get_converter_class`` directly, so this assertion is now the only
        thing holding the convention in place on that path.
        """
        with pytest.raises(ConversionRefused, match="No converter for format"):
            get_converter_class("nonexistent_format")


# The two below live outside TestRegistry on purpose: its setup_method
# clears _registry._converters and teardown_method restores it, so the same
# assertions inside the class would only ever test that fixture.
#
# Both were measured to pass against the pre-#310 code as well, and that is
# not a flaw in them: on an install where spatialdata imports, the flag they
# used to be gated behind was True and the behaviour was already correct.
# They state the invariant so a future reviewer can see it asserted
# somewhere; the test that actually separates the two trees is
# tests/unit/test_hard_dependency.py, which fails before the fix.


def test_spatialdata_converter_is_registered_on_import():
    """Importing thyra registers the one output format the docs describe.

    Registration used to sit behind ``if SPATIALDATA_AVAILABLE:``, a flag
    set by a try/except that swallowed the ImportError. When spatialdata
    could not be imported, ``import thyra`` still succeeded and this lookup
    raised "No converter for format 'spatialdata'. Available: []" -- an
    empty registry, naming neither the missing package nor the cause
    (issue #310). spatialdata is a hard dependency, so registration is now
    unconditional and a broken install fails at ``import thyra`` instead.
    """
    assert get_converter_class("spatialdata") is thyra.SpatialDataConverter


def test_spatialdata_converter_is_always_a_class():
    """``thyra.SpatialDataConverter`` is a class, never None.

    ``thyra/__init__.py`` carried an ``except ImportError`` branch rebinding
    this name to ``None``, and issue #282 item 3 described the name as "a
    class or None depending on installed extras". It was measured to be the
    class even with spatialdata blocked -- the branch was dead, because the
    swallow one layer down meant the import it guarded never raised. Both
    the branch and the flag are gone; the name has one type.
    """
    assert isinstance(thyra.SpatialDataConverter, type)
