# tests/unit/utils/test_windows_paths.py
"""Tests for extended-length output paths on Windows.

A Zarr store's deepest key sits far below the path the caller named, so an
output path well inside the 260 character Windows limit can still push an
internal key past it. The path rewriting is pure string work and is tested
on every platform; the end-to-end conversion is Windows-only.
"""

from __future__ import annotations

import ntpath
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from thyra.utils import windows_paths
from thyra.utils.windows_paths import (
    DEEPEST_KEY_RESERVE,
    WINDOWS_MAX_PATH,
    prepare_zarr_output_path,
    prepare_zarr_read_path,
    projected_deepest_key_length,
    to_extended_length_path,
)

EXTENDED_PREFIX = "\\\\?\\"


class TestPrepareZarrReadPath:
    """Reading a store back is subject to the same limit as writing it.

    A store written through an extended-length path is intact, but its deep
    keys are invisible to a plain read, which makes it look corrupt.
    """

    @pytest.fixture
    def on_windows(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(windows_paths, "_long_paths_enabled", lambda: False)

    @staticmethod
    def _store_with_key_length(monkeypatch, total: int, walked=None) -> Path:
        """A store whose longest key is ``total`` characters.

        The walk is faked rather than written to disk: creating a key past
        the limit is exactly the thing that needs the prefix, so a real file
        cannot be used to test the code that decides to apply it. The fake
        is handed the extended path and sizes its one key so that the plain
        spelling measures ``total``; ``walked`` collects the paths it saw.
        """
        store = Path("C:\\stores\\store.zarr")

        def walk(path):
            root = str(path)
            if walked is not None:
                walked.append(root)
            plain_root_length = len(root) - len(EXTENDED_PREFIX)
            return [(root, [], ["k" * max(1, total - plain_root_length - 1)])]

        monkeypatch.setattr(windows_paths.os, "walk", walk)
        return store

    def test_shallow_store_is_untouched(self, on_windows, monkeypatch):
        store = self._store_with_key_length(monkeypatch, 120)

        assert prepare_zarr_read_path(store) == store

    def test_deep_store_is_extended(self, on_windows, monkeypatch):
        store = self._store_with_key_length(monkeypatch, WINDOWS_MAX_PATH + 20)

        result = prepare_zarr_read_path(store)

        assert str(result).startswith(EXTENDED_PREFIX)
        assert str(result).endswith(str(store))

    def test_key_at_the_limit_is_untouched(self, on_windows, monkeypatch):
        store = self._store_with_key_length(monkeypatch, WINDOWS_MAX_PATH)

        assert prepare_zarr_read_path(store) == store

    def test_key_one_past_the_limit_is_extended(self, on_windows, monkeypatch):
        """The prefix on the walked path must not be counted as key length."""
        store = self._store_with_key_length(monkeypatch, WINDOWS_MAX_PATH + 1)

        assert str(prepare_zarr_read_path(store)).startswith(EXTENDED_PREFIX)

    def test_the_walk_goes_through_the_extended_path(self, on_windows, monkeypatch):
        """A plain walk cannot list a directory past the limit and skips it
        silently, so it would miss exactly the keys being measured."""
        walked: list[str] = []
        store = self._store_with_key_length(monkeypatch, 120, walked)

        prepare_zarr_read_path(store)

        assert walked and all(p.startswith(EXTENDED_PREFIX) for p in walked)

    def test_relative_path_is_resolved_before_extending(self, on_windows, monkeypatch):
        """The length that matters is the absolute one, and the prefix is
        only valid on an absolute path."""
        self._store_with_key_length(monkeypatch, WINDOWS_MAX_PATH + 20)

        result = prepare_zarr_read_path(Path("store.zarr"))

        text = str(result)
        assert text.startswith(EXTENDED_PREFIX)
        assert Path(text[len(EXTENDED_PREFIX) :]).is_absolute()

    def test_a_relative_path_that_fits_still_comes_back_absolute(
        self, on_windows, monkeypatch, tmp_path
    ):
        r"""Regression: a store that fits was still read through the
        caller's relative spelling.

        Windows measures a relative path as ``<cwd> + "\" + <spelling>``
        *before* collapsing the ``..`` segments, so a spelling that reaches
        an absolute path of 252 characters can be refused at 260. Returning
        ``store_path`` unchanged on this branch handed that spelling to Zarr,
        which reports the over-long keys as absent and drops them from
        ``array_keys()`` with only a ``UserWarning``. Observed on a real
        store: ``mobility_edges`` vanished while its siblings, identical in
        shape, dtype, codecs and shards, survived.
        """
        monkeypatch.setattr(windows_paths.os, "walk", lambda path: [])
        monkeypatch.chdir(tmp_path)

        result = prepare_zarr_read_path(Path("store.zarr"))

        assert result != Path("store.zarr")
        assert ntpath.isabs(str(result))
        assert result.name == "store.zarr"

    def test_a_relative_path_is_resolved_when_long_paths_are_enabled(
        self, monkeypatch, tmp_path
    ):
        """Long-path support only covers fully qualified paths, so it is no
        reason to hand a relative spelling back."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(windows_paths, "_long_paths_enabled", lambda: True)
        self._store_with_key_length(monkeypatch, WINDOWS_MAX_PATH + 20)
        monkeypatch.chdir(tmp_path)

        result = prepare_zarr_read_path(Path("store.zarr"))

        assert ntpath.isabs(str(result))
        assert not str(result).startswith(EXTENDED_PREFIX)

    def test_already_extended_path_is_returned_without_walking(
        self, on_windows, monkeypatch
    ):
        def refuse(path):
            raise AssertionError(f"walked {path}")

        monkeypatch.setattr(windows_paths.os, "walk", refuse)
        store = Path(EXTENDED_PREFIX + "C:\\stores\\store.zarr")

        assert prepare_zarr_read_path(store) == store

    def test_no_op_off_windows(self, monkeypatch):
        store = self._store_with_key_length(monkeypatch, WINDOWS_MAX_PATH + 20)
        monkeypatch.setattr(sys, "platform", "linux")

        assert prepare_zarr_read_path(store) == store

    def test_no_op_when_long_paths_are_enabled(self, monkeypatch):
        store = self._store_with_key_length(monkeypatch, WINDOWS_MAX_PATH + 20)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(windows_paths, "_long_paths_enabled", lambda: True)

        assert prepare_zarr_read_path(store) == store

    def test_missing_store_is_left_alone(self, on_windows, tmp_path):
        """os.walk yields nothing for a missing store, so there is nothing
        to protect and the caller should see its own path in the error."""
        missing = tmp_path / "does_not_exist.zarr"

        assert prepare_zarr_read_path(missing) == missing


class TestToExtendedLengthPath:
    """Rewriting must produce syntax Windows actually accepts."""

    def test_drive_path_gets_the_prefix(self):
        result = to_extended_length_path(Path("C:\\data\\out.zarr"))

        assert str(result) == "\\\\?\\C:\\data\\out.zarr"

    def test_already_extended_path_is_unchanged(self):
        original = Path("\\\\?\\C:\\data\\out.zarr")

        assert str(to_extended_length_path(original)) == str(original)

    def test_applying_twice_is_idempotent(self):
        once = to_extended_length_path(Path("C:\\data\\out.zarr"))
        twice = to_extended_length_path(once)

        assert str(twice) == str(once)

    def test_unc_share_uses_the_unc_spelling(self):
        """A bare prefix on a UNC path is invalid syntax."""
        result = to_extended_length_path(Path("\\\\server\\share\\out.zarr"))

        assert str(result) == "\\\\?\\UNC\\server\\share\\out.zarr"
        assert not str(result).startswith("\\\\?\\\\\\")


class TestProjectedLength:
    """The projection must account for the store's internal depth."""

    def test_includes_the_dataset_id_and_reserve(self):
        out = Path("C:\\data\\out.zarr")

        projected = projected_deepest_key_length(out, "msi_dataset")

        assert projected == len(str(out)) + 1 + len("msi_dataset") + DEEPEST_KEY_RESERVE

    def test_longer_dataset_id_projects_further(self):
        out = Path("C:\\data\\out.zarr")

        short = projected_deepest_key_length(out, "ds")
        long = projected_deepest_key_length(out, "a_much_longer_dataset_id")

        assert long > short

    def test_reserve_covers_the_observed_deepest_key(self):
        """The deepest key measured in a real conversion was 95 characters
        below the store root, excluding the dataset id."""
        assert DEEPEST_KEY_RESERVE >= 95


class TestPrepareZarrOutputPath:
    """The prefix must appear only when it is needed."""

    @pytest.fixture
    def on_windows(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(windows_paths, "_long_paths_enabled", lambda: False)

    def _long_path(self, dataset_id="msi_dataset"):
        """A path just long enough to need the prefix."""
        needed = WINDOWS_MAX_PATH - DEEPEST_KEY_RESERVE - len(dataset_id)
        return Path("C:\\" + "d" * (needed + 10))

    def _short_path(self):
        return Path("C:\\data\\out.zarr")

    def test_short_path_is_untouched_on_windows(self, on_windows):
        out = self._short_path()

        assert prepare_zarr_output_path(out, "msi_dataset") == out

    def test_long_path_is_extended_on_windows(self, on_windows):
        out = self._long_path()

        result = prepare_zarr_output_path(out, "msi_dataset")

        assert str(result).startswith(EXTENDED_PREFIX)
        assert str(result).endswith(str(out))

    def test_no_op_off_windows(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        out = self._long_path()

        assert prepare_zarr_output_path(out, "msi_dataset") == out

    def test_no_op_when_long_paths_are_enabled(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(windows_paths, "_long_paths_enabled", lambda: True)
        out = self._long_path()

        assert prepare_zarr_output_path(out, "msi_dataset") == out

    def test_a_relative_output_path_comes_back_absolute(
        self, on_windows, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)

        result = prepare_zarr_output_path(Path("out.zarr"), "msi_dataset")

        assert ntpath.isabs(str(result))
        assert result.name == "out.zarr"

    def test_a_relative_output_path_is_measured_after_resolving(
        self, on_windows, monkeypatch, tmp_path
    ):
        """``len("out.zarr")`` is the spelling, not the path.

        Projecting from the relative spelling understates the deepest key by
        the whole working directory, so a location that cannot hold the
        store is waved through and the write fails part-way with a
        ``FileNotFoundError`` naming a file the caller never chose.
        """
        deep = tmp_path / ("d" * 100)
        deep.mkdir()
        monkeypatch.chdir(deep)

        result = prepare_zarr_output_path(Path("out.zarr"), "msi_dataset")

        assert str(result).startswith(EXTENDED_PREFIX)

    def test_the_registry_is_only_consulted_when_needed(self, monkeypatch):
        """A short path must not pay for a registry read."""
        monkeypatch.setattr(sys, "platform", "win32")
        calls = {"n": 0}

        def counting():
            calls["n"] += 1
            return False

        monkeypatch.setattr(windows_paths, "_long_paths_enabled", counting)

        prepare_zarr_output_path(self._short_path(), "msi_dataset")

        assert calls["n"] == 0

    def test_a_longer_dataset_id_can_tip_a_path_over(self, on_windows):
        """The dataset id is part of the deepest key, so it matters."""
        dataset_id = "a" * 60
        needed = WINDOWS_MAX_PATH - DEEPEST_KEY_RESERVE
        out = Path("C:\\" + "d" * (needed - 30))

        with_short = prepare_zarr_output_path(out, "ds")
        with_long = prepare_zarr_output_path(out, dataset_id)

        assert not str(with_short).startswith(EXTENDED_PREFIX)
        assert str(with_long).startswith(EXTENDED_PREFIX)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path limit only")
class TestLongPathConversion:
    """End to end: a store deeper than MAX_PATH must still convert."""

    def test_conversion_succeeds_past_the_limit(self):
        from thyra.convert import convert_msi

        base = Path(tempfile.mkdtemp())
        try:
            deep = base
            while len(str(deep)) < 175:
                deep = deep / "nested_directory_segment"
                deep.mkdir(exist_ok=True)

            src = base / "src"
            src.mkdir()
            imzml = self._write_imzml(src)

            out = deep / "out.zarr"
            assert len(str(out)) > 165, "path must be long enough to matter"

            assert convert_msi(
                str(imzml), str(out), dataset_id="msi_dataset", pixel_size_um=2.5
            ), "a long output path must not fail the conversion"

            import spatialdata as sd

            sdata = sd.read_zarr(EXTENDED_PREFIX + str(out))
            assert "msi_dataset_z0" in sdata.tables
            assert len(sdata.images) >= 1
        finally:
            # rmtree itself trips the limit without the prefix.
            shutil.rmtree(EXTENDED_PREFIX + str(base), ignore_errors=True)

    @staticmethod
    def _write_imzml(directory: Path) -> Path:
        import numpy as np
        from pyimzml.ImzMLWriter import ImzMLWriter

        path = directory / "minimal.imzML"
        mzs = np.linspace(100, 1000, 50)
        with ImzMLWriter(str(path), mode="processed") as writer:
            for x, y, z in [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1)]:
                intensities = np.zeros_like(mzs)
                intensities[10] = 100.0 * x
                intensities[30] = 150.0 * y
                writer.addSpectrum(mzs, intensities, (x, y, z))
        return path
