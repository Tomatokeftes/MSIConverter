# tests/unit/utils/test_zarr_atomic_write.py
"""Tests for the bounded retry around Zarr's atomic metadata writes.

The failure being guarded against is Windows-only, but the retry logic
itself is plain Python and is tested on every platform by injecting the
errors rather than racing for them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from thyra.utils import zarr_atomic_write
from thyra.utils.zarr_atomic_write import (
    _is_transient,
    _move_with_retry,
    install_windows_atomic_write_retry,
)


def _unwrapped_stub(*_args, **_kwargs):
    """Stand-in for an unpatched zarr._atomic_write."""
    raise AssertionError("should not be called")


def _oserror(winerror: int | None) -> OSError:
    """Build an OSError shaped like the one MoveFileEx failures produce."""
    error = OSError("Access is denied")
    if winerror is not None:
        error.winerror = winerror  # type: ignore[attr-defined]
    return error


class TestIsTransient:
    """Only a busy destination counts as retryable."""

    @pytest.mark.parametrize("winerror", [5, 32])
    def test_busy_destination_is_transient(self, winerror):
        assert _is_transient(_oserror(winerror)) is True

    @pytest.mark.parametrize("winerror", [2, 3, 13, 183, None])
    def test_other_errors_are_not_transient(self, winerror):
        assert _is_transient(_oserror(winerror)) is False


class TestMoveWithRetry:
    """A move blocked by a transient lock must be retried, then give up."""

    def test_succeeds_without_retry_when_the_move_works(self):
        calls = []

        _move_with_retry(Path("src"), Path("dst"), lambda s, d: calls.append((s, d)))

        assert calls == [(Path("src"), Path("dst"))]

    @pytest.mark.parametrize("failures_before_success", [1, 2, 3])
    def test_recovers_after_transient_failures(self, failures_before_success):
        attempts = {"n": 0}

        def flaky(src, dst):
            attempts["n"] += 1
            if attempts["n"] <= failures_before_success:
                raise _oserror(5)

        _move_with_retry(Path("src"), Path("dst"), flaky)

        assert attempts["n"] == failures_before_success + 1

    def test_gives_up_after_a_bounded_number_of_attempts(self):
        attempts = {"n": 0}

        def always_locked(src, dst):
            attempts["n"] += 1
            raise _oserror(5)

        with pytest.raises(OSError) as excinfo:
            _move_with_retry(Path("src"), Path("dst"), always_locked)

        assert attempts["n"] == len(zarr_atomic_write._RETRY_DELAYS)
        assert getattr(excinfo.value, "winerror", None) == 5

    def test_non_transient_errors_are_not_retried(self):
        attempts = {"n": 0}

        def missing(src, dst):
            attempts["n"] += 1
            raise _oserror(2)  # ERROR_FILE_NOT_FOUND

        with pytest.raises(OSError):
            _move_with_retry(Path("src"), Path("dst"), missing)

        assert attempts["n"] == 1, "a real error must surface immediately"

    def test_retry_delays_are_bounded(self):
        """Retries must not stall a conversion for long."""
        assert sum(zarr_atomic_write._RETRY_DELAYS) < 1.0


@pytest.fixture
def pristine_zarr(monkeypatch):
    """Restore zarr's real _atomic_write, and undo any install afterwards.

    Earlier tests (or an earlier conversion in the same session) may have
    already installed the shim, so capture and restore the exact object
    rather than assuming it is unpatched.
    """
    import zarr.storage._local as zarr_local

    saved = zarr_local._atomic_write
    monkeypatch.setattr(zarr_atomic_write, "_installed", False, raising=False)
    yield zarr_local
    zarr_local._atomic_write = saved
    zarr_atomic_write._installed = False


class TestInstall:
    """Installation must be idempotent and confined to Windows."""

    def test_is_a_noop_off_windows(self, pristine_zarr, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        pristine_zarr._atomic_write = _unwrapped_stub
        install_windows_atomic_write_retry()

        assert pristine_zarr._atomic_write is _unwrapped_stub

    def test_patches_zarr_on_windows(self, pristine_zarr, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        pristine_zarr._atomic_write = _unwrapped_stub

        install_windows_atomic_write_retry()

        assert (
            getattr(pristine_zarr._atomic_write, "_thyra_retry_wrapped", False) is True
        )

    def test_repeated_installs_do_not_stack_wrappers(self, pristine_zarr, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        pristine_zarr._atomic_write = _unwrapped_stub

        install_windows_atomic_write_retry()
        first = pristine_zarr._atomic_write
        zarr_atomic_write._installed = False
        install_windows_atomic_write_retry()

        assert pristine_zarr._atomic_write is first

    def test_an_already_wrapped_zarr_is_left_alone(self, pristine_zarr, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        already = zarr_atomic_write.make_atomic_write_with_retry(lambda s, d: None)
        pristine_zarr._atomic_write = already

        install_windows_atomic_write_retry()

        assert pristine_zarr._atomic_write is already

    def test_missing_zarr_internals_leaves_zarr_alone(self, pristine_zarr, monkeypatch):
        """A future Zarr that reworks _atomic_write must not be patched."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delattr(pristine_zarr, "_atomic_write", raising=False)

        install_windows_atomic_write_retry()

        assert not hasattr(pristine_zarr, "_atomic_write")


class TestPatchedWriteBehaviour:
    """The replacement must behave like Zarr's own atomic write."""

    @pytest.fixture
    def real_move(self):
        """Zarr's own non-clobbering move, for the exclusive path."""
        import zarr.storage._local as zarr_local

        return zarr_local._safe_move

    @pytest.fixture
    def patched(self, real_move):
        return zarr_atomic_write.make_atomic_write_with_retry(real_move)

    def test_writes_content_to_the_destination(self, patched, tmp_path):
        target = tmp_path / "zarr.json"

        with patched(target, "wb") as f:
            f.write(b'{"zarr_format":3}')

        assert target.read_bytes() == b'{"zarr_format":3}'

    def test_replaces_existing_content(self, patched, tmp_path):
        target = tmp_path / "zarr.json"
        target.write_bytes(b"stale")

        with patched(target, "wb") as f:
            f.write(b"fresh")

        assert target.read_bytes() == b"fresh"

    def test_leaves_no_partial_file_behind(self, patched, tmp_path):
        target = tmp_path / "zarr.json"

        with patched(target, "wb") as f:
            f.write(b"{}")

        assert list(tmp_path.glob("*.partial")) == []

    def test_partial_file_is_cleaned_up_when_the_body_raises(self, patched, tmp_path):
        target = tmp_path / "zarr.json"

        with pytest.raises(RuntimeError):
            with patched(target, "wb") as f:
                f.write(b"half")
                raise RuntimeError("writer blew up")

        assert list(tmp_path.glob("*.partial")) == []
        assert not target.exists()

    def test_exclusive_writes_use_the_non_clobbering_move(self, patched, tmp_path):
        target = tmp_path / "zarr.json"

        with patched(target, "wb", exclusive=True) as f:
            f.write(b"fresh")

        assert target.read_bytes() == b"fresh"

    def test_exclusive_moves_also_retry(self, real_move, tmp_path):
        """Zarr's non-clobbering move must get the same treatment."""
        state = {"blocked": 1}

        def flaky_safe_move(src, dst):
            if state["blocked"]:
                state["blocked"] -= 1
                raise _oserror(5)
            return real_move(src, dst)

        write = zarr_atomic_write.make_atomic_write_with_retry(flaky_safe_move)
        target = tmp_path / "zarr.json"

        with write(target, "wb", exclusive=True) as f:
            f.write(b"fresh")

        assert target.read_bytes() == b"fresh"
        assert state["blocked"] == 0, "the flaky move was never exercised"

    def test_recovers_from_a_transient_lock_on_the_destination(
        self, patched, tmp_path, monkeypatch
    ):
        """The end-to-end path: a blocked rename must still land."""
        target = tmp_path / "zarr.json"
        target.write_bytes(b"stale")

        real_replace = Path.replace
        state = {"blocked": 1}

        def flaky_replace(self, dst):
            if state["blocked"]:
                state["blocked"] -= 1
                raise _oserror(5)
            return real_replace(self, dst)

        monkeypatch.setattr(Path, "replace", flaky_replace)

        with patched(target, "wb") as f:
            f.write(b"fresh")

        assert target.read_bytes() == b"fresh"
        assert list(tmp_path.glob("*.partial")) == []
        assert state["blocked"] == 0, "the flaky replace was never exercised"

    def test_a_permanently_locked_destination_still_raises(
        self, patched, tmp_path, monkeypatch
    ):
        target = tmp_path / "zarr.json"

        def always_locked(self, dst):
            raise _oserror(5)

        monkeypatch.setattr(Path, "replace", always_locked)

        with pytest.raises(OSError):
            with patched(target, "wb") as f:
                f.write(b"fresh")

        assert list(tmp_path.glob("*.partial")) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only behaviour")
class TestWiring:
    """The retry has to be in place by the time a converter writes."""

    def test_importing_thyra_does_not_patch_zarr(self):
        """A library should not reach into another library on import."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import zarr.storage._local as z; import thyra; "
                "print(getattr(z._atomic_write, '_thyra_retry_wrapped', False))",
            ],
            capture_output=True,
            text=True,
        )

        assert result.stdout.strip().endswith("False"), result.stdout

    def test_constructing_a_converter_installs_the_retry(
        self, create_minimal_imzml, temp_dir
    ):
        import zarr.storage._local as zarr_local

        from thyra.converters.spatialdata import SpatialDataConverter
        from thyra.readers.imzml import ImzMLReader

        imzml_path, _, _, _ = create_minimal_imzml
        reader = ImzMLReader(imzml_path)
        try:
            SpatialDataConverter(
                reader, temp_dir / "out.zarr", dataset_id="ds", pixel_size_um=2.5
            )
        finally:
            reader.close()

        assert getattr(zarr_local._atomic_write, "_thyra_retry_wrapped", False) is True
