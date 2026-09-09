# tests/unit/test_cli_exit_status.py
"""Tests for the exit status the ``thyra`` command reports to the shell.

A conversion that fails must not look like a success to a calling script
or CI job, and it must not leave an unreadable store sitting at the
destination path where it can be mistaken for a finished conversion.
"""

from __future__ import annotations

import importlib

import pytest
from click.testing import CliRunner

from thyra.__main__ import main


@pytest.fixture
def runner():
    return CliRunner()


def _invoke(runner, imzml_path, output_path, *extra):
    return runner.invoke(
        main,
        [str(imzml_path), str(output_path), "--pixel-size", "1.0", *extra],
    )


class TestExitStatus:
    """The process exit code must track conversion success."""

    def test_failed_conversion_exits_nonzero(
        self, create_minimal_imzml, temp_dir, monkeypatch, runner
    ):
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "out.zarr"

        monkeypatch.setattr("thyra.__main__.convert_msi", lambda *a, **k: False)

        result = _invoke(runner, imzml_path, output_path)

        assert result.exit_code != 0, (
            "a failed conversion must report a non-zero exit status; "
            f"got {result.exit_code}"
        )

    def test_successful_conversion_exits_zero(
        self, create_minimal_imzml, temp_dir, monkeypatch, runner
    ):
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "out.zarr"

        monkeypatch.setattr("thyra.__main__.convert_msi", lambda *a, **k: True)

        result = _invoke(runner, imzml_path, output_path)

        assert result.exit_code == 0, result.output

    def test_real_failing_conversion_exits_one(self, temp_dir, runner):
        """End-to-end: an undetectable input format must exit 1.

        ``convert_msi`` catches the format-detection error and returns
        False, so this exercises the real CLI path rather than a
        monkeypatched stub. Exit code 1 rather than click's 2 confirms the
        failure came from the conversion, not from argument parsing.
        """
        unknown_input = temp_dir / "not_an_msi_file.txt"
        unknown_input.write_text("not MSI data")
        output_path = temp_dir / "out.zarr"

        result = _invoke(runner, unknown_input, output_path)

        assert result.exit_code == 1, result.output


class TestPartialOutputQuarantine:
    """A failed conversion must not leave an unreadable store in place."""

    def test_partial_output_is_moved_aside(
        self, create_minimal_imzml, temp_dir, monkeypatch, runner
    ):
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "out.zarr"

        def fake_convert(*_args, **_kwargs):
            # Stand in for a conversion that dies part-way through the
            # zarr write, leaving an incomplete store behind.
            output_path.mkdir(parents=True)
            (output_path / "zarr.json").write_text("{}")
            return False

        monkeypatch.setattr("thyra.__main__.convert_msi", fake_convert)

        result = _invoke(runner, imzml_path, output_path)

        assert result.exit_code != 0
        assert not output_path.exists(), (
            "the destination path must be cleared so the partial store "
            "cannot be mistaken for a finished conversion"
        )
        assert (temp_dir / "out.zarr.failed").is_dir()

    def test_an_interrupt_is_quarantined_like_any_other_failure(
        self, create_minimal_imzml, temp_dir, monkeypatch, runner
    ):
        """Ctrl-C is a failed conversion, not a separate kind of exit.

        ``KeyboardInterrupt`` used to propagate past every handler: click
        printed ``Aborted!``, the partial store stayed where a finished one
        belongs, the retry was refused with "Output path already exists",
        and the CSC scratch memmaps -- 330 MB on a small TIMS set with
        ``--mobility-grid``, 18 GB on a whole slide -- were left behind
        (issue #245). Verified with a real CTRL_BREAK as well; this pins
        the exit path it has to reach.
        """
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "out.zarr"

        def interrupted(*_args, **_kwargs):
            output_path.mkdir(parents=True)
            (output_path / "zarr.json").write_text("{}")
            raise KeyboardInterrupt

        monkeypatch.setattr("thyra.__main__.convert_msi", interrupted)

        result = _invoke(runner, imzml_path, output_path)

        assert result.exit_code == 1, result.output
        assert not output_path.exists()
        assert (temp_dir / "out.zarr.failed").is_dir()

    def test_quarantine_does_not_overwrite_an_earlier_failure(
        self, create_minimal_imzml, temp_dir, monkeypatch, runner
    ):
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "out.zarr"
        (temp_dir / "out.zarr.failed").mkdir()

        def fake_convert(*_args, **_kwargs):
            output_path.mkdir(parents=True)
            return False

        monkeypatch.setattr("thyra.__main__.convert_msi", fake_convert)

        result = _invoke(runner, imzml_path, output_path)

        assert result.exit_code != 0
        assert not output_path.exists()
        assert (temp_dir / "out.zarr.failed2").is_dir()

    def test_nothing_to_quarantine_is_not_an_error(
        self, create_minimal_imzml, temp_dir, monkeypatch, runner
    ):
        """Failing before anything is written must still exit non-zero."""
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "out.zarr"

        monkeypatch.setattr("thyra.__main__.convert_msi", lambda *a, **k: False)

        result = _invoke(runner, imzml_path, output_path)

        assert result.exit_code != 0
        assert not output_path.exists()
        assert not (temp_dir / "out.zarr.failed").exists()
        assert result.exception is None or isinstance(result.exception, SystemExit)


class TestOptimizeChunksDeprecation:
    """``--optimize-chunks`` is a no-op that warns, and is no longer advertised.

    It never worked: the post-hoc pass it invoked was written for a dense 4-D
    image layout and could not read the sparse ``tables/<id>/X`` group the
    converter writes, so it failed on every conversion while the CLI still
    exited 0. The flag stays accepted so scripts passing it keep running.
    """

    def test_flag_is_still_accepted(
        self, create_minimal_imzml, temp_dir, monkeypatch, runner
    ):
        """A script passing the flag must not die on an unknown option."""
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "out.zarr"

        monkeypatch.setattr("thyra.__main__.convert_msi", lambda *a, **k: True)

        result = _invoke(runner, imzml_path, output_path, "--optimize-chunks")

        # click reports an unknown option as exit code 2.
        assert result.exit_code == 0, result.output

    def test_flag_warns_that_it_does_nothing(
        self, create_minimal_imzml, temp_dir, monkeypatch, runner
    ):
        """The user must be told the flag is dead.

        Asserted against the CLI's own output rather than ``caplog``:
        ``setup_logging`` sets ``propagate = False`` on the ``thyra`` logger and
        attaches a ``StreamHandler(sys.stdout)``, so records never reach the root
        handler ``caplog`` installs. Reading stdout also asserts on what the user
        actually sees.
        """
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "out.zarr"

        monkeypatch.setattr("thyra.__main__.convert_msi", lambda *a, **k: True)

        result = _invoke(runner, imzml_path, output_path, "--optimize-chunks")

        assert result.exit_code == 0, result.output
        assert "--optimize-chunks is deprecated" in result.output
        assert "WARNING" in result.output

    def test_flag_is_silent_when_not_passed(
        self, create_minimal_imzml, temp_dir, monkeypatch, runner
    ):
        """The warning must only fire for users who actually pass the flag."""
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "out.zarr"

        monkeypatch.setattr("thyra.__main__.convert_msi", lambda *a, **k: True)

        result = _invoke(runner, imzml_path, output_path)

        assert result.exit_code == 0, result.output
        assert "--optimize-chunks" not in result.output

    def test_flag_does_not_mask_a_failed_conversion(
        self, create_minimal_imzml, temp_dir, monkeypatch, runner
    ):
        """The deprecation branch must not swallow the non-zero exit."""
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "out.zarr"

        monkeypatch.setattr("thyra.__main__.convert_msi", lambda *a, **k: False)

        result = _invoke(runner, imzml_path, output_path, "--optimize-chunks")

        assert result.exit_code != 0, result.output

    def test_the_dead_helper_is_gone(self):
        """``optimize_zarr_chunks`` and its module must not come back."""
        import thyra.utils

        assert not hasattr(thyra.utils, "optimize_zarr_chunks")
        assert "optimize_zarr_chunks" not in thyra.utils.__all__

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("thyra.utils.data_processors")


class TestVersionOption:
    """``thyra --version`` must report the installed package version."""

    def test_version_flag_reports_package_version(self, runner):
        from thyra import __version__

        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_flag_does_not_require_arguments(self, runner):
        """--version must work without INPUT and OUTPUT."""
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "Missing argument" not in result.output
