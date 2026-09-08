"""The `thyra validate` / `thyra export-metaspace` subcommands."""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from thyra.metadata.schema import build_msi_metadata
from thyra.metadata.schema.cli import export_metaspace_command, validate_command
from thyra.utils.windows_paths import WINDOWS_MAX_PATH, to_extended_length_path


def _make_runner() -> CliRunner:
    """A runner whose results expose stderr on every supported click.

    click < 8.2 mixes stderr into output unless asked not to (and
    ``result.stderr`` raises); 8.2 removed the ``mix_stderr`` kwarg and
    always separates.  The project supports both (see the click pin in
    pyproject.toml).
    """
    try:
        return CliRunner(mix_stderr=False)  # type: ignore[call-arg]
    except TypeError:
        return CliRunner()


@pytest.fixture
def runner():
    return _make_runner()


def _write_doc(tmp_path, name="meta.json", mutate=None):
    doc = build_msi_metadata(
        None, pixel_size_um=(20.0, 20.0), source_format="imzml"
    ).to_uns_dict()
    if mutate:
        mutate(doc)
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class TestValidateCommand:
    def test_valid_document_exits_zero(self, runner, tmp_path):
        result = runner.invoke(validate_command, [str(_write_doc(tmp_path))])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_invalid_document_exits_one_and_names_the_field(self, runner, tmp_path):
        def corrupt(doc):
            doc["ms_analysis"]["pixel_size_um"]["x"] = -5.0

        path = _write_doc(tmp_path, mutate=corrupt)
        result = runner.invoke(validate_command, [str(path)])
        assert result.exit_code == 1
        assert "ms_analysis.pixel_size_um.x" in result.output

    def test_missing_input_exits_two(self, runner, tmp_path):
        result = runner.invoke(validate_command, [str(tmp_path / "nope.json")])
        assert result.exit_code == 2

    def test_merge_overlays_user_fields(self, runner, tmp_path):
        # An overlay can also break the document; that must be caught.
        path = _write_doc(tmp_path)
        overlay = tmp_path / "user.json"
        overlay.write_text(
            json.dumps({"ms_analysis": {"polarity": "sideways"}}),
            encoding="utf-8",
        )
        result = runner.invoke(validate_command, [str(path), "--merge", str(overlay)])
        assert result.exit_code == 1
        assert "polarity" in result.output

    def test_json_report_is_machine_readable(self, runner, tmp_path):
        path = _write_doc(tmp_path)
        result = runner.invoke(validate_command, [str(path), "--json"])
        assert result.exit_code == 0
        report = json.loads(result.stdout)
        assert report[path.name]["valid"] is True
        assert report[path.name]["issues"] == []


class TestExportMetaspaceCommand:
    def test_writes_the_submission_json(self, runner, tmp_path):
        path = _write_doc(tmp_path)
        output = tmp_path / "out.json"
        result = runner.invoke(export_metaspace_command, [str(path), "-o", str(output)])
        assert result.exit_code == 0
        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["Data_Type"] == "Imaging MS"
        assert document["MS_Analysis"]["Pixel_Size"] == {"Xaxis": 20, "Yaxis": 20}

    def test_default_output_lands_next_to_the_input(self, runner, tmp_path):
        path = _write_doc(tmp_path)
        result = runner.invoke(export_metaspace_command, [str(path)])
        assert result.exit_code == 0
        assert (tmp_path / "meta.metaspace.json").exists()

    def test_stdout_output(self, runner, tmp_path):
        path = _write_doc(tmp_path)
        result = runner.invoke(export_metaspace_command, [str(path), "-o", "-"])
        assert result.exit_code == 0
        assert json.loads(result.stdout)["Data_Type"] == "Imaging MS"

    def test_missing_required_fields_are_warned_on_stderr(self, runner, tmp_path):
        path = _write_doc(tmp_path)
        result = runner.invoke(export_metaspace_command, [str(path), "-o", "-"])
        assert result.exit_code == 0
        assert "Organism" in (result.stderr or "")

    def test_merge_completes_the_submission(self, runner, tmp_path):
        path = _write_doc(tmp_path)
        overlay = tmp_path / "user.json"
        overlay.write_text(
            json.dumps(
                {
                    "sample": {
                        "organism": "Mus musculus",
                        "organism_part": "liver",
                        "condition": "wildtype",
                    }
                }
            ),
            encoding="utf-8",
        )
        result = runner.invoke(
            export_metaspace_command,
            [str(path), "--merge", str(overlay), "-o", "-"],
        )
        assert result.exit_code == 0
        document = json.loads(result.stdout)
        assert document["Sample_Information"]["Organism"] == "Mus musculus"

    def test_non_conforming_document_is_refused(self, runner, tmp_path):
        def corrupt(doc):
            doc["ms_analysis"]["pixel_size_um"]["x"] = -5.0

        path = _write_doc(tmp_path, mutate=corrupt)
        result = runner.invoke(export_metaspace_command, [str(path), "-o", "-"])
        assert result.exit_code == 1


class TestDispatcher:
    def test_subcommands_are_dispatched(self, monkeypatch, capsys):
        from thyra.__main__ import cli

        monkeypatch.setattr("sys.argv", ["thyra", "validate", "--help"], raising=False)
        with pytest.raises(SystemExit) as excinfo:
            cli()
        assert excinfo.value.code == 0
        assert "Validate MSI metadata" in capsys.readouterr().out

    def test_conversion_interface_still_owns_bare_help(self, monkeypatch, capsys):
        from thyra.__main__ import cli

        monkeypatch.setattr("sys.argv", ["thyra", "--help"], raising=False)
        with pytest.raises(SystemExit) as excinfo:
            cli()
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "INPUT" in out and "OUTPUT" in out
        assert "export-metaspace" in out


class TestTheStorePathIsPreparedInsideTheCommand:
    """Issue #257: click's existence check ran before the path was prepared.

    The converter writes a store to a deep Windows path through an
    extended-length path, so ``os.path.exists`` on the plain spelling is
    False and ``click.Path(exists=True)`` refused a store the CLI itself
    had just written -- while ``validate_store`` on the prepared path
    validated it fine.
    """

    def test_a_missing_path_is_still_a_usage_error(self, runner, tmp_path):
        result = runner.invoke(validate_command, [str(tmp_path / "nope.zarr")])
        assert result.exit_code == 2

    def test_the_prepared_path_is_what_gets_read(self, tmp_path, monkeypatch):
        """Proven by preparation that redirects: the command must follow it."""
        from thyra.metadata.schema import cli as cli_module

        real = _write_doc(tmp_path, name="real.json")
        typed = tmp_path / "typed.json"

        monkeypatch.setattr(
            cli_module, "prepare_zarr_read_path", lambda path: real, raising=True
        )
        assert cli_module._resolve_store_path(typed) == real

    def test_export_prepares_it_too(self, runner, tmp_path, monkeypatch):
        from thyra.metadata.schema import cli as cli_module

        real = _write_doc(tmp_path, name="real.json")
        monkeypatch.setattr(
            cli_module, "prepare_zarr_read_path", lambda path: real, raising=True
        )

        result = runner.invoke(
            export_metaspace_command, [str(tmp_path / "typed.json"), "-o", "-"]
        )
        assert result.exit_code == 0, result.output


@pytest.mark.skipif(sys.platform != "win32", reason="the 260 character limit")
class TestAPathPastTheWindowsLimit:
    """The case the issue reproduced: a path click cannot even stat."""

    @pytest.fixture
    def deep_document(self):
        base = Path(tempfile.mkdtemp(prefix="thyra_deep_cli_"))
        try:
            deep = base
            while len(str(deep / "metadata.json")) <= WINDOWS_MAX_PATH:
                deep = deep / "nested_directory_segment"
                to_extended_length_path(deep).mkdir()
            document = deep / "metadata.json"
            to_extended_length_path(document).write_text(
                json.dumps(
                    build_msi_metadata(
                        None, pixel_size_um=(20.0, 20.0), source_format="imzml"
                    ).to_uns_dict()
                ),
                encoding="utf-8",
            )
            if document.exists():
                # A machine with LongPathsEnabled=1 -- the GitHub Windows
                # runner is one -- resolves the plain path fine, so the
                # case this reproduces cannot occur on it.
                pytest.skip("Windows long-path support is on; the limit does not apply")
            yield document
        finally:
            shutil.rmtree(to_extended_length_path(base), ignore_errors=True)

    def test_validate_reaches_it(self, runner, deep_document):
        result = runner.invoke(validate_command, [str(deep_document)])
        assert result.exit_code == 0, result.output
        assert "OK" in result.output
