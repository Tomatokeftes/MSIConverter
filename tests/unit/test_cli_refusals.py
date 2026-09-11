"""A refusal Thyra planned for reads as a refusal, not as a crash.

``convert_msi`` and ``BaseMSIConverter.convert`` both wrapped everything
in one ``except Exception`` that logged the message *and* a full
traceback at ERROR, so every carefully written ``ValueError`` reached the
user as roughly thirty lines that read like a crash (issue #234). The
front-door checks that used to fail late -- an output path whose parent
is a file (#255), a dataset id SpatialData cannot name an element with
and a mass range with no extent (#250) -- are here too, because the
defect in each was when and how the refusal arrived.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from thyra.__main__ import _validate_basic_params, _validate_output_path, main
from thyra.convert import convert_msi, dataset_id_problem
from thyra.errors import ConversionRefused

_CONVERTER_MODULE = "thyra.converters.spatialdata.base_spatialdata_converter"


@pytest.fixture
def runner():
    return CliRunner()


class TestTheRefusalType:
    def test_a_refusal_is_still_a_value_error(self):
        """Everything that raises one used to raise ValueError; callers catch it."""
        with pytest.raises(ValueError):
            raise ConversionRefused("nope")


class TestPresentation:
    """The message once, at ERROR; the traceback only at DEBUG."""

    def _convert(self, tmp_path, monkeypatch, thyra_logs, exc):
        def boom(*_args, **_kwargs):
            raise exc

        monkeypatch.setattr("thyra.convert.detect_format", boom)
        with thyra_logs("thyra.convert", logging.DEBUG) as records:
            result = convert_msi(tmp_path, tmp_path / "out.zarr")
        return result, records

    def test_a_refusal_logs_its_message_without_a_traceback(
        self, tmp_path, monkeypatch, thyra_logs
    ):
        result, records = self._convert(
            tmp_path,
            monkeypatch,
            thyra_logs,
            ConversionRefused("this file is a spot run"),
        )
        assert result is False

        errors = [r for r in records if r.levelno == logging.ERROR]
        assert [r.getMessage() for r in errors] == ["this file is a spot run"]
        assert not any("Traceback" in r.getMessage() for r in errors)

    def test_the_traceback_is_still_available_at_debug(
        self, tmp_path, monkeypatch, thyra_logs
    ):
        _, records = self._convert(
            tmp_path, monkeypatch, thyra_logs, ConversionRefused("a spot run")
        )

        debug = [r.getMessage() for r in records if r.levelno == logging.DEBUG]
        assert any("Traceback" in message for message in debug)

    def test_an_unexpected_exception_keeps_its_traceback_at_error(
        self, tmp_path, monkeypatch, thyra_logs
    ):
        """The traceback is the explanation for anything nobody planned for."""
        result, records = self._convert(
            tmp_path, monkeypatch, thyra_logs, KeyError("frame")
        )
        assert result is False

        errors = [r.getMessage() for r in records if r.levelno == logging.ERROR]
        assert any("Traceback" in message for message in errors)

    def test_a_plain_value_error_is_not_treated_as_a_refusal(
        self, tmp_path, monkeypatch, thyra_logs
    ):
        """numpy raises ValueError too, and those really are surprises.

        ``np.min`` on an empty array raises ``ValueError`` with a message
        nobody wrote for a user ("zero-size array to reduction operation
        minimum which has no identity"), which is exactly why the
        refusals are marked by type rather than matched on ValueError.
        """
        _, records = self._convert(
            tmp_path, monkeypatch, thyra_logs, ValueError("zero-size array")
        )

        errors = [r.getMessage() for r in records if r.levelno == logging.ERROR]
        assert any("Traceback" in message for message in errors)


class TestAnUnregisteredOutputFormat:
    """A ``format_type`` nobody registered is a refusal all the way out.

    ``_create_converter`` used to reach the registry through a wrapper,
    ``_resolve_converter_class``, which caught the lookup's ``ValueError``
    and, for any name containing "spatialdata", logged five lines of
    zarr-upgrade advice and raised ``ConversionRefused("SpatialData
    converter unavailable")`` -- discarding the registry's own message,
    which names the format asked for and the formats there are. The
    wrapper is gone; ``MSIRegistry.get_converter_class`` raises
    ``ConversionRefused`` itself, so the convention survives the deletion
    and does so for every caller of the registry rather than for the one
    caller that remembered to map the error.

    Python-API only in practice -- the CLI's ``--format`` is a
    ``click.Choice`` with one value -- but a refusal is a refusal, and
    nothing asserted this.
    """

    def _convert(self, tmp_path, monkeypatch, thyra_logs, format_type):
        class _StubReader:
            """Enough reader for _create_converter to be reached."""

            def get_essential_metadata(self):
                return SimpleNamespace(pixel_size=(25.0, 25.0))

            def close(self):
                pass

        monkeypatch.setattr(
            "thyra.convert._create_reader",
            lambda *_args, **_kwargs: (_StubReader(), "imzml"),
        )
        source = tmp_path / "sample.imzML"
        source.write_bytes(b"")

        with thyra_logs("thyra.convert", logging.DEBUG) as records:
            result = convert_msi(
                source,
                tmp_path / "out.zarr",
                format_type=format_type,
                pixel_size_um=25.0,
            )
        return result, records

    @pytest.mark.parametrize("format_type", ["spatialdata2", "nonsense"])
    def test_it_is_one_line_at_error_naming_what_there_is(
        self, tmp_path, monkeypatch, thyra_logs, format_type
    ):
        """One ERROR record, no traceback in it, and it names the choices.

        "spatialdata2" is the interesting half of the parametrisation: it
        is the shape of name the deleted wrapper special-cased on a
        substring test, so it is the one that would regress first.
        """
        result, records = self._convert(tmp_path, monkeypatch, thyra_logs, format_type)
        assert result is False

        errors = [r.getMessage() for r in records if r.levelno == logging.ERROR]
        assert errors == [
            f"No converter for format '{format_type}'. Available: ['spatialdata']"
        ]

    def test_the_traceback_is_kept_for_debug(self, tmp_path, monkeypatch, thyra_logs):
        _, records = self._convert(tmp_path, monkeypatch, thyra_logs, "spatialdata2")

        debug = [r.getMessage() for r in records if r.levelno == logging.DEBUG]
        assert any("Traceback" in message for message in debug)


class TestOutputPathParent:
    """Issue #255: refuse before the reader is opened, naming the cause."""

    def test_a_parent_that_is_a_file_is_refused(self, tmp_path):
        blocker = tmp_path / "afile.txt"
        blocker.write_text("not a directory", encoding="utf-8")

        with pytest.raises(click.BadParameter) as excinfo:
            _validate_output_path(blocker / "out.zarr")

        message = str(excinfo.value)
        assert "afile.txt" in message and "is a file, not a directory" in message

    def test_an_ancestor_that_is_a_file_is_refused(self, tmp_path):
        blocker = tmp_path / "afile.txt"
        blocker.write_text("not a directory", encoding="utf-8")

        with pytest.raises(click.BadParameter):
            _validate_output_path(blocker / "deeper" / "out.zarr")

    def test_a_missing_parent_is_still_accepted(self, tmp_path):
        """Directories on the way are created at write time, as they were."""
        _validate_output_path(tmp_path / "not" / "there" / "yet" / "out.zarr")

    def test_an_ordinary_output_path_is_accepted(self, tmp_path):
        _validate_output_path(tmp_path / "out.zarr")

    def test_the_cli_refuses_it_before_opening_the_reader(
        self, create_minimal_imzml, tmp_path, monkeypatch, runner
    ):
        imzml_path, _, _, _ = create_minimal_imzml
        blocker = tmp_path / "afile.txt"
        blocker.write_text("not a directory", encoding="utf-8")

        def never(*_args, **_kwargs):  # pragma: no cover - must not run
            raise AssertionError("the conversion started")

        monkeypatch.setattr("thyra.__main__.convert_msi", never)
        result = runner.invoke(main, [str(imzml_path), str(blocker / "out.zarr")])

        assert result.exit_code == 2, result.output
        assert "is a file, not a directory" in result.output
        assert "Traceback" not in result.output


class TestDatasetId:
    """Issue #250: the id names every element, so SpatialData's rule applies."""

    @pytest.mark.parametrize("bad", ["has space", "a/b", "a\\b", ".", "..", "__x", ""])
    def test_an_unusable_id_is_named(self, bad):
        assert dataset_id_problem(bad) is not None

    @pytest.mark.parametrize("good", ["msi_dataset", "brain-1", "run.2", "cerveau"])
    def test_a_usable_id_passes(self, good):
        assert dataset_id_problem(good) is None

    def test_the_cli_refuses_before_any_pass(self):
        with pytest.raises(click.BadParameter) as excinfo:
            _validate_basic_params(None, "has space")
        assert "letters, digits" in str(excinfo.value)

    def test_the_api_refuses_it_too(self, tmp_path):
        assert convert_msi(tmp_path, tmp_path / "out.zarr", dataset_id="a/b") is False


def _plan_stub(min_mz, max_mz, target_bins):
    """A stand-in for the converter, carrying only what the plan reads."""
    from thyra.resampling.types import AxisType

    return SimpleNamespace(
        _essential_metadata_cached=SimpleNamespace(mass_range=(100.0, 1000.0)),
        _min_mz=min_mz,
        _max_mz=max_mz,
        _manual_axis_type=AxisType.CONSTANT,
        _width_at_mz=None,
        _target_bins=target_bins,
    )


class TestTheResamplingPlan:
    """Issue #250: the range and the bin count, checked before any pass."""

    def _resolve(self, stub):
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            BaseSpatialDataConverter,
        )

        return BaseSpatialDataConverter._resolve_resampling_plan(stub)

    def test_an_inverted_range_is_refused(self):
        """It used to build a descending axis, drop every peak against it,
        report "4 of 3 ... affected" from a negative count, and succeed."""
        with pytest.raises(ConversionRefused, match="empty"):
            self._resolve(_plan_stub(600.0, 300.0, 4000))

    def test_an_empty_range_is_refused(self):
        with pytest.raises(ConversionRefused, match="empty"):
            self._resolve(_plan_stub(500.0, 500.0, 4000))

    def test_one_bin_is_refused_rather_than_crashing(self):
        """``np.min(np.diff(axis))`` on a one-point axis is not a message."""
        with pytest.raises(ConversionRefused, match="at least 2 bins"):
            self._resolve(_plan_stub(None, None, 1))

    def test_two_bins_and_a_real_range_still_plan(self):
        min_mz, max_mz, _axis, bins = self._resolve(_plan_stub(300.0, 600.0, 2))
        assert (min_mz, max_mz, bins) == (300.0, 600.0, 2)


class TestUnknownResamplingConfigKeys:
    """Issue #250: a typo in the dict is warned about, not swallowed."""

    def test_an_unknown_key_is_named(self, thyra_logs):
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            _normalize_resampling_config,
        )

        with thyra_logs(_CONVERTER_MODULE, logging.WARNING) as records:
            _normalize_resampling_config(
                {"method": "nearest_neighbor", "target_bin": 10}
            )

        assert any("'target_bin'" in r.getMessage() for r in records)

    def test_the_keys_it_reads_are_quiet(self, thyra_logs):
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            _normalize_resampling_config,
        )

        with thyra_logs(_CONVERTER_MODULE, logging.WARNING) as records:
            _normalize_resampling_config(
                {
                    "method": "nearest_neighbor",
                    "axis_type": "constant",
                    "target_bins": 10,
                    "min_mz": None,
                    "max_mz": None,
                    "width_at_mz": None,
                    "reference_mz": 1000.0,
                    "gap_tolerance_da": None,
                    "tof_a": None,
                    "tof_b": None,
                    "bins_per_fwhm": None,
                }
            )

        assert not records
