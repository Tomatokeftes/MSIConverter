"""The CSC scratch directories go, on every way out of a conversion.

Each table is an AnnData built over memmapped CSC arrays in a
``.thyra_*`` directory next to the output. Windows will not delete a
mapped file, so the directory survives for exactly as long as something
still references the table -- and two things kept one alive:

**A log handler that retains records** (issue #249). The finalize path
passed exception *objects* as logger arguments at 19 sites. A retained
record holds the exception, the exception holds its traceback, the
traceback's frames reach the AnnData, and the scratch cannot be removed.
pytest's ``caplog`` retains records; so does Ousia's per-session log
capture, which is the consumer that met it. A plain ``StreamHandler``
never showed it.

**An interrupt** (issue #245). ``KeyboardInterrupt`` propagated past
``except Exception``, so nothing released the tables and nothing ran the
CLI's quarantine step: a whole-slide mobility grid left 18 GB of scratch
and an unopenable store blocking the retry. Measured on
``TIMS-test-data/02_tiny_longramp_1465px`` with ``--mobility-grid``:
330 MB of scratch left behind, exit status 0xC000013A.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import pytest

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata.streaming_converter import (
    StreamingSpatialDataConverter,
)


class _Retaining(logging.Handler):
    """What pytest's caplog and a GUI's session log both do."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _config() -> MockMSIConfig:
    return MockMSIConfig(
        n_x=3, n_y=2, n_z=1, n_mz_bins=200, peaks_per_spectrum=(5, 9), seed=3
    )


def _converter(reader, output: Path) -> StreamingSpatialDataConverter:
    return StreamingSpatialDataConverter(
        reader=reader,
        output_path=output,
        dataset_id="m",
        pixel_size_um=10.0,
        include_optical=False,
    )


def _scratch(parent: Path) -> list[str]:
    return sorted(p.name for p in parent.iterdir() if p.name.startswith(".thyra_"))


@pytest.fixture
def retaining_handler():
    """A record-retaining handler on the ``thyra`` logger, not on the root.

    ``setup_logging`` sets ``propagate = False`` on that logger and it is
    process-global, so a handler on the root sees Thyra records only until
    some other test has invoked the CLI -- which passes alone and fails in
    the full suite. Same reason ``tests/conftest.py`` has ``thyra_logs``.
    """
    handler = _Retaining()
    logger = logging.getLogger("thyra")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def _reader_that_loses_its_metadata():
    """A reader that makes one finalize-path warning fire.

    ``build_uns_metadata`` catches this and logs it while the table's
    AnnData is a live local, which is the frame a retained exception
    reaches through ``tb_frame.f_back``.
    """
    reader = MockMSIReader(_config())

    def raising(*_args, **_kwargs):
        raise RuntimeError("simulated metadata failure")

    reader.get_comprehensive_metadata = raising
    return reader


class TestAFailedSave:
    def test_the_scratch_goes_even_with_a_retaining_handler(
        self, tmp_path, monkeypatch, retaining_handler
    ):
        import spatialdata

        def boom(self, *_args, **_kwargs):
            raise OSError("simulated write failure")

        monkeypatch.setattr(spatialdata.SpatialData, "write", boom)

        converter = _converter(_reader_that_loses_its_metadata(), tmp_path / "s.zarr")
        assert converter.convert() is False

        del converter
        gc.collect()
        assert _scratch(tmp_path) == []

    def test_the_warning_that_used_to_pin_it_still_reaches_the_log(
        self, tmp_path, monkeypatch, retaining_handler
    ):
        """Logging ``str(e)`` must not have silenced the message itself."""
        import spatialdata

        def boom(self, *_args, **_kwargs):
            raise OSError("simulated write failure")

        monkeypatch.setattr(spatialdata.SpatialData, "write", boom)

        _converter(_reader_that_loses_its_metadata(), tmp_path / "s.zarr").convert()

        assert any(
            "simulated metadata failure" in record.getMessage()
            for record in retaining_handler.records
        )

    def test_no_record_holds_an_exception_object(
        self, tmp_path, monkeypatch, retaining_handler
    ):
        """The rule, not just its effect: an object in ``args`` is the defect."""
        import spatialdata

        def boom(self, *_args, **_kwargs):
            raise OSError("simulated write failure")

        monkeypatch.setattr(spatialdata.SpatialData, "write", boom)

        _converter(_reader_that_loses_its_metadata(), tmp_path / "s.zarr").convert()

        offenders = [
            record.name
            for record in retaining_handler.records
            if record.name.startswith("thyra")
            and any(isinstance(arg, BaseException) for arg in (record.args or ()))
        ]
        assert offenders == []


class TestAnInterrupt:
    def test_it_is_reported_as_a_failed_conversion(self, tmp_path, monkeypatch):
        import spatialdata

        def interrupt(self, *_args, **_kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(spatialdata.SpatialData, "write", interrupt)

        assert _converter(MockMSIReader(_config()), tmp_path / "i.zarr").convert() is (
            False
        )

    def test_the_scratch_goes(self, tmp_path, monkeypatch, retaining_handler):
        import spatialdata

        def interrupt(self, *_args, **_kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(spatialdata.SpatialData, "write", interrupt)

        converter = _converter(MockMSIReader(_config()), tmp_path / "i.zarr")
        assert converter.convert() is False

        del converter
        gc.collect()
        assert _scratch(tmp_path) == []

    def test_the_reader_is_closed(self, tmp_path, monkeypatch):
        import spatialdata

        def interrupt(self, *_args, **_kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(spatialdata.SpatialData, "write", interrupt)

        reader = MockMSIReader(_config())
        closed = []
        reader.close = lambda: closed.append(True)

        _converter(reader, tmp_path / "i.zarr").convert()

        assert closed == [True]

    def test_an_interrupt_during_the_passes_is_the_same(self, tmp_path):
        """Not only the write: the scatter is where a long run spends its time."""

        class _InterruptingReader(MockMSIReader):
            def iter_spectra(self, batch_size=None):
                for n, spectrum in enumerate(super().iter_spectra(batch_size)):
                    if self._scattering and n == 2:
                        raise KeyboardInterrupt
                    yield spectrum

            _scattering = False

            def reset(self):
                super().reset()
                self._scattering = True

        converter = _converter(_InterruptingReader(_config()), tmp_path / "p.zarr")
        assert converter.convert() is False

        del converter
        gc.collect()
        assert _scratch(tmp_path) == []
