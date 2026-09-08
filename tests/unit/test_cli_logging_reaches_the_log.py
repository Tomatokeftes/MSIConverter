"""The CLI's own log lines reach the CLI's own handlers.

``thyra/__main__.py`` logged to ``getLogger(__name__)``, which under
``python -m thyra`` is ``"__main__"`` while ``setup_logging`` configures
``"thyra"`` with ``propagate=False``. "Conversion completed successfully"
appeared in 0 of 67 successful ``python -m thyra`` runs and never in
``--log-file`` (issue #258).

The console handler is here too: on a Windows console encoding in cp1252,
one character outside it turned every write of that line into a logging
traceback on stderr while the conversion itself was fine (issue #259).
"""

from __future__ import annotations

import io
import logging
import sys

import pytest

from thyra.utils.logging_config import _console_stream, setup_logging


class TestTheCliLoggerIsUnderThyra:
    def test_the_module_logger_is_configured_by_setup_logging(self):
        """Whatever the module is called, its logger has to be a "thyra" child."""
        import thyra.__main__ as cli

        assert cli.logger.name.startswith("thyra.")

    def test_its_records_reach_the_configured_handlers(self, tmp_path):
        import thyra.__main__ as cli

        log_file = tmp_path / "run.log"
        setup_logging(log_level=logging.INFO, log_file=str(log_file))
        try:
            cli.logger.info("Conversion completed successfully. Output stored at x")
        finally:
            for handler in logging.getLogger("thyra").handlers:
                handler.close()
            logging.getLogger("thyra").handlers.clear()

        assert "Conversion completed successfully" in log_file.read_text(
            encoding="utf-8"
        )


class _Cp1252Stream(io.TextIOWrapper):
    """A stdout that only speaks cp1252, like a redirected Windows console."""


class TestTheConsoleStream:
    def test_a_stream_that_cannot_encode_gets_replacement(self, monkeypatch):
        buffer = io.BytesIO()
        stream = _Cp1252Stream(buffer, encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", stream)

        console = _console_stream()
        console.write("path: uni 日本 ö\n")
        console.flush()

        assert console.errors == "replace"
        assert b"path: uni" in buffer.getvalue()

    def test_a_stream_without_reconfigure_is_left_alone(self, monkeypatch):
        class Bare:
            def write(self, _text):  # pragma: no cover - never called
                return 0

        bare = Bare()
        monkeypatch.setattr(sys, "stdout", bare)
        assert _console_stream() is bare

    def test_a_non_ascii_message_does_not_raise(self, monkeypatch):
        buffer = io.BytesIO()
        monkeypatch.setattr(
            sys, "stdout", _Cp1252Stream(buffer, encoding="cp1252", errors="strict")
        )
        setup_logging(log_level=logging.INFO)
        logger = logging.getLogger("thyra")
        try:
            logger.info("Successfully saved SpatialData to out/uni 日本 ö.zarr")
            for handler in logger.handlers:
                handler.flush()
        finally:
            logger.handlers.clear()

        written = buffer.getvalue()
        assert b"Successfully saved SpatialData" in written


class TestTheLogFile:
    def test_it_is_written_as_utf8(self, tmp_path):
        """A path the platform encoding cannot spell must not cost the line."""
        log_file = tmp_path / "run.log"
        setup_logging(log_level=logging.INFO, log_file=str(log_file))
        logger = logging.getLogger("thyra")
        try:
            logger.info("saved to 日本.zarr")
        finally:
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()

        assert "日本.zarr" in log_file.read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _restore_thyra_logger():
    logger = logging.getLogger("thyra")
    handlers = list(logger.handlers)
    propagate = logger.propagate
    level = logger.level
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers = handlers
    logger.propagate = propagate
    logger.setLevel(level)
