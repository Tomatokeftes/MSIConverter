import logging
import logging.handlers
import sys

from ..config import LOG_BACKUP_COUNT, LOG_FILE_MAX_SIZE_MB, MB_TO_BYTES


def _console_stream():
    """``sys.stdout``, told to replace what it cannot encode.

    A log line carries whatever the source gave it -- an output path, a
    dataset id -- and on Windows a redirected stdout encodes in the ANSI
    code page. One character outside cp1252 turned every write of that
    line into a ``--- Logging error ---`` block and a ``UnicodeEncodeError``
    traceback on stderr, once per handler, while the conversion itself
    ran fine (issue #259).

    ``reconfigure`` changes the error handler of the stream already in
    place rather than wrapping it, so nothing else that writes to stdout
    (``click.echo``, a progress bar) ends up behind a second buffer. A
    stream without it -- pytest's capture object, a caller's own file --
    is left alone.
    """
    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return stream
    try:
        reconfigure(errors="replace")
    except (ValueError, OSError):  # pragma: no cover - stream-defined
        pass
    return stream


def setup_logging(log_level=logging.INFO, log_file=None):
    """Set up logging for the application.

    Args:
        log_level (int): The minimum logging level to display.
        log_file (str): Path to the log file. If None, logs are not saved to a file.
    """
    # Get the root logger. Everything Thyra logs sits under this name,
    # the CLI's own messages included -- see the note in __main__.py on
    # why they are logged to "thyra.cli" and not to __name__.
    logger = logging.getLogger("thyra")
    logger.setLevel(log_level)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    # Remove all existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler
    console_handler = logging.StreamHandler(_console_stream())
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a file handler if a log file is specified. UTF-8 rather than
    # the platform's preferred encoding for the same reason as above: a
    # log file is read later, often elsewhere, and a path it cannot spell
    # should not cost the lines around it.
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOG_FILE_MAX_SIZE_MB * MB_TO_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("Logging configured")
