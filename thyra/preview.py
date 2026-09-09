"""Metadata-only preview of an MSI input.

Used by the Ousia Import Wizard's step 2 to populate the per-sample
preview card (m/z range, grid dimensions, pixel size, instrument
guess, EscDat folder probe) without running any conversion.

Design constraints:

- No spectra are read; no zarr is written.  The function only invokes
  ``BaseMSIReader.get_essential_metadata`` and
  ``get_comprehensive_metadata``.
- The function never raises for "we couldn't read it".  On any
  exception it returns an :class:`MsiPreview` with
  ``readable=False`` and ``error`` set to the exception message,
  so the wizard can render an inline error in the per-sample card.
- Target runtime: <500 ms for inputs up to ~50 GB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .core.registry import detect_format, get_reader_class
from .resampling.decision_tree import ResamplingDecisionTree
from .resampling.types import AxisType, ResamplingMethod

logger = logging.getLogger(__name__)


@dataclass
class MsiPreview:
    """Metadata-only snapshot of an MSI input.

    Attributes:
        mz_range: ``(min_mz, max_mz)`` of the source mass axis, in Da.
            ``(0.0, 0.0)`` when ``readable=False``.
        n_pixels: Total number of spectra (pixels) in the dataset.
            ``0`` when ``readable=False``.
        grid_dims: Grid dimensions as ``(width, height)`` in pixels --
            i.e. ``(x, y)`` from :attr:`EssentialMetadata.dimensions`.
            ``(0, 0)`` when ``readable=False``.
        instrument_type: The :class:`AxisType` the
            :class:`ResamplingDecisionTree` picks for this input, or
            ``None`` if metadata could not be extracted.  The wizard
            uses this as a default in step 4; the user may always
            override.
        pixel_size_um: Pixel pitch in micrometres.  Thyra stores a
            ``(x_um, y_um)`` tuple internally; preview returns the
            ``x`` component and logs a debug message when ``x != y``
            (real MSI rasters are essentially always isotropic).
            ``None`` when the source metadata does not declare a
            pixel size.
        has_escdat_folder: ``True`` iff ``<path>/EscDat/`` exists and
            is a directory.  Used by step 3 to decide whether the
            EscDat-derived registration is available for this sample.
            For non-Bruker inputs this is virtually always ``False``,
            which is the intended behaviour.
        readable: ``True`` iff format detection succeeded, a reader
            was built, and ``get_essential_metadata`` returned
            without raising.
        error: The exception's ``str()`` when ``readable=False``;
            ``None`` otherwise.
        resampling_method: The :class:`ResamplingMethod` the
            :class:`ResamplingDecisionTree` picks for this input, or
            ``None`` if metadata could not be extracted.  Report this
            rather than re-deriving a method from
            :attr:`instrument_type`: the two are separate answers from
            the detector, and inferring one from the other gets it
            wrong wherever a detector pairs them unconventionally.  A
            caller that assumed ``constant`` implied profile MALDI --
            and so ``tic_preserving`` -- reached exactly that bug on
            PHI ToF-SIMS, which the catch-all default reports as
            ``constant`` while asking for ``nearest_neighbor``.
    """

    mz_range: Tuple[float, float]
    n_pixels: int
    grid_dims: Tuple[int, int]
    instrument_type: Optional[AxisType]
    pixel_size_um: Optional[float]
    has_escdat_folder: bool
    readable: bool
    error: Optional[str] = None
    resampling_method: Optional[ResamplingMethod] = None


def _probe_escdat(path: Path) -> bool:
    """Return ``True`` iff ``<path>/EscDat/`` exists as a directory.

    Robust to the path itself being a file (e.g. an ``.imzML``) or
    missing entirely.
    """
    try:
        return (path / "EscDat").is_dir()
    except OSError:
        return False


def _resampling_metadata_dict(essential: Any, comprehensive: Any) -> Dict[str, Any]:
    """Build the dict shape that ``ResamplingDecisionTree`` expects.

    Mirrors ``BaseSpatialDataConverter._get_reader_metadata_for_resampling``
    so the preview's auto-detection matches what the conversion would
    actually do.
    """
    metadata: Dict[str, Any] = {}

    metadata["source_path"] = str(getattr(essential, "source_path", ""))
    metadata["essential_metadata"] = {
        "spectrum_type": getattr(essential, "spectrum_type", None),
        "dimensions": essential.dimensions,
        "mass_range": essential.mass_range,
        "source_path": str(getattr(essential, "source_path", "")),
        "total_peaks": getattr(essential, "total_peaks", None),
        "n_spectra": getattr(essential, "n_spectra", None),
    }

    if comprehensive is None:
        return metadata

    raw_metadata = getattr(comprehensive, "raw_metadata", None) or {}
    if isinstance(raw_metadata, dict) and "global_metadata" in raw_metadata:
        metadata["GlobalMetadata"] = raw_metadata["global_metadata"]

    instrument_info = getattr(comprehensive, "instrument_info", None)
    if instrument_info is not None:
        metadata["instrument_info"] = instrument_info

    format_specific = getattr(comprehensive, "format_specific", None)
    if format_specific is not None:
        metadata["format_specific"] = format_specific

    acquisition_params = getattr(comprehensive, "acquisition_params", None)
    if acquisition_params is not None:
        metadata["acquisition_params"] = acquisition_params

    return metadata


def _pixel_size_um(pixel_size: Optional[Tuple[float, float]]) -> Optional[float]:
    """Collapse ``(x_um, y_um)`` to a single float, preferring ``x``.

    Real MSI rasters are essentially always isotropic.  A debug log
    fires if ``|x - y| > 0.01 um`` so anomalies are visible without
    surfacing a UX-confusing flag in the preview.

    Deliberately unlike the converter, which carries both pitches into
    every block of the store that states one (issue #228). This is a
    one-line summary of a folder someone is deciding whether to convert,
    not a coordinate system; the conversion it precedes reads the pair.
    """
    if pixel_size is None:
        return None
    x_um, y_um = float(pixel_size[0]), float(pixel_size[1])
    if abs(x_um - y_um) > 0.01:
        logger.debug(
            "Anisotropic pixel size detected (%.4f x %.4f um); "
            "preview reports the x component only.",
            x_um,
            y_um,
        )
    return x_um


def _unreadable(path: Path, error: str) -> MsiPreview:
    """Build the ``readable=False`` MsiPreview for a failed probe."""
    return MsiPreview(
        mz_range=(0.0, 0.0),
        n_pixels=0,
        grid_dims=(0, 0),
        instrument_type=None,
        pixel_size_um=None,
        has_escdat_folder=_probe_escdat(path),
        readable=False,
        error=error,
    )


def _build_reader(path: Path) -> tuple[Optional[Any], Optional[str]]:
    """Detect format + construct a reader.  Returns (reader, error).

    Either ``reader`` is a live :class:`BaseMSIReader` and ``error`` is
    ``None``, or ``reader`` is ``None`` and ``error`` carries the
    failure message.  Splitting this out keeps :func:`preview_msi`
    below the C901 complexity threshold.

    We pass ``metadata_only=True`` as a kwarg so readers that have
    expensive setup (notably :class:`BrukerReader`'s SDK DLL load)
    can short-circuit it.  Readers that don't accept the kwarg
    swallow it via their ``**kwargs`` -- :class:`BaseMSIReader` and
    every concrete reader takes ``**kwargs``.
    """
    try:
        format_name = detect_format(path)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Format detection failed: {exc}"
    try:
        reader_class = get_reader_class(format_name)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"No reader for format '{format_name}': {exc}"
    try:
        return reader_class(path, metadata_only=True), None
    except Exception as exc:
        return None, f"Reader construction failed: {exc}"


def _guess_axis_type(essential: Any, comprehensive: Any) -> Optional[AxisType]:
    """Best-effort AxisType pick.  Returns ``None`` on hard failure.

    Option B from the design: when the decision tree completes
    successfully, return whatever it picked (including its
    ``DefaultDetector`` fall-through of ``CONSTANT`` for genuinely
    unknown instruments).  Only return ``None`` if metadata extraction
    raised.
    """
    try:
        metadata_dict = _resampling_metadata_dict(essential, comprehensive)
        return ResamplingDecisionTree().select_axis_type(metadata_dict)
    except Exception as exc:
        logger.debug("AxisType auto-detection failed for preview: %s", exc)
        return None


def _guess_resampling_method(
    essential: Any, comprehensive: Any
) -> Optional[ResamplingMethod]:
    """Best-effort ResamplingMethod pick.  Returns ``None`` on hard failure.

    Same contract as :func:`_guess_axis_type`, and deliberately a separate
    call rather than something a caller derives from the axis type -- the
    detector answers the two questions independently.
    """
    try:
        metadata_dict = _resampling_metadata_dict(essential, comprehensive)
        return ResamplingDecisionTree().select_strategy(metadata_dict)
    except Exception as exc:
        logger.debug("ResamplingMethod auto-detection failed for preview: %s", exc)
        return None


def _close_quietly(reader: Optional[Any]) -> None:
    """Close a reader, swallowing any errors.  Best-effort cleanup."""
    if reader is None:
        return
    try:
        reader.close()
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("Reader.close() raised during preview: %s", exc)


def preview_msi(path: Path) -> MsiPreview:
    """Return a metadata-only snapshot of an MSI input.

    This is the entry point used by the Ousia Import Wizard to drive
    the per-sample preview card in step 2.  It detects the format,
    constructs the appropriate reader, and calls
    :meth:`BaseMSIReader.get_essential_metadata` (plus
    :meth:`get_comprehensive_metadata` for the instrument-type guess).
    No spectra are decoded.

    Args:
        path: Filesystem path to an ``.imzML`` file or a Bruker ``.d``
            directory.  May or may not exist; nonexistent paths are
            returned as ``readable=False`` rather than raising.

    Returns:
        :class:`MsiPreview` -- always returns; never raises.  Check
        :attr:`MsiPreview.readable` before using the numeric fields.

    Example:
        >>> from thyra import preview_msi
        >>> p = preview_msi(Path("slice_01.imzML"))
        >>> if p.readable:
        ...     print(p.grid_dims, p.mz_range)
    """
    path = Path(path)

    if not path.exists():
        return _unreadable(path, f"Path does not exist: {path}")

    reader, error = _build_reader(path)
    if reader is None:
        return _unreadable(path, error or "Reader unavailable")

    try:
        try:
            essential = reader.get_essential_metadata()
        except Exception as exc:
            return _unreadable(path, f"Could not read essential metadata: {exc}")

        # Comprehensive metadata is needed only for the AxisType guess.
        # If it fails, the rest of the preview is still valid.
        comprehensive = None
        try:
            comprehensive = reader.get_comprehensive_metadata()
        except Exception as exc:
            logger.debug("Comprehensive metadata unavailable for preview: %s", exc)

        dims = essential.dimensions
        return MsiPreview(
            mz_range=(float(essential.mass_range[0]), float(essential.mass_range[1])),
            n_pixels=int(essential.n_spectra),
            grid_dims=(int(dims[0]), int(dims[1])),
            instrument_type=_guess_axis_type(essential, comprehensive),
            pixel_size_um=_pixel_size_um(essential.pixel_size),
            has_escdat_folder=_probe_escdat(path),
            readable=True,
            error=None,
            resampling_method=_guess_resampling_method(essential, comprehensive),
        )
    finally:
        _close_quietly(reader)
