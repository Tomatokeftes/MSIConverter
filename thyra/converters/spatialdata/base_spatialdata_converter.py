# thyra/converters/spatialdata/base_spatialdata_converter.py

import json
import logging
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...alignment import AreaAlignmentResult, TeachingPointAlignment
from ...core.base_converter import BaseMSIConverter, PixelSizeSource
from ...core.base_reader import BaseMSIReader
from ...metadata.types import ComprehensiveMetadata, EssentialMetadata
from ...resampling import ResamplingDecisionTree, ResamplingMethod
from ...resampling.gaps import zero_across_gaps
from ...resampling.mass_axis.tof_generator import (
    DEFAULT_BINS_PER_FWHM,
    TOFAxisGenerator,
)
from ...resampling.mobility_grid import (
    MOBILITY_CHANNELS,
    MobilityGrid,
    build_mobility_grid,
    report_channel_width,
)
from ...resampling.tic import preserved_tic, rescale_to_preserved_tic
from ...resampling.types import AxisType, ResamplingConfig
from ...utils.zarr_atomic_write import install_windows_atomic_write_retry
from ._chunking import image_chunks, table_write_config

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_upstream_warnings():
    """Suppress known upstream warnings from ome_zarr, zarr v3, and spatialdata."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Passing storage-related arguments",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore", message="Object at.*is not recognized", category=UserWarning
        )
        warnings.filterwarnings(
            "ignore",
            message="Consolidated metadata is currently not",
            category=UserWarning,
        )
        yield


def _numeric_only(value: Any) -> bool:
    """True when a list round-trips through AnnData/zarr as a numeric array."""
    if isinstance(value, list):
        return all(_numeric_only(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "iufb"
    return isinstance(value, (bool, int, float, np.integer, np.floating, np.bool_))


def _json_fallback(obj: Any) -> Any:
    """Last-resort encoder for values ``json.dumps`` does not know."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def _jsonify_string_lists(obj: Any) -> Any:
    """Replace any list that is not purely numeric with its JSON encoding.

    AnnData/zarr cannot round-trip such lists: a list of dicts is
    stringified entry by entry into Python ``repr`` strings, and any list
    of strings comes back as a numpy string array -- whose ``deepcopy``
    segfaults outright on numpy 2.1-2.2 (numpy#28609). Since every table
    copy deepcopies ``uns`` (``AnnData.copy``, spatial queries, joins),
    one such array in ``uns`` kills the reader's process with no
    traceback. JSON side-steps both: it stores a single scalar string,
    and hands consumers the actual structure back through
    ``json.loads`` instead of ``repr`` output.

    Purely numeric lists (nested included) are kept: they become plain
    numeric arrays, which are safe and more useful as arrays.
    """
    if isinstance(obj, dict):
        return {key: _jsonify_string_lists(value) for key, value in obj.items()}
    if isinstance(obj, list):
        if _numeric_only(obj):
            return obj
        return json.dumps(obj, default=_json_fallback)
    return obj


def _resolve_config_enum(raw: Any, by_name: Dict[str, Any], key: str) -> Any:
    """Resolve one resampling-config value to its enum member.

    ``None``, ``"auto"`` and ``""`` all mean "decide this automatically"
    and resolve to ``None``.

    The CLI is protected by ``click.Choice``, but the Python API takes
    whatever the caller passes. Reject anything unrecognised rather than
    dropping it: silently treating ``"tic_preserving "`` or a typo as
    "auto-detect" hands back a conversion that ignored the request
    without saying so.

    Args:
        raw: The value as supplied by the caller.
        by_name: Accepted string spellings mapped to enum members.
        key: The config key, used in error messages.

    Raises:
        ValueError: If ``raw`` names no accepted value, or is neither a
            string nor one of the accepted enum members.
    """
    if raw is None:
        return None

    valid = ", ".join(repr(name) for name in ["auto", *by_name])

    if isinstance(raw, str):
        if raw in ("auto", ""):
            return None
        if raw not in by_name:
            raise ValueError(
                f"Unknown resampling_config[{key!r}] value {raw!r}. "
                f"Valid values are: {valid}."
            )
        return by_name[raw]

    if raw in by_name.values():
        return raw

    raise ValueError(
        f"Unsupported resampling_config[{key!r}] value {raw!r}. "
        f"Pass one of {valid}, or the matching enum member."
    )


def _normalize_resampling_config(
    config: Union[Dict[str, Any], "ResamplingConfig"],
) -> "ResamplingConfig":
    """Normalise a resampling config dict or dataclass to a ResamplingConfig.

    Accepts either a plain dict (as produced by _build_resampling_config in
    __main__.py) or an already-constructed ResamplingConfig dataclass and
    returns a ResamplingConfig in both cases.

    Raises:
        ValueError: If ``method`` or ``axis_type`` is not a recognised
            value.
    """
    if isinstance(config, ResamplingConfig):
        return config

    from ...resampling.types import DEFAULT_REFERENCE_MZ, AxisType

    # Only the methods the resampling pipeline actually implements, and
    # only the axis types CommonAxisBuilder has a generator for. These
    # deliberately match the CLI's click.Choice lists.
    method_by_name = {
        "nearest_neighbor": ResamplingMethod.NEAREST_NEIGHBOR,
        "tic_preserving": ResamplingMethod.TIC_PRESERVING,
    }
    axis_type_by_name = {
        "constant": AxisType.CONSTANT,
        "linear_tof": AxisType.LINEAR_TOF,
        "reflector_tof": AxisType.REFLECTOR_TOF,
        "tof": AxisType.TOF,
        "orbitrap": AxisType.ORBITRAP,
        "fticr": AxisType.FTICR,
    }

    reference_mz = config.get("reference_mz")

    def _optional_float(key: str) -> Optional[float]:
        value = config.get(key)
        return None if value is None else float(value)

    return ResamplingConfig(
        method=_resolve_config_enum(config.get("method"), method_by_name, "method"),
        axis_type=_resolve_config_enum(
            config.get("axis_type"), axis_type_by_name, "axis_type"
        ),
        target_bins=config.get("target_bins"),
        mass_width_da=config.get("width_at_mz"),
        reference_mz=(
            DEFAULT_REFERENCE_MZ if reference_mz is None else float(reference_mz)
        ),
        min_mz=config.get("min_mz"),
        max_mz=config.get("max_mz"),
        gap_tolerance_da=config.get("gap_tolerance_da"),
        tof_a=_optional_float("tof_a"),
        tof_b=_optional_float("tof_b"),
        bins_per_fwhm=_optional_float("bins_per_fwhm"),
    )


# Check SpatialData availability (defer imports to avoid issues)
SPATIALDATA_AVAILABLE = False
_import_error_msg = None
try:
    import geopandas as gpd
    import zarr
    from anndata import AnnData
    from shapely.geometry import box
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, ShapesModel, TableModel
    from spatialdata.transformations import Affine, Identity, Scale, Sequence

    from .optical_image import OpticalTiffSource, StreamedOpticalImage

    SPATIALDATA_AVAILABLE = True
except (ImportError, NotImplementedError) as e:
    _import_error_msg = str(e)
    logger.warning(f"SpatialData dependencies not available: {e}")
    SPATIALDATA_AVAILABLE = False

    # Create dummy classes for registration
    AnnData = None
    SpatialData = None
    TableModel = None
    ShapesModel = None
    Image2DModel = None
    Affine = None
    Identity = None
    Scale = None
    box = None
    gpd = None
    OpticalTiffSource = None  # type: ignore[misc]
    StreamedOpticalImage = None  # type: ignore[misc]


def _calc_optical_scale_factors(
    smallest_dim: int,
    min_coarsest_size: int = 1000,
    factor: int = 2,
) -> list:
    """Pick pyramid scale factors for an optical image of given size.

    Decides how many cumulative-doubling downsample levels to generate
    based on the smallest spatial dimension; stops once the next
    halving would drop the short side below ``min_coarsest_size``.

    Mirrors :func:`spatialdata_io.readers._utils._utils.calc_scale_factors`
    so wizard-converted microscopy and Thyra-bundled FlexImaging
    brightfield share the same pyramid shape Xenium's own morphology
    image gets out of spatialdata-io.

    Returns a list of (cumulative) downsample factors to feed to
    ``Image2DModel.parse(scale_factors=...)``.  Empty list means
    "no pyramid needed" (image is already at or below the coarsest
    target size).

    Args:
        smallest_dim: ``min(width, height)`` of the source image.
        min_coarsest_size: stop adding levels once the next halving
            would drop the short side below this value.  Default
            ~1000 px matches spatialdata-io's convention and gives a
            coarsest level that comfortably fits a single viewport
            paint in a few hundred KB.
        factor: downsample factor per step.  Default 2 (mip-map style).
    """
    if smallest_dim <= 0:
        return []
    factors: list = []
    cur = smallest_dim / factor
    while cur >= min_coarsest_size:
        factors.append(factor)
        cur /= factor
    return factors


#: How far a marginal may sit from the column it mirrors, relative to the
#: largest value in the summed table, and still be called exact. Summing
#: the same float64 values in a different order is the only difference
#: under ``--tdf-spectrum scan_sum``.
_MARGINAL_TOLERANCE = 1e-9


def _current_ratio_block(
    table: Any, summed_key: str, summed: Any
) -> Optional[Dict[str, Any]]:
    """How much of the summed table's ion current a sibling holds, per pixel.

    A sibling's row sum over every one of its columns is that pixel's ion
    current, and so is the summed table's, so the two row sums compare
    directly. Both matrices sit on memmaps, and a row sum is one pass
    over each -- nothing the size of a matrix is held in RAM, which is
    why this is the whole comparison: a cell-by-cell deviation between
    the grid table's marginal and the summed table needs the product and
    the difference materialised, each as large as the summed table, and
    the route that writes both never holds either. ``None`` when the two
    cannot be compared (a row count mismatch, no ion current at all),
    which is not a disagreement.
    """
    if summed is None:
        return None
    totals = np.asarray(summed.X.sum(axis=1)).ravel().astype(np.float64)
    split = np.asarray(table.X.sum(axis=1)).ravel().astype(np.float64)
    if split.size != totals.size or not totals.any():
        return None
    per_pixel = split / np.where(totals == 0, np.nan, totals)
    return {
        "summed_table": summed_key,
        "current_ratio": float(split.sum() / totals.sum()),
        "current_ratio_pixel_min": float(np.nanmin(per_pixel)),
        "current_ratio_pixel_max": float(np.nanmax(per_pixel)),
    }


def _nn_map_to_bins(
    axis: NDArray[np.float64], mzs: NDArray[np.float64]
) -> NDArray[np.int_]:
    """Map in-range m/z values to their nearest bin on ``axis``.

    Ties (a peak exactly between two bins) resolve to the right bin,
    matching the strict ``<`` comparison this code has always used.

    A module-level function rather than a method so the unbound-call test
    harnesses (a ``SimpleNamespace`` posing as the converter) keep working.

    Args:
        axis: The target mass axis, ascending.
        mzs: m/z values, all within ``[axis[0], axis[-1]]``.

    Returns:
        The nearest-bin index of each m/z value, same length as ``mzs``.
    """
    # Find insertion points using vectorized binary search
    indices = np.searchsorted(axis, mzs)

    # Clip to valid range. Everything reaching here is inside the axis,
    # so this only pins searchsorted's one-past-the-end result for a
    # value equal to axis[-1]; it can no longer pull an outside peak in.
    indices_clipped = np.clip(indices, 0, len(axis) - 1)

    # For non-boundary points, check if left is closer
    # Only check where we're not at the left edge
    check_left = indices > 0
    if np.any(check_left):
        # Get distances only for points that need checking
        mz_values = axis[indices_clipped[check_left]]
        mz_values_left = axis[indices_clipped[check_left] - 1]
        mz_query = mzs[check_left]

        # Use left if it's closer
        use_left = np.abs(mz_values_left - mz_query) < np.abs(mz_values - mz_query)
        indices_clipped[check_left] = np.where(
            use_left, indices_clipped[check_left] - 1, indices_clipped[check_left]
        )
    return indices_clipped


def _nn_accumulate(
    idx: NDArray[np.int_], intensities: NDArray[np.float64]
) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Sum intensities per bin and return the non-zero bins, ascending.

    Two equivalent routes. When ``idx`` is non-decreasing -- true whenever
    the spectrum's m/z values are ascending, which every format seen so
    far produces -- equal bins form contiguous runs, so the per-bin sums
    are ``np.add.reduceat`` over the run starts. That is O(n_peaks),
    where the ``np.bincount`` fallback is O(n_bins): for a centroid
    spectrum of hundreds of peaks against an axis of 10^5 bins the
    difference is roughly 7x per spectrum. Both sum left-to-right over
    the same float64 values, so the results are bit-identical; the
    fallback keeps unsorted input correct.

    The kept-bin test is ``!= 0`` rather than ``> 0`` because a bin whose
    accumulated value is negative is still a measurement: dropping it
    silently raises the stored TIC above the input's. Baseline-subtracted
    data can carry negative intensities, though every export seen so far
    filters them out upstream.
    """
    # bincount always promotes its weights to float64; match it so both
    # accumulation routes return the same dtype and the same rounding.
    vals = intensities.astype(np.float64, copy=False)

    if idx.size and bool(np.all(idx[1:] >= idx[:-1])):
        starts = np.concatenate(([0], np.flatnonzero(np.diff(idx)) + 1))
        if starts.size == idx.size:
            # Every peak already sits in its own bin (the common case
            # for both centroid and profile data): nothing to sum.
            bins = idx
            sums = vals
        else:
            bins = idx[starts]
            sums = np.add.reduceat(vals, starts)
        keep = sums != 0
        return bins[keep].astype(np.int_, copy=False), sums[keep]

    accumulated = np.bincount(idx, weights=vals)
    nonzero_mask = accumulated != 0
    nonzero_indices = np.where(nonzero_mask)[0].astype(np.int_)
    nonzero_values = accumulated[nonzero_mask]
    return nonzero_indices, nonzero_values.astype(np.float64)


def _reference_params(converter: Any, axis_name: str) -> Tuple[float, float]:
    """``(width_da, reference_mz)`` for the axis about to be built.

    Precedence: the caller's ``--resample-width-at-mz`` /
    ``--resample-reference-mz``; then the width the detected instrument
    declared (``_detected_reference_width``, set on the auto path only);
    then the per-axis-type default -- 17 mDa at m/z 300 for ``linear_tof``,
    chosen to be close to the axis SCiLS Lab produces for FlexImaging data,
    and 5 mDa at m/z 1000 for everything else.

    Module-level so that ``_calculate_bins_from_width`` and
    ``_get_reference_params`` cannot drift apart, and so that either can be
    driven on a bare stub carrying only the three attributes.
    """
    if converter._width_at_mz is not None:
        return converter._width_at_mz, converter._reference_mz
    if axis_name == "tof":
        # The law and the bins-per-FWHM fix the width everywhere; report
        # the one realised at the reference m/z.
        a, b, k = _tof_plan(converter)
        reference_mz = float(converter._reference_mz)
        return float(TOFAxisGenerator(a, b).bin_width_at(reference_mz, k)), reference_mz
    detected = getattr(converter, "_detected_reference_width", None)
    if detected is not None:
        return float(detected[0]), float(detected[1])
    if axis_name == "linear_tof":
        return 0.017, 300.0
    return 0.005, 1000.0


def _tof_plan(converter: Any) -> Tuple[float, float, float]:
    """``(A, B, bins_per_fwhm)`` for an ``AxisType.TOF`` axis.

    The law is the caller's ``tof_a``/``tof_b`` pair, else the pair the
    detected instrument declared (``_detected_tof_law``). ``bins_per_fwhm``
    is derived from ``--resample-width-at-mz`` at the reference m/z when
    that was given, so the width flag means the same thing on every axis
    type; otherwise it is the API's ``bins_per_fwhm``, else 3.
    """
    a = getattr(converter, "_tof_a", None)
    b = getattr(converter, "_tof_b", None)
    if a is None or b is None:
        law = getattr(converter, "_detected_tof_law", None)
        if law is None:
            raise ValueError(
                "A 'tof' mass axis needs the width law's coefficients: pass "
                "--tof-law A B, or convert a run whose instrument declares "
                "them (SELECT SERIES MRT centroid, timsTOF)."
            )
        a, b = law
    generator = TOFAxisGenerator(float(a), float(b))
    if converter._width_at_mz is not None:
        k = generator.bins_per_fwhm_for(
            float(converter._reference_mz), float(converter._width_at_mz)
        )
    else:
        k = getattr(converter, "_bins_per_fwhm", None)
        if k is None:
            k = DEFAULT_BINS_PER_FWHM
    return float(a), float(b), float(k)


def _tic_support_bins(
    axis: NDArray[np.float64],
    mzs: NDArray[np.float64],
    intensities: NDArray[np.float64],
) -> NDArray[np.int_]:
    """Axis indices at which the linear interpolant of a spectrum can be non-zero.

    ``np.interp`` draws straight lines between consecutive source points, so
    the interpolant is non-zero only on segments with a non-zero endpoint.
    Consecutive non-zero samples form runs; each run's support is the open
    interval from the sample before it to the sample after it (those two are
    where the trace touches zero), closed at the spectrum's own ends where
    there is no such neighbour. Runs are separated by at least one zero
    sample, so their supports never overlap and the indices come out sorted.

    A zero-suppressed profile -- Waters MassLynx stores samples in clusters
    around each peak with an explicit zero at either edge -- has supports
    covering a small fraction of a fine axis, which is what makes the sparse
    evaluation cheap. A spectrum with no zeros in it is one run, and its
    support is every axis point the dense evaluation would have populated.

    Args:
        axis: Target mass axis, ascending.
        mzs: Source m/z values, ascending.
        intensities: Source intensities, parallel to ``mzs``.

    Returns:
        Sorted, unique axis indices. Empty when nothing is non-zero.
    """
    nonzero = intensities != 0
    if not nonzero.any():
        return np.array([], dtype=np.int_)

    last = mzs.size - 1
    edges = np.diff(np.concatenate(([False], nonzero, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1

    # Open at a neighbouring zero sample (the interpolant is exactly zero
    # there), closed at the spectrum's own first or last sample.
    lo = np.where(
        starts == 0,
        np.searchsorted(axis, mzs[0], side="left"),
        np.searchsorted(axis, mzs[np.maximum(starts - 1, 0)], side="right"),
    )
    hi = np.where(
        ends == last,
        np.searchsorted(axis, mzs[last], side="right"),
        np.searchsorted(axis, mzs[np.minimum(ends + 1, last)], side="left"),
    )

    lengths = hi - lo
    keep = lengths > 0
    if not keep.any():
        return np.array([], dtype=np.int_)
    lo = lo[keep]
    lengths = lengths[keep]
    offsets = np.cumsum(lengths) - lengths
    total = int(lengths.sum())
    return np.repeat(lo - offsets, lengths) + np.arange(total, dtype=np.int_)


def _tic_preserving_sparse(
    axis: NDArray[np.float64],
    mzs: NDArray[np.float64],
    intensities: NDArray[np.float64],
    gap_tolerance_da: Optional[float],
) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """TIC-preserving resampling, evaluated only where it can be non-zero.

    This is the operator ``BaseSpatialDataConverter._tic_preserving_resample``
    documents -- interpolate onto the axis, zero unsupported bins, rescale to
    the preserved TIC -- restricted to the axis points
    :func:`_tic_support_bins` reports. Every other axis point interpolates
    to exactly zero, so scattering the result into a zero array reproduces
    the dense evaluation bin for bin; the sole difference is the order in
    which the rescale sums its terms.

    Measured on a Waters SELECT SERIES MRT run (13,398 pixels, ~15,000
    stored samples per strong pixel, 1.05M-bin axis): the dense form cost
    570 s against 16 s for nearest-neighbour binning, almost all of it in
    interpolating onto and then scanning a million bins per pixel of which
    ~13,000 were ever non-zero.

    Args:
        axis: Target mass axis, ascending.
        mzs: Source m/z values, any order.
        intensities: Source intensities, parallel to ``mzs``.
        gap_tolerance_da: See :func:`thyra.resampling.gaps.zero_across_gaps`.

    Returns:
        ``(bin_indices, intensities)`` holding only the non-zero bins,
        indices ascending.
    """
    empty = (np.array([], dtype=np.int_), np.array([], dtype=np.float64))
    if mzs.size == 0:
        return empty

    if np.all(mzs[:-1] <= mzs[1:]):
        mzs_sorted = mzs
        intensities_sorted = intensities
    else:
        order = np.argsort(mzs)
        mzs_sorted = mzs[order]
        intensities_sorted = intensities[order]

    if mzs_sorted.size == 1:
        # np.interp cannot interpolate a lone point onto a grid that does
        # not contain it -- it would return all zeros and lose the peak.
        # Place it in its nearest bin, as the nearest_neighbor path and
        # TICPreservingStrategy both do.
        target_tic = preserved_tic(
            mzs_sorted, intensities_sorted, float(axis[0]), float(axis[-1])
        )
        if target_tic <= 0.0:
            return empty
        nearest = int(np.argmin(np.abs(axis - mzs_sorted[0])))
        return (
            np.array([nearest], dtype=np.int_),
            np.array([target_tic], dtype=np.float64),
        )

    indices = _tic_support_bins(axis, mzs_sorted, intensities_sorted)
    if indices.size == 0:
        return empty

    targets = axis[indices]
    values = np.interp(targets, mzs_sorted, intensities_sorted, left=0.0, right=0.0)

    # Discard bins no source point vouches for, before the rescale so the
    # intensity returns to the bins that were measured rather than being
    # deleted. No-op when no tolerance was configured.
    zero_across_gaps(values, targets, mzs_sorted, gap_tolerance_da)

    # Rescale to the required TIC -- the step that makes the method live up
    # to its name. Reads only the axis endpoints and the sum of ``values``,
    # so the subset evaluation rescales exactly as the dense one would.
    rescale_to_preserved_tic(values, axis, mzs_sorted, intensities_sorted)

    keep = values != 0
    return indices[keep], values[keep]


class _SharedAxisNNCache:
    """Precomputed nearest-neighbor mapping for one recurring m/z array.

    Built the first time :meth:`_nearest_neighbor_resample` sees a spectrum,
    and reused for every later spectrum that carries the same m/z array.
    ``key`` is a private copy so a caller-side mutation of the original
    array cannot fool the equality check; ``key_ref`` keeps the original
    object for the O(1) identity test that readers yielding one shared
    array (Rapiflex, Waters, PHI) hit every time.
    """

    __slots__ = (
        "key",
        "key_ref",
        "n_total",
        "n_dropped",
        "lo",
        "hi",
        "starts",
        "bins",
    )

    key: NDArray[np.float64]
    key_ref: NDArray[np.float64]
    n_total: int
    n_dropped: int
    lo: int
    hi: int
    starts: Optional[NDArray[np.int_]]
    bins: NDArray[np.int_]


class BaseSpatialDataConverter(BaseMSIConverter, ABC):
    """Base converter for MSI data to SpatialData format with shared functionality."""

    def __init__(
        self,
        reader: BaseMSIReader,
        output_path: Path,
        dataset_id: str = "msi_dataset",
        pixel_size_um: float = 1.0,
        pixel_size_source: PixelSizeSource = PixelSizeSource.DEFAULT,
        handle_3d: bool = False,
        z_spacing_um: Optional[float] = None,
        pixel_size_detection_info: Optional[Dict[str, Any]] = None,
        resampling_config: Optional[Union[Dict[str, Any], ResamplingConfig]] = None,
        include_optical: bool = True,
        apply_optical_alignment: bool = True,
        write_mobility_table: bool = True,
        mobility_heatmap: bool = True,
        mobility_grid: bool = False,
        mobility_bins: int = MOBILITY_CHANNELS,
        mobility_min: Optional[float] = None,
        mobility_max: Optional[float] = None,
        msms_table: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the base SpatialData converter.

        Args:
            reader: MSI data reader
            output_path: Path for output file
            dataset_id: Identifier for the dataset
            pixel_size_um: In-plane size of each pixel in micrometers
            pixel_size_source: How pixel size was determined
            handle_3d: Whether to process as 3D data (True) or 2D slices
                (False)
            z_spacing_um: Distance between consecutive slices in
                micrometers.  Only meaningful when a true volume is
                written (``handle_3d=True`` and more than one slice).
                ``None`` (default) falls back to the in-plane pitch and
                records that as an assumption rather than a measurement;
                see :meth:`BaseMSIConverter._resolve_z_spacing`.
            pixel_size_detection_info: Optional metadata about pixel size
                detection
            resampling_config: Optional resampling configuration dict
            include_optical: Whether to include optical images in output
                (default: True)
            write_mobility_table: When the reader shares one set of
                (m/z, mobility) feature pairs across pixels, also write
                them as a mobility-resolved sibling table
                (``{table}_mobility``) beside the summed MSI table
                (default: True). Never changes the MSI table itself.
            mobility_heatmap: When the reader has an ion mobility
                dimension, accumulate the mean mass-mobility frame from
                the raw scan read and store it on the summed table as
                ``uns["mobility_heatmap"]`` (default: True). Costs one
                extra pass over the source; see ``mobility_heatmap.py``.
            mobility_grid: When the reader carries mobility per pixel
                rather than as a shared feature axis (Bruker TDF), bin
                the point cloud onto a common mobility grid and write the
                result as the same ``{table}_mobility`` sibling
                (default: False -- opt in, it costs an extra pass and a
                far larger table). Ignored by a source that already
                shares a feature axis, which needs no grid.
            mobility_bins: Channels the grid divides the mobility range
                into (default: 256). The default is the mass-mobility
                heatmap's own channel count over the same edges, which is
                what lets a box drawn on the heatmap index grid channels
                directly; changing it gives that up.
            mobility_min: Lower edge of the grid, in the axis unit.
                ``None`` (default) takes the smallest mobility value the
                source's axis actually holds, which is also where the
                heatmap starts.
            mobility_max: Upper edge of the grid; ``None`` takes the
                largest value the axis holds.
            msms_table: When the source isolates several precursors per
                pixel in disjoint mobility slices (Bruker PASEF), also
                write them split apart as a demultiplexed sibling table
                (``{table}_msms``) beside the summed MSI table (default:
                True -- the summed spectrum of such a pixel is a mixture
                of unrelated fragment spectra, so the split is the
                accurate representation; design decision D2). Refused
                with a reason rather than approximated when the schedule
                is not separable, which is also what happens on every
                source that is not MS/MS, so the default costs a source
                that has nothing to split nothing; see ``msms_table.py``.
                Never changes the MSI table itself.
            apply_optical_alignment: If True (default) and the MSI source
                has FlexImaging Area metadata, compute an alignment that
                places MSI raster coordinates in optical-image pixel
                space.  Set to False when a downstream tool owns the
                alignment (e.g. Ousia's wizard registers MSI to Xenium
                via EscDat and does not want Thyra to pre-rotate the
                MSI into FlexImaging's optical frame).  When False the
                MSI elements land in pure micrometer coordinates at
                ``"global"``.
            **kwargs: Additional keyword arguments

        Raises:
            ImportError: If SpatialData dependencies are not available
            ValueError: If pixel_size_um is not positive, dataset_id is
                empty, or ``sparse_format`` is passed
        """
        # ``sparse_format`` chose between CSC and CSR until v3.22. CSC is now
        # the only layout written, so the keyword has nothing left to select --
        # but an unknown keyword lands in ``self.options`` without a word, and a
        # caller who asked for CSR would get CSC and no signal. That is the
        # failure this removal was meant to end, so say it instead.
        if "sparse_format" in kwargs:
            raise ValueError(
                "sparse_format was removed: every converter writes CSC, which "
                "is the layout an ion image reads down. Drop the argument; for "
                "row-wise access call X.tocsr() on the matrix you read back."
            )

        # Check if SpatialData is available
        if not SPATIALDATA_AVAILABLE:
            error_msg = (
                f"SpatialData dependencies not available: "
                f"{_import_error_msg}. "
                f"Please install required packages or fix dependency "
                f"conflicts."
            )
            raise ImportError(error_msg)

        # Every Zarr write below this point goes through Zarr's atomic
        # rename, which on Windows intermittently loses a race against
        # whatever else has the destination open. Idempotent and a no-op
        # off Windows.
        install_windows_atomic_write_retry()

        # Validate inputs
        if pixel_size_um <= 0:
            raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")
        if not dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")

        # Extract pixel_size_detection_info from kwargs if provided
        kwargs_filtered = dict(kwargs)
        if (
            pixel_size_detection_info is None
            and "pixel_size_detection_info" in kwargs_filtered
        ):
            pixel_size_detection_info = kwargs_filtered.pop("pixel_size_detection_info")

        super().__init__(
            reader,
            output_path,
            dataset_id=dataset_id,
            pixel_size_um=pixel_size_um,
            pixel_size_source=pixel_size_source,
            handle_3d=handle_3d,
            z_spacing_um=z_spacing_um,
            **kwargs_filtered,
        )

        self._non_empty_pixel_count: int = 0
        # Peaks discarded for falling outside the target mass axis, and
        # whether the one-line summary has been emitted yet. See
        # _count_out_of_range().
        self._out_of_range_peaks: int = 0
        self._out_of_range_warned: bool = False
        self._pixel_size_detection_info = pixel_size_detection_info
        self._resampling_config = (
            _normalize_resampling_config(resampling_config)
            if resampling_config is not None
            else None
        )
        self._include_optical = include_optical
        self._apply_optical_alignment = apply_optical_alignment
        # Filled by _build_resampled_mass_axis(); consumed by
        # _processing_provenance().
        self._resolved_resampling_plan: Optional[Dict[str, Any]] = None

        # The mobility-resolved sibling table (see mobility_table.py): whether
        # to write one, and -- once a finalize step has decided for its slice
        # -- the element key it gets, so the MSI table's uns can name it.
        self._write_mobility_table = bool(write_mobility_table)
        self._mobility_table_key: Optional[str] = None
        # The common mobility grid (see resampling/mobility_grid.py): the
        # second way to fill the same sibling, for a source whose pixels
        # each carry their own mobility values. Resolved once by
        # _plan_mobility_table, so the MSI table's metadata block and the
        # sibling describe the same grid.
        self._mobility_grid_enabled = bool(mobility_grid)
        self._mobility_bins = int(mobility_bins)
        self._mobility_bounds = (mobility_min, mobility_max)
        self._mobility_grid: Optional[MobilityGrid] = None
        self._mobility_grid_resolved = False
        # The grid's discovery pass for the slice being written, when the
        # converter ran it fused with the heatmap's pass (see
        # _prepare_sibling_scans); consumed by _attach_sibling_tables.
        self._grid_discovery: Any = None
        # The MS/MS table's accumulator when the converter fed it from the
        # summed table's own passes (see fused_passes.py); consumed by
        # _attach_sibling_tables like the grid's discovery.
        self._msms_accumulator: Any = None
        # Whether the sibling sinks were fed from the summed table's passes
        # already, so _prepare_sibling_scans has nothing left to scan; and
        # whether the sibling tables were planned before those passes.
        self._sibling_scans_done = False
        self._siblings_planned = False
        # Scratch directories holding the memmapped matrices of every table
        # until it is written; released by _release_table_scratch.
        self._table_scratch: List[Tuple[Any, Path]] = []
        # The mass-mobility heatmap (see mobility_heatmap.py): built once
        # per conversion, on first demand, and shared by every uns block
        # that asks for it. ``_built`` distinguishes "not yet" from
        # "tried, nothing to write".
        self._mobility_heatmap_enabled = bool(mobility_heatmap)
        self._mobility_heatmap_block: Optional[Dict[str, Any]] = None
        self._mobility_heatmap_built = False
        # The demultiplexed MS/MS sibling (see msms_table.py): on by default, and
        # -- once a finalize step has decided for its slice -- the element
        # key it gets, so the MSI table's uns can name it.
        self._write_msms_table = bool(msms_table)
        self._msms_table_key: Optional[str] = None
        self._fragmentation_schedule: Any = None
        self._fragmentation_read = False

        # Metadata caches (populated lazily during conversion). These have
        # to exist before _setup_resampling below: its strategy selection
        # extracts the reader's metadata through them, and every extractor
        # swallows failures at DEBUG. With the caches assigned after it,
        # that first extraction hit AttributeError on every field, the
        # decision tree saw an empty dict, and *every* reader's method was
        # chosen by DefaultDetector -- nearest_neighbor -- while the axis
        # type, resolved later on the properly cached metadata, came from
        # the right detector. Nothing noticed while the detectors agreed;
        # the Waters profile route is the first to ask for tic_preserving.
        self._essential_metadata_cached: Optional[EssentialMetadata] = None
        self._comprehensive_metadata_cached: Optional[ComprehensiveMetadata] = None
        self._spectrum_metadata_cached: Optional[Dict[str, Any]] = None
        self._resampling_metadata_cached: Optional[Dict[str, Any]] = None

        # Set up resampling if enabled
        if self._resampling_config:
            self._setup_resampling()
            # Note: _build_resampled_mass_axis() will be called in _initialize_conversion()
            # after reader metadata is fully loaded

        # Shared-axis nearest-neighbor cache. Continuous imzML, Rapiflex,
        # Waters and PHI all hand every spectrum the same m/z array, so the
        # peak-to-bin mapping is computed once and verified per spectrum by
        # array equality instead of being re-derived by searchsorted. None
        # means "not built yet"; False means "tried and the data does not
        # share an axis, stop checking". See _nearest_neighbor_resample.
        self._nn_shared_cache: Any = None
        self._nn_cache_misses: int = 0

        # Optical-MSI alignment (computed from FlexImaging Area definitions)
        self._alignment_result: Optional[AreaAlignmentResult] = None
        # Affine matrix mapping TIC raster indices to optical image pixels
        self._tic_to_image_matrix: Optional[NDArray[np.float64]] = None
        # Primary optical image filename from .mis <ImageFile> and its dimensions
        self._primary_optical_filename: Optional[str] = None
        self._primary_optical_dims: Optional[Tuple[int, int]] = None  # (width, height)
        # Optical images declared to SpatialData as placeholders whose pixels
        # still have to be streamed into the store once it is written. See
        # optical_image.py and _stream_pending_optical_pixels().
        self._pending_optical_images: Dict[str, "StreamedOpticalImage"] = {}

    def _setup_resampling(self) -> None:
        """Set up resampling configuration and strategy."""
        if not self._resampling_config:
            return

        config = self._resampling_config
        method = config.method
        axis_type = config.axis_type

        # If method is None or "auto", use DecisionTree to determine strategy
        if method is None:
            try:
                # Get metadata from reader for instrument detection
                metadata = self._get_reader_metadata_for_resampling()
                tree = ResamplingDecisionTree()
                detected_method = tree.select_strategy(metadata)
                logger.info(f"Auto-detected resampling method: {detected_method}")
                self._resampling_method = detected_method
            except NotImplementedError as e:
                logger.error(f"Auto-detection failed: {e}")
                logger.info("Falling back to nearest_neighbor for resampling")
                self._resampling_method = ResamplingMethod.NEAREST_NEIGHBOR
        else:
            # Use provided method directly (already an enum)
            self._resampling_method = method

        logger.info(f"Using resampling method: {self._resampling_method}")

        # Store axis_type override if provided (will be used in _build_resampled_mass_axis)
        self._manual_axis_type = axis_type

        # Store resampling parameters from the ResamplingConfig dataclass
        self._target_bins = config.target_bins
        self._min_mz = config.min_mz
        self._max_mz = config.max_mz
        self._width_at_mz = config.mass_width_da
        self._reference_mz = config.reference_mz
        # Filled by _resolve_resampling_plan when the detected instrument
        # declares a bin width and the caller set none.
        self._detected_reference_width: Optional[Tuple[float, float]] = None
        # The two-term TOF width law (AxisType.TOF): the caller's pair, or
        # the detected instrument's when the axis resolves to TOF without one.
        self._tof_a = config.tof_a
        self._tof_b = config.tof_b
        self._bins_per_fwhm = config.bins_per_fwhm
        self._detected_tof_law: Optional[Tuple[float, float]] = None
        self._gap_tolerance_da = config.gap_tolerance_da
        if self._gap_tolerance_da is not None:
            logger.info(
                f"Interpolation gap tolerance: {self._gap_tolerance_da} Da "
                "(target bins farther than this from any source m/z are zeroed)"
            )

    def _get_cached_metadata_for_resampling(self) -> Dict[str, Any]:
        """Get cached metadata for resampling decision tree to avoid multiple reader calls."""
        if self._resampling_metadata_cached is not None:
            return self._resampling_metadata_cached

        # If not cached yet, extract and cache it
        return self._get_reader_metadata_for_resampling()

    def _get_reader_metadata_for_resampling(self) -> Dict[str, Any]:
        """Extract metadata from reader for resampling decision tree."""
        metadata: Dict[str, Any] = {}

        # Extract different types of metadata
        self._extract_essential_metadata(metadata)
        self._extract_comprehensive_metadata(metadata)
        self._extract_spectrum_metadata(metadata)

        # Cache for later reuse
        self._resampling_metadata_cached = metadata
        return metadata

    def _extract_essential_metadata(self, metadata: Dict[str, Any]) -> None:
        """Extract essential metadata for resampling decisions."""
        try:
            # Use cached essential metadata if available
            if self._essential_metadata_cached is not None:
                essential = self._essential_metadata_cached
            else:
                essential = self.reader.get_essential_metadata()
                self._essential_metadata_cached = essential

            if hasattr(essential, "source_path"):
                metadata["source_path"] = str(essential.source_path)

            # Add essential metadata for resampling decisions
            metadata["essential_metadata"] = {
                "spectrum_type": getattr(essential, "spectrum_type", None),
                "dimensions": essential.dimensions,
                "mass_range": essential.mass_range,
                "source_path": str(essential.source_path),
                "total_peaks": getattr(essential, "total_peaks", None),
                "n_spectra": getattr(essential, "n_spectra", None),
            }
        except Exception as e:
            logger.debug(f"Could not extract essential metadata: {e}")

    def _extract_comprehensive_metadata(self, metadata: Dict[str, Any]) -> None:
        """Extract comprehensive metadata including Bruker GlobalMetadata."""
        try:
            # Use cached comprehensive metadata if available
            if self._comprehensive_metadata_cached is not None:
                comp_meta = self._comprehensive_metadata_cached
            else:
                comp_meta = self.reader.get_comprehensive_metadata()
                self._comprehensive_metadata_cached = comp_meta

            self._extract_bruker_metadata(metadata, comp_meta)
            self._extract_instrument_info(metadata, comp_meta)
        except Exception as e:
            logger.debug(f"Could not extract comprehensive metadata: {e}")

    def _extract_bruker_metadata(self, metadata: Dict[str, Any], comp_meta) -> None:
        """Extract Bruker GlobalMetadata from comprehensive metadata."""
        if (
            hasattr(comp_meta, "raw_metadata")
            and "global_metadata" in comp_meta.raw_metadata
        ):
            metadata["GlobalMetadata"] = comp_meta.raw_metadata["global_metadata"]
            logger.debug(
                f"Extracted Bruker GlobalMetadata with keys: "
                f"{list(metadata['GlobalMetadata'].keys())}"
            )

    def _extract_instrument_info(self, metadata: Dict[str, Any], comp_meta) -> None:
        """Extract instrument_info for fallback detection."""
        if hasattr(comp_meta, "instrument_info"):
            metadata["instrument_info"] = comp_meta.instrument_info
            logger.debug(f"Extracted instrument_info: {comp_meta.instrument_info}")

        # Extract format_specific for FlexImaging detection
        if hasattr(comp_meta, "format_specific"):
            metadata["format_specific"] = comp_meta.format_specific
            logger.debug(f"Extracted format_specific: {comp_meta.format_specific}")

        # Extract acquisition_params for additional detection
        if hasattr(comp_meta, "acquisition_params"):
            metadata["acquisition_params"] = comp_meta.acquisition_params
            logger.debug(
                f"Extracted acquisition_params: {comp_meta.acquisition_params}"
            )

    def _extract_spectrum_metadata(self, metadata: Dict[str, Any]) -> None:
        """Extract ImzML-specific spectrum metadata."""
        try:
            if hasattr(self.reader, "get_spectrum_metadata"):
                # Use cached spectrum metadata if available
                if self._spectrum_metadata_cached is not None:
                    spec_meta = self._spectrum_metadata_cached
                else:
                    spec_meta = self.reader.get_spectrum_metadata()
                    self._spectrum_metadata_cached = spec_meta

                if spec_meta:
                    metadata.update(spec_meta)
        except Exception as e:
            logger.debug(f"Could not extract spectrum metadata: {e}")

    def convert(self) -> bool:
        """Run the base workflow; release the tables' scratch on every exit path.

        The scratch cleanup must run on success, on an exception and on a
        KeyboardInterrupt alike: a route that released its temp directory
        on the success path only leaked 79.5 GiB into one user's system
        temp before a manual sweep.
        """
        try:
            return super().convert()
        finally:
            self._release_table_scratch()

    def build_uns_metadata(self) -> Dict[str, Any]:
        """The provenance block every write path must persist, identically.

        Single source of truth for what lands in the table's ``uns``:

        - ``essential_metadata`` -- dimensions, mass range, source path,
          spectrum type and the Thyra version that wrote the store.
        - ``format_specific`` -- vendor metadata (FlexImaging areas,
          teaching points, imzML file mode, ...).
        - ``acquisition_params`` / ``instrument_info`` -- when the reader
          has them.
        - ``raw_metadata`` -- the source metadata as read.
        - ``regions`` -- the acquisition region summary, as JSON.

        This exists because the converters once had two write paths and
        they drifted. The in-memory converters handed the table to
        ``anndata``'s writer, which serialises whatever is in
        ``adata.uns``; the streaming path hand-wrote the Zarr layout and
        composed its own, much smaller block -- with ``spectrum_type``
        hardcoded to ``"processed"``, which is not even a value the
        extractors produce. Routing was on a size threshold at the time,
        so a dataset large enough to reach the streaming path came out
        claiming a spectrum representation it did not have, and without
        any of the other sections, while a slightly smaller one from the
        same instrument came out complete. There is one write path now,
        through ``anndata``'s writer, and every table -- the summed one
        and its siblings -- renders this mapping, so a section added here
        reaches every store.

        Sections the reader has nothing for are omitted rather than
        written empty, so consumers can tell "not available from this
        format" from "available and empty".

        Returns:
            Mapping of ``uns`` key to the value to store. Empty if the
            reader cannot produce comprehensive metadata at all.
        """
        try:
            comp_meta = self.reader.get_comprehensive_metadata()
        except Exception as e:
            # Non-fatal: a store without provenance still holds the
            # spectra. But it is not a debug-level event -- the whole
            # point of the block is that a consumer can say where the
            # data came from, so losing it has to be visible in the log.
            logger.warning("Could not read metadata for uns provenance: %s", e)
            return {}

        uns: Dict[str, Any] = {}
        try:
            self._collect_essential_metadata(uns, comp_meta)
            self._collect_optional_sections(uns, comp_meta)
            self._collect_region_info(uns)
        except Exception as e:
            logger.warning("Could not build the full uns provenance block: %s", e)

        self._collect_msi_metadata_block(uns, comp_meta)
        self._collect_mobility_axis(uns)
        self._collect_mobility_heatmap(uns)
        self._collect_msms_schedule(uns)

        return uns

    def _fragmentation(self) -> Any:
        """The reader's fragmentation schedule, read once and cached.

        ``None`` when the reader cannot say. Asked for through the base
        reader contract, so a format that learns to report it later needs
        no change here.
        """
        if not self._fragmentation_read:
            self._fragmentation_read = True
            # getattr, not a direct call: a reader predating this part of
            # the contract simply has nothing to say, which is the same
            # answer as ``None`` and not worth a warning.
            describe = getattr(self.reader, "get_fragmentation", None)
            if callable(describe):
                try:
                    self._fragmentation_schedule = describe()
                except Exception as e:  # pragma: no cover - reader-defined
                    logger.warning("Could not describe the fragmentation: %s", e)
                    self._fragmentation_schedule = None
            self._warn_if_precursors_merge()
        return self._fragmentation_schedule

    def _fragmentation_report(self) -> Any:
        """The schedule in the shape the schema builder reads, or ``None``."""
        schedule = self._fragmentation()
        return None if schedule is None else schedule.to_extractor_report()

    def _warn_if_precursors_merge(self) -> None:
        """Say out loud when a stored spectrum sums several precursors.

        The stored spectrum of such a pixel holds fragments of every
        precursor the frame isolated, with nothing marking which came
        from which. That is not visible in the output -- it looks like an
        ordinary spectrum -- so it is said once, at WARNING, rather than
        left for a reader of the peaks to work out.
        """
        schedule = self._fragmentation_schedule
        if schedule is None or not schedule.merges_precursors:
            return
        targets = ", ".join(f"{w.target:g}" for w in schedule.windows[:6])
        if len(schedule.windows) > 6:
            targets += ", ..."
        logger.warning(
            "This acquisition isolates %d precursors per pixel (%s). Thyra "
            "sums them into one spectrum per pixel, so the stored spectrum "
            "holds fragments of all of them and cannot be attributed to a "
            "single precursor. The schedule is recorded in "
            "uns['msms_schedule'].",
            len(schedule.windows),
            targets,
        )

    def _collect_msms_schedule(self, uns: Dict[str, Any]) -> None:
        """Add ``msms_schedule`` when the source fragmented anything.

        Written on the summed MSI table so a consumer can tell fragment
        m/z from intact m/z, and see which precursors a chimeric spectrum
        merges. Kept out of the versioned ``msi_metadata`` block for the
        same reason ``mobility_axis`` is: that block is versioned, this
        one carries arrays.
        """
        schedule = self._fragmentation()
        if schedule is None or not schedule.is_msms:
            return
        block = schedule.to_uns()
        if self._msms_table_key is not None:
            block["resolved_table"] = self._msms_table_key
        uns["msms_schedule"] = _jsonify_string_lists(self._serialize_for_zarr(block))

    def _collect_mobility_axis(self, uns: Dict[str, Any]) -> None:
        """Add ``mobility_axis`` when the source has a mobility dimension.

        Written on the summed MSI table so a consumer can tell "summed over
        mobility" from "never had any", and so it can find the
        mobility-resolved sibling (``resolved_table``) when one was written.
        Kept out of the ``msi_metadata`` schema block: that block is
        versioned, this one carries arrays.
        """
        try:
            if not getattr(self.reader, "has_ion_mobility", False):
                return
            axis = self.reader.get_mobility_axis()
        except Exception as e:  # pragma: no cover - reader-defined
            logger.warning("Could not describe the mobility axis: %s", e)
            return
        if axis is None:
            return
        block = axis.to_uns()
        if self._mobility_table_key is not None:
            block["resolved_table"] = self._mobility_table_key
        uns["mobility_axis"] = _jsonify_string_lists(self._serialize_for_zarr(block))

    def _collect_mobility_heatmap(self, uns: Dict[str, Any]) -> None:
        """Add ``mobility_heatmap`` when the source has a mobility dimension.

        The mean (m/z, mobility) frame of the whole dataset, the surface
        a consumer looks at to decide whether mobility separates anything
        before asking for a mobility-resolved table. Arrays, not lists,
        and plain-name keys; nothing in it is per pixel.
        """
        block = self._ensure_mobility_heatmap()
        if block is not None:
            uns["mobility_heatmap"] = block

    def _ensure_mobility_heatmap(self) -> Optional[Dict[str, Any]]:
        """Build the heatmap the first time it is asked for; cache the result.

        Needs the common mass axis, so it can only run after
        ``_initialize_conversion``. A failure is logged and leaves the
        summed table untouched: the block is additive, and a store
        without it is still complete.
        """
        if self._mobility_heatmap_built:
            return self._mobility_heatmap_block
        self._mobility_heatmap_built = True
        if not self._mobility_heatmap_enabled:
            return None
        try:
            if not getattr(self.reader, "has_ion_mobility", False):
                return None
        except Exception as e:  # pragma: no cover - reader-defined
            logger.warning("Could not inspect the mobility axis: %s", e)
            return None
        if self._common_mass_axis is None:
            logger.warning(
                "No mass-mobility heatmap: the common mass axis is not built yet"
            )
            return None
        from .mobility_heatmap import build_mobility_heatmap

        try:
            self._mobility_heatmap_block = build_mobility_heatmap(
                self.reader,
                self._common_mass_axis,
                n_spectra=self._get_total_spectra_count(),
            )
        except Exception as e:
            logger.error("Could not build the mass-mobility heatmap: %s", e)
            self._mobility_heatmap_block = None
        return self._mobility_heatmap_block

    def _plan_mobility_table(self, table_key: str) -> Optional[str]:
        """The key of the mobility table this slice gets, or ``None``.

        Decided before the MSI table's ``uns`` is built so the two agree,
        which for the grid route also means the grid itself is resolved
        here: the summed table's metadata block names it, and the sibling
        must be binned onto the very grid that was named.

        Two mechanisms can fill the same key -- a shared feature axis
        (nothing to decide) or a common grid (opt in) -- and a source that
        allows neither gets no table, said by name rather than silently.
        """
        if not self._write_mobility_table:
            return None
        try:
            if not getattr(self.reader, "has_ion_mobility", False):
                return None
            shared = bool(self.reader.has_shared_mobility_axis)
        except Exception as e:  # pragma: no cover - reader-defined
            logger.warning("Could not inspect the mobility axis: %s", e)
            return None
        from .mobility_table import mobility_table_key

        if shared:
            return mobility_table_key(table_key)
        if not self._mobility_grid_enabled:
            logger.info(
                "No mobility-resolved table: %s carries mobility per pixel "
                "rather than as a shared feature axis. Pass --mobility-grid "
                "to bin it onto a common mobility grid.",
                type(self.reader).__name__,
            )
            return None
        if not self._mobility_grid_resolved:
            self._mobility_grid_resolved = True
            self._mobility_grid = self._resolve_mobility_grid()
        if self._mobility_grid is None:
            return None
        return mobility_table_key(table_key)

    def _resolve_mobility_grid(self) -> Optional[MobilityGrid]:
        """The common mobility grid this conversion bins onto, or ``None``.

        Bounds come from the axis values unless the caller overrode them:
        the per-scan 1/K0 of a real file overhangs its declared
        acquisition range, and the mass-mobility heatmap already bins over
        the values, so anything else breaks the index-for-index mapping
        between the two. Every refusal is said by name.
        """
        from .mobility_table import (
            MAX_GRID_VAR_ENTRIES,
            grid_refusal,
            grid_var_bound,
            mobility_grid_range,
        )

        if self._common_mass_axis is None:
            logger.warning(
                "No mobility-resolved table: the common mass axis is not "
                "built yet, so the grid's m/z bins are unknown"
            )
            return None
        try:
            measured = mobility_grid_range(self.reader)
        except Exception as e:  # pragma: no cover - reader-defined
            logger.warning("Could not read the mobility axis values: %s", e)
            return None
        lower, upper = self._mobility_bounds
        if measured is None and (lower is None or upper is None):
            logger.warning(
                "No mobility-resolved table: the source's mobility axis "
                "carries no per-scan values to bin over (a reader opened "
                "without its vendor library cannot supply them). Give "
                "--mobility-min and --mobility-max to bin over a stated range."
            )
            return None
        span = measured or (0.0, 0.0)
        try:
            grid = build_mobility_grid(
                span[0] if lower is None else float(lower),
                span[1] if upper is None else float(upper),
                self._mobility_bins,
            )
        except ValueError as e:
            logger.warning("No mobility-resolved table: %s", e)
            return None
        refusal = grid_refusal(self.reader, self._common_mass_axis, grid)
        if refusal is not None:
            logger.warning("No mobility-resolved table: %s", refusal)
            return None
        unit = None
        axis = self.reader.get_mobility_axis()
        if axis is not None and axis.unit_name:
            unit = str(axis.unit_name)
        logger.info(
            "Mobility grid: %d %s channels over [%.5f, %.5f]%s",
            grid.n_channels,
            grid.law,
            grid.lower,
            grid.upper,
            "" if unit is None else f" {unit}",
        )
        report_channel_width(grid)
        bound = grid_var_bound(self._common_mass_axis, grid)
        if bound > MAX_GRID_VAR_ENTRIES:
            # A bound above the ceiling settles nothing -- real occupancy
            # runs an order of magnitude below it -- so it is said and the
            # source is read; the count decides (see var_ceiling_refusal).
            logger.info(
                "The mobility grid spans %s (m/z bin, channel) pairs, above "
                "the var ceiling of %s. Most of them will be empty; the "
                "table is refused only if the pairs that carry signal pass "
                "the ceiling too.",
                f"{bound:,}",
                f"{MAX_GRID_VAR_ENTRIES:,}",
            )
        return grid

    def _prepare_sibling_scans(
        self, obs: pd.DataFrame, z_value: Optional[int] = None
    ) -> None:
        """Run the raw mobility pass once for everything that needs it.

        Called by a finalize step right after the sibling tables are
        planned and before the summed table's ``uns`` is built, with the
        ``obs`` the siblings will mirror. When a grid table is planned its
        discovery pass -- which occupied cells there are, and how many
        rows each holds -- is fused into the heatmap's pass, so the two
        share one read *and* one mapping of every point onto the mass
        axis; the mapping is the larger cost of the two. The heatmap is
        built here once and cached for every ``uns`` block that asks; a
        route that never calls this still gets it from
        :meth:`_ensure_mobility_heatmap` on first demand.

        A no-op when the sinks were already fed from the summed table's
        own passes (a reader that hands its frames over as records; see
        ``fused_passes.py``).
        """
        if self._sibling_scans_done:
            return
        self._grid_discovery = None
        try:
            if not getattr(self.reader, "has_ion_mobility", False):
                return
        except Exception as e:  # pragma: no cover - reader-defined
            logger.warning("Could not inspect the mobility axis: %s", e)
            return
        if self._common_mass_axis is None:
            return
        from .mobility_heatmap import finish_mobility_heatmap, scan_mobility

        heatmap = self._pending_heatmap()
        discovery = self._pending_grid_discovery(obs, z_value)
        sinks = [sink for sink in (heatmap, discovery) if sink is not None]
        if not sinks:
            return
        try:
            scan_mobility(
                self.reader,
                self._common_mass_axis,
                *sinks,
                n_spectra=self._get_total_spectra_count(),
                description=(
                    "Mobility heatmap + grid"
                    if discovery is not None
                    else "Mobility heatmap"
                ),
            )
        except Exception as e:
            logger.error("Could not scan the mobility spectra: %s", e)
            if heatmap is not None:
                self._mobility_heatmap_built = True
                self._mobility_heatmap_block = None
            return
        if heatmap is not None:
            self._mobility_heatmap_built = True
            self._mobility_heatmap_block = finish_mobility_heatmap(heatmap)
        if discovery is not None:
            discovery.finish()
            self._grid_discovery = discovery

    def _pending_heatmap(self) -> Any:
        """An empty heatmap accumulator, when one is wanted and not yet built."""
        if not self._mobility_heatmap_enabled or self._mobility_heatmap_built:
            return None
        from .mobility_heatmap import prepare_mobility_heatmap

        return prepare_mobility_heatmap(self.reader, self._common_mass_axis)

    def _pending_grid_discovery(self, obs: pd.DataFrame, z_value: Optional[int]) -> Any:
        """The grid's discovery accumulator, when a grid table is planned."""
        if self._mobility_table_key is None or self._mobility_grid is None:
            return None
        from .mobility_table import GridDiscovery, row_lookup

        try:
            return GridDiscovery(
                self._common_mass_axis,
                self._mobility_grid,
                row_lookup(obs, z_value, None),
                int(len(obs)),
            )
        except MemoryError as e:
            logger.warning("No mobility-resolved table: %s", e)
            return None

    def _new_sibling_scratch(self, prefix: str) -> Path:
        """A scratch directory for one sibling's memmaps, next to the output."""
        from .csc_assembly import scratch_directory

        return scratch_directory(f".thyra_{prefix}_", parent=self.output_path.parent)

    def _register_table_scratch(self, prefix: str, assembly: Any) -> Path:
        """A scratch directory for ``assembly``, released with the others once written."""
        scratch = self._new_sibling_scratch(prefix)
        self._table_scratch.append((assembly, scratch))
        return scratch

    def _fused_sibling_passes(self, table_key: str) -> Any:
        """The sibling sinks to feed from the summed table's own passes, or ``None``.

        Only for a reader that hands its frames over as records
        (:attr:`~thyra.core.base_reader.BaseMSIReader.has_frame_scans`);
        plans the siblings of ``table_key`` first, since the sinks are
        theirs. ``None`` when nothing wants the frames, in which case the
        passes read the summed spectra as they always did and
        :meth:`_prepare_sibling_scans` scans on its own later.
        """
        if not getattr(self.reader, "has_frame_scans", False):
            return None
        if self._common_mass_axis is None or self._dimensions is None:
            return None
        self._mobility_table_key = self._plan_mobility_table(table_key)
        self._msms_table_key = self._plan_msms_table(table_key)
        self._siblings_planned = True
        from .fused_passes import SiblingPasses
        from .mobility_table import GridDiscovery
        from .msms_table import new_msms_accumulator

        n_x, n_y, n_z = self._dimensions
        n_grid = int(n_x * n_y * n_z)
        heatmap = self._pending_heatmap()
        discovery = None
        if self._mobility_table_key is not None and self._mobility_grid is not None:
            try:
                # Rows are handed to the sinks by the passes themselves,
                # so the lookup a standalone pass would use is not needed.
                discovery = GridDiscovery(
                    self._common_mass_axis,
                    self._mobility_grid,
                    lambda coords: None,
                    n_grid,
                )
            except MemoryError as e:
                logger.warning("No mobility-resolved table: %s", e)
        msms = None
        if self._msms_table_key is not None:
            msms = new_msms_accumulator(self.reader, self._common_mass_axis, n_grid)
        if heatmap is None and discovery is None and msms is None:
            return None
        self._sibling_scans_done = True
        return SiblingPasses(
            self._common_mass_axis, heatmap=heatmap, discovery=discovery, msms=msms
        )

    def _take_fused_results(self, passes: Any) -> None:
        """Keep what the fused passes built for the finalize step to write."""
        if passes.heatmap_wanted:
            self._mobility_heatmap_built = True
            self._mobility_heatmap_block = passes.heatmap_block
        self._grid_discovery = passes.discovery
        self._msms_accumulator = passes.msms

    def _release_table_scratch(self, tables: Optional[Dict[str, Any]] = None) -> None:
        """Drop every table's memmaps and remove their scratch directories.

        Called once the store is written. ``tables`` is the mapping that
        still holds the tables; it is emptied first, because a mapped file
        cannot be deleted on Windows and the AnnData is what keeps it
        mapped. Idempotent, so the ``finally`` of ``convert`` can call it
        too.
        """
        from .csc_assembly import remove_scratch

        if not self._table_scratch:
            return
        if tables is not None:
            tables.clear()
        pending = self._table_scratch
        self._table_scratch = []
        for assembly, path in pending:
            if assembly is not None:
                assembly.release()
            remove_scratch(path)

    def _attach_sibling_tables(
        self,
        data_structures: Dict[str, Any],
        table_key: str,
        region_key: str,
        obs: pd.DataFrame,
        z_value: Optional[int] = None,
    ) -> None:
        """Build the sibling tables of ``table_key`` and add them.

        No-op for a sibling :meth:`_plan_mobility_table` or
        :meth:`_plan_msms_table` did not name for this slice. A failure
        is logged and leaves the summed table untouched: the siblings are
        additive, and a store without them is still complete.
        """
        if self._common_mass_axis is None:
            return
        if self._mobility_table_key is None and self._msms_table_key is None:
            return
        # The siblings carry the same provenance as the summed table,
        # minus the heatmap: that block is the summed table's navigator
        # over the very data the siblings hold resolved or split.
        sibling_uns = self.build_uns_metadata()
        sibling_uns.pop("mobility_heatmap", None)
        summed = data_structures["tables"].get(table_key)
        if self._mobility_table_key is not None:
            table = self._build_mobility_sibling(
                obs, table_key, region_key, dict(sibling_uns), z_value
            )
            if table is not None:
                if self._mobility_grid is not None:
                    self._record_mobility_marginal(table, summed, table_key)
                data_structures["tables"][self._mobility_table_key] = table
        if self._msms_table_key is not None:
            table = self._build_msms_sibling(
                obs, table_key, region_key, dict(sibling_uns), z_value
            )
            if table is not None:
                self._record_demultiplexed_current(table, summed, table_key)
                data_structures["tables"][self._msms_table_key] = table

    def _build_mobility_sibling(
        self,
        obs: pd.DataFrame,
        table_key: str,
        region_key: str,
        uns: Dict[str, Any],
        z_value: Optional[int],
    ) -> Optional[Any]:
        """The mobility sibling of one slice, on a scratch directory of its own."""
        from .mobility_table import build_mobility_table

        discovery, self._grid_discovery = self._grid_discovery, None
        # The fused passes allocate and register their own scratch; a
        # discovery without one is scattered by the builder on a new one.
        scratch = None if discovery is None else discovery.scratch
        if scratch is None:
            scratch = self._new_sibling_scratch("mobility")
            self._table_scratch.append(
                (None if discovery is None else discovery.assembly, scratch)
            )
        try:
            return build_mobility_table(
                self.reader,
                obs,
                self._common_mass_axis,
                table_key,
                region_key,
                uns,
                z_value=z_value,
                grid=self._mobility_grid,
                discovery=discovery,
                scratch=scratch,
            )
        except Exception as e:
            logger.error("Could not build the mobility-resolved table: %s", e)
            return None

    def _build_msms_sibling(
        self,
        obs: pd.DataFrame,
        table_key: str,
        region_key: str,
        uns: Dict[str, Any],
        z_value: Optional[int],
    ) -> Optional[Any]:
        """The demultiplexed sibling of one slice, on a scratch directory of its own."""
        from .msms_table import build_msms_table

        accumulator, self._msms_accumulator = self._msms_accumulator, None
        scratch = None if accumulator is None else accumulator.scratch
        if scratch is None:
            scratch = self._new_sibling_scratch("msms")
            self._table_scratch.append((None, scratch))
        try:
            return build_msms_table(
                self.reader,
                obs,
                self._common_mass_axis,
                table_key,
                region_key,
                uns,
                z_value=z_value,
                scratch=scratch,
                accumulator=accumulator,
            )
        except Exception as e:
            logger.error("Could not build the demultiplexed MS/MS table: %s", e)
            return None

    @staticmethod
    def _record_mobility_marginal(table: Any, summed: Any, summed_key: str) -> None:
        """Say how far the grid table's marginal is from the summed table.

        Summing a grid table's channels within one m/z bin must reproduce
        that bin's column of the summed table: both are the same points,
        binned the same way, differing only in whether mobility was kept.
        Under ``--tdf-spectrum scan_sum`` that holds exactly; under the
        vendor centroid it cannot, because the centroid is a peak-picked
        spectrum over the same ramp and keeps only the current inside the
        peaks it picks (87-96% on measured acquisitions) while the grid
        reads raw scans. A store whose two tables disagree
        must say by how much rather than leave a reader to find it by
        subtraction.

        The comparison is per pixel: the grid table's row sums against
        the summed table's, which is one bounded pass over each memmap.
        The per-cell deviation the in-memory converters used to record
        needed the marginal and its difference from the summed table
        materialised, each the size of the summed table, and went with
        them (design decision D11); the current ratio is the same number
        it always was.
        """
        try:
            block = _current_ratio_block(table, summed_key, summed)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Could not compare the mobility marginal: %s", e)
            return
        if block is None:
            return
        table.uns["mobility_marginal"] = block
        exact = abs(float(block["current_ratio"]) - 1.0) <= _MARGINAL_TOLERANCE
        log = logger.info if exact else logger.warning
        log(
            "The mobility grid table holds %.4fx the summed table's ion "
            "current (per pixel %.4f to %.4f). They agree exactly only "
            "under --tdf-spectrum scan_sum; the vendor centroid is a "
            "peak-picked spectrum over the same scans and keeps less of "
            "the current.",
            block["current_ratio"],
            block["current_ratio_pixel_min"],
            block["current_ratio_pixel_max"],
        )

    def _plan_msms_table(self, table_key: str) -> Optional[str]:
        """The key of the demultiplexed MS/MS table this slice gets, or ``None``.

        Decided before the MSI table's ``uns`` is built so the two agree.
        Every refusal is said by name: writing no table is the right
        answer for an acquisition whose precursors cannot be told apart,
        but silently writing none is not.
        """
        if not self._write_msms_table:
            return None
        from .msms_table import demultiplex_refusal, msms_table_key

        if not callable(getattr(self.reader, "iter_precursor_spectra", None)):
            logger.info(
                "No demultiplexed MS/MS table: %s cannot separate the "
                "precursors of a pixel",
                type(self.reader).__name__,
            )
            return None
        refusal = demultiplex_refusal(self._fragmentation())
        if refusal is not None:
            logger.info("No demultiplexed MS/MS table: %s", refusal)
            return None
        # The fragment axis is the summed table's axis whatever that axis
        # is (design decision D6): resampled, it is the grid the user chose
        # for the whole store; raw, it is the union of the very fragment
        # m/z values the split re-reads, so the mapping is exact either way.
        return msms_table_key(table_key)

    @staticmethod
    def _record_demultiplexed_current(table: Any, summed: Any, summed_key: str) -> None:
        """Say how much of the summed table's ion current the split holds.

        The two tables agree exactly under ``--tdf-spectrum scan_sum``,
        which writing this table now selects; under an explicit
        ``vendor_centroid`` they do not, because the vendor peak picker
        drops the index bins it assigns to no peak while the split reads
        raw scans, so the
        demultiplexed table holds *more*. That is not a defect, but a
        store whose two tables disagree must say so rather than leave a
        reader to find it by subtraction.
        """
        try:
            block = _current_ratio_block(table, summed_key, summed)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Could not compare the demultiplexed current: %s", e)
            return
        if block is None:
            return
        table.uns["demultiplexed_current"] = block
        if abs(float(block["current_ratio"]) - 1.0) > _MARGINAL_TOLERANCE:
            logger.info(
                "The demultiplexed table holds %.4fx the summed table's ion "
                "current (per pixel %.4f to %.4f). Above 1 under an explicit "
                "--tdf-spectrum vendor_centroid, which discards counts the "
                "raw scans keep; below 1 when the schedule's windows do not "
                "cover every scan that carries current.",
                block["current_ratio"],
                block["current_ratio_pixel_min"],
                block["current_ratio_pixel_max"],
            )

    def _resolved_pixel_size_xy(self) -> Tuple[float, float]:
        """The in-plane pixel pitch as ``(x_um, y_um)``.

        The converter itself carries a single float (auto-detection
        keeps the source's x pitch), but the detection info still has
        the true per-axis values for anisotropic rasters -- the
        metadata block records those, since it describes the
        acquisition rather than the rendering.
        """
        info = self._pixel_size_detection_info or {}
        if "detected_x_um" in info and "detected_y_um" in info:
            return (float(info["detected_x_um"]), float(info["detected_y_um"]))
        return (float(self.pixel_size_um), float(self.pixel_size_um))

    def _collect_msi_metadata_block(self, uns: Dict[str, Any], comp_meta: Any) -> None:
        """Add the versioned ``msi_metadata`` schema block.

        Built in its own try so a schema failure cannot take the other
        provenance sections down with it.  See docs/metadata-schema.md
        for the storage contract and ``thyra validate`` for the
        consumer side.
        """
        from thyra.metadata.schema import MSI_METADATA_UNS_KEY, build_msi_metadata

        try:
            info = self._pixel_size_detection_info or {}
            meta = build_msi_metadata(
                comp_meta,
                pixel_size_um=self._resolved_pixel_size_xy(),
                pixel_size_source=self.pixel_size_source.value,
                source_format=info.get("source_format"),
                processing=self._processing_provenance(),
                mobility_resolved_table=self._mobility_table_key,
                mobility_grid=(
                    None
                    if self._mobility_grid is None
                    else self._mobility_grid.to_schema_report()
                ),
                fragmentation=self._fragmentation_report(),
                msms_resolved_table=self._msms_table_key,
            )
            uns[MSI_METADATA_UNS_KEY] = meta.to_uns_dict()
        except Exception as e:
            logger.warning("Could not build the msi_metadata block: %s", e)

    def _processing_provenance(self) -> List[Any]:
        """The processing steps this conversion performed, oldest first.

        Modeled on mzQC provenance.  The list describes what was done to
        the data, so nothing about how the store was written -- the sparse
        layout, the number of passes -- belongs here.
        """
        from thyra import __version__
        from thyra.metadata.schema import ProcessingStep, SoftwareRef

        thyra_ref = SoftwareRef(name="thyra", version=__version__)
        conversion_parameters: Dict[str, Any] = {}
        # A TDF frame's mobility ramp is collapsed into one spectrum, and
        # the two correct ways to do that differ in TIC by up to a fifth
        # (vendor centroid vs. lossless scan sum), so the store must say
        # which one it holds.
        tdf_spectrum = getattr(self.reader, "tdf_spectrum", None)
        if (
            tdf_spectrum is not None
            and getattr(self.reader, "file_type", None) == "tdf"
        ):
            conversion_parameters["tdf_spectrum"] = str(tdf_spectrum)
        steps = [
            ProcessingStep(
                name="conversion",
                software=thyra_ref,
                parameters=conversion_parameters,
            )
        ]

        config = self._resampling_config
        if config is not None:
            parameters: Dict[str, Any] = {}
            for field_name, value in vars(config).items():
                if value is None:
                    continue
                parameters[field_name] = getattr(value, "value", value)
            # The resolved plan wins over the requested config: with
            # "auto" settings the config says nothing about the method
            # and axis the decision tree actually picked, and the step
            # must declare what was done, not what was asked for.
            for field_name, value in (self._resolved_resampling_plan or {}).items():
                if value is None:
                    continue
                parameters[field_name] = getattr(value, "value", value)
            steps.append(
                ProcessingStep(
                    name="mass axis resampling",
                    software=thyra_ref,
                    parameters=parameters,
                )
            )
        return steps

    def _collect_essential_metadata(self, uns: Dict[str, Any], comp_meta: Any) -> None:
        """Add ``essential_metadata`` (tuples become lists for Zarr)."""
        essential = getattr(comp_meta, "essential", None)
        if essential is None:
            return

        dims = essential.dimensions
        mrange = essential.mass_range
        from thyra import __version__

        uns["essential_metadata"] = {
            "source_path": str(essential.source_path),
            "dimensions": list(dims) if dims else None,
            "mass_range": list(mrange) if mrange else None,
            "spectrum_type": getattr(essential, "spectrum_type", None),
            "thyra_version": __version__,
        }

    def _collect_optional_sections(self, uns: Dict[str, Any], comp_meta: Any) -> None:
        """Add the vendor sections the reader actually populated.

        Lists that are not purely numeric (imzML ``cvParams``, or any
        string list a vendor extractor reports) are stored as JSON
        strings -- see :func:`_jsonify_string_lists` for why letting
        them reach the writer as lists corrupts them and crashes
        readers on numpy 2.1-2.2.
        """
        for key in ("format_specific", "acquisition_params", "instrument_info"):
            value = getattr(comp_meta, key, None)
            if value:
                uns[key] = _jsonify_string_lists(self._serialize_for_zarr(value))

        raw_metadata = getattr(comp_meta, "raw_metadata", None)
        if raw_metadata:
            uns["raw_metadata"] = _jsonify_string_lists(
                self._serialize_for_zarr(raw_metadata)
            )

    def _collect_region_info(self, uns: Dict[str, Any]) -> None:
        """Add the acquisition region summary as JSON.

        Stored as a JSON string because AnnData/zarr cannot round-trip
        a list of dicts (they get stringified individually). JSON
        preserves the structure and can be parsed with json.loads().

        Always written for a consistent schema. Single-region datasets
        get a single-entry list with region_number=1.
        """
        region_info = getattr(self, "_region_info", None)
        if region_info:
            uns["regions"] = json.dumps(self._serialize_for_zarr(region_info))

    def _add_metadata_to_uns(self, adata) -> None:
        """Apply :meth:`build_uns_metadata` to an AnnData about to be written."""
        uns = self.build_uns_metadata()
        adata.uns.update(uns)
        logger.debug("Added MSI metadata to AnnData .uns: %s", sorted(uns))

    def _serialize_for_zarr(self, obj):
        """Recursively convert tuples to lists for Zarr serialization.

        Dict keys are coerced to strings: Zarr group members must be named,
        and a non-string key otherwise fails at write time, after the whole
        conversion has already been done.
        """
        if isinstance(obj, dict):
            return {str(k): self._serialize_for_zarr(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_zarr(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            # Convert dataclass/object to dict
            return self._serialize_for_zarr(vars(obj))
        else:
            return obj

    def _calculate_bins_from_width(
        self, min_mz: float, max_mz: float, axis_type
    ) -> int:
        """Calculate optimal number of bins from desired width at reference m/z.

        Args:
            min_mz: Minimum m/z of the mass range
            max_mz: Maximum m/z of the mass range
            axis_type: The axis type (determines physics-based spacing)

        Returns:
            Calculated number of bins
        """
        # Calculate bins based on axis type physics
        if hasattr(axis_type, "value"):
            axis_name = axis_type.value
        else:
            axis_name = str(axis_type).split(".")[-1].lower()

        width_at_mz, reference_mz = _reference_params(self, axis_name)

        logger.info(
            f"Calculating bins for {width_at_mz*1000:.1f} mDa width at m/z {reference_mz:.1f}"
        )

        if axis_name == "reflector_tof":
            # REFLECTOR_TOF: constant relative resolution (width ∝ m/z)
            # relative_resolution = reference_mz / width_at_mz
            # For logarithmic spacing: bins ≈ ln(max_mz/min_mz) * (reference_mz / width_at_mz)
            relative_resolution = reference_mz / width_at_mz
            bins = int(np.log(max_mz / min_mz) * relative_resolution)

        elif axis_name == "tof":
            # TOF: bin width = sqrt(A m + B m^2) / k. The generator carries
            # the closed-form integral of 1 / width, so the count is exact.
            a, b, k = _tof_plan(self)
            bins = TOFAxisGenerator(a, b).bin_count(min_mz, max_mz, k)
            logger.info(
                "TOF width law A=%.4g mDa^2/Da, B=%.4g, %.2f bins per FWHM",
                a,
                b,
                k,
            )

        elif axis_name == "linear_tof":
            # LINEAR_TOF: bin width = k * sqrt(m/z), where k = width_at_mz / sqrt(reference_mz)
            # Number of bins: n = (2/k) * (sqrt(max_mz) - sqrt(min_mz))
            # This matches SCiLS Lab's "Linear TOF" mass axis calculation
            k = width_at_mz / np.sqrt(reference_mz)
            bins = int((2.0 / k) * (np.sqrt(max_mz) - np.sqrt(min_mz)))

        elif axis_name == "orbitrap":
            # ORBITRAP: bin_width = k * (m/z)^1.5 with k = width_at_mz /
            # reference_mz^1.5. Integrating dm / w(m) over the range gives
            #   bins = 2 * (1/sqrt(min_mz) - 1/sqrt(max_mz)) *
            #          (reference_mz^1.5 / width_at_mz)
            # The factor 2 comes from d(m^-0.5)/dm = -1/2 * m^-1.5 and was
            # missing, which halved the bin count and made every bin twice
            # the requested width.
            scaling_factor = (reference_mz**1.5) / width_at_mz
            bins = int(2 * (1 / np.sqrt(min_mz) - 1 / np.sqrt(max_mz)) * scaling_factor)

        elif axis_name == "fticr":
            # FTICR: width ∝ m/z^2, i.e. bin_width = k * (m/z)^2 with
            # k = width_at_mz / reference_mz^2. FTICRAxisGenerator lays the
            # axis out uniformly in 1/mz, so the bin count is the 1/mz span
            # divided by the step k:
            #   bins = (1/min_mz - 1/max_mz) * (reference_mz^2 / width_at_mz)
            # Without this branch an FT-ICR axis took its bin count from the
            # uniform formula below, so the realized width at reference_mz was
            # not the width that was asked for.
            scaling_factor = (reference_mz**2) / width_at_mz
            bins = int((1 / min_mz - 1 / max_mz) * scaling_factor)

        else:
            # LINEAR/CONSTANT: uniform spacing
            # bins = (max_mz - min_mz) / width_at_mz
            bins = int((max_mz - min_mz) / width_at_mz)

        # Ensure minimum bin count
        bins = max(100, bins)

        logger.info(f"Calculated {bins} bins for {axis_name} axis type")
        return bins

    def _get_reference_params(self, axis_type) -> Tuple[float, float]:
        """Get reference width and m/z for the given axis type.

        Explicit setting first, then the width the detected instrument
        asked for, then the axis-type default; see :func:`_reference_params`.
        """
        if hasattr(axis_type, "value"):
            axis_name = axis_type.value
        else:
            axis_name = str(axis_type).split(".")[-1].lower()
        return _reference_params(self, axis_name)

    def _resolve_resampling_plan(
        self,
    ) -> Tuple[float, float, Any, int]:
        """Resolve mass range, axis type, and target bin count for resampling.

        Single source of truth used by both ``_build_resampled_mass_axis`` and
        the size estimator. Caches essential metadata lazily.

        Returns:
            (min_mz, max_mz, axis_type, target_bins)
        """
        if self._essential_metadata_cached is None:
            self._essential_metadata_cached = self.reader.get_essential_metadata()

        mass_range = self._essential_metadata_cached.mass_range
        min_mz = mass_range[0] if self._min_mz is None else self._min_mz
        max_mz = mass_range[1] if self._max_mz is None else self._max_mz

        tree = ResamplingDecisionTree()
        if hasattr(self, "_manual_axis_type") and self._manual_axis_type is not None:
            axis_type = self._manual_axis_type
        else:
            metadata = self._get_cached_metadata_for_resampling()
            axis_type = tree.select_axis_type(metadata)
            # A detector's width goes with the axis law it chose, so it is
            # consulted only on the auto path and only when the caller has
            # not set a width of their own.
            if self._width_at_mz is None:
                self._detected_reference_width = tree.select_reference_width(metadata)

        # A TOF axis without the caller's own coefficients takes the pair
        # the instrument declares -- on the auto path (an MRT centroid
        # conversion) and when --mass-axis-type tof was asked for by name
        # (a timsTOF opting in).
        if axis_type is AxisType.TOF and (
            getattr(self, "_tof_a", None) is None
            or getattr(self, "_tof_b", None) is None
        ):
            self._detected_tof_law = tree.select_tof_law(
                self._get_cached_metadata_for_resampling()
            )
        elif (
            axis_type is not AxisType.TOF and getattr(self, "_tof_a", None) is not None
        ):
            logger.warning(
                "--tof-law was given but the mass axis resolved to %s, which "
                "does not use a width law; it is ignored. Pass "
                "--mass-axis-type tof to use it.",
                getattr(axis_type, "value", axis_type),
            )

        if self._width_at_mz is not None or self._target_bins is None:
            target_bins = self._calculate_bins_from_width(min_mz, max_mz, axis_type)
        else:
            target_bins = self._target_bins

        return min_mz, max_mz, axis_type, target_bins

    def _build_resampled_mass_axis(self) -> None:
        """Build resampled mass axis using physics-based generators."""
        from ...resampling.common_axis import CommonAxisBuilder

        min_mz, max_mz, axis_type, target_bins = self._resolve_resampling_plan()

        # Determine reference parameters for physics generators
        reference_width, reference_mz = self._get_reference_params(axis_type)

        # Kept for provenance: the processing step must declare what was
        # actually done, and with "auto" settings the requested config
        # says nothing about the method, axis and bin width the decision
        # tree resolved to.  See _processing_provenance().
        self._resolved_resampling_plan = {
            "method": getattr(self, "_resampling_method", None),
            "axis_type": axis_type,
            "target_bins": target_bins,
            "min_mz": min_mz,
            "max_mz": max_mz,
            "mass_width_da": reference_width,
            "reference_mz": reference_mz,
        }
        tof_law: Optional[Tuple[float, float]] = None
        if axis_type is AxisType.TOF:
            a, b, k = _tof_plan(self)
            tof_law = (a, b)
            self._resolved_resampling_plan.update(
                {"tof_a": a, "tof_b": b, "bins_per_fwhm": k}
            )

        if hasattr(self, "_manual_axis_type") and self._manual_axis_type is not None:
            logger.info(f"Using manually specified axis type: {axis_type}")
        else:
            logger.info(f"Auto-detected axis type: {axis_type}")

        logger.info(
            f"Building resampled mass axis: {min_mz:.2f} - {max_mz:.2f} m/z, "
            f"{target_bins} bins"
        )

        # Build the physics-based axis
        builder = CommonAxisBuilder()

        if hasattr(axis_type, "value") and axis_type.value != "constant":
            # Use physics-based generator with reference parameters
            mass_axis = builder.build_physics_axis(
                min_mz=min_mz,
                max_mz=max_mz,
                num_bins=target_bins,
                axis_type=axis_type,
                reference_mz=reference_mz,
                reference_width=reference_width,
                tof_law=tof_law,
            )
            logger.info(
                f"Built physics-based {axis_type} mass axis with "
                f"{len(mass_axis.mz_values)} points"
            )
        else:
            # Fall back to uniform axis
            mass_axis = builder.build_uniform_axis(min_mz, max_mz, target_bins)
            logger.info(
                f"Built uniform mass axis with " f"{len(mass_axis.mz_values)} points"
            )

        # Override the parent's common mass axis
        self._common_mass_axis = mass_axis.mz_values.astype(np.float64)
        if self._common_mass_axis is None:
            raise RuntimeError("Common mass axis is None after assignment")

        # Calculate bin sizes for informative logging
        bin_widths = np.diff(self._common_mass_axis)
        min_bin_size = np.min(bin_widths) * 1000  # Convert to mDa
        max_bin_size = np.max(bin_widths) * 1000  # Convert to mDa

        logger.info(
            f"Resampled mass axis created: {len(self._common_mass_axis)} bins, "
            f"range {self._common_mass_axis[0]:.2f}-{self._common_mass_axis[-1]:.2f} m/z, "
            f"bin sizes {min_bin_size:.2f}-{max_bin_size:.2f} mDa ({axis_type})"
        )

    def _initialize_conversion(self) -> None:
        """Override parent initialization to preserve resampled mass axis."""
        logger.info("Loading essential dataset information...")
        try:
            # Load essential metadata first (fast, single query for Bruker)
            essential = self.reader.get_essential_metadata()
            # Cache for reuse during resampling setup
            self._essential_metadata_cached = essential

            self._dimensions = essential.dimensions
            if any(d <= 0 for d in self._dimensions):
                raise ValueError(
                    f"Invalid dimensions: {self._dimensions}. All dimensions "
                    f"must be positive."
                )

            # Store essential metadata for use throughout conversion
            self._coordinate_bounds = essential.coordinate_bounds
            self._n_spectra = essential.n_spectra
            self._estimated_memory_gb = essential.estimated_memory_gb

            # Override pixel size only if using default and metadata is available
            if (
                self.pixel_size_source == PixelSizeSource.DEFAULT
                and essential.pixel_size
            ):
                old_size = self.pixel_size_um
                self.pixel_size_um = essential.pixel_size[0]
                self.pixel_size_source = PixelSizeSource.AUTO_DETECTED
                logger.info(
                    f"Auto-detected pixel size: {self.pixel_size_um} um "
                    f"(was default: {old_size} um)"
                )
            elif self.pixel_size_source == PixelSizeSource.USER_PROVIDED:
                logger.info(f"Using user-specified pixel size: {self.pixel_size_um} um")

            # After pixel size, because the fallback is the pixel size.
            self._resolve_z_spacing(essential)

            # Handle mass axis setup
            self._setup_mass_axis()

            # Only load comprehensive metadata if needed (lazy loading)
            self._metadata = None  # Will be loaded on demand

            # Fetch region data from reader (available for multi-region datasets)
            # Always populate region metadata for a consistent schema:
            # - obs["region_number"] exists on every dataset
            # - uns["regions"] always has at least one entry
            # Single-region or unknown-region datasets default to region 1.
            self._region_map = self.reader.get_region_map()
            self._region_info = self.reader.get_region_info()
            if self._region_info:
                logger.info(
                    f"Region metadata available: {len(self._region_info)} regions"
                )
            else:
                self._region_info = [{"region_number": 1, "n_spectra": self._n_spectra}]

            # Compute optical alignment for FlexImaging data when
            # available.  The alignment data is computed REGARDLESS of
            # apply_optical_alignment so it can be used to place the
            # FlexImaging optical image relative to the MSI -- even
            # when a downstream tool (e.g. Ousia's wizard) is doing
            # its own MSI-to-target registration and does not want
            # Thyra to pre-rotate the MSI itself.  The flag only
            # controls whether MSI elements use the alignment; the
            # optical image always uses it (if available) to land in
            # the same "global" frame as the MSI.
            self._compute_optical_alignment()
            self._build_tic_to_image_affine()
            if not self._apply_optical_alignment:
                logger.info(
                    "apply_optical_alignment=False: MSI elements will "
                    "use micrometer coordinates; optical image (if "
                    "any) will use inverse-alignment to land in the "
                    "same micrometer frame."
                )

            logger.info(f"Dataset dimensions: {self._dimensions}")
            logger.info(f"Coordinate bounds: {self._coordinate_bounds}")
            logger.info(f"Total spectra: {self._n_spectra}")
            logger.info(f"Estimated memory: {self._estimated_memory_gb:.2f} GB")
            if self._common_mass_axis is None:
                raise RuntimeError("Common mass axis is None after initialization")
            logger.info(f"Common mass axis length: {len(self._common_mass_axis)}")
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    def _setup_mass_axis(self) -> None:
        """Set up the common mass axis (resampled or raw)."""
        config_status = "SET" if self._resampling_config else "NOT SET"
        logger.info(f"Mass axis mode: resampling_config={config_status}")
        if self._resampling_config:
            # Build resampled mass axis now that reader metadata is loaded
            logger.info(
                "Building RESAMPLED mass axis (resampling enabled) - "
                "will NOT iterate through all spectra"
            )
            self._build_resampled_mass_axis()
            if self._common_mass_axis is None:
                raise RuntimeError(
                    "Common mass axis is None after resampled axis build"
                )
            logger.info(
                f"Built resampled mass axis with " f"{len(self._common_mass_axis)} bins"
            )
        else:
            # No resampling - load raw mass axis as usual
            if self.reader.has_shared_mass_axis:
                logger.info(
                    "Loading RAW mass axis (no resampling) - "
                    "continuous mode, reading m/z from first spectrum only"
                )
            else:
                logger.warning(
                    "Building RAW mass axis (no resampling) - "
                    "processed mode, iterating ALL spectra to collect unique m/z values. "
                    "This is slow for large datasets!"
                )
            self._common_mass_axis = self.reader.get_common_mass_axis()
            if len(self._common_mass_axis) == 0:
                raise ValueError(
                    "Common mass axis is empty. Cannot proceed with conversion."
                )
            logger.info(
                f"Using raw mass axis with "
                f"{len(self._common_mass_axis)} unique m/z values"
            )

    @staticmethod
    def _coalesce_duplicate_bins(
        indices: NDArray[np.int_], intensities: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
        """Sum intensities that landed on the same axis bin within one spectrum.

        Without resampling every m/z maps to its own axis entry, so this is
        the identity for ordinary data. A spectrum that repeats an m/z value
        -- an ion mobility export lists a feature once per mobility -- maps
        two entries to one column, and writing both would leave a
        duplicate ``(row, col)`` in the sparse matrix (the scatter also
        requires a row's bins to be unique). Summing here is what the old
        in-memory route did through scipy's COO conversion, so a store
        reads back the same.
        """
        if indices.size < 2:
            return indices, intensities
        diffs = np.diff(indices)
        if bool(np.all(diffs > 0)):
            return indices, intensities
        unique, inverse = np.unique(indices, return_inverse=True)
        summed = np.bincount(
            np.asarray(inverse).ravel(),
            weights=np.asarray(intensities, dtype=np.float64),
            minlength=unique.size,
        )
        return unique.astype(indices.dtype, copy=False), summed

    def _count_out_of_range(self, n_dropped: int, n_total: int) -> None:
        """Record peaks discarded for lying outside the target mass axis.

        Narrowing the mass range is deliberate, so dropping the peaks
        outside it is the correct answer and not an error -- but it is not
        something a user should have to infer either, since the previous
        behaviour conserved the total exactly and so left no trace a TIC
        check could find.

        Warns once per conversion rather than once per spectrum: a
        narrowed range typically excludes peaks in every spectrum, and on
        xenium that is 918,855 identical lines. ``_out_of_range_peaks``
        keeps the running total; note it counts resample *calls*, and the
        converter resamples every spectrum twice (once per pass), so it
        is a lower bound on nothing and an upper bound on nothing -- read
        the warning, not the counter, if you want a per-spectrum figure.

        Args:
            n_dropped: Peaks outside the axis in this spectrum.
            n_total: Peaks in this spectrum before filtering.
        """
        self._out_of_range_peaks += n_dropped

        if self._out_of_range_warned or self._common_mass_axis is None:
            return
        self._out_of_range_warned = True
        logger.warning(
            "Dropping peaks that fall outside the target mass axis "
            "[%.4f, %.4f] m/z -- %d of %d in the first spectrum affected. "
            "They are discarded, not folded into the edge bins. Widen the "
            "resampling range to keep them.",
            float(self._common_mass_axis[0]),
            float(self._common_mass_axis[-1]),
            n_dropped,
            n_total,
        )

    def _nearest_neighbor_resample(
        self, mzs: NDArray[np.float64], intensities: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
        """Resample spectrum using nearest neighbor interpolation.

        Maps each m/z value to its nearest bin in the common mass axis and
        accumulates intensities. Returns only non-zero bins for efficiency.

        Peaks outside the axis are **dropped**, not folded into the edge
        bins. They used to be clipped to bin 0 or the last bin and then
        accumulated there, so narrowing the mass range -- the most ordinary
        thing ``--resample-min-mz`` / ``--resample-max-mz`` are for --
        piled everything below the floor onto the first bin and everything
        above the ceiling onto the last. On real ``pea.imzML`` resampled to
        400-800 m/z, bin 0 held 654,158 counts where a real peak there is
        around 80. The total was conserved exactly, so a TIC check could
        not see it; the peak was simply in the wrong place, 1,634x the
        median interior bin.

        "In range" is the strict axis span, ``[axis[0], axis[-1]]``: a peak
        is kept when the axis covers it, not when it is within half a bin of
        the end. That is the same rule ``_tic_preserving_resample`` already
        follows -- ``np.interp(..., left=0, right=0)`` and
        ``thyra.resampling.tic.preserved_tic`` both cut at the endpoints.

        Args:
            mzs: Original m/z values from spectrum
            intensities: Corresponding intensity values

        Returns:
            Tuple of (bin_indices, accumulated_intensities) containing only non-zero bins
        """
        if mzs.size == 0:
            return np.array([], dtype=np.int_), np.array([], dtype=np.float64)

        # Ensure common mass axis is initialized
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        axis = self._common_mass_axis

        # Shared-axis fast path: when every spectrum carries the same m/z
        # array (continuous imzML, Rapiflex, Waters, PHI), the peak-to-bin
        # mapping is a property of the axis pair, not of the spectrum. It is
        # computed once; each later spectrum only proves its m/z array is
        # the same one -- an exact array comparison, which is far cheaper
        # than re-deriving the mapping by binary search per spectrum.
        # ``False`` means the cache disabled itself (processed-mode data).
        # getattr rather than a bare read: test harnesses drive this method
        # unbound on a stub that predates the cache, and they exercise the
        # generic path below, which is exactly what "no cache" selects.
        if getattr(self, "_nn_shared_cache", False) is not False:
            result = self._nn_resample_via_cache(axis, mzs, intensities)
            if result is not None:
                return result

        in_range = (mzs >= axis[0]) & (mzs <= axis[-1])
        if not in_range.all():
            self._count_out_of_range(int(mzs.size - in_range.sum()), int(mzs.size))
            mzs = mzs[in_range]
            intensities = intensities[in_range]
            if mzs.size == 0:
                return np.array([], dtype=np.int_), np.array([], dtype=np.float64)

        return _nn_accumulate(_nn_map_to_bins(axis, mzs), intensities)

    def _build_nn_shared_cache(
        self, axis: NDArray[np.float64], mzs: NDArray[np.float64]
    ) -> Optional[_SharedAxisNNCache]:
        """Precompute the nearest-neighbor mapping for one m/z array.

        Returns None when the array cannot be cached -- empty, or not
        ascending, which the contiguous in-range slice below relies on.
        The mapping itself comes from the same :meth:`_nn_map_to_bins`
        the generic path uses, so a cache hit and a fresh computation
        agree bin for bin.
        """
        if mzs.size == 0 or not bool(np.all(mzs[1:] >= mzs[:-1])):
            return None

        # Ascending m/z makes the in-range subset one contiguous slice,
        # with the same inclusive endpoints as the generic path's mask.
        lo = int(np.searchsorted(mzs, axis[0], side="left"))
        hi = int(np.searchsorted(mzs, axis[-1], side="right"))

        cache = _SharedAxisNNCache()
        cache.key = mzs.copy()
        cache.key_ref = mzs
        cache.n_total = int(mzs.size)
        cache.n_dropped = int(mzs.size - (hi - lo))
        cache.lo = lo
        cache.hi = hi
        if hi <= lo:
            cache.starts = None
            cache.bins = np.array([], dtype=np.int_)
            return cache

        idx = _nn_map_to_bins(axis, mzs[lo:hi])
        # Ascending m/z onto an ascending axis gives non-decreasing bins,
        # so equal bins form contiguous runs.
        starts = np.concatenate(([0], np.flatnonzero(np.diff(idx)) + 1))
        if starts.size == idx.size:
            # Every peak in its own bin: per-spectrum work reduces to a
            # zero-filter over the raw intensities.
            cache.starts = None
            cache.bins = idx.astype(np.int_, copy=False)
        else:
            cache.starts = starts
            cache.bins = idx[starts].astype(np.int_, copy=False)
        return cache

    def _nn_resample_via_cache(
        self,
        axis: NDArray[np.float64],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> Optional[Tuple[NDArray[np.int_], NDArray[np.float64]]]:
        """Resample through the shared-axis cache, or return None to decline.

        The cache is built from the first spectrum seen. A hit requires the
        spectrum's m/z array to be the cached one -- same object, or equal
        element for element -- so a lying reader cannot get a stale mapping;
        it can only miss. After five consecutive misses the cache disables
        itself so processed-mode data stops paying for the comparison
        (its size check is O(1) in the common case anyway).
        """
        cache = self._nn_shared_cache
        if cache is False:
            return None
        if cache is None:
            cache = self._build_nn_shared_cache(axis, mzs)
            if cache is None:
                self._nn_shared_cache = False
                return None
            self._nn_shared_cache = cache

        if not (
            mzs is cache.key_ref
            or (mzs.size == cache.n_total and bool(np.array_equal(mzs, cache.key)))
        ):
            self._nn_cache_misses += 1
            if self._nn_cache_misses > 4:
                self._nn_shared_cache = False
            return None

        if cache.n_dropped:
            self._count_out_of_range(cache.n_dropped, cache.n_total)
        if cache.hi <= cache.lo:
            return np.array([], dtype=np.int_), np.array([], dtype=np.float64)

        # Match bincount's float64 promotion so cached and generic results
        # carry the same rounding.
        vals = intensities[cache.lo : cache.hi].astype(np.float64, copy=False)
        if cache.starts is None:
            sums = vals
        else:
            sums = np.add.reduceat(vals, cache.starts)
        keep = sums != 0
        return cache.bins[keep], sums[keep]

    def _tic_preserving_resample(
        self, mzs: NDArray[np.float64], intensities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Resample onto the common axis, preserving total ion current.

        Linear interpolation onto the target axis, followed by rescaling so
        the resampled spectrum carries the same total ion current as the
        input -- the behaviour ``ResamplingMethod.TIC_PRESERVING`` and
        docs/resampling.md both describe.

        The rescaling is not cosmetic. Interpolation samples the spectrum at
        every target point, so the raw interpolated sum scales with the
        density of the target axis rather than staying fixed. Onto the
        default 190,000-bin axis, 4,000 source points came back with 47x the
        input TIC, and a 150-peak centroid spectrum with over 1000x.

        The TIC preserved is the share of the spectrum lying inside the
        target axis range -- see ``thyra.resampling.tic``, which holds the
        rule and the reasoning, and which ``TICPreservingStrategy`` uses too.
        When the axis spans the spectrum, which is the default, that share
        is the whole input TIC.

        When ``ResamplingConfig.gap_tolerance_da`` is set, bins farther than
        that from any source m/z are zeroed before the rescale, so
        interpolation cannot claim regions nothing was measured in. See
        ``thyra.resampling.gaps``.

        Args:
            mzs: Original m/z values from the spectrum.
            intensities: Corresponding intensity values.

        Returns:
            Intensities on the common mass axis, of the axis's length.
        """
        if self._common_mass_axis is None:
            raise RuntimeError("Common mass axis is not initialized")

        axis = self._common_mass_axis
        indices, values = _tic_preserving_sparse(
            axis, mzs, intensities, getattr(self, "_gap_tolerance_da", None)
        )
        resampled = np.zeros(len(axis))
        resampled[indices] = values
        return resampled

    def _tic_preserving_resample_sparse(
        self, mzs: NDArray[np.float64], intensities: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
        """The sparse form of :meth:`_tic_preserving_resample`.

        Same operator, same numbers: only the axis points the interpolant
        can be non-zero at are evaluated, and only the non-zero results are
        returned, as ``(bin_indices, intensities)`` the way the
        nearest-neighbour path does. On a zero-suppressed profile source --
        a Waters MRT pixel stores ~15,000 samples in clusters around its
        peaks, on a 1.05M-bin axis -- this is what turns a 570 s conversion
        into one that is bounded by reading the file. See
        :func:`_tic_preserving_sparse`.
        """
        if self._common_mass_axis is None:
            raise RuntimeError("Common mass axis is not initialized")
        return _tic_preserving_sparse(
            self._common_mass_axis,
            mzs,
            intensities,
            getattr(self, "_gap_tolerance_da", None),
        )

    def build_region_numbers(self, x_values, y_values) -> NDArray[np.int32]:
        """``obs["region_number"]`` for the given pixel positions, in row order.

        Written on every dataset for a consistent schema, so a consumer
        does not have to branch on whether the acquisition had regions:
        without a region map every pixel is region 1, which is also what
        ``uns["regions"]`` reports for that case. Only the Bruker timsTOF
        reader produces a map today; a position missing from it gets -1.

        Kept as one method because the three write paths that used to
        build an obs table each had their own copy of this rule, and the
        hand-written streaming layout had no copy at all and simply
        omitted the column. One obs builder remains; the rule stays here
        so the sibling tables' builders can share it too.

        Args:
            x_values: Pixel x index per obs row.
            y_values: Pixel y index per obs row.

        Returns:
            Region number per obs row.
        """
        n = len(x_values)
        region_map = getattr(self, "_region_map", None)
        if region_map is None:
            return np.ones(n, dtype=np.int32)

        keys = zip(x_values.tolist(), y_values.tolist())
        return np.fromiter(
            (region_map.get(key, -1) for key in keys), dtype=np.int32, count=n
        )

    def _create_mass_dataframe(self) -> pd.DataFrame:
        """Create m/z dataframe for variable metadata.

        Returns:
            DataFrame with m/z values

        Raises:
            ValueError: If common mass axis is not initialized
        """
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        n_channels = len(self._common_mass_axis)
        columns: Dict[str, Any] = {"mz": self._common_mass_axis}
        columns.update(self._validated_mass_axis_annotations())

        return pd.DataFrame(
            columns,
            index=[f"mz_{i}" for i in range(n_channels)],
        )

    def _validated_mass_axis_annotations(self) -> Dict[str, Any]:
        """Reader-supplied per-channel columns that fit the axis being written.

        A reader whose native axis is not m/z (flight time, drift time) can
        keep that axis alongside ``mz`` so the conversion stays reversible.
        Annotations whose length does not match are dropped rather than
        raising: that is the expected outcome when resampling rebuilds the
        axis, and it must not fail an otherwise good conversion.
        """
        if self._common_mass_axis is None:
            return {}
        n_channels = len(self._common_mass_axis)

        # getattr rather than a bare call: readers predating this hook, and
        # test doubles that do not subclass BaseMSIReader, simply have nothing
        # to contribute and should not have to raise to say so.
        getter = getattr(self.reader, "get_mass_axis_annotations", None)
        if getter is None:
            return {}

        try:
            annotations = getter()
        except Exception as exc:  # pragma: no cover - reader-defined
            # Format eagerly. Passing the exception itself to the logger keeps
            # it alive inside the log record, and with it its traceback, the
            # caller frames reachable through tb_frame.f_back, and any memmap
            # those frames hold -- which on Windows leaves the backing file
            # locked long after the conversion has finished.
            detail = f"{type(exc).__name__}: {exc}"
            logger.warning("Reader failed to supply mass axis annotations: %s", detail)
            return {}

        from thyra.metadata.schema import MSI_VAR_RESERVED_COLUMNS

        validated: Dict[str, Any] = {}
        for name, values in (annotations or {}).items():
            if name == "mz":
                logger.warning("Ignoring mass axis annotation named 'mz'")
                continue
            if name in MSI_VAR_RESERVED_COLUMNS:
                logger.info(
                    "Mass axis annotation %r uses a var column name the "
                    "metadata spec reserves for annotation results; it must "
                    "carry that meaning (see docs/metadata-schema.md)",
                    name,
                )
            if len(values) != n_channels:
                logger.info(
                    "Dropping mass axis annotation %r: %d values for a "
                    "%d channel axis (expected when resampling rebuilds it)",
                    name,
                    len(values),
                    n_channels,
                )
                continue
            validated[name] = values
        return validated

    def _create_pixel_shapes(self, adata: AnnData) -> "ShapesModel":
        """Create geometric shapes for pixels with proper transformations.

        When optical alignment is available (FlexImaging with Area definitions),
        shapes are created in optical image pixel coordinates for proper overlay.
        Otherwise, shapes use physical (micrometer) coordinates.

        **Footprints are two-dimensional, including on a multi-slice
        volume.** A slice's depth is carried by the TIC image's ``Scale``
        and by ``obs["spatial_z"]``; it is deliberately not also put on
        the polygon geometry.

        Three options were measured. All three are imperfect, so what
        follows is the reasoning rather than a claim that this one is
        free:

        * **``POLYGON Z``** -- geometrically the most honest, and what
          7317792 shipped in v3.2.0. It breaks ``spatialdata``'s spatial
          queries: measured on a 5x3x2 volume, a bounding box enclosing
          the whole dataset with 1000um of margin returned 26 of 30
          footprints and 26 of 30 table rows, with no exception and no
          warning. ``shapely.force_2d`` on the same geometry restored
          30 of 30. A z-restricted query returned the same rows whether
          z was inside or far outside the data, so z is ignored by the
          query path rather than honoured.
        * **Flat 2D shapes carrying the depth in a transformation**
          (one element per slice, each with a ``Translation`` in z).
          The element parses, coexists with a 3D image and round-trips,
          but the transform **silently drops z** and every slice comes
          back at the same depth -- a depth in the metadata that never
          reaches the geometry. ``test_3d_pixel_shapes_z.py`` pins this,
          because it is the change someone will otherwise propose.
        * **Flat, with the depth on the image and in ``obs`` only** --
          what this does.

        The deciding argument is that ``spatialdata`` itself asks for
        this. ``ShapesModel.validate`` warns that a 3-dimensional
        geometry column "could led to unexpected behaviors" and names
        ``force_2d()`` as the remedy; the query result above is that
        behaviour. 3D shapes are not on the upstream roadmap
        (scverse/spatialdata#109 has been idle since June 2023 and
        covers images, labels and transformations only), and the live
        2.5D discussion (#961) scopes itself to points, images and
        labels. Serial-section MSI is 2.5D in that taxonomy.

        What this costs: ``docs/coordinate-systems.md``'s promise that
        every element agrees at ``"global"`` is exact in x and y and
        silent in z for the shapes element. That is a documented gap
        rather than a wrong answer, which a truncated query is not. See
        ``docs/output-format.md`` for the consumer-facing statement.

        Reinstating z means restoring the ``is_3d`` parameter at all
        three call sites as well; it was removed with the geometry so it
        could not sit unread, which is how the original defect arose.

        Args:
            adata: AnnData object containing coordinates

        Returns:
            SpatialData shapes model

        Raises:
            ImportError: If required SpatialData dependencies are not available
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")

        geometries = []

        # Track valid indices (for alignment mode where we skip empty positions)
        valid_indices: Optional[List[int]] = None

        # Use FlexImaging alignment for MSI shapes only when both
        # data is available AND the caller hasn't opted out via
        # apply_optical_alignment=False.  Opt-out leaves MSI in pure
        # micrometer coordinates so a downstream alignment step
        # (e.g. Ousia's EscDat registration) is the canonical mapping.
        use_msi_alignment = (
            self._apply_optical_alignment and self._alignment_result is not None
        )

        if use_msi_alignment:
            # Use optical alignment - transform raster coords to image pixels
            # Only create shapes for positions that have actual spectra (in pos_to_region)
            raster_x: NDArray[np.int_] = adata.obs["x"].values
            raster_y: NDArray[np.int_] = adata.obs["y"].values

            # Default half-pixel size (fallback if region lookup fails)
            default_half_pixel = (10.0, 10.0)
            if self._alignment_result.region_mappings:
                # Use first region as default
                rm = self._alignment_result.region_mappings[0]
                default_half_pixel = rm.get_half_pixel_size()

            valid_indices = []
            for i in range(len(adata)):
                rx, ry = int(raster_x[i]), int(raster_y[i])
                img_coords = self._alignment_result.transform_point(rx, ry)

                if img_coords is not None:
                    ix, iy = img_coords
                    # Get region-specific half-pixel size (may be non-square)
                    half_pixel = self._alignment_result.get_half_pixel_size(rx, ry)
                    if half_pixel is None:
                        half_pixel = default_half_pixel

                    half_x, half_y = half_pixel
                    pixel_box = box(
                        ix - half_x,
                        iy - half_y,
                        ix + half_x,
                        iy + half_y,
                    )
                    geometries.append(pixel_box)
                    valid_indices.append(i)
                # Skip positions without spectra - empty grid cells

            n_skipped = len(adata) - len(valid_indices)
            if n_skipped > 0:
                logger.info(
                    f"Created {len(geometries)} shapes using optical alignment "
                    f"(skipped {n_skipped} empty grid positions)"
                )
            else:
                logger.info(
                    f"Created {len(geometries)} shapes using optical alignment "
                    f"(image pixel coordinates)"
                )
        else:
            # Standard physical coordinates (micrometers)
            from shapely import box as shapely_box_vectorized

            x_coords: NDArray[np.float64] = adata.obs["spatial_x"].values
            y_coords: NDArray[np.float64] = adata.obs["spatial_y"].values
            half_pixel_um = self.pixel_size_um / 2

            # Footprints are flat, on every route including volumes. A
            # slice's depth lives on the TIC image's Scale and in
            # obs["spatial_z"]; see the docstring for why it is not also
            # put on the geometry. Built through shapely's vectorised box
            # constructor -- one C call for the whole table instead of one
            # Python-level geometry per pixel.
            geometries = shapely_box_vectorized(
                x_coords - half_pixel_um,
                y_coords - half_pixel_um,
                x_coords + half_pixel_um,
                y_coords + half_pixel_um,
            )

        # Create GeoDataFrame with appropriate index
        if valid_indices is not None:
            # Use filtered indices for alignment mode
            filtered_index = adata.obs.index[valid_indices]
            gdf = gpd.GeoDataFrame(geometry=geometries, index=filtered_index)
        else:
            gdf = gpd.GeoDataFrame(geometry=geometries, index=adata.obs.index)

        # Set up transform
        transform = Identity()
        transformations = {self.dataset_id: transform, "global": transform}

        # Parse shapes
        shapes = ShapesModel.parse(gdf, transformations=transformations)
        return shapes

    def _compute_optical_alignment(self) -> None:
        """Compute optical-MSI alignment from reader metadata.

        For FlexImaging data with Area definitions, this computes the
        transformation that maps MSI raster coordinates to optical image
        pixel coordinates.
        """
        # Check if reader has FlexImaging-specific metadata
        if not hasattr(self.reader, "mis_metadata"):
            logger.debug("Reader does not have mis_metadata, skipping alignment")
            return

        mis_metadata = getattr(self.reader, "mis_metadata", {})
        areas = mis_metadata.get("areas", [])

        if not areas:
            logger.debug("No Area definitions found, skipping alignment")
            return

        # Get required data for alignment
        positions = getattr(self.reader, "_positions", [])
        header = getattr(self.reader, "_header", {})

        if not positions:
            logger.warning("No position data available for alignment")
            return

        first_raster_x = header.get("first_raster_x", 0)
        first_raster_y = header.get("first_raster_y", 0)

        # Store the primary optical image filename from <ImageFile>
        image_file = mis_metadata.get("ImageFile", "")
        if image_file:
            self._primary_optical_filename = Path(image_file).stem.lower()
            logger.info(f"Primary alignment image from .mis: {image_file}")

        # Compute area-based alignment
        try:
            aligner = TeachingPointAlignment()
            self._alignment_result = aligner.compute_area_alignment(
                areas=areas,
                poslog_positions=positions,
                first_raster_x=first_raster_x,
                first_raster_y=first_raster_y,
            )
            logger.info(
                f"Computed optical alignment with "
                f"{len(self._alignment_result.region_mappings)} region mappings"
            )
        except Exception as e:
            logger.warning(f"Failed to compute optical alignment: {e}")
            self._alignment_result = None

    def _build_tic_to_image_affine(self) -> None:
        """Build affine matrix mapping TIC raster-index coords to image pixels.

        When optical alignment is available, this creates a 3x3 affine matrix
        that transforms TIC image coordinates (integer raster indices) into
        optical image pixel coordinates, so the TIC overlays correctly on the
        optical image in SpatialData.

        For single-region data, uses that region's mapping directly.
        For multi-region data, computes a global affine from the overall
        raster bounds and overall image bounds across all regions.

        The matrix encodes: image_pixel = scale * raster_index + offset
        where offset places the first pixel center at image_min + half_pixel.
        """
        if self._alignment_result is None:
            return
        if not self._alignment_result.region_mappings:
            return

        mappings = self._alignment_result.region_mappings

        if len(mappings) == 1:
            # Single region: use its mapping directly
            rm = mappings[0]
            n_raster_x = rm.raster_max_x - rm.raster_min_x + 1
            image_width = rm.image_max_x - rm.image_min_x
            scale_x = image_width / max(1, n_raster_x)
            n_raster_y = rm.raster_max_y - rm.raster_min_y + 1
            image_height = rm.image_max_y - rm.image_min_y
            scale_y = image_height / max(1, n_raster_y)
            half_x = scale_x / 2.0
            half_y = scale_y / 2.0
            tx = rm.image_min_x + half_x
            ty = rm.image_min_y + half_y
        else:
            # Multi-region: compute global affine from overall bounds.
            # The TIC grid covers the full normalized raster space
            # (0..n_x-1, 0..n_y-1). We map this to the bounding box
            # of all region image areas.
            first_rx = self._alignment_result.first_raster_x
            first_ry = self._alignment_result.first_raster_y

            # Global raster bounds (original coords)
            global_raster_min_x = min(rm.raster_min_x for rm in mappings)
            global_raster_max_x = max(rm.raster_max_x for rm in mappings)
            global_raster_min_y = min(rm.raster_min_y for rm in mappings)
            global_raster_max_y = max(rm.raster_max_y for rm in mappings)

            # Global image bounds
            global_img_min_x = min(rm.image_min_x for rm in mappings)
            global_img_max_x = max(rm.image_max_x for rm in mappings)
            global_img_min_y = min(rm.image_min_y for rm in mappings)
            global_img_max_y = max(rm.image_max_y for rm in mappings)

            n_raster_x = global_raster_max_x - global_raster_min_x + 1
            n_raster_y = global_raster_max_y - global_raster_min_y + 1
            image_width = global_img_max_x - global_img_min_x
            image_height = global_img_max_y - global_img_min_y

            scale_x = image_width / max(1, n_raster_x)
            scale_y = image_height / max(1, n_raster_y)
            half_x = scale_x / 2.0
            half_y = scale_y / 2.0

            # The TIC grid index (0,0) corresponds to original raster
            # position (first_rx, first_ry). We need to account for
            # any gap between first_rx and global_raster_min_x.
            offset_raster_x = first_rx - global_raster_min_x
            offset_raster_y = first_ry - global_raster_min_y

            tx = global_img_min_x + half_x + offset_raster_x * scale_x
            ty = global_img_min_y + half_y + offset_raster_y * scale_y

        # 3x3 affine: [[sx, 0, tx], [0, sy, ty], [0, 0, 1]]
        self._tic_to_image_matrix = np.array(
            [
                [scale_x, 0, tx],
                [0, scale_y, ty],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        logger.info(
            f"Built TIC-to-image affine: "
            f"scale=({scale_x:.2f}, {scale_y:.2f}), "
            f"offset=({tx:.1f}, {ty:.1f})"
        )

    def _add_optical_images(self, data_structures: Dict[str, Any]) -> None:
        """Load and add optical images from the reader to data structures.

        Finds optical TIFF files associated with the MSI data and adds them
        as image layers in the SpatialData output. The primary alignment image
        (from .mis <ImageFile>) is loaded first so its dimensions are known
        when computing Scale transforms for the other images.

        Args:
            data_structures: Data structures dict to add images to
        """
        if not self._include_optical:
            return

        optical_paths = self.reader.get_optical_image_paths()
        if not optical_paths:
            logger.debug("No optical images found")
            return

        logger.info(f"Found {len(optical_paths)} optical image(s)")

        # Load primary image first so we know its dimensions for scaling others
        primary_paths = [p for p in optical_paths if self._is_primary_optical(p)]
        other_paths = [p for p in optical_paths if not self._is_primary_optical(p)]

        for tiff_path in primary_paths + other_paths:
            try:
                self._load_single_optical_image(tiff_path, data_structures)
            except Exception as e:
                logger.warning(f"Failed to load optical image {tiff_path.name}: {e}")

    def _is_primary_optical(self, tiff_path: Path) -> bool:
        """Check if a TIFF file is the primary alignment image from .mis."""
        if not self._primary_optical_filename:
            return False
        return tiff_path.stem.lower() == self._primary_optical_filename

    def _compute_optical_scale_transform(self, x_size: int, y_size: int) -> Any:
        """Compute a Scale transform for a non-primary optical image.

        Maps the image's pixel coordinates to the primary alignment image's
        coordinate space using the dimension ratio.

        Args:
            x_size: Width of the non-primary image
            y_size: Height of the non-primary image

        Returns:
            Scale transform, or Identity if no primary dimensions available
        """
        if self._primary_optical_dims is None:
            return Identity()

        primary_w, primary_h = self._primary_optical_dims
        scale_x = primary_w / x_size
        scale_y = primary_h / y_size

        logger.info(f"  Scale to primary: ({scale_x:.4f}, {scale_y:.4f})")
        return Scale([scale_x, scale_y], axes=("x", "y"))

    def _build_optical_to_um_transform(self) -> Any:
        """Build an Affine mapping primary-optical pixels to MSI um.

        Composes the inverse of the tic-to-image affine (so optical
        pixel -> MSI raster index) with the MSI pixel size (so raster
        index -> um).  Used only when ``apply_optical_alignment=False``
        and FlexImaging metadata is available -- it places the optical
        image into the same micrometer "global" frame as the MSI so a
        downstream registration step (e.g. Ousia's EscDat wizard) can
        map both elements together with a single composed affine.

        The math: ``tic_to_image_matrix`` is a 3x3 affine encoding
        ``image_pixel = scale * raster_index + offset``.  Inverting
        and composing with scale-by-pixel-size yields::

            um = pixel_size_um * inv(tic_to_image) @ optical_pixel

        Returns an :class:`Affine` over ``(x, y)`` input + output axes.
        Caller should not invoke when ``_tic_to_image_matrix`` is None.
        """
        if self._tic_to_image_matrix is None:
            raise RuntimeError(
                "_build_optical_to_um_transform called without "
                "a tic_to_image_matrix; check call-site guard."
            )
        inv = np.linalg.inv(self._tic_to_image_matrix)
        # Scale matrix: [[ps, 0, 0], [0, ps, 0], [0, 0, 1]]
        ps = float(self.pixel_size_um)
        scale_mat = np.array(
            [[ps, 0.0, 0.0], [0.0, ps, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        matrix = scale_mat @ inv
        return Affine(
            matrix,
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )

    def _load_single_optical_image(
        self, tiff_path: Path, data_structures: Dict[str, Any]
    ) -> None:
        """Load a single optical TIFF and add it to data structures.

        The primary image (identified by .mis <ImageFile>) gets an Identity
        transform. Other images get a Scale transform mapping their pixel
        coordinates to the primary image's coordinate space.

        Args:
            tiff_path: Path to the TIFF file
            data_structures: Data structures dict to add the image to
        """
        # Generate a clean name for the image layer
        image_name = self._generate_optical_image_name(tiff_path)

        logger.info(f"Loading optical image: {tiff_path.name} as '{image_name}'")

        # Only the page header is read here. The pixels never enter this
        # process whole: the element is declared to SpatialData as a lazy
        # placeholder with the final shape, dtype, chunking and pyramid,
        # and _stream_pending_optical_pixels() fills it in bands once the
        # store exists. See optical_image.py for why (the whole-page route
        # cost ~5x the decoded image in transient memory).
        # probe raises for a layout or sample format it cannot read; the
        # per-image guard in _add_optical_images turns that into the same
        # "skip with a warning" the whole-page decode used to give.
        source = OpticalTiffSource.probe(tiff_path)
        n_channels, y_size, x_size = source.shape

        # Determine transform.  Two cases:
        #
        # 1. apply_optical_alignment=True (default): "global" is
        #    optical-image pixel space.  Primary image is Identity;
        #    non-primary images Scale to match primary dims.
        #
        # 2. apply_optical_alignment=False (e.g. Ousia wizard):
        #    "global" is MSI micrometer space.  Map the primary
        #    image's pixel coordinates into MSI um using the inverse
        #    of the tic-to-image affine, then scale by pixel_size_um.
        #    This way the optical image lands alongside the MSI in
        #    the same um frame and downstream registration steps
        #    map both together.
        um_mode = (
            not self._apply_optical_alignment and self._tic_to_image_matrix is not None
        )
        is_primary = self._is_primary_optical(tiff_path)
        if is_primary:
            self._primary_optical_dims = (x_size, y_size)
            if um_mode:
                transform = self._build_optical_to_um_transform()
                logger.info(
                    f"  Primary image -> um via inverse alignment: {x_size}x{y_size}"
                )
            else:
                transform = Identity()
                logger.info(f"  Primary alignment image: {x_size}x{y_size}")
        elif self._primary_optical_dims is not None:
            # Non-primary: first scale to match primary, then if
            # we're in um-mode, chain through the same um affine.
            base = self._compute_optical_scale_transform(x_size, y_size)
            if um_mode:
                transform = Sequence([base, self._build_optical_to_um_transform()])
            else:
                transform = base
        else:
            transform = Identity()

        # Multi-scale pyramid + chunked layout.
        #
        # Without scale_factors a single-scale image is written and any
        # downstream viewer has to read full-resolution tiles at every
        # zoom level.  For a typical FlexImaging brightfield (10k x 10k+
        # pixels) that is the difference between an instant first paint
        # and a multi-second stall every time the user pans or zooms.
        #
        # We mirror what spatialdata-io's xenium reader does for its
        # morphology images: scale_factors=[2, 2, 2, 2] gives the viewer
        # five pyramid levels.  Here we adapt the level count to the
        # image's smallest spatial dimension so tiny images don't waste
        # levels and huge ones get enough to keep the coarsest level
        # fast (< ~1000 px short side).
        #
        # chunks=(1, 4096, 4096) stores each channel as 4k x 4k blocks
        # so a viewer's 512 x 512 tile read decompresses at most one
        # chunk per request.
        smallest = min(y_size, x_size)
        scale_factors = _calc_optical_scale_factors(smallest)
        streamed = StreamedOpticalImage(
            source=source,
            name=image_name,
            chunks=image_chunks(2),  # (1, 4096, 4096); sharding seam, see _chunking
            scale_factors=scale_factors,
            transformations={
                self.dataset_id: transform,
                "global": transform,
            },
            attrs={
                "source_file": tiff_path.name,
                "original_path": str(tiff_path),
            },
        )
        # Keyed by element name, as the images dict is: a second file that
        # maps to the same name replaces the first, the way the dict
        # assignment always did, only now with a warning.
        earlier = self._pending_optical_images.get(image_name)
        if earlier is not None:
            logger.warning(
                f"Optical image '{image_name}' from {earlier.source.path.name} "
                f"is replaced by {tiff_path.name}, which maps to the same name"
            )
        data_structures["images"][image_name] = streamed.placeholder()
        self._pending_optical_images[image_name] = streamed

        pyramid_desc = (
            f", {len(scale_factors)} pyramid level{'s' if len(scale_factors) != 1 else ''}"
            if scale_factors
            else " (no pyramid; image small enough)"
        )
        logger.info(
            f"Added optical image '{image_name}': {x_size}x{y_size} "
            f"({n_channels} channel{'s' if n_channels > 1 else ''}){pyramid_desc}"
            "; pixels stream in once the store is written"
        )

    def _stream_pending_optical_pixels(self) -> int:
        """Fill every optical image declared so far with its pixels.

        Call once the SpatialData write that carried the placeholders has
        returned and before metadata is consolidated. Each image streams
        from its TIFF in bands and builds its pyramid level by level on
        disk, so memory stays bounded by one band, not by the image.

        A TIFF whose pixels cannot be read is dropped from the store with a
        warning and the conversion goes on without it -- the tolerance the
        whole-page decode had, when the same failure happened before
        anything was written. Only a failure to drop the element propagates,
        because an image with metadata and no pixels is a corrupt store.

        Returns:
            The number of images whose pixels are now in the store.
        """
        pending, self._pending_optical_images = self._pending_optical_images, {}
        streamed = 0
        for image in pending.values():
            logger.info(f"Streaming optical image pixels: '{image.name}'")
            try:
                image.stream_pixels(self.output_path)
            except Exception as e:  # mirrors the per-image guard in _add_optical_images
                logger.warning(
                    f"Failed to load optical image {image.source.path.name}: {e}; "
                    f"dropping '{image.name}' from the store"
                )
                image.discard(self.output_path)
                continue
            streamed += 1
        return streamed

    def _generate_optical_image_name(self, tiff_path: Path) -> str:
        """Generate a clean name for an optical image layer.

        Args:
            tiff_path: Path to the TIFF file

        Returns:
            Clean name for the image layer (e.g., 'optical_0000', 'optical_deriv')
        """
        stem = tiff_path.stem.lower()

        # Extract meaningful suffix from filename
        if "_0000" in stem:
            suffix = "highres"
        elif "_0001" in stem:
            suffix = "derived"
        elif "deriv" in stem:
            suffix = "overview"
        else:
            # Use stem with special chars replaced
            suffix = stem.replace(" ", "_").replace("-", "_")
            # Truncate if too long
            if len(suffix) > 30:
                suffix = suffix[:30]

        return f"{self.dataset_id}_optical_{suffix}"

    def _save_output(self, data_structures: Dict[str, Any]) -> bool:
        """Save the data to SpatialData format.

        Args:
            data_structures: Data structures to save

        Returns:
            True if saving was successful, False otherwise
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")

        try:
            # Create SpatialData object with images included
            sdata = SpatialData(
                tables=data_structures["tables"],
                shapes=data_structures["shapes"],
                images=data_structures["images"],
            )

            # Add metadata
            self.add_metadata(sdata)

            # Write to disk. table_write_config() must wrap the write itself,
            # not just the SpatialData construction: anndata creates the table's
            # zarr arrays lazily inside sdata.write, and that is where the shard
            # budget is read. Without it the table lands in ~1 KB shards -- one
            # file per KB -- and the write dominates conversion wall-clock.
            with _suppress_upstream_warnings(), table_write_config():
                sdata.write(str(self.output_path))
                # The optical images above are placeholders; their pixels
                # stream into the store now that it exists.
                self._stream_pending_optical_pixels()
                zarr.consolidate_metadata(str(self.output_path))
            logger.info(f"Successfully saved SpatialData to {self.output_path}")
            # Every table was written from its memmaps; nothing holds them
            # now but this object and the caller's mapping.
            del sdata
            self._release_table_scratch(data_structures.get("tables"))
            return True
        except Exception as e:
            logger.error(f"Error saving SpatialData: {e}")
            import traceback

            logger.debug(f"Detailed traceback:\n{traceback.format_exc()}")
            return False

    def add_metadata(self, metadata: "SpatialData") -> None:
        """Add comprehensive metadata to the SpatialData object.

        Args:
            metadata: SpatialData object to add metadata to
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        # Call parent to prepare structured metadata
        super().add_metadata(metadata)

        # Get comprehensive metadata object for detailed access
        comprehensive_metadata_obj = self.reader.get_comprehensive_metadata()

        # Setup attributes and add pixel size metadata
        self._setup_spatialdata_attrs(metadata, comprehensive_metadata_obj)

        # Add comprehensive dataset metadata if supported
        self._add_comprehensive_metadata(metadata)

    def build_root_attrs(
        self, comprehensive_metadata_obj: Any = None
    ) -> Dict[str, Any]:
        """The root-level attributes every write path must persist, identically.

        The store's own attrs, as opposed to the table's ``uns`` block
        :meth:`build_uns_metadata` owns. Sibling of that method and here for
        the same reason: the streaming path used to hand-write its Zarr
        layout and composed its own, shorter set -- 7 attributes against
        10 on real ``pea.imzML``, missing ``coordinate_systems``,
        ``format_specific_metadata`` and ``msi_dataset_info``.

        ``coordinate_systems`` is the one that matters most in practice: it
        is the structured contract saying what unit ``"global"`` is in, and
        Ousia and the registration tooling read it rather than guessing.
        A streaming store simply did not have it, and at the time the route
        was chosen by size, so the datasets that lost it were the largest
        ones. Every store is written through :meth:`_save_output` now, so
        this is the one place the attrs are composed.

        Sections the reader has nothing for are omitted rather than written
        empty, matching :meth:`build_uns_metadata`.

        Args:
            comprehensive_metadata_obj: Already-read comprehensive metadata,
                when the caller has it. ``None`` reads it from the reader.

        Returns:
            Mapping of root attribute name to value.
        """
        if comprehensive_metadata_obj is None:
            comprehensive_metadata_obj = self.reader.get_comprehensive_metadata()

        attrs = self._create_pixel_size_attrs()
        self._add_comprehensive_sections(attrs, comprehensive_metadata_obj)
        return attrs

    def _setup_spatialdata_attrs(
        self, metadata: "SpatialData", comprehensive_metadata_obj
    ) -> None:
        """Setup SpatialData attributes with pixel size and metadata."""
        if not hasattr(metadata, "attrs") or metadata.attrs is None:
            metadata.attrs = {}

        logger.info("Adding comprehensive metadata to SpatialData.attrs")

        metadata.attrs.update(self.build_root_attrs(comprehensive_metadata_obj))

    def _create_pixel_size_attrs(self) -> Dict[str, Any]:
        """Create pixel size and conversion metadata attributes."""
        # Import version dynamically
        try:
            from ... import __version__

            version = __version__
        except ImportError:
            version = "unknown"

        # Base pixel size metadata
        pixel_size_attrs = {
            "pixel_size_x_um": float(self.pixel_size_um),
            "pixel_size_y_um": float(self.pixel_size_um),
            "pixel_size_units": "micrometers",
            "coordinate_system": "physical_micrometers",
            "msi_converter_version": version,
            "conversion_timestamp": pd.Timestamp.now().isoformat(),
        }

        # Structured coordinate-system contract describing what "global"
        # actually means in this zarr. Consumers (e.g. Ousia, registration
        # tooling) read this to know what unit "global" is in and how to
        # convert to micrometers without guessing.
        pixel_size_attrs["coordinate_systems"] = self._build_coordinate_systems_attr(
            version
        )

        # Add pixel size detection provenance if available
        if self._pixel_size_detection_info is not None:
            pixel_size_attrs["pixel_size_detection_info"] = dict(
                self._pixel_size_detection_info
            )
            logger.info(
                f"Added pixel size detection info: "
                f"{self._pixel_size_detection_info}"
            )

        # Add conversion metadata
        if self._dimensions is None:
            raise RuntimeError("Dimensions are not initialized")
        pixel_size_attrs["msi_dataset_info"] = {
            "dataset_id": self.dataset_id,
            "total_grid_pixels": self._dimensions[0]
            * self._dimensions[1]
            * self._dimensions[2],
            "non_empty_pixels": self._non_empty_pixel_count,
            "dimensions_xyz": list(self._dimensions),
        }

        return pixel_size_attrs

    # Schema version for the structured `coordinate_systems` attr below.
    # Bump when the schema shape changes in a way consumers need to notice.
    _COORDINATE_SYSTEMS_SCHEMA_VERSION: int = 1

    def _build_coordinate_systems_attr(self, thyra_version: str) -> Dict[str, Any]:
        """Build the structured coordinate-system contract attr.

        This describes what `"global"` means in the produced zarr so that
        downstream consumers can render and convert without guessing.

        Two variants are emitted depending on whether FlexImaging optical
        alignment was applied during conversion:

        - No alignment (`global = micrometer`): the TIC image carries a
          `Scale(pixel_size_um)` and pixel-polygon shapes are stored in
          micrometers with `Identity`. Both elements agree at `global`.
          `pixel_size_um_x/y` are filled with the MSI grid pixel size,
          since "global" is in physical micrometers and there is no
          canonical raster image other than the MSI itself.

        - With alignment (`global = pixel`): the TIC image carries an
          `Affine` mapping raster indices to optical-image pixels and
          shapes are stored directly in optical-image pixels with
          `Identity`. The optical image is the canonical reference.
          `pixel_size_um_x/y` are typically unknown at conversion time
          (FlexImaging does not generally calibrate the optical photo
          to um); leave them null and let the consumer fill in.

        Multi-slice volumes additionally get `z_spacing_um` and
        `z_spacing_source`. These are written **only** for volumes, so a
        2D store is byte-identical to what earlier versions produced and
        their absence is itself the signal that no z axis exists. That
        is also why `convention_version` does not move: the keys are
        purely additive, a consumer that does not do 3D is unaffected,
        and bumping the version would make every existing consumer log a
        "newer than I understand" warning on ordinary 2D datasets.

        Note `z_spacing_um` is an absolute micrometre distance even when
        `unit="pixel"`, because the 3D route always scales z by it
        directly -- the optical affine only ever governs x and y.

        Returns:
            Dict suitable for storing under
            `zarr.attrs["coordinate_systems"]`.
        """
        if self._tic_to_image_matrix is not None:
            unit = "pixel"
            pixel_size_um_x: Optional[float] = None
            pixel_size_um_y: Optional[float] = None
            reference_element: Optional[str] = (
                self._primary_optical_filename
                if self._primary_optical_filename
                else None
            )
        else:
            unit = "micrometer"
            pixel_size_um_x = float(self.pixel_size_um)
            pixel_size_um_y = float(self.pixel_size_um)
            reference_element = None

        global_cs: Dict[str, Any] = {
            "unit": unit,
            "pixel_size_um_x": pixel_size_um_x,
            "pixel_size_um_y": pixel_size_um_y,
            "reference_element": reference_element,
            "convention_version": self._COORDINATE_SYSTEMS_SCHEMA_VERSION,
            "produced_by": f"thyra/{thyra_version}",
        }

        # Additive keys; `convention_version` stays 1 for the same
        # reason as z_spacing_um below.  `raster_to_global_affine` is
        # the explicit 3x3 row-major affine from TIC raster indices to
        # "global" (the same mapping the TIC element's transform
        # expresses), so a consumer that reads only attrs still gets
        # the full placement.  `coordinate_offsets_px` preserves the
        # source's raw acquisition-index offsets, which 0-based
        # normalisation otherwise erases; `stage_offset_um` is their
        # physical equivalent, written only when "global" is in
        # micrometers so it cannot be misread in the optical-pixel
        # variant.
        if self._tic_to_image_matrix is not None:
            global_cs["raster_to_global_affine"] = [
                [float(v) for v in row] for row in self._tic_to_image_matrix
            ]
        else:
            in_plane = float(self.pixel_size_um)
            global_cs["raster_to_global_affine"] = [
                [in_plane, 0.0, 0.0],
                [0.0, in_plane, 0.0],
                [0.0, 0.0, 1.0],
            ]
        offsets = self._source_coordinate_offsets()
        if offsets is not None:
            global_cs["coordinate_offsets_px"] = [int(v) for v in offsets]
            if unit == "micrometer":
                global_cs["stage_offset_um"] = [
                    float(offsets[0]) * float(self.pixel_size_um),
                    float(offsets[1]) * float(self.pixel_size_um),
                ]

        if self._is_volume:
            global_cs["z_spacing_um"] = float(self.z_spacing_um)
            global_cs["z_spacing_source"] = self.z_spacing_source.value

        return {"global": global_cs}

    def _source_coordinate_offsets(self) -> Optional[Tuple[int, int, int]]:
        """The reader's raw coordinate offsets, if it reported any."""
        try:
            essential = self.reader.get_essential_metadata()
        except Exception as e:
            logger.debug("Could not read coordinate offsets: %s", e)
            return None
        offsets = getattr(essential, "coordinate_offsets", None)
        if offsets is None:
            return None
        x, y, z = offsets
        return (int(x), int(y), int(z))

    def _add_comprehensive_sections(
        self, pixel_size_attrs: Dict[str, Any], comprehensive_metadata_obj
    ) -> None:
        """Add comprehensive metadata sections to attributes."""
        if comprehensive_metadata_obj.format_specific:
            pixel_size_attrs["format_specific_metadata"] = (
                comprehensive_metadata_obj.format_specific
            )

        if comprehensive_metadata_obj.acquisition_params:
            pixel_size_attrs["acquisition_parameters"] = (
                comprehensive_metadata_obj.acquisition_params
            )

        if comprehensive_metadata_obj.instrument_info:
            pixel_size_attrs["instrument_information"] = (
                comprehensive_metadata_obj.instrument_info
            )

    def _add_comprehensive_metadata(self, metadata: "SpatialData") -> None:
        """Add comprehensive dataset metadata if SpatialData supports it."""
        if not hasattr(metadata, "metadata"):
            return

        # Start with structured metadata from base class
        metadata_dict = self._structured_metadata.copy()

        # Add SpatialData-specific enhancements
        metadata_dict["non_empty_pixels"] = self._non_empty_pixel_count  # type: ignore[assignment]
        metadata_dict.update(
            {
                "spatialdata_specific": {
                    "zarr_compression_level": self.compression_level,
                    "tables_count": len(getattr(metadata, "tables", {})),
                    "shapes_count": len(getattr(metadata, "shapes", {})),
                    "images_count": len(getattr(metadata, "images", {})),
                },
            }
        )

        # Add pixel size detection provenance if available
        if self._pixel_size_detection_info is not None:
            metadata_dict["pixel_size_provenance"] = self._pixel_size_detection_info

        # Add conversion options used
        metadata_dict["conversion_options"] = {
            "handle_3d": self.handle_3d,
            "pixel_size_um": self.pixel_size_um,
            "z_spacing_um": self.z_spacing_um,
            "z_spacing_source": self.z_spacing_source.value,
            "dataset_id": self.dataset_id,
            **self.options,
        }

        metadata.metadata = metadata_dict

        logger.info(
            f"Comprehensive metadata persisted to SpatialData with "
            f"{len(metadata_dict)} top-level sections"
        )

    @abstractmethod
    def _create_data_structures(self) -> Dict[str, Any]:
        """Create data structures for the specific converter type."""
        pass

    @abstractmethod
    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Finalize data structures for the specific converter type."""
        pass
