"""The demultiplexed MS/MS table: pixels x (precursor, fragment) features.

A MALDI PASEF acquisition isolates several precursors inside one frame,
each in its own slice of the mobility ramp, and the MSI table Thyra always
writes sums that frame into one spectrum per pixel -- so that spectrum
holds fragments of every precursor at once, with nothing marking which
came from which. When the windows are disjoint and identical at every
pixel, splitting them apart is exact: each point of the frame falls in one
window's scan range or in none, so the split is a filter, never an
estimate. This module writes the result as a second table in the same
store:

- same ``obs`` rows and the same ``region`` as the MSI table, so every ROI,
  transform and registration already resolves against it;
- ``var`` sorted lexicographically by ``(precursor_mz, mz)``, so one
  precursor's fragments are a contiguous column block and its ion image is
  a slice of that block;
- ``var["precursor_mz"]`` is the structural marker a consumer
  discriminates on. The MSI table never carries it.

A precursor axis is **discrete**: the schedule names the precursors, so
there is no grid, no channel count and nothing to bin. That is what makes
this independent of the mobility-resolved table, which bins a continuum --
and it is why ``var`` carries no ``mobility`` column. The scan range is how
the precursors are *separated*, not what they are *indexed by*.

The fragment axis is the MSI table's own mass axis: a feature is a
``(precursor, mass axis column)`` pair, so a fragment column of this table
and the corresponding column of the summed table are the same m/z bin, and
the demultiplexed columns of a pixel add back up to what the summed table
holds there.

Like its mobility sibling, the table is built in two passes through
:class:`~thyra.converters.spatialdata.csc_assembly.CscAssembly` -- count
the occupied ``(precursor, bin)`` pairs, then scatter every pixel's values
into memmapped CSC arrays -- so a 100,000-pixel acquisition with 25
precursors costs the same memory as a 700-pixel one with 15.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from ...core.base_reader import BaseMSIReader
from ...core.msms import FragmentationSchedule, windows_overlap
from .csc_assembly import (
    CscAssembly,
    release_when_collected,
    remove_scratch,
    scratch_directory,
)
from .mobility_table import (
    Coords,
    RowLookup,
    assemble_sibling_table,
    collapse_row,
    disambiguate_labels,
    int_strings,
    row_lookup,
)

logger = logging.getLogger(__name__)

#: Suffix appended to the MSI table's key for its demultiplexed sibling.
MSMS_TABLE_SUFFIX = "_msms"

_STRING = np.dtypes.StringDType()


def msms_table_key(table_key: str) -> str:
    """The element key of the demultiplexed MS/MS sibling of ``table_key``."""
    return f"{table_key}{MSMS_TABLE_SUFFIX}"


def feature_axis_block(
    summed_table_key: str, with_mobility: bool = True
) -> Dict[str, Any]:
    """The ``uns["feature_axis"]`` descriptor of a demultiplexed table.

    ``dims`` names the ``var`` columns the sort runs over, in order --
    and only the ones the table carries: ``precursor_mobility`` is absent
    when the source has no per-scan mobility axis to look the isolation
    position up in, and a descriptor naming a column that is not there
    would be a lie a consumer could act on.
    """
    dims = ["precursor_mz", "precursor_mobility", "mz"]
    if not with_mobility:
        dims.remove("precursor_mobility")
    return {"dims": dims, "sorted": True, "summed_table": summed_table_key}


def demultiplex_refusal(schedule: Optional[FragmentationSchedule]) -> Optional[str]:
    """Why this acquisition must not be demultiplexed, or ``None``.

    Each condition is a property of the acquisition that would make the
    split an approximation rather than a filter, so the answer is to
    refuse and say which one failed -- never to apportion ion current
    between precursors that cannot be told apart.

    The conditions are asked in order of how much they say about the
    acquisition, not in order of how cheap they are to check. A schedule
    that varies is a fact about the method; an empty or one-entry window
    list is a fact about what the source recorded, and reporting either of
    those first would describe a run whose precursors Thyra simply cannot
    read -- a diaPASEF file with survey and fragment frames interleaved,
    say -- as one that isolated a single precursor.
    """
    if schedule is None or not schedule.is_msms:
        return "the acquisition is not MS/MS"
    if not schedule.constant_across_pixels:
        return (
            "the precursor schedule is not constant across pixels, so the "
            "precursors are not a global feature axis"
        )
    if not schedule.windows:
        return (
            "the source records no precursor for the fragment frames, so "
            "there is nothing to separate them by"
        )
    if len(schedule.windows) < 2:
        return (
            "the acquisition isolates a single precursor, so the summed table "
            "is already its fragment spectrum"
        )
    if windows_overlap(schedule.windows):
        return (
            "the isolation windows overlap or carry no mobility scan range, "
            "so they cannot be separated by scan number alone"
        )
    return None


def _window_mobility(reader: BaseMSIReader) -> Callable[[Any], float]:
    """A function from an isolation window to the mobility it was isolated at.

    The 1/K0 at the middle of the window's scan range: a window spans a
    slice of the ramp rather than a point, and a scheduled method reports
    no apex, so the midpoint is the honest representative. ``NaN`` when
    the source has no per-scan mobility axis to look it up in.
    """
    values: Optional[NDArray[np.float64]] = None
    try:
        axis = reader.get_mobility_axis()
        if axis is not None:
            values = axis.values
    except Exception as e:  # pragma: no cover - reader-defined
        logger.debug("No mobility axis for the precursor axis: %s", str(e))

    def mobility_of(window: Any) -> float:
        if values is None or not window.is_mobility_resolved:
            return float("nan")
        middle = (int(window.scan_begin) + int(window.scan_end) - 1) // 2
        if middle < 0 or middle >= int(values.size):
            return float("nan")
        return float(values[middle])

    return mobility_of


def _precursor_axis(
    schedule: FragmentationSchedule, mobility_of: Callable[[Any], float]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """The precursor axis: one entry per isolation window, m/z then mobility.

    **Windows are never merged.** Two that isolate the same m/z at
    different mobility positions are two precursors, not one: that is how
    an isomer pair is targeted on this instrument, and summing them would
    undo exactly the separation the mobility ramp provided. They stay two
    column blocks, told apart by ``precursor_mobility``.

    Returns the per-precursor ``m/z`` and mobility in axis order, and the
    position each of the reader's windows takes in that order.
    """
    targets = np.array([w.target for w in schedule.windows], dtype=np.float64)
    mobility = np.array([mobility_of(w) for w in schedule.windows], dtype=np.float64)
    # Primary key m/z, secondary the mobility it was isolated at: the
    # refusals guarantee disjoint scan ranges, so the pair is unique.
    order = np.lexsort((mobility, targets))
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    shared = int(targets.size - np.unique(targets).size)
    if shared:
        logger.info(
            "%d isolation windows share a precursor m/z with another and are "
            "kept apart by the mobility they were isolated at",
            shared,
        )
    return targets[order], mobility[order], rank


def _var_labels(
    precursor_mz: NDArray[np.float64],
    precursor_index: NDArray[np.int64],
    mz_index: NDArray[np.int64],
) -> NDArray[Any]:
    """``p{precursor}_mz{i}`` per feature, disambiguated where two collide.

    Named after the precursor's m/z rather than its position in the
    schedule: a rank is only meaningful inside one dataset, so labelling
    by it would let two samples with different schedules concatenate the
    wrong precursors onto each other without an error. Isomers share an
    m/z and so share a stem; they are disambiguated in mobility order,
    which is the same order in any dataset acquired the same way.
    """
    stems = np.array([f"p{mz:g}" for mz in precursor_mz.tolist()], dtype=_STRING)
    labels = np.strings.add(
        np.strings.add(stems[precursor_index], "_mz"), int_strings(mz_index)
    )
    _unique_stems, stem_id = np.unique(stems, return_inverse=True)
    stem_id = np.asarray(stem_id).ravel().astype(np.int64)
    stride = int(mz_index.max()) + 1 if mz_index.size else 1
    return disambiguate_labels(labels, stem_id[precursor_index] * stride + mz_index)


def _bin_indices(
    axis: NDArray[np.float64], mzs: NDArray[np.float64]
) -> Tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Nearest mass-axis column of each fragment m/z, and which were kept.

    The same rule the summed table's resampling follows: "in range" is the
    strict axis span, and a peak outside it is dropped rather than folded
    onto an edge bin.
    """
    from .base_spatialdata_converter import _nn_map_to_bins

    in_range = (mzs >= axis[0]) & (mzs <= axis[-1])
    kept = mzs if in_range.all() else mzs[in_range]
    if kept.size == 0:
        return np.array([], dtype=np.int64), in_range
    return _nn_map_to_bins(axis, kept).astype(np.int64), in_range


class _Demultiplexer:
    """The ``(precursor, mass axis column)`` entries of one pixel-precursor pair.

    One key per pair: ascending key is ascending ``(precursor_mz, mz)``,
    which is the order ``var`` wants. This is the one place the key is
    derived; the counting pass and the scatter pass both come here, so
    they cannot disagree on what a row holds.
    """

    def __init__(
        self, axis: NDArray[np.float64], window_rank: NDArray[np.int64]
    ) -> None:
        self.axis = axis
        self.window_rank = window_rank
        self.n_dropped = 0

    def entries(
        self,
        window_index: int,
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        columns, in_range = _bin_indices(self.axis, mzs)
        if columns.size == 0:
            self.n_dropped += int(mzs.size)
            return columns, np.zeros(0, dtype=np.float64)
        if columns.size != mzs.size:
            self.n_dropped += int(mzs.size - columns.size)
            intensities = intensities[in_range]
        rank = int(self.window_rank[window_index])
        return collapse_row(
            rank * int(self.axis.size) + columns,
            np.asarray(intensities, dtype=np.float64),
        )


def _feature_var(
    unique_keys: NDArray[np.int64],
    axis: NDArray[np.float64],
    precursor_mz: NDArray[np.float64],
    precursor_mobility: NDArray[np.float64],
) -> pd.DataFrame:
    """The ``var`` of the demultiplexed table, one row per feature.

    ``precursor_index`` is a position in *this* store's precursor axis and
    means nothing outside it. Two datasets are aligned on
    ``(precursor_mz, precursor_mobility)``, which name the same precursor
    wherever it was acquired.
    """
    precursor_index = (unique_keys // axis.size).astype(np.int64)
    mz_index = (unique_keys % axis.size).astype(np.int64)
    columns = {
        "precursor_mz": precursor_mz[precursor_index],
        "mz": axis[mz_index],
        "precursor_index": precursor_index,
        "mz_index": mz_index,
    }
    if np.isfinite(precursor_mobility).any():
        columns["precursor_mobility"] = precursor_mobility[precursor_index]
    return pd.DataFrame(
        columns,
        index=_var_labels(precursor_mz, precursor_index, mz_index),
        copy=False,
    )


class MsmsAccumulator:
    """The two passes of the demultiplexed table, one pixel-precursor pair at a time.

    Pass 1 (:meth:`count`) records which ``(precursor, column)`` keys each
    row occupies; :meth:`finish_counting` says whether anything was
    accumulated and names the columns; :meth:`allocate` backs the CSC
    arrays with files; pass 2 (:meth:`scatter`) writes the values. The
    standalone :func:`_demultiplex` drives it from
    ``iter_precursor_spectra``; the converter's fused passes drive it from
    frame records. Either way every key comes from the same
    :class:`_Demultiplexer`, so the two cannot disagree on what a row holds.
    """

    def __init__(
        self, axis: NDArray[np.float64], window_rank: NDArray[np.int64], n_rows: int
    ) -> None:
        """Allocate the count over ``windows x axis``; refuses a span too wide."""
        n_windows = int(window_rank.size)
        self.demux = _Demultiplexer(axis, window_rank)
        self.assembly = CscAssembly(n_windows * int(axis.size), n_rows)
        self.n_skipped = 0
        self.n_rows_seen = 0
        #: :meth:`finish_counting`'s verdict, ``None`` until it has run.
        self.has_rows: Optional[bool] = None
        #: Whether pass 2 has been run (the converter's fused passes set it).
        self.scattered = False
        #: The scratch directory the fused passes allocated on, if any.
        self.scratch: Optional[Path] = None

    def count(
        self,
        row: Optional[int],
        spectra: Sequence[Tuple[int, NDArray[np.float64], NDArray[np.float64]]],
    ) -> None:
        """Pass 1 for one pixel: its precursor spectra, or none when it has no row."""
        if row is None:
            self.n_skipped += len(spectra)
            return
        for window_index, mzs, intensities in spectra:
            keys, _values = self.demux.entries(window_index, mzs, intensities)
            if keys.size:
                self.n_rows_seen += 1
            self.assembly.count(row, keys)

    def finish_counting(self) -> bool:
        """Say what pass 1 found; whether there is a table to build at all."""
        if self.n_skipped:
            logger.warning(
                "%d demultiplexed spectra had no row in the MSI table and were skipped",
                self.n_skipped,
            )
        if self.demux.n_dropped:
            logger.warning(
                "%d fragment peaks fell outside the mass axis and were dropped "
                "from the demultiplexed table",
                self.demux.n_dropped,
            )
        if self.n_rows_seen == 0:
            logger.warning(
                "No demultiplexed spectra matched the MSI table; no MS/MS table written"
            )
            self.has_rows = False
            return False
        self.assembly.finish_counting()
        self.has_rows = True
        return True

    def allocate(self, scratch: Path) -> None:
        """Back the CSC arrays with files in ``scratch``."""
        self.assembly.allocate(scratch)

    def scatter(
        self,
        row: Optional[int],
        spectra: Sequence[Tuple[int, NDArray[np.float64], NDArray[np.float64]]],
    ) -> None:
        """Pass 2 for one pixel."""
        if row is None:
            return
        for window_index, mzs, intensities in spectra:
            keys, values = self.demux.entries(window_index, mzs, intensities)
            self.assembly.scatter(row, keys, values)

    def result(self) -> Tuple[sparse.csc_matrix, NDArray[np.int64], CscAssembly]:
        """``(matrix, unique keys, assembly)`` once pass 2 is complete."""
        assert self.assembly.unique_keys is not None
        return self.assembly.matrix(), self.assembly.unique_keys, self.assembly


def _demultiplex(
    reader: BaseMSIReader,
    row_for: RowLookup,
    axis: NDArray[np.float64],
    window_rank: NDArray[np.int64],
    n_obs: int,
    scratch: Path,
) -> Optional[Tuple[sparse.csc_matrix, NDArray[np.int64], CscAssembly]]:
    """Two passes over the precursor spectra; ``(matrix, keys, assembly)`` or ``None``."""
    from tqdm import tqdm

    accumulator = MsmsAccumulator(axis, window_rank, n_obs)
    with tqdm(desc="MS/MS table: counting", unit="spectrum") as pbar:
        for coords, window_index, mzs, intensities in reader.iter_precursor_spectra():
            pbar.update(1)
            accumulator.count(row_for(coords), [(window_index, mzs, intensities)])
    if not accumulator.finish_counting():
        return None
    accumulator.allocate(scratch)
    with tqdm(desc="MS/MS table: scattering", unit="spectrum") as pbar:
        for coords, window_index, mzs, intensities in reader.iter_precursor_spectra():
            pbar.update(1)
            accumulator.scatter(row_for(coords), [(window_index, mzs, intensities)])
    return accumulator.result()


def new_msms_accumulator(
    reader: BaseMSIReader, common_mass_axis: NDArray[np.float64], n_rows: int
) -> Optional[MsmsAccumulator]:
    """An accumulator for ``reader``'s schedule, or ``None`` when it must not be split.

    What the converter's fused passes feed. Same refusals and the same
    precursor axis as :func:`build_msms_table`, which then takes the
    accumulator back once the passes have run.
    """
    schedule = reader.get_fragmentation()
    if schedule is None or demultiplex_refusal(schedule) is not None:
        return None
    axis = np.asarray(common_mass_axis, dtype=np.float64)
    if axis.size == 0:
        return None
    _mz, _mobility, window_rank = _precursor_axis(schedule, _window_mobility(reader))
    return MsmsAccumulator(axis, window_rank, n_rows)


def build_msms_table(
    reader: BaseMSIReader,
    obs: pd.DataFrame,
    common_mass_axis: NDArray[np.float64],
    slice_key: str,
    region_key: str,
    uns: Dict[str, Any],
    z_value: Optional[int] = None,
    pixel_key: Optional[Callable[[Coords], Optional[str]]] = None,
    scratch: Optional[Path] = None,
    accumulator: Optional[MsmsAccumulator] = None,
) -> Optional[Any]:
    """Build the demultiplexed MS/MS table for one MSI table, or ``None``.

    Args:
        reader: The source reader; must separate precursors within a pixel.
        obs: The MSI table's ``obs`` (its rows define this table's rows; it
            needs ``x`` and ``y`` columns, and ``z`` when the store is 3D).
        common_mass_axis: The MSI table's ``var["mz"]``, which is also this
            table's fragment axis.
        slice_key: The MSI table's element key (``{id}_z0``).
        region_key: The shapes element both tables annotate.
        uns: The provenance block to store on the table (already built).
        z_value: The plane this table covers when ``obs`` has no ``z``
            column; pixels on other planes are skipped.
        pixel_key: Optional override mapping a reader coordinate to an
            ``obs`` index label; the default matches on ``(x, y[, z])``.
        scratch: Directory for the memmapped CSC arrays the table is built
            on; see :func:`~thyra.converters.spatialdata.mobility_table.build_mobility_table`.
        accumulator: The two passes already run by the converter's fused
            passes (:func:`new_msms_accumulator`, fed frame by frame);
            ``None`` runs them here over ``iter_precursor_spectra``.

    Returns:
        A ``TableModel``-parsed AnnData, or ``None`` when the acquisition
        must not be demultiplexed (logged at info level with the reason)
        or nothing was accumulated.
    """
    schedule = reader.get_fragmentation()
    refusal = demultiplex_refusal(schedule)
    if schedule is None or refusal is not None:
        logger.info("No demultiplexed MS/MS table: %s", refusal)
        return None
    axis = np.asarray(common_mass_axis, dtype=np.float64)
    if axis.size == 0:
        return None

    precursor_mz, precursor_mobility, window_rank = _precursor_axis(
        schedule, _window_mobility(reader)
    )
    n_obs = int(len(obs))
    if accumulator is not None:
        if not accumulator.has_rows or not accumulator.scattered:
            return None
        built: Optional[Tuple[Any, NDArray[np.int64], CscAssembly]] = (
            accumulator.result()
        )
        owns_scratch = False
        workdir = Path(accumulator.scratch) if accumulator.scratch else Path()
    else:
        owns_scratch = scratch is None
        workdir = scratch_directory("thyra_msms_") if scratch is None else Path(scratch)
        built = None
        try:
            built = _demultiplex(
                reader,
                row_lookup(obs, z_value, pixel_key),
                axis,
                window_rank,
                n_obs,
                workdir,
            )
        finally:
            if built is None and owns_scratch:
                remove_scratch(workdir)
    if built is None:
        return None
    matrix, unique_keys, assembly = built

    var = _feature_var(unique_keys, axis, precursor_mz, precursor_mobility)
    table = assemble_sibling_table(
        matrix,
        var,
        obs,
        region_key,
        uns,
        feature_axis_block(
            slice_key, with_mobility="precursor_mobility" in var.columns
        ),
    )
    if owns_scratch:
        release_when_collected(table, assembly, workdir)
    logger.info(
        "Demultiplexed MS/MS table: %d pixels x %d (precursor, fragment) "
        "features over %d precursors, %d non-zeros",
        n_obs,
        int(var.shape[0]),
        int(precursor_mz.size),
        int(matrix.nnz),
    )
    return table
