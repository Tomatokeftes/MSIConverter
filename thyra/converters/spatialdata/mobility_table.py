"""The mobility-resolved table: pixels x (m/z, mobility) features.

The MSI table Thyra always writes is summed over ion mobility. This
module writes what it summed over as a second table in the same store,
filled by whichever of two mechanisms the source allows:

- **a shared feature axis.** A continuous imzML export with a mobility
  array (TIMSImaging, TIMSCONVERT) hands every pixel the same set of
  (m/z, mobility) pairs, so the pairs are already a feature list and
  nothing is binned.
- **a common mobility grid.** A Bruker TDF pixel is its own point cloud
  with no pairs shared across pixels, so the continuum is binned onto the
  channels of a :class:`~thyra.resampling.mobility_grid.MobilityGrid`
  and ``(m/z bin, channel)`` becomes the feature axis.

The two produce the *same kind of table* -- same key, same columns, same
sort, same discriminator -- and a consumer must not need to tell them
apart to read either. What says which mechanism filled it is
``uns["mobility_grid"]``, present only for a binned one, and that is a
description rather than a structural difference.

Either way:

- same ``obs`` rows and the same ``region`` as the MSI table, so every ROI,
  transform and registration already resolves against it;
- ``var`` sorted lexicographically by ``(mz, mobility)``, so an m/z window
  is a contiguous column block and a mobility window a mask inside it;
- ``var["mobility"]`` is the structural marker a consumer discriminates on.
  The MSI table never carries it, and its ``mz`` stays strictly
  increasing; this table's ``mz`` is non-decreasing with duplicates, which
  is exactly what mobility splits.

Both mechanisms are built in two passes over the source through
:class:`~thyra.converters.spatialdata.csc_assembly.CscAssembly` -- count
the occupied features, then scatter each pixel's values straight into
memmapped CSC arrays -- so the table's memory is bounded by its feature
space and one pixel, never by its size. That is the same shape the
summed table takes, and it is what lets a whole acquisition convert
rather than the few hundred frames that fit in RAM.

Mobility is a feature coordinate: nothing here touches ``obs`` beyond
copying it, and nothing enters a coordinate system.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from ...core.base_reader import BaseMSIReader
from ...errors import ConversionRefused
from ...resampling.mobility_grid import MobilityGrid
from .csc_assembly import (
    CscAssembly,
    available_memory_gb,
    count_refusal,
    release_when_collected,
    remove_scratch,
    scratch_directory,
)
from .mobility_heatmap import Coords, scan_mobility

logger = logging.getLogger(__name__)

#: Suffix appended to the MSI table's key for its mobility-resolved sibling.
MOBILITY_TABLE_SUFFIX = "_mobility"

RowLookup = Callable[[Coords], Optional[int]]

_STRING = np.dtypes.StringDType()


def mobility_table_key(table_key: str) -> str:
    """The element key of the mobility-resolved sibling of ``table_key``."""
    return f"{table_key}{MOBILITY_TABLE_SUFFIX}"


def feature_axis_block(summed_table_key: str) -> Dict[str, Any]:
    """The ``uns["feature_axis"]`` descriptor of a mobility-resolved table."""
    return {
        "dims": ["mz", "mobility"],
        "sorted": True,
        "summed_table": summed_table_key,
    }


# ----------------------------------------------------------------------
# Feature labels, and the row mirror both siblings share
# ----------------------------------------------------------------------


def int_strings(values: Any) -> NDArray[Any]:
    """Integers as a variable-width string array, without Python objects."""
    return np.asarray(values, dtype=np.int64).astype(_STRING)


def disambiguate_labels(labels: NDArray[Any], group: NDArray[np.int64]) -> NDArray[Any]:
    """Suffix ``_n`` on the n-th label (n >= 1) of every group, in order seen.

    ``group`` names which labels collide (equal groups, equal labels).
    Vectorised: the labels of a grid table are unique by construction and
    there are millions of them, so this must cost a sort rather than a
    Python dict of every string.
    """
    if group.size == 0:
        return labels
    order = np.argsort(group, kind="stable")
    sorted_group = group[order]
    starts = np.empty(sorted_group.size, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_group[1:] != sorted_group[:-1]
    if starts.all():
        return labels
    positions = np.arange(sorted_group.size, dtype=np.int64)
    run_start = np.maximum.accumulate(np.where(starts, positions, 0))
    ordinal = positions - run_start
    repeated = ordinal > 0
    out = labels.copy()
    where = order[repeated]
    out[where] = np.strings.add(
        np.strings.add(labels[where], "_"), int_strings(ordinal[repeated])
    )
    return out


def _var_labels(
    mz_index: NDArray[np.int64], mobility_index: NDArray[np.int64], unique: bool
) -> NDArray[Any]:
    """``mz{i}_im{j}`` per feature, disambiguated when two features share both.

    ``unique`` says the caller knows every pair is distinct (one feature
    per occupied grid cell), so the collision check is skipped.
    """
    labels = np.strings.add(
        np.strings.add(np.strings.add("mz", int_strings(mz_index)), "_im"),
        int_strings(mobility_index),
    )
    if unique or mz_index.size == 0:
        return labels
    stride = int(mobility_index.max()) + 1
    return disambiguate_labels(
        labels, mz_index.astype(np.int64) * stride + mobility_index
    )


def nearest_axis_index(
    axis: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.int64]:
    """The column of the MSI table each of ``values`` maps to.

    The summed table's own rule -- nearest axis entry, ties to the right --
    so ``mz_index`` names the column that table put the same m/z in. A
    value outside the axis span has no such column (the summed table
    dropped it); it takes the nearest edge, which is the only honest
    integer there is.
    """
    from .base_spatialdata_converter import _nn_map_to_bins

    values = np.asarray(values, dtype=np.float64)
    if axis.size == 0:
        return np.zeros(values.size, dtype=np.int64)
    clipped = np.clip(values, axis[0], axis[-1])
    return _nn_map_to_bins(axis, clipped).astype(np.int64)


def row_lookup(
    obs: pd.DataFrame,
    z_value: Optional[int],
    pixel_key: Optional[Callable[[Coords], Optional[str]]],
) -> RowLookup:
    """A function from reader coordinates to the MSI table's row position.

    Shared with the demultiplexed MS/MS sibling (``msms_table.py``): both
    tables mirror the MSI table's rows, so both resolve a reader's
    coordinates the same way.
    """
    if pixel_key is not None:
        label_to_row = {label: row for row, label in enumerate(obs.index.astype(str))}

        def by_label(coords: Coords) -> Optional[int]:
            label = pixel_key(coords)
            return None if label is None else label_to_row.get(str(label))

        return by_label

    xs = obs["x"].to_numpy().astype(int).tolist()
    ys = obs["y"].to_numpy().astype(int).tolist()
    if "z" in obs.columns:
        zs = obs["z"].to_numpy().astype(int).tolist()
        row_of_xyz = {(x, y, z): row for row, (x, y, z) in enumerate(zip(xs, ys, zs))}

        def by_xyz(coords: Coords) -> Optional[int]:
            return row_of_xyz.get(coords)

        return by_xyz

    row_of_xy = {(x, y): row for row, (x, y) in enumerate(zip(xs, ys))}

    def by_xy(coords: Coords) -> Optional[int]:
        x, y, z = coords
        if z_value is not None and z != z_value:
            return None
        return row_of_xy.get((x, y))

    return by_xy


def collapse_row(
    keys: NDArray[np.int64], values: NDArray[np.float64]
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """One entry per distinct key, ascending, with its values summed.

    The shape :class:`CscAssembly` takes a row in. Keys already unique
    and ascending -- a feature list read in its own order -- pass through
    untouched; anything else is sorted and reduced, which is the merge
    the ``COO -> CSC`` conversion used to perform.
    """
    if keys.size < 2 or bool(np.all(keys[1:] > keys[:-1])):
        return keys, values
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    summed = np.bincount(
        np.asarray(inverse).ravel(), weights=values, minlength=unique_keys.size
    )
    return unique_keys, summed


# ----------------------------------------------------------------------
# The shared-axis mechanism: a feature list every pixel is read off
# ----------------------------------------------------------------------


class _SharedFeatureAxis:
    """The source's ``(mz, mobility)`` pairs, sorted, and the way back to them.

    ``np.unique`` on rows sorts lexicographically by (mz, mobility), which
    is the order the table wants. A source that lists one pair twice has
    its entries merged (summed) into one feature.
    """

    def __init__(
        self, feature_mz: NDArray[np.float64], feature_mobility: NDArray[np.float64]
    ) -> None:
        """Sort the source's pairs and index them for exact lookup."""
        pairs = np.stack(
            [feature_mz.astype(np.float64), feature_mobility.astype(np.float64)],
            axis=1,
        )
        self.unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        self.source_to_feature = np.asarray(inverse).ravel().astype(np.int64)
        self.n_source = int(pairs.shape[0])
        n_merged = int(self.n_source - self.unique_pairs.shape[0])
        if n_merged:
            logger.info(
                "%d feature entries repeat an (m/z, mobility) pair and are merged",
                n_merged,
            )
        # Rank keys for a thresholded spectrum, which carries a subset of
        # the pairs: a pair's column is found by its m/z rank and mobility
        # rank, exact because the values come from the very arrays the
        # axis was built from.
        self._unique_mz, mz_rank = np.unique(
            self.unique_pairs[:, 0], return_inverse=True
        )
        self._unique_mobility, mobility_rank = np.unique(
            self.unique_pairs[:, 1], return_inverse=True
        )
        self._stride = int(self._unique_mobility.size)
        self._pair_rank = np.asarray(mz_rank).ravel().astype(
            np.int64
        ) * self._stride + np.asarray(mobility_rank).ravel().astype(np.int64)

    @property
    def n_features(self) -> int:
        return int(self.unique_pairs.shape[0])

    def columns(
        self, coords: Coords, mzs: NDArray[np.float64], mobility: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        """Feature column of every point of one pixel."""
        if mzs.size == self.n_source:
            return self.source_to_feature
        mz_rank = np.clip(
            np.searchsorted(self._unique_mz, mzs), 0, self._unique_mz.size - 1
        )
        mobility_rank = np.clip(
            np.searchsorted(self._unique_mobility, mobility),
            0,
            self._unique_mobility.size - 1,
        )
        key = mz_rank.astype(np.int64) * self._stride + mobility_rank.astype(np.int64)
        column = np.clip(
            np.searchsorted(self._pair_rank, key), 0, self._pair_rank.size - 1
        )
        on_axis = (
            (self._unique_mz[mz_rank] == mzs)
            & (self._unique_mobility[mobility_rank] == mobility)
            & (self._pair_rank[column] == key)
        )
        if not on_axis.all():
            first = int(np.flatnonzero(~on_axis)[0])
            raise ConversionRefused(
                f"Pixel {coords}: (m/z {mzs[first]}, mobility {mobility[first]}) "
                "is not on the shared feature axis"
            )
        return column.astype(np.int64)


def _shared_axis_row(
    features: _SharedFeatureAxis,
    coords: Coords,
    mzs: NDArray[np.float64],
    mobility: NDArray[np.float64],
    intensities: NDArray[np.float64],
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """One pixel's ``(column, value)`` entries, collapsed, zeros left out."""
    columns = features.columns(coords, mzs, mobility)
    intensities = np.asarray(intensities, dtype=np.float64)
    nonzero = intensities != 0
    if not np.all(nonzero):
        columns = columns[nonzero]
        intensities = intensities[nonzero]
    return collapse_row(np.asarray(columns, dtype=np.int64), intensities)


def _build_from_shared_axis(
    reader: BaseMSIReader,
    row_for: RowLookup,
    common_mass_axis: NDArray[np.float64],
    n_obs: int,
    scratch: Path,
) -> Optional[Tuple[sparse.csc_matrix, pd.DataFrame, CscAssembly]]:
    """Scatter the source's own feature pairs; ``(matrix, var, assembly)`` or ``None``."""
    listed = reader.get_shared_mobility_features()
    if listed is None or listed[0].size == 0:
        return None
    features = _SharedFeatureAxis(*listed)
    # The source's feature list is the table's var, empty features
    # included: a feature no pixel carries is still a feature the export
    # declared, and a consumer aligning on the list must find it.
    assembly = CscAssembly(features.n_features, n_obs, keep_empty_columns=True)
    n_skipped = 0
    n_pixels = 0
    for coords, mzs, mobility, intensities in reader.iter_mobility_spectra():
        row = row_for(coords)
        if row is None:
            n_skipped += 1
            continue
        n_pixels += 1
        keys, _values = _shared_axis_row(features, coords, mzs, mobility, intensities)
        assembly.count(row, keys)
    if n_skipped:
        logger.warning(
            "%d mobility spectra had no row in the MSI table and were skipped",
            n_skipped,
        )
    if n_pixels == 0:
        logger.warning(
            "No mobility spectra matched the MSI table; no mobility table written"
        )
        return None
    assembly.finish_counting()
    assembly.allocate(scratch)
    for coords, mzs, mobility, intensities in reader.iter_mobility_spectra():
        row = row_for(coords)
        if row is None:
            continue
        keys, values = _shared_axis_row(features, coords, mzs, mobility, intensities)
        assembly.scatter(row, keys, values)
    matrix = assembly.matrix()
    var, n_mobility_values = _feature_var(features.unique_pairs, common_mass_axis)
    logger.info(
        "Mobility-resolved table: %d pixels x %d (m/z, mobility) features, "
        "%d non-zeros, %d distinct mobility values",
        n_obs,
        int(var.shape[0]),
        int(matrix.nnz),
        n_mobility_values,
    )
    return matrix, var, assembly


def _feature_var(
    unique_pairs: NDArray[np.float64], common_mass_axis: NDArray[np.float64]
) -> Tuple[pd.DataFrame, int]:
    """The ``var`` of a shared-axis table and its count of distinct mobilities."""
    var_mz = unique_pairs[:, 0]
    var_mobility = unique_pairs[:, 1]
    mz_index = nearest_axis_index(
        np.asarray(common_mass_axis, dtype=np.float64), var_mz
    )
    unique_mobility, mobility_index = np.unique(var_mobility, return_inverse=True)
    mobility_index = np.asarray(mobility_index).ravel().astype(np.int64)
    var = pd.DataFrame(
        {
            "mz": var_mz,
            "mobility": var_mobility,
            "mz_index": mz_index.astype(np.int64),
            "mobility_index": mobility_index,
        },
        index=_var_labels(mz_index, mobility_index, unique=False),
        copy=False,
    )
    return var, int(unique_mobility.size)


# ----------------------------------------------------------------------
# The common-grid mechanism: a per-pixel point cloud, binned
# ----------------------------------------------------------------------

#: Absolute cap on the features a grid table's ``var`` may hold -- occupied
#: ``(m/z bin, mobility channel)`` pairs, not the pairs the grid spans. The
#: two differ by an order of magnitude on real data (a measured 200-frame
#: timsTOF acquisition occupied 3.9M of a possible 35.5M), which is why the
#: count is checked and not the bound. The cap is a statement of what a
#: downstream tool can be expected to open, not the operative guard: that
#: is the memory projection below (design decision D4), which is what
#: refuses on a real machine long before this number.
MAX_GRID_VAR_ENTRIES = 100_000_000

#: Process memory the ``var`` frame costs per feature while it is built,
#: AnnData copies included. Measured 2026-09-07 on a real grid: a private
#: peak 4.4 GB above baseline at 13.27M features.
VAR_BYTES_PER_FEATURE = 330

#: Fractions of the machine's free memory the projected ``var`` frame may
#: take before the conversion warns, and before it refuses. Fractions and
#: not sizes: a table that is routine on a workstation is fatal on a
#: laptop, and the same number cannot be right for both. The mass axis's
#: own guard (``csc_assembly.mass_axis_refusal``) uses the same pair.
GRID_VAR_WARN_FRACTION = 0.25
GRID_VAR_REFUSE_FRACTION = 0.5


def projected_var_gb(n_features: int) -> float:
    """What building a ``var`` frame of this many features is expected to cost."""
    return float(n_features) * VAR_BYTES_PER_FEATURE / 1024**3


def grid_var_bound(common_mass_axis: NDArray[np.float64], grid: MobilityGrid) -> int:
    """Every ``(m/z bin, channel)`` pair the grid spans: the size to beat.

    Knowable before reading anything, and a true upper bound on the
    table's ``var`` -- so a grid whose bound already fits certainly fits.
    A bound above the ceiling decides nothing on its own, since real
    occupancy is far below it; that case is read and then counted.
    """
    return int(np.asarray(common_mass_axis).size) * int(grid.n_channels)


def var_ceiling_refusal(
    n_features: int, available_gb: Optional[float] = None
) -> Optional[str]:
    """Why a grid table this wide must not be built, or ``None``.

    The operative test is memory: the ``var`` frame is the peak of an
    out-of-core build, its cost per feature is measured, and the machine's
    free memory is known, so the projection is compared with what is
    there -- refused past :data:`GRID_VAR_REFUSE_FRACTION` of it, warned
    about past :data:`GRID_VAR_WARN_FRACTION`. The absolute cap is checked
    first and is far above any real table.

    ``available_gb`` overrides the machine's own answer, for tests.
    """
    levers = (
        "resample to fewer mass bins (--resample-bins) or ask for fewer "
        "mobility channels (--mobility-bins)"
    )
    if n_features > MAX_GRID_VAR_ENTRIES:
        return (
            f"the grid occupies {n_features:,} (m/z bin, mobility channel) "
            f"pairs, above the var ceiling of {MAX_GRID_VAR_ENTRIES:,}; {levers}"
        )
    gb = projected_var_gb(n_features)
    free = available_memory_gb() if available_gb is None else float(available_gb)
    if gb > free * GRID_VAR_REFUSE_FRACTION:
        return (
            f"the grid occupies {n_features:,} (m/z bin, mobility channel) "
            f"pairs, and building their var frame is projected to need "
            f"{gb:.1f} GB ({VAR_BYTES_PER_FEATURE} bytes per feature, measured) "
            f"against {free:.1f} GB free, more than the "
            f"{GRID_VAR_REFUSE_FRACTION:.0%} of free memory a conversion may "
            f"take; {levers}"
        )
    if gb > free * GRID_VAR_WARN_FRACTION:
        logger.warning(
            "The mobility grid occupies %s pairs; building their var frame "
            "is projected to need %.1f GB of the %.1f GB free. It will be "
            "attempted; %s to make it smaller.",
            f"{n_features:,}",
            gb,
            free,
            levers,
        )
    return None


def grid_refusal(
    reader: BaseMSIReader,
    common_mass_axis: NDArray[np.float64],
    grid: Optional[MobilityGrid],
) -> Optional[str]:
    """Why this source must not be binned onto a common grid, or ``None``.

    The refusals knowable before the source is read. Every answer is a
    property of the source, so the answer is to say which one failed and
    write no table -- never to bin something that is not a continuum. The
    size ceiling is not among them: it is a property of the data, checked
    on the real count in :func:`var_ceiling_refusal`.
    """
    if not reader.has_ion_mobility:
        return "the source has no ion mobility dimension"
    if grid is None:
        return (
            "the source's mobility axis gives no range to bin over; a grid "
            "needs the per-scan mobility values, which a reader opened "
            "without its vendor library cannot supply"
        )
    if int(np.asarray(common_mass_axis).size) == 0:
        return "the common mass axis is empty"
    return count_refusal(grid_var_bound(common_mass_axis, grid))


def grid_cells(
    bins: NDArray[np.int64],
    channels: NDArray[np.int64],
    intensities: NDArray[np.float64],
    n_channels: int,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """One entry per occupied cell of a pixel, with its summed current.

    A pixel's raw cloud carries many points per cell -- that is what a
    TIMS ramp is -- so the pixel is collapsed onto its own occupied cells
    before anything is counted or scattered. The flat key
    ``bin * n_channels + channel`` ascends with ``(mz, mobility)``, which
    is the order ``var`` wants, for free. This is the one place the key is
    derived: the discovery pass and the scatter pass both come here.
    """
    keys = bins * int(n_channels) + channels
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    summed = np.bincount(
        np.asarray(inverse).ravel(), weights=intensities, minlength=unique_keys.size
    )
    nonzero = summed != 0
    return unique_keys[nonzero], summed[nonzero]


class GridDiscovery:
    """Pass 1 of the grid route: which cells are occupied, and by how many rows.

    A sink for :func:`~thyra.converters.spatialdata.mobility_heatmap.scan_mobility`,
    so a conversion that writes the heatmap can discover the grid in the
    same raw pass. Its memory is the count array over the grid's span --
    a few hundred megabytes at most, fixed before the first pixel is read
    -- and nothing is ever buffered per pixel.
    """

    def __init__(
        self,
        axis: NDArray[np.float64],
        grid: MobilityGrid,
        row_for: RowLookup,
        n_obs: int,
    ) -> None:
        """Allocate the count over the grid's span; refuses one too wide."""
        self.axis = np.asarray(axis, dtype=np.float64)
        self.grid = grid
        self.row_for = row_for
        self.assembly = CscAssembly(grid_var_bound(self.axis, grid), n_obs)
        self.n_pixels = 0
        self.n_skipped = 0
        self.n_dropped = 0
        self.n_points = 0
        self.n_features: Optional[int] = None
        #: The var ceiling's answer once the count is known.
        self.refusal: Optional[str] = None
        #: What :func:`_report_discovery` decided, once it has (``None``
        #: until then), so the verdict is reached and logged once.
        self.decided: Optional[bool] = None
        #: Whether pass 2 has already been run through this discovery's
        #: assembly (the converter's fused passes do that); the builder
        #: then takes the matrix as it is.
        self.scattered = False
        #: The scratch directory the fused passes allocated on, if any.
        self.scratch: Optional[Path] = None

    def add_mapped(
        self,
        coords: Coords,
        bins: NDArray[np.int64],
        mobility: NDArray[np.float64],
        intensities: NDArray[np.float64],
        n_dropped: int,
    ) -> None:
        """Count one pixel already mapped by :func:`map_points_to_axis`."""
        self.add_mapped_row(
            self.row_for(coords), bins, mobility, intensities, n_dropped
        )

    def add_mapped_row(
        self,
        row: Optional[int],
        bins: NDArray[np.int64],
        mobility: NDArray[np.float64],
        intensities: NDArray[np.float64],
        n_dropped: int,
    ) -> None:
        """Count one mapped pixel whose row the caller already knows.

        ``None`` is a pixel with no row in the table, skipped exactly as
        :meth:`add_mapped` skips one ``row_for`` cannot place.
        """
        if row is None:
            self.n_skipped += 1
            return
        self.n_pixels += 1
        self.n_dropped += int(n_dropped)
        self.n_points += int(bins.size)
        keys, _values = grid_cells(
            bins,
            self.grid.assign(mobility).astype(np.int64),
            intensities,
            self.grid.n_channels,
        )
        self.assembly.count(row, keys)

    def finish(self) -> Optional[str]:
        """Close the count; the var ceiling's refusal, if any, is the answer.

        Checked here, before the labels, the memmaps and the ``var`` exist:
        this is the first moment the real number is known and nothing has
        been committed to yet, which is the whole point of the ceiling.
        """
        self.n_features = self.assembly.finish_counting()
        self.refusal = var_ceiling_refusal(self.n_features)
        return self.refusal


def _grid_feature_var(
    unique_keys: NDArray[np.int64],
    axis: NDArray[np.float64],
    grid: MobilityGrid,
) -> pd.DataFrame:
    """The ``var`` of a grid table -- the same columns a shared axis writes.

    ``mz`` is the common axis entry the points were mapped to and
    ``mobility`` the centre of the channel they fell in, so both are bin
    representatives rather than measured values. The pair is unique by
    construction (one feature per occupied cell) and the flat key's order
    is already lexicographic ``(mz, mobility)``.
    """
    n_channels = int(grid.n_channels)
    mz_index = (unique_keys // n_channels).astype(np.int64)
    mobility_index = (unique_keys % n_channels).astype(np.int64)
    centres = grid.centres
    # copy=False: a frame of millions of rows is built once from arrays
    # nothing else holds, and pandas would otherwise duplicate every column.
    return pd.DataFrame(
        {
            "mz": axis[mz_index],
            "mobility": centres[mobility_index],
            "mz_index": mz_index,
            "mobility_index": mobility_index,
        },
        index=_var_labels(mz_index, mobility_index, unique=True),
        copy=False,
    )


class GridScatter:
    """Pass 2 of the grid route: one mapped pixel into the allocated CSC arrays.

    Built on a :class:`GridDiscovery` whose assembly has been allocated;
    the same cell keys pass 1 counted, now with their values.
    """

    def __init__(self, discovery: GridDiscovery) -> None:
        """Scatter into ``discovery``'s assembly, which must be allocated."""
        self.discovery = discovery
        self.grid = discovery.grid
        self.assembly = discovery.assembly
        self.n_channels = int(discovery.grid.n_channels)

    def add_mapped(
        self,
        coords: Coords,
        bins: NDArray[np.int64],
        mobility: NDArray[np.float64],
        intensities: NDArray[np.float64],
        _n_dropped: int,
    ) -> None:
        """Scatter one pixel already mapped by :func:`map_points_to_axis`."""
        self.add_mapped_row(
            self.discovery.row_for(coords), bins, mobility, intensities, _n_dropped
        )

    def add_mapped_row(
        self,
        row: Optional[int],
        bins: NDArray[np.int64],
        mobility: NDArray[np.float64],
        intensities: NDArray[np.float64],
        _n_dropped: int,
    ) -> None:
        """Scatter one mapped pixel whose row the caller already knows."""
        if row is None:
            return
        keys, values = grid_cells(
            bins,
            self.grid.assign(mobility).astype(np.int64),
            intensities,
            self.n_channels,
        )
        self.assembly.scatter(row, keys, values)


def _report_discovery(discovery: GridDiscovery) -> bool:
    """Say what pass 1 found; whether there is a table to build at all.

    Decided and logged once: a second call returns the first verdict.
    """
    if discovery.decided is not None:
        return discovery.decided
    discovery.decided = _decide_discovery(discovery)
    return discovery.decided


def _decide_discovery(discovery: GridDiscovery) -> bool:
    if discovery.n_skipped:
        logger.warning(
            "%d mobility spectra had no row in the MSI table and were skipped",
            discovery.n_skipped,
        )
    if discovery.n_dropped:
        logger.warning(
            "%d mobility points fell outside the mass axis and were "
            "dropped from the mobility-resolved table",
            discovery.n_dropped,
        )
    if discovery.n_pixels == 0 or discovery.assembly.n_nonzeros == 0:
        logger.warning(
            "No mobility spectra matched the MSI table; no mobility table written"
        )
        return False
    if discovery.refusal is not None:
        logger.warning("No mobility-resolved table: %s", discovery.refusal)
        return False
    return True


def _build_from_grid(
    reader: BaseMSIReader,
    row_for: RowLookup,
    common_mass_axis: NDArray[np.float64],
    grid: MobilityGrid,
    n_obs: int,
    discovery: Optional[GridDiscovery],
    scratch: Path,
) -> Optional[Tuple[sparse.csc_matrix, pd.DataFrame, CscAssembly]]:
    """Bin every pixel's point cloud onto the grid; ``(matrix, var, assembly)`` or ``None``.

    ``discovery`` is pass 1 already run (fused into the heatmap's pass by
    the converter); without it the pass runs here. Pass 2 then re-reads
    the source and scatters each pixel straight into the CSC arrays --
    unless the converter's fused passes have scattered already
    (``discovery.scattered``), in which case the matrix is taken as it is.
    """
    axis = np.asarray(common_mass_axis, dtype=np.float64)
    if discovery is None:
        discovery = GridDiscovery(axis, grid, row_for, n_obs)
        scan_mobility(reader, axis, discovery, description="Mobility grid: counting")
        discovery.finish()
    elif discovery.n_features is None:
        discovery.finish()
    if not _report_discovery(discovery):
        return None
    assembly = discovery.assembly
    n_channels = int(grid.n_channels)
    if not discovery.scattered:
        assembly.allocate(scratch)
        scan_mobility(
            reader,
            axis,
            GridScatter(discovery),
            description="Mobility grid: scattering",
        )
    matrix = assembly.matrix()
    var = _grid_feature_var(assembly.unique_keys, axis, grid)
    logger.info(
        "Mobility grid: %d points over %d pixels onto %d occupied "
        "(m/z bin, channel) cells of %d x %d, %d channels used",
        discovery.n_points,
        discovery.n_pixels,
        int(var.shape[0]),
        int(axis.size),
        n_channels,
        int(np.unique(var["mobility_index"].to_numpy()).size),
    )
    return matrix, var, assembly


# ----------------------------------------------------------------------
# The entry point over both mechanisms
# ----------------------------------------------------------------------


def build_mobility_table(
    reader: BaseMSIReader,
    obs: pd.DataFrame,
    common_mass_axis: NDArray[np.float64],
    slice_key: str,
    region_key: str,
    uns: Dict[str, Any],
    z_value: Optional[int] = None,
    pixel_key: Optional[Callable[[Coords], Optional[str]]] = None,
    grid: Optional[MobilityGrid] = None,
    discovery: Optional[GridDiscovery] = None,
    scratch: Optional[Path] = None,
) -> Optional[Any]:
    """Build the mobility-resolved table for one MSI table, or ``None``.

    A source with a shared feature axis is read off it; one without is
    binned onto ``grid`` when the caller supplied one, and gets no table
    when it did not. The two tables are indistinguishable apart from
    ``uns["mobility_grid"]``.

    Args:
        reader: The source reader; must report an ion mobility dimension.
        obs: The MSI table's ``obs`` (its rows define this table's rows; it
            needs ``x`` and ``y`` columns, and ``z`` when the store is 3D).
        common_mass_axis: The MSI table's ``var["mz"]``, for ``mz_index``
            and, on the grid route, for the m/z binning itself.
        slice_key: The MSI table's element key (``{id}_z0``).
        region_key: The shapes element both tables annotate.
        uns: The provenance block to store on the table (already built).
        z_value: The plane this table covers when ``obs`` has no ``z``
            column; pixels on other planes are skipped.
        pixel_key: Optional override mapping a reader coordinate to an
            ``obs`` index label; the default matches on ``(x, y[, z])``.
        grid: The common mobility grid to bin a per-pixel source onto.
            ``None`` (the default) leaves such a source with the summed
            table only.
        discovery: The grid's discovery pass, when the caller already ran
            it (the converter fuses it into the heatmap's pass); ``None``
            runs it here.
        scratch: Directory for the memmapped CSC arrays the table is built
            on. The table's ``X`` stays backed by them until it is written,
            so the directory must outlive the write; a caller who passes
            one owns its removal. ``None`` makes a temporary one that is
            removed when the returned table is garbage collected.

    Returns:
        A ``TableModel``-parsed AnnData, or ``None`` when no table can be
        written (logged with the reason).
    """
    if not reader.has_ion_mobility:
        return None
    n_obs = int(len(obs))
    row_for = row_lookup(obs, z_value, pixel_key)
    axis = np.asarray(common_mass_axis, dtype=np.float64)
    grid_uns: Optional[Dict[str, Any]] = None
    owns_scratch = scratch is None
    workdir = scratch_directory("thyra_mobility_") if scratch is None else Path(scratch)
    built = None
    try:
        if reader.has_shared_mobility_axis:
            built = _build_from_shared_axis(reader, row_for, axis, n_obs, workdir)
        else:
            refusal = grid_refusal(reader, axis, grid)
            if refusal is not None or grid is None:
                logger.info(
                    "No mobility-resolved table: the source carries mobility per "
                    "pixel rather than as a shared feature axis, and %s",
                    refusal or "no common mobility grid was asked for",
                )
                return None
            built = _build_from_grid(
                reader, row_for, axis, grid, n_obs, discovery, workdir
            )
            grid_uns = grid.to_uns()
    finally:
        if built is None and owns_scratch:
            remove_scratch(workdir)
    if built is None:
        return None
    matrix, var, assembly = built
    table = _assemble(matrix, var, obs, region_key, slice_key, uns, grid_uns)
    if owns_scratch:
        release_when_collected(table, assembly, workdir)
    return table


def _assemble(
    matrix: sparse.csc_matrix,
    var: pd.DataFrame,
    obs: pd.DataFrame,
    region_key: str,
    slice_key: str,
    uns: Dict[str, Any],
    grid_uns: Optional[Dict[str, Any]],
) -> Any:
    """Wrap a matrix and its ``var`` as the store's mobility table."""
    extra: Dict[str, Any] = {}
    if grid_uns is not None:
        extra["mobility_grid"] = grid_uns
    return assemble_sibling_table(
        matrix, var, obs, region_key, uns, feature_axis_block(slice_key), extra
    )


def assemble_sibling_table(
    matrix: sparse.csc_matrix,
    var: pd.DataFrame,
    obs: pd.DataFrame,
    region_key: str,
    uns: Dict[str, Any],
    feature_axis: Dict[str, Any],
    extra_uns: Optional[Dict[str, Any]] = None,
) -> Any:
    """Wrap a matrix and its ``var`` as a sibling of the MSI table.

    Shared by both siblings: the same ``obs`` rows and ``region`` as the
    summed table, the provenance block it was given, ``uns["feature_axis"]``
    saying what the columns are, and whatever block says which mechanism
    filled it. ``X`` is taken as it comes -- on the assembly's memmaps --
    so the table is written from disk to disk.
    """
    from anndata import AnnData
    from spatialdata.models import TableModel

    from .base_spatialdata_converter import _jsonify_string_lists

    n_obs = int(len(obs))
    table_obs = obs.copy()
    table_obs["region"] = pd.Categorical([region_key] * n_obs)
    table_obs["instance_key"] = table_obs.index.astype(str)

    adata = AnnData(X=matrix, obs=table_obs, var=var)
    adata.uns.update(uns)
    # String lists become JSON strings, as everywhere else in uns: a list of
    # strings does not round-trip through zarr on numpy 2.1-2.2.
    adata.uns["feature_axis"] = _jsonify_string_lists(feature_axis)
    for key, value in (extra_uns or {}).items():
        adata.uns[key] = value
    return TableModel.parse(
        adata,
        region=region_key,
        region_key="region",
        instance_key="instance_key",
    )


def mobility_grid_range(reader: BaseMSIReader) -> Optional[Tuple[float, float]]:
    """The range to bin a per-pixel source's mobility over, or ``None``.

    The axis *values* decide, and only the values: the per-scan 1/K0 of a
    real Bruker file overhangs its declared ``OneOverK0AcqRange`` by a few
    scans, the mass-mobility heatmap already bins over the values, and a
    grid that used the declared range instead would not index the heatmap.
    A reader that cannot supply per-scan values (opened without its vendor
    library) has no grid, which is a refusal rather than a fallback.
    """
    axis = reader.get_mobility_axis()
    if axis is None or axis.values is None:
        return None
    finite = axis.values[np.isfinite(axis.values)]
    if not finite.size:
        return None
    lower, upper = float(finite.min()), float(finite.max())
    return (lower, upper) if upper > lower else None
