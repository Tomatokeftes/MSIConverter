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

Mobility is a feature coordinate: nothing here touches ``obs`` beyond
copying it, and nothing enters a coordinate system.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from ...core.base_reader import BaseMSIReader
from ...resampling.mobility_grid import MobilityGrid

logger = logging.getLogger(__name__)

#: Suffix appended to the MSI table's key for its mobility-resolved sibling.
MOBILITY_TABLE_SUFFIX = "_mobility"

Coords = Tuple[int, int, int]
RowLookup = Callable[[Coords], Optional[int]]


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


def _var_labels(mz_index: NDArray[np.int64], mobility_index: NDArray[np.int64]) -> list:
    """``mz{i}_im{j}`` per feature, disambiguated when two features share both."""
    labels = [
        f"mz{i}_im{j}" for i, j in zip(mz_index.tolist(), mobility_index.tolist())
    ]
    seen: Dict[str, int] = {}
    out = []
    for label in labels:
        n = seen.get(label, 0)
        seen[label] = n + 1
        out.append(label if n == 0 else f"{label}_{n}")
    return out


def nearest_axis_index(
    axis: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Index of the nearest entry of a sorted ``axis`` for each of ``values``."""
    if axis.size == 0:
        return np.zeros(values.size, dtype=np.int64)
    right = np.clip(np.searchsorted(axis, values), 0, axis.size - 1)
    left = np.maximum(right - 1, 0)
    pick_left = np.abs(axis[left] - values) <= np.abs(axis[right] - values)
    return np.where(pick_left, left, right).astype(np.int64)


def _feature_axis(
    feature_mz: NDArray[np.float64], feature_mobility: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Unique ``(mz, mobility)`` pairs, sorted, and the source-to-feature map.

    ``np.unique`` on rows sorts lexicographically by (mz, mobility), which
    is the order the table wants. A source that lists one pair twice has its
    entries merged (summed) into one feature.
    """
    pairs = np.stack(
        [feature_mz.astype(np.float64), feature_mobility.astype(np.float64)], axis=1
    )
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    n_merged = int(pairs.shape[0] - unique_pairs.shape[0])
    if n_merged:
        logger.info(
            "%d feature entries repeat an (m/z, mobility) pair and are merged",
            n_merged,
        )
    return unique_pairs, np.asarray(inverse).ravel().astype(np.int64)


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


def _columns_for_subset(
    coords: Coords,
    mzs: NDArray[np.float64],
    mobility: NDArray[np.float64],
    unique_pairs: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Feature columns of a thresholded spectrum, matched pair by pair.

    Exact matching is correct here: the values come from the very arrays
    the feature axis was built from.
    """
    var_mz = unique_pairs[:, 0]
    var_mobility = unique_pairs[:, 1]
    n_features = int(var_mz.size)
    starts = np.searchsorted(var_mz, mzs, side="left")
    cols = np.empty(mzs.size, dtype=np.int64)
    for i, (m, b) in enumerate(zip(mzs.tolist(), mobility.tolist())):
        j = int(starts[i])
        while j < n_features and var_mz[j] == m and var_mobility[j] != b:
            j += 1
        if j >= n_features or var_mz[j] != m or var_mobility[j] != b:
            raise ValueError(
                f"Pixel {coords}: (m/z {m}, mobility {b}) is not on the shared "
                "feature axis"
            )
        cols[i] = j
    return cols


def _accumulate(
    reader: BaseMSIReader,
    row_for: RowLookup,
    unique_pairs: NDArray[np.float64],
    source_to_feature: NDArray[np.int64],
    n_obs: int,
) -> Optional[sparse.csc_matrix]:
    """Scatter every pixel's intensities onto the feature axis; CSC result."""
    rows_acc: List[NDArray[np.int64]] = []
    cols_acc: List[NDArray[np.int64]] = []
    data_acc: List[NDArray[np.float64]] = []
    n_skipped = 0
    n_pixels = 0
    n_source = int(source_to_feature.size)
    for coords, mzs, mobility, intensities in reader.iter_mobility_spectra():
        row = row_for(coords)
        if row is None:
            n_skipped += 1
            continue
        n_pixels += 1
        if mzs.size == n_source:
            cols = source_to_feature
        else:
            cols = _columns_for_subset(coords, mzs, mobility, unique_pairs)
        nonzero = intensities != 0
        if not np.all(nonzero):
            cols = cols[nonzero]
            intensities = intensities[nonzero]
        rows_acc.append(np.full(cols.size, row, dtype=np.int64))
        cols_acc.append(np.asarray(cols, dtype=np.int64))
        data_acc.append(np.asarray(intensities, dtype=np.float64))

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

    rows = np.concatenate(rows_acc)
    cols = np.concatenate(cols_acc)
    data = np.concatenate(data_acc)
    # COO -> CSC sums coincident entries, which is the merge the unique
    # pairs call for.
    n_features = int(unique_pairs.shape[0])
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_obs, n_features)).tocsc()


def _feature_var(
    unique_pairs: NDArray[np.float64], common_mass_axis: NDArray[np.float64]
) -> Tuple[pd.DataFrame, int]:
    """The ``var`` of the mobility table and its count of distinct mobilities."""
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
        index=_var_labels(mz_index, mobility_index),
    )
    return var, int(unique_mobility.size)


# ----------------------------------------------------------------------
# The common-grid mechanism: a per-pixel point cloud, binned
# ----------------------------------------------------------------------

#: Features a grid table's ``var`` may hold before the grid is refused --
#: occupied ``(m/z bin, mobility channel)`` pairs, not the pairs the grid
#: spans. The two differ by an order of magnitude on real data (a measured
#: 200-frame timsTOF acquisition occupied 3.9M of a possible 35.5M), which
#: is why the count is checked and not the bound: refusing on the bound
#: would turn away conversions that fit ninefold over.
MAX_GRID_VAR_ENTRIES = 20_000_000


def grid_var_bound(common_mass_axis: NDArray[np.float64], grid: MobilityGrid) -> int:
    """Every ``(m/z bin, channel)`` pair the grid spans: the size to beat.

    Knowable before reading anything, and a true upper bound on the
    table's ``var`` -- so a grid whose bound already fits certainly fits.
    A bound above the ceiling decides nothing on its own, since real
    occupancy is far below it; that case is read and then counted.
    """
    return int(np.asarray(common_mass_axis).size) * int(grid.n_channels)


def var_ceiling_refusal(n_features: int) -> Optional[str]:
    """Why a grid table this wide must not be built, or ``None``."""
    if n_features <= MAX_GRID_VAR_ENTRIES:
        return None
    return (
        f"the grid occupies {n_features:,} (m/z bin, mobility channel) pairs, "
        f"above the var ceiling of {MAX_GRID_VAR_ENTRIES:,}; resample to "
        f"fewer mass bins or ask for fewer mobility channels "
        f"(--mobility-bins)"
    )


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
    return None


class _GridAccumulator:
    """The ``(pixel, m/z bin, mobility channel)`` cells of one slice.

    A pixel's raw cloud carries many points per cell -- that is what a
    TIMS ramp is -- so each pixel is collapsed onto its own occupied cells
    before anything is buffered. The matrix would sum the duplicates
    anyway; doing it per pixel is what keeps the accumulator the size of
    the answer rather than the size of the source.
    """

    def __init__(
        self,
        axis: NDArray[np.float64],
        row_for: RowLookup,
        grid: MobilityGrid,
    ) -> None:
        self._axis = axis
        self._row_for = row_for
        self._grid = grid
        self._rows: List[NDArray[np.int64]] = []
        self._keys: List[NDArray[np.int64]] = []
        self._data: List[NDArray[np.float64]] = []
        self.n_pixels = 0
        self.n_skipped = 0
        self.n_dropped = 0
        self.n_points = 0

    def add(
        self,
        coords: Coords,
        mzs: NDArray[np.float64],
        mobility: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        """Fold one pixel's ``(m/z, mobility, intensity)`` points in."""
        row = self._row_for(coords)
        if row is None:
            self.n_skipped += 1
            return
        self.n_pixels += 1
        mzs = np.asarray(mzs, dtype=np.float64)
        if mzs.size == 0:
            return
        mobility = np.asarray(mobility, dtype=np.float64)
        intensities = np.asarray(intensities, dtype=np.float64)
        # "In range" is the summed table's own rule: a peak outside the
        # mass axis is dropped there and must be dropped here, or the
        # marginal over channels would exceed the column it mirrors.
        in_range = (mzs >= self._axis[0]) & (mzs <= self._axis[-1])
        if not in_range.all():
            self.n_dropped += int(mzs.size - in_range.sum())
            mzs = mzs[in_range]
            mobility = mobility[in_range]
            intensities = intensities[in_range]
            if mzs.size == 0:
                return
        self.n_points += int(mzs.size)
        keys, values = self._cells(mzs, mobility, intensities)
        if keys.size == 0:
            return
        self._rows.append(np.full(keys.size, row, dtype=np.int64))
        self._keys.append(keys)
        self._data.append(values)

    def _cells(
        self,
        mzs: NDArray[np.float64],
        mobility: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        """One entry per occupied cell of this pixel, with its summed current."""
        from .base_spatialdata_converter import _nn_map_to_bins

        # Ascending flat key is ascending (mz, mobility), which is the
        # order var wants, for free.
        keys = _nn_map_to_bins(self._axis, mzs).astype(
            np.int64
        ) * self._grid.n_channels + self._grid.assign(mobility).astype(np.int64)
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        summed = np.bincount(
            np.asarray(inverse).ravel(), weights=intensities, minlength=unique_keys.size
        )
        nonzero = summed != 0
        return unique_keys[nonzero], summed[nonzero]

    def matrix(
        self, n_obs: int
    ) -> Optional[Tuple[sparse.csc_matrix, NDArray[np.int64]]]:
        """The pixels x features matrix and the flat cell keys it is built on.

        Only cells that carry ion current somewhere become features: the
        grid spans mass bins x channels, and a real acquisition occupies a
        small, structured part of it.
        """
        if self.n_skipped:
            logger.warning(
                "%d mobility spectra had no row in the MSI table and were skipped",
                self.n_skipped,
            )
        if self.n_dropped:
            logger.warning(
                "%d mobility points fell outside the mass axis and were "
                "dropped from the mobility-resolved table",
                self.n_dropped,
            )
        if not self._rows:
            logger.warning(
                "No mobility spectra matched the MSI table; no mobility table written"
            )
            return None
        keys = np.concatenate(self._keys)
        unique_keys = np.unique(keys)
        # The var ceiling, on the count rather than on the bound: this is
        # the first moment the real number is known, and it is checked
        # before the labels and the matrix are built rather than after.
        refusal = var_ceiling_refusal(int(unique_keys.size))
        if refusal is not None:
            logger.warning("No mobility-resolved table: %s", refusal)
            return None
        columns = np.searchsorted(unique_keys, keys)
        matrix = sparse.coo_matrix(
            (np.concatenate(self._data), (np.concatenate(self._rows), columns)),
            shape=(n_obs, int(unique_keys.size)),
        ).tocsc()
        return matrix, unique_keys


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
    return pd.DataFrame(
        {
            "mz": axis[mz_index],
            "mobility": centres[mobility_index],
            "mz_index": mz_index,
            "mobility_index": mobility_index,
        },
        index=_var_labels(mz_index, mobility_index),
    )


def _build_from_grid(
    reader: BaseMSIReader,
    row_for: RowLookup,
    common_mass_axis: NDArray[np.float64],
    grid: MobilityGrid,
    n_obs: int,
) -> Optional[Tuple[sparse.csc_matrix, pd.DataFrame]]:
    """Bin every pixel's point cloud onto the grid; ``(matrix, var)`` or ``None``."""
    axis = np.asarray(common_mass_axis, dtype=np.float64)
    accumulator = _GridAccumulator(axis, row_for, grid)
    for coords, mzs, mobility, intensities in reader.iter_mobility_spectra():
        accumulator.add(coords, mzs, mobility, intensities)
    accumulated = accumulator.matrix(n_obs)
    if accumulated is None:
        return None
    matrix, unique_keys = accumulated
    var = _grid_feature_var(unique_keys, axis, grid)
    logger.info(
        "Mobility grid: %d points over %d pixels onto %d occupied "
        "(m/z bin, channel) cells of %d x %d, %d channels used",
        accumulator.n_points,
        accumulator.n_pixels,
        int(var.shape[0]),
        int(axis.size),
        grid.n_channels,
        int(np.unique(var["mobility_index"].to_numpy()).size),
    )
    return matrix, var


# ----------------------------------------------------------------------
# The shared-axis mechanism, and the entry point over both
# ----------------------------------------------------------------------


def _build_from_shared_axis(
    reader: BaseMSIReader,
    row_for: RowLookup,
    common_mass_axis: NDArray[np.float64],
    n_obs: int,
) -> Optional[Tuple[sparse.csc_matrix, pd.DataFrame]]:
    """Scatter the source's own feature pairs; ``(matrix, var)`` or ``None``."""
    features = reader.get_shared_mobility_features()
    if features is None or features[0].size == 0:
        return None
    unique_pairs, source_to_feature = _feature_axis(*features)
    matrix = _accumulate(reader, row_for, unique_pairs, source_to_feature, n_obs)
    if matrix is None:
        return None
    var, n_mobility_values = _feature_var(unique_pairs, common_mass_axis)
    logger.info(
        "Mobility-resolved table: %d pixels x %d (m/z, mobility) features, "
        "%d non-zeros, %d distinct mobility values",
        n_obs,
        int(var.shape[0]),
        int(matrix.nnz),
        n_mobility_values,
    )
    return matrix, var


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

    Returns:
        A ``TableModel``-parsed AnnData, or ``None`` when no table can be
        written (logged at info level with the reason).
    """
    if not reader.has_ion_mobility:
        return None
    n_obs = int(len(obs))
    row_for = row_lookup(obs, z_value, pixel_key)
    grid_uns: Optional[Dict[str, Any]] = None
    if reader.has_shared_mobility_axis:
        built = _build_from_shared_axis(reader, row_for, common_mass_axis, n_obs)
    else:
        refusal = grid_refusal(reader, common_mass_axis, grid)
        if refusal is not None or grid is None:
            logger.info(
                "No mobility-resolved table: the source carries mobility per "
                "pixel rather than as a shared feature axis, and %s",
                refusal or "no common mobility grid was asked for",
            )
            return None
        built = _build_from_grid(reader, row_for, common_mass_axis, grid, n_obs)
        grid_uns = grid.to_uns()
    if built is None:
        return None
    matrix, var = built
    return _assemble(matrix, var, obs, region_key, slice_key, uns, grid_uns)


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
    adata.uns["feature_axis"] = _jsonify_string_lists(feature_axis_block(slice_key))
    if grid_uns is not None:
        adata.uns["mobility_grid"] = grid_uns
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
