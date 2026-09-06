"""The mobility-resolved table: pixels x (m/z, mobility) features.

The MSI table Thyra always writes is summed over ion mobility. When the
source shares one set of (m/z, mobility) feature pairs across every pixel
-- a continuous imzML export with a mobility array, as TIMSImaging and
TIMSCONVERT write -- those pairs are already a feature list, and this
module writes them as a second table in the same store:

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


def build_mobility_table(
    reader: BaseMSIReader,
    obs: pd.DataFrame,
    common_mass_axis: NDArray[np.float64],
    slice_key: str,
    region_key: str,
    uns: Dict[str, Any],
    z_value: Optional[int] = None,
    pixel_key: Optional[Callable[[Coords], Optional[str]]] = None,
) -> Optional[Any]:
    """Build the mobility-resolved table for one MSI table, or ``None``.

    Args:
        reader: The source reader; must report a shared mobility axis.
        obs: The MSI table's ``obs`` (its rows define this table's rows; it
            needs ``x`` and ``y`` columns, and ``z`` when the store is 3D).
        common_mass_axis: The MSI table's ``var["mz"]``, for ``mz_index``.
        slice_key: The MSI table's element key (``{id}_z0``).
        region_key: The shapes element both tables annotate.
        uns: The provenance block to store on the table (already built).
        z_value: The plane this table covers when ``obs`` has no ``z``
            column; pixels on other planes are skipped.
        pixel_key: Optional override mapping a reader coordinate to an
            ``obs`` index label; the default matches on ``(x, y[, z])``.

    Returns:
        A ``TableModel``-parsed AnnData, or ``None`` when the reader has no
        shared mobility axis (logged at info level with the reason).
    """
    if not reader.has_ion_mobility:
        return None
    if not reader.has_shared_mobility_axis:
        logger.info(
            "Source carries ion mobility per pixel rather than a shared feature "
            "axis; the mobility-resolved table needs a common mobility grid, "
            "which is not available yet, so only the summed MSI table is written"
        )
        return None
    features = reader.get_shared_mobility_features()
    if features is None or features[0].size == 0:
        return None

    from anndata import AnnData
    from spatialdata.models import TableModel

    from .base_spatialdata_converter import _jsonify_string_lists

    unique_pairs, source_to_feature = _feature_axis(*features)
    n_obs = int(len(obs))
    matrix = _accumulate(
        reader,
        row_lookup(obs, z_value, pixel_key),
        unique_pairs,
        source_to_feature,
        n_obs,
    )
    if matrix is None:
        return None

    var, n_mobility_values = _feature_var(unique_pairs, common_mass_axis)

    table_obs = obs.copy()
    table_obs["region"] = pd.Categorical([region_key] * n_obs)
    table_obs["instance_key"] = table_obs.index.astype(str)

    adata = AnnData(X=matrix, obs=table_obs, var=var)
    adata.uns.update(uns)
    # String lists become JSON strings, as everywhere else in uns: a list of
    # strings does not round-trip through zarr on numpy 2.1-2.2.
    adata.uns["feature_axis"] = _jsonify_string_lists(feature_axis_block(slice_key))

    logger.info(
        "Mobility-resolved table: %d pixels x %d (m/z, mobility) features, "
        "%d non-zeros, %d distinct mobility values",
        n_obs,
        int(var.shape[0]),
        int(matrix.nnz),
        n_mobility_values,
    )
    return TableModel.parse(
        adata,
        region=region_key,
        region_key="region",
        instance_key="instance_key",
    )
