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
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from ...core.base_reader import BaseMSIReader
from ...core.msms import FragmentationSchedule, windows_overlap
from .mobility_table import Coords, RowLookup, row_lookup

logger = logging.getLogger(__name__)

#: Suffix appended to the MSI table's key for its demultiplexed sibling.
MSMS_TABLE_SUFFIX = "_msms"


def msms_table_key(table_key: str) -> str:
    """The element key of the demultiplexed MS/MS sibling of ``table_key``."""
    return f"{table_key}{MSMS_TABLE_SUFFIX}"


def feature_axis_block(summed_table_key: str) -> Dict[str, Any]:
    """The ``uns["feature_axis"]`` descriptor of a demultiplexed table."""
    return {
        "dims": ["precursor_mz", "mz"],
        "sorted": True,
        "summed_table": summed_table_key,
    }


def demultiplex_refusal(schedule: Optional[FragmentationSchedule]) -> Optional[str]:
    """Why this acquisition must not be demultiplexed, or ``None``.

    Each condition is a property of the acquisition that would make the
    split an approximation rather than a filter, so the answer is to
    refuse and say which one failed -- never to apportion ion current
    between precursors that cannot be told apart.
    """
    if schedule is None or not schedule.is_msms:
        return "the acquisition is not MS/MS"
    if len(schedule.windows) < 2:
        return (
            "the acquisition isolates a single precursor, so the summed table "
            "is already its fragment spectrum"
        )
    if not schedule.constant_across_pixels:
        return (
            "the precursor schedule is not constant across pixels, so the "
            "precursors are not a global feature axis"
        )
    if windows_overlap(schedule.windows):
        return (
            "the isolation windows overlap or carry no mobility scan range, "
            "so they cannot be separated by scan number alone"
        )
    return None


def _precursor_ranks(
    schedule: FragmentationSchedule,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """The distinct precursor m/z, ascending, and each window's rank in them.

    Two windows that isolate the same m/z in two mobility slices are one
    precursor, not two: summing them is exact, and keeping them apart
    would put the same ``(precursor_mz, mz)`` pair in ``var`` twice.
    """
    targets = np.array([w.target for w in schedule.windows], dtype=np.float64)
    unique_targets, ranks = np.unique(targets, return_inverse=True)
    ranks = np.asarray(ranks).ravel().astype(np.int64)
    if unique_targets.size != targets.size:
        logger.info(
            "%d isolation windows share a precursor m/z with another and are "
            "merged into one precursor each",
            int(targets.size - unique_targets.size),
        )
    return unique_targets, ranks


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


class _Accumulator:
    """The ``(pixel, precursor, mass axis column)`` triples of one slice."""

    def __init__(
        self,
        axis: NDArray[np.float64],
        row_for: RowLookup,
        window_rank: NDArray[np.int64],
    ) -> None:
        self._axis = axis
        self._row_for = row_for
        self._window_rank = window_rank
        self._rows: List[NDArray[np.int64]] = []
        self._keys: List[NDArray[np.int64]] = []
        self._data: List[NDArray[np.float64]] = []
        self.n_skipped = 0
        self.n_dropped = 0

    def add(
        self,
        coords: Coords,
        window_index: int,
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        row = self._row_for(coords)
        if row is None:
            self.n_skipped += 1
            return
        columns, in_range = _bin_indices(self._axis, mzs)
        if columns.size == 0:
            self.n_dropped += int(mzs.size)
            return
        if columns.size != mzs.size:
            self.n_dropped += int(mzs.size - columns.size)
            intensities = intensities[in_range]
        rank = int(self._window_rank[window_index])
        self._rows.append(np.full(columns.size, row, dtype=np.int64))
        # One key per (precursor, mass axis column) pair. Ascending key is
        # ascending (precursor_mz, mz), which is the order var wants.
        self._keys.append(rank * self._axis.size + columns)
        self._data.append(np.asarray(intensities, dtype=np.float64))

    def matrix(
        self, n_obs: int
    ) -> Optional[Tuple[sparse.csc_matrix, NDArray[np.int64]]]:
        """The pixels x features matrix and the feature keys it is built on.

        Only the ``(precursor, column)`` pairs that carry ion current
        somewhere become features: the precursor axis is discrete but the
        fragment axis is the whole MSI mass axis, and a precursor's
        fragments occupy a small part of it.
        """
        if self.n_skipped:
            logger.warning(
                "%d demultiplexed spectra had no row in the MSI table and "
                "were skipped",
                self.n_skipped,
            )
        if self.n_dropped:
            logger.warning(
                "%d fragment peaks fell outside the mass axis and were "
                "dropped from the demultiplexed table",
                self.n_dropped,
            )
        if not self._rows:
            logger.warning(
                "No demultiplexed spectra matched the MSI table; no MS/MS "
                "table written"
            )
            return None

        keys = np.concatenate(self._keys)
        unique_keys = np.unique(keys)
        columns = np.searchsorted(unique_keys, keys)
        matrix = sparse.coo_matrix(
            (
                np.concatenate(self._data),
                (np.concatenate(self._rows), columns),
            ),
            shape=(n_obs, int(unique_keys.size)),
        ).tocsc()
        return matrix, unique_keys


def _feature_var(
    unique_keys: NDArray[np.int64],
    axis: NDArray[np.float64],
    precursor_mz: NDArray[np.float64],
) -> pd.DataFrame:
    """The ``var`` of the demultiplexed table, one row per feature."""
    precursor_index = (unique_keys // axis.size).astype(np.int64)
    mz_index = (unique_keys % axis.size).astype(np.int64)
    return pd.DataFrame(
        {
            "precursor_mz": precursor_mz[precursor_index],
            "mz": axis[mz_index],
            "precursor_index": precursor_index,
            "mz_index": mz_index,
        },
        index=[
            f"p{p}_mz{i}" for p, i in zip(precursor_index.tolist(), mz_index.tolist())
        ],
    )


def build_msms_table(
    reader: BaseMSIReader,
    obs: pd.DataFrame,
    common_mass_axis: NDArray[np.float64],
    slice_key: str,
    region_key: str,
    uns: Dict[str, Any],
    z_value: Optional[int] = None,
    pixel_key: Optional[Callable[[Coords], Optional[str]]] = None,
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

    from anndata import AnnData
    from spatialdata.models import TableModel

    from .base_spatialdata_converter import _jsonify_string_lists

    precursor_mz, window_rank = _precursor_ranks(schedule)
    accumulator = _Accumulator(axis, row_lookup(obs, z_value, pixel_key), window_rank)
    for coords, window_index, mzs, intensities in reader.iter_precursor_spectra():
        accumulator.add(coords, window_index, mzs, intensities)
    accumulated = accumulator.matrix(int(len(obs)))
    if accumulated is None:
        return None
    matrix, unique_keys = accumulated

    var = _feature_var(unique_keys, axis, precursor_mz)
    table_obs = obs.copy()
    n_obs = int(len(obs))
    table_obs["region"] = pd.Categorical([region_key] * n_obs)
    table_obs["instance_key"] = table_obs.index.astype(str)

    adata = AnnData(X=matrix, obs=table_obs, var=var)
    adata.uns.update(uns)
    # String lists become JSON strings, as everywhere else in uns: a list of
    # strings does not round-trip through zarr on numpy 2.1-2.2.
    adata.uns["feature_axis"] = _jsonify_string_lists(feature_axis_block(slice_key))

    logger.info(
        "Demultiplexed MS/MS table: %d pixels x %d (precursor, fragment) "
        "features over %d precursors, %d non-zeros",
        n_obs,
        int(var.shape[0]),
        int(precursor_mz.size),
        int(matrix.nnz),
    )
    return TableModel.parse(
        adata,
        region=region_key,
        region_key="region",
        instance_key="instance_key",
    )
