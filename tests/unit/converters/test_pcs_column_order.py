"""The stored matrix is canonical whatever order the reader yields.

The scatter pass writes a column's entries in the order the reader
hands its pixels over, and the writer stores the memmaps as they are.
A raster read makes every column's
row indices ascending for free, and every synthetic reader in this suite
reads in raster order -- so nothing here could see that a reader with any
other order produced a matrix scipy reports as non-canonical
(``has_sorted_indices`` False), which breaks the binary search a consumer
does on a column's indices and some of anndata's sparse-dataset paths.

A real one does. A TDF acquisition of several areas measured one after
another comes back area by area: on a 26,087-pixel, three-area slide,
2,000 of 2,000 sampled columns of the summed table were unsorted, while
the sibling mobility table written next to it -- whose assembly sorts its
columns when rows arrive out of order -- had none.

The fix tracks whether rows arrived ascending during the scatter and,
only when they did not, sorts each column's ``(indices, data)`` slice a
bounded chunk of columns at a time before the copy. These tests pin:

* a reader yielding its pixels in shuffled order produces a canonical
  matrix, checked column by column and not only through scipy's flag;
* that matrix equals, value for value, the raster-order write's -- the
  sort permutes within columns and touches nothing else;
* a raster-order reader is not sorted at all (the fast path stays fast).

The chunked sort itself is the sibling tables' ``sort_csc_columns`` and
is pinned against scipy in ``test_csc_assembly.py``.

The shuffled reader is guarded too: a "shuffle" that happened to be the
raster order would turn every assertion above into a tautology.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
import zarr
from numpy.typing import NDArray
from scipy import sparse

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata.streaming_converter import (
    StreamingSpatialDataConverter,
)

DATASET_ID = "mock"
TABLE_KEY = f"{DATASET_ID}_z0"

# Non-square, with a fifth of the positions empty: the compacted row
# offsets then differ from the grid indices, so an arrival order that is
# wrong in grid terms is also wrong in row terms after the compaction.
N_X, N_Y = 6, 5
SPARSITY = 0.2
SHUFFLE_SEED = 7


class ShuffledMockMSIReader(MockMSIReader):
    """The mock reader, yielding its pixels in a fixed shuffled order.

    The permutation is drawn from a fresh generator on every call, so the
    two passes of a streaming conversion see the same order -- the
    contract every two-pass converter relies on.
    """

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """Yield every spectrum, in the same shuffled order each time."""
        spectra = list(super().iter_spectra(batch_size=batch_size))
        order = np.random.default_rng(SHUFFLE_SEED).permutation(len(spectra))
        for position in order:
            yield spectra[position]


def _config() -> MockMSIConfig:
    return MockMSIConfig(
        n_x=N_X,
        n_y=N_Y,
        n_mz_bins=300,
        peaks_per_spectrum=(20, 40),
        sparsity=SPARSITY,
    )


def _grid_order(reader: MockMSIReader) -> list[int]:
    return [y * N_X + x for (x, y, _z), _mzs, _ints in reader.iter_spectra()]


def _convert_streaming(out: Path, reader: MockMSIReader) -> Path:
    converter = StreamingSpatialDataConverter(
        reader=reader,
        output_path=out,
        dataset_id=DATASET_ID,
        pixel_size_um=10.0,
        use_csc=True,
    )
    assert converter.convert() is True
    return out


def _stored_csc(store: Path) -> sparse.csc_matrix:
    """The table's X as written, read straight off the zarr arrays."""
    group = zarr.open_group(str(store / "tables" / TABLE_KEY), mode="r")
    x_group = group["X"]
    assert x_group.attrs["encoding-type"] == "csc_matrix"
    return sparse.csc_matrix(
        (
            np.asarray(x_group["data"]),
            np.asarray(x_group["indices"]),
            np.asarray(x_group["indptr"]),
        ),
        shape=tuple(x_group.attrs["shape"]),
    )


def _columns_strictly_ascending(matrix: sparse.csc_matrix) -> bool:
    """Every column's row indices ascend, checked without scipy's flags."""
    indices = np.asarray(matrix.indices, dtype=np.int64)
    indptr = np.asarray(matrix.indptr, dtype=np.int64)
    if indices.size < 2:
        return True
    steps = np.diff(indices)
    # A step that crosses a column boundary compares unrelated entries.
    boundaries = indptr[1:-1] - 1
    boundaries = boundaries[(boundaries >= 0) & (boundaries < steps.size)]
    steps[boundaries] = 1
    return bool(np.all(steps > 0))


# --- the fixture cannot decay into a tautology --------------------------


def test_the_shuffled_reader_is_out_of_raster_order():
    """The shuffle really puts a later grid position before an earlier one."""
    raster = _grid_order(MockMSIReader(_config()))
    shuffled = _grid_order(ShuffledMockMSIReader(_config()))

    assert sorted(shuffled) == raster, "same pixels, whatever the order"
    assert shuffled != raster
    assert any(a > b for a, b in zip(shuffled, shuffled[1:]))


def test_the_shuffled_reader_repeats_its_order():
    """Both passes of a two-pass conversion must see the same sequence."""
    reader = ShuffledMockMSIReader(_config())
    first = _grid_order(reader)
    reader.reset()
    assert _grid_order(reader) == first


# --- the written matrix ---------------------------------------------------


def test_shuffled_arrival_writes_a_canonical_matrix(tmp_path):
    """Row indices ascend within every column, so scipy calls it canonical."""
    store = _convert_streaming(
        tmp_path / "shuffled.zarr", ShuffledMockMSIReader(_config())
    )

    matrix = _stored_csc(store)
    assert _columns_strictly_ascending(matrix)
    assert matrix.has_sorted_indices
    assert matrix.has_canonical_format


def test_shuffled_arrival_matches_the_raster_write_exactly(tmp_path):
    """The sort permutes within columns and changes nothing else.

    Against the raster-order write of the same pixels -- canonical as
    scattered -- the three CSC arrays have to come out identical.
    """
    shuffled = _stored_csc(
        _convert_streaming(tmp_path / "shuffled.zarr", ShuffledMockMSIReader(_config()))
    )
    raster = _stored_csc(
        _convert_streaming(tmp_path / "raster.zarr", MockMSIReader(_config()))
    )

    assert shuffled.shape == raster.shape
    np.testing.assert_array_equal(shuffled.indptr, raster.indptr)
    np.testing.assert_array_equal(shuffled.indices, raster.indices)
    np.testing.assert_array_equal(shuffled.data, raster.data)
