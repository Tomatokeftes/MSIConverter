"""The streaming PCS matrix is canonical whatever order the reader yields.

``_scatter_spectra_direct`` writes a column's entries in the order the
reader hands its pixels over, and ``_write_csc_arrays_to_zarr`` copies
the memmaps to the store unchanged. A raster read makes every column's
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
* that matrix equals, value for value, both the in-memory route's and
  the raster-order streaming write's -- the sort permutes within columns
  and touches nothing else;
* a raster-order reader is not sorted at all (the fast path stays fast);
* the chunked sort itself agrees with scipy across chunk boundaries,
  including a budget smaller than a single column.

The shuffled reader is guarded too: a "shuffle" that happened to be the
raster order would turn every assertion above into a tautology.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray
from scipy import sparse

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata import streaming_converter as mod
from thyra.converters.spatialdata.spatialdata_2d_converter import SpatialData2DConverter
from thyra.converters.spatialdata.streaming_converter import (
    SPATIALDATA_AVAILABLE,
    StreamingSpatialDataConverter,
    _sort_csc_columns,
)

pytestmark = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
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


def _convert_in_memory(out: Path, reader: MockMSIReader) -> Path:
    converter = SpatialData2DConverter(
        reader=reader,
        output_path=out,
        dataset_id=DATASET_ID,
        pixel_size_um=10.0,
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


def _dense_by_instance(store: Path) -> Tuple[np.ndarray, list[str]]:
    """The table densified, with rows in ``instance_id`` order."""
    import spatialdata

    table = spatialdata.read_zarr(str(store)).tables[TABLE_KEY]
    matrix = table.X
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    ids = [str(i) for i in table.obs.index]
    order = np.argsort(np.asarray(ids, dtype=np.int64))
    return dense[order], [ids[i] for i in order]


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


def test_shuffled_arrival_equals_the_in_memory_route(tmp_path):
    """Pixel for pixel, value for value, the two routes agree.

    Compared densely, aligned on ``instance_id``: the in-memory route
    drops explicit zeros where the streaming route keeps them, so neither
    ``nnz`` nor a positional row lookup would be a fair comparison.
    """
    streamed, streamed_ids = _dense_by_instance(
        _convert_streaming(tmp_path / "shuffled.zarr", ShuffledMockMSIReader(_config()))
    )
    in_memory, in_memory_ids = _dense_by_instance(
        _convert_in_memory(tmp_path / "in_memory.zarr", MockMSIReader(_config()))
    )

    assert streamed_ids == in_memory_ids
    assert streamed.shape == in_memory.shape
    np.testing.assert_array_equal(streamed, in_memory)


# --- the fast path stays fast ---------------------------------------------


@pytest.mark.parametrize(
    "reader_type,expected_sorts",
    [(MockMSIReader, 0), (ShuffledMockMSIReader, 1)],
    ids=["raster", "shuffled"],
)
def test_only_an_out_of_order_arrival_is_sorted(
    tmp_path, monkeypatch, reader_type, expected_sorts
):
    """A raster read is canonical as scattered and must not pay for a sort."""
    calls: list = []
    real_sort = mod._sort_csc_columns

    def _counting_sort(*args, **kwargs):
        calls.append(args)
        return real_sort(*args, **kwargs)

    monkeypatch.setattr(mod, "_sort_csc_columns", _counting_sort)
    store = _convert_streaming(tmp_path / "out.zarr", reader_type(_config()))

    assert len(calls) == expected_sorts
    assert _columns_strictly_ascending(_stored_csc(store))


# --- the chunked sort itself ----------------------------------------------


def _scrambled_within_columns(
    matrix: sparse.csc_matrix, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """``(indices, data)`` of ``matrix`` with each column's entries permuted."""
    rng = np.random.default_rng(seed)
    indices = matrix.indices.astype(np.int32).copy()
    data = matrix.data.copy()
    indptr = matrix.indptr
    for column in range(matrix.shape[1]):
        lo, hi = int(indptr[column]), int(indptr[column + 1])
        if hi - lo > 1:
            order = rng.permutation(hi - lo)
            indices[lo:hi] = indices[lo:hi][order]
            data[lo:hi] = data[lo:hi][order]
    return indices, data


@pytest.mark.parametrize(
    "chunk_entries", [1, 7, 10**6], ids=["one-column", "mid-column", "all"]
)
def test_sort_csc_columns_matches_scipy_across_chunk_boundaries(
    tmp_path, chunk_entries
):
    """In place on memmaps, the chunked sort reproduces ``sort_indices``.

    A budget of 1 forces one column per chunk (the branch that takes a
    column exceeding the budget on its own); 7 lands chunk boundaries in
    the middle of a run of columns; the last sorts everything at once.
    """
    n_rows, n_cols = 53, 41
    expected = sparse.random(
        n_rows, n_cols, density=0.3, format="csc", random_state=3, dtype=np.float64
    )
    expected.sort_indices()
    # Some empty columns, so the budget arithmetic meets zero-width columns.
    expected = sparse.csc_matrix(expected)
    expected[:, [0, 5, 6, n_cols - 1]] = 0
    expected.eliminate_zeros()
    expected.sort_indices()
    assert expected.nnz > 0

    scrambled_indices, scrambled_data = _scrambled_within_columns(expected, seed=11)
    assert not _columns_strictly_ascending(
        sparse.csc_matrix(
            (scrambled_data, scrambled_indices, expected.indptr), shape=expected.shape
        )
    )

    indices = np.memmap(
        tmp_path / "indices.bin", dtype=np.int32, mode="w+", shape=(expected.nnz,)
    )
    data = np.memmap(
        tmp_path / "data.bin", dtype=np.float64, mode="w+", shape=(expected.nnz,)
    )
    indices[:] = scrambled_indices
    data[:] = scrambled_data

    _sort_csc_columns(
        indices,
        data,
        expected.indptr.astype(np.int64),
        n_rows,
        chunk_entries=chunk_entries,
    )

    np.testing.assert_array_equal(np.asarray(indices), expected.indices)
    np.testing.assert_array_equal(np.asarray(data), expected.data)
