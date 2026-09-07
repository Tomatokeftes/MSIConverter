"""The out-of-core CSC engine both sibling tables are built on.

Pinned against scipy: whatever the engine writes through its two passes
must be the matrix ``coo_matrix(...).tocsc()`` would have built from the
same triples in one, entry for entry and dtype for dtype -- that is the
route it replaced, and the stores written by the two must not differ.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from thyra.converters.spatialdata.csc_assembly import (
    MAX_COUNT_BYTES,
    CscAssembly,
    count_refusal,
    index_dtype,
    remove_scratch,
)
from thyra.converters.spatialdata.mobility_table import (
    collapse_row,
    disambiguate_labels,
    int_strings,
)

SPAN = 1000


def _rows(rng, n_rows, keys_per_row=40):
    """One row's unique ascending keys and values, for every row."""
    rows = []
    for row in range(n_rows):
        keys = np.unique(rng.integers(0, SPAN, keys_per_row, dtype=np.int64))
        values = rng.random(keys.size) + 0.5
        rows.append((row, keys, values))
    return rows


def _reference(rows, n_rows, keep_empty):
    r = np.concatenate([np.full(k.size, row) for row, k, _v in rows])
    c = np.concatenate([k for _row, k, _v in rows])
    d = np.concatenate([v for _row, _k, v in rows])
    if keep_empty:
        return sparse.coo_matrix((d, (r, c)), shape=(n_rows, SPAN)).tocsc()
    occupied = np.unique(c)
    columns = np.searchsorted(occupied, c)
    return sparse.coo_matrix((d, (r, columns)), shape=(n_rows, occupied.size)).tocsc()


def _build(rows, n_rows, scratch, keep_empty=False, order=None):
    assembly = CscAssembly(SPAN, n_rows, keep_empty_columns=keep_empty)
    sequence = rows if order is None else [rows[i] for i in order]
    for row, keys, _values in sequence:
        assembly.count(row, keys)
    assembly.finish_counting()
    assembly.allocate(scratch)
    for row, keys, values in sequence:
        assembly.scatter(row, keys, values)
    return assembly


def _assert_same(built, expected):
    assert built.shape == expected.shape
    assert built.indices.dtype == expected.indices.dtype
    assert built.indptr.dtype == expected.indptr.dtype
    np.testing.assert_array_equal(built.indptr, expected.indptr)
    np.testing.assert_array_equal(built.indices, expected.indices)
    np.testing.assert_array_equal(built.data, expected.data)
    assert built.has_canonical_format


class TestAgainstScipy:
    @pytest.mark.parametrize("keep_empty", [False, True], ids=["occupied", "all"])
    def test_rows_in_order_are_the_coo_to_csc_matrix(self, tmp_path, keep_empty):
        rng = np.random.default_rng(1)
        rows = _rows(rng, 60)
        assembly = _build(rows, 60, tmp_path / "s", keep_empty)
        assert assembly.rows_in_order
        _assert_same(assembly.matrix(), _reference(rows, 60, keep_empty))
        assembly.release()

    def test_rows_out_of_order_are_sorted_into_the_same_matrix(self, tmp_path):
        # A source read in some order other than the table's rows: the
        # columns are put in row order afterwards, chunk by chunk, so the
        # stored matrix is canonical either way.
        from thyra.converters.spatialdata import csc_assembly

        rng = np.random.default_rng(2)
        rows = _rows(rng, 80)
        order = rng.permutation(80)
        assembly = _build(rows, 80, tmp_path / "s", order=order)
        assert not assembly.rows_in_order
        csc_assembly.SORT_CHUNK_ENTRIES, kept = 97, csc_assembly.SORT_CHUNK_ENTRIES
        try:
            matrix = assembly.matrix()
        finally:
            csc_assembly.SORT_CHUNK_ENTRIES = kept
        _assert_same(matrix, _reference(rows, 80, False))
        assembly.release()

    def test_a_row_in_several_groups_is_one_row(self, tmp_path):
        # The demultiplexer hands a pixel in once per precursor, each with
        # its own disjoint keys; the engine sees the same row twice.
        rows = [
            (0, np.array([1, 5]), np.array([1.0, 2.0])),
            (0, np.array([7, 9]), np.array([3.0, 4.0])),
            (1, np.array([5]), np.array([5.0])),
        ]
        assembly = _build(rows, 2, tmp_path / "s")
        assert assembly.rows_in_order
        _assert_same(assembly.matrix(), _reference(rows, 2, False))
        assembly.release()

    def test_the_matrix_does_not_copy_the_memmaps(self, tmp_path):
        rows = _rows(np.random.default_rng(3), 5)
        assembly = _build(rows, 5, tmp_path / "s")
        matrix = assembly.matrix()
        assert np.shares_memory(matrix.data, assembly._data)
        assert np.shares_memory(matrix.indices, assembly._indices)
        assembly.release()

    def test_an_empty_row_is_a_row(self, tmp_path):
        rows = [
            (0, np.array([], dtype=np.int64), np.array([])),
            (1, np.array([3]), np.array([1.0])),
        ]
        assembly = _build(rows, 2, tmp_path / "s")
        matrix = assembly.matrix()
        assert matrix.shape == (2, 1)
        np.testing.assert_array_equal(matrix.toarray(), [[0.0], [1.0]])
        assembly.release()


class TestTheGuards:
    def test_the_count_array_has_a_budget_naming_the_lever(self):
        assert count_refusal(MAX_COUNT_BYTES // 4) is None
        refusal = count_refusal(MAX_COUNT_BYTES // 4 + 1)
        assert refusal is not None and "--resample-bins" in refusal
        with pytest.raises(MemoryError):
            CscAssembly(MAX_COUNT_BYTES // 4 + 1, 1)

    def test_the_index_dtype_is_scipys_rule(self):
        assert index_dtype(10, 10) is np.int32
        assert index_dtype(2**31, 10) is np.int64
        assert index_dtype(10, 2**31) is np.int64

    def test_the_passes_must_agree_on_the_rows(self, tmp_path):
        assembly = CscAssembly(SPAN, 3)
        assembly.count(0, np.array([1, 2]))
        assembly.count(1, np.array([2]))
        assembly.finish_counting()
        assembly.allocate(tmp_path / "s")
        assembly.scatter(0, np.array([1, 2]), np.array([1.0, 1.0]))
        # Row 1 was counted and never scattered: a reserved slot nothing
        # wrote, which must not become a silent zero at row 0.
        with pytest.raises(RuntimeError, match="disagree"):
            assembly.matrix()
        assembly.release()

    def test_the_feature_ceiling_is_checkable_before_allocation(self, tmp_path):
        assembly = CscAssembly(SPAN, 2)
        assembly.count(0, np.array([1, 4, 9]))
        assert assembly.finish_counting() == 3
        np.testing.assert_array_equal(assembly.unique_keys, [1, 4, 9])
        assert assembly.indptr is None  # nothing sized yet

    def test_release_lets_the_scratch_go(self, tmp_path):
        scratch = Path(tmp_path / "s")
        rows = _rows(np.random.default_rng(4), 3)
        assembly = _build(rows, 3, scratch)
        matrix = assembly.matrix()
        del matrix
        assembly.release()
        remove_scratch(scratch)
        assert not scratch.exists()


class TestRowHelpers:
    def test_collapse_row_leaves_a_canonical_row_alone(self):
        keys = np.array([2, 5, 9])
        values = np.array([1.0, 2.0, 3.0])
        out_keys, out_values = collapse_row(keys, values)
        assert out_keys is keys and out_values is values

    def test_collapse_row_merges_repeats_and_sorts(self):
        keys = np.array([9, 2, 9, 5, 2])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out_keys, out_values = collapse_row(keys, values)
        np.testing.assert_array_equal(out_keys, [2, 5, 9])
        np.testing.assert_array_equal(out_values, [7.0, 4.0, 4.0])

    def test_disambiguation_matches_the_dict_it_replaced(self):
        rng = np.random.default_rng(5)
        group = rng.integers(0, 20, 200, dtype=np.int64)
        labels = np.strings.add("k", int_strings(group))
        seen = {}
        expected = []
        for label in labels.tolist():
            n = seen.get(label, 0)
            seen[label] = n + 1
            expected.append(label if n == 0 else f"{label}_{n}")
        assert disambiguate_labels(labels, group).tolist() == expected

    def test_unique_labels_pass_through_untouched(self):
        labels = np.strings.add("k", int_strings([3, 1, 2]))
        assert disambiguate_labels(labels, np.array([3, 1, 2])) is labels
