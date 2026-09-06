"""The fixed var column contract, checked straight off the zarr layout.

These tests fabricate minimal zarr stores rather than converting, so
they run without the SpatialData stack and cover the failure shapes a
conversion can never produce.
"""

import numpy as np
import pytest
import zarr

from thyra.metadata.schema import check_store_var_conventions


def _store_with_var(tmp_path, mz=None, extra=None, skip_var=False):
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="a")
    table = root.create_group("tables").create_group("t")
    if not skip_var:
        var = table.create_group("var")
        if mz is not None:
            var.create_array("mz", data=np.asarray(mz))
        if extra is not None:
            var.create_array(extra, data=np.zeros(3))
    return tmp_path / "store.zarr"


def _messages(issues):
    return " ".join(i.message for i in issues)


class TestCheckStoreVarConventions:
    def test_conforming_var_has_no_issues(self, tmp_path):
        store = _store_with_var(tmp_path, mz=[100.0, 200.5, 300.0])
        assert check_store_var_conventions(store) == {"t": []}

    def test_missing_var_group_is_an_error(self, tmp_path):
        store = _store_with_var(tmp_path, skip_var=True)
        issues = check_store_var_conventions(store)["t"]
        assert issues and issues[0].severity == "error"

    def test_missing_mz_column_is_an_error(self, tmp_path):
        store = _store_with_var(tmp_path, mz=None, extra="flight_time")
        issues = check_store_var_conventions(store)["t"]
        assert "'mz' is missing" in _messages(issues)

    def test_non_monotonic_mz_is_an_error(self, tmp_path):
        store = _store_with_var(tmp_path, mz=[100.0, 300.0, 200.0])
        issues = check_store_var_conventions(store)["t"]
        assert "strictly increasing" in _messages(issues)

    def test_non_finite_mz_is_an_error(self, tmp_path):
        store = _store_with_var(tmp_path, mz=[100.0, np.nan, 300.0])
        issues = check_store_var_conventions(store)["t"]
        assert "non-finite" in _messages(issues)

    def test_not_a_store_raises(self, tmp_path):
        zarr.open_group(str(tmp_path / "plain.zarr"), mode="a")
        with pytest.raises(ValueError, match="tables"):
            check_store_var_conventions(tmp_path / "plain.zarr")


def _store_with_pairs(tmp_path, mz, mobility):
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="a")
    var = root.create_group("tables").create_group("t").create_group("var")
    var.create_array("mz", data=np.asarray(mz, dtype=np.float64))
    var.create_array("mobility", data=np.asarray(mobility, dtype=np.float64))
    return tmp_path / "store.zarr"


class TestMobilityTableVar:
    """A table carrying ``mobility`` is validated on the (mz, mobility) pair."""

    def test_sorted_unique_pairs_pass(self, tmp_path):
        store = _store_with_pairs(tmp_path, [300.0, 300.0, 450.5], [0.95, 1.1, 1.02])
        assert check_store_var_conventions(store) == {"t": []}

    def test_repeated_mz_is_legal_here(self, tmp_path):
        store = _store_with_pairs(tmp_path, [300.0, 300.0], [0.9, 1.0])
        assert check_store_var_conventions(store)["t"] == []

    def test_duplicate_pair_is_an_error(self, tmp_path):
        store = _store_with_pairs(tmp_path, [300.0, 300.0], [0.95, 0.95])
        assert "not unique and sorted" in _messages(
            check_store_var_conventions(store)["t"]
        )

    def test_unsorted_mobility_within_an_mz_is_an_error(self, tmp_path):
        store = _store_with_pairs(tmp_path, [300.0, 300.0], [1.1, 0.95])
        assert "not unique and sorted" in _messages(
            check_store_var_conventions(store)["t"]
        )

    def test_decreasing_mz_is_an_error(self, tmp_path):
        store = _store_with_pairs(tmp_path, [450.5, 300.0], [1.0, 1.0])
        assert "non-decreasing" in _messages(check_store_var_conventions(store)["t"])

    def test_length_mismatch_is_an_error(self, tmp_path):
        store = _store_with_pairs(tmp_path, [300.0, 450.5], [1.0])
        assert "has 1 values for 2" in _messages(
            check_store_var_conventions(store)["t"]
        )

    def test_non_finite_mobility_is_an_error(self, tmp_path):
        store = _store_with_pairs(tmp_path, [300.0, 450.5], [1.0, np.nan])
        assert "non-finite" in _messages(check_store_var_conventions(store)["t"])


def _store_with_precursors(tmp_path, precursor_mz, mz):
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="a")
    var = root.create_group("tables").create_group("t").create_group("var")
    var.create_array("mz", data=np.asarray(mz, dtype=np.float64))
    var.create_array("precursor_mz", data=np.asarray(precursor_mz, dtype=np.float64))
    return tmp_path / "store.zarr"


class TestMsMsTableVar:
    """A table carrying ``precursor_mz`` is validated on the pair.

    ``mz`` restarts at every precursor, so the strict single-column
    contract would reject a perfectly correct demultiplexed table; the
    pair contract is what says the blocks are in order.
    """

    def test_sorted_unique_pairs_pass(self, tmp_path):
        store = _store_with_precursors(
            tmp_path, [313.275, 313.275, 936.578], [80.1, 200.4, 91.2]
        )
        assert check_store_var_conventions(store) == {"t": []}

    def test_mz_restarting_at_the_next_precursor_is_legal(self, tmp_path):
        store = _store_with_precursors(tmp_path, [313.275, 936.578], [500.0, 80.0])
        assert check_store_var_conventions(store)["t"] == []

    def test_duplicate_pair_is_an_error(self, tmp_path):
        store = _store_with_precursors(tmp_path, [313.275, 313.275], [80.1, 80.1])
        assert "not unique and sorted" in _messages(
            check_store_var_conventions(store)["t"]
        )

    def test_unsorted_mz_within_a_precursor_is_an_error(self, tmp_path):
        store = _store_with_precursors(tmp_path, [313.275, 313.275], [200.4, 80.1])
        assert "not unique and sorted" in _messages(
            check_store_var_conventions(store)["t"]
        )

    def test_decreasing_precursor_mz_is_an_error(self, tmp_path):
        store = _store_with_precursors(tmp_path, [936.578, 313.275], [80.0, 80.0])
        assert "non-decreasing" in _messages(check_store_var_conventions(store)["t"])

    def test_length_mismatch_is_an_error(self, tmp_path):
        store = _store_with_precursors(tmp_path, [313.275], [80.1, 200.4])
        assert "has 1 values for 2" in _messages(
            check_store_var_conventions(store)["t"]
        )

    def test_non_finite_precursor_mz_is_an_error(self, tmp_path):
        store = _store_with_precursors(tmp_path, [313.275, np.nan], [80.1, 80.1])
        assert "non-finite" in _messages(check_store_var_conventions(store)["t"])


class TestReservedColumnNames:
    def test_spec_reserves_the_annotation_columns(self):
        from thyra.metadata.schema import (
            MSI_VAR_REQUIRED_COLUMNS,
            MSI_VAR_RESERVED_COLUMNS,
        )

        assert MSI_VAR_REQUIRED_COLUMNS == ("mz",)
        assert set(MSI_VAR_RESERVED_COLUMNS) == {
            "mz",
            "mobility",
            "mz_index",
            "mobility_index",
            "precursor_mz",
            "precursor_index",
            "formula",
            "adduct",
            "annotation_source",
            "fdr",
        }
