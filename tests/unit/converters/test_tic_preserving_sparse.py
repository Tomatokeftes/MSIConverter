"""The sparse form of ``tic_preserving`` is the dense form, bin for bin.

``_tic_preserving_resample`` used to interpolate onto every bin of the axis
and hand a dense array downstream, where it was scanned for non-zeros. On a
zero-suppressed profile source -- a Waters MRT pixel stores ~15,000 samples
in clusters around its peaks and the default axis has 1.05M bins -- that
cost 570 s for a conversion nearest-neighbour binning finished in 16 s. The
sparse form evaluates only the axis points the interpolant can be non-zero
at. These tests pin that it is the same operator: same values, same TIC,
same handling of gaps, cropping, unsorted input and degenerate spectra.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from thyra.converters.spatialdata.base_spatialdata_converter import (
    BaseSpatialDataConverter,
    _tic_preserving_sparse,
    _tic_support_bins,
)


def _sqrt_axis(lo, hi, width_at_1000):
    """A linear_tof axis: uniform in sqrt(m/z), like the digitiser grid."""
    k = width_at_1000 / np.sqrt(1000.0)
    n = int((2.0 / k) * (np.sqrt(hi) - np.sqrt(lo)))
    edges = np.linspace(np.sqrt(lo), np.sqrt(hi), n + 1) ** 2
    return (edges[:-1] + edges[1:]) / 2


def _zero_suppressed_profile(seed=0, n_peaks=40, spacing_at_1000=1.14e-3):
    """Clusters of samples on a sqrt-spaced grid, an explicit zero at each edge.

    This is the shape MassLynx returns for a profile pixel: only samples
    around peaks are stored, the sample positions lie on one global grid,
    and every cluster starts and ends with a stored zero.
    """
    rng = np.random.default_rng(seed)
    grid = _sqrt_axis(300.0, 1000.0, spacing_at_1000)
    centres = np.sort(rng.uniform(320.0, 980.0, n_peaks))
    mzs, its = [], []
    for c in centres:
        sigma = 1.9e-3 * np.sqrt(c / 1000.0)  # ~4 samples per FWHM
        i = np.searchsorted(grid, c)
        lo, hi = max(i - 12, 0), min(i + 13, grid.size)
        m = grid[lo:hi]
        y = rng.uniform(50, 5000) * np.exp(-((m - c) ** 2) / (2 * sigma**2))
        y[y < 20.0] = 0.0
        if y[0] != 0.0 or y[-1] != 0.0:
            continue
        # Trim to one zero on either side, as MassLynx stores it.
        nz = np.flatnonzero(y)
        m, y = m[nz[0] - 1 : nz[-1] + 2], y[nz[0] - 1 : nz[-1] + 2]
        if mzs and m[0] <= mzs[-1][-1]:
            continue
        mzs.append(m)
        its.append(y)
    return np.concatenate(mzs), np.concatenate(its)


AXIS = _sqrt_axis(300.0, 1000.0, 1.3e-3)


def _dense(axis, mzs, its, tol=None):
    stub = SimpleNamespace(_common_mass_axis=axis, _gap_tolerance_da=tol)
    return BaseSpatialDataConverter._tic_preserving_resample(stub, mzs, its)


def _sparse(axis, mzs, its, tol=None):
    stub = SimpleNamespace(_common_mass_axis=axis, _gap_tolerance_da=tol)
    return BaseSpatialDataConverter._tic_preserving_resample_sparse(stub, mzs, its)


def _scatter(axis, idx, vals):
    out = np.zeros(axis.size)
    out[idx] = vals
    return out


class TestSupportBins:
    def test_open_at_the_zero_neighbours_closed_at_the_ends(self):
        axis = np.arange(0.0, 10.5, 0.5)
        mzs = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0])
        its = np.array([5.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0])
        idx = _tic_support_bins(axis, mzs, its)
        # First run [1, 2] starts at the spectrum's own first sample, so
        # axis == 1.0 is included; it ends before the zero at 3.0.
        # Run [7] sits between zeros at 6.0 and 8.0, open at both.
        # Run [9] ends at the spectrum's last sample: axis == 9.0 included.
        expected = np.flatnonzero(
            ((axis >= 1.0) & (axis < 3.0))
            | ((axis > 6.0) & (axis < 8.0))
            | ((axis > 8.0) & (axis <= 9.0))
        )
        np.testing.assert_array_equal(idx, expected)

    def test_no_zeros_means_the_whole_span(self):
        axis = np.linspace(0.0, 10.0, 41)
        mzs = np.array([2.0, 3.0, 5.0])
        idx = _tic_support_bins(axis, mzs, np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(
            idx, np.flatnonzero((axis >= 2.0) & (axis <= 5.0))
        )

    def test_all_zero_is_empty(self):
        idx = _tic_support_bins(np.linspace(0, 1, 5), np.array([0.2, 0.4]), np.zeros(2))
        assert idx.size == 0

    def test_indices_are_sorted_and_unique(self):
        mzs, its = _zero_suppressed_profile()
        idx = _tic_support_bins(AXIS, mzs, its)
        assert np.all(np.diff(idx) > 0)


class TestAgreesWithTheDenseForm:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_zero_suppressed_profile(self, seed):
        mzs, its = _zero_suppressed_profile(seed)
        dense = _dense(AXIS, mzs, its)
        idx, vals = _sparse(AXIS, mzs, its)
        np.testing.assert_allclose(_scatter(AXIS, idx, vals), dense, rtol=1e-12, atol=0)
        assert np.all(vals != 0)
        assert vals.sum() == pytest.approx(its.sum(), rel=1e-12)

    def test_only_a_small_fraction_of_the_axis_is_touched(self):
        """The point of the exercise: work proportional to the samples."""
        mzs, its = _zero_suppressed_profile()
        idx, _ = _sparse(AXIS, mzs, its)
        assert idx.size < 2 * np.count_nonzero(its)
        assert idx.size < 0.05 * AXIS.size

    def test_centroid_like_input_without_zeros(self):
        rng = np.random.default_rng(3)
        mzs = np.sort(rng.uniform(300.0, 1000.0, 150))
        its = rng.uniform(10.0, 1000.0, 150)
        dense = _dense(AXIS, mzs, its)
        idx, vals = _sparse(AXIS, mzs, its)
        np.testing.assert_allclose(_scatter(AXIS, idx, vals), dense, rtol=1e-12, atol=0)

    def test_unsorted_input(self):
        mzs, its = _zero_suppressed_profile(4)
        order = np.random.default_rng(4).permutation(mzs.size)
        dense = _dense(AXIS, mzs, its)
        idx, vals = _sparse(AXIS, mzs[order], its[order])
        np.testing.assert_allclose(_scatter(AXIS, idx, vals), dense, rtol=1e-12, atol=0)

    def test_gap_tolerance_is_applied_the_same_way(self):
        rng = np.random.default_rng(5)
        mzs = np.sort(rng.uniform(300.0, 1000.0, 60))
        its = rng.uniform(10.0, 1000.0, 60)
        dense = _dense(AXIS, mzs, its, tol=0.05)
        idx, vals = _sparse(AXIS, mzs, its, tol=0.05)
        np.testing.assert_allclose(_scatter(AXIS, idx, vals), dense, rtol=1e-12, atol=0)
        assert vals.sum() == pytest.approx(its.sum(), rel=1e-12)

    def test_cropped_axis_keeps_the_same_share(self):
        mzs, its = _zero_suppressed_profile(6)
        axis = _sqrt_axis(500.0, 800.0, 1.3e-3)
        dense = _dense(axis, mzs, its)
        idx, vals = _sparse(axis, mzs, its)
        np.testing.assert_allclose(_scatter(axis, idx, vals), dense, rtol=1e-12, atol=0)
        assert 0 < vals.sum() < its.sum()

    def test_single_point(self):
        idx, vals = _sparse(AXIS, np.array([650.0]), np.array([42.0]))
        assert idx.size == 1 and vals[0] == 42.0
        assert AXIS[idx[0]] == pytest.approx(650.0, abs=1e-3)
        idx, vals = _sparse(AXIS, np.array([1200.0]), np.array([42.0]))
        assert idx.size == 0

    def test_empty_and_all_zero(self):
        idx, vals = _sparse(AXIS, np.array([]), np.array([]))
        assert idx.size == 0 and vals.size == 0
        idx, vals = _sparse(AXIS, np.array([400.0, 401.0]), np.zeros(2))
        assert idx.size == 0
        assert _dense(AXIS, np.array([400.0, 401.0]), np.zeros(2)).sum() == 0.0

    def test_module_function_is_what_both_methods_call(self):
        mzs, its = _zero_suppressed_profile(7)
        idx, vals = _tic_preserving_sparse(AXIS, mzs, its, None)
        idx2, vals2 = _sparse(AXIS, mzs, its)
        np.testing.assert_array_equal(idx, idx2)
        np.testing.assert_array_equal(vals, vals2)


class TestOnTheDefaultMRTAxis:
    """Bins at 1.14x the sample spacing: nothing doubled, nothing skipped."""

    def test_no_holes_inside_clusters(self):
        mzs, its = _zero_suppressed_profile(8)
        idx, vals = _sparse(AXIS, mzs, its)
        stored = np.zeros(AXIS.size, dtype=bool)
        stored[idx] = True
        nz = its != 0
        for k in np.flatnonzero(nz[1:-1]) + 1:
            inside = (AXIS > mzs[k - 1]) & (AXIS < mzs[k + 1])
            assert stored[inside].all()

    def test_no_bin_exceeds_its_neighbouring_samples(self):
        """Interpolation cannot put two samples' worth into one bin."""
        mzs, its = _zero_suppressed_profile(9)
        idx, vals = _sparse(AXIS, mzs, its)
        scale = vals.sum() / np.interp(AXIS[idx], mzs, its).sum()
        right = np.searchsorted(mzs, AXIS[idx])
        left = np.clip(right - 1, 0, mzs.size - 1)
        right = np.clip(right, 0, mzs.size - 1)
        ceiling = np.maximum(its[left], its[right]) * scale
        assert np.all(vals <= ceiling * (1 + 1e-12))
