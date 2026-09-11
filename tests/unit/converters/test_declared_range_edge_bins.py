# tests/unit/converters/test_declared_range_edge_bins.py
"""A peak exactly on a declared mass-range bound lands in the edge bin.

Every physics generator lays ``target_bins + 1`` bin *edges* across the
requested ``[min_mz, max_mz]`` and returns the midpoints, so the first and
last axis point sit half a bin inside the range that was asked for: a
source declaring 50-1000 built the axis ``[50.0001, 999.9975]``. The keep
rule tested membership against those axis points, so a peak sitting
exactly on a declared bound was always discarded (issue #239).

It costs most on the sources that declare their range *as* their first and
last sample. PHI TOF-SIMS sets ``mass_range`` from the first and last
detector channel, so both were dropped in every pixel of every resampled
variant of the mock fixture -- 12 counts stored against 14 in the source,
where ``--no-resample`` stored all 14.

The fix must not become the clamp it replaced
---------------------------------------------

``_nearest_neighbor_resample`` used to clip every out-of-range index into
the axis and accumulate, so narrowing the range piled the whole discarded
part of the spectrum onto two bins: on real ``pea.imzML`` resampled to
400-800 m/z, bin 0 held 654,158 counts where a real peak there is around
80. The total was conserved exactly, so no TIC check could see it.

The new rule is bounded to the **declared** range -- at most half a bin
beyond the first and last centre -- which is why
:class:`TestBinZeroIsStillNotADumpingGround` sits alongside the
edge-bin tests rather than instead of them.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata
import numpy as np
import pytest

from thyra.converters.spatialdata.base_spatialdata_converter import (
    BaseSpatialDataConverter,
    _kept_mz_range,
)
from thyra.converters.spatialdata.streaming_converter import (
    StreamingSpatialDataConverter,
)
from thyra.readers.imzml import ImzMLReader
from thyra.resampling.common_axis import CommonAxisBuilder
from thyra.resampling.types import AxisType

MIN_MZ = 100.0
MAX_MZ = 1000.0
N_BINS = 5_000


def _physics_axis() -> np.ndarray:
    """A linear-TOF axis over ``[MIN_MZ, MAX_MZ]``, as the converter builds it."""
    return (
        CommonAxisBuilder()
        .build_physics_axis(
            min_mz=MIN_MZ,
            max_mz=MAX_MZ,
            num_bins=N_BINS,
            axis_type=AxisType.LINEAR_TOF,
        )
        .mz_values.astype(np.float64)
    )


def _stub(axis, axis_range):
    """The converter's resampling surface, without building a converter.

    The same ``SimpleNamespace`` pattern the other resampling tests use,
    with the reporting helper bound on so the count and the one-shot
    warning are the real ones.
    """
    stub = SimpleNamespace(
        _common_mass_axis=np.asarray(axis, float),
        _axis_range=axis_range,
        _gap_tolerance_da=None,
        _out_of_range_peaks=0,
        _out_of_range_warned=False,
    )
    stub._count_out_of_range = (
        lambda n_dropped, n_total: BaseSpatialDataConverter._count_out_of_range(
            stub, n_dropped, n_total
        )
    )
    return stub


def _nearest_neighbor(stub, mzs, intensities):
    return BaseSpatialDataConverter._nearest_neighbor_resample(
        stub, np.asarray(mzs, float), np.asarray(intensities, float)
    )


def _tic_preserving(stub, mzs, intensities):
    return BaseSpatialDataConverter._tic_preserving_resample(
        stub, np.asarray(mzs, float), np.asarray(intensities, float)
    )


class TestTheAxisReallyStopsShortOfTheDeclaredRange:
    """Guard the guard: without this gap there is nothing to fix."""

    def test_the_first_and_last_centre_sit_inside_the_range(self):
        axis = _physics_axis()

        assert axis[0] > MIN_MZ
        assert axis[-1] < MAX_MZ

    def test_the_gap_is_no_more_than_half_a_bin(self):
        """The bound the fix is allowed to reach, and no further."""
        axis = _physics_axis()

        assert axis[0] - MIN_MZ < (axis[1] - axis[0])
        assert MAX_MZ - axis[-1] < (axis[-1] - axis[-2])


class TestAPeakOnTheBoundSurvives:
    """Both methods, because a rule only one follows is a disagreement."""

    def test_nearest_neighbor_puts_the_bounds_in_the_edge_bins(self):
        axis = _physics_axis()
        stub = _stub(axis, (MIN_MZ, MAX_MZ))

        indices, values = _nearest_neighbor(stub, [MIN_MZ, MAX_MZ], [7.0, 11.0])

        assert sorted(indices.tolist()) == [0, len(axis) - 1]
        assert values.sum() == pytest.approx(18.0)
        assert stub._out_of_range_peaks == 0
        assert not stub._out_of_range_warned

    def test_nearest_neighbor_dropped_them_before(self):
        """The same call with no declared range is the old behaviour."""
        axis = _physics_axis()
        stub = _stub(axis, None)

        indices, _ = _nearest_neighbor(stub, [MIN_MZ, MAX_MZ], [7.0, 11.0])

        assert indices.size == 0
        assert stub._out_of_range_peaks == 2

    def test_tic_preserving_keeps_the_bound_peaks_too(self):
        axis = _physics_axis()
        stub = _stub(axis, (MIN_MZ, MAX_MZ))

        resampled = _tic_preserving(stub, [MIN_MZ, 500.0, MAX_MZ], [7.0, 5.0, 11.0])

        assert resampled.sum() == pytest.approx(23.0)
        assert resampled[0] > 0.0
        assert resampled[-1] > 0.0

    def test_the_two_methods_preserve_the_same_total(self):
        """#248's rule: they must agree on what the axis covers."""
        axis = _physics_axis()
        mzs = [MIN_MZ, 250.0, 700.0, MAX_MZ]
        intensities = [7.0, 3.0, 5.0, 11.0]

        _, nn_values = _nearest_neighbor(
            _stub(axis, (MIN_MZ, MAX_MZ)), mzs, intensities
        )
        tic_values = _tic_preserving(_stub(axis, (MIN_MZ, MAX_MZ)), mzs, intensities)

        assert nn_values.sum() == pytest.approx(sum(intensities))
        assert tic_values.sum() == pytest.approx(sum(intensities))

    def test_the_shared_axis_cache_agrees_with_the_generic_path(self):
        """The fast path for a shared m/z array carries its own keep rule.

        It slices the in-range subset with ``searchsorted`` rather than a
        mask, so the two rules are written twice and can disagree once.
        """
        axis = _physics_axis()
        mzs = np.array([MIN_MZ, 300.0, MAX_MZ])
        intensities = np.array([7.0, 3.0, 11.0])

        cached = _stub(axis, (MIN_MZ, MAX_MZ))
        cached._nn_shared_cache = None
        cached._nn_cache_misses = 0
        cached._build_nn_shared_cache = (
            lambda a, m: BaseSpatialDataConverter._build_nn_shared_cache(cached, a, m)
        )
        cached._nn_resample_via_cache = (
            lambda a, m, i: BaseSpatialDataConverter._nn_resample_via_cache(
                cached, a, m, i
            )
        )

        # First call builds the cache, second one hits it.
        first = _nearest_neighbor(cached, mzs, intensities)
        second = _nearest_neighbor(cached, mzs, intensities)
        generic = _nearest_neighbor(_stub(axis, (MIN_MZ, MAX_MZ)), mzs, intensities)

        for got in (first, second):
            assert got[0].tolist() == generic[0].tolist()
            assert got[1] == pytest.approx(generic[1])


class TestBinZeroIsStillNotADumpingGround:
    """The regression the current rule exists to prevent."""

    def test_a_peak_below_the_declared_minimum_is_dropped(self):
        axis = _physics_axis()
        stub = _stub(axis, (MIN_MZ, MAX_MZ))

        indices, values = _nearest_neighbor(
            stub, [50.0, 500.0, 5_000.0], [654_158.0, 80.0, 654_158.0]
        )

        assert values.sum() == pytest.approx(80.0)
        assert 0 not in indices.tolist()
        assert stub._out_of_range_peaks == 2

    def test_the_slack_stops_at_the_declared_bound(self):
        """Half a bin outside the *centre*, not half a bin outside the range."""
        axis = _physics_axis()
        stub = _stub(axis, (MIN_MZ, MAX_MZ))

        indices, _ = _nearest_neighbor(stub, [MIN_MZ - 1e-6, MAX_MZ + 1e-6], [3.0, 4.0])

        assert indices.size == 0

    def test_a_spectrum_wholly_outside_the_range_stores_nothing(self):
        """Both methods, and the one assertion they can share here.

        The clamp only ever existed on the nearest-neighbour path, so
        "bin 0 does not collect the floor" is a claim about that method
        alone. ``tic_preserving`` interpolates *between* source points by
        construction, so a peak below the range still raises the bins
        between it and the next measured point -- that is the method
        working as documented, and the hazard
        ``--resample-gap-tolerance`` exists for (issue #246), not this
        one. What both must agree on is a spectrum the range does not
        reach at all.
        """
        axis = _physics_axis()

        indices, values = _nearest_neighbor(
            _stub(axis, (MIN_MZ, MAX_MZ)), [10.0, 20.0], [654_158.0, 9.0]
        )
        resampled = _tic_preserving(
            _stub(axis, (MIN_MZ, MAX_MZ)), [10.0, 20.0], [654_158.0, 9.0]
        )

        assert indices.size == 0
        assert values.size == 0
        assert resampled.sum() == 0.0


class TestAUniformAxisIsUnchanged:
    """``np.linspace`` end points *are* the declared bounds, so nothing moves."""

    def test_the_kept_range_is_the_axis_span(self):
        axis = np.linspace(100.0, 110.0, 11)

        assert _kept_mz_range(axis, (100.0, 110.0)) == (100.0, 110.0)
        assert _kept_mz_range(axis, None) == (100.0, 110.0)

    def test_a_peak_just_outside_is_still_out(self):
        axis = np.linspace(100.0, 110.0, 11)
        stub = _stub(axis, (100.0, 110.0))

        indices, _ = _nearest_neighbor(stub, [100.0 - 1e-9, 110.0 + 1e-9], [3.0, 4.0])

        assert indices.size == 0


class TestTheKeptRangeNeverNarrowsTheAxis:
    def test_a_range_inside_the_axis_is_widened_to_it(self):
        """A generator that overshot must not cost a peak with a bin waiting."""
        axis = np.linspace(100.0, 110.0, 11)

        assert _kept_mz_range(axis, (101.0, 109.0)) == (100.0, 110.0)


class TestEndToEndOnAPhysicsAxis:
    """A conversion whose only peaks sit on the declared bounds.

    Before #239 this stored nothing at all, and batch 4's empty-store
    guard turned that into a refusal: the whole spectrum was "outside the
    target mass axis" although every peak was inside the range asked for.
    """

    @staticmethod
    def _convert(imzml_path: Path, output_path: Path, mzs) -> bool:
        reader = ImzMLReader(imzml_path)
        try:
            return StreamingSpatialDataConverter(
                reader=reader,
                output_path=output_path,
                dataset_id="edge",
                pixel_size_um=10.0,
                resampling_config={
                    "method": "nearest_neighbor",
                    "axis_type": "linear_tof",
                    "target_bins": 2_000,
                    # The two m/z the fixture actually puts intensity on.
                    "min_mz": float(mzs[10]),
                    "max_mz": float(mzs[30]),
                },
            ).convert()
        finally:
            reader.close()

    def test_the_bound_peaks_reach_the_store(self, create_minimal_imzml, temp_dir):
        imzml_path, _, mzs, all_intensities = create_minimal_imzml
        output_path = temp_dir / "edges.zarr"

        assert self._convert(imzml_path, output_path, mzs) is True

        adata = anndata.read_zarr(output_path / "tables" / "edge_z0")
        stored_total = float(np.asarray(adata.X.sum()))
        expected = sum(
            float(intensities[10] + intensities[30]) for intensities in all_intensities
        )

        assert adata.n_obs == 4
        assert stored_total == pytest.approx(expected)

    def test_both_bounds_land_in_the_edge_bins(self, create_minimal_imzml, temp_dir):
        """Not merely kept: kept in the bin the declared range gives them."""
        imzml_path, _, mzs, _ = create_minimal_imzml
        output_path = temp_dir / "edge_bins.zarr"

        assert self._convert(imzml_path, output_path, mzs) is True

        adata = anndata.read_zarr(output_path / "tables" / "edge_z0")
        occupied = sorted(
            int(c) for c in np.unique(adata.X.tocoo().col)  # type: ignore[union-attr]
        )

        assert occupied == [0, adata.n_vars - 1]
