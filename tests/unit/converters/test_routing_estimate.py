# tests/unit/converters/test_routing_estimate.py
"""The size the log reports must be the size that gets written.

**What used to be here.** ``_estimate_output_size_gb`` scored a dataset as
``n_pixels * n_mz_bins * 4`` -- the *bounding box* by the bin count, as
though the matrix were dense. The bin count itself had been fixed once
already (issue #87): the raw-axis branch guessed it from ``(max_mass -
min_mass) / 0.01``, a 10 mDa spacing the data need not have, which scored
a continuous file carrying 4,000 points over 250-1200 m/z as though it had
95,000. While that number drove the routing it mis-sent the largest
datasets to the method that held the most in memory.

**Why it is gone.** Fixing the bin count never fixed the *dense*. On
``TIMS-test-data/02_tiny_longramp_1465px`` the line read
``Estimated output size: 40.3 GB (513,339 pixels x 21,072 m/z bins)`` for
a store of 106 MB; 2.5 GB for a 7.6 MB store, 240.7 GB for a 329 MB one
(issue #254). Since the route stopped being chosen by size (design
decision D11) the number selected nothing, so all it could still do was
talk somebody out of a conversion they had room for.

Nothing has to be estimated. The pre-scan counts every entry before the
scatter, and ``_matrix_size_gb`` reports those: eight bytes of value plus
four or eight of row index each, which is an upper bound on what lands on
disk because zarr compresses it. These tests pin that arithmetic, that it
is reported per table rather than per grid position, and that neither the
dense estimate nor the routing machinery came back.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

from thyra.converters.spatialdata.streaming_converter import (
    StreamingSpatialDataConverter,
    _TableUnit,
)


class _Meta:
    def __init__(self, dimensions, mass_range):
        self.dimensions = dimensions
        self.mass_range = mass_range
        self.n_spectra = dimensions[0] * dimensions[1] * dimensions[2]
        self.pixel_size = (20.0, 20.0)
        self.estimated_memory_gb = 1.0
        self.coordinate_bounds = (0, dimensions[0], 0, dimensions[1])
        self.is_3d = dimensions[2] > 1
        self.has_mass_axis = True
        self.source_path = "mock"


class _Reader:
    def __init__(self, dimensions, mass_range):
        self._meta = _Meta(dimensions, mass_range)

    def get_essential_metadata(self):
        return self._meta


def _converter(
    dimensions: Tuple[int, int, int],
    mass_range: Tuple[float, float] = (250.0, 1200.0),
    axis: Optional[np.ndarray] = None,
    resampling_config=None,
    **kwargs,
):
    """Build a converter, optionally with the mass axis already resolved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        conv = StreamingSpatialDataConverter(
            reader=_Reader(dimensions, mass_range),
            output_path=Path(tmpdir) / "out.zarr",
            resampling_config=resampling_config,
            **kwargs,
        )
    conv._common_mass_axis = axis
    return conv


def _unit(n_grid: int, n_cols: int, rows: dict[int, int]) -> _TableUnit:
    """A table whose pre-scan found ``rows[grid]`` entries at each position."""
    unit = _TableUnit("t", "t_pixels", 0, n_grid, n_cols, (1, n_grid))
    for grid, n_entries in rows.items():
        keys = np.arange(n_entries, dtype=np.int64)
        unit.count(grid, keys, np.ones(n_entries))
    unit.finish_counting()
    return unit


class TestTheReportedSize:
    """It is the counted entries, not the bounding box."""

    def test_entries_times_twelve_bytes(self):
        conv = _converter((4, 1, 1), axis=np.linspace(250.0, 1200.0, 50))
        unit = _unit(4, 50, {0: 10, 1: 7, 3: 3})

        assert unit.assembly.n_nonzeros == 20
        expected = 20 * (8 + 4) / 1024**3
        assert conv._matrix_size_gb([unit]) == pytest.approx(expected, rel=1e-12)

    def test_empty_positions_cost_nothing(self):
        """The bounding box is mostly empty on a polygon-shaped acquisition."""
        conv = _converter((1000, 1000, 1), axis=np.linspace(250.0, 1200.0, 21_072))
        unit = _unit(1_000_000, 21_072, {5: 4})

        # Dense arithmetic would have said 1e6 x 21072 x 4 = 78 GB.
        assert conv._matrix_size_gb([unit]) < 1e-6

    def test_several_tables_are_added_up(self):
        conv = _converter((4, 1, 2), axis=np.linspace(250.0, 1200.0, 50))
        units = [_unit(4, 50, {0: 6}), _unit(4, 50, {1: 9})]

        expected = (6 + 9) * (8 + 4) / 1024**3
        assert conv._matrix_size_gb(units) == pytest.approx(expected, rel=1e-12)

    def test_nothing_counted_is_zero(self):
        conv = _converter((4, 1, 1), axis=np.linspace(250.0, 1200.0, 50))
        assert conv._matrix_size_gb([_unit(4, 50, {})]) == 0.0


class TestTheDenseEstimateIsGone:
    """Left behind, somebody would fix its bin count again instead of it."""

    def test_the_estimator_is_gone(self):
        assert not hasattr(StreamingSpatialDataConverter, "_estimate_output_size_gb")

    def test_the_route_machinery_is_gone(self):
        """A leftover predicate or constant would read as a live gate.

        Left behind, the next person tunes it and nothing happens.
        """
        for name in ("PCS_SIZE_THRESHOLD_GB", "_should_use_pcs", "_stream_build_coo"):
            assert not hasattr(StreamingSpatialDataConverter, name), name


class TestUseCscIsCompatibilityOnly:
    """``use_csc`` names a route that no longer has a sibling.

    Ousia's import wizard pins ``use_csc=True`` from when a second route
    existed, so the keyword stays accepted. ``False`` selected the COO
    route, which is gone, and must say so rather than fall through to the
    only route as if it had been chosen.
    """

    @pytest.mark.parametrize("value", ["auto", True])
    def test_auto_and_true_are_accepted(self, value):
        conv = _converter((10, 10, 1), use_csc=value)
        assert conv._matrix_size_gb([]) == 0.0

    def test_false_names_the_removed_route(self):
        with pytest.raises(ValueError, match=r"COO route, which has been removed"):
            _converter((10, 10, 1), use_csc=False)

    def test_sparse_format_is_refused_by_the_base(self):
        """The keyword is gone from every converter, not just this one.

        This route used to accept ``sparse_format="csr"`` and store CSC
        anyway, which nothing reported; a caller who asked for row access
        got the opposite. Now no route takes the keyword, and the base
        refuses it rather than letting ``**kwargs`` eat it in silence.
        """
        with pytest.raises(ValueError, match=r"sparse_format was removed"):
            _converter((10, 10, 1), sparse_format="csr")
