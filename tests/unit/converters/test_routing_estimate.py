# tests/unit/converters/test_routing_estimate.py
"""The size estimate must be honest, and routing must not depend on it.

``_estimate_output_size_gb`` scores a dataset as
``n_pixels * n_mz_bins * 4``. The resampling branch already resolved the
real bin count (issue #87), but the raw-axis branch still guessed it from
``(max_mass - min_mass) / 0.01`` -- a 10 mDa spacing the data need not have.

That guess is wrong in both directions. A continuous file carrying 4,000
points over 250-1200 m/z was scored as though it had 95,000, inflating it
24x; a processed file whose spectra share no m/z values was scored far too
low, which routed the very largest datasets to the method that holds the
most in memory.

**The routing half of that is now history.** PCS was faster and lighter at
every size measured, so ``"auto"`` first picked it unconditionally and
``PCS_SIZE_THRESHOLD_GB`` went; then the COO route itself went, and with it
the predicate. The estimate survives as a log line, and these tests survive
with it -- an inaccurate number in a support log is a smaller problem than
an inaccurate route, but it is still a problem, and the 24x inflation is
the kind of thing that gets re-derived if nobody wrote down that the
fallback is unreliable.

``convert()`` runs ``_initialize_conversion()`` before reaching the
estimate, so the axis is already built and there is nothing to guess. These
tests pin that the built axis is preferred, that the old heuristics still
apply when it is not available -- which is the case when the estimator is
called directly, as the older tests in ``test_streaming_converter.py`` do --
and that ``use_csc`` survives only as a compatibility keyword.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

from thyra.converters.spatialdata.streaming_converter import (
    StreamingSpatialDataConverter,
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


class TestPrefersTheBuiltAxis:
    """When the axis exists, its length is what counts."""

    def test_uses_real_axis_length(self):
        n_pixels = 10_000
        axis = np.linspace(250.0, 1200.0, 4_000)
        conv = _converter((100, 100, 1), axis=axis)

        expected = n_pixels * 4_000 * 4 / (1024**3)
        assert conv._estimate_output_size_gb() == pytest.approx(expected, rel=1e-9)

    def test_real_axis_beats_the_raw_heuristic(self):
        """The heuristic would say 95,000 bins; the axis says 4,000."""
        axis = np.linspace(250.0, 1200.0, 4_000)
        conv = _converter((100, 100, 1), axis=axis)
        with_axis = conv._estimate_output_size_gb()

        conv._common_mass_axis = None
        without_axis = conv._estimate_output_size_gb()

        # (1200 - 250) / 0.01 = 95,000, i.e. 23.75x the real count.
        assert without_axis == pytest.approx(with_axis * 95_000 / 4_000, rel=1e-6)

    def test_real_axis_beats_the_resampling_plan_too(self):
        """A built axis is authoritative even when resampling is configured."""
        axis = np.linspace(250.0, 1200.0, 1_000)
        conv = _converter(
            (100, 100, 1),
            axis=axis,
            resampling_config={"method": "nearest_neighbor", "target_bins": 500_000},
        )
        expected = 10_000 * 1_000 * 4 / (1024**3)
        assert conv._estimate_output_size_gb() == pytest.approx(expected, rel=1e-9)


class TestFallbacksStillApply:
    """Without a built axis, the previous behaviour is unchanged."""

    def test_raw_heuristic_when_no_axis_and_no_resampling(self):
        conv = _converter((100, 100, 1), mass_range=(250.0, 1200.0), axis=None)
        expected = 10_000 * 95_000 * 4 / (1024**3)
        assert conv._estimate_output_size_gb() == pytest.approx(expected, rel=1e-9)

    def test_resampling_plan_when_no_axis(self):
        conv = _converter(
            (50, 50, 1),
            mass_range=(100.0, 1000.0),
            axis=None,
            resampling_config={"method": "nearest_neighbor", "target_bins": 5_000},
        )
        expected = 2_500 * 5_000 * 4 / (1024**3)
        assert conv._estimate_output_size_gb() == pytest.approx(expected, rel=1e-9)


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
        assert conv._estimate_output_size_gb() >= 0.0

    def test_false_names_the_removed_route(self):
        with pytest.raises(ValueError, match=r"COO route, which has been removed"):
            _converter((10, 10, 1), use_csc=False)

    def test_csr_is_refused_rather_than_stored_as_csc(self):
        """The route has no CSR layout.

        It used to accept ``sparse_format="csr"`` and store CSC anyway,
        which nothing reported; a caller who asked for row access got the
        opposite. The refusal names the converter that does write CSR.
        """
        with pytest.raises(ValueError, match=r"CSC only.*streaming=False"):
            _converter((10, 10, 1), sparse_format="csr")

    def test_the_route_machinery_is_gone(self):
        """A leftover predicate or constant would read as a live gate.

        Left behind, the next person tunes it and nothing happens.
        """
        for name in ("PCS_SIZE_THRESHOLD_GB", "_should_use_pcs", "_stream_build_coo"):
            assert not hasattr(StreamingSpatialDataConverter, name), name


class TestDegenerate:
    """Empty and tiny axes must not raise."""

    def test_empty_axis_is_zero_sized(self):
        conv = _converter((10, 10, 1), axis=np.array([]))
        assert conv._estimate_output_size_gb() == 0.0

    def test_single_bin_axis(self):
        conv = _converter((10, 10, 1), axis=np.array([500.0]))
        expected = 100 * 1 * 4 / (1024**3)
        assert conv._estimate_output_size_gb() == pytest.approx(expected, rel=1e-9)
