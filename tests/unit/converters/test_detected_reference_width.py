"""A detector's bin width reaches the axis, behind the caller's own setting.

``_reference_params`` is the one place the width and reference m/z are
decided, for both the bin count and the axis generator. Precedence is the
caller's ``--resample-width-at-mz``, then the width the detected instrument
declared (Waters: 2 mDa for the vendor centroid, 1.3 mDa for the MRT
profile), then the per-axis-type default that applied before any detector
could speak.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from thyra.converters.spatialdata.base_spatialdata_converter import (
    BaseSpatialDataConverter,
    _reference_params,
)
from thyra.resampling.types import AxisType


def _stub(width=None, ref=1000.0, detected=None):
    return SimpleNamespace(
        _width_at_mz=width, _reference_mz=ref, _detected_reference_width=detected
    )


class TestPrecedence:
    def test_explicit_width_wins(self):
        assert _reference_params(
            _stub(0.01, 500.0, (0.002, 1000.0)), "reflector_tof"
        ) == (
            0.01,
            500.0,
        )

    def test_detected_width_beats_the_axis_default(self):
        assert _reference_params(_stub(detected=(0.0013, 1000.0)), "linear_tof") == (
            0.0013,
            1000.0,
        )

    @pytest.mark.parametrize(
        "axis_name, expected",
        [
            ("linear_tof", (0.017, 300.0)),
            ("reflector_tof", (0.005, 1000.0)),
            ("constant", (0.005, 1000.0)),
            ("fticr", (0.005, 1000.0)),
        ],
    )
    def test_axis_defaults_when_nothing_was_declared(self, axis_name, expected):
        assert _reference_params(_stub(), axis_name) == expected

    def test_a_stub_without_the_attribute_still_works(self):
        """Older test stubs carry only the two explicit attributes."""
        stub = SimpleNamespace(_width_at_mz=None, _reference_mz=1000.0)
        assert _reference_params(stub, "reflector_tof") == (0.005, 1000.0)


class TestBothConsumersAgree:
    def test_bin_count_and_generator_read_the_same_width(self):
        stub = _stub(detected=(0.002, 1000.0))
        assert BaseSpatialDataConverter._get_reference_params(
            stub, AxisType.REFLECTOR_TOF
        ) == (0.002, 1000.0)
        bins = BaseSpatialDataConverter._calculate_bins_from_width(
            stub, 100.0, 1000.0, AxisType.REFLECTOR_TOF
        )
        # ln(10) * 1000 / 0.002
        assert bins == pytest.approx(1_151_292, rel=1e-3)

    def test_mrt_profile_axis_size_on_the_reference_run(self):
        """1.3 mDa at m/z 1000 over 100-1000 on linear_tof: ~1.05M bins."""
        stub = _stub(detected=(0.0013, 1000.0))
        bins = BaseSpatialDataConverter._calculate_bins_from_width(
            stub, 100.0, 1000.0, AxisType.LINEAR_TOF
        )
        assert bins == pytest.approx(1_051_941, rel=1e-3)
