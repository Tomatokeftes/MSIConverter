"""What a store does with values that are not measurements (issue #248).

Nothing on the read path tested finiteness. Ten NaN intensities reached
``X.data``, turned ``uns["average_spectrum"]`` into NaN and passed
``thyra validate``; a source with NaN m/z converted at exit 0 and was
then refused by that same validator, so the converter wrote a store its
own validator rejects.

Negative intensities needed a decision rather than a guard: they were
stored under ``nearest_neighbor`` (giving the pixel a TIC of 0) while
``tic_preserving`` found a non-positive total and zeroed the whole
spectrum, so the two methods disagreed in kind and neither said so.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from thyra.converters.spatialdata.base_spatialdata_converter import (
    BaseSpatialDataConverter,
)
from thyra.errors import ConversionRefused

_MODULE_LOGGER = "thyra.converters.spatialdata.base_spatialdata_converter"


def _stub():
    """The converter reduced to what the drop reads and writes."""
    stub = SimpleNamespace(
        _unusable_intensities=0,
        _unusable_intensities_warned=False,
    )
    stub._count_unusable_intensities = (
        lambda *args: BaseSpatialDataConverter._count_unusable_intensities(stub, *args)
    )
    return stub


def _drop(stub, mzs, intensities):
    return BaseSpatialDataConverter._drop_unusable_intensities(
        stub, np.asarray(mzs, dtype=np.float64), np.asarray(intensities, np.float64)
    )


class TestTheAxisIsRefusedRatherThanWritten:
    def _refuse(self, axis):
        stub = SimpleNamespace(_common_mass_axis=np.asarray(axis, dtype=np.float64))
        BaseSpatialDataConverter._refuse_non_finite_axis(stub)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_axis_value_is_refused(self, bad):
        with pytest.raises(ConversionRefused, match="non-finite"):
            self._refuse([100.0, bad, 300.0])

    def test_the_message_names_var_mz_and_the_count(self):
        with pytest.raises(ConversionRefused) as excinfo:
            self._refuse([np.nan, np.nan, 300.0])
        message = str(excinfo.value)
        assert "2" in message and "var['mz']" in message

    def test_a_real_axis_passes(self):
        self._refuse([100.0, 200.0, 300.0])

    def test_an_empty_axis_is_left_to_the_check_that_owns_it(self):
        self._refuse([])


class TestNonFiniteIntensities:
    def test_they_are_dropped_with_their_m_z(self):
        mzs, intensities = _drop(_stub(), [1.0, 2.0, 3.0], [10.0, np.nan, 30.0])
        np.testing.assert_array_equal(mzs, [1.0, 3.0])
        np.testing.assert_array_equal(intensities, [10.0, 30.0])

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_every_kind_goes(self, bad):
        _, intensities = _drop(_stub(), [1.0, 2.0], [bad, 5.0])
        np.testing.assert_array_equal(intensities, [5.0])

    def test_an_ordinary_spectrum_is_returned_untouched(self):
        """Same objects, so the shared-axis identity fast path survives."""
        stub = _stub()
        mzs = np.asarray([1.0, 2.0])
        intensities = np.asarray([10.0, 20.0])
        out_mz, out_int = BaseSpatialDataConverter._drop_unusable_intensities(
            stub, mzs, intensities
        )
        assert out_mz is mzs and out_int is intensities
        assert stub._unusable_intensities == 0


class TestNegativeIntensities:
    def test_they_are_dropped_too(self):
        _, intensities = _drop(_stub(), [1.0, 2.0, 3.0], [10.0, -5.0, 30.0])
        np.testing.assert_array_equal(intensities, [10.0, 30.0])

    def test_zero_is_kept_here(self):
        """Zeros are a separate matter; the sparse write drops them."""
        _, intensities = _drop(_stub(), [1.0, 2.0], [0.0, 3.0])
        np.testing.assert_array_equal(intensities, [0.0, 3.0])

    def test_both_methods_now_see_the_same_spectrum(self):
        """The whole point: neither method decides for itself what a
        negative value means, because neither one ever sees one."""
        mzs, intensities = _drop(_stub(), [1.0, 2.0, 3.0], [-1.0, -2.0, 30.0])
        assert intensities.min() >= 0
        np.testing.assert_array_equal(mzs, [3.0])


class TestTheCount:
    def test_the_first_affected_spectrum_warns_once(self, thyra_logs):
        stub = _stub()
        with thyra_logs(_MODULE_LOGGER, logging.WARNING) as records:
            _drop(stub, [1.0, 2.0, 3.0], [np.nan, -5.0, 30.0])
            _drop(stub, [1.0, 2.0], [np.nan, 30.0])

        warnings = [r.getMessage() for r in records]
        assert len(warnings) == 1
        assert "2 of 3" in warnings[0]
        assert "1 non-finite, 1 negative" in warnings[0]

    def test_the_running_total_accumulates(self):
        stub = _stub()
        _drop(stub, [1.0, 2.0], [np.nan, 30.0])
        _drop(stub, [1.0, 2.0], [-1.0, 30.0])
        assert stub._unusable_intensities == 2


class TestTheSiblingTablesUseTheSameRule:
    """A grid's marginal reproduces the summed column only if it drops the
    same points, so the raw-scan mapper applies the rule too."""

    def _axis(self):
        return np.asarray([100.0, 200.0, 300.0], dtype=np.float64)

    def test_map_points_drops_them(self):
        from thyra.converters.spatialdata.mobility_heatmap import map_points_to_axis

        bins, mobility, intensities, n_dropped = map_points_to_axis(
            self._axis(),
            np.asarray([100.0, 200.0, 300.0]),
            np.asarray([1.0, 1.1, 1.2]),
            np.asarray([5.0, np.nan, -3.0]),
        )
        np.testing.assert_array_equal(bins, [0])
        np.testing.assert_array_equal(mobility, [1.0])
        np.testing.assert_array_equal(intensities, [5.0])
        # Not an out-of-range drop; that count means something else.
        assert n_dropped == 0

    def test_the_indexed_mapper_agrees_point_for_point(self):
        """The two mappers are pinned against each other; so is this rule."""
        from thyra.converters.spatialdata.mobility_heatmap import (
            map_indexed_points_to_axis,
            map_points_to_axis,
        )

        axis = self._axis()
        mzs = np.asarray([100.0, 200.0, 200.0, 300.0, 400.0])
        mobility = np.asarray([1.0, 1.1, 1.2, 1.3, 1.4])
        intensities = np.asarray([5.0, np.nan, 7.0, -3.0, 9.0])
        unique_mz, inverse = np.unique(mzs, return_inverse=True)

        plain = map_points_to_axis(axis, mzs, mobility, intensities)
        indexed = map_indexed_points_to_axis(
            axis, unique_mz, inverse, mobility, intensities
        )

        for a, b in zip(plain[:3], indexed[:3]):
            np.testing.assert_array_equal(a, b)
        assert plain[3] == indexed[3] == 1  # the m/z 400 point, out of range

    def test_an_ordinary_pixel_is_unchanged(self):
        from thyra.converters.spatialdata.mobility_heatmap import usable_intensities

        assert usable_intensities(np.asarray([1.0, 0.0, 3.0])) is None
