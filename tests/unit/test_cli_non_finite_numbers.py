"""Non-finite numeric options are refused before any file is opened.

``value <= 0`` is False for NaN and for +infinity, so every guard written
that way admitted both: ``--pixel-size nan`` reached the store as the
dataset's pixel size, and ``--tof-law nan 1`` was refused only by the axis
generator, as a traceback, after a 47 s metadata scan (issue #231).
"""

from __future__ import annotations

import click
import pytest

from thyra.__main__ import (
    _validate_basic_params,
    _validate_positive_float,
    _validate_tof_law,
)
from thyra.convert import _validate_numeric_parameters

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


class TestPixelSize:
    @pytest.mark.parametrize("value", NON_FINITE)
    def test_a_non_finite_pixel_size_is_refused(self, value):
        with pytest.raises(click.BadParameter, match="finite"):
            _validate_basic_params(value, "dataset")

    def test_a_real_pixel_size_is_still_accepted(self):
        _validate_basic_params(25.0, "dataset")

    def test_no_pixel_size_is_still_accepted(self):
        _validate_basic_params(None, "dataset")

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_a_non_positive_pixel_size_is_still_refused(self, value):
        with pytest.raises(click.BadParameter):
            _validate_basic_params(value, "dataset")


class TestPositiveFloats:
    """--resample-min-mz, --resample-max-mz, --resample-width-at-mz, --z-spacing."""

    @pytest.mark.parametrize("value", NON_FINITE)
    def test_a_non_finite_value_is_refused(self, value):
        with pytest.raises(click.BadParameter, match="finite"):
            _validate_positive_float(value, "z_spacing", "Z spacing")

    def test_a_real_value_is_still_accepted(self):
        _validate_positive_float(10.0, "z_spacing", "Z spacing")
        _validate_positive_float(None, "z_spacing", "Z spacing")


class TestTofLaw:
    @pytest.mark.parametrize(
        "law",
        [
            (float("nan"), 1e-6),
            (0.02, float("nan")),
            (float("inf"), 0.0),
            (0.0, float("inf")),
        ],
    )
    def test_a_non_finite_law_is_refused(self, law):
        with pytest.raises(click.BadParameter, match="finite"):
            _validate_tof_law(law)

    def test_a_real_law_is_still_accepted(self):
        _validate_tof_law((0.0185, 9.1e-6))

    def test_the_width_check_still_applies(self):
        with pytest.raises(click.BadParameter):
            _validate_tof_law((0.0, 0.0))


class TestTheApiValidatesToo:
    """convert_msi can be called directly, without the CLI's guards."""

    @pytest.mark.parametrize("value", NON_FINITE)
    def test_a_non_finite_pixel_size_is_refused(self, value):
        assert _validate_numeric_parameters(value) is False

    @pytest.mark.parametrize("value", NON_FINITE)
    def test_a_non_finite_z_spacing_is_refused(self, value):
        assert _validate_numeric_parameters(25.0, value) is False

    def test_real_numbers_pass(self):
        assert _validate_numeric_parameters(25.0, 10.0) is True
        assert _validate_numeric_parameters(None, None) is True
