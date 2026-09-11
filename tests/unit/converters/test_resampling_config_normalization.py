# tests/unit/converters/test_resampling_config_normalization.py
"""Tests for normalising a ``resampling_config`` dict into ResamplingConfig.

The CLI is protected by ``click.Choice``, so bad method and axis-type
values can only arrive through the Python API. They used to be dropped
silently, which produced a conversion that ignored the caller's request
without reporting anything.
"""

from __future__ import annotations

import pytest

from thyra.converters.spatialdata.base_spatialdata_converter import (
    _normalize_resampling_config,
)
from thyra.errors import ConversionRefused
from thyra.resampling.types import (
    DEFAULT_REFERENCE_MZ,
    AxisType,
    ResamplingConfig,
    ResamplingMethod,
)


class TestRecognisedValues:
    """Accepted spellings must resolve to the matching enum member."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("nearest_neighbor", ResamplingMethod.NEAREST_NEIGHBOR),
            ("tic_preserving", ResamplingMethod.TIC_PRESERVING),
        ],
    )
    def test_method_strings(self, name, expected):
        assert _normalize_resampling_config({"method": name}).method is expected

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("constant", AxisType.CONSTANT),
            ("linear_tof", AxisType.LINEAR_TOF),
            ("reflector_tof", AxisType.REFLECTOR_TOF),
            ("tof", AxisType.TOF),
            ("orbitrap", AxisType.ORBITRAP),
            ("fticr", AxisType.FTICR),
        ],
    )
    def test_axis_type_strings(self, name, expected):
        assert _normalize_resampling_config({"axis_type": name}).axis_type is expected

    def test_enum_members_pass_through(self):
        config = _normalize_resampling_config(
            {
                "method": ResamplingMethod.TIC_PRESERVING,
                "axis_type": AxisType.FTICR,
            }
        )

        assert config.method is ResamplingMethod.TIC_PRESERVING
        assert config.axis_type is AxisType.FTICR

    @pytest.mark.parametrize("sentinel", ["auto", "", None])
    def test_auto_sentinels_mean_autodetect(self, sentinel):
        config = _normalize_resampling_config(
            {"method": sentinel, "axis_type": sentinel}
        )

        assert config.method is None
        assert config.axis_type is None

    def test_missing_keys_mean_autodetect(self):
        config = _normalize_resampling_config({})

        assert config.method is None
        assert config.axis_type is None

    def test_dataclass_is_returned_unchanged(self):
        original = ResamplingConfig(method=ResamplingMethod.NEAREST_NEIGHBOR)

        assert _normalize_resampling_config(original) is original


class TestUnrecognisedValues:
    """Unknown values must raise, not fall back to auto-detection.

    All seven assertions below were tightened from ``ValueError`` to
    ``ConversionRefused`` alongside the docstring correction on
    ``_normalize_resampling_config``, and **all seven pass before and
    after**: ``ConversionRefused`` subclasses ``ValueError``, and the code
    already raised the subclass -- only the ``Raises`` block named the
    parent. They pin the refusal type the PR #264 convention makes the
    contract, so a later widening back to a bare ``ValueError`` cannot pass
    unnoticed; none of them is a regression guard.
    """

    @pytest.mark.parametrize(
        "bad",
        [
            "tic_preserving ",  # stray trailing space
            " tic_preserving",
            "TIC_PRESERVING",  # wrong case
            "tic-preserving",  # wrong separator
            "nearest",  # truncated
            "linear_interpolation",  # advertised once, never implemented
            "none",
            "typo",
        ],
    )
    def test_unknown_method_raises(self, bad):
        with pytest.raises(ConversionRefused, match="method"):
            _normalize_resampling_config({"method": bad})

    @pytest.mark.parametrize(
        "bad",
        [
            "fticr ",
            "FTICR",
            "ft-icr",
            "TOF",
            "unknown",  # enum member exists but has no axis generator
            "typo",
        ],
    )
    def test_unknown_axis_type_raises(self, bad):
        with pytest.raises(ConversionRefused, match="axis_type"):
            _normalize_resampling_config({"axis_type": bad})

    def test_error_names_the_valid_values(self):
        with pytest.raises(ConversionRefused) as excinfo:
            _normalize_resampling_config({"method": "typo"})

        message = str(excinfo.value)
        assert "'auto'" in message
        assert "'nearest_neighbor'" in message
        assert "'tic_preserving'" in message

    def test_error_names_the_offending_value(self):
        with pytest.raises(ConversionRefused, match="tic_preserving "):
            _normalize_resampling_config({"method": "tic_preserving "})

    def test_unimplemented_method_enum_raises(self):
        """ResamplingMethod.NONE has no strategy behind it.

        Passing it used to reach the dense path and silently perform
        TIC-preserving resampling. Callers who want no resampling pass
        ``resampling_config=None`` instead.
        """
        with pytest.raises(ConversionRefused, match="method"):
            _normalize_resampling_config({"method": ResamplingMethod.NONE})

    def test_wrong_type_raises(self):
        with pytest.raises(ConversionRefused, match="method"):
            _normalize_resampling_config({"method": 3})

    def test_axis_type_enum_without_a_generator_raises(self):
        with pytest.raises(ConversionRefused, match="axis_type"):
            _normalize_resampling_config({"axis_type": AxisType.UNKNOWN})


class TestReferenceMz:
    """The dict path and the dataclass must agree on the default."""

    def test_default_matches_the_dataclass_default(self):
        assert _normalize_resampling_config({}).reference_mz == (
            ResamplingConfig().reference_mz
        )

    def test_default_is_the_documented_cli_default(self):
        assert ResamplingConfig().reference_mz == 1000.0
        assert DEFAULT_REFERENCE_MZ == 1000.0

    def test_explicit_value_wins(self):
        assert (
            _normalize_resampling_config({"reference_mz": 250.0}).reference_mz == 250.0
        )

    def test_explicit_value_is_coerced_to_float(self):
        config = _normalize_resampling_config({"reference_mz": 500})

        assert isinstance(config.reference_mz, float)
        assert config.reference_mz == 500.0


class TestOtherFields:
    """The remaining keys are passed through by name."""

    def test_width_at_mz_maps_to_mass_width_da(self):
        config = _normalize_resampling_config({"width_at_mz": 0.01})

        assert config.mass_width_da == 0.01

    def test_range_and_bins_pass_through(self):
        config = _normalize_resampling_config(
            {"target_bins": 4096, "min_mz": 300.0, "max_mz": 900.0}
        )

        assert config.target_bins == 4096
        assert config.min_mz == 300.0
        assert config.max_mz == 900.0
