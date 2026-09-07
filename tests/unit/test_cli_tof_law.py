"""``--mass-axis-type tof``, ``--tof-a``, ``--tof-b`` and ``--bins-per-fwhm``.

The CLI only transports the two-term width law; what is pinned here is the
transport and the validation that keeps a half-specified law from reaching
the converter.
"""

from __future__ import annotations

import click
import pytest

from thyra.__main__ import _build_resampling_config, _validate_tof_params


class TestValidation:
    def test_a_and_b_come_together(self):
        with pytest.raises(click.BadParameter, match="together"):
            _validate_tof_params(0.0185, None, None, None)
        with pytest.raises(click.BadParameter, match="together"):
            _validate_tof_params(None, 9.1e-6, None, None)

    @pytest.mark.parametrize("a, b", [(-1.0, 1e-6), (0.01, -1e-6), (0.0, 0.0)])
    def test_the_law_must_have_a_width(self, a, b):
        with pytest.raises(click.BadParameter):
            _validate_tof_params(a, b, None, None)

    def test_one_limit_alone_is_fine(self):
        _validate_tof_params(0.0, 1e-6, None, None)
        _validate_tof_params(0.02, 0.0, None, None)

    def test_bins_per_fwhm_and_width_are_exclusive(self):
        with pytest.raises(click.BadParameter, match="mutually exclusive"):
            _validate_tof_params(None, None, 3.0, 0.002)

    def test_bins_per_fwhm_must_be_positive(self):
        with pytest.raises(click.BadParameter):
            _validate_tof_params(None, None, 0.0, None)

    def test_nothing_given_is_fine(self):
        _validate_tof_params(None, None, None, None)


class TestTransport:
    def test_defaults_are_none(self):
        cfg = _build_resampling_config("auto", "auto", None, None, None, None, 1000.0)
        assert cfg["tof_a"] is None and cfg["tof_b"] is None
        assert cfg["bins_per_fwhm"] is None

    def test_values_are_forwarded(self):
        cfg = _build_resampling_config(
            "auto", "tof", None, None, None, None, 1000.0, None, 0.0185, 9.1e-6, 4.0
        )
        assert cfg["axis_type"] == "tof"
        assert cfg["tof_a"] == 0.0185
        assert cfg["tof_b"] == 9.1e-6
        assert cfg["bins_per_fwhm"] == 4.0
