"""``--mass-axis-type tof`` and ``--tof-law A B``.

One flag carries the two-term width law; the bin width is not a second
flag, because ``--resample-width-at-mz`` at the reference m/z already says
how wide a bin is on every axis type. What is pinned here is the transport
into the config dict and the validation that keeps a law with no width out.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from thyra.__main__ import _build_resampling_config, _validate_tof_law, main


class TestValidation:
    @pytest.mark.parametrize("a, b", [(-1.0, 1e-6), (0.01, -1e-6), (0.0, 0.0)])
    def test_the_law_must_have_a_width(self, a, b):
        with pytest.raises(click.BadParameter):
            _validate_tof_law((a, b))

    def test_one_limit_alone_is_fine(self):
        _validate_tof_law((0.0, 1e-6))
        _validate_tof_law((0.02, 0.0))

    def test_nothing_given_is_fine(self):
        _validate_tof_law(None)


class TestTransport:
    def test_defaults_are_none(self):
        cfg = _build_resampling_config("auto", "auto", None, None, None, None, 1000.0)
        assert cfg["tof_a"] is None and cfg["tof_b"] is None
        assert "bins_per_fwhm" not in cfg

    def test_the_pair_is_split_into_the_api_fields(self):
        cfg = _build_resampling_config(
            "auto", "tof", None, None, None, None, 1000.0, None, (0.0185, 9.1e-6)
        )
        assert cfg["axis_type"] == "tof"
        assert cfg["tof_a"] == 0.0185
        assert cfg["tof_b"] == 9.1e-6

    def test_the_flag_takes_two_numbers(self, tmp_path):
        """Click rejects a lone coefficient before any conversion starts."""
        result = CliRunner().invoke(
            main,
            [
                str(tmp_path / "in.imzML"),
                str(tmp_path / "out.zarr"),
                "--tof-law",
                "0.0185",
            ],
        )
        assert result.exit_code != 0
        assert (
            "tof-law" in result.output.lower() or "2 arguments" in result.output.lower()
        )
