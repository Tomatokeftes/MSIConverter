"""An option given on a source that ignores it says so (issue #260).

Every vendor and mobility-grid option was accepted on an imzML input and
produced a store byte-identical to the plain run, with no log line.
docs/cli.md documents that the vendor groups are "ignored on other
formats"; the silence was never documented, and ``--tof-law`` on a
non-tof axis and ``--msms-table`` on imzML already warned when ignored.

The grid's own numbers are checked here too: ``--mobility-bins 0`` and
``--mobility-min 2 --mobility-max 1`` reached the grid builder unexamined
even on a source where the options do apply.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import click
import pytest

from thyra.__main__ import (
    GRID_SIZING_FLAGS,
    IGNORED_ELSEWHERE,
    _validate_mobility_grid_params,
    _warn_ignored_flags,
)


class _Ctx:
    """A click context stand-in that answers where a value came from."""

    def __init__(self, *given: str):
        self._given = set(given)

    def get_parameter_source(self, name: str):
        if name in self._given:
            return click.core.ParameterSource.COMMANDLINE
        return click.core.ParameterSource.DEFAULT


@pytest.fixture
def warnings(thyra_logs):
    """What the CLI logged, read off its own logger rather than caplog."""
    with thyra_logs("thyra.cli", logging.WARNING) as records:
        yield SimpleNamespace(records=records)


class TestIgnoredOnAnotherFormat:
    @pytest.mark.parametrize("name", sorted(IGNORED_ELSEWHERE))
    def test_every_listed_option_warns_where_it_does_not_apply(self, name, warnings):
        spelling, formats = IGNORED_ELSEWHERE[name]
        elsewhere = next(f for f in ("imzml", "waters", "phi") if f not in formats)

        _warn_ignored_flags(_Ctx(name), elsewhere, mobility_grid=False)

        messages = [r.getMessage() for r in warnings.records]
        assert any(spelling in m and "ignored" in m for m in messages), messages

    @pytest.mark.parametrize("name", sorted(IGNORED_ELSEWHERE))
    def test_no_option_warns_where_it_does_apply(self, name, warnings):
        _, formats = IGNORED_ELSEWHERE[name]
        _warn_ignored_flags(_Ctx(name), formats[0], mobility_grid=True)
        assert not warnings.records

    def test_an_option_left_at_its_default_is_quiet(self, warnings):
        """Reading click's parameter source, not comparing to the default.

        ``--use-recalibrated`` defaults to on and ``--mobility-bins`` to
        the value it takes, so a value comparison would warn about every
        imzML conversion ever run.
        """
        _warn_ignored_flags(_Ctx(), "imzml", mobility_grid=False)
        assert not warnings.records

    def test_the_message_names_the_source_and_the_format_it_needs(self, warnings):
        _warn_ignored_flags(_Ctx("tdf_spectrum"), "imzml", mobility_grid=False)

        message = warnings.records[0].getMessage()
        assert "--tdf-spectrum" in message
        assert "imzML" in message and "Bruker timsTOF" in message

    def test_no_context_is_not_an_error(self, warnings):
        """The API has no click context, and calls nothing here."""
        _warn_ignored_flags(None, "imzml", mobility_grid=False)
        assert not warnings.records


class TestGridSizingWithoutAGrid:
    """docs/cli.md: "They do nothing without it." Now the log says it too."""

    @pytest.mark.parametrize("name", GRID_SIZING_FLAGS)
    def test_sizing_without_the_grid_warns(self, name, warnings):
        _warn_ignored_flags(_Ctx(name), "bruker", mobility_grid=False)

        messages = [r.getMessage() for r in warnings.records]
        assert any("--mobility-grid" in m for m in messages), messages

    @pytest.mark.parametrize("name", GRID_SIZING_FLAGS)
    def test_sizing_with_the_grid_is_quiet(self, name, warnings):
        _warn_ignored_flags(_Ctx(name), "bruker", mobility_grid=True)
        assert not warnings.records

    def test_it_is_not_said_twice_on_a_format_that_ignores_them(self, warnings):
        """On imzML the option is already reported as ignored outright."""
        _warn_ignored_flags(_Ctx("mobility_bins"), "imzml", mobility_grid=False)
        assert len(warnings.records) == 1


class TestGridNumbers:
    def test_zero_channels_is_refused(self):
        with pytest.raises(click.BadParameter, match="at least one channel"):
            _validate_mobility_grid_params(0, None, None)

    def test_negative_channels_are_refused(self):
        with pytest.raises(click.BadParameter):
            _validate_mobility_grid_params(-4, None, None)

    def test_an_inverted_range_is_refused(self):
        with pytest.raises(click.BadParameter, match="below"):
            _validate_mobility_grid_params(256, 2.0, 1.0)

    def test_an_empty_range_is_refused(self):
        with pytest.raises(click.BadParameter):
            _validate_mobility_grid_params(256, 1.0, 1.0)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_a_non_finite_edge_is_refused(self, value):
        with pytest.raises(click.BadParameter, match="finite"):
            _validate_mobility_grid_params(256, value, None)

    def test_the_edges_may_be_zero_or_negative(self):
        """They are positions on an axis, not quantities."""
        _validate_mobility_grid_params(256, -1.0, 0.0)

    def test_the_defaults_pass(self):
        _validate_mobility_grid_params(256, None, None)
