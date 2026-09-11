# tests/unit/converters/test_tic_gate_uses_resolved_axis.py
"""The TIC-preserving gate governs the axis the conversion actually builds.

``_gate_tic_preserving`` permits ``TIC_PRESERVING`` only when the source
grid law and the target axis are the same law, because that is the one
condition under which Thyra's operator -- interpolate, then rescale by one
global factor -- is exact.

Until #286 the gate read that target axis from ``detector.get_axis_type()``,
the detector's own preference. The converter resolves the axis separately
and later, and ``--mass-axis-type`` overrides it there. The two detectors
that reach ``TIC_PRESERVING`` at all -- ``RapiflexDetector`` (CONSTANT) and
``WatersProfileDetector`` (LINEAR_TOF) -- each declare a source law equal to
their own axis, so the early gate always cleared and the conversion then
interpolated onto whatever axis had been asked for.

``docs/resampling.md`` measures the cost: across 300-1100 m/z, two ions of
equal abundance come back with their ratio distorted by 1.9x on
``linear_tof``, 3.7x on ``reflector_tof``, 7.0x on ``orbitrap`` and 13.4x on
``fticr``. That page said you had to ask for such a pairing with two explicit
flags. One was enough: ``--mass-axis-type fticr`` alone, with
``--resample-method`` left at its ``auto`` default.

Nothing downstream could see it. Every stored intensity in the main table is
affected, and the per-pixel TIC still balances exactly -- preserving it is
what the operator does -- so neither the TIC identity nor the two-pass
agreement check registers anything.
"""

from __future__ import annotations

import pytest

from thyra.converters.spatialdata.base_spatialdata_converter import (
    BaseSpatialDataConverter,
)
from thyra.resampling.decision_tree import ResamplingDecisionTree
from thyra.resampling.types import AxisType, ResamplingConfig, ResamplingMethod

#: Matched by ``RapiflexDetector`` on the format flag alone. It is the
#: cheapest source that reaches ``TIC_PRESERVING``: method TIC_PRESERVING,
#: axis CONSTANT, source grid law CONSTANT.
_RAPIFLEX = {"format_specific": {"format": "Rapiflex"}}


class _EssentialMetadata:
    """Only the field ``_resolve_resampling_plan`` reads."""

    def __init__(self, mass_range=(300.0, 1100.0)):
        self.mass_range = mass_range


class _Converter:
    """Just enough converter to run the two methods under test.

    ``_setup_resampling`` picks the method; ``_resolve_resampling_plan``
    settles the axis. The bug lives in the gap between them, so the test has
    to run both in order rather than either alone.
    """

    def __init__(self, metadata, *, axis_type=None, method=None):
        self._resampling_config = ResamplingConfig(
            method=method, axis_type=axis_type, mass_width_da=0.1
        )
        self._metadata = metadata
        self._resampling_method = None
        self._resampling_metadata_cached = metadata
        self._essential_metadata_cached = _EssentialMetadata()

    def _get_reader_metadata_for_resampling(self):
        return self._metadata

    _get_cached_metadata_for_resampling = (
        BaseSpatialDataConverter._get_cached_metadata_for_resampling
    )
    _setup_resampling = BaseSpatialDataConverter._setup_resampling
    _warn_if_override_contradicts_detector = (
        BaseSpatialDataConverter._warn_if_override_contradicts_detector
    )
    _resolve_resampling_plan = BaseSpatialDataConverter._resolve_resampling_plan
    _regate_tic_preserving = BaseSpatialDataConverter._regate_tic_preserving
    _get_reference_params = BaseSpatialDataConverter._get_reference_params
    _calculate_bins_from_width = BaseSpatialDataConverter._calculate_bins_from_width

    def plan(self):
        self._setup_resampling()
        return self._resolve_resampling_plan()


class TestTheGateAtTheDecisionTree:
    """``select_strategy_for_axis`` asks the gate about a named axis."""

    def test_keeps_tic_preserving_when_the_axis_is_the_source_law(self):
        tree = ResamplingDecisionTree()
        assert (
            tree.select_strategy_for_axis(_RAPIFLEX, AxisType.CONSTANT)
            is ResamplingMethod.TIC_PRESERVING
        )

    @pytest.mark.parametrize(
        "axis_type",
        [AxisType.FTICR, AxisType.ORBITRAP, AxisType.LINEAR_TOF, AxisType.TOF],
    )
    def test_downgrades_on_every_axis_that_is_not_the_source_law(self, axis_type):
        tree = ResamplingDecisionTree()
        assert (
            tree.select_strategy_for_axis(_RAPIFLEX, axis_type)
            is ResamplingMethod.NEAREST_NEIGHBOR
        )

    def test_agrees_with_select_strategy_when_the_axis_is_not_overridden(self):
        """The no-override case has to be a no-op, or auto conversions change."""
        tree = ResamplingDecisionTree()
        detected_axis = tree.select_axis_type(_RAPIFLEX)
        assert tree.select_strategy_for_axis(
            _RAPIFLEX, detected_axis
        ) is tree.select_strategy(_RAPIFLEX)


class TestTheGateThroughTheConverter:
    """The regression: one flag used to be enough."""

    def test_axis_override_downgrades_the_auto_selected_method(self):
        """#286. ``--mass-axis-type fticr`` alone, method left on auto."""
        converter = _Converter(_RAPIFLEX, axis_type=AxisType.FTICR)
        _, _, axis_type, _ = converter.plan()

        assert axis_type is AxisType.FTICR, "the override still decides the axis"
        assert converter._resampling_method is ResamplingMethod.NEAREST_NEIGHBOR

    def test_without_an_override_tic_preserving_survives(self):
        """The control. Rapiflex on its own axis is the exact case."""
        converter = _Converter(_RAPIFLEX)
        _, _, axis_type, _ = converter.plan()

        assert axis_type is AxisType.CONSTANT
        assert converter._resampling_method is ResamplingMethod.TIC_PRESERVING

    def test_an_explicit_method_is_still_honoured(self):
        """D15: an explicit method is the caller's decision, warned not moved.

        The gate governs auto-selection only. Re-gating an explicit
        ``tic_preserving`` here would silently overrule the caller, which is
        the behaviour #246 deliberately rejected.
        """
        converter = _Converter(
            _RAPIFLEX,
            axis_type=AxisType.FTICR,
            method=ResamplingMethod.TIC_PRESERVING,
        )
        _, _, axis_type, _ = converter.plan()

        assert axis_type is AxisType.FTICR
        assert converter._resampling_method is ResamplingMethod.TIC_PRESERVING

    def test_a_matching_explicit_axis_is_not_disturbed(self):
        """Naming the axis the detector would have picked changes nothing."""
        converter = _Converter(_RAPIFLEX, axis_type=AxisType.CONSTANT)
        _, _, axis_type, _ = converter.plan()

        assert axis_type is AxisType.CONSTANT
        assert converter._resampling_method is ResamplingMethod.TIC_PRESERVING
