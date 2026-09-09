"""What ``convert_msi`` does with arguments nobody meant to pass.

The CLI validates its own options with click, so these gaps were only ever
reachable through the Python API -- which is how Ousia and every notebook
call Thyra. Three shapes were found in the 2026-09-08 sweep (issue #261):

- a no-op argument accepted in silence when misspelled (``streaming``);
- an argument accepted and then used, turning every file into a refusal
  that named the caller's nonsense value back at them
  (``max_mass_axis_length``);
- a removed argument whose refusal reaches the caller as ``False`` rather
  than as an exception (``sparse_format``), which design decision D10
  described the other way round.

All three end the same way at the front door, which is the point: one
``ERROR`` line saying what to do, and ``False``. That is what every other
refusal does since issue #234, and what the CLI turns into exit 1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from thyra.convert import _validate_streaming, convert_msi
from thyra.errors import ConversionRefused
from thyra.readers.imzml.imzml_reader import _validate_max_mass_axis_length


class TestStreaming:
    """``streaming`` selects nothing, and still has to say when it is wrong.

    Every conversion streams since v3.23 (design decision D11), so the
    argument is kept only so existing calls keep running. That is not a
    licence to accept anything: a caller who writes ``streaming="flase"``
    believes they changed something.
    """

    @pytest.mark.parametrize("value", [True, False, "auto"])
    def test_the_three_real_values_are_accepted(self, value):
        assert _validate_streaming(value) is True

    @pytest.mark.parametrize("value", ["yes", "true", "false", None, 1, 2.0, []])
    def test_anything_else_is_refused(self, value, thyra_logs):
        with thyra_logs("thyra.convert", logging.ERROR) as records:
            assert _validate_streaming(value) is False
        said = " ".join(r.getMessage() for r in records)
        assert "streaming must be True, False or 'auto'" in said

    def test_zero_is_refused_rather_than_read_as_false(self, thyra_logs):
        """The sharpest case, and the reason this check exists.

        ``0 == False`` but ``0 is not False``, so ``streaming=0`` -- which a
        caller writes meaning ``False`` -- slipped past the ``is False``
        test that exists precisely to answer that request, and converted in
        silence.
        """
        with thyra_logs("thyra.convert", logging.ERROR) as records:
            assert _validate_streaming(0) is False
        assert "got 0" in " ".join(r.getMessage() for r in records)

    def test_it_is_refused_before_the_source_is_opened(self, tmp_path, thyra_logs):
        """A path that does not exist must not be the complaint.

        The argument check runs in ``_validate_input_parameters``, ahead of
        every path check and every read, so a typo fails while the caller is
        still looking at their own arguments rather than after a metadata
        scan.
        """
        with thyra_logs("thyra.convert", logging.ERROR) as records:
            assert (
                convert_msi(
                    tmp_path / "no-such-file.imzML",
                    tmp_path / "out.zarr",
                    streaming="yes",
                )
                is False
            )
        said = " ".join(r.getMessage() for r in records)
        assert "streaming must be" in said
        assert "does not exist" not in said


class TestMaxMassAxisLength:
    """The raw-axis cap is a count, so it is a positive integer or nothing."""

    @pytest.mark.parametrize("value", [None, 1, 10_000_000])
    def test_a_count_or_no_limit_is_accepted(self, value):
        assert _validate_max_mass_axis_length(value) == value

    @pytest.mark.parametrize("value", [-1, 0, 2.5, "ten", [10]])
    def test_anything_else_is_refused(self, value):
        with pytest.raises(ConversionRefused, match="positive integer or None"):
            _validate_max_mass_axis_length(value)

    @pytest.mark.parametrize("value", [True, False])
    def test_a_bool_is_refused_rather_than_read_as_a_count(self, value):
        """``bool`` is an ``int`` subclass; ``True`` would mean a cap of one."""
        with pytest.raises(ConversionRefused, match="positive integer or None"):
            _validate_max_mass_axis_length(value)

    def test_the_message_does_not_blame_the_file(self):
        """It used to.

        A cap of ``-1`` was accepted and then compared against a growing
        axis, so every file was refused for "exceeding -1 unique m/z
        values" -- a sentence about the data, for a mistake in the call.
        """
        with pytest.raises(ConversionRefused) as excinfo:
            _validate_max_mass_axis_length(-1)
        assert "exceeded" not in str(excinfo.value)
        assert "max_mass_axis_length must be" in str(excinfo.value)


class TestSparseFormatReachesTheCaller:
    """``sparse_format`` was removed in v3.22, and the removal still speaks.

    Design decision D10 said passing the keyword "raises". The converter
    class does raise (pinned in
    ``tests/unit/converters/test_spatialdata_converter.py``), but nothing
    called ``convert_msi`` and looked: its ``except ConversionRefused``
    turns every refusal into one logged ``ERROR`` and ``False``, which is
    the whole point of that handler (issue #234).

    So the front door reports this the same way it reports a Waters raster
    whose stage never moved, and D10's sentence was the thing that was
    wrong, not the code. Pinned here so the two cannot drift apart again:
    what matters is that the removal is *not silent*, and that the message
    is the one the caller needs.
    """

    FIXTURE = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "fixtures"
        / "iontof_sparse.imzML"
    )

    def test_it_is_reported_and_returns_false(self, tmp_path, thyra_logs):
        with thyra_logs("thyra.convert", logging.ERROR) as records:
            result = convert_msi(
                self.FIXTURE,
                tmp_path / "out.zarr",
                dataset_id="pinned",
                sparse_format="csr",
            )

        assert result is False
        said = " ".join(r.getMessage() for r in records)
        assert "sparse_format was removed" in said
        assert "tocsr()" in said, "the message must say what to call instead"

    def test_nothing_is_written(self, tmp_path):
        out = tmp_path / "out.zarr"
        assert (
            convert_msi(self.FIXTURE, out, dataset_id="pinned", sparse_format="csc")
            is False
        )
        assert not out.exists(), "a refused conversion must leave no store"
