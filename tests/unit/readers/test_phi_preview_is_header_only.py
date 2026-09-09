# tests/unit/readers/test_phi_preview_is_header_only.py
"""``preview_msi`` on a PHI ``.raw`` decodes no ion events.

``preview_msi`` promises "No spectra are decoded" and passes
``metadata_only=True`` for that purpose. ``PhiReader.__init__`` swallowed
the kwarg through ``**kwargs``, and ``PhiMetadataExtractor`` called
``get_peak_counts_per_pixel()``, which aggregates every 8-byte event in
the stream to learn which pixels carry one. Measured at 0.16 s for a 16 MB
file with 2.02 M events and linear in file size, so a multi-gigabyte
SmartSoft acquisition previewed as slowly as it converted (issue #240).

The assertion here is that the event stream is never touched, not that the
preview was quick: a wall-clock threshold on a synthetic file measures the
machine, not the fix. ``iter_event_batches`` is the single door into the
stream -- :meth:`PhiReader._aggregate` is its only caller -- so counting
calls to it settles the question exactly.

What the preview reports instead is the second half of the fix. The header
knows the tile geometry but not which pixels fired, and PHI is the only
format that cannot answer that cheaply (imzML counts its coordinate list,
Bruker runs a SQL count). Rather than quietly reporting the raster size in
a column that means "spectra present" everywhere else, ``n_pixels`` comes
back as ``None``.
"""

from __future__ import annotations

import numpy as np
import pytest

from thyra.preview import preview_msi
from thyra.readers.phi import PhiReader

from .test_phi_reader import block, event, events_block, write_raw


@pytest.fixture
def phi_raw(tmp_path):
    """A 4x4 raster in which only two of the sixteen pixels fired."""
    blocks = (
        events_block([event(0, 0, 3_000_000), event(0, 0, 3_000_000)])
        + events_block([event(1, 2, 5_000_000)])
        + block(2)
    )
    return write_raw(tmp_path / "preview.raw", blocks)


@pytest.fixture
def event_reads(monkeypatch):
    """Count every entry into the ion-event stream."""
    from thyra.readers.phi import phi_reader as phi_reader_module

    calls: list[str] = []
    original = phi_reader_module.iter_event_batches

    def counting(path, index):
        calls.append(str(path))
        return original(path, index)

    monkeypatch.setattr(phi_reader_module, "iter_event_batches", counting)
    return calls


class TestThePreviewNeverReachesTheEventStream:
    def test_preview_decodes_no_events(self, phi_raw, event_reads):
        """The whole point: a preview reads the header, not the file."""
        preview = preview_msi(phi_raw)

        assert preview.readable, preview.error
        assert event_reads == [], "preview aggregated the event stream"

    def test_a_real_read_still_decodes_them(self, phi_raw, event_reads):
        """Guard the guard: the counter must be able to see an aggregate.

        Without this, a fixture that silently stopped intercepting
        ``iter_event_batches`` would make the test above pass for the
        wrong reason.
        """
        reader = PhiReader(phi_raw)
        try:
            counts = reader.get_peak_counts_per_pixel()
        finally:
            reader.close()

        assert event_reads, "the counter never saw the aggregate"
        assert counts is not None
        assert int(np.count_nonzero(counts)) == 2


class TestWhatThePreviewReportsWithoutTheAggregate:
    def test_n_pixels_is_unknown_rather_than_the_raster_size(self, phi_raw):
        """``None`` means "not counted"; it must not be filled in with 16.

        The raster extent is already reported, in ``grid_dims``. Putting it
        in ``n_pixels`` as well would make that field mean "positions the
        raster covers" for PHI and "spectra present" for every other
        format, which is the one outcome worse than not answering.
        """
        preview = preview_msi(phi_raw)

        assert preview.n_pixels is None
        assert preview.grid_dims == (4, 4)

    def test_the_header_derived_fields_are_still_answered(self, phi_raw):
        """Everything the header knows still comes back."""
        preview = preview_msi(phi_raw)

        assert preview.readable
        assert preview.mz_range[0] < preview.mz_range[1]
        assert preview.pixel_size_um == pytest.approx(2.0)
        assert preview.error is None

    def test_the_resampling_verdict_survives(self, phi_raw):
        """The detector matches on the format flag, not on peak density.

        ``PhiToFSIMSDetector`` exists because interpolating PHI's sparse
        pixels fabricates signal (#168), so the preview losing its
        nearest-neighbour verdict would be a regression of that fix -- and
        the detector reads ``total_peaks``/``n_spectra`` off the same
        metadata this change zeroes.
        """
        from thyra.resampling.types import ResamplingMethod

        preview = preview_msi(phi_raw)

        assert preview.resampling_method is ResamplingMethod.NEAREST_NEIGHBOR


class TestAFullReadIsUnchanged:
    """A reader built for conversion still counts everything it used to."""

    def test_counts_are_exact_without_metadata_only(self, phi_raw):
        reader = PhiReader(phi_raw)
        try:
            essential = reader.get_essential_metadata()
        finally:
            reader.close()

        assert essential.n_spectra_counted is True
        assert essential.n_spectra == 2
        # Two events in one channel of pixel (0, 0), one in pixel (1, 2):
        # two occupied channels in total.
        assert essential.total_peaks == 2
        assert essential.peak_counts_per_pixel is not None

    def test_metadata_only_says_the_counts_are_absent(self, phi_raw):
        """0 with the flag down, never 0 pretending to be a measurement."""
        reader = PhiReader(phi_raw, metadata_only=True)
        try:
            essential = reader.get_essential_metadata()
        finally:
            reader.close()

        assert essential.n_spectra_counted is False
        assert essential.n_spectra == 0
        assert essential.total_peaks == 0
        assert essential.peak_counts_per_pixel is None
        # Header-derived, so still exact.
        assert essential.dimensions == (4, 4, 1)
        assert essential.coordinate_bounds == (0.0, 3.0, 0.0, 3.0)
