# tests/unit/readers/test_imzml_zero_based.py
"""A 0-based imzML keeps its first row and column.

The imzML specification numbers pixels from 1, and three sites subtracted
a constant 1 to reach the 0-based indices everything downstream uses: the
reader's cached coordinate array, its per-spectrum fallback, and the
extractor's grid sizing. Files written 0-based exist, and on one of those
the constant produced ``x = -1`` and ``y = -1`` for the first row and
column. A negative index is a legal *negative* numpy index, so the
converter's ``_locate`` guard dropped those spectra:

    preview_msi: n_pixels 9, grid (2, 2)
    WARNING - 5 spectra sat outside the declared 2x2x1 grid and were skipped
    store: 4 rows, x in {0, 1}, y in {0, 1}

Exit 0, with the warning blaming a grid the file never declared (#244).

The rule that replaced the constant is **not** the one z uses. z rebases on
its observed minimum because z has no origin; x and y do, so they fold a 0
down and otherwise keep the specification's base of 1. The
three fixtures below are the whole argument: the 0-based file must gain
its lost row and column, the ordinary 1-based file must not move, and the
*cropped* 1-based file -- the one an observed-minimum rebase would slide
to the origin -- must not move either.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata
import numpy as np
import pytest

from thyra.converters.spatialdata.streaming_converter import (
    SPATIALDATA_AVAILABLE,
    StreamingSpatialDataConverter,
)
from thyra.readers.imzml import ImzMLReader

pytestmark = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)

_TABLE_KEY = "imzml_z0"
_OUT_OF_GRID_LOGGER = "thyra.converters.spatialdata.streaming_converter"


def _convert(imzml_path: Path, output_path: Path) -> bool:
    reader = ImzMLReader(imzml_path)
    try:
        return StreamingSpatialDataConverter(
            reader=reader,
            output_path=output_path,
            dataset_id="imzml",
            pixel_size_um=10.0,
        ).convert()
    finally:
        reader.close()


def _stored(output_path: Path) -> dict:
    """Stored intensity per ``(x, y)``, keyed by the obs coordinates."""
    adata = anndata.read_zarr(output_path / "tables" / _TABLE_KEY)
    row_sums = np.asarray(adata.X.sum(axis=1)).ravel()
    xs = np.asarray(adata.obs["x"]).astype(int)
    ys = np.asarray(adata.obs["y"]).astype(int)
    return {(int(x), int(y)): float(v) for x, y, v in zip(xs, ys, row_sums)}


class TestTheReaderRebasesOnWhatTheFileHolds:
    def test_a_zero_based_file_yields_every_pixel_at_or_above_the_origin(
        self, zero_based_imzml
    ):
        """No coordinate is negative, and all nine are distinct."""
        path, _, _ = zero_based_imzml
        reader = ImzMLReader(path)
        try:
            coords = [c for c, _mzs, _its in reader.iter_spectra()]
        finally:
            reader.close()

        assert len(coords) == 9
        assert len(set(coords)) == 9
        assert min(x for x, _y, _z in coords) == 0
        assert min(y for _x, y, _z in coords) == 0
        assert max(x for x, _y, _z in coords) == 2
        assert max(y for _x, y, _z in coords) == 2

    def test_a_one_based_file_is_unchanged(self, one_based_imzml):
        """The ordinary case, and the whole risk of the rebase decision."""
        path, _, _ = one_based_imzml
        reader = ImzMLReader(path)
        try:
            coords = [c for c, _mzs, _its in reader.iter_spectra()]
        finally:
            reader.close()

        assert min(x for x, _y, _z in coords) == 0
        assert max(x for x, _y, _z in coords) == 2

    def test_a_cropped_acquisition_keeps_its_offset(self, cropped_one_based_imzml):
        """x = 5 stays at 4, it does not slide to 0.

        This is what separates the x/y rule from z's. Rebasing on the
        observed minimum would report this three-pixel-wide crop at the
        origin, changing its grid width and every stored coordinate.
        """
        path, _, _ = cropped_one_based_imzml
        reader = ImzMLReader(path)
        try:
            coords = [c for c, _mzs, _its in reader.iter_spectra()]
        finally:
            reader.close()

        assert min(x for x, _y, _z in coords) == 4
        assert max(x for x, _y, _z in coords) == 6

    def test_the_fallback_path_agrees_with_the_cached_array(self, zero_based_imzml):
        """``_get_spectrum_coordinates`` has two routes; both are rebased.

        The cached numpy array is the fast one. The per-spectrum fallback
        recomputes, and carried its own copy of the constant.
        """
        path, _, _ = zero_based_imzml
        reader = ImzMLReader(path)
        try:
            reader._ensure_parser_initialized()
            cached = [
                reader._get_spectrum_coordinates(reader.parser, i) for i in range(9)
            ]
            reader._coordinates_array = None
            fallback = [
                reader._get_spectrum_coordinates(reader.parser, i) for i in range(9)
            ]
        finally:
            reader.close()

        assert cached == fallback


class TestTheGridIsSizedFromTheSameBase:
    def test_a_zero_based_file_declares_its_full_grid(self, zero_based_imzml):
        """The 2x2 in the warning came from sizing a 0..2 file by its max."""
        path, _, _ = zero_based_imzml
        reader = ImzMLReader(path)
        try:
            essential = reader.get_essential_metadata()
        finally:
            reader.close()

        assert essential.dimensions == (3, 3, 1)
        assert essential.n_spectra == 9

    def test_a_one_based_file_declares_the_same_grid(self, one_based_imzml):
        path, _, _ = one_based_imzml
        reader = ImzMLReader(path)
        try:
            essential = reader.get_essential_metadata()
        finally:
            reader.close()

        assert essential.dimensions == (3, 3, 1)

    def test_a_cropped_file_is_sized_from_the_spec_base(self, cropped_one_based_imzml):
        """Seven columns wide, of which the first four are empty.

        Not three: the acquisition sits at x = 5..7 on a slide whose origin
        is x = 1, and that offset is real.
        """
        path, _, _ = cropped_one_based_imzml
        reader = ImzMLReader(path)
        try:
            essential = reader.get_essential_metadata()
        finally:
            reader.close()

        assert essential.dimensions == (7, 7, 1)

    def test_what_was_subtracted_is_recorded(self, zero_based_imzml, one_based_imzml):
        """A shift nothing records is a shift nobody can undo.

        The imzML extractor never set ``coordinate_offsets`` before #244,
        so the store could not say where its origin came from. The
        converter writes this to
        ``coordinate_systems.global.coordinate_offsets_px``.
        """
        offsets = {}
        for name, fixture in (("zero", zero_based_imzml), ("one", one_based_imzml)):
            path, _, _ = fixture
            reader = ImzMLReader(path)
            try:
                offsets[name] = reader.get_essential_metadata().coordinate_offsets
            finally:
                reader.close()

        assert offsets["zero"][:2] == (0, 0)
        assert offsets["one"][:2] == (1, 1)


class TestTheStoreKeepsEveryPixel:
    def test_a_zero_based_file_stores_all_nine_rows(self, zero_based_imzml, temp_dir):
        """Four rows before the fix, and no error to say so."""
        path, _, expected = zero_based_imzml
        output_path = temp_dir / "zero.zarr"

        assert _convert(path, output_path) is True

        stored = _stored(output_path)
        assert len(stored) == 9
        assert set(stored) == set(expected)
        for position, value in expected.items():
            assert stored[position] == pytest.approx(value), position

    def test_nothing_is_reported_as_off_grid(
        self, zero_based_imzml, temp_dir, thyra_logs
    ):
        """The warning blamed a grid the file never declared."""
        path, _, _ = zero_based_imzml
        output_path = temp_dir / "quiet.zarr"

        with thyra_logs(_OUT_OF_GRID_LOGGER, logging.WARNING) as records:
            assert _convert(path, output_path) is True

        off_grid = [
            r.getMessage() for r in records if "outside the declared" in r.getMessage()
        ]
        assert off_grid == []

    def test_a_one_based_file_stores_the_same_thing(self, one_based_imzml, temp_dir):
        """The two files describe one acquisition; the stores must match."""
        path, _, expected = one_based_imzml
        output_path = temp_dir / "one.zarr"

        assert _convert(path, output_path) is True

        stored = _stored(output_path)
        assert len(stored) == 9
        for position, value in expected.items():
            assert stored[position] == pytest.approx(value), position
