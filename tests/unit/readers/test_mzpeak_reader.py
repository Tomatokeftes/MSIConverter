"""Reader-level tests for the experimental mzPeak reader.

Every archive here is built by :mod:`tests.fixtures.mzpeak_builder`, which
spells the container out literally rather than asking the reader how to write
it -- see that module's docstring for why that separation matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.mzpeak_builder import Spectrum, build_mzpeak, grid_spectra
from thyra.readers.mzpeak import MzPeakReader


@pytest.fixture
def simple_archive(tmp_path):
    """A 3x2 acquisition, 1-based positions, eight points per spectrum."""
    return build_mzpeak(tmp_path / "simple.mzpeak", grid_spectra(3, 2))


class TestIteration:
    """Coordinates, payloads and ordering coming out of iter_spectra."""

    def test_yields_every_spectrum_in_index_order(self, simple_archive):
        """All spectra arrive, ascending, with 0-based coordinates."""
        with MzPeakReader(simple_archive) as reader:
            emitted = list(reader.iter_spectra())

        assert len(emitted) == 6
        assert [coords for coords, _, _ in emitted] == [
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (2, 1, 0),
        ]

    def test_payload_matches_what_was_written(self, tmp_path):
        """m/z and intensity survive the round trip, as float64."""
        spectra = [Spectrum(1, 1, [100.0, 200.5, 300.25], [7.0, 8.0, 9.0])]
        archive = build_mzpeak(tmp_path / "one.mzpeak", spectra)

        with MzPeakReader(archive) as reader:
            ((_, mzs, intensities),) = list(reader.iter_spectra())

        np.testing.assert_array_equal(mzs, [100.0, 200.5, 300.25])
        np.testing.assert_allclose(intensities, [7.0, 8.0, 9.0])
        assert mzs.dtype == np.float64
        assert intensities.dtype == np.float64

    def test_spectrum_straddling_row_groups_is_emitted_once(self, tmp_path):
        """A spectrum split across row groups is reassembled, not duplicated.

        Row groups cut at a fixed row count with no regard for spectrum
        boundaries, so with eight points per spectrum and three rows per group
        every spectrum spans several groups.
        """
        archive = build_mzpeak(
            tmp_path / "straddle.mzpeak",
            grid_spectra(2, 2, n_points=8),
            row_group_size=3,
        )

        with MzPeakReader(archive) as reader:
            emitted = list(reader.iter_spectra())

        assert len(emitted) == 4
        assert [coords for coords, _, _ in emitted] == [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
        ]
        for _, mzs, intensities in emitted:
            assert mzs.size == 8
            assert intensities.size == 8
            # Ascending within the spectrum proves the pieces were joined in
            # order rather than concatenated arbitrarily.
            assert np.all(np.diff(mzs) > 0)

    def test_row_group_size_does_not_change_the_result(self, tmp_path):
        """Reading is invariant to how the writer chose its row groups."""
        spectra = grid_spectra(3, 2, n_points=7)
        whole = build_mzpeak(tmp_path / "whole.mzpeak", spectra)
        split = build_mzpeak(tmp_path / "split.mzpeak", spectra, row_group_size=2)

        with MzPeakReader(whole) as reader:
            reference = list(reader.iter_spectra())
        with MzPeakReader(split) as reader:
            candidate = list(reader.iter_spectra())

        assert len(reference) == len(candidate)
        for (coords_a, mz_a, int_a), (coords_b, mz_b, int_b) in zip(
            reference, candidate
        ):
            assert coords_a == coords_b
            np.testing.assert_array_equal(mz_a, mz_b)
            np.testing.assert_array_equal(int_a, int_b)

    def test_zero_based_positions_are_normalised(self, tmp_path):
        """A 0-based file lands on the same grid as a 1-based one.

        Normalisation subtracts the observed minimum rather than a constant,
        because the format promises nothing about the origin.
        """
        archive = build_mzpeak(tmp_path / "zero.mzpeak", grid_spectra(2, 2, base=0))

        with MzPeakReader(archive) as reader:
            coords = [c for c, _, _ in reader.iter_spectra()]

        assert coords == [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]

    def test_offset_positions_are_normalised(self, tmp_path):
        """A file whose positions start at 5 still converts to a 0-based grid."""
        spectra = [
            Spectrum(5, 9, [100.0, 101.0], [1.0, 2.0]),
            Spectrum(6, 9, [100.0, 101.0], [3.0, 4.0]),
            Spectrum(5, 10, [100.0, 101.0], [5.0, 6.0]),
        ]
        archive = build_mzpeak(tmp_path / "offset.mzpeak", spectra)

        with MzPeakReader(archive) as reader:
            coords = [c for c, _, _ in reader.iter_spectra()]
            essential = reader.get_essential_metadata()

        assert coords == [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        assert essential.coordinate_offsets == (5, 9, 0)

    def test_missing_pixels_are_left_missing(self, tmp_path):
        """A sparse acquisition yields only acquired pixels.

        Missing pixels are ordinary in imaging mzPeak -- they were 38% of the
        real dataset this reader was designed against -- and the converter's
        sparse grid handles the gaps. Densifying here would invent spectra.
        """
        spectra = grid_spectra(3, 3, skip=[(2, 2), (3, 1)])
        archive = build_mzpeak(tmp_path / "sparse.mzpeak", spectra)

        with MzPeakReader(archive) as reader:
            coords = [c for c, _, _ in reader.iter_spectra()]
            essential = reader.get_essential_metadata()

        assert len(coords) == 7
        assert (1, 1, 0) not in coords  # the skipped (2, 2) in file frame
        assert (2, 0, 0) not in coords  # the skipped (3, 1) in file frame
        assert essential.dimensions == (3, 3, 1)
        assert essential.n_spectra == 7


class TestMassAxis:
    """The reader's view of the m/z axis."""

    def test_never_claims_a_shared_axis(self, simple_archive):
        """mzPeak has no shared-axis concept anywhere in the format."""
        with MzPeakReader(simple_archive) as reader:
            assert reader.has_shared_mass_axis is False

    def test_common_axis_is_the_sorted_union(self, tmp_path):
        """Per-spectrum axes combine into one ascending, deduplicated axis."""
        spectra = [
            Spectrum(1, 1, [100.0, 102.0], [1.0, 2.0]),
            Spectrum(2, 1, [101.0, 102.0], [3.0, 4.0]),
        ]
        archive = build_mzpeak(tmp_path / "union.mzpeak", spectra)

        with MzPeakReader(archive) as reader:
            axis = reader.get_common_mass_axis()

        np.testing.assert_array_equal(axis, [100.0, 101.0, 102.0])


class TestNullPairPadding:
    """Null pairs mark removed zero runs and must not reach the converter."""

    def test_padding_is_dropped_from_the_payload(self, tmp_path):
        """Rows whose m/z and intensity are both null never surface."""
        spectra = grid_spectra(2, 1, n_points=6)
        archive = build_mzpeak(tmp_path / "padded.mzpeak", spectra, null_pair_after=3)

        with MzPeakReader(archive) as reader:
            emitted = list(reader.iter_spectra())

        for _, mzs, intensities in emitted:
            assert mzs.size == 6
            assert intensities.size == 6
            assert not np.isnan(mzs).any()
            assert not np.isnan(intensities).any()

    def test_the_dropped_count_is_per_iteration(self, tmp_path):
        """Every conversion iterates twice, and the count must not accumulate.

        ``_dropped_points`` was set once in ``__init__`` and only ever
        incremented, so the second pass logged the sum of both passes --
        12 dropped points, then 24, then 36 on a third iteration of the
        same file (issue #238). Cleared as ``iter_spectra`` starts, which
        also covers a caller that iterates again without ``reset()``.
        """
        spectra = grid_spectra(3, 2, n_points=6)
        archive = build_mzpeak(tmp_path / "counted.mzpeak", spectra, null_pair_after=3)

        counts = []
        with MzPeakReader(archive) as reader:
            for _ in range(3):
                reader.reset()
                list(reader.iter_spectra())
                counts.append(reader._dropped_points)
            list(reader.iter_spectra())  # no reset() at all
            counts.append(reader._dropped_points)

        assert counts[0] > 0
        assert counts == [counts[0]] * 4

    def test_padding_is_excluded_from_the_mass_axis(self, tmp_path):
        """The axis holds only channels that can take a value."""
        spectra = grid_spectra(2, 1, n_points=6)
        archive = build_mzpeak(
            tmp_path / "padded_axis.mzpeak", spectra, null_pair_after=3
        )

        with MzPeakReader(archive) as reader:
            axis = reader.get_common_mass_axis()

        assert not np.isnan(axis).any()
        assert np.all(np.diff(axis) > 0)

    def test_peak_counts_are_corrected_for_padding(self, tmp_path):
        """Recorded point counts include padding; the reported total does not.

        ``number_of_data_points`` counts stored rows, so a file with a null
        pair per spectrum records two more per spectrum than it can deliver.
        Reporting the recorded figure would have the converter pre-allocate
        for points that never arrive.
        """
        spectra = grid_spectra(2, 1, n_points=6)
        archive = build_mzpeak(
            tmp_path / "padded_counts.mzpeak", spectra, null_pair_after=3
        )

        with MzPeakReader(archive) as reader:
            essential = reader.get_essential_metadata()
            delivered = sum(mzs.size for _, mzs, _ in reader.iter_spectra())

        assert essential.total_peaks == delivered == 12

    def test_per_pixel_counts_declined_when_padded(self, tmp_path):
        """Per-pixel counts are withheld rather than reported inflated.

        The padding cannot be attributed to individual spectra without a full
        pass over the point data, which is the pass these counts exist to
        avoid, so the reader declines and the converter measures instead.
        """
        spectra = grid_spectra(2, 1, n_points=6)
        padded = build_mzpeak(
            tmp_path / "padded_ppp.mzpeak", spectra, null_pair_after=3
        )
        clean = build_mzpeak(tmp_path / "clean_ppp.mzpeak", spectra)

        with MzPeakReader(padded) as reader:
            assert reader.get_peak_counts_per_pixel() is None
        with MzPeakReader(clean) as reader:
            counts = reader.get_peak_counts_per_pixel()

        assert counts is not None
        np.testing.assert_array_equal(counts, [6, 6])


class TestIndexTolerance:
    """The index vocabulary is parsed as tolerantly as the reference parser."""

    @pytest.mark.parametrize("spelling", ["data_arrays", "data arrays", "DATA ARRAYS"])
    def test_data_kind_spellings_all_resolve(self, tmp_path, spelling):
        """Underscore, space and case variants name the same role.

        ``DataKind::from_str`` lowercases and trims before matching and
        carries ``data arrays`` as an explicit alias, so a reader that
        compares the raw string rejects valid archives.
        """
        archive = build_mzpeak(
            tmp_path / f"kind_{abs(hash(spelling))}.mzpeak",
            grid_spectra(2, 1),
            data_kind=spelling,
        )

        with MzPeakReader(archive) as reader:
            assert len(list(reader.iter_spectra())) == 2

    def test_metadata_mapping_alias_is_accepted(self, tmp_path):
        """``metadata_mapping`` is a serde alias for ``column_mapping``."""
        archive = build_mzpeak(
            tmp_path / "alias.mzpeak",
            grid_spectra(2, 1),
            column_mapping_key="metadata_mapping",
        )

        with MzPeakReader(archive) as reader:
            assert len(list(reader.iter_spectra())) == 2

    def test_absent_column_mapping_falls_back_to_conventional_names(self, tmp_path):
        """Bindings are ``serde(default)``, so an archive may omit them."""
        archive = build_mzpeak(
            tmp_path / "nomapping.mzpeak",
            grid_spectra(2, 1),
            column_mapping_key=None,
        )

        with MzPeakReader(archive) as reader:
            assert [c for c, _, _ in reader.iter_spectra()] == [
                (0, 0, 0),
                (1, 0, 0),
            ]


class TestRefusals:
    """Layouts and acquisitions the reader will not pretend to handle."""

    def test_chunked_layout_is_refused_by_name(self, tmp_path):
        """The chunked encoding is a different layout, not a variant."""
        archive = build_mzpeak(
            tmp_path / "chunked.mzpeak", grid_spectra(2, 1), layout="chunk"
        )

        with pytest.raises(NotImplementedError, match="chunked layout"):
            with MzPeakReader(archive) as reader:
                reader.get_essential_metadata()

    def test_non_imaging_archive_is_refused(self, tmp_path):
        """No positions means no pixels, and Thyra is MSI-only.

        A valid mzPeak file can carry no positions at all -- the reference
        converter only emits them when it happens to see imaging input.
        """
        archive = build_mzpeak(
            tmp_path / "nonimaging.mzpeak",
            grid_spectra(2, 1),
            include_positions=False,
        )

        with pytest.raises(ValueError, match="not an imaging mzPeak archive"):
            with MzPeakReader(archive) as reader:
                reader.get_essential_metadata()

    def test_region_map_is_none(self, simple_archive):
        """mzPeak carries no region or ROI identity of any kind."""
        with MzPeakReader(simple_archive) as reader:
            assert reader.get_region_map() is None
            assert reader.get_region_info() is None
