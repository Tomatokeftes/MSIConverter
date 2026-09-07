"""The mass-mobility heatmap: binning rules, the accumulator, the stored block.

The accumulator is driven with hand-picked points first, then a stub
reader with a known (m/z, 1/K0, intensity) cloud per pixel goes down the
in-memory and streaming write routes so the block round-trips through
zarr -- on Windows too, which is where a colon in a key would fail.
"""

from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from thyra.converters.spatialdata import mobility_heatmap as mh
from thyra.converters.spatialdata.mobility_heatmap import (
    HEATMAP_MOBILITY_CHANNELS,
    HEATMAP_MZ_BINS,
    MobilityHeatmap,
    build_mobility_heatmap,
    map_indexed_points_to_axis,
    map_points_to_axis,
    mobility_bin_edges,
    mz_bin_edges,
)
from thyra.core.base_extractor import MetadataExtractor
from thyra.core.base_reader import BaseMSIReader
from thyra.core.mobility import MobilityAxis
from thyra.metadata.types import ComprehensiveMetadata, EssentialMetadata


class TestConstants:
    def test_the_channel_count_is_the_alignment_anchor(self):
        # The opt-in grid table defaults to the same channel count so a box
        # on the heatmap maps onto grid channels by index. Not a tunable.
        assert HEATMAP_MOBILITY_CHANNELS == 256
        assert HEATMAP_MZ_BINS == 4000


class TestMzBinEdges:
    def test_a_short_axis_keeps_every_entry(self):
        axis = np.linspace(100.0, 200.0, 11)
        edges, step = mz_bin_edges(axis)
        assert step == 1
        assert edges.shape == (12,)
        assert np.all(edges[:-1] < axis) and np.all(axis < edges[1:])
        np.testing.assert_allclose(edges[1:-1], (axis[:-1] + axis[1:]) / 2)
        assert edges[0] == pytest.approx(95.0) and edges[-1] == pytest.approx(205.0)

    def test_a_long_axis_is_coarsened_by_an_integer_step(self):
        axis = np.arange(10, dtype=np.float64)
        edges, step = mz_bin_edges(axis, target_bins=4)
        assert step == 3  # ceil(10 / 4)
        assert edges.shape == (5,)  # ceil(10 / 3) = 4 bins
        # Entry i lands in bin i // step, and the outer edges are the axis'.
        for i, value in enumerate(axis):
            b = i // step
            assert edges[b] < value < edges[b + 1]
        assert edges[0] == pytest.approx(-0.5) and edges[-1] == pytest.approx(9.5)

    def test_the_target_is_approximate_never_exceeded_by_much(self):
        axis = np.linspace(50.0, 1000.0, 123_457)
        edges, step = mz_bin_edges(axis)
        assert step == 31
        assert HEATMAP_MZ_BINS * 0.9 <= edges.size - 1 <= HEATMAP_MZ_BINS

    def test_single_entry_axis(self):
        edges, step = mz_bin_edges(np.array([500.0]))
        assert step == 1
        np.testing.assert_array_equal(edges, [499.5, 500.5])

    def test_empty_axis_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            mz_bin_edges(np.array([]))


class TestMobilityBinEdges:
    def test_default_is_256_equal_channels(self):
        edges = mobility_bin_edges(1.0, 1.5)
        assert edges.shape == (257,)
        assert edges[0] == 1.0 and edges[-1] == 1.5
        np.testing.assert_allclose(np.diff(edges), 0.5 / 256)

    @pytest.mark.parametrize("lower,upper", [(1.0, 1.0), (1.2, 1.0), (np.nan, 1.0)])
    def test_a_range_without_extent_is_refused(self, lower, upper):
        with pytest.raises(ValueError, match="extent"):
            mobility_bin_edges(lower, upper)


def _heatmap(channels: int = 4) -> MobilityHeatmap:
    return MobilityHeatmap(np.linspace(100.0, 200.0, 11), (1.0, 2.0), channels=channels)


class TestAccumulator:
    def test_shapes_and_dtypes(self):
        heat = _heatmap()
        heat.add([150.0], [1.3], [7.0])
        block = heat.finalize()
        assert set(block) == {"mz_edges", "mobility_edges", "counts"}
        assert block["mz_edges"].dtype == np.float64 and block["mz_edges"].shape == (
            12,
        )
        assert block["mobility_edges"].dtype == np.float64
        assert block["mobility_edges"].shape == (5,)
        assert block["counts"].dtype == np.float32 and block["counts"].shape == (11, 4)

    def test_a_point_lands_on_its_axis_entry_and_channel(self):
        heat = _heatmap()
        heat.add([150.0], [1.3], [7.0])  # entry 5; channel floor(0.3 * 4) = 1
        counts = heat.finalize()["counts"]
        assert counts[5, 1] == 7.0
        assert counts.sum() == 7.0

    def test_nearest_entry_ties_go_right_like_the_summed_spectrum(self):
        heat = _heatmap()
        heat.add([105.0, 104.9], [1.0, 1.0], [1.0, 2.0])
        counts = heat.finalize()["counts"]
        assert counts[1, 0] == 1.0  # exactly midway: the right neighbour
        assert counts[0, 0] == 2.0

    def test_the_upper_mobility_edge_belongs_to_the_last_channel(self):
        heat = _heatmap()
        heat.add([150.0, 150.0], [2.0, 1.0], [1.0, 2.0])
        counts = heat.finalize()["counts"]
        assert counts[5, 3] == 1.0 and counts[5, 0] == 2.0

    def test_mobility_beyond_the_range_is_clipped_not_lost(self):
        heat = _heatmap()
        heat.add([150.0, 150.0], [2.4, 0.6], [1.0, 2.0])
        counts = heat.finalize()["counts"]
        assert counts[5, 3] == 1.0 and counts[5, 0] == 2.0
        assert counts.sum() == 3.0

    def test_mz_outside_the_axis_is_dropped_and_counted(self):
        heat = _heatmap()
        heat.add([99.9, 150.0, 200.1], [1.5, 1.5, 1.5], [1.0, 2.0, 3.0])
        assert heat.n_out_of_range == 2
        assert heat.n_points == 1
        assert heat.finalize()["counts"].sum() == 2.0

    def test_counts_are_the_mean_over_pixels_added(self):
        heat = _heatmap()
        heat.add([150.0], [1.3], [8.0])
        heat.add([150.0], [1.3], [4.0])
        heat.add([], [], [])  # a pixel with nothing in range still counts
        assert heat.n_pixels == 3
        assert heat.finalize()["counts"][5, 1] == pytest.approx(4.0)

    def test_coincident_points_sum(self):
        heat = _heatmap()
        heat.add([150.0, 150.0, 150.0], [1.3, 1.31, 1.3], [1.0, 2.0, 3.0])
        assert heat.finalize()["counts"][5, 1] == 6.0

    def test_buffer_flushes_do_not_change_the_result(self, monkeypatch):
        rng = np.random.default_rng(3)
        mzs = rng.uniform(100.0, 200.0, 5000)
        mob = rng.uniform(1.0, 2.0, 5000)
        w = rng.uniform(0.0, 10.0, 5000)

        unbuffered = _heatmap()
        unbuffered.add(mzs, mob, w)
        expected = unbuffered.finalize()["counts"]

        monkeypatch.setattr(mh, "_FLUSH_POINTS", 700)
        buffered = _heatmap()
        for start in range(0, 5000, 300):
            sl = slice(start, start + 300)
            buffered.add(mzs[sl], mob[sl], w[sl])
        buffered.n_pixels = 1
        np.testing.assert_allclose(buffered.finalize()["counts"], expected, rtol=1e-6)

    def test_coarse_bins_follow_the_axis_step(self):
        heat = MobilityHeatmap(np.arange(10, dtype=np.float64), (0.0, 1.0), mz_bins=4)
        heat.add([0.0, 2.0, 3.0, 9.0], [0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0])
        counts = heat.finalize()["counts"]
        assert counts.shape[0] == 4
        assert counts[0].sum() == 2.0  # entries 0..2
        assert counts[1].sum() == 1.0  # entries 3..5
        assert counts[3].sum() == 1.0  # entry 9


class TestIndexedMapping:
    """``map_indexed_points_to_axis`` is the flat mapping, gathered.

    A TDF frame reports its points as digitizer indices, so its m/z
    values repeat; mapping the distinct ones and gathering must give what
    mapping every point gives, bins, masked arrays and drop count alike.
    """

    AXIS = np.array([100.0, 110.0, 120.0, 130.0])

    def _both(self, mzs, mobility, intensities):
        """Map one point cloud both ways: point by point, and factored."""
        flat = map_points_to_axis(
            self.AXIS,
            np.asarray(mzs, dtype=np.float64),
            np.asarray(mobility, dtype=np.float64),
            np.asarray(intensities, dtype=np.float64),
        )
        unique, inverse = np.unique(
            np.asarray(mzs, dtype=np.float64), return_inverse=True
        )
        indexed = map_indexed_points_to_axis(
            self.AXIS,
            unique,
            np.asarray(inverse).ravel(),
            np.asarray(mobility, dtype=np.float64),
            np.asarray(intensities, dtype=np.float64),
        )
        return flat, indexed

    @staticmethod
    def _same(flat, indexed):
        np.testing.assert_array_equal(indexed[0], flat[0])
        np.testing.assert_array_equal(indexed[1], flat[1])
        np.testing.assert_array_equal(indexed[2], flat[2])
        assert indexed[3] == flat[3]
        assert indexed[0].dtype == flat[0].dtype == np.int64

    def test_repeated_mz_values_map_to_the_same_bins(self):
        mzs = [100.0, 121.0, 100.0, 130.0, 121.0, 104.9, 121.0]
        mobility = np.linspace(1.5, 1.1, len(mzs))
        intensities = np.arange(1.0, len(mzs) + 1.0)
        flat, indexed = self._both(mzs, mobility, intensities)
        self._same(flat, indexed)
        assert flat[3] == 0 and flat[0].size == len(mzs)

    def test_out_of_range_points_are_dropped_the_same_way(self):
        # Below the axis, above it, and repeats of both.
        mzs = [99.9, 100.0, 140.0, 121.0, 99.9, 140.0, 130.0]
        mobility = np.linspace(1.5, 1.1, len(mzs))
        intensities = np.arange(1.0, len(mzs) + 1.0)
        flat, indexed = self._both(mzs, mobility, intensities)
        self._same(flat, indexed)
        assert flat[3] == 4 and flat[0].size == 3

    def test_every_point_out_of_range(self):
        flat, indexed = self._both(
            [10.0, 10.0, 500.0], [1.5, 1.4, 1.3], [1.0, 2.0, 3.0]
        )
        self._same(flat, indexed)
        assert flat[0].size == 0 and flat[3] == 3

    def test_a_frame_with_no_points(self):
        empty = np.zeros(0, dtype=np.float64)
        flat = map_points_to_axis(self.AXIS, empty, empty, empty)
        indexed = map_indexed_points_to_axis(
            self.AXIS, empty, np.zeros(0, dtype=np.int64), empty, empty
        )
        self._same(flat, indexed)

    def test_ties_go_right_through_the_gather_too(self):
        # 105.0 is exactly between two entries; the flat mapping takes the
        # right one, and the gathered mapping is the same mapping.
        flat, indexed = self._both([105.0, 105.0], [1.5, 1.4], [1.0, 2.0])
        self._same(flat, indexed)
        np.testing.assert_array_equal(flat[0], [1, 1])

    def test_the_unique_values_are_mapped_once(self, monkeypatch):
        calls = []
        original = mh._nn_map_to_bins

        def counted(axis, mzs):
            calls.append(np.asarray(mzs).size)
            return original(axis, mzs)

        monkeypatch.setattr(mh, "_nn_map_to_bins", counted)
        mzs = np.array([100.0, 121.0] * 50)
        map_indexed_points_to_axis(
            self.AXIS,
            *np.unique(mzs, return_inverse=True),
            np.ones(mzs.size),
            np.ones(mzs.size),
        )
        assert calls == [2]

    def test_an_accumulator_fed_either_way_agrees(self):
        mzs = np.array([100.0, 121.0, 100.0, 140.0, 130.0, 121.0])
        mobility = np.linspace(1.5, 1.1, mzs.size)
        intensities = np.arange(1.0, mzs.size + 1.0)
        flat, indexed = self._both(mzs, mobility, intensities)
        a = MobilityHeatmap(self.AXIS, (1.1, 1.5))
        b = MobilityHeatmap(self.AXIS, (1.1, 1.5))
        a.add_mapped(None, *flat)
        b.add_mapped(None, *indexed)
        np.testing.assert_array_equal(a.finalize()["counts"], b.finalize()["counts"])
        assert a.n_out_of_range == b.n_out_of_range == 1


# ----------------------------------------------------------------------
# A stub reader: four pixels, a shared four-entry mass axis, three
# "scans" of 1/K0, and a known point cloud per pixel. iter_spectra yields
# the cloud summed over mobility -- the lossless summed spectrum -- so the
# heatmap's marginal must equal the stored mean spectrum exactly.
# ----------------------------------------------------------------------

MASS_AXIS = np.array([100.0, 110.0, 120.0, 130.0])
SCAN_K0 = np.array([1.5, 1.3, 1.1])  # decreasing with scan number, as on TIMS
PIXELS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def _cloud(p: int) -> Tuple[NDArray, NDArray, NDArray]:
    """(m/z, 1/K0, intensity) of pixel ``p``; m/z 100 splits over two scans."""
    mzs = np.array([100.0, 100.0, 120.0, 130.0])
    mob = SCAN_K0[[0, 2, 1, 1]]
    it = np.array([10.0 + p, 1.0 + p, 5.0 + p, 2.0 * p])
    return mzs, mob, it


def _summed(p: int) -> Tuple[NDArray, NDArray]:
    mzs, _, it = _cloud(p)
    unique, inverse = np.unique(mzs, return_inverse=True)
    sums = np.bincount(inverse.ravel(), weights=it)
    keep = sums != 0
    return unique[keep], sums[keep]


class _StubExtractor(MetadataExtractor):
    def __init__(self):
        super().__init__(data_source=None)

    def _extract_essential_impl(self) -> EssentialMetadata:
        return EssentialMetadata(
            dimensions=(2, 2, 1),
            coordinate_bounds=(0.0, 1.0, 0.0, 1.0),
            mass_range=(100.0, 130.0),
            pixel_size=(10.0, 10.0),
            n_spectra=4,
            total_peaks=12,
            estimated_memory_gb=0.0,
            source_path="stub_mobility",
            spectrum_type="centroid spectrum",
        )

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        return ComprehensiveMetadata(
            essential=self._extract_essential_impl(),
            format_specific={"format": "stub"},
            acquisition_params={},
            instrument_info={"instrument": "stub"},
            raw_metadata={},
        )


class MobilityStubReader(BaseMSIReader):
    def __init__(self, with_values: bool = True, acq_range=(1.0, 1.6)):
        super().__init__(Path("stub_mobility"))
        self._with_values = with_values
        self._acq_range = acq_range
        self.mobility_passes = 0

    def _create_metadata_extractor(self) -> MetadataExtractor:
        return _StubExtractor()

    @property
    def has_shared_mass_axis(self) -> bool:
        return True

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        return MASS_AXIS.copy()

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator:
        for p, (x, y) in enumerate(PIXELS):
            mzs, it = _summed(p)
            yield (x, y, 0), mzs, it

    @property
    def has_ion_mobility(self) -> bool:
        return True

    def get_mobility_axis(self) -> Optional[MobilityAxis]:
        return MobilityAxis(
            kind_accession="MS:1002815",
            kind_name="inverse reduced ion mobility",
            unit_accession="MS:1002814",
            unit_name="volt-second per square centimeter",
            values=SCAN_K0.copy() if self._with_values else None,
            acq_range=self._acq_range,
            source="stub",
        )

    def iter_mobility_spectra(self, batch_size: Optional[int] = None) -> Generator:
        self.mobility_passes += 1
        for p, (x, y) in enumerate(PIXELS):
            mzs, mob, it = _cloud(p)
            yield (x, y, 0), mzs, mob, it

    def close(self) -> None:
        pass


class _NoMobilityReader(MobilityStubReader):
    @property
    def has_ion_mobility(self) -> bool:
        return False


class TestBuildFromReader:
    def test_reader_without_mobility_gives_nothing(self):
        assert build_mobility_heatmap(_NoMobilityReader(), MASS_AXIS) is None

    def test_axis_without_values_falls_back_to_the_declared_range(self):
        block = build_mobility_heatmap(MobilityStubReader(with_values=False), MASS_AXIS)
        assert block is not None
        assert block["mobility_edges"][0] == 1.0 and block["mobility_edges"][-1] == 1.6

    def test_axis_without_values_or_range_gives_nothing(self):
        reader = MobilityStubReader(with_values=False, acq_range=None)
        assert build_mobility_heatmap(reader, MASS_AXIS) is None

    def test_values_decide_the_range_when_present(self):
        # The declared range overhangs the values; the values win, so no
        # scan is piled into an edge channel.
        block = build_mobility_heatmap(MobilityStubReader(), MASS_AXIS)
        assert block["mobility_edges"][0] == pytest.approx(1.1)
        assert block["mobility_edges"][-1] == pytest.approx(1.5)
        assert block["counts"].shape == (4, 256)

    def test_marginal_over_mobility_is_the_mean_summed_spectrum(self):
        block = build_mobility_heatmap(MobilityStubReader(), MASS_AXIS)
        mean = np.zeros(4)
        for p in range(4):
            mzs, it = _summed(p)
            mean[np.searchsorted(MASS_AXIS, mzs)] += it
        mean /= 4
        np.testing.assert_allclose(block["counts"].sum(axis=1), mean, rtol=1e-6)

    def test_the_split_feature_occupies_two_channels(self):
        block = build_mobility_heatmap(MobilityStubReader(), MASS_AXIS)
        row = block["counts"][0]
        occupied = np.flatnonzero(row)
        assert occupied.size == 2
        assert occupied[0] == 0 and occupied[-1] == 255  # 1/K0 1.1 and 1.5


# ----------------------------------------------------------------------
# End to end: the block on the summed table, down both write routes.
# ----------------------------------------------------------------------

spatialdata = pytest.importorskip("spatialdata")


def _convert(reader: BaseMSIReader, out: Path, streaming: bool, **kwargs):
    from thyra.utils.windows_paths import prepare_zarr_output_path

    out = prepare_zarr_output_path(out, "stub")
    if streaming:
        from thyra.converters.spatialdata.streaming_converter import (
            StreamingSpatialDataConverter,
        )

        converter = StreamingSpatialDataConverter(
            reader, out, dataset_id="stub", pixel_size_um=10.0, **kwargs
        )
    else:
        from thyra.converters.spatialdata.spatialdata_2d_converter import (
            SpatialData2DConverter,
        )

        converter = SpatialData2DConverter(
            reader, out, dataset_id="stub", pixel_size_um=10.0, **kwargs
        )
    assert converter.convert(), "conversion reported failure"
    return out


def _read(out: Path):
    from thyra.utils.windows_paths import prepare_zarr_read_path

    return spatialdata.read_zarr(prepare_zarr_read_path(out))


@pytest.mark.parametrize("streaming", [False, True], ids=["in-memory", "streaming"])
class TestStoredBlock:
    def test_round_trips_through_zarr(self, tmp_path, streaming):
        reader = MobilityStubReader()
        out = _convert(reader, tmp_path / "stub.zarr", streaming)
        table = _read(out).tables["stub_z0"]

        block = table.uns["mobility_heatmap"]
        assert set(block) == {"mz_edges", "mobility_edges", "counts"}
        counts = np.asarray(block["counts"])
        assert counts.dtype == np.float32 and counts.shape == (4, 256)
        assert np.asarray(block["mz_edges"]).shape == (5,)
        assert np.asarray(block["mobility_edges"]).shape == (257,)
        assert not any(":" in key for key in table.uns)

        # The marginal is the stored mean spectrum: same points, same
        # nearest-bin rule, lossless sum.
        np.testing.assert_allclose(
            counts.sum(axis=1), np.asarray(table.uns["average_spectrum"]), rtol=1e-6
        )
        # Built exactly once, however many uns blocks asked for it.
        assert reader.mobility_passes == 1

    def test_axis_block_carries_the_contract_keys(self, tmp_path, streaming):
        out = _convert(MobilityStubReader(), tmp_path / "stub.zarr", streaming)
        axis = _read(out).tables["stub_z0"].uns["mobility_axis"]
        assert axis["type_accession"] == "MS:1002815"
        assert axis["n_scans"] == 3
        np.testing.assert_array_equal(np.asarray(axis["values"]), SCAN_K0)
        np.testing.assert_array_equal(np.asarray(axis["acq_range"]), [1.0, 1.6])
        assert axis["source"] == "stub"

    def test_read_lazy_reaches_the_block(self, tmp_path, streaming):
        anndata = pytest.importorskip("anndata")
        from thyra.utils.windows_paths import prepare_zarr_read_path

        out = _convert(MobilityStubReader(), tmp_path / "stub.zarr", streaming)
        lazy = anndata.experimental.read_lazy(
            str(prepare_zarr_read_path(out) / "tables" / "stub_z0")
        )
        block = lazy.uns["mobility_heatmap"]
        assert set(block) == {"mz_edges", "mobility_edges", "counts"}
        assert np.asarray(block["counts"]).shape == (4, 256)

    def test_opt_out_leaves_the_axis_and_drops_the_heatmap(self, tmp_path, streaming):
        reader = MobilityStubReader()
        out = _convert(
            reader, tmp_path / "stub.zarr", streaming, mobility_heatmap=False
        )
        uns = _read(out).tables["stub_z0"].uns
        assert "mobility_heatmap" not in uns
        assert uns["mobility_axis"]["type_accession"] == "MS:1002815"
        assert reader.mobility_passes == 0

    def test_a_source_without_mobility_writes_neither_block(self, tmp_path, streaming):
        out = _convert(_NoMobilityReader(), tmp_path / "stub.zarr", streaming)
        uns = _read(out).tables["stub_z0"].uns
        assert "mobility_heatmap" not in uns and "mobility_axis" not in uns
