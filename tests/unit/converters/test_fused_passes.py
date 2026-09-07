"""The sibling tables fed from the summed table's own passes (design decision D5).

A stand-in source hands its frames over as records, the way a Bruker TDF
does, and also answers the three iterators the standalone passes use.
Converted both ways on the streaming route, the stores must be identical
in every table -- and the fused way must not have touched the iterators
the sinks used to read from, which is the whole saving.
"""

from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("spatialdata")

import spatialdata  # noqa: E402

from thyra.core.base_extractor import MetadataExtractor  # noqa: E402
from thyra.core.base_reader import BaseMSIReader  # noqa: E402
from thyra.core.mobility import MobilityAxis  # noqa: E402
from thyra.core.msms import FragmentationSchedule, IsolationWindow  # noqa: E402
from thyra.metadata.types import ComprehensiveMetadata, EssentialMetadata  # noqa: E402

PIXELS = [(0, 0), (1, 0), (0, 1), (1, 1)]
#: A decreasing 1/K0 ramp of 8 scans, as a TDF's is; window 0 owns scans
#: 0..4 (1/K0 1.5 .. 1.27), window 1 scans 4..8 (1.21 .. 1.1).
SCAN_K0 = np.linspace(1.5, 1.1, 8)
WINDOWS = (
    IsolationWindow(313.275, 0.5, 0.5, 34.3, scan_begin=0, scan_end=4),
    IsolationWindow(936.578, 0.5, 0.5, 49.6, scan_begin=4, scan_end=8),
)
SCHEDULE = FragmentationSchedule(ms_level=2, windows=WINDOWS, source="stub")

#: One point cloud per pixel: (m/z, scan, intensity). Two points at one
#: m/z on different scans, one m/z in both windows, one pixel sparse.
CLOUDS = {
    0: [(100.0, 1, 10.0), (100.0, 5, 4.0), (300.0, 2, 20.0), (200.0, 6, 30.0)],
    1: [(300.0, 0, 40.0), (200.0, 7, 50.0), (500.0, 5, 60.0)],
    2: [(100.0, 3, 5.0), (400.0, 4, 7.0)],
    3: [(500.0, 2, 1.0), (100.0, 6, 2.0), (500.0, 7, 3.0)],
}


def _summed(p: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    totals: dict = {}
    for mz, _scan, intensity in CLOUDS[p]:
        totals[mz] = totals.get(mz, 0.0) + intensity
    mzs = np.array(sorted(totals), dtype=np.float64)
    return mzs, np.array([totals[m] for m in mzs], dtype=np.float64)


def _cloud(p: int):
    points = sorted(CLOUDS[p], key=lambda t: (t[1], t[0]))
    mzs = np.array([t[0] for t in points], dtype=np.float64)
    mobility = np.array([SCAN_K0[t[1]] for t in points], dtype=np.float64)
    intensities = np.array([t[2] for t in points], dtype=np.float64)
    return mzs, mobility, intensities


def _precursors(p: int) -> List[Tuple[int, NDArray[np.float64], NDArray[np.float64]]]:
    out = []
    for w, window in enumerate(WINDOWS):
        totals: dict = {}
        for mz, scan, intensity in CLOUDS[p]:
            if window.scan_begin <= scan < window.scan_end:
                totals[mz] = totals.get(mz, 0.0) + intensity
        if totals:
            mzs = np.array(sorted(totals), dtype=np.float64)
            out.append((w, mzs, np.array([totals[m] for m in mzs], dtype=np.float64)))
    return out


class _Frame:
    def __init__(self, p: int, coords):
        self.coords = coords
        self._p = p

    def spectrum(self):
        return _summed(self._p)

    def mobility_points(self):
        return _cloud(self._p)

    def precursor_spectra(self):
        return _precursors(self._p)


class _StubExtractor(MetadataExtractor):
    def __init__(self):
        super().__init__(data_source=None)

    def _extract_essential_impl(self) -> EssentialMetadata:
        return EssentialMetadata(
            dimensions=(2, 2, 1),
            coordinate_bounds=(0.0, 1.0, 0.0, 1.0),
            mass_range=(100.0, 500.0),
            pixel_size=(10.0, 10.0),
            n_spectra=4,
            total_peaks=12,
            estimated_memory_gb=0.0,
            source_path="stub_fused",
            spectrum_type="centroid spectrum",
        )

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        return ComprehensiveMetadata(
            essential=self._extract_essential_impl(),
            format_specific={"format": "stub"},
            acquisition_params={},
            instrument_info={},
            raw_metadata={},
        )


class FusedStubReader(BaseMSIReader):
    """A TIMS PASEF source in miniature that also hands its frames over as records."""

    frame_scans = True

    def __init__(self):
        super().__init__(Path("stub_fused"))
        self.mobility_passes = 0
        self.precursor_passes = 0
        self.frame_passes = 0

    def _create_metadata_extractor(self) -> MetadataExtractor:
        return _StubExtractor()

    @property
    def has_shared_mass_axis(self) -> bool:
        return False

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        return np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator:
        for p, (x, y) in enumerate(PIXELS):
            mzs, intensities = _summed(p)
            yield (x, y, 0), mzs, intensities

    # -- mobility ----------------------------------------------------------

    @property
    def has_ion_mobility(self) -> bool:
        return True

    @property
    def has_shared_mobility_axis(self) -> bool:
        return False

    def get_mobility_axis(self) -> Optional[MobilityAxis]:
        return MobilityAxis(
            kind_accession="MS:1002815",
            kind_name="inverse reduced ion mobility",
            unit_accession="MS:1002814",
            unit_name="volt-second per square centimeter",
            values=SCAN_K0.copy(),
            acq_range=(1.1, 1.5),
            source="stub",
        )

    def iter_mobility_spectra(self, batch_size: Optional[int] = None) -> Generator:
        self.mobility_passes += 1
        for p, (x, y) in enumerate(PIXELS):
            mzs, mobility, intensities = _cloud(p)
            yield (x, y, 0), mzs, mobility, intensities

    # -- MS/MS -------------------------------------------------------------

    def get_fragmentation(self) -> Optional[FragmentationSchedule]:
        return SCHEDULE

    def iter_precursor_spectra(self, batch_size: Optional[int] = None) -> Generator:
        self.precursor_passes += 1
        for p, (x, y) in enumerate(PIXELS):
            for window, mzs, intensities in _precursors(p):
                yield (x, y, 0), window, mzs, intensities

    # -- the records --------------------------------------------------------

    @property
    def has_frame_scans(self) -> bool:
        return self.frame_scans

    def iter_frame_scans(self, batch_size: Optional[int] = None) -> Generator:
        self.frame_passes += 1
        for p, (x, y) in enumerate(PIXELS):
            yield _Frame(p, (x, y, 0))

    def close(self) -> None:
        pass


class UnfusedStubReader(FusedStubReader):
    frame_scans = False


RESAMPLED = {
    "method": "nearest_neighbor",
    "axis_type": "constant",
    "target_bins": 43,
    "min_mz": 90.0,
    "max_mz": 510.0,
}


def _convert(reader, out: Path, **kwargs) -> Path:
    from thyra.converters.spatialdata.streaming_converter import (
        StreamingSpatialDataConverter,
    )
    from thyra.utils.windows_paths import prepare_zarr_output_path

    out = prepare_zarr_output_path(out, "stub")
    converter = StreamingSpatialDataConverter(
        reader,
        out,
        dataset_id="stub",
        pixel_size_um=10.0,
        resampling_config=RESAMPLED,
        mobility_grid=True,
        **kwargs,
    )
    assert converter.convert(), "conversion reported failure"
    return out


def _read(out: Path):
    from thyra.utils.windows_paths import prepare_zarr_read_path

    return spatialdata.read_zarr(prepare_zarr_read_path(out))


def _dense(table) -> np.ndarray:
    X = table.X
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X)


class TestFusedPasses:
    def test_the_fused_store_is_the_unfused_store(self, tmp_path):
        fused, unfused = FusedStubReader(), UnfusedStubReader()
        a = _read(_convert(fused, tmp_path / "fused.zarr"))
        b = _read(_convert(unfused, tmp_path / "unfused.zarr"))

        assert set(a.tables) == set(b.tables)
        assert set(a.tables) == {"stub_z0", "stub_z0_mobility", "stub_z0_msms"}
        for key in a.tables:
            ta, tb = a.tables[key], b.tables[key]
            np.testing.assert_array_equal(_dense(ta), _dense(tb))
            assert list(ta.var.index) == list(tb.var.index)
            assert list(ta.obs.index) == list(tb.obs.index)
            for column in ta.var.columns:
                np.testing.assert_array_equal(
                    ta.var[column].to_numpy(), tb.var[column].to_numpy()
                )
        heat_a = a.tables["stub_z0"].uns["mobility_heatmap"]["counts"]
        heat_b = b.tables["stub_z0"].uns["mobility_heatmap"]["counts"]
        np.testing.assert_array_equal(np.asarray(heat_a), np.asarray(heat_b))

    def test_the_fused_route_reads_the_source_twice_and_never_the_iterators(
        self, tmp_path
    ):
        fused = FusedStubReader()
        _convert(fused, tmp_path / "fused.zarr")
        # Count and scatter: one read each, serving the summed table, the
        # heatmap, the grid and the split alike.
        assert fused.frame_passes == 2
        assert fused.mobility_passes == 0
        assert fused.precursor_passes == 0

        unfused = UnfusedStubReader()
        _convert(unfused, tmp_path / "unfused.zarr")
        assert unfused.frame_passes == 0
        # Heatmap + discovery fused into one, then the grid's scatter.
        assert unfused.mobility_passes == 2
        assert unfused.precursor_passes == 2

    def test_the_marginals_are_exact_either_way(self, tmp_path):
        for reader, name in ((FusedStubReader(), "f"), (UnfusedStubReader(), "u")):
            sdata = _read(_convert(reader, tmp_path / f"{name}.zarr"))
            grid = sdata.tables["stub_z0_mobility"].uns["mobility_marginal"]
            split = sdata.tables["stub_z0_msms"].uns["demultiplexed_current"]
            assert float(grid["current_ratio"]) == pytest.approx(1.0)
            assert float(split["current_ratio"]) == pytest.approx(1.0)
