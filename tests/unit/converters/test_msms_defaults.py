"""The demultiplexed MS/MS table's defaults, on a real conversion.

Design decisions D2 and D6 (``docs/design-decisions.md``): the table is
written by default when the schedule allows an exact split, opted out of
with ``msms_table=False``, and exact on a resampled axis and on the raw
one alike, because the fragment axis is the summed table's axis and both
are built from the same points. A stand-in PASEF reader with two
precursors on disjoint ramp slices goes down both write routes.
"""

from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("spatialdata")

import spatialdata  # noqa: E402

from thyra.core.base_extractor import MetadataExtractor  # noqa: E402
from thyra.core.base_reader import BaseMSIReader  # noqa: E402
from thyra.core.msms import FragmentationSchedule, IsolationWindow  # noqa: E402
from thyra.metadata.types import ComprehensiveMetadata, EssentialMetadata  # noqa: E402

WINDOWS = (
    IsolationWindow(313.275, 0.5, 0.5, 34.3, scan_begin=150, scan_end=200),
    IsolationWindow(936.578, 0.5, 0.5, 49.6, scan_begin=20, scan_end=60),
)
SCHEDULE = FragmentationSchedule(ms_level=2, windows=WINDOWS, source="stub")

PIXELS = [(0, 0), (1, 0), (0, 1), (1, 1)]

# (window index, m/z values, intensities) per pixel; the summed spectrum
# is their sum, so the split adds back up exactly.
SPLIT = {
    0: [(0, [100.0, 300.0], [10.0, 20.0]), (1, [200.0], [30.0])],
    1: [(0, [300.0], [40.0]), (1, [200.0, 500.0], [50.0, 60.0])],
    2: [(0, [100.0], [5.0]), (1, [400.0], [7.0])],
    3: [(0, [500.0], [1.0]), (1, [100.0, 500.0], [2.0, 3.0])],
}


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
            source_path="stub_msms",
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


class MsmsStubReader(BaseMSIReader):
    """A PASEF acquisition in miniature: two precursors per pixel."""

    def __init__(self):
        super().__init__(Path("stub_msms"))

    def _create_metadata_extractor(self) -> MetadataExtractor:
        return _StubExtractor()

    @property
    def has_shared_mass_axis(self) -> bool:
        return False

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        return np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator:
        for p, (x, y) in enumerate(PIXELS):
            summed: dict = {}
            for _window, mzs, intensities in SPLIT[p]:
                for mz, i in zip(mzs, intensities):
                    summed[mz] = summed.get(mz, 0.0) + i
            mzs = np.array(sorted(summed), dtype=np.float64)
            yield (x, y, 0), mzs, np.array([summed[m] for m in mzs])

    def get_fragmentation(self) -> Optional[FragmentationSchedule]:
        return SCHEDULE

    def iter_precursor_spectra(self, batch_size: Optional[int] = None) -> Generator:
        for p, (x, y) in enumerate(PIXELS):
            for window, mzs, intensities in SPLIT[p]:
                yield (
                    (x, y, 0),
                    window,
                    np.asarray(mzs, dtype=np.float64),
                    np.asarray(intensities, dtype=np.float64),
                )

    def close(self) -> None:
        pass


#: A constant-width axis at 10 Da from 90 to 510, so every m/z the stub
#: emits sits on a bin centre and nearest-bin mapping is exact.
RESAMPLED = {
    "method": "nearest_neighbor",
    "axis_type": "constant",
    "target_bins": 43,
    "min_mz": 90.0,
    "max_mz": 510.0,
}


def _convert(out: Path, streaming: bool, **kwargs):
    from thyra.utils.windows_paths import prepare_zarr_output_path

    out = prepare_zarr_output_path(out, "stub")
    if streaming:
        from thyra.converters.spatialdata.streaming_converter import (
            StreamingSpatialDataConverter as Converter,
        )
    else:
        from thyra.converters.spatialdata.spatialdata_2d_converter import (
            SpatialData2DConverter as Converter,
        )

    converter = Converter(
        MsmsStubReader(), out, dataset_id="stub", pixel_size_um=10.0, **kwargs
    )
    assert converter.convert(), "conversion reported failure"
    return out


def _read(out: Path):
    from thyra.utils.windows_paths import prepare_zarr_read_path

    return spatialdata.read_zarr(prepare_zarr_read_path(out))


@pytest.mark.parametrize("streaming", [False, True], ids=["in_memory", "streaming"])
class TestTheDefault:
    def test_a_qualifying_source_gets_the_split_without_asking(
        self, tmp_path, streaming
    ):
        sdata = _read(
            _convert(tmp_path / "s.zarr", streaming, resampling_config=RESAMPLED)
        )
        assert set(sdata.tables) == {"stub_z0", "stub_z0_msms"}
        split = sdata.tables["stub_z0_msms"]
        assert split.n_obs == 4
        assert set(split.var["precursor_index"].to_numpy()) == {0, 1}
        ratio = split.uns["demultiplexed_current"]["current_ratio"]
        assert float(ratio) == pytest.approx(1.0)

    def test_opting_out_writes_only_the_summed_table(self, tmp_path, streaming):
        sdata = _read(
            _convert(
                tmp_path / "s.zarr",
                streaming,
                resampling_config=RESAMPLED,
                msms_table=False,
            )
        )
        assert set(sdata.tables) == {"stub_z0"}

    def test_a_raw_axis_gets_the_split_exactly(self, tmp_path, streaming):
        # D6: the fragment axis is the summed table's axis. Unresampled,
        # that axis is the union of the fragment m/z values the split
        # re-reads, so the mapping is exact and the blocks add back up.
        sdata = _read(_convert(tmp_path / "s.zarr", streaming))
        assert set(sdata.tables) == {"stub_z0", "stub_z0_msms"}
        split = sdata.tables["stub_z0_msms"]
        ratio = split.uns["demultiplexed_current"]["current_ratio"]
        assert float(ratio) == pytest.approx(1.0)
        assert float(split.uns["demultiplexed_current"]["current_ratio_pixel_max"]) == (
            pytest.approx(1.0)
        )
