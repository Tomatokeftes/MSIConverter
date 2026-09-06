"""The TDF reader end to end through the real Bruker library.

``tests/data/fixtures/synthetic_tims.d`` is a hand-written TIMS acquisition
(see ``build_tdf_fixture.py``); ``synthetic_tims_expected.json`` is what was
written into it. The library is bundled for Windows and Linux; anywhere it
cannot be loaded these tests skip rather than fail.

A second group runs only against a real acquisition named by
``THYRA_BRUKER_TDF_DATASET`` and checks the reader against the database's own
per-frame ``SummedIntensities``, and the stored mass-mobility heatmap against
the stored mean spectrum.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from thyra.core.mobility import ccs_from_one_over_k0, mason_schamp_ccs
from thyra.readers.bruker.timstof.timstof_reader import BrukerReader
from thyra.utils.bruker_exceptions import SDKError

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "synthetic_tims.d"
EXPECTED = FIXTURE.with_name("synthetic_tims_expected.json")


def _open(mode: str, path: Path = FIXTURE) -> BrukerReader:
    try:
        return BrukerReader(path, tdf_spectrum=mode)
    except (SDKError, OSError) as exc:  # the vendor library is not loadable here
        pytest.skip(f"Bruker library not loadable on this platform: {exc}")


@pytest.fixture(scope="module")
def expected() -> Dict:
    return json.loads(EXPECTED.read_text(encoding="utf-8"))


def _frames_by_coord(reader: BrukerReader, expected: Dict) -> Dict[tuple, Dict]:
    """Map normalised (x, y) -> the fixture's frame record."""
    offsets = reader.get_essential_metadata().coordinate_offsets or (0, 0, 0)
    return {(f["x"] - offsets[0], f["y"] - offsets[1]): f for f in expected["frames"]}


def _spectra_by_frame(reader: BrukerReader, expected: Dict) -> Dict[int, tuple]:
    """Map frame id -> (mz, intensity) using the fixture's grid."""
    by_coord = _frames_by_coord(reader, expected)
    out = {}
    for (x, y, _z), mzs, intensities in reader.iter_spectra():
        out[by_coord[(x, y)]["frame"]] = (mzs, intensities)
    return out


def _no_colon_keys(block) -> bool:
    """No CV accession anywhere as a key: a colon is not a zarr key on Windows."""
    if isinstance(block, dict):
        return all(":" not in str(k) and _no_colon_keys(v) for k, v in block.items())
    return True


def _convert(tmp_path: Path, mode: str, **kwargs) -> Path:
    from thyra.convert import convert_msi

    out = tmp_path / f"synthetic_tims_{mode}.zarr"
    ok = convert_msi(
        str(FIXTURE),
        str(out),
        dataset_id="tims",
        pixel_size_um=20.0,
        reader_options={"tdf_spectrum": mode},
        **kwargs,
    )
    assert ok
    return out


def _read_table(out: Path, key: str = "tims_z0"):
    spatialdata = pytest.importorskip("spatialdata")
    from thyra.utils.windows_paths import prepare_zarr_read_path

    return spatialdata.read_zarr(prepare_zarr_read_path(out)).tables[key]


class TestSyntheticFixture:
    def test_scan_sum_reads_every_scan_of_the_right_frame(self, expected):
        with _open("scan_sum") as reader:
            spectra = _spectra_by_frame(reader, expected)

        assert sorted(spectra) == [f["frame"] for f in expected["frames"]]
        for frame in expected["frames"]:
            mzs, intensities = spectra[frame["frame"]]
            # AccumulationTime is 100 ms in the fixture, so the SDK's
            # intensity scale is exactly 1 and the TIC is the written sum.
            assert intensities.sum() == pytest.approx(frame["tic"])
            assert mzs.size == frame["unique_indices"]
            assert np.all(np.diff(mzs) > 0)

    def test_first_frame_is_present(self, expected):
        # The old reader asked the SDK for frame_id - 1, which does not exist
        # for the first frame, and skipped that pixel with a warning.
        with _open("scan_sum") as reader:
            spectra = _spectra_by_frame(reader, expected)
        assert 1 in spectra

    def test_planted_ion_lands_on_its_calibrated_mz(self, expected):
        frame = expected["frames"][0]
        with _open("scan_sum") as reader:
            spectra = _spectra_by_frame(reader, expected)
            planted_mz = reader.sdk._convert_indices_to_mz(
                reader.handle, frame["frame"], np.array([float(frame["planted_index"])])
            )[0]
        mzs, intensities = spectra[frame["frame"]]
        hit = int(np.argmin(np.abs(mzs - planted_mz)))
        assert mzs[hit] == pytest.approx(planted_mz, abs=1e-9)
        assert intensities[hit] == pytest.approx(frame["planted_intensity"])

    def test_vendor_centroid_yields_one_spectrum_per_pixel(self, expected):
        with _open("scan_sum") as reader:
            summed = _spectra_by_frame(reader, expected)
        with _open("vendor_centroid") as reader:
            centroided = _spectra_by_frame(reader, expected)

        assert sorted(centroided) == sorted(summed)
        for frame_id, (mzs, intensities) in centroided.items():
            assert mzs.size > 0
            assert np.all(np.diff(mzs) > 0)
            # The vendor picker merges bins and drops single counts: never
            # more ion current than the lossless sum, never none at all.
            assert 0 < intensities.sum() <= summed[frame_id][1].sum() * (1 + 1e-9)
            assert mzs.size <= summed[frame_id][0].size

    def test_mobility_axis_decreases_with_scan_number(self, expected):
        with _open("scan_sum") as reader:
            scans = np.arange(expected["n_scans"], dtype=np.float64)
            k0 = reader.sdk.scannum_to_oneoverk0(reader.handle, 1, scans)
        assert k0.shape == scans.shape
        assert np.all(np.diff(k0) < 0)
        assert 0.5 < k0.min() < k0.max() < 2.5

    def test_no_per_pixel_peak_counts_for_tdf(self):
        with _open("scan_sum") as reader:
            assert reader.get_peak_counts_per_pixel() is None

    def test_mobility_axis_is_the_per_scan_calibration(self, expected):
        with _open("scan_sum") as reader:
            assert reader.has_ion_mobility is True
            # Each pixel is its own point cloud: no shared feature list, so
            # no mobility-resolved sibling table in this phase.
            assert reader.has_shared_mobility_axis is False
            axis = reader.get_mobility_axis()
            scans = np.arange(expected["n_scans"], dtype=np.float64)
            direct = reader.sdk.scannum_to_oneoverk0(reader.handle, 1, scans)
        assert axis.kind_accession == "MS:1002815"
        assert axis.unit_accession == "MS:1002814"
        assert axis.values.size == expected["n_scans"]
        np.testing.assert_array_equal(axis.values, direct)
        assert np.all(np.diff(axis.values) < 0)
        assert axis.acq_range == (0.9, 1.99)
        assert axis.calibration["model_type"] == 2
        assert len(axis.calibration["coefficients"]) == 10
        assert axis.source == "bruker_tdf"

    def test_mobility_spectra_are_the_written_pairs(self, expected):
        with _open("scan_sum") as reader:
            by_coord = _frames_by_coord(reader, expected)
            values = reader.get_mobility_axis().values
            seen = []
            for (
                (x, y, _z),
                mzs,
                mobility,
                intensities,
            ) in reader.iter_mobility_spectra():
                frame = by_coord[(x, y)]
                # (index, scan, intensity), written scan by scan, index ascending.
                pairs = np.asarray(frame["pairs"], dtype=np.int64)
                expected_mz = reader.sdk.index_to_mz(
                    reader.handle, frame["frame"], pairs[:, 0].astype(np.float64)
                )
                assert mzs.shape == mobility.shape == intensities.shape
                np.testing.assert_array_equal(mzs, expected_mz)
                np.testing.assert_array_equal(mobility, values[pairs[:, 1]])
                np.testing.assert_array_equal(intensities, pairs[:, 2])
                assert intensities.sum() == pytest.approx(frame["tic"])
                seen.append(frame["frame"])
        assert sorted(seen) == [f["frame"] for f in expected["frames"]]

    def test_mobility_cloud_sums_to_the_scan_sum_spectrum(self, expected):
        with _open("scan_sum") as reader:
            summed = {c: (m, i) for c, m, i in reader.iter_spectra()}
            n = 0
            for coords, mzs, _mobility, intensities in reader.iter_mobility_spectra():
                unique, inverse = np.unique(mzs, return_inverse=True)
                sums = np.bincount(np.asarray(inverse).ravel(), weights=intensities)
                np.testing.assert_array_equal(unique, summed[coords][0])
                np.testing.assert_allclose(sums, summed[coords][1])
                n += 1
        assert n == len(summed) == 6

    def test_ccs_sdk_and_formula_agree(self):
        ook0 = np.array([0.8, 1.0, 1.2, 1.5, 1.9])
        mz = np.array([300.0, 500.0, 760.5, 1000.0, 1500.0])
        with _open("scan_sum") as reader:
            for charge in (1, 2):
                vendor = reader.sdk.oneoverk0_to_ccs(ook0, charge, mz)
                np.testing.assert_allclose(
                    mason_schamp_ccs(ook0, mz, charge), vendor, rtol=1e-6
                )
                # With the library loaded the dispatcher goes through it.
                np.testing.assert_array_equal(
                    ccs_from_one_over_k0(ook0, mz, charge, sdk=reader.sdk), vendor
                )
        assert 150.0 < vendor[0] < 400.0

    def test_extractor_reports_the_mobility_dimension(self, expected):
        with _open("scan_sum") as reader:
            info = reader.get_comprehensive_metadata().format_specific["ion_mobility"]
        assert info["present"] is True
        assert info["separation_accession"] == "MS:1002815"
        assert info["num_scans_min"] == info["num_scans_max"] == expected["n_scans"]
        assert info["one_over_k0_range"] == [0.9, 1.99]

    def test_conversion_records_the_mobility_block_and_the_summation(self, tmp_path):
        spatialdata = pytest.importorskip("spatialdata")
        from thyra.convert import convert_msi

        _open("scan_sum").close()  # skip early where the library is missing
        out = tmp_path / "synthetic_tims.zarr"
        ok = convert_msi(
            str(FIXTURE),
            str(out),
            dataset_id="tims",
            pixel_size_um=20.0,
            reader_options={"tdf_spectrum": "scan_sum"},
        )
        assert ok

        from thyra.metadata.schema import read_msi_metadata_blocks

        sdata = spatialdata.read_zarr(out)
        table = next(iter(sdata.tables.values()))
        assert table.n_obs == 6
        block = next(iter(read_msi_metadata_blocks(out).values()))
        mobility = block["ms_analysis"]["ion_mobility"]
        assert mobility["present"] is True
        assert mobility["num_scans"] == 240
        assert mobility["separation_term"]["accession"] == "MS:1002815"
        conversion = block["processing"][0]
        assert conversion["name"] == "conversion"
        assert conversion["parameters"]["tdf_spectrum"] == "scan_sum"
        assert "resolved_table" not in mobility and "grid" not in mobility
        assert block["schema_version"] == "0.4.0"

    def test_a_survey_acquisition_is_recorded_as_unfragmented(self, tmp_path, expected):
        """The fixture is MS1, and the store says so rather than staying silent.

        ``present: false`` is a claim the database supports -- ``MsMsType``
        is there and says 0 -- and it is what tells a consumer the m/z axis
        is intact-ion m/z. The array block beside it is for MS/MS only, so
        an MS1 store must not carry one.
        """
        from thyra.metadata.schema import read_msi_metadata_blocks

        out = _convert(tmp_path, "scan_sum")
        table = _read_table(out)
        block = next(iter(read_msi_metadata_blocks(out).values()))

        fragmentation = block["ms_analysis"]["fragmentation"]
        assert fragmentation["present"] is False
        assert fragmentation["ms_level"] == 1
        assert not fragmentation.get("windows")
        assert "msms_schedule" not in table.uns

    @pytest.mark.parametrize("mode", ["scan_sum", "vendor_centroid"])
    def test_conversion_writes_the_axis_and_the_heatmap(self, tmp_path, mode, expected):
        _open(mode).close()  # skip early where the library is missing
        table = _read_table(_convert(tmp_path, mode))

        axis = table.uns["mobility_axis"]
        assert _no_colon_keys(table.uns)
        assert axis["type_accession"] == "MS:1002815"
        assert axis["type_name"] == "inverse reduced ion mobility"
        assert axis["unit_accession"] == "MS:1002814"
        assert axis["n_scans"] == expected["n_scans"]
        values = np.asarray(axis["values"])
        assert values.dtype == np.float64 and values.size == expected["n_scans"]
        np.testing.assert_array_equal(np.asarray(axis["acq_range"]), [0.9, 1.99])
        assert axis["calibration"]["model_type"] == 2
        assert np.asarray(axis["calibration"]["coefficients"]).size == 10
        assert axis["source"] == "bruker_tdf"
        assert "resolved_table" not in axis

        heat = table.uns["mobility_heatmap"]
        assert set(heat) == {"mz_edges", "mobility_edges", "counts"}
        counts = np.asarray(heat["counts"])
        mz_edges = np.asarray(heat["mz_edges"])
        mobility_edges = np.asarray(heat["mobility_edges"])
        assert counts.dtype == np.float32
        # The fixture's axis is far shorter than 4,000 bins: one bin per entry.
        assert counts.shape == (table.n_vars, 256)
        assert mz_edges.shape == (table.n_vars + 1,)
        assert mobility_edges.shape == (257,)
        # Binned over the axis values, which overhang the declared range.
        assert mobility_edges[0] == pytest.approx(values.min())
        assert mobility_edges[-1] == pytest.approx(values.max())
        mz = table.var["mz"].to_numpy()
        assert np.all(mz_edges[:-1] < mz) and np.all(mz < mz_edges[1:])

        marginal = counts.sum(axis=1).astype(np.float64)
        mean_spectrum = np.asarray(table.uns["average_spectrum"])
        if mode == "scan_sum":
            # Lossless sum: the heatmap over mobility IS the mean spectrum.
            np.testing.assert_allclose(marginal, mean_spectrum, rtol=1e-6)
        else:
            # The vendor centroid discards part of the current the raw
            # cloud carries; the heatmap is built from the cloud.
            assert marginal.sum() >= mean_spectrum.sum() * (1 - 1e-6)
            assert marginal.sum() > 0

    def test_heatmap_can_be_switched_off(self, tmp_path):
        _open("scan_sum").close()
        table = _read_table(_convert(tmp_path, "scan_sum", mobility_heatmap=False))
        assert "mobility_heatmap" not in table.uns
        assert table.uns["mobility_axis"]["type_accession"] == "MS:1002815"


REAL_DATASET = os.environ.get("THYRA_BRUKER_TDF_DATASET")


@pytest.mark.skipif(
    not REAL_DATASET,
    reason="Set THYRA_BRUKER_TDF_DATASET to a TIMS .d directory to run",
)
class TestRealAcquisition:
    def test_every_frame_is_read_and_the_lossless_tic_matches_the_database(self):
        path = Path(REAL_DATASET)  # type: ignore[arg-type]
        con = sqlite3.connect(
            f"file:{(path / 'analysis.tdf').as_posix()}?mode=ro&immutable=1", uri=True
        )
        frames = dict(
            con.execute(
                "SELECT m.Frame, f.SummedIntensities FROM MaldiFrameInfo m "
                "JOIN Frames f ON f.Id = m.Frame ORDER BY m.Frame LIMIT 5"
            ).fetchall()
        )
        n_frames = con.execute("SELECT COUNT(*) FROM MaldiFrameInfo").fetchone()[0]
        con.close()

        with _open("scan_sum", path) as reader:
            n_pixels = sum(1 for _ in reader.iter_spectra())
            per_frame = {
                frame_id: reader.sdk.read_spectrum(
                    reader.handle, frame_id, num_scans=reader._frame_num_scans(frame_id)
                )
                for frame_id in frames
            }
        assert n_pixels == n_frames
        for frame_id, summed in frames.items():
            # SummedIntensities is the unrounded scaled sum; the SDK rounds
            # each pair after scaling, so allow a few counts per pair.
            tic = per_frame[frame_id][1].sum()
            assert tic == pytest.approx(summed, rel=5e-3)

        with _open("vendor_centroid", path) as reader:
            for frame_id in frames:
                mzs, intensities = reader.sdk.read_spectrum(
                    reader.handle, frame_id, num_scans=reader._frame_num_scans(frame_id)
                )
                assert mzs.size > 0
                assert (
                    0.5 * frames[frame_id]
                    < intensities.sum()
                    <= frames[frame_id] * 1.01
                )

    def test_heatmap_marginal_is_the_mean_spectrum_and_shows_the_trend(self, tmp_path):
        """Valid under scan_sum only: the vendor centroid is not the marginal."""
        from scipy.stats import spearmanr

        from thyra.convert import convert_msi

        path = Path(REAL_DATASET)  # type: ignore[arg-type]
        out = tmp_path / "real_scan_sum.zarr"
        assert convert_msi(
            str(path),
            str(out),
            dataset_id="real",
            reader_options={"tdf_spectrum": "scan_sum"},
        )
        table = _read_table(out, "real_z0")
        heat = table.uns["mobility_heatmap"]
        counts = np.asarray(heat["counts"], dtype=np.float64)
        mz_edges = np.asarray(heat["mz_edges"])
        mobility_edges = np.asarray(heat["mobility_edges"])
        assert counts.shape[1] == 256
        assert 0.9 * 4000 <= counts.shape[0] <= 4000

        # The stored mean spectrum coarsened onto the heatmap's m/z bins.
        mean_spectrum = np.asarray(table.uns["average_spectrum"])
        mz = table.var["mz"].to_numpy()
        which = np.clip(
            np.searchsorted(mz_edges, mz, side="right") - 1, 0, counts.shape[0] - 1
        )
        coarse = np.bincount(which, weights=mean_spectrum, minlength=counts.shape[0])
        marginal = counts.sum(axis=1)
        signal = coarse > 0
        np.testing.assert_allclose(marginal[signal], coarse[signal], rtol=1e-5)
        assert marginal.sum() == pytest.approx(coarse.sum(), rel=1e-6)

        # Heavier ions drift slower: the intensity-weighted mean 1/K0 of the
        # stronger m/z bins rises with m/z (the mass-mobility trend line).
        mz_centres = 0.5 * (mz_edges[:-1] + mz_edges[1:])
        k0_centres = 0.5 * (mobility_edges[:-1] + mobility_edges[1:])
        strong = marginal > np.percentile(marginal[marginal > 0], 50)
        mean_k0 = (counts[strong] @ k0_centres) / marginal[strong]
        rho = spearmanr(mz_centres[strong], mean_k0).statistic
        assert rho > 0.3, f"no mass-mobility trend: Spearman {rho:.2f}"
