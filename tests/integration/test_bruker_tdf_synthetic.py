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
import logging
import os
import shutil
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

    def test_the_indexed_points_are_the_flat_points_factored(self, expected):
        # The record's two views of the same read: the indexed one leaves
        # the m/z as (distinct values, index of each point), which is what
        # the fused passes map by. unique_mz[inverse] must be the flat
        # view's m/z, value for value, or a sibling table would bin
        # points differently depending on which view it asked for.
        with _open("scan_sum") as reader:
            n = 0
            for frame in reader.iter_frame_scans():
                flat = frame.mobility_points()
                indexed = frame.mobility_points_indexed()
                assert (flat is None) == (indexed is None)
                if flat is None:
                    continue
                unique_mz, inverse, mobility, intensities = indexed
                np.testing.assert_array_equal(unique_mz[inverse], flat[0])
                np.testing.assert_array_equal(mobility, flat[1])
                np.testing.assert_array_equal(intensities, flat[2])
                assert unique_mz.size <= flat[0].size
                n += 1
        assert n == len(expected["frames"])

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
        assert block["schema_version"] == "0.5.0"

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


# Three isolation windows partitioning the fixture's 240-scan ramp:
# disjoint, gapless, and listed here in scan order while the schedule
# orders them by precursor m/z. A gapless partition is what lets the
# conservation test below be an equality rather than an inequality.
_PASEF_PARTITION = [
    (936.578, 1.0, 49.638, 0, 80),
    (353.320, 1.0, 35.172, 80, 160),
    (313.275, 1.0, 34.307, 160, 240),
]

_PASEF_DDL = """
CREATE TABLE PasefFrameMsMsInfo (
    Frame INTEGER NOT NULL, ScanNumBegin INTEGER NOT NULL,
    ScanNumEnd INTEGER NOT NULL, IsolationMz REAL NOT NULL,
    IsolationWidth REAL NOT NULL, CollisionEnergy REAL NOT NULL,
    Precursor INTEGER)
"""


def _pasef_copy(tmp_path: Path, rows=_PASEF_PARTITION) -> Path:
    """A copy of the fixture turned into a PASEF MS/MS acquisition.

    The committed fixture has ``FrameMsMsInfo`` but no
    ``PasefFrameMsMsInfo``; writing the schedule here rather than
    committing a second acquisition keeps the demultiplexer covered
    without new binary data.
    """
    target = tmp_path / "pasef.d"
    shutil.copytree(FIXTURE, target)
    with sqlite3.connect(target / "analysis.tdf") as conn:
        conn.execute("UPDATE Frames SET MsMsType = 8")
        conn.execute(_PASEF_DDL)
        frames = [row[0] for row in conn.execute("SELECT Id FROM Frames")]
        conn.executemany(
            "INSERT INTO PasefFrameMsMsInfo (Frame, ScanNumBegin, ScanNumEnd, "
            "IsolationMz, IsolationWidth, CollisionEnergy, Precursor) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL)",
            [
                (frame, lo, hi, mz, width, ce)
                for frame in frames
                for mz, width, ce, lo, hi in rows
            ],
        )
    return target


def _convert_path(source: Path, out: Path, **kwargs) -> Path:
    from thyra.convert import convert_msi

    ok = convert_msi(
        str(source),
        str(out),
        dataset_id="tims",
        pixel_size_um=20.0,
        reader_options={"tdf_spectrum": "scan_sum"},
        **kwargs,
    )
    assert ok
    return out


def _rows(table) -> np.ndarray:
    X = table.X
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X)


class TestDemultiplexedStore:
    """A PASEF acquisition converted with ``--msms-table``.

    ``scan_sum`` and no resampling, so the summed table and the
    demultiplexed one are built from the very same ion current on the very
    same axis and the conservation check below is an exact equality rather
    than a tolerance.
    """

    def test_both_tables_are_written_and_agree_on_rows(self, tmp_path):
        spatialdata = pytest.importorskip("spatialdata")
        _open("scan_sum").close()
        out = _convert_path(
            _pasef_copy(tmp_path), tmp_path / "pasef.zarr", msms_table=True
        )
        sdata = spatialdata.read_zarr(out)

        assert set(sdata.tables) == {"tims_z0", "tims_z0_msms"}
        summed, msms = sdata.tables["tims_z0"], sdata.tables["tims_z0_msms"]
        assert summed.n_obs == msms.n_obs == 6
        assert list(summed.obs.index) == list(msms.obs.index)
        assert set(msms.obs["region"].astype(str)) == {"tims_z0_pixels"}

    def test_the_precursors_are_contiguous_blocks_on_the_msi_axis(self, tmp_path):
        spatialdata = pytest.importorskip("spatialdata")
        _open("scan_sum").close()
        out = _convert_path(
            _pasef_copy(tmp_path), tmp_path / "pasef.zarr", msms_table=True
        )
        sdata = spatialdata.read_zarr(out)
        var = sdata.tables["tims_z0_msms"].var
        axis = sdata.tables["tims_z0"].var["mz"].to_numpy()

        assert list(var["precursor_mz"].unique()) == [313.275, 353.320, 936.578]
        assert np.all(np.diff(var["precursor_index"].to_numpy()) >= 0)
        # The 1/K0 each precursor was isolated at, from the vendor
        # calibration: the coordinate that keeps two isomers apart.
        mobility = var["precursor_mobility"].to_numpy()
        assert np.all(np.isfinite(mobility))
        assert 0.5 < mobility.min() <= mobility.max() < 2.5
        assert var.index.is_unique
        assert all(label.startswith("p313.275_") for label in var.index[:1])
        # Fragment m/z is the MSI table's own axis, pointed at by mz_index.
        np.testing.assert_array_equal(
            var["mz"].to_numpy(), axis[var["mz_index"].to_numpy()]
        )
        assert "mobility" not in var.columns

    def test_the_precursors_add_back_up_to_the_summed_table(self, tmp_path):
        """The assertion that proves a demultiplexing rather than an output.

        Every point of the frame is inside exactly one window, so summing
        all fragment columns of every precursor must reproduce the summed
        table's TIC pixel by pixel: nothing dropped, nothing counted twice.
        """
        spatialdata = pytest.importorskip("spatialdata")
        _open("scan_sum").close()
        out = _convert_path(
            _pasef_copy(tmp_path), tmp_path / "pasef.zarr", msms_table=True
        )
        sdata = spatialdata.read_zarr(out)

        summed = _rows(sdata.tables["tims_z0"]).sum(axis=1)
        demultiplexed = _rows(sdata.tables["tims_z0_msms"]).sum(axis=1)
        assert summed.sum() > 0
        np.testing.assert_allclose(demultiplexed, summed, rtol=1e-12)

    def test_the_summed_table_names_the_sibling_and_still_validates(self, tmp_path):
        spatialdata = pytest.importorskip("spatialdata")
        from thyra.metadata.schema import (
            check_store_var_conventions,
            read_msi_metadata_blocks,
        )

        _open("scan_sum").close()
        out = _convert_path(
            _pasef_copy(tmp_path), tmp_path / "pasef.zarr", msms_table=True
        )
        sdata = spatialdata.read_zarr(out)
        schedule = sdata.tables["tims_z0"].uns["msms_schedule"]

        assert schedule["resolved_table"] == "tims_z0_msms"
        assert schedule["n_windows"] == 3
        # The versioned block must name it too: a consumer reading only
        # msi_metadata finds the mobility sibling, so it must find this one.
        blocks = read_msi_metadata_blocks(out)
        for name, block in blocks.items():
            fragmentation = block["ms_analysis"]["fragmentation"]
            assert fragmentation["present"] is True, name
            assert fragmentation["resolved_table"] == "tims_z0_msms", name
        assert _no_colon_keys(sdata.tables["tims_z0_msms"].uns)
        issues = check_store_var_conventions(out)
        assert set(issues) == {"tims_z0", "tims_z0_msms"}
        assert all(not table_issues for table_issues in issues.values()), issues

    def test_an_isomer_pair_stays_two_precursors(self, tmp_path):
        """One m/z isolated at two mobility positions is two precursors.

        Summing them back together would undo exactly the separation the
        mobility ramp provided, which on a targeted method is the reason
        the same mass is scheduled twice.
        """
        spatialdata = pytest.importorskip("spatialdata")
        _open("scan_sum").close()
        isomers = [
            (700.0, 1.0, 40.0, 0, 120),
            (700.0, 1.0, 45.0, 120, 240),
        ]
        out = _convert_path(
            _pasef_copy(tmp_path, isomers),
            tmp_path / "isomers.zarr",
            msms_table=True,
        )
        sdata = spatialdata.read_zarr(out)
        var = sdata.tables["tims_z0_msms"].var

        np.testing.assert_array_equal(var["precursor_mz"].unique(), [700.0])
        assert sorted(var["precursor_index"].unique()) == [0, 1]
        assert var["precursor_mobility"].nunique() == 2
        assert var.index.is_unique
        # Still a partition: nothing was dropped by keeping them apart.
        summed = _rows(sdata.tables["tims_z0"]).sum(axis=1)
        split = _rows(sdata.tables["tims_z0_msms"]).sum(axis=1)
        np.testing.assert_allclose(split, summed, rtol=1e-12)

        from thyra.metadata.schema import check_store_var_conventions

        assert all(not v for v in check_store_var_conventions(out).values())

    def test_the_store_records_how_much_current_the_split_holds(self, tmp_path):
        """Exact under scan_sum; the block is where a store says so."""
        spatialdata = pytest.importorskip("spatialdata")
        _open("scan_sum").close()
        out = _convert_path(
            _pasef_copy(tmp_path), tmp_path / "pasef.zarr", msms_table=True
        )
        block = (
            spatialdata.read_zarr(out)
            .tables["tims_z0_msms"]
            .uns["demultiplexed_current"]
        )

        assert block["summed_table"] == "tims_z0"
        assert block["current_ratio"] == pytest.approx(1.0, abs=1e-12)
        assert block["current_ratio_pixel_max"] == pytest.approx(1.0, abs=1e-12)

    def test_on_by_default_and_off_on_request(self, tmp_path):
        """The split is the default (design decision D2); the schedule is recorded either way."""
        spatialdata = pytest.importorskip("spatialdata")
        _open("scan_sum").close()
        out = _convert_path(_pasef_copy(tmp_path), tmp_path / "pasef.zarr")
        sdata = spatialdata.read_zarr(out)
        assert set(sdata.tables) == {"tims_z0", "tims_z0_msms"}
        assert sdata.tables["tims_z0"].uns["msms_schedule"]["resolved_table"] == (
            "tims_z0_msms"
        )

        out = _convert_path(
            _pasef_copy(tmp_path / "again"), tmp_path / "off.zarr", msms_table=False
        )
        sdata = spatialdata.read_zarr(out)
        assert set(sdata.tables) == {"tims_z0"}
        assert "resolved_table" not in sdata.tables["tims_z0"].uns["msms_schedule"]

    def test_a_single_precursor_acquisition_is_refused(self, tmp_path, caplog):
        """Its summed table already is the fragment spectrum of 1046.54."""
        spatialdata = pytest.importorskip("spatialdata")
        _open("scan_sum").close()
        source = tmp_path / "single.d"
        shutil.copytree(FIXTURE, source)
        with sqlite3.connect(source / "analysis.tdf") as conn:
            conn.execute("UPDATE Frames SET MsMsType = 2")
            conn.executemany(
                "INSERT INTO FrameMsMsInfo (Frame, Parent, TriggerMass, "
                "IsolationWidth, PrecursorCharge, CollisionEnergy) "
                "VALUES (?, NULL, 1046.54, 1.5, NULL, 57.327)",
                [(row[0],) for row in conn.execute("SELECT Id FROM Frames")],
            )
        with caplog.at_level(logging.INFO):
            out = _convert_path(source, tmp_path / "single.zarr", msms_table=True)
        sdata = spatialdata.read_zarr(out)

        assert set(sdata.tables) == {"tims_z0"}
        assert "single precursor" in caplog.text

    def test_an_ms1_acquisition_is_untouched(self, tmp_path):
        """Asking for the table on a survey run changes nothing at all."""
        spatialdata = pytest.importorskip("spatialdata")
        _open("scan_sum").close()
        plain = spatialdata.read_zarr(_convert(tmp_path, "scan_sum"))
        asked = spatialdata.read_zarr(
            _convert_path(FIXTURE, tmp_path / "asked.zarr", msms_table=True)
        )

        assert set(asked.tables) == set(plain.tables) == {"tims_z0"}
        np.testing.assert_array_equal(
            _rows(asked.tables["tims_z0"]), _rows(plain.tables["tims_z0"])
        )
        np.testing.assert_array_equal(
            asked.tables["tims_z0"].var["mz"].to_numpy(),
            plain.tables["tims_z0"].var["mz"].to_numpy(),
        )
        assert "msms_schedule" not in asked.tables["tims_z0"].uns


# ----------------------------------------------------------------------
# The mobility grid table on a TDF, which has no shared feature axis
# ----------------------------------------------------------------------


def _marginal(grid, summed) -> np.ndarray:
    """The grid table collapsed over mobility channels, per (pixel, m/z bin)."""
    mz_index = grid.var["mz_index"].to_numpy()
    out = np.zeros((grid.n_obs, summed.n_vars), dtype=np.float64)
    rows = _rows(grid)
    for column, bin_index in enumerate(mz_index):
        out[:, bin_index] += rows[:, column]
    return out


class TestMobilityGridStore:
    """The opt-in grid table, built by binning each frame's point cloud.

    A TDF pixel is its own point cloud, so this is the second mechanism
    that fills ``{table}_mobility`` -- and the first one a Bruker source
    can use at all. Everything here runs on the committed synthetic
    fixture through the real library; no new binary data.
    """

    def test_a_default_conversion_is_untouched(self, tmp_path):
        out = _convert(tmp_path, "vendor_centroid")
        spatialdata = pytest.importorskip("spatialdata")
        from thyra.utils.windows_paths import prepare_zarr_read_path

        sdata = spatialdata.read_zarr(prepare_zarr_read_path(out))
        assert "tims_z0_mobility" not in sdata.tables

    def test_the_flag_writes_the_sibling_under_the_default_lossless_sum(self, tmp_path):
        from thyra.convert import convert_msi

        out = tmp_path / "grid.zarr"
        # No reader_options: the default is scan_sum (design decision D1),
        # which is what makes the grid's marginal exact.
        assert convert_msi(
            str(FIXTURE),
            str(out),
            dataset_id="tims",
            pixel_size_um=20.0,
            mobility_grid=True,
        )
        from thyra.metadata.schema import read_msi_metadata_blocks
        from thyra.utils.windows_paths import prepare_zarr_read_path

        summed = _read_table(out, "tims_z0")
        grid = _read_table(out, "tims_z0_mobility")
        blocks = read_msi_metadata_blocks(prepare_zarr_read_path(out))
        step = blocks["tims_z0"]["processing"][0]
        assert step["parameters"]["tdf_spectrum"] == "scan_sum"
        assert summed.n_obs == grid.n_obs
        assert list(summed.obs.index) == list(grid.obs.index)
        assert "mobility" in grid.var.columns
        assert "mobility" not in summed.var.columns

    def test_the_marginal_reproduces_the_summed_table_per_pixel(self, tmp_path):
        from thyra.convert import convert_msi

        out = tmp_path / "grid.zarr"
        assert convert_msi(
            str(FIXTURE),
            str(out),
            dataset_id="tims",
            pixel_size_um=20.0,
            mobility_grid=True,
        )
        summed = _read_table(out, "tims_z0")
        grid = _read_table(out, "tims_z0_mobility")
        np.testing.assert_allclose(
            _marginal(grid, summed), _rows(summed), rtol=0, atol=1e-9
        )
        block = grid.uns["mobility_marginal"]
        assert float(block["current_ratio"]) == pytest.approx(1.0, abs=1e-12)
        assert float(block["max_absolute_deviation"]) == pytest.approx(0.0, abs=1e-9)

    def test_the_grid_indexes_the_heatmap_by_integer_channel(self, tmp_path):
        from thyra.convert import convert_msi

        out = tmp_path / "grid.zarr"
        assert convert_msi(
            str(FIXTURE),
            str(out),
            dataset_id="tims",
            pixel_size_um=20.0,
            mobility_grid=True,
        )
        summed = _read_table(out, "tims_z0")
        grid = _read_table(out, "tims_z0_mobility")
        block = grid.uns["mobility_grid"]
        assert int(block["n_channels"]) == 256
        # Equal arrays, not merely close.
        np.testing.assert_array_equal(
            np.asarray(block["edges"]),
            np.asarray(summed.uns["mobility_heatmap"]["mobility_edges"]),
        )
        channels = grid.var["mobility_index"].to_numpy()
        assert channels.min() >= 0 and channels.max() <= 255
        assert _no_colon_keys(block)
        # And the schema block names the same grid.
        schema_grid = summed.uns["msi_metadata"]["ms_analysis"]["ion_mobility"]["grid"]
        assert int(schema_grid["n_channels"]) == 256
        assert float(schema_grid["lower"]) == pytest.approx(float(block["lower"]))

    def test_a_tsf_file_is_refused_by_name(self, tmp_path, caplog):
        """No mobility dimension, so no grid -- and no exception either."""
        from thyra.converters.spatialdata.mobility_table import grid_refusal
        from thyra.resampling.mobility_grid import build_mobility_grid

        with _open("vendor_centroid") as reader:
            assert reader.has_ion_mobility
            assert not reader.has_shared_mobility_axis
        grid = build_mobility_grid(1.0, 1.3)

        class _Tsf:
            has_ion_mobility = False

        assert "no ion mobility" in str(grid_refusal(_Tsf(), np.arange(10.0), grid))

    def test_the_var_ceiling_refuses_the_table_and_keeps_the_store(
        self, tmp_path, caplog, monkeypatch
    ):
        """The ceiling is on occupied pairs, so it is checked on the count."""
        import thyra.converters.spatialdata.mobility_table as module
        from thyra.convert import convert_msi

        monkeypatch.setattr(module, "MAX_GRID_VAR_ENTRIES", 8)
        out = tmp_path / "ceiling.zarr"
        with caplog.at_level(logging.WARNING):
            assert convert_msi(
                str(FIXTURE),
                str(out),
                dataset_id="tims",
                pixel_size_um=20.0,
                mobility_grid=True,
            )
        assert "above the var ceiling" in caplog.text
        spatialdata = pytest.importorskip("spatialdata")
        from thyra.utils.windows_paths import prepare_zarr_read_path

        sdata = spatialdata.read_zarr(prepare_zarr_read_path(out))
        assert "tims_z0_mobility" not in sdata.tables
        # The summed table is untouched by the refusal.
        assert sdata.tables["tims_z0"].n_obs == 6


def _capped_copy(source: Path, tmp_path: Path, n_frames: int) -> Path:
    """A copy of a ``.d`` limited to its first ``n_frames`` MALDI frames.

    Everything but ``analysis.tdf`` is hard-linked, so a 27 GB acquisition
    costs nothing to cap; the database itself is copied because the frame
    list is trimmed in it. Falls back to a plain copy where hard links are
    not available.
    """
    target = tmp_path / "capped.d"
    try:
        shutil.copytree(source, target, copy_function=os.link)
        (target / "analysis.tdf").unlink()
        shutil.copy2(source / "analysis.tdf", target / "analysis.tdf")
    except OSError:
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)
    with sqlite3.connect(target / "analysis.tdf") as conn:
        conn.execute(
            "DELETE FROM MaldiFrameInfo WHERE Frame NOT IN "
            "(SELECT Frame FROM MaldiFrameInfo ORDER BY Frame LIMIT ?)",
            (int(n_frames),),
        )
    return target


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

    def test_the_grid_table_holds_the_summed_table_resolved(self, tmp_path):
        """Acceptance for the grid on real data: both tables, and they agree.

        Frame-capped so the conversion stays short: the ``.tdf_bin`` is
        hard-linked rather than copied and the frame list is trimmed, so
        the cap costs no disk and no read of the frames it drops.
        """
        from thyra.convert import convert_msi

        source = _capped_copy(Path(REAL_DATASET), tmp_path, n_frames=60)  # type: ignore[arg-type]
        out = tmp_path / "real_grid.zarr"
        assert convert_msi(
            str(source),
            str(out),
            dataset_id="real",
            # As the CLI converts: without resampling a TDF's raw axis is
            # millions of bins and the grid is refused by the var ceiling,
            # which is the ceiling doing its job rather than a bad default.
            resampling_config={
                "method": "auto",
                "axis_type": "auto",
                "reference_mz": 1000.0,
            },
            mobility_grid=True,
        )
        summed = _read_table(out, "real_z0")
        grid = _read_table(out, "real_z0_mobility")

        assert summed.n_obs == grid.n_obs
        assert list(summed.obs.index) == list(grid.obs.index)

        # var: the frozen contract, and channels on the 256-channel grid.
        mz = grid.var["mz"].to_numpy()
        mobility = grid.var["mobility"].to_numpy()
        assert np.all(np.diff(mz) >= 0)
        order = np.lexsort((mobility, mz))
        np.testing.assert_array_equal(order, np.arange(order.size))
        channels = grid.var["mobility_index"].to_numpy()
        assert channels.min() >= 0 and channels.max() <= 255

        # The heatmap's edges are the grid's, as arrays.
        np.testing.assert_array_equal(
            np.asarray(grid.uns["mobility_grid"]["edges"]),
            np.asarray(summed.uns["mobility_heatmap"]["mobility_edges"]),
        )

        # The marginal invariant, per pixel and per m/z bin. Row by row:
        # this table is millions of columns wide and densifying it whole
        # would need tens of gigabytes.
        mz_index = grid.var["mz_index"].to_numpy()
        summed_csr = summed.X.tocsr()
        grid_csr = grid.X.tocsr()
        scale = float(np.abs(summed_csr.data).max())
        for row in range(summed.n_obs):
            marginal = np.zeros(summed.n_vars, dtype=np.float64)
            block_row = grid_csr[row]
            np.add.at(marginal, mz_index[block_row.indices], block_row.data)
            np.testing.assert_allclose(
                marginal,
                np.asarray(summed_csr[row].todense()).ravel(),
                rtol=0,
                atol=scale * 1e-9,
            )
        block = grid.uns["mobility_marginal"]
        assert float(block["current_ratio"]) == pytest.approx(1.0, abs=1e-9)

        ratio = grid.X.nnz / summed.X.nnz
        assert 1.2 <= ratio <= 5, f"unexpected non-zero ratio {ratio:.2f}"
