"""
Tests for the Rapiflex reader.
"""

import struct

import numpy as np
import pytest

from thyra.readers.bruker.rapiflex import RapiflexReader


@pytest.fixture
def create_mock_rapiflex_data(tmp_path):
    """Create mock Rapiflex data files for testing.

    Creates:
    - sample.dat: Binary file with header, offset table, and spectral data
    - sample_info.txt: Metadata file
    - sample_poslog.txt: Position log file
    - sample.mis: Optional XML method file
    """
    folder = tmp_path / "mock_rapiflex"
    folder.mkdir()

    # Parameters
    n_spots = 9  # 3x3 grid
    n_datapoints = 100
    mass_start = 100.0
    mass_end = 500.0
    raster_x = 20
    raster_y = 20

    # Create position log
    poslog_path = folder / "sample_poslog.txt"
    positions = []
    with open(poslog_path, "w") as f:
        f.write("#Timestamp Pos X Y Z\n")
        idx = 0
        for y in range(3):
            for x in range(3):
                pos_id = f"R00X{x}Y{y}"
                phys_x = 1000.0 + x * raster_x
                phys_y = 2000.0 + y * raster_y
                f.write(f"2023-01-01 12:00:00.000 {pos_id} {phys_x} {phys_y} 0.0\n")
                positions.append((x, y, phys_x, phys_y))
                idx += 1

    # Create info file
    info_path = folder / "sample_info.txt"
    with open(info_path, "w") as f:
        f.write("Rapiflex Info File\n")
        f.write("Name of Sample: Mock Sample\n")
        f.write(f"Number of Spots: {n_spots}\n")
        f.write("Number of Shots: 100\n")
        f.write("Spectrum Size: 100\n")
        f.write("Detector Gain: 2.0\n")
        f.write(f"Mass Start: {mass_start}\n")
        f.write(f"Mass End: {mass_end}\n")
        f.write("Acquisition Mode: REFLECTOR\n")
        f.write("Instrument Serial Number: TEST123\n")
        f.write("Laser Power: 50\n")
        f.write("Sample Rate: 1.0\n")
        f.write(f"DataPoints: {n_datapoints}\n")
        f.write("Method: TestMethod.par\n")
        f.write("flexImaging Version: 5.0.0\n")
        f.write("flexControl Version: 4.0.0\n")
        f.write(f"Raster: {raster_x},{raster_y}\n")
        f.write("Start Time: Mon, 01.01.2023 12:00:00\n")
        f.write("End Time: Mon, 01.01.2023 12:30:00\n")

    # Create .mis file (optional but good to test)
    mis_path = folder / "sample.mis"
    with open(mis_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<ImagingSequence flexImagingVersion="5.0.0">\n')
        f.write("  <Method>TestMethod.par</Method>\n")
        f.write("  <ImageFile>sample.tif</ImageFile>\n")
        f.write(f"  <Raster>{raster_x},{raster_y}</Raster>\n")
        f.write("  <TeachPoint>100,200;1000,2000</TeachPoint>\n")
        f.write("  <TeachPoint>300,200;1040,2000</TeachPoint>\n")
        f.write("  <TeachPoint>100,400;1000,2040</TeachPoint>\n")
        f.write("</ImagingSequence>\n")

    # Create .dat file
    dat_path = folder / "sample.dat"

    # Header (48 bytes)
    header_size = 48
    first_raster_x = 0
    first_raster_y = 0
    raster_width = 3
    raster_height = 3

    # Calculate offsets
    offset_table_start = header_size
    data_start = offset_table_start + n_spots * 4

    # Generate mock spectra
    spectra = []
    for i in range(n_spots):
        # Create spectrum with some peaks
        spectrum = np.zeros(n_datapoints, dtype=np.float32)
        # Add some peaks at different positions for each spectrum
        peak_pos = [20 + i * 5, 50, 80 - i * 3]
        for pos in peak_pos:
            if 0 <= pos < n_datapoints:
                spectrum[pos] = 100.0 + i * 10
                if pos > 0:
                    spectrum[pos - 1] = 50.0 + i * 5
                if pos < n_datapoints - 1:
                    spectrum[pos + 1] = 50.0 + i * 5
        spectra.append(spectrum)

    with open(dat_path, "wb") as f:
        # Write header
        header = struct.pack(
            "<12I",
            header_size,  # header size
            256,  # unknown
            first_raster_x,
            first_raster_y,
            raster_width,
            raster_height,
            n_datapoints,
            0,
            0,
            0,
            0,
            0,  # padding
        )
        f.write(header)

        # Write offset table
        current_offset = data_start
        for i in range(n_spots):
            f.write(struct.pack("<I", current_offset))
            current_offset += n_datapoints * 4

        # Write spectral data
        for spectrum in spectra:
            f.write(spectrum.tobytes())

    return folder, n_spots, n_datapoints, mass_start, mass_end, spectra


class TestRapiflexReader:
    """Test the Rapiflex reader functionality."""

    def test_initialization(self, create_mock_rapiflex_data):
        """Test initializing the reader with valid files."""
        folder, n_spots, n_datapoints, mass_start, mass_end, _ = (
            create_mock_rapiflex_data
        )

        reader = RapiflexReader(folder)

        assert reader.n_spectra == n_spots
        assert reader.n_datapoints == n_datapoints
        assert reader.mass_range == (mass_start, mass_end)

        reader.close()

    def test_initialization_with_string_path(self, create_mock_rapiflex_data):
        """Test initializing with string path."""
        folder, _, _, _, _, _ = create_mock_rapiflex_data

        reader = RapiflexReader(str(folder))
        assert reader.n_spectra > 0
        reader.close()

    def test_missing_dat_file(self, tmp_path):
        """Test error when .dat file is missing."""
        folder = tmp_path / "missing_dat"
        folder.mkdir()

        # Create only info and poslog files
        (folder / "sample_info.txt").write_text("Rapiflex Info File\n")
        (folder / "sample_poslog.txt").write_text("#Timestamp Pos X Y Z\n")

        with pytest.raises(ValueError, match="No .dat file found"):
            RapiflexReader(folder)

    def test_missing_poslog_file(self, tmp_path):
        """Test error when _poslog.txt file is missing."""
        folder = tmp_path / "missing_poslog"
        folder.mkdir()

        # Create only dat and info files
        (folder / "sample.dat").write_bytes(b"\x00" * 100)
        (folder / "sample_info.txt").write_text("Rapiflex Info File\n")

        with pytest.raises(ValueError, match="No .*_poslog.txt file found"):
            RapiflexReader(folder)

    def test_get_common_mass_axis(self, create_mock_rapiflex_data):
        """Test getting common mass axis."""
        folder, _, n_datapoints, mass_start, mass_end, _ = create_mock_rapiflex_data

        reader = RapiflexReader(folder)
        mz_axis = reader.get_common_mass_axis()

        assert len(mz_axis) == n_datapoints
        assert mz_axis[0] == pytest.approx(mass_start)
        assert mz_axis[-1] == pytest.approx(mass_end)

        # Check linear spacing
        expected_step = (mass_end - mass_start) / (n_datapoints - 1)
        actual_step = mz_axis[1] - mz_axis[0]
        assert actual_step == pytest.approx(expected_step)

        reader.close()

    def test_iter_spectra(self, create_mock_rapiflex_data):
        """Test iterating through spectra."""
        folder, n_spots, n_datapoints, _, _, expected_spectra = (
            create_mock_rapiflex_data
        )

        reader = RapiflexReader(folder)

        count = 0
        for coords, mz, intensities in reader.iter_spectra():
            assert len(coords) == 3  # (x, y, z)
            assert len(mz) == n_datapoints
            assert len(intensities) == n_datapoints

            # Check that intensities match expected
            expected = expected_spectra[count].astype(np.float64)
            np.testing.assert_array_almost_equal(intensities, expected)

            count += 1

        assert count == n_spots
        reader.close()

    def test_iter_spectra_coordinates(self, create_mock_rapiflex_data):
        """Test that coordinates are correctly parsed."""
        folder, _, _, _, _, _ = create_mock_rapiflex_data

        reader = RapiflexReader(folder)

        coords_list = []
        for coords, _, _ in reader.iter_spectra():
            coords_list.append(coords)

        # Should have 9 spectra in a 3x3 grid
        assert len(coords_list) == 9

        # Coordinates should be 0-based
        xs = [c[0] for c in coords_list]
        ys = [c[1] for c in coords_list]

        assert min(xs) == 0
        assert max(xs) == 2
        assert min(ys) == 0
        assert max(ys) == 2

        reader.close()

    def test_metadata_extraction(self, create_mock_rapiflex_data):
        """Test metadata extraction."""
        folder, _, _, _, _, _ = create_mock_rapiflex_data

        reader = RapiflexReader(folder)

        # Check info metadata
        info = reader.info_metadata
        assert info["Name of Sample"] == "Mock Sample"
        assert info["Acquisition Mode"] == "REFLECTOR"
        assert info["raster_x"] == 20.0
        assert info["raster_y"] == 20.0

        # Check mis metadata
        mis = reader.mis_metadata
        assert "teaching_points" in mis
        assert len(mis["teaching_points"]) == 3

        reader.close()

    def test_essential_metadata(self, create_mock_rapiflex_data):
        """Test essential metadata through extractor."""
        folder, n_spots, n_datapoints, mass_start, mass_end, _ = (
            create_mock_rapiflex_data
        )

        reader = RapiflexReader(folder)
        essential = reader.get_essential_metadata()

        # Check EssentialMetadata attributes
        assert essential.n_spectra == n_spots
        assert essential.mass_range == (mass_start, mass_end)
        assert essential.dimensions[0] == 3  # 3x3 grid
        assert essential.dimensions[1] == 3
        assert essential.source_path == str(folder)

        reader.close()

    def test_context_manager(self, create_mock_rapiflex_data):
        """Test reader as context manager."""
        folder, _, _, _, _, _ = create_mock_rapiflex_data

        with RapiflexReader(folder) as reader:
            assert reader.n_spectra > 0
            mz = reader.get_common_mass_axis()
            assert len(mz) > 0

        # After context exit, reader should be closed
        assert reader._closed

    def test_close_idempotent(self, create_mock_rapiflex_data):
        """Test that close() can be called multiple times."""
        folder, _, _, _, _, _ = create_mock_rapiflex_data

        reader = RapiflexReader(folder)
        reader.close()
        reader.close()  # Should not raise

    def test_repr(self, create_mock_rapiflex_data):
        """Test string representation."""
        folder, n_spots, n_datapoints, _, _, _ = create_mock_rapiflex_data

        reader = RapiflexReader(folder)
        repr_str = repr(reader)

        assert "RapiflexReader" in repr_str
        assert str(n_spots) in repr_str
        assert str(n_datapoints) in repr_str

        reader.close()

    def test_intensity_threshold_filtering(self, create_mock_rapiflex_data):
        """Test that intensity_threshold filters low values during iteration."""
        folder, n_spots, n_datapoints, _, _, expected_spectra = (
            create_mock_rapiflex_data
        )

        # Calculate max intensity to set a reasonable threshold
        max_intensity = max(s.max() for s in expected_spectra)
        threshold = max_intensity * 0.5  # Filter out bottom 50%

        # First, read without threshold to get baseline
        reader_no_thresh = RapiflexReader(folder)
        total_values_no_thresh = 0
        for _, _, intensities in reader_no_thresh.iter_spectra():
            total_values_no_thresh += len(intensities)
        reader_no_thresh.close()

        # Now read with threshold
        reader_with_thresh = RapiflexReader(folder, intensity_threshold=threshold)
        total_values_with_thresh = 0
        for _, _, intensities in reader_with_thresh.iter_spectra():
            # All returned intensities should be >= threshold
            assert np.all(intensities >= threshold), (
                f"Found intensities below threshold: "
                f"{intensities[intensities < threshold]}"
            )
            total_values_with_thresh += len(intensities)
        reader_with_thresh.close()

        # With threshold, we should have fewer values (continuous mode = same m/z count)
        # Note: Rapiflex is continuous mode, so filtering reduces values per spectrum
        assert total_values_with_thresh < total_values_no_thresh, (
            f"Expected fewer values with threshold. "
            f"No threshold: {total_values_no_thresh}, "
            f"With threshold: {total_values_with_thresh}"
        )

    def test_intensity_threshold_none_returns_all(self, create_mock_rapiflex_data):
        """Test that intensity_threshold=None returns all values."""
        folder, n_spots, n_datapoints, _, _, _ = create_mock_rapiflex_data

        # Read with explicit None threshold
        reader = RapiflexReader(folder, intensity_threshold=None)

        count = 0
        for _, mz, intensities in reader.iter_spectra():
            # Should return all n_datapoints values per spectrum
            assert len(mz) == n_datapoints
            assert len(intensities) == n_datapoints
            count += 1

        assert count == n_spots, f"Expected {n_spots} spectra, got {count}"
        reader.close()


class TestRapiflexFormatDetection:
    """Test format detection for Rapiflex data."""

    def test_detect_rapiflex_format(self, create_mock_rapiflex_data):
        """Test that Rapiflex format is correctly detected."""
        from thyra.core.registry import detect_format

        folder, _, _, _, _, _ = create_mock_rapiflex_data
        format_name = detect_format(folder)

        assert format_name == "rapiflex"

    def test_extract_areas_rectangular(self, create_mock_rapiflex_data):
        """Test area extraction with rectangular (Type=0) 2-point areas."""
        folder, _, _, _, _, _ = create_mock_rapiflex_data
        reader = RapiflexReader(folder)

        import xml.etree.ElementTree as ET

        xml_str = """<Root>
            <Area Type="0" Name="01">
                <Point>9768,6372</Point>
                <Point>10672,6676</Point>
            </Area>
        </Root>"""
        root = ET.fromstring(xml_str)
        metadata = {}
        reader._extract_areas(root, metadata)

        assert len(metadata["areas"]) == 1
        area = metadata["areas"][0]
        assert area["name"] == "01"
        assert area["p1"] == [9768, 6372]
        assert area["p2"] == [10672, 6676]
        reader.close()

    def test_extract_areas_polygon(self, create_mock_rapiflex_data):
        """Test area extraction with polygon (Type=3) N-point areas."""
        folder, _, _, _, _, _ = create_mock_rapiflex_data
        reader = RapiflexReader(folder)

        import xml.etree.ElementTree as ET

        xml_str = """<Root>
            <Area Type="3" Name="01">
                <Point>24470,4585</Point>
                <Point>24420,3818</Point>
                <Point>24862,3543</Point>
                <Point>25353,3168</Point>
                <Point>26228,3043</Point>
                <Point>26753,4193</Point>
                <Point>26462,5485</Point>
                <Point>25362,5777</Point>
                <Point>24737,4943</Point>
                <Point>24487,4552</Point>
            </Area>
        </Root>"""
        root = ET.fromstring(xml_str)
        metadata = {}
        reader._extract_areas(root, metadata)

        assert len(metadata["areas"]) == 1
        area = metadata["areas"][0]
        assert area["name"] == "01"
        # Bounding box should span ALL points, not just first two
        assert area["p1"] == [24420, 3043]
        assert area["p2"] == [26753, 5777]
        reader.close()

    def test_not_rapiflex_without_dat(self, tmp_path):
        """Test that folder without .dat is not detected as Rapiflex."""
        from thyra.core.registry import detect_format

        folder = tmp_path / "not_rapiflex"
        folder.mkdir()

        # Create some files but not .dat
        (folder / "sample_info.txt").write_text("info")
        (folder / "sample_poslog.txt").write_text("poslog")

        with pytest.raises(ValueError):
            detect_format(folder)


class TestMissingRasterDeclaration:
    """A pitch nobody declared must not be invented (issue #236).

    The reader used to fall back to 20 um, which convert.py then recorded
    as ``pixel_size_source: automatic`` with ``detection_successful: True``
    while ``format_specific.raster_step_um`` sat empty -- a guess the store
    called a measurement. Every other reader returns None here and the CLI
    refuses with the actionable "use --pixel-size" hint.
    """

    @staticmethod
    def _strip_raster(folder):
        """Remove every declaration of the raster step from the acquisition."""
        info_path = folder / "sample_info.txt"
        kept = [
            line
            for line in info_path.read_text().splitlines(keepends=True)
            if not line.startswith("Raster:")
        ]
        info_path.write_text("".join(kept))
        (folder / "sample.mis").unlink()

    def test_no_raster_declaration_means_no_pixel_size(self, create_mock_rapiflex_data):
        folder, _, _, _, _, _ = create_mock_rapiflex_data
        self._strip_raster(folder)

        reader = RapiflexReader(folder)
        essential = reader._create_metadata_extractor().get_essential()

        assert essential.pixel_size is None
        reader.close()

    def test_a_declared_raster_is_still_read(self, create_mock_rapiflex_data):
        folder, _, _, _, _, _ = create_mock_rapiflex_data

        reader = RapiflexReader(folder)
        essential = reader._create_metadata_extractor().get_essential()

        assert essential.pixel_size == (20.0, 20.0)
        reader.close()

    def test_the_missing_declaration_is_reported(
        self, create_mock_rapiflex_data, caplog
    ):
        import logging

        folder, _, _, _, _, _ = create_mock_rapiflex_data
        self._strip_raster(folder)

        reader = RapiflexReader(folder)
        with caplog.at_level(logging.WARNING):
            reader._create_metadata_extractor().get_essential()

        assert "declares no raster step" in caplog.text
        assert "--pixel-size" in caplog.text
        reader.close()


class TestRasterOrigin:
    """Where on the slide the raster starts (issue #237).

    The .dat header carries it and it was discarded for a hard-coded
    (0, 0, 0), so no Rapiflex store could be placed against its optical
    image the way a solariX one can -- that reader records
    ``coordinate_offsets_px [516, 520, 0]`` for a section whose raster
    starts at stage (516, 520).
    """

    @staticmethod
    def _set_first_raster(folder, first_x, first_y):
        """Rewrite the header's first_raster_x/y, at byte offsets 8 and 12."""
        dat_path = folder / "sample.dat"
        raw = bytearray(dat_path.read_bytes())
        raw[8:16] = struct.pack("<2I", first_x, first_y)
        dat_path.write_bytes(bytes(raw))

    def test_the_raster_origin_reaches_the_metadata(self, create_mock_rapiflex_data):
        folder, _, _, _, _, _ = create_mock_rapiflex_data
        self._set_first_raster(folder, 100, 50)

        reader = RapiflexReader(folder)
        essential = reader._create_metadata_extractor().get_essential()

        assert essential.coordinate_offsets == (100, 50, 0)
        reader.close()

    def test_a_raster_starting_at_the_origin_still_reads_zero(
        self, create_mock_rapiflex_data
    ):
        folder, _, _, _, _, _ = create_mock_rapiflex_data

        reader = RapiflexReader(folder)
        essential = reader._create_metadata_extractor().get_essential()

        assert essential.coordinate_offsets == (0, 0, 0)
        reader.close()

    def test_the_pixel_coordinates_stay_zero_based(self, create_mock_rapiflex_data):
        """The offset is metadata; the stored coordinates are normalised."""
        folder, _, _, _, _, _ = create_mock_rapiflex_data
        self._set_first_raster(folder, 100, 50)

        reader = RapiflexReader(folder)
        coords = [c for c, _, _ in reader.iter_spectra()]

        assert min(x for x, _, _ in coords) == 0
        assert min(y for _, y, _ in coords) == 0
        reader.close()
