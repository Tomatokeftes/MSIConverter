"""Which Waters instrument wrote a run, and what the reader does about it.

The three cases the detection has to get right are the ones that exist on
disk: a SELECT SERIES MRT run (``OpticMode = MRT``, ``Resolution`` ~225,000,
``$$ Instrument: MRT#``), a Synapt G2-Si run (no ``OpticMode``, no
instrument line, ``Resolution`` 10,000, tab-run key padding), and a directory
with neither side file at all. The field values and formatting below are
copied from those two runs.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from thyra.readers.waters.imaging_grid import ImagingGrid
from thyra.readers.waters.instrument import (
    MRT_RESOLUTION_THRESHOLD,
    WatersInstrument,
    identify_waters_instrument,
    parse_extern_inf,
    parse_header_txt,
)
from thyra.readers.waters.masslynx_lib import FunctionType, ScanInfoData
from thyra.readers.waters.waters_reader import WatersReader

# The MRT pads keys with spaces then one tab; the Synapt with runs of tabs.
MRT_EXTERN = (
    "Instrument Configuration:\n"
    "Lteff                    \t48400.0\n"
    "Veff                    \t7221.664\n"
    "Resolution              \t225909.053\n"
    "Acquisition Device      \tWatersADC\n"
    "ADC Sample Frequency (GHz)\t1.35\n"
    "\n"
    "Function Parameters - Function 1 - TOF MS FUNCTION\n"
    "ADC Sample Frequency (GHz)                   \t1.35\n"
    "ADC Pusher Period (us)                       \t187.0\n"
    "OpticMode                                    \tMRT\n"
    "Polarity                                     \tPositive\n"
)
MRT_HEADER = (
    "$$ Version: 01.00\n"
    "$$ Acquired Name: 20260821_MRT_FTW2\n"
    "$$ Instrument: MRT#\n"
    "$$ Cal Function 1: 0.000000000000000E+00,1.000000000000000E+00,T1\n"
)
SYNAPT_EXTERN = (
    "Parameters for E:\\Britt.PRO\\Acqudb\\synapt on MALDI MS.exp\n"
    "Created by 4.1 SCN957\n"
    " \n"
    "Instrument Configuration:\n"
    "Lteff\t\t\t\t\t\t1800.0\n"
    "Veff\t\t\t\t\t\t7200.50\n"
    "Resolution\t\t\t\t\t10000\n"
    "Min Points in Peak\t\t\t\t2\n"
    "Acquisition Device\t\t\t\tWatersADC\n"
    "ADC Sample Frequency (GHz)\t\t\t3.0\n"
    "Pusher Cycle Time (\xb5s)\t\t\t\tAutomatic\n"
    "PusherInterval\t\t\t\t\t54.000000\n"
)
SYNAPT_HEADER = (
    "$$ Version: 01.00\n"
    "$$ Acquired Name: 20201216_MDA468_slide20200630_synapt\n"
    "$$ Job Code: 20201216_MDA468_slide20200630_synapt\n"
)


def _raw_dir(tmp_path, extern=None, header=None):
    raw = tmp_path / "run.raw"
    raw.mkdir(parents=True)
    (raw / "_FUNC001.DAT").write_bytes(b"\x00" * 64)
    if extern is not None:
        (raw / "_extern.inf").write_text(extern, encoding="latin-1")
    if header is not None:
        (raw / "_header.txt").write_text(header, encoding="latin-1")
    return raw


class TestParsing:
    def test_mrt_formatting(self):
        fields = parse_extern_inf(MRT_EXTERN)
        assert fields["OpticMode"] == "MRT"
        assert fields["Resolution"] == "225909.053"
        assert fields["ADC Pusher Period (us)"] == "187.0"

    def test_synapt_tab_runs_and_latin1(self):
        fields = parse_extern_inf(SYNAPT_EXTERN)
        assert fields["Resolution"] == "10000"
        assert fields["Lteff"] == "1800.0"
        assert fields["Pusher Cycle Time (\xb5s)"] == "Automatic"
        assert "OpticMode" not in fields

    def test_first_occurrence_wins(self):
        fields = parse_extern_inf("K\t1\nK\t2\n")
        assert fields["K"] == "1"

    def test_section_headers_are_skipped(self):
        assert "Instrument Configuration:" not in parse_extern_inf(MRT_EXTERN)

    def test_header(self):
        fields = parse_header_txt(MRT_HEADER)
        assert fields["Instrument"] == "MRT#"
        assert fields["Acquired Name"] == "20260821_MRT_FTW2"
        assert "Instrument" not in parse_header_txt(SYNAPT_HEADER)


class TestDecision:
    def test_mrt_run(self, tmp_path):
        inst = identify_waters_instrument(_raw_dir(tmp_path, MRT_EXTERN, MRT_HEADER))
        assert inst.is_mrt
        assert inst.decided_by == "OpticMode=MRT"
        assert inst.name == "SELECT SERIES MRT"
        assert inst.resolution == pytest.approx(225909.053)

    def test_synapt_run(self, tmp_path):
        inst = identify_waters_instrument(
            _raw_dir(tmp_path, SYNAPT_EXTERN, SYNAPT_HEADER)
        )
        assert not inst.is_mrt
        assert inst.decided_by == "Resolution=10000"
        assert inst.optic_mode is None
        assert inst.instrument_label is None

    def test_neither_field(self, tmp_path):
        inst = identify_waters_instrument(_raw_dir(tmp_path))
        assert not inst.is_mrt
        assert inst.decided_by == "no instrument field found"
        assert inst.resolution is None

    def test_optic_mode_outranks_resolution(self, tmp_path):
        # OpticMode says a non-MRT optic even though the resolution is high.
        extern = MRT_EXTERN.replace("\tMRT\n", "\tV\n")
        inst = identify_waters_instrument(_raw_dir(tmp_path, extern))
        assert not inst.is_mrt
        assert inst.decided_by == "OpticMode=V"

    def test_header_label_when_optic_mode_is_absent(self, tmp_path):
        inst = identify_waters_instrument(_raw_dir(tmp_path, header=MRT_HEADER))
        assert inst.is_mrt
        assert inst.decided_by == "$$ Instrument=MRT#"

    def test_resolution_fallback(self, tmp_path):
        extern = "Resolution\t225909.053\n"
        inst = identify_waters_instrument(_raw_dir(tmp_path, extern))
        assert inst.is_mrt
        assert inst.decided_by == "Resolution=225909"
        assert MRT_RESOLUTION_THRESHOLD == 100_000.0

    def test_unparseable_resolution_is_not_a_decision(self, tmp_path):
        inst = identify_waters_instrument(_raw_dir(tmp_path, "Resolution\tn/a\n"))
        assert not inst.is_mrt
        assert inst.decided_by == "no instrument field found"

    def test_decision_is_logged(self, tmp_path, caplog):
        with caplog.at_level(logging.INFO, logger="thyra.readers.waters.instrument"):
            identify_waters_instrument(_raw_dir(tmp_path, MRT_EXTERN, MRT_HEADER))
        assert "SELECT SERIES MRT (OpticMode=MRT)" in caplog.text


class TestSampleSpacing:
    """Predicted from Lteff, Veff and the ADC clock; measured values beside."""

    def test_mrt(self, tmp_path):
        inst = identify_waters_instrument(_raw_dir(tmp_path, MRT_EXTERN))
        # 1.16 mDa measured at m/z 990-1000, 1.04 at 795-805.
        assert inst.profile_sample_spacing_da(1000.0) == pytest.approx(
            1.143e-3, rel=0.01
        )
        assert inst.profile_sample_spacing_da(800.0) == pytest.approx(
            1.022e-3, rel=0.01
        )

    def test_synapt(self, tmp_path):
        inst = identify_waters_instrument(_raw_dir(tmp_path, SYNAPT_EXTERN))
        # 13.4 mDa measured at m/z 900-1000.
        assert inst.profile_sample_spacing_da(1000.0) == pytest.approx(
            13.8e-3, rel=0.01
        )

    def test_sqrt_law(self, tmp_path):
        inst = identify_waters_instrument(_raw_dir(tmp_path, MRT_EXTERN))
        ratio = inst.profile_sample_spacing_da(400.0) / inst.profile_sample_spacing_da(
            1600.0
        )
        assert ratio == pytest.approx(0.5)

    def test_missing_constants(self, tmp_path):
        inst = identify_waters_instrument(_raw_dir(tmp_path))
        assert inst.profile_sample_spacing_da(1000.0) is None
        assert WatersInstrument(is_mrt=False, decided_by="x", flight_path_mm=1.0)
        assert (
            WatersInstrument(
                is_mrt=False,
                decided_by="x",
                flight_path_mm=0.0,
                effective_voltage_v=1.0,
                adc_sample_frequency_ghz=1.0,
            ).profile_sample_spacing_da(500.0)
            is None
        )


def _grid_one_pixel():
    scan_map = {
        (0, 0): ScanInfoData(
            ms_level=1,
            polarity=0,
            drift_scan_count=0,
            is_profile=1,
            precursor_mz=0.0,
            rt=1.0,
            laser_x_pos=0.1,
            laser_y_pos=0.1,
        )
    }
    return ImagingGrid(
        x_index_map={100.0: 0},
        y_index_map={100.0: 0},
        pixel_count_x=1,
        pixel_count_y=1,
        pixel_size_x=0.0,
        pixel_size_y=0.0,
        lateral_width=0.0,
        lateral_height=0.0,
        scan_map=scan_map,
    )


class TestReaderDefault:
    """``use_centroid=None`` follows the instrument; an explicit value wins."""

    def test_mrt_defaults_to_profile(self, tmp_path):
        reader = WatersReader(_raw_dir(tmp_path, MRT_EXTERN, MRT_HEADER))
        assert reader.use_centroid is False
        assert reader.instrument.is_mrt

    def test_synapt_defaults_to_centroid(self, tmp_path):
        reader = WatersReader(_raw_dir(tmp_path, SYNAPT_EXTERN, SYNAPT_HEADER))
        assert reader.use_centroid is True

    def test_unknown_defaults_to_centroid(self, tmp_path):
        reader = WatersReader(_raw_dir(tmp_path))
        assert reader.use_centroid is True
        assert reader.instrument.decided_by == "no instrument field found"

    @pytest.mark.parametrize("explicit", [True, False])
    def test_explicit_choice_overrides_either_way(self, tmp_path, explicit):
        mrt = WatersReader(
            _raw_dir(tmp_path / "a", MRT_EXTERN, MRT_HEADER), use_centroid=explicit
        )
        synapt = WatersReader(
            _raw_dir(tmp_path / "b", SYNAPT_EXTERN, SYNAPT_HEADER),
            use_centroid=explicit,
        )
        assert mrt.use_centroid is explicit
        assert synapt.use_centroid is explicit

    def test_default_is_logged_with_its_reason(self, tmp_path, caplog):
        with caplog.at_level(logging.INFO, logger="thyra.readers.waters.waters_reader"):
            WatersReader(_raw_dir(tmp_path, MRT_EXTERN, MRT_HEADER))
        assert "profile trace by default (OpticMode=MRT" in caplog.text

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_mrt_default_reaches_the_native_mode_switch(
        self, mock_build_grid, mock_ml_cls, tmp_path
    ):
        """The decision is only real once ``set_centroid`` sees it."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.open_file.return_value = "handle"
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS
        mock_build_grid.return_value = _grid_one_pixel()

        reader = WatersReader(_raw_dir(tmp_path, MRT_EXTERN, MRT_HEADER))
        reader._ensure_initialized()
        mock_ml.set_centroid.assert_called_once_with("handle", False)

        extractor = reader._create_metadata_extractor()
        assert extractor._use_centroid is False
        assert extractor._instrument is reader.instrument
