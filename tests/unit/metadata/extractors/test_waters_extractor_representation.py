"""The Waters extractor reports what the reader delivers, on the axis it acquired.

Two facts the resampling chain acts on come from here. The spectrum
representation must be the one the open handle is set to hand back -- the
profile trace or the vendor centroid -- not the one the file was acquired
in, since a profile-acquired file read in centroid mode delivers centroids.
And the mass range must be the acquisition setting, not the span of the
stored values, so that runs acquired with the same method share one
generated axis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from thyra.metadata.extractors.waters_extractor import WatersMetadataExtractor
from thyra.readers.waters.instrument import WatersInstrument
from thyra.resampling.constants import SpectrumType
from thyra.resampling.data_characteristics import DataCharacteristics

from .test_waters_extractor import _make_grid_and_ml

MRT = WatersInstrument(
    is_mrt=True,
    decided_by="OpticMode=MRT",
    optic_mode="MRT",
    resolution=225909.053,
    flight_path_mm=48400.0,
    effective_voltage_v=7221.664,
    adc_sample_frequency_ghz=1.35,
)


def _extractor(mock_ml, handle, grid, ft, ms, **kwargs):
    return WatersMetadataExtractor(
        mock_ml, handle, Path("/test/data.raw"), grid, ft, ms, **kwargs
    )


class TestDeliveredRepresentation:
    def test_profile_file_read_as_profile(self):
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        mock_ml.is_raw_spectrum_profile.return_value = True
        essential = _extractor(
            mock_ml, handle, grid, ft, ms, use_centroid=False
        ).get_essential()
        assert essential.spectrum_type == SpectrumType.PROFILE

    def test_profile_file_read_as_centroid(self):
        """What MassLynx hands back is centroids, so that is what is declared."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        mock_ml.is_raw_spectrum_profile.return_value = True
        essential = _extractor(
            mock_ml, handle, grid, ft, ms, use_centroid=True
        ).get_essential()
        assert essential.spectrum_type == SpectrumType.CENTROID

    def test_centroid_file_has_no_trace_to_give(self, caplog):
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        mock_ml.is_raw_spectrum_profile.return_value = False
        with caplog.at_level(logging.WARNING):
            essential = _extractor(
                mock_ml, handle, grid, ft, ms, use_centroid=False
            ).get_essential()
        assert essential.spectrum_type == SpectrumType.CENTROID
        assert "no profile trace to read" in caplog.text

    def test_default_is_centroid_for_callers_that_predate_the_argument(self):
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        mock_ml.is_raw_spectrum_profile.return_value = True
        essential = _extractor(mock_ml, handle, grid, ft, ms).get_essential()
        assert essential.spectrum_type == SpectrumType.CENTROID


class TestAxisMassRange:
    def test_acquisition_range_when_it_holds_the_stored_span(self):
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_peaks=10)
        # Stored values span 100-1000 (linspace); the method was 50-1200.
        mock_ml.get_acquisition_range.return_value = (50.0, 1200.0)
        essential = _extractor(mock_ml, handle, grid, ft, ms).get_essential()
        assert essential.mass_range == (50.0, 1200.0)

    def test_stored_span_when_no_function_reports_a_range(self):
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_peaks=10)
        mock_ml.get_acquisition_range.return_value = None
        essential = _extractor(mock_ml, handle, grid, ft, ms).get_essential()
        assert essential.mass_range[0] == pytest.approx(100.0)
        assert essential.mass_range[1] == pytest.approx(1000.0)

    def test_stored_span_when_a_value_falls_outside_the_range(self, caplog):
        """A range that would drop measured data is not used."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_peaks=10)
        mock_ml.get_acquisition_range.return_value = (200.0, 1000.0)
        with caplog.at_level(logging.WARNING):
            essential = _extractor(mock_ml, handle, grid, ft, ms).get_essential()
        assert essential.mass_range[0] == pytest.approx(100.0)
        assert "outside the acquisition range" in caplog.text

    def test_union_over_ms_functions(self):
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_peaks=10)
        ft = {0: ft[0], 1: ft[0]}
        ms = [0, 1]
        mock_ml.get_acquisition_range.side_effect = lambda h, f: {
            0: (50.0, 600.0),
            1: (400.0, 1100.0),
        }[f]
        essential = _extractor(mock_ml, handle, grid, ft, ms).get_essential()
        assert essential.mass_range == (50.0, 1100.0)


class TestInstrumentReachesTheDetectorChain:
    """format_specific carries what DataCharacteristics.from_metadata reads."""

    def test_mrt_profile_stamps(self):
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        mock_ml.is_raw_spectrum_profile.return_value = True
        comp = _extractor(
            mock_ml, handle, grid, ft, ms, instrument=MRT, use_centroid=False
        ).get_comprehensive()

        specific = comp.format_specific
        assert specific["format"] == "Waters MassLynx raw"
        assert specific["is_mrt"] is True
        assert specific["instrument"] == "SELECT SERIES MRT"
        assert specific["instrument_decided_by"] == "OpticMode=MRT"
        assert specific["spectrum_source"] == "profile_trace"
        assert specific["profile_sample_spacing_da_at_1000"] == pytest.approx(
            1.143e-3, rel=0.01
        )
        # The layout facts that were always here are still here.
        assert specific["n_functions"] == 1
        assert specific["pixel_count_x"] == 3
        assert comp.instrument_info["instrument_model"] == "SELECT SERIES MRT"
        assert comp.instrument_info["declared_resolution"] == pytest.approx(225909.053)

        characteristics = DataCharacteristics.from_metadata(
            {
                "format_specific": specific,
                "instrument_info": comp.instrument_info,
                "essential_metadata": {"spectrum_type": comp.essential.spectrum_type},
            }
        )
        assert characteristics.is_waters_raw
        assert characteristics.is_waters_mrt
        assert characteristics.is_profile_data
        assert characteristics.profile_sample_spacing_da_at_1000 == pytest.approx(
            1.143e-3, rel=0.01
        )

    def test_centroid_read_stamps(self):
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        comp = _extractor(
            mock_ml, handle, grid, ft, ms, instrument=MRT, use_centroid=True
        ).get_comprehensive()
        assert comp.format_specific["spectrum_source"] == "vendor_centroid"
        assert comp.essential.spectrum_type == SpectrumType.CENTROID

    def test_without_an_instrument_nothing_is_claimed(self):
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        comp = _extractor(mock_ml, handle, grid, ft, ms).get_comprehensive()
        assert "is_mrt" not in comp.format_specific
        assert "instrument_model" not in comp.instrument_info
        characteristics = DataCharacteristics.from_metadata(
            {"format_specific": comp.format_specific}
        )
        assert characteristics.is_waters_raw
        assert not characteristics.is_waters_mrt
        assert characteristics.profile_sample_spacing_da_at_1000 is None

    def test_spacing_is_omitted_when_it_cannot_be_predicted(self):
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        bare = WatersInstrument(is_mrt=False, decided_by="Resolution=10000")
        comp = _extractor(
            mock_ml, handle, grid, ft, ms, instrument=bare
        ).get_comprehensive()
        assert comp.format_specific["profile_sample_spacing_da_at_1000"] is None
        characteristics = DataCharacteristics.from_metadata(
            {"format_specific": comp.format_specific}
        )
        assert characteristics.profile_sample_spacing_da_at_1000 is None
        assert np.isfinite(comp.essential.mass_range).all()
