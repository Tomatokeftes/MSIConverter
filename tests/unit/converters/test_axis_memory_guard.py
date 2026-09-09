"""A mass axis is refused for what it really costs, before it is built.

The only guard was the count array's 1 GiB ceiling, which sizes 4 bytes
per bin. Everything else the conversion builds per bin -- the axis, the
``var`` frame with its string index, ``total_intensity``, ``avg_spectrum``,
the column pointers, anndata's copies during the write -- comes to about
200 bytes, measured. So the ceiling let through axes that could not
possibly fit: 200M bins on a six-pixel dataset needed only 800 MB of count
array, passed, and took 217 s and 67.5 GB. 300M bins were refused, but the
process had reached 7.44 GB building the axis before anything looked at it
(issue #251).

The guard now reads the bin count the resampling plan resolved, before the
axis is materialised, and compares the projection with the machine's free
memory the way the mobility grid's own guard does.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from thyra.converters.spatialdata import base_spatialdata_converter as base_converter
from thyra.converters.spatialdata import csc_assembly
from thyra.converters.spatialdata.base_spatialdata_converter import _bin_width_range
from thyra.converters.spatialdata.csc_assembly import (
    AXIS_BYTES_PER_BIN,
    AXIS_REFUSE_FRACTION,
    AXIS_WARN_FRACTION,
    mass_axis_refusal,
    projected_axis_gb,
)
from thyra.converters.spatialdata.streaming_converter import (
    StreamingSpatialDataConverter,
)
from thyra.errors import ConversionRefused


class _Meta:
    def __init__(self):
        self.dimensions = (2, 3, 1)
        self.mass_range = (100.0, 1000.0)
        self.n_spectra = 6
        self.pixel_size = (20.0, 20.0)
        self.estimated_memory_gb = 1.0
        self.coordinate_bounds = (0, 2, 0, 3)
        self.is_3d = False
        self.has_mass_axis = True
        self.source_path = "mock"
        self.total_peaks = 60
        self.coordinate_offsets = (0, 0, 0)
        self.spectrum_type = "centroid spectrum"
        self.peak_counts_per_pixel = None
        self.z_spacing_um = None


class _Reader:
    """Just enough reader to reach ``_setup_mass_axis``."""

    def __init__(self):
        self._meta = _Meta()
        self.axis_built = False

    def get_essential_metadata(self):
        return self._meta

    def get_common_mass_axis(self):
        self.axis_built = True
        return np.linspace(100.0, 1000.0, 1000)

    @property
    def has_shared_mass_axis(self):
        return True

    def close(self):
        pass


class TestTheProjection:
    def test_it_is_the_measured_cost_per_bin(self):
        assert projected_axis_gb(1_000_000) == pytest.approx(
            1_000_000 * AXIS_BYTES_PER_BIN / 1024**3
        )

    def test_the_count_array_alone_would_have_let_it_through(self):
        """The number the old ceiling saw, against the number it costs."""
        n_bins = 200_000_000
        assert csc_assembly.count_refusal(n_bins) is None
        assert projected_axis_gb(n_bins) > 30.0

    def test_an_axis_that_fits_is_not_refused(self):
        assert mass_axis_refusal(1_000_000, available_gb=32.0) is None

    def test_an_axis_that_does_not_fit_names_the_levers(self):
        refusal = mass_axis_refusal(200_000_000, available_gb=32.0)

        assert refusal is not None
        assert "200,000,000 bins" in refusal
        assert "--resample-bins" in refusal
        assert f"{AXIS_BYTES_PER_BIN} bytes per bin" in refusal

    def test_the_boundary_is_the_refuse_fraction(self):
        free = 8.0
        bins = int(free * AXIS_REFUSE_FRACTION * 1024**3 / AXIS_BYTES_PER_BIN)
        assert mass_axis_refusal(bins, available_gb=free) is None
        assert mass_axis_refusal(bins + 1, available_gb=free) is not None

    def test_the_band_below_it_warns(self, thyra_logs):
        """``thyra_logs`` rather than ``caplog``: see ``tests/conftest.py``."""
        free = 8.0
        bins = int(free * AXIS_WARN_FRACTION * 1024**3 / AXIS_BYTES_PER_BIN) + 1

        with thyra_logs("thyra.converters.spatialdata.csc_assembly") as records:
            assert mass_axis_refusal(bins, available_gb=free) is None

        assert any("projected to need" in r.getMessage() for r in records)


class TestTheConverterRefusesBeforeBuilding:
    def _converter(self, reader, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            return StreamingSpatialDataConverter(
                reader=reader,
                output_path=Path(tmpdir) / "out.zarr",
                dataset_id="m",
                pixel_size_um=10.0,
                **kwargs,
            )

    def test_a_resampled_axis_is_refused_unbuilt(self, monkeypatch):
        """The point of the guard: nothing wide is allocated first."""
        monkeypatch.setattr(csc_assembly, "available_memory_gb", lambda: 1.0)
        reader = _Reader()
        converter = self._converter(
            reader,
            resampling_config={
                "method": "nearest_neighbor",
                "target_bins": 200_000_000,
            },
        )

        with pytest.raises(ConversionRefused, match=r"200,000,000 bins"):
            converter._setup_mass_axis()

        assert converter._common_mass_axis is None

    def test_a_raw_axis_is_refused_too(self, monkeypatch):
        """It is already built, but every per-bin structure is still ahead."""
        monkeypatch.setattr(csc_assembly, "available_memory_gb", lambda: 1e-6)
        reader = _Reader()
        converter = self._converter(reader)

        with pytest.raises(ConversionRefused, match=r"1,000 bins"):
            converter._setup_mass_axis()

        assert reader.axis_built

    def test_an_axis_that_fits_is_built(self, monkeypatch):
        monkeypatch.setattr(csc_assembly, "available_memory_gb", lambda: 32.0)
        converter = self._converter(_Reader())

        converter._setup_mass_axis()

        assert converter._common_mass_axis is not None
        assert len(converter._common_mass_axis) == 1000


class TestBinWidthsAreNotMaterialised:
    """``np.diff`` over the whole axis is 1.6 GB on a 200M-bin one, for a log line."""

    @pytest.mark.parametrize("n", [2, 3, 1000, 100_000])
    def test_it_is_the_min_and_max_of_the_diff(self, n):
        rng = np.random.default_rng(4)
        axis = np.sort(rng.random(n) * 900.0 + 100.0)
        widths = np.diff(axis)

        low, high = _bin_width_range(axis)

        assert low == pytest.approx(float(widths.min()))
        assert high == pytest.approx(float(widths.max()))

    @pytest.mark.parametrize("gap_at", [1, 99, 100, 101, 199, 998])
    def test_a_gap_at_a_chunk_edge_is_still_seen(self, monkeypatch, gap_at):
        """Chunks overlap by one entry; a gap between two must not be lost."""
        monkeypatch.setattr(base_converter, "BIN_WIDTH_CHUNK", 100)
        axis = np.arange(1000, dtype=np.float64)
        axis[gap_at:] += 5.0  # one wide gap, placed on and around an edge

        low, high = _bin_width_range(axis)

        assert low == pytest.approx(1.0)
        assert high == pytest.approx(6.0)

    def test_a_degenerate_axis_is_zero(self):
        assert _bin_width_range(np.array([500.0])) == (0.0, 0.0)
        assert _bin_width_range(np.array([])) == (0.0, 0.0)
