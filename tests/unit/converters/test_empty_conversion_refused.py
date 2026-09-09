# tests/unit/converters/test_empty_conversion_refused.py
"""A conversion that would store no spectrum is refused, and which one is not.

``convert_msi`` used to return ``True`` for a source in which no position
carries a spectrum: exit 0, and a store at the output path with no table,
no image and no shapes -- while the log said both "No non-zero entries
found!" and, per plane, "no position carries a spectrum" (issue #242).
The exit status and what is left at the path are checked through the CLI
in ``tests/unit/test_cli_exit_status.py``; what is checked here is where
the refusal fires and, more importantly, where it must not.

The line it must not cross is #88: an acquisition is polygon-shaped and
the grid is its bounding box, so empty positions -- and, on a multi-slice
source, whole empty planes -- are ordinary. A plane with no row is dropped
with a warning and the rest of the store is written. Only a conversion
with no row *anywhere* has nothing to write.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import spatialdata

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata.streaming_converter import (
    SPATIALDATA_AVAILABLE,
    StreamingSpatialDataConverter,
)
from thyra.errors import ConversionRefused

pytestmark = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)

N_X, N_Y = 3, 2


def _config(n_z: int = 1) -> MockMSIConfig:
    return MockMSIConfig(
        n_x=N_X,
        n_y=N_Y,
        n_z=n_z,
        n_mz_bins=64,
        peaks_per_spectrum=(3, 5),
        seed=5,
    )


def _converter(reader, output: Path) -> StreamingSpatialDataConverter:
    return StreamingSpatialDataConverter(
        reader=reader,
        output_path=output,
        dataset_id="m",
        pixel_size_um=10.0,
        include_optical=False,
    )


class _NoSpectraAtAll(MockMSIReader):
    def iter_spectra(self, batch_size=None):
        return iter(())


class _EverySpectrumEmpty(MockMSIReader):
    def iter_spectra(self, batch_size=None):
        for coords, _mzs, _intensities in super().iter_spectra(batch_size):
            yield coords, np.array([]), np.array([])


class _EveryCoordinateOffTheGrid(MockMSIReader):
    def iter_spectra(self, batch_size=None):
        for _coords, mzs, intensities in super().iter_spectra(batch_size):
            yield (N_X + 3, N_Y + 3, 0), mzs, intensities


class _OnlyThePlaneInTheMiddle(MockMSIReader):
    """Planes 0 and 2 carry nothing; plane 1 is a full raster."""

    def iter_spectra(self, batch_size=None):
        for coords, mzs, intensities in super().iter_spectra(batch_size):
            if coords[2] != 1:
                continue
            yield coords, mzs, intensities


class TestNothingToWriteIsRefused:
    """Every route into the empty store, and the sentence each gets."""

    def test_no_spectra_at_all(self, tmp_path):
        converter = _converter(_NoSpectraAtAll(_config()), tmp_path / "a.zarr")
        with pytest.raises(ConversionRefused, match="no spectra at all"):
            converter._initialize_conversion()
            converter._process_spectra(converter._create_data_structures())

    def test_every_spectrum_empty(self, tmp_path):
        converter = _converter(_EverySpectrumEmpty(_config()), tmp_path / "b.zarr")
        with pytest.raises(ConversionRefused, match="every spectrum.*was empty"):
            converter._initialize_conversion()
            converter._process_spectra(converter._create_data_structures())

    def test_every_coordinate_off_the_grid(self, tmp_path):
        converter = _converter(
            _EveryCoordinateOffTheGrid(_config()), tmp_path / "c.zarr"
        )
        with pytest.raises(ConversionRefused, match="outside the declared 3x2x1 grid"):
            converter._initialize_conversion()
            converter._process_spectra(converter._create_data_structures())

    def test_the_refusal_reaches_the_caller_as_a_failure(self, tmp_path):
        """``convert`` catches it, so the caller sees False, not a traceback."""
        output = tmp_path / "d.zarr"
        assert _converter(_EverySpectrumEmpty(_config()), output).convert() is False
        assert not output.exists()

    def test_it_refuses_before_the_second_pass(self, tmp_path):
        """The source is read once, not twice, before being turned away.

        Refusing here rather than on the empty ``tables`` mapping at write
        time is what saves a full second read of a source that has already
        shown it has nothing to give.
        """
        reader = _EverySpectrumEmpty(_config())
        passes = []
        original = reader.iter_spectra

        def counted(batch_size=None):
            passes.append(1)
            return original(batch_size)

        reader.iter_spectra = counted  # type: ignore[method-assign]
        assert _converter(reader, tmp_path / "e.zarr").convert() is False
        assert sum(passes) == 1, f"the source was read {sum(passes)} times"


class TestSomeEmptyPlanesAreNotRefused:
    """#88: an empty plane is dropped, an empty conversion is refused."""

    def test_a_source_with_one_live_plane_converts(self, tmp_path):
        output = tmp_path / "planes.zarr"
        assert (
            _converter(_OnlyThePlaneInTheMiddle(_config(n_z=3)), output).convert()
            is True
        )

        sdata = spatialdata.SpatialData.read(str(output))
        assert set(sdata.tables) == {"m_z1"}
        assert sdata.tables["m_z1"].n_obs == N_X * N_Y
