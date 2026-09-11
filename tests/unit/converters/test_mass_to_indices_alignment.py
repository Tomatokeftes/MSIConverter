"""``_map_mass_to_indices`` must stay parallel to the intensity array.

``BaseMSIConverter._map_mass_to_indices`` used to validate its
``searchsorted`` result against a 1e-6 tolerance and return only the
indices that passed -- but every caller pairs the returned indices with
the ORIGINAL, unfiltered intensity array.  The streaming converter's
``_process_spectrum`` returns ``(mz_indices, intensities)`` straight into
the PCS/COO passes, which do ``np.add.at(total_intensity, mz_indices,
ints)`` and positional row writes on the assumption that the two arrays
are parallel.  Had the tolerance mask ever fired, the arrays would have
desynced: a length mismatch crash in some paths, intensities summed into
the wrong m/z bins in others.

The mask never fired on well-formed data (continuous mode yields the
axis itself; a processed-mode raw axis is the union of every spectrum's
m/z values, so every lookup matches exactly), which is why the desync
was never observed.  These tests pin the corrected contract:

- the returned index array is always parallel to the input ``mzs``;
- a value with no axis entry within tolerance raises ``ValueError``
  instead of being dropped silently (a near-miss means the axis and the
  data have diverged -- that is corrupt input, not something to paper
  over);
- a value within tolerance maps to its NEAREST axis entry, even when
  ``searchsorted`` lands on the far side of it.

Contrast ``BaseMSIReader.map_mz_to_common_axis``, which filters indices
and intensities together and so was never affected.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata.streaming_converter import (
    StreamingSpatialDataConverter,
)


@pytest.fixture
def converter(tmp_path: Path) -> StreamingSpatialDataConverter:
    """A streaming converter on the no-resample path with a loaded axis."""
    reader = MockMSIReader(MockMSIConfig(n_x=3, n_y=3, n_mz_bins=1001))
    conv = StreamingSpatialDataConverter(
        reader=reader,
        output_path=tmp_path / "out.zarr",
        dataset_id="mock",
        pixel_size_um=10.0,
    )
    conv._initialize_conversion()
    assert conv._resampling_config is None
    return conv


class TestOffAxisMzRaises:
    """One off-axis m/z value must abort the spectrum, not vanish."""

    @pytest.mark.parametrize(
        "off_axis",
        [
            pytest.param(lambda axis: (axis[200] + axis[201]) / 2, id="mid-bin"),
            pytest.param(lambda axis: axis[0] - 1.0, id="below-range"),
            pytest.param(lambda axis: axis[-1] + 1.0, id="above-range"),
        ],
    )
    def test_streaming_no_resample_path(self, converter, off_axis):
        axis = converter._common_mass_axis
        mzs = np.array([axis[10], axis[50], off_axis(axis), axis[400]])
        intensities = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="no common mass axis entry"):
            converter._process_spectrum(mzs, intensities)


class TestParallelReturn:
    """Indices come back parallel to the input arrays."""

    def test_exact_matches_map_positionally(self, converter):
        axis = converter._common_mass_axis
        positions = np.array([0, 17, 500, 1000])
        mzs = axis[positions]
        intensities = np.array([10.0, 20.0, 30.0, 40.0])

        mz_indices, out_intensities = converter._process_spectrum(mzs, intensities)

        np.testing.assert_array_equal(mz_indices, positions)
        np.testing.assert_array_equal(out_intensities, intensities)

    def test_fp_noise_maps_to_nearest_entry(self, converter):
        # A value a hair ABOVE an axis entry lands on the far neighbor via
        # searchsorted; the old one-sided check dropped it silently even
        # though it sits well within the documented 1e-6 tolerance.
        axis = converter._common_mass_axis
        mzs = np.array([axis[5] + 1e-9, axis[7] - 1e-9])
        intensities = np.array([1.0, 2.0])

        mz_indices, out_intensities = converter._process_spectrum(mzs, intensities)

        np.testing.assert_array_equal(mz_indices, [5, 7])
        np.testing.assert_array_equal(out_intensities, intensities)
