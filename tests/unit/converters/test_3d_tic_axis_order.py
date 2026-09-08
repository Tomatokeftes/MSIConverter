# tests/unit/converters/test_3d_tic_axis_order.py
"""The 3D TIC volume must be stored on the axes it declares.

The 3D converter used to accumulate TIC into an array indexed ``[y, x, z]``
and hand it to ``Image3DModel``, which declares ``(c, z, y, x)``. Those are
different orders, so the array had to be transposed. ``_create_tic_image``
used to *relabel* it instead::

    z_size, y_size, x_size = tic_values.shape          # (n_y, n_x, n_z)
    tic_values.reshape(1, z_size, y_size, x_size)      # a no-op reshape

Unpacking ``(n_y, n_x, n_z)`` into names reading ``z, y, x`` renames the axes
without moving anything, and the reshape that follows only prepends the channel
axis. A 5(x) x 3(y) x 2(z) acquisition was written as ``(1, 3, 5, 2)`` and
``convert()`` returned ``True``: x and z were swapped and y took x's extent, so
a volume came back sliced along the wrong axis with no error anywhere.

Three properties of the old tests let it through, and this file is built to
have none of them:

* **A square grid.** The only test reaching the branch used 3 x 3 x 2, where two
  of the three extents agree and a permutation is half-invisible. The grid here
  is **pairwise different** (5 x 3 x 2), so no transpose can coincidentally
  produce the right shape.
* **Asserting that mocks were called.** ``AnnData`` and ``TableModel`` were
  patched out and the assertions were ``assert mock.called``, which cannot see
  the contents of an array. Nothing is mocked here; the store is converted for
  real and read back off disk, because what is wrong is what is *stored*.
* **Totals only.** ``test_tic_image_totals_the_matrix`` compares ``sum(TIC)`` to
  ``sum(X)``, and permuting an array does not change its sum. Every voxel is
  checked at its own index instead.

The expected value is **computed, never read back**: each spatial axis writes to
its own m/z channel with a different order of magnitude, so a voxel's TIC
decodes uniquely to the coordinate that produced it (``235`` is x=4, y=2, z=1
and nothing else). An assertion against a value read out of the same store would
pass for any permutation that corrupted both sides equally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from thyra.converters.spatialdata import SpatialDataConverter
from thyra.converters.spatialdata.base_spatialdata_converter import (
    SPATIALDATA_AVAILABLE,
)
from thyra.core.base_extractor import MetadataExtractor
from thyra.core.base_reader import BaseMSIReader
from thyra.metadata.types import ComprehensiveMetadata, EssentialMetadata

pytestmark = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)

_DATASET_ID = "vol"

# Pairwise different on purpose -- see the module docstring. A cubic or
# square grid cannot distinguish a correct volume from a transposed one.
_N_X, _N_Y, _N_Z = 5, 3, 2

# Three m/z channels, one per spatial axis. Values sit exactly on the axis the
# reader reports, so the no-resampling path maps them by exact match and the
# stored TIC is the sum of what was emitted, to the bit.
_MASS_AXIS = np.array([100.0, 200.0, 300.0], dtype=np.float64)


def _expected_tic(x: int, y: int, z: int) -> float:
    """TIC of the voxel at ``(x, y, z)``, in closed form.

    ``111 + x + 10y + 100z`` over the extents here: units carry x, tens carry
    y, hundreds carry z, so the value names its own coordinate. Every term is
    strictly positive, so no voxel is empty and none is dropped as a phantom.
    """
    return float((1 + x) + 10 * (1 + y) + 100 * (1 + z))


def _intensities(x: int, y: int, z: int) -> NDArray[np.float64]:
    """One channel per spatial axis; sums to ``_expected_tic`` exactly."""
    return np.array(
        [1 + x, 10 * (1 + y), 100 * (1 + z)],
        dtype=np.float64,
    )


class _AxisProbeExtractor(MetadataExtractor):
    """Reports the grid the probe reader emits, and nothing else."""

    def __init__(self, dimensions: Tuple[int, int, int]) -> None:
        super().__init__(None)
        self._dimensions = dimensions

    def _extract_essential_impl(self) -> EssentialMetadata:
        n_x, n_y, n_z = self._dimensions
        n_spectra = n_x * n_y * n_z
        return EssentialMetadata(
            dimensions=self._dimensions,
            coordinate_bounds=(0.0, float(n_x - 1), 0.0, float(n_y - 1)),
            mass_range=(float(_MASS_AXIS[0]), float(_MASS_AXIS[-1])),
            pixel_size=None,
            n_spectra=n_spectra,
            total_peaks=n_spectra * len(_MASS_AXIS),
            estimated_memory_gb=0.001,
            source_path="axis_order_probe",
        )

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        return ComprehensiveMetadata(
            essential=self._extract_essential_impl(),
            format_specific={"format": "probe"},
            acquisition_params={},
            instrument_info={"instrument": "probe"},
            raw_metadata={"source": "probe"},
        )


class _AxisProbeReader(BaseMSIReader):
    """A full grid whose every voxel carries its own coordinate as intensity."""

    def __init__(self, dimensions: Tuple[int, int, int]) -> None:
        super().__init__(Path("axis_order_probe"))
        self._dimensions = dimensions

    def _create_metadata_extractor(self) -> MetadataExtractor:
        return _AxisProbeExtractor(self._dimensions)

    @property
    def has_shared_mass_axis(self) -> bool:
        return True

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        return _MASS_AXIS.copy()

    def get_optical_image_paths(self) -> List[Path]:
        return []

    def get_peak_counts_per_pixel(self) -> Optional[NDArray[np.int32]]:
        return None

    def reset(self) -> None:
        """Spectra are a pure function of the coordinate; nothing to reset."""

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        n_x, n_y, n_z = self._dimensions
        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    yield (x, y, z), _MASS_AXIS.copy(), _intensities(x, y, z)

    def close(self) -> None:
        """No resource to release."""


def _convert(tmp_path_factory, dimensions: Tuple[int, int, int]):
    """Convert one probe grid through the 3D route; hand back the store root."""
    n_x, n_y, n_z = dimensions
    output_path = tmp_path_factory.mktemp(f"tic_axes_{n_x}x{n_y}x{n_z}") / "out.zarr"
    converter = SpatialDataConverter(
        _AxisProbeReader(dimensions),
        output_path,
        handle_3d=True,
        dataset_id=_DATASET_ID,
        pixel_size_um=10.0,
    )
    assert converter.convert() is True, f"{dimensions} conversion failed"
    return output_path


@pytest.fixture(scope="module")
def volume_store(tmp_path_factory):
    """A real multi-slice volume: the branch that was permuted."""
    return _convert(tmp_path_factory, (_N_X, _N_Y, _N_Z))


@pytest.fixture(scope="module")
def single_slice_store(tmp_path_factory):
    """``n_z == 1``, which takes the other branch of ``_create_tic_image``."""
    return _convert(tmp_path_factory, (_N_X, _N_Y, 1))


def _tic_image(store_path):
    """The TIC image, read back off disk rather than out of the converter."""
    import spatialdata

    sdata = spatialdata.read_zarr(str(store_path))
    return sdata.images[f"{_DATASET_ID}_tic"]


def test_volume_tic_has_the_declared_axis_order(volume_store):
    """Shape must match the extents the axis labels claim.

    This is the reproduction. The dims have always read ``(c, z, y, x)``; what
    was wrong was the array underneath them, so comparing the two is the whole
    test. On the old code this is ``(1, 3, 5, 2)``.
    """
    image = _tic_image(volume_store)

    assert image.dims == ("c", "z", "y", "x")
    assert np.asarray(image.data).shape == (1, _N_Z, _N_Y, _N_X)


def test_every_voxel_lands_at_its_own_coordinate(volume_store):
    """The assertion a shape check cannot make.

    A shape of ``(1, 2, 3, 5)`` is necessary but not sufficient: a volume can
    have the right extents and still be transposed within them, and on a grid
    where two extents happened to agree it would be. Checking the whole volume
    against the closed form pins the axes *and* their direction.
    """
    volume = np.asarray(_tic_image(volume_store).data)

    expected = np.array(
        [
            [[_expected_tic(x, y, z) for x in range(_N_X)] for y in range(_N_Y)]
            for z in range(_N_Z)
        ],
        dtype=np.float64,
    )[np.newaxis, ...]

    np.testing.assert_allclose(volume, expected, rtol=1e-9)


def test_named_asymmetric_voxel(volume_store):
    """One voxel spelled out, so the intent survives without running anything.

    ``(x=4, y=2, z=1)`` is the far corner and is only addressable at all if the
    extents are right, since 4 exceeds both of the other two axes' lengths.
    Under the old permutation this index raised ``IndexError`` rather than
    returning a wrong number, which is worth stating: the failure was not
    subtle once anything actually indexed the volume by coordinate.
    """
    volume = np.asarray(_tic_image(volume_store).data)

    assert volume[0, 1, 2, 4] == pytest.approx(235.0)
    assert _expected_tic(x=4, y=2, z=1) == 235.0


def test_single_slice_keeps_its_axis_order(single_slice_store):
    """The ``n_z == 1`` branch, which was already correct and must stay so.

    It reads the same ``[y, x, z]`` accumulator through a different path, so a
    later change to the accumulator's layout would silently break this one
    while the volume branch above stayed green.
    """
    image = _tic_image(single_slice_store)
    plane = np.asarray(image.data)

    assert image.dims == ("c", "y", "x")
    assert plane.shape == (1, _N_Y, _N_X)

    expected = np.array(
        [[_expected_tic(x, y, 0) for x in range(_N_X)] for y in range(_N_Y)],
        dtype=np.float64,
    )[np.newaxis, ...]

    np.testing.assert_allclose(plane, expected, rtol=1e-9)
