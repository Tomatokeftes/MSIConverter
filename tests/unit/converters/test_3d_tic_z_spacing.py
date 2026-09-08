# tests/unit/converters/test_3d_tic_z_spacing.py
"""The 3D TIC volume must be scaled by a real slice-to-slice distance.

``_create_tic_image`` used to build the volume's transform as::

    Scale([pixel_size_um, pixel_size_um, pixel_size_um], axes=("x", "y", "z"))

``pixel_size_um`` is the **in-plane** raster pitch. Reusing it for z asserts
that consecutive slices sit exactly one pixel width apart -- which is true only
by coincidence, because sections are cut by a microtome while the raster is set
by the stage. The voxel *values* were right (PR #146 fixed the permutation that
made them wrong); the volume was simply the wrong depth, so any consumer
reading it in micrometres -- Ousia does -- rendered the stack squashed or
stretched along z.

What these tests pin, and why each is here:

* **The affine, not the absence of an exception.** A converter that raises
  nothing tells you nothing about what it wrote. Every assertion below reads
  the store back off disk and checks
  ``get_transformation(...).to_affine_matrix(...)`` entry by entry.
* **Three pairwise-different lengths.** ``pixel_size_um``, ``z_spacing_um`` and
  the grid extents are all distinct and none is a multiple of another, so a
  value landing on the wrong axis cannot coincidentally produce the expected
  matrix. With the old code ``test_explicit_z_spacing_reaches_the_global_affine``
  reports 10.0 where 45.0 is required.
* **The table, not just the image.** ``spatial_z`` in ``obs`` had the same
  hardcoded pitch. Fixing only the image would leave the volume and the table
  disagreeing about depth at ``"global"``, which is worse than both being
  consistently wrong, so their agreement is asserted directly.
* **The assumption is machine-readable.** The no-spacing-given case still falls
  back to the in-plane pitch -- what changes is that the store now says so, via
  ``coordinate_systems.global.z_spacing_source``. A warning in a log nobody kept
  is not a record.

Note ``Scale`` pairs values with axis *names*, not by position, so the
converter's ``axes=("x", "y", "z")`` against a ``(c, z, y, x)`` image is correct
and deliberate. The expected matrices below are written out in ``(c, z, y, x)``
order to make that mapping explicit rather than implied.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from thyra.converters.spatialdata import SpatialDataConverter
from thyra.converters.spatialdata.base_spatialdata_converter import (
    SPATIALDATA_AVAILABLE,
)
from thyra.core.base_converter import ZSpacingSource
from thyra.core.base_extractor import MetadataExtractor
from thyra.core.base_reader import BaseMSIReader
from thyra.metadata.types import ComprehensiveMetadata, EssentialMetadata

pytestmark = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)

_DATASET_ID = "vol"

# Pairwise different, as in test_3d_tic_axis_order: a cubic grid cannot tell a
# correct volume from a permuted one.
_N_X, _N_Y, _N_Z = 5, 3, 2

# Deliberately unrelated numbers. 45 is not a multiple of 10, so reusing the
# in-plane pitch for z (the defect) cannot produce the expected matrix, and
# neither can swapping the two.
_PIXEL_SIZE_UM = 10.0
_Z_SPACING_UM = 45.0

_MASS_AXIS = np.array([100.0, 200.0], dtype=np.float64)


class _VolumeProbeExtractor(MetadataExtractor):
    """Reports the grid the probe reader emits, and nothing else.

    ``pixel_size`` is ``None`` on purpose: it forces the converter to keep the
    ``pixel_size_um`` the test passed in, rather than auto-detecting over it,
    so the in-plane pitch under test is the one the test chose.
    """

    def __init__(
        self,
        dimensions: Tuple[int, int, int],
        z_spacing_um: Optional[float] = None,
    ) -> None:
        super().__init__(None)
        self._dimensions = dimensions
        self._z_spacing_um = z_spacing_um

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
            source_path="z_spacing_probe",
            z_spacing_um=self._z_spacing_um,
        )

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        return ComprehensiveMetadata(
            essential=self._extract_essential_impl(),
            format_specific={"format": "probe"},
            acquisition_params={},
            instrument_info={"instrument": "probe"},
            raw_metadata={"source": "probe"},
        )


class _VolumeProbeReader(BaseMSIReader):
    """A full grid; every voxel carries a positive intensity so none is dropped."""

    def __init__(
        self,
        dimensions: Tuple[int, int, int],
        z_spacing_um: Optional[float] = None,
    ) -> None:
        super().__init__(Path("z_spacing_probe"))
        self._dimensions = dimensions
        self._z_spacing_um = z_spacing_um

    def _create_metadata_extractor(self) -> MetadataExtractor:
        return _VolumeProbeExtractor(self._dimensions, self._z_spacing_um)

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
                    intensities = np.array(
                        [1.0 + x + 10.0 * y, 100.0 * (1.0 + z)], dtype=np.float64
                    )
                    yield (x, y, z), _MASS_AXIS.copy(), intensities

    def close(self) -> None:
        """No resource to release."""


def _convert(
    tmp_path: Path,
    dimensions: Tuple[int, int, int] = (_N_X, _N_Y, _N_Z),
    z_spacing_um: Optional[float] = None,
    reader_z_spacing_um: Optional[float] = None,
) -> Path:
    """Convert one probe grid through the 3D route; hand back the store root."""
    output_path = tmp_path / "out.zarr"
    converter = SpatialDataConverter(
        _VolumeProbeReader(dimensions, z_spacing_um=reader_z_spacing_um),
        output_path,
        handle_3d=True,
        dataset_id=_DATASET_ID,
        pixel_size_um=_PIXEL_SIZE_UM,
        z_spacing_um=z_spacing_um,
    )
    assert converter.convert() is True, f"{dimensions} conversion failed"
    return output_path


def _read_store(store_path: Path):
    import spatialdata

    return spatialdata.read_zarr(str(store_path))


def _global_affine(store_path: Path) -> NDArray[np.float64]:
    """The TIC element's affine to ``"global"``, in ``(c, z, y, x)`` order."""
    from spatialdata.transformations import get_transformation

    image = _read_store(store_path).images[f"{_DATASET_ID}_tic"]
    transform = get_transformation(image, to_coordinate_system="global")
    axes = ("c", "z", "y", "x")
    return np.asarray(
        transform.to_affine_matrix(input_axes=axes, output_axes=axes),
        dtype=np.float64,
    )


def _expected_affine(pixel_size_um: float, z_spacing_um: float) -> NDArray[np.float64]:
    """The matrix a correct ``Scale`` produces, written out in ``(c, z, y, x)``.

    Spelled as a literal diagonal rather than derived from a ``Scale`` so the
    test cannot agree with the code by sharing its mistake.
    """
    return np.diag([1.0, z_spacing_um, pixel_size_um, pixel_size_um, 1.0])


def _zarr_attrs(store_path: Path) -> Dict[str, Any]:
    """Top-level zarr attrs, for both the v2 and v3 layouts."""
    z3 = store_path / "zarr.json"
    if z3.exists():
        with open(z3, encoding="utf-8") as f:
            return (json.load(f) or {}).get("attributes", {}) or {}
    z2 = store_path / ".zattrs"
    if z2.exists():
        with open(z2, encoding="utf-8") as f:
            return json.load(f) or {}
    raise FileNotFoundError(f"No zarr attrs in {store_path}")


def _global_cs(store_path: Path) -> Dict[str, Any]:
    return _zarr_attrs(store_path)["coordinate_systems"]["global"]


@contextmanager
def _captured_warnings(logger_name: str = "thyra"):
    """Collect warnings from ``logger_name``, whatever the logging state.

    Deliberately not ``caplog``. ``setup_logging`` sets ``propagate = False`` on
    the ``thyra`` logger, and it is process-global: once any test has invoked
    the CLI, ``caplog``'s root handler never sees another Thyra record. That
    made this assertion pass alone and fail in the full suite, which is the
    worst way for a test to be wrong.

    Attaching a handler directly to the logger sidesteps propagation entirely,
    so the result does not depend on which tests ran first.
    """
    import logging

    records: List[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Collector(level=logging.WARNING)
    logger = logging.getLogger(logger_name)
    previous_level = logger.level
    logger.addHandler(handler)
    if previous_level > logging.WARNING or previous_level == logging.NOTSET:
        logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


class TestTheGlobalAffine:
    """What the volume actually claims about its own physical extent."""

    def test_explicit_z_spacing_reaches_the_global_affine(self, tmp_path):
        """The reproduction. On the old code the z entry is 10.0, not 45.0."""
        store = _convert(tmp_path, z_spacing_um=_Z_SPACING_UM)

        np.testing.assert_allclose(
            _global_affine(store),
            _expected_affine(_PIXEL_SIZE_UM, _Z_SPACING_UM),
            rtol=1e-12,
        )

    def test_the_z_entry_is_not_the_in_plane_pitch(self, tmp_path):
        """Stated on its own, so the intent survives a rewrite of the matrix.

        The whole defect is one number being reused for a quantity it does not
        describe. Naming that here means a future refactor that reintroduces it
        fails on a test whose message says why.
        """
        matrix = _global_affine(_convert(tmp_path, z_spacing_um=_Z_SPACING_UM))

        z_entry = matrix[1, 1]
        assert z_entry == pytest.approx(_Z_SPACING_UM)
        assert z_entry != pytest.approx(_PIXEL_SIZE_UM), (
            "z was scaled by the in-plane pixel pitch, which asserts slices "
            "are one pixel width apart -- true only by coincidence"
        )

    def test_the_in_plane_axes_are_untouched(self, tmp_path):
        """Giving z its own spacing must not disturb x or y."""
        matrix = _global_affine(_convert(tmp_path, z_spacing_um=_Z_SPACING_UM))

        assert matrix[2, 2] == pytest.approx(_PIXEL_SIZE_UM)  # y
        assert matrix[3, 3] == pytest.approx(_PIXEL_SIZE_UM)  # x

    def test_a_reader_reported_spacing_is_used(self, tmp_path):
        """``EssentialMetadata.z_spacing_um`` beats the fallback.

        No shipped reader populates it yet -- imzML has no term for it -- but
        the precedence chain is live code, so it is tested like live code.
        """
        store = _convert(tmp_path, reader_z_spacing_um=_Z_SPACING_UM)

        np.testing.assert_allclose(
            _global_affine(store),
            _expected_affine(_PIXEL_SIZE_UM, _Z_SPACING_UM),
            rtol=1e-12,
        )

    def test_an_explicit_spacing_outranks_the_reader(self, tmp_path):
        """The caller knows their sections better than the file does."""
        store = _convert(tmp_path, z_spacing_um=_Z_SPACING_UM, reader_z_spacing_um=7.5)

        assert _global_affine(store)[1, 1] == pytest.approx(_Z_SPACING_UM)


class TestTheAssumedCase:
    """What happens when nobody supplies a spacing -- the default path."""

    def test_it_falls_back_to_the_in_plane_pitch(self, tmp_path):
        """Unchanged behaviour: this is what every existing 3D store did.

        Kept deliberately rather than made an error, so converting a volume
        without knowing its section thickness still works. What is new is the
        record asserted in the next test.
        """
        store = _convert(tmp_path)

        np.testing.assert_allclose(
            _global_affine(store),
            _expected_affine(_PIXEL_SIZE_UM, _PIXEL_SIZE_UM),
            rtol=1e-12,
        )

    def test_the_assumption_is_recorded_in_the_store(self, tmp_path):
        """The guess must be distinguishable from a measurement, on disk.

        Without this a consumer sees a plain number and no way to tell whether
        anyone ever measured it. A warning at conversion time does not help
        someone opening the store six months later.
        """
        cs = _global_cs(_convert(tmp_path))

        assert cs["z_spacing_source"] == ZSpacingSource.ASSUMED_ISOTROPIC.value
        assert cs["z_spacing_um"] == pytest.approx(_PIXEL_SIZE_UM)

    def test_a_supplied_spacing_is_recorded_as_supplied(self, tmp_path):
        cs = _global_cs(_convert(tmp_path, z_spacing_um=_Z_SPACING_UM))

        assert cs["z_spacing_source"] == ZSpacingSource.USER_PROVIDED.value
        assert cs["z_spacing_um"] == pytest.approx(_Z_SPACING_UM)

    def test_a_reader_reported_spacing_is_recorded_as_detected(self, tmp_path):
        cs = _global_cs(_convert(tmp_path, reader_z_spacing_um=_Z_SPACING_UM))

        assert cs["z_spacing_source"] == ZSpacingSource.AUTO_DETECTED.value
        assert cs["z_spacing_um"] == pytest.approx(_Z_SPACING_UM)

    def test_the_warning_names_the_flag_that_fixes_it(self, tmp_path):
        """A warning that does not say what to do instead is noise."""
        with _captured_warnings() as records:
            _convert(tmp_path)

        assumed = [m for m in records if "z spacing" in m]
        assert assumed, "converting a volume with no z spacing warned about nothing"
        assert any("--z-spacing" in message for message in assumed)

    def test_supplying_a_spacing_silences_the_warning(self, tmp_path):
        with _captured_warnings() as records:
            _convert(tmp_path, z_spacing_um=_Z_SPACING_UM)

        assert not [m for m in records if "No z spacing was supplied" in m]

    def test_two_dimensional_conversions_do_not_warn(self, tmp_path):
        """A 2D dataset has no depth to get wrong.

        Warning on every ordinary conversion would train people to ignore the
        message on the volumes where it actually matters.
        """
        with _captured_warnings() as records:
            _convert(tmp_path, dimensions=(_N_X, _N_Y, 1))

        assert not [m for m in records if "z spacing" in m]


class TestTheTableAgreesWithTheVolume:
    """``obs.spatial_z`` and the image's ``Scale`` are one contract."""

    def test_slice_depth_uses_the_z_spacing(self, tmp_path):
        """Both elements resolve to the same depth at ``"global"``.

        ``spatial_z`` carried the same hardcoded in-plane pitch, so before this
        change a volume and its own table disagreed the moment the two numbers
        differed.
        """
        table = _read_store(_convert(tmp_path, z_spacing_um=_Z_SPACING_UM)).tables[
            _DATASET_ID
        ]

        for z_index in range(_N_Z):
            rows = table.obs[table.obs["z"] == z_index]
            assert not rows.empty, f"no pixels at z={z_index}"
            np.testing.assert_allclose(
                rows["spatial_z"].to_numpy(),
                z_index * _Z_SPACING_UM,
                rtol=1e-12,
            )

    def test_the_table_depth_is_not_the_in_plane_pitch(self, tmp_path):
        """The specific wrong answer, named."""
        table = _read_store(_convert(tmp_path, z_spacing_um=_Z_SPACING_UM)).tables[
            _DATASET_ID
        ]

        top = table.obs[table.obs["z"] == _N_Z - 1]["spatial_z"].to_numpy()
        assert top[0] == pytest.approx((_N_Z - 1) * _Z_SPACING_UM)
        assert top[0] != pytest.approx((_N_Z - 1) * _PIXEL_SIZE_UM)


class TestTwoDimensionalStoresAreUnchanged:
    """A 2D conversion must be byte-identical to what earlier versions wrote."""

    def test_a_single_slice_carries_no_z_keys(self, tmp_path):
        """Absence is the signal that there is no z axis.

        This is also why ``convention_version`` does not move for this change:
        the keys are additive and only volumes carry them, so a consumer that
        never does 3D sees exactly the schema it already knows.
        """
        cs = _global_cs(_convert(tmp_path, dimensions=(_N_X, _N_Y, 1)))

        assert "z_spacing_um" not in cs
        assert "z_spacing_source" not in cs
        assert cs["convention_version"] == 1

    def test_the_single_slice_transform_is_still_two_dimensional(self, tmp_path):
        """The ``n_z == 1`` branch takes a different path and must not gain a z."""
        store = _convert(tmp_path, dimensions=(_N_X, _N_Y, 1))
        image = _read_store(store).images[f"{_DATASET_ID}_tic"]

        assert image.dims == ("c", "y", "x")


class TestRejectedInput:
    def test_a_non_positive_spacing_is_rejected(self, tmp_path):
        """Caught in the constructor, before anything is read or written."""
        for bad in (0.0, -1.0, -0.5):
            with pytest.raises(ValueError, match="z_spacing_um must be positive"):
                SpatialDataConverter(
                    _VolumeProbeReader((_N_X, _N_Y, _N_Z)),
                    tmp_path / "rejected.zarr",
                    handle_3d=True,
                    dataset_id=_DATASET_ID,
                    pixel_size_um=_PIXEL_SIZE_UM,
                    z_spacing_um=bad,
                )

    def test_nothing_is_written_when_it_is_rejected(self, tmp_path):
        output = tmp_path / "rejected.zarr"
        with pytest.raises(ValueError):
            SpatialDataConverter(
                _VolumeProbeReader((_N_X, _N_Y, _N_Z)),
                output,
                handle_3d=True,
                dataset_id=_DATASET_ID,
                pixel_size_um=_PIXEL_SIZE_UM,
                z_spacing_um=-1.0,
            )
        assert not output.exists()
