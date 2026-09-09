# tests/unit/converters/test_anisotropic_pixel_size.py
"""One pitch per axis, in every block of the store that states one.

An anisotropic raster -- a DESI method with ``DesiXStep != DesiYStep`` --
used to be written down twice with two different answers. ``convert.py``
kept only the detected x pitch (``# Use X size``), the root attrs wrote
that same value as ``pixel_size_y_um``, and the single float went into
``coordinate_systems.global``, the image and shapes transforms and
``obs["spatial_x"]``/``["spatial_y"]``; only
``msi_metadata.ms_analysis.pixel_size_um`` carried the true ``(x, y)``.
With a detected 30 x 50 um the store said 30 x 30 in one place and
(30, 50) in another, and the raster rendered squashed by y/x (issue #228).

The schema was already per-axis -- ``pixel_size_um`` is ``{x, y}`` and
required, mapped to IMS:1000046 and IMS:1000047 -- so the fix is the
converter reading both halves back, not a format change. These assertions
are on the *stored artefacts* rather than on a helper's return value,
because the defect was that those artefacts disagreed with each other.

No registry dataset is anisotropic, so the pitch here is synthetic: the
reader is asked for one, exactly as ``convert.py`` asks a real Waters or
DESI extractor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import spatialdata
from spatialdata.transformations import get_transformation

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata.streaming_converter import (
    SPATIALDATA_AVAILABLE,
    StreamingSpatialDataConverter,
)
from thyra.core.base_converter import PixelSizeSource
from thyra.metadata.schema import MSI_METADATA_UNS_KEY

pytestmark = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)

N_X, N_Y = 3, 2
PITCH_X, PITCH_Y = 30.0, 50.0
KEY = "m_z0"


def _config() -> MockMSIConfig:
    return MockMSIConfig(
        n_x=N_X,
        n_y=N_Y,
        n_z=1,
        n_mz_bins=64,
        peaks_per_spectrum=(3, 5),
        seed=3,
    )


class _AnisotropicReader(MockMSIReader):
    """A source whose essential metadata declares 30 x 50 um."""

    def get_essential_metadata(self):
        essential = super().get_essential_metadata()
        object.__setattr__(essential, "pixel_size", (PITCH_X, PITCH_Y))
        return essential


def _detection_info() -> dict:
    """What ``convert.py`` builds when auto-detection succeeds."""
    return {
        "method": "automatic",
        "detected_x_um": PITCH_X,
        "detected_y_um": PITCH_Y,
        "source_format": "mock",
        "detection_successful": True,
    }


def _convert(output: Path, **kwargs) -> Path:
    """The CLI's own construction: the x pitch plus the detected pair."""
    converter = StreamingSpatialDataConverter(
        reader=MockMSIReader(_config()),
        output_path=output,
        dataset_id="m",
        pixel_size_um=PITCH_X,
        pixel_size_source=PixelSizeSource.AUTO_DETECTED,
        pixel_size_detection_info=_detection_info(),
        include_optical=False,
        **kwargs,
    )
    assert converter.convert() is True
    return output


def _root_attrs(store: Path) -> dict:
    doc = json.loads((store / "zarr.json").read_text())
    return doc.get("attributes", {}) or {}


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return _convert(tmp_path_factory.mktemp("aniso") / "out.zarr")


class TestTheStoredArtefactsAgree:
    """Every block that states a pitch states the same pair."""

    def test_root_attrs_carry_both_pitches(self, store):
        attrs = _root_attrs(store)
        assert attrs["pixel_size_x_um"] == PITCH_X
        assert attrs["pixel_size_y_um"] == PITCH_Y

    def test_the_coordinate_system_carries_both_pitches(self, store):
        cs = _root_attrs(store)["coordinate_systems"]["global"]
        assert cs["unit"] == "micrometer"
        assert cs["pixel_size_um_x"] == PITCH_X
        assert cs["pixel_size_um_y"] == PITCH_Y

    def test_the_raster_affine_is_not_square(self, store):
        affine = _root_attrs(store)["coordinate_systems"]["global"][
            "raster_to_global_affine"
        ]
        np.testing.assert_allclose(
            affine,
            [[PITCH_X, 0.0, 0.0], [0.0, PITCH_Y, 0.0], [0.0, 0.0, 1.0]],
        )

    def test_the_image_transform_scales_each_axis_by_its_own_pitch(self, store):
        sdata = spatialdata.SpatialData.read(str(store))
        image = sdata.images[f"{KEY}_tic"]
        matrix = get_transformation(
            image, to_coordinate_system="global"
        ).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        assert matrix[0, 0] == PITCH_X
        assert matrix[1, 1] == PITCH_Y

    def test_obs_positions_use_each_axis_pitch(self, store):
        table = spatialdata.SpatialData.read(str(store)).tables[KEY]
        np.testing.assert_allclose(
            table.obs["spatial_x"].values, table.obs["x"].values * PITCH_X
        )
        np.testing.assert_allclose(
            table.obs["spatial_y"].values, table.obs["y"].values * PITCH_Y
        )

    def test_the_pixel_footprints_are_not_square(self, store):
        shapes = spatialdata.SpatialData.read(str(store)).shapes[f"{KEY}_pixels"]
        xmin, ymin, xmax, ymax = shapes.geometry.iloc[0].bounds
        assert xmax - xmin == pytest.approx(PITCH_X)
        assert ymax - ymin == pytest.approx(PITCH_Y)

    def test_the_metadata_block_still_carries_the_pair(self, store):
        """The one block that was right before, and must stay right."""
        table = spatialdata.SpatialData.read(str(store)).tables[KEY]
        block = table.uns[MSI_METADATA_UNS_KEY]
        assert block["ms_analysis"]["pixel_size_um"] == {"x": PITCH_X, "y": PITCH_Y}

    def test_the_shapes_span_the_image_extent(self, store):
        """The contract the issue's "squashed by y/x" broke.

        The footprints are in micrometres and the image is scaled into
        them, so the two describe the same rectangle to within the half
        pixel the boxes add on each side.
        """
        sdata = spatialdata.SpatialData.read(str(store))
        shapes = sdata.shapes[f"{KEY}_pixels"]
        xmin, ymin, xmax, ymax = shapes.total_bounds
        assert xmax - xmin == pytest.approx((N_X - 1) * PITCH_X + PITCH_X)
        assert ymax - ymin == pytest.approx((N_Y - 1) * PITCH_Y + PITCH_Y)


class TestWhereTheYPitchComesFrom:
    """Both detection routes, and the declaration that overrides them."""

    def test_a_reader_detected_pair_is_taken_whole(self, tmp_path):
        """The API-direct route: no detection info, the reader is asked.

        ``essential.pixel_size`` is a pair on every reader and taking only
        ``[0]`` of it is the same defect one layer down.
        """
        converter = StreamingSpatialDataConverter(
            reader=_AnisotropicReader(_config()),
            output_path=tmp_path / "reader.zarr",
            dataset_id="m",
            include_optical=False,
        )
        assert converter.convert() is True

        assert converter.pixel_size_um == PITCH_X
        assert converter.pixel_size_y_um == PITCH_Y
        attrs = _root_attrs(tmp_path / "reader.zarr")
        assert attrs["pixel_size_x_um"] == PITCH_X
        assert attrs["pixel_size_y_um"] == PITCH_Y

    def test_a_stated_pitch_makes_the_raster_square(self, tmp_path):
        """``--pixel-size`` is a declaration and applies to both axes.

        Someone who knows the file's y step is wrong -- or who simply
        wants square pixels -- says so with one number, and the store then
        says that number twice rather than mixing it with a detected one.
        """
        converter = StreamingSpatialDataConverter(
            reader=_AnisotropicReader(_config()),
            output_path=tmp_path / "stated.zarr",
            dataset_id="m",
            pixel_size_um=17.0,
            pixel_size_source=PixelSizeSource.USER_PROVIDED,
            pixel_size_detection_info=_detection_info(),
            include_optical=False,
        )
        assert converter.convert() is True

        assert converter.pixel_size_um == 17.0
        assert converter.pixel_size_y_um == 17.0
        attrs = _root_attrs(tmp_path / "stated.zarr")
        assert attrs["pixel_size_x_um"] == 17.0
        assert attrs["pixel_size_y_um"] == 17.0
        table = spatialdata.SpatialData.read(str(tmp_path / "stated.zarr")).tables[KEY]
        assert table.uns[MSI_METADATA_UNS_KEY]["ms_analysis"]["pixel_size_um"] == {
            "x": 17.0,
            "y": 17.0,
        }

    def test_a_square_raster_is_unchanged(self, tmp_path):
        """The ordinary case: one number, everywhere, as before."""
        converter = StreamingSpatialDataConverter(
            reader=MockMSIReader(_config()),
            output_path=tmp_path / "square.zarr",
            dataset_id="m",
            pixel_size_um=10.0,
            include_optical=False,
        )
        assert converter.convert() is True

        attrs = _root_attrs(tmp_path / "square.zarr")
        assert attrs["pixel_size_x_um"] == attrs["pixel_size_y_um"] == 10.0

    def test_it_says_so_when_the_raster_is_anisotropic(self, tmp_path, thyra_logs):
        with thyra_logs("thyra.core.base_converter") as records:
            StreamingSpatialDataConverter(
                reader=MockMSIReader(_config()),
                output_path=tmp_path / "logged.zarr",
                dataset_id="m",
                pixel_size_um=PITCH_X,
                pixel_size_source=PixelSizeSource.AUTO_DETECTED,
                pixel_size_detection_info=_detection_info(),
                include_optical=False,
            )
        assert any("Anisotropic raster" in r.getMessage() for r in records), [
            r.getMessage() for r in records
        ]

    def test_float_noise_is_not_an_anisotropic_raster(self, tmp_path, thyra_logs):
        """A detector dividing an extent by a count can land a few ULP apart.

        Until #217 the Waters extractor returned 29.66 x 28.93 for every
        square raster; that is 2.5 percent and genuinely anisotropic, and
        it was fixed there. What must not be reported is the last bit of a
        float.
        """
        info = _detection_info()
        info["detected_y_um"] = np.nextafter(PITCH_X, PITCH_X + 1)
        with thyra_logs("thyra.core.base_converter") as records:
            StreamingSpatialDataConverter(
                reader=MockMSIReader(_config()),
                output_path=tmp_path / "noise.zarr",
                dataset_id="m",
                pixel_size_um=PITCH_X,
                pixel_size_source=PixelSizeSource.AUTO_DETECTED,
                pixel_size_detection_info=info,
                include_optical=False,
            )
        assert not any("Anisotropic raster" in r.getMessage() for r in records)
