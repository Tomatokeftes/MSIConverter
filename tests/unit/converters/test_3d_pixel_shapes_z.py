# tests/unit/converters/test_3d_pixel_shapes_z.py
"""A volume's pixel footprints are flat, and its depth lives elsewhere.

The history here is worth keeping, because the obvious reading of it is
wrong in both directions.

``_create_pixel_shapes`` originally took an ``is_3d`` argument and never
read it, so a volume's polygons stacked at z=0 while its TIC image spanned
``n_z * z_spacing_um``. 7317792 fixed that by making the footprints
``POLYGON Z`` at the depth of their slice. Geometrically that was right,
and it broke ``spatialdata``'s spatial queries: measured on this fixture,
a bounding box enclosing the entire dataset with 1000um of margin returned
26 of 30 footprints and 26 of 30 table rows, with no exception and no
warning. ``shapely.force_2d`` on the same geometry returned 30 of 30.

So the depth was moved back off the geometry, and this file pins the
resulting arrangement:

* the polygons are 2D -- one square per pixel, in x and y only;
* the depth is on the TIC image's ``Scale`` and in ``obs["spatial_z"]``,
  both of which 255c177 made honest and neither of which changed here;
* spatial queries return every pixel again.

**This is not a claim that flat is ideal.** It is a claim that a
documented gap beats a silently truncated query. ``spatialdata`` asks for
it too: ``ShapesModel.validate`` warns that a 3-dimensional geometry
column "could led to unexpected behaviors" and names ``force_2d()`` as the
remedy. 3D shapes are absent from the upstream roadmap
(scverse/spatialdata#109, idle since June 2023, covers images, labels and
transformations), and the live 2.5D discussion (#961) scopes itself to
points, images and labels. Serial-section MSI is 2.5D in that taxonomy.

Two tests here are tripwires rather than assertions about Thyra:
``test_polygon_z_still_breaks_spatial_queries`` and
``test_a_translation_would_not_have_worked``. If either fails, upstream
has changed and this decision is worth reopening -- that is a signal, not
a regression.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata import SpatialDataConverter

_DATASET_ID = "vol"

# Pairwise different, as elsewhere in the 3D tests: no axis length can stand in
# for another and make a wrong answer look right.
_N_X, _N_Y, _N_Z = 5, 3, 2

# 25x apart, so a depth placed with the in-plane pitch instead of the slice
# spacing is off by more than any rounding could explain.
_PIXEL_SIZE_UM = 10.0
_Z_SPACING_UM = 250.0

# Margin on the query box, in micrometres. Far larger than the dataset, so a
# footprint can only be missing because the query mishandled it.
_GENEROUS_MARGIN_UM = 1000.0


def _config(n_z: int = _N_Z) -> MockMSIConfig:
    """A full grid -- every voxel present, so obs covers the whole volume."""
    return MockMSIConfig(
        n_x=_N_X,
        n_y=_N_Y,
        n_z=n_z,
        n_mz_bins=64,
        peaks_per_spectrum=(4, 8),
        sparsity=0.0,
    )


def _convert_volume(tmp_path, n_z: int = _N_Z, **kwargs: Any):
    output_path = tmp_path / "out.zarr"
    converter = SpatialDataConverter(
        MockMSIReader(_config(n_z)),
        output_path,
        handle_3d=True,
        dataset_id=_DATASET_ID,
        pixel_size_um=_PIXEL_SIZE_UM,
        z_spacing_um=_Z_SPACING_UM,
        **kwargs,
    )
    assert converter.convert() is True
    return output_path


@pytest.fixture(scope="module")
def volume(tmp_path_factory) -> Dict[str, Any]:
    """One converted volume, read back off disk."""
    import spatialdata

    store = _convert_volume(tmp_path_factory.mktemp("shapes_z"))
    sdata = spatialdata.read_zarr(str(store))
    return {
        "sdata": sdata,
        "shapes": sdata.shapes[f"{_DATASET_ID}_pixels"],
        "obs": sdata.tables[_DATASET_ID].obs,
        "image": sdata.images[f"{_DATASET_ID}_tic"],
    }


def _enclosing_box(shapes):
    """A box around every footprint, with room to spare, as (min, max)."""
    bounds = shapes.geometry.bounds
    return (
        [
            float(bounds.minx.min()) - _GENEROUS_MARGIN_UM,
            float(bounds.miny.min()) - _GENEROUS_MARGIN_UM,
        ],
        [
            float(bounds.maxx.max()) + _GENEROUS_MARGIN_UM,
            float(bounds.maxy.max()) + _GENEROUS_MARGIN_UM,
        ],
    )


def test_volume_footprints_are_flat(volume):
    """The decision, stated directly.

    One footprint per pixel across every slice, none of them carrying a z
    ordinate. The count matters as much as the flatness: dropping z must
    not also drop a slice.
    """
    shapes = volume["shapes"]

    assert len(shapes) == _N_X * _N_Y * _N_Z
    assert not any(geometry.has_z for geometry in shapes.geometry)


def test_a_whole_dataset_query_returns_every_footprint(volume):
    """The regression this arrangement exists for.

    Under POLYGON Z this returned 26 of 30 shapes and 26 of 30 table rows
    for a box enclosing everything, silently. No exception, no warning --
    a user filtering a volume to a region simply got fewer pixels than
    they asked for, with nothing to indicate it.
    """
    from spatialdata import bounding_box_query

    sdata, shapes = volume["sdata"], volume["shapes"]
    minimum, maximum = _enclosing_box(shapes)

    result = bounding_box_query(
        sdata,
        axes=("x", "y"),
        min_coordinate=minimum,
        max_coordinate=maximum,
        target_coordinate_system="global",
    )

    assert result is not None, "a box enclosing the data returned nothing"
    assert len(result.shapes[f"{_DATASET_ID}_pixels"]) == len(shapes)
    assert len(result.tables[_DATASET_ID]) == len(shapes)


def test_the_depth_is_still_recorded_on_the_table_and_the_image(volume):
    """Flat geometry must not mean the depth is gone from the store.

    ``obs["spatial_z"]`` and the image's ``Scale`` are where it lives now,
    and they have to keep agreeing with each other: a consumer reads one
    or the other and must land in the same place.
    """
    from spatialdata.transformations import get_transformation

    obs, image = volume["obs"], volume["image"]

    axes = ("c", "z", "y", "x")
    matrix = get_transformation(image, "global").to_affine_matrix(
        input_axes=axes, output_axes=axes
    )
    z_scale = float(matrix[1, 1])
    assert z_scale == pytest.approx(_Z_SPACING_UM)

    n_z_planes = np.asarray(image.data).shape[1]
    assert n_z_planes == _N_Z

    depths = sorted(set(float(v) for v in obs["spatial_z"]))
    assert depths == [z * _Z_SPACING_UM for z in range(_N_Z)]

    # The table's deepest pixel and the image's last plane are the same place.
    assert max(depths) == pytest.approx((n_z_planes - 1) * z_scale)

    # And each row's depth is its own slice index, not plane 0 for everyone.
    for label in obs.index:
        z_index = int(obs.loc[label, "z"])
        assert float(obs.loc[label, "spatial_z"]) == pytest.approx(
            z_index * _Z_SPACING_UM
        )


def test_the_footprint_is_still_pixel_sized(volume):
    """The in-plane extent is the pitch, unchanged by any of this."""
    geometry = volume["shapes"].geometry.iloc[0]
    xs = [c[0] for c in geometry.exterior.coords]
    ys = [c[1] for c in geometry.exterior.coords]

    assert max(xs) - min(xs) == pytest.approx(_PIXEL_SIZE_UM)
    assert max(ys) - min(ys) == pytest.approx(_PIXEL_SIZE_UM)


def test_the_per_slice_tables_are_unchanged(tmp_path):
    """The per-slice layout writes flat elements and must keep doing so.

    It emits one table and one image per plane, so each element genuinely has
    no third dimension. A z here would be inventing one.
    """
    import spatialdata

    output_path = tmp_path / "flat.zarr"
    converter = SpatialDataConverter(
        MockMSIReader(_config()),
        output_path,
        handle_3d=False,
        dataset_id=_DATASET_ID,
        pixel_size_um=_PIXEL_SIZE_UM,
    )
    assert converter.convert() is True

    sdata = spatialdata.read_zarr(str(output_path))
    for key, shapes in sdata.shapes.items():
        assert not any(
            geometry.has_z for geometry in shapes.geometry
        ), f"{key} grew a z coordinate"


def test_polygon_z_still_breaks_spatial_queries():
    """Tripwire: the measured reason the footprints are flat.

    Builds the geometry 7317792 shipped -- flat squares at two different
    depths -- and queries a box that encloses all of them in x and y. Some
    come back missing. ``spatialdata``'s own ``ShapesModel.validate`` warns
    that a 3-dimensional geometry column "could led to unexpected
    behaviors"; this is that behaviour, and it is silent.

    If this test fails, upstream now handles ``POLYGON Z`` in queries and
    the flat-footprint decision is worth reopening. That is the point of
    the test: it is a signal, not a regression.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon
    from spatialdata import SpatialData, bounding_box_query
    from spatialdata.models import ShapesModel
    from spatialdata.transformations import Identity

    half = _PIXEL_SIZE_UM / 2
    geometries = []
    for z_index in range(_N_Z):
        z = z_index * _Z_SPACING_UM
        for y_index in range(_N_Y):
            for x_index in range(_N_X):
                x = x_index * _PIXEL_SIZE_UM
                y = y_index * _PIXEL_SIZE_UM
                geometries.append(
                    Polygon(
                        [
                            (x - half, y - half, z),
                            (x + half, y - half, z),
                            (x + half, y + half, z),
                            (x - half, y + half, z),
                        ]
                    )
                )

    gdf = gpd.GeoDataFrame(
        geometry=geometries, index=[str(i) for i in range(len(geometries))]
    )
    with pytest.warns(UserWarning, match="dimensions"):
        parsed = ShapesModel.parse(gdf, transformations={"global": Identity()})

    sdata = SpatialData(shapes={"pixels": parsed})
    minimum, maximum = _enclosing_box(parsed)

    result = bounding_box_query(
        sdata,
        axes=("x", "y"),
        min_coordinate=minimum,
        max_coordinate=maximum,
        target_coordinate_system="global",
    )
    returned = 0 if result is None else len(result.shapes["pixels"])

    assert returned < len(geometries), (
        "a POLYGON Z bounding-box query now returns every footprint; "
        "spatialdata may have gained 3D shape support, so revisit whether "
        "the pixel footprints should carry their slice depth again"
    )


def test_a_translation_would_not_have_worked():
    """Tripwire: the other representation, also measured, also rejected.

    Keeping flat geometry and expressing the depth as a ``Translation`` is
    the natural reading of spatialdata's 2D shapes model, and it is the
    change someone will propose on seeing that the footprints are flat. It
    does not work, and it fails *silently*: the transform drops z and
    returns flat geometry, so the depth would live only in metadata.

    If a future spatialdata makes this work, this test fails -- and that is
    the signal to revisit, not a regression.
    """
    import geopandas as gpd
    import spatialdata
    from shapely.geometry import box
    from spatialdata.models import ShapesModel
    from spatialdata.transformations import Scale, Sequence, Translation

    gdf = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 1.0, 1.0)], index=["p0"])
    parsed = ShapesModel.parse(
        gdf,
        transformations={
            "global": Sequence(
                [
                    Scale([_PIXEL_SIZE_UM, _PIXEL_SIZE_UM], axes=("x", "y")),
                    Translation([2 * _Z_SPACING_UM], axes=("z",)),
                ]
            )
        },
    )

    transformed = spatialdata.transform(parsed, to_coordinate_system="global")
    geometry = transformed.geometry.iloc[0]

    # x/y scaling lands; the z translation vanishes without a word.
    assert max(c[0] for c in geometry.exterior.coords) == pytest.approx(_PIXEL_SIZE_UM)
    assert not geometry.has_z, (
        "spatialdata now carries z through a Translation on 2D shapes; "
        "revisit how a volume's footprints should record their depth"
    )
