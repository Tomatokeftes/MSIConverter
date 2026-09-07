"""The bounded optical-image path writes what the whole-page path wrote.

Three contracts, each pinned against the library code it replaces rather
than against a stored fixture:

* :func:`block_mean` / :func:`trim_excess` reproduce xarray's
  ``coarsen(boundary="trim", side="right").mean().astype(dtype)`` value for
  value, odd sizes and all dtypes an optical TIFF can carry included;
* :meth:`StreamedOpticalImage.placeholder` is structurally what
  ``Image2DModel.parse(..., scale_factors=...)`` builds: same levels, shapes,
  chunks, dtype, per-level transformations and coordinates;
* declaring the placeholder through ``SpatialData.write`` and then streaming
  the pixels leaves a store whose optical element -- metadata, chunk grid,
  every pixel of every level -- equals the one the parse-and-write route
  produces, and the store never sees the page whole.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("spatialdata")
tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

from spatialdata import SpatialData  # noqa: E402
from spatialdata.models import Image2DModel  # noqa: E402
from spatialdata.transformations import Scale, get_transformation  # noqa: E402

from thyra.converters.spatialdata.optical_image import (  # noqa: E402
    OpticalTiffSource,
    StreamedOpticalImage,
    band_rows,
    block_mean,
    level_shapes,
    trim_excess,
)


def _xarray_reference(block: np.ndarray, factor: int) -> np.ndarray:
    """What multiscale_spatial_image computes for one pyramid step."""
    image = xr.DataArray(block, dims=("c", "y", "x"))
    return (
        image.coarsen(dim={"y": factor, "x": factor}, boundary="trim", side="right")
        .mean()
        .astype(block.dtype)
        .values
    )


def _stream_reference(block: np.ndarray, factor: int) -> np.ndarray:
    """The same step the way stream_pixels computes it: trim, then block mean."""
    ey = trim_excess(block.shape[1], factor)
    ex = trim_excess(block.shape[2], factor)
    return block_mean(block[:, ey:, ex:], (1, factor, factor))


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.int16, np.uint32, np.float32, np.float64]
)
@pytest.mark.parametrize("shape", [(3, 8, 8), (1, 9, 7), (2, 15, 33), (3, 5, 6)])
@pytest.mark.parametrize("factor", [2, 3])
def test_block_mean_matches_xarray_coarsen(dtype, shape, factor):
    rng = np.random.default_rng(int(np.prod(shape)) * factor)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        block = rng.integers(info.min, info.max, size=shape, dtype=dtype)
    else:
        block = rng.random(size=shape).astype(dtype) * 300 - 100
    expected = _xarray_reference(block, factor)
    actual = _stream_reference(block, factor)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_trim_drops_the_leading_rows():
    """side="right" keeps the LAST full windows; the excess comes off the front."""
    block = np.arange(3 * 5 * 4, dtype=np.uint8).reshape(3, 5, 4)
    reference = _xarray_reference(block, 2)
    assert trim_excess(5, 2) == 1
    assert trim_excess(4, 2) == 0
    # Row 0 is dropped: the first output row averages rows 1 and 2.
    np.testing.assert_array_equal(
        reference[0, 0],
        block[0, 1:3, :].reshape(2, 2, 2).mean(axis=(0, 2)).astype(np.uint8),
    )
    np.testing.assert_array_equal(_stream_reference(block, 2), reference)


def test_block_mean_rejects_untrimmed_input():
    with pytest.raises(ValueError, match="not a multiple"):
        block_mean(np.zeros((1, 5, 4), dtype=np.uint8), (1, 2, 2))
    with pytest.raises(ValueError, match="window factors"):
        block_mean(np.zeros((1, 4, 4), dtype=np.uint8), (2, 2))


def test_level_shapes_floor_like_coarsen():
    assert level_shapes((3, 11748, 36736), [2, 2, 2], ("c", "y", "x")) == [
        (3, 11748, 36736),
        (3, 5874, 18368),
        (3, 2937, 9184),
        (3, 1468, 4592),
    ]
    assert level_shapes((1, 10, 10), [], ("c", "y", "x")) == [(1, 10, 10)]


def test_band_rows_tile_the_chunk_rows():
    # A whole chunk row fits: one band per chunk row.
    assert band_rows(row_bytes=1000, chunk_rows=4096, budget=10_000_000) == 4096
    # 110 KB rows (the 36736 x 3 uint8 brightfield): 2048 of them per 256 MiB.
    assert (
        band_rows(row_bytes=36736 * 3, chunk_rows=4096, budget=256 * 1024 * 1024)
        == 2048
    )
    # Never a divisor that straddles: 4096 rows only split into powers of two.
    assert band_rows(row_bytes=1 << 20, chunk_rows=4096, budget=3000 << 20) == 2048
    assert band_rows(row_bytes=1 << 30, chunk_rows=4096, budget=1) == 1
    with pytest.raises(ValueError):
        band_rows(row_bytes=0, chunk_rows=4096)


@pytest.fixture
def rgb_tiff(tmp_path: Path) -> Path:
    """An odd-sized RGB TIFF with strips of one row, like a FlexImaging export."""
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
    # A fully black band, so an all-fill chunk exercises write_empty_chunks.
    image[10:20, :, :] = 0
    path = tmp_path / "brightfield_0000.tiff"
    tifffile.imwrite(str(path), image, rowsperstrip=1, compression="lzw")
    return path


def _source_and_transform(path: Path):
    source = OpticalTiffSource.probe(path)
    assert source is not None
    transform = Scale([2.0, 0.5], axes=("x", "y"))
    return source, {"slide": transform, "global": transform}


def test_probe_reads_only_the_header(rgb_tiff: Path):
    source = OpticalTiffSource.probe(rgb_tiff)
    assert source is not None
    assert source.shape == (3, 37, 53)
    assert source.dtype == np.uint8
    assert source.streamable
    assert source.page_axes == "YXS"
    assert source.row_bytes == 53 * 3


def test_bands_decode_row_ranges(rgb_tiff: Path):
    source = OpticalTiffSource.probe(rgb_tiff)
    whole = np.moveaxis(tifffile.imread(str(rgb_tiff)), -1, 0)
    starts = []
    for start, band in source.bands(rows=16):
        starts.append(start)
        assert band.shape[0] == 3
        np.testing.assert_array_equal(band, whole[:, start : start + band.shape[1], :])
    assert starts == [0, 16, 32]


def test_placeholder_mirrors_image2dmodel_parse(rgb_tiff: Path):
    source, transformations = _source_and_transform(rgb_tiff)
    chunks = (1, 16, 16)
    scale_factors = [2, 2]
    streamed = StreamedOpticalImage(
        source=source,
        name="optical",
        chunks=chunks,
        scale_factors=scale_factors,
        transformations=transformations,
    )
    placeholder = streamed.placeholder()

    whole = np.moveaxis(tifffile.imread(str(rgb_tiff)), -1, 0)
    reference = Image2DModel.parse(
        xr.DataArray(
            whole,
            dims=("c", "y", "x"),
            coords={"c": np.arange(3), "y": np.arange(37), "x": np.arange(53)},
        ),
        transformations=dict(transformations),
        chunks=chunks,
        scale_factors=scale_factors,
    )

    assert list(placeholder.children) == list(reference.children)
    for level in reference.children:
        ours = placeholder[level]["image"]
        theirs = reference[level]["image"]
        assert ours.dims == theirs.dims
        assert ours.shape == theirs.shape
        assert ours.dtype == theirs.dtype
        assert ours.data.chunks == theirs.data.chunks
        for axis in ("c", "y", "x"):
            np.testing.assert_array_equal(
                ours.coords[axis].values, theirs.coords[axis].values
            )
        for system in transformations:
            assert get_transformation(ours, system) == get_transformation(
                theirs, system
            )


def _read_element(store: Path, name: str):
    group = zarr.open_group(str(store), mode="r", use_consolidated=False)[
        f"images/{name}"
    ]
    attrs = group.attrs.asdict()
    datasets = attrs["ome"]["multiscales"][0]["datasets"]
    levels = {}
    for dataset in datasets:
        array = group[dataset["path"]]
        levels[dataset["path"]] = (array.metadata.to_dict(), array[:])
    files = sorted(
        str(f.relative_to(store / "images" / name))
        for f in (store / "images" / name).rglob("*")
        if f.is_file()
    )
    return attrs, levels, files


def test_declared_then_streamed_store_equals_parse_and_write(
    rgb_tiff: Path, tmp_path: Path
):
    source, transformations = _source_and_transform(rgb_tiff)
    chunks = (1, 16, 16)
    scale_factors = [2, 2]
    name = "optical"

    # The route this module replaces: whole page -> parse -> write.
    whole = np.moveaxis(tifffile.imread(str(rgb_tiff)), -1, 0)
    reference = Image2DModel.parse(
        xr.DataArray(
            whole,
            dims=("c", "y", "x"),
            coords={"c": np.arange(3), "y": np.arange(37), "x": np.arange(53)},
        ),
        transformations=dict(transformations),
        chunks=chunks,
        scale_factors=scale_factors,
    )
    expected_store = tmp_path / "expected.zarr"
    SpatialData(images={name: reference}).write(expected_store)

    # Declare, then stream.
    streamed = StreamedOpticalImage(
        source=source,
        name=name,
        chunks=chunks,
        scale_factors=scale_factors,
        transformations=transformations,
    )
    actual_store = tmp_path / "actual.zarr"
    SpatialData(images={name: streamed.placeholder()}).write(actual_store)
    declared_files = [
        f for f in (actual_store / "images" / name).rglob("*") if f.is_file()
    ]
    # Declaring stores metadata only: one zarr.json per level plus the group's.
    assert all(f.name == "zarr.json" for f in declared_files)
    assert len(declared_files) == len(scale_factors) + 2

    streamed.stream_pixels(actual_store)

    expected_attrs, expected_levels, expected_files = _read_element(
        expected_store, name
    )
    actual_attrs, actual_levels, actual_files = _read_element(actual_store, name)
    assert json.dumps(actual_attrs, sort_keys=True) == json.dumps(
        expected_attrs, sort_keys=True
    )
    assert list(actual_levels) == list(expected_levels) == ["s0", "s1", "s2"]
    for path in expected_levels:
        expected_meta, expected_pixels = expected_levels[path]
        actual_meta, actual_pixels = actual_levels[path]
        assert json.dumps(actual_meta, sort_keys=True, default=str) == json.dumps(
            expected_meta, sort_keys=True, default=str
        )
        np.testing.assert_array_equal(actual_pixels, expected_pixels)
    # Same chunk files, so all-fill chunks are skipped the same way too.
    assert actual_files == expected_files

    # And the pixels are the TIFF's.
    np.testing.assert_array_equal(actual_levels["s0"][1], whole)
    # Reading back through spatialdata still sees a valid multiscale image.
    sdata = SpatialData.read(str(actual_store))
    assert list(sdata.images[name].children) == ["scale0", "scale1", "scale2"]
    axes = ("x", "y")
    np.testing.assert_array_equal(
        get_transformation(sdata.images[name], "slide").to_affine_matrix(axes, axes),
        transformations["slide"].to_affine_matrix(axes, axes),
    )


def test_streaming_a_small_band_budget_still_matches(
    rgb_tiff: Path, tmp_path: Path, monkeypatch
):
    """Bands narrower than a chunk row (partial-chunk writes) change nothing."""
    from thyra.converters.spatialdata import optical_image

    monkeypatch.setattr(optical_image, "BAND_BUDGET_BYTES", 53 * 3 * 3)  # three rows
    monkeypatch.setattr(optical_image, "REDUCE_STRIP_ROWS", 2)
    source, transformations = _source_and_transform(rgb_tiff)
    streamed = StreamedOpticalImage(
        source=source,
        name="optical",
        chunks=(1, 16, 16),
        scale_factors=[2, 2],
        transformations=transformations,
    )
    store = tmp_path / "narrow.zarr"
    SpatialData(images={"optical": streamed.placeholder()}).write(store)
    streamed.stream_pixels(store)

    whole = np.moveaxis(tifffile.imread(str(rgb_tiff)), -1, 0)
    _, levels, _ = _read_element(store, "optical")
    np.testing.assert_array_equal(levels["s0"][1], whole)
    np.testing.assert_array_equal(levels["s1"][1], _stream_reference(whole, 2))
    np.testing.assert_array_equal(
        levels["s2"][1], _stream_reference(_stream_reference(whole, 2), 2)
    )


def test_stream_pixels_refuses_a_store_without_the_element(
    rgb_tiff: Path, tmp_path: Path
):
    source, transformations = _source_and_transform(rgb_tiff)
    streamed = StreamedOpticalImage(
        source=source,
        name="optical",
        chunks=(1, 16, 16),
        scale_factors=[2],
        transformations=transformations,
    )
    store = tmp_path / "wrong.zarr"
    # A different pyramid was declared under that name.
    other = StreamedOpticalImage(
        source=source,
        name="optical",
        chunks=(1, 16, 16),
        scale_factors=[2, 2],
        transformations=transformations,
    )
    SpatialData(images={"optical": other.placeholder()}).write(store)
    with pytest.raises(RuntimeError, match="pyramid levels"):
        streamed.stream_pixels(store)
    shutil.rmtree(store)
    with pytest.raises((KeyError, FileNotFoundError)):
        streamed.stream_pixels(store)


def test_grayscale_and_non_streamable_layouts(tmp_path: Path):
    rng = np.random.default_rng(3)
    gray = rng.integers(0, 65535, size=(21, 30), dtype=np.uint16)
    gray_path = tmp_path / "gray.tif"
    tifffile.imwrite(str(gray_path), gray)
    source = OpticalTiffSource.probe(gray_path)
    assert source.shape == (1, 21, 30)
    assert source.dtype == np.uint16
    bands = list(source.bands(rows=8))
    assert [start for start, _ in bands] == [0, 8, 16]
    np.testing.assert_array_equal(
        np.concatenate([b for _, b in bands], axis=1), gray[np.newaxis]
    )

    # Planar (separate samples) pages cannot be row-streamed; they decode whole,
    # reshaped the way the converter always reshaped a 3-d page.
    planar = rng.integers(0, 256, size=(3, 11, 13), dtype=np.uint8)
    planar_path = tmp_path / "planar.tif"
    tifffile.imwrite(
        str(planar_path), planar, photometric="rgb", planarconfig="separate"
    )
    source = OpticalTiffSource.probe(planar_path)
    assert not source.streamable
    bands = list(source.bands(rows=4))
    assert len(bands) == 1 and bands[0][0] == 0
    np.testing.assert_array_equal(bands[0][1], np.moveaxis(planar, -1, 0))


@pytest.mark.parametrize("route", ["2d", "streaming"])
def test_converters_stream_the_pixels_after_the_write(
    rgb_tiff: Path, tmp_path: Path, route
):
    """Both write paths declare the placeholder, then fill it: the store has the pixels."""
    from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
    from thyra.converters.spatialdata.spatialdata_2d_converter import (
        SpatialData2DConverter,
    )
    from thyra.converters.spatialdata.streaming_converter import (
        StreamingSpatialDataConverter,
    )

    reader = MockMSIReader(
        MockMSIConfig(n_x=4, n_y=3, n_mz_bins=32, peaks_per_spectrum=(2, 4)),
        optical_image_paths=[rgb_tiff],
    )
    output_path = tmp_path / f"{route}.zarr"
    if route == "2d":
        converter = SpatialData2DConverter(
            reader, output_path, dataset_id="ds", pixel_size_um=10.0
        )
    else:
        converter = StreamingSpatialDataConverter(
            reader, output_path, dataset_id="ds", pixel_size_um=10.0, use_csc=True
        )
    assert converter.convert() is True
    # Nothing is left waiting: every declared image was streamed.
    assert converter._pending_optical_images == []

    whole = np.moveaxis(tifffile.imread(str(rgb_tiff)), -1, 0)
    _, levels, _ = _read_element(output_path, "ds_optical_highres")
    # 37 x 53 is far below the 1000 px coarsest-level floor: a single level.
    assert list(levels) == ["s0"]
    np.testing.assert_array_equal(levels["s0"][1], whole)
    sdata = SpatialData.read(str(output_path))
    np.testing.assert_array_equal(sdata.images["ds_optical_highres"].values, whole)
