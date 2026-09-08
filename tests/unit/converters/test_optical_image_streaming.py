"""The bounded optical-image path writes what the whole-page path wrote.

Three contracts, each pinned against the library code it replaces rather
than against a stored fixture:

* :func:`block_mean` / :func:`trim_excess` reproduce xarray's
  ``coarsen(boundary="trim", side="right").mean().astype(dtype)`` value for
  value, odd sizes and all dtypes an optical TIFF can carry included;
* :meth:`StreamedOpticalImage.placeholder` is structurally what
  ``Image2DModel.parse`` builds, with and without ``scale_factors``: same
  levels, shapes, chunks, dtype, per-level transformations and coordinates;
* declaring the placeholder through ``SpatialData.write`` and then streaming
  the pixels leaves a store whose optical element -- metadata, chunk grid,
  every pixel of every level -- equals the one the parse-and-write route
  produces, and the store never sees the page whole.

Plus the tolerance the old route had: a TIFF whose pixels cannot be read
is dropped with a warning, not fatal, on both converter routes.
"""

from __future__ import annotations

import json
import logging
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

from thyra.converters.spatialdata import optical_image  # noqa: E402
from thyra.converters.spatialdata.optical_image import (  # noqa: E402
    OpticalTiffSource,
    StreamedOpticalImage,
    band_rows,
    block_mean,
    level_shapes,
    trim_excess,
)

CHUNKS = (1, 16, 16)
TRANSFORM = Scale([2.0, 0.5], axes=("x", "y"))
TRANSFORMATIONS = {"slide": TRANSFORM, "global": TRANSFORM}


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
    return block_mean(block[:, ey:, ex:], factor)


def _parse_reference(whole: np.ndarray, scale_factors):
    """The route this module replaces: whole page -> Image2DModel.parse."""
    kwargs = {"transformations": dict(TRANSFORMATIONS), "chunks": CHUNKS}
    if scale_factors:
        kwargs["scale_factors"] = scale_factors
    return Image2DModel.parse(
        xr.DataArray(
            whole,
            dims=("c", "y", "x"),
            coords={
                "c": np.arange(whole.shape[0]),
                "y": np.arange(whole.shape[1]),
                "x": np.arange(whole.shape[2]),
            },
        ),
        **kwargs,
    )


def _streamed(source: OpticalTiffSource, scale_factors, name="optical"):
    return StreamedOpticalImage(
        source=source,
        name=name,
        chunks=CHUNKS,
        scale_factors=scale_factors,
        transformations=dict(TRANSFORMATIONS),
    )


def _cyx(path: Path) -> np.ndarray:
    decoded = tifffile.imread(str(path))
    return decoded[np.newaxis] if decoded.ndim == 2 else np.moveaxis(decoded, -1, 0)


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.int16, np.uint32, np.int32, np.float32, np.float64],
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


def test_block_mean_extremes_and_nans():
    # Saturated unsigned windows: the exact accumulator must not overflow.
    full = np.full((2, 6, 4), 255, dtype=np.uint8)
    np.testing.assert_array_equal(
        _stream_reference(full, 2), _xarray_reference(full, 2)
    )
    full32 = np.full((1, 4, 6), np.iinfo(np.uint32).max, dtype=np.uint32)
    np.testing.assert_array_equal(
        _stream_reference(full32, 3), _xarray_reference(full32, 3)
    )
    # NaNs are skipped the way xarray's nanmean skips them.
    floats = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    floats[0, 0, 0] = np.nan
    floats[1, 2:4, 2:4] = np.nan
    expected = _xarray_reference(floats, 2)
    actual = _stream_reference(floats, 2)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(
        actual[~np.isnan(actual)], expected[~np.isnan(expected)]
    )


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
        block_mean(np.zeros((1, 5, 4), dtype=np.uint8), 2)
    with pytest.raises(ValueError, match="\\(c, y, x\\)"):
        block_mean(np.zeros((4, 4), dtype=np.uint8), 2)


def test_level_shapes_floor_like_coarsen():
    assert level_shapes((3, 11748, 36736), [2, 2, 2]) == [
        (3, 11748, 36736),
        (3, 5874, 18368),
        (3, 2937, 9184),
        (3, 1468, 4592),
    ]
    assert level_shapes((1, 10, 10), []) == [(1, 10, 10)]


def test_band_rows_are_whole_strips_that_tile_the_chunk_rows():
    row = 36736 * 3  # the 110 KB rows of the real brightfield
    # A whole chunk row fits the budget: one band per chunk row.
    assert band_rows(1000, 4096, budget=10_000_000) == 4096
    assert band_rows(row, 4096, strip_rows=1, budget=512 << 20) == 4096
    # Half a chunk row fits: the largest divisor, still a whole number of strips.
    assert band_rows(row, 4096, strip_rows=1, budget=256 << 20) == 2048
    assert band_rows(row, 4096, strip_rows=8, budget=256 << 20) == 2048
    # Strips that do not divide the chunk: whole strips within the budget.
    assert band_rows(row, 4096, strip_rows=1000, budget=256 << 20) == 2000
    # A single-strip page (rowsperstrip == height): one band, the whole page.
    assert band_rows(row, 4096, strip_rows=11748, budget=256 << 20) == 11748
    # Never less than one strip, even when a strip is over budget.
    assert band_rows(1 << 30, 4096, strip_rows=64, budget=1) == 64
    assert band_rows(1 << 30, 4096, strip_rows=1, budget=1) == 1
    with pytest.raises(ValueError):
        band_rows(0, 4096)


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


def test_probe_reads_only_the_header(rgb_tiff: Path):
    source = OpticalTiffSource.probe(rgb_tiff)
    assert source.shape == (3, 37, 53)
    assert source.dtype == np.uint8
    assert source.page_axes == "YXS"
    assert source.strip_rows == 1
    assert source.row_bytes == 53 * 3


def test_bands_decode_row_ranges(rgb_tiff: Path):
    source = OpticalTiffSource.probe(rgb_tiff)
    whole = _cyx(rgb_tiff)
    starts = []
    for start, band in source.bands(rows=16):
        starts.append(start)
        assert band.shape[0] == 3
        np.testing.assert_array_equal(band, whole[:, start : start + band.shape[1], :])
    assert starts == [0, 16, 32]


def test_probe_reports_strip_and_tile_heights(tmp_path: Path):
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(70, 90, 3), dtype=np.uint8)
    default = tmp_path / "default.tif"
    tifffile.imwrite(str(default), image)  # tifffile's default: one strip
    assert OpticalTiffSource.probe(default).strip_rows == 70
    strips = tmp_path / "strips.tif"
    tifffile.imwrite(str(strips), image, rowsperstrip=8)
    assert OpticalTiffSource.probe(strips).strip_rows == 8
    tiles = tmp_path / "tiles.tif"
    tifffile.imwrite(str(tiles), image, tile=(16, 16))
    assert OpticalTiffSource.probe(tiles).strip_rows == 16


def test_single_strip_page_is_decoded_once_whole(tmp_path: Path, monkeypatch):
    """A page stored as one strip cannot be read in pieces: one band, one decode."""
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(64, 40, 3), dtype=np.uint8)
    path = tmp_path / "single_strip.tif"
    tifffile.imwrite(str(path), image)
    source = OpticalTiffSource.probe(path)
    assert source.strip_rows == 64
    rows = band_rows(source.row_bytes, 16, source.strip_rows, budget=40 * 3 * 4)
    assert rows == 64
    calls = []
    original = tifffile.TiffPage.asarray

    def counting_asarray(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(tifffile.TiffPage, "asarray", counting_asarray)
    bands = list(source.bands(rows))
    assert [start for start, _ in bands] == [0]
    np.testing.assert_array_equal(bands[0][1], np.moveaxis(image, -1, 0))
    assert len(calls) == 1


def test_probe_rejects_what_the_decoder_would_have(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(3)
    volume = rng.integers(0, 256, size=(4, 16, 16), dtype=np.uint8)
    path = tmp_path / "volume.tif"
    # One ZYX page (ImageDepth), the kind of 3-d page that is not an image.
    tifffile.imwrite(str(path), volume, photometric="minisblack", volumetric=True)
    with pytest.raises(ValueError, match="layout"):
        OpticalTiffSource.probe(path)

    # tifffile reports a (SampleFormat, BitsPerSample) pair it cannot map as
    # dtype None; np.dtype(None) would silently be float64.
    class _UntypedPage:
        shape = (16, 16, 3)
        dtype = None
        axes = "YXS"
        chunks = (1, 16, 3)

    class _UntypedTiff:
        pages = [_UntypedPage()]

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(optical_image.tifffile, "TiffFile", _UntypedTiff)
    with pytest.raises(ValueError, match="sample format"):
        OpticalTiffSource.probe(path)


def test_grayscale_and_planar_layouts(tmp_path: Path):
    rng = np.random.default_rng(3)
    gray = rng.integers(0, 65535, size=(21, 30), dtype=np.uint16)
    gray_path = tmp_path / "gray.tif"
    tifffile.imwrite(str(gray_path), gray, rowsperstrip=4)
    source = OpticalTiffSource.probe(gray_path)
    assert source.shape == (1, 21, 30)
    assert source.dtype == np.uint16
    bands = list(source.bands(rows=8))
    assert [start for start, _ in bands] == [0, 8, 16]
    np.testing.assert_array_equal(
        np.concatenate([b for _, b in bands], axis=1), gray[np.newaxis]
    )

    # One plane per sample: already (c, y, x), and streamed by rows too.
    planar = rng.integers(0, 256, size=(3, 40, 13), dtype=np.uint8)
    planar_path = tmp_path / "planar.tif"
    tifffile.imwrite(
        str(planar_path),
        planar,
        photometric="rgb",
        planarconfig="separate",
        rowsperstrip=8,
    )
    source = OpticalTiffSource.probe(planar_path)
    assert source.page_axes == "SYX"
    assert source.shape == (3, 40, 13)
    assert source.strip_rows == 8
    bands = list(source.bands(rows=16))
    assert [start for start, _ in bands] == [0, 16, 32]
    np.testing.assert_array_equal(np.concatenate([b for _, b in bands], axis=1), planar)


@pytest.mark.parametrize("scale_factors", [[2, 2], []])
def test_placeholder_mirrors_image2dmodel_parse(rgb_tiff: Path, scale_factors):
    placeholder = _streamed(
        OpticalTiffSource.probe(rgb_tiff), scale_factors
    ).placeholder()
    reference = _parse_reference(_cyx(rgb_tiff), scale_factors)
    assert type(placeholder) is type(reference)

    if scale_factors:
        assert list(placeholder.children) == list(reference.children)
        pairs = [
            (placeholder[level]["image"], reference[level]["image"])
            for level in reference.children
        ]
    else:
        pairs = [(placeholder, reference)]
    for ours, theirs in pairs:
        assert ours.dims == theirs.dims
        assert ours.shape == theirs.shape
        assert ours.dtype == theirs.dtype
        assert ours.data.chunks == theirs.data.chunks
        for axis in ("c", "y", "x"):
            np.testing.assert_array_equal(
                ours.coords[axis].values, theirs.coords[axis].values
            )
        for system in TRANSFORMATIONS:
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


@pytest.mark.parametrize("scale_factors", [[2, 2], []])
def test_declared_then_streamed_store_equals_parse_and_write(
    rgb_tiff: Path, tmp_path: Path, scale_factors
):
    name = "optical"
    whole = _cyx(rgb_tiff)
    expected_store = tmp_path / "expected.zarr"
    SpatialData(images={name: _parse_reference(whole, scale_factors)}).write(
        expected_store
    )

    streamed = _streamed(OpticalTiffSource.probe(rgb_tiff), scale_factors, name)
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
    assert list(actual_levels) == list(expected_levels)
    assert list(actual_levels) == [f"s{i}" for i in range(len(scale_factors) + 1)]
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
    # Reading back through spatialdata still sees a valid image.
    sdata = SpatialData.read(str(actual_store))
    element = sdata.images[name]
    if scale_factors:
        assert list(element.children) == ["scale0", "scale1", "scale2"]
    else:
        assert isinstance(element, xr.DataArray)
    axes = ("x", "y")
    np.testing.assert_array_equal(
        get_transformation(element, "slide").to_affine_matrix(axes, axes),
        TRANSFORM.to_affine_matrix(axes, axes),
    )


def test_streaming_a_small_band_budget_still_matches(
    rgb_tiff: Path, tmp_path: Path, monkeypatch
):
    """Bands narrower than a chunk row (partial-chunk writes) change nothing."""
    monkeypatch.setattr(optical_image, "BAND_BUDGET_BYTES", 53 * 3 * 3)  # three rows
    monkeypatch.setattr(optical_image, "REDUCE_STRIP_ROWS", 2)
    source = OpticalTiffSource.probe(rgb_tiff)
    assert band_rows(source.row_bytes, 16, source.strip_rows, budget=53 * 3 * 3) == 2
    streamed = _streamed(source, [2, 2])
    store = tmp_path / "narrow.zarr"
    SpatialData(images={"optical": streamed.placeholder()}).write(store)
    bands_seen = []
    original = OpticalTiffSource.bands

    def counting_bands(self, rows):
        bands_seen.append(rows)
        return original(self, rows)

    monkeypatch.setattr(OpticalTiffSource, "bands", counting_bands)
    streamed.stream_pixels(store)
    assert bands_seen == [2]

    whole = _cyx(rgb_tiff)
    _, levels, _ = _read_element(store, "optical")
    np.testing.assert_array_equal(levels["s0"][1], whole)
    np.testing.assert_array_equal(levels["s1"][1], _stream_reference(whole, 2))
    np.testing.assert_array_equal(
        levels["s2"][1], _stream_reference(_stream_reference(whole, 2), 2)
    )


def test_stream_pixels_refuses_a_store_without_the_element(
    rgb_tiff: Path, tmp_path: Path
):
    source = OpticalTiffSource.probe(rgb_tiff)
    streamed = _streamed(source, [2])
    store = tmp_path / "wrong.zarr"
    # A different pyramid was declared under that name.
    SpatialData(images={"optical": _streamed(source, [2, 2]).placeholder()}).write(
        store
    )
    with pytest.raises(RuntimeError, match="pyramid levels"):
        streamed.stream_pixels(store)
    shutil.rmtree(store)
    with pytest.raises((KeyError, FileNotFoundError)):
        streamed.stream_pixels(store)


def test_discard_removes_the_declared_element(rgb_tiff: Path, tmp_path: Path):
    streamed = _streamed(OpticalTiffSource.probe(rgb_tiff), [2])
    store = tmp_path / "declared.zarr"
    SpatialData(images={"optical": streamed.placeholder()}).write(store)
    assert (store / "images" / "optical").exists()
    streamed.discard(store)
    assert not (store / "images" / "optical").exists()
    streamed.discard(store)  # idempotent
    zarr.consolidate_metadata(str(store))
    assert list(SpatialData.read(str(store)).images) == []


# ---- the converter routes ----------------------------------------------


def _convert(reader, output_path: Path):
    from thyra.converters.spatialdata.streaming_converter import (
        StreamingSpatialDataConverter,
    )

    converter = StreamingSpatialDataConverter(
        reader, output_path, dataset_id="ds", pixel_size_um=10.0, use_csc=True
    )
    return converter, converter.convert()


def _mock_reader(*optical_paths: Path):
    from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader

    return MockMSIReader(
        MockMSIConfig(n_x=4, n_y=3, n_mz_bins=32, peaks_per_spectrum=(2, 4)),
        optical_image_paths=list(optical_paths),
    )


def test_converters_stream_the_pixels_after_the_write(rgb_tiff: Path, tmp_path: Path):
    """Both write paths declare the placeholder, then fill it: the store has the pixels."""
    output_path = tmp_path / "rgb.zarr"
    converter, success = _convert(_mock_reader(rgb_tiff), output_path)
    assert success is True
    # Nothing is left waiting: every declared image was streamed.
    assert converter._pending_optical_images == {}

    whole = _cyx(rgb_tiff)
    _, levels, _ = _read_element(output_path, "ds_optical_highres")
    # 37 x 53 is far below the 1000 px coarsest-level floor: a single level.
    assert list(levels) == ["s0"]
    np.testing.assert_array_equal(levels["s0"][1], whole)
    sdata = SpatialData.read(str(output_path))
    np.testing.assert_array_equal(sdata.images["ds_optical_highres"].values, whole)


@pytest.fixture
def truncated_tiff(tmp_path: Path) -> Path:
    """Header intact, strips cut: probe succeeds, decoding fails."""
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, size=(40, 30, 3), dtype=np.uint8)
    path = tmp_path / "truncated_0000.tiff"
    tifffile.imwrite(str(path), image, rowsperstrip=1, compression="lzw")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    source = OpticalTiffSource.probe(path)
    assert source.shape == (3, 40, 30)
    return path


def test_unreadable_pixels_drop_the_image_not_the_conversion(
    truncated_tiff: Path, tmp_path: Path, caplog
):
    """A TIFF that cannot be decoded is skipped with a warning, as it always was."""
    output_path = tmp_path / "truncated.zarr"
    with caplog.at_level(logging.WARNING):
        converter, success = _convert(_mock_reader(truncated_tiff), output_path)
    assert success is True
    assert converter._pending_optical_images == {}
    assert any(
        "Failed to load optical image truncated_0000.tiff" in record.message
        for record in caplog.records
    )
    sdata = SpatialData.read(str(output_path))
    assert [k for k in sdata.images if "optical" in k] == []
    assert not (output_path / "images" / "ds_optical_highres").exists()
    # The rest of the store is intact and consolidated.
    assert "ds_z0_tic" in sdata.images
    assert (output_path / "zarr.json").exists()


def test_same_name_keeps_the_last_file(tmp_path: Path, caplog):
    """Two TIFFs mapping to one element name: last wins, like the dict always did."""
    rng = np.random.default_rng(5)
    first = tmp_path / "a_0000.tif"
    second = tmp_path / "b_0000.tif"
    tifffile.imwrite(str(first), rng.integers(0, 256, size=(20, 24, 3), dtype=np.uint8))
    last = rng.integers(0, 256, size=(18, 22, 3), dtype=np.uint8)
    tifffile.imwrite(str(second), last)
    output_path = tmp_path / "out.zarr"
    with caplog.at_level(logging.WARNING):
        converter, success = _convert(_mock_reader(first, second), output_path)
    assert success is True
    assert converter._pending_optical_images == {}
    assert any("is replaced by b_0000.tif" in r.message for r in caplog.records)
    _, levels, _ = _read_element(output_path, "ds_optical_highres")
    np.testing.assert_array_equal(levels["s0"][1], np.moveaxis(last, -1, 0))
