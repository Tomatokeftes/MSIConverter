"""Bounded-memory loading of optical TIFFs into a SpatialData store.

The optical images Thyra bundles with a conversion (FlexImaging brightfield,
typically a 10k x 40k RGB TIFF that decodes to 1-2 GB) used to travel the
same road as any other raster: decode the whole page into numpy, hand it to
``Image2DModel.parse`` with ``scale_factors``, and let ``SpatialData.write``
compute the pyramid. Measured on a 1.29 GB decode that road costs ~6.6 GB
over the conversion's baseline, for three reasons, none of them the image:

* dask's ``from_array`` copies every numpy array it is given (``x.copy()``
  in both ``dask.array.core.from_array`` and its expression-based twin, so
  no dask configuration avoids it), and ``Image2DModel.parse`` calls it for
  numpy input: the decoded page is in memory twice before anything is
  written;
* ``multiscale_spatial_image`` builds each pyramid level as
  ``coarsen(...).mean().astype(dtype)`` -- the mean promotes every 4096 x 4096
  block to float64 (128 MB per 16 MB of uint8) and the threaded scheduler
  runs one such block per core;
* all levels are computed in a single ``dask.compute``, so the scheduler is
  free to hold as many of those intermediates as it has threads.

This module keeps the store what that road produced while bounding memory
by one *band* of the source, not by the image:

1. **Declare, then stream.** The converter hands ``SpatialData``
   :meth:`StreamedOpticalImage.placeholder` -- the element with its final
   shapes, dtype, chunks and transformations, whose pixels are lazy zeros.
   spatialdata and ome-zarr create the arrays and write every piece of
   metadata exactly as before; zarr skips the all-zero chunks (its
   ``write_empty_chunks`` default), so declaring stores no chunk.
2. **Level 0 from the TIFF in bands.** :meth:`StreamedOpticalImage.stream_pixels`
   reads row bands through tifffile's zarr adapter, which decodes only the
   strips (or tile rows) a band needs, and writes them into the level-0
   array under :data:`BAND_BUDGET_BYTES`. Bands are whole strips: a page
   stored as one strip cannot be read in pieces, so it is decoded once,
   whole, as before.
3. **Each pyramid level from the one below, on disk.** Every level-*k* write
   unit is assembled one level-*k-1* chunk at a time, each reduced with the
   exact :func:`block_mean`, so a worker holds one source chunk, one
   accumulator and the chunk it is building whatever the image size.

The pixel values are those ``xarray``'s ``coarsen`` produces: same
``boundary="trim", side="right"`` rule (an odd-sized axis drops its FIRST
row or column), same float64 mean, same truncating cast back to the source
dtype. :func:`block_mean` is unit-tested against xarray for exactly that.

**Failure policy.** A TIFF that cannot be decoded used to be skipped with a
warning before anything was written; now the header is read up front and
the pixels only after the store exists, so
:meth:`BaseSpatialDataConverter._stream_pending_optical_pixels` keeps that
contract by dropping the declared element from the store and warning. A
store never keeps an image whose pixels were not written.
"""

from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import tifffile
import xarray as xr
import zarr
from spatialdata.models import Image2DModel
from spatialdata.transformations import set_transformation

logger = logging.getLogger(__name__)

#: Decoded bytes one band of the source TIFF may occupy while level 0 is
#: streamed in. Measured on a 36736 px wide RGB brightfield: a whole
#: 4096-row chunk row (451 MB) fits, so every level-0 chunk is encoded once;
#: a 256 MiB budget split it in two and cost a read-modify-write per chunk
#: (+0.7 s per chunk row) for a stage peak only 29 MB lower.
BAND_BUDGET_BYTES = 512 * 1024 * 1024

#: Output rows reduced at a time inside one source chunk. Bounds the
#: accumulator of the block mean to (rows x chunk width) values.
REDUCE_STRIP_ROWS = 512

#: Pyramid write units reduced concurrently. Each worker holds one source
#: chunk, one strip accumulator and the unit it is assembling.
REDUCE_WORKERS = 4

#: The layouts of a TIFF page this module reads: samples interleaved per
#: pixel (what FlexImaging exports and practically every RGB TIFF use),
#: a single sample, or one plane per sample.
PAGE_LAYOUTS = ("YXS", "YX", "SYX")

Shape = Tuple[int, int, int]


def block_mean(block: np.ndarray, factor: int) -> np.ndarray:
    """Mean over ``factor`` x ``factor`` windows of a ``(c, y, x)`` block.

    Reproduces ``xarray``'s ``coarsen(window).mean().astype(block.dtype)``
    value for value. Unsigned integers are summed exactly and floor-divided,
    which equals the truncated float64 mean for values that are never
    negative; signed integers take numpy's float64 mean (the accumulator
    dask picks for the same reduction); floating input keeps its own
    precision and skips NaNs the way xarray's ``nanmean`` does when any are
    present. The cast truncates the way ``ndarray.astype`` does.

    Args:
        block: ``(c, y, x)`` array whose ``y`` and ``x`` are multiples of
            ``factor``. Trim before calling; see :func:`trim_excess`.
        factor: Window edge along ``y`` and ``x``.

    Returns:
        Array of shape ``(c, y // factor, x // factor)`` and ``block.dtype``.
    """
    if block.ndim != 3:
        raise ValueError(f"block_mean expects a (c, y, x) block, got {block.ndim}-d")
    if factor < 1:
        raise ValueError(f"window factor must be positive, got {factor}")
    channels, rows, cols = block.shape
    if rows % factor or cols % factor:
        raise ValueError(
            f"block of shape {block.shape} is not a multiple of window {factor}"
        )
    if block.dtype.kind in "ub":
        # Exact integer path: the float64 sum of these values is exact and
        # /n truncates to the floor, so floor(sum / n) is the same number.
        n = factor * factor
        acc_dtype = (
            np.uint32 if int(np.iinfo(block.dtype).max) * n < 2**32 else np.uint64
        )
        if block.dtype.kind == "b":
            acc_dtype = np.uint32
        acc = np.zeros((channels, rows // factor, cols // factor), dtype=acc_dtype)
        for dy in range(factor):
            for dx in range(factor):
                acc += block[:, dy::factor, dx::factor]
        return (acc // n).astype(block.dtype)
    windows = block.reshape(channels, rows // factor, factor, cols // factor, factor)
    if block.dtype.kind in "fc" and np.isnan(windows).any():
        return np.nanmean(windows, axis=(2, 4)).astype(block.dtype)
    return np.mean(windows, axis=(2, 4)).astype(block.dtype)


def trim_excess(size: int, factor: int) -> int:
    """Leading rows/columns xarray's ``coarsen`` drops for ``boundary="trim"``.

    With ``side="right"`` the excess comes off the FRONT of the axis, so the
    last window always ends on the last element. This is the rule
    ``multiscale_spatial_image`` uses for every pyramid level.
    """
    return size - (size // factor) * factor


def level_shapes(shape: Shape, scale_factors: Sequence[int]) -> List[Shape]:
    """``(c, y, x)`` of every pyramid level, level 0 first.

    Each factor divides the spatial axes of the level before it, flooring
    the way ``coarsen(boundary="trim")`` does. The channel axis is never
    reduced.
    """
    shapes = [(int(shape[0]), int(shape[1]), int(shape[2]))]
    for factor in scale_factors:
        c, y, x = shapes[-1]
        shapes.append((c, y // factor, x // factor))
    return shapes


def band_rows(
    row_bytes: int,
    chunk_rows: int,
    strip_rows: int = 1,
    budget: int = BAND_BUDGET_BYTES,
) -> int:
    """Rows per band: whole strips, within ``budget``, tiling the chunk rows.

    A strip (or tile row) is the smallest unit the TIFF decoder produces, so
    a band is always a whole number of them and never shorter than one --
    a single-strip page yields one band, the whole page, decoded once.
    Within that, the largest divisor of ``chunk_rows`` that fits is
    preferred so bands complete whole level-0 chunks instead of straddling
    them; when the strip height does not divide the chunk height no such
    divisor exists and the budget alone decides, which only means the
    chunks are completed by more than one band.
    """
    if row_bytes <= 0 or chunk_rows <= 0 or strip_rows <= 0:
        raise ValueError("row_bytes, chunk_rows and strip_rows must be positive")
    rows = max(strip_rows, (budget // row_bytes) // strip_rows * strip_rows)
    if rows >= chunk_rows and chunk_rows % strip_rows == 0:
        return chunk_rows
    for candidate in range(min(rows, chunk_rows), 0, -1):
        if chunk_rows % candidate == 0 and candidate % strip_rows == 0:
            return candidate
    return rows


def _write_unit(array: zarr.Array) -> Tuple[int, ...]:
    """The block one write of ``array`` should cover: its shard, else its chunk.

    ``zarr.Array.chunks`` is the inner chunk once an array is sharded, and
    concurrent writes inside one shard clobber each other, so the streaming
    and the reducer key on the shard whenever there is one.
    """
    return tuple(int(n) for n in (array.shards or array.chunks))


def _write_regions(array: zarr.Array) -> Iterator[Tuple[slice, slice, slice]]:
    """Every write unit of a ``(c, y, x)`` array as slices, C order."""
    unit = _write_unit(array)
    ranges = [range(0, size, step) for size, step in zip(array.shape, unit)]
    for starts in itertools.product(*ranges):
        c, y, x = (
            slice(start, min(start + step, size))
            for start, step, size in zip(starts, unit, array.shape)
        )
        yield c, y, x


@dataclass(frozen=True)
class OpticalTiffSource:
    """What the first page of an optical TIFF looks like, without decoding it.

    ``shape`` is always ``(c, y, x)``. ``page_axes`` is one of
    :data:`PAGE_LAYOUTS` and says how a decoded array maps onto that;
    ``strip_rows`` is the height of the page's strips or tiles, the smallest
    row range the decoder can produce on its own.
    """

    path: Path
    shape: Shape
    dtype: np.dtype
    page_axes: str
    strip_rows: int

    @classmethod
    def probe(cls, path: Union[str, Path]) -> "OpticalTiffSource":
        """Read the page header.

        Raises:
            ValueError: for a page layout other than :data:`PAGE_LAYOUTS`
                or a sample format tifffile cannot map to a dtype -- the
                cases the whole-page decode used to reject, at the same
                point, before anything is written.
        """
        path = Path(path)
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            page_shape = tuple(int(n) for n in page.shape)
            dtype = page.dtype
            axes = str(page.axes)
            chunks = tuple(int(n) for n in page.chunks)
        if dtype is None:
            raise ValueError(f"{path.name}: sample format not supported by tifffile")
        if axes == "YX":
            shape: Shape = (1, page_shape[0], page_shape[1])
            strip_rows = chunks[0]
        elif axes == "YXS":
            shape = (page_shape[2], page_shape[0], page_shape[1])
            strip_rows = chunks[0]
        elif axes == "SYX":
            shape = (page_shape[0], page_shape[1], page_shape[2])
            # tifffile reports planar strips as (rows, width).
            strip_rows = chunks[-2]
        else:
            raise ValueError(
                f"{path.name}: unsupported TIFF page layout {axes} {page_shape}"
            )
        return cls(
            path=path,
            shape=shape,
            dtype=np.dtype(dtype),
            page_axes=axes,
            strip_rows=max(1, min(strip_rows, shape[1])),
        )

    @property
    def row_bytes(self) -> int:
        """Decoded bytes of one full-width row across all channels."""
        return int(self.shape[0]) * int(self.shape[2]) * int(self.dtype.itemsize)

    def bands(self, rows: int) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(first_row, band)`` with ``band`` shaped ``(c, rows, x)``.

        Row ranges are decoded through tifffile's zarr adapter, so only the
        strips of a band are ever decoded. When ``rows`` covers the page the
        page is decoded once, whole, with tifffile's own parallel decoder.
        """
        n_rows = self.shape[1]
        with tifffile.TiffFile(self.path) as tif:
            page = tif.pages[0]
            if rows >= n_rows:
                yield 0, self._to_cyx(page.asarray())
                return
            source = zarr.open_array(store=page.aszarr(), mode="r")
            for start in range(0, n_rows, rows):
                stop = min(start + rows, n_rows)
                if self.page_axes == "SYX":
                    decoded = source[:, start:stop]
                else:
                    decoded = source[start:stop]
                yield start, self._to_cyx(decoded)

    def _to_cyx(self, decoded: np.ndarray) -> np.ndarray:
        if self.page_axes == "YX":
            return decoded[np.newaxis, :, :]
        if self.page_axes == "YXS":
            return np.moveaxis(decoded, -1, 0)
        return decoded


@dataclass
class StreamedOpticalImage:
    """One optical image: declared to SpatialData up front, pixels streamed after.

    Build it, put :meth:`placeholder` in the images the SpatialData write
    carries, and call :meth:`stream_pixels` on the store once that write
    has returned. Between the two the element on disk has all its metadata
    and no pixels; :meth:`discard` removes it again if the pixels cannot
    be streamed.
    """

    source: OpticalTiffSource
    name: str
    chunks: Tuple[int, ...]
    scale_factors: Sequence[int]
    transformations: Dict[str, Any]
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.chunks) != 3:
            raise ValueError(f"chunks must be (c, y, x), got {self.chunks}")
        self.scale_factors = [int(f) for f in self.scale_factors]

    @property
    def level_shapes(self) -> List[Shape]:
        """``(c, y, x)`` of every pyramid level of this image, level 0 first."""
        return level_shapes(self.source.shape, self.scale_factors)

    def placeholder(self) -> Union[xr.DataArray, xr.DataTree]:
        """The element as ``Image2DModel.parse`` would build it, pixels lazy zeros.

        With scale factors this is the multiscale DataTree ``parse`` builds
        for ``scale_factors``: same levels, chunks, per-level transformations
        and coordinates. Without, it is the single-scale DataArray ``parse``
        returns, so the store gets the single-scale writer it always had.
        Either way the write creates the arrays and metadata and stores no
        chunk.
        """
        dims = ("c", "y", "x")
        if not self.scale_factors:
            image = Image2DModel.parse(
                da.zeros(
                    self.source.shape, chunks=self.chunks, dtype=self.source.dtype
                ),
                dims=dims,
                transformations=dict(self.transformations),
                chunks=self.chunks,
            )
            image.attrs.update(self.attrs)
            return image
        base = self.level_shapes[0]
        levels: Dict[str, xr.Dataset] = {}
        for index, shape in enumerate(self.level_shapes):
            coords: Dict[str, Any] = {"c": np.arange(shape[0])}
            for axis, full, size in zip(dims[1:], base[1:], shape[1:]):
                # Pixel centres in level-0 units; spatialdata's own
                # compute_coordinates for a DataTree, verbatim.
                coords[axis] = np.linspace(0, full, size + 1)[:-1] + full / size / 2
            image = xr.DataArray(
                da.zeros(shape, chunks=self.chunks, dtype=self.source.dtype),
                dims=dims,
                coords=coords,
                attrs=dict(self.attrs),
            )
            levels[f"scale{index}"] = xr.Dataset({"image": image})
        tree = xr.DataTree.from_dict(levels)
        set_transformation(tree, dict(self.transformations), set_all=True)
        Image2DModel.validate(tree)
        return tree

    def stream_pixels(self, store_path: Union[str, Path]) -> None:
        """Fill the element's arrays in ``store_path`` with the real pixels.

        Requires the store to hold the element :meth:`placeholder` declared.
        Level 0 comes from the TIFF in bands; each further level from the
        level below, one write unit at a time.
        """
        arrays = self._open_level_arrays(store_path)
        self._stream_level0(arrays[0])
        for index, factor in enumerate(self.scale_factors, start=1):
            self._reduce_level(arrays[index - 1], arrays[index], factor)
        logger.info(
            f"Streamed optical image '{self.name}': level 0 plus "
            f"{len(self.scale_factors)} pyramid level"
            f"{'s' if len(self.scale_factors) != 1 else ''}"
        )

    def discard(self, store_path: Union[str, Path]) -> None:
        """Remove the declared element from ``store_path``, if it is there."""
        root = zarr.open_group(str(store_path), mode="r+", use_consolidated=False)
        images = root["images"]
        if self.name in images:
            del images[self.name]

    def _open_level_arrays(self, store_path: Union[str, Path]) -> List[zarr.Array]:
        root = zarr.open_group(str(store_path), mode="r+", use_consolidated=False)
        element = root[f"images/{self.name}"]
        try:
            datasets = element.attrs["ome"]["multiscales"][0]["datasets"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"images/{self.name} in {store_path} carries no multiscales metadata"
            ) from exc
        paths = [dataset["path"] for dataset in datasets]
        expected = self.level_shapes
        if len(paths) != len(expected):
            raise RuntimeError(
                f"images/{self.name} has {len(paths)} pyramid levels on disk, "
                f"expected {len(expected)}"
            )
        arrays: List[zarr.Array] = []
        for path, shape in zip(paths, expected):
            array = element[path]
            if not isinstance(array, zarr.Array) or tuple(array.shape) != tuple(shape):
                raise RuntimeError(
                    f"images/{self.name}/{path} is {getattr(array, 'shape', None)}, "
                    f"expected {shape}"
                )
            arrays.append(array)
        return arrays

    def _stream_level0(self, target: zarr.Array) -> None:
        chunk_rows = _write_unit(target)[1]
        # The budget is looked up here, not bound as a default, so a test
        # (or a caller) can narrow it.
        rows = band_rows(
            self.source.row_bytes,
            chunk_rows,
            self.source.strip_rows,
            budget=BAND_BUDGET_BYTES,
        )
        n_bands = -(-self.source.shape[1] // rows)
        logger.debug(
            f"  level 0 of '{self.name}': {n_bands} band(s) of {rows} rows "
            f"({rows * self.source.row_bytes / 1e6:.0f} MB decoded each)"
        )
        for start, band in self.source.bands(rows):
            target[:, start : start + band.shape[1], :] = band
            # The next band is decoded before this name is rebound; drop
            # it now so only one band is ever alive.
            del band

    def _reduce_level(
        self, source: zarr.Array, target: zarr.Array, factor: int
    ) -> None:
        """Write every unit of ``target`` from its footprint in ``source``.

        A unit's footprint is walked one source write unit at a time (an
        odd trim shifts the walk by the excess, so the last step of a row or
        column may touch a second source chunk), and each piece is reduced
        in strips of :data:`REDUCE_STRIP_ROWS` output rows.
        """
        excess_y = trim_excess(source.shape[1], factor)
        excess_x = trim_excess(source.shape[2], factor)
        _, src_rows, src_cols = _write_unit(source)
        out_rows_per_read = max(1, src_rows // factor)
        out_cols_per_read = max(1, src_cols // factor)
        regions = list(_write_regions(target))
        workers = max(1, min(REDUCE_WORKERS, os.cpu_count() or 1, len(regions)))

        def reduce_region(region: Tuple[slice, slice, slice]) -> None:
            c_sel, y_sel, x_sel = region
            unit = np.empty(
                (
                    c_sel.stop - c_sel.start,
                    y_sel.stop - y_sel.start,
                    x_sel.stop - x_sel.start,
                ),
                dtype=target.dtype,
            )
            for y0 in range(0, unit.shape[1], out_rows_per_read):
                y1 = min(y0 + out_rows_per_read, unit.shape[1])
                for x0 in range(0, unit.shape[2], out_cols_per_read):
                    x1 = min(x0 + out_cols_per_read, unit.shape[2])
                    piece = source[
                        c_sel,
                        excess_y
                        + (y_sel.start + y0) * factor : excess_y
                        + (y_sel.start + y1) * factor,
                        excess_x
                        + (x_sel.start + x0) * factor : excess_x
                        + (x_sel.start + x1) * factor,
                    ]
                    for r0 in range(0, y1 - y0, REDUCE_STRIP_ROWS):
                        r1 = min(r0 + REDUCE_STRIP_ROWS, y1 - y0)
                        unit[:, y0 + r0 : y0 + r1, x0:x1] = block_mean(
                            piece[:, r0 * factor : r1 * factor, :], factor
                        )
                    del piece
            target[region] = unit

        with ThreadPoolExecutor(max_workers=workers) as pool:
            # list() so a failing unit raises here, not silently.
            list(pool.map(reduce_region, regions))
