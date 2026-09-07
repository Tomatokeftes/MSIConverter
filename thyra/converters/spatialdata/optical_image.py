"""Bounded-memory loading of optical TIFFs into a SpatialData store.

The optical images Thyra bundles with a conversion (FlexImaging brightfield,
typically a 10k x 40k RGB TIFF that decodes to 1-2 GB) used to travel the
same road as any other raster: decode the whole page into numpy, hand it to
``Image2DModel.parse`` with ``scale_factors``, and let ``SpatialData.write``
compute the pyramid. Measured on a 1.29 GB decode that road costs ~6.6 GB
over the conversion's baseline, for three reasons, none of them the image:

* dask's expression-based ``from_array`` copies every numpy array it is
  given (``x.copy()`` in ``dask.array._array_expr._collection.from_array``),
  so the decoded page is in memory twice before anything is written;
* ``multiscale_spatial_image`` builds each pyramid level as
  ``coarsen(...).mean().astype(dtype)`` -- the mean promotes every 4096 x 4096
  block to float64 (128 MB per 16 MB of uint8) and the threaded scheduler
  runs one such block per core;
* all levels are computed in a single ``dask.compute``, so the scheduler is
  free to hold as many of those intermediates as it has threads.

This module keeps the store byte-for-byte what that road produced while
bounding memory by one *band* of the source, not by the image:

1. **Declare, then stream.** The converter hands ``SpatialData`` a
   :meth:`StreamedOpticalImage.placeholder` -- a DataTree with the right
   shapes, dtype, chunks and transformations per level whose pixels are
   lazy zeros.
   spatialdata and ome-zarr create the arrays and write every piece of
   metadata exactly as before; zarr skips the all-zero chunks (its
   ``write_empty_chunks`` default), so declaring costs nothing on disk.
2. **Level 0 from the TIFF in bands.** :meth:`StreamedOpticalImage.stream_pixels`
   opens the page through tifffile's zarr adapter, which decodes only the
   strips a row range needs, and writes the bands into the level-0 array
   under :data:`BAND_BUDGET_BYTES`.
3. **Each pyramid level from the one below, on disk.** Every level-*k* chunk
   is the exact block mean of its level-*k-1* footprint, read back from the
   store and reduced in strips (:func:`block_mean`), so a chunk's working set
   is tens of MB whatever the image size.

The pixel values are those ``xarray``'s ``coarsen`` produces: same
``boundary="trim", side="right"`` rule (an odd-sized axis drops its FIRST
row or column), same float64 mean, same truncating cast back to the source
dtype. :func:`block_mean` is unit-tested against xarray for exactly that.
"""

from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import tifffile
import xarray as xr
import zarr
from spatialdata.models import Image2DModel
from spatialdata.transformations import set_transformation

logger = logging.getLogger(__name__)

#: Decoded bytes one band of the source TIFF may occupy while level 0 is
#: streamed in. Bands tile the level-0 chunk rows (never straddle them), so
#: a narrow budget only means more partial-chunk writes, never a wrong one.
BAND_BUDGET_BYTES = 256 * 1024 * 1024

#: Output rows reduced at a time when a pyramid chunk is assembled from the
#: level below. Bounds the float64 intermediate of the block mean.
REDUCE_STRIP_ROWS = 512

#: Pyramid chunks reduced concurrently. Each holds its level-below footprint
#: (four source chunks), the chunk being assembled and one strip's float64
#: mean, so this is the multiplier on that working set.
REDUCE_WORKERS = 4

Axes = Tuple[str, ...]
Shape = Tuple[int, ...]


def block_mean(block: np.ndarray, factors: Sequence[int]) -> np.ndarray:
    """Mean over non-overlapping windows, cast back to ``block``'s dtype.

    Reproduces ``xarray``'s ``coarsen(window).mean().astype(block.dtype)``
    value for value: integer input is averaged in float64 (numpy's own
    ``mean`` accumulator for integers, the dtype dask picks for the same
    reduction), floating input in its own precision with NaNs skipped as
    xarray's ``nanmean`` does, and the cast truncates the way
    ``ndarray.astype`` does.

    Args:
        block: Array whose every axis is a multiple of its window. Trim
            before calling; see :func:`trim_excess`.
        factors: Window per axis. ``1`` leaves an axis alone.

    Returns:
        Array of shape ``block.shape // factors`` and ``block.dtype``.
    """
    if len(factors) != block.ndim:
        raise ValueError(f"{len(factors)} window factors for a {block.ndim}-d block")
    shape: List[int] = []
    reduce_axes: List[int] = []
    for axis, (size, factor) in enumerate(zip(block.shape, factors)):
        if factor < 1 or size % factor:
            raise ValueError(
                f"axis {axis} of size {size} is not a multiple of window {factor}"
            )
        shape.extend((size // factor, factor))
        reduce_axes.append(2 * axis + 1)
    windows = block.reshape(shape)
    mean_of = np.nanmean if block.dtype.kind in "fc" else np.mean
    return mean_of(windows, axis=tuple(reduce_axes)).astype(block.dtype)


def trim_excess(size: int, factor: int) -> int:
    """Leading rows/columns xarray's ``coarsen`` drops for ``boundary="trim"``.

    With ``side="right"`` the excess comes off the FRONT of the axis, so the
    last window always ends on the last element. This is the rule
    ``multiscale_spatial_image`` uses for every pyramid level.
    """
    return size - (size // factor) * factor


def level_shapes(shape: Shape, scale_factors: Sequence[int], axes: Axes) -> List[Shape]:
    """Shape of every pyramid level, level 0 first.

    Each factor halves (or whatever the factor is) the spatial axes of the
    level before it, flooring the way ``coarsen(boundary="trim")`` does. The
    channel axis is never reduced.
    """
    shapes = [tuple(shape)]
    for factor in scale_factors:
        previous = shapes[-1]
        shapes.append(
            tuple(
                n // factor if ax in ("x", "y", "z") else n
                for n, ax in zip(previous, axes)
            )
        )
    return shapes


def band_rows(row_bytes: int, chunk_rows: int, budget: int = BAND_BUDGET_BYTES) -> int:
    """Rows per band so a decoded band fits ``budget`` and tiles the chunk rows.

    Returns ``chunk_rows`` whenever a whole chunk row of the source fits;
    otherwise the largest divisor of ``chunk_rows`` that does, so bands never
    straddle a chunk boundary and every chunk is completed by a contiguous
    run of bands.
    """
    if row_bytes <= 0 or chunk_rows <= 0:
        raise ValueError("row_bytes and chunk_rows must be positive")
    rows = max(1, budget // row_bytes)
    if rows >= chunk_rows:
        return chunk_rows
    for candidate in range(rows, 0, -1):
        if chunk_rows % candidate == 0:
            return candidate
    return 1


def _chunk_slices(array: zarr.Array) -> Iterator[Tuple[slice, ...]]:
    """Every chunk of ``array`` as a tuple of slices, C order."""
    ranges = [range(0, size, chunk) for size, chunk in zip(array.shape, array.chunks)]
    for starts in itertools.product(*ranges):
        yield tuple(
            slice(start, min(start + chunk, size))
            for start, chunk, size in zip(starts, array.chunks, array.shape)
        )


@dataclass(frozen=True)
class OpticalTiffSource:
    """What the first page of an optical TIFF looks like, without decoding it.

    ``shape`` is always ``(c, y, x)``. ``streamable`` is True when the page
    is stored ``YX`` or ``YXS`` (samples interleaved per pixel, the layout
    every FlexImaging export and practically every RGB TIFF uses), which is
    what lets a row range be decoded on its own. Any other layout is
    decoded whole, exactly as before, and reshaped the same way.
    """

    path: Path
    shape: Shape
    dtype: np.dtype
    streamable: bool
    page_axes: str

    @classmethod
    def probe(cls, path: Union[str, Path]) -> Optional["OpticalTiffSource"]:
        """Read the page header. ``None`` when the page is not 2-D or 3-D."""
        path = Path(path)
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            page_shape = tuple(int(n) for n in page.shape)
            dtype = np.dtype(page.dtype)
            axes = str(page.axes)
        if len(page_shape) == 2:
            shape: Shape = (1, page_shape[0], page_shape[1])
        elif len(page_shape) == 3:
            # Trailing axis is the channel axis, as the moveaxis(-1, 0) the
            # converter always applied assumed.
            shape = (page_shape[2], page_shape[0], page_shape[1])
        else:
            return None
        return cls(
            path=path,
            shape=shape,
            dtype=dtype,
            streamable=axes in ("YX", "YXS"),
            page_axes=axes,
        )

    @property
    def row_bytes(self) -> int:
        """Decoded bytes of one full-width row across all channels."""
        return int(self.shape[0]) * int(self.shape[2]) * int(self.dtype.itemsize)

    def bands(self, rows: int) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(first_row, band)`` with ``band`` shaped ``(c, rows, x)``.

        Streams row ranges through tifffile's zarr adapter when the layout
        allows it, so only the strips of a band are ever decoded; otherwise
        decodes the page once and yields it as a single band.
        """
        with tifffile.TiffFile(self.path) as tif:
            page = tif.pages[0]
            if not self.streamable:
                yield 0, self._to_cyx(page.asarray())
                return
            source = zarr.open_array(store=page.aszarr(), mode="r")
            n_rows = self.shape[1]
            for start in range(0, n_rows, rows):
                stop = min(start + rows, n_rows)
                yield start, self._to_cyx(source[start:stop])

    @staticmethod
    def _to_cyx(decoded: np.ndarray) -> np.ndarray:
        if decoded.ndim == 2:
            return decoded[np.newaxis, :, :]
        return np.moveaxis(decoded, -1, 0)


@dataclass
class StreamedOpticalImage:
    """One optical image: declared to SpatialData up front, pixels streamed after.

    Build it, put :meth:`placeholder` in the images the SpatialData write
    carries, and call :meth:`stream_pixels` on the store once that write
    has returned. Between the two the element on disk has all its metadata
    and no pixels.
    """

    source: OpticalTiffSource
    name: str
    chunks: Shape
    scale_factors: Sequence[int]
    transformations: Dict[str, Any]
    axes: Axes = ("c", "y", "x")
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The band and chunk arithmetic below indexes rows as axis 1 and
        # columns as axis 2; that is the (c, y, x) layout and nothing else.
        if tuple(self.axes) != ("c", "y", "x"):
            raise ValueError(f"optical images are (c, y, x), got axes {self.axes}")
        if len(self.chunks) != len(self.axes):
            raise ValueError(f"chunks {self.chunks} do not match axes {self.axes}")
        self.scale_factors = [int(f) for f in self.scale_factors]

    @property
    def level_shapes(self) -> List[Shape]:
        """Shape of every pyramid level of this image, level 0 first."""
        return level_shapes(self.source.shape, self.scale_factors, self.axes)

    def placeholder(self) -> xr.DataTree:
        """The multiscale element as ``Image2DModel.parse`` would shape it.

        Every level is lazy zeros with the store's chunking, so the write
        creates the arrays and metadata and stores no chunk. Structure,
        per-level transformations and coordinates mirror what ``parse``
        builds for ``scale_factors`` so the in-memory element is a valid
        spatialdata image as well.
        """
        base = self.level_shapes[0]
        levels: Dict[str, xr.Dataset] = {}
        for index, shape in enumerate(self.level_shapes):
            coords: Dict[str, Any] = {"c": np.arange(shape[0])}
            for axis, size in zip(self.axes, shape):
                if axis == "c":
                    continue
                full = base[self.axes.index(axis)]
                # Pixel centres in level-0 units; spatialdata's own
                # compute_coordinates for a DataTree, verbatim.
                coords[axis] = np.linspace(0, full, size + 1)[:-1] + full / size / 2
            image = xr.DataArray(
                da.zeros(shape, chunks=self.chunks, dtype=self.source.dtype),
                dims=self.axes,
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
        level below, chunk by chunk.
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

    def _open_level_arrays(self, store_path: Union[str, Path]) -> List[zarr.Array]:
        root = zarr.open_group(str(store_path), mode="r+", use_consolidated=False)
        element = root[f"images/{self.name}"]
        attrs = element.attrs.asdict()
        multiscales = attrs.get("ome", {}).get("multiscales") or attrs.get(
            "multiscales"
        )
        if not multiscales:
            raise RuntimeError(
                f"images/{self.name} in {store_path} carries no multiscales metadata"
            )
        paths = [dataset["path"] for dataset in multiscales[0]["datasets"]]
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
        chunk_rows = int(target.chunks[1])
        rows = band_rows(self.source.row_bytes, chunk_rows)
        n_bands = -(-self.source.shape[1] // rows)
        logger.debug(
            f"  level 0 of '{self.name}': {n_bands} band(s) of {rows} rows "
            f"({rows * self.source.row_bytes / 1e6:.0f} MB decoded each)"
        )
        for start, band in self.source.bands(rows):
            target[:, start : start + band.shape[1], :] = band
            del band

    def _reduce_level(
        self, source: zarr.Array, target: zarr.Array, factor: int
    ) -> None:
        factors = tuple(factor if ax in ("x", "y", "z") else 1 for ax in self.axes)
        excess = tuple(trim_excess(size, f) for size, f in zip(source.shape, factors))
        chunks = list(_chunk_slices(target))
        workers = max(1, min(REDUCE_WORKERS, os.cpu_count() or 1, len(chunks)))

        def reduce_chunk(out_sel: Tuple[slice, ...]) -> None:
            in_sel = tuple(
                slice(e + s.start * f, e + s.stop * f)
                for s, e, f in zip(out_sel, excess, factors)
            )
            footprint = source[in_sel]
            chunk = np.empty(
                tuple(s.stop - s.start for s in out_sel), dtype=target.dtype
            )
            n_rows = chunk.shape[1]
            for row in range(0, n_rows, REDUCE_STRIP_ROWS):
                stop = min(row + REDUCE_STRIP_ROWS, n_rows)
                strip = footprint[:, row * factors[1] : stop * factors[1], :]
                chunk[:, row:stop, :] = block_mean(strip, factors)
            del footprint
            target[out_sel] = chunk

        with ThreadPoolExecutor(max_workers=workers) as pool:
            # list() so a failing chunk raises here, not silently.
            list(pool.map(reduce_chunk, chunks))
