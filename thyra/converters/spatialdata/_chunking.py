"""Chunk/shard policy for the zarr Thyra writes: rasters and the MSI table.

Single source of truth for how Thyra chunks the raster images it writes to
SpatialData zarr -- and the designated seam for tile-aligned zarr v3 sharding
once scverse/spatialdata PR #1106 (``raster_write_kwargs``) lands.

Cross-repo contract: the viewer (Ousia) morphology tile server streams
``DEFAULT_TILE_SIZE`` (= 512) edge tiles. Once sharding lands, the zarr INNER
chunk edge must equal that so one served tile == one decompress. Thyra is a
separate package and CANNOT import Ousia, so the coupling is DUPLICATED BY
CONTRACT here -- keep ``INNER_TILE_EDGE`` numerically in sync with Ousia's
``DEFAULT_TILE_SIZE``; a silent divergence reintroduces cold-tile amplification
on the viewer. See the Ousia ADR docs/ADR-pyramid-tile-sharding.md.

The raster policy above is passed per-array. The TABLE policy below cannot be:
``SpatialData.write`` hands the table straight to ``anndata.write_elem`` and
exposes no ``dataset_kwargs`` seam, so the only supported lever is zarr's own
config. ``table_write_config()`` is that lever, and the one writer that
reaches zarr through spatialdata (``BaseSpatialDataConverter._save_output``,
which every table goes through) must hold it open across the write.
"""

from contextlib import contextmanager
from typing import Iterator, Tuple

import zarr

# Outer chunk edge used TODAY (pre-sharding the chunk IS the outer block): a 4096
# block means a viewer 512-tile read decompresses at most one chunk per request.
IMAGE_CHUNK_EDGE = 4096

# Future zarr INNER chunk edge once #1106 lands == Ousia DEFAULT_TILE_SIZE (512).
# Duplicated by contract (Thyra cannot import Ousia).
INNER_TILE_EDGE = 512

# Uncompressed byte budget for ONE SHARD of a table array -- X/data, X/indices,
# X/indptr and every obs/var column anndata writes. One shard is one FILE, so
# this is what keeps a 36M-nnz table (test_data/bellini.imzML) in 11 files
# instead of 394,699.
#
# anndata >= 0.13 turns on ``shards="auto"`` for zarr v3 and would otherwise
# pick its own 1 GB budget; it explicitly yields to a caller-set value (see
# ``anndata._io.specs.methods.zarr_v3_sharding``). We set a smaller one because
# a shard is assembled in memory before it is written -- anndata's own comment
# says so -- and Thyra converts multi-GB acquisitions on workstations, where a
# 1 GB write buffer is a real cost. 128 MiB is still enormous relative to the
# file counts that hurt: test_data/pea.imzML (60.1M nnz, 481 MB of X/data)
# lands in 4 data files, so even a table 100x that size stays in the hundreds.
#
# The INNER chunk stays zarr's ~1 MiB auto pick: that is the unit a random read
# decompresses, and it is the size the non-sharded path used before anndata
# began auto-sharding, so read amplification is unchanged.
TABLE_SHARD_TARGET_BYTES = 128 * 1024 * 1024

# zarr < 3.1.6 computed the auto chunk with ``max_bytes=1024`` where 1 MiB was
# meant (zarr-python#3603, first released in 3.1.6 -- it landed after both 3.1.4
# and 3.1.5). zarr < 3.1.4 additionally predates ``array.target_shard_size_bytes``,
# making the budget above unenforceable, and there ``shards="auto"`` produced
# ~1 KB SHARDS -- one file per KB of table. pyproject pins the floor at 3.1.6 for
# both reasons.
#
# This constant guards the shard, i.e. the file, so it only ever catches that
# second, pre-3.1.4 half: on 3.1.4/3.1.5 the budget above still lands large
# shards and it is the chunk inside them that collapses. MIN_TABLE_CHUNK_BYTES
# is the half that catches those two.
MIN_TABLE_SHARD_BYTES = 64 * 1024

# Minimum size for the INNER chunk of a table array -- the unit that sits inside
# a shard and the unit a random read decompresses. This is the half of the defect
# a large shard hides: on 3.1.4/3.1.5 the file count looks fine while each file
# quietly holds ~146,000 separately compressed chunks instead of ~142, and pays
# for all of them in the shard index. 64 KiB sits well above the ~800 B those
# versions hand the regression-test fixture and well below the ~200 KB 3.1.6
# hands it, so the two are never close to the threshold.
MIN_TABLE_CHUNK_BYTES = 64 * 1024


@contextmanager
def table_write_config() -> Iterator[None]:
    """Hold Thyra's table shard budget open across a ``SpatialData.write``.

    Wraps zarr's global config, so it must stay open for the *whole* write --
    the arrays are created lazily inside anndata, not up front.
    """
    with zarr.config.set({"array.target_shard_size_bytes": TABLE_SHARD_TARGET_BYTES}):
        yield


def image_chunks(spatial_ndim: int, edge: int = IMAGE_CHUNK_EDGE) -> Tuple[int, ...]:
    """The zarr chunk tuple for a Thyra-written raster image (leading channel axis).

    ``spatial_ndim == 2`` (c, y, x) -> ``(1, edge, edge)``;
    ``spatial_ndim == 3`` (c, z, y, x) -> ``(1, 1, edge, edge)``.

    This is the ONE place the future inner=``INNER_TILE_EDGE`` + outer-shard policy
    will live when spatialdata PR #1106 ships ``raster_write_kwargs``. Today it
    returns the pre-sharding behaviour, byte-for-byte.
    """
    if spatial_ndim == 2:
        return (1, edge, edge)
    if spatial_ndim == 3:
        return (1, 1, edge, edge)
    raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}")
