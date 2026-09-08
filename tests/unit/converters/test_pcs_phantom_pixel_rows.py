"""Rows follow the acquired spectra, and depth goes to its own table.

**The rows.** The streaming route used to emit one row per grid position.
Acquisitions are polygon-shaped and the grid is their bounding box, so the
corners came out as all-zero rows -- the same #88 the in-memory converters
fixed by dropping empty rows after the fact. On real ``pea.imzML``: 17,423
rows against 12,737 spectra, 4,686 of them empty, with
``shapes/ds_z0_pixels`` carrying a polygon for each phantom. The rows are
now decided in the counting pass (``_TableUnit.finish_counting``), before
anything is scattered.

**The depth.** The scatter used to index rows by ``y * n_x + x`` with no
``z`` term, and the obs-length mismatch that made a multi-plane dataset
crash was the only thing stopping two planes from summing silently onto
one row set once the phantom rows were dropped. So the route refused
``n_z > 1`` and named the in-memory converters as the ones that wrote
depth. They are gone, and this route writes depth the way they did: one
table per plane by default, the whole volume as one table with
``handle_3d=True``. ``test_two_planes_stay_apart`` below is the
measurement that used to justify the refusal, now asserting the separation.
"""

from pathlib import Path

import anndata
import numpy as np
import pytest
import spatialdata

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata.streaming_converter import (
    SPATIALDATA_AVAILABLE,
    StreamingSpatialDataConverter,
)

pytestmark = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)

_N_X = 4
_N_Y = 4


def _config(n_z: int = 1, sparsity: float = 0.0) -> MockMSIConfig:
    return MockMSIConfig(
        n_x=_N_X,
        n_y=_N_Y,
        n_z=n_z,
        n_mz_bins=100,
        peaks_per_spectrum=(8, 15),
        sparsity=sparsity,
    )


def _converter(output_path: Path, config: MockMSIConfig, **kwargs):
    return StreamingSpatialDataConverter(
        reader=MockMSIReader(config),
        output_path=output_path,
        dataset_id="mock",
        pixel_size_um=10.0,
        **kwargs,
    )


def _plane_totals(config: MockMSIConfig) -> dict:
    """Total intensity the reader yields, per z plane."""
    totals: dict = {}
    for (_x, _y, z), _mzs, intensities in MockMSIReader(config).iter_spectra():
        totals[z] = totals.get(z, 0.0) + float(intensities.sum())
    return totals


def _table(store_path: Path, key: str = "mock_z0") -> anndata.AnnData:
    return anndata.read_zarr(store_path / "tables" / key)


def test_pcs_drops_the_empty_grid_positions(tmp_path):
    """One row per acquired spectrum, not one per bounding-box position."""
    config = _config(sparsity=0.25)
    n_spectra = _N_X * _N_Y - int(_N_X * _N_Y * 0.25)

    converter = _converter(tmp_path / "sparse.zarr", config)
    assert converter.convert() is True

    assert _table(tmp_path / "sparse.zarr").n_obs == n_spectra


def test_pcs_keeps_the_grid_index_as_the_row_identity(tmp_path):
    """Dropped rows leave gaps in ``instance_id``, as they always did.

    The surviving rows keep the ``instance_id`` they were built with --
    the grid index, with holes where the empties were -- so a consumer
    can still recover the position from it. The row *offsets* compact,
    because a matrix has to be dense in its rows; the identities do not.
    """
    config = _config(sparsity=0.25)
    out = tmp_path / "sparse.zarr"
    assert _converter(out, config).convert() is True

    table = _table(out)
    instance_ids = [int(v) for v in table.obs.index]
    x_values = np.asarray(table.obs["x"])
    y_values = np.asarray(table.obs["y"])

    # Gaps, not 0..n-1.
    assert instance_ids != list(range(len(instance_ids)))
    # And each surviving id still names its own grid position.
    assert instance_ids == [int(y) * _N_X + int(x) for x, y in zip(x_values, y_values)]


def test_fully_populated_grid_is_unchanged(tmp_path):
    """The common case keeps every row: nothing is dropped that had data."""
    config = _config(sparsity=0.0)
    out = tmp_path / "dense.zarr"
    assert _converter(out, config).convert() is True

    assert _table(out).n_obs == _N_X * _N_Y


def test_two_planes_stay_apart(tmp_path):
    """Each plane's table holds that plane's intensity and nothing else.

    This is the measurement that used to justify refusing ``n_z > 1``:
    with a row index of ``y * n_x + x`` the two planes summed onto one
    set of rows, every number stayed plausible, and only the total gave
    it away. The same total now says the planes are apart.
    """
    config = _config(n_z=2)
    totals = _plane_totals(config)
    assert len(totals) == 2, "fixture must actually have two planes"

    out = tmp_path / "z2.zarr"
    assert _converter(out, config).convert() is True

    sdata = spatialdata.read_zarr(str(out))
    assert set(sdata.tables) == {"mock_z0", "mock_z1"}
    assert set(sdata.shapes) == {"mock_z0_pixels", "mock_z1_pixels"}
    assert set(sdata.images) == {"mock_z0_tic", "mock_z1_tic"}
    for z in (0, 1):
        table = _table(out, f"mock_z{z}")
        assert table.n_obs == _N_X * _N_Y, "one plane's worth of rows"
        assert float(table.X.sum()) == pytest.approx(totals[z], rel=1e-9)
        assert float(np.asarray(sdata.images[f"mock_z{z}_tic"].data).sum()) == (
            pytest.approx(totals[z], rel=1e-9)
        )
    assert totals[0] != pytest.approx(totals[1], rel=1e-9)


def test_a_volume_carries_both_planes_with_a_z_term(tmp_path):
    """``handle_3d=True``: one table, rows indexed with the z term.

    Every plane's spectra are there, under the whole-volume grid index,
    and ``obs`` says which plane each row sits on.
    """
    config = _config(n_z=2)
    totals = _plane_totals(config)

    out = tmp_path / "volume.zarr"
    assert _converter(out, config, handle_3d=True).convert() is True

    sdata = spatialdata.read_zarr(str(out))
    assert set(sdata.tables) == {"mock"}
    table = _table(out, "mock")
    assert table.n_obs == _N_X * _N_Y * 2
    assert float(table.X.sum()) == pytest.approx(totals[0] + totals[1], rel=1e-9)

    z = np.asarray(table.obs["z"]).astype(int)
    x = np.asarray(table.obs["x"]).astype(int)
    y = np.asarray(table.obs["y"]).astype(int)
    ids = np.asarray([int(v) for v in table.obs.index])
    np.testing.assert_array_equal(ids, z * _N_X * _N_Y + y * _N_X + x)
    for plane in (0, 1):
        rows = table[z == plane]
        assert float(rows.X.sum()) == pytest.approx(totals[plane], rel=1e-9)

    volume = np.asarray(sdata.images["mock_tic"].data)
    assert volume.shape == (1, 2, _N_Y, _N_X)


class _PlaneStrippedReader(MockMSIReader):
    """Declares two planes but acquires only the first."""

    def iter_spectra(self, batch_size=None):
        for coords, mzs, intensities in super().iter_spectra(batch_size):
            if coords[2] == 0:
                yield coords, mzs, intensities


def test_a_plane_with_no_spectra_gets_no_table(tmp_path):
    """An empty plane is left out rather than written as a 0-row table."""
    out = tmp_path / "half.zarr"
    converter = StreamingSpatialDataConverter(
        reader=_PlaneStrippedReader(_config(n_z=2)),
        output_path=out,
        dataset_id="mock",
        pixel_size_um=10.0,
    )
    assert converter.convert() is True

    sdata = spatialdata.read_zarr(str(out))
    assert set(sdata.tables) == {"mock_z0"}
    assert set(sdata.shapes) == {"mock_z0_pixels"}
    assert set(sdata.images) == {"mock_z0_tic"}


def test_the_scratch_directory_is_gone_afterwards(tmp_path):
    """The memmaps live next to the output and must not outlive the write."""
    out = tmp_path / "clean.zarr"
    assert _converter(out, _config()).convert() is True

    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".thyra_")]
    assert not leftovers, leftovers
