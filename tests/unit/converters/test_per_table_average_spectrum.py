# tests/unit/converters/test_per_table_average_spectrum.py
"""What ``uns["average_spectrum"]`` is the mean of.

``docs/output-format.md`` promises each table "the mean over the acquired
spectra" of that table, and Ousia reads it as that table's own mean. A
multi-slice source converted as 2D -- the default, one table per plane --
stored one dataset-wide vector in every plane's table instead. Measured on
a 3-plane source with intensities scaled by z, the ratio of the stored
vector to ``X.mean(axis=0)`` was 1.96 / 0.996 / 0.67 across the planes,
with the same vector in all three (issue #243). ``#199`` had already fixed
this key for the 3D volume path.

Two things the denominator has to get right, both checked here against
``X.mean(axis=0)`` rather than a hand-computed expectation, because that
is the number the docs promise:

* a **dropped** row -- an all-zero spectrum gets no row, so it belongs in
  neither the sum nor the count. Counting the spectra fed in made a plane
  with one of four rows dropped come out at 0.75 of its mean.
* a **repeated** position -- since #241 the source measuring one pixel
  twice gives one row holding the sum of both spectra, so its contribution
  is two spectra of current over one row. That is the row's real ion
  current, and counting the position twice would be the error.
"""

from __future__ import annotations

from pathlib import Path

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

N_X, N_Y = 3, 2


def _config(n_z: int = 3) -> MockMSIConfig:
    return MockMSIConfig(
        n_x=N_X,
        n_y=N_Y,
        n_z=n_z,
        n_mz_bins=64,
        peaks_per_spectrum=(3, 5),
        seed=11,
    )


def _convert(reader, output: Path, **kwargs) -> Path:
    converter = StreamingSpatialDataConverter(
        reader=reader,
        output_path=output,
        dataset_id="m",
        pixel_size_um=10.0,
        include_optical=False,
        **kwargs,
    )
    assert converter.convert() is True
    return output


class _ScaledByPlane(MockMSIReader):
    """Every plane is plane 0's raster, scaled by ``z + 1``.

    The same spectra rather than merely the same intensity scale, so the
    planes' mean spectra stand in an exactly known ratio and a shared
    vector cannot pass for any of them.
    """

    def iter_spectra(self, batch_size=None):
        plane_zero: dict = {}
        for coords, mzs, intensities in super().iter_spectra(batch_size):
            x, y, z = coords
            if z == 0:
                plane_zero[(x, y)] = (mzs, intensities)
            mzs, intensities = plane_zero[(x, y)]
            yield coords, mzs, intensities * (z + 1)


class _DroppedRowAndRepeat(MockMSIReader):
    """Plane 0 loses a row to an all-zero spectrum, plane 1 repeats a pixel."""

    def iter_spectra(self, batch_size=None):
        extra = None
        for coords, mzs, intensities in super().iter_spectra(batch_size):
            if coords == (0, 0, 0):
                yield coords, mzs, np.zeros_like(intensities)
                continue
            if coords == (1, 0, 1):
                extra = mzs[:2]
            yield coords, mzs, intensities
        assert extra is not None
        yield (1, 0, 1), extra, np.array([3.0, 5.0])


class _TwoRegions(MockMSIReader):
    """The left column is region 1, the rest region 2."""

    def get_region_map(self):
        return {
            (x, y): (1 if x == 0 else 2)
            for y in range(self.config.n_y)
            for x in range(self.config.n_x)
        }


def _tables(store: Path):
    return spatialdata.SpatialData.read(str(store)).tables


def _dense(table) -> np.ndarray:
    X = table.X
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


class TestEachTableStoresItsOwnMean:
    """The promise in docs/output-format.md, per table."""

    def test_every_plane_matches_its_own_matrix(self, tmp_path):
        store = _convert(_ScaledByPlane(_config()), tmp_path / "scaled.zarr")

        tables = _tables(store)
        assert set(tables) == {"m_z0", "m_z1", "m_z2"}
        for key, table in tables.items():
            stored = np.asarray(table.uns["average_spectrum"])
            np.testing.assert_allclose(
                stored,
                _dense(table).mean(axis=0),
                err_msg=f"{key} does not store its own mean spectrum",
            )

    def test_the_planes_do_not_share_one_vector(self, tmp_path):
        """The symptom the issue reported: the same vector in every table."""
        tables = _tables(_convert(_ScaledByPlane(_config()), tmp_path / "s.zarr"))

        z0 = np.asarray(tables["m_z0"].uns["average_spectrum"])
        z1 = np.asarray(tables["m_z1"].uns["average_spectrum"])
        z2 = np.asarray(tables["m_z2"].uns["average_spectrum"])

        # The source scales plane z by (z + 1), so the planes' means are in
        # the ratio 1 : 2 : 3 and cannot be one shared vector.
        np.testing.assert_allclose(z1, 2.0 * z0)
        np.testing.assert_allclose(z2, 3.0 * z0)

    def test_a_dropped_row_and_a_repeat_both_land_on_the_matrix(self, tmp_path):
        """The two denominators the sub-finding names, in one store."""
        store = _convert(_DroppedRowAndRepeat(_config(n_z=2)), tmp_path / "mixed.zarr")
        tables = _tables(store)

        # The all-zero spectrum got no row; the repeat got one, not two.
        assert tables["m_z0"].n_obs == N_X * N_Y - 1
        assert tables["m_z1"].n_obs == N_X * N_Y

        for key, table in tables.items():
            np.testing.assert_allclose(
                np.asarray(table.uns["average_spectrum"]),
                _dense(table).mean(axis=0),
                err_msg=f"{key} does not store its own mean spectrum",
            )

    def test_a_volume_stores_the_volume_mean(self, tmp_path):
        """``handle_3d=True`` is one table, so its mean is the whole stack's."""
        store = _convert(
            _ScaledByPlane(_config()), tmp_path / "vol.zarr", handle_3d=True
        )

        tables = _tables(store)
        assert set(tables) == {"m"}
        table = tables["m"]
        np.testing.assert_allclose(
            np.asarray(table.uns["average_spectrum"]), _dense(table).mean(axis=0)
        )


class TestThePerRegionMeans:
    """``average_spectrum_per_region`` is a mean too, over the same rows."""

    def test_each_region_is_the_mean_of_its_rows(self, tmp_path):
        store = _convert(_TwoRegions(_config(n_z=1)), tmp_path / "regions.zarr")

        table = _tables(store)["m_z0"]
        per_region = table.uns["average_spectrum_per_region"]
        assert set(per_region) == {"1", "2"}

        X = _dense(table)
        for region in ("1", "2"):
            rows = table.obs["region_number"].values == int(region)
            np.testing.assert_allclose(
                np.asarray(per_region[region]),
                X[rows].mean(axis=0),
                err_msg=f"region {region} is not the mean of its rows",
            )

    def test_a_repeated_position_does_not_dilute_its_region(self, tmp_path):
        """The denominator is rows, so the summed row counts once."""

        class _RegionsWithRepeat(_TwoRegions):
            def iter_spectra(self, batch_size=None):
                extra = None
                for coords, mzs, intensities in super().iter_spectra(batch_size):
                    if coords == (0, 0, 0):
                        extra = mzs[:2]
                    yield coords, mzs, intensities
                assert extra is not None
                yield (0, 0, 0), extra, np.array([7.0, 11.0])

        store = _convert(_RegionsWithRepeat(_config(n_z=1)), tmp_path / "repeat.zarr")

        table = _tables(store)["m_z0"]
        X = _dense(table)
        rows = table.obs["region_number"].values == 1
        np.testing.assert_allclose(
            np.asarray(table.uns["average_spectrum_per_region"]["1"]),
            X[rows].mean(axis=0),
        )

    def test_the_regions_span_the_planes(self, tmp_path):
        """A region has no z, so its mean is over every plane's rows.

        ``get_region_map`` is keyed on ``(x, y)``, so one region is one
        in-plane footprint sampled on every plane. The per-region key is
        therefore dataset-wide while ``average_spectrum`` is per table --
        which is the one place the two disagree, and it is deliberate.
        """
        store = _convert(_TwoRegions(_config(n_z=2)), tmp_path / "span.zarr")

        tables = _tables(store)
        stacked = np.vstack([_dense(tables[k]) for k in ("m_z0", "m_z1")])
        regions = np.concatenate(
            [tables[k].obs["region_number"].values for k in ("m_z0", "m_z1")]
        )

        per_region = tables["m_z0"].uns["average_spectrum_per_region"]
        for region in ("1", "2"):
            np.testing.assert_allclose(
                np.asarray(per_region[region]),
                stacked[regions == int(region)].mean(axis=0),
                err_msg=f"region {region} is not the mean of its rows on every plane",
            )

        # Same dict on every plane's table, because it describes them all.
        for region in ("1", "2"):
            np.testing.assert_allclose(
                np.asarray(per_region[region]),
                np.asarray(tables["m_z1"].uns["average_spectrum_per_region"][region]),
            )
