"""The streaming PCS path drops empty pixel rows, and refuses depth.

Two changes that have to travel together, which is the whole point of
this module.

**The rows.** ``_write_csc_arrays_to_zarr`` used to emit one row per grid
position. Acquisitions are polygon-shaped and the grid is their bounding
box, so the corners came out as all-zero rows -- the same #88 the other
three write paths fix with ``_drop_empty_pixels``. On real ``pea.imzML``:
17,423 rows against 12,737 spectra, 4,686 of them empty, with
``shapes/ds_z0_pixels`` carrying a polygon for each phantom.

**The depth.** ``_scatter_spectra_direct`` computes its row index as
``y * n_x + x``, with no ``z`` term (the COO route it sat beside used
``z * (n_x * n_y) + y * n_x + x``). Nothing caught that, because the
*other* consequence of the full-grid layout did: ``obs`` was built over
``n_y * n_x`` positions while ``X`` was sized ``n_x * n_y * n_z``, so a
multi-plane dataset died on the length mismatch before anyone could
notice the summing. Drop the phantom rows and those lengths line up --
the crash disappears and the planes start silently summing onto one.
``test_dropping_rows_alone_would_merge_z_planes`` below measures exactly
that, with the refusal disabled, so the reason for the refusal is a
number in the suite and not a claim in a commit message.

The streaming route was never able to write depth anyway: ``__init__``
forces ``handle_3d=False``, the table is named ``_z0``, the TIC is
``(n_y, n_x)``, and obs is built for one plane. Refusing names a
restriction that already existed.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr

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


def _converter(output_path: Path, config: MockMSIConfig):
    return StreamingSpatialDataConverter(
        reader=MockMSIReader(config),
        output_path=output_path,
        dataset_id="mock",
        pixel_size_um=10.0,
    )


def _plane_totals(config: MockMSIConfig) -> dict:
    """Total intensity the reader yields, per z plane."""
    totals: dict = {}
    for (_x, _y, z), _mzs, intensities in MockMSIReader(config).iter_spectra():
        totals[z] = totals.get(z, 0.0) + float(intensities.sum())
    return totals


def _x_total(store_path: Path) -> float:
    """Sum of the stored matrix, read straight off the CSC data array."""
    group = zarr.open_group(str(store_path / "tables" / "mock_z0"), mode="r")
    return float(np.asarray(group["X"]["data"]).sum())


def _n_obs(store_path: Path) -> int:
    group = zarr.open_group(str(store_path / "tables" / "mock_z0"), mode="r")
    return int(group["obs"]["x"].shape[0])


def test_pcs_drops_the_empty_grid_positions(tmp_path):
    """One row per acquired spectrum, not one per bounding-box position."""
    config = _config(sparsity=0.25)
    n_spectra = _N_X * _N_Y - int(_N_X * _N_Y * 0.25)

    converter = _converter(tmp_path / "sparse.zarr", config)
    assert converter.convert() is True

    assert _n_obs(tmp_path / "sparse.zarr") == n_spectra


def test_pcs_keeps_the_grid_index_as_the_row_identity(tmp_path):
    """Dropped rows leave gaps in ``instance_id``, as they do elsewhere.

    ``_drop_empty_pixels`` subsets an AnnData, so on the other paths the
    surviving rows keep the ``instance_id`` they were built with -- the
    grid index, with holes where the empties were. A consumer can still
    recover the position from it. The PCS path compacts row *offsets*,
    because a matrix has to be dense in its rows, and must not compact
    the identities with them.
    """
    config = _config(sparsity=0.25)
    out = tmp_path / "sparse.zarr"
    assert _converter(out, config).convert() is True

    group = zarr.open_group(str(out / "tables" / "mock_z0"), mode="r")
    instance_ids = [int(v) for v in np.asarray(group["obs"]["instance_id"])]
    x_values = np.asarray(group["obs"]["x"])
    y_values = np.asarray(group["obs"]["y"])

    # Gaps, not 0..n-1.
    assert instance_ids != list(range(len(instance_ids)))
    # And each surviving id still names its own grid position.
    assert instance_ids == [int(y) * _N_X + int(x) for x, y in zip(x_values, y_values)]


def test_fully_populated_grid_is_unchanged(tmp_path):
    """The common case keeps every row: nothing is dropped that had data."""
    config = _config(sparsity=0.0)
    out = tmp_path / "dense.zarr"
    assert _converter(out, config).convert() is True

    assert _n_obs(out) == _N_X * _N_Y


def test_streaming_refuses_more_than_one_z_plane(tmp_path):
    """The streaming route refuses depth, and writes nothing.

    Before this it failed too, but only after two full passes over the
    spectra, and with a pandas length complaint ("Length of values (9)
    does not match length of index (18)") that says nothing about what
    to do instead.
    """
    out = tmp_path / "z2.zarr"
    converter = _converter(out, _config(n_z=2))

    assert converter.convert() is False
    assert not out.exists(), "a refused conversion must leave no store behind"


def test_the_refusal_names_the_count_and_the_alternative(tmp_path):
    """The message, checked where it is raised rather than where it is logged.

    ``convert()`` catches everything and returns False, so the text only
    reaches a log record; asserting on the exception keeps this
    independent of whatever log configuration the rest of the suite has
    left behind.
    """
    converter = _converter(tmp_path / "z2.zarr", _config(n_z=3))
    converter._initialize_conversion()

    with pytest.raises(ValueError, match=r"single z plane.*declares 3"):
        converter._refuse_multiple_z_planes()


def test_dropping_rows_alone_would_merge_z_planes(tmp_path):
    """Why the refusal ships in the same commit as the drop.

    With the refusal disabled, a two-plane dataset now *converts*, and
    the store it produces holds both planes summed onto one set of rows
    -- ``y * n_x + x`` ignores z, and the obs-length mismatch that used
    to make that crash is gone once the empty rows go.

    Asserted as a total rather than an error, because that is the shape
    of the failure: the numbers are all plausible, nothing warns, and
    the only tell is that the intensity of two planes is sitting in one.
    """
    config = _config(n_z=2)
    totals = _plane_totals(config)
    assert len(totals) == 2, "fixture must actually have two planes"
    both_planes = totals[0] + totals[1]

    out = tmp_path / "z2_unguarded.zarr"
    converter = _converter(out, config)
    converter._refuse_multiple_z_planes = lambda: None  # type: ignore[method-assign]

    assert converter.convert() is True, "the crash that used to stop this is gone"
    assert _n_obs(out) == _N_X * _N_Y, "one plane's worth of rows"
    assert _x_total(out) == pytest.approx(both_planes, rel=1e-9)
    assert _x_total(out) != pytest.approx(totals[0], rel=1e-9)
