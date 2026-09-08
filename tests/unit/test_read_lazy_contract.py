# tests/unit/test_read_lazy_contract.py
"""The converted store must stay readable by ``anndata.experimental.read_lazy``.

Downstream software (Ousia) opens Thyra's output through the lazy path, so the
AnnData encoding metadata on ``tables/<id>/X`` is a load-bearing part of the
output contract, not an implementation detail.

The assertions here are deliberately stronger than "read_lazy did not raise".
Ablating the attrs one at a time on a real converted store shows two distinct
failure classes:

* **Loud** -- deleting ``encoding-type``/``encoding-version`` on the ``X`` group
  or on the table group raises ``IORegistryError``; deleting ``shape`` raises
  ``TypeError``.
* **Silent** -- flipping ``encoding-type`` from ``csc_matrix`` to ``csr_matrix``,
  or transposing/shortening ``shape``, leaves ``read_lazy`` succeeding and
  returning the *wrong data*. Likewise, a rechunk that forgets
  ``zarr.consolidate_metadata`` leaves a stale chunk grid in the root index, so
  reads through the consolidated root silently substitute ``fill_value``.

Two design choices follow from mutation-testing this file against those cases:

**The expectation is computed, never read back.** ``expected_dense`` reproduces
the intensities the fixture wrote into the imzML rather than reading the store,
so the store cannot grade its own homework. Comparing a lazy read against an
eager read of the *same* store passes for any corruption that degrades both
paths identically -- verified: on a square matrix, flipping ``encoding-type`` to
``csr_matrix`` returns exactly the transpose and a self-referential test stays
green.

**``nnz`` is never asserted.** Deleting a chunk file preserves *structural* nnz
via ``fill_value`` substitution while zeroing every value, so ``nnz > 0`` passes
on a store that has lost its data. Compare values instead.

One thing deliberately *not* asserted: ``encoding-type``/``encoding-version`` on
the ``data``/``indices``/``indptr`` members. Only the hand-rolled streaming-CSC
writer sets those; the spatialdata-written paths leave them empty and read back
fine, so pinning them would fail the default converter.
"""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pytest
import zarr
from scipy import sparse

from thyra.convert import convert_msi

DATASET_ID = "ds"
TABLE_KEY = f"{DATASET_ID}_z0"

# The synthetic acquisition the fixture writes, and therefore the independent
# oracle every value assertion below is checked against. Peaks land on two m/z
# channels with pixel-dependent intensity, so a transposed or shifted read
# cannot coincidentally match.
COORDINATES = [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1)]  # 2x2 grid
N_MZ = 50
PEAK_X_CHANNEL = 10  # intensity = 100.0 * x
PEAK_Y_CHANNEL = 30  # intensity = 150.0 * y


def _dense(matrix) -> np.ndarray:
    """Densify whatever sparse/array flavour a read surface hands back."""
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


@pytest.fixture(scope="module")
def converted_store(tmp_path_factory) -> Path:
    """A real converted SpatialData store from the standard (non-streaming) path.

    Module-scoped: the conversion is the slow part and no test here mutates the
    store.
    """
    from pyimzml.ImzMLWriter import ImzMLWriter

    tmp_path = tmp_path_factory.mktemp("read_lazy")
    imzml_path = tmp_path / "minimal.imzML"

    mzs = np.linspace(100, 1000, N_MZ)
    with ImzMLWriter(str(imzml_path), mode="processed") as writer:
        for x, y, z in COORDINATES:
            intensities = np.zeros_like(mzs)
            intensities[PEAK_X_CHANNEL] = 100.0 * x
            intensities[PEAK_Y_CHANNEL] = 150.0 * y
            writer.addSpectrum(mzs, intensities, (x, y, z))

    output_path = tmp_path / "out.zarr"
    assert convert_msi(
        str(imzml_path),
        str(output_path),
        format_type="spatialdata",
        dataset_id=DATASET_ID,
        pixel_size_um=10.0,
    ), "conversion fixture failed"

    return output_path


@pytest.fixture(scope="module")
def expected_dense(converted_store: Path) -> np.ndarray:
    """X as it *should* be, rebuilt from the fixture's own input.

    Only ``obs`` is read from the store, to learn which pixel each row holds --
    the intensities themselves come from the acquisition above. Conversion does
    not resample by default, so the raw m/z axis survives in order and the two
    peak channels keep their positions.
    """
    obs = anndata.read_zarr(converted_store / "tables" / TABLE_KEY).obs

    expected = np.zeros((len(COORDINATES), N_MZ), dtype=np.float64)
    for row, (x, y) in enumerate(zip(obs["x"].to_numpy(), obs["y"].to_numpy())):
        # obs x/y are 0-based pixel indices; the acquisition used 1-based coords.
        expected[row, PEAK_X_CHANNEL] = 100.0 * (int(x) + 1)
        expected[row, PEAK_Y_CHANNEL] = 150.0 * (int(y) + 1)
    return expected


class TestReadLazyContract:
    """``anndata.experimental.read_lazy`` must open the store and yield real data."""

    def test_read_lazy_opens_the_table(self, converted_store, expected_dense):
        adata = anndata.experimental.read_lazy(
            str(converted_store / "tables" / TABLE_KEY)
        )

        assert adata.shape == expected_dense.shape

    def test_read_lazy_materialises_the_right_values(
        self, converted_store, expected_dense
    ):
        """Materialise X: metadata corruption is invisible until you do.

        A flipped ``encoding-type`` or a transposed ``shape`` leaves the lazy
        handle looking healthy and only shows up once the blocks are computed.
        """
        adata = anndata.experimental.read_lazy(
            str(converted_store / "tables" / TABLE_KEY)
        )

        materialised = adata.X.compute()

        assert sparse.issparse(materialised)
        assert materialised.shape == expected_dense.shape
        np.testing.assert_allclose(_dense(materialised), expected_dense)

    def test_eager_and_lazy_reads_agree(self, converted_store, expected_dense):
        """The two read paths must not diverge from each other or from the input."""
        eager = anndata.read_zarr(converted_store / "tables" / TABLE_KEY)

        np.testing.assert_allclose(_dense(eager.X), expected_dense)

    def test_read_lazy_exposes_the_spatial_obs_columns(self, converted_store):
        """obs must survive the lazy path.

        Asserted as a subset of a set: column *order* is not stable across runs,
        and the streaming writer emits a different column set.
        """
        adata = anndata.experimental.read_lazy(
            str(converted_store / "tables" / TABLE_KEY)
        )

        assert {"x", "y", "region"}.issubset(set(adata.obs.columns))


class TestEncodingMetadata:
    """The attrs ``read_lazy`` dispatches on must be present and exact."""

    def test_x_group_carries_the_sparse_encoding(self, converted_store):
        x_group = zarr.open_group(
            str(converted_store / "tables" / TABLE_KEY / "X"), mode="r"
        )

        assert isinstance(x_group, zarr.Group), "X must stay a Group, not an Array"
        assert set(x_group.keys()) == {"data", "indices", "indptr"}
        # CSC is the only layout any route writes since sparse_format went.
        assert x_group.attrs["encoding-type"] == "csc_matrix"
        # anndata matches this exactly; '0.2.0' raises IORegistryError.
        assert x_group.attrs["encoding-version"] == "0.1.0"
        assert list(x_group.attrs["shape"]) == [len(COORDINATES), N_MZ]

    def test_indptr_length_matches_the_declared_orientation(self, converted_store):
        """A CSC store needs ``n_cols + 1`` pointers, a CSR one ``n_rows + 1``.

        Catches an ``encoding-type`` flip on its own terms, without waiting for
        the value comparison to notice the transpose.
        """
        x_group = zarr.open_group(
            str(converted_store / "tables" / TABLE_KEY / "X"), mode="r"
        )
        n_rows, n_cols = tuple(x_group.attrs["shape"])

        expected_len = (
            n_cols + 1 if x_group.attrs["encoding-type"] == "csc_matrix" else n_rows + 1
        )
        assert x_group["indptr"].shape[0] == expected_len

    def test_table_group_carries_the_anndata_encoding(self, converted_store):
        """Deleting these breaks read_lazy even with X itself untouched."""
        table_group = zarr.open_group(
            str(converted_store / "tables" / TABLE_KEY), mode="r"
        )

        assert table_group.attrs["encoding-type"] == "anndata"
        assert table_group.attrs["encoding-version"] == "0.1.0"

    def test_consolidated_root_agrees_with_the_table(
        self, converted_store, expected_dense
    ):
        """Read X through the root's consolidated index, not the table directly.

        The root store carries a consolidated metadata index duplicating the
        chunk grid of every array under ``tables/``. Rewriting those arrays
        without re-running ``zarr.consolidate_metadata`` leaves the index stale,
        and reads through it silently return ``fill_value`` for chunks whose keys
        no longer exist. This is the only assertion in the file that catches
        that, so do not drop it as redundant.
        """
        root = zarr.open_group(str(converted_store), mode="r")
        x_group = root[f"tables/{TABLE_KEY}/X"]

        data = np.asarray(x_group["data"][:])
        indices = np.asarray(x_group["indices"][:])
        indptr = np.asarray(x_group["indptr"][:])
        shape = tuple(x_group.attrs["shape"])

        ctor = (
            sparse.csc_matrix
            if x_group.attrs["encoding-type"] == "csc_matrix"
            else sparse.csr_matrix
        )
        reconstructed = ctor((data, indices, indptr), shape=shape)

        np.testing.assert_allclose(_dense(reconstructed), expected_dense)
