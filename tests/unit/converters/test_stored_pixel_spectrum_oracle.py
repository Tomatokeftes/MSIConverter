"""A named pixel's stored spectrum must be the spectrum that pixel acquired.

Every other converter test in this tree checks something a permutation
survives. Shapes, the var index, dtypes and ``uns`` keys
(``test_lazy_loading_encoding``); ``ome.version`` and ``n_obs``
(``test_streaming_pcs_roundtrip``); a row count and ``min(nnz) > 0``, or
``sum(TIC) == sum(X)`` (``test_uns_provenance_parity``); geometry, whose
own docstring records that the values are assumed rather than checked
(``test_chunking_policy``). Shuffle which pixel owns which row and all of
them stay green, which is how two wrong-data defects sat in a passing
suite.

``test_read_lazy_contract`` is the one file that builds an independent
expectation, but only for a fully populated grid.

This file closes that gap. It is a regression net, not a bug hunt: it
is expected to pass. The specific failure it exists to catch is that
``obs`` positions come from a table unit's ``kept_grid`` while X's row
indices come independently from its ``row_of_grid`` (both in
``streaming_converter``). Should those two ever disagree, every pixel
permutes while the row count, the instance ids, ``uns`` and the TIC
total all still agree.

Four choices make the assertions load-bearing rather than decorative;
each has a test of its own below guarding it, because each degrades
silently into a tautology if a later edit "simplifies" the fixture.

**The expectation is computed, never read back.** ``expected_spectrum``
rebuilds the intensities from the acquisition this module writes. A store
compared against itself -- a lazy read against an eager one, X against
any quantity computed from X -- agrees with itself under every
corruption that reaches both sides equally.

**The grid is sparse and non-square.** 16 acquired positions in a 7x4
bounding box. On a square or fully-populated grid a transposed or
shifted read can land on the right value by coincidence; commit 2c719d4
hit exactly that and gave its own mock 25% sparsity in response. The
matrix is non-square too (16 rows, 40 columns), so a transpose cannot
even produce a conformable shape.

**Pixels are found by ``instance_id``, never by row offset.** v3.0.0
compacted the row offsets while keeping the grid index as the
``instance_id``, so the two differ for every acquired position past the
first gap -- the named pixel below is grid 11 at row 5. A positional
lookup would silently re-encode the pre-v3.0.0 layout and assert the
wrong pixel.

**Values are compared densely, never by ``nnz``.** Whether a writer drops
zero intensities or keeps them as explicit entries is a layout choice,
and neither count means the values are right.

Each of those choices was mutation-tested, not reasoned about, and the
grid changed as a result: at an earlier 11 acquired positions the named
pixel sat on row 5 of 11 -- the one row a reversal maps to itself -- and
a mutation reversing every row left the headline
assertion green. See
``TestTheFixtureCannotDecayIntoATautology`` for the guards that now
prevent it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import anndata
import numpy as np
import pytest
from scipy import sparse

from thyra.convert import convert_msi

DATASET_ID = "ds"
TABLE_KEY = f"{DATASET_ID}_z0"

# --- the synthetic acquisition -------------------------------------------
#
# A 7x4 bounding box, deliberately not square, holding 16 acquired
# positions out of 28. The four corners are all acquired, so the reader
# derives the full 7x4 box and the unacquired positions really are
# interior holes rather than a smaller box.
#
# Both counts are chosen, not arbitrary. An EVEN row count (16) leaves a
# reversal of the rows with no fixed point, and an EVEN N_Y leaves a
# vertical flip with none -- measured: at 11 rows the named pixel landed
# on row 5 of 11, the one row a reversal maps to itself, and a mutation
# that reversed every row sailed past the headline
# assertion. Guarded by the tests in
# TestTheFixtureCannotDecayIntoATautology.
N_X, N_Y = 7, 4

ACQUIRED: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (2, 0),
    (3, 0),
    (6, 0),
    (1, 1),
    (4, 1),
    (5, 1),
    (0, 2),
    (2, 2),
    (3, 2),
    (5, 2),
    (6, 2),
    (0, 3),
    (3, 3),
    (4, 3),
    (6, 3),
)

# The m/z axis every spectrum is written on. Conversion does not resample
# by default, so this survives into var in order and channel index means
# column index -- pinned by test_the_mass_axis_survives_in_order.
N_MZ = 40
MZ_AXIS = np.linspace(100.0, 1000.0, N_MZ)

# Three peaks per spectrum, each catching a different way of being wrong.
# All intensities are small integers, hence exact in the float32 the
# imzML writer stores.
X_CHANNEL_BASE = 2  # channel 2+x, intensity 100*(x+1): moves with x
Y_CHANNEL_BASE = 20  # channel 20+y, intensity 1000*(y+1): moves with y
TAG_CHANNEL = 30  # fixed channel, intensity unique to the pixel
TAG_BASE = 10_000.0

# The pixel the headline assertion names, grid index 11 stored at row 5.
# Interior, off the origin, off x == y, off both anti-diagonals (x+y is
# 5, against 6 for a 7-wide box and 3 for a 4-tall one) and off the
# column a horizontal flip fixes (x == 3) -- so no symmetry of the grid
# and no reversal of the rows maps it onto itself, and a wrong answer
# cannot come back looking right. Guarded by
# test_the_named_pixel_cannot_be_hit_by_accident.
NAMED_PIXEL = (4, 1)

# convert_msi kwargs per converted store. One route since the in-memory
# converters were folded in; ``use_csc`` is the keyword Ousia's wizard
# still passes, so it is exercised once.
WRITE_PATHS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "wizard_kwargs": {"streaming": True, "use_csc": True},
}

# Both surfaces a consumer opens the store with. Ousia uses the lazy one.
READ_SURFACES = ("eager", "lazy")


def grid_index(x: int, y: int) -> int:
    """Flat index of a pixel, which is also its ``instance_id``."""
    return y * N_X + x


def expected_spectrum(x: int, y: int) -> np.ndarray:
    """The intensity vector this module wrote at ``(x, y)``.

    The independent oracle: built from the acquisition, never from the
    store. Positions other than the three peaks are asserted zero, so a
    stray extra value is a failure too.
    """
    spectrum = np.zeros(N_MZ, dtype=np.float64)
    spectrum[X_CHANNEL_BASE + x] = 100.0 * (x + 1)
    spectrum[Y_CHANNEL_BASE + y] = 1000.0 * (y + 1)
    spectrum[TAG_CHANNEL] = TAG_BASE + grid_index(x, y)
    return spectrum


def _dense_matrix(adata: "anndata.AnnData") -> np.ndarray:
    """X as a dense array, whichever surface handed it over.

    ``read_lazy`` yields a dask array wrapping a sparse block; the eager
    read yields the sparse matrix directly.
    """
    matrix = adata.X
    if hasattr(matrix, "compute"):
        matrix = matrix.compute()
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _row_of_pixel(adata: "anndata.AnnData", x: int, y: int) -> int:
    """Row offset holding pixel ``(x, y)``, resolved through ``instance_id``.

    The obs index is the grid index as a string on both paths. This
    is the whole point of the file: the row offset is derived from the
    identity, never assumed equal to it.
    """
    instance_id = str(grid_index(x, y))
    index = adata.obs.index
    assert instance_id in index, (
        f"pixel (x={x}, y={y}) has instance_id {instance_id!r}, which is "
        f"absent from obs: {list(index)}"
    )
    return int(index.get_loc(instance_id))


@pytest.fixture(scope="module")
def acquisition(tmp_path_factory) -> Path:
    """Write the synthetic acquisition to a processed-mode imzML."""
    from pyimzml.ImzMLWriter import ImzMLWriter

    imzml_path = tmp_path_factory.mktemp("pixel_oracle") / "acquisition.imzML"

    with ImzMLWriter(str(imzml_path), mode="processed") as writer:
        for x, y in ACQUIRED:
            # imzML coordinates are 1-based; obs x/y come back 0-based.
            writer.addSpectrum(MZ_AXIS, expected_spectrum(x, y), (x + 1, y + 1, 1))

    return imzml_path


@pytest.fixture(scope="module", params=list(WRITE_PATHS))
def converted_store(request, acquisition: Path, tmp_path_factory) -> Path:
    """One converted store per write path.

    Module-scoped: converting is the slow part and nothing here mutates
    the result.
    """
    write_path = request.param
    output_path = tmp_path_factory.mktemp(write_path) / "out.zarr"

    assert convert_msi(
        str(acquisition),
        str(output_path),
        format_type="spatialdata",
        dataset_id=DATASET_ID,
        pixel_size_um=10.0,
        **WRITE_PATHS[write_path],
    ), f"conversion fixture failed for the {write_path} path"

    return output_path


@pytest.fixture(params=READ_SURFACES)
def table(request, converted_store: Path) -> "anndata.AnnData":
    """The converted table, opened through one of the two read surfaces."""
    table_path = converted_store / "tables" / TABLE_KEY
    if request.param == "lazy":
        return anndata.experimental.read_lazy(str(table_path))
    return anndata.read_zarr(table_path)


class TestStoredPixelSpectrum:
    """What the reader yields for a pixel must be what that pixel acquired."""

    def test_named_pixel_spectrum_matches_the_acquisition(self, table):
        """The headline assertion, on one pixel named up front.

        A row/identity disagreement permutes the rows, and this pixel's
        row then carries some other pixel's
        spectrum -- a different ``TAG_CHANNEL`` value, since that channel
        is unique per pixel by construction.
        """
        x, y = NAMED_PIXEL
        row = _row_of_pixel(table, x, y)

        np.testing.assert_allclose(
            _dense_matrix(table)[row],
            expected_spectrum(x, y),
            err_msg=(
                f"pixel (x={x}, y={y}), instance_id {grid_index(x, y)}, "
                f"stored at row {row}, does not hold the spectrum it acquired"
            ),
        )

    def test_every_acquired_pixel_spectrum_matches_the_acquisition(self, table):
        """The same check across every acquired position.

        The named test above is the one that documents intent; this one
        makes a partial permutation -- two rows swapped, one row shifted
        -- as visible as a total one.
        """
        dense = _dense_matrix(table)

        for x, y in ACQUIRED:
            row = _row_of_pixel(table, x, y)
            np.testing.assert_allclose(
                dense[row],
                expected_spectrum(x, y),
                err_msg=(
                    f"pixel (x={x}, y={y}), instance_id {grid_index(x, y)}, "
                    f"stored at row {row}"
                ),
            )

    def test_obs_positions_agree_with_the_instance_ids(self, table):
        """obs x/y must decode to the same pixel the ``instance_id`` names.

        X is indexed by row, ``obs`` carries both the identity and the
        position, and a consumer scattering values back onto a grid uses
        the position. If the two ever disagree the values above can be
        right while the picture is wrong.
        """
        obs_x = np.asarray(table.obs["x"])
        obs_y = np.asarray(table.obs["y"])

        for x, y in ACQUIRED:
            row = _row_of_pixel(table, x, y)
            assert (int(obs_x[row]), int(obs_y[row])) == (x, y), (
                f"instance_id {grid_index(x, y)} sits at row {row}, which "
                f"reports position (x={obs_x[row]}, y={obs_y[row]}) rather "
                f"than (x={x}, y={y})"
            )

    def test_only_the_acquired_positions_have_rows(self, table):
        """No row for a position that acquired nothing, and none missing.

        The unacquired positions are inside the bounding box, so they
        exist as candidate rows until they are dropped. Asserted as a set
        because row order is not the contract; the identities are.
        """
        assert set(table.obs.index) == {str(grid_index(x, y)) for x, y in ACQUIRED}


class TestTheMassAxis:
    """The column axis the value assertions index into."""

    def test_the_mass_axis_survives_in_order(self, table):
        """var must be the acquisition's m/z axis, in order and complete.

        ``expected_spectrum`` indexes channels positionally, which is
        only meaningful if conversion left the raw axis alone. Without
        this the value tests would still pass on a store whose columns
        were reordered together with the expectation.
        """
        np.testing.assert_array_equal(np.asarray(table.var["mz"]), MZ_AXIS)


class TestTheFixtureCannotDecayIntoATautology:
    """Guards on the properties that make the assertions above bite.

    Each of these would fail loudly if a later edit made the grid square,
    filled in the holes, or moved the named pixel onto an axis of
    symmetry -- rather than leaving the value tests quietly unable to
    detect anything.
    """

    def test_the_named_pixel_cannot_be_hit_by_accident(self):
        """No symmetry of the grid maps the named pixel onto itself.

        Every entry here is a mapping some real coordinate bug applies.
        A pixel the mapping fixes reads back correct under it, and the
        headline assertion is then decorative.
        """
        x, y = NAMED_PIXEL

        assert (x, y) in ACQUIRED
        assert (x, y) != (0, 0), "the origin is fixed by too many wrong mappings"
        assert x != y, "on x == y, fixed by a transposed lookup"
        assert x + y != N_X - 1, "on the anti-diagonal of the 7-wide box"
        assert x + y != N_Y - 1, "on the anti-diagonal of the 4-tall box"
        assert 2 * x != N_X - 1, "on the column a horizontal flip fixes"
        assert 2 * y != N_Y - 1, "on the row a vertical flip fixes"
        assert 0 < x < N_X - 1 and 0 < y < N_Y - 1, "on the edge of the box"

    def test_the_grid_is_sparse_and_non_square(self):
        """Acquired positions are a strict, non-square subset of the box."""
        assert N_X != N_Y, "a square grid makes x/y confusion undetectable"
        assert len(ACQUIRED) < N_X * N_Y, "a full grid drops no rows at all"
        assert len(set(ACQUIRED)) == len(ACQUIRED), "duplicate coordinate"
        # Every corner acquired, so the bounding box really is N_X x N_Y
        # and the gaps are interior.
        assert {(0, 0), (N_X - 1, 0), (0, N_Y - 1), (N_X - 1, N_Y - 1)} <= set(ACQUIRED)

    def test_the_matrix_is_not_square(self, table):
        """A transpose must not even be conformable."""
        n_rows, n_cols = table.shape

        assert n_rows == len(ACQUIRED)
        assert n_cols == N_MZ
        assert n_rows != n_cols, "a square X lets a transpose read back cleanly"

    def test_instance_id_lookup_is_not_the_row_offset(self, table):
        """The named pixel's identity and its row offset must differ.

        If they ever coincide, ``_row_of_pixel`` stops proving anything a
        positional lookup would not, and the file no longer covers the
        v3.0.0 row compaction it was written for.
        """
        x, y = NAMED_PIXEL

        assert grid_index(x, y) != _row_of_pixel(table, x, y)

    def test_the_named_pixels_row_is_not_fixed_by_a_reversal(self, table):
        """The named pixel must not sit on the row a reversal maps to itself.

        Measured, not hypothetical: at 11 acquired positions the named
        pixel landed on row 5 of 11, and a mutation reversing every row
        left that row alone -- the headline assertion
        passed on a store where all ten other pixels were wrong. An even
        row count leaves a reversal no fixed point at all, so this also
        pins ``len(ACQUIRED)`` even.
        """
        n_rows = table.shape[0]
        row = _row_of_pixel(table, *NAMED_PIXEL)

        assert n_rows % 2 == 0, "an odd row count gives a reversal a fixed point"
        assert row != n_rows - 1 - row, "the named pixel survives a full reversal"
        assert row not in (0, n_rows - 1), "first and last rows are too easy to hit"
