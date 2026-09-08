"""The streaming converter skips spectra outside the declared grid.

Both passes index a row by ``y * n_x + x`` and, before 87f287c, neither
checked the result, so a negative coordinate was a legal negative numpy
index and wrapped silently onto an unrelated pixel. The result was neither
the intruder's value nor the victim's but a mixture of the two, which is
the shape of failure that makes this worth a test: every number stayed
plausible and nothing warned. On the 4x4 fixture below, a route without
the guard stored 39,508.05 for pixel (3, 3) against a truth of 47,363.47.

**Reachability.** ``imzml_reader`` subtracts 1 from the 1-based positions
an imzML declares, so a file that is already 0-based yields ``x = -1``
-- the same base-convention hazard ``_z_base()`` documents for z. That
route is inferred from the reader source rather than measured against a
vendor file, so the fixture reproduces the coordinate directly.

Only x and y are exercised. ``_refuse_multiple_z_planes`` runs before
either pass, so ``n_z == 1`` and a z term cannot reach here.

**The guard was itself incomplete at first**, which the per-pixel
assertions here could not see. The pre-scan counted ``col_counts`` and
``total_nnz`` before the bounds check it already had, so it sized the CSC
arrays for spectra its own scatter pass then skipped. The totals stayed
right -- an unwritten slot is a zero and adds nothing to a row sum -- while
the matrix itself was non-canonical. That is why the structural assertions
at the bottom of this file exist alongside the value ones.

This file used to run every assertion on both streaming routes and compare
them against each other. The COO route is gone, and the comparisons went
with it; the in-memory converters are the reference now, in
``test_stored_pixel_spectrum_oracle``.
"""

import logging
from pathlib import Path

import anndata
import numpy as np
import pytest

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
_TABLE_KEY = "mock_z0"

# The spectrum that gets relocated, and where it is yielded instead.
_RELOCATED_FROM = (0, 0, 0)
_OUT_OF_GRID = (-1, 0, 0)
# y * n_x + x == -1 for that coordinate, which indexes the *last* row.
_WRAP_TARGET = (_N_X - 1, _N_Y - 1)

_WARNING_FRAGMENT = "outside the declared"


class _OutOfGridReader(MockMSIReader):
    """Yields one spectrum at ``x = -1``, the rest on the declared grid.

    Dimensions and spectrum count still say 4x4x1: the point is a reader
    whose *coordinates* disagree with the grid it declares, which is what
    an off-by-one in base convention produces.
    """

    def iter_spectra(self, batch_size=None):
        for coords, mzs, intensities in super().iter_spectra(batch_size):
            yield (
                _OUT_OF_GRID if coords == _RELOCATED_FROM else coords
            ), mzs, intensities


def _config() -> MockMSIConfig:
    return MockMSIConfig(
        n_x=_N_X,
        n_y=_N_Y,
        n_z=1,
        n_mz_bins=100,
        peaks_per_spectrum=(8, 15),
        sparsity=0.0,
    )


def _convert(output_path: Path, reader) -> bool:
    return StreamingSpatialDataConverter(
        reader=reader,
        output_path=output_path,
        dataset_id="mock",
        pixel_size_um=10.0,
    ).convert()


class _CaptureWarnings(logging.Handler):
    """Collect warnings straight off the converter's own logger.

    Attached to the module logger rather than going through ``caplog``,
    which sees nothing here: the converters configure logging themselves
    and the suite's global state is whatever ran before this file.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def __enter__(self) -> "_CaptureWarnings":
        self._logger = logging.getLogger(
            "thyra.converters.spatialdata.streaming_converter"
        )
        self._logger.addHandler(self)
        return self

    def __exit__(self, *exc) -> None:
        self._logger.removeHandler(self)

    @property
    def out_of_grid(self) -> list[str]:
        return [m for m in self.messages if _WARNING_FRAGMENT in m]


def _reader_totals(reader) -> dict:
    """Total intensity per (x, y) as the reader yields it.

    The independent oracle: computed from the input, never read back off
    the store, so the store cannot grade its own homework. Mock peaks sit
    exactly on the common mass axis, so resampling is lossless and these
    totals are directly comparable to the stored ones.
    """
    totals: dict = {}
    for (x, y, _z), _mzs, intensities in reader.iter_spectra():
        totals[(x, y)] = totals.get((x, y), 0.0) + float(intensities.sum())
    return totals


def _stored_totals(store_path: Path) -> dict:
    """Stored intensity per (x, y), keyed by the obs coordinates."""
    adata = anndata.read_zarr(store_path / "tables" / _TABLE_KEY)
    row_sums = np.asarray(adata.X.sum(axis=1)).ravel()
    xs = np.asarray(adata.obs["x"]).astype(int)
    ys = np.asarray(adata.obs["y"]).astype(int)
    return {(int(x), int(y)): float(v) for x, y, v in zip(xs, ys, row_sums)}


def test_the_fixture_actually_wraps_onto_a_populated_pixel():
    """Guard the guard: the setup must reproduce the collision.

    If the mock ever stops emitting the corner spectrum, or the row
    formula changes, every assertion below would pass vacuously against
    a fixture that no longer collides with anything.
    """
    truth = _reader_totals(_OutOfGridReader(_config()))

    assert (_OUT_OF_GRID[0], _OUT_OF_GRID[1]) in truth, "no out-of-grid spectrum"
    assert _WRAP_TARGET in truth, "nothing at the position it wraps onto"
    assert truth[_WRAP_TARGET] > 0.0


def test_out_of_grid_spectrum_leaves_the_wrapped_pixel_alone(tmp_path):
    """Every in-grid pixel keeps its own intensity.

    Asserted over the whole grid rather than just the victim: a guard
    that skipped too much would show up as some *other* pixel going
    missing or short.
    """
    truth = _reader_totals(_OutOfGridReader(_config()))
    in_grid = {
        (x, y): total
        for (x, y), total in truth.items()
        if 0 <= x < _N_X and 0 <= y < _N_Y
    }

    out = tmp_path / "oog.zarr"
    assert _convert(out, _OutOfGridReader(_config())) is True

    stored = _stored_totals(out)

    assert set(stored) == set(in_grid), "the stored grid positions changed"
    assert stored[_WRAP_TARGET] == pytest.approx(truth[_WRAP_TARGET], rel=1e-9)
    for position, expected in in_grid.items():
        assert stored[position] == pytest.approx(expected, rel=1e-9), position


def test_the_skipped_spectrum_is_reported(tmp_path):
    """Dropping data silently is what made this expensive to find."""
    with _CaptureWarnings() as captured:
        out = tmp_path / "warn.zarr"
        assert _convert(out, _OutOfGridReader(_config())) is True

    assert len(captured.out_of_grid) == 1, captured.messages
    assert "1 spectra" in captured.out_of_grid[0]
    assert f"{_N_X}x{_N_Y}" in captured.out_of_grid[0]


def test_a_clean_grid_is_untouched_and_silent(tmp_path):
    """No false positives: the ordinary case keeps every spectrum.

    The stored values for an in-grid dataset are unchanged by this fix,
    which is the claim that matters to anyone holding an existing store.
    """
    truth = _reader_totals(MockMSIReader(_config()))

    with _CaptureWarnings() as captured:
        out = tmp_path / "clean.zarr"
        assert _convert(out, MockMSIReader(_config())) is True

    assert captured.out_of_grid == []

    stored = _stored_totals(out)
    assert len(stored) == _N_X * _N_Y
    for position, expected in truth.items():
        assert stored[position] == pytest.approx(expected, rel=1e-9), position


def test_the_store_still_opens_lazily(tmp_path):
    """Ousia opens these stores through ``anndata.experimental.read_lazy``.

    The pre-scan sizes the CSC arrays from a ``total_nnz`` that excludes
    the skipped spectrum, and ``indptr`` is built from the same counts.
    Those two have to stay consistent or the CSC is malformed -- a
    mismatch that an eager read can absorb but the lazy path need not.
    Values are materialised rather than trusting the handle: encoding
    corruption is invisible until the blocks are computed (see
    ``tests/unit/test_read_lazy_contract.py``).
    """
    truth = _reader_totals(_OutOfGridReader(_config()))

    out = tmp_path / "lazy.zarr"
    assert _convert(out, _OutOfGridReader(_config())) is True

    adata = anndata.experimental.read_lazy(str(out / "tables" / _TABLE_KEY))
    materialised = adata.X.compute()

    assert materialised.shape[0] == _N_X * _N_Y - 1, "one position had no spectrum"

    row_sums = np.asarray(materialised.sum(axis=1)).ravel()
    xs = np.asarray(adata.obs["x"]).astype(int)
    ys = np.asarray(adata.obs["y"]).astype(int)
    lazy_totals = {(int(x), int(y)): float(v) for x, y, v in zip(xs, ys, row_sums)}

    assert lazy_totals[_WRAP_TARGET] == pytest.approx(truth[_WRAP_TARGET], rel=1e-9)


def _reader_nnz_in_grid(reader) -> int:
    """Non-zeros the store should hold, counted off the reader.

    Mock peaks sit exactly on the common mass axis, so resampling neither
    merges nor drops one and a spectrum contributes as many entries as it
    yields. Out-of-grid spectra contribute none: they get no row.
    """
    return sum(
        len(mzs)
        for (x, y, _z), mzs, _intensities in reader.iter_spectra()
        if 0 <= x < _N_X and 0 <= y < _N_Y
    )


def test_the_stored_matrix_is_structurally_clean(tmp_path):
    """Skipping a spectrum must not leave slots reserved for it.

    The pre-scan counted ``col_counts`` and ``total_nnz`` for every
    spectrum with peaks, *including* the ones its own bounds check then
    rejected, while the scatter pass skipped exactly those. The memmap is
    zero-filled, so the reserved slots reached disk as explicit zeros at
    row 0, out of order inside their column.

    Measured on this fixture before the fix: 174 entries with 7 explicit
    zeros and ``has_canonical_format`` False, against the 167 and 0 the
    reader accounts for. ``X.sum()`` raised from scipy, but ``todense()``,
    ``read_zarr()`` and ``read_lazy()`` all succeeded silently.

    No per-pixel value assertion can see this. Explicit zeros contribute
    nothing to a row sum, which is why the totals-based tests above passed
    throughout -- they compared totals, and the totals were right.
    """
    expected_nnz = _reader_nnz_in_grid(_OutOfGridReader(_config()))

    out = tmp_path / "structure.zarr"
    assert _convert(out, _OutOfGridReader(_config())) is True

    csc = anndata.read_zarr(out / "tables" / _TABLE_KEY).X
    assert csc.format == "csc"

    assert csc.nnz == expected_nnz, "slots reserved for a skipped spectrum"
    assert (csc.data == 0).sum() == 0, "explicit zeros from unwritten slots"
    assert csc.has_canonical_format, "unsorted or duplicate indices"

    # indptr has to agree with the array it indexes, or the CSC is
    # malformed in a way an eager read can absorb and a lazy one need not.
    assert int(csc.indptr[-1]) == csc.nnz
