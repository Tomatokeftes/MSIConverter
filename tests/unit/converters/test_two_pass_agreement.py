"""What the second pass is allowed to hand the assembly.

Every conversion reads the source twice: pass 1 counts the entries each
m/z column will hold and records each position's TIC, pass 2 scatters the
values. Two things were taken on trust.

**A position handed in twice** (issue #241). Two spectra at one pixel were
written as two ``(row, col)`` entries rather than one holding their sum,
which is what ``coo_matrix(...).tocsc()`` -- the conversion the streaming
engine replaced -- produced. The stored matrix is then non-canonical, and
invisibly so: scipy and dask both merge duplicates as they read, so
``read_zarr`` shows the sum while a consumer that binary-searches
``X/indices`` directly sees one of the two values. The TIC image held the
last spectrum's total against a row holding the sum, and
``non_empty_pixels`` counted spectra rather than rows. So these tests read
the zarr arrays themselves.

**A pass that returns different values** (issue #247). The only agreement
check was per-column entry counts, so a reader whose second iteration
returned the same counts with different values wrote a store whose ``X``
came from pass 2 while its TIC image and ``average_spectrum`` came from
pass 1 -- measured at ``TIC sum=147066`` against ``X.sum=294132``, with
nothing said. Pass 2 now compares each position's total with the one pass
1 recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import anndata
import numpy as np
import pytest
import spatialdata
import zarr

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata.streaming_converter import (
    SPATIALDATA_AVAILABLE,
    StreamingSpatialDataConverter,
)
from thyra.errors import ConversionRefused

pytestmark = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)

N_X, N_Y = 3, 2
KEY = "m_z0"


def _config() -> MockMSIConfig:
    return MockMSIConfig(
        n_x=N_X, n_y=N_Y, n_z=1, n_mz_bins=64, peaks_per_spectrum=(3, 5), seed=11
    )


def _converter(reader, output: Path) -> StreamingSpatialDataConverter:
    return StreamingSpatialDataConverter(
        reader=reader,
        output_path=output,
        dataset_id="m",
        pixel_size_um=10.0,
        include_optical=False,
    )


class _RepeatedCoordinate(MockMSIReader):
    """The raster, then one more spectrum at the pixel (1, 0) already has.

    It carries m/z values that pixel already carries, so the extra entries
    land in columns it already occupies -- which is the case the stored
    arrays cannot represent twice.
    """

    def iter_spectra(self, batch_size=None):
        extra = None
        for coords, mzs, intensities in super().iter_spectra(batch_size):
            if coords == (1, 0, 0):
                extra = mzs[:2]
            yield coords, mzs, intensities
        assert extra is not None
        yield (1, 0, 0), extra, np.array([1.0, 2.0])


class _ScaledSecondPass(MockMSIReader):
    """Same entry counts on the second pass, every value doubled."""

    def __init__(self, config):
        super().__init__(config)
        self._pass = 0

    def reset(self):
        super().reset()
        self._pass += 1

    def iter_spectra(self, batch_size=None):
        scale = 2.0 if self._pass else 1.0
        for coords, mzs, intensities in super().iter_spectra(batch_size):
            yield coords, mzs, intensities * scale


class _SwappedPixels(MockMSIReader):
    """The second pass exchanges two pixels' spectra: counts unchanged."""

    def __init__(self, config):
        super().__init__(config)
        self._pass = 0

    def reset(self):
        super().reset()
        self._pass += 1

    def iter_spectra(self, batch_size=None):
        spectra = list(super().iter_spectra(batch_size))
        if self._pass:
            first, second = spectra[0], spectra[1]
            spectra[0] = (first[0], second[1], second[2])
            spectra[1] = (second[0], first[1], first[2])
        yield from spectra


def _stored_arrays(store: Path, key: str = KEY):
    """``X`` exactly as written -- not what scipy or dask hand back."""
    group = zarr.open_group(str(store / "tables" / key / "X"), mode="r")
    return (
        np.asarray(group["indices"][:]),
        np.asarray(group["data"][:]),
        np.asarray(group["indptr"][:]),
    )


def _first_non_canonical_column(indices, indptr) -> Optional[int]:
    for column, (lo, hi) in enumerate(zip(indptr[:-1], indptr[1:])):
        rows = indices[lo:hi]
        if rows.size > 1 and not np.all(np.diff(rows) > 0):
            return column
    return None


class TestRepeatedCoordinates:
    def test_the_store_holds_one_entry_per_pixel_and_bin(self, tmp_path):
        store = tmp_path / "dup.zarr"
        assert _converter(_RepeatedCoordinate(_config()), store).convert() is True

        indices, _data, indptr = _stored_arrays(store)
        assert _first_non_canonical_column(indices, indptr) is None

    def test_the_repeated_pixel_holds_the_sum(self, tmp_path):
        store = tmp_path / "sum.zarr"
        plain = tmp_path / "plain.zarr"
        assert _converter(_RepeatedCoordinate(_config()), store).convert() is True
        assert _converter(MockMSIReader(_config()), plain).convert() is True

        with_repeat = anndata.read_zarr(store / "tables" / KEY)
        without = anndata.read_zarr(plain / "tables" / KEY)
        row = int(np.flatnonzero(with_repeat.obs.index.astype(int) == 1)[0])

        # The extra spectrum carried 1.0 and 2.0 into columns the pixel had.
        assert with_repeat.n_obs == without.n_obs
        assert float(with_repeat.X[row].sum()) == pytest.approx(
            float(without.X[row].sum()) + 3.0
        )

    def test_the_tic_image_agrees_with_the_row(self, tmp_path):
        store = tmp_path / "tic.zarr"
        assert _converter(_RepeatedCoordinate(_config()), store).convert() is True

        table = anndata.read_zarr(store / "tables" / KEY)
        sdata = spatialdata.read_zarr(store)
        tic = np.asarray(sdata.images[f"{KEY}_tic"].data).reshape(N_Y, N_X)
        row = int(np.flatnonzero(table.obs.index.astype(int) == 1)[0])

        assert float(tic[0, 1]) == pytest.approx(float(table.X[row].sum()))

    def test_non_empty_pixels_counts_rows_not_spectra(self, tmp_path):
        store = tmp_path / "count.zarr"
        assert _converter(_RepeatedCoordinate(_config()), store).convert() is True

        sdata = spatialdata.read_zarr(store)
        table = anndata.read_zarr(store / "tables" / KEY)
        reported = sdata.attrs["msi_dataset_info"]["non_empty_pixels"]

        assert reported == table.n_obs == N_X * N_Y

    def test_the_repeat_is_reported(self, tmp_path, thyra_logs):
        """``thyra_logs`` rather than ``caplog``: see ``tests/conftest.py``."""
        store = tmp_path / "logged.zarr"
        with thyra_logs("thyra.converters.spatialdata.streaming_converter") as records:
            assert _converter(_RepeatedCoordinate(_config()), store).convert() is True

        assert any(
            "carry more than one spectrum" in record.getMessage() for record in records
        )


class TestTheSecondPassMustRepeatTheFirst:
    def test_a_scaled_second_pass_is_refused(self, tmp_path):
        store = tmp_path / "scaled.zarr"
        converter = _converter(_ScaledSecondPass(_config()), store)
        converter._initialize_conversion()

        with pytest.raises(ConversionRefused, match=r"disagree at pixel"):
            converter._process_spectra(converter._create_data_structures())

    def test_the_cli_reports_it_as_a_failure(self, tmp_path):
        """Through ``convert()``, which turns a refusal into ``False``."""
        store = tmp_path / "failed.zarr"
        assert _converter(_ScaledSecondPass(_config()), store).convert() is False
        assert not store.exists()

    def test_exchanged_pixels_are_refused(self, tmp_path):
        """No count can see this: the entries are all still there."""
        store = tmp_path / "swapped.zarr"
        assert _converter(_SwappedPixels(_config()), store).convert() is False

    def test_a_repeatable_reader_is_not_refused(self, tmp_path):
        store = tmp_path / "fine.zarr"
        assert _converter(MockMSIReader(_config()), store).convert() is True
        assert anndata.read_zarr(store / "tables" / KEY).n_obs == N_X * N_Y

    def test_a_repeated_position_is_checked_on_its_total(self, tmp_path):
        """The summed positions are compared once the pass is over."""

        class _RepeatAndDiverge(_RepeatedCoordinate):
            def __init__(self, config):
                super().__init__(config)
                self._pass = 0

            def reset(self):
                super().reset()
                self._pass += 1

            def iter_spectra(self, batch_size=None):
                for coords, mzs, intensities in super().iter_spectra(batch_size):
                    if self._pass and coords == (1, 0, 0):
                        intensities = intensities * 3.0
                    yield coords, mzs, intensities

        store = tmp_path / "repeat_diverge.zarr"
        assert _converter(_RepeatAndDiverge(_config()), store).convert() is False
