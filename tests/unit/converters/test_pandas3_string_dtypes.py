"""Write-path coverage under pandas 3's string dtypes.

Under pandas 3.0 -- or pandas 2.x with ``future.infer_string=True``, which is
what 3.0 makes the default -- the table's obs index (``instance_id``), the
``instance_key`` column and the var index (``mz_*``) are inferred as pandas'
``str`` dtype rather than ``object``.

anndata could not serialize that until 0.13.0 (anndata #2221, fixed by anndata
#2133), so ``SpatialData.write()`` raised ``IORegistryError`` and Thyra carried
a write-time coercion to ``object`` in ``_save_output``.  The anndata floor has
since moved past 0.13, the coercion is gone, and these tests now cover
anndata's own handling instead of Thyra's workaround.

They are still worth keeping.  The rest of the suite runs with pandas' default
inference, so nothing else in CI would notice a regression here; every test in
this module turns ``future.infer_string`` on for its duration and restores it
afterwards.  Every table reaches anndata's writer now -- the streaming path
used to hand-write the AnnData layout straight to Zarr and sidestep it, which
is why this module once covered two paths.
"""

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata.base_spatialdata_converter import (
    SPATIALDATA_AVAILABLE,
)
from thyra.converters.spatialdata.streaming_converter import (
    StreamingSpatialDataConverter,
)

pytestmark = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)

_N_X = 6
_N_Y = 6
_N_MZ_BINS = 500
_N_PIXELS = _N_X * _N_Y


@pytest.fixture(autouse=True)
def infer_string():
    """Run every test in this module with pandas 3 string inference on.

    ``pd.option_context`` restores the previous value on exit, so the option
    cannot leak into the rest of the suite.  The option still exists on pandas
    3 (where it defaults to ``True``), so this is a no-op there rather than a
    behaviour change.
    """
    with pd.option_context("future.infer_string", True):
        yield


def _small_config() -> MockMSIConfig:
    return MockMSIConfig(
        n_x=_N_X, n_y=_N_Y, n_mz_bins=_N_MZ_BINS, peaks_per_spectrum=(20, 40)
    )


def _converter(output_path):
    return StreamingSpatialDataConverter(
        reader=MockMSIReader(_small_config()),
        output_path=output_path,
        dataset_id="mock",
        pixel_size_um=10.0,
        use_csc=True,
    )


def _string_obs_table() -> pd.DataFrame:
    """An obs table shaped like the converters build it, with pandas 3 dtypes.

    Mirrors what the converter's obs builder plus the ``region`` /
    ``instance_key`` fixups produce: a string index, a string column, a
    string-backed categorical with more than one category, and numeric
    columns that must be left alone.
    """
    return pd.DataFrame(
        {
            "x": np.arange(3, dtype=np.int32),
            "spatial_x": np.arange(3, dtype=np.float64) * 10.0,
            "region": pd.Categorical(np.array(["mock_z0_pixels", "other", "other"])),
            "instance_key": np.array(["0", "1", "2"]),
        },
        index=pd.Index(np.array(["0", "1", "2"]), name="instance_id"),
    )


def test_conversion_under_infer_string(tmp_path):
    """The converter must write with pandas 3 string dtypes.

    The writer route failed here before anndata 0.13 without the coercion.
    """
    output_path = tmp_path / "infer_string.zarr"

    converter = _converter(output_path)
    assert converter.convert() is True, "conversion failed"

    import spatialdata

    sdata = spatialdata.read_zarr(str(output_path))
    assert len(sdata.tables) == 1
    table = list(sdata.tables.values())[0]
    assert table.shape == (_N_PIXELS, _N_MZ_BINS)
    assert "mock_z0_tic" in sdata.images
    assert "mock_z0_pixels" in sdata.shapes


def test_written_store_reads_back_with_pandas_string_index(tmp_path):
    """The coercion is write-time only -- the on-disk format is unchanged.

    Readers still get pandas' native string dtype for the obs and var indices,
    exactly as they would from a pandas 2 write, so the workaround does not
    leak ``object`` dtypes into the stored layout.
    """
    output_path = tmp_path / "readback.zarr"
    assert _converter(output_path).convert() is True

    import spatialdata

    table = list(spatialdata.read_zarr(str(output_path)).tables.values())[0]

    assert isinstance(table.obs.index.dtype, pd.StringDtype)
    assert isinstance(table.var.index.dtype, pd.StringDtype)
    assert table.obs.index.tolist() == [str(i) for i in range(_N_PIXELS)]
    assert table.var.index.tolist() == [f"mz_{i}" for i in range(_N_MZ_BINS)]
    assert isinstance(table.obs["region"].dtype, pd.CategoricalDtype)
    assert table.obs["region"].tolist() == ["mock_z0_pixels"] * _N_PIXELS


def test_anndata_writes_arrow_backed_strings(tmp_path):
    """anndata serializes pandas' ``str`` dtype directly.

    This was the removal trigger for the write-time coercion Thyra used to
    carry: it was an ``xfail`` that XPASSed once the anndata floor moved to
    0.13.  Kept as a plain assertion so a regression in anndata's string
    support is caught here, at the smallest possible scope, rather than as a
    confusing failure somewhere in the write paths above.
    """
    import anndata as ad
    import zarr

    df = _string_obs_table()
    assert isinstance(df.index.dtype, pd.StringDtype)  # precondition

    group = zarr.open_group(str(tmp_path / "probe.zarr"), mode="w")
    ad.io.write_elem(group, "obs", df)

    written_back = ad.io.read_elem(group["obs"])
    assert written_back.index.tolist() == df.index.tolist()
    assert written_back["instance_key"].tolist() == df["instance_key"].tolist()
