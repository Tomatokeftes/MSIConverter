"""What a store records about itself, on both table shapes.

A converted store is supposed to be able to say where it came from and how
it was interpreted: ``uns["essential_metadata"]`` carries ``source_path``,
``dimensions``, ``mass_range``, ``spectrum_type`` and the ``thyra_version``
that wrote it, and the sections beside it carry the vendor metadata.  Ousia
reads those back out of the store; nothing else records them.

This module used to compare three write paths against each other, because
they had drifted: the streaming path hand-wrote its Zarr layout and
composed its own provenance block (``spectrum_type`` hardcoded to
``"processed"``, ``mass_range`` from the resampled target axis rather than
the source, the vendor sections dropped), its own root attributes (7
against 10, missing ``coordinate_systems``), and its own ``obs`` (no
``region_number``), and it wrote one row per grid position where the other
paths wrote one per acquired spectrum.  Which path a dataset took was
decided by size, so two acquisitions off the same instrument came out
described differently.

Those paths are one now: every table goes through spatialdata's writer
from memory-mapped arrays (design decision D11).  What is left to guard is
that the store keeps saying these things, on both shapes the one converter
writes -- one table per z plane, or one for the volume -- and that the
two shapes agree with each other.

Nothing caught the original drift because ``MockMSIReader`` reported
``spectrum_type="processed"`` too -- the fixture agreed with the hardcoded
literal, so a test could compare them and pass.  The fixture now reports a
value from the vocabulary the real extractors produce; see the comment on
it.

The mock is deliberately **sparse** (``sparsity`` below): on a fully
populated grid "one row per grid position" and "one row per spectrum" are
the same number, which is exactly why the phantom rows survived a green
suite.
"""

from typing import Any, Dict

import numpy as np
import pytest

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata import SpatialDataConverter
from thyra.converters.spatialdata.base_spatialdata_converter import (
    SPATIALDATA_AVAILABLE,
)

pytestmark = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)

_DATASET_ID = "mock"
_N_X = 6
_N_Y = 6
_N_MZ_BINS = 500
# A quarter of the grid carries no spectrum, so a writer that emits one row
# per bounding-box position disagrees with one that emits one per spectrum.
_SPARSITY = 0.25
_N_SPECTRA = _N_X * _N_Y - int(_N_X * _N_Y * _SPARSITY)

# What the mock reader reports, and therefore what must come back out of
# every store. Kept as literals rather than read off the fixture: the point
# is that the stored value tracks the source, and a test that derives the
# expectation from the same place the writer does cannot show that.
_EXPECTED_SPECTRUM_TYPE = "centroid spectrum"
_EXPECTED_SOURCE_PATH = "mock_msi_data"
_EXPECTED_ESSENTIAL_KEYS = {
    "source_path",
    "dimensions",
    "mass_range",
    "spectrum_type",
    "thyra_version",
}

# Shape name -> (handle_3d, the table it writes). ``handle_3d=True`` emits a
# single volume table named after the dataset; the default emits one per z
# plane and there is only ever a z0 here.
TABLE_SHAPES: Dict[str, tuple] = {
    "per_plane": (False, f"{_DATASET_ID}_z0"),
    "volume": (True, _DATASET_ID),
}


def _config() -> MockMSIConfig:
    return MockMSIConfig(
        n_x=_N_X,
        n_y=_N_Y,
        n_mz_bins=_N_MZ_BINS,
        peaks_per_spectrum=(20, 40),
        sparsity=_SPARSITY,
    )


def _convert(tmp_path_factory, shape: str):
    """Convert once into one table shape; hand back the store root."""
    handle_3d, _ = TABLE_SHAPES[shape]
    output_path = tmp_path_factory.mktemp(f"prov_{shape}") / "out.zarr"
    converter = SpatialDataConverter(
        reader=MockMSIReader(_config()),
        output_path=output_path,
        dataset_id=_DATASET_ID,
        pixel_size_um=10.0,
        handle_3d=handle_3d,
    )
    assert converter.convert() is True, f"{shape} conversion failed"
    return output_path


@pytest.fixture(scope="module")
def stores(tmp_path_factory) -> Dict[str, Any]:
    """One conversion per table shape. Module-scoped: converting is the slow part."""
    return {name: _convert(tmp_path_factory, name) for name in TABLE_SHAPES}


def _table_path(stores, shape: str):
    """The table group inside a converted store."""
    return stores[shape] / "tables" / TABLE_SHAPES[shape][1]


def _read_uns(table_path) -> Dict[str, Any]:
    """Read ``uns`` back eagerly, as a plain dict."""
    import anndata as ad
    import zarr

    return ad.io.read_elem(zarr.open_group(str(table_path), mode="r")["uns"])


def _plain(value: Any) -> Any:
    """Normalise for comparison.

    A list written by one path and an ndarray by another are the same
    stored value; ``==`` on ndarrays is not a bool. Recurse so nested
    sections compare too.
    """
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@pytest.mark.parametrize("shape", list(TABLE_SHAPES))
def test_essential_metadata_is_not_empty(stores, shape):
    """The reported symptom: the block present but with nothing in it.

    Deliberately the weakest assertion here, and deliberately first --
    a store whose provenance is empty cannot say what it came from at
    all, which is worse than one that says something inaccurate.
    """
    uns = _read_uns(_table_path(stores, shape))

    assert "essential_metadata" in uns, f"{shape} wrote no essential_metadata"
    essential = uns["essential_metadata"]
    assert essential, f"{shape} wrote an EMPTY essential_metadata block"
    assert set(essential) == _EXPECTED_ESSENTIAL_KEYS


@pytest.mark.parametrize("shape", list(TABLE_SHAPES))
def test_stored_spectrum_type_is_the_readers(stores, shape):
    """``spectrum_type`` must come from the data, not from a literal.

    The hand-written layout wrote ``"processed"`` for everything. That is
    not a value any extractor produces, so a consumer could neither trust
    it nor tell it apart from a real reading -- which is the whole
    failure: losing provenance *silently*.
    """
    essential = _read_uns(_table_path(stores, shape))["essential_metadata"]

    assert essential["spectrum_type"] == _EXPECTED_SPECTRUM_TYPE, (
        f"{shape} stored spectrum_type={essential['spectrum_type']!r}; "
        f"the reader reports {_EXPECTED_SPECTRUM_TYPE!r}"
    )
    assert essential["spectrum_type"] != "processed"


@pytest.mark.parametrize("shape", list(TABLE_SHAPES))
def test_essential_metadata_values_track_the_source(stores, shape):
    """The rest of the block must describe the source, not the output."""
    essential = _read_uns(_table_path(stores, shape))["essential_metadata"]
    cfg = _config()

    assert essential["source_path"] == _EXPECTED_SOURCE_PATH
    assert _plain(essential["dimensions"]) == [cfg.n_x, cfg.n_y, cfg.n_z]
    # mass_range is the SOURCE range. The hand-written layout took it from
    # the resampled target axis, which is a different quantity whenever
    # resampling clips or extends the range.
    assert _plain(essential["mass_range"]) == pytest.approx([cfg.mz_min, cfg.mz_max])
    assert essential["thyra_version"]


@pytest.mark.parametrize("shape", list(TABLE_SHAPES))
def test_vendor_sections_are_stored(stores, shape):
    """The sections beside essential_metadata, which the old layout dropped.

    ``acquisition_params`` is deliberately not asserted: the mock leaves
    it empty and empty sections are omitted, so that a consumer can tell
    "this format has none" from "this one has none recorded".
    """
    uns = _read_uns(_table_path(stores, shape))

    assert uns.get("format_specific") == {"format": "mock"}
    assert uns.get("instrument_info") == {"instrument": "mock"}
    raw = uns.get("raw_metadata") or {}
    assert raw.get("source") == "mock"
    assert "acquisition_params" not in uns
    assert uns.get("regions"), f"{shape} stored no region summary"


@pytest.mark.parametrize("shape", list(TABLE_SHAPES))
def test_string_lists_are_stored_as_json(stores, shape):
    """Non-numeric lists in the vendor sections come back as JSON strings.

    Stored as lists they do not survive the writer: a list of dicts is
    stringified entry by entry into ``repr`` output, and any list of
    strings reads back as a numpy string array -- whose ``deepcopy``
    segfaults the whole process on numpy 2.1-2.2 (numpy#28609), killing
    every consumer that copies the table (``AnnData.copy``,
    ``polygon_query``, joins). Purely numeric lists stay arrays.
    """
    import json

    raw = _read_uns(_table_path(stores, shape))["raw_metadata"]

    assert isinstance(raw["cvParams"], str), (
        f"{shape} stored cvParams as {type(raw['cvParams']).__name__}; "
        "a non-numeric list must be stored as a JSON string"
    )
    assert json.loads(raw["cvParams"]) == [
        {"name": "MS1 spectrum", "accession": "MS:1000579", "value": True},
    ]
    assert _plain(raw["scan_window"]) == [100.0, 1000.0]


@pytest.mark.parametrize("shape", list(TABLE_SHAPES))
def test_uns_contains_no_string_arrays(stores, shape):
    """No value anywhere in ``uns`` reads back as a string array.

    The blanket form of the assertion above, over the whole block rather
    than the known offender: one string array anywhere in ``uns`` is
    enough to crash a numpy 2.1-2.2 reader on the first table copy, so
    the contract is their absence, not just cvParams' encoding.
    """

    def _string_arrays(value, path):
        if isinstance(value, dict):
            for k, v in value.items():
                yield from _string_arrays(v, f"{path}.{k}")
        elif isinstance(value, np.ndarray) and value.dtype.kind in "TUSO":
            yield path

    uns = _read_uns(_table_path(stores, shape))
    offenders = list(_string_arrays(uns, "uns"))
    assert not offenders, f"{shape} stored string arrays at: {offenders}"


def test_both_shapes_store_the_same_provenance(stores):
    """The plane table and the volume table describe the source identically.

    Compares whole blocks rather than named keys, so a section reaching
    one shape and not the other fails here without anyone having to
    remember to extend this file.
    """
    provenance_keys = (
        "essential_metadata",
        "format_specific",
        "acquisition_params",
        "instrument_info",
        "raw_metadata",
        "regions",
        "msi_metadata",
    )
    blocks = {
        name: {
            k: _plain(v)
            for k, v in _read_uns(_table_path(stores, name)).items()
            if k in provenance_keys
        }
        for name in TABLE_SHAPES
    }

    assert blocks["volume"] == blocks["per_plane"]


def _root_attrs(store_path) -> Dict[str, Any]:
    """The store's own root attributes, as a plain dict."""
    import zarr

    return dict(zarr.open_group(str(store_path), mode="r").attrs)


@pytest.mark.parametrize("shape", list(TABLE_SHAPES))
def test_root_attrs_carry_the_coordinate_contract(stores, shape):
    """The store's own attrs, the second half of what a store says about itself.

    The hand-written layout composed its root attrs itself and was short
    by ``coordinate_systems``, ``format_specific_metadata`` and
    ``msi_dataset_info``. The first of those is the structured contract
    naming the unit ``"global"`` is in, which Ousia and the registration
    tooling read rather than guess.
    """
    attrs = _root_attrs(stores[shape])

    assert {
        "coordinate_systems",
        "format_specific_metadata",
        "msi_dataset_info",
    } <= set(attrs)
    assert attrs["msi_dataset_info"]["non_empty_pixels"] == _N_SPECTRA


def test_both_shapes_write_the_same_root_attrs(stores):
    """Same key set on both shapes; values may legitimately differ."""
    assert set(_root_attrs(stores["volume"])) == set(_root_attrs(stores["per_plane"]))


def _read_obs(table_path):
    """Read ``obs`` back eagerly, as a DataFrame."""
    import anndata as ad
    import zarr

    return ad.io.read_elem(zarr.open_group(str(table_path), mode="r")["obs"])


def _read_x(table_path):
    """Read ``X`` back eagerly, as a scipy sparse matrix."""
    import anndata as ad
    import zarr

    return ad.io.read_elem(zarr.open_group(str(table_path), mode="r")["X"])


def test_the_plane_table_writes_the_obs_schema(stores):
    """``obs`` schema of a plane table: positions, region, and region_number.

    The hand-written layout simply had no ``region_number`` column, so a
    consumer that branches on it saw a different schema depending on
    which side of the old size threshold the dataset fell.
    """
    columns = set(_read_obs(_table_path(stores, "per_plane")).columns)

    assert columns == {
        "y",
        "x",
        "region",
        "spatial_x",
        "spatial_y",
        "region_number",
        "instance_key",
    }


def test_volume_obs_adds_only_the_z_columns(stores):
    """The volume table's schema is the plane schema plus depth, nothing else.

    Asserted rather than skipped: "3D is different" is true only for the
    two columns a volume needs, and anything else appearing or vanishing
    there is the same class of drift this file exists to catch.
    """
    plane_columns = set(_read_obs(_table_path(stores, "per_plane")).columns)
    volume_columns = set(_read_obs(_table_path(stores, "volume")).columns)

    assert volume_columns - plane_columns == {"z", "spatial_z"}
    assert not plane_columns - volume_columns


@pytest.mark.parametrize("shape", list(TABLE_SHAPES))
def test_one_row_per_spectrum(stores, shape):
    """Row count: one row per acquired spectrum, on both shapes.

    The hand-written layout emitted the whole bounding box, so a quarter
    of this fixture's rows were all-zero phantoms. On real ``pea.imzML``
    that was 17,423 rows against 12,737 spectra.

    Also asserts that no surviving row is empty, which is the property
    the count is a proxy for: a writer could reach the right *number* of
    rows while keeping the wrong ones.
    """
    obs = _read_obs(_table_path(stores, shape))
    assert len(obs) == _N_SPECTRA, (
        f"{shape} wrote {len(obs)} rows for {_N_SPECTRA} spectra "
        f"({_N_X * _N_Y} grid positions)"
    )

    x = _read_x(_table_path(stores, shape))
    assert int(np.asarray(x.getnnz(axis=1)).min()) > 0, f"{shape} kept empty rows"


@pytest.mark.parametrize("shape", list(TABLE_SHAPES))
def test_shapes_annotate_exactly_the_rows(stores, shape):
    """One pixel polygon per obs row, under the same identity.

    obs and shapes are built from two different places, and compacting
    one without the other is the obvious way to break a store while
    every count still looks plausible. Compares the labels rather than
    the count for the same reason: the index is the *grid* index and
    keeps its gaps, since what compacts is the row offset, not the
    identity.
    """
    import spatialdata

    sdata = spatialdata.read_zarr(str(stores[shape]))
    obs = _read_obs(_table_path(stores, shape))
    region_key = f"{TABLE_SHAPES[shape][1]}_pixels"

    assert list(sdata.shapes[region_key].index) == list(obs.index)


@pytest.mark.parametrize("shape", list(TABLE_SHAPES))
def test_tic_image_totals_the_matrix(stores, shape):
    """``sum(TIC image) == sum(X)``, which dropping rows must not disturb.

    The TIC image stays full-grid while the table drops empty positions,
    which is only sound because a dropped position contributes zero; if
    a route ever dropped a row that carried signal, or scattered one
    row's peaks onto another, these two totals would part company. It is
    the cheapest end-to-end check that the compaction moved rows rather
    than data.
    """
    import spatialdata

    sdata = spatialdata.read_zarr(str(stores[shape]))
    tic_name = f"{TABLE_SHAPES[shape][1]}_tic"
    tic_total = float(np.asarray(sdata.images[tic_name].data).sum())
    x_total = float(_read_x(_table_path(stores, shape)).sum())

    assert tic_total == pytest.approx(x_total, rel=1e-9)


@pytest.mark.parametrize("shape", list(TABLE_SHAPES))
def test_average_spectrum_is_a_mean(stores, shape):
    """``uns["average_spectrum"]`` is the per-pixel mean, not the sum.

    The old volume converter stored ``total_intensity`` unscaled -- the
    sum over pixels, under a key named "average" -- for years, because
    nothing read this key off that path and the scale factor is
    invisible in a plot with an unlabelled y-axis.

    Derived from ``X`` rather than from ``total_intensity`` so the
    expectation comes from the stored matrix a consumer can see, not
    from the accumulator the writer used.
    """
    table_path = _table_path(stores, shape)
    x = _read_x(table_path)
    total = np.asarray(x.sum(axis=0)).ravel()
    stored = np.asarray(_read_uns(table_path)["average_spectrum"], dtype=np.float64)

    np.testing.assert_allclose(
        stored,
        total / _N_SPECTRA,
        rtol=1e-9,
        err_msg=(
            f"{shape} stored an average_spectrum that is not the mean "
            f"over the {_N_SPECTRA} acquired spectra"
        ),
    )

    # Guard the guard: with more than one spectrum the mean and the total
    # are different arrays, so the assertion above can tell them apart.
    assert _N_SPECTRA > 1
    assert not np.allclose(stored, total), f"{shape} stored the total, not the mean"


@pytest.mark.parametrize("shape", list(TABLE_SHAPES))
def test_read_lazy_sees_the_block(stores, shape):
    """The access mode a consumer actually uses.

    ``read_lazy`` is how Ousia opens a converted store without pulling
    the matrix into memory, and it is where the loss was noticed. Array
    entries come back as dask, so this checks the keys are reachable --
    the values are covered eagerly above.
    """
    anndata = pytest.importorskip("anndata")

    lazy = anndata.experimental.read_lazy(str(_table_path(stores, shape)))
    essential = lazy.uns["essential_metadata"]

    assert set(essential) == _EXPECTED_ESSENTIAL_KEYS
    assert essential["spectrum_type"] == _EXPECTED_SPECTRUM_TYPE
