"""What a store records about itself must agree on every write path.

A converted store is supposed to be able to say where it came from and how
it was interpreted: ``uns["essential_metadata"]`` carries ``source_path``,
``dimensions``, ``mass_range``, ``spectrum_type`` and the ``thyra_version``
that wrote it, and the sections beside it carry the vendor metadata.  Ousia
reads those back out of the store; nothing else records them.

There are four write paths and they had drifted:

* the in-memory converters and the streaming-COO path hand the table to
  anndata's writer, which serialises ``adata.uns`` as-is, and
* the streaming-PCS path hand-writes the Zarr layout, and used to compose
  its own, much smaller block -- ``spectrum_type`` hardcoded to
  ``"processed"``, ``mass_range`` taken from the *resampled target axis*
  rather than the source, ``source_path`` from a different attribute, and
  ``format_specific`` / ``instrument_info`` / ``raw_metadata`` / ``regions``
  dropped entirely.

Which path a dataset took was decided by size at the time, in
``StreamingSpatialDataConverter._should_use_pcs`` -- so two acquisitions
off the same instrument landed on opposite sides of the split and came out
described differently.  On ``test_data/``: ``pea.imzML`` estimated 29.9 GB
(COO, complete provenance) and ``bellini.imzML`` 43.3 GB (PCS, wrong
``spectrum_type``, everything else missing).  Ousia's import wizard forces
``use_csc=True``, so *every* wizard conversion took the PCS path.

That size split is gone -- ``"auto"`` is PCS now, unconditionally -- which
removes the *mechanism* that made two sibling datasets disagree but not the
hazard: the paths still compose provenance separately, so a section added
to one and not the other diverges just as silently. Hence this module keeps
asserting parity across all of them rather than being retired with the
threshold.

Nothing caught it because ``MockMSIReader`` reported
``spectrum_type="processed"`` too -- the fixture agreed with the hardcoded
literal, so a test could compare them and pass.  The fixture now reports a
value from the vocabulary the real extractors produce; see the comment on
it.

``uns`` is not the only thing that drifted, and the same size gate decides
all of it, so this file compares everything a store says about itself:

* the ``uns`` provenance block,
* the **root attributes** -- ``coordinate_systems``,
  ``format_specific_metadata`` and ``msi_dataset_info`` were missing from a
  PCS store, 7 attributes against 10 on real ``pea.imzML``.  The first of
  those is the structured contract naming the unit ``"global"`` is in,
  which Ousia and the registration tooling read rather than guess,
* the **obs columns** -- PCS never wrote ``region_number``, and
* the **row count** -- PCS wrote one row per grid position where the other
  paths write one per acquired spectrum, so a polygon-shaped acquisition
  came out with all-zero phantom rows (real ``pea``: 17,423 against
  12,737, and ``shapes/`` tracked the phantoms).

There is a fourth path, ``SpatialData3DConverter``, which the original
version of this file did not cover.

These tests convert the same mock dataset down all four paths and compare
what each one stored.  The mock is deliberately **sparse** (``sparsity``
below): on a fully populated grid "one row per grid position" and "one row
per spectrum" are the same number, which is exactly why the phantom rows
survived a green suite.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata.base_spatialdata_converter import (
    SPATIALDATA_AVAILABLE,
)
from thyra.converters.spatialdata.spatialdata_2d_converter import SpatialData2DConverter
from thyra.converters.spatialdata.spatialdata_3d_converter import SpatialData3DConverter
from thyra.converters.spatialdata.streaming_converter import (
    StreamingSpatialDataConverter,
)

pytestmark = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)

_DATASET_ID = "mock"
_TABLE_NAME = f"{_DATASET_ID}_z0"
_N_X = 6
_N_Y = 6
_N_MZ_BINS = 500
# A quarter of the grid carries no spectrum, so the routes that write one
# row per acquired spectrum and one that writes the whole grid disagree.
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


def _config() -> MockMSIConfig:
    return MockMSIConfig(
        n_x=_N_X,
        n_y=_N_Y,
        n_mz_bins=_N_MZ_BINS,
        peaks_per_spectrum=(20, 40),
        sparsity=_SPARSITY,
    )


def _common(output_path) -> Dict[str, Any]:
    return {
        "reader": MockMSIReader(_config()),
        "output_path": output_path,
        "dataset_id": _DATASET_ID,
        "pixel_size_um": 10.0,
    }


def _in_memory(output_path):
    """The default converter -- anything under the streaming threshold."""
    return SpatialData2DConverter(**_common(output_path))


def _streaming_coo(output_path):
    """``streaming=True, use_csc=False`` -- writes via ``SpatialData.write()``."""
    return StreamingSpatialDataConverter(**_common(output_path), use_csc=False)


def _streaming_pcs(output_path):
    """``streaming=True, use_csc=True`` -- the hand-written Zarr layout."""
    return StreamingSpatialDataConverter(**_common(output_path), use_csc=True)


def _volume_3d(output_path):
    """``handle_3d=True`` -- one table for the whole volume, not per z-slice."""
    return SpatialData3DConverter(**_common(output_path))


# Name -> (factory, the table this path writes). The 3D converter emits a
# single volume table named after the dataset; the others emit one per
# z-slice and there is only ever a z0 here.
WRITE_PATHS: Dict[str, Tuple[Callable, str]] = {
    "in_memory": (_in_memory, _TABLE_NAME),
    "streaming_coo": (_streaming_coo, _TABLE_NAME),
    "streaming_pcs": (_streaming_pcs, _TABLE_NAME),
    "volume_3d": (_volume_3d, _DATASET_ID),
}

# The three paths that write a 2D slice table. Their obs schemas must match
# each other exactly; the volume table legitimately carries z as well.
_SLICE_PATHS = ("in_memory", "streaming_coo", "streaming_pcs")


def _convert(tmp_path_factory, path_name: str):
    """Convert once through one write path; hand back the store root."""
    factory, _ = WRITE_PATHS[path_name]
    output_path = tmp_path_factory.mktemp(f"prov_{path_name}") / "out.zarr"
    assert factory(output_path).convert() is True, f"{path_name} conversion failed"
    return output_path


@pytest.fixture(scope="module")
def stores(tmp_path_factory) -> Dict[str, Any]:
    """One conversion per write path. Module-scoped: converting is the slow part."""
    return {name: _convert(tmp_path_factory, name) for name in WRITE_PATHS}


def _table_path(stores, path_name: str):
    """The table group inside a converted store."""
    return stores[path_name] / "tables" / WRITE_PATHS[path_name][1]


def _read_uns(table_path) -> Dict[str, Any]:
    """Read ``uns`` back eagerly, as a plain dict."""
    import anndata as ad
    import zarr

    return ad.io.read_elem(zarr.open_group(str(table_path), mode="r")["uns"])


def _plain(value: Any) -> Any:
    """Normalise for comparison across paths.

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


@pytest.mark.parametrize("path_name", list(WRITE_PATHS))
def test_essential_metadata_is_not_empty(stores, path_name):
    """The reported symptom: the block present but with nothing in it.

    Deliberately the weakest assertion here, and deliberately first --
    a store whose provenance is empty cannot say what it came from at
    all, which is worse than one that says something inaccurate.
    """
    uns = _read_uns(_table_path(stores, path_name))

    assert "essential_metadata" in uns, f"{path_name} wrote no essential_metadata"
    essential = uns["essential_metadata"]
    assert essential, f"{path_name} wrote an EMPTY essential_metadata block"
    assert set(essential) == _EXPECTED_ESSENTIAL_KEYS


@pytest.mark.parametrize("path_name", list(WRITE_PATHS))
def test_stored_spectrum_type_is_the_readers(stores, path_name):
    """``spectrum_type`` must come from the data, not from a literal.

    The PCS path wrote ``"processed"`` for everything. That is not a
    value any extractor produces, so a consumer could neither trust it
    nor tell it apart from a real reading -- which is the whole failure:
    losing provenance *silently*.
    """
    essential = _read_uns(_table_path(stores, path_name))["essential_metadata"]

    assert essential["spectrum_type"] == _EXPECTED_SPECTRUM_TYPE, (
        f"{path_name} stored spectrum_type={essential['spectrum_type']!r}; "
        f"the reader reports {_EXPECTED_SPECTRUM_TYPE!r}"
    )
    assert essential["spectrum_type"] != "processed"


@pytest.mark.parametrize("path_name", list(WRITE_PATHS))
def test_essential_metadata_values_track_the_source(stores, path_name):
    """The rest of the block must describe the source, not the output."""
    essential = _read_uns(_table_path(stores, path_name))["essential_metadata"]
    cfg = _config()

    assert essential["source_path"] == _EXPECTED_SOURCE_PATH
    assert _plain(essential["dimensions"]) == [cfg.n_x, cfg.n_y, cfg.n_z]
    # mass_range is the SOURCE range. The PCS path used to take it from
    # the resampled target axis, which is a different quantity whenever
    # resampling clips or extends the range.
    assert _plain(essential["mass_range"]) == pytest.approx([cfg.mz_min, cfg.mz_max])
    assert essential["thyra_version"]


@pytest.mark.parametrize("path_name", list(WRITE_PATHS))
def test_vendor_sections_are_stored(stores, path_name):
    """The sections beside essential_metadata, which PCS dropped entirely.

    ``acquisition_params`` is deliberately not asserted: the mock leaves
    it empty and empty sections are omitted, so that a consumer can tell
    "this format has none" from "this one has none recorded".
    """
    uns = _read_uns(_table_path(stores, path_name))

    assert uns.get("format_specific") == {"format": "mock"}
    assert uns.get("instrument_info") == {"instrument": "mock"}
    raw = uns.get("raw_metadata") or {}
    assert raw.get("source") == "mock"
    assert "acquisition_params" not in uns
    assert uns.get("regions"), f"{path_name} stored no region summary"


@pytest.mark.parametrize("path_name", list(WRITE_PATHS))
def test_string_lists_are_stored_as_json(stores, path_name):
    """Non-numeric lists in the vendor sections come back as JSON strings.

    Stored as lists they do not survive the writer: a list of dicts is
    stringified entry by entry into ``repr`` output, and any list of
    strings reads back as a numpy string array -- whose ``deepcopy``
    segfaults the whole process on numpy 2.1-2.2 (numpy#28609), killing
    every consumer that copies the table (``AnnData.copy``,
    ``polygon_query``, joins). Purely numeric lists stay arrays.
    """
    import json

    raw = _read_uns(_table_path(stores, path_name))["raw_metadata"]

    assert isinstance(raw["cvParams"], str), (
        f"{path_name} stored cvParams as {type(raw['cvParams']).__name__}; "
        "a non-numeric list must be stored as a JSON string"
    )
    assert json.loads(raw["cvParams"]) == [
        {"name": "MS1 spectrum", "accession": "MS:1000579", "value": True},
    ]
    assert _plain(raw["scan_window"]) == [100.0, 1000.0]


@pytest.mark.parametrize("path_name", list(WRITE_PATHS))
def test_uns_contains_no_string_arrays(stores, path_name):
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

    uns = _read_uns(_table_path(stores, path_name))
    offenders = list(_string_arrays(uns, "uns"))
    assert not offenders, f"{path_name} stored string arrays at: {offenders}"


def test_every_path_stores_the_same_provenance(stores):
    """The invariant the individual assertions above are instances of.

    Compares whole blocks rather than named keys, so a section added to
    one path and forgotten on another fails here without anyone having
    to remember to extend this file.
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
        for name in WRITE_PATHS
    }

    reference_name = "in_memory"
    reference = blocks[reference_name]
    for name, block in blocks.items():
        if name == reference_name:
            continue
        assert (
            block == reference
        ), f"{name} stored different provenance from {reference_name}"


def _root_attrs(store_path) -> Dict[str, Any]:
    """The store's own root attributes, as a plain dict."""
    import zarr

    return dict(zarr.open_group(str(store_path), mode="r").attrs)


def test_every_path_writes_the_same_root_attrs(stores):
    """The store's own attrs, the second half of what a store says about itself.

    Same shape of invariant as the provenance comparison above and same
    reason for existing: the PCS path composed its root attrs by hand and
    was short by ``coordinate_systems``, ``format_specific_metadata`` and
    ``msi_dataset_info``.  Compares the key *set*, since three of these
    are timestamps, counts and versions that legitimately differ.
    """
    reference_name = "in_memory"
    reference = set(_root_attrs(stores[reference_name]))

    # Guard the guard: an empty or near-empty reference would make this
    # pass for every path while checking nothing.
    assert {
        "coordinate_systems",
        "format_specific_metadata",
        "msi_dataset_info",
    } <= reference

    for name in WRITE_PATHS:
        assert set(_root_attrs(stores[name])) == reference, (
            f"{name} root attrs differ from {reference_name}: "
            f"missing {sorted(reference - set(_root_attrs(stores[name])))}, "
            f"extra {sorted(set(_root_attrs(stores[name])) - reference)}"
        )


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


def test_slice_paths_write_the_same_obs_columns(stores):
    """``obs`` schema parity across the three paths that write a slice table.

    PCS hand-writes its ``obs`` group and simply had no ``region_number``
    column, so a consumer that branches on it saw a different schema
    depending on which side of the old size threshold the dataset fell.
    """
    reference_name = "in_memory"
    reference = set(_read_obs(_table_path(stores, reference_name)).columns)

    assert "region_number" in reference

    for name in _SLICE_PATHS:
        assert (
            set(_read_obs(_table_path(stores, name)).columns) == reference
        ), f"{name} obs columns differ from {reference_name}"


def test_volume_obs_adds_only_the_z_columns(stores):
    """The volume table's schema is the slice schema plus depth, nothing else.

    Asserted rather than skipped: "3D is different" is true only for the
    two columns a volume needs, and anything else appearing or vanishing
    there is the same class of drift this file exists to catch.
    """
    slice_columns = set(_read_obs(_table_path(stores, "in_memory")).columns)
    volume_columns = set(_read_obs(_table_path(stores, "volume_3d")).columns)

    assert volume_columns - slice_columns == {"z", "spatial_z"}
    assert not slice_columns - volume_columns


@pytest.mark.parametrize("path_name", list(WRITE_PATHS))
def test_every_path_writes_one_row_per_spectrum(stores, path_name):
    """Row count parity: one row per acquired spectrum, on every path.

    Three of the four dropped the grid positions that carry no spectrum
    (#88); the PCS path emitted the whole bounding box, so a quarter of
    this fixture's rows were all-zero phantoms. On real ``pea.imzML``
    that was 17,423 rows against 12,737 spectra.

    Also asserts that no surviving row is empty, which is the property
    the count is a proxy for: a path could reach the right *number* of
    rows while keeping the wrong ones.
    """
    obs = _read_obs(_table_path(stores, path_name))
    assert len(obs) == _N_SPECTRA, (
        f"{path_name} wrote {len(obs)} rows for {_N_SPECTRA} spectra "
        f"({_N_X * _N_Y} grid positions)"
    )

    x = _read_x(_table_path(stores, path_name))
    assert int(np.asarray(x.getnnz(axis=1)).min()) > 0, f"{path_name} kept empty rows"


@pytest.mark.parametrize("path_name", list(WRITE_PATHS))
def test_shapes_annotate_exactly_the_rows(stores, path_name):
    """One pixel polygon per obs row, under the same identity.

    Green before the phantom rows were dropped as well as after -- the
    PCS shapes tracked the phantom rows, so they were consistently wrong
    together. It is here as the guard on the fix: obs and shapes are
    built from two different places, and compacting one without the
    other is the obvious way to break a store while every count still
    looks plausible. Compares the labels rather than the count for the
    same reason: the index is the *grid* index and keeps its gaps, since
    what compacts is the row offset, not the identity.
    """
    import spatialdata

    sdata = spatialdata.read_zarr(str(stores[path_name]))
    obs = _read_obs(_table_path(stores, path_name))
    region_key = f"{WRITE_PATHS[path_name][1]}_pixels"

    assert list(sdata.shapes[region_key].index) == list(obs.index)


@pytest.mark.parametrize("path_name", list(WRITE_PATHS))
def test_tic_image_totals_the_matrix(stores, path_name):
    """``sum(TIC image) == sum(X)``, which dropping rows must not disturb.

    Also green on every route before the drop, and also here as a guard
    rather than a reproduction. The TIC image stays full-grid while the
    table drops empty positions, which is only sound because a dropped
    position contributes zero; if a route ever dropped a row that
    carried signal, or scattered one row's peaks onto another, these two
    totals would part company. It is the cheapest end-to-end check that
    the compaction moved rows rather than data.
    """
    import spatialdata

    sdata = spatialdata.read_zarr(str(stores[path_name]))
    tic_name = f"{WRITE_PATHS[path_name][1]}_tic"
    tic_total = float(np.asarray(sdata.images[tic_name].data).sum())
    x_total = float(_read_x(_table_path(stores, path_name)).sum())

    assert tic_total == pytest.approx(x_total, rel=1e-9)


@pytest.mark.parametrize("path_name", list(WRITE_PATHS))
def test_average_spectrum_is_a_mean_on_every_path(stores, path_name):
    """``uns["average_spectrum"]`` is the per-pixel mean, not the sum.

    The volume path stored ``total_intensity`` unscaled -- the sum over
    pixels, under a key named "average". The comment on it said "use
    total_intensity to match original behavior", carried through the
    refactor that split the converters and never revisited; the slice
    paths divide by ``pixel_count`` two lines from the same place.

    So a volume came out with a spectrum ``_N_SPECTRA`` times the one
    the other three paths wrote from identical data -- and, on a
    multi-region volume, a factor apart from the
    ``average_spectrum_per_region`` block beside it in the same ``uns``,
    which divides by the region's pixel count and always did. Nothing
    pinned it: no test read this key off the 3D path, and the scale
    factor is invisible in a plot with an unlabelled y-axis, so it reads
    as correct until a consumer compares two stores or trusts the axis.

    Derived from ``X`` rather than from ``total_intensity`` so the
    expectation comes from the stored matrix a consumer can see, not
    from the accumulator the writer used.
    """
    table_path = _table_path(stores, path_name)
    x = _read_x(table_path)
    total = np.asarray(x.sum(axis=0)).ravel()
    stored = np.asarray(_read_uns(table_path)["average_spectrum"], dtype=np.float64)

    np.testing.assert_allclose(
        stored,
        total / _N_SPECTRA,
        rtol=1e-9,
        err_msg=(
            f"{path_name} stored an average_spectrum that is not the mean "
            f"over the {_N_SPECTRA} acquired spectra"
        ),
    )

    # Guard the guard: with more than one spectrum the mean and the total
    # are different arrays, so the assertion above can tell them apart.
    assert _N_SPECTRA > 1
    assert not np.allclose(stored, total), f"{path_name} stored the total, not the mean"


@pytest.mark.parametrize("path_name", list(WRITE_PATHS))
def test_read_lazy_sees_the_block(stores, path_name):
    """The access mode a consumer actually uses.

    ``read_lazy`` is how Ousia opens a converted store without pulling
    the matrix into memory, and it is where the loss was noticed. Array
    entries come back as dask, so this checks the keys are reachable --
    the values are covered eagerly above.
    """
    anndata = pytest.importorskip("anndata")

    lazy = anndata.experimental.read_lazy(str(_table_path(stores, path_name)))
    essential = lazy.uns["essential_metadata"]

    assert set(essential) == _EXPECTED_ESSENTIAL_KEYS
    assert essential["spectrum_type"] == _EXPECTED_SPECTRUM_TYPE
