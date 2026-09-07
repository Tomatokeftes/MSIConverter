"""The common mobility grid, and the table it fills for a per-pixel source.

The grid itself is exercised on hand-picked values first: edges, channel
assignment, the clamp band, and the alignment with the mass-mobility
heatmap that the whole 256-channel anchor exists for. Then a stub reader
whose pixels each carry their own ``(m/z, 1/K0, intensity)`` cloud -- a
Bruker TDF in miniature, no SDK and no committed data -- goes down the
in-memory and streaming write routes, so the marginal invariant, the
``uns`` blocks and the refusals are checked on a real store.
"""

from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from thyra.converters.spatialdata.mobility_heatmap import (
    HEATMAP_MOBILITY_CHANNELS,
    MobilityHeatmap,
    mobility_bin_edges,
)
from thyra.converters.spatialdata.mobility_table import (
    GRID_VAR_REFUSE_FRACTION,
    MAX_GRID_VAR_ENTRIES,
    VAR_BYTES_PER_FEATURE,
    build_mobility_table,
    grid_refusal,
    grid_var_bound,
    mobility_grid_range,
    projected_var_gb,
    var_ceiling_refusal,
)
from thyra.core.base_extractor import MetadataExtractor
from thyra.core.base_reader import BaseMSIReader
from thyra.core.mobility import MobilityAxis
from thyra.metadata.types import ComprehensiveMetadata, EssentialMetadata
from thyra.resampling.mobility_grid import (
    MAX_CHANNEL_WIDTH,
    MIN_CHANNEL_WIDTH,
    MOBILITY_CHANNELS,
    LinearMobilityGridGenerator,
    MobilityGrid,
    build_mobility_grid,
    linear_channel,
)

# ----------------------------------------------------------------------
# The grid on its own
# ----------------------------------------------------------------------


class TestTheAnchor:
    def test_the_channel_count_is_the_heatmap_s_own_constant(self):
        # Not two constants that happen to agree: one, so a box drawn on
        # the heatmap maps onto grid channels by integer index.
        assert MOBILITY_CHANNELS == 256
        assert HEATMAP_MOBILITY_CHANNELS is MOBILITY_CHANNELS

    def test_default_edges_are_the_heatmap_s_edges_exactly(self):
        grid = build_mobility_grid(1.00003, 1.29133)
        edges = mobility_bin_edges(1.00003, 1.29133)
        # Equal arrays, not merely close: an approximate match would put a
        # heatmap box half a channel off the grid it is meant to index.
        np.testing.assert_array_equal(grid.edges, edges)

    def test_assignment_matches_the_heatmap_s_own_channels(self):
        lower, upper = 1.00003, 1.29133
        heatmap = MobilityHeatmap(np.linspace(100.0, 200.0, 11), (lower, upper))
        grid = build_mobility_grid(lower, upper)
        values = np.linspace(lower - 0.01, upper + 0.01, 5000)
        heatmap.add(np.full(values.size, 150.0), values, np.ones(values.size))
        counts = heatmap.finalize()["counts"]
        occupied_by_heatmap = np.flatnonzero(counts.sum(axis=0))
        occupied_by_grid = np.unique(grid.assign(values))
        np.testing.assert_array_equal(occupied_by_grid, occupied_by_heatmap)


class TestLinearGrid:
    def test_edges_and_centres(self):
        grid = build_mobility_grid(1.0, 1.4, n_channels=4)
        np.testing.assert_allclose(grid.edges, [1.0, 1.1, 1.2, 1.3, 1.4])
        np.testing.assert_allclose(grid.centres, [1.05, 1.15, 1.25, 1.35])
        assert grid.law == "linear"
        assert grid.channel_width == pytest.approx(0.1)

    def test_assign_places_a_value_in_the_channel_containing_it(self):
        grid = build_mobility_grid(1.0, 1.4, n_channels=4)
        channels = grid.assign([1.0, 1.05, 1.1, 1.25, 1.39999])
        np.testing.assert_array_equal(channels, [0, 0, 1, 2, 3])
        assert channels.dtype == np.int32

    def test_the_upper_edge_belongs_to_the_last_channel(self):
        grid = build_mobility_grid(1.0, 1.4, n_channels=4)
        assert int(grid.assign([1.4])[0]) == 3

    def test_a_value_past_either_edge_is_clipped_not_lost(self):
        # The per-scan axis overhangs its declared range, and a reader
        # may clip a scan number; a marginal must still keep every count.
        grid = build_mobility_grid(1.0, 1.4, n_channels=4)
        np.testing.assert_array_equal(grid.assign([0.5, 9.0]), [0, 3])

    @pytest.mark.parametrize("lower,upper", [(1.0, 1.0), (1.4, 1.0), (np.nan, 1.0)])
    def test_a_range_without_extent_is_refused(self, lower, upper):
        with pytest.raises(ValueError, match="extent"):
            build_mobility_grid(lower, upper)

    def test_a_channel_count_below_one_is_refused(self):
        with pytest.raises(ValueError, match="at least one channel"):
            build_mobility_grid(1.0, 1.4, n_channels=0)

    def test_an_unknown_law_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="Unknown mobility grid law"):
            build_mobility_grid(1.0, 1.4, law="constant_relative")

    def test_a_second_law_is_a_drop_in(self):
        # A law that supplies its own edges inherits assign() through the
        # binary search, which is the whole point of the split.
        class _Quadratic(LinearMobilityGridGenerator):
            def generate(self, lower, upper, n_channels):
                fractions = np.linspace(0.0, 1.0, n_channels + 1) ** 2
                return MobilityGrid(
                    law="quadratic",
                    lower=float(lower),
                    upper=float(upper),
                    n_channels=int(n_channels),
                    edges=lower + (upper - lower) * fractions,
                )

        grid = _Quadratic().generate(1.0, 1.4, 4)
        np.testing.assert_allclose(grid.edges, [1.0, 1.025, 1.1, 1.225, 1.4])
        np.testing.assert_array_equal(grid.assign([1.0, 1.03, 1.2, 1.4]), [0, 1, 2, 3])


class TestChannelWidthBand:
    def test_the_band_is_reported_never_moves_the_anchor(self, caplog):
        # A real acquisition's 0.29 span over 256 channels is finer than
        # the band's floor. The count is the anchor and does not move for
        # it; the width is said out loud instead.
        grid = build_mobility_grid(1.0, 1.29)
        assert grid.n_channels == MOBILITY_CHANNELS
        assert grid.channel_width < MIN_CHANNEL_WIDTH

    def test_a_coarse_grid_warns_that_separations_may_merge(self, caplog):
        from thyra.resampling.mobility_grid import report_channel_width

        grid = build_mobility_grid(1.0, 11.0, n_channels=4)
        assert grid.channel_width > MAX_CHANNEL_WIDTH
        with caplog.at_level("WARNING"):
            report_channel_width(grid)
        assert "above" in caplog.text


class TestLinearChannel:
    def test_is_the_shared_expression(self):
        values = np.array([1.0, 1.2, 1.4])
        np.testing.assert_array_equal(
            linear_channel(values, 1.0, 1.4, 4),
            build_mobility_grid(1.0, 1.4, 4).assign(values),
        )


# ----------------------------------------------------------------------
# A per-pixel mobility source, in miniature
# ----------------------------------------------------------------------

MASS_AXIS = np.array([100.0, 110.0, 120.0, 130.0])
PIXELS = [(0, 0), (1, 0), (0, 1), (1, 1)]

#: The per-scan 1/K0 axis: decreasing with scan number, as a TDF's is.
SCAN_K0 = np.linspace(1.5, 1.1, 9)

#: One pixel's raw cloud. m/z 100 is one column of the summed table and
#: two channels of the grid; m/z 120 likewise; the two points at 110 fall
#: in one cell and are collapsed before anything is buffered.
CLOUD_MZ = np.array([100.0, 100.0, 110.0, 110.0, 120.0, 120.0])
CLOUD_K0 = np.array([1.12, 1.48, 1.303, 1.3031, 1.201, 1.207])


def _cloud(pixel: int) -> Tuple[NDArray, NDArray, NDArray]:
    intensities = np.array([10.0, 1.0, 5.0, 2.0, 20.0, 3.0]) + pixel
    return CLOUD_MZ.copy(), CLOUD_K0.copy(), intensities


def _summed(pixel: int) -> Tuple[NDArray, NDArray]:
    """The summed spectrum of that cloud: what ``scan_sum`` produces."""
    mzs, _, intensities = _cloud(pixel)
    unique, inverse = np.unique(mzs, return_inverse=True)
    sums = np.bincount(np.asarray(inverse).ravel(), weights=intensities)
    keep = sums != 0
    return unique[keep], sums[keep]


class _StubExtractor(MetadataExtractor):
    def __init__(self):
        super().__init__(data_source=None)

    def _extract_essential_impl(self) -> EssentialMetadata:
        return EssentialMetadata(
            dimensions=(2, 2, 1),
            coordinate_bounds=(0.0, 1.0, 0.0, 1.0),
            mass_range=(100.0, 130.0),
            pixel_size=(10.0, 10.0),
            n_spectra=4,
            total_peaks=12,
            estimated_memory_gb=0.0,
            source_path="stub_grid",
            spectrum_type="centroid spectrum",
        )

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        return ComprehensiveMetadata(
            essential=self._extract_essential_impl(),
            format_specific={
                "format": "stub",
                "ion_mobility": {
                    "present": True,
                    "separation": "inverse reduced ion mobility",
                    "separation_accession": "MS:1002815",
                    "unit_accession": "MS:1002814",
                    "range": [1.1, 1.5],
                    "num_scans": 9,
                },
            },
            acquisition_params={},
            instrument_info={"instrument": "stub"},
            raw_metadata={},
        )


class GridStubReader(BaseMSIReader):
    """A source whose pixels each carry their own mobility point cloud.

    ``has_shared_mobility_axis`` is False, as a Bruker TDF reader's is: a
    frame is a point cloud and no two pixels are promised the same pairs.
    """

    def __init__(self, with_values: bool = True, mass_axis: Optional[NDArray] = None):
        super().__init__(Path("stub_grid"))
        self._with_values = with_values
        self._mass_axis = MASS_AXIS if mass_axis is None else mass_axis
        self.mobility_passes = 0

    def _create_metadata_extractor(self) -> MetadataExtractor:
        return _StubExtractor()

    @property
    def has_shared_mass_axis(self) -> bool:
        return True

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        return self._mass_axis.copy()

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator:
        for p, (x, y) in enumerate(PIXELS):
            mzs, intensities = _summed(p)
            yield (x, y, 0), mzs, intensities

    @property
    def has_ion_mobility(self) -> bool:
        return True

    @property
    def has_shared_mobility_axis(self) -> bool:
        return False

    def get_mobility_axis(self) -> Optional[MobilityAxis]:
        return MobilityAxis(
            kind_accession="MS:1002815",
            kind_name="inverse reduced ion mobility",
            unit_accession="MS:1002814",
            unit_name="volt-second per square centimeter",
            values=SCAN_K0.copy() if self._with_values else None,
            acq_range=(1.1, 1.5),
            source="stub",
        )

    def iter_mobility_spectra(self, batch_size: Optional[int] = None) -> Generator:
        self.mobility_passes += 1
        for p, (x, y) in enumerate(PIXELS):
            mzs, mobility, intensities = _cloud(p)
            yield (x, y, 0), mzs, mobility, intensities

    def close(self) -> None:
        pass


class _NoMobilityReader(GridStubReader):
    @property
    def has_ion_mobility(self) -> bool:
        return False


class TestGridRange:
    def test_comes_from_the_values_not_the_declared_range(self):
        # The declared acq_range is (1.1, 1.5) and so are the values here,
        # but the values are what is read -- a reader that has none gets
        # no grid rather than the declared range as a fallback.
        assert mobility_grid_range(GridStubReader()) == pytest.approx((1.1, 1.5))
        assert mobility_grid_range(GridStubReader(with_values=False)) is None


class TestRefusals:
    def test_a_source_without_mobility(self):
        grid = build_mobility_grid(1.1, 1.5)
        assert "no ion mobility" in str(
            grid_refusal(_NoMobilityReader(), MASS_AXIS, grid)
        )

    def test_no_grid_at_all(self):
        assert "no range to bin over" in str(
            grid_refusal(GridStubReader(), MASS_AXIS, None)
        )

    def test_an_empty_mass_axis(self):
        grid = build_mobility_grid(1.1, 1.5)
        assert "mass axis is empty" in str(
            grid_refusal(GridStubReader(), np.array([]), grid)
        )

    def test_a_grid_that_fits_is_not_refused(self):
        axis = np.linspace(100.0, 1000.0, 78_125)
        assert (
            grid_refusal(GridStubReader(), axis, build_mobility_grid(1.1, 1.5)) is None
        )


class TestVarCeiling:
    def test_the_bound_is_the_pairs_the_grid_spans(self):
        axis = np.linspace(100.0, 1000.0, 100_000)
        assert grid_var_bound(axis, build_mobility_grid(1.1, 1.5)) == 100_000 * 256

    def test_a_bound_over_the_ceiling_is_not_itself_a_refusal(self):
        # Real occupancy runs an order of magnitude below the bound (a
        # measured 200-frame timsTOF acquisition occupied 3.9M of a
        # possible 35.5M), so refusing on the bound would turn away
        # conversions that fit ninefold over.
        axis = np.linspace(100.0, 1000.0, 400_000)
        grid = build_mobility_grid(1.1, 1.5)
        assert grid_var_bound(axis, grid) > MAX_GRID_VAR_ENTRIES
        assert grid_refusal(GridStubReader(), axis, grid) is None

    def test_the_count_refuses_before_anything_is_allocated(self, monkeypatch, caplog):
        # The ceiling is checked the moment the count is known -- the end
        # of the discovery pass -- and before the memmaps, the labels or
        # the var exist. Nothing has been committed to when it fires,
        # which is the whole reason it exists; this pins that allocate()
        # is never reached on a refusal.
        import thyra.converters.spatialdata.mobility_table as module
        from thyra.converters.spatialdata import csc_assembly

        monkeypatch.setattr(module, "MAX_GRID_VAR_ENTRIES", 4)

        def never(self, scratch):
            raise AssertionError("allocate() ran after the ceiling refused")

        monkeypatch.setattr(csc_assembly.CscAssembly, "allocate", never)
        with caplog.at_level("WARNING"):
            table = build_mobility_table(
                GridStubReader(),
                _stub_obs(),
                MASS_AXIS,
                "stub_z0",
                "stub_z0_pixels",
                {},
                grid=build_mobility_grid(1.1, 1.5),
            )
        assert table is None
        assert "above the var ceiling" in caplog.text

    def test_a_span_too_wide_to_count_is_refused_before_the_read(self):
        # The discovery pass counts every (m/z bin, channel) pair the grid
        # spans in a dense array; a raw, unresampled axis of millions of
        # bins would push that into gigabytes. Said before anything is
        # read, as a property of the source, naming the lever.
        from thyra.converters.spatialdata.csc_assembly import MAX_COUNT_BYTES

        too_many_bins = MAX_COUNT_BYTES // 4 // MOBILITY_CHANNELS + 1
        refusal = grid_refusal(
            GridStubReader(),
            np.linspace(100.0, 1000.0, too_many_bins),
            build_mobility_grid(1.1, 1.5),
        )
        assert refusal is not None
        assert "--resample-bins" in refusal

    def test_the_count_is_what_is_refused_with_the_number_printed(self):
        # The absolute cap, with memory taken out of the question.
        plenty = 1e9
        assert var_ceiling_refusal(MAX_GRID_VAR_ENTRIES, available_gb=plenty) is None
        refusal = var_ceiling_refusal(MAX_GRID_VAR_ENTRIES + 1, available_gb=plenty)
        assert refusal is not None
        assert f"{MAX_GRID_VAR_ENTRIES + 1:,}" in refusal
        assert f"{MAX_GRID_VAR_ENTRIES:,}" in refusal
        assert "--mobility-bins" in refusal

    def test_the_operative_guard_is_the_projected_var_memory(self, caplog):
        # Design decision D4: the ceiling is a memory guard, not a constant
        # fitted to a dataset. 10M features project to ~3.1 GB at the
        # measured bytes-per-feature; that is refused on a machine with
        # 4 GB free (over half), warned about with 10 GB free (over a
        # quarter), and silent with 100 GB free.
        n = 10_000_000
        gb = projected_var_gb(n)
        assert gb == pytest.approx(n * VAR_BYTES_PER_FEATURE / 1024**3)

        refusal = var_ceiling_refusal(n, available_gb=4.0)
        assert refusal is not None
        assert f"{gb:.1f} GB" in refusal
        assert "4.0 GB free" in refusal
        assert "--resample-bins" in refusal and "--mobility-bins" in refusal

        with caplog.at_level("WARNING"):
            assert var_ceiling_refusal(n, available_gb=10.0) is None
        assert "projected to need" in caplog.text

        caplog.clear()
        with caplog.at_level("WARNING"):
            assert var_ceiling_refusal(n, available_gb=100.0) is None
        assert "projected to need" not in caplog.text

    def test_the_fractions_are_of_free_memory_not_fixed_sizes(self):
        # The same table is fine on a workstation and refused on a laptop.
        n = 10_000_000
        gb = projected_var_gb(n)
        just_enough = gb / GRID_VAR_REFUSE_FRACTION + 1e-6
        assert var_ceiling_refusal(n, available_gb=just_enough) is None
        assert var_ceiling_refusal(n, available_gb=just_enough * 0.9) is not None


def _stub_obs():
    import pandas as pd

    return pd.DataFrame(
        {
            "x": [x for x, _y in PIXELS],
            "y": [y for _x, y in PIXELS],
        },
        index=[str(i) for i in range(len(PIXELS))],
    )


# ----------------------------------------------------------------------
# End to end: the table on a real store, down both write routes
# ----------------------------------------------------------------------

spatialdata = pytest.importorskip("spatialdata")


def _convert(reader: BaseMSIReader, out: Path, streaming: bool, **kwargs):
    from thyra.utils.windows_paths import prepare_zarr_output_path

    out = prepare_zarr_output_path(out, "stub")
    if streaming:
        from thyra.converters.spatialdata.streaming_converter import (
            StreamingSpatialDataConverter as Converter,
        )
    else:
        from thyra.converters.spatialdata.spatialdata_2d_converter import (
            SpatialData2DConverter as Converter,
        )

    converter = Converter(reader, out, dataset_id="stub", pixel_size_um=10.0, **kwargs)
    assert converter.convert(), "conversion reported failure"
    return out


def _read(out: Path):
    from thyra.utils.windows_paths import prepare_zarr_read_path

    return spatialdata.read_zarr(prepare_zarr_read_path(out))


def _row(table, x: int, y: int) -> np.ndarray:
    obs = table.obs
    mask = (obs["x"].to_numpy().astype(int) == x) & (
        obs["y"].to_numpy().astype(int) == y
    )
    rows = np.flatnonzero(mask)
    assert rows.size == 1
    X = table.X[rows[0]]
    return (
        np.asarray(X.toarray()).ravel()
        if hasattr(X, "toarray")
        else np.asarray(X).ravel()
    )


@pytest.mark.parametrize("streaming", [False, True], ids=["in-memory", "streaming"])
class TestGridTable:
    def test_off_by_default_the_store_is_exactly_what_it_was(self, tmp_path, streaming):
        reader = GridStubReader()
        sdata = _read(_convert(reader, tmp_path / "s.zarr", streaming))
        assert set(sdata.tables) == {"stub_z0"}
        # The heatmap still costs its one pass; the grid costs none.
        assert reader.mobility_passes == 1

    def test_on_request_the_sibling_appears(self, tmp_path, streaming):
        reader = GridStubReader()
        sdata = _read(
            _convert(reader, tmp_path / "s.zarr", streaming, mobility_grid=True)
        )
        assert set(sdata.tables) == {"stub_z0", "stub_z0_mobility"}
        # The grid's discovery shares the heatmap's pass; only the scatter
        # is a pass of its own. Two reads of the raw points in all.
        assert reader.mobility_passes == 2
        summed, grid = sdata.tables["stub_z0"], sdata.tables["stub_z0_mobility"]
        assert summed.n_obs == grid.n_obs == 4
        assert list(summed.obs.index) == list(grid.obs.index)
        assert set(grid.obs["region"].astype(str)) == {"stub_z0_pixels"}

    def test_var_is_the_frozen_contract(self, tmp_path, streaming):
        sdata = _read(
            _convert(
                GridStubReader(), tmp_path / "s.zarr", streaming, mobility_grid=True
            )
        )
        var = sdata.tables["stub_z0_mobility"].var
        # Exactly the columns the shared-axis mechanism writes; a consumer
        # must not be able to tell the two apart from the table.
        assert list(var.columns) == ["mz", "mobility", "mz_index", "mobility_index"]
        mz = var["mz"].to_numpy()
        mobility = var["mobility"].to_numpy()
        assert np.all(np.diff(mz) >= 0)
        pairs = np.stack([mz, mobility], axis=1)
        assert np.array_equal(pairs, pairs[np.lexsort((mobility, mz))])
        assert len(set(map(tuple, pairs.tolist()))) == pairs.shape[0]
        channels = var["mobility_index"].to_numpy()
        assert channels.min() >= 0 and channels.max() <= MOBILITY_CHANNELS - 1
        # 100 and 120 each split into two channels; the two points at 110
        # share a cell and are one feature.
        np.testing.assert_array_equal(var["mz_index"].to_numpy(), [0, 0, 1, 2, 2])
        np.testing.assert_array_equal(channels, [12, 243, 129, 64, 68])
        assert list(var.index) == [
            "mz0_im12",
            "mz0_im243",
            "mz1_im129",
            "mz2_im64",
            "mz2_im68",
        ]

    def test_the_marginal_reproduces_the_summed_table_per_pixel(
        self, tmp_path, streaming
    ):
        sdata = _read(
            _convert(
                GridStubReader(), tmp_path / "s.zarr", streaming, mobility_grid=True
            )
        )
        summed = sdata.tables["stub_z0"]
        grid = sdata.tables["stub_z0_mobility"]
        mz_index = grid.var["mz_index"].to_numpy()
        for x, y in PIXELS:
            marginal = np.zeros(summed.n_vars)
            np.add.at(marginal, mz_index, _row(grid, x, y))
            np.testing.assert_allclose(marginal, _row(summed, x, y), rtol=0, atol=1e-12)

    def test_the_mobility_dimension_buys_a_split(self, tmp_path, streaming):
        sdata = _read(
            _convert(
                GridStubReader(), tmp_path / "s.zarr", streaming, mobility_grid=True
            )
        )
        grid = sdata.tables["stub_z0_mobility"]
        columns = np.flatnonzero(grid.var["mz_index"].to_numpy() == 0)
        assert columns.size == 2
        # One column of the summed table, two images that differ.
        first = np.asarray(grid.X[:, columns[0]].todense()).ravel()
        second = np.asarray(grid.X[:, columns[1]].todense()).ravel()
        assert not np.allclose(first, second)

    def test_the_uns_blocks_round_trip(self, tmp_path, streaming):
        sdata = _read(
            _convert(
                GridStubReader(), tmp_path / "s.zarr", streaming, mobility_grid=True
            )
        )
        summed = sdata.tables["stub_z0"]
        grid = sdata.tables["stub_z0_mobility"]

        block = grid.uns["mobility_grid"]
        assert set(block) == {
            "law",
            "lower",
            "upper",
            "n_channels",
            "channel_width",
            "edges",
        }
        assert str(block["law"]) == "linear"
        assert int(block["n_channels"]) == MOBILITY_CHANNELS
        # The heatmap on the summed table and the grid on the sibling share
        # their edges, which is what makes a box on one index the other.
        np.testing.assert_array_equal(
            np.asarray(block["edges"]),
            np.asarray(summed.uns["mobility_heatmap"]["mobility_edges"]),
        )
        assert not any(":" in key for key in grid.uns)
        # A shared-axis table has no such block; that absence is the only
        # thing that says which mechanism filled the table.
        assert "mobility_grid" not in summed.uns

    def test_the_marginal_ratio_is_recorded_not_only_asserted(
        self, tmp_path, streaming
    ):
        sdata = _read(
            _convert(
                GridStubReader(), tmp_path / "s.zarr", streaming, mobility_grid=True
            )
        )
        block = sdata.tables["stub_z0_mobility"].uns["mobility_marginal"]
        assert str(block["summed_table"]) == "stub_z0"
        assert float(block["current_ratio"]) == pytest.approx(1.0)
        assert float(block["current_ratio_pixel_min"]) == pytest.approx(1.0)
        assert float(block["current_ratio_pixel_max"]) == pytest.approx(1.0)
        if streaming:
            # The streaming route writes the summed table straight to disk
            # and never holds it, so the per-column deviation -- which
            # needs both matrices -- is the one field it cannot state.
            assert "max_relative_deviation" not in block
        else:
            assert float(block["max_relative_deviation"]) < 1e-12
            assert float(block["max_absolute_deviation"]) < 1e-9

    def test_the_metadata_block_names_the_grid(self, tmp_path, streaming):
        from thyra.metadata.schema import read_msi_metadata_blocks
        from thyra.utils.windows_paths import prepare_zarr_read_path

        out = _convert(
            GridStubReader(), tmp_path / "s.zarr", streaming, mobility_grid=True
        )
        blocks = read_msi_metadata_blocks(prepare_zarr_read_path(out))
        mobility = blocks["stub_z0"]["ms_analysis"]["ion_mobility"]
        assert mobility["resolved_table"] == "stub_z0_mobility"
        assert mobility["grid"] == {
            "law": "linear",
            "lower": pytest.approx(1.1),
            "upper": pytest.approx(1.5),
            "n_channels": MOBILITY_CHANNELS,
        }

    def test_validate_accepts_the_table(self, tmp_path, streaming):
        from thyra.metadata.schema import check_store_var_conventions
        from thyra.utils.windows_paths import prepare_zarr_read_path

        out = _convert(
            GridStubReader(), tmp_path / "s.zarr", streaming, mobility_grid=True
        )
        results = check_store_var_conventions(prepare_zarr_read_path(out))
        assert results, "no tables were checked"
        for issues in results.values():
            assert [i for i in issues if i.severity == "error"] == []

    def test_the_channel_count_and_range_can_be_overridden(self, tmp_path, streaming):
        sdata = _read(
            _convert(
                GridStubReader(),
                tmp_path / "s.zarr",
                streaming,
                mobility_grid=True,
                mobility_bins=8,
                mobility_min=1.0,
                mobility_max=1.5,
            )
        )
        grid = sdata.tables["stub_z0_mobility"]
        block = grid.uns["mobility_grid"]
        assert int(block["n_channels"]) == 8
        assert float(block["lower"]) == pytest.approx(1.0)
        np.testing.assert_array_equal(
            grid.var["mobility_index"].to_numpy(), [1, 7, 4, 3]
        )
        # Coarser channels merge the 1.201/1.207 pair into one feature.
        assert grid.n_vars == 4

    def test_a_source_with_no_axis_values_gets_no_table(self, tmp_path, streaming):
        sdata = _read(
            _convert(
                GridStubReader(with_values=False),
                tmp_path / "s.zarr",
                streaming,
                mobility_grid=True,
            )
        )
        assert set(sdata.tables) == {"stub_z0"}

    def test_a_source_over_the_var_ceiling_gets_no_table(
        self, tmp_path, streaming, monkeypatch
    ):
        # The count decides, so the ceiling is lowered to below the count
        # this fixture reaches rather than the axis being blown up to a
        # size no test should convert.
        import thyra.converters.spatialdata.mobility_table as module

        monkeypatch.setattr(module, "MAX_GRID_VAR_ENTRIES", 4)
        sdata = _read(
            _convert(
                GridStubReader(), tmp_path / "s.zarr", streaming, mobility_grid=True
            )
        )
        assert set(sdata.tables) == {"stub_z0"}

    def test_a_source_at_the_var_ceiling_still_gets_its_table(
        self, tmp_path, streaming, monkeypatch
    ):
        import thyra.converters.spatialdata.mobility_table as module

        monkeypatch.setattr(module, "MAX_GRID_VAR_ENTRIES", 5)
        sdata = _read(
            _convert(
                GridStubReader(), tmp_path / "s.zarr", streaming, mobility_grid=True
            )
        )
        assert sdata.tables["stub_z0_mobility"].n_vars == 5

    def test_a_source_without_mobility_gets_no_table(self, tmp_path, streaming):
        sdata = _read(
            _convert(
                _NoMobilityReader(), tmp_path / "s.zarr", streaming, mobility_grid=True
            )
        )
        assert set(sdata.tables) == {"stub_z0"}
        assert "mobility_grid" not in sdata.tables["stub_z0"].uns
