# Output Format

Thyra converts MSI data into [SpatialData](https://spatialdata.scverse.org/)
objects stored as Zarr directories. This page describes what the output
contains and how to work with it.

---

## Loading a Dataset

```python
import spatialdata as sd

sdata = sd.read_zarr("output.zarr")

print("Tables:", list(sdata.tables.keys()))
print("Images:", list(sdata.images.keys()))
print("Shapes:", list(sdata.shapes.keys()))
```

---

## Structure Overview

A converted dataset contains the following elements:

| Element | Key Pattern | Description |
|---------|------------|-------------|
| **Table** | `{dataset_id}_z{z}` | AnnData with intensity matrix (pixels x m/z), coordinates in `.obs`, m/z axis in `.var` |
| **TIC Image** | `{dataset_id}_z{z}_tic` | 2D total ion current image, shape `(1, y, x)` |
| **Pixel Shapes** | `{dataset_id}_z{z}_pixels` | GeoDataFrame with pixel box geometries |
| **Optical Images** | `{dataset_id}_optical_{name}` | Microscopy images (when available) |

!!! note "3D mode"
    When converted with `--handle-3d`, the `_z{z}` suffix is dropped and all
    slices are merged into a single table with `x`, `y`, `z` coordinates in
    `.obs`. The TIC image becomes a single **volume** of shape `(c, z, y, x)`
    rather than one 2D image per slice — see
    [3D Data / Z-Slices](#3d-data-z-slices).

    The pixel shapes stay two-dimensional: one flat square per pixel per slice,
    carrying no depth. Read a pixel's depth from the table's `spatial_z`
    instead. See [Pixel footprints in a volume](#pixel-footprints-in-a-volume)
    for why, and [Coordinate systems](coordinate-systems.md) for how the three
    elements line up.

!!! tip "Default dataset ID"
    The default `dataset_id` is `msi_dataset`, so typical keys look like
    `msi_dataset_z0`, `msi_dataset_z0_tic`, etc. Change it with `--dataset-id`.

!!! info "Coordinate systems"
    Every element above carries a transform to a single ``"global"``
    coordinate system, and Thyra writes a self-describing
    ``coordinate_systems`` metadata attr at the zarr top level so
    consumers know what ``"global"`` is in (micrometers or pixels).
    See [Coordinate Systems](coordinate-systems.md) for the contract
    and how to read it.

---

## TIC Images

The TIC (Total Ion Current) image shows the sum of all intensities per pixel.

```python
import numpy as np
import matplotlib.pyplot as plt

tic_key = [k for k in sdata.images if k.endswith("_tic")][0]
tic_array = np.asarray(sdata.images[tic_key])[0]  # drop channel dim -> (y, x)

plt.imshow(tic_array, cmap="viridis")
plt.colorbar(label="TIC Intensity")
plt.title("Total Ion Current")
plt.show()
```

---

## Optical Images

When converted with `--include-optical` (the default for Bruker data),
microscopy images are stored alongside the MSI data.

```python
optical_keys = [k for k in sdata.images if "optical" in k]
print("Optical images:", optical_keys)

if optical_keys:
    opt_image = np.asarray(sdata.images[optical_keys[0]])
    # Shape is (channels, y, x) -- transpose for display
    opt_rgb = np.moveaxis(opt_image[:3], 0, -1)
    plt.imshow(opt_rgb)
    plt.title("Optical Image")
    plt.show()
```

### TIC-to-Optical Overlay

The TIC image carries an affine transform that maps it into the optical image's
coordinate space. This means overlays work automatically in tools like napari.

To inspect the transform:

```python
from spatialdata.transformations import get_transformation

tic_element = sdata.images[tic_key]
transforms = get_transformation(tic_element, get_all=True)
affine = list(transforms.values())[0]

matrix = np.array(affine.to_affine_matrix(
    input_axes=("x", "y"), output_axes=("x", "y")
))
print(f"Scale: {matrix[0,0]:.2f}x, {matrix[1,1]:.2f}x")
print(f"Offset: ({matrix[0,2]:.0f}, {matrix[1,2]:.0f})")
```

!!! info "How alignment works"
    The optical image has an Identity transform and defines the reference
    coordinate system. The TIC image has an Affine transform (scale + offset)
    that positions it in the optical coordinate space. This comes from the
    teaching point calibration in the `.mis` file (Bruker data).

---

## Mass Spectrum Data

### Average Spectrum

Each table stores a pre-computed average spectrum in `uns`:

```python
table_key = list(sdata.tables.keys())[0]
msi_table = sdata.tables[table_key]

mz_values = msi_table.var["mz"].values
avg_spectrum = msi_table.uns["average_spectrum"]

plt.plot(mz_values, avg_spectrum, linewidth=0.5)
plt.xlabel("m/z")
plt.ylabel("Average Intensity")
plt.title("Average Mass Spectrum")
plt.show()
```

It is the **mean over the acquired spectra** -- the intensity summed down the
m/z axis and divided by the number of pixels that carry a spectrum, not by the
number of grid positions. Every write path stores it on that scale, including
the 3D volume path, which previously stored the undivided sum here.

### Per-Region Average Spectrum

For multi-region datasets, Thyra also stores a mean spectrum per acquisition
region in `uns["average_spectrum_per_region"]`. Each key is the region number
(as a string), and the value is a 1-D array matching the m/z axis.

```python
if "average_spectrum_per_region" in msi_table.uns:
    per_region = msi_table.uns["average_spectrum_per_region"]
    for region_id, spectrum in per_region.items():
        plt.plot(mz_values, spectrum, label=f"Region {region_id}", linewidth=0.5)
    plt.xlabel("m/z")
    plt.ylabel("Average Intensity")
    plt.legend()
    plt.title("Average Spectrum per Region")
    plt.show()
```

!!! note
    This key is only present when the dataset contains multiple acquisition
    regions. Single-region datasets only have the global `average_spectrum`.

### Intensity Matrix

The intensity matrix is stored as a sparse matrix. Each row is one pixel, each
column is one m/z bin:

```python
X = msi_table.X  # sparse (pixels x m/z)
print(f"Shape: {X.shape}")
print(f"Non-zero: {X.nnz:,} ({X.nnz / (X.shape[0] * X.shape[1]) * 100:.2f}%)")
```

!!! tip "Sparse format"
    The storage is CSC (Compressed Sparse Column) on every route, which is fast
    for extracting ion images (column = one m/z across all pixels). If you need
    fast per-pixel access, call `X.tocsr()` on the matrix you read back -- that
    is one conversion in memory, and it is why there is no format choice at
    write time.

    The stored matrix is canonical: within every column the row indices are in
    ascending order and each appears once, whatever order the source delivered
    its pixels in (a multi-area Bruker acquisition comes back area by area,
    for instance). A consumer can binary-search a column's indices directly,
    and scipy reports `has_sorted_indices` as true without a `sort_indices()`
    pass.

    One pixel is one row. A source that measures the same coordinate twice --
    a processed imzML with a repeated coordinate -- has both spectra **summed**
    into that row, which is what the pixel's TIC image and its
    `non_empty_pixels` count then describe as well. The conversion says so in
    the log, naming how many positions it applied to. Reading the store is not
    how you would find out otherwise: scipy and dask both merge repeated
    entries as they read, so `read_zarr` would show the sum whether or not the
    stored arrays held one entry or two.

!!! note "Every stored intensity is a real, non-negative number"
    Non-finite and negative intensities are dropped from a spectrum before it
    is resampled, and the conversion says how many at `WARNING` the first time
    it happens.

    A NaN is not a measurement, and one of them makes every aggregate over its
    column NaN -- the average spectrum, the TIC image, any ion image. A
    negative value is a baseline subtraction that overshot rather than a
    smaller measurement, and nothing that reads a store treats it as signal: a
    TIC, an ion image and a mean spectrum all read it as removing current that
    was never there.

    Dropping them before either resampling method runs is also what makes the
    two agree. Given a negative value, `nearest_neighbor` stored it (leaving
    the pixel with a TIC of 0) while `tic_preserving` found a non-positive
    total for the spectrum and zeroed the whole thing, so the same file
    converted two ways gave two different stores with nothing in either saying
    why.

    The sibling tables built from the raw scans -- the mobility grid, the
    demultiplexed MS/MS split -- drop the same points by the same rule, so a
    grid's marginal over channels still reproduces the summed table's column
    on a source that carries them.

    Zero is not affected by this; the sparse write drops explicit zeros as it
    always did. A source whose m/z values are non-finite is refused outright
    instead, since those values *are* `var["mz"]` and dropping them would
    desynchronise every index paired with them.

### Ion Images

To visualise the spatial distribution of a specific m/z value:

```python
target_mz = 760.5
mz_idx = np.abs(mz_values - target_mz).argmin()

# Extract column from sparse matrix
ion_values = np.asarray(X[:, mz_idx].toarray()).flatten()

# Reconstruct image from pixel coordinates
x_coords = msi_table.obs["x"].values.astype(int)
y_coords = msi_table.obs["y"].values.astype(int)

ion_image = np.zeros((y_coords.max() + 1, x_coords.max() + 1))
ion_image[y_coords, x_coords] = ion_values

plt.imshow(ion_image, cmap="hot")
plt.colorbar(label="Intensity")
plt.title(f"m/z {mz_values[mz_idx]:.4f}")
plt.show()
```

### Individual Spectra

```python
pixel_idx = 0
spectrum = X[pixel_idx].toarray().flatten()

plt.plot(mz_values, spectrum, linewidth=0.5)
plt.xlabel("m/z")
plt.ylabel("Intensity")
plt.title(f"Pixel {pixel_idx}")
plt.show()
```

### var columns

`var` always carries `mz` (numeric, finite, strictly increasing).
Readers whose native axis is not m/z keep that axis alongside it, and
annotation tools add `formula`, `adduct`, `annotation_source` and
`fdr` -- names fixed by the
[metadata schema](metadata-schema.md#var-column-conventions) so they
mean the same thing in every store. `thyra validate` checks the `mz`
contract on every table.

### Ion mobility

Mobility is a **feature coordinate**, next to m/z. It never enters `obs`, a
coordinate system or a transform: those say where a pixel is, mobility says
what was measured there. Two things follow.

The MSI table above is always **summed over mobility**. When the source had
a mobility dimension (Bruker TDF with TIMS engaged, or an imzML export that
carries a mobility array), the table's `uns` gains two blocks, and
`uns["msi_metadata"]["ms_analysis"]["ion_mobility"]` carries the schema-level
summary. A source without mobility writes none of them, so a consumer can
tell "summed over mobility" from "never had any".

**`uns["mobility_axis"]`** describes the axis the table was summed over.
Plain-name keys (never a CV accession: a colon is not a legal zarr key on
Windows), numpy arrays rather than lists:

| key | value |
|---|---|
| `present` | `True` |
| `type_name`, `type_accession` | the quantity: `inverse reduced ion mobility` / `MS:1002815` (TIMS), or drift time / `MS:1002476` |
| `unit_name`, `unit_accession` | e.g. `volt-second per square centimeter` / `MS:1002814` for 1/K0 |
| `n_scans` | length of `values` (the TIMS ramp length on a Bruker source) |
| `values` | `float64[n_scans]`: the native axis when it is shared across pixels -- the 1/K0 of every TIMS scan (from the vendor calibration, decreasing with scan number) or of every feature of a continuous imzML export; absent when mobility is per pixel |
| `acq_range` | `float64[2]`, `[lower, upper]`: the acquired range as declared, else the span of `values` |
| `calibration` | `{model_type, coefficients}`: the vendor `TimsCalibration` row, provenance only |
| `source` | `"bruker_tdf"` or `"imzml"` |
| `resolved_table` | element key of the mobility-resolved sibling table, when one was written |

**`uns["mobility_heatmap"]`** is the dataset's mean mass-mobility frame: the
raw `(m/z, mobility, intensity)` points of every pixel, binned and averaged
over pixels, accumulated from the raw scan read during conversion. It is the
discovery surface -- look at it to see whether mobility separates anything
before asking for a mobility-resolved table -- and what a viewer's mobility
panel draws.

| key | value |
|---|---|
| `mz_edges` | `float64[m + 1]`: bin edges on the common m/z axis, which is coarsened by an integer factor to about 4,000 bins (`m` is the axis length itself when it is shorter) |
| `mobility_edges` | `float64[k + 1]`, **`k = 256`**: equal-width bins in the axis unit, ascending, spanning the axis `values` |
| `counts` | `float32[m, k]`: mean intensity per bin over pixels |
| `current_ratio` | `float`: what fraction of `uns["average_spectrum"]`'s ion current the heatmap holds -- 1.0 under `scan_sum`, about 0.85 under `vendor_centroid` |

`k = 256` is fixed on purpose: the mobility grid table below defaults to the
same 256 channels over the same edges -- literally the same constant and the
same generator -- so a box drawn on the heatmap maps onto grid channels by
integer index in both directions.
The m/z binning is the converter's own nearest-bin rule, coarsened, so under a
lossless summed spectrum (`--tdf-spectrum scan_sum`) the heatmap summed over
mobility, `counts.sum(axis=1)`, equals `uns["average_spectrum"]` coarsened to
`mz_edges`. Under `vendor_centroid` it does not: the centroid keeps only the current
inside the peaks its picker assigns (87 to 96 percent on the acquisitions
measured, see [Design Decisions](design-decisions.md#d1-which-spectrum-a-reader-takes))
and merges bins, while the heatmap is built from every raw point. Which case a given store is in is recorded rather than left to be found by subtraction: `current_ratio` is the heatmap's total over the stored mean spectrum's, the same number `uns["mobility_marginal"]` carries for the grid table, and the conversion says it at `WARNING` when it is not 1. On a Bruker source the heatmap costs one extra
library call per frame (about a millisecond); `--no-mobility-heatmap` skips
it.

```python
heat = table.uns["mobility_heatmap"]
counts = np.asarray(heat["counts"])              # (m, 256)
mz_centres = np.asarray(heat["mz_edges"])
mz_centres = (mz_centres[:-1] + mz_centres[1:]) / 2
k0_centres = np.asarray(heat["mobility_edges"])
k0_centres = (k0_centres[:-1] + k0_centres[1:]) / 2
plt.pcolormesh(k0_centres, mz_centres, np.log1p(counts))
plt.xlabel("1/K0 (V s cm^-2)"); plt.ylabel("m/z")
```

Both blocks are summaries of the whole dataset. Nothing in them is per pixel,
and nothing in them enters `obs` or a coordinate system.

When every pixel shares one set of `(m/z, mobility)` feature pairs -- a
continuous imzML export with a mobility array, as TIMSCONVERT and
TIMSImaging write -- Thyra also writes a **mobility-resolved sibling
table**, `{table}_mobility`, unless `--no-mobility-table` is given:

| | MSI table `{id}_z0` | mobility table `{id}_z0_mobility` |
|---|---|---|
| rows | pixels | the same pixels, same `obs`, same `region` |
| `var["mz"]` | strictly increasing, unique | **non-decreasing with duplicates** |
| `var["mobility"]` | absent | the feature's 1/K0 (or drift time) |
| sort | by `mz` | lexicographic `(mz, mobility)` |
| also | | `mz_index` (column on the MSI axis), `mobility_index` (rank of the mobility value, or the grid channel), `uns["feature_axis"]`, `uns["mobility_axis"]` |

Two isomers at one m/z that separate in mobility are one column in the MSI
table and two in the mobility table. The `(mz, mobility)` sort means an m/z
window is one contiguous column block whose mobility-summed image equals the
MSI table's, and a mobility window is a mask inside that block.

**Consumers must discriminate on `"mobility" in var.columns`**, never on the
element name: the mobility table is not a second MSI dataset, and code that
assumes a unique, strictly increasing `var["mz"]` must not be pointed at it.
`thyra validate` applies the pair contract (non-decreasing `mz`, unique
sorted pairs) to tables carrying `mobility` and the strict contract to all
others.

#### The same table from a common mobility grid

A source whose mobility values differ per pixel -- a Bruker TDF, where a
frame is a point cloud and no two pixels are promised the same `(m/z, 1/K0)`
pairs -- has no shared feature axis to read off. `--mobility-grid` fills the
**same** `{table}_mobility` element for it by binning: every pixel's points
go onto one set of mobility channels shared across the conversion, and
`(m/z bin, channel)` becomes the feature axis.

The two mechanisms produce the same kind of table -- same element key, same
`var` columns, same sort, same discriminator -- and a consumer does not need
to tell them apart to read either. What says which one filled it is
`uns["mobility_grid"]`, present only on a binned table:

| key | value |
|---|---|
| `law` | how the channel edges are spaced; `"linear"` (equal width in the axis unit) is the only law today |
| `lower`, `upper` | the range the channels span, in the axis unit |
| `n_channels` | how many channels, **256 by default** |
| `channel_width` | `(upper - lower) / n_channels`, recorded so a reader can see it without arithmetic |
| `edges` | `float64[n_channels + 1]`, the channel edges themselves |

`msi_metadata.ms_analysis.ion_mobility.grid` carries `law`, `lower`, `upper`
and `n_channels` too, so a consumer reading only the summed table finds them.

Three things are worth knowing before asking for one:

- **The edges come from the axis values, never the declared acquisition
  range.** A real file's per-scan 1/K0 overhangs its declared
  `OneOverK0AcqRange` by a few scans (1.00003 to 1.29133 against a declared
  1.0 to 1.29 on one measured acquisition), and `uns["mobility_heatmap"]`
  already bins over the values. `--mobility-min` / `--mobility-max` override
  them, at the cost of the alignment below.
- **256 channels is an alignment anchor, not a tuning knob.** It is the
  heatmap's own channel count over the heatmap's own edges, so a box drawn on
  the heatmap selects grid channels by integer index with no resampling and
  no edge off-by-one. `--mobility-bins` changes it and gives that up. The
  width the anchor realizes on a typical 0.29 1/K0 span is 0.0011, finer than
  the 0.002 to 0.02 band TIMS resolving power supports; that is said at
  `INFO` and the count is not moved for it, because the alignment is worth
  more than the size.
- **Its marginal reproduces the summed table under the default
  `--tdf-spectrum scan_sum`.** A grid table is built from raw scans, so its
  marginal reproduces the summed table only when the summed table was built
  from the same scans, which the default is. An explicit `vendor_centroid`,
  a peak-picked spectrum over the same ramp, is kept and said at `WARNING`;
  `uns["mobility_marginal"]` then records by how much the two differ.

**The marginal invariant.** Summing a grid table's channels within one m/z
bin reproduces that bin's column of the summed table, per pixel. That is what
`uns["mobility_marginal"]` records rather than merely asserting:

| key | value |
|---|---|
| `summed_table` | element key of the table the marginal is compared against |
| `current_ratio` | total ion current of the grid table over the summed table's; exactly `1.0` under `scan_sum` |
| `current_ratio_pixel_min` / `_max` | the same ratio across pixels |

The comparison is per pixel, one bounded pass over each table's memmaps. A
per-cell deviation (`max_absolute_deviation` / `max_relative_deviation`)
was recorded up to v3.21 by the in-memory converter only; it needed the
marginal and its difference from the summed table materialised, each as
large as the summed table, and went with that converter (see
[Design Decisions](design-decisions.md#d11-one-converter)). The snippet
below computes it from the two stored matrices when it is wanted.

```python
grid = sdata.tables["msi_dataset_z0_mobility"]
mz_index = grid.var["mz_index"].to_numpy()
marginal = np.zeros(sdata.tables["msi_dataset_z0"].n_vars)
np.add.at(marginal, mz_index, np.asarray(grid.X[0].todense()).ravel())
# equals the summed table's row 0, to floating point
```

A grid is refused, at `INFO` or `WARNING` and never as an exception, when:

| refused when | because |
|---|---|
| the source has no mobility dimension | there is nothing to bin |
| `--mobility-grid` was not given | binning is opt in: it costs a pass over the source and a much larger table |
| the mobility axis carries no per-scan values | a reader opened without its vendor library cannot supply them, and the declared range is not a substitute |
| the grid spans more pairs than the counting pass can hold (a raw, unresampled axis of millions of bins) | the count array is `4 bytes x bins x channels`, capped at 1 GB; resample to fewer mass bins |
| the `var` frame of the occupied `(m/z bin, channel)` pairs is projected to take more than half of the machine's free memory (330 bytes per pair, measured), or the pairs pass an absolute cap of 100,000,000 | the count, the projection and the free memory are printed; resample to fewer mass bins or ask for fewer channels. A projection past a quarter of free memory is attempted with a `WARNING`. See [Design Decisions](design-decisions.md#d4-the-grids-feature-ceiling-is-a-memory-guard-not-a-format-limit) |

**How it is built, and why its size does not matter.** The table is built
the way the summed table is built, in two passes over the raw scans. The first pass counts, per `(m/z bin, channel)` cell of the
grid, how many pixels occupy it -- a dense count over the grid's span, 142 MB
on a default-resampled timsTOF axis, sized before the first pixel is read and
independent of how many pixels there are. The second pass scatters each
pixel's cells straight into memmapped CSC arrays in a scratch directory next
to the output (`.thyra_mobility_*`, removed once the table is written), 12
bytes of disk per stored non-zero. Nothing is ever held for the whole image:
memory is the count array plus one frame. On a Bruker TDF both passes are
the summed table's own: the converter reads each frame once per pass
and derives the summed spectrum, the heatmap's points, the grid's cells and
the MS/MS split from that one read, so none of the siblings adds a read of
the source (see [Design Decisions](design-decisions.md#d5-one-raw-read-per-frame-per-pass-serves-every-table)).
Every point is mapped onto the mass axis once per pass and shared between
the heatmap and the grid, since the mapping costs more than the vendor read.

The size ceiling is on the pairs that carry signal, not on the pairs the grid
spans. The two differ by an order of magnitude -- 200 frames of a measured
timsTOF acquisition occupied 3.9M of a possible 35.5M -- so refusing on the
span would turn away conversions that fit ninefold over. The span is said at
`INFO` when it passes the ceiling; the count is what refuses, checked the
moment the first pass ends and before a single value has been scattered, so a
refusal costs one read and no memory.

Bruker TDF is the only source that needs a grid today; an imzML export with a
mobility array already has a shared feature axis and is read off it.

### Fragmentation (MS/MS)

An MS/MS imaging run measures fragments, so `var["mz"]` is fragment m/z.
Nothing about the axis says so -- a converted MS/MS store is otherwise shaped
exactly like an MS1 one -- so when the source reports fragmentation the table
carries **`uns["msms_schedule"]`**:

| key | value |
|---|---|
| `ms_level` | `2` for a fragment spectrum; the block is absent for MS1 |
| `n_windows` | number of precursors isolated per pixel |
| `merges_precursors` | `True` when more than one, so the stored spectrum sums them |
| `constant_across_pixels` | whether every pixel was fragmented on the same schedule |
| `isolation_window_target` | `float64[n]`: the isolated m/z of each window |
| `isolation_window_lower_offset` / `_upper_offset` | `float64[n]`: the window spans `target - lower` to `target + upper` |
| `collision_energy` | `float64[n]`: in electronvolts |
| `scan_begin` / `scan_end` | `int64[n]`: the mobility scans each window occupies, when the source separates them that way (Bruker PASEF) |
| `resolved_table` | element key of the demultiplexed sibling table, when one was written |

Field names and CV terms follow
[mzPeak](https://github.com/HUPO-PSI/mzPeak)'s `spectra_metadata_precursors`:
`ms_level` is `MS:1000511`, the isolation terms are `MS:1000827` / `828` /
`829`, and activation is `MS:1000133` with `MS:1000045` collision energy in
`UO:0000266`. The accessions travel as *values* (`..._accession` keys), never
as dict keys -- a colon is not a legal Windows path character and zarr writes
a key as a directory name. A source reporting a single full isolation width
has it halved into two equal offsets.

The same facts appear in the versioned schema block as
`uns["msi_metadata"]["ms_analysis"]["fragmentation"]`, where the precursor
list is a JSON string (a list of objects does not round-trip through
AnnData/zarr); `read_msi_metadata_blocks` and `thyra validate` decode it.

!!! warning "A multi-precursor pixel is a chimera"
    The MSI table sums a frame into one spectrum per pixel. When
    `merges_precursors` is `True`, that spectrum holds fragments of every
    precursor the frame isolated, with nothing marking which came from
    which -- so it must not be read as the fragment spectrum of any one of
    them. Conversion says so at `WARNING`, and by default also writes the
    split apart as a second table, below (`--no-msms-table` turns that
    off); the MSI table itself is unchanged either way.

```python
if "msms_schedule" in table.uns:
    sched = table.uns["msms_schedule"]
    print("MS level:", sched["ms_level"])
    for mz, ce in zip(sched["isolation_window_target"], sched["collision_energy"]):
        print(f"  precursor {mz:.3f} at {ce:.1f} eV")
```

#### Demultiplexed MS/MS table

When the source isolates several precursors per pixel in **disjoint
mobility scan ranges** (Bruker PASEF -- the targeted MALDI variant is
Bruker's `iprm-PASEF`, which serially fragments a scheduled list of
precursors at every pixel), Thyra also writes them split apart as a
sibling table, `{table}_msms`, by default (`--no-msms-table` opts out; see
[Design Decisions](design-decisions.md#d2-the-msms-table-is-written-by-default-when-the-schedule-qualifies)).
Each precursor's block is its **precursor
ion image**, and each column inside the block is one fragment's image:

| | MSI table `{id}_z0` | MS/MS table `{id}_z0_msms` |
|---|---|---|
| rows | pixels | the same pixels, same `obs`, same `region` |
| a column | one m/z bin, all precursors summed | one m/z bin **of one precursor** |
| `var["mz"]` | strictly increasing, unique | the fragment m/z, **restarting at every precursor** |
| `var["precursor_mz"]` | absent | the isolated m/z the fragments came from |
| sort | by `mz` | lexicographic `(precursor_mz, precursor_mobility, mz)` |
| also | | `precursor_mobility` (the 1/K0 it was isolated at), `precursor_index` (its position in this store's precursor axis), `mz_index` (column on the MSI axis), `uns["feature_axis"]`, `uns["msms_schedule"]`, `uns["demultiplexed_current"]` |

The fragment axis is the MSI table's own mass axis: `var["mz"]` is
`msi.var["mz"][var["mz_index"]]`, so a column of this table and the
corresponding column of the summed table are the same m/z bin. One
precursor's fragments are therefore a contiguous column block whose row
sums are its ion image, and **the blocks add back up**: summing all
fragment columns of every precursor reproduces the summed table's TIC per
pixel, because each recorded point falls in exactly one isolation window.
Exactly, under the default `--tdf-spectrum scan_sum`: the split is built
from the raw scans, and only a summed spectrum built from the same scans
can add back up to it. An explicit `vendor_centroid` keeps only the current
inside the peaks it picks (87 to 96 percent on the acquisitions measured),
so the two tables of one store would genuinely not agree; that is said at
`WARNING` and recorded in `uns["demultiplexed_current"]`.

```python
msms = sdata.tables["msi_z0_msms"]
# Look the precursor up once, then slice on its block index: never
# compare precursor_mz with == , and never assume an m/z is unique.
var = msms.var
index = var["precursor_index"][np.argmin(np.abs(var["precursor_mz"] - 936.578))]
block = (var["precursor_index"] == index).to_numpy()
image = np.asarray(msms.X[:, block].sum(axis=1)).ravel()
```

**Consumers must discriminate on `"precursor_mz" in var.columns`**, never
on the element name. `thyra validate` checks that `precursor_mz` is
non-decreasing, that each `precursor_index` owns one contiguous block, and
that `mz` increases strictly inside it; the strict single-column contract
applies to every other table.

!!! warning "Two precursors can share an m/z"
    A method may isolate the same mass at two mobility positions -- that is
    how an **isomer pair** is targeted, and separating them is what the
    mobility dimension is for. Thyra keeps them as two column blocks and
    never sums them back together. So:

    - the block identity is **`precursor_index`**, not `precursor_mz`;
    - `(precursor_mz, mz)` may legitimately repeat, which is why
      validation does not use that pair;
    - `precursor_mobility` is what tells the two apart (the 1/K0 at the
      middle of the window's scan range -- a window spans a slice of the
      ramp, and a scheduled method reports no apex).

!!! danger "Aligning two datasets"
    `precursor_index` is a position in **one store's** precursor axis and
    means nothing outside it: two samples whose schedules differ in length
    give the same index to different precursors. Align on
    `(precursor_mz, precursor_mobility)` -- by matching the pair, never by
    comparing it with `==`: two runs acquired back to back agreed on the
    1/K0 bitwise, while two of the same method on different days came out
    up to 4.0e-3 apart, because each file carries its own TIMS calibration.
    The `var` index labels are named
    after the precursor's m/z for the same reason -- `p936.578_mz1732`,
    never `p14_mz1732` -- so `anndata.concat` cannot silently merge two
    unrelated precursors. Measured on two real acquisitions in opposite
    polarities, 15 precursors against 13 with none in common: **zero**
    labels shared, against **5,262** a rank-named scheme would have shared,
    every one of them naming two different precursors. Differing mass axes
    do not save you there -- both started at m/z 50, so the low `mz_index`
    values line up. Concatenate with `join="inner"` to keep
    a `var` at all: on an outer join every column missing from one side
    makes `merge="unique"` drop the whole column, while on an inner join it
    keeps `precursor_mz`, `mz` and `mz_index` and drops exactly
    `precursor_index` and `precursor_mobility` -- anndata finding, on its
    own, which of the four travel between stores. The table carries no
    `mobility`
column: the scan range is how the precursors are *separated*, not what
they are *indexed by*, and a table matching both discriminators would tell
a consumer nothing about which kind it holds.

The split is a filter on the scan number, never an estimate, so Thyra
refuses rather than approximates and says at `INFO` which condition
failed. They are asked in this order, which is by how much each says about
the acquisition rather than by how cheap it is to check:

| refused when | because |
|---|---|
| the schedule varies from pixel to pixel | the precursors are not a global feature axis |
| the source records no precursor for the fragment frames | there is nothing to separate them by |
| there is one precursor | the summed table already *is* its fragment spectrum |
| the isolation windows overlap, or carry no scan range | they cannot be separated by mobility alone |

The order matters because the last three are facts about what the source
recorded and the first is a fact about the method. A diaPASEF file puts its
windows in `DiaFrameMsMsWindows`, which the TDF reader does not read, so it
arrives with no windows at all *and* survey frames mixed in among the
fragment ones -- and reporting a count first would describe a run isolating
32 precursors as one isolating a single precursor. A data-dependent
(DDA) PASEF run trips the first row and the last at once -- one measured
holds 14,952 distinct windows, each in one to eight frames, overlapping on
the ramp -- and the varying schedule is the design while the overlap is
only its consequence.

A refusal writes no sibling table, is never an exception and never touches
the summed table. Bruker TDF is the only source that reports what the
split needs today.

**`uns["demultiplexed_current"]`** records how much of the summed table's
ion current the split holds: `current_ratio` over the whole image, and
`current_ratio_pixel_min` / `_max` across pixels. Under the default
`--tdf-spectrum scan_sum` it is exactly `1.0`.
Under an explicit `--tdf-spectrum vendor_centroid` it is **above** 1 -- the
vendor peak picker drops the index bins it assigns to no peak while the
split reads raw scans,
which on a real acquisition is about 1.5% overall and up to 1.14x on a
single pixel. The two tables genuinely do not add up in that mode, and this
block is where the store says so. It compares the two tables' per-pixel
ion current, one bounded pass over each.

!!! note "Deliberate limits"
    - **The feature axis depends on the data.** Only `(precursor, bin)`
      pairs that carry signal become columns, so two datasets converted
      with identical settings get different `var`. The alternative is
      precursors x the whole mass axis -- millions of empty columns -- and
      is not worth it. Align on the intrinsic columns, as above.
    - **The fragment axis is borrowed from MS1.** It is the summed table's
      mass axis, so a resampling grid chosen for intact ions also sets
      fragment resolution. That coupling is what makes `mz_index`
      meaningful and the conservation check exact; it is a choice, not a
      necessity.
    - **Do not select precursors by float equality.** Look the precursor up
      once and slice on `precursor_index` within that store.
    - **Built out of core, like the mobility grid.** Two passes over the
      precursor spectra -- count the occupied `(precursor, bin)` pairs,
      then scatter into memmapped CSC arrays in a scratch directory next to
      the output (`.thyra_msms_*`) -- so memory is the count array
      (`4 bytes x precursors x mass bins`) plus one frame, whatever the
      pixel count. Those are the summed table's own two passes, fed from
      the same frame read, so the split adds no read of the source. The largest acquisition this has run on is still 713
      pixels with 15 precursors; `var` grows with the occupied pairs.
    - **Two acquired schedules have been tested, from one instrument.** 15
      precursors in positive mode and 13 in negative sharing none of them,
      over four images of 487 to 713 pixels, plus a third shape made by
      deleting one window from a copy, the single-precursor refusal on a
      real `MsMsType = 2` acquisition and the varying-schedule refusal on
      real DDA-PASEF (see
      [Design Decisions](design-decisions.md#d2-the-msms-table-is-written-by-default-when-the-schedule-qualifies)).
      All four come from the same instrument and method family, and none
      exceeds 713 pixels.
    - **A label means the same bin only when the axes match.** `mz_index`
      is a column of *that store's* mass axis, and the default resampling
      grid follows the acquisition range, so two runs share an axis only
      when they share that range -- 599,146 bins to m/z 1000 against
      635,610 to 1200 on the two measured. Check `var["mz"]`, not just the
      label, before reading across stores.

!!! note "Relation to other MS/MS imaging representations"
    The open formats solve this at the raw layer by never merging: an
    imzML or mzML spectrum carries its own precursor, and mzPeak links a
    spectrum row to a precursor table by index. The chimera is created by
    the analysis layer's one-spectrum-per-pixel model, so this table is
    that per-spectrum precursor reference translated onto a feature axis.

    Feature-based workflows (MZmine's SIMSEF, for instance) instead attach
    one representative MS2 spectrum to each MS1 feature, which identifies
    the feature but keeps no spatial information about the fragments. This
    table is the stronger form of the same data: summing a precursor's
    block collapses it to that per-precursor ion image, while keeping the
    block gives every fragment its own image -- which is what a spatial
    co-localisation check between a fragment and its precursor needs.

    The `(precursor_mz, mz)` feature-pair layout is Thyra's own; no open
    analysis-layer convention for per-precursor ion images exists to
    follow. The vocabulary is not: `ms_level`, the isolation window terms
    and the activation terms are PSI-MS, spelled as mzPeak spells them.

---

## Pixel Coordinates

Coordinates are stored in the table's `.obs` DataFrame:

```python
print("Columns:", list(msi_table.obs.columns))
```

| Column | Type | Description |
|--------|------|-------------|
| `x`, `y` | int | Raster grid coordinates (pixel indices) |
| `spatial_x`, `spatial_y` | float | Physical coordinates in micrometers |
| `region` | categorical | SpatialData region key |
| `region_number` | int | Acquisition region number |

The DataFrame index is `instance_id` (a string pixel identifier).

---

## Regions

Datasets acquired from multi-region slides (e.g., multiple tissue sections on
one slide) store region information in two places.

### Per-Pixel Region Number

The `region_number` column in `.obs` indicates which acquisition region each
pixel belongs to:

```python
print(msi_table.obs["region_number"].value_counts())
```

### Region Summary

Region metadata -- including human-readable names from the instrument's Area
definitions -- is stored as JSON in `uns`:

```python
import json

regions = json.loads(msi_table.uns["regions"])
for r in regions:
    print(f"Region {r['region_number']}: {r.get('name', 'unnamed')} "
          f"({r['n_spectra']:,} spectra)")
```

Example output:

```
Region 0: E2506 (104,321 spectra)
Region 1: Matrix (1,053 spectra)
```

!!! tip "Filtering by region"
    To work with only one region:
    ```python
    mask = msi_table.obs["region_number"] == 0
    tissue_table = msi_table[mask]
    ```

---

## 3D Data / Z-Slices

By default, Thyra stores each z-slice as a separate table and TIC image:

```python
slice_tables = sorted(k for k in sdata.tables if "_z" in k)
print(f"{len(slice_tables)} z-slices: {slice_tables}")

# Access a single slice
z0_table = sdata.tables[slice_tables[0]]
print(f"Slice 0: {z0_table.shape}")
```

### 3D mode (`--handle-3d`)

When converted with `--handle-3d`, all slices are combined into a single table
with `x`, `y`, `z` coordinates in `.obs`, and the per-slice TIC images are
replaced by one **volume**:

| Element | Key | Shape / dims |
|---------|-----|--------------|
| **Table** | `{dataset_id}` | pixels x m/z, with `x`, `y`, `z` and `spatial_x`, `spatial_y`, `spatial_z` in `.obs` |
| **TIC volume** | `{dataset_id}_tic` | `(c, z, y, x)` — one channel, then the three spatial axes |
| **Pixel Shapes** | `{dataset_id}_pixels` | GeoDataFrame of 2D pixel boxes — one per pixel per slice, no z |

```python
volume = sdata.images[f"{dataset_id}_tic"]
print(volume.dims)                      # ('c', 'z', 'y', 'x')

arr = np.asarray(volume.data)[0]        # drop channel -> (z, y, x)
plt.imshow(arr[0], cmap="viridis")      # first slice
```

Note the axis order is `(c, z, y, x)`, not `(c, x, y, z)`: index a slice with
`arr[z]`, not `arr[..., z]`.

#### Voxel depth

The volume carries a `Scale` to `"global"` built from **two** distinct numbers —
the in-plane pixel pitch for `x` and `y`, and the slice spacing for `z`:

```python
from spatialdata.transformations import get_transformation

axes = ("c", "z", "y", "x")
matrix = get_transformation(volume, to_coordinate_system="global").to_affine_matrix(
    input_axes=axes, output_axes=axes
)
print(matrix[1, 1])   # um per slice step
print(matrix[3, 3])   # um per pixel in x
```

The slice spacing comes from `--z-spacing`. When nothing supplies one, Thyra
falls back to the in-plane pitch and records that it did:

```python
cs = sdata.attrs["coordinate_systems"]["global"]
cs["z_spacing_um"]      # the number used
cs["z_spacing_source"]  # "manual" | "automatic" | "assumed_isotropic"
```

A `z_spacing_source` of `"assumed_isotropic"` means **nobody supplied a spacing
and the in-plane pitch was reused** — treat the depth as unknown rather than as
measured. Section thickness is set by the microtome, not by the raster, so the
two agree only by coincidence. See
[`--z-spacing`](cli.md#set-z-spacing-whenever-you-know-it).

!!! note "These keys only appear on volumes"
    `z_spacing_um` and `z_spacing_source` are written only when the store
    actually holds a multi-slice volume. Their absence is how a 2D store says it
    has no z axis, which is why `convention_version` stays at `1` — the keys are
    additive and a consumer that never reads 3D sees the schema it already
    knows.

#### Pixel footprints in a volume

The pixel polygons are **two-dimensional**, on every route including a volume.
A slice's depth is not on the geometry; read it from the table's `spatial_z`,
or from the volume's own `Scale`:

```python
shapes = sdata.shapes[f"{dataset_id}_pixels"]
print(shapes.geometry.iloc[0].has_z)    # False, on 2D and 3D alike

# Depth per pixel, in micrometres, from the table:
obs = sdata.tables[dataset_id].obs
print(obs.loc["0", "spatial_z"])        # z_index * z_spacing_um
```

Every footprint therefore resolves to the correct place in x and y, and carries
no claim about z. Overlaying the shapes on the volume is exact in-plane and
needs `spatial_z` to pick the slice.

!!! note "Why the footprints are flat"
    They briefly were not. v3.2.0 made them `POLYGON Z` at the depth of their
    slice, which is geometrically the more honest representation, and it broke
    `spatialdata`'s spatial queries: a bounding box enclosing an entire test
    volume returned 26 of 30 footprints and 26 of 30 table rows, with no
    exception and no warning. A z-restricted query returned the same rows
    whether z was inside or far outside the data.

    `spatialdata` asks for 2D here: `ShapesModel.validate` warns that a
    3-dimensional geometry column "could led to unexpected behaviors" and names
    `force_2d()` as the remedy. 3D shapes are not on its roadmap
    ([#109](https://github.com/scverse/spatialdata/issues/109) has been idle
    since June 2023), and the live 2.5D discussion
    ([#961](https://github.com/scverse/spatialdata/issues/961)) covers points,
    images and labels only. Serial-section MSI is 2.5D in that sense.

    So this is a deliberate trade: a documented gap in z, in exchange for
    queries that return every pixel. Expressing the depth as a `Translation` on
    flat geometry does not close the gap either — the transform silently drops
    z. Both negative results are pinned by
    `tests/unit/converters/test_3d_pixel_shapes_z.py`, which will fail if
    upstream changes. See [Coordinate systems](coordinate-systems.md).

---

## Dataset Metadata

### Global metadata

Stored in `sdata.attrs`:

```python
if "pixel_size_x_um" in sdata.attrs:
    print(f"Pixel size: {sdata.attrs['pixel_size_x_um']} um")

if "msi_dataset_info" in sdata.attrs:
    info = sdata.attrs["msi_dataset_info"]
    print(f"Dimensions: {info.get('dimensions_xyz')}")
    print(f"Non-empty pixels: {info.get('non_empty_pixels'):,}")
```

### Table-level metadata

Instrument info, acquisition parameters, and resampling config are in
`msi_table.uns`:

```python
print("uns keys:", list(msi_table.uns.keys()))
```

### Provenance

`uns["essential_metadata"]` records what the store was made from, and how
the source was interpreted. It is the only place a converted dataset says
where it came from, so it is written the same way by every converter path:

```python
provenance = msi_table.uns["essential_metadata"]

print(provenance["source_path"])     # the file this was converted from
print(provenance["dimensions"])      # source grid, [x, y, z]
print(provenance["mass_range"])      # SOURCE m/z range, not the target axis
print(provenance["spectrum_type"])   # "centroid spectrum" / "profile spectrum"
print(provenance["thyra_version"])   # the Thyra that wrote the store
```

`mass_range` describes the source, not the resampled axis -- for the axis
the data actually sits on, read `msi_table.var["mz"]`.

Beside it, when the source format provides them:

| Key | Contents |
|-----|----------|
| `format_specific` | Vendor metadata (imzML file mode and UUID, FlexImaging areas, teaching points) |
| `acquisition_params` | Polarity, scan range, laser settings |
| `instrument_info` | Instrument model, serial, software version |
| `raw_metadata` | Source metadata as read, for round-trip fidelity |
| `regions` | Acquisition region summary, as a JSON string (see [Regions](#regions)) |

A section the source format has nothing for is omitted rather than written
empty, so `"instrument_info" not in uns` means "this format does not carry
it" rather than "it was carried and lost".

Within these sections, a list that holds anything besides numbers -- imzML
`cvParams` (a list of objects) is the main case -- is stored as a **JSON
string**; decode it with `json.loads`:

```python
import json

cv_params = json.loads(msi_table.uns["raw_metadata"]["cvParams"])
print(cv_params[0])   # {"name": ..., "accession": "MS:...", "value": ...}
```

Purely numeric lists stay plain arrays. The JSON encoding exists because
AnnData/zarr cannot round-trip such lists faithfully (a list of objects
comes back as `repr` strings), and because a stored list of strings reads
back as a numpy string array -- and on numpy 2.1-2.2, deepcopying such an
array segfaults the Python process outright
([numpy#28609](https://github.com/numpy/numpy/issues/28609), fixed in
numpy 2.3). Every table copy deepcopies `uns` (`AnnData.copy`,
`spatialdata.polygon_query`, joins), so a store carrying one would crash
those readers with no traceback.

!!! note "Stores written by Thyra <= 3.7.2 on numpy 2.1-2.2"
    Older stores still carry `cvParams` as a string array. On numpy
    2.1.x-2.2.x (Google Colab ships 2.1.3), convert those arrays to
    plain lists right after loading, before anything copies the table:

    ```python
    from thyra.metadata import sanitize_uns_string_arrays

    for table in sdata.tables.values():
        table.uns = sanitize_uns_string_arrays(table.uns)
    ```

    Environments on numpy 2.0 or >= 2.3 are unaffected either way.

### Structured metadata: `uns["msi_metadata"]`

The sections above preserve what the source said, in the source's own
vocabulary. `uns["msi_metadata"]` is the normalised, versioned view of the
same facts: fixed field names, PSI-MS/NCBITaxon/UBERON/CHEBI ontology
terms, and a schema a validator can hold it to.

```python
block = msi_table.uns["msi_metadata"]

print(block["schema_version"])                    # "0.1.0"
print(block["ms_analysis"]["pixel_size_um"])      # {"x": 20.0, "y": 20.0}
print(block["provenance"]["source_format"])       # "imzml"
```

It is written by every converter path identically, validated by
`thyra validate`, and exported to a METASPACE submission by
`thyra export-metaspace`. See [Metadata Schema](metadata-schema.md) for
the full contract.

---

## Recipes

### Side-by-side TIC and ion image

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(tic_array, cmap="viridis")
ax1.set_title("TIC")

target_mz = 760.5
mz_idx = np.abs(mz_values - target_mz).argmin()
ion_values = np.asarray(X[:, mz_idx].toarray()).flatten()

x_coords = msi_table.obs["x"].values.astype(int)
y_coords = msi_table.obs["y"].values.astype(int)
ion_image = np.zeros((y_coords.max() + 1, x_coords.max() + 1))
ion_image[y_coords, x_coords] = ion_values

ax2.imshow(ion_image, cmap="hot")
ax2.set_title(f"m/z {mz_values[mz_idx]:.2f}")

plt.tight_layout()
plt.show()
```

### Export ion image to TIFF

```python
from PIL import Image

# Normalise to 0-255
ion_norm = (ion_image / ion_image.max() * 255).astype(np.uint8)
Image.fromarray(ion_norm).save("ion_image.tiff")
```

### Top N most intense m/z values

```python
avg = msi_table.uns["average_spectrum"]
top_n = 10
top_indices = np.argsort(avg)[-top_n:][::-1]

for idx in top_indices:
    print(f"  m/z {mz_values[idx]:.4f}  avg intensity {avg[idx]:.1f}")
```

### Summary statistics

```python
print(f"Dataset: {table_key}")
print(f"  Pixels: {msi_table.n_obs:,}")
print(f"  m/z bins: {msi_table.n_vars:,}")
print(f"  m/z range: {mz_values.min():.2f} -- {mz_values.max():.2f}")
print(f"  Sparsity: {(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.1f}%")
if "pixel_size_x_um" in sdata.attrs:
    print(f"  Pixel size: {sdata.attrs['pixel_size_x_um']} um")
```
