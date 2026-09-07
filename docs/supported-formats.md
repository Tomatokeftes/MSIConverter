# Supported Formats

Thyra reads seven MSI formats and writes all of them into the same
SpatialData/Zarr layout. The input format is detected from the path -- there is
no format flag on the CLI, and `format_type` in the Python API selects the
*output* format, not the input.

| Format | Path shape | Detected by | Vendor SDK |
|---|---|---|---|
| **imzML** | `.imzML` file + `.ibd` | extension, `.ibd` must exist | none |
| **Bruker timsTOF** | `.d` directory | `analysis.tsf` or `analysis.tdf` | bundled DLL |
| **Bruker solariX** | `.d` directory | `peaks.sqlite` + `ImagingInfo.xml` | none |
| **Bruker Rapiflex** | directory | `*.dat` + `*_poslog.txt` | none |
| **Waters MassLynx** | `.raw` **directory** | `_FUNC*.DAT` files inside | bundled DLL |
| **PHI SmartSoft-TOF** | `.raw` **file** | `SOFH` magic in first 4 bytes | none |
| **mzPeak** | `.mzpeak` file | ZIP magic + `mzpeak_index.json` member | none |

```bash
thyra sample.imzML       out.zarr   # imzML
thyra sample.d           out.zarr   # Bruker timsTOF or solariX (see below)
thyra rapiflex_folder/   out.zarr   # Bruker Rapiflex
thyra waters_run.raw/    out.zarr   # Waters (a directory)
thyra tofsims_run.raw    out.zarr   # PHI (a file)
thyra sample.mzpeak      out.zarr   # mzPeak (experimental)
```

Two Bruker instrument families share the `.d` extension and are told apart by
what the directory contains: timsTOF writes `analysis.tsf`/`analysis.tdf`,
solariX (FT-ICR / MRMS) writes `peaks.sqlite` alongside `ImagingInfo.xml`.

---

## The `.raw` collision

Two vendors claim `.raw`, and they are told apart by **shape, not extension**:

- **Waters** `.raw` is a *directory* containing `_FUNC001.DAT`, `_FUNC002.DAT`, …
- **PHI** `.raw` is a *single file* whose header begins with the ASCII magic
  `SOFH`

Detection checks the directory case first, then the file magic. A `.raw` path
that is neither raises an error naming both possibilities, rather than a
confusing Waters-specific complaint:

```
Unrecognised .raw file: <path>. Expected either a Waters directory containing
_FUNC*.DAT files, or a PHI SmartSoft-TOF file beginning with the SOFH magic.
```

---

## What each format provides

Not every source carries every kind of metadata. This is what Thyra can
actually populate from each.

| | imzML | timsTOF | solariX | Rapiflex | Waters | PHI | mzPeak |
|---|---|---|---|---|---|---|---|
| Pixel size from metadata | yes | yes | yes (`.mis`) | yes | yes | yes | sometimes |
| Optical image | -- | yes | -- | yes | -- | -- | -- |
| Optical alignment | -- | yes (`.mis`) | -- | -- | -- | -- | -- |
| Multi-region | -- | yes | recorded | -- | -- | mosaic tiles | -- |
| 3D / multi-slice | yes | yes | -- | -- | -- | -- | -- |
| Native non-m/z axis kept | -- | -- | -- | -- | -- | flight time | -- |

Anything a format does not supply is simply absent from the output rather than
guessed at. Pixel size is the one exception worth knowing about: when a source
cannot report it, the CLI falls back to a default and records that it did so in
`uns` (see [Output Format](output-format.md)).

---

## imzML

The open interchange format, read through
[pyimzML](https://github.com/alexandrovteam/pyimzML). Both storage modes work:

- **continuous** -- every spectrum shares one m/z array, so the common mass
  axis is read once
- **processed** -- each spectrum carries its own m/z values, so building the
  common axis requires a pass over every spectrum

An `.imzML` without its `.ibd` beside it is rejected up front, because the XML
holds only offsets and the binary holds the data.

**Ion mobility.** imzML defines two binary arrays, but TIMSCONVERT and
TIMSImaging add a third for mobility, declared through a param group bound to
`MS:1003006` (mean inverse reduced ion mobility array, unit `MS:1002814`).
Thyra reads it. The MSI table is summed over mobility as always -- a shared
m/z block that repeats a value where mobility splits a feature collapses to
one column -- and, for a continuous export (one shared feature list), the
`(m/z, mobility)` pairs are also written as a mobility-resolved sibling
table; see [Output Format](output-format.md#ion-mobility). A processed export
(per-pixel point cloud) gets the summed table only -- `--mobility-grid`
covers per-pixel mobility for Bruker TDF, not yet for imzML. A mobility array declared
with zlib compression is refused like the other two.

See [imzML Parser Notes](imzml-parser-notes.md) for the hazards Thyra works
around in the underlying library.

## Bruker timsTOF

`.d` directories containing `analysis.tsf` (TOF only) or `analysis.tdf` (TIMS
engaged). Reading goes through Bruker's `timsdata` library, which is bundled
for Windows and Linux. This is the richest source Thyra handles: it carries
optical microscopy images, FlexImaging `.mis` teaching points for MSI-to-optical
registration, and per-pixel region annotations for multi-region slides.

A TDF frame is one pixel whose scans are the ion mobility dimension. Thyra
reads **every scan of the ramp** and collapses them into the one spectrum per
pixel the MSI table holds; `--mobility-grid` writes what was collapsed as a
separate table (below), and `--msms-table` slices the ramp by precursor
instead (below). Two collapses are available through `--tdf-spectrum`:

- `vendor_centroid` (default): Bruker's frame-level peak picker over the full
  ramp, the same one behind the TSF line spectrum and SCiLS Lab's import. It
  merges neighbouring digitizer bins and drops single-count noise, which on
  real imaging frames keeps roughly 80 to 90 percent of the raw ion current.
- `scan_sum`: every scan summed per digitizer index. Lossless, three to four
  times as many points per frame, and exactly the mobility marginal of the
  per-scan data.

The choice is recorded in the store's processing provenance
(`msi_metadata.processing[0].parameters.tdf_spectrum`), and the acquisition's
mobility range and ramp length are recorded in
`msi_metadata.ms_analysis.ion_mobility`, so a consumer can tell a summed-over-
mobility spectrum from one that never had a mobility dimension.

The mobility dimension itself is not thrown away. The summed table carries
`uns["mobility_axis"]` -- the 1/K0 of every scan from the vendor calibration,
the declared range and the `TimsCalibration` row -- and `uns["mobility_heatmap"]`,
the dataset's mean mass-mobility frame (about 4,000 m/z bins by 256 mobility
channels) accumulated from the raw scan read of every frame. The heatmap is
where to look to see whether mobility separates anything; it costs one extra
library call per frame, about a millisecond, and `--no-mobility-heatmap`
skips it. Under `scan_sum` the heatmap summed over mobility is exactly the
stored mean spectrum; under `vendor_centroid` the two differ by what the
centroid discards. See [Output Format](output-format.md#ion-mobility). A TSF
file has no mobility dimension and gets none of this.

**`--mobility-grid` writes the mobility-resolved table.** A TDF pixel is its
own point cloud, so there is no feature list shared across pixels to write
directly the way an imzML mobility export has; the flag bins every pixel's
`(m/z, 1/K0, intensity)` points onto one set of mobility channels shared
across the conversion and writes the result as `{table}_mobility` -- the same
element, columns and sort a shared-axis source produces. The default 256
channels over the axis' own value range are the heatmap's, so a box on the
heatmap indexes the table's channels. It costs a second pass over the source
(the first is shared with the heatmap), a table with 1.3 to 4 times the summed
table's non-zeros -- built out of core, so a whole acquisition converts
whatever its size -- and a switch to
`--tdf-spectrum scan_sum` (said at `WARNING`) so the table's marginal over
channels reproduces the summed table exactly. See
[Output Format](output-format.md#the-same-table-from-a-common-mobility-grid).

**MS/MS acquisitions convert, and say so.** `Frames.MsMsType` tells a survey
frame from a fragment one; the precursor detail comes from `PasefFrameMsMsInfo`
(PASEF frames) or `FrameMsMsInfo` (single-precursor frames). What is read lands
in `uns["msms_schedule"]` and in
`msi_metadata.ms_analysis.fragmentation`: the MS level, each isolation window's
target m/z and offsets, the collision energy, and -- for PASEF -- the mobility
scan range each window occupies.

The MSI table is still the whole frame summed, so on a scheduled PASEF method
that isolates several precursors at every pixel it holds fragments of all of
them at once, and conversion says so at `WARNING`. **`--msms-table` splits them
apart** into a `{table}_msms` sibling of `(precursor, fragment)` features: each
window owns a disjoint slice of the mobility ramp, so every recorded point
belongs to exactly one precursor and the split is a filter on the scan number,
not a deconvolution -- the per-precursor columns of a pixel add back up to what
the summed table holds there. This is the shape a targeted MALDI MS/MS imaging
run has, the one Bruker acquires as `iprm-PASEF`: a scheduled precursor list,
serially fragmented at every pixel. Thyra refuses, by name and without writing a
table, when the schedule varies per pixel, when the windows overlap or carry no
scan range, or when there is only one precursor to begin with. TDF is the only
source that reports what the split needs; an imzML export of the same
acquisition does not. See
[Output Format](output-format.md#fragmentation-msms).

## Bruker solariX

`.d` directories from solariX / MRMS (FT-ICR) instruments running ftmsControl,
detected by `peaks.sqlite` together with `ImagingInfo.xml`. Thyra reads the
processed peak store the acquisition software writes into every imaging `.d`:
centroided, calibrated per-pixel peak lists with stage-raster coordinates and
the instrument identity. **No vendor SDK** and no FT processing are involved --
the raw transient block (`ser`) and the `.mcf` containers are never touched.
See [solariX Notes](solarix-notes.md) for the verified layout, what the reader
refuses, and the pixel-size story.

Two things to know up front:

- The pixel size lives **only** in the flexImaging `.mis` file next to the
  `.d` (same stem). Without it, pass `--pixel-size` explicitly -- Thyra never
  guesses.
- A solariX-family `.d` that carries raw transients but no `peaks.sqlite`
  cannot be read natively; the error says so and names the imzML export
  fallback (DataAnalysis, SCiLS Lab, or flexImaging).

## Bruker Rapiflex

A folder of `*.dat` files with `*_poslog.txt` and `*_info.txt` alongside. The
position log supplies the pixel grid. No SDK required.

## Waters MassLynx

A `.raw` **directory** of `_FUNC*.DAT` files, read through the MassLynxRaw and
MLReader native libraries (bundled for Windows and Linux). The pixel grid is
reconstructed from the laser X/Y position recorded on each scan, and only MS
functions are converted -- lockmass, MRM and ion-mobility functions are
classified and skipped.

## PHI SmartSoft-TOF (ToF-SIMS)

A single `.raw` **file** from PHI (Physical Electronics) nanoTOF instruments.
Unlike Waters and Bruker this needs **no vendor SDK** -- the format is parsed
directly, so it behaves identically on every platform.

It differs from the other formats in one important way: it records **individual
ion arrivals**, not per-pixel spectra. Thyra aggregates those events into sparse
spectra on the detector's time-channel grid. Because flight time is what the
instrument actually measures, Thyra also stores it as `var["tof_us"]` so the
mass calibration stays reversible.

```bash
thyra tofsims_run.raw out.zarr
```

```python
from thyra.readers.phi import PhiReader

with PhiReader("tofsims_run.raw") as reader:
    print(reader.dimensions)          # (512, 512, 1)
    print(reader.pixel_size_um)       # 1.0
    print(reader.calibration_source)  # 'appended' or 'header'
```

Verified against the instrument software's own exports: the total ion image is
reconstructed bit-exactly, and the exported peak images to 99.7%. Mosaic,
MS/MS and depth-profiling acquisitions are implemented but have only been
tested against synthetic files -- if you have real data in one of those modes,
please [open an issue](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/issues).

See [PHI ToF-SIMS Notes](phi-tofsims-notes.md) for the file layout, the
calibration behaviour, and why the time axis is binned the way it is.

---

## mzPeak (experimental)

A single `.mzpeak` **file**: a ZIP of Parquet members plus an
`mzpeak_index.json` that maps each member to a role. mzPeak is the HUPO-PSI
working group's intended successor to mzML/imzML at the raw/archival layer.

Thyra treats it as an **input only** and never writes one. Support is marked
experimental because the container is a v0.9 draft -- column names, the index
vocabulary and the placement of file-level metadata have all moved between
prototype revisions and are expected to move again before v1.0. The reader
validates what it depends on, so a drifted archive raises a named error rather
than converting to something plausible but wrong.

```bash
thyra sample.mzpeak out.zarr
```

Data is shaped like processed imzML: one m/z per point, per-spectrum axes, and
no shared-axis concept anywhere in the format. The resampling decision tree
therefore treats these files exactly as it treats processed imzML.

Three things are refused rather than guessed at:

- The **chunked layout** (`chunk` instead of `point` in the signal member) is a
  different physical encoding and raises `NotImplementedError` naming the file.
- **Non-imaging archives** are rejected. Positions are optional in mzPeak --
  the reference converter only writes them when it happens to see imaging
  input -- and an archive without them has no pixels for Thyra to place.
- **Unrecognised layouts** fail with the schema they actually carry.

Two behaviours worth knowing:

- **Pixel size is often absent.** When `IMS:1000046`/`IMS:1000047` are missing
  the CLI falls back to `--pixel-size` exactly as it does for imzML. Terms are
  matched on accession rather than name, because the controlled vocabulary
  spells the two axes inconsistently.
- **Null-pair padding is dropped.** mzPeak compresses profile spectra by
  removing interior runs of zero intensity and marking each gap with two rows
  whose m/z *and* intensity are both null; the reference reader regenerates the
  missing m/z from a per-spectrum polynomial. Those regenerated values are
  extrapolations that carry zero intensity, so in a sparse matrix they would
  only add mass-axis channels that can never hold a value. Thyra omits them and
  logs how many it dropped. Recorded point counts include the padding, so peak
  totals are corrected against it.

mzPeak carries no region or ROI identity of any kind, so `get_region_map()`
returns `None`. Missing pixels are ordinary and are left missing rather than
densified.

---

## Adding another format

A reader subclasses `BaseMSIReader` and implements four methods --
`_create_metadata_extractor`, `get_common_mass_axis`, `iter_spectra` and
`close` -- then registers itself with `@register_reader("name")`. The converter
is format-agnostic, so 2D/3D handling, resampling, chunking and the whole
SpatialData output path come for free.

Optional overrides that are worth implementing when the format allows it:

| Method | Buys you |
|---|---|
| `has_shared_mass_axis` | skips a full pass when the axis is fixed |
| `get_peak_counts_per_pixel` | single-pass CSR build instead of two passes |
| `get_mass_axis_annotations` | keeps a native non-m/z axis in `var` |
| `get_region_map` / `get_region_info` | per-pixel region annotation |
| `get_optical_image_paths` | optical images carried into the output |

See [Contributing](contributing.md).
