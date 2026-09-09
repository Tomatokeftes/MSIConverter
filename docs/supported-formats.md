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

**Pixel numbering.** The specification numbers x and y from 1, and Thyra
subtracts that to reach the 0-based indices the store uses. Exports numbered
from **0** exist, and on those the subtraction used to produce `x = -1` for the
first column, which the grid guard then dropped -- a 3x3 acquisition stored as
4 pixels, with a warning naming a 2x2 grid the file never declared. Since
v3.24.0 the base is measured: a file whose smallest coordinate is 0 is rebased
on 0, and a file starting at 1 -- or at 5, because it is a crop of a larger
slide -- keeps the base of 1 and does not move. z is separate and rebases on
the smallest plane present, because z has no origin to preserve. Whatever was
subtracted is recorded in `coordinate_systems.global.coordinate_offsets_px`.

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

- `scan_sum` (default): every scan summed per digitizer index. Lossless,
  1.5 to 3 times as many points per frame as the centroid, exactly the
  mobility marginal of the per-scan data, and exactly Bruker's own
  per-frame total (`Frames.SummedIntensities`) and its quasi-profile
  export. This is the instrument's record, which is why it is the default;
  see [Design Decisions](design-decisions.md#d1-which-spectrum-a-reader-takes).
- `vendor_centroid`: Bruker's frame-level peak picker over the full ramp,
  the same one behind the TSF line spectrum and SCiLS Lab's import. It
  reports peak areas, merges neighbouring digitizer bins, and drops every
  index bin its picker assigns to no peak, which on measured imaging
  acquisitions keeps 87 to 96 percent of the raw ion current. Choose it to
  reproduce the vendor software's numbers.

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
where to look to see whether mobility separates anything; it is fed from the
same frame read that builds the summed table, so it costs the mapping of
every point onto the mass axis and no extra read, and
`--no-mobility-heatmap` skips it. Under `scan_sum` the heatmap summed over mobility is exactly the
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
heatmap indexes the table's channels. It is fed from the summed table's own
two passes -- one raw read per frame per pass serves
every table, so the grid adds no read of the source -- and it is a table
with 1.3 to 4 times the summed table's non-zeros, built out of core, so a
whole acquisition converts whatever its size. Under the default `--tdf-spectrum scan_sum` the table's
marginal over channels reproduces the summed table exactly; an explicit
`vendor_centroid` is accepted with a `WARNING` and the mismatch recorded. See
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

The raster step comes from the `Raster:` line in `*_info.txt` or the
`<Raster>` element of the `.mis`. Without either, Thyra refuses and asks for
`--pixel-size`, the same way the solariX reader does -- it used to fall back
to 20 um and record that guess in the store as an automatically detected
measurement.

The `.dat` header's raster origin reaches the store as
`coordinate_offsets_px`, with `stage_offset_um` beside it, so a Rapiflex
store can be placed against its optical image the way a solariX or timsTOF
one can. The stored pixel coordinates themselves stay 0-based.

## Waters MassLynx

A `.raw` **directory** of `_FUNC*.DAT` files, read through the MassLynxRaw and
MLReader native libraries (bundled for Windows and Linux). The pixel grid is
reconstructed from the laser X/Y position recorded on each scan; MRM and
ion-mobility functions are classified and skipped, and which of the rest hold
the image is decided from those same positions (below).

The grid is a *lattice* fitted to those positions -- an origin, a pitch and a
count per axis -- not a ranking of the distinct ones. The pitch is the
neighbour interval, so a raster missing an interior row keeps its true pitch
and leaves that row empty, rather than inflating the pitch and shifting every
row past the gap up by one. Positions that do not lie on the fitted raster
are refused, naming the axis and how far off the worst reading sits, instead
of becoming a grid with a pixel per stage wobble. A scan reporting a
non-finite position is unpositioned; two scans reporting the same position
are summed into one pixel and the log names it.

A `.raw` directory is refused when the library declares more functions than
it has `_FUNC*.DAT` files. MassLynx keeps reporting a function whose data
file is gone, with 0 scans and no error, so a chunk lost from a split raster
would otherwise drop its rows of the image in silence.

### Which representation is read

MassLynx can hand back either of two things for a profile-acquired pixel: the
**vendor centroid** list, computed on demand by its peak picker, or the
**profile trace** behind it -- the digitiser's samples, zero-suppressed to the
clusters around each peak. Thyra reads `_extern.inf` and `_header.txt` to
decide which one is the default, and logs the field it decided on:

| Instrument | Identified by | Default | Method | Axis | Default width |
|---|---|---|---|---|---|
| SELECT SERIES MRT | `OpticMode = MRT`, or `$$ Instrument: MRT#`, or `Resolution` above 100,000 | **profile trace** | `tic_preserving` | `linear_tof` | 1.3 mDa at m/z 1000 |
| SELECT SERIES MRT, `--waters-spectrum centroid` | as above | vendor centroid | `nearest_neighbor` | `tof` (measured width law, see [Resampling](resampling.md#the-two-term-tof-law)) | 3 bins per peak width |
| every other Waters instrument | none of the above | vendor centroid | `nearest_neighbor` | `reflector_tof` | 2 mDa at m/z 1000 |

`--waters-spectrum centroid|profile` overrides the default in either
direction. A run that is not an MRT but is asked for its profile gets the
profile treatment with a bin width taken from its own digitiser: 1.14 times
the sample spacing predicted from `Lteff`, `Veff` and the ADC clock in
`_extern.inf` (15.8 mDa on a Synapt G2-Si), so the store is not oversampled.

**Why the MRT defaults to the profile.** Its multi-reflecting flight path
gives a measured resolving power of 130,000 at m/z 300 rising to 190,000 at
m/z 1000 (median 168,000), and at that resolution the vendor peak picker, not
the analyser, is what limits the data. On a 13,398-pixel MALDI brain section,
four separate 8-9 mDa doublets between m/z 760 and 830 -- including the
<sup>13</sup>C<sub>2</sub> isotopologue of PC 34:1 [M+K]<sup>+</sup> at
800.5477 against PC 34:0 [M+K]<sup>+</sup> at 800.5566 -- were resolved as two
maxima in the profile in 19,227 pixels between them. The vendor returned
**exactly one** centroid in 96-100% of those pixels, never two, landing 4-8 ppm
from either true mass. No centroid-side setting recovers that: a finer axis
just places the merged centroid more precisely.

**Why the others do not.** The same test on a Synapt G2-Si MALDI imaging run
(7,007 pixels, resolving power about 26,000) found the vendor centroider
merging **none** of the pairs its profile resolves: it reported three or more
peaks in 99.4-99.7% of the pixels where the profile showed two. At that
resolving power the analyser is the limit and the peak picker keeps everything
the trace has, so the profile would cost 2-3x the store for nothing. Both
instruments sample the trace on the same `sqrt(m/z)` grid (measured
`(m/z)^0.494` on the MRT, `(m/z)^0.498` on the Synapt), which is why the
profile route uses `linear_tof` whichever instrument it is asked for on.

**What the profile default costs.** On the reference run (13,398 pixels) the
store is 289 MB against 52 MB for the centroid default, and it converts in
15 s against 96 s, because MassLynx centroids on demand and a profile read
skips that work. Two things make the profile store that size: it holds 4.5x
the non-zeros (31.3M against 6.9M), and interpolated values are not the
integer ADC counts the raw samples are, so they compress about half as well
-- nearest-neighbour binning of the same trace onto the same axis gives
138 MB. The profile is stored on a fixed, generated
axis at about 1.14 times the digitiser's own sample spacing (1.3 mDa at m/z
1000 against 1.14 mDa per sample), so every MRT run with the same acquisition
mass range lands on the same bins -- the axis is built over the acquisition
setting, not the span of stored values, exactly as the timsTOF route does.
The accepted loss is one bin of smoothing: intensity is interpolated between
samples, so a peak apex in the stored mean spectrum sits within about 2 ppm of
the raw apex before any peak picking, and a centroider run on the stored
profile recovers it. The raw digitiser grid itself is still available with
`--no-resample`, where binning is the identity and the apex error is zero,
at the cost of an axis that differs from run to run.

`uns["essential_metadata"]["spectrum_type"]` records what was stored
(`profile spectrum` or `centroid spectrum`), `format_specific.spectrum_source`
says which MassLynx representation it came from, and `format_specific.is_mrt`
with `instrument_decided_by` record the instrument decision.
### Which functions hold the image

A Waters *function* is not necessarily one acquisition function. MassLynx
caps a `_FUNC*.DAT` file at about 1.6 GB and opens a **new function** when a
long imaging run reaches it, so one raster commonly arrives as several
functions that tile the stage. It reports MS level 1 for the first of them,
2 for the middle ones and 0 for the last, and `getLockmassFunction` names
that last one as the file's lockmass function -- none of which is true. Nine
real imaging runs from two instruments and five users were checked and every
multi-function one was a chunked single-function raster; see the
[D7 entry](design-decisions.md#d7-waters-which-functions-hold-the-image).

Thyra therefore decides from the laser positions, the same measurement the
pixel grid is built from:

- A function landing on pixels no earlier function covers **extends the
  raster** and is converted, whatever level MassLynx reports and whether or
  not MassLynx calls it the lockmass function -- provided its positions lie
  on the raster. The tail of a split raster continues it, so it shares the
  fitted lattice; a reference or calibration spot parked off the sample also
  covers pixels nothing else covers but does not, and is excluded and named
  rather than converted as an image pixel.
- The lattice itself is fitted to the MS functions, then re-fitted to
  whatever is finally converted, so a function that contributes no pixel
  never leaves its pitch behind in the grid.
- Functions **competing for the same pixels** were acquired in parallel:
  MSe low and high energy, a data-dependent run, a co-acquired lockmass
  reference. Only one of them can be the pixel's spectrum, and summing an
  intact-ion and a fragment spectrum into one pixel would make a spectrum of
  nothing, so the MS1 ones win where the file has any. The rest are listed,
  with their MS level and precursor m/z, under `excluded_functions` in the
  Waters-specific metadata block.

`format_specific.function_types` keeps MassLynx's own classification next to
`format_specific.ms_functions`, so a rescued chunk is visible in the store.

**One catch.** The library will not centroid the function it names the
lockmass function -- the request is honoured for every other function and
ignored for that one, which returns its profile trace either way. Putting it
into a store of centroids would lay a band of profile rows across the top of
the image at about 3x the neighbouring TIC, so while a run is read as
centroids that chunk stays out, and `excluded_functions` records its scan
count, the pixels it would have added and the reason. `--waters-spectrum
profile` converts every chunk, because then they all come back the same way.
The log names the cost and the flags:

```
Function(s) 2 hold 1274 pixels (16.6% of the image) that no other function
covers, but MassLynx names them the lockmass function and will not centroid
them. They stay out rather than put profile rows in a table of centroids:
pass --waters-spectrum profile to convert the whole image.
```

Budget the disk for it: the profile store for that run is estimated at 74 GB
against 241 MB for the centroid one. Memory is not the constraint, since every
conversion streams.

A file whose converted functions carry a precursor m/z holds fragment
spectra and reports them through `ms_analysis.fragmentation`, exactly as a
Bruker MS/MS acquisition does. A reported MS level with no precursor behind
it does not: that is the chunk artefact above.

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

**Previewing costs nothing.** Recording events rather than spectra means the
header cannot say which pixels carry one; counting them means decoding the
whole stream, which `preview_msi` used to do despite promising otherwise --
linear in file size, so a multi-gigabyte acquisition previewed as slowly as it
converted. Since v3.24.0 a preview answers from the header and the block chain
alone, and reports `n_pixels` as `None`: unknown, rather than quietly filled in
with the raster size that `grid_dims` already carries. A conversion is
unaffected and still counts every pixel exactly.

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
