# CLI Reference

```
thyra [OPTIONS] INPUT OUTPUT
```

**INPUT** -- Path to input MSI file or directory (`.imzML`, `.d`, `.raw`)

**OUTPUT** -- Path for output `.zarr` directory

`thyra` also has two metadata subcommands, `thyra validate` and
`thyra export-metaspace` -- see [Metadata subcommands](#metadata-subcommands).

!!! tip "Grouped help"
    `thyra --help` lists every option under a category heading -- Conversion,
    Logging, Resampling (advanced), Performance, imzML-specific,
    Bruker-specific, Waters-specific, Other, and a General section holding
    `--version` and `--help` -- in the same order as the sections on this page.

---

## Exit status

| Code | Meaning |
|------|---------|
| `0` | Conversion completed and the output store was written |
| `1` | Conversion failed |
| `2` | Invalid command-line arguments (click usage error) |

A failed conversion renames any partially written store to
`<output>.zarr.failed`, so the output path stays free for a retry and an
incomplete store is never left where a finished one is expected. This makes
`thyra` safe to chain in a shell script or CI job:

```bash
thyra input.imzML output.zarr && python analyse.py output.zarr
```

---

## Conversion

| Option | Default | Description |
|--------|---------|-------------|
| `--format [spatialdata]` | `spatialdata` | Output format |
| `--pixel-size FLOAT` | auto-detect | Pixel size in micrometers |
| `--region TEXT` | all | Convert one region, by `.mis` Area Name or by DB RegionNumber |
| `--resample / --no-resample` | enabled | Mass axis resampling |
| `--include-optical / --no-optical` | enabled | Include optical images in output |
| `--mobility-table / --no-mobility-table` | enabled | Also write the mobility-resolved sibling table when the source shares one set of (m/z, ion mobility) features across pixels (see [Output Format](output-format.md#ion-mobility)) |
| `--mobility-heatmap / --no-mobility-heatmap` | enabled | When the source has an ion mobility dimension, store the mean mass-mobility frame on the summed table as `uns["mobility_heatmap"]`; one extra pass over the source (see [Output Format](output-format.md#ion-mobility)) |
| `--mobility-grid / --no-mobility-grid` | **disabled** | When the source carries ion mobility per pixel rather than as a shared feature list (Bruker TDF), bin the point cloud onto a common mobility grid and write the same mobility-resolved sibling table; one extra pass over the source beyond the heatmap's, a much larger table built out of core so any acquisition fits, and it forces `--tdf-spectrum scan_sum` (see [Output Format](output-format.md#the-same-table-from-a-common-mobility-grid)) |
| `--msms-table / --no-msms-table` | **disabled** | When the source isolates several precursors per pixel in disjoint mobility slices (Bruker PASEF), also write them split apart as a demultiplexed sibling table; two extra passes over the source, and it forces `--tdf-spectrum scan_sum` so the split adds back up to the summed table exactly (see [Output Format](output-format.md#demultiplexed-msms-table)) |

### Examples

```bash
# Basic conversion -- format, pixel size, and resampling all auto-detected
thyra input.imzML output.zarr

# Specify pixel size manually (when metadata is unavailable)
thyra input.imzML output.zarr --pixel-size 25

# Convert one region, named the way flexImaging names it
thyra data.d output.zarr --region 03

# Or by the database's own region number
thyra data.d output.zarr --region 0

# Skip optical images
thyra data.d output.zarr --no-optical
```

!!! note "Naming a region"
    `--region` takes a string, and the value is matched against the `.mis`
    Area Names first. Only if nothing matches is it parsed as a DB
    RegionNumber, which starts at 0. Passing something that is neither is an
    error that lists the area names it did find.

    The two numbering schemes do not have to agree, which is exactly why area
    `'03'` need not be RegionNumber 3. For a multi-region dataset Thyra logs
    the whole mapping at `INFO` during startup -- a header line
    `Region mapping (DB RegionNumber -> .mis Area Name):` followed by one
    `RegionNumber <n> -> Area '<name>' (<n> frames)` line per region -- so a
    plain run already tells you which is which before you pick one.

    Use `-v DEBUG` for the per-region spectrum counts as well.

---

## Logging

| Option | Default | Description |
|--------|---------|-------------|
| `-v, --log-level LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--log-file PATH` | none | Write logs to file |

### Examples

```bash
# Verbose output -- shows pixel size detection, resampling config, timing
thyra input.imzML output.zarr -v DEBUG

# Save logs to file for later review
thyra input.imzML output.zarr --log-file conversion.log
```

!!! tip "Debugging conversions"
    When something looks wrong in the output, re-run with `-v DEBUG --log-file
    debug.log`. The log will contain pixel size detection details, resampling
    parameters, region info, and timing for each step.

---

## Ion Mobility Grid (Advanced)

These size the grid `--mobility-grid` bins onto. They do nothing without it.

| Option | Default | Description |
|--------|---------|-------------|
| `--mobility-bins INTEGER` | `256` | Mobility channels the grid divides the range into |
| `--mobility-min FLOAT` | axis minimum | Lower edge of the grid, in the axis unit (1/K0 for TIMS) |
| `--mobility-max FLOAT` | axis maximum | Upper edge of the grid |

### Examples

```bash
# The mobility-resolved table for a TIMS acquisition, at the default
# 256 channels over the range the acquisition actually used
thyra tims_data.d output.zarr --mobility-grid

# A narrower range at finer resolution, giving up the heatmap alignment
thyra tims_data.d output.zarr --mobility-grid --mobility-bins 512     --mobility-min 1.05 --mobility-max 1.25
```

!!! note "The grid table is built out of core"
    Like the summed table on the streaming route, the grid table is built in
    two passes: the first counts the occupied `(m/z bin, channel)` cells and
    the second scatters every pixel straight into memmapped CSC arrays in a
    scratch directory next to the output (`.thyra_mobility_*`, removed once
    the table is written). Memory is the count array over the grid's span --
    142 MB on a default-resampled timsTOF axis -- plus one frame, whatever the
    number of pixels; disk is 12 bytes per stored non-zero while the table is
    being built. Measured on a whole 26,087-pixel timsTOF acquisition on a
    40,000-bin axis (7.3M features, 1.59 billion non-zeros, 18 GB of scratch):
    the table's own phases peaked at 3.2 GB of private memory above the
    interpreter's baseline, in the `var` frame rather than in the matrix.

!!! note "A mobility-resolved table wants a coarser mass axis than the default"
    The table's features are `(m/z bin, mobility channel)` pairs, so the mass
    axis' width is multiplied by 256. A default resampled timsTOF axis is
    around 140,000 bins; 200 frames of one measured acquisition filled 3.9
    million of the resulting pairs, and a whole 26,000-pixel acquisition
    would pass the 20,000,000 ceiling the table is refused at. Reach for
    `--resample-bins` when the whole image is wanted resolved.

!!! warning "The defaults are an alignment, and the grid changes the TIC"
    256 channels over the mobility axis' own value range is exactly what
    `uns["mobility_heatmap"]` bins over, so a box drawn on the heatmap
    selects grid channels by integer index. `--mobility-bins`,
    `--mobility-min` and `--mobility-max` each give that up.

    Asking for a grid also switches a TDF conversion to `--tdf-spectrum
    scan_sum` unless that option was given explicitly, and says so at
    `WARNING`: the grid is built from raw scans, and its marginal over
    channels reproduces the summed table only when that table was built the
    same way. The switch moves the stored TIC by 13 to 21 percent against a
    default conversion of the same file.

---

## Resampling (Advanced)

These options control how spectra are mapped onto a common mass axis. In most
cases the defaults work well -- Thyra auto-detects the instrument type and
chooses an appropriate method and bin count.

| Option | Default | Description |
|--------|---------|-------------|
| `--resample-method METHOD` | `auto` | `auto`, `nearest_neighbor`, or `tic_preserving` |
| `--mass-axis-type TYPE` | `auto` | `auto`, `constant`, `linear_tof`, `reflector_tof`, `orbitrap`, `fticr` |
| `--resample-bins INTEGER` | auto | Number of bins (mutually exclusive with `--resample-width-at-mz`) |
| `--resample-min-mz FLOAT` | auto | Minimum m/z value |
| `--resample-max-mz FLOAT` | auto | Maximum m/z value |
| `--resample-width-at-mz FLOAT` | auto | Mass width in Da at reference m/z for physics-based binning |
| `--resample-reference-mz FLOAT` | `1000.0` | Reference m/z for width specification |
| `--resample-gap-tolerance FLOAT` | none | `tic_preserving` only: discard target bins farther than this many Da from any measured m/z, instead of interpolating across the gap |

!!! info "Choosing a resampling method"
    - **`nearest_neighbor`** -- Each target bin takes the nearest original m/z
      value. Correct for **centroid** data, where peaks are discrete masses.
    - **`tic_preserving`** -- Linear interpolation, rescaled so the total ion
      current is unchanged. Correct for **profile** data on a target axis
      whose bin widths scale the same way the source points are spaced --
      pair it only with `constant` unless you know otherwise.
    - **`auto`** -- Picks `tic_preserving` only for Bruker flexImaging /
      Rapiflex data, whose source grid is uniform in m/z, and
      `nearest_neighbor` for everything else -- including profile data from
      an instrument Thyra cannot identify, because the interpolating method
      is only exact when the two axis laws match. See
      [Resampling](resampling.md#which-detector-wins) for the full decision
      table.

!!! info "Choosing a mass axis type"
    The axis type determines how bin widths scale with m/z:

    - **`constant`** -- Uniform bin width (Da). Suitable for MALDI-TOF in linear mode.
    - **`linear_tof`** -- Width scales as sqrt(m/z). Matches TOF resolution.
    - **`reflector_tof`** -- Width scales linearly with m/z (constant relative resolution). Matches reflector TOF.
    - **`orbitrap`** -- Width scales as m/z^(3/2). Matches Orbitrap resolution.
    - **`fticr`** -- Width scales as m/z^2. Matches FTICR resolution.
    - **`auto`** -- Detected from instrument metadata.

### Examples

```bash
# Physics-based resampling for Orbitrap data
thyra input.imzML output.zarr \
    --resample-method tic_preserving \
    --mass-axis-type orbitrap

# Fixed number of bins
thyra input.imzML output.zarr --resample-bins 50000

# Restrict mass range
thyra input.imzML output.zarr --resample-min-mz 100 --resample-max-mz 1000

# Specify bin width at a reference m/z (physics-based)
thyra input.imzML output.zarr \
    --resample-width-at-mz 0.01 \
    --resample-reference-mz 500
```

---

## Performance

| Option | Default | Description |
|--------|---------|-------------|
| `--streaming [auto\|true\|false]` | `auto` | Streaming mode for large datasets |
| `--sparse-format [csc\|csr]` | `csc` | Sparse matrix storage format |

!!! info "Streaming mode"
    - **`auto`** (default) -- Thyra estimates dataset size and enables streaming
      for datasets over ~10 GB.
    - **`true`** -- Force streaming. Useful if auto-detection underestimates.
    - **`false`** -- Force standard (in-memory) conversion.

    Streaming processes spectra in chunks and writes incrementally to disk. The
    output is identical to standard mode.

### Examples

```bash
# Force streaming for a large dataset
thyra large.d output.zarr --streaming true

# Use CSR format (faster row access, slower column access)
thyra input.imzML output.zarr --sparse-format csr
```

!!! tip "CSC vs CSR"
    **CSC** (default) is optimised for extracting ion images (one m/z across all
    pixels). **CSR** is optimised for extracting spectra (one pixel across all
    m/z values). Choose based on your downstream access pattern.

!!! warning "`--optimize-chunks` is deprecated"
    The flag is still accepted, so existing scripts keep running, but it does
    nothing and now logs a warning. It will be dropped in a future release.

    It never did anything: the post-hoc pass it invoked was written for a dense
    4-D image layout and could not read the sparse table the converter actually
    writes, so it failed on every conversion and the CLI still exited 0. Chunk
    sizes are chosen at write time instead.

---

## imzML-Specific

| Option | Default | Description |
|--------|---------|-------------|
| `--spectrum-type TYPE` | `auto` | `auto`, `profile`, or `centroid` -- declare the spectrum representation instead of detecting it |

By default Thyra reads the representation the file declares (`MS:1000127`
centroid / `MS:1000128` profile), wherever in the document it is written, and
only guesses when the file declares neither. `--spectrum-type` overrides all of
that.

SCiLS Lab exposes the same control as `--rep_type PROFILE|CENTROID` (2026b User
Guide, p.81).

!!! warning "This changes stored values"
    The representation is not just a label: it feeds instrument detection,
    which picks the mass-axis type, which decides the bin spacing. Overriding
    it changes the axis a dataset is resampled onto, and therefore the numbers
    written to the store. It changes nothing for anyone who does not pass it.

    Use it when a file declares the wrong thing -- that does happen. When the
    override contradicts an explicit declaration Thyra logs a warning naming
    both values; run with `-v INFO` to see it, and check the result.

### Examples

```bash
# The file says centroid but it is really profile data
thyra input.imzML output.zarr --spectrum-type profile

# See what detection would have concluded before overriding it
thyra input.imzML output.zarr -v INFO
```

---

## Bruker-Specific

These options only apply when converting Bruker `.d` directories.

| Option | Default | Description |
|--------|---------|-------------|
| `--use-recalibrated / --no-recalibrated` | enabled | Use recalibrated m/z state |
| `--interactive-calibration` | off | Display available calibration states |
| `--intensity-threshold FLOAT` | none | Minimum intensity filter |
| `--tdf-spectrum {vendor_centroid,scan_sum}` | `vendor_centroid`; `scan_sum` when `--mobility-grid` or `--msms-table` is given | How a TDF (TIMS) frame's mobility scans collapse into one spectrum per pixel |

### Examples

```bash
# Use raw (non-recalibrated) m/z values
thyra data.d output.zarr --no-recalibrated

# Interactively choose calibration state
thyra data.d output.zarr --interactive-calibration

# Filter low-intensity signals (useful for continuous-mode Bruker data)
thyra data.d output.zarr --intensity-threshold 100

# TIMS data: keep every ion count instead of Bruker's centroided spectrum
thyra tims_data.d output.zarr --tdf-spectrum scan_sum
```

!!! note "TDF: the mobility ramp is summed, not sliced"
    Every scan of a TIMS frame is read. `vendor_centroid` is Bruker's own
    frame-level peak picker over the full ramp (parity with TSF line spectra
    and SCiLS Lab); `scan_sum` sums every scan per digitizer index and keeps
    all of the ion current, at three to four times the points. See
    [Supported Formats](supported-formats.md#bruker-timstof).

!!! warning "Intensity threshold"
    The `--intensity-threshold` option drops all peaks below the given value
    **before** writing to zarr. This reduces file size but is irreversible.
    Use with care -- inspect the data with `-v DEBUG` first to choose an
    appropriate threshold.

---

## Waters-Specific

This option only applies when converting Waters `.raw` directories.

| Option | Default | Description |
|--------|---------|-------------|
| `--waters-spectrum {centroid,profile}` | `centroid` | What MassLynx hands back for one pixel: the vendor peak picker, or the sampled trace behind it |

### Examples

```bash
# Keep near-isobars the vendor centroider merges
thyra data.raw output.zarr --waters-spectrum profile \
    --resample-width-at-mz 0.001 --resample-reference-mz 1000
```

!!! note "When the profile is worth its size"
    MassLynx centroiding reports a single peak wherever the sampled trace has
    two maxima a few mDa apart, and puts it partway between them. On a
    SELECT SERIES MRT brain section, four separate 8-9 mDa doublets between
    m/z 760 and 830 -- including the <sup>13</sup>C<sub>2</sub> isotopologue of
    PC 34:1 [M+K]<sup>+</sup> against PC 34:0 [M+K]<sup>+</sup> -- came back as
    one centroid in 96-100% of the pixels that resolved them, 4-8 ppm from
    either true mass. `--waters-spectrum profile` keeps them apart, for about
    3.5 times the stored non-zeros and 2.3 times the store. It is not slower:
    MassLynx centroids on demand, so a profile read skips that work.

---

## Other

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-id TEXT` | `msi_dataset` | Dataset identifier used in element keys |
| `--handle-3d` | off | Process as 3D volume instead of 2D slices |
| `--z-spacing FLOAT` | in-plane pixel size | Distance between consecutive slices, in um. Only used with `--handle-3d` |

### Examples

```bash
# Custom dataset ID (affects table and image key names)
thyra input.imzML output.zarr --dataset-id hippocampus
# -> table key: hippocampus_z0, TIC key: hippocampus_z0_tic

# Combine z-slices into a single 3D table
thyra volume.imzML output.zarr --handle-3d

# 20 um sections acquired on a 50 um raster
thyra volume.imzML output.zarr --handle-3d --pixel-size 50 --z-spacing 20
```

### Set `--z-spacing` whenever you know it

`--pixel-size` is the **in-plane** raster pitch. It says nothing about how far
apart consecutive slices are: that distance is set by the microtome that cut the
sections, not by the stage that rastered them, and the two match only by
coincidence.

With no `--z-spacing`, Thyra reuses the in-plane pitch, warns, and records
`z_spacing_source: "assumed_isotropic"` in the store so the guess stays
distinguishable from a measurement. The volume's voxel values are unaffected
either way — what is wrong is its depth, so a viewer reading it in micrometres
renders the stack squashed or stretched along z.

There is no way to detect this from the data. imzML has no term for slice
spacing, and 3D MSI acquisitions are frequently non-consecutive sections, so
the number has to come from whoever cut them.

!!! warning "It does nothing without `--handle-3d`"
    Without 3D handling each slice is written as its own 2D image and there is
    no z axis to space out. Passing `--z-spacing` on its own is logged as
    ignored rather than silently accepted.

---

## Metadata subcommands

Both take either a converted `.zarr` store or a standalone metadata
`.json` document. Only the metadata block is read, never the intensity
data, so both are instant on stores of any size. See
[Metadata Schema](metadata-schema.md) for the schema itself.

### `thyra validate`

```
thyra validate PATH [OPTIONS]
```

Validates the `uns["msi_metadata"]` block (or a JSON document) against
the schema: structure, schema version, and ontology terms.

| Option | Default | Description |
|--------|---------|-------------|
| `--merge PATH` | none | JSON file overlaid onto the metadata before validation |
| `--json` | off | Machine-readable report on stdout instead of text |

Exit status follows the conversion command's convention: `0` when every
document conforms (warnings allowed), `1` otherwise, `2` for usage
errors -- so `thyra validate` can gate a CI pipeline.

```bash
thyra validate output.zarr
thyra validate output.zarr --merge sample.json
thyra validate metadata.json --json
```

### `thyra export-metaspace`

```
thyra export-metaspace PATH [OPTIONS]
```

Writes the [METASPACE](https://metaspace2020.org) submission metadata
JSON for a dataset. Required fields the store cannot know (organism,
condition, ...) are emitted empty and reported as warnings on stderr;
supply them with `--merge` or fill them on the submission form.

| Option | Default | Description |
|--------|---------|-------------|
| `--merge PATH` | none | JSON file overlaid onto the metadata before export |
| `--table NAME` | the only one | Table to export when the store has several |
| `-o, --output PATH` | `<input>.metaspace.json` | Output file; `-` for stdout |

```bash
thyra export-metaspace output.zarr --merge sample.json
thyra export-metaspace output.zarr -o -          # print to stdout
```

---

## The other two commands

Installing Thyra puts three commands on your path, not one.

### `thyra-example-data`

Writes the small synthetic imzML dataset the [tutorial](tutorial.md) uses. No
download, no vendor SDK.

```
thyra-example-data [OUTPUT] [OPTIONS]
```

**OUTPUT** -- Path for the `.imzML` (default:
`example_data/synthetic_brain.imzML`). A non-`.imzML` suffix is replaced.

| Option | Default | Description |
|--------|---------|-------------|
| `--pixels-x INTEGER` | `48` | Grid width in pixels |
| `--pixels-y INTEGER` | `36` | Grid height in pixels |
| `--pixel-size FLOAT` | `25.0` | Pixel size in micrometers written to the metadata |
| `--mz-bins INTEGER` | `4000` | Number of points on the m/z axis |
| `--seed INTEGER` | `0` | Random seed |
| `--verbose` | off | Verbose output |

```bash
# The tutorial's dataset: 48 x 36 pixels, 4,000 m/z points, seed 0
thyra-example-data example_data/synthetic_brain.imzML

# A larger phantom on a coarser axis
thyra-example-data big.imzML --pixels-x 200 --pixels-y 150 --mz-bins 2000
```

!!! note "`--seed` is what makes the tutorial reproducible"
    Generation is deterministic given the seed, so the default `--seed 0`
    produces byte-identical data on every machine -- which is why the
    tutorial can print exact numbers and expect yours to match. Change the
    seed and the numbers change with it.

### `thyra-check-ontology`

Validates the CV (controlled vocabulary) terms in an imzML file against
Thyra's ontology cache, and reports any it does not recognise.

```
thyra-check-ontology INPUT [OPTIONS]
```

**INPUT** -- An `.imzML` file, or a directory to check every file in.

| Option | Default | Description |
|--------|---------|-------------|
| `--output PATH` | none | Save results as JSON instead of printing them |
| `--verbose` | off | Verbose output, including the per-file summary |

```bash
# Check one file
thyra-check-ontology sample.imzML

# Check a whole directory and keep the report
thyra-check-ontology data/ --output ontology_report.json
```
