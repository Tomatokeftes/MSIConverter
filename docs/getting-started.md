# Getting Started

## Installation

=== "pip (recommended)"

    ```bash
    pip install thyra
    ```

=== "From source"

    ```bash
    git clone https://github.com/M4i-Imaging-Mass-Spectrometry/thyra.git
    cd thyra
    uv sync
    ```

!!! note "Requirements"
    Python 3.12 or 3.13. Windows, macOS, and Linux are all supported.
    Bruker readers require the vendor SDK DLLs (bundled for Windows).

---

## Your First Conversion

### Command Line

```bash
# ImzML file
thyra input.imzML output.zarr

# Bruker .d folder
thyra data.d output.zarr

# Waters .raw folder
thyra data.raw output.zarr

# PHI SmartSoft-TOF ToF-SIMS .raw file
thyra tofsims_run.raw output.zarr
```

That's it -- Thyra auto-detects the format, reads the pixel size from metadata,
resamples onto a common mass axis, and writes a SpatialData/Zarr directory.

See [Supported Formats](supported-formats.md) for the full list, how each one
is detected, and what metadata each can supply.

### Python API

```python
from thyra import convert_msi

success = convert_msi("data/sample.imzML", "output/sample.zarr")
```

You can pass explicit parameters when needed:

```python
success = convert_msi(
    "data/experiment.d",
    "output/experiment.zarr",
    dataset_id="hippocampus",
    pixel_size_um=10.0,
)
```

!!! tip "Checking the result"
    After conversion, load the output and inspect:
    ```python
    import spatialdata as sd

    sdata = sd.read_zarr("output/sample.zarr")
    print(list(sdata.tables.keys()))
    print(list(sdata.images.keys()))
    ```
    See [Output Format](output-format.md) for the full structure.

---

## Multi-Dataset Folders

When you point Thyra at a folder containing multiple `.d` datasets (e.g., a slide
with several tissue sections), it prompts you to choose:

```
$ thyra "slide_folder/" output.zarr

Found 3 datasets in slide_folder:
  [1] C1501.d
  [2] E2501.d
  [3] E2506.d

Select dataset to convert: 3
  -> E2506.d
```

Each `.d` dataset is matched to its own `.mis` file for correct optical alignment,
so the TIC overlay lands on the right tissue section in the shared optical image.

!!! note "Converting a specific region"
    If a single dataset contains multiple acquisition regions (e.g., tissue +
    matrix), you can select one with `--region`:
    ```bash
    thyra data.d output.zarr --region 0
    ```

---

## Resampling

Mass axis resampling is **enabled by default**. This maps all spectra onto a
common m/z axis, which is required for most downstream analysis tools.

```bash
# Default -- resampling is on, method and binning auto-detected
thyra input.imzML output.zarr

# Disable resampling (keeps original m/z values per spectrum)
thyra input.imzML output.zarr --no-resample
```

For advanced control you can specify the method and instrument type:

```bash
# Physics-based resampling for Orbitrap data
thyra input.imzML output.zarr \
    --resample-method nearest_neighbor \
    --mass-axis-type orbitrap
```

`nearest_neighbor` is the right pairing for a non-uniform axis:
`tic_preserving` applies one scaling factor to the whole spectrum, which
cannot account for bin widths that vary across the mass range. See
[Resampling](resampling.md#methods) for the measured cost of getting that
combination wrong.

!!! info "When to disable resampling"
    Disable resampling (`--no-resample`) if you need the raw, unmodified spectra
    -- for example, when doing your own peak picking or centroiding downstream.
    Note that without resampling, each pixel may have a different m/z axis.

See [Resampling](resampling.md) for how the method, axis type, and bin count are
chosen, and the [CLI Reference](cli.md#resampling-advanced) for all options.

---

## Optical Images

For Bruker data, Thyra automatically includes optical (microscopy) images and
aligns them to the MSI data using teaching point calibration from the `.mis` file.

```bash
# Optical images included by default
thyra data.d output.zarr

# Skip optical images
thyra data.d output.zarr --no-optical
```

The TIC image is stored with an affine transform that maps it into the optical
image's coordinate space, so overlays work out of the box.

---

## 3D Data

By default, Thyra treats each z-slice as a separate 2D dataset with `_z{i}` suffixes:

```bash
# Default: separate 2D slices (msi_dataset_z0, msi_dataset_z1, ...)
thyra volume.imzML output.zarr

# Combined 3D volume (single table with x, y, z coordinates)
thyra volume.imzML output.zarr --handle-3d
```

---

## Memory

Every conversion streams, whatever the dataset's size: Thyra makes two passes
over the source -- one to count, one to scatter straight into memory-mapped
arrays next to the output -- and writes each table from those arrays, so the
intensity matrix is never held in RAM. There is no mode to switch on; the
`--streaming` flag older scripts pass is accepted and ignored.

!!! info "What does take memory"
    The `var` frame (one row per m/z bin) and the transient shard buffers of
    the write. A few hundred megabytes on a default-resampled axis; a raw,
    unresampled axis of millions of bins is what to resample.

---

## Troubleshooting

### "WinError 5: Access is denied"

Windows has no atomic file replace, so Zarr's metadata writes have to delete the
destination before renaming the new copy over it. If any handle is open on that
file at that instant the rename fails outright instead of waiting.

Thyra retries these renames a few times, which clears the transient contention
Zarr's own concurrent writes create. A failure that survives the retries means
something is holding the store open for longer than that -- typically a Python
session, napari, or a Jupyter notebook that loaded it.

**Fix:** Close any program that has the zarr open, or write to a different output
path.

### Windows: long paths

Windows caps a normal path at 260 characters, and the limit applies to every
file inside the `.zarr` directory rather than to the path you typed. Thyra's
deepest metadata key sits roughly 134 characters below the output path, so an
output path over about 125 characters would once fail part-way through the
write with a confusing error naming a file you never asked for:

```
Error saving SpatialData: [Errno 2] No such file or directory:
  '...\output.zarr\tables\msi_dataset_z0\uns\format_specific\imzml_version\zarr.<hash>.partial'
```

Thyra now detects this before writing and opens the store through an
extended-length (`\\?\`) path, which is exempt from the 260-character limit, so
long output paths convert normally. A log line records when this happens.

!!! note "Reading a store at a long path"
    The store is written correctly, but reading it is subject to the same
    limit, and a read past it does not fail: Windows reports the deep files as
    missing and Zarr returns its fill value for a missing chunk by design.
    `spatialdata.read_zarr` then reports a structurally invalid store, and a
    bare `zarr.open_group` hands back empty metadata (every ontology term as
    `{"accession": "", "name": ""}`) without any error.

    `thyra validate`, `thyra export-metaspace` and
    `thyra.metadata.schema.read_msi_metadata_blocks` detect this and read
    through an extended-length path on their own. For *other* tools either
    enable long path support system-wide
    (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`, `LongPathsEnabled` = 1;
    needs administrator rights and a reboot), pass the store through
    `thyra.utils.windows_paths.prepare_zarr_read_path` or add the `\\?\`
    prefix yourself:
    ```python
    import spatialdata as sd
    sdata = sd.read_zarr(r"\\?\C:\very\long\path\output.zarr")
    ```
    or simply choose a shorter output path such as `C:\msi\out.zarr`.

!!! warning "A relative path is measured differently, and can lose an array"
    Windows applies the limit to the working directory, a separator and the
    relative path **as you spelled it**, before the `..` segments are
    collapsed. A store you can open perfectly well by its absolute path can
    therefore be unreadable through a relative one, and the failure is the
    silent kind described above: the array simply does not appear, with at
    most a `UserWarning` from Zarr about an object it does not recognise.

    Two arrays sitting side by side in the same group of a real store:

    | key | relative | absolute | result |
    | --- | --- | --- | --- |
    | `current_ratio` | 259 | 251 | reads |
    | `mobility_edges` | **260** | 252 | **missing, no error** |

    Nothing about the two differed but the length of the name -- same shape,
    dtype, codecs and shards. Enabling long path support does **not** help
    here, because it only applies to fully qualified paths.

    Pass an absolute path, or pass the store through
    `thyra.utils.windows_paths.prepare_zarr_read_path`, which resolves one for
    you. If something looks missing, compare the two spellings before
    suspecting the data:

    ```python
    import os
    os.path.exists(p), os.path.exists(os.path.abspath(p))
    ```

    A `False, True` result means the path, not the store.

!!! info "Failed conversions exit non-zero and move the partial store aside"
    Any failed conversion exits with status 1, so a script or CI job wrapping
    `thyra` sees the failure. The partially written store is renamed to
    `<output>.zarr.failed` rather than left at the destination, which keeps it
    available for diagnosis and leaves the output path free for a retry.

### An imzML is refused before conversion starts

Thyra checks what the imzML declares against the `.ibd` on disk before it reads
a single spectrum, and refuses rather than converting something it cannot decode
correctly. A truncated `.ibd` is reported with the spectrum index and both byte
figures:

```
imzML spectrum 3 declares a m/z array ending at byte 236, but sample.ibd is 196
bytes (3 spectra are affected; the furthest ends at 356). The .ibd is truncated
or its offsets are wrong -- pyimzml would return empty arrays for these spectra
without raising, and they would simply be missing from the output.
```

The usual cause is an **incomplete copy of the `.ibd`** -- copy it again and
compare the byte count.

The other refusals are declarations Thyra cannot honour:

- `MS:1000574 zlib compression` on either binary array. The underlying parser
  has no decompression path, so it would read the compressed bytes and decode
  them as numbers.
- A param group declaring no precision term, two at once, or one that disagrees
  with the precision the parser resolved. The decode width would be arbitrary.
- `64-bit integer` arrays, whose width differs between Windows and Linux.
  `32-bit integer` is spec-legal and is *not* refused.
- A first spectrum whose `IMS:1000104` encoded byte length contradicts its value
  count at the declared precision -- the signature of an array that decodes at
  the wrong width but exactly the right length.
- A negative offset or length, or a spectrum whose m/z and intensity arrays
  declare different numbers of values.

Re-export the file from the vendor software with 64-bit float m/z and no
compression.

Not everything unusual is refused. Trailing bytes in the `.ibd`, offsets that do
not follow document order, and more than one `<scanSettings>` block are legal
and are logged as warnings.

Before this check these files converted "successfully": a truncated `.ibd`
produced a store containing only the pixels before the cut, with no error.

### "No module named 'timsdata'" or Bruker SDK errors

The Bruker SDK DLLs are bundled for Windows. On Linux/macOS, Bruker data requires
the vendor's `libtimsdata.so` / `libtimsdata.dylib` to be installed separately.

### Pixel size not detected

Thyra never prompts for one. If the source metadata declares no pixel size the
run fails immediately -- before the converter is built, so nothing is written
and there is no partial store to clean up -- logging `Pixel size not found in
metadata` followed by `Use --pixel-size parameter (e.g., --pixel-size 25)`, and
exits 1.

Pass the value yourself and re-run:

```bash
thyra input.imzML output.zarr --pixel-size 50
```

### Memory errors on large datasets

The intensity matrix is never held in memory; what grows with the axis is the
`var` frame. Reduce the number of resampling bins:

```bash
thyra large.d output.zarr --resample-bins 20000
```

### Verbose logging for debugging

```bash
thyra input.imzML output.zarr -v DEBUG --log-file conversion.log
```

---

## What Next?

- **[Tutorial](tutorial.md)** -- a full walkthrough on real and example data
- **[CLI Reference](cli.md)** -- all command-line options
- **[Resampling](resampling.md)** -- how the mass axis is chosen, and how to control it
- **[Output Format](output-format.md)** -- understanding the zarr output structure
- **[API Reference](api.md)** -- Python API documentation
