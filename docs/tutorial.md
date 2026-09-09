# Tutorial

This is a complete, step-by-step walkthrough of the Thyra pipeline: from a raw
vendor dataset to a SpatialData/Zarr store you can query, plot, and hand to any
tool in the scverse ecosystem.

It comes in two parts, and they are independent:

| | What you need | Time | Runs on |
|---|---|---|---|
| **[Part 1](#part-1-a-five-minute-check-with-synthetic-data)** -- synthetic data | Nothing beyond `pip install thyra` | ~1 minute | Windows, macOS, Linux |
| **[Part 2](#part-2-the-published-example-dataset)** -- published example dataset | 19 GB download, Bruker SDK | Hours | Windows (see [caveat](#platform-requirements)) |

Start with Part 1 to confirm your installation works and to see the shape of the
output. Move to Part 2 when you want to run the real acquisition from the
paper.

---

## Before you start

```bash
pip install thyra
```

Thyra requires Python 3.12 or 3.13. Check what you have:

```bash
thyra --version
```

Every figure below was last re-verified against `thyra` 3.0.0 with
`spatialdata` 0.8.0, `anndata` 0.13.2, `zarr` 3.1.6 and `pandas` 2.3.2 --
the set `pyproject.toml` pins. To check the `spatialdata` version too:

```bash
python -c "import spatialdata; print(spatialdata.__version__)"
```

---

## Part 1: A five-minute check with synthetic data

No download, no vendor SDK, no configuration. Thyra ships a generator that
writes a small synthetic imzML dataset, so you can exercise the whole pipeline
immediately.

### Step 1: Generate a dataset

```bash
thyra-example-data example_data/synthetic_brain.imzML
```

```
Wrote example_data/synthetic_brain.imzML
Wrote example_data/synthetic_brain.ibd
1728 pixels (48 x 36), 4000 m/z bins, 25.0 um, 30.4 MB total
```

This is a brain-like phantom: an elliptical "section" containing a distinct
inner structure, over a m/z range of 250-1200. Three groups of peaks are
written into it -- one shared across the whole section, one confined to the
outer region, one confined to the inner region, plus low-mass matrix ions
covering the entire slide. That gives the images below real spatial structure
to show.

The output is deterministic: the same `--seed` always produces byte-identical
data, so your numbers should match the ones printed in this tutorial exactly.

!!! note "This is synthetic data"
    The phantom is for verifying the software and learning the output layout.
    The peak positions are in a plausible phospholipid range but the intensities
    are invented. For real data, see [Part 2](#part-2-the-published-example-dataset).

### Step 2: Convert it

```bash
thyra example_data/synthetic_brain.imzML example_data/synthetic_brain.zarr
```

Nothing else is required -- no format flag, no pixel size. Thyra detects the
format from the file, reads the pixel size out of the imzML metadata, picks a
resampling strategy, and writes the store. The interesting lines in the log:

```
INFO - Detected format: imzml
INFO - Using reader: ImzMLReader
INFO - Attempting automatic pixel size detection...
INFO - Detected pixel size: 25.0 um
INFO - Selected resampling method: NEAREST_NEIGHBOR
INFO - Building resampled mass axis: 250.00 - 1200.00 m/z, 190000 bins
INFO - Converted sparse matrix for msi_dataset_z0: 6,885,838 non-zero entries (CSC)
INFO - Conversion completed successfully
```

Conversion takes a few seconds and produces a roughly 31 MB `.zarr` directory.

!!! info "Why 190,000 m/z bins from 4,000 input points?"
    Resampling is on by default and builds a uniform axis with 5 mDa bins
    across the mass range, which is finer than this phantom's 0.24 Da spacing.
    The result is correct and stays compact because it is stored sparsely --
    only 4,000 bins per spectrum are populated. Pass `--no-resample` to keep
    the original axis instead. See [Resampling](resampling.md) for how the axis
    type and bin count are chosen.

### Step 3: Open the output

```python
import spatialdata as sd

sdata = sd.read_zarr("example_data/synthetic_brain.zarr")

print("Tables:", list(sdata.tables.keys()))
print("Images:", list(sdata.images.keys()))
print("Shapes:", list(sdata.shapes.keys()))
```

```
Tables: ['msi_dataset_z0']
Images: ['msi_dataset_z0_tic']
Shapes: ['msi_dataset_z0_pixels']
```

Three elements, all named after the dataset id (`msi_dataset` by default,
settable with `--dataset-id`, and `_z0` marking the first z-slice):

- the **table**, an AnnData object holding the intensity matrix
- the **TIC image**, total ion current per pixel
- the **pixel shapes**, one polygon per acquired pixel

Confirm the pixel size survived the round trip:

```python
print(sdata.attrs["pixel_size_x_um"], sdata.attrs["pixel_size_y_um"])
print(sdata.attrs["pixel_size_units"])
print(sdata.attrs["pixel_size_detection_info"]["method"])
```

```
25.0 25.0
micrometers
automatic
```

The two are equal here because this raster is square. They are stored
separately because not every raster is.

### Step 4: The intensity matrix

```python
table = sdata.tables["msi_dataset_z0"]
print(f"Shape: {table.shape}  (pixels x m/z bins)")
print(f"m/z range: {table.var['mz'].min():.1f} -- {table.var['mz'].max():.1f}")
print(f"Sparse: {table.X.nnz:,} non-zero of {table.X.shape[0] * table.X.shape[1]:,}")
```

```
Shape: (1728, 190000)  (pixels x m/z bins)
m/z range: 250.0 -- 1200.0
Sparse: 6885838 non-zero of 328320000
```

1728 pixels matches the 48 x 36 grid. The matrix is a `csc_matrix`; keep it
sparse and never call `.toarray()` on the whole thing for a real dataset.

### Step 5: The TIC image

```python
import numpy as np
import matplotlib.pyplot as plt

tic = np.asarray(sdata.images["msi_dataset_z0_tic"])[0]  # (c, y, x) -> (y, x)
print(f"Shape: {tic.shape}, range {tic.min():.0f} -- {tic.max():.0f}")

plt.imshow(tic, cmap="viridis")
plt.colorbar(label="TIC")
plt.title("Total ion current")
plt.show()
```

```
Shape: (36, 48), range 16177 -- 146472
```

You should see a bright ellipse -- the phantom "section" -- against a dim
background of matrix signal.

### Step 6: The average spectrum

```python
mz = table.var["mz"].values
avg = np.asarray(table.uns["average_spectrum"])

plt.plot(mz, avg, linewidth=0.6)
plt.xlim(740, 920)
plt.xlabel("m/z")
plt.ylabel("Average intensity")
plt.show()

peak_mz = mz[np.argmax(avg)]
print(f"Most intense average peak: m/z {peak_mz:.3f}")
```

```
Most intense average peak: m/z 760.513
```

The average is computed over non-empty pixels only, so empty grid positions do
not dilute it.

### Step 7: Ion images

To map one ion, sum the matrix columns inside an m/z window and scatter the
result back onto the pixel grid:

```python
def ion_image(table, target_mz, tol=0.25):
    """Sum intensities in [target-tol, target+tol] and lay them out on the grid."""
    mz = table.var["mz"].values
    lo, hi = np.searchsorted(mz, [target_mz - tol, target_mz + tol])
    values = np.asarray(table.X[:, lo:hi].sum(axis=1)).ravel()

    x = table.obs["x"].values.astype(int)
    y = table.obs["y"].values.astype(int)
    image = np.zeros((y.max() + 1, x.max() + 1))
    image[y, x] = values
    return image


for target, label in [(760.6, "whole section"),
                      (772.5, "outer region"),
                      (888.6, "inner region")]:
    plt.figure()
    plt.imshow(ion_image(table, target), cmap="inferno")
    plt.colorbar(label="Intensity")
    plt.title(f"m/z {target} -- {label}")
    plt.show()
```

The three images differ: m/z 760.6 covers the whole section, 772.5 covers the
outer region with the inner structure punched out as a hole, and 888.6 lights
up only the inner structure. That contrast is what makes MSI worth doing, and
seeing it here confirms intensities landed on the right pixels.

!!! warning "Always integrate over a window, never a single bin"
    On a resampled axis most bins are empty by construction. Picking the single
    nearest bin to a target m/z -- `np.abs(mz - target).argmin()` -- will
    usually land on an empty one and return an all-zero image. Sum over a
    tolerance window, as `ion_image` does above.

That is the full pipeline. Continue to the
[notebook](explore-the-output.ipynb) for optical overlays, per-pixel spectra,
z-slices, and metadata, or to [Part 2](#part-2-the-published-example-dataset)
for real data.

---

## Part 2: The published example dataset

The dataset behind the paper is archived on Zenodo under
[10.5281/zenodo.18326569](https://doi.org/10.5281/zenodo.18326569), licensed
CC-BY-4.0.

| File | Size | MD5 |
|---|---|---|
| `MALDI-MSI_Sagittal_Mouse_Brain.tar.gz` | 19.1 GB | `1dbd2ea14dcb6bd99baf7af6ca843018` |
| `Xenium_Sagittal_Mouse_Brain.tar.gz` | 13.0 GB | `14416994287008b691d43de9dbe34ca3` |
| `he_mouse_brain_adjacent.tar.gz` | 851.5 MB | `8393a3ee8ba59219fe1c9f7f21acb710` |

Only the first is needed to run Thyra. It is a sagittal mouse brain section
acquired on a Bruker timsTOF fleX: MALDI, positive mode, 250-1200 Da, 5 um
raster, 1007 x 1469 pixels. The other two files are the Xenium spatial
transcriptomics and H&E histology of the same specimen, included for
cross-modal work and not used by Thyra.

### Platform requirements

!!! warning "Bruker data needs the vendor SDK"
    This is a Bruker `.d` dataset, so reading it requires the Bruker SDK. The
    DLLs are bundled for **Windows**. On Linux and macOS the vendor's
    `libtimsdata.so` / `libtimsdata.dylib` must be installed separately -- see
    [Troubleshooting](getting-started.md#no-module-named-timsdata-or-bruker-sdk-errors).
    Hosted notebook services such as Colab and Binder cannot run this part.

    If you only need to verify that Thyra works, [Part 1](#part-1-a-five-minute-check-with-synthetic-data)
    is platform-independent and requires no SDK.

You will also need roughly 40 GB of free disk: 19 GB for the archive, 19 GB
extracted, plus the output store.

### Step 1: Download

```bash
pip install zenodo_get
zenodo_get 10.5281/zenodo.18326569 --glob "MALDI-MSI*"
```

Or directly:

```bash
curl -L -O "https://zenodo.org/api/records/18326569/files/MALDI-MSI_Sagittal_Mouse_Brain.tar.gz/content"
```

Verify the download before spending time on conversion:

```bash
md5sum MALDI-MSI_Sagittal_Mouse_Brain.tar.gz
```

It must print `1dbd2ea14dcb6bd99baf7af6ca843018`. On Windows use
`CertUtil -hashfile MALDI-MSI_Sagittal_Mouse_Brain.tar.gz MD5`.

### Step 2: Extract

```bash
tar -xzf MALDI-MSI_Sagittal_Mouse_Brain.tar.gz
```

This gives you:

```
MALDI-MSI Sagittal Mouse Brain/
├── 20240826_Xenium_0040000.tiff       optical image
├── 20240826_Xenium_0040001.tiff       optical image, high resolution
├── 20240826_Xenium_0041899.bak
└── 20240826_Xenium_0041899.d/         the MSI acquisition
    └── 250-1200Pos_Maldi_5um.m/       acquisition method
```

!!! note "The folder name contains spaces"
    Quote the path in every command below, or the shell will split it.

### Step 3: Convert

```bash
thyra "MALDI-MSI Sagittal Mouse Brain/20240826_Xenium_0041899.d" mouse_brain.zarr
```

Same command as Part 1 -- only the input path changes. Thyra detects the Bruker
format, reads the 5 um pixel size from the acquisition metadata, and finds the
optical images next to the `.d` folder.

Size is not a concern: every conversion makes two passes over the source and
writes from memory-mapped arrays, so peak memory stays roughly flat instead of
scaling with the dataset. Expect a long run; add logging so you can watch
progress and keep a record:

```bash
thyra "MALDI-MSI Sagittal Mouse Brain/20240826_Xenium_0041899.d" mouse_brain.zarr \
    -v INFO --log-file conversion.log
```

If memory is tight, reduce the resampling bin count:

```bash
thyra "MALDI-MSI Sagittal Mouse Brain/20240826_Xenium_0041899.d" mouse_brain.zarr \
    --resample-bins 20000
```

### Step 4: Optical alignment

For Bruker data Thyra also writes the microscopy images and aligns the MSI to
them using the teaching points from the FlexImaging metadata. After conversion
you will see additional image elements:

```python
import spatialdata as sd

sdata = sd.read_zarr("mouse_brain.zarr")
print([k for k in sdata.images if "optical" in k])
```

The TIC image carries an affine transform mapping it into the optical image's
pixel space, so overlays line up without manual registration. The
[notebook](explore-the-output.ipynb) shows how to read that transform and plot
the overlay, and [Coordinate Systems](coordinate-systems.md) documents the
contract.

---

## What just happened

Each stage of the pipeline maps onto something you can observe in the steps
above:

```
  input                thyra                        output
  ─────                ─────                        ──────
 .imzML  ──┐      ┌── format detection ──┐      ┌── table      (pixels x m/z)
 .d      ──┼──────┼── pixel size         ┼──────┼── TIC image
 .raw    ──┘      ├── mass resampling    │      ├── optical images + transform
                  └── optical alignment ─┘      └── pixel shapes
                                                    + metadata in .attrs
```

| Stage | Where you saw it |
|---|---|
| Format detection | `Detected format: imzml` / `Using reader: ImzMLReader` |
| Pixel size detection | `Detected pixel size: 25.0 um`, then `sdata.attrs["pixel_size_detection_info"]` |
| Mass resampling | `Building resampled mass axis: ... 190000 bins` |
| Sparse table | `table.X.nnz` of 6,885,838 |
| TIC image | Step 5 |
| Optical alignment | Part 2, Step 4 |
| Metadata preservation | `sdata.attrs`, `table.uns` |

The result is a plain SpatialData store. Nothing in it is Thyra-specific --
`spatialdata.read_zarr` is the only Thyra-aware step, and after that napari,
squidpy, and scanpy work on it directly.

---

## Where to go next

- **[Explore the output notebook](explore-the-output.ipynb)** -- optical
  overlays, per-pixel spectra, z-slices, metadata
- **[Output Format](output-format.md)** -- the full element and metadata layout
- **[CLI Reference](cli.md)** -- every option
- **[Coordinate Systems](coordinate-systems.md)** -- how MSI, optical, and
  global coordinate spaces relate
- **[Getting Started](getting-started.md#troubleshooting)** -- troubleshooting
