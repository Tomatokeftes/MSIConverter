<p align="center">
  <img src="assets/thyra-logotype.svg" alt="Thyra" width="420">
</p>

[![PyPI](https://img.shields.io/pypi/v/thyra?logo=pypi&logoColor=white)](https://pypi.org/project/thyra/)
[![Tests](https://img.shields.io/github/actions/workflow/status/M4i-Imaging-Mass-Spectrometry/thyra/tests.yml?branch=main&logo=github)](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/M4i-Imaging-Mass-Spectrometry/thyra/blob/main/notebooks/Thyra_Validation_Workflow.ipynb)

**Thyra** (from Greek *thyra*, meaning "door" or "portal") converts Mass Spectrometry Imaging (MSI) data into the standardized [SpatialData](https://spatialdata.scverse.org/)/Zarr format -- your portal to spatial omics analysis workflows.

---

## Why Thyra?

Mass spectrometry imaging produces rich spatial-molecular data, but every vendor stores it differently. Downstream tools -- napari, squidpy, scanpy -- expect a common format. Thyra bridges that gap:

```
 .imzML  ──┐                        ┌── napari visualisation
 .d      ──┼──  thyra  ──> .zarr  ──┼── squidpy / scanpy analysis
 .raw    ──┘   (SpatialData)        └── custom Python workflows
```

The output is a single SpatialData/Zarr directory containing intensity matrices, TIC images, optical images, pixel geometries, and full metadata -- ready for any tool in the scverse ecosystem.

---

## Features

| | Feature | Description |
|---|---------|-------------|
| **Formats** | Multiple inputs | ImzML, Bruker (.d timsTOF + solariX, Rapiflex), Waters (.raw directory), PHI SmartSoft-TOF (.raw file) |
| **Output** | SpatialData/Zarr | Cloud-ready, chunked, standardised |
| **Scale** | Memory efficient | Streaming mode for 100+ GB datasets |
| **Optics** | Optical alignment | Automatic MSI-to-microscopy registration (Bruker) |
| **Regions** | Multi-region | Handles slides with multiple tissue sections |
| **Resampling** | Physics-aware | Instrument-specific mass axis resampling (on by default in the CLI, opt-in from the Python API) |
| **Ion mobility** | TIMS-aware | The summed spectrum is the instrument's own scan sum; the mass-mobility heatmap is stored by default and a mobility-resolved sibling table on request, fed from the same frame reads |
| **MS/MS** | Demultiplexed | A scheduled PASEF acquisition is written split apart by precursor by default, next to the summed table; the schedule is recorded either way |
| **Defaults** | Decided in the open | Every default a reasonable person could argue with is recorded with its reason and measurements in [Design Decisions](design-decisions.md) |
| **3D** | Volume support | Process as 3D volume or separate 2D slices |
| **Platform** | Cross-platform | Windows, macOS, Linux |

---

## Quick Start

### Install

```bash
pip install thyra
```

### Convert

=== "CLI"

    ```bash
    thyra input.imzML output.zarr
    ```

=== "Python"

    ```python
    from thyra import convert_msi

    success = convert_msi("input.imzML", "output.zarr")
    ```

### Explore the output

```python
import spatialdata as sd

sdata = sd.read_zarr("output.zarr")

# Intensity matrix (pixels x m/z bins)
table = sdata.tables["msi_dataset_z0"]
print(f"Shape: {table.shape}")
print(f"m/z range: {table.var['mz'].min():.1f} -- {table.var['mz'].max():.1f}")

# TIC image
import numpy as np
tic = np.asarray(sdata.images["msi_dataset_z0_tic"])[0]
```

!!! tip "What is in the output?"
    See [Output Format](output-format.md) for the full structure: tables, TIC images, optical images, pixel shapes, regions, and metadata.

---

## Supported Formats

### Input

| Format | Path | Instruments |
|--------|------|-------------|
| ImzML  | `.imzML` file | Any vendor exporting to the open standard |
| Bruker | `.d` directory | timsTOF fleX, solariX/MRMS FT-ICR, Rapiflex MALDI-TOF |
| Waters | `.raw` directory | MassLynx imaging (DESI, MALDI) |
| PHI    | `.raw` file | SmartSoft-TOF nanoTOF (ToF-SIMS) |

`.raw` is claimed by two vendors and resolved by shape: Waters writes a
directory, PHI writes a single file. See
[Supported Formats](supported-formats.md).

### Output

| Format | Description |
|--------|-------------|
| **SpatialData/Zarr** | The [scverse](https://scverse.org/) standard for spatial omics -- cloud-ready, chunked, with coordinate transforms |

---

## Next Steps

- **[Getting Started](getting-started.md)** -- installation, first conversion, common workflows
- **[Tutorial](tutorial.md)** -- step-by-step walkthrough, from an example dataset to ion images
- **[Supported Formats](supported-formats.md)** -- every input format, how it is detected, what metadata it supplies
- **[CLI Reference](cli.md)** -- every command-line option explained
- **[Resampling](resampling.md)** -- how the common mass axis is chosen, and how to control it
- **[Output Format](output-format.md)** -- what the .zarr contains and how to use it
- **[API Reference](api.md)** -- Python API documentation

---

## Acknowledgments

### Visual identity

The Thyra logomark and logotype were designed by **Nepsis Scriptorium**.

[![Instagram @nepsis.scriptorium](https://img.shields.io/badge/Instagram-%40nepsis.scriptorium-E4405F?logo=instagram&logoColor=white)](https://www.instagram.com/nepsis.scriptorium/)
[![Email nepsisscriptorium@gmail.com](https://img.shields.io/badge/Email-nepsisscriptorium%40gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:nepsisscriptorium@gmail.com)

### Built on

- [SpatialData](https://spatialdata.scverse.org/) ecosystem
- [Zarr](https://zarr.readthedocs.io/) for efficient storage
- [pyimzML](https://github.com/alexandrovteam/pyimzML) for ImzML parsing
