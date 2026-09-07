<p align="center">
  <img src="docs/assets/thyra-logotype.svg" alt="Thyra" width="420">
</p>

[![Tests](https://img.shields.io/github/actions/workflow/status/M4i-Imaging-Mass-Spectrometry/thyra/tests.yml?branch=main&logo=github)](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/thyra?logo=pypi&logoColor=white)](https://pypi.org/project/thyra/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/M4i-Imaging-Mass-Spectrometry/thyra/blob/main/notebooks/Thyra_Validation_Workflow.ipynb)

**Thyra** (from Greek thyra, meaning "door" or "portal") -- a modern Python library for converting Mass Spectrometry Imaging (MSI) data into the standardized **SpatialData/Zarr format**, serving as your portal to spatial omics analysis workflows.

**[Read the documentation](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra)** | [Getting Started](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/getting-started/) | [Tutorial](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/tutorial/) | [CLI Reference](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/cli/) | [API Reference](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/api/)

### Try it without any data

```bash
pip install thyra
thyra-example-data example_data/synthetic_brain.imzML   # generates a small synthetic dataset
thyra example_data/synthetic_brain.imzML example_data/synthetic_brain.zarr
```

See the **[Tutorial](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/tutorial/)**
for a full walkthrough, including the published example dataset
([10.5281/zenodo.18326569](https://doi.org/10.5281/zenodo.18326569)).

## Features

- **Multiple Input Formats**: ImzML, Bruker (.d directories), Waters (.raw directories), PHI SmartSoft-TOF ToF-SIMS (.raw files)
- **SpatialData Output**: Modern, cloud-ready format with Zarr backend
- **Memory Efficient**: Handles large datasets (100+ GB) through streaming processing
- **Optical Alignment**: Automatic MSI-to-optical image registration for Bruker data
- **Multi-Region Support**: Handles slides with multiple tissue sections
- **Resampling**: Physics-aware mass axis resampling (on by default in the CLI; opt-in from the Python API)
- **Ion Mobility and MS/MS**: TIMS acquisitions keep their mass-mobility heatmap and can carry a mobility-resolved sibling table; scheduled PASEF MS/MS acquisitions are written split apart by precursor by default. The defaults and the measurements behind them are recorded in the [design decisions](https://m4i-imaging-mass-spectrometry.github.io/thyra/design-decisions/)
- **Validated Metadata**: Versioned, ontology-mapped metadata schema (PSI-MS, NCBITaxon, UBERON, CHEBI) with `thyra validate` and one-command METASPACE export
- **3D Support**: Process volume data or treat as 2D slices
- **Cross-Platform**: Windows, macOS, and Linux

## Installation

```bash
pip install thyra
```

## Quick Start

### Command Line

```bash
# Basic conversion (resampling enabled by default)
thyra input.imzML output.zarr

# Bruker data with verbose logging
thyra data.d output.zarr -v DEBUG

# PHI SmartSoft-TOF ToF-SIMS (a .raw file, not a directory)
thyra tofsims_run.raw output.zarr

# Disable resampling
thyra input.imzML output.zarr --no-resample
```

Thyra auto-detects the input format. Note that `.raw` is claimed by two
vendors and resolved by shape: Waters writes a directory, PHI writes a single
file. See [Supported Formats](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/supported-formats/).

### Python API

```python
from thyra import convert_msi

success = convert_msi("data/sample.imzML", "output/sample.zarr")
```

### Working with the Output

```python
import spatialdata as sd

sdata = sd.read_zarr("output/sample.zarr")
msi_table = sdata.tables["msi_dataset_z0"]

print(f"Shape: {msi_table.shape}")  # (pixels, m/z bins)
print(f"m/z range: {msi_table.var['mz'].min():.1f} -- {msi_table.var['mz'].max():.1f}")
```

### Metadata

Every converted store carries a versioned, ontology-mapped metadata block
(`uns["msi_metadata"]`), auto-populated from the source file:

```bash
thyra validate output.zarr                               # schema + ontology checks
thyra export-metaspace output.zarr --merge sample.json   # METASPACE submission JSON
```

See [Metadata Schema](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/metadata-schema/).

## Documentation

Full documentation: **[M4i-Imaging-Mass-Spectrometry.github.io/thyra](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra)**

- [Getting Started](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/getting-started/) -- installation, first conversion, common workflows
- [CLI Reference](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/cli/) -- all command-line options
- [Output Format](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/output-format/) -- understanding the zarr structure
- [Metadata Schema](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/metadata-schema/) -- the validated, ontology-mapped metadata block and METASPACE export
- [Coordinate Systems](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/coordinate-systems/) -- the ``"global"`` contract Thyra writes for downstream consumers
- [API Reference](https://M4i-Imaging-Mass-Spectrometry.github.io/thyra/api/) -- Python API documentation

## Supported Formats

| Input | Extension | Status |
|-------|-----------|--------|
| ImzML | `.imzML` | Full support |
| Bruker | `.d` | Full support (timsTOF + solariX + Rapiflex) |
| Waters | `.raw` directory | Full support |
| PHI SmartSoft-TOF | `.raw` file | Full support (nanoTOF, ToF-SIMS) |
| mzPeak | `.mzpeak` | Experimental (HUPO-PSI v0.9 draft, read-only) |
| Shimadzu | `.imdx`, `.kbd` | In development (workaround: imzML export from IMAGEREVEAL MS) |

PHI mosaic, MS/MS and depth-profiling acquisitions are implemented but so far
tested only against synthetic files -- real data in those modes is very welcome.

Output: **SpatialData/Zarr** -- cloud-ready, efficient, standardized

## Development

```bash
git clone https://github.com/M4i-Imaging-Mass-Spectrometry/thyra.git
cd thyra
uv sync
uv run pre-commit install
uv run pytest
```

## Contributing

See [CONTRIBUTING.md](docs/contributing.md) for guidelines.

## License

MIT -- see [LICENSE](LICENSE).

## Citation

```bibtex
@software{thyra2024,
  title = {Thyra: Modern Mass Spectrometry Imaging Data Conversion},
  author = {Visvikis, Theodoros},
  year = {2024},
  url = {https://github.com/M4i-Imaging-Mass-Spectrometry/thyra}
}
```

## Acknowledgments

- Built with [SpatialData](https://spatialdata.scverse.org/) ecosystem
- Powered by [Zarr](https://zarr.readthedocs.io/) for efficient storage
- Uses [pyimzML](https://github.com/alexandrovteam/pyimzML) for ImzML parsing

### Visual identity

Logomark and logotype designed by **Nepsis Scriptorium**.

[![Instagram @nepsis.scriptorium](https://img.shields.io/badge/Instagram-%40nepsis.scriptorium-E4405F?logo=instagram&logoColor=white)](https://www.instagram.com/nepsis.scriptorium/)
[![Email nepsisscriptorium@gmail.com](https://img.shields.io/badge/Email-nepsisscriptorium%40gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:nepsisscriptorium@gmail.com)
