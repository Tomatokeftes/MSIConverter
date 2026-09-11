# Thyra Test Suite

This directory contains the test suite for the `thyra` package, providing both unit tests and integration tests.

## Test Structure

The tests are organized as follows:

- `unit/`: Fast tests for individual components
  - Core functionality (registry, base classes)
  - Readers (imzML and Bruker)
  - Converters (SpatialData)
  - Utility functions

- `integration/`: End-to-end tests for the full conversion workflow
  - imzML format conversion tests
  - Bruker format conversion tests
  - Command-line interface tests

- `data/`: Test data for running the tests
  - Contains minimal test data for both imzML and Bruker formats

- `conftest.py`: Common fixtures for all tests

## Running the Tests

### Running Unit Tests

To run only the unit tests (fast, no external dependencies):

```bash
pytest
```

or explicitly:

```bash
pytest -m "not integration"
```

### Running Integration Tests

To run the integration tests:

```bash
pytest -m integration
```

Most of the lane builds its own inputs and needs nothing further. The tests
that need a file no repository can carry -- a vendor acquisition, or an archive
written by someone else's converter -- are opt-in through an environment
variable naming the path, and **skip cleanly when it is unset**, which is what
CI sees. The variable is the only way to reach them; there is no default
location and no auto-discovery.

| Variable | Points at | Unlocks |
| --- | --- | --- |
| `THYRA_BRUKER_TDF_DATASET` | a real TIMS `.d` directory containing `analysis.tdf` | `integration/test_bruker_real_data.py` (the whole file) and `TestRealAcquisition` in `integration/test_bruker_tdf_synthetic.py`, which checks every frame is read and the lossless TIC matches the database. One variable serves both files |
| `THYRA_BRUKER_PASEF_DATASET` | a `.d` directory from a PASEF (MS/MS) acquisition | the demultiplexed-table tests at the end of `integration/test_bruker_tdf_synthetic.py`, including the cross-store comparison of two schedules of differing shape |
| `THYRA_MZPEAK_REFERENCE_ARCHIVE` | an archive written by the HUPO-PSI reference converter | `test_reference_archive_converts` in `integration/test_convert_mzpeak.py`. Not vendored: that repository publishes no licence, so the check runs against a path the operator supplies |
| `THYRA_TEST_DATA` | the directory holding the real imzML corpus (default: `test_data/` at the repository root, which is gitignored -- see below) | the "a real file is still accepted" cases in `unit/readers/test_imzml_parser_state_validation.py`. Unit lane, not integration, but it fails the same way if you expect it to find files on its own |

The two Bruker variables additionally need the vendor library, which is bundled
for Windows and Linux only; anywhere it cannot be loaded those tests skip
rather than fail. A path that exists but is not a Bruker acquisition still
fails loudly -- the reader rejects it before the library is ever reached.

So a full opt-in run looks like this (POSIX shell; use `$env:NAME = "..."` in
PowerShell):

```bash
THYRA_BRUKER_TDF_DATASET=/path/to/acquisition.d \
THYRA_BRUKER_PASEF_DATASET=/path/to/pasef.d \
THYRA_MZPEAK_REFERENCE_ARCHIVE=/path/to/reference.mzpeak \
  pytest -m integration
```

### Running All Tests

To run both unit and integration tests:

```bash
pytest -m "unit or integration"
```

### Running Specific Test Files

To run tests from a specific file:

```bash
pytest tests/unit/test_registry.py
```

### Running Tests With Coverage Report

To run tests with coverage:

```bash
pytest --cov=thyra
```

For a detailed coverage report:

```bash
pytest --cov=thyra --cov-report=html
```

## Test Dependencies

In addition to the regular package dependencies, the test suite requires:

- pytest
- pytest-cov (optional, for coverage reports)
- mock (for mocking and patching)

These can be installed with:

```bash
pip install pytest pytest-cov mock
```

## Special Test Considerations

- **Bruker Tests**: Some Bruker tests require the Bruker timsdata DLL/shared library to be available. Tests will be skipped if these dependencies are not found.

- **SpatialData Tests**: Tests for SpatialData format conversion require the `spatialdata` package to be installed. Tests will be skipped if this package is not available.

- **Mock Data**: The test suite uses fixtures to create minimal test data rather than requiring large real-world datasets.

## What the local imzML corpus can and cannot exercise

`test_data/` is gitignored, so the three real imzML files live only on
developer machines and CI never sees them. Their properties were measured once,
in full, and are recorded here because re-deriving them costs a streaming scan
of a 2.1 GB XML and an 8-minute conversion — and because **the gaps are the
useful part**. Several code paths look covered and are not.

| | bellini | pea | 20240826_xenium_0041899 |
| --- | --- | --- | --- |
| imzML / `.ibd` | 21 MB / 582 MB | 29 MB / 722 MB | 2.1 GB / 14.0 GB |
| spectra | 16,384 | 12,737 | 918,855 |
| vendor | IONTOF SurfaceLab | SCiLS | SCiLS |
| layout | unindented, CRLF | indented, LF | indented, LF |
| mode | profile | centroid | centroid |
| m/z precision | 64-bit float | 64-bit float | 64-bit float |
| intensity precision | 64-bit float | 32-bit float | 32-bit float |
| convert (default settings) | 5.8 s | 8.7 s | 494 s |

### Nothing here exercises these

Any change to the paths below is untested against real data, whatever the suite
says:

- **Continuous mode.** All three files are `IMS:1000031 processed`. There is no
  continuous file in the corpus, so `get_common_mass_axis`'s spectrum-0 branch
  and `_get_mass_range_continuous` have never met one.
- **Any z dimension.** `IMS:1000052` appears zero times, so pyimzml synthesises
  `z=1` everywhere and the z-handling code only ever sees the one shape real
  data never has.
- **Compression.** All three declare `MS:1000576 no compression`.
- **Integer binary arrays.** No file declares `IMS:1000141/1000142` or
  `MS:1000519/1000522`.
- **More than one `<scanSettings>`.** All three declare `count="1"`.
- **Wrapped 32-bit offsets.** Zero negative offsets and zero inversions across
  **1,896,000** offsets, so pyimzml's `__fix_offsets` repair is dormant on the
  entire corpus. Xenium is the interesting case and it is *correct*: 71.7% of
  its offsets exceed 2^32, the largest is 14,032,038,028, and they are written
  as proper unsigned 64-bit.

Some of these are reachable through the hand-authored corpus in
`data/fixtures/` — see that directory's README for which.

### Two properties worth relying on

- **All three `.ibd` files are packed exactly.** `max(offset + length x itemsize)`
  equals the on-disk size to the byte: 581,940,736 / 721,795,516 /
  14,032,053,520. A file that fails that check is damaged, not unusual.
- **`IMS:1000104` is consistent.** The encoded byte length equals
  `IMS:1000103 x itemsize` for all ~1.9 million arrays, zero violations, which
  is what makes it usable as a cross-check on a declared precision.

### One inconsistency to know about

**bellini's spatial metadata contradicts itself.** `max count of pixels x` x
`pixel size x` is `128 x 4.40625 = 564`, but `max dimension x` says `500`.
Deriving um/pixel from `max dimension / max count` gives `3.90625`; reading
`pixel size` gives `4.40625` — 12.8% apart, with no unit on either cvParam to
arbitrate. pea and xenium are exactly self-consistent, so code that reads the
wrong one is invisible on SCiLS files and silently wrong on IONTOF files.

Unrelated but easy to trip over: bellini's m/z values all lie on a single
global linear-TOF grid (666,526 values from 300 spectra fit
`sqrt(m/z) = s0 + k*g` to within float64 noise), so it is a sparse subset of a
common axis despite being stored as `processed`. pea and xenium are genuinely
arbitrary centroids.

### Every file is 1-based in x and y — measured, not assumed

`_get_spectrum_coordinates` converts to 0-based with a hardcoded `x - 1, y - 1`,
where z is instead rebased on the smallest value present (`_z_base`). PR #147
guarded the converter against the coordinate that asymmetry would produce, and
left open whether any real file actually produces one. It does not:

| | min x | max x | min y | max y | cvParams seen |
| --- | --- | --- | --- | --- | --- |
| bellini | 1 | 128 | 1 | 128 | 16,384 |
| pea | 1 | 131 | 1 | 133 | 12,737 |
| 20240826_xenium_0041899 | 1 | 1007 | 1 | 1469 | 918,855 |

The cvParam count equals each file's spectrum count exactly, which is what makes
the minima trustworthy — a partial scan would undercount and could miss the one
spectrum that mattered. `IMS:1000052` appears zero times in all three, as
recorded above.

**So a 0-based file is unreachable in this corpus, across two vendors.** Do not
re-derive this; it costs a full scan of a 2.1 GB XML.

Three things keep it from being the whole answer, though:

- **The CV permits 0.** `IMS:1000050`, `IMS:1000051` and `IMS:1000052` all carry
  value type `xsd:nonNegativeInteger` (see `thyra/metadata/ontology/_ims.py`),
  which includes zero, and it is the *same* type for all three. There is no
  ontological basis for rebasing z but not x and y — the reason z got
  `_z_base()` was pyimzml's own inconsistency, not a vendor file.
- **Out-of-grid is only reachable downward.** `_calculate_dimensions` sets
  `n_x = max(raw x)` while the reader yields `raw_x - 1`, so a 1-based file fits
  the grid exactly and *cannot* land outside it in either direction. A 0-based
  file would send its entire `x = 0` column to `-1`, which #147 now drops with a
  warning instead of wrapping onto the far edge.
- **Rebasing x/y like z would not be a free fix.** Subtracting the observed
  minimum is safe for z because z is a plane index. For x and y it moves the
  spatial origin: a legitimately 1-based acquisition cropped to `x >= 50`
  currently keeps its absolute position (with an empty left margin that
  `_drop_empty_pixels` trims from obs), and would instead be shifted to 0.
  Whether the origin is the slide or the acquisition is a real decision, not a
  bug — which is why nothing was changed here.

Worth noting the codebase already contains both habits: the Bruker rapiflex
reader *measures* its origin (`x = raster_idx % raster_width`, documented as
"normalize to 0-based by subtracting first_x/first_y", so it cannot go
negative), while the imzML reader assumes its own.

## Adding New Tests

When adding new functionality to the `thyra` package, please follow these guidelines for testing:

1. Add unit tests for new components, functions, or methods
2. Update integration tests if the conversion workflow is affected
3. Add new test fixtures to `conftest.py` if needed
4. Document any special requirements or dependencies for new tests
