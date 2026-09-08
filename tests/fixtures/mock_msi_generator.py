"""Generate mock MSI data for testing SpatialData conversion.

This module provides a synthetic MSI reader so the converters can be
exercised end-to-end without real datasets.  ``MockMSIReader`` implements
the full :class:`thyra.core.base_reader.BaseMSIReader` interface (including
``get_region_map``/``get_region_info``/``reset``/``close`` and the metadata
extractor contract), so it is a drop-in stand-in for a real reader.

Run as a script for a quick streaming-conversion smoke test:

    uv run python tests/fixtures/mock_msi_generator.py [--size small|medium|large|huge]

Sizes:
    small  : 100x100 pixels, ~10k spectra (quick test, <1s)
    medium : 500x500 pixels, ~250k spectra (realistic, ~5s)
    large  : 1000x1000 pixels, ~1M spectra (stress test, ~30s)
    huge   : 2000x2000 pixels, ~4M spectra (memory test, ~2min)

The mock data simulates processed MSI data with:
- Sparse spectra (typical ~50-200 peaks per pixel)
- Realistic m/z range (100-2000 Da)
- Peaks placed exactly on the common mass axis so they map without loss
"""

import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from thyra.core.base_extractor import MetadataExtractor
from thyra.core.base_reader import BaseMSIReader
from thyra.metadata.types import ComprehensiveMetadata, EssentialMetadata


@dataclass
class MockMSIConfig:
    """Configuration for mock MSI data generation."""

    n_x: int = 100
    n_y: int = 100
    n_z: int = 1
    mz_min: float = 100.0
    mz_max: float = 2000.0
    n_mz_bins: int = 50000  # Common mass axis bins
    peaks_per_spectrum: Tuple[int, int] = (50, 200)  # min, max peaks
    intensity_range: Tuple[float, float] = (100.0, 10000.0)
    noise_level: float = 0.1
    sparsity: float = 0.0  # Fraction of empty pixels (0-1)
    seed: Optional[int] = 42
    pixel_size_um: float = 10.0


PRESETS = {
    "small": MockMSIConfig(n_x=100, n_y=100, n_mz_bins=30000),
    "medium": MockMSIConfig(n_x=500, n_y=500, n_mz_bins=50000),
    "large": MockMSIConfig(n_x=1000, n_y=1000, n_mz_bins=100000),
    "huge": MockMSIConfig(n_x=2000, n_y=2000, n_mz_bins=200000),
}


class _MockMetadataExtractor(MetadataExtractor):
    """Metadata extractor backed by a :class:`MockMSIConfig`."""

    def __init__(self, config: MockMSIConfig, n_spectra: int):
        super().__init__(data_source=None)
        self._config = config
        self._n_spectra = n_spectra

    def _extract_essential_impl(self) -> EssentialMetadata:
        cfg = self._config
        n_pixels = cfg.n_x * cfg.n_y * cfg.n_z
        avg_peaks = sum(cfg.peaks_per_spectrum) // 2
        return EssentialMetadata(
            dimensions=(cfg.n_x, cfg.n_y, cfg.n_z),
            coordinate_bounds=(
                0.0,
                float(cfg.n_x - 1),
                0.0,
                float(cfg.n_y - 1),
            ),
            mass_range=(cfg.mz_min, cfg.mz_max),
            pixel_size=(cfg.pixel_size_um, cfg.pixel_size_um),
            n_spectra=self._n_spectra,
            total_peaks=self._n_spectra * avg_peaks,
            estimated_memory_gb=n_pixels * 150 * 8 / (1024**3),
            source_path="mock_msi_data",
            # Deliberately a value from the vocabulary the real extractors
            # produce ("centroid spectrum" / "profile spectrum"), and
            # deliberately NOT "processed".  The streaming-PCS writer used
            # to hardcode "processed" into the stored provenance, and this
            # fixture used to say "processed" too -- so every test agreed
            # with the bug and none of them could see it.  Keep these two
            # different.
            spectrum_type="centroid spectrum",
        )

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        return ComprehensiveMetadata(
            essential=self._extract_essential_impl(),
            format_specific={"format": "mock"},
            acquisition_params={},
            instrument_info={"instrument": "mock"},
            raw_metadata={
                "source": "mock",
                # The two list shapes the store contract distinguishes:
                # cvParams is the shape imzML extractors report (a list of
                # dicts, stored as JSON), scan_window a purely numeric
                # list (stored as an array). MS:1000579 carries no
                # polarity/spectrum-type semantics, so the msi_metadata
                # builder ignores it.
                "cvParams": [
                    {"name": "MS1 spectrum", "accession": "MS:1000579", "value": True},
                ],
                "scan_window": [100.0, 1000.0],
            },
        )


class MockMSIReader(BaseMSIReader):
    """Mock MSI reader that generates synthetic data on-the-fly.

    Implements the full :class:`BaseMSIReader` interface so it can be passed
    directly to the converters.  Generated peaks sit exactly on the common
    mass axis, so they survive nearest-neighbour mapping and produce a
    non-empty table (mirroring continuous-then-binned real data).
    """

    def __init__(
        self,
        config: MockMSIConfig,
        optical_image_paths: Optional[List[Path]] = None,
    ):
        """Initialize the mock MSI reader.

        Args:
            config: Mock data generation configuration.
            optical_image_paths: Optional list of TIFF paths to expose via
                ``get_optical_image_paths`` for testing optical handling.
        """
        super().__init__(data_path=Path("mock_msi_data"))
        self.config = config
        self._seed = config.seed
        self._rng = np.random.default_rng(config.seed)

        # Build common mass axis
        self._common_mass_axis = np.linspace(
            config.mz_min, config.mz_max, config.n_mz_bins
        )

        # Simulate some "real" peaks that appear in multiple spectra
        # (like biomarker peaks).
        n_common_peaks = 20
        self._common_peak_indices = self._rng.choice(
            config.n_mz_bins, size=n_common_peaks, replace=False
        )

        # Pre-generate which pixels are empty (for sparse datasets).
        self._n_pixels = config.n_x * config.n_y * config.n_z
        self._empty_pixels: set = set()
        if config.sparsity > 0:
            n_empty = int(self._n_pixels * config.sparsity)
            self._empty_pixels = set(
                self._rng.choice(self._n_pixels, size=n_empty, replace=False).tolist()
            )

        self._optical_image_paths: List[Path] = list(optical_image_paths or [])

    @property
    def _n_spectra(self) -> int:
        return self._n_pixels - len(self._empty_pixels)

    def _create_metadata_extractor(self) -> MetadataExtractor:
        """Create the mock metadata extractor."""
        return _MockMetadataExtractor(self.config, self._n_spectra)

    @property
    def has_shared_mass_axis(self) -> bool:
        """Mock data is processed mode (no shared m/z axis)."""
        return False

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """Return the common mass axis."""
        return self._common_mass_axis

    def get_optical_image_paths(self) -> List[Path]:
        """Return configured optical image paths (empty by default)."""
        return list(self._optical_image_paths)

    def get_peak_counts_per_pixel(self) -> Optional[NDArray[np.int32]]:
        """Return None to exercise the two-pass counting fallback."""
        return None

    def reset(self) -> None:
        """Reset reader state for re-iteration.

        Spectra are a pure function of pixel coordinates (see
        ``_generate_spectrum``), so re-iteration is already reproducible and
        this is effectively a no-op.  It exists to satisfy readers whose
        two-pass converters call ``reset()`` between passes.
        """
        self._rng = np.random.default_rng(self._seed)

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """Iterate through synthetic spectra.

        Yields:
            Tuple of ``((x, y, z), mz_values, intensities)`` for each
            non-empty pixel.
        """
        cfg = self.config
        for z in range(cfg.n_z):
            for y in range(cfg.n_y):
                for x in range(cfg.n_x):
                    pixel_idx = z * (cfg.n_x * cfg.n_y) + y * cfg.n_x + x
                    if pixel_idx in self._empty_pixels:
                        continue
                    mz_indices, intensities = self._generate_spectrum(pixel_idx)
                    mz_values = self._common_mass_axis[mz_indices]
                    yield (x, y, z), mz_values, intensities

    def _generate_spectrum(
        self, pixel_idx: int
    ) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
        """Generate a single synthetic spectrum for a pixel.

        Deterministic in ``pixel_idx``: the streaming PCS path iterates
        the reader twice (count then scatter) and requires the second pass to
        reproduce the first exactly, so the spectrum must not depend on RNG
        history or iteration order.
        """
        cfg = self.config
        base_seed = self._seed if self._seed is not None else 0
        rng = np.random.default_rng([base_seed, pixel_idx])

        n_peaks = rng.integers(cfg.peaks_per_spectrum[0], cfg.peaks_per_spectrum[1])

        # Include some common peaks (biomarkers).
        n_common = min(len(self._common_peak_indices), n_peaks // 3)
        common_indices = rng.choice(
            self._common_peak_indices, size=n_common, replace=False
        )

        # Random peaks.
        n_random = n_peaks - n_common
        random_indices = rng.choice(cfg.n_mz_bins, size=n_random, replace=False)

        all_indices = np.unique(np.concatenate([common_indices, random_indices]))

        # Log-normal intensities clipped to the configured range.
        intensities = rng.lognormal(mean=np.log(1000), sigma=1.5, size=len(all_indices))
        intensities = np.clip(
            intensities, cfg.intensity_range[0], cfg.intensity_range[1]
        )

        if cfg.noise_level > 0:
            noise = rng.normal(0, cfg.noise_level * intensities)
            intensities = np.maximum(intensities + noise, 0)

        return all_indices.astype(np.int32), intensities.astype(np.float64)

    def close(self) -> None:
        """No-op close (mock holds no file handles)."""
        return None


def run_streaming_demo(config: MockMSIConfig, output_dir: Path) -> Path:
    """Run the streaming converter on mock data and report basic stats."""
    import tracemalloc

    from thyra.converters.spatialdata.streaming_converter import (
        StreamingSpatialDataConverter,
    )

    print("\nMock data config:")
    print(f"  Dimensions: {config.n_x} x {config.n_y} x {config.n_z}")
    print(f"  Total pixels: {config.n_x * config.n_y * config.n_z:,}")
    print(f"  M/z bins: {config.n_mz_bins:,}")
    print(f"  Peaks per spectrum: {config.peaks_per_spectrum}")
    print(f"  Sparsity: {config.sparsity * 100:.1f}%")

    reader = MockMSIReader(config)
    output_path = output_dir / "mock_streaming.zarr"

    tracemalloc.start()
    print("\nConverting with streaming method...")
    start_time = time.time()

    converter = StreamingSpatialDataConverter(
        reader=reader,
        output_path=output_path,
        dataset_id="mock",
        pixel_size_um=config.pixel_size_um,
    )
    success = converter.convert()

    elapsed = time.time() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nConversion: {'SUCCESS' if success else 'FAILED'}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Peak memory: {peak / (1024 * 1024):.1f} MB")

    if success:
        print("\nVerifying output with spatialdata.read_zarr ...")
        try:
            import spatialdata

            sdata = spatialdata.read_zarr(str(output_path))
            table = list(sdata.tables.values())[0]
            print("  read_zarr: SUCCESS")
            print(f"  Table shape: {table.X.shape}")
            print(f"  Table NNZ: {table.X.nnz:,}")
            print(f"  Images: {list(sdata.images.keys())}")
            print(f"  Shapes: {list(sdata.shapes.keys())}")
        except Exception as e:  # pragma: no cover - demo diagnostics
            print(f"  read_zarr: FAILED - {e}")

    return output_path


def main() -> None:
    """Run the mock MSI data generator smoke test."""
    print("=" * 70)
    print("  MOCK MSI DATA GENERATOR FOR STREAMING CONVERSION TEST")
    print("=" * 70)

    size = "small"
    for arg in sys.argv[1:]:
        if arg.startswith("--size="):
            size = arg.split("=")[1]
        elif arg in PRESETS:
            size = arg

    if size not in PRESETS:
        print(f"Unknown size: {size}")
        print(f"Available: {list(PRESETS.keys())}")
        return

    config = PRESETS[size]
    print(f"\nUsing preset: {size}")

    temp_dir = Path(tempfile.mkdtemp(prefix="mock_msi_"))
    print(f"Output directory: {temp_dir}")

    output_path = run_streaming_demo(config, temp_dir)

    print("\n" + "=" * 70)
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
