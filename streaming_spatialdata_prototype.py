"""
Proof-of-concept: Streaming SpatialData converter for MSI data

This prototype demonstrates how to:
1. Create placeholder SpatialData structure
2. Stream pixel data in chunks
3. Interpolate and write incrementally to Zarr
4. Maintain memory efficiency for large datasets

Based on successful test.py validation of SpatialData incremental writing.
"""

import logging
import shutil
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import zarr
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm

from msiconvert.converters.spatialdata.base_spatialdata_converter import (
    SPATIALDATA_AVAILABLE,
)

# Import MSI converter components
from msiconvert.core.base_reader import BaseMSIReader

if SPATIALDATA_AVAILABLE:
    import geopandas as gpd
    import pandas as pd
    import spatialdata as sd
    from anndata import AnnData
    from shapely.geometry import Polygon
    from spatialdata.models import Image2DModel, Image3DModel, TableModel
    from spatialdata.transformations import Identity


class StreamingSpatialDataConverter:
    """
    Memory-efficient streaming converter for large MSI datasets.

    Key features:
    - Processes data in configurable chunks
    - Parallel interpolation
    - Direct Zarr writing without memory accumulation
    - Supports datasets larger than available RAM
    """

    def __init__(
        self,
        reader: BaseMSIReader,
        output_path: Path,
        dataset_id: str = "msi_dataset",
        pixel_size_um: float = 1.0,
        chunk_size: int = 1000,
        n_workers: Optional[int] = None,
        interpolation_method: str = "linear",
        target_memory_gb: float = 8.0,
    ):
        """
        Initialize streaming converter.

        Args:
            reader: MSI data reader
            output_path: Output zarr path
            dataset_id: Dataset identifier
            pixel_size_um: Pixel size in micrometers
            chunk_size: Number of pixels to process per chunk
            n_workers: Number of parallel workers (default: cpu_count())
            interpolation_method: Interpolation method ('linear', 'cubic')
            target_memory_gb: Target memory usage limit
        """
        self.reader = reader
        self.output_path = Path(output_path)
        self.dataset_id = dataset_id
        self.pixel_size_um = pixel_size_um
        self.chunk_size = chunk_size
        self.n_workers = n_workers or cpu_count()
        self.interpolation_method = interpolation_method
        self.target_memory_gb = target_memory_gb

        # Initialize metadata
        self.metadata = reader.get_essential_metadata()
        self.dimensions = self.metadata.dimensions
        self.total_pixels = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

        # Generate common mass axis
        self.common_mass_axis = self._generate_common_mass_axis()
        self.n_masses = len(self.common_mass_axis)

        # Adaptive chunk sizing
        self.chunk_size = self._estimate_optimal_chunk_size()

        logging.info(f"StreamingSpatialDataConverter initialized:")
        logging.info(f"  Dataset: {self.dimensions} pixels, {self.n_masses} mass bins")
        logging.info(f"  Chunk size: {self.chunk_size} pixels")
        logging.info(f"  Workers: {self.n_workers}")

    def _generate_common_mass_axis(self) -> np.ndarray:
        """
        Generate common mass axis for interpolation.

        TODO: Implement proper mass axis generation based on:
        - Metadata min/max m/z (preferred)
        - Sparse sampling if metadata unavailable
        - Non-linear binning with higher resolution at lower masses
        """
        # Placeholder - use simple linear spacing for now
        # In real implementation, this would use equivalent mass formula
        min_mz = 100.0  # Should come from metadata or sampling
        max_mz = 1000.0  # Should come from metadata or sampling
        n_bins = 2000  # Should be configurable

        return np.linspace(min_mz, max_mz, n_bins)

    def _estimate_optimal_chunk_size(self) -> int:
        """
        Estimate optimal chunk size based on memory constraints.

        Considers:
        - Available memory
        - Interpolation overhead (~3x spectrum size)
        - Multiprocessing memory duplication
        """
        # Estimate memory per pixel (spectrum + interpolated + overhead)
        avg_spectrum_size = 1000  # Should estimate from sample pixels
        memory_per_pixel_mb = (avg_spectrum_size * 8 * 3) / (1024 * 1024)  # 3x overhead

        # Account for multiprocessing overhead
        memory_per_pixel_mb *= self.n_workers + 1

        # Calculate chunk size for target memory usage
        target_memory_mb = self.target_memory_gb * 1024
        optimal_chunk_size = int(target_memory_mb / memory_per_pixel_mb)

        # Ensure reasonable bounds
        return max(100, min(optimal_chunk_size, self.chunk_size))

    def _create_placeholder_spatialdata(self) -> None:
        """
        Create placeholder SpatialData structure with correct dimensions.

        This creates the zarr structure that we'll write to incrementally.
        """
        logging.info("Creating placeholder SpatialData structure...")

        # Create placeholder sparse matrix (empty)
        n_pixels = np.prod(self.dimensions)
        placeholder_sparse = csr_matrix((n_pixels, self.n_masses), dtype=np.float32)

        # Create placeholder coordinates
        coords_data = {
            "spatial_x": np.zeros(n_pixels, dtype=np.float32),
            "spatial_y": np.zeros(n_pixels, dtype=np.float32),
            "spatial_z": np.zeros(n_pixels, dtype=np.float32),
            "x": np.zeros(n_pixels, dtype=np.int32),
            "y": np.zeros(n_pixels, dtype=np.int32),
            "z": np.zeros(n_pixels, dtype=np.int32),
            "region": [f"{self.dataset_id}_pixels"] * n_pixels,
            "instance_key": [str(i) for i in range(n_pixels)],
        }
        coords_df = pd.DataFrame(coords_data)

        # Create mass axis dataframe
        var_df = pd.DataFrame(
            {
                "mz": self.common_mass_axis,
                "_index": [f"mass_{i}" for i in range(self.n_masses)],
            }
        )
        var_df.index = var_df["_index"]

        # Create AnnData with placeholder data
        adata = AnnData(X=placeholder_sparse, obs=coords_df, var=var_df)
        adata.uns["average_spectrum"] = np.zeros(self.n_masses, dtype=np.float64)

        # Create TableModel
        table = TableModel.parse(
            adata,
            region=f"{self.dataset_id}_pixels",
            region_key="region",
            instance_key="instance_key",
        )

        # Create placeholder TIC image
        n_x, n_y, n_z = self.dimensions
        if n_z > 1:
            # 3D TIC image
            tic_placeholder = np.zeros((1, n_z, n_y, n_x), dtype=np.float32)
            import xarray as xr

            tic_image = xr.DataArray(
                tic_placeholder,
                dims=("c", "z", "y", "x"),
                coords={
                    "c": [0],
                    "z": np.arange(n_z) * self.pixel_size_um,
                    "y": np.arange(n_y) * self.pixel_size_um,
                    "x": np.arange(n_x) * self.pixel_size_um,
                },
            )
            tic_model = Image3DModel.parse(
                tic_image, transformations={"global": Identity()}
            )
        else:
            # 2D TIC image
            tic_placeholder = np.zeros((1, n_y, n_x), dtype=np.float32)
            import xarray as xr

            tic_image = xr.DataArray(
                tic_placeholder,
                dims=("c", "y", "x"),
                coords={
                    "c": [0],
                    "y": np.arange(n_y) * self.pixel_size_um,
                    "x": np.arange(n_x) * self.pixel_size_um,
                },
            )
            tic_model = Image2DModel.parse(
                tic_image, transformations={"global": Identity()}
            )

        # Create placeholder shapes
        shapes_data = []
        for i in range(n_pixels):
            # Create small placeholder polygon for each pixel
            x = i % n_x
            y = i // n_x
            poly = Polygon(
                [
                    (x * self.pixel_size_um, y * self.pixel_size_um),
                    ((x + 1) * self.pixel_size_um, y * self.pixel_size_um),
                    ((x + 1) * self.pixel_size_um, (y + 1) * self.pixel_size_um),
                    (x * self.pixel_size_um, (y + 1) * self.pixel_size_um),
                ]
            )
            shapes_data.append(poly)

        shapes_gdf = gpd.GeoDataFrame({"geometry": shapes_data})

        # Create initial SpatialData object
        sdata = sd.SpatialData(
            tables={self.dataset_id: table},
            images={f"{self.dataset_id}_tic": tic_model},
            shapes={f"{self.dataset_id}_pixels": shapes_gdf},
        )

        # Write placeholder structure
        sdata.write(str(self.output_path))
        logging.info(f"Placeholder structure written to {self.output_path}")

    def _get_pixel_chunks(self) -> Iterator[Tuple[int, int]]:
        """
        Generator yielding (start_idx, end_idx) for pixel chunks.
        """
        for start in range(0, self.total_pixels, self.chunk_size):
            end = min(start + self.chunk_size, self.total_pixels)
            yield start, end

    def _interpolate_pixel_chunk(
        self, pixel_data_chunk
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate a chunk of pixels to common mass axis.

        Args:
            pixel_data_chunk: List of (coords, mzs, intensities) tuples

        Returns:
            Tuple of (interpolated_intensities, tic_values)
        """
        chunk_size = len(pixel_data_chunk)
        interpolated_chunk = np.zeros((chunk_size, self.n_masses), dtype=np.float32)
        tic_values = np.zeros(chunk_size, dtype=np.float32)

        for i, (coords, mzs, intensities) in enumerate(pixel_data_chunk):
            # Calculate TIC
            tic_values[i] = np.sum(intensities)

            # Interpolate to common mass axis
            if len(mzs) > 0 and len(intensities) > 0:
                interpolated_intensities = np.interp(
                    self.common_mass_axis, mzs, intensities, left=0, right=0
                )
                interpolated_chunk[i] = interpolated_intensities

        return interpolated_chunk, tic_values

    def convert(self) -> None:
        """
        Main conversion method using streaming approach.
        """
        logging.info("Starting streaming conversion...")

        # Step 1: Create placeholder structure
        self._create_placeholder_spatialdata()

        # Step 2: Open zarr for incremental writing
        root = zarr.open(str(self.output_path), mode="r+")

        # Get handles to arrays we'll write to
        sparse_data_handle = root[f"tables/{self.dataset_id}/X/data"]
        sparse_indices_handle = root[f"tables/{self.dataset_id}/X/indices"]
        sparse_indptr_handle = root[f"tables/{self.dataset_id}/X/indptr"]
        tic_image_handle = root[f"images/{self.dataset_id}_tic/0"]

        # Initialize tracking variables
        total_intensity = np.zeros(self.n_masses, dtype=np.float64)
        processed_pixels = 0
        current_data_offset = 0

        # Step 3: Process pixels in chunks
        with tqdm(total=self.total_pixels, desc="Processing pixels") as pbar:
            for start_idx, end_idx in self._get_pixel_chunks():

                # Read pixel chunk from reader
                pixel_chunk = []
                for pixel_idx in range(start_idx, end_idx):
                    coords, mzs, intensities = self._read_pixel(pixel_idx)
                    pixel_chunk.append((coords, mzs, intensities))

                # Interpolate chunk (parallelizable in future)
                interpolated_chunk, tic_chunk = self._interpolate_pixel_chunk(
                    pixel_chunk
                )

                # Update running totals
                total_intensity += np.sum(interpolated_chunk, axis=0)
                processed_pixels += len(pixel_chunk)

                # Write interpolated data to zarr incrementally
                self._write_chunk_to_zarr(
                    root,
                    interpolated_chunk,
                    tic_chunk,
                    start_idx,
                    end_idx,
                    current_data_offset,
                )

                pbar.update(len(pixel_chunk))

        # Step 4: Finalize average spectrum and metadata
        average_spectrum = (
            total_intensity / processed_pixels
            if processed_pixels > 0
            else total_intensity
        )
        root[f"tables/{self.dataset_id}/uns/average_spectrum"][:] = average_spectrum

        logging.info(
            f"Streaming conversion completed: {processed_pixels} pixels processed"
        )

    def _read_pixel(
        self, pixel_idx: int
    ) -> Tuple[Tuple[int, int, int], np.ndarray, np.ndarray]:
        """
        Read a single pixel from the reader.

        TODO: This needs to interface with your actual reader API
        """
        # Convert linear index to 3D coordinates
        n_x, n_y, n_z = self.dimensions
        z = pixel_idx // (n_x * n_y)
        y = (pixel_idx % (n_x * n_y)) // n_x
        x = pixel_idx % n_x
        coords = (x, y, z)

        # TODO: Use actual reader API
        # For now, return dummy data
        mzs = np.random.uniform(100, 1000, 50)
        intensities = np.random.exponential(100, 50)

        return coords, mzs, intensities

    def _write_chunk_to_zarr(
        self,
        root,
        interpolated_chunk: np.ndarray,
        tic_chunk: np.ndarray,
        start_idx: int,
        end_idx: int,
        data_offset: int,
    ) -> None:
        """
        Write interpolated chunk directly to zarr arrays.

        This is where the magic happens - incremental writing without memory accumulation.
        """
        # TODO: Implement sparse matrix incremental writing
        # This is complex because CSR format requires careful index management

        # For TIC image, we can write directly
        n_x, n_y, n_z = self.dimensions
        for i, pixel_idx in enumerate(range(start_idx, end_idx)):
            z = pixel_idx // (n_x * n_y)
            y = (pixel_idx % (n_x * n_y)) // n_x
            x = pixel_idx % n_x

            if n_z > 1:
                # 3D TIC
                tic_image_handle[0, z, y, x] = tic_chunk[i]
            else:
                # 2D TIC
                tic_image_handle[0, y, x] = tic_chunk[i]


def interpolate_chunk_parallel(args):
    """
    Parallel worker function for chunk interpolation.

    Args:
        args: Tuple of (pixel_chunk, common_mass_axis)
    """
    pixel_chunk, common_mass_axis = args
    # Implementation would go here
    pass


# Example usage
if __name__ == "__main__":
    # This would be integrated with your existing converter registration
    logging.basicConfig(level=logging.INFO)

    print("StreamingSpatialDataConverter prototype ready!")
    print("Key features demonstrated:")
    print("✅ Placeholder SpatialData creation")
    print("✅ Chunk-based processing architecture")
    print("✅ Incremental zarr writing framework")
    print("✅ Memory-efficient design")
    print("\nNext steps:")
    print("- Integrate with actual reader API")
    print("- Implement sparse matrix incremental writing")
    print("- Add parallel interpolation")
    print("- Test with real MSI datasets")
