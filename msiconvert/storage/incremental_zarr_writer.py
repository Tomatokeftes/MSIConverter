# msiconvert/storage/incremental_zarr_writer.py

"""
Incremental Zarr writer for streaming MSI data processing.

This module provides the IncrementalZarrWriter class that enables true out-of-core
processing by writing sparse matrix data incrementally to Zarr storage without
accumulating all data in memory.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

try:
    import zarr

    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    zarr = None

from ..processing.interpolation import InterpolationResult


@dataclass
class ZarrStorageConfig:
    """Configuration for Zarr storage settings."""

    chunk_size: int = 10000
    compression: str = "zstd"
    compression_level: int = 3
    sparse_data_dtype: str = "float32"
    sparse_index_dtype: str = "int32"
    resize_increment: int = 100000


class IncrementalZarrWriter:
    """
    Thread-safe incremental Zarr writer for sparse MSI data.

    This class enables true streaming processing by writing interpolated pixel data
    directly to Zarr storage without accumulating all data in memory. It maintains
    sparse matrix components (data, row indices, column indices) in resizable
    Zarr arrays.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        total_pixels: int,
        n_masses: int,
        config: Optional[ZarrStorageConfig] = None,
    ):
        """
        Initialize incremental Zarr writer.

        Args:
            output_path: Path to Zarr store
            total_pixels: Total number of pixels in dataset
            n_masses: Number of mass channels
            config: Storage configuration
        """
        if not ZARR_AVAILABLE:
            raise ImportError(
                "Zarr is not available but is required for incremental writing"
            )

        self.output_path = Path(output_path)
        self.total_pixels = total_pixels
        self.n_masses = n_masses
        self.config = config or ZarrStorageConfig()

        # Thread safety
        self.write_lock = threading.Lock()

        # Current write positions
        self._current_sparse_pos = 0
        self._pixels_written = 0

        # Initialize Zarr store
        self._init_zarr_store()

        logging.info(
            f"Initialized incremental Zarr writer for {total_pixels} pixels, {n_masses} masses"
        )

    def _init_zarr_store(self) -> None:
        """Initialize Zarr store with sparse matrix arrays."""
        # Create or open Zarr store
        self.zarr_root = zarr.open(str(self.output_path), mode="w")

        # Create resizable arrays for sparse matrix in COO format
        self.sparse_data_array = self.zarr_root.create_dataset(
            "sparse_data",
            shape=(0,),
            maxshape=(None,),
            dtype=self.config.sparse_data_dtype,
            chunks=(self.config.chunk_size,),
            compression=self.config.compression,
            compression_level=self.config.compression_level,
        )

        self.sparse_rows_array = self.zarr_root.create_dataset(
            "sparse_rows",
            shape=(0,),
            maxshape=(None,),
            dtype=self.config.sparse_index_dtype,
            chunks=(self.config.chunk_size,),
            compression=self.config.compression,
            compression_level=self.config.compression_level,
        )

        self.sparse_cols_array = self.zarr_root.create_dataset(
            "sparse_cols",
            shape=(0,),
            maxshape=(None,),
            dtype=self.config.sparse_index_dtype,
            chunks=(self.config.chunk_size,),
            compression=self.config.compression,
            compression_level=self.config.compression_level,
        )

        # Create arrays for dense data
        self.tic_values_array = self.zarr_root.create_dataset(
            "tic_values",
            shape=(self.total_pixels,),
            dtype="float64",
            chunks=(min(self.config.chunk_size, self.total_pixels),),
            compression=self.config.compression,
            compression_level=self.config.compression_level,
            fill_value=0.0,
        )

        self.total_intensity_array = self.zarr_root.create_dataset(
            "total_intensity",
            shape=(self.n_masses,),
            dtype="float64",
            chunks=(min(self.config.chunk_size, self.n_masses),),
            compression=self.config.compression,
            compression_level=self.config.compression_level,
            fill_value=0.0,
        )

        # Metadata
        self.zarr_root.attrs["total_pixels"] = self.total_pixels
        self.zarr_root.attrs["n_masses"] = self.n_masses
        self.zarr_root.attrs["sparse_format"] = "coo"
        self.zarr_root.attrs["pixels_written"] = 0
        self.zarr_root.attrs["sparse_nnz"] = 0

        logging.debug(f"Created Zarr arrays for incremental writing")

    def write_interpolation_results(self, results: List[InterpolationResult]) -> None:
        """
        Write a batch of interpolation results to Zarr storage.

        Args:
            results: List of InterpolationResult objects from Dask processing
        """
        if not results:
            return

        with self.write_lock:
            # Count total sparse elements needed
            total_sparse_elements = sum(len(result.sparse_values) for result in results)

            if total_sparse_elements == 0:
                # Still update pixel count and TIC values for empty pixels
                for result in results:
                    self.tic_values_array[result.pixel_idx] = result.tic_value
                    self._pixels_written += 1
                self.zarr_root.attrs["pixels_written"] = self._pixels_written
                return

            # Ensure arrays are large enough
            self._ensure_sparse_capacity(total_sparse_elements)

            # Write sparse data
            write_pos = self._current_sparse_pos
            for result in results:
                if len(result.sparse_values) > 0:
                    # Write sparse matrix data
                    end_pos = write_pos + len(result.sparse_values)

                    self.sparse_data_array[write_pos:end_pos] = result.sparse_values
                    self.sparse_rows_array[write_pos:end_pos] = result.pixel_idx
                    self.sparse_cols_array[write_pos:end_pos] = result.sparse_indices

                    write_pos = end_pos

                    # Update total intensity for average spectrum
                    self.total_intensity_array[
                        result.sparse_indices
                    ] += result.sparse_values

                # Write TIC value
                self.tic_values_array[result.pixel_idx] = result.tic_value
                self._pixels_written += 1

            self._current_sparse_pos = write_pos

            # Update metadata
            self.zarr_root.attrs["pixels_written"] = self._pixels_written
            self.zarr_root.attrs["sparse_nnz"] = self._current_sparse_pos

            logging.debug(
                f"Wrote {len(results)} pixels, {total_sparse_elements} sparse elements to Zarr"
            )

    def _ensure_sparse_capacity(self, additional_elements: int) -> None:
        """Ensure sparse arrays have enough capacity for additional elements."""
        required_size = self._current_sparse_pos + additional_elements
        current_size = self.sparse_data_array.shape[0]

        if required_size > current_size:
            # Calculate new size with some headroom
            new_size = max(required_size, current_size + self.config.resize_increment)

            # Resize all sparse arrays
            self.sparse_data_array.resize((new_size,))
            self.sparse_rows_array.resize((new_size,))
            self.sparse_cols_array.resize((new_size,))

            logging.debug(f"Resized sparse arrays from {current_size} to {new_size}")

    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the Zarr store and return data structure information.

        Returns:
            Dictionary containing sparse matrix data and metadata
        """
        with self.write_lock:
            # Trim sparse arrays to actual size
            if self._current_sparse_pos < self.sparse_data_array.shape[0]:
                self.sparse_data_array.resize((self._current_sparse_pos,))
                self.sparse_rows_array.resize((self._current_sparse_pos,))
                self.sparse_cols_array.resize((self._current_sparse_pos,))

            # Update final metadata
            self.zarr_root.attrs["finalized"] = True
            self.zarr_root.attrs["final_pixels_written"] = self._pixels_written
            self.zarr_root.attrs["final_sparse_nnz"] = self._current_sparse_pos

            # Read data back for creating sparse matrix
            if self._current_sparse_pos > 0:
                sparse_data = np.array(
                    self.sparse_data_array[: self._current_sparse_pos]
                )
                sparse_rows = np.array(
                    self.sparse_rows_array[: self._current_sparse_pos]
                )
                sparse_cols = np.array(
                    self.sparse_cols_array[: self._current_sparse_pos]
                )

                # Create sparse matrix
                from scipy.sparse import coo_matrix

                sparse_matrix = coo_matrix(
                    (sparse_data, (sparse_rows, sparse_cols)),
                    shape=(self.total_pixels, self.n_masses),
                    dtype=np.float64,
                ).tocsr()
            else:
                # Empty sparse matrix
                from scipy.sparse import csr_matrix

                sparse_matrix = csr_matrix(
                    (self.total_pixels, self.n_masses), dtype=np.float64
                )

            # Get dense arrays
            tic_values = np.array(self.tic_values_array[:])
            total_intensity = np.array(self.total_intensity_array[:])

            logging.info(
                f"Finalized Zarr writer: {self._pixels_written} pixels, "
                f"{self._current_sparse_pos} sparse elements"
            )

            return {
                "sparse_matrix": sparse_matrix,
                "tic_values": tic_values,
                "total_intensity": total_intensity,
                "pixels_written": self._pixels_written,
                "sparse_nnz": self._current_sparse_pos,
            }

    def get_progress(self) -> Dict[str, int]:
        """Get current progress information."""
        return {
            "pixels_written": self._pixels_written,
            "total_pixels": self.total_pixels,
            "sparse_elements": self._current_sparse_pos,
            "progress_percent": int(100 * self._pixels_written / self.total_pixels),
        }

    def close(self) -> None:
        """Close the Zarr store."""
        if hasattr(self, "zarr_root"):
            # Zarr stores are automatically closed when Python exits
            # But we can explicitly sync to ensure data is written
            self.zarr_root.store.close()
            logging.debug("Closed Zarr store")


def create_incremental_zarr_writer(
    output_path: Union[str, Path],
    dimensions: Tuple[int, int, int],
    n_masses: int,
    config: Optional[ZarrStorageConfig] = None,
) -> IncrementalZarrWriter:
    """
    Factory function to create an incremental Zarr writer.

    Args:
        output_path: Path to Zarr store
        dimensions: (n_x, n_y, n_z) dimensions
        n_masses: Number of mass channels
        config: Storage configuration

    Returns:
        Configured IncrementalZarrWriter instance
    """
    n_x, n_y, n_z = dimensions
    total_pixels = n_x * n_y * n_z

    # Create temporary Zarr path for streaming
    zarr_path = Path(output_path).with_suffix(".zarr.tmp")

    return IncrementalZarrWriter(
        output_path=zarr_path,
        total_pixels=total_pixels,
        n_masses=n_masses,
        config=config,
    )
