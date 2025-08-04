# msiconvert/converters/spatialdata/spatialdata_3d_converter.py

import logging
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .base_spatialdata_converter import (
    SPATIALDATA_AVAILABLE,
    BaseSpatialDataConverter,
)

if SPATIALDATA_AVAILABLE:
    import xarray as xr
    from anndata import AnnData
    from spatialdata.models import Image2DModel, TableModel
    from spatialdata.transformations import Identity


class SpatialData3DConverter(BaseSpatialDataConverter):
    """Converter for MSI data to SpatialData format as true 3D volume or single 2D slice."""

    def __init__(self, *args, **kwargs):
        """Initialize 3D converter with handle_3d=True."""
        kwargs["handle_3d"] = True  # Force 3D mode
        super().__init__(*args, **kwargs)

    def _create_data_structures(self) -> Dict[str, Any]:
        """
        Create data structures for 3D volume format.

        Returns:
            Dict containing tables, shapes, images, and data arrays for 3D volume
        """
        # Check processing mode
        self._use_dask_processing = self._should_use_dask_processing()
        self._use_chunked_processing = self._should_use_chunked_processing()
        
        # Return dictionaries to store tables, shapes, and images
        tables: Dict[str, Any] = {}
        shapes: Dict[str, Any] = {}
        images: Dict[str, Any] = {}

        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        n_x, n_y, n_z = self._dimensions

        if self._use_dask_processing or self._use_chunked_processing:
            # For Dask/chunked processing, initialize with sparse storage lists
            sparse_data = {
                "sparse_rows": [],
                "sparse_cols": [], 
                "sparse_data": [],
            }
        else:
            # Standard processing with upfront sparse matrix allocation
            sparse_data = self._create_sparse_matrix()

        return {
            "mode": "3d_volume",
            "sparse_data": sparse_data,
            "coords_df": self._create_coordinates_dataframe(),
            "var_df": self._create_mass_dataframe(),
            "tables": tables,
            "shapes": shapes,
            "images": images,
            "tic_values": np.zeros((n_y, n_x, n_z), dtype=np.float64),
            "total_intensity": np.zeros(
                len(self._common_mass_axis), dtype=np.float64
            ),
            "pixel_count": 0,
        }

    def _process_single_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        """
        Process a single spectrum for 3D volume format.

        Args:
            data_structures: Data structures for storing processed data
            coords: (x, y, z) pixel coordinates
            mzs: Array of m/z values
            intensities: Array of intensity values
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        x, y, z = coords

        # Zero out negative intensities to prevent data corruption
        intensities = np.maximum(intensities, 0.0)

        # Calculate TIC for this pixel after zeroing negatives
        tic_value = float(np.sum(intensities))

        # Update total intensity for average spectrum calculation
        mz_indices = self._map_mass_to_indices(mzs)
        
        # Only update if we have valid indices and matching array sizes
        if len(mz_indices) > 0 and len(mz_indices) == len(intensities):
            data_structures["total_intensity"][mz_indices] += intensities
            data_structures["pixel_count"] += 1

        # Get pixel index for 3D volume
        pixel_idx = self._get_pixel_index(x, y, z)

        # Store TIC value for this pixel
        data_structures["tic_values"][y, x, z] = tic_value

        # Add to sparse matrix
        if self._use_dask_processing or self._use_chunked_processing:
            # For Dask/chunked processing, accumulate in lists
            nonzero_mask = intensities > 0
            if np.any(nonzero_mask):
                data_structures["sparse_data"]["sparse_rows"].extend([pixel_idx] * np.sum(nonzero_mask))
                data_structures["sparse_data"]["sparse_cols"].extend(mz_indices[nonzero_mask])
                data_structures["sparse_data"]["sparse_data"].extend(intensities[nonzero_mask])
        else:
            # Standard processing with direct sparse matrix addition
            self._add_to_sparse_matrix(
                data_structures["sparse_data"], pixel_idx, mz_indices, intensities
            )

    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """
        Finalize 3D volume data by creating tables, shapes, and images.

        Args:
            data_structures: Data structures containing processed data
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")

        try:
            # Store pixel count for metadata
            self._non_empty_pixel_count = data_structures["pixel_count"]

            # Handle sparse data creation based on processing mode
            if not hasattr(data_structures["sparse_data"], 'tocsr'):
                # Data is in dictionary format (chunked/standard processing)
                if (isinstance(data_structures["sparse_data"], dict) and 
                    data_structures["sparse_data"]["sparse_data"]):  # Check if we have data
                    from scipy.sparse import coo_matrix
                    n_x, n_y, n_z = self._dimensions
                    n_pixels = n_x * n_y * n_z
                    n_masses = len(self._common_mass_axis)
                    
                    sparse_matrix = coo_matrix(
                        (data_structures["sparse_data"]["sparse_data"], 
                         (data_structures["sparse_data"]["sparse_rows"], 
                          data_structures["sparse_data"]["sparse_cols"])),
                        shape=(n_pixels, n_masses),
                        dtype=np.float64
                    ).tocsr()
                    logging.info(f"3D volume: Created sparse matrix with {sparse_matrix.nnz} non-zero elements")
                else:
                    # Empty volume
                    n_x, n_y, n_z = self._dimensions
                    n_pixels = n_x * n_y * n_z
                    n_masses = len(self._common_mass_axis)
                    sparse_matrix = sparse.csr_matrix((n_pixels, n_masses), dtype=np.float64)
                    logging.info("3D volume: Created empty sparse matrix")
                
                # Update data structures
                data_structures["sparse_data"] = sparse_matrix
            else:
                # Data is already a sparse matrix (streaming processing)
                if hasattr(data_structures["sparse_data"], 'tocsr'):
                    data_structures["sparse_data"] = data_structures["sparse_data"].tocsr()
                logging.info(f"3D volume: Using pre-created sparse matrix with {data_structures['sparse_data'].nnz} non-zero elements")

            # Create AnnData
            adata = AnnData(
                X=data_structures["sparse_data"],
                obs=data_structures["coords_df"],
                var=data_structures["var_df"],
            )

            # Add average spectrum to .uns (use total_intensity to match original behavior)
            adata.uns["average_spectrum"] = data_structures["total_intensity"]

            # Make sure region column exists and is correct
            region_key = f"{self.dataset_id}_pixels"
            if "region" not in adata.obs.columns:
                adata.obs["region"] = region_key

            # Ensure instance_key is a string column
            adata.obs["instance_key"] = adata.obs.index.astype(str)

            # Create table model
            table = TableModel.parse(
                adata,
                region=region_key,
                region_key="region",
                instance_key="instance_key",
            )

            # Add to tables and create shapes
            data_structures["tables"][self.dataset_id] = table
            data_structures["shapes"][region_key] = self._create_pixel_shapes(
                adata, is_3d=True
            )

            # Create TIC image
            self._create_tic_image(data_structures)

        except Exception as e:
            logging.error(f"Error processing 3D volume: {e}")
            import traceback

            logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
            raise

    def _create_tic_image(self, data_structures: Dict[str, Any]) -> None:
        """
        Create TIC image for 3D volume or 2D slice.

        Args:
            data_structures: Data structures containing TIC values
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_x, n_y, n_z = self._dimensions

        if n_z > 1:
            # True 3D TIC image
            tic_values = data_structures["tic_values"]
            z_size, y_size, x_size = tic_values.shape

            # Add channel dimension for 3D image
            tic_values_with_channel = tic_values.reshape(
                1, z_size, y_size, x_size
            )

            tic_image = xr.DataArray(
                tic_values_with_channel,
                dims=("c", "z", "y", "x"),
                coords={
                    "c": [0],  # Single channel
                    "z": np.arange(z_size) * self.pixel_size_um,
                    "y": np.arange(y_size) * self.pixel_size_um,
                    "x": np.arange(x_size) * self.pixel_size_um,
                },
            )

            # Create Image model for 3D image
            transform = Identity()
            try:
                from spatialdata.models import Image3DModel

                data_structures["images"][f"{self.dataset_id}_tic"] = (
                    Image3DModel.parse(
                        tic_image,
                        transformations={
                            self.dataset_id: transform,
                            "global": transform,
                        },
                    )
                )
            except (ImportError, AttributeError):
                # Fallback if Image3DModel is not available
                logging.warning(
                    "Image3DModel not available, using generic image model"
                )
                from spatialdata.models import ImageModel

                data_structures["images"][f"{self.dataset_id}_tic"] = (
                    ImageModel.parse(
                        tic_image,
                        transformations={
                            self.dataset_id: transform,
                            "global": transform,
                        },
                    )
                )
        else:
            # Single 2D slice
            tic_values = data_structures["tic_values"]

            # Handle both 3D array with single z-slice and 2D array
            if len(tic_values.shape) == 3:
                tic_values = tic_values[:, :, 0]

            y_size, x_size = tic_values.shape

            # Add channel dimension to make it (c, y, x)
            tic_values_with_channel = tic_values.reshape(1, y_size, x_size)

            tic_image = xr.DataArray(
                tic_values_with_channel,
                dims=("c", "y", "x"),
                coords={
                    "c": [0],  # Single channel
                    "y": np.arange(y_size) * self.pixel_size_um,
                    "x": np.arange(x_size) * self.pixel_size_um,
                },
            )

            # Create Image2DModel for the TIC image
            transform = Identity()
            data_structures["images"][f"{self.dataset_id}_tic"] = (
                Image2DModel.parse(
                    tic_image,
                    transformations={
                        self.dataset_id: transform,
                        "global": transform,
                    },
                )
            )

    def _process_dask_result_specific(self, data_structures: Dict[str, Any], result) -> None:
        """Process Dask result for 3D volume format."""
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
            
        # Process each InterpolationResult from Dask result
        for interpolation_result in result:
            coords = interpolation_result.coords
            pixel_idx = interpolation_result.pixel_idx
            sparse_indices = interpolation_result.sparse_indices
            sparse_values = interpolation_result.sparse_values
            tic_value = interpolation_result.tic_value
            
            x, y, z = coords
            
            # Store TIC value
            data_structures["tic_values"][y, x, z] = tic_value
            
            # Add sparse data
            if len(sparse_values) > 0:
                data_structures["sparse_data"]["sparse_rows"].extend([pixel_idx] * len(sparse_values))
                data_structures["sparse_data"]["sparse_cols"].extend(sparse_indices)
                data_structures["sparse_data"]["sparse_data"].extend(sparse_values)
                
            # Update total intensity for average spectrum
            data_structures["total_intensity"][sparse_indices] += sparse_values
            data_structures["pixel_count"] += 1
        
        logging.info(f"Processed {len(result)} pixels from Dask result for 3D volume")
        
    def _add_interpolation_result_to_data_structures(self, data_structures: Dict[str, Any], result) -> None:
        """Add interpolation result to 3D volume data structures."""
        coords = result.coords
        pixel_idx = result.pixel_idx
        sparse_indices = result.sparse_indices
        sparse_values = result.sparse_values
        tic_value = result.tic_value
        
        x, y, z = coords
        
        # Store TIC value
        data_structures["tic_values"][y, x, z] = tic_value
        
        # Add sparse data
        if len(sparse_values) > 0:
            data_structures["sparse_data"]["sparse_rows"].extend([pixel_idx] * len(sparse_values))
            data_structures["sparse_data"]["sparse_cols"].extend(sparse_indices)
            data_structures["sparse_data"]["sparse_data"].extend(sparse_values)
