# thyra/converters/spatialdata/spatialdata_3d_converter.py

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base_spatialdata_converter import SPATIALDATA_AVAILABLE, BaseSpatialDataConverter

logger = logging.getLogger(__name__)

if SPATIALDATA_AVAILABLE:
    import xarray as xr
    from anndata import AnnData
    from spatialdata.models import Image2DModel, TableModel
    from spatialdata.transformations import Scale


class SpatialData3DConverter(BaseSpatialDataConverter):
    """Converter for MSI data to SpatialData format as true 3D volume or single 2D slice."""

    def __init__(self, *args, **kwargs):
        """Initialize 3D converter with handle_3d=True."""
        kwargs["handle_3d"] = True  # Force 3D mode
        super().__init__(*args, **kwargs)

    def _create_data_structures(self) -> Dict[str, Any]:
        """Create data structures for 3D volume format.

        Returns:
            Dict containing tables, shapes, images, and data arrays for
            3D volume
        """
        # Return dictionaries to store tables, shapes, and images
        tables: Dict[str, Any] = {}
        shapes: Dict[str, Any] = {}
        images: Dict[str, Any] = {}

        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        n_x, n_y, n_z = self._dimensions

        n_cols = len(self._common_mass_axis)

        data_structures: Dict[str, Any] = {
            "mode": "3d_volume",
            "sparse_matrix": self._create_sparse_matrix(),
            "coords_df": self._create_coordinates_dataframe(),
            "var_df": self._create_mass_dataframe(),
            "tables": tables,
            "shapes": shapes,
            "images": images,
            "tic_values": np.zeros((n_y, n_x, n_z), dtype=np.float64),
            "total_intensity": np.zeros(n_cols, dtype=np.float64),
            "pixel_count": 0,
        }

        # Per-region accumulators for multi-region datasets
        if self._region_map is not None:
            unique_regions = sorted(set(self._region_map.values()))
            data_structures["region_total_intensity"] = {
                r: np.zeros(n_cols, dtype=np.float64) for r in unique_regions
            }
            data_structures["region_pixel_count"] = {r: 0 for r in unique_regions}

        return data_structures

    def _process_single_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        """Process a single spectrum for 3D volume format.

        Delegates to parent's resampling-aware processing.

        Args:
            data_structures: Data structures for storing processed data
            coords: (x, y, z) pixel coordinates
            mzs: Array of m/z values
            intensities: Array of intensity values
        """
        # Delegate to parent's resampling-aware processing
        super()._process_single_spectrum(data_structures, coords, mzs, intensities)

    def _process_resampled_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mz_indices: NDArray[np.int_],
        intensities: NDArray[np.float64],
    ) -> None:
        """Process a spectrum with resampled intensities for 3D volume format.

        Args:
            data_structures: Data structures for storing processed data
            coords: (x, y, z) pixel coordinates
            mz_indices: Indices in the common mass axis (all indices for
            resampled)
            intensities: Resampled intensity values
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        x, y, z = coords

        # Calculate TIC for this pixel
        tic_value = float(np.sum(intensities))

        # Update total intensity for average spectrum calculation
        # Handle both resampled and non-resampled cases
        if len(intensities) == len(data_structures["total_intensity"]):
            # Resampled case - intensities match common mass axis length
            data_structures["total_intensity"] += intensities
        else:
            # Sparse case - scatter onto the mapped indices. Guard the
            # lengths the way the old per-element loop did: only pairs
            # that exist in both arrays, only in-bounds indices.
            n = min(len(intensities), len(mz_indices))
            idx = mz_indices[:n]
            in_bounds = idx < len(data_structures["total_intensity"])
            np.add.at(
                data_structures["total_intensity"],
                idx[in_bounds],
                intensities[:n][in_bounds],
            )
        data_structures["pixel_count"] += 1

        # Per-region accumulation for multi-region datasets
        if "region_total_intensity" in data_structures:
            region_num = self._region_map.get((x, y), -1)
            if region_num in data_structures["region_total_intensity"]:
                np.add.at(
                    data_structures["region_total_intensity"][region_num],
                    mz_indices,
                    intensities,
                )
                data_structures["region_pixel_count"][region_num] += 1

        # Get pixel index for 3D volume
        pixel_idx = self._get_pixel_index(x, y, z)

        # Store TIC value for this pixel
        data_structures["tic_values"][y, x, z] = tic_value

        # Add to sparse matrix
        self._add_to_sparse_matrix(
            data_structures["sparse_matrix"],
            pixel_idx,
            mz_indices,
            intensities,
        )

        self._non_empty_pixel_count += 1

    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Finalize 3D volume data by creating tables, shapes, and images.

        Args:
            data_structures: Data structures containing processed data
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")

        try:
            # Store pixel count for metadata
            self._non_empty_pixel_count = data_structures["pixel_count"]

            # Convert COO arrays to sparse matrix (CSC or CSR based on config)
            format_name = "CSC" if self._sparse_format == "csc" else "CSR"
            logger.info(f"Converting COO arrays to {format_name} format...")
            coo_arrays = data_structures["sparse_matrix"]
            current_idx = coo_arrays["current_idx"]

            # Trim arrays to actual size
            from scipy import sparse

            coo = sparse.coo_matrix(
                (
                    coo_arrays["data"][:current_idx],
                    (
                        coo_arrays["rows"][:current_idx],
                        coo_arrays["cols"][:current_idx],
                    ),
                ),
                shape=(coo_arrays["n_rows"], coo_arrays["n_cols"]),
                dtype=np.float64,
            )
            # Convert to configured sparse format
            sparse_matrix: Any
            if self._sparse_format == "csc":
                sparse_matrix = coo.tocsc()
            else:
                sparse_matrix = coo.tocsr()

            logger.info(
                f"Converted sparse matrix: {sparse_matrix.nnz:,} non-zero entries ({format_name})"
            )

            # Create AnnData
            adata = AnnData(
                X=sparse_matrix,
                obs=data_structures["coords_df"],
                var=data_structures["var_df"],
            )

            # Drop bbox positions that have no spectrum (#88)
            adata = self._drop_empty_pixels(adata)

            # Add average spectrum to .uns. Divided by the pixel count,
            # as the slice paths do: this key is the per-pixel mean
            # everywhere else, including the per-region block below.
            adata.uns["average_spectrum"] = data_structures["total_intensity"] / max(
                data_structures["pixel_count"], 1
            )

            # Add per-region mean spectra for multi-region datasets
            if "region_total_intensity" in data_structures:
                per_region: Dict[str, Any] = {}
                for r, total in data_structures["region_total_intensity"].items():
                    count = data_structures["region_pixel_count"].get(r, 0)
                    per_region[str(r)] = total / max(count, 1)
                adata.uns["average_spectrum_per_region"] = per_region

            # Decide on the sibling tables first so uns can name them.
            self._mobility_table_key = self._plan_mobility_table(self.dataset_id)
            self._msms_table_key = self._plan_msms_table(self.dataset_id)

            # Add MSI metadata to .uns. The 2D and streaming paths have
            # always done this; the 3D one never did, so a volume came
            # out with no record of where it came from at all.
            self._add_metadata_to_uns(adata)

            # Make sure region column exists and is correct
            region_key = f"{self.dataset_id}_pixels"
            if "region" not in adata.obs.columns:
                adata.obs["region"] = pd.Categorical([region_key] * len(adata))
            elif not isinstance(adata.obs["region"].dtype, pd.CategoricalDtype):
                adata.obs["region"] = pd.Categorical(adata.obs["region"])

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
            data_structures["shapes"][region_key] = self._create_pixel_shapes(adata)
            self._attach_mobility_table(
                data_structures, self.dataset_id, region_key, adata.obs
            )
            self._attach_msms_table(
                data_structures, self.dataset_id, region_key, adata.obs
            )

            # Create TIC image
            self._create_tic_image(data_structures)

        except Exception as e:
            logger.error(f"Error processing 3D volume: {e}")
            import traceback

            logger.debug(f"Detailed traceback:\n{traceback.format_exc()}")
            raise

    def _create_tic_image(self, data_structures: Dict[str, Any]) -> None:
        """Create TIC image for 3D volume or 2D slice.

        Args:
            data_structures: Data structures containing TIC values
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_x, n_y, n_z = self._dimensions

        if n_z > 1:
            # True 3D TIC image. The accumulator is indexed [y, x, z] (see
            # _create_data_structures), while Image3DModel declares
            # (c, z, y, x) -- so the axes have to be *moved*. This used to
            # relabel the shape instead, which reshapes nothing and left
            # every volume whose extents were not all equal permuted: a
            # 5(x) x 3(y) x 2(z) acquisition was stored as (1, 3, 5, 2).
            tic_values = data_structures["tic_values"]
            tic_values_zyx = np.transpose(tic_values, (2, 0, 1))

            # Add channel dimension for 3D image
            tic_values_with_channel = tic_values_zyx[np.newaxis, ...]

            # The image is intrinsically in raster pixel indices.
            # Scale into physical micrometers so "global" agrees with
            # pixel-polygon shapes (which are stored in um).
            tic_image = xr.DataArray(
                tic_values_with_channel,
                dims=("c", "z", "y", "x"),
            )

            # z gets its own spacing. It used to reuse the in-plane pitch,
            # which asserts that consecutive slices sit exactly one pixel
            # width apart -- true only by coincidence, since sections are
            # cut by a microtome and the raster is set by the stage. The
            # voxel values were right; the volume was simply the wrong
            # depth. See BaseMSIConverter._resolve_z_spacing for where the
            # number comes from when the caller supplies nothing.
            #
            # Scale pairs values with axis *names*, not by position, so
            # this reads ("x", "y", "z") against a (c, z, y, x) image
            # deliberately. Do not "reorder" it to match the dims.
            transform = Scale(
                [self.pixel_size_um, self.pixel_size_um, self.z_spacing_um],
                axes=("x", "y", "z"),
            )
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
                logger.warning("Image3DModel not available, using generic image model")
                from spatialdata.models import ImageModel

                data_structures["images"][f"{self.dataset_id}_tic"] = ImageModel.parse(
                    tic_image,
                    transformations={
                        self.dataset_id: transform,
                        "global": transform,
                    },
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

            # Image intrinsic CS is raster pixel indices; Scale into
            # physical micrometers so "global" agrees with shapes (um).
            tic_image = xr.DataArray(
                tic_values_with_channel,
                dims=("c", "y", "x"),
            )

            transform = Scale(
                [self.pixel_size_um, self.pixel_size_um],
                axes=("x", "y"),
            )
            data_structures["images"][f"{self.dataset_id}_tic"] = Image2DModel.parse(
                tic_image,
                transformations={
                    self.dataset_id: transform,
                    "global": transform,
                },
            )

        # Add optical images if available
        self._add_optical_images(data_structures)
