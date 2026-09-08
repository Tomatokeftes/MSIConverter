# thyra/converters/spatialdata/spatialdata_2d_converter.py

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from .base_spatialdata_converter import SPATIALDATA_AVAILABLE, BaseSpatialDataConverter

logger = logging.getLogger(__name__)

if SPATIALDATA_AVAILABLE:
    import xarray as xr
    from anndata import AnnData
    from spatialdata.models import Image2DModel, TableModel
    from spatialdata.transformations import Affine, Scale


class SpatialData2DConverter(BaseSpatialDataConverter):
    """Converter for MSI data to SpatialData format treating 3D data as separate 2D slices."""

    def __init__(self, *args, **kwargs):
        """Initialize 2D converter with handle_3d=False."""
        kwargs["handle_3d"] = False  # Force 2D mode
        super().__init__(*args, **kwargs)

    def _create_data_structures(self) -> Dict[str, Any]:
        """Create data structures for 2D slices format.

        Returns:
            Dict containing tables, shapes, images, and data arrays for
            2D slices
        """
        # Return dictionaries to store tables, shapes, and sparse matrices
        tables: Dict[str, Any] = {}
        shapes: Dict[str, Any] = {}
        images: Dict[str, Any] = {}

        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_x, n_y, n_z = self._dimensions

        # Create a structure for each slice
        slices_data: Dict[str, Dict[str, Any]] = {}
        for z in range(n_z):
            slice_id = f"{self.dataset_id}_z{z}"
            slices_data[slice_id] = {
                "sparse_data": self._create_sparse_matrix_for_slice(z),
                "coords_df": self._create_coordinates_dataframe_for_slice(z),
                "tic_values": np.zeros((n_y, n_x), dtype=np.float64),
            }

        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        n_cols = len(self._common_mass_axis)

        data_structures: Dict[str, Any] = {
            "mode": "2d_slices",
            "slices_data": slices_data,
            "tables": tables,
            "shapes": shapes,
            "images": images,
            "var_df": self._create_mass_dataframe(),
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

    def _create_sparse_matrix_for_slice(self, z_value: int) -> Dict[str, Any]:
        """Create COO arrays for a single slice.

        Args:
            z_value: Z-index of the slice

        Returns:
            Dictionary containing pre-allocated COO arrays for this slice
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        n_x, n_y, _ = self._dimensions
        n_pixels = n_x * n_y
        n_masses = len(self._common_mass_axis)

        # Estimate number of non-zero values (assume ~2000 peaks per pixel)
        avg_peaks_per_pixel = 2000
        estimated_nnz = int(n_pixels * avg_peaks_per_pixel * 1.1)  # 10% buffer

        logger.info(
            f"Pre-allocating COO arrays for slice z={z_value}: {n_pixels:,} pixels x "
            f"{n_masses:,} m/z bins"
        )

        # Pre-allocate arrays (int32 for indices saves memory)
        return {
            "rows": np.empty(estimated_nnz, dtype=np.int32),
            "cols": np.empty(estimated_nnz, dtype=np.int32),
            "data": np.empty(estimated_nnz, dtype=np.float64),
            "current_idx": 0,
            "n_rows": n_pixels,
            "n_cols": n_masses,
        }

    def _create_coordinates_dataframe_for_slice(self, z_value: int) -> pd.DataFrame:
        """Create a coordinates dataframe for a single Z-slice.

        Args:
            z_value: Z-index of the slice

        Returns:
            DataFrame with pixel coordinates
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_x, n_y, _ = self._dimensions

        # Pre-allocate arrays for better performance
        pixel_count = n_x * n_y
        y_values: NDArray[np.int32] = np.repeat(np.arange(n_y, dtype=np.int32), n_x)
        x_values: NDArray[np.int32] = np.tile(np.arange(n_x, dtype=np.int32), n_y)
        instance_ids: NDArray[np.int32] = np.arange(pixel_count, dtype=np.int32)

        # Create DataFrame in one operation
        coords_df = pd.DataFrame(
            {
                "y": y_values,
                "x": x_values,
                "instance_id": instance_ids,
                "region": f"{self.dataset_id}_z{z_value}_pixels",
            }
        )

        # Set index efficiently
        coords_df["instance_id"] = coords_df["instance_id"].astype(str)
        coords_df.set_index("instance_id", inplace=True)

        # Add spatial coordinates in a vectorized operation
        coords_df["spatial_x"] = coords_df["x"] * self.pixel_size_um
        coords_df["spatial_y"] = coords_df["y"] * self.pixel_size_um

        # Always add per-pixel region numbers for a consistent schema.
        # Multi-region datasets use the reader's region map; single-region
        # or unknown-region datasets default to region 1.
        coords_df["region_number"] = self.build_region_numbers(x_values, y_values)

        return coords_df

    def _process_single_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        """Process a single spectrum for 2D slices format.

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
        """Process a spectrum with resampled intensities for 2D slices format.

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

        # Update total intensity for average spectrum calculation. The dense
        # resampling path hands the cached full-axis arange, where a plain
        # vector add is ~5x cheaper than the scatter np.add.at performs;
        # sparse indices keep the scatter.
        total_intensity = data_structures["total_intensity"]
        is_dense = (
            mz_indices is self._cached_mass_axis_indices
            or mz_indices is self._identity_mass_indices
        ) and intensities.size == total_intensity.size
        if is_dense:
            total_intensity += intensities
        else:
            np.add.at(total_intensity, mz_indices, intensities)
        data_structures["pixel_count"] += 1

        # Per-region accumulation for multi-region datasets
        if "region_total_intensity" in data_structures:
            region_num = self._region_map.get((x, y), -1)
            if region_num in data_structures["region_total_intensity"]:
                region_total = data_structures["region_total_intensity"][region_num]
                if is_dense:
                    region_total += intensities
                else:
                    np.add.at(region_total, mz_indices, intensities)
                data_structures["region_pixel_count"][region_num] += 1

        # Add data to the appropriate slice
        slice_id = f"{self.dataset_id}_z{z}"
        if slice_id in data_structures["slices_data"]:
            slice_data = data_structures["slices_data"][slice_id]
            pixel_idx = y * self._dimensions[0] + x

            # Store TIC value for this pixel
            slice_data["tic_values"][y, x] = tic_value

            # Add to sparse matrix for this slice
            self._add_to_sparse_matrix(
                slice_data["sparse_data"], pixel_idx, mz_indices, intensities
            )

    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Finalize 2D slices data by creating tables, shapes, and images.

        Args:
            data_structures: Data structures containing processed data
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")

        # Calculate average mass spectrum
        if data_structures["pixel_count"] > 0:
            avg_spectrum = (
                data_structures["total_intensity"] / data_structures["pixel_count"]
            )
        else:
            avg_spectrum = data_structures["total_intensity"].copy()

        # Store pixel count for metadata
        self._non_empty_pixel_count = data_structures["pixel_count"]

        # Process each slice separately
        for slice_id, slice_data in data_structures["slices_data"].items():
            try:
                # Convert the COO arrays to CSC, the one layout written
                logger.info(f"Converting COO arrays to CSC for {slice_id}...")
                coo_arrays = slice_data["sparse_data"]
                current_idx = coo_arrays["current_idx"]

                # Trim arrays to actual size and create COO matrix
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
                sparse_matrix = coo.tocsc()

                logger.info(
                    f"Converted sparse matrix for {slice_id}: "
                    f"{sparse_matrix.nnz:,} non-zero entries (CSC)"
                )

                # Create AnnData for this slice
                adata = AnnData(
                    X=sparse_matrix,
                    obs=slice_data["coords_df"],
                    var=data_structures["var_df"],
                )

                # Drop bbox positions that have no spectrum (#88)
                adata = self._drop_empty_pixels(adata)

                # Add average spectrum to .uns
                adata.uns["average_spectrum"] = avg_spectrum

                # Add per-region mean spectra for multi-region datasets
                self._store_per_region_avg(adata, data_structures)

                # Decide on the sibling tables first so uns can name them,
                # then run the raw mobility pass once for the heatmap and
                # the grid's discovery together, before uns is built.
                self._mobility_table_key = self._plan_mobility_table(slice_id)
                self._msms_table_key = self._plan_msms_table(slice_id)
                self._prepare_sibling_scans(
                    adata.obs, z_value=slice_data.get("z_value")
                )

                # Add MSI metadata to .uns
                self._add_metadata_to_uns(adata)

                # Make sure region column exists and is correct
                region_key = f"{slice_id}_pixels"
                if "region" not in adata.obs.columns:
                    adata.obs["region"] = pd.Categorical([region_key] * len(adata))
                elif not isinstance(adata.obs["region"].dtype, pd.CategoricalDtype):
                    adata.obs["region"] = pd.Categorical(adata.obs["region"])

                # Make sure instance_key is a string column
                adata.obs["instance_key"] = adata.obs.index.astype(str)

                # Create table model
                table = TableModel.parse(
                    adata,
                    region=region_key,
                    region_key="region",
                    instance_key="instance_key",
                )

                # Add to tables and create shapes
                data_structures["tables"][slice_id] = table
                data_structures["shapes"][region_key] = self._create_pixel_shapes(adata)
                self._attach_sibling_tables(
                    data_structures,
                    slice_id,
                    region_key,
                    adata.obs,
                    z_value=slice_data.get("z_value"),
                )

                # Create TIC image for this slice
                tic_values = slice_data["tic_values"]
                y_size, x_size = tic_values.shape

                # Add channel dimension to make it (c, y, x) as required by
                # SpatialData
                tic_values_with_channel = tic_values.reshape(1, y_size, x_size)

                # The image array is intrinsically in raster pixel indices.
                # The transform to "global" expresses the conversion from
                # raster indices into the chosen global frame:
                #   - With FlexImaging optical alignment: Affine into optical
                #     image pixel space (so TIC overlays correctly on the
                #     optical image). global = optical pixels.
                #   - Without alignment: Scale into physical micrometers, so
                #     "global" agrees with the pixel-polygon shapes (which
                #     are stored in um). global = micrometers.
                tic_image = xr.DataArray(
                    tic_values_with_channel,
                    dims=("c", "y", "x"),
                )
                # Gate the alignment-based affine on apply_optical_alignment.
                # When the caller opts out (e.g. Ousia's wizard), MSI lands
                # in pure micrometer coordinates so downstream registration
                # is the canonical alignment step.  The optical image (in
                # _add_optical_images) still uses the alignment internally
                # to land in this same um frame.
                if (
                    self._apply_optical_alignment
                    and self._tic_to_image_matrix is not None
                ):
                    transform = Affine(
                        self._tic_to_image_matrix,
                        input_axes=("x", "y"),
                        output_axes=("x", "y"),
                    )
                else:
                    transform = Scale(
                        [self.pixel_size_um, self.pixel_size_um],
                        axes=("x", "y"),
                    )

                # Create Image2DModel for the TIC image
                data_structures["images"][f"{slice_id}_tic"] = Image2DModel.parse(
                    tic_image,
                    transformations={
                        self.dataset_id: transform,
                        "global": transform,
                    },
                )

            except Exception as e:
                logger.error(f"Error processing slice {slice_id}: {e}")
                import traceback

                logger.debug(f"Detailed traceback:\n{traceback.format_exc()}")
                raise

        # Add optical images if available
        self._add_optical_images(data_structures)

    @staticmethod
    def _store_per_region_avg(
        adata: "AnnData", data_structures: Dict[str, Any]
    ) -> None:
        """Compute and store per-region mean spectra in uns."""
        if "region_total_intensity" not in data_structures:
            return
        per_region: Dict[str, Any] = {}
        for r, total in data_structures["region_total_intensity"].items():
            count = data_structures["region_pixel_count"].get(r, 0)
            per_region[str(r)] = total / max(count, 1)
        adata.uns["average_spectrum_per_region"] = per_region
