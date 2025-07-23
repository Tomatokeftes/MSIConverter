import logging
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from scipy import sparse
from tqdm import tqdm

from .base_reader import BaseMSIReader


class BaseMSIConverter(ABC):
    """
    Base class for MSI data converters with shared functionality.
    Implements common processing steps while allowing format-specific customization.
    """

    def __init__(
        self,
        reader: BaseMSIReader,
        output_path: Union[str, Path, PathLike[str]],
        dataset_id: str = "msi_dataset",
        pixel_size_um: float = 1.0,
        compression_level: int = 5,
        handle_3d: bool = False,
        **kwargs: Any,
    ):
        self.reader = reader
        self.output_path = Path(output_path)
        self.dataset_id = dataset_id
        self.pixel_size_um = pixel_size_um
        self.compression_level = compression_level
        self.handle_3d = handle_3d
        self.options: dict[str, Any] = kwargs
        self._common_mass_axis: Optional[NDArray[np.float64]] = None
        self._dimensions: Optional[Tuple[int, int, int]] = None
        self._metadata: Optional[dict[str, Any]] = None
        from ..config import DEFAULT_BUFFER_SIZE

        self._buffer_size = DEFAULT_BUFFER_SIZE

    def convert(self) -> bool:
        """
        Template method defining the conversion workflow.

        Returns:
        --------
        bool: True if conversion was successful, False otherwise.
        """
        try:
            self._initialize_conversion()
            data_structures = self._create_data_structures()
            self._process_spectra(data_structures)
            self._finalize_data(data_structures)
            success = self._save_output(data_structures)

            return success
        except Exception as e:
            logging.error(f"Error during conversion: {e}")
            import traceback

            logging.error(f"Detailed traceback:\n{traceback.format_exc()}")
            return False
        finally:
            self.reader.close()

    def _initialize_conversion(self) -> None:
        """Initialize conversion by loading dimensions, mass axis and metadata."""
        logging.info("Initializing conversion...")
        try:
            self._dimensions = self.reader.get_dimensions()
            if any(d <= 0 for d in self._dimensions):
                raise ValueError(
                    f"Invalid dimensions: {self._dimensions}. All dimensions must be positive."
                )

            self._common_mass_axis = self.reader.get_common_mass_axis()
            if len(self._common_mass_axis) == 0:
                raise ValueError(
                    "Common mass axis is empty. Cannot proceed with conversion."
                )

            self._metadata = self.reader.get_metadata()
            if self._metadata is None:  # type: ignore
                self._metadata = {}  # Initialize with empty dict if None

            logging.info(f"Dataset dimensions: {self._dimensions}")
            logging.info(f"Common mass axis length: {len(self._common_mass_axis)}")
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise

    @abstractmethod
    def _create_data_structures(self) -> Any:
        """
        Create format-specific data structures.

        Returns:
        --------
        Any: Format-specific data structures to be used in subsequent steps.
        """
        pass

    def _process_spectra(self, data_structures: Any) -> None:
        """
        Process all spectra from the reader and integrate into data structures.

        Parameters:
        -----------
        data_structures: Format-specific data containers created by _create_data_structures.
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized.")

        total_spectra = self._get_total_spectra_count()
        logging.info(
            f"Converting {total_spectra} spectra to {self.__class__.__name__.replace('Converter', '')} format..."
        )

        setattr(self.reader, "_quiet_mode", True)

        # Process spectra with unified progress tracking
        with tqdm(
            total=total_spectra, desc="Converting spectra", unit="spectrum"
        ) as pbar:
            for coords, mzs, intensities in self.reader.iter_spectra(
                batch_size=self._buffer_size
            ):
                self._process_single_spectrum(data_structures, coords, mzs, intensities)
                pbar.update(1)

    def _get_total_spectra_count(self) -> int:
        """
        Get the total number of spectra for progress tracking.

        This is a helper method to calculate the total spectra count since
        different readers may store this information differently.
        """
        # Try common patterns for getting spectra count
        if hasattr(self.reader, "n_spectra"):
            return self.reader.n_spectra

        # For ImzML readers, count coordinates
        if hasattr(self.reader, "parser") and self.reader.parser is not None:
            if hasattr(self.reader.parser, "coordinates"):
                return len(self.reader.parser.coordinates)

        # For Bruker readers, try frame count methods
        if hasattr(self.reader, "_get_frame_count"):
            return self.reader._get_frame_count()

        # Fallback: calculate dimensions and assume all pixels have data
        # This is less accurate but provides a reasonable estimate
        dimensions = self.reader.get_dimensions()
        total_pixels = dimensions[0] * dimensions[1] * dimensions[2]

        # Log a warning since this is an estimate
        logging.warning(
            f"Could not determine exact spectra count, estimating {total_pixels} from dimensions"
        )
        return total_pixels

    def _process_single_spectrum(
        self,
        data_structures: Any,
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        """
        Process a single spectrum.

        Parameters:
        -----------
        data_structures: Format-specific data containers
        coords: (x, y, z) coordinates
        mzs: m/z values
        intensities: Intensity values
        """
        # Default implementation - to be overridden by subclasses if needed
        pass

    def _finalize_data(self, data_structures: Any) -> None:
        """
        Perform any final processing on the data structures before saving.

        Parameters:
        -----------
        data_structures: Format-specific data containers
        """
        # Default implementation - to be overridden by subclasses if needed
        pass

    @abstractmethod
    def _save_output(self, data_structures: Any) -> bool:
        """
        Save the processed data to the output format.

        Parameters:
        -----------
        data_structures: Format-specific data containers

        Returns:
        --------
        bool: True if saving was successful, False otherwise
        """
        pass

    def add_metadata(self, metadata: Any) -> None:
        """
        Add metadata to the output.
        Base implementation to be extended by subclasses.

        Parameters:
        -----------
        metadata: Any object that can store metadata
        """
        # This will be implemented by subclasses
        pass

    # --- Common Utility Methods ---

    def _create_sparse_matrix(self) -> sparse.lil_matrix:
        """
        Create a sparse matrix for storing spectral data.

        Returns:
        --------
        sparse.lil_matrix: Empty sparse matrix sized for the dataset
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized.")
        n_x, n_y, n_z = self._dimensions
        n_pixels = n_x * n_y * n_z
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized.")
        n_masses = len(self._common_mass_axis)

        logging.info(
            f"Creating sparse matrix for {n_pixels} pixels and {n_masses} mass values"
        )
        return sparse.lil_matrix((n_pixels, n_masses), dtype=np.float64)

    def _create_coordinates_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame containing pixel coordinates.

        Returns:
        --------
        pd.DataFrame: DataFrame with pixel coordinates
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized.")
        n_x, n_y, n_z = self._dimensions

        coords = []
        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    pixel_idx = z * (n_y * n_x) + y * n_x + x
                    coords.append(
                        {  # type: ignore
                            "z": z,
                            "y": y,
                            "x": x,
                            "pixel_id": str(
                                pixel_idx
                            ),  # Convert to string for compatibility
                        }
                    )

        coords_df: pd.DataFrame = pd.DataFrame(coords)
        coords_df.set_index("pixel_id", inplace=True)  # type: ignore

        # Add spatial coordinates
        coords_df["spatial_x"] = coords_df["x"] * self.pixel_size_um
        coords_df["spatial_y"] = coords_df["y"] * self.pixel_size_um
        coords_df["spatial_z"] = coords_df["z"] * self.pixel_size_um

        return coords_df

    def _create_mass_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame containing mass values.

        Returns:
        --------
        pd.DataFrame: DataFrame with mass values
        """
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized.")
        var_df: DataFrame = pd.DataFrame({"mz": self._common_mass_axis})
        # Convert to string index for compatibility
        var_df["mz_str"] = var_df["mz"].astype(str)
        var_df.set_index("mz_str", inplace=True)  # type: ignore

        return var_df

    def _get_pixel_index(self, x: int, y: int, z: int) -> int:
        """
        Convert 3D coordinates to a flat array index.

        Parameters:
        -----------
        x, y, z: Pixel coordinates

        Returns:
        --------
        int: Flat index
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized.")
        n_x, n_y, _ = self._dimensions
        return z * (n_y * n_x) + y * n_x + x

    def _map_mass_to_indices(self, mzs: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Map m/z values to indices in the common mass axis with high accuracy.

        Parameters:
        -----------
        mzs: Array of m/z values

        Returns:
        --------
        NDArray[np.int_]: Array of indices in common mass axis
        """
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized.")

        if mzs.size == 0:
            return np.array([], dtype=int)

        # Use searchsorted for exact mapping
        indices = np.searchsorted(self._common_mass_axis, mzs)

        # Ensure indices are within bounds
        indices = np.clip(indices, 0, len(self._common_mass_axis) - 1)

        # For complete accuracy, validate the indices
        max_diff = 1e-6  # Very small tolerance threshold for floating point differences
        mask = np.abs(self._common_mass_axis[indices] - mzs) <= max_diff

        return indices[mask]

    def _add_to_sparse_matrix(
        self,
        sparse_matrix: sparse.lil_matrix,
        pixel_idx: int,
        mz_indices: NDArray[np.int_],
        intensities: NDArray[np.float64],
    ) -> None:
        """
        Add intensity values to a sparse matrix efficiently.

        Parameters:
        -----------
        sparse_matrix: Target sparse matrix
        pixel_idx: Flat pixel index
        mz_indices: Indices in common mass axis
        intensities: Intensity values
        """
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized.")

        if mz_indices.size == 0 or intensities.size == 0:
            return

        n_masses = len(self._common_mass_axis)

        # Filter out invalid indices and zero intensities in a single pass
        valid_mask = (mz_indices < n_masses) & (intensities > 0)
        if not np.any(valid_mask):
            return

        # Extract valid values
        valid_indices = mz_indices[valid_mask]
        valid_intensities = intensities[valid_mask]

        # Use bulk assignment for better performance
        sparse_matrix[pixel_idx, valid_indices] = valid_intensities

    # --- Interpolation Integration Methods ---

    def _should_interpolate(self) -> bool:
        """
        Decide if interpolation should be applied based on configuration and reader capabilities.
        
        Returns:
            bool: True if interpolation should be applied, False otherwise
        """
        # Check if interpolation is enabled in configuration
        interpolation_config = getattr(self, 'interpolation_config', None)
        if not interpolation_config or not getattr(interpolation_config, 'enabled', False):
            return False
            
        # Check if reader supports required methods for interpolation
        required_methods = ['get_mass_bounds', 'get_spatial_bounds']
        for method in required_methods:
            if not hasattr(self.reader, method):
                logging.warning(f"Reader missing {method}, skipping interpolation")
                return False
                
        return True
        
    def _setup_interpolation(self) -> Optional['InterpolationConfig']:
        """
        Setup interpolation configuration based on reader capabilities and user settings.
        
        Returns:
            InterpolationConfig or None if interpolation not available
        """
        if not self._should_interpolate():
            return None
            
        try:
            from ..interpolators.streaming_engine import InterpolationConfig
            from ..interpolators.bounds_detector import detect_bounds_from_reader
            from ..interpolators.physics_models import create_physics_model
            
            # Get bounds from reader
            bounds_info = detect_bounds_from_reader(self.reader)
            
            # Get instrument info for physics model
            metadata = self.reader.get_metadata()
            instrument_info = metadata.get('instrument', {}) if metadata else {}
            instrument_type = instrument_info.get('type', 'tof').lower()
            
            # Create physics model
            physics_model = create_physics_model(instrument_type)
            
            # Get interpolation settings from converter configuration
            interp_config = getattr(self, 'interpolation_config', None)
            
            # Create optimal mass axis
            if hasattr(interp_config, 'interpolation_width') and interp_config.interpolation_width:
                # Option 2: Use width at reference m/z (like SCiLS approach)
                width_at_mz = (
                    interp_config.interpolation_width, 
                    getattr(interp_config, 'interpolation_width_mz', 400.0)
                )
                target_mass_axis = physics_model.create_optimal_mass_axis(
                    bounds_info.min_mz, bounds_info.max_mz, width_at_mz=width_at_mz
                )
            else:
                # Option 1: Use number of bins (default approach)
                target_bins = getattr(interp_config, 'interpolation_bins', 90000)
                target_mass_axis = physics_model.create_optimal_mass_axis(
                    bounds_info.min_mz, bounds_info.max_mz, target_bins=target_bins
                )
            
            # Create interpolation configuration
            config = InterpolationConfig(
                method=getattr(interp_config, 'method', 'pchip'),
                target_mass_axis=target_mass_axis,
                n_workers=self._calculate_optimal_workers(),
                buffer_size=getattr(interp_config, 'buffer_size', 1000),
                max_memory_gb=getattr(interp_config, 'max_memory_gb', 8.0),
                validate_quality=getattr(interp_config, 'validate_quality', True),
                adaptive_workers=getattr(interp_config, 'adaptive_workers', True)
            )
            
            logging.info(f"Interpolation setup: {len(target_mass_axis)} bins, "
                        f"{config.n_workers} workers, method={config.method}")
            
            return config
            
        except Exception as e:
            logging.error(f"Failed to setup interpolation: {e}")
            return None
            
    def _calculate_optimal_workers(self) -> int:
        """
        Calculate optimal number of workers based on system capabilities.
        
        Returns:
            int: Optimal number of workers
        """
        import os
        
        # Start with CPU count
        cpu_count = os.cpu_count() or 4
        
        # Get interpolation configuration if available
        interp_config = getattr(self, 'interpolation_config', None)
        if interp_config:
            max_workers = getattr(interp_config, 'max_workers', 80)
            min_workers = getattr(interp_config, 'min_workers', 4)
            
            # Use CPU count but respect limits
            optimal = max(min_workers, min(cpu_count, max_workers))
        else:
            # Default conservative approach
            optimal = min(cpu_count, 8)
            
        return optimal
        
    def _create_physics_model(self, instrument_info: dict) -> 'InstrumentPhysics':
        """
        Create appropriate physics model based on instrument information.
        
        Args:
            instrument_info: Dictionary with instrument metadata
            
        Returns:
            InstrumentPhysics: Appropriate physics model
        """
        from ..interpolators.physics_models import create_physics_model
        
        instrument_type = instrument_info.get('type', 'tof').lower()
        resolution_at_400 = instrument_info.get('resolution_at_400')
        
        return create_physics_model(instrument_type, resolution_at_400)
        
    def _initialize_interpolation_conversion(self, interp_config) -> None:
        """Initialize conversion for interpolation path - bypasses common mass axis building."""
        logging.info("Initializing interpolation conversion...")
        try:
            # Get dimensions (fast operation)
            self._dimensions = self.reader.get_dimensions()
            if any(d <= 0 for d in self._dimensions):
                raise ValueError(
                    f"Invalid dimensions: {self._dimensions}. All dimensions must be positive."
                )

            # Get metadata (fast operation)
            self._metadata = self.reader.get_metadata()
            if self._metadata is None:
                self._metadata = {}

            # Skip get_common_mass_axis() - use target axis from interpolation config instead!
            logging.info(f"Dataset dimensions: {self._dimensions}")
            logging.info(f"Interpolation target bins: {len(interp_config.target_mass_axis)}")
            
        except Exception as e:
            logging.error(f"Error during interpolation initialization: {e}")
            raise
        
    def _convert_with_interpolation(self) -> bool:
        """
        Alternative conversion path using interpolation.
        
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        try:
            # Setup interpolation
            interp_config = self._setup_interpolation()
            if interp_config is None:
                logging.warning("Interpolation setup failed, falling back to standard conversion")
                return self.convert()
                
            from ..interpolators.streaming_engine import StreamingInterpolationEngine
            
            # Initialize conversion WITHOUT building common mass axis
            self._initialize_interpolation_conversion(interp_config)
            
            # Use the target mass axis from interpolation config
            self._common_mass_axis = interp_config.target_mass_axis
            
            # Create data structures with interpolated mass axis
            data_structures = self._create_data_structures()
            
            # Create streaming engine
            engine = StreamingInterpolationEngine(interp_config)
            
            # Create output writer specific to the converter format
            output_writer = self._create_interpolation_writer(data_structures)
            
            # Process dataset with interpolation
            engine.process_dataset(
                reader=self.reader,
                output_writer=output_writer,
                progress_callback=self._interpolation_progress_callback
            )
            
            # Finalize with interpolation statistics
            self._finalize_interpolated_output(data_structures, engine.get_performance_stats())
            
            # Save output
            success = self._save_output(data_structures)
            
            if success:
                # Log interpolation performance statistics
                stats = engine.get_performance_stats()
                logging.info(f"Interpolation completed: {stats['spectra_written']} spectra, "
                           f"{stats['overall_throughput_per_sec']:.1f} spectra/sec")
                           
                if stats['quality_summary']:
                    quality = stats['quality_summary']
                    logging.info(f"Quality metrics: TIC ratio={quality.get('avg_tic_ratio', 'N/A'):.3f}, "
                               f"peak preservation={quality.get('avg_peak_preservation', 'N/A'):.3f}")
            
            return success
            
        except Exception as e:
            logging.error(f"Interpolation conversion failed: {e}")
            import traceback
            logging.error(f"Detailed traceback:\n{traceback.format_exc()}")
            return False
        finally:
            self.reader.close()
            
    def _create_interpolation_writer(self, data_structures: Any) -> callable:
        """
        Create writer function for interpolated data.
        This should be overridden by subclasses to provide format-specific writing.
        
        Args:
            data_structures: Format-specific data structures
            
        Returns:
            callable: Writer function that accepts (coords, mass_axis, intensities)
        """
        def default_writer(coords: Tuple[int, int, int], 
                          mass_axis: NDArray[np.float64], 
                          intensities: NDArray[np.float32]):
            """Default writer that falls back to standard processing"""
            # Convert back to the format expected by _process_single_spectrum
            self._process_single_spectrum(data_structures, coords, mass_axis, intensities)
            
        return default_writer
        
    def _interpolation_progress_callback(self, completed: int, total: int):
        """
        Progress callback for interpolation process.
        
        Args:
            completed: Number of completed spectra
            total: Total number of spectra
        """
        # This can be overridden by subclasses for custom progress reporting
        if completed % 1000 == 0:  # Log every 1000 spectra
            progress = (completed / total) * 100 if total > 0 else 0
            logging.info(f"Interpolation progress: {completed}/{total} ({progress:.1f}%)")
            
    def _finalize_interpolated_output(self, data_structures: Any, stats: dict):
        """
        Finalize output with interpolation metadata and statistics.
        This should be overridden by subclasses to add format-specific metadata.
        
        Args:
            data_structures: Format-specific data structures
            stats: Interpolation performance statistics
        """
        # Default implementation just calls standard finalization
        self._finalize_data(data_structures)
        
        # Store interpolation metadata for potential use by subclasses
        self._interpolation_stats = stats
