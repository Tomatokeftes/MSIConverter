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
        
        # Initialize interpolation configuration from kwargs
        self.interpolation_config = self._create_interpolation_config(**kwargs)

    def _create_interpolation_config(self, **kwargs) -> Any:
        """
        Create interpolation configuration from kwargs.
        
        Returns:
            Simple configuration object with interpolation settings
        """
        # Simple configuration object to avoid circular imports
        class SimpleInterpolationConfig:
            def __init__(self, **kwargs):
                self.enabled = kwargs.get('enable_interpolation', False)
                self.method = kwargs.get('interpolation_method', 'pchip')
                self.interpolation_bins = kwargs.get('interpolation_bins')
                self.interpolation_width = kwargs.get('interpolation_width')
                self.interpolation_width_mz = kwargs.get('interpolation_width_mz', 400.0)
                self.max_memory_gb = kwargs.get('max_memory_gb', 8.0)
                self.buffer_size = kwargs.get('buffer_size', 1000)
                self.validate_quality = kwargs.get('validate_quality', True)
                self.adaptive_workers = kwargs.get('adaptive_workers', True)
                self.max_workers = kwargs.get('max_workers', 80)
                self.min_workers = kwargs.get('min_workers', 4)
                
        return SimpleInterpolationConfig(**kwargs)

    def convert(self) -> bool:
        """
        Template method defining the conversion workflow.
        
        Automatically detects if interpolation should be used and routes to
        the appropriate conversion path.

        Returns:
        --------
        bool: True if conversion was successful, False otherwise.
        """
        try:
            logging.info("[DEBUG] BaseMSIConverter.convert() started")
            # Check if interpolation should be used
            if self._should_interpolate():
                logging.info("Using interpolation-based conversion path")
                return self._convert_with_interpolation()
            else:
                logging.info("Using standard conversion path")
                
            # Standard conversion path
            logging.info("[DEBUG] Starting _initialize_conversion...")
            self._initialize_conversion()
            logging.info("[DEBUG] Starting _create_data_structures...")
            data_structures = self._create_data_structures()
            logging.info("[DEBUG] Starting _process_spectra...")
            self._process_spectra(data_structures)
            logging.info("[DEBUG] Starting _finalize_data...")
            self._finalize_data(data_structures)
            logging.info("[DEBUG] Starting _save_output...")
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
            logging.info("[DEBUG] Getting dimensions...")
            self._dimensions = self.reader.get_dimensions()
            if any(d <= 0 for d in self._dimensions):
                raise ValueError(
                    f"Invalid dimensions: {self._dimensions}. All dimensions must be positive."
                )
            logging.info(f"[DEBUG] Dimensions: {self._dimensions}")

            logging.info("[DEBUG] Getting common mass axis (this may take several minutes for large datasets)...")
            import time
            start_time = time.time()
            self._common_mass_axis = self.reader.get_common_mass_axis()
            elapsed_time = time.time() - start_time
            logging.info(f"[DEBUG] Common mass axis retrieved in {elapsed_time:.1f} seconds, length: {len(self._common_mass_axis)}")
            
            if len(self._common_mass_axis) == 0:
                raise ValueError(
                    "Common mass axis is empty. Cannot proceed with conversion."
                )

            logging.info("[DEBUG] Getting metadata...")
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
        Create a DataFrame containing pixel coordinates using a vectorized approach.

        Returns:
        --------
        pd.DataFrame: DataFrame with pixel coordinates
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized.")
        n_x, n_y, n_z = self._dimensions
        
        # 1. Create coordinate arrays for each dimension using NumPy
        # np.meshgrid creates coordinate matrices from coordinate vectors.
        # 'indexing="ij"' ensures the output order matches the nested loop (z, y, x).
        zz, yy, xx = np.meshgrid(
            np.arange(n_z),
            np.arange(n_y),
            np.arange(n_x),
            indexing="ij"
        )

        # 2. Create the DataFrame in a single, efficient operation
        # .ravel() flattens the 3D grid arrays into 1D arrays (columns).
        coords_df = pd.DataFrame({
            "z": zz.ravel(),
            "y": yy.ravel(),
            "x": xx.ravel()
        })

        # 3. Create the pixel_id index (also vectorized)
        pixel_count = n_x * n_y * n_z
        coords_df.index = np.arange(pixel_count).astype(str)
        coords_df.index.name = "pixel_id"
        
        # 4. Calculate spatial coordinates using vectorized multiplication
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
            logging.info("[DEBUG] Interpolation not enabled in config")
            return False
            
        logging.info(f"[DEBUG] Interpolation config found: enabled={interpolation_config.enabled}")
        
        # Check if reader supports required methods for interpolation
        required_methods = ['get_mass_bounds', 'get_spatial_bounds']
        for method in required_methods:
            if not hasattr(self.reader, method):
                logging.warning(f"Reader missing {method}, skipping interpolation")
                return False
        
        logging.info("[DEBUG] Reader has all required methods for interpolation")
        return True
        
    def _setup_interpolation(self) -> Optional['StreamingInterpolationConfig']:
        """
        Setup interpolation configuration based on reader capabilities and user settings.
        
        Returns:
            StreamingInterpolationConfig or None if interpolation not available
        """
        if not self._should_interpolate():
            return None
            
        try:
            from ..interpolators.streaming_engine import StreamingInterpolationConfig
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
            config = StreamingInterpolationConfig(
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
            logging.info("[DEBUG] Starting _convert_with_interpolation")
            # Setup interpolation
            logging.info("[DEBUG] Setting up interpolation...")
            interp_config = self._setup_interpolation()
            if interp_config is None:
                logging.warning("Interpolation setup failed, falling back to standard conversion")
                return self.convert()
                
            logging.info("[DEBUG] Interpolation config created successfully")
            # Try to use Dask pipeline first, fall back to streaming engine
            try:
                from ..processing.dask_pipeline import DaskInterpolationPipeline
                use_dask = True
                logging.info("Using Dask-based interpolation pipeline")
            except ImportError:
                from ..interpolators.streaming_engine import StreamingInterpolationEngine
                use_dask = False
                logging.info("Using threading-based interpolation pipeline")
            
            # Initialize conversion WITHOUT building common mass axis
            logging.info("[DEBUG] Initializing interpolation conversion...")
            self._initialize_interpolation_conversion(interp_config)
            
            # Use the target mass axis from interpolation config
            self._common_mass_axis = interp_config.target_mass_axis
            logging.info(f"[DEBUG] Set common mass axis with {len(self._common_mass_axis)} bins")
            
            # Create data structures with interpolated mass axis
            logging.info("[DEBUG] Creating data structures...")
            data_structures = self._create_data_structures()
            logging.info("[DEBUG] Data structures created")
            
            # Create processing engine
            logging.info(f"[DEBUG] Creating {'Dask' if use_dask else 'streaming'} engine...")
            if use_dask:
                engine = DaskInterpolationPipeline(interp_config)
            else:
                engine = StreamingInterpolationEngine(interp_config)
            logging.info("[DEBUG] Engine created")
            
            # Create output writer specific to the converter format
            logging.info("[DEBUG] Creating interpolation writer...")
            output_writer = self._create_interpolation_writer(data_structures)
            logging.info("[DEBUG] Writer created")
            
            # Process dataset with interpolation
            try:
                # Track start time for progress reporting
                import time
                self._interpolation_start_time = time.time()
                
                logging.info("[DEBUG] Starting engine.process_dataset...")
                engine.process_dataset(
                    reader=self.reader,
                    output_writer=output_writer,
                    progress_callback=self._interpolation_progress_callback
                )
                logging.info("[DEBUG] engine.process_dataset completed")
                
                # Finalize with interpolation statistics
                self._finalize_interpolated_output(data_structures, engine.get_performance_stats())
                
                # Save output
                success = self._save_output(data_structures)
                
                if success:
                    # Log interpolation performance statistics
                    stats = engine.get_performance_stats()
                    logging.info(f"Interpolation completed: {stats['spectra_written']} spectra, "
                               f"{stats['overall_throughput_per_sec']:.1f} spectra/sec")
                               
                    if stats.get('quality_summary'):
                        quality = stats['quality_summary']
                        logging.info(f"Quality metrics: TIC ratio={quality.get('avg_tic_ratio', 'N/A'):.3f}, "
                               f"peak preservation={quality.get('avg_peak_preservation', 'N/A'):.3f}")
                
                return success
                
            finally:
                # Clean shutdown of processing engine
                if hasattr(engine, 'shutdown'):
                    engine.shutdown()
            
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
                          intensities: NDArray[np.float64]):
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
        if completed % 1000 == 0 or completed == total:  # Log every 1000 spectra and at completion
            progress = (completed / total) * 100 if total > 0 else 0
            import time
            rate = completed / ((time.time() - getattr(self, '_interpolation_start_time', time.time())) or 1)
            logging.info(f"Interpolation progress: {completed}/{total} ({progress:.1f}%) - {rate:.0f} spectra/sec")
            
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
