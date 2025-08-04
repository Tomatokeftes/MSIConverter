# msiconvert/converters/spatialdata/base_spatialdata_converter.py

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from tqdm import tqdm

from ...core.base_converter import BaseMSIConverter
from ...core.base_reader import BaseMSIReader
from ...processing.interpolation import InterpolationResult, SpectralInterpolator
from ...storage.incremental_zarr_writer import (
    IncrementalZarrWriter,
    ZarrStorageConfig,
    create_incremental_zarr_writer,
)

# Check Dask availability
DASK_AVAILABLE = False
try:
    import dask
    import dask.array as da
    from dask import delayed

    DASK_AVAILABLE = True
except ImportError:
    logging.warning(
        "Dask not available - streaming processing will use chunked fallback"
    )
    DASK_AVAILABLE = False

    # Create dummy delayed decorator for compatibility
    def delayed(func):
        return func

    dask = None
    da = None

# Check SpatialData availability (defer imports to avoid issues)
SPATIALDATA_AVAILABLE = False
_import_error_msg = None
try:
    import geopandas as gpd
    from anndata import AnnData  # type: ignore
    from shapely.geometry import box
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, ShapesModel, TableModel
    from spatialdata.transformations import Identity

    SPATIALDATA_AVAILABLE = True
except (ImportError, NotImplementedError) as e:
    _import_error_msg = str(e)
    logging.warning(f"SpatialData dependencies not available: {e}")
    SPATIALDATA_AVAILABLE = False

    # Create dummy classes for registration
    class AnnData:
        pass

    SpatialData = None
    TableModel = None
    ShapesModel = None
    Image2DModel = None
    Identity = None
    box = None
    gpd = None


class BaseSpatialDataConverter(BaseMSIConverter, ABC):
    """Base converter for MSI data to SpatialData format with shared functionality."""

    def __init__(
        self,
        reader: BaseMSIReader,
        output_path: Path,
        dataset_id: str = "msi_dataset",
        pixel_size_um: float = 1.0,
        handle_3d: bool = False,
        pixel_size_detection_info: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        target_memory_gb: float = 8.0,
        use_dask: Optional[bool] = None,
        dask_chunk_size: Optional[int] = None,
        dask_memory_limit: str = "2GB",
        enable_streaming: bool = False,
        zarr_config: Optional[ZarrStorageConfig] = None,
        analyzer_type: str = "tof",
        tof_bin_width_da: float = 0.1,
        tof_reference_mz: float = 1000.0,
        tof_n_bins: Optional[int] = None,
        fourier_bin_width_da: float = 0.01,
        fourier_n_bins: Optional[int] = None,
        quadrupole_step_size_da: float = 0.1,
        quadrupole_n_bins: Optional[int] = None,
        default_n_bins: int = 10000,
        n_bins: Optional[int] = None,
        disable_interpolation: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the base SpatialData converter.

        Args:
            reader: MSI data reader
            output_path: Path for output file
            dataset_id: Identifier for the dataset
            pixel_size_um: Size of each pixel in micrometers
            handle_3d: Whether to process as 3D data (True) or 2D slices (False)
            pixel_size_detection_info: Optional metadata about pixel size detection
            chunk_size: Number of pixels to process per chunk (None for auto-detection)
            target_memory_gb: Target memory usage limit for chunking decisions
            use_dask: Whether to use Dask streaming (None for auto-detection)
            dask_chunk_size: Dask chunk size in pixels (None for auto-detection)
            dask_memory_limit: Dask memory limit for workers
            analyzer_type: Mass analyzer type ('tof', 'orbitrap', 'ft-icr', 'quadrupole')
            tof_bin_width_da: TOF bin width in Da at reference mass
            tof_reference_mz: Reference m/z for TOF bin width scaling
            tof_n_bins: Alternative to bin_width - specify total number of TOF bins
            fourier_bin_width_da: Constant bin width for Orbitrap/FT-ICR in Da
            fourier_n_bins: Alternative - specify total number of Fourier bins
            quadrupole_step_size_da: Constant step size for quadrupole in Da
            quadrupole_n_bins: Alternative - specify total number of quadrupole bins
            default_n_bins: Default number of bins for unknown analyzer types
            n_bins: Universal override for number of bins (supersedes analyzer-specific settings)
            disable_interpolation: If True, skip interpolation and keep original mass axes
            **kwargs: Additional keyword arguments

        Raises:
            ImportError: If SpatialData dependencies are not available
            ValueError: If pixel_size_um is not positive or dataset_id is empty
        """
        # Check if SpatialData is available
        if not SPATIALDATA_AVAILABLE:
            error_msg = (
                f"SpatialData dependencies not available: {_import_error_msg}. "
                f"Please install required packages or fix dependency conflicts."
            )
            raise ImportError(error_msg)

        # Validate inputs
        if pixel_size_um <= 0:
            raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")
        if not dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")

        # Extract pixel_size_detection_info from kwargs if provided
        kwargs_filtered = dict(kwargs)
        if (
            pixel_size_detection_info is None
            and "pixel_size_detection_info" in kwargs_filtered
        ):
            pixel_size_detection_info = kwargs_filtered.pop("pixel_size_detection_info")

        super().__init__(
            reader,
            output_path,
            dataset_id=dataset_id,
            pixel_size_um=pixel_size_um,
            handle_3d=handle_3d,
            **kwargs_filtered,
        )

        self._non_empty_pixel_count: int = 0
        self._pixel_size_detection_info = pixel_size_detection_info

        # Chunking parameters
        self.chunk_size = chunk_size
        self.target_memory_gb = target_memory_gb
        self._use_chunked_processing = False

        # Dask streaming parameters
        self.use_dask = use_dask
        self.dask_chunk_size = dask_chunk_size
        self.dask_memory_limit = dask_memory_limit
        self._use_dask_processing = False

        # Streaming Zarr parameters
        self.enable_streaming = enable_streaming
        self.zarr_config = zarr_config or ZarrStorageConfig()
        self._zarr_writer: Optional[IncrementalZarrWriter] = None

        # Interpolation components
        self._interpolator: Optional[SpectralInterpolator] = None

        # Mass analyzer parameters for enhanced mass axis generation
        self.analyzer_type = analyzer_type.lower()
        self.tof_bin_width_da = tof_bin_width_da
        self.tof_reference_mz = tof_reference_mz
        self.tof_n_bins = tof_n_bins
        self.fourier_bin_width_da = fourier_bin_width_da
        self.fourier_n_bins = fourier_n_bins
        self.quadrupole_step_size_da = quadrupole_step_size_da
        self.quadrupole_n_bins = quadrupole_n_bins
        self.default_n_bins = default_n_bins
        self.n_bins = n_bins
        self.disable_interpolation = disable_interpolation

    def _initialize_conversion(self) -> None:
        """Initialize conversion with enhanced mass axis generation."""
        logging.info("Loading essential dataset information...")
        try:
            # Load essential metadata first (fast, single query for Bruker)
            essential = self.reader.get_essential_metadata()

            self._dimensions = essential.dimensions
            if any(d <= 0 for d in self._dimensions):
                raise ValueError(
                    f"Invalid dimensions: {self._dimensions}. All dimensions must be positive."
                )

            # Store essential metadata for use throughout conversion
            self._coordinate_bounds = essential.coordinate_bounds
            self._n_spectra = essential.n_spectra
            self._estimated_memory_gb = essential.estimated_memory_gb

            # Override pixel size if not provided and available in metadata
            if self.pixel_size_um == 1.0 and essential.pixel_size:
                self.pixel_size_um = essential.pixel_size[0]
                logging.info(f"Using detected pixel size: {self.pixel_size_um} μm")

            if self.disable_interpolation:
                logging.info(
                    "Interpolation disabled - will preserve original mass axes"
                )
                self._common_mass_axis = None
                self._interpolator = None
            else:
                # Use enhanced mass axis generation instead of reader's method
                self._common_mass_axis = self._generate_common_mass_axis()
                if len(self._common_mass_axis) == 0:
                    raise ValueError(
                        "Common mass axis is empty. Cannot proceed with conversion."
                    )

                # Initialize interpolator with the generated mass axis
                self._interpolator = SpectralInterpolator(
                    common_mass_axis=self._common_mass_axis,
                    interpolation_method="linear",
                    fill_value=0.0,
                    sparsity_threshold=1e-10,
                )

            # Only load comprehensive metadata if needed (lazy loading)
            self._metadata = None  # Will be loaded on demand

            logging.info(f"Dataset dimensions: {self._dimensions}")
            logging.info(f"Coordinate bounds: {self._coordinate_bounds}")
            logging.info(f"Total spectra: {self._n_spectra}")
            logging.info(f"Estimated memory: {self._estimated_memory_gb:.2f} GB")
            logging.info(f"Enhanced mass axis length: {len(self._common_mass_axis)}")
            logging.info(
                f"Mass range: {self._common_mass_axis[0]:.4f} - {self._common_mass_axis[-1]:.4f} m/z"
            )
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise

    def _should_use_dask_processing(self) -> bool:
        """Determine if Dask streaming should be used based on dataset size and availability."""
        # Manual override
        if self.use_dask is not None:
            return self.use_dask and DASK_AVAILABLE

        # Default to Dask if available - it provides better performance for most datasets
        if not DASK_AVAILABLE:
            logging.info(
                "Dask not available, falling back to chunked/standard processing"
            )
            return False

        if self._dimensions is None or self._common_mass_axis is None:
            return False

        # Use Dask by default for better performance and memory management
        # Only fall back to standard processing for very small datasets
        estimated_memory_gb = self._estimate_memory_requirements()

        # Use standard processing only for very small datasets (< 100MB estimated)
        is_very_small = estimated_memory_gb < 0.1
        should_use_dask = not is_very_small

        if should_use_dask:
            logging.info(
                f"Using Dask streaming (default): estimated {estimated_memory_gb:.2f}GB"
            )
        else:
            logging.info(
                f"Using standard processing for small dataset: estimated {estimated_memory_gb:.2f}GB < 0.1GB threshold"
            )

        return should_use_dask

    def _should_use_chunked_processing(self) -> bool:
        """Determine if chunked processing should be used based on memory requirements."""
        if self._dimensions is None or self._common_mass_axis is None:
            return False

        # If using Dask, chunking is handled by Dask itself
        if self._should_use_dask_processing():
            return False

        estimated_memory_gb = self._estimate_memory_requirements()
        should_chunk = estimated_memory_gb > (
            self.target_memory_gb * 0.8
        )  # 80% safety margin

        if should_chunk:
            logging.info(
                f"Using chunked processing: estimated {estimated_memory_gb:.2f}GB > target {self.target_memory_gb}GB"
            )
        else:
            logging.info(
                f"Using standard processing: estimated {estimated_memory_gb:.2f}GB <= target {self.target_memory_gb}GB"
            )

        return should_chunk

    def _estimate_memory_requirements(self) -> float:
        """Estimate total memory requirements for the dataset in GB."""
        if self._dimensions is None or self._common_mass_axis is None:
            return 0.0

        n_x, n_y, n_z = self._dimensions
        n_pixels = n_x * n_y * n_z
        n_masses = len(self._common_mass_axis)

        # Estimate sparse matrix memory (assuming ~10% sparsity for typical MSI data)
        sparsity_factor = 0.1
        sparse_matrix_mb = (n_pixels * n_masses * sparsity_factor * 8) / (
            1024 * 1024
        )  # float64

        # Additional memory for coordinates, mass axis, etc.
        overhead_mb = (n_pixels * 8 * 6) / (1024 * 1024)  # coordinates dataframe

        total_memory_gb = (sparse_matrix_mb + overhead_mb) / 1024
        return total_memory_gb

    def _determine_chunk_size(self) -> int:
        """Determine optimal chunk size based on target memory usage."""
        if self.chunk_size is not None:
            return self.chunk_size

        if self._dimensions is None or self._common_mass_axis is None:
            return 1000  # default fallback

        n_x, n_y, n_z = self._dimensions
        n_pixels = n_x * n_y * n_z
        n_masses = len(self._common_mass_axis)

        # Calculate chunk size to stay within memory target
        target_memory_mb = self.target_memory_gb * 1024

        # Estimate memory per pixel (sparse storage + overhead)
        sparsity_factor = 0.1
        memory_per_pixel_mb = (n_masses * sparsity_factor * 8) / (
            1024 * 1024
        )  # float64

        optimal_chunk_size = int(target_memory_mb / memory_per_pixel_mb)
        chunk_size = max(100, min(optimal_chunk_size, n_pixels))  # reasonable bounds

        logging.info(f"Determined chunk size: {chunk_size} pixels")
        return chunk_size

    def _generate_common_mass_axis(self) -> NDArray[np.float64]:
        """Generate mass axis efficiently with appropriate binning strategy."""

        # Get essential metadata (should contain mass range)
        essential = self.reader.get_essential_metadata()

        # Extract mass range from essential metadata
        if hasattr(essential, "mass_range") and essential.mass_range:
            min_mz, max_mz = essential.mass_range
            logging.info(f"Mass range from metadata: {min_mz:.4f} - {max_mz:.4f} m/z")
        else:
            # Fallback: Quick scan for datasets without mass range in metadata
            logging.warning("No mass range in metadata, scanning dataset...")
            min_mz, max_mz = self._scan_for_mass_range()

        logging.info(f"Using analyzer type: {self.analyzer_type}")

        return self._create_mass_axis(min_mz, max_mz, self.analyzer_type)

    def _create_mass_axis(
        self, min_mz: float, max_mz: float, analyzer_type: str = "unknown"
    ) -> NDArray[np.float64]:
        """Create mass axis with analyzer-specific binning strategy."""

        if analyzer_type in ["tof", "time-of-flight"]:
            return self._create_tof_mass_axis(min_mz, max_mz)
        elif analyzer_type in ["orbitrap", "ft-icr", "fticr"]:
            return self._create_fourier_mass_axis(min_mz, max_mz)
        elif analyzer_type in ["quadrupole", "triple-quad", "qtof"]:
            return self._create_quadrupole_mass_axis(min_mz, max_mz)
        else:
            # Default: linear binning with configurable resolution
            logging.warning(
                f"Unknown analyzer type '{analyzer_type}', using default linear binning"
            )
            return self._create_linear_mass_axis(min_mz, max_mz)

    def _create_tof_mass_axis(
        self, min_mz: float, max_mz: float
    ) -> NDArray[np.float64]:
        """Create mass axis for TOF analyzer with variable bin width."""

        # Universal n_bins override takes priority
        if self.n_bins is not None:
            return self._create_axis_from_n_bins(min_mz, max_mz, self.n_bins, "tof")

        # Alternative: specify number of bins
        if self.tof_n_bins is not None:
            return self._create_axis_from_n_bins(min_mz, max_mz, self.tof_n_bins, "tof")

        # Generate variable-width bins (bin width ∝ √m/z)
        bin_edges = []
        current_mz = min_mz

        while current_mz <= max_mz:
            bin_edges.append(current_mz)
            # Scale bin width: width ∝ √(current_mz / reference_mz)
            scaling_factor = np.sqrt(current_mz / self.tof_reference_mz)
            bin_width = self.tof_bin_width_da * scaling_factor
            current_mz += bin_width

        bin_edges.append(max_mz)  # Ensure we cover the full range

        # Convert edges to centroids
        bin_edges = np.array(bin_edges)
        centroids = (bin_edges[:-1] + bin_edges[1:]) / 2

        logging.info(
            f"TOF mass axis: {len(centroids)} bins, {self.tof_bin_width_da} Da at {self.tof_reference_mz} m/z"
        )
        return centroids

    def _create_fourier_mass_axis(
        self, min_mz: float, max_mz: float
    ) -> NDArray[np.float64]:
        """Create mass axis for Orbitrap/FT-ICR with constant bin width."""

        # Universal n_bins override takes priority
        if self.n_bins is not None:
            return self._create_axis_from_n_bins(min_mz, max_mz, self.n_bins, "linear")

        # Alternative: specify number of bins
        if self.fourier_n_bins is not None:
            return self._create_axis_from_n_bins(
                min_mz, max_mz, self.fourier_n_bins, "linear"
            )

        # Generate linear bins with constant width
        n_bins = int((max_mz - min_mz) / self.fourier_bin_width_da) + 1
        bin_edges = np.linspace(min_mz, max_mz, n_bins + 1)

        # Convert edges to centroids
        centroids = (bin_edges[:-1] + bin_edges[1:]) / 2

        logging.info(
            f"Fourier mass axis: {len(centroids)} bins, {self.fourier_bin_width_da} Da constant width"
        )
        return centroids

    def _create_quadrupole_mass_axis(
        self, min_mz: float, max_mz: float
    ) -> NDArray[np.float64]:
        """Create mass axis for quadrupole with constant step size."""

        # Universal n_bins override takes priority
        if self.n_bins is not None:
            return self._create_axis_from_n_bins(min_mz, max_mz, self.n_bins, "linear")

        # Alternative: specify number of bins
        if self.quadrupole_n_bins is not None:
            return self._create_axis_from_n_bins(
                min_mz, max_mz, self.quadrupole_n_bins, "linear"
            )

        # Generate linear bins
        n_bins = int((max_mz - min_mz) / self.quadrupole_step_size_da) + 1
        bin_edges = np.linspace(min_mz, max_mz, n_bins + 1)

        # Convert edges to centroids
        centroids = (bin_edges[:-1] + bin_edges[1:]) / 2

        logging.info(
            f"Quadrupole mass axis: {len(centroids)} bins, {self.quadrupole_step_size_da} Da steps"
        )
        return centroids

    def _create_linear_mass_axis(
        self, min_mz: float, max_mz: float
    ) -> NDArray[np.float64]:
        """Create linear mass axis (fallback/default)."""
        # Universal n_bins override takes priority
        if self.n_bins is not None:
            return self._create_axis_from_n_bins(min_mz, max_mz, self.n_bins, "linear")

        return self._create_axis_from_n_bins(
            min_mz, max_mz, self.default_n_bins, "linear"
        )

    def _create_axis_from_n_bins(
        self, min_mz: float, max_mz: float, n_bins: int, spacing_type: str
    ) -> NDArray[np.float64]:
        """Create mass axis from specified number of bins."""
        if spacing_type == "linear":
            bin_edges = np.linspace(min_mz, max_mz, n_bins + 1)
            centroids = (bin_edges[:-1] + bin_edges[1:]) / 2
            logging.info(f"Linear mass axis: {len(centroids)} bins")
            return centroids
        elif spacing_type == "tof":
            # Variable spacing for TOF with specified number of bins
            # Use logarithmic-like distribution
            # This is more complex - for now, fall back to linear
            bin_edges = np.linspace(min_mz, max_mz, n_bins + 1)
            centroids = (bin_edges[:-1] + bin_edges[1:]) / 2
            logging.info(f"TOF mass axis: {len(centroids)} bins (linear approximation)")
            return centroids
        else:
            raise ValueError(f"Unknown spacing type: {spacing_type}")

    def _detect_mass_analyzer_type(self, metadata=None) -> str:
        """Detect mass analyzer type from metadata or heuristics."""
        # For now, this method is placeholder - analyzer type is set manually
        # Will be enhanced when metadata structure is updated

        # Future implementation will check:
        # - metadata.instrument_info.mass_analyzer
        # - ImzML cvParam values for analyzer type
        # - Bruker instrument configuration
        # - File format heuristics

        logging.debug("Analyzer detection not yet implemented - using manual setting")
        return "tof"  # Default fallback

    def _scan_for_mass_range(self) -> Tuple[float, float]:
        """Simple full scan for datasets without metadata (typically small)."""
        min_mz, max_mz = float("inf"), float("-inf")

        # Direct iteration - fast enough for small datasets
        for coords, mzs, intensities in self.reader.iter_spectra():
            if len(mzs) > 0:
                min_mz = min(min_mz, np.min(mzs))
                max_mz = max(max_mz, np.max(mzs))

        logging.info(f"Mass range from scan: {min_mz:.4f} - {max_mz:.4f} m/z")
        return min_mz, max_mz

    def _create_sparse_matrix(self) -> sparse.lil_matrix:
        """Create sparse matrix for storing intensity values.

        Returns:
            Sparse matrix for storing intensity values (or None for chunked processing)

        Raises:
            ValueError: If dimensions or common mass axis are not initialized
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        n_x, n_y, n_z = self._dimensions
        n_pixels = n_x * n_y * n_z
        n_masses = len(self._common_mass_axis)

        # Check processing mode
        self._use_dask_processing = self._should_use_dask_processing()
        self._use_chunked_processing = self._should_use_chunked_processing()

        if self._use_dask_processing:
            logging.info(
                f"Using Dask streaming for {n_pixels} pixels and {n_masses} mass values"
            )
            return None  # No upfront sparse matrix allocation for Dask
        elif self._use_chunked_processing:
            logging.info(
                f"Using chunked processing for {n_pixels} pixels and {n_masses} mass values"
            )
            return None  # No upfront sparse matrix allocation
        else:
            logging.info(
                f"Creating sparse matrix with {n_pixels} pixels and {n_masses} mass values"
            )
            return sparse.lil_matrix((n_pixels, n_masses), dtype=np.float64)

    def _create_coordinates_dataframe(self) -> pd.DataFrame:
        """Create coordinates dataframe with pixel positions.

        Returns:
            DataFrame with pixel coordinates

        Raises:
            ValueError: If dimensions are not initialized
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_x, n_y, n_z = self._dimensions

        # Pre-allocate arrays for better performance
        coords_data = []

        pixel_idx = 0
        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    coords_data.append(
                        {
                            "x": x,
                            "y": y,
                            "z": z if n_z > 1 else 0,
                            "instance_id": str(pixel_idx),
                            "region": f"{self.dataset_id}_pixels",
                            "spatial_x": x * self.pixel_size_um,
                            "spatial_y": y * self.pixel_size_um,
                            "spatial_z": (z * self.pixel_size_um if n_z > 1 else 0.0),
                        }
                    )
                    pixel_idx += 1

        coords_df = pd.DataFrame(coords_data)
        coords_df.set_index("instance_id", inplace=True)
        return coords_df

    def _create_mass_dataframe(self) -> pd.DataFrame:
        """Create m/z dataframe for variable metadata.

        Returns:
            DataFrame with m/z values

        Raises:
            ValueError: If common mass axis is not initialized
        """
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        return pd.DataFrame(
            {"mz": self._common_mass_axis},
            index=[f"mz_{i}" for i in range(len(self._common_mass_axis))],
        )

    def _get_pixel_index(self, x: int, y: int, z: int) -> int:
        """Calculate linear pixel index from 3D coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            Linear pixel index

        Raises:
            ValueError: If dimensions are not initialized
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_x, n_y, _ = self._dimensions
        return z * (n_x * n_y) + y * n_x + x

    def _add_to_sparse_matrix(
        self,
        sparse_matrix: sparse.lil_matrix,
        pixel_idx: int,
        mz_indices: NDArray[np.int_],
        intensities: NDArray[np.float64],
    ) -> None:
        """Add intensity data to sparse matrix.

        Args:
            sparse_matrix: Sparse matrix to update
            pixel_idx: Linear pixel index
            mz_indices: Indices for mass values
            intensities: Intensity values to add
        """
        # Only add if we have valid indices and matching array sizes
        if len(mz_indices) > 0 and len(mz_indices) == len(intensities):
            sparse_matrix[pixel_idx, mz_indices] = intensities

    def _process_chunked(self, data_structures: Dict[str, Any]) -> None:
        """Process dataset using chunked approach for memory efficiency."""
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_x, n_y, n_z = self._dimensions
        total_pixels = n_x * n_y * n_z
        chunk_size = self._determine_chunk_size()

        # Initialize temporary storage for chunked processing
        temp_data = self._initialize_chunked_storage(data_structures)

        # Process pixels in chunks
        with tqdm(total=total_pixels, desc="Processing chunks") as pbar:
            for chunk_start in range(0, total_pixels, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_pixels)
                self._process_pixel_chunk(temp_data, chunk_start, chunk_end)
                pbar.update(chunk_end - chunk_start)

        # Finalize chunked data into SpatialData structures
        self._finalize_chunked_data(data_structures, temp_data)

    def _initialize_chunked_storage(
        self, data_structures: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize temporary storage for chunked processing."""
        # Just pass through the data_structures and let specific converters handle initialization
        logging.info("Initialized chunked storage - delegating to specific converter")
        return data_structures

    def _process_pixel_chunk(
        self, temp_data: Dict[str, Any], start_idx: int, end_idx: int
    ) -> None:
        """Process a chunk of pixels and accumulate data."""
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_x, n_y, n_z = self._dimensions

        for pixel_idx in range(start_idx, end_idx):
            # Convert linear index to 3D coordinates
            z = pixel_idx // (n_x * n_y)
            y = (pixel_idx % (n_x * n_y)) // n_x
            x = pixel_idx % n_x
            coords = (x, y, z)

            # Get spectrum data
            try:
                mzs, intensities = self.reader.get_spectrum(x, y, z)
                if mzs is not None and intensities is not None and len(intensities) > 0:
                    self._non_empty_pixel_count += 1

                    # Process spectrum using existing converter logic
                    self._process_single_spectrum(temp_data, coords, mzs, intensities)

            except Exception as e:
                logging.warning(f"Error processing pixel ({x}, {y}, {z}): {e}")
                continue

    def _finalize_chunked_data(
        self, data_structures: Dict[str, Any], temp_data: Dict[str, Any]
    ) -> None:
        """Convert temporary chunked data into final SpatialData structures."""
        # Let the specific converter handle finalization
        self._finalize_data(data_structures)
        logging.info(
            f"Finalized chunked data: {self._non_empty_pixel_count} non-empty pixels"
        )

    # Dask Streaming Methods

    def _process_with_dask(self, data_structures: Dict[str, Any]) -> None:
        """Process dataset using Dask streaming for true out-of-core processing."""
        if not DASK_AVAILABLE:
            raise RuntimeError(
                "Dask is not available but Dask processing was requested"
            )

        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        # Initialize streaming Zarr writer if enabled
        if self.enable_streaming:
            self._zarr_writer = create_incremental_zarr_writer(
                output_path=self.output_path,
                dimensions=self._dimensions,
                n_masses=len(self._common_mass_axis),
                config=self.zarr_config,
            )
            logging.info("Initialized incremental Zarr writer for streaming")

        logging.info(
            f"Starting Dask {'streaming' if self.enable_streaming else 'standard'} processing..."
        )

        # Configure Dask for memory management
        with dask.config.set({"array.chunk-size": self.dask_memory_limit}):
            # Create Dask pipeline for streaming processing
            pipeline = self._create_dask_pipeline(data_structures)

            # Execute pipeline with progress tracking
            result = pipeline.compute()

            # Process result
            self._process_dask_result(data_structures, result)

            # Clean up Zarr writer
            if self._zarr_writer is not None:
                self._zarr_writer.close()
                self._zarr_writer = None

        logging.info(
            f"Completed Dask streaming processing: {self._non_empty_pixel_count} non-empty pixels"
        )

    def _create_dask_pipeline(self, data_structures: Dict[str, Any]):
        """Create Dask computation graph using sequential iter_spectra() for optimal performance."""
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        if self._interpolator is None:
            raise ValueError("Interpolator is not initialized")

        # Check if this is a Bruker reader (which has SQLite threading issues)
        reader_type = type(self.reader).__name__
        is_bruker = reader_type == "BrukerReader"

        # Determine optimal chunk size for Dask
        dask_chunk_size = self._determine_dask_chunk_size()

        if is_bruker:
            logging.info(
                "Using single Dask worker for Bruker data to maintain SQLite thread consistency"
            )
            # For Bruker, use single worker that does both reading and interpolation in same thread
            processed_result = self._create_sequential_processing_task_single_threaded()
        else:
            # For other readers (like ImzML), use optimized sequential processing
            processed_result = self._create_sequential_processing_task(dask_chunk_size)

        if self.enable_streaming and self._zarr_writer is not None:
            # Streaming mode: process and write to Zarr incrementally
            streaming_result = self._create_streaming_pipeline([processed_result])
            return streaming_result
        else:
            # Standard mode: return the processed result
            return processed_result

    def _determine_dask_chunk_size(self) -> int:
        """Determine optimal chunk size for Dask processing."""
        if self.dask_chunk_size is not None:
            return self.dask_chunk_size

        if self._dimensions is None or self._common_mass_axis is None:
            return 500  # default fallback

        # Calculate chunk size to keep memory per chunk reasonable
        n_masses = len(self._common_mass_axis)

        # Target ~100MB per chunk for sparse data
        target_memory_mb = 100
        sparsity_factor = 0.1
        memory_per_pixel_mb = (n_masses * sparsity_factor * 8) / (
            1024 * 1024
        )  # float64

        optimal_chunk_size = int(target_memory_mb / memory_per_pixel_mb)
        chunk_size = max(
            50, min(optimal_chunk_size, 500)
        )  # reasonable bounds, smaller max for better performance

        logging.info(f"Determined Dask chunk size: {chunk_size} pixels")
        return chunk_size

    def _create_pixel_chunk_task(self, start_idx: int, end_idx: int):
        """Create a delayed task for processing a chunk of pixels with combined read+interpolate."""
        if not DASK_AVAILABLE:
            raise RuntimeError("Dask is not available")
        if self._interpolator is None:
            raise ValueError("Interpolator is not initialized")

        @delayed
        def _delayed_combined_chunk_task():
            if self._dimensions is None:
                raise ValueError("Dimensions are not initialized")

            n_x, n_y, n_z = self._dimensions
            results = []

            for pixel_idx in range(start_idx, end_idx):
                # Convert linear index to 3D coordinates
                z = pixel_idx // (n_x * n_y)
                y = (pixel_idx % (n_x * n_y)) // n_x
                x = pixel_idx % n_x
                coords = (x, y, z)

                try:
                    mzs, intensities = self.reader.get_spectrum(x, y, z)
                    if (
                        mzs is not None
                        and intensities is not None
                        and len(intensities) > 0
                    ):
                        # Direct interpolation without intermediate dictionaries
                        result = self._interpolator.interpolate_spectrum(
                            mzs, intensities, coords, pixel_idx
                        )
                        results.append(result)

                except Exception as e:
                    logging.warning(f"Error processing pixel ({x}, {y}, {z}): {e}")
                    continue

            return results

        return _delayed_combined_chunk_task()

    def _create_sequential_processing_task(self, chunk_size: int):
        """Create a single delayed task that processes all spectra sequentially using iter_spectra()."""
        if not DASK_AVAILABLE:
            raise RuntimeError("Dask is not available")
        if self._interpolator is None:
            raise ValueError("Interpolator is not initialized")

        @delayed
        def _delayed_sequential_task():
            results = []
            pixel_idx = 0

            # Use the reader's optimized sequential iteration
            for coords, mzs, intensities in self.reader.iter_spectra():
                if mzs is not None and intensities is not None and len(intensities) > 0:
                    try:
                        # Direct interpolation without intermediate data structures
                        result = self._interpolator.interpolate_spectrum(
                            mzs, intensities, coords, pixel_idx
                        )
                        results.append(result)

                    except Exception as e:
                        logging.warning(f"Error processing pixel {coords}: {e}")
                        continue

                pixel_idx += 1

            logging.info(
                f"Sequential processing completed: {len(results)} interpolated pixels"
            )
            return results

        return _delayed_sequential_task()

    def _create_sequential_processing_task_single_threaded(self):
        """Create a single delayed task that processes all spectra in one Dask worker (for SQLite safety)."""
        if not DASK_AVAILABLE:
            raise RuntimeError("Dask is not available")
        if self._interpolator is None:
            raise ValueError("Interpolator is not initialized")

        @delayed
        def _delayed_single_worker_task():
            results = []
            pixel_idx = 0

            # All SQLite operations and interpolation happen in the same Dask worker thread
            # This ensures SQLite connections are not shared across threads
            for coords, mzs, intensities in self.reader.iter_spectra():
                if mzs is not None and intensities is not None and len(intensities) > 0:
                    try:
                        # Direct interpolation - both reading and interpolation in same worker
                        result = self._interpolator.interpolate_spectrum(
                            mzs, intensities, coords, pixel_idx
                        )
                        results.append(result)

                    except Exception as e:
                        logging.warning(f"Error processing pixel {coords}: {e}")
                        continue

                pixel_idx += 1

            logging.info(
                f"Single-worker processing completed: {len(results)} interpolated pixels"
            )
            return results

        return _delayed_single_worker_task()

    def _process_dask_chunk(self, pixel_chunk):
        """Process chunk of pixels with interpolation using the interpolation module."""
        if not DASK_AVAILABLE:
            raise RuntimeError("Dask is not available")
        if self._interpolator is None:
            raise ValueError("Interpolator is not initialized")

        # Use the interpolation module to create a delayed task
        return self._interpolator.create_dask_interpolation_task(pixel_chunk)

    def _combine_dask_results(self, processed_chunks):
        """Combine processed chunks into final format."""
        if not DASK_AVAILABLE:
            raise RuntimeError("Dask is not available")

        @delayed
        def _delayed_combine_results(*chunks):
            all_processed_data = []
            for chunk in chunks:
                # chunk is now a List[InterpolationResult] from the interpolation module
                all_processed_data.extend(chunk)

            logging.info(
                f"Combined {len(all_processed_data)} processed pixels from Dask chunks"
            )
            return all_processed_data

        return _delayed_combine_results(*processed_chunks)

    def _create_streaming_pipeline(self, processed_chunks):
        """Create streaming pipeline that writes directly to Zarr."""
        if not DASK_AVAILABLE:
            raise RuntimeError("Dask is not available")
        if self._zarr_writer is None:
            raise ValueError("Zarr writer is not initialized")

        @delayed
        def _delayed_streaming_process(*chunks):
            """Process chunks and write to Zarr incrementally."""
            total_pixels_processed = 0

            for chunk_results in chunks:
                # Write chunk to Zarr
                self._zarr_writer.write_interpolation_results(chunk_results)
                total_pixels_processed += len(chunk_results)

                # Log progress
                if total_pixels_processed % 1000 == 0:
                    progress = self._zarr_writer.get_progress()
                    logging.info(
                        f"Streaming progress: {progress['progress_percent']}% "
                        f"({progress['pixels_written']}/{progress['total_pixels']} pixels)"
                    )

            # Finalize and get data for final processing
            zarr_data = self._zarr_writer.finalize()

            logging.info(
                f"Streaming completed: {zarr_data['pixels_written']} pixels written, "
                f"{zarr_data['sparse_nnz']} sparse elements"
            )

            return zarr_data

        return _delayed_streaming_process(*processed_chunks)

    def _process_dask_result(self, data_structures: Dict[str, Any], result) -> None:
        """Process the result from Dask pipeline and update data structures."""
        if (
            self.enable_streaming
            and isinstance(result, dict)
            and "sparse_matrix" in result
        ):
            # Streaming mode: result is finalized Zarr data
            logging.info(
                f"Processing streaming Dask result: {result['pixels_written']} pixels, "
                f"{result['sparse_nnz']} sparse elements"
            )
            self._non_empty_pixel_count = result["pixels_written"]

            # Update data structures with finalized data from Zarr
            self._process_streaming_result(data_structures, result)
        else:
            # Standard mode: result is list of InterpolationResult objects
            logging.info(f"Processing standard Dask result with {len(result)} pixels")
            self._non_empty_pixel_count = len(result)

            # Let the specific converter handle the result
            self._process_dask_result_specific(data_structures, result)

    def _process_streaming_result(
        self, data_structures: Dict[str, Any], zarr_data: Dict[str, Any]
    ) -> None:
        """Process streaming result from Zarr writer."""
        # Update data structures with pre-computed sparse matrix and arrays
        data_structures["sparse_data"] = zarr_data["sparse_matrix"]
        data_structures["total_intensity"] = zarr_data["total_intensity"]
        data_structures["pixel_count"] = zarr_data["pixels_written"]

        # Handle TIC values based on data structure format
        if "tic_values" in data_structures:
            # For formats that expect shaped TIC values
            if self._dimensions is not None:
                n_x, n_y, n_z = self._dimensions
                if n_z == 1:
                    # 2D data
                    data_structures["tic_values"] = zarr_data["tic_values"].reshape(
                        n_y, n_x
                    )
                else:
                    # 3D data
                    data_structures["tic_values"] = zarr_data["tic_values"].reshape(
                        n_y, n_x, n_z
                    )

        logging.info("Updated data structures with streaming Zarr result")

    def _process_dask_result_specific(
        self, data_structures: Dict[str, Any], result
    ) -> None:
        """Process Dask result for specific converter type - to be overridden."""
        # Default implementation converts InterpolationResult to chunked format for compatibility
        for interpolation_result in result:
            if isinstance(interpolation_result, InterpolationResult):
                # Extract data from InterpolationResult
                coords = interpolation_result.coords
                sparse_indices = interpolation_result.sparse_indices
                sparse_values = interpolation_result.sparse_values

                # Convert sparse indices back to m/z values for compatibility
                if len(sparse_indices) > 0:
                    mzs = self._common_mass_axis[sparse_indices]
                    self._process_single_spectrum(
                        data_structures, coords, mzs, sparse_values
                    )
            else:
                # Legacy format support
                self._process_single_spectrum(
                    data_structures,
                    interpolation_result["coords"],
                    self._common_mass_axis[interpolation_result["sparse_indices"]],
                    interpolation_result["sparse_values"],
                )

    def _process_spectra(self, data_structures: Dict[str, Any]) -> None:
        """Process all spectra with routing to appropriate processing method."""
        # Determine processing mode
        self._use_dask_processing = self._should_use_dask_processing()
        self._use_chunked_processing = self._should_use_chunked_processing()

        if self._use_dask_processing:
            logging.info("Using Dask streaming processing")
            self._process_with_dask(data_structures)
        elif self._use_chunked_processing:
            logging.info("Using chunked processing")
            self._process_chunked(data_structures)
        else:
            logging.info("Using standard processing")
            # Call parent implementation for standard processing
            super()._process_spectra(data_structures)

    def _process_chunked(self, data_structures: Dict[str, Any]) -> None:
        """Process spectra in chunks to manage memory usage."""
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        total_spectra = self._get_total_spectra_count()
        chunk_size = self._determine_chunk_size()

        logging.info(f"Processing {total_spectra} spectra in chunks of {chunk_size}")

        # Prepare reader
        setattr(self.reader, "_quiet_mode", True)

        # Process in chunks
        processed_count = 0
        chunk_data = []

        with tqdm(
            total=total_spectra, desc="Converting spectra (chunked)", unit="spectrum"
        ) as pbar:
            for coords, mzs, intensities in self.reader.iter_spectra(batch_size=1):
                # Zero out negative intensities
                intensities = np.maximum(intensities, 0.0)

                # Add to current chunk
                chunk_data.append(
                    {"coords": coords, "mzs": mzs, "intensities": intensities}
                )

                processed_count += 1
                pbar.update(1)

                # Process chunk when full or at end
                if len(chunk_data) >= chunk_size or processed_count >= total_spectra:
                    self._process_chunk(data_structures, chunk_data)
                    chunk_data = []  # Clear chunk data to free memory

                    # Force garbage collection every few chunks
                    if processed_count % (chunk_size * 10) == 0:
                        import gc

                        gc.collect()

    def _process_chunk(
        self, data_structures: Dict[str, Any], chunk_data: List[Dict[str, Any]]
    ) -> None:
        """Process a single chunk of spectra."""
        if not chunk_data:
            return

        # Use interpolator for efficient processing
        if self._interpolator is None:
            raise ValueError("Interpolator not initialized")

        # Process each spectrum in the chunk
        for pixel_data in chunk_data:
            coords = pixel_data["coords"]
            mzs = pixel_data["mzs"]
            intensities = pixel_data["intensities"]

            # Calculate pixel index
            if len(self._dimensions) == 3:
                x, y, z = coords
                pixel_idx = (
                    z * (self._dimensions[0] * self._dimensions[1])
                    + y * self._dimensions[0]
                    + x
                )
            else:
                x, y = coords[:2]
                pixel_idx = y * self._dimensions[0] + x

            # Interpolate spectrum
            result = self._interpolator.interpolate_spectrum(
                mzs, intensities, coords, pixel_idx
            )

            # Process result using format-specific logic
            self._process_interpolation_result(data_structures, result)

    def _process_interpolation_result(
        self, data_structures: Dict[str, Any], result
    ) -> None:
        """Process a single interpolation result into data structures."""
        # Process directly without the overhead of Dask result processing
        coords = result.coords
        sparse_indices = result.sparse_indices
        sparse_values = result.sparse_values
        tic_value = result.tic_value

        # Update total intensity for average spectrum
        if len(sparse_values) > 0:
            data_structures["total_intensity"][sparse_indices] += sparse_values
            data_structures["pixel_count"] += 1

        # Format-specific processing based on converter type
        self._add_interpolation_result_to_data_structures(data_structures, result)

    def _map_mass_to_indices(self, mzs: NDArray[np.float64]) -> NDArray[np.int_]:
        """Map m/z values to indices in the common mass axis for binned data.

        Override the base method to handle binned mass axes properly.
        For binned mass axes, we need to map each m/z to its nearest bin.
        """
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized.")

        if mzs.size == 0:
            return np.array([], dtype=int)

        # Use searchsorted for finding insertion points
        indices = np.searchsorted(self._common_mass_axis, mzs)

        # For binned data, use nearest neighbor mapping
        # Check both left and right neighbors and pick the closest
        left_indices = np.clip(indices - 1, 0, len(self._common_mass_axis) - 1)
        right_indices = np.clip(indices, 0, len(self._common_mass_axis) - 1)

        # Calculate distances to left and right bins
        left_distances = np.abs(mzs - self._common_mass_axis[left_indices])
        right_distances = np.abs(mzs - self._common_mass_axis[right_indices])

        # Choose the closer bin for each m/z value
        final_indices = np.where(
            left_distances <= right_distances, left_indices, right_indices
        )

        return final_indices.astype(np.int_)

    def _add_interpolation_result_to_data_structures(
        self, data_structures: Dict[str, Any], result
    ) -> None:
        """Add interpolation result to format-specific data structures."""
        # Default implementation - to be overridden by subclasses
        pass

    def _create_pixel_shapes(
        self, adata: AnnData, is_3d: bool = False
    ) -> "ShapesModel":
        """Create geometric shapes for pixels with proper transformations.

        Args:
            adata: AnnData object containing coordinates
            is_3d: Whether to handle as 3D data

        Returns:
            SpatialData shapes model

        Raises:
            ImportError: If required SpatialData dependencies are not available
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")

        # Extract coordinates directly from obs
        x_coords: NDArray[np.float64] = adata.obs["spatial_x"].values
        y_coords: NDArray[np.float64] = adata.obs["spatial_y"].values

        # Create geometries efficiently - this loop could be optimized but kept for clarity
        half_pixel = self.pixel_size_um / 2
        geometries = []

        for i in range(len(adata)):
            x, y = x_coords[i], y_coords[i]
            pixel_box = box(
                x - half_pixel, y - half_pixel, x + half_pixel, y + half_pixel
            )
            geometries.append(pixel_box)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=geometries, index=adata.obs.index)

        # Set up transform
        transform = Identity()
        transformations = {self.dataset_id: transform, "global": transform}

        # Parse shapes
        shapes = ShapesModel.parse(gdf, transformations=transformations)
        return shapes

    def _save_output(self, data_structures: Dict[str, Any]) -> bool:
        """Save the data to SpatialData format.

        Args:
            data_structures: Data structures to save

        Returns:
            True if saving was successful, False otherwise
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")

        try:
            # Create SpatialData object with images included
            sdata = SpatialData(
                tables=data_structures["tables"],
                shapes=data_structures["shapes"],
                images=data_structures["images"],
            )

            # Add metadata
            self.add_metadata(sdata)

            # Write to disk
            sdata.write(str(self.output_path))
            logging.info(f"Successfully saved SpatialData to {self.output_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving SpatialData: {e}")
            import traceback

            logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
            return False

    def add_metadata(self, metadata: "SpatialData") -> None:
        """Add comprehensive metadata to the SpatialData object.

        Args:
            metadata: SpatialData object to add metadata to
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        # Call parent to prepare structured metadata
        super().add_metadata(metadata)

        # Get comprehensive metadata object for detailed access
        comprehensive_metadata_obj = self.reader.get_comprehensive_metadata()

        # Setup attributes and add pixel size metadata
        self._setup_spatialdata_attrs(metadata, comprehensive_metadata_obj)

        # Add comprehensive dataset metadata if supported
        self._add_comprehensive_metadata(metadata)

    def _setup_spatialdata_attrs(
        self, metadata: "SpatialData", comprehensive_metadata_obj
    ) -> None:
        """Setup SpatialData attributes with pixel size and metadata."""
        if not hasattr(metadata, "attrs") or metadata.attrs is None:
            metadata.attrs = {}

        logging.info("Adding comprehensive metadata to SpatialData.attrs")

        # Create pixel size attributes
        pixel_size_attrs = self._create_pixel_size_attrs()

        # Add comprehensive metadata sections
        self._add_comprehensive_sections(pixel_size_attrs, comprehensive_metadata_obj)

        # Update SpatialData attributes
        metadata.attrs.update(pixel_size_attrs)

    def _create_pixel_size_attrs(self) -> Dict[str, Any]:
        """Create pixel size and conversion metadata attributes."""
        # Import version dynamically
        try:
            from ... import __version__

            version = __version__
        except ImportError:
            version = "unknown"

        # Base pixel size metadata
        pixel_size_attrs = {
            "pixel_size_x_um": float(self.pixel_size_um),
            "pixel_size_y_um": float(self.pixel_size_um),
            "pixel_size_units": "micrometers",
            "coordinate_system": "physical_micrometers",
            "msi_converter_version": version,
            "conversion_timestamp": pd.Timestamp.now().isoformat(),
        }

        # Add pixel size detection provenance if available
        if self._pixel_size_detection_info is not None:
            pixel_size_attrs["pixel_size_detection_info"] = dict(
                self._pixel_size_detection_info
            )
            logging.info(
                f"Added pixel size detection info: {self._pixel_size_detection_info}"
            )

        # Add conversion metadata
        pixel_size_attrs["msi_dataset_info"] = {
            "dataset_id": self.dataset_id,
            "total_grid_pixels": self._dimensions[0]
            * self._dimensions[1]
            * self._dimensions[2],
            "non_empty_pixels": self._non_empty_pixel_count,
            "dimensions_xyz": list(self._dimensions),
        }

        return pixel_size_attrs

    def _add_comprehensive_sections(
        self, pixel_size_attrs: Dict[str, Any], comprehensive_metadata_obj
    ) -> None:
        """Add comprehensive metadata sections to attributes."""
        if comprehensive_metadata_obj.format_specific:
            pixel_size_attrs["format_specific_metadata"] = (
                comprehensive_metadata_obj.format_specific
            )

        if comprehensive_metadata_obj.acquisition_params:
            pixel_size_attrs["acquisition_parameters"] = (
                comprehensive_metadata_obj.acquisition_params
            )

        if comprehensive_metadata_obj.instrument_info:
            pixel_size_attrs["instrument_information"] = (
                comprehensive_metadata_obj.instrument_info
            )

    def _add_comprehensive_metadata(self, metadata: "SpatialData") -> None:
        """Add comprehensive dataset metadata if SpatialData supports it."""
        if not hasattr(metadata, "metadata"):
            return

        # Start with structured metadata from base class
        metadata_dict = self._structured_metadata.copy()

        # Add SpatialData-specific enhancements
        metadata_dict.update(
            {
                "non_empty_pixels": self._non_empty_pixel_count,
                "spatialdata_specific": {
                    "zarr_compression_level": self.compression_level,
                    "tables_count": len(getattr(metadata, "tables", {})),
                    "shapes_count": len(getattr(metadata, "shapes", {})),
                    "images_count": len(getattr(metadata, "images", {})),
                },
            }
        )

        # Add pixel size detection provenance if available
        if self._pixel_size_detection_info is not None:
            metadata_dict["pixel_size_provenance"] = self._pixel_size_detection_info

        # Add conversion options used
        metadata_dict["conversion_options"] = {
            "handle_3d": self.handle_3d,
            "pixel_size_um": self.pixel_size_um,
            "dataset_id": self.dataset_id,
            **self.options,
        }

        metadata.metadata = metadata_dict

        logging.info(
            f"Comprehensive metadata persisted to SpatialData with "
            f"{len(metadata_dict)} top-level sections"
        )

    @abstractmethod
    def _create_data_structures(self) -> Dict[str, Any]:
        """Create data structures for the specific converter type."""
        pass

    @abstractmethod
    def _process_single_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        """Process a single spectrum for the specific converter type."""
        pass

    @abstractmethod
    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Finalize data structures for the specific converter type."""
        pass
