from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple, Union, Optional, Dict
from os import PathLike
import numpy as np
import logging
from scipy import sparse
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from numpy.typing import NDArray

from .base_reader import BaseMSIReader

class BaseMSIConverter(ABC):
    """
    Base class for MSI data converters with shared functionality.
    Implements common processing steps while allowing format-specific customization.
    """
    
    def __init__(self, reader: BaseMSIReader, output_path: Union[str, Path, PathLike[str]], 
                 dataset_id: str = "msi_dataset",
                 pixel_size_um: float = 1.0,
                 compression_level: int = 5,
                 handle_3d: bool = False,
                 **kwargs: Any):
        self.reader = reader
        self.output_path = Path(output_path)
        self.dataset_id = dataset_id
        self.pixel_size_um = pixel_size_um
        self.compression_level = compression_level
        self.handle_3d = handle_3d
        self.options: Dict[str, Any] = kwargs
        self._common_mass_axis: Optional[NDArray[np.float64]] = None
        self._dimensions: Optional[Tuple[int, int, int]] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._buffer_size = 100000  # Default buffer size for processing spectra
    
    def convert(self) -> bool:
        """
        Template method defining the conversion workflow.
        
        Returns:
        --------
        bool: True if conversion was successful, False otherwise.
        """
        try:
            # 1. Initialize and prepare data
            self._initialize_conversion()
            
            # 2. Create output-specific data structures
            data_structures = self._create_data_structures()
            
            # 3. Process spectra
            self._process_spectra(data_structures)
            
            # 4. Post-process and finalize data
            self._finalize_data(data_structures)
            
            # 5. Save to disk in format-specific way
            success = self._save_output(data_structures)
            
            return success
        except Exception as e:
            logging.error(f"Error during conversion: {e}")
            import traceback
            logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
            return False
        finally:
            self.reader.close()
    
    def _initialize_conversion(self) -> None:
        """Initialize conversion by loading dimensions, mass axis and metadata."""
        logging.info("Initializing conversion...")
        try:
            self._dimensions = self.reader.get_dimensions()
            if any(d <= 0 for d in self._dimensions):
                raise ValueError(f"Invalid dimensions: {self._dimensions}. All dimensions must be positive.")
                
            self._common_mass_axis = self.reader.get_common_mass_axis()
            if len(self._common_mass_axis) == 0:
                raise ValueError("Common mass axis is empty. Cannot proceed with conversion.")
                
            self._metadata = self.reader.get_metadata()
            if self._metadata is None:
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
        logging.info("Processing spectra...")
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized.")
        
        # Calculate total number of spectra for progress tracking
        # total_spectra = int(np.prod(self._dimensions))
        
        # Process spectra with progress tracking
        # with tqdm(desc="Processing spectra", unit="spectrum", total=total_spectra) as pbar:
        for coords, mzs, intensities in self.reader.iter_spectra(batch_size=self._buffer_size):
            self._process_single_spectrum(data_structures, coords, mzs, intensities)
                # pbar.update(1)
    
    def _process_single_spectrum(
        self,
        data_structures: Any,
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64]
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
        
        logging.info(f"Creating sparse matrix for {n_pixels} pixels and {n_masses} mass values")
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
                    coords.append({
                        'z': z,
                        'y': y, 
                        'x': x,
                        'pixel_id': str(pixel_idx)  # Convert to string for compatibility
                    })
        
        coords_df: pd.DataFrame = pd.DataFrame(coords)
        coords_df.set_index('pixel_id', inplace=True)
        
        # Add spatial coordinates
        coords_df['spatial_x'] = coords_df['x'] * self.pixel_size_um
        coords_df['spatial_y'] = coords_df['y'] * self.pixel_size_um
        coords_df['spatial_z'] = coords_df['z'] * self.pixel_size_um
        
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
        var_df: DataFrame = pd.DataFrame({'mz': self._common_mass_axis})
        # Convert to string index for compatibility
        var_df['mz_str'] = var_df['mz'].astype(str)
        var_df.set_index('mz_str', inplace=True)

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

    def _add_to_sparse_matrix(self, sparse_matrix: sparse.lil_matrix, 
                            pixel_idx: int, mz_indices: NDArray[np.int_], 
                            intensities: NDArray[np.float64]) -> None:
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