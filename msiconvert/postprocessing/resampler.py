# msiconvert/postprocessing/resampler.py
"""Post-processing resampler for MSI data stored in zarr format."""

import numpy as np
import zarr
import dask.array as da
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from tqdm import tqdm
import shutil
from scipy import sparse
import pandas as pd

from ..resamplers import ResamplingRequest, ResamplingService, StrategyFactory

logger = logging.getLogger(__name__)


class PostProcessResampler:
    """
    Post-processes zarr files to apply resampling to MSI data.
    
    This class reads already-converted data, applies resampling via interpolation,
    and updates the zarr file in-place or creates a new file.
    """
    
    def __init__(self, resampling_params: Dict[str, Any], chunk_size: int = 1000):
        """
        Initialize the post-process resampler.
        
        Parameters
        ----------
        resampling_params : Dict[str, Any]
            Resampling parameters including mode, size, etc.
        chunk_size : int
            Number of spectra to process at once (default: 1000)
        """
        self.resampling_params = resampling_params
        self.chunk_size = chunk_size
        self._resampled_mass_axis: Optional[np.ndarray] = None
        self._resampling_result = None
    
    def process_zarr_file(
        self, 
        input_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        in_place: bool = True
    ) -> bool:
        """
        Process a zarr file to apply resampling.
        
        Parameters
        ----------
        input_path : Union[str, Path]
            Path to the input zarr file
        output_path : Optional[Union[str, Path]]
            Path for output. If None and in_place=False, adds '_resampled' suffix
        in_place : bool
            If True, modifies the input file. If False, creates a copy.
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return False
        
        # Determine output path
        if in_place:
            output_path = input_path
            logger.info(f"Resampling {input_path} in-place")
        else:
            if output_path is None:
                output_path = input_path.parent / f"{input_path.stem}_resampled.zarr"
            else:
                output_path = Path(output_path)
            
            logger.info(f"Resampling {input_path} to {output_path}")
            
            # Copy the zarr file first
            if output_path.exists():
                logger.error(f"Output path already exists: {output_path}")
                return False
            
            logger.info("Copying zarr file...")
            shutil.copytree(input_path, output_path)
        
        try:
            # Open the zarr file
            store = zarr.open(str(output_path), mode='r+')
            
            # Detect format and process accordingly
            if 'tables' in store:
                # SpatialData format
                logger.info("Detected SpatialData format")
                return self._process_spatialdata(store)
            else:
                # Try other formats if implemented
                logger.error("Unsupported zarr format")
                return False
                
        except Exception as e:
            logger.error(f"Error during resampling: {e}")
            if not in_place and output_path.exists():
                # Clean up failed output
                shutil.rmtree(output_path)
            return False
    
    def _process_spatialdata(self, store: zarr.Group) -> bool:
        """
        Process SpatialData format zarr file.
        
        Parameters
        ----------
        store : zarr.Group
            Opened zarr store
            
        Returns
        -------
        bool
            True if successful
        """
        tables_group = store['tables']
        
        # Find all MSI tables (excluding average spectrum tables)
        msi_tables = [key for key in tables_group.keys() if "average" not in key.lower()]
        
        if not msi_tables:
            logger.error("No MSI tables found in zarr file")
            return False
        
        # Process each table
        for table_name in msi_tables:
            logger.info(f"Processing table: {table_name}")
            
            if not self._process_single_table(tables_group, table_name):
                return False
        
        # Update metadata
        self._update_metadata(store)
        
        logger.info("Resampling completed successfully")
        return True
    
    def _process_single_table(self, tables_group: zarr.Group, table_name: str) -> bool:
        """
        Process a single MSI table in the zarr file.
        
        Parameters
        ----------
        tables_group : zarr.Group
            The tables group in the zarr file
        table_name : str
            Name of the table to process
            
        Returns
        -------
        bool
            True if successful
        """
        table_group = tables_group[table_name]
        
        # Load the var dataframe to get m/z values
        var_group = table_group['var']
        old_mz = self._load_var_mz(var_group)
        
        if old_mz is None or len(old_mz) == 0:
            logger.error(f"Could not load m/z values for table {table_name}")
            return False
        
        # Create resampled mass axis
        new_mz = self._create_resampled_mass_axis(np.min(old_mz), np.max(old_mz))
        logger.info(f"Resampling from {len(old_mz)} to {len(new_mz)} m/z values")
        
        # Load intensity data
        X_group = table_group['X']
        
        # Check if it's sparse
        if 'data' in X_group:
            # Sparse matrix
            logger.info("Processing sparse matrix data")
            success = self._process_sparse_matrix(X_group, old_mz, new_mz)
        else:
            # Dense matrix
            logger.info("Processing dense matrix data")
            success = self._process_dense_matrix(X_group, old_mz, new_mz)
        
        if not success:
            return False
        
        # Update var dataframe with new m/z values
        self._update_var_mz(var_group, new_mz)
        
        # Update average spectrum if present
        if 'uns' in table_group and 'average_spectrum' in table_group['uns']:
            self._update_average_spectrum(table_group['uns'], old_mz, new_mz)
        
        return True
    
    def _load_var_mz(self, var_group: zarr.Group) -> Optional[np.ndarray]:
        """Load m/z values from var group."""
        if 'mz' in var_group:
            # Try different possible structures
            if isinstance(var_group['mz'], zarr.Array):
                return var_group['mz'][:]
            elif '0' in var_group['mz']:  # Column-based storage
                return var_group['mz']['0'][:]
        
        # Try to reconstruct from index
        if '_index' in var_group:
            try:
                mz_strings = var_group['_index'][:]
                return np.array([float(mz) for mz in mz_strings])
            except:
                pass
        
        return None
    
    def _create_resampled_mass_axis(self, min_mz: float, max_mz: float) -> np.ndarray:
        """Create the resampled mass axis."""
        if self._resampled_mass_axis is None:
            logger.info(f"Creating resampled mass axis for range [{min_mz:.2f}, {max_mz:.2f}]")
            
            # Create resampling request
            request = ResamplingRequest(
                min_mz=min_mz,
                max_mz=max_mz,
                model_type=self.resampling_params['mode'],
                num_bins=self.resampling_params.get('num_bins'),
                bin_size_mu=self.resampling_params.get('bin_size_mu'),
                reference_mz=self.resampling_params.get('reference_mz', 1000.0)
            )
            
            # Generate bins
            strategy = StrategyFactory.create_strategy(request.model_type)
            service = ResamplingService(strategy)
            self._resampling_result = service.generate_resampled_axis(request)
            
            # Calculate bin centers as new mass axis
            edges = self._resampling_result.bin_edges
            self._resampled_mass_axis = (edges[:-1] + edges[1:]) / 2.0
            
            logger.info(f"Created resampled mass axis with {len(self._resampled_mass_axis)} bins")
        
        return self._resampled_mass_axis
    
    def _process_sparse_matrix(
        self, 
        X_group: zarr.Group, 
        old_mz: np.ndarray, 
        new_mz: np.ndarray
    ) -> bool:
        """
        Process sparse matrix data in chunks.
        
        Parameters
        ----------
        X_group : zarr.Group
            The X data group containing sparse matrix
        old_mz : np.ndarray
            Original m/z values
        new_mz : np.ndarray
            New resampled m/z values
            
        Returns
        -------
        bool
            True if successful
        """
        # Load sparse matrix structure
        data = X_group['data']
        indices = X_group['indices']
        indptr = X_group['indptr']
        
        n_pixels = len(indptr) - 1
        n_old_mz = len(old_mz)
        n_new_mz = len(new_mz)
        
        # Create new sparse matrix arrays
        new_data_list = []
        new_indices_list = []
        new_indptr = [0]
        
        # Process in chunks
        with tqdm(total=n_pixels, desc="Resampling spectra") as pbar:
            for start_idx in range(0, n_pixels, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, n_pixels)
                
                # Process chunk of pixels
                for pixel_idx in range(start_idx, end_idx):
                    # Get spectrum for this pixel
                    start_ptr = indptr[pixel_idx]
                    end_ptr = indptr[pixel_idx + 1]
                    
                    if start_ptr == end_ptr:
                        # Empty spectrum
                        new_indptr.append(new_indptr[-1])
                        pbar.update(1)
                        continue
                    
                    # Extract spectrum
                    pixel_indices = indices[start_ptr:end_ptr]
                    pixel_data = data[start_ptr:end_ptr]
                    
                    # Convert to dense for interpolation
                    dense_spectrum = np.zeros(n_old_mz)
                    dense_spectrum[pixel_indices] = pixel_data
                    
                    # Interpolate
                    new_spectrum = np.interp(new_mz, old_mz, dense_spectrum)
                    
                    # Convert back to sparse (only non-zero values)
                    non_zero_mask = new_spectrum > 0
                    if np.any(non_zero_mask):
                        new_data_list.extend(new_spectrum[non_zero_mask])
                        new_indices_list.extend(np.where(non_zero_mask)[0])
                    
                    new_indptr.append(len(new_data_list))
                    pbar.update(1)
        
        # Update the sparse matrix in zarr
        logger.info("Updating sparse matrix in zarr file...")
        
        # Convert to numpy arrays
        new_data = np.array(new_data_list, dtype=np.float32)
        new_indices = np.array(new_indices_list, dtype=np.int32)
        new_indptr = np.array(new_indptr, dtype=np.int64)
        
        # Update shape attribute
        X_group.attrs['shape'] = [n_pixels, n_new_mz]
        
        # Resize and update arrays
        del X_group['data']
        del X_group['indices']
        del X_group['indptr']
        
        X_group.create_dataset('data', data=new_data, chunks=(10000,))
        X_group.create_dataset('indices', data=new_indices, chunks=(10000,))
        X_group.create_dataset('indptr', data=new_indptr, chunks=(1000,))
        
        return True
    
    def _process_dense_matrix(
        self, 
        X_group: zarr.Group, 
        old_mz: np.ndarray, 
        new_mz: np.ndarray
    ) -> bool:
        """
        Process dense matrix data using Dask.
        
        Parameters
        ----------
        X_group : zarr.Group
            The X data array
        old_mz : np.ndarray
            Original m/z values
        new_mz : np.ndarray
            New resampled m/z values
            
        Returns
        -------
        bool
            True if successful
        """
        # Load as Dask array
        X_dask = da.from_zarr(X_group)
        n_pixels, n_old_mz = X_dask.shape
        n_new_mz = len(new_mz)
        
        # Create output array
        logger.info(f"Creating output array of shape ({n_pixels}, {n_new_mz})")
        
        # Process in chunks
        chunk_results = []
        
        with tqdm(total=n_pixels, desc="Resampling spectra") as pbar:
            for start_idx in range(0, n_pixels, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, n_pixels)
                
                # Load chunk into memory
                chunk = X_dask[start_idx:end_idx].compute()
                
                # Resample each spectrum in the chunk
                resampled_chunk = np.zeros((end_idx - start_idx, n_new_mz))
                for i, spectrum in enumerate(chunk):
                    resampled_chunk[i] = np.interp(new_mz, old_mz, spectrum)
                
                chunk_results.append(resampled_chunk)
                pbar.update(end_idx - start_idx)
        
        # Combine results
        new_data = np.vstack(chunk_results)
        
        # Update zarr array
        logger.info("Updating dense matrix in zarr file...")
        del X_group[:]
        X_group.resize((n_pixels, n_new_mz))
        X_group[:] = new_data
        
        return True
    
    def _update_var_mz(self, var_group: zarr.Group, new_mz: np.ndarray) -> None:
        """Update the var group with new m/z values."""
        # Update mz column
        if 'mz' in var_group:
            if isinstance(var_group['mz'], zarr.Array):
                var_group['mz'].resize(len(new_mz))
                var_group['mz'][:] = new_mz
            elif '0' in var_group['mz']:
                var_group['mz']['0'].resize(len(new_mz))
                var_group['mz']['0'][:] = new_mz
        
        # Update index if needed
        if '_index' in var_group:
            new_index = [str(mz) for mz in new_mz]
            var_group['_index'].resize(len(new_index))
            var_group['_index'][:] = new_index
        
        # Update length attribute
        if 'attrs' in dir(var_group):
            var_group.attrs['length'] = len(new_mz)
    
    def _update_average_spectrum(
        self, 
        uns_group: zarr.Group, 
        old_mz: np.ndarray, 
        new_mz: np.ndarray
    ) -> None:
        """Update the average spectrum with resampled values."""
        if 'average_spectrum' in uns_group:
            old_spectrum = uns_group['average_spectrum'][:]
            new_spectrum = np.interp(new_mz, old_mz, old_spectrum)
            
            uns_group['average_spectrum'].resize(len(new_spectrum))
            uns_group['average_spectrum'][:] = new_spectrum
    
    def _update_metadata(self, store: zarr.Group) -> None:
        """Update metadata to indicate resampling."""
        if 'attrs' not in dir(store):
            return
        
        # Add resampling information
        metadata = dict(store.attrs)
        metadata['resampling'] = {
            'enabled': True,
            'mode': self.resampling_params['mode'],
            'parameters': self.resampling_params,
            'timestamp': str(np.datetime64('now'))
        }
        
        if self._resampling_result:
            metadata['resampling']['final_num_bins'] = self._resampling_result.final_num_bins
            metadata['resampling']['achieved_width_mda'] = (
                self._resampling_result.achieved_width_at_ref_mz_da * 1000
            )
        
        store.attrs.update(metadata) 