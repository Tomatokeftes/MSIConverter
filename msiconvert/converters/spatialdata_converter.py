# msiconvert/converters/spatialdata_converter.py (refactored)
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import logging
from numpy.typing import NDArray
from dataclasses import dataclass, field

# Import SpatialData dependencies
try:
    from spatialdata import SpatialData
    from spatialdata.models import ShapesModel, TableModel
    from spatialdata.transformations import Identity
    from shapely.geometry import box
    import geopandas as gpd
    import xarray as xr
    from geopandas import GeoDataFrame
    SPATIALDATA_AVAILABLE = True
except ImportError:
    logging.warning("SpatialData dependencies not available. SpatialDataConverter will not work.")
    SPATIALDATA_AVAILABLE = False

from ..core.base_converter import BaseMSIConverter
from ..core.base_reader import BaseMSIReader
from ..core.registry import register_converter


@dataclass
class MSIDataUnit:
    """Holds all the data for a single processing unit (a slice or a volume)."""
    unit_id: str
    sparse_matrix: sparse.lil_matrix
    coords_df: pd.DataFrame
    tic_values: NDArray[np.float64]
    is_3d: bool
    pixel_count: int = 0
    region_key: str = field(init=False)

    def __post_init__(self):
        self.region_key = f"{self.unit_id}_pixels"


class DataStructureBuilder:
    """Handles creation of data structures for SpatialData format."""
    
    def __init__(self, converter: 'SpatialDataConverter'):
        self.converter = converter
        
    def create_structures(self) -> Dict[str, Any]:
        """Create data structures based on data dimensions and processing mode."""
        tables: Dict[str, Any] = {}
        shapes: Dict[str, Any] = {}
        images: Dict[str, Any] = {}
        
        if self.converter._dimensions is None:
            raise ValueError("Dimensions are not initialized")
            
        n_x, n_y, n_z = self.converter._dimensions
        
        if n_z > 1 and not self.converter.handle_3d:
            return self._create_2d_slice_structures(tables, shapes, images, n_x, n_y, n_z)
        else:
            return self._create_3d_volume_structures(tables, shapes, images, n_x, n_y, n_z)
    
    def _create_2d_slice_structures(
        self, 
        tables: Dict[str, Any], 
        shapes: Dict[str, Any], 
        images: Dict[str, Any],
        n_x: int, 
        n_y: int, 
        n_z: int
    ) -> Dict[str, Any]:
        """Create structures for 3D data treated as 2D slices."""
        slices_data: Dict[str, Dict[str, Any]] = {}
        
        for z in range(n_z):
            slice_id = f"{self.converter.dataset_id}_z{z}"
            slices_data[slice_id] = {
                'sparse_data': self._create_sparse_matrix_for_slice(n_x, n_y),
                'coords_df': self._create_coordinates_dataframe_for_slice(n_x, n_y, z),
                'tic_values': np.zeros((n_y, n_x), dtype=np.float64)
            }
        
        if self.converter._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")
            
        return {
            'mode': '2d_slices',
            'slices_data': slices_data,
            'tables': tables,
            'shapes': shapes,
            'images': images,
            'var_df': self.converter._create_mass_dataframe(),
            'total_intensity': np.zeros(len(self.converter._common_mass_axis), dtype=np.float64),
            'pixel_count': 0,
        }
    
    def _create_3d_volume_structures(
        self, 
        tables: Dict[str, Any], 
        shapes: Dict[str, Any], 
        images: Dict[str, Any],
        n_x: int, 
        n_y: int, 
        n_z: int
    ) -> Dict[str, Any]:
        """Create structures for full 3D dataset or single 2D slice."""
        if self.converter._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")
            
        return {
            'mode': '3d_volume',
            'sparse_data': self.converter._create_sparse_matrix(),
            'coords_df': self.converter._create_coordinates_dataframe(),
            'var_df': self.converter._create_mass_dataframe(),
            'tables': tables,
            'shapes': shapes,
            'images': images,
            'tic_values': np.zeros((n_y, n_x, n_z), dtype=np.float64),
            'total_intensity': np.zeros(len(self.converter._common_mass_axis), dtype=np.float64),
            'pixel_count': 0,
        }
    
    def _create_sparse_matrix_for_slice(self, n_x: int, n_y: int) -> sparse.lil_matrix:
        """Create a sparse matrix for a single Z-slice."""
        if self.converter._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")
            
        n_pixels = n_x * n_y
        n_masses = len(self.converter._common_mass_axis)
        
        logging.info(f"Creating sparse matrix with {n_pixels} pixels and {n_masses} mass values")
        return sparse.lil_matrix((n_pixels, n_masses), dtype=np.float64)
    
    def _create_coordinates_dataframe_for_slice(self, n_x: int, n_y: int, z_value: int) -> pd.DataFrame:
        """Create a coordinates dataframe for a single Z-slice."""
        # Pre-allocate arrays for better performance
        pixel_count = n_x * n_y
        y_values = np.repeat(np.arange(n_y, dtype=np.int32), n_x)
        x_values = np.tile(np.arange(n_x, dtype=np.int32), n_y)
        instance_ids = np.arange(pixel_count, dtype=np.int32)
        
        # Create DataFrame in one operation
        coords_df = pd.DataFrame({
            'y': y_values,
            'x': x_values,
            'instance_id': instance_ids,
            'region': f"{self.converter.dataset_id}_z{z_value}_pixels"
        })
        
        # Set index efficiently
        coords_df['instance_id'] = coords_df['instance_id'].astype(str)
        coords_df.set_index('instance_id', inplace=True)
        
        # Add spatial coordinates in a vectorized operation
        coords_df['spatial_x'] = coords_df['x'] * self.converter.pixel_size_um
        coords_df['spatial_y'] = coords_df['y'] * self.converter.pixel_size_um
        
        return coords_df


class SpatialDataBuilder:
    """Handles building of SpatialData components (tables, shapes, images)."""
    
    def __init__(self, converter: 'SpatialDataConverter'):
        self.converter = converter
        
    def build_anndata(
        self, 
        unit: MSIDataUnit, 
        var_df: pd.DataFrame, 
        avg_spectrum: NDArray[np.float64]
    ) -> AnnData:
        """Builds an AnnData object from a single MSIDataUnit."""
        logging.info(f"Building AnnData for unit: {unit.unit_id}")
        
        # Convert to CSR for efficiency
        adata = AnnData(
            X=unit.sparse_matrix.tocsr(),
            obs=unit.coords_df.copy(),
            var=var_df
        )
        adata.uns['average_spectrum'] = avg_spectrum
        adata.obs['region'] = unit.region_key
        adata.obs['instance_key'] = adata.obs.index.astype(str)
        return adata

    def build_table(self, adata: AnnData) -> 'TableModel':
        """Parses an AnnData object into a SpatialData TableModel."""
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")
            
        from spatialdata.models import TableModel
        return TableModel.parse(
            adata,
            region=adata.obs['region'].iloc[0],
            region_key="region",
            instance_key="instance_key"
        )

    def build_shapes(self, adata: AnnData) -> 'ShapesModel':
        """Creates pixel shapes from the coordinates in an AnnData object."""
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")
            
        x_coords = adata.obs['spatial_x'].values
        y_coords = adata.obs['spatial_y'].values
        half_pixel = self.converter.pixel_size_um / 2.0
        
        geometries = [
            box(x - half_pixel, y - half_pixel, x + half_pixel, y + half_pixel)
            for x, y in zip(x_coords, y_coords)
        ]
        
        gdf = gpd.GeoDataFrame(geometry=geometries, index=adata.obs.index)
        
        transform = Identity()
        transformations = {self.converter.dataset_id: transform, "global": transform}
        
        return ShapesModel.parse(gdf, transformations=transformations)

    def build_tic_image(self, unit: MSIDataUnit, unit_id: str) -> Any:
        """Builds a 2D or 3D TIC ImageModel from an MSIDataUnit."""
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")
            
        transform = Identity()
        
        if unit.is_3d:
            return self._build_3d_tic_image(unit.tic_values, transform)
        else:
            return self._build_2d_tic_image(unit.tic_values, transform, unit_id)
    
    def _build_2d_tic_image(self, tic_values: NDArray[np.float64], transform: 'Identity', unit_id: str) -> Any:
        """Build a 2D TIC image."""
        from spatialdata.models import Image2DModel
        
        y_size, x_size = tic_values.shape
        tic_values_with_channel = tic_values.reshape(1, y_size, x_size)
        
        tic_image = xr.DataArray(
            tic_values_with_channel,
            dims=('c', 'y', 'x'),
            coords={
                'c': [0],
                'y': np.arange(y_size) * self.converter.pixel_size_um,
                'x': np.arange(x_size) * self.converter.pixel_size_um,
            }
        )
        
        return Image2DModel.parse(
            tic_image,
            transformations={unit_id: transform, "global": transform}
        )
    
    def _build_3d_tic_image(self, tic_values: NDArray[np.float64], transform: 'Identity') -> Any:
        """Build a 3D TIC image."""
        # The tic_values array has shape (y, x, z) as created in data structures
        y_size, x_size, z_size = tic_values.shape
        
        # Reshape to add channel dimension: (c, z, y, x)
        # We need to transpose from (y, x, z) to (z, y, x) first
        tic_values_transposed = np.transpose(tic_values, (2, 0, 1))  # Now (z, y, x)
        tic_values_with_channel = tic_values_transposed.reshape(1, z_size, y_size, x_size)
        
        tic_image = xr.DataArray(
            tic_values_with_channel,
            dims=('c', 'z', 'y', 'x'),
            coords={
                'c': [0],
                'z': np.arange(z_size) * self.converter.pixel_size_um,
                'y': np.arange(y_size) * self.converter.pixel_size_um,
                'x': np.arange(x_size) * self.converter.pixel_size_um,
            }
        )
        
        try:
            from spatialdata.models import Image3DModel
            return Image3DModel.parse(
                tic_image,
                transformations={self.converter.dataset_id: transform, "global": transform}
            )
        except (ImportError, AttributeError):
            logging.warning("Image3DModel not available, using generic image model")
            from spatialdata.models import ImageModel
            return ImageModel.parse(
                tic_image,
                transformations={self.converter.dataset_id: transform, "global": transform}
            )


class DataProcessor:
    """Handles processing of spectra and finalization of data structures."""
    
    def __init__(self, converter: 'SpatialDataConverter'):
        self.converter = converter
        self.spatial_builder = SpatialDataBuilder(converter)
        
    def process_single_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64]
    ) -> None:
        """Process a single spectrum based on the data mode."""
        x, y, z = coords
        
        # Update total intensity for average spectrum
        mz_indices = self.converter._map_mass_to_indices(mzs)
        for idx, intensity in zip(mz_indices, intensities):
            data_structures['total_intensity'][idx] += intensity
        
        if data_structures['mode'] == '2d_slices':
            self._process_2d_slice_spectrum(data_structures, coords, mzs, intensities)
        else:
            self._process_3d_volume_spectrum(data_structures, coords, mzs, intensities)
            
        # Increment pixel count only if we have non-zero intensities
        if np.sum(intensities) > 0:
            data_structures['pixel_count'] += 1
    
    def _process_2d_slice_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64]
    ) -> None:
        """Process spectrum for 2D slice mode."""
        x, y, z = coords
        slice_id = f"{self.converter.dataset_id}_z{z}"
        slice_data = data_structures['slices_data'][slice_id]
        
        # Get pixel index for 2D slice
        if self.converter._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        n_x = self.converter._dimensions[0]
        pixel_idx = y * n_x + x
        
        # Update TIC
        tic_value = float(np.sum(intensities))
        slice_data['tic_values'][y, x] = tic_value
        
        # Update sparse matrix
        mz_indices = self.converter._map_mass_to_indices(mzs)
        self.converter._add_to_sparse_matrix(slice_data['sparse_data'], pixel_idx, mz_indices, intensities)
    
    def _process_3d_volume_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64]
    ) -> None:
        """Process spectrum for 3D volume mode."""
        x, y, z = coords
        pixel_idx = self.converter._get_pixel_index(x, y, z)
        
        # Update TIC
        tic_value = float(np.sum(intensities))
        data_structures['tic_values'][y, x, z] = tic_value
        
        # Update sparse matrix
        mz_indices = self.converter._map_mass_to_indices(mzs)
        self.converter._add_to_sparse_matrix(data_structures['sparse_data'], pixel_idx, mz_indices, intensities)
    
    def finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Finalize SpatialData structures by creating tables, shapes, and images."""
        # Calculate average mass spectrum
        if data_structures['pixel_count'] > 0:
            avg_spectrum = data_structures['total_intensity'] / data_structures['pixel_count']
        else:
            avg_spectrum = data_structures['total_intensity'].copy()
        
        # Store pixel count for metadata
        self.converter._non_empty_pixel_count = data_structures['pixel_count']
        
        if data_structures['mode'] == '2d_slices':
            self._finalize_2d_slices(data_structures, avg_spectrum)
        else:
            self._finalize_3d_volume(data_structures, avg_spectrum)
    
    def _finalize_2d_slices(self, data_structures: Dict[str, Any], avg_spectrum: NDArray[np.float64]) -> None:
        """Finalize data for 2D slices mode."""
        for slice_id, slice_data in data_structures['slices_data'].items():
            try:
                # Create MSIDataUnit
                unit = MSIDataUnit(
                    unit_id=slice_id,
                    sparse_matrix=slice_data['sparse_data'],
                    coords_df=slice_data['coords_df'],
                    tic_values=slice_data['tic_values'],
                    is_3d=False
                )
                
                # Build components
                adata = self.spatial_builder.build_anndata(unit, data_structures['var_df'], avg_spectrum)
                table = self.spatial_builder.build_table(adata)
                shapes = self.spatial_builder.build_shapes(adata)
                tic_image = self.spatial_builder.build_tic_image(unit, slice_id)
                
                # Store in data structures
                data_structures['tables'][slice_id] = table
                data_structures['shapes'][unit.region_key] = shapes
                data_structures['images'][f"{slice_id}_tic"] = tic_image
                
            except Exception as e:
                logging.error(f"Error processing slice {slice_id}: {e}")
                import traceback
                logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
                raise
    
    def _finalize_3d_volume(self, data_structures: Dict[str, Any], avg_spectrum: NDArray[np.float64]) -> None:
        """Finalize data for 3D volume mode."""
        try:
            # Convert sparse matrix to CSR format
            data_structures['sparse_data'] = data_structures['sparse_data'].tocsr()
            
            # Create AnnData
            adata = AnnData(
                X=data_structures['sparse_data'],
                obs=data_structures['coords_df'],
                var=data_structures['var_df']
            )
            
            # Add average spectrum to .uns
            adata.uns['average_spectrum'] = data_structures['total_intensity']
            
            # Make sure region column exists and is correct
            region_key = f"{self.converter.dataset_id}_pixels"
            if 'region' not in adata.obs.columns:
                adata.obs['region'] = region_key
            
            # Ensure instance_key is a string column
            adata.obs['instance_key'] = adata.obs.index.astype(str)
            
            # Create table model
            table = self.spatial_builder.build_table(adata)
            
            # Add to tables and create shapes
            data_structures['tables'][self.converter.dataset_id] = table
            data_structures['shapes'][region_key] = self.spatial_builder.build_shapes(adata)
            
            # Create TIC image
            if self.converter.handle_3d:
                # 3D TIC image
                tic_values = data_structures['tic_values']
                # Build 3D image directly here to match original logic
                transform = Identity()
                y_size, x_size, z_size = tic_values.shape
                
                # Reshape to add channel dimension: (c, z, y, x)
                tic_values_transposed = np.transpose(tic_values, (2, 0, 1))  # (z, y, x)
                tic_values_with_channel = tic_values_transposed.reshape(1, z_size, y_size, x_size)
                
                tic_image = xr.DataArray(
                    tic_values_with_channel,
                    dims=('c', 'z', 'y', 'x'),
                    coords={
                        'c': [0],
                        'z': np.arange(z_size) * self.converter.pixel_size_um,
                        'y': np.arange(y_size) * self.converter.pixel_size_um,
                        'x': np.arange(x_size) * self.converter.pixel_size_um,
                    }
                )
                
                # Create Image model for 3D image
                try:
                    from spatialdata.models import Image3DModel
                    data_structures['images'][f"{self.converter.dataset_id}_tic"] = Image3DModel.parse(
                        tic_image,
                        transformations={self.converter.dataset_id: transform, "global": transform}
                    )
                except (ImportError, AttributeError):
                    logging.warning("Image3DModel not available, using generic image model")
                    from spatialdata.models import ImageModel
                    data_structures['images'][f"{self.converter.dataset_id}_tic"] = ImageModel.parse(
                        tic_image,
                        transformations={self.converter.dataset_id: transform, "global": transform}
                    )
            else:
                # 2D TIC image
                if len(data_structures['tic_values'].shape) == 3:
                    # Take first z-plane from 3D array
                    tic_values = data_structures['tic_values'][:, :, 0]
                else:
                    tic_values = data_structures['tic_values']
                
                y_size, x_size = tic_values.shape
                
                # Add channel dimension to make it (c, y, x)
                tic_values_with_channel = tic_values.reshape(1, y_size, x_size)
                
                tic_image = xr.DataArray(
                    tic_values_with_channel,
                    dims=('c', 'y', 'x'),
                    coords={
                        'c': [0],
                        'y': np.arange(y_size) * self.converter.pixel_size_um,
                        'x': np.arange(x_size) * self.converter.pixel_size_um,
                    }
                )
                
                # Create Image2DModel for the TIC image
                transform = Identity()
                from spatialdata.models import Image2DModel
                data_structures['images'][f"{self.converter.dataset_id}_tic"] = Image2DModel.parse(
                    tic_image,
                    transformations={self.converter.dataset_id: transform, "global": transform}
                )
            
        except Exception as e:
            logging.error(f"Error processing 3D volume: {e}")
            import traceback
            logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
            raise


@register_converter('spatialdata')
class SpatialDataConverter(BaseMSIConverter):
    """Converter for MSI data to SpatialData format."""
    
    def __init__(
        self, 
        reader: BaseMSIReader, 
        output_path: Path, 
        dataset_id: str = "msi_dataset",
        pixel_size_um: float = 1.0,
        handle_3d: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the SpatialData converter.
        
        Args:
            reader: MSI data reader
            output_path: Path for output file
            dataset_id: Identifier for the dataset
            pixel_size_um: Size of each pixel in micrometers
            handle_3d: Whether to process as 3D data (True) or 2D slices (False)
            **kwargs: Additional keyword arguments
            
        Raises:
            ImportError: If SpatialData dependencies are not available
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available. Please install required packages.")
            
        super().__init__(
            reader, 
            output_path, 
            dataset_id=dataset_id,
            pixel_size_um=pixel_size_um,
            handle_3d=handle_3d,
            **kwargs
        )
        
        self._non_empty_pixel_count: int = 0
        
        # Initialize helper components
        self._structure_builder = DataStructureBuilder(self)
        self._data_processor = DataProcessor(self)
    
    def _create_data_structures(self) -> Dict[str, Any]:
        """Create data structures for SpatialData format."""
        return self._structure_builder.create_structures()
    
    def _process_single_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64]
    ) -> None:
        """Process a single spectrum."""
        self._data_processor.process_single_spectrum(data_structures, coords, mzs, intensities)
    
    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Finalize SpatialData structures by creating tables, shapes, and images."""
        self._data_processor.finalize_data(data_structures)
    
    def _save_output(self, data_structures: Dict[str, Any]) -> bool:
        """Save the data to SpatialData format."""
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")
            
        try:
            # Create SpatialData object with images included
            sdata = SpatialData(
                tables=data_structures['tables'],
                shapes=data_structures['shapes'],
                images=data_structures['images']
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
    
    def add_metadata(self, sdata: 'SpatialData') -> None:
        """Add metadata to the SpatialData object."""
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        if self._metadata is None:
            raise ValueError("Metadata is not initialized")
            
        # Add dataset metadata if SpatialData supports it
        if hasattr(sdata, 'metadata'):
            sdata.metadata = {
                'dataset_id': self.dataset_id,
                'pixel_size_um': self.pixel_size_um,
                'source': self._metadata.get('source', 'unknown'),
                'msi_metadata': self._metadata,
                'total_grid_pixels': self._dimensions[0] * self._dimensions[1] * self._dimensions[2],
                'non_empty_pixels': self._non_empty_pixel_count
            }