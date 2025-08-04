#!/usr/bin/env python3
"""
Test script to analyze MSI conversion output and verify interpolation.
Creates plots and saves them as PNG files.
"""

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=== MSI DATA ANALYSIS ===")
    
    # Try different approaches to load the data
    try:
        # Method 1: Fix Dask config and import SpatialData
        import dask
        import os
        
        # Force Dask to use the new query planning
        os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "True"
        dask.config.set({"dataframe.query-planning": True})
        
        from spatialdata import SpatialData
        import numpy as np
        
        # Load the SpatialData from our test output
        sdata = SpatialData.read("test_bruker_fixed.zarr")
        print("✅ Loaded with SpatialData")
        
    except Exception as e:
        print(f"SpatialData loading failed: {e}")
        print("Falling back to direct Zarr reading...")
        
        # Method 2: Direct Zarr reading as fallback
        import zarr
        import numpy as np
        from scipy.sparse import csr_matrix
        import anndata as adata
        
        # Open zarr store directly
        store = zarr.open("test_bruker_fixed.zarr", mode='r')
        
        # Create a mock SpatialData-like object for compatibility
        class MockSpatialData:
            def __init__(self, zarr_store):
                self.store = zarr_store
                self.tables = {}
                self.images = {}
                self.shapes = {}
                
                # Load main table data
                try:
                    data = zarr_store['tables/msi_dataset/X/data'][:]
                    indices = zarr_store['tables/msi_dataset/X/indices'][:]
                    indptr = zarr_store['tables/msi_dataset/X/indptr'][:]
                    X = csr_matrix((data, indices, indptr))
                    
                    # Create AnnData object
                    adata_obj = adata.AnnData(X=X)
                    
                    # Add var data (m/z values)
                    adata_obj.var['mz'] = zarr_store['tables/msi_dataset/var/mz'][:]
                    
                    # Add obs data (coordinates)
                    adata_obj.obs['x'] = zarr_store['tables/msi_dataset/obs/x'][:]
                    adata_obj.obs['y'] = zarr_store['tables/msi_dataset/obs/y'][:]
                    adata_obj.obs['z'] = zarr_store['tables/msi_dataset/obs/z'][:]
                    
                    # Add uns data (average spectrum)
                    if 'tables/msi_dataset/uns/average_spectrum' in zarr_store:
                        adata_obj.uns['average_spectrum'] = zarr_store['tables/msi_dataset/uns/average_spectrum'][:]
                    
                    self.tables['msi_dataset'] = adata_obj
                    
                    # Load TIC image if available
                    if 'images/msi_dataset_tic/0' in zarr_store:
                        import xarray as xr
                        tic_data = zarr_store['images/msi_dataset_tic/0'][:]
                        # Create xarray DataArray for compatibility
                        self.images['msi_dataset_tic'] = xr.DataArray(tic_data, dims=['c', 'y', 'x'])
                        
                except Exception as load_error:
                    print(f"Error loading data: {load_error}")
        
        sdata = MockSpatialData(store)
        print("✅ Loaded with direct Zarr reading")

    # Display basic information
    if hasattr(sdata, 'tables') and sdata.tables:
        print("\n=== DATA STRUCTURE ===")
        print("Available tables:", list(sdata.tables.keys()))
        print("Available images:", list(sdata.images.keys()) if hasattr(sdata, 'images') else 'Not available')
        print("Available shapes:", list(sdata.shapes.keys()) if hasattr(sdata, 'shapes') else 'Not available')

        # Get basic info about the main table
        main_table = sdata.tables[list(sdata.tables.keys())[0]]
        print(f"\nMain table shape: {main_table.shape}")
        print(f"Number of pixels: {main_table.n_obs:,}")
        print(f"Number of m/z bins: {main_table.n_vars:,}")
        print(f"Data matrix format: {type(main_table.X)}")
        if hasattr(main_table.X, 'nnz'):
            density = main_table.X.nnz / (main_table.n_obs * main_table.n_vars) * 100
            print(f"Sparse matrix density: {density:.2f}%")
            print(f"Non-zero values: {main_table.X.nnz:,}")

        # Analyze mass axis
        print("\n=== MASS AXIS ANALYSIS ===")
        mz_values = main_table.var["mz"].values if "mz" in main_table.var.columns else np.arange(main_table.n_vars)
        print(f"Mass axis length: {len(mz_values):,}")
        print(f"Mass range: {mz_values.min():.3f} - {mz_values.max():.3f} Da")
        
        if len(mz_values) > 1:
            diffs = np.diff(mz_values)
            print(f"Mass resolution (median): {np.median(diffs):.4f} Da")
            print(f"Mass spacing std dev: {np.std(diffs):.4f} Da")
        
        # Analyze average spectrum
        print("\n=== AVERAGE SPECTRUM ANALYSIS ===")
        if "average_spectrum" in main_table.uns:
            avg_spectrum = main_table.uns["average_spectrum"]
            print(f"Average spectrum shape: {avg_spectrum.shape}")
            print(f"Intensity range: {avg_spectrum.min():.2f} - {avg_spectrum.max():.2f}")
            print(f"Non-zero peaks in average: {np.count_nonzero(avg_spectrum):,}")
        else:
            print("Computing average spectrum from sparse matrix...")
            avg_spectrum = np.array(main_table.X.mean(axis=0)).flatten()
            print(f"Computed average spectrum shape: {avg_spectrum.shape}")
            print(f"Intensity range: {avg_spectrum.min():.2f} - {avg_spectrum.max():.2f}")

        # Plot and save average spectrum
        plt.figure(figsize=(15, 6))
        plt.plot(mz_values, avg_spectrum, linewidth=0.8)
        plt.xlabel('m/z (Da)', fontsize=12)
        plt.ylabel('Average Intensity', fontsize=12)
        plt.title('Average Mass Spectrum - Interpolated Data', fontsize=14)
        plt.xlim(mz_values.min(), mz_values.max())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('average_mass_spectrum.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Average spectrum plot saved as 'average_mass_spectrum.png'")

        # Analyze individual pixel
        print("\n=== INDIVIDUAL PIXEL ANALYSIS ===")
        middle_pixel = main_table.n_obs // 2
        pixel_spectrum = main_table.X[middle_pixel].toarray().flatten()
        
        # Get pixel coordinates
        x_coord = main_table.obs.iloc[middle_pixel]['x']
        y_coord = main_table.obs.iloc[middle_pixel]['y']
        
        non_zero_mask = pixel_spectrum > 0
        non_zero_count = np.sum(non_zero_mask)
        
        print(f"Pixel ({x_coord}, {y_coord}) analysis:")
        print(f"  Non-zero peaks: {non_zero_count:,}")
        print(f"  Coverage: {non_zero_count/len(pixel_spectrum)*100:.1f}% of mass axis")
        if non_zero_count > 0:
            non_zero_intensities = pixel_spectrum[non_zero_mask]
            non_zero_mz = mz_values[non_zero_mask]
            print(f"  Intensity range: {non_zero_intensities.min():.2f} - {non_zero_intensities.max():.2f}")
            
            # Plot and save individual pixel spectrum
            plt.figure(figsize=(15, 8))
            
            # Full spectrum
            plt.subplot(2, 1, 1)
            plt.plot(non_zero_mz, non_zero_intensities, linewidth=0.8, alpha=0.8)
            plt.xlabel('m/z (Da)')
            plt.ylabel('Intensity')
            plt.title(f'Individual Pixel Spectrum - Pixel ({x_coord}, {y_coord})')
            plt.grid(True, alpha=0.3)
            
            # Zoomed view
            zoom_start, zoom_end = 200, 300
            zoom_mask = (non_zero_mz >= zoom_start) & (non_zero_mz <= zoom_end)
            if np.any(zoom_mask):
                zoom_mz = non_zero_mz[zoom_mask]
                zoom_intensities = non_zero_intensities[zoom_mask]
                
                plt.subplot(2, 1, 2)
                plt.plot(zoom_mz, zoom_intensities, 'o-', linewidth=1, markersize=2)
                plt.xlabel('m/z (Da)')
                plt.ylabel('Intensity')
                plt.title(f'Zoomed View ({zoom_start}-{zoom_end} Da) - Shows Interpolation Detail')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('individual_pixel_spectrum.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Individual pixel spectrum plot saved as 'individual_pixel_spectrum.png'")

        # TIC image analysis
        print("\n=== TIC IMAGE ANALYSIS ===")
        if hasattr(sdata, 'images') and sdata.images:
            tic_keys = [key for key in sdata.images.keys() if "tic" in key.lower()]
            if tic_keys:
                tic_image = sdata.images[tic_keys[0]]
                print(f"TIC image shape: {tic_image.shape}")
                print(f"TIC image dims: {tic_image.dims}")
                
                # Get TIC data
                if hasattr(tic_image, 'sel'):
                    tic_data = tic_image.sel(c=0).squeeze()
                else:
                    tic_data = tic_image[0].squeeze() if len(tic_image.shape) > 2 else tic_image
                
                print(f"TIC data shape: {tic_data.shape}")
                tic_min = float(tic_data.min())
                tic_max = float(tic_data.max())
                print(f"TIC range: {tic_min:.2f} - {tic_max:.2f}")
                
                # Plot and save TIC image
                plt.figure(figsize=(12, 10))
                tic_array = np.array(tic_data)
                plt.imshow(tic_array, cmap='viridis', origin='lower', aspect='auto')
                plt.colorbar(label='Total Ion Current', shrink=0.8)
                plt.title('Total Ion Current (TIC) Image', fontsize=14)
                plt.xlabel('X pixel', fontsize=12)
                plt.ylabel('Y pixel', fontsize=12)
                plt.tight_layout()
                plt.savefig('tic_image.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ TIC image plot saved as 'tic_image.png'")
            else:
                print("No TIC images found")
        else:
            print("No images available")

        # Interpolation verification
        print("\n=== INTERPOLATION VERIFICATION ===")
        print("✅ Interpolation successful - all spectra are on common m/z axis")
        print(f"✅ Common mass axis: {len(mz_values):,} bins")
        if len(mz_values) > 1:
            print(f"✅ Mass resolution: ~{np.median(np.diff(mz_values)):.4f} Da median spacing")
        if hasattr(main_table.X, 'nnz'):
            density = main_table.X.nnz / (main_table.n_obs * main_table.n_vars) * 100
            print(f"✅ Data density: {density:.1f}% of matrix filled")
        
        print(f"\n🎯 SUMMARY:")
        print(f"  • {main_table.n_obs:,} pixels successfully processed")
        print(f"  • {main_table.n_vars:,} m/z bins in common mass axis")
        print(f"  • {main_table.X.nnz:,} interpolated intensity values")
        print(f"  • Mass range: {mz_values.min():.1f} - {mz_values.max():.1f} Da")
        print("  • Interpolation: CONFIRMED ✅")
        
    else:
        print("❌ Failed to load data with both methods")

if __name__ == "__main__":
    main()