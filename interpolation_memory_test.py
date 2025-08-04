#!/usr/bin/env python3
"""
Test script to analyze interpolation memory usage and sparsity.
"""

import numpy as np
import sys

def test_interpolation_memory():
    print("=== INTERPOLATION MEMORY ANALYSIS ===")
    
    # Simulate typical MSI data
    n_bins_common = 300000  # Your target
    n_peaks_original = 2500  # Typical peaks per pixel
    
    print(f"Common mass axis bins: {n_bins_common:,}")
    print(f"Original peaks per pixel: {n_peaks_original:,}")
    
    # Create synthetic original spectrum (sparse) - more realistic
    np.random.seed(42)
    # Create clusters of peaks (more realistic for MSI data)
    cluster_centers = np.random.uniform(100, 1000, n_peaks_original // 10)
    original_mz = []
    original_intensities = []
    
    for center in cluster_centers:
        # Add 10 peaks around each center
        cluster_mz = np.random.normal(center, 0.5, 10)
        cluster_intensities = np.random.exponential(1000, 10)
        original_mz.extend(cluster_mz)
        original_intensities.extend(cluster_intensities)
    
    original_mz = np.sort(np.array(original_mz))
    original_intensities = np.array(original_intensities)
    
    print(f"Original spectrum memory: {original_mz.nbytes + original_intensities.nbytes:,} bytes")
    
    # Create common mass axis
    common_mass_axis = np.linspace(100, 1000, n_bins_common)
    print(f"Common mass axis memory: {common_mass_axis.nbytes:,} bytes")
    
    # CURRENT APPROACH: Dense interpolation
    print("\n=== CURRENT DENSE INTERPOLATION ===")
    dense_interpolated = np.interp(common_mass_axis, original_mz, original_intensities, left=0.0, right=0.0)
    print(f"Dense interpolated array memory: {dense_interpolated.nbytes:,} bytes ({dense_interpolated.nbytes / 1024 / 1024:.1f} MB)")
    
    # Check sparsity
    sparsity_threshold = 1e-10
    nonzero_mask = dense_interpolated > sparsity_threshold
    n_nonzero = np.sum(nonzero_mask)
    sparsity_percent = (n_nonzero / len(dense_interpolated)) * 100
    
    print(f"Non-zero values after interpolation: {n_nonzero:,}")
    print(f"Sparsity: {sparsity_percent:.2f}% (most values are zero!)")
    
    # Final sparse representation
    sparse_indices = np.where(nonzero_mask)[0]
    sparse_values = dense_interpolated[nonzero_mask]
    sparse_memory = sparse_indices.nbytes + sparse_values.nbytes
    
    print(f"Final sparse representation memory: {sparse_memory:,} bytes")
    print(f"Memory waste ratio: {dense_interpolated.nbytes / sparse_memory:.1f}x")
    
    # Memory usage for full dataset
    n_pixels = 30000
    total_dense_memory_gb = (dense_interpolated.nbytes * n_pixels) / (1024**3)
    total_sparse_memory_gb = (sparse_memory * n_pixels) / (1024**3)
    
    print(f"\n=== FULL DATASET MEMORY ===")
    print(f"Pixels: {n_pixels:,}")
    print(f"Dense interpolation memory: {total_dense_memory_gb:.1f} GB")
    print(f"Final sparse memory: {total_sparse_memory_gb:.1f} GB")
    print(f"Temporary memory waste: {total_dense_memory_gb - total_sparse_memory_gb:.1f} GB")

if __name__ == "__main__":
    test_interpolation_memory()