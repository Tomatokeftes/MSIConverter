#!/usr/bin/env python3
"""
Test with real MSI interpolation characteristics.
"""

import numpy as np

def test_real_interpolation():
    print("=== REAL MSI INTERPOLATION ANALYSIS ===")
    
    # Real parameters from our Bruker test
    n_pixels = 33800
    n_bins_5k = 5000
    n_bins_300k = 300000
    sparsity_5k = 0.885  # 88.5% from real data
    
    print(f"Real dataset: {n_pixels:,} pixels")
    print(f"Current test (5K bins): {sparsity_5k*100:.1f}% density")
    
    # Memory per pixel for different bin counts
    def calculate_memory(n_bins, sparsity):
        n_nonzero_per_pixel = int(n_bins * sparsity)
        
        # Dense interpolation (temporary)
        dense_memory_per_pixel = n_bins * 8  # 8 bytes per float64
        
        # Final sparse storage
        sparse_memory_per_pixel = n_nonzero_per_pixel * (4 + 8)  # int32 index + float64 value
        
        return dense_memory_per_pixel, sparse_memory_per_pixel, n_nonzero_per_pixel
    
    # 5K bins analysis
    dense_5k, sparse_5k, nonzero_5k = calculate_memory(n_bins_5k, sparsity_5k)
    print(f"\n=== 5K BINS ===")
    print(f"Non-zero values per pixel: {nonzero_5k:,}")
    print(f"Dense memory per pixel: {dense_5k:,} bytes ({dense_5k/1024:.1f} KB)")
    print(f"Sparse memory per pixel: {sparse_5k:,} bytes ({sparse_5k/1024:.1f} KB)")
    print(f"Total dense memory: {(dense_5k * n_pixels)/1024**3:.1f} GB")
    print(f"Total sparse memory: {(sparse_5k * n_pixels)/1024**3:.1f} GB")
    
    # 300K bins analysis (assuming same sparsity ratio)
    dense_300k, sparse_300k, nonzero_300k = calculate_memory(n_bins_300k, sparsity_5k)
    print(f"\n=== 300K BINS (Current approach) ===")
    print(f"Non-zero values per pixel: {nonzero_300k:,}")
    print(f"Dense memory per pixel: {dense_300k:,} bytes ({dense_300k/1024:.1f} KB)")
    print(f"Sparse memory per pixel: {sparse_300k:,} bytes ({sparse_300k/1024:.1f} KB)")
    print(f"Total dense memory: {(dense_300k * n_pixels)/1024**3:.1f} GB")
    print(f"Total sparse memory: {(sparse_300k * n_pixels)/1024**3:.1f} GB")
    print(f"🚨 TEMPORARY MEMORY REQUIRED: {(dense_300k * n_pixels)/1024**3:.1f} GB")
    
    # More realistic sparsity for 300K (peaks don't scale linearly)
    realistic_sparsity_300k = 0.05  # Much more realistic for 300K bins
    dense_300k_real, sparse_300k_real, nonzero_300k_real = calculate_memory(n_bins_300k, realistic_sparsity_300k)
    print(f"\n=== 300K BINS (Realistic 5% sparsity) ===")
    print(f"Non-zero values per pixel: {nonzero_300k_real:,}")
    print(f"Dense memory per pixel: {dense_300k_real:,} bytes ({dense_300k_real/1024:.1f} KB)")
    print(f"Sparse memory per pixel: {sparse_300k_real:,} bytes ({sparse_300k_real/1024:.1f} KB)")
    print(f"Total dense memory: {(dense_300k_real * n_pixels)/1024**3:.1f} GB")
    print(f"Total sparse memory: {(sparse_300k_real * n_pixels)/1024**3:.1f} GB")
    print(f"🚨 STILL TEMPORARY MEMORY REQUIRED: {(dense_300k_real * n_pixels)/1024**3:.1f} GB")

if __name__ == "__main__":
    test_real_interpolation()