#!/usr/bin/env python3
"""
Direct comparison of old dense vs new sparse interpolation methods.
Run this to see the memory and accuracy differences yourself.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple
from numpy.typing import NDArray

def old_dense_interpolation(
    common_mass_axis: NDArray[np.float64],
    original_mz: NDArray[np.float64], 
    original_intensities: NDArray[np.float64],
    sparsity_threshold: float = 1e-10
) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """
    OLD METHOD: Create dense array, then convert to sparse (memory hog).
    """
    print("  🔄 Creating dense interpolated array...")
    
    # This creates a HUGE dense array temporarily
    dense_interpolated = np.interp(
        common_mass_axis, original_mz, original_intensities,
        left=0.0, right=0.0
    )
    
    dense_memory_mb = dense_interpolated.nbytes / (1024 * 1024)
    print(f"  🚨 Temporary dense array: {dense_memory_mb:.1f} MB")
    
    # Convert to sparse
    print("  🔄 Converting to sparse...")
    nonzero_mask = dense_interpolated > sparsity_threshold
    sparse_indices = np.where(nonzero_mask)[0].astype(np.int_)
    sparse_values = dense_interpolated[nonzero_mask].astype(np.float64)
    
    return sparse_indices, sparse_values

def new_sparse_interpolation(
    common_mass_axis: NDArray[np.float64],
    original_mz: NDArray[np.float64], 
    original_intensities: NDArray[np.float64],
    sparsity_threshold: float = 1e-10
) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """
    NEW METHOD: Direct sparse-to-sparse interpolation (memory efficient).
    """
    print("  🚀 Direct sparse interpolation (no dense arrays)...")
    
    if len(original_mz) == 0:
        return np.array([], dtype=np.int_), np.array([], dtype=np.float64)
    
    sparse_indices = []
    sparse_values = []
    
    # For each original peak, find which common axis bins it affects
    for mz, intensity in zip(original_mz, original_intensities):
        if intensity <= sparsity_threshold:
            continue
            
        # Find the insertion point in common mass axis
        insert_idx = np.searchsorted(common_mass_axis, mz)
        
        # Handle edge cases
        if insert_idx == 0:
            # Before first bin
            sparse_indices.append(0)
            sparse_values.append(intensity)
        elif insert_idx >= len(common_mass_axis):
            # After last bin
            sparse_indices.append(len(common_mass_axis) - 1)
            sparse_values.append(intensity)
        else:
            # Linear interpolation between two bins
            left_idx = insert_idx - 1
            right_idx = insert_idx
            
            left_mz = common_mass_axis[left_idx]
            right_mz = common_mass_axis[right_idx]
            
            # Calculate interpolation weights
            total_distance = right_mz - left_mz
            if total_distance > 0:
                left_weight = (right_mz - mz) / total_distance
                right_weight = (mz - left_mz) / total_distance
                
                # Add contributions to both bins
                if left_weight > 0:
                    sparse_indices.append(left_idx)
                    sparse_values.append(intensity * left_weight)
                if right_weight > 0:
                    sparse_indices.append(right_idx)
                    sparse_values.append(intensity * right_weight)
            else:
                # Exact match
                sparse_indices.append(left_idx)
                sparse_values.append(intensity)
    
    if not sparse_indices:
        return np.array([], dtype=np.int_), np.array([], dtype=np.float64)
    
    # Combine contributions to same bins
    sparse_indices = np.array(sparse_indices, dtype=np.int_)
    sparse_values = np.array(sparse_values, dtype=np.float64)
    
    # Group by index and sum values
    unique_indices, inverse_indices = np.unique(sparse_indices, return_inverse=True)
    summed_values = np.zeros(len(unique_indices), dtype=np.float64)
    np.add.at(summed_values, inverse_indices, sparse_values)
    
    # Filter out values below threshold
    keep_mask = summed_values > sparsity_threshold
    final_indices = unique_indices[keep_mask]
    final_values = summed_values[keep_mask]
    
    return final_indices, final_values

def run_comparison():
    """Run side-by-side comparison of both methods."""
    print("=" * 60)
    print("🧪 INTERPOLATION METHOD COMPARISON")
    print("=" * 60)
    
    # Test parameters - adjust these to see different scenarios
    n_bins = 50000  # Start smaller to see the effect clearly
    n_original_peaks = 2500
    mass_range = (100, 1000)
    
    print(f"📊 Test Setup:")
    print(f"  Common mass axis bins: {n_bins:,}")
    print(f"  Original peaks per pixel: {n_original_peaks:,}")
    print(f"  Mass range: {mass_range[0]}-{mass_range[1]} Da")
    
    # Create realistic test data
    print(f"\n🔧 Creating test data...")
    np.random.seed(42)
    
    # Create clustered peaks (more realistic)
    cluster_centers = np.random.uniform(mass_range[0], mass_range[1], n_original_peaks // 10)
    original_mz = []
    original_intensities = []
    
    for center in cluster_centers:
        # Add peaks around each center
        cluster_mz = np.random.normal(center, 2.0, 10)  # 2 Da spread
        cluster_mz = np.clip(cluster_mz, mass_range[0], mass_range[1])
        cluster_intensities = np.random.exponential(1000, 10)
        original_mz.extend(cluster_mz)
        original_intensities.extend(cluster_intensities)
    
    original_mz = np.sort(np.array(original_mz))
    original_intensities = np.array(original_intensities)
    
    # Create common mass axis
    common_mass_axis = np.linspace(mass_range[0], mass_range[1], n_bins)
    
    print(f"  ✅ Created {len(original_mz):,} original peaks")
    print(f"  ✅ Created {len(common_mass_axis):,} common mass axis bins")
    
    # Test OLD method
    print(f"\n🔴 OLD DENSE METHOD:")
    start_time = time.time()
    old_indices, old_values = old_dense_interpolation(
        common_mass_axis, original_mz, original_intensities
    )
    old_time = time.time() - start_time
    
    old_memory = (old_indices.nbytes + old_values.nbytes) / 1024
    print(f"  ⏱️  Time: {old_time:.3f} seconds")
    print(f"  📦 Final sparse elements: {len(old_indices):,}")
    print(f"  💾 Final memory: {old_memory:.1f} KB")
    
    # Test NEW method
    print(f"\n🟢 NEW SPARSE METHOD:")
    start_time = time.time()
    new_indices, new_values = new_sparse_interpolation(
        common_mass_axis, original_mz, original_intensities
    )
    new_time = time.time() - start_time
    
    new_memory = (new_indices.nbytes + new_values.nbytes) / 1024
    print(f"  ⏱️  Time: {new_time:.3f} seconds")
    print(f"  📦 Final sparse elements: {len(new_indices):,}")
    print(f"  💾 Final memory: {new_memory:.1f} KB")
    
    # Comparison
    print(f"\n📊 COMPARISON:")
    print(f"  🔥 Sparse elements reduction: {len(old_indices) / len(new_indices):.1f}x")
    print(f"  💾 Memory reduction: {old_memory / new_memory:.1f}x")
    print(f"  ⚡ Speed improvement: {old_time / new_time:.1f}x")
    
    # Quality check - compare total intensity
    old_total = np.sum(old_values)
    new_total = np.sum(new_values)
    intensity_diff = abs(old_total - new_total) / old_total * 100
    print(f"  🎯 Total intensity difference: {intensity_diff:.2f}%")
    
    # Scaling projection
    print(f"\n🚀 SCALING TO 300K BINS:")
    scale_factor = 300000 / n_bins
    
    old_300k_memory_gb = (old_memory * scale_factor * 33800) / (1024 * 1024)  # 33800 pixels
    new_300k_memory_gb = (new_memory * scale_factor * 33800) / (1024 * 1024)
    
    print(f"  Old method @ 300K bins: ~{old_300k_memory_gb:.1f} GB peak memory")
    print(f"  New method @ 300K bins: ~{new_300k_memory_gb:.1f} GB peak memory")
    print(f"  Memory savings: ~{old_300k_memory_gb - new_300k_memory_gb:.1f} GB")
    
    # Create visualization
    print(f"\n📈 Creating comparison plot...")
    
    # Convert sparse to dense for plotting (just small regions)
    plot_start_idx = len(common_mass_axis) // 4
    plot_end_idx = plot_start_idx + 1000  # Show 1000 bins
    
    plot_range = np.arange(plot_start_idx, min(plot_end_idx, len(common_mass_axis)))
    plot_axis = common_mass_axis[plot_range]
    
    # Old method dense representation
    old_dense = np.zeros(len(plot_range))
    old_mask = (old_indices >= plot_start_idx) & (old_indices < plot_end_idx)
    if np.any(old_mask):
        old_plot_indices = old_indices[old_mask] - plot_start_idx
        old_dense[old_plot_indices] = old_values[old_mask]
    
    # New method dense representation  
    new_dense = np.zeros(len(plot_range))
    new_mask = (new_indices >= plot_start_idx) & (new_indices < plot_end_idx)
    if np.any(new_mask):
        new_plot_indices = new_indices[new_mask] - plot_start_idx
        new_dense[new_plot_indices] = new_values[new_mask]
    
    # Plot comparison
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(plot_axis, old_dense, 'r-', linewidth=1, alpha=0.8, label=f'Old Dense Method ({len(old_indices):,} peaks)')
    plt.title('Old Dense Interpolation Method')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(plot_axis, new_dense, 'g-', linewidth=1, alpha=0.8, label=f'New Sparse Method ({len(new_indices):,} peaks)')
    plt.title('New Sparse Interpolation Method')
    plt.xlabel('m/z (Da)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved comparison plot as 'interpolation_comparison.png'")
    
    print(f"\n" + "=" * 60)
    print(f"🎯 CONCLUSION: New sparse method is {old_memory / new_memory:.0f}x more memory efficient!")
    print("=" * 60)

if __name__ == "__main__":
    run_comparison()