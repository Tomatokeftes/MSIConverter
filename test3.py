#!/usr/bin/env python3
"""
Compare three approaches: Dense interpolation vs Sparse interpolation vs Nearest bin assignment.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple
from numpy.typing import NDArray

def dense_interpolation(
    common_mass_axis: NDArray[np.float64],
    original_mz: NDArray[np.float64], 
    original_intensities: NDArray[np.float64],
    sparsity_threshold: float = 1e-10
) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """CURRENT METHOD: Dense interpolation (memory hog)."""
    print("  🔄 Creating dense interpolated array...")
    
    # This creates a HUGE dense array temporarily
    dense_interpolated = np.interp(
        common_mass_axis, original_mz, original_intensities,
        left=0.0, right=0.0
    )
    
    dense_memory_mb = dense_interpolated.nbytes / (1024 * 1024)
    print(f"  🚨 Temporary dense array: {dense_memory_mb:.1f} MB")
    
    # Convert to sparse
    nonzero_mask = dense_interpolated > sparsity_threshold
    sparse_indices = np.where(nonzero_mask)[0].astype(np.int_)
    sparse_values = dense_interpolated[nonzero_mask].astype(np.float64)
    
    return sparse_indices, sparse_values

def sparse_interpolation(
    common_mass_axis: NDArray[np.float64],
    original_mz: NDArray[np.float64], 
    original_intensities: NDArray[np.float64],
    sparsity_threshold: float = 1e-10
) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """SPARSE INTERPOLATION: True linear interpolation, but memory efficient."""
    print("  🚀 Sparse linear interpolation...")
    
    if len(original_mz) == 0:
        return np.array([], dtype=np.int_), np.array([], dtype=np.float64)
    
    sparse_indices = []
    sparse_values = []
    
    for mz, intensity in zip(original_mz, original_intensities):
        if intensity <= sparsity_threshold:
            continue
            
        # Find the insertion point
        insert_idx = np.searchsorted(common_mass_axis, mz)
        
        if insert_idx == 0:
            # Before first bin
            sparse_indices.append(0)
            sparse_values.append(intensity)
        elif insert_idx >= len(common_mass_axis):
            # After last bin
            sparse_indices.append(len(common_mass_axis) - 1)
            sparse_values.append(intensity)
        else:
            # TRUE LINEAR INTERPOLATION between two bins
            left_idx = insert_idx - 1
            right_idx = insert_idx
            
            left_mz = common_mass_axis[left_idx]
            right_mz = common_mass_axis[right_idx]
            
            # Calculate interpolation weights
            total_distance = right_mz - left_mz
            if total_distance > 0:
                left_weight = (right_mz - mz) / total_distance
                right_weight = (mz - left_mz) / total_distance
                
                # Split intensity between both bins
                sparse_indices.append(left_idx)
                sparse_values.append(intensity * left_weight)
                sparse_indices.append(right_idx)
                sparse_values.append(intensity * right_weight)
            else:
                sparse_indices.append(left_idx)
                sparse_values.append(intensity)
    
    if not sparse_indices:
        return np.array([], dtype=np.int_), np.array([], dtype=np.float64)
    
    # Combine contributions to same bins
    sparse_indices = np.array(sparse_indices, dtype=np.int_)
    sparse_values = np.array(sparse_values, dtype=np.float64)
    
    unique_indices, inverse_indices = np.unique(sparse_indices, return_inverse=True)
    summed_values = np.zeros(len(unique_indices), dtype=np.float64)
    np.add.at(summed_values, inverse_indices, sparse_values)
    
    # Filter out values below threshold
    keep_mask = summed_values > sparsity_threshold
    return unique_indices[keep_mask], summed_values[keep_mask]

def nearest_bin_assignment(
    common_mass_axis: NDArray[np.float64],
    original_mz: NDArray[np.float64], 
    original_intensities: NDArray[np.float64],
    sparsity_threshold: float = 1e-10
) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """NEAREST BIN: Just assign each peak to the closest bin (no interpolation)."""
    print("  📍 Nearest bin assignment (no interpolation)...")
    
    if len(original_mz) == 0:
        return np.array([], dtype=np.int_), np.array([], dtype=np.float64)
    
    # Find nearest bin for each peak
    bin_indices = np.searchsorted(common_mass_axis, original_mz)
    
    # Handle edge cases
    bin_indices = np.clip(bin_indices, 0, len(common_mass_axis) - 1)
    
    # For peaks exactly between two bins, choose the closer one
    for i, (mz, bin_idx) in enumerate(zip(original_mz, bin_indices)):
        if bin_idx > 0 and bin_idx < len(common_mass_axis):
            left_distance = abs(mz - common_mass_axis[bin_idx - 1])
            right_distance = abs(mz - common_mass_axis[bin_idx])
            if left_distance < right_distance:
                bin_indices[i] = bin_idx - 1
    
    # Filter out low intensities
    keep_mask = original_intensities > sparsity_threshold
    final_indices = bin_indices[keep_mask]
    final_intensities = original_intensities[keep_mask]
    
    # Combine intensities for peaks assigned to same bin
    unique_indices, inverse_indices = np.unique(final_indices, return_inverse=True)
    summed_intensities = np.zeros(len(unique_indices), dtype=np.float64)
    np.add.at(summed_intensities, inverse_indices, final_intensities)
    
    return unique_indices.astype(np.int_), summed_intensities

def run_three_way_comparison():
    """Compare all three approaches."""
    print("=" * 70)
    print("🧪 THREE-WAY INTERPOLATION COMPARISON")
    print("=" * 70)
    
    # Test parameters
    n_bins = 50000
    n_original_peaks = 2500
    mass_range = (100, 1000)
    
    print(f"📊 Test Setup:")
    print(f"  Common mass axis bins: {n_bins:,}")
    print(f"  Original peaks: {n_original_peaks:,}")
    print(f"  Mass range: {mass_range[0]}-{mass_range[1]} Da")
    
    # Create test data
    print(f"\n🔧 Creating test data...")
    np.random.seed(42)
    
    # Create realistic clustered peaks
    cluster_centers = np.random.uniform(mass_range[0], mass_range[1], n_original_peaks // 10)
    original_mz = []
    original_intensities = []
    
    for center in cluster_centers:
        cluster_mz = np.random.normal(center, 1.0, 10)
        cluster_mz = np.clip(cluster_mz, mass_range[0], mass_range[1])
        cluster_intensities = np.random.exponential(1000, 10)
        original_mz.extend(cluster_mz)
        original_intensities.extend(cluster_intensities)
    
    original_mz = np.sort(np.array(original_mz))
    original_intensities = np.array(original_intensities)
    common_mass_axis = np.linspace(mass_range[0], mass_range[1], n_bins)
    
    print(f"  ✅ Created {len(original_mz):,} original peaks")
    
    results = {}
    
    # Test Method 1: Dense interpolation
    print(f"\n🔴 METHOD 1: DENSE INTERPOLATION (Current)")
    start_time = time.time()
    dense_indices, dense_values = dense_interpolation(
        common_mass_axis, original_mz, original_intensities
    )
    dense_time = time.time() - start_time
    
    results['dense'] = {
        'indices': dense_indices,
        'values': dense_values,
        'time': dense_time,
        'memory_kb': (dense_indices.nbytes + dense_values.nbytes) / 1024,
        'count': len(dense_indices),
        'total_intensity': np.sum(dense_values)
    }
    
    print(f"  ⏱️  Time: {dense_time:.3f}s")
    print(f"  📦 Peaks: {results['dense']['count']:,}")
    print(f"  💾 Memory: {results['dense']['memory_kb']:.1f} KB")
    print(f"  🎯 Total intensity: {results['dense']['total_intensity']:.0f}")
    
    # Test Method 2: Sparse interpolation
    print(f"\n🟡 METHOD 2: SPARSE INTERPOLATION (New)")
    start_time = time.time()
    sparse_indices, sparse_values = sparse_interpolation(
        common_mass_axis, original_mz, original_intensities
    )
    sparse_time = time.time() - start_time
    
    results['sparse'] = {
        'indices': sparse_indices,
        'values': sparse_values,
        'time': sparse_time,
        'memory_kb': (sparse_indices.nbytes + sparse_values.nbytes) / 1024,
        'count': len(sparse_indices),
        'total_intensity': np.sum(sparse_values)
    }
    
    print(f"  ⏱️  Time: {sparse_time:.3f}s")
    print(f"  📦 Peaks: {results['sparse']['count']:,}")
    print(f"  💾 Memory: {results['sparse']['memory_kb']:.1f} KB")
    print(f"  🎯 Total intensity: {results['sparse']['total_intensity']:.0f}")
    
    # Test Method 3: Nearest bin assignment
    print(f"\n🟢 METHOD 3: NEAREST BIN (No Interpolation)")
    start_time = time.time()
    nearest_indices, nearest_values = nearest_bin_assignment(
        common_mass_axis, original_mz, original_intensities
    )
    nearest_time = time.time() - start_time
    
    results['nearest'] = {
        'indices': nearest_indices,
        'values': nearest_values,
        'time': nearest_time,
        'memory_kb': (nearest_indices.nbytes + nearest_values.nbytes) / 1024,
        'count': len(nearest_indices),
        'total_intensity': np.sum(nearest_values)
    }
    
    print(f"  ⏱️  Time: {nearest_time:.3f}s")
    print(f"  📦 Peaks: {results['nearest']['count']:,}")
    print(f"  💾 Memory: {results['nearest']['memory_kb']:.1f} KB")
    print(f"  🎯 Total intensity: {results['nearest']['total_intensity']:.0f}")
    
    # Comparison
    print(f"\n📊 COMPARISON (vs Dense):")
    sparse_speedup = dense_time / sparse_time if sparse_time > 0 else float('inf')
    nearest_speedup = dense_time / nearest_time if nearest_time > 0 else float('inf')
    
    print(f"  Sparse interpolation:")
    print(f"    🚀 {sparse_speedup:.1f}x faster")
    print(f"    💾 {results['dense']['memory_kb'] / results['sparse']['memory_kb']:.1f}x less memory")
    print(f"    📦 {results['dense']['count'] / results['sparse']['count']:.1f}x fewer peaks")
    print(f"    🎯 {abs(results['dense']['total_intensity'] - results['sparse']['total_intensity']) / results['dense']['total_intensity'] * 100:.2f}% intensity difference")
    
    print(f"  Nearest bin assignment:")
    print(f"    🚀 {nearest_speedup:.1f}x faster")
    print(f"    💾 {results['dense']['memory_kb'] / results['nearest']['memory_kb']:.1f}x less memory")
    print(f"    📦 {results['dense']['count'] / results['nearest']['count']:.1f}x fewer peaks")
    print(f"    🎯 {abs(results['dense']['total_intensity'] - results['nearest']['total_intensity']) / results['dense']['total_intensity'] * 100:.2f}% intensity difference")
    
    # Create visualization
    print(f"\n📈 Creating comparison plot...")
    
    plot_start_idx = len(common_mass_axis) // 4
    plot_end_idx = plot_start_idx + 1000
    plot_range = np.arange(plot_start_idx, min(plot_end_idx, len(common_mass_axis)))
    plot_axis = common_mass_axis[plot_range]
    
    plt.figure(figsize=(15, 12))
    
    for i, (method_name, method_data) in enumerate([
        ('Dense Interpolation', results['dense']),
        ('Sparse Interpolation', results['sparse']),
        ('Nearest Bin Assignment', results['nearest'])
    ], 1):
        
        plt.subplot(3, 1, i)
        
        # Convert sparse to dense for plotting
        plot_dense = np.zeros(len(plot_range))
        mask = (method_data['indices'] >= plot_start_idx) & (method_data['indices'] < plot_end_idx)
        if np.any(mask):
            plot_indices = method_data['indices'][mask] - plot_start_idx
            plot_dense[plot_indices] = method_data['values'][mask]
        
        plt.plot(plot_axis, plot_dense, linewidth=1, alpha=0.8)
        plt.title(f'{method_name} ({method_data["count"]:,} peaks, {method_data["memory_kb"]:.1f} KB)')
        plt.ylabel('Intensity')
        if i == 3:
            plt.xlabel('m/z (Da)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('three_way_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved comparison plot as 'three_way_comparison.png'")
    
    print(f"\n" + "=" * 70)
    print(f"💡 INSIGHTS:")
    print(f"  • Dense interpolation creates {results['dense']['count']:,} peaks")
    print(f"  • Sparse interpolation creates {results['sparse']['count']:,} peaks (still interpolating)")
    print(f"  • Nearest bin creates {results['nearest']['count']:,} peaks (no interpolation)")
    print(f"  • Question: Do we really need interpolation at all?")
    print("=" * 70)

if __name__ == "__main__":
    run_three_way_comparison()