#!/usr/bin/env python3
"""
Proposed sparse interpolation method to fix memory issues.
"""

import numpy as np
from typing import Tuple
from numpy.typing import NDArray

def sparse_linear_interpolation(
    common_mass_axis: NDArray[np.float64],
    original_mz: NDArray[np.float64], 
    original_intensities: NDArray[np.float64],
    sparsity_threshold: float = 1e-10
) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """
    Perform sparse linear interpolation without creating dense arrays.
    
    Returns:
        Tuple of (sparse_indices, sparse_values) directly
    """
    if len(original_mz) == 0:
        return np.array([], dtype=np.int_), np.array([], dtype=np.float64)
    
    sparse_indices = []
    sparse_values = []
    
    # For each original peak, find which common axis bins it affects
    for i, (mz, intensity) in enumerate(zip(original_mz, original_intensities)):
        if intensity <= sparsity_threshold:
            continue
            
        # Find the insertion point in common mass axis
        insert_idx = np.searchsorted(common_mass_axis, mz)
        
        # Handle edge cases
        if insert_idx == 0:
            # Before first bin - assign to first bin
            sparse_indices.append(0)
            sparse_values.append(intensity)
        elif insert_idx >= len(common_mass_axis):
            # After last bin - assign to last bin  
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

def test_sparse_interpolation():
    print("=== SPARSE INTERPOLATION TEST ===")
    
    # Test parameters
    n_bins = 300000
    n_original_peaks = 2500
    
    # Create test data
    np.random.seed(42)
    original_mz = np.sort(np.random.uniform(100, 1000, n_original_peaks))
    original_intensities = np.random.exponential(1000, n_original_peaks)
    common_mass_axis = np.linspace(100, 1000, n_bins)
    
    print(f"Original peaks: {len(original_mz):,}")
    print(f"Common mass axis: {len(common_mass_axis):,} bins")
    
    # Memory usage of inputs
    input_memory = original_mz.nbytes + original_intensities.nbytes
    print(f"Input memory: {input_memory:,} bytes ({input_memory/1024:.1f} KB)")
    
    # OLD METHOD (dense)
    dense_interpolated = np.interp(common_mass_axis, original_mz, original_intensities, left=0.0, right=0.0)
    dense_memory = dense_interpolated.nbytes
    
    nonzero_mask = dense_interpolated > 1e-10
    dense_sparse_indices = np.where(nonzero_mask)[0]
    dense_sparse_values = dense_interpolated[nonzero_mask]
    
    print(f"\n=== OLD DENSE METHOD ===")
    print(f"Temporary dense array: {dense_memory:,} bytes ({dense_memory/1024:.1f} KB)")
    print(f"Final sparse elements: {len(dense_sparse_indices):,}")
    print(f"Final sparse memory: {dense_sparse_indices.nbytes + dense_sparse_values.nbytes:,} bytes")
    
    # NEW METHOD (sparse)
    sparse_indices, sparse_values = sparse_linear_interpolation(
        common_mass_axis, original_mz, original_intensities
    )
    
    sparse_memory = sparse_indices.nbytes + sparse_values.nbytes
    
    print(f"\n=== NEW SPARSE METHOD ===")
    print(f"No temporary dense array!")
    print(f"Final sparse elements: {len(sparse_indices):,}")
    print(f"Final sparse memory: {sparse_memory:,} bytes")
    print(f"Memory reduction: {dense_memory / sparse_memory:.1f}x")
    
    # For full dataset
    n_pixels = 33800
    total_dense_gb = (dense_memory * n_pixels) / (1024**3)
    total_sparse_gb = (sparse_memory * n_pixels) / (1024**3)
    
    print(f"\n=== FULL DATASET IMPACT ===")
    print(f"Old method peak memory: {total_dense_gb:.1f} GB")
    print(f"New method peak memory: {total_sparse_gb:.1f} GB") 
    print(f"Memory savings: {total_dense_gb - total_sparse_gb:.1f} GB")

if __name__ == "__main__":
    test_sparse_interpolation()