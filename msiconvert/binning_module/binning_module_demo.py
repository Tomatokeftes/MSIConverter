# examples/binning_module_demo.py
"""
Demonstration of the binning module functionality.

This script shows various use cases for the modular binning system.
"""

import numpy as np
import matplotlib.pyplot as plt
from msiconvert.binning_module import (
    BinningRequest, BinningService, StrategyFactory, BinningConfig
)


def plot_bin_edges(result, title):
    """Plot bin edges and widths."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot bin edges
    ax1.scatter(range(len(result.bin_edges)), result.bin_edges, s=1)
    ax1.set_xlabel('Bin Index')
    ax1.set_ylabel('m/z')
    ax1.set_title(f'{title} - Bin Edges')
    ax1.grid(True, alpha=0.3)
    
    # Plot bin widths
    widths = np.diff(result.bin_edges)
    centers = (result.bin_edges[:-1] + result.bin_edges[1:]) / 2
    ax2.plot(centers, widths * 1000, linewidth=1)  # Convert to mDa
    ax2.set_xlabel('m/z')
    ax2.set_ylabel('Bin Width (mDa)')
    ax2.set_title(f'{title} - Bin Widths')
    ax2.grid(True, alpha=0.3)
    
    # Mark reference m/z
    ref_mz = result.parameters_used.reference_mz
    ax2.axvline(ref_mz, color='red', linestyle='--', alpha=0.5, 
                label=f'Reference m/z = {ref_mz}')
    ax2.axhline(result.achieved_width_at_ref_mz_da * 1000, 
                color='red', linestyle='--', alpha=0.5,
                label=f'Width at ref = {result.achieved_width_at_ref_mz_da*1000:.2f} mDa')
    ax2.legend()
    
    plt.tight_layout()
    return fig


def example_1_linear_tof_with_bin_size():
    """Example 1: Linear TOF with specified bin size."""
    print("=" * 60)
    print("Example 1: Linear TOF with 5 mDa bin size at m/z 1000")
    print("=" * 60)
    
    # Create request
    request = BinningRequest(
        min_mz=100.0,
        max_mz=2000.0,
        model_type='linear',
        bin_size_mu=5.0,  # 5 milli-Daltons
        reference_mz=1000.0
    )
    
    # Create service
    strategy = StrategyFactory.create_strategy(request.model_type)
    service = BinningService(strategy)
    
    # Generate bins
    result = service.generate_binned_axis(request)
    
    # Display results
    print(f"Generated {result.final_num_bins} bins")
    print(f"Bin width at m/z {request.reference_mz}: "
          f"{result.achieved_width_at_ref_mz_da*1000:.3f} mDa")
    print(f"First 5 edges: {result.bin_edges[:5]}")
    print(f"Last 5 edges: {result.bin_edges[-5:]}")
    
    # Plot
    fig = plot_bin_edges(result, "Linear TOF")
    plt.show()
    
    return result


def example_2_reflector_tof_with_num_bins():
    """Example 2: Reflector TOF with specified number of bins."""
    print("\n" + "=" * 60)
    print("Example 2: Reflector TOF with 5000 bins")
    print("=" * 60)
    
    # Create request
    request = BinningRequest(
        min_mz=50.0,
        max_mz=3000.0,
        model_type='reflector',
        num_bins=5000
    )
    
    # Create service
    strategy = StrategyFactory.create_strategy(request.model_type)
    service = BinningService(strategy)
    
    # Generate bins
    result = service.generate_binned_axis(request)
    
    # Display results
    print(f"Generated {result.final_num_bins} bins")
    print(f"Bin width at m/z {request.reference_mz}: "
          f"{result.achieved_width_at_ref_mz_da*1000:.3f} mDa")
    
    # Show bin width variation
    widths = np.diff(result.bin_edges)
    print(f"Minimum bin width: {np.min(widths)*1000:.3f} mDa at m/z {result.bin_edges[np.argmin(widths)]:.1f}")
    print(f"Maximum bin width: {np.max(widths)*1000:.3f} mDa at m/z {result.bin_edges[np.argmax(widths)]:.1f}")
    
    # Plot
    fig = plot_bin_edges(result, "Reflector TOF")
    plt.show()
    
    return result


def example_3_compare_strategies():
    """Example 3: Compare Linear and Reflector TOF strategies."""
    print("\n" + "=" * 60)
    print("Example 3: Comparing Linear vs Reflector TOF strategies")
    print("=" * 60)
    
    # Common parameters
    params = {
        'min_mz': 100.0,
        'max_mz': 2000.0,
        'bin_size_mu': 5.0,
        'reference_mz': 1000.0
    }
    
    # Create both strategies
    results = {}
    for model_type in ['linear', 'reflector']:
        request = BinningRequest(**params, model_type=model_type)
        strategy = StrategyFactory.create_strategy(model_type)
        service = BinningService(strategy)
        results[model_type] = service.generate_binned_axis(request)
    
    # Compare results
    print(f"Linear TOF: {results['linear'].final_num_bins} bins")
    print(f"Reflector TOF: {results['reflector'].final_num_bins} bins")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_type, result in results.items():
        widths = np.diff(result.bin_edges)
        centers = (result.bin_edges[:-1] + result.bin_edges[1:]) / 2
        ax.plot(centers, widths * 1000, label=f'{model_type.capitalize()} TOF', 
                linewidth=2, alpha=0.7)
    
    ax.set_xlabel('m/z')
    ax.set_ylabel('Bin Width (mDa)')
    ax.set_title('Comparison of Binning Strategies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(100, 2000)
    
    plt.show()
    
    return results


def example_4_custom_configuration():
    """Example 4: Using custom configuration."""
    print("\n" + "=" * 60)
    print("Example 4: Custom configuration with strict limits")
    print("=" * 60)
    
    # Create custom configuration
    config = BinningConfig(
        MAX_ALLOWED_BINS=1000,
        MIN_MZ_VALUE=50.0,
        MAX_MZ_VALUE=1500.0
    )
    
    # Try to create request that would exceed limits
    request = BinningRequest(
        min_mz=100.0,
        max_mz=1400.0,
        model_type='linear',
        bin_size_mu=1.0  # Very fine bins
    )
    
    # Create service with custom config
    strategy = StrategyFactory.create_strategy(request.model_type)
    service = BinningService(strategy, config)
    
    # This might fail if it exceeds MAX_ALLOWED_BINS
    try:
        result = service.generate_binned_axis(request)
        print(f"Success! Generated {result.final_num_bins} bins (within limit)")
    except Exception as e:
        print(f"Failed as expected: {e}")
        
        # Try again with larger bin size
        request = BinningRequest(
            min_mz=100.0,
            max_mz=1400.0,
            model_type='linear',
            bin_size_mu=5.0  # Larger bins
        )
        result = service.generate_binned_axis(request)
        print(f"Success with larger bins! Generated {result.final_num_bins} bins")
    
    return result


def example_5_binning_real_data():
    """Example 5: Binning for real MSI data processing."""
    print("\n" + "=" * 60)
    print("Example 5: Practical binning for MSI data")
    print("=" * 60)
    
    # Typical MALDI-TOF MSI parameters
    request = BinningRequest(
        min_mz=500.0,
        max_mz=3000.0,
        model_type='linear',  # MALDI-TOF is typically linear
        bin_size_mu=20.0,     # 20 mDa for reasonable data size
        reference_mz=1500.0   # Middle of typical peptide range
    )
    
    # Generate bins
    strategy = StrategyFactory.create_strategy(request.model_type)
    service = BinningService(strategy)
    result = service.generate_binned_axis(request)
    
    print(f"Binning configuration for MSI:")
    print(f"  - m/z range: {request.min_mz} - {request.max_mz}")
    print(f"  - Number of bins: {result.final_num_bins}")
    print(f"  - Target resolution: {request.bin_size_mu} mDa at m/z {request.reference_mz}")
    print(f"  - Achieved resolution: {result.achieved_width_at_ref_mz_da*1000:.2f} mDa")
    
    # Estimate memory usage
    num_pixels = 256 * 256  # Typical small MSI dataset
    bytes_per_value = 4  # float32
    memory_mb = (num_pixels * result.final_num_bins * bytes_per_value) / 1024 / 1024
    print(f"\nEstimated memory for 256x256 image: {memory_mb:.1f} MB")
    
    return result


if __name__ == "__main__":
    # Run all examples
    example_1_linear_tof_with_bin_size()
    example_2_reflector_tof_with_num_bins()
    example_3_compare_strategies()
    example_4_custom_configuration()
    example_5_binning_real_data()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)