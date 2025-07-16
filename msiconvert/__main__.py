# msiconvert/__main__.py
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import argparse
import logging
import sys
from .convert import convert_msi
from .utils.data_processors import optimize_zarr_chunks
# This import will be needed for the new resampling step
from .postprocessing import PostProcessResampler 

def main():
    """Main entry point for msiconvert CLI."""
    parser = argparse.ArgumentParser(description='Convert MSI data to SpatialData format')
    
    # --- Argument parsing remains the same as your file ---
    parser.add_argument('input', help='Path to input MSI file or directory')
    parser.add_argument('output', help='Path for output file')
    parser.add_argument(
        '--format', 
        choices=['spatialdata'], 
        default='spatialdata',
        help='Output format type: spatialdata (full SpatialData format)'
    )
    parser.add_argument(
        '--dataset-id',
        default='msi_dataset',
        help='Identifier for the dataset'
    )
    parser.add_argument(
        '--pixel-size',
        type=float,
        default=1.0,
        help='Size of each pixel in micrometers'
    )
    parser.add_argument(
        '--handle-3d',
        action='store_true',
        help='Process as 3D data (default: treat as 2D slices)'
    )
    parser.add_argument(
        '--optimize-chunks',
        action='store_true',
        help='Optimize Zarr chunks after all processing'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    # Resampling parameters
    parser.add_argument(
        '--resample-mz',
        action='store_true',
        help='Enable m/z resampling to reduce data size'
    )
    parser.add_argument(
        '--resample-mode',
        choices=['linear', 'reflector'],
        default='linear',
        help='resample mode based on instrument type (default: linear)'
    )
    bin_group = parser.add_mutually_exclusive_group()
    bin_group.add_argument(
        '--bin-size',
        type=float,
        help='Bin size in milli-u (mu)'
    )
    bin_group.add_argument(
        '--num-bins',
        type=int,
        help='Total number of bins'
    )
    parser.add_argument(
        '--bin-reference-mz',
        type=float,
        default=1000.0,
        help='Reference m/z for bin size calculation (default: 1000.0)'
    )
    parser.add_argument(
        '--bin-min-mz',
        type=float,
        help='Minimum m/z for resampling (auto-detected if not specified)'
    )
    parser.add_argument(
        '--bin-max-mz',
        type=float,
        help='Maximum m/z for resampling (auto-detected if not specified)'
    )

    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # --- Logic from saved function: Build resampling params separately ---
    resampling_params = None
    if args.resample_mz:
        if not args.bin_size and not args.num_bins:
            logging.error("Error: --bin-size or --num-bins must be specified when --resample-mz is enabled")
            sys.exit(1)
            
        resampling_params = {
            'mode': args.resample_mode,
            'bin_size_mu': args.bin_size,
            'num_bins': args.num_bins,
            'reference_mz': args.bin_reference_mz,
            'min_mz': args.bin_min_mz,
            'max_mz': args.bin_max_mz
        }

    # --- Step 1: Perform the initial conversion WITHOUT resampling ---
    logging.info(f"Starting conversion of {args.input}...")
    success = convert_msi(
        args.input,
        args.output,
        format_type=args.format,
        dataset_id=args.dataset_id,
        pixel_size_um=args.pixel_size,
        handle_3d=args.handle_3d,
        resampling_params=None  # Resampling is now a post-processing step
    )
    
    if not success:
        logging.error("Initial conversion failed.")
        sys.exit(1)
    
    logging.info("Initial conversion completed successfully.")

    # --- Step 2: Apply post-processing resampling if requested ---
    if success and args.resample_mz and resampling_params:
        logging.info("Starting post-processing resampling...")
        try:
            resampler = PostProcessResampler(resampling_params)
            resample_result = resampler.process_zarr_file(
                args.output,
                in_place=True  # Modify the output file in-place
            )
            
            if not resample_result:
                logging.error("Resampling failed.")
                sys.exit(1)
            
            logging.info("Post-processing resampling completed successfully!")
            
        except Exception as e:
            logging.error(f"Error during resampling: {e}", exc_info=True)
            sys.exit(1)

    # --- Step 3: Optimize Zarr chunks if requested ---
    if success and args.optimize_chunks:
        logging.info("Optimizing Zarr chunks for better performance...")
        # For SpatialData format, optimize the table's X array
        optimize_zarr_chunks(args.output, f'tables/{args.dataset_id}/X')
        logging.info("Chunk optimization complete.")
    
    logging.info(f"All processing completed successfully. Output stored at {args.output}")

if __name__ == '__main__':
    main()