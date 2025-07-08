# msiconvert/__main__.py
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import argparse
import logging
from .convert import convert_msi
from .utils.data_processors import optimize_zarr_chunks


def main():
    parser = argparse.ArgumentParser(description='Convert MSI data to SpatialData format')
    
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
        help='Optimize Zarr chunks after conversion'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Convert MSI data
    success = convert_msi(
        args.input,
        args.output,
        format_type=args.format,
        dataset_id=args.dataset_id,
        pixel_size_um=args.pixel_size,
        handle_3d=args.handle_3d
    )
    
    if success and args.optimize_chunks:
        # Optimize chunks for better performance
        if args.format == 'spatialdata':
            # For SpatialData format, optimize the table's X array
            optimize_zarr_chunks(args.output, f'tables/{args.dataset_id}/X')
    
    if success:
        print(f"Conversion completed successfully. Output stored at {args.output}")
    else:
        print("Conversion failed.")

if __name__ == '__main__':
    main()