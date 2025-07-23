# msiconvert/__main__.py
import argparse
import logging
import sys
from pathlib import Path

from msiconvert.convert import convert_msi
from msiconvert.core.registry import detect_format, get_reader_class
from msiconvert.utils.data_processors import optimize_zarr_chunks
from msiconvert.utils.logging_config import setup_logging
from msiconvert.config import InterpolationConfig


def prompt_for_pixel_size(detected_size=None):
    """
    Interactively prompt user for pixel size.

    Args:
        detected_size: Optional tuple of (x_size, y_size) if detection succeeded

    Returns:
        float: Pixel size in micrometers
    """
    if detected_size is not None:
        print(
            f"\nOK Automatically detected pixel size: {detected_size[0]:.1f} x {detected_size[1]:.1f} um"
        )
        while True:
            response = input("Use detected pixel size? [Y/n]: ").strip().lower()
            if response in ["", "y", "yes"]:
                return detected_size[0]  # Use X size (assuming square pixels)
            elif response in ["n", "no"]:
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

    # Manual input
    while True:
        try:
            print("\nPixel size could not be automatically detected.")
            pixel_size_input = input(
                "Please enter pixel size in micrometers (um): "
            ).strip()
            pixel_size = float(pixel_size_input)
            if pixel_size <= 0:
                print("Error: Pixel size must be positive. Please try again.")
                continue
            return pixel_size
        except ValueError:
            print("Error: Please enter a valid number. Please try again.")
        except KeyboardInterrupt:
            print("\nConversion cancelled by user.")
            sys.exit(1)


def detect_pixel_size_interactive(reader, input_format):
    """
    Try automatic detection and fall back to interactive prompt if needed.

    Args:
        reader: MSI reader instance
        input_format: Format of the input file

    Returns:
        tuple: (pixel_size, detection_info_dict)
    """
    logging.info("Attempting automatic pixel size detection...")
    detected_pixel_size = reader.get_pixel_size()

    if detected_pixel_size is not None:
        logging.info(
            f"OK Automatically detected pixel size: {detected_pixel_size[0]:.1f} x {detected_pixel_size[1]:.1f} um"
        )
        selected_pixel_size = prompt_for_pixel_size(detected_pixel_size)

        # If user selected the detected size, create automatic detection info
        from msiconvert.config import PIXEL_SIZE_TOLERANCE

        if (
            abs(selected_pixel_size - detected_pixel_size[0]) < PIXEL_SIZE_TOLERANCE
        ):  # Used detected size
            detection_info = {
                "method": "automatic_interactive",
                "detected_x_um": float(detected_pixel_size[0]),
                "detected_y_um": float(detected_pixel_size[1]),
                "source_format": input_format,
                "detection_successful": True,
                "note": "Pixel size automatically detected from source metadata and confirmed by user",
            }
        else:  # User entered different value
            detection_info = {
                "method": "manual_override",
                "detected_x_um": float(detected_pixel_size[0]),
                "detected_y_um": float(detected_pixel_size[1]),
                "user_specified_um": float(selected_pixel_size),
                "source_format": input_format,
                "detection_successful": True,
                "note": "Pixel size was auto-detected but user specified a different value",
            }

        return selected_pixel_size, detection_info
    else:
        logging.warning("Could not automatically detect pixel size from metadata")
        selected_pixel_size = prompt_for_pixel_size(None)

        detection_info = {
            "method": "manual",
            "source_format": input_format,
            "detection_successful": False,
            "note": "Pixel size could not be auto-detected, manually specified by user",
        }

        return selected_pixel_size, detection_info


def dry_run_conversion(
    input_path,
    output_path,
    format_type="spatialdata",
    dataset_id="msi_dataset",
    pixel_size_um=None,
    handle_3d=False,
):
    """Simulate conversion process without writing files."""
    logging.info("=== DRY RUN MODE ===")

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return False

    # Check if output would overwrite
    if output_path.exists():
        logging.warning(
            f"Output path already exists and would be overwritten: " f"{output_path}"
        )

    try:
        # Detect format
        input_format = detect_format(input_path)
        logging.info(f"Detected input format: {input_format}")

        # Create reader to get metadata
        reader_class = get_reader_class(input_format)
        logging.info(f"Would use reader: {reader_class.__name__}")

        reader = reader_class(input_path)

        # Try automatic pixel size detection if not provided
        final_pixel_size = pixel_size_um
        if pixel_size_um is None:
            logging.info("Attempting automatic pixel size detection...")
            detected_pixel_size = reader.get_pixel_size()
            if detected_pixel_size is not None:
                final_pixel_size = detected_pixel_size[
                    0
                ]  # Use X size (assuming square pixels)
                logging.info(
                    f"OK Automatically detected pixel size: {detected_pixel_size[0]:.1f} x {detected_pixel_size[1]:.1f} um"
                )
            else:
                logging.warning("Could not automatically detect pixel size")
                final_pixel_size = 1.0  # Default fallback for dry-run
                logging.info(
                    f"Using default pixel size: {final_pixel_size} um (dry-run mode)"
                )

        # Get basic metadata
        logging.info(f"Dataset dimensions: {reader.shape}")
        logging.info(f"Number of spectra: {reader.n_spectra}")
        logging.info(f"Mass range: {reader.mass_range}")

        # Estimate output size (very rough)
        common_mass_axis = reader.get_common_mass_axis()
        from msiconvert.config import ESTIMATED_BYTES_PER_SPECTRUM_POINT, MB_TO_BYTES

        estimated_size_mb = (
            reader.n_spectra
            * len(common_mass_axis)
            * ESTIMATED_BYTES_PER_SPECTRUM_POINT
        ) / MB_TO_BYTES
        logging.info(f"Estimated output size: ~{estimated_size_mb:.1f} MB")

        # Show conversion parameters
        logging.info(f"Output format: {format_type}")
        logging.info(f"Dataset ID: {dataset_id}")
        logging.info(f"Pixel size: {final_pixel_size} um")
        logging.info(f"3D handling: {handle_3d}")
        logging.info(f"Output path: {output_path}")

        reader.close()
        logging.info("=== DRY RUN COMPLETED SUCCESSFULLY ===")
        return True

    except Exception as e:
        logging.error(f"Dry run failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert MSI data to SpatialData format"
    )

    parser.add_argument("input", help="Path to input MSI file or directory")
    parser.add_argument("output", help="Path for output file")
    parser.add_argument(
        "--format",
        choices=["spatialdata"],
        default="spatialdata",
        help="Output format type: spatialdata (full SpatialData format)",
    )
    parser.add_argument(
        "--dataset-id", default="msi_dataset", help="Identifier for the dataset"
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Size of each pixel in micrometers (if not specified, automatic detection will be attempted)",
    )
    parser.add_argument(
        "--handle-3d",
        action="store_true",
        help="Process as 3D data (default: treat as 2D slices)",
    )
    parser.add_argument(
        "--optimize-chunks",
        action="store_true",
        help="Optimize Zarr chunks after conversion",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument("--log-file", default=None, help="Path to the log file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate conversion process without writing output files",
    )
    
    # Interpolation arguments
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Enable intelligent interpolation for data size reduction and performance",
    )
    parser.add_argument(
        "--interpolation-bins",
        type=int,
        default=None,
        help="Number of bins for interpolated mass axis (default: 90000)",
    )
    parser.add_argument(
        "--interpolation-width",
        type=float,
        default=None,
        help="Mass axis width specification in Da (alternative to --interpolation-bins)",
    )
    parser.add_argument(
        "--interpolation-width-mz",
        type=float,
        default=400.0,
        help="Reference m/z for width specification (default: 400.0)",
    )
    parser.add_argument(
        "--interpolation-method",
        choices=["pchip", "linear", "adaptive"],
        default="pchip",
        help="Interpolation method (default: pchip for scientific data integrity)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of interpolation workers (default: auto-detect)",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=4,
        help="Minimum number of interpolation workers (default: 4)",
    )
    parser.add_argument(
        "--no-quality-validation",
        action="store_true",
        help="Disable interpolation quality validation (not recommended)",
    )
    parser.add_argument(
        "--physics-model",
        choices=["auto", "tof", "orbitrap", "fticr"],
        default="auto",
        help="Physics model for optimal mass axis generation (default: auto)",
    )

    args = parser.parse_args()

    # Input validation - check early to give better error messages
    if args.pixel_size is not None and args.pixel_size <= 0:
        parser.error("Pixel size must be positive (got: {})".format(args.pixel_size))

    if not args.dataset_id.strip():
        parser.error("Dataset ID cannot be empty")

    # Validate input path exists and is accessible
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    # Check for common input format issues
    if input_path.is_file() and input_path.suffix.lower() == ".imzml":
        ibd_path = input_path.with_suffix(".ibd")
        if not ibd_path.exists():
            parser.error(
                f"ImzML file requires corresponding .ibd file, but not found: {ibd_path}"
            )
    elif input_path.is_dir() and input_path.suffix.lower() == ".d":
        if (
            not (input_path / "analysis.tsf").exists()
            and not (input_path / "analysis.tdf").exists()
        ):
            parser.error(
                f"Bruker .d directory requires analysis.tsf or analysis.tdf file: {input_path}"
            )

    # Warn about output path conflicts early (but allow dry-run)
    output_path = Path(args.output)
    if not args.dry_run and output_path.exists():
        parser.error(
            f"Output path already exists (use --dry-run to preview): {output_path}"
        )

    # Validate interpolation arguments
    if args.interpolation_bins is not None and args.interpolation_bins <= 0:
        parser.error("Interpolation bins must be positive (got: {})".format(args.interpolation_bins))
    
    if args.interpolation_width is not None and args.interpolation_width <= 0:
        parser.error("Interpolation width must be positive (got: {})".format(args.interpolation_width))
        
    if args.interpolation_bins and args.interpolation_width:
        parser.error("Cannot specify both --interpolation-bins and --interpolation-width")
        
    if args.min_workers <= 0:
        parser.error("Minimum workers must be positive (got: {})".format(args.min_workers))
        
    if args.max_workers is not None and args.max_workers < args.min_workers:
        parser.error("Maximum workers must be >= minimum workers ({} < {})".format(args.max_workers, args.min_workers))

    # Configure logging
    setup_logging(log_level=getattr(logging, args.log_level), log_file=args.log_file)

    # Determine pixel size (auto-detect or use provided value)
    final_pixel_size = args.pixel_size
    if args.pixel_size is None and not args.dry_run:
        # For actual conversion, use interactive detection
        try:
            # Detect format and create reader for pixel size detection
            input_path = Path(args.input).resolve()
            input_format = detect_format(input_path)
            reader_class = get_reader_class(input_format)
            reader = reader_class(input_path)

            # Get pixel size interactively
            final_pixel_size, detection_info = detect_pixel_size_interactive(
                reader, input_format
            )
            reader.close()

            logging.info(f"Using pixel size: {final_pixel_size} um")
        except Exception as e:
            logging.error(f"Error during pixel size detection: {e}")
            logging.error("Conversion aborted.")
            return

    # Create interpolation configuration if requested
    interpolation_config = None
    if args.interpolate:
        interpolation_config = InterpolationConfig(
            enabled=True,
            method=args.interpolation_method,
            interpolation_bins=args.interpolation_bins,
            interpolation_width=args.interpolation_width,
            interpolation_width_mz=args.interpolation_width_mz,
            max_workers=args.max_workers or 80,  # Use default max if not specified
            min_workers=args.min_workers,
            validate_quality=not args.no_quality_validation,
            physics_model=args.physics_model,
        )
        logging.info(f"Interpolation enabled: {interpolation_config.get_summary()}")

    # Handle dry-run mode
    if args.dry_run:
        success = dry_run_conversion(
            args.input,
            args.output,
            format_type=args.format,
            dataset_id=args.dataset_id,
            pixel_size_um=final_pixel_size,
            handle_3d=args.handle_3d,
        )
    else:
        # Convert MSI data - pass detection_info if available
        detection_info_override = (
            detection_info if "detection_info" in locals() else None
        )
        success = convert_msi(
            args.input,
            args.output,
            format_type=args.format,
            dataset_id=args.dataset_id,
            pixel_size_um=final_pixel_size,
            handle_3d=args.handle_3d,
            pixel_size_detection_info_override=detection_info_override,
            interpolation_config=interpolation_config,
        )

    if success and args.optimize_chunks and not args.dry_run:
        # Optimize chunks for better performance
        if args.format == "spatialdata":
            # For SpatialData format, optimize the table's X array
            optimize_zarr_chunks(args.output, f"tables/{args.dataset_id}/X")

    if success:
        if args.dry_run:
            logging.info("Dry run completed successfully. No files were written.")
        else:
            logging.info(
                f"Conversion completed successfully. Output stored at {args.output}"
            )
    else:
        if args.dry_run:
            logging.error("Dry run failed.")
        else:
            logging.error("Conversion failed.")


if __name__ == "__main__":
    main()
