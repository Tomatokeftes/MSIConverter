# msiconvert/__main__.py
import argparse
import logging
from pathlib import Path

from msiconvert.convert import convert_msi
from msiconvert.utils.data_processors import optimize_zarr_chunks
from msiconvert.utils.logging_config import setup_logging


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
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
        "--dataset-id",
        default="msi_dataset",
        help="Identifier for the dataset",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in micrometers. If not specified, automatic detection from "
        "metadata will be attempted. Required if detection fails.",
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

    # Advanced processing options
    parser.add_argument(
        "--enable-streaming",
        action="store_true",
        help="Enable true streaming mode with incremental Zarr writing for very large datasets",
    )
    parser.add_argument(
        "--dask-chunk-size",
        type=int,
        help="Chunk size for Dask processing (auto-detected by default)",
    )
    parser.add_argument(
        "--target-memory-gb",
        type=float,
        default=8.0,
        help="Target memory usage in GB for processing mode selection (default: 8.0)",
    )
    parser.add_argument(
        "--analyzer-type",
        choices=["tof", "orbitrap", "quadrupole"],
        default="tof",
        help="Mass analyzer type for enhanced mass axis generation (default: tof)",
    )

    # Interpolation parameters
    parser.add_argument(
        "--n-bins",
        type=int,
        help="Number of bins for mass axis interpolation (overrides analyzer-specific defaults). Default varies by analyzer: TOF=10000, Orbitrap=50000, Quadrupole=20000",
    )
    parser.add_argument(
        "--disable-interpolation",
        action="store_true",
        help="Disable interpolation and keep original individual mass axes",
    )

    return parser


def _validate_arguments(parser: argparse.ArgumentParser, args) -> None:
    """Validate command line arguments."""
    if args.pixel_size is not None and args.pixel_size <= 0:
        parser.error("Pixel size must be positive (got: {})".format(args.pixel_size))

    if not args.dataset_id.strip():
        parser.error("Dataset ID cannot be empty")

    if args.dask_chunk_size is not None and args.dask_chunk_size <= 0:
        parser.error(
            "Dask chunk size must be positive (got: {})".format(args.dask_chunk_size)
        )

    if args.target_memory_gb <= 0:
        parser.error(
            "Target memory must be positive (got: {})".format(args.target_memory_gb)
        )

    # Validate interpolation parameters
    if args.n_bins is not None and args.n_bins <= 0:
        parser.error("N-bins must be positive (got: {})".format(args.n_bins))

    # Dask is always enabled - no validation needed


def _check_imzml_requirements(
    parser: argparse.ArgumentParser, input_path: Path
) -> None:
    """Check ImzML format requirements."""
    ibd_path = input_path.with_suffix(".ibd")
    if not ibd_path.exists():
        parser.error(
            f"ImzML file requires corresponding .ibd file, but not found: {ibd_path}"
        )


def _check_bruker_requirements(
    parser: argparse.ArgumentParser, input_path: Path
) -> None:
    """Check Bruker format requirements."""
    if (
        not (input_path / "analysis.tsf").exists()
        and not (input_path / "analysis.tdf").exists()
    ):
        parser.error(
            f"Bruker .d directory requires analysis.tsf or analysis.tdf file: {input_path}"
        )


def _validate_input_path(parser: argparse.ArgumentParser, input_path: Path) -> None:
    """Validate input path and format requirements."""
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    if input_path.is_file() and input_path.suffix.lower() == ".imzml":
        _check_imzml_requirements(parser, input_path)
    elif input_path.is_dir() and input_path.suffix.lower() == ".d":
        _check_bruker_requirements(parser, input_path)


def _validate_output_path(parser: argparse.ArgumentParser, output_path: Path) -> None:
    """Validate output path."""
    if output_path.exists():
        parser.error(f"Output path already exists: {output_path}")


def _perform_conversion(args) -> bool:
    """Perform the MSI data conversion."""
    return convert_msi(
        args.input,
        args.output,
        format_type=args.format,
        dataset_id=args.dataset_id,
        pixel_size_um=args.pixel_size,
        handle_3d=args.handle_3d,
        use_dask=True,
        enable_streaming=args.enable_streaming,
        dask_chunk_size=args.dask_chunk_size,
        target_memory_gb=args.target_memory_gb,
        analyzer_type=args.analyzer_type,
        n_bins=args.n_bins,
        disable_interpolation=args.disable_interpolation,
    )


def _optimize_output(args) -> None:
    """Optimize output chunks if requested."""
    if args.format == "spatialdata":
        optimize_zarr_chunks(args.output, f"tables/{args.dataset_id}/X")


def main() -> None:
    """Main entry point for the CLI."""
    parser = _create_argument_parser()
    args = parser.parse_args()

    # Validate arguments and paths
    _validate_arguments(parser, args)
    input_path = Path(args.input)
    output_path = Path(args.output)
    _validate_input_path(parser, input_path)
    _validate_output_path(parser, output_path)

    # Configure logging
    setup_logging(log_level=getattr(logging, args.log_level), log_file=args.log_file)

    # Perform conversion
    success = _perform_conversion(args)

    # Optimize chunks if requested and conversion succeeded
    if success and args.optimize_chunks:
        _optimize_output(args)

    # Log final result
    if success:
        logging.info(
            f"Conversion completed successfully. Output stored at {args.output}"
        )
    else:
        logging.error("Conversion failed.")


if __name__ == "__main__":
    main()
