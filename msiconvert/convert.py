# msiconvert/convert.py
import logging
import traceback
import warnings
from pathlib import Path

from .core.registry import detect_format, get_converter_class, get_reader_class


def _log_interpolation_performance(interpolation_config, reader_stats, total_time_sec, reader):
    """Log detailed interpolation performance statistics"""
    
    logging.info("=" * 60)
    logging.info("INTERPOLATION PERFORMANCE SUMMARY")
    logging.info("=" * 60)
    
    # Basic configuration
    logging.info(f"Method: {interpolation_config.method}")
    if interpolation_config.interpolation_bins:
        logging.info(f"Target bins: {interpolation_config.interpolation_bins:,}")
    if interpolation_config.interpolation_width:
        logging.info(f"Target width: {interpolation_config.interpolation_width} Da at {interpolation_config.interpolation_width_mz} m/z")
    
    # Dataset information
    n_spectra = 0
    try:
        dimensions = reader.get_dimensions()
        n_spectra = reader.n_spectra
        mass_axis_length = len(reader.get_common_mass_axis())
        
        logging.info(f"Dataset dimensions: {dimensions[0]} x {dimensions[1]} x {dimensions[2]}")  
        logging.info(f"Total spectra processed: {n_spectra:,}")
        logging.info(f"Original mass axis length: {mass_axis_length:,}")
        
        # Calculate size reduction
        if interpolation_config.interpolation_bins:
            reduction_factor = mass_axis_length / interpolation_config.interpolation_bins
            reduction_percent = (1 - 1/reduction_factor) * 100 if reduction_factor > 1 else 0
            logging.info(f"Mass axis reduction: {reduction_factor:.1f}x ({reduction_percent:.1f}% size reduction)")
        
    except Exception as e:
        logging.warning(f"Could not retrieve dataset statistics: {e}")
    
    # Performance metrics
    logging.info(f"Total conversion time: {total_time_sec:.1f} seconds")
    
    if n_spectra > 0:
        spectra_per_sec = n_spectra / total_time_sec
        logging.info(f"Processing throughput: {spectra_per_sec:.0f} spectra/second")
    
    # Memory and worker information
    logging.info(f"Workers used: {interpolation_config.min_workers}-{interpolation_config.max_workers}")
    logging.info(f"Quality validation: {'enabled' if interpolation_config.validate_quality else 'disabled'}")
    logging.info(f"Physics model: {interpolation_config.physics_model}")
    
    # Reader-specific performance stats
    if reader_stats:
        if 'average_read_time_ms' in reader_stats:
            logging.info(f"Average spectrum read time: {reader_stats['average_read_time_ms']:.2f} ms")
        if 'memory_stats' in reader_stats:
            memory_stats = reader_stats['memory_stats']
            if 'process_memory_gb' in memory_stats:
                logging.info(f"Peak memory usage: {memory_stats['process_memory_gb']:.1f} GB")
    
    logging.info("=" * 60)



warnings.filterwarnings(
    "ignore",
    message=r"Accession IMS:1000046.*",  # or just "ignore" all UserWarning from that module
    category=UserWarning,
    module=r"pyimzml.ontology.ontology",
)

# warnings.filterwarnings(
#     "ignore",
#     category=CryptographyDeprecationWarning
# )


def convert_msi(
    input_path: str,
    output_path: str,
    format_type: str = "spatialdata",
    dataset_id: str = "msi_dataset",
    pixel_size_um: float = None,
    handle_3d: bool = False,
    pixel_size_detection_info_override: dict = None,
    interpolation_config: 'InterpolationConfig' = None,
    **kwargs,
) -> bool:
    """
    Convert MSI data to the specified format with enhanced error handling and automatic pixel size detection.
    
    Args:
        input_path: Path to input MSI file or directory
        output_path: Path for output file
        format_type: Output format type (default: "spatialdata")
        dataset_id: Identifier for the dataset (default: "msi_dataset")
        pixel_size_um: Size of each pixel in micrometers (None for auto-detection)
        handle_3d: Process as 3D data (default: False for 2D slices)
        pixel_size_detection_info_override: Override pixel size detection metadata
        interpolation_config: Configuration for interpolation functionality (None to disable)
        **kwargs: Additional arguments passed to converter
        
    Returns:
        bool: True if conversion succeeded, False otherwise
    """

    if not input_path or not isinstance(input_path, (str, Path)):
        logging.error("Input path must be a valid string or Path object")
        return False

    if not output_path or not isinstance(output_path, (str, Path)):
        logging.error("Output path must be a valid string or Path object")
        return False

    if not isinstance(format_type, str) or not format_type.strip():
        logging.error("Format type must be a non-empty string")
        return False

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        logging.error("Dataset ID must be a non-empty string")
        return False

    if pixel_size_um is not None and (
        not isinstance(pixel_size_um, (int, float)) or pixel_size_um <= 0
    ):
        logging.error("Pixel size must be a positive number")
        return False

    if not isinstance(handle_3d, bool):
        logging.error("handle_3d must be a boolean value")
        return False

    # Validate interpolation config if provided
    if interpolation_config is not None:
        from .config import InterpolationConfig
        if not isinstance(interpolation_config, InterpolationConfig):
            logging.error("interpolation_config must be an InterpolationConfig instance")
            return False

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    logging.info(f"Processing input file: {input_path}")

    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return False

    if output_path.exists():
        logging.error(f"Destination {output_path} already exists.")
        return False

    try:
        input_format = detect_format(input_path)
        logging.info(f"Detected format: {input_format}")

        reader_class = get_reader_class(input_format)
        logging.info(f"Using reader: {reader_class.__name__}")
        reader = reader_class(input_path)

        # Handle automatic pixel size detection if not provided
        final_pixel_size = pixel_size_um
        pixel_size_detection_info = pixel_size_detection_info_override

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

                # Create pixel size detection provenance metadata
                pixel_size_detection_info = {
                    "method": "automatic",
                    "detected_x_um": float(detected_pixel_size[0]),
                    "detected_y_um": float(detected_pixel_size[1]),
                    "source_format": input_format,
                    "detection_successful": True,
                    "note": "Pixel size automatically detected from source metadata and applied to coordinate systems",
                }
            else:
                logging.error(
                    "ERROR Could not automatically detect pixel size from metadata"
                )
                logging.error(
                    "Please specify --pixel-size manually or ensure the input file contains pixel size metadata"
                )
                return False
        elif pixel_size_detection_info_override is None:
            # Manual pixel size was provided and no override info provided
            pixel_size_detection_info = {
                "method": "manual",
                "source_format": input_format,
                "detection_successful": False,
                "note": "Pixel size manually specified via --pixel-size parameter and applied to coordinate systems",
            }

        # Create converter
        converter_class = get_converter_class(format_type.lower())
        logging.info(f"Using converter: {converter_class.__name__}")
        converter = converter_class(
            reader,
            output_path,
            dataset_id=dataset_id,
            pixel_size_um=final_pixel_size,
            handle_3d=handle_3d,
            pixel_size_detection_info=pixel_size_detection_info,
            **kwargs,
        )

        # Run conversion (with interpolation if configured)
        if interpolation_config is not None and interpolation_config.enabled:
            logging.info(f"Starting conversion with interpolation: {interpolation_config.get_summary()}")
            
            # Get reader stats for performance comparison
            reader_stats = getattr(reader, 'get_performance_stats', lambda: {})()
            
            import time
            start_time = time.time()
            result = converter.convert_with_interpolation(interpolation_config)
            total_time = time.time() - start_time
            
            if result:
                # Log performance statistics
                _log_interpolation_performance(interpolation_config, reader_stats, total_time, reader)
        else:
            logging.info("Starting standard conversion...")
            result = converter.convert()
            
        logging.info(f"Conversion {'completed successfully' if result else 'failed'}")
        return result
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        # Log detailed traceback for debugging
        logging.error(f"Detailed traceback:\n{traceback.format_exc()}")
        return False
