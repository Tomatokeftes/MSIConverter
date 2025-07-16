# msiconvert/convert.py
from pathlib import Path
import logging
import traceback
import warnings
from typing import Dict, Any, Optional
from cryptography.utils import CryptographyDeprecationWarning

from .core.registry import detect_format, get_reader_class, get_converter_class
# The ResampledReader is no longer needed here as resampling is a post-processing step
# from .postprocessing.resampler import ResampledReader 

warnings.filterwarnings(
    "ignore", 
    message=r"Accession IMS:1000046.*",
    category=UserWarning,
    module=r"pyimzml.ontology.ontology"
)

warnings.filterwarnings(
    "ignore", 
    category=CryptographyDeprecationWarning
)


def convert_msi(
    input_path: str,
    output_path: str,
    format_type: str = "spatialdata",
    dataset_id: str = "msi_dataset",
    pixel_size_um: float = 1.0,
    handle_3d: bool = False,
    # The resampling_params parameter is removed from the function signature
    **kwargs
) -> bool:
    """Convert MSI data to the specified format with enhanced error handling."""
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
        # Detect input format
        input_format = detect_format(input_path)
        logging.info(f"Detected format: {input_format}")
        
        # Create reader
        reader_class = get_reader_class(input_format)
        logging.info(f"Using reader: {reader_class.__name__}")
        reader = reader_class(input_path)

        # All resampling-related code, including the ResampledReader wrapper, is removed.
        # The reader is now passed directly to the converter.

        # Create converter
        converter_class = get_converter_class(format_type.lower())
        logging.info(f"Using converter: {converter_class.__name__}")
        converter = converter_class(
            reader, 
            output_path,
            dataset_id=dataset_id,
            pixel_size_um=pixel_size_um,
            handle_3d=handle_3d,
            **kwargs
        )
        
        # Run conversion
        logging.info("Starting conversion...")
        result = converter.convert()
        logging.info(f"Conversion {'completed successfully' if result else 'failed'}")
        return result
    
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        # Log detailed traceback for debugging
        logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
        return False