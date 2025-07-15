# msiconvert/convert.py
from pathlib import Path
import logging
import traceback
import warnings
from typing import Dict, Any, Optional
from cryptography.utils import CryptographyDeprecationWarning

from .core.registry import detect_format, get_reader_class, get_converter_class
from .core.resampled_reader import ResampledReader

warnings.filterwarnings(
    "ignore", 
    message=r"Accession IMS:1000046.*",  # or just "ignore" all UserWarning from that module
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
    resampling_params: Optional[Dict[str, Any]] = None,
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
        
        # Apply resampling wrapper if requested
        if resampling_params and resampling_params.get('enabled', False):
            logging.info(f"Applying {resampling_params['mode']} resampling to reduce data size")
            reader = ResampledReader(reader, resampling_params)
            
            # Log resampling results
            mass_axis = reader.get_common_mass_axis()
            logging.info(f"Resampled mass axis: {len(mass_axis)} bins")
            
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