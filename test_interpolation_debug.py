"""
Debug script to test interpolation hanging issue
"""

import logging
import sys
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from msiconvert.config import InterpolationConfig
from msiconvert import convert_msi

def test_interpolation():
    """Test interpolation with a small dataset"""
    
    # Create test configuration
    config = InterpolationConfig(
        enabled=True,
        method="pchip",
        interpolation_bins=1000,  # Very small for testing
        max_workers=2,  # Minimal workers
        validate_quality=False,  # Skip validation for speed
        adaptive_workers=False
    )
    
    # Test with example data
    test_files = [
        r"C:\Users\P70078823\Desktop\MSIConverter\tests\data\test_imzml\Example_Continuous.imzML",
        r"C:\Users\P70078823\Desktop\MSIConverter\tests\data\bruker_tdf"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            logging.info(f"\n{'='*60}")
            logging.info(f"Testing interpolation with: {test_file}")
            logging.info(f"{'='*60}")
            
            output_path = Path(test_file).parent / "test_interpolation_output.zarr"
            
            try:
                result = convert_msi(
                    input_path=test_file,
                    output_path=str(output_path),
                    format_type="spatialdata",
                    dataset_id="test_dataset",
                    pixel_size_um=20.0,
                    interpolation_config=config
                )
                
                logging.info(f"Conversion result: {result}")
                
                # Clean up
                if output_path.exists():
                    import shutil
                    shutil.rmtree(output_path)
                    
            except Exception as e:
                logging.error(f"Error during conversion: {e}")
                import traceback
                traceback.print_exc()
            
            break  # Only test the first available file

if __name__ == "__main__":
    test_interpolation()