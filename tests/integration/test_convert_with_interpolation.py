# test_interpolation.py
from msiconvert import convert_msi
from msiconvert.config import InterpolationConfig
from pathlib import Path

# Configure interpolation
interp_config = InterpolationConfig(
    enabled=True,
    method="pchip",
    interpolation_bins=50000,  # Reduce file size significantly
    max_workers=8,
    validate_quality=True
)

# Run conversion with interpolation
success = convert_msi(
    input_path="C:\\Users\\P70078823\\Desktop\\MSIConverter\\data\\20231109_PEA_NEDC.d",
    output_path="C:\\Users\\P70078823\\Desktop\\MSIConverter\\output_interpolated.zarr",
    output_format="spatialdata",
    interpolation_config=interp_config,
    pixel_size_um= None  # Replace with your pixel size or None for auto-detection
)

print(f"Conversion {'succeeded' if success else 'failed'}")