# thyra/converters/spatialdata/converter.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.base_reader import BaseMSIReader
from ...core.registry import register_converter
from .base_spatialdata_converter import SPATIALDATA_AVAILABLE, _import_error_msg
from .spatialdata_2d_converter import SpatialData2DConverter
from .spatialdata_3d_converter import SpatialData3DConverter

logger = logging.getLogger(__name__)


class SpatialDataConverter:
    """Factory converter for MSI data to SpatialData format.

    Creates appropriate 2D or 3D converter based on handle_3d parameter
    and data dimensions.
    """

    def __new__(
        cls,
        reader: BaseMSIReader,
        output_path: Path,
        dataset_id: str = "msi_dataset",
        pixel_size_um: float = 1.0,
        handle_3d: bool = False,
        pixel_size_detection_info: Optional[Dict[str, Any]] = None,
        resampling_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Create appropriate converter based on handle_3d parameter and data dimensions.

        Args:
            reader: MSI data reader
            output_path: Path for output file
            dataset_id: Identifier for the dataset
            pixel_size_um: Size of each pixel in micrometers
            handle_3d: Whether to process as 3D data (True) or 2D slices
                (False)
            pixel_size_detection_info: Optional metadata about pixel size
                detection
            resampling_config: Optional resampling configuration dict
            **kwargs: Additional keyword arguments

        Returns:
            SpatialData2DConverter or SpatialData3DConverter instance

        Raises:
            ImportError: If SpatialData dependencies are not available
        """
        # Check if we have 3D data and determine converter type
        dimensions = reader.get_essential_metadata().dimensions
        if dimensions[2] > 1 and not handle_3d:
            # Multi-slice 3D data treated as 2D slices
            return SpatialData2DConverter(
                reader=reader,
                output_path=output_path,
                dataset_id=dataset_id,
                pixel_size_um=pixel_size_um,
                pixel_size_detection_info=pixel_size_detection_info,
                resampling_config=resampling_config,
                **kwargs,
            )
        elif dimensions[2] == 1:
            # Single 2D slice - always use 2D converter
            return SpatialData2DConverter(
                reader=reader,
                output_path=output_path,
                dataset_id=dataset_id,
                pixel_size_um=pixel_size_um,
                pixel_size_detection_info=pixel_size_detection_info,
                resampling_config=resampling_config,
                **kwargs,
            )
        else:
            # True 3D volume (dimensions[2] > 1 and handle_3d=True)
            return SpatialData3DConverter(
                reader=reader,
                output_path=output_path,
                dataset_id=dataset_id,
                pixel_size_um=pixel_size_um,
                pixel_size_detection_info=pixel_size_detection_info,
                resampling_config=resampling_config,
                **kwargs,
            )


# Only register the converter if SpatialData dependencies are available
if SPATIALDATA_AVAILABLE:
    register_converter("spatialdata")(SpatialDataConverter)
    logger.debug("SpatialDataConverter registered successfully")
else:
    logger.warning(
        f"SpatialDataConverter not registered due to dependency issues: "
        f"{_import_error_msg}"
    )
