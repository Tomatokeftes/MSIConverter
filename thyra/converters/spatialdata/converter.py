# thyra/converters/spatialdata/converter.py

import logging

from ...core.registry import register_converter
from .streaming_converter import StreamingSpatialDataConverter

logger = logging.getLogger(__name__)


class SpatialDataConverter(StreamingSpatialDataConverter):
    """The converter registered as ``"spatialdata"``.

    There is one converter (design decision D11):
    :class:`StreamingSpatialDataConverter`, which makes two passes over
    the source and writes every table from memory-mapped arrays through
    spatialdata's writer. This name is what the registry and
    ``convert_msi`` reach it by. ``handle_3d`` decides the shape of the
    store -- one table per z plane (the default) or one table for the
    whole volume -- where this used to be a factory picking one of two
    in-memory converters on the same argument.
    """


register_converter("spatialdata")(SpatialDataConverter)
logger.debug("SpatialDataConverter registered successfully")
