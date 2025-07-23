from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class InstrumentMetadata(BaseModel):
    """Instrument-specific metadata"""
    instrument_type: str  # "TOF", "Orbitrap", "FTICR"
    model: Optional[str] = None
    resolution_at_400: Optional[float] = None
    
class AcquisitionMetadata(BaseModel):
    """Acquisition parameters"""
    mass_range_lower: float = Field(alias="MzAcqRangeLower")
    mass_range_upper: float = Field(alias="MzAcqRangeUpper")
    pixel_size_x: float = Field(alias="SpotSizeX")
    pixel_size_y: float = Field(alias="SpotSizeY")
    
class SpatialMetadata(BaseModel):
    """Spatial bounds from metadata"""
    min_x: int = Field(alias="ImagingAreaMinXIndexPos")
    max_x: int = Field(alias="ImagingAreaMaxXIndexPos")
    min_y: int = Field(alias="ImagingAreaMinYIndexPos")
    max_y: int = Field(alias="ImagingAreaMaxYIndexPos")
    
class DatasetMetadata(BaseModel):
    """Complete dataset metadata"""
    instrument: InstrumentMetadata
    acquisition: AcquisitionMetadata
    spatial: SpatialMetadata
    raw_metadata: Dict[str, Any] = {}