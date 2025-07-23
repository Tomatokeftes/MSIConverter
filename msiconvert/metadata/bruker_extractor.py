import sqlite3
from pathlib import Path
from typing import Dict, Any, Tuple
from .base_extractor import BaseMetadataExtractor
from .metadata_models import DatasetMetadata, InstrumentMetadata, AcquisitionMetadata, SpatialMetadata

class BrukerMetadataExtractor(BaseMetadataExtractor):
    """Extract metadata from Bruker .d folders WITHOUT reading spectra"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.db_path = data_path / "analysis.tsf"  # or .tdf
        if not self.db_path.exists():
            self.db_path = data_path / "analysis.tdf"
            
    def extract_mass_bounds(self) -> Tuple[float, float]:
        """Get mass bounds from GlobalMetadata table"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Query for mass bounds
            query = """
            SELECT Value FROM GlobalMetadata 
            WHERE Key IN ('MzAcqRangeLower', 'MzAcqRangeUpper')
            ORDER BY Key
            """
            results = cursor.execute(query).fetchall()
            
            if len(results) != 2:
                raise ValueError("Mass bounds not found in metadata")
                
            min_mz = float(results[0][0])
            max_mz = float(results[1][0])
            
            return min_mz, max_mz
            
    def extract_spatial_bounds(self) -> Dict[str, int]:
        """Get spatial bounds from GlobalMetadata table"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Query for spatial bounds
            keys = [
                'ImagingAreaMinXIndexPos',
                'ImagingAreaMaxXIndexPos', 
                'ImagingAreaMinYIndexPos',
                'ImagingAreaMaxYIndexPos'
            ]
            
            query = f"""
            SELECT Key, Value FROM GlobalMetadata 
            WHERE Key IN ({','.join(['?' for _ in keys])})
            """
            
            results = cursor.execute(query, keys).fetchall()
            bounds = {row[0]: int(row[1]) for row in results}
            
            return {
                'min_x': bounds['ImagingAreaMinXIndexPos'],
                'max_x': bounds['ImagingAreaMaxXIndexPos'],
                'min_y': bounds['ImagingAreaMinYIndexPos'],
                'max_y': bounds['ImagingAreaMaxYIndexPos']
            }
            
    def extract_instrument_info(self) -> Dict[str, Any]:
        """Get instrument type from metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get instrument model
            query = "SELECT Value FROM GlobalMetadata WHERE Key = 'InstrumentName'"
            result = cursor.execute(query).fetchone()
            
            instrument_name = result[0] if result else "Unknown"
            
            # Determine type from name
            instrument_type = "TOF"  # default
            if "orbitrap" in instrument_name.lower():
                instrument_type = "Orbitrap"
            elif "fticr" in instrument_name.lower():
                instrument_type = "FTICR"
                
            return {
                "type": instrument_type,
                "model": instrument_name
            }