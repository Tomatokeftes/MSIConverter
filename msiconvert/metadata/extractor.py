from typing import Any, Dict
from ..core.base_reader import BaseMSIReader

class MetadataExtractor:
    """
    Extracts metadata from an MSI data reader.
    """
    def __init__(self, reader: BaseMSIReader):
        """
        Initializes the MetadataExtractor with a reader.

        Args:
            reader: An instance of a class inheriting from BaseMSIReader.
        """
        self.reader = reader

    def extract(self) -> Dict[str, Any]:
        """
        Extracts metadata from the reader.

        Returns:
            A dictionary containing the extracted metadata.
        """
        metadata = self.reader.get_metadata()
        return metadata
