# msiconvert/readers/imzml_reader.py
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from ..interpolators import SpectrumBufferPool, SpectrumBuffer

import numpy as np
from numpy.typing import NDArray
from pyimzml.ImzMLParser import ImzMLParser  # type: ignore
from tqdm import tqdm

from ..core.base_reader import BaseMSIReader
from ..core.registry import register_reader


@register_reader("imzml")
class ImzMLReader(BaseMSIReader):
    """Reader for imzML format files with optimizations for performance."""

    def __init__(
        self,
        data_path: Path,
        batch_size: int = 50,
        cache_coordinates: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize an ImzML reader.

        Args:
            data_path: Path to the imzML file
            batch_size: Default batch size for spectrum iteration
            cache_coordinates: Whether to cache coordinates upfront
            **kwargs: Additional arguments
        """
        super().__init__(data_path, **kwargs)
        self.filepath: Optional[Union[str, Path]] = data_path
        self.batch_size: int = batch_size
        self.cache_coordinates: bool = cache_coordinates
        self.parser: Optional[ImzMLParser] = None
        self.ibd_file: Optional[Any] = None
        self.imzml_path: Optional[Path] = None
        self.ibd_path: Optional[Path] = None
        self.is_continuous: bool = False
        self.is_processed: bool = False

        # Cached properties
        self._common_mass_axis: Optional[NDArray[np.float64]] = None
        self._dimensions: Optional[Tuple[int, int, int]] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._coordinates_cache: Dict[int, Tuple[int, int, int]] = {}

        # Initialize if path provided
        if data_path is not None:
            self._initialize_parser(data_path)

    def _initialize_parser(self, imzml_path: Union[str, Path]) -> None:
        """
        Initialize the ImzML parser with the given path.

        Args:
            imzml_path: Path to the imzML file to parse

        Raises:
            ValueError: If the corresponding .ibd file is not found or metadata parsing fails
            Exception: If parser initialization fails
        """
        if isinstance(imzml_path, str):
            imzml_path = Path(imzml_path)

        self.imzml_path = imzml_path
        self.ibd_path = imzml_path.with_suffix(".ibd")

        if not self.ibd_path.exists():
            raise ValueError(f"Corresponding .ibd file not found for {imzml_path}")

        # Open the .ibd file for reading
        self.ibd_file = open(self.ibd_path, mode="rb")

        # Initialize the parser
        logging.info(f"Initializing ImzML parser for {imzml_path}")
        try:
            self.parser = ImzMLParser(
                filename=str(imzml_path), parse_lib="lxml", ibd_file=self.ibd_file
            )
        except Exception as e:
            if self.ibd_file:
                self.ibd_file.close()
            logging.error(f"Failed to initialize ImzML parser: {e}")
            raise

        if self.parser.metadata is None:
            raise ValueError("Failed to parse metadata from imzML file.")

        # Determine file mode
        self.is_continuous = "continuous" in self.parser.metadata.file_description.param_by_name  # type: ignore
        self.is_processed = "processed" in self.parser.metadata.file_description.param_by_name  # type: ignore

        if self.is_continuous == self.is_processed:
            raise ValueError(
                "Invalid file mode, expected either 'continuous' or 'processed'."
            )

        # Cache coordinates if requested
        if self.cache_coordinates:
            self._cache_all_coordinates()

    def _cache_all_coordinates(self) -> None:
        """
        Cache all coordinates for faster access.

        Converts 1-based coordinates from imzML to 0-based coordinates for internal use.
        """
        if not hasattr(self, "parser") or self.parser is None:
            return

        logging.info("Caching all coordinates for faster access")
        self._coordinates_cache = {}

        for idx, (x, y, z) in enumerate(self.parser.coordinates):  # type: ignore
            self._coordinates_cache[idx] = (x - 1, y - 1, z - 1 if z > 0 else 0)

        logging.info(f"Cached {len(self._coordinates_cache)} coordinates")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about the imzML dataset.

        Returns:
            Dict[str, Any]: Dictionary containing metadata from the imzML file

        Raises:
            ValueError: If parser is not initialized and no filepath is available
        """
        if not hasattr(self, "parser") or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")

        if self._metadata is None:
            # We know parser is not None at this point
            parser = cast(ImzMLParser, self.parser)
            imzml_path = cast(Path, self.imzml_path)

            self._metadata = {
                "source": str(imzml_path),
                "uuid": parser.metadata.file_description.cv_params[0][2],  # type: ignore
                "file_mode": "continuous" if self.is_continuous else "processed",
            }

            # Add additional metadata if available
            if hasattr(parser, "imzmldict"):
                for key, value in parser.imzmldict.items():  # type: ignore
                    self._metadata[key] = value

        return self._metadata

    def get_pixel_size(self) -> Optional[Tuple[float, float]]:
        """
        Extract pixel size from ImzML metadata.

        Searches for cvParam elements with accessions:
        - IMS:1000046: pixel size x
        - IMS:1000047: pixel size y

        Returns:
            Optional[Tuple[float, float]]: Pixel size as (x_size, y_size) in micrometers,
                                         or None if not found in metadata.
        """
        if not hasattr(self, "parser") or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                return None

        parser = cast(ImzMLParser, self.parser)

        # Look for pixel size in imzmldict first (parsed cvParams)
        x_size = None
        y_size = None

        if hasattr(parser, "imzmldict") and parser.imzmldict:
            # Check for pixel size parameters in the parsed dictionary
            x_size = parser.imzmldict.get("pixel size x")
            y_size = parser.imzmldict.get("pixel size y")

        # If not found in imzmldict, try accessing the raw XML metadata
        if x_size is None or y_size is None:
            try:
                import xml.etree.ElementTree as ET

                # Access the XML root if available
                if hasattr(parser, "metadata") and hasattr(parser.metadata, "root"):
                    root = parser.metadata.root

                    # Define namespaces for XML parsing
                    namespaces = {
                        "mzml": "http://psi.hupo.org/ms/mzml",
                        "ims": "http://www.maldi-msi.org/download/imzml/imagingMS.obo",
                    }

                    # Search for cvParam elements with the pixel size accessions
                    for cvparam in root.findall(".//mzml:cvParam", namespaces):
                        accession = cvparam.get("accession")
                        if accession == "IMS:1000046":  # pixel size x
                            x_size = float(cvparam.get("value", 0))
                        elif accession == "IMS:1000047":  # pixel size y
                            y_size = float(cvparam.get("value", 0))

            except Exception as e:
                logging.warning(f"Failed to parse XML metadata for pixel size: {e}")

        # Convert to float if found as strings
        try:
            if x_size is not None:
                x_size = float(x_size)
            if y_size is not None:
                y_size = float(y_size)
        except (ValueError, TypeError):
            logging.warning("Invalid pixel size values found in metadata")
            return None

        # Return pixel sizes if both found
        if x_size is not None and y_size is not None:
            logging.info(
                f"Detected pixel size from ImzML metadata: x={x_size} μm, y={y_size} μm"
            )
            return (x_size, y_size)

        # Log if only partial information found
        if x_size is not None or y_size is not None:
            logging.warning(
                f"Partial pixel size information found: x={x_size}, y={y_size}"
            )

        return None

    def get_dimensions(self) -> Tuple[int, int, int]:
        """
        Return the dimensions of the imzML dataset (x, y, z).

        Returns:
            Tuple[int, int, int]: The x, y, z dimensions of the dataset

        Raises:
            ValueError: If parser is not initialized and no filepath is available
        """
        if not hasattr(self, "parser") or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")

        if self._dimensions is None:
            # We know parser is not None at this point
            parser = cast(ImzMLParser, self.parser)

            # Get dimensions from metadata if available
            if hasattr(parser, "imzmldict"):
                x_max = int(parser.imzmldict.get("max count of pixels x", 1))  # type: ignore
                y_max = int(parser.imzmldict.get("max count of pixels y", 1))  # type: ignore
                z_max = int(parser.imzmldict.get("max count of pixels z", 1))  # type: ignore
            else:
                # Calculate from coordinates
                x_coordinates: List[int] = [x for x, _, _ in parser.coordinates]  # type: ignore
                y_coordinates: List[int] = [y for _, y, _ in parser.coordinates]  # type: ignore
                z_coordinates: List[int] = [z for _, _, z in parser.coordinates]  # type: ignore

                x_max = max(x_coordinates) if x_coordinates else 1
                y_max = max(y_coordinates) if y_coordinates else 1
                z_max = max(z_coordinates) if z_coordinates else 1

            self._dimensions = (x_max, y_max, z_max)

        return self._dimensions

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """
        Return the common mass axis composed of all unique m/z values.

        For continuous mode, returns the m/z values from the first spectrum.
        For processed mode, collects all unique m/z values across spectra.

        Returns:
            NDArray[np.float64]: Array of m/z values in ascending order

        Raises:
            ValueError: If the common mass axis cannot be created
        """
        if not hasattr(self, "parser") or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")

        if self._common_mass_axis is None:
            # We know parser is not None at this point
            parser = cast(ImzMLParser, self.parser)

            if self.is_continuous:
                logging.info("Using m/z values from first spectrum (continuous mode)")
                spectrum_data = parser.getspectrum(0)  # type: ignore
                if spectrum_data is None or len(spectrum_data) < 1:  # type: ignore
                    raise ValueError("Could not get first spectrum")

                mzs = spectrum_data[0]
                if mzs.size == 0:
                    raise ValueError("First spectrum contains no m/z values")

                self._common_mass_axis = mzs
            else:
                # For processed data, collect unique m/z values across spectra
                logging.info(
                    "Building common mass axis from all unique m/z values (processed mode)"
                )

                total_spectra = len(parser.coordinates)  # type: ignore

                all_mzs: List[NDArray[np.float64]] = []

                with tqdm(
                    total=total_spectra,
                    desc="Building common mass axis",
                    unit="spectrum",
                ) as pbar:
                    for idx in range(total_spectra):
                        try:
                            spectrum_data = parser.getspectrum(idx)  # type: ignore
                            if spectrum_data is None or len(spectrum_data) < 1:  # type: ignore
                                continue

                            mzs = spectrum_data[0]
                            if mzs.size > 0:
                                all_mzs.append(mzs)
                        except Exception as e:
                            logging.warning(f"Error getting spectrum {idx}: {e}")
                        pbar.update(1)

                if not all_mzs:
                    # No spectra found - raise exception instead of returning empty array
                    raise ValueError("No spectra found to build common mass axis")

                try:
                    combined_mzs = np.concatenate(all_mzs)
                    unique_mzs = np.unique(combined_mzs)

                    if unique_mzs.size == 0:
                        raise ValueError("Failed to extract any m/z values")

                    self._common_mass_axis = unique_mzs
                    logging.info(
                        f"Created common mass axis with {len(self._common_mass_axis)} unique m/z values"
                    )
                except Exception as e:
                    # Re-raise with more context
                    raise ValueError(f"Error creating common mass axis: {e}") from e

        # Return the common mass axis
        return self._common_mass_axis

    def iter_spectra(
        self, batch_size: Optional[int] = None
    ) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """
        Iterate through spectra with progress monitoring and batch processing.

        Maps m/z values to the common mass axis using searchsorted for accurate
        representation in the output data structures.

        Args:
            batch_size: Number of spectra to process in each batch (None for default)

        Yields:
            Tuple containing:
                - Tuple[int, int, int]: Coordinates (x, y, z) - 0-based
                - NDArray[np.float64]: m/z values array
                - NDArray[np.float64]: Intensity values array

        Raises:
            ValueError: If parser is not initialized and no filepath is available
        """
        if not hasattr(self, "parser") or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")

        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.batch_size

        # We know parser is not None at this point
        parser = cast(ImzMLParser, self.parser)

        total_spectra = len(parser.coordinates)  # type: ignore
        dimensions = self.get_dimensions()
        total_pixels = dimensions[0] * dimensions[1] * dimensions[2]

        # Log information about spectra vs pixels
        logging.info(
            f"Processing {total_spectra} spectra in a grid of {total_pixels} pixels"
        )

        # Process in batches
        with tqdm(
            total=total_spectra,
            desc="Reading spectra",
            unit="spectrum",
            disable=getattr(self, "_quiet_mode", False),
        ) as pbar:
            if batch_size <= 1:
                # Process one at a time
                for idx in range(total_spectra):
                    try:
                        # Get coordinates (using cached 0-based coordinates if available)
                        if idx in self._coordinates_cache:
                            coords = self._coordinates_cache[idx]
                        else:
                            x, y, z = parser.coordinates[idx]  # type: ignore
                            coords = cast(
                                Tuple[int, int, int],
                                (x - 1, y - 1, z - 1 if z > 0 else 0),
                            )

                        # Get spectrum data
                        mzs, intensities = parser.getspectrum(idx)  # type: ignore

                        if mzs.size > 0 and intensities.size > 0:
                            yield coords, mzs, intensities

                        pbar.update(1)
                    except Exception as e:
                        logging.warning(f"Error processing spectrum {idx}: {e}")
                        pbar.update(1)
            else:
                # Process in batches
                for batch_start in range(0, total_spectra, batch_size):
                    batch_end = min(batch_start + batch_size, total_spectra)
                    batch_size_actual = batch_end - batch_start

                    # Process each spectrum in the batch
                    for offset in range(batch_size_actual):
                        idx = batch_start + offset
                        try:
                            # Get coordinates (using cached 0-based coordinates if available)
                            if idx in self._coordinates_cache:
                                coords = self._coordinates_cache[idx]
                            else:
                                x, y, z = parser.coordinates[idx]  # type: ignore
                                coords = cast(
                                    Tuple[int, int, int],
                                    (x - 1, y - 1, z - 1 if z > 0 else 0),
                                )

                            # Get spectrum data
                            mzs, intensities = parser.getspectrum(idx)  # type: ignore

                            if mzs.size > 0 and intensities.size > 0:
                                yield coords, mzs, intensities

                            pbar.update(1)
                        except Exception as e:
                            logging.warning(f"Error processing spectrum {idx}: {e}")
                            pbar.update(1)

    def read(self) -> Dict[str, Any]:
        """
        Read the entire imzML file and return a structured data dictionary.

        Returns:
            Dict containing:
                - mzs: NDArray[np.float64] - common m/z values array
                - intensities: NDArray[np.float64] - array of intensity arrays
                - coordinates: List[Tuple[int, int, int]] - list of (x,y,z) coordinates
                - width: int - number of pixels in x dimension
                - height: int - number of pixels in y dimension
                - depth: int - number of pixels in z dimension

        Raises:
            ValueError: If parser is not initialized and no filepath is available
        """
        if not hasattr(self, "parser") or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")

        # Get common mass axis
        mzs = self.get_common_mass_axis()

        # Get dimensions
        width, height, depth = self.get_dimensions()

        # Collect all spectra
        coordinates: List[Tuple[int, int, int]] = []
        intensities: List[NDArray[np.float64]] = []

        # Iterate through all spectra
        for coords, spectrum_mzs, spectrum_intensities in self.iter_spectra():
            coordinates.append(coords)

            # Convert sparse representation to full array
            full_spectrum = np.zeros(len(mzs), dtype=np.float64)

            # Find indices in the common mass axis using searchsorted
            indices = np.searchsorted(mzs, spectrum_mzs)

            # Ensure indices are within bounds
            valid_indices = indices < len(mzs)
            indices = indices[valid_indices]
            valid_intensities = spectrum_intensities[valid_indices]

            # Fill spectrum array
            full_spectrum[indices] = valid_intensities
            intensities.append(full_spectrum)

        return {
            "mzs": mzs,
            "intensities": np.array(intensities, dtype=np.float64),
            "coordinates": coordinates,
            "width": width,
            "height": height,
            "depth": depth,
        }

    def close(self) -> None:
        """Close all open file handles."""
        if hasattr(self, "ibd_file") and self.ibd_file is not None:
            self.ibd_file.close()
            self.ibd_file = None

        if hasattr(self, "parser") and self.parser is not None:
            if hasattr(self.parser, "m") and self.parser.m is not None:
                self.parser.m.close()  # type: ignore
            self.parser = None

    @property
    def n_spectra(self) -> int:
        """
        Return the total number of spectra in the dataset.

        Returns:
            Total number of spectra (efficient implementation using parser)
        """
        if not hasattr(self, "parser") or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")

        # Use parser coordinates which is efficient
        parser = cast(ImzMLParser, self.parser)
        return len(parser.coordinates)  # type: ignore

    def get_mass_bounds(self) -> Tuple[float, float]:
        """
        Get mass bounds by scanning through spectra.
        
        For ImzML files, we need to scan through spectra since metadata 
        doesn't always contain reliable m/z bounds.
        
        Returns:
            Tuple of (min_mz, max_mz)
        """
        if not hasattr(self, "parser") or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")

        parser = cast(ImzMLParser, self.parser)
        
        min_mz = float('inf')
        max_mz = float('-inf')
        
        logging.info("Scanning spectra to determine mass bounds...")
        
        # For continuous mode, we can get bounds from the first spectrum
        if self.is_continuous:
            try:
                spectrum_data = parser.getspectrum(0)
                if spectrum_data is not None and len(spectrum_data) >= 1:
                    mzs = spectrum_data[0]
                    if mzs.size > 0:
                        min_mz = float(np.min(mzs))
                        max_mz = float(np.max(mzs))
                        logging.info(f"Mass bounds from continuous mode: {min_mz:.4f} - {max_mz:.4f}")
                        return (min_mz, max_mz)
            except Exception as e:
                logging.warning(f"Error getting mass bounds from first spectrum: {e}")
        
        # For processed mode or if continuous mode failed, scan all spectra
        total_spectra = len(parser.coordinates)
        sample_size = min(100, total_spectra)  # Sample first 100 spectra for efficiency
        
        for idx in range(sample_size):
            try:
                spectrum_data = parser.getspectrum(idx)
                if spectrum_data is not None and len(spectrum_data) >= 1:
                    mzs = spectrum_data[0]
                    if mzs.size > 0:
                        spec_min = float(np.min(mzs))
                        spec_max = float(np.max(mzs))
                        min_mz = min(min_mz, spec_min)
                        max_mz = max(max_mz, spec_max)
            except Exception as e:
                logging.warning(f"Error processing spectrum {idx} for mass bounds: {e}")
                continue
        
        if min_mz == float('inf') or max_mz == float('-inf'):
            # Fallback to common mass axis if available
            try:
                mass_axis = self.get_common_mass_axis()
                if len(mass_axis) > 0:
                    min_mz = float(np.min(mass_axis))
                    max_mz = float(np.max(mass_axis))
                else:
                    raise ValueError("No valid m/z values found")
            except Exception:
                raise ValueError("Could not determine mass bounds from ImzML file")
        
        logging.info(f"Determined mass bounds: {min_mz:.4f} - {max_mz:.4f}")
        return (min_mz, max_mz)

    def get_spatial_bounds(self) -> Dict[str, int]:
        """
        Get spatial bounds from coordinates.
        
        Returns:
            Dictionary with keys: min_x, max_x, min_y, max_y
        """
        if not hasattr(self, "parser") or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")

        parser = cast(ImzMLParser, self.parser)
        coordinates = parser.coordinates
        
        if not coordinates:
            raise ValueError("No coordinates found in ImzML file")
        
        # Extract x and y coordinates
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        bounds = {
            'min_x': min(x_coords) - 1,  # Convert to 0-based indexing
            'max_x': max(x_coords) - 1,
            'min_y': min(y_coords) - 1,
            'max_y': max(y_coords) - 1
        }
        
        logging.info(f"Spatial bounds: x=[{bounds['min_x']}, {bounds['max_x']}], "
                    f"y=[{bounds['min_y']}, {bounds['max_y']}]")
        
        return bounds

    def iter_spectra_buffered(self, buffer_pool: 'SpectrumBufferPool') -> Generator['SpectrumBuffer', None, None]:
        """
        Iterate through spectra using pre-allocated buffers for zero-copy operations.
        
        Args:
            buffer_pool: Pool of pre-allocated spectrum buffers
            
        Yields:
            SpectrumBuffer: Buffer containing spectrum data
        """
        if not hasattr(self, "parser") or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")

        parser = cast(ImzMLParser, self.parser)
        total_spectra = len(parser.coordinates)
        
        for idx in range(total_spectra):
            try:
                # Get spectrum data from parser
                spectrum_data = parser.getspectrum(idx)
                if spectrum_data is None or len(spectrum_data) < 2:
                    continue
                
                mzs = spectrum_data[0]
                intensities = spectrum_data[1]
                
                if mzs.size == 0 or intensities.size == 0:
                    continue
                
                # Get coordinates (convert to 0-based)
                coords_1based = parser.coordinates[idx]
                coords = (coords_1based[0] - 1, coords_1based[1] - 1, coords_1based[2] - 1)
                
                # Get buffer from pool
                buffer = buffer_pool.get_buffer()
                
                # Fill buffer with spectrum data
                buffer.coords = coords
                buffer.fill(mzs, intensities.astype(np.float64))
                
                yield buffer
                
            except Exception as e:
                logging.warning(f"Error processing spectrum {idx}: {e}")
                continue
