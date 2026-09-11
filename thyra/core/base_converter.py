import logging
import math
from abc import ABC, abstractmethod
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from tqdm import tqdm

from ..errors import ConversionRefused
from .base_reader import BaseMSIReader

if TYPE_CHECKING:
    from ..metadata.types import EssentialMetadata

logger = logging.getLogger(__name__)


class PixelSizeSource(Enum):
    """Enum to track how pixel size was determined."""

    DEFAULT = "default"  # Using default 1.0 (fallback)
    USER_PROVIDED = "manual"  # User explicitly provided via parameter
    AUTO_DETECTED = "automatic"  # Detected from metadata


class ZSpacingSource(Enum):
    """How the slice-to-slice (z) spacing was determined.

    Deliberately separate from :class:`PixelSizeSource`: the in-plane
    pitch is a property of the raster and is nearly always recoverable
    from the file, while the z spacing is a property of how the sections
    were physically cut and usually is not recoverable at all.

    ``ASSUMED_ISOTROPIC`` is the one that matters. It records that
    nothing supplied a spacing and the in-plane pitch was reused, which
    is what this code has always done -- the point of naming it is that
    a consumer can now tell a measured spacing from a guessed one
    instead of receiving both as bare numbers.
    """

    ASSUMED_ISOTROPIC = "assumed_isotropic"  # Nothing supplied one; reused x pitch
    USER_PROVIDED = "manual"  # Explicit z_spacing_um argument
    AUTO_DETECTED = "automatic"  # Reported by the source metadata


class BaseMSIConverter(ABC):
    """Base class for MSI data converters with shared functionality.

    Implements common processing steps while allowing format-specific
    customization.
    """

    def __init__(
        self,
        reader: BaseMSIReader,
        output_path: Union[str, Path, PathLike[str]],
        dataset_id: str = "msi_dataset",
        pixel_size_um: float = 1.0,
        pixel_size_source: PixelSizeSource = PixelSizeSource.DEFAULT,
        compression_level: int = 5,
        handle_3d: bool = False,
        z_spacing_um: Optional[float] = None,
        **kwargs: Any,
    ):
        """Initialize the MSI converter.

        Args:
            reader: MSI data reader instance
            output_path: Path for output file
            dataset_id: Identifier for the dataset
            pixel_size_um: In-plane pixel pitch in micrometers **along
                x**. The y pitch starts equal to it and is replaced by
                the detected one when the source declares an anisotropic
                raster -- see :attr:`pixel_size_y_um`.
            pixel_size_source: How pixel size was determined
            compression_level: Compression level for output
            handle_3d: Whether to process as 3D data
            z_spacing_um: Distance between consecutive slices in
                micrometers.  Only meaningful with ``handle_3d=True``.
                ``None`` (default) means "nobody said", which falls back
                to the in-plane pitch and records that it was assumed --
                see :meth:`_resolve_z_spacing`.
            **kwargs: Additional keyword arguments

        Raises:
            ConversionRefused: If ``z_spacing_um`` is given and is not
                positive, a non-number included.
        """
        if z_spacing_um is not None and (
            not isinstance(z_spacing_um, (int, float)) or z_spacing_um <= 0
        ):
            raise ConversionRefused(
                f"z_spacing_um must be positive, got {z_spacing_um}"
            )

        self.reader = reader
        self.output_path = Path(output_path)
        self.dataset_id = dataset_id
        self.pixel_size_um = pixel_size_um
        #: The pitch along y, in micrometers. A raster is usually square
        #: and then this is :attr:`pixel_size_um`; a DESI method with
        #: ``DesiXStep != DesiYStep`` is not, and until issue #228 only
        #: the ``msi_metadata`` block recorded the difference while the
        #: root attrs, the affines, the pixel footprints and
        #: ``obs["spatial_y"]`` all carried the x pitch on both axes --
        #: so an anisotropic raster rendered squashed by y/x and the
        #: store contradicted itself. Settled from the detected pair (see
        #: ``BaseSpatialDataConverter.__init__`` and
        #: :meth:`_initialize_conversion`); a pitch the caller states with
        #: ``--pixel-size`` applies to both axes, which is how someone
        #: declares a raster square whatever the file says.
        self.pixel_size_y_um: float = float(pixel_size_um)
        self.pixel_size_source = pixel_size_source
        self.compression_level = compression_level
        self.handle_3d = handle_3d

        # Settled in _resolve_z_spacing() once metadata is loaded, because
        # the fallback is the in-plane pitch and that may itself still be
        # the placeholder 1.0 until auto-detection has run.
        self._z_spacing_um_arg = (
            float(z_spacing_um) if z_spacing_um is not None else None
        )
        self.z_spacing_um: float = float(pixel_size_um)
        self.z_spacing_source = ZSpacingSource.ASSUMED_ISOTROPIC
        self.options: Dict[str, Any] = kwargs
        self._common_mass_axis: Optional[NDArray[np.float64]] = None
        # Identity-mapping cache for _map_mass_to_indices: shared-axis
        # readers hand every spectrum the very m/z array the common axis
        # was built from, so the exact-match search is the identity map.
        self._identity_mass_indices: Optional[NDArray[np.int_]] = None
        self._axis_strictly_increasing: Optional[bool] = None
        self._dimensions: Optional[Tuple[int, int, int]] = None
        self._metadata: Optional[dict[str, Any]] = None
        from ..config import DEFAULT_BUFFER_SIZE

        self._buffer_size = DEFAULT_BUFFER_SIZE

        # Essential metadata properties (loaded during initialization)
        self._coordinate_bounds: Optional[Tuple[float, float, float, float]] = None
        self._n_spectra: Optional[int] = None
        self._estimated_memory_gb: Optional[float] = None

    def convert(self) -> bool:
        """Template method defining the conversion workflow.

        An interrupt is a failed conversion, not a separate kind of exit.
        ``KeyboardInterrupt`` is caught here so the caller gets ``False``
        and the CLI's ``_handle_post_conversion`` runs, renaming the
        partial store to ``.failed`` and leaving the output path free for
        a retry -- which is what ``docs/cli.md`` promises of *any* failed
        conversion. Before this it propagated past ``except Exception``,
        click printed ``Aborted!``, and an interrupted run left an
        unopenable store where a finished one belongs, a blocked retry,
        and (on a whole-slide mobility grid) 18 GB of scratch memmaps
        nothing came back to remove (issue #245).

        Returns:
        --------
        bool: True if conversion was successful, False otherwise.
        """
        try:
            self._initialize_conversion()
            data_structures = self._create_data_structures()
            self._process_spectra(data_structures)
            self._finalize_data(data_structures)
            success = self._save_output(data_structures)

            return success
        except ConversionRefused as e:
            # The other half of issue #234: convert_msi's catch-all was
            # not the only one. This one runs first for anything raised
            # inside the workflow -- the axis plan, the reader's refusal
            # of a frame -- so a refusal reaching it has to be presented
            # as a refusal here too, or nothing downstream ever sees it.
            #
            # ``str(e)`` rather than ``e`` for the reason issue #249 gives:
            # a retained log record would hold the exception's traceback
            # and through it the reader, whose files then stay open.
            import traceback

            logger.error("%s", str(e))
            logger.debug("Refusal raised at:\n%s", traceback.format_exc())
            return False
        except KeyboardInterrupt:
            # No ``as e``: the exception is cleared when this block ends,
            # and with it the traceback whose frames still hold the tables
            # built over the CSC scratch memmaps. Windows will not delete a
            # mapped file, so letting those frames go here is what lets the
            # scratch directories go in the converter's own cleanup.
            logger.error(
                "Conversion interrupted. Nothing usable was written: any "
                "partial store is moved aside and the scratch directories "
                "are removed."
            )
            return False
        except Exception as e:
            logger.error(f"Error during conversion: {e}")
            import traceback

            logger.error(f"Detailed traceback:\n{traceback.format_exc()}")
            return False
        finally:
            self.reader.close()

    def _initialize_conversion(self) -> None:
        """Initialize conversion by loading essential metadata first, then other data."""
        logger.info("Loading essential dataset information...")
        try:
            # Load essential metadata first (fast, single query for Bruker)
            essential = self.reader.get_essential_metadata()

            self._dimensions = essential.dimensions
            if any(d <= 0 for d in self._dimensions):
                raise ConversionRefused(
                    f"Invalid dimensions: {self._dimensions}. All dimensions "
                    f"must be positive."
                )

            # Store essential metadata for use throughout conversion
            self._coordinate_bounds = essential.coordinate_bounds
            self._n_spectra = essential.n_spectra
            self._estimated_memory_gb = essential.estimated_memory_gb

            # Override pixel size only if using default value and metadata
            # is available
            self._adopt_detected_pixel_size(essential)

            # After pixel size, because the fallback is the pixel size.
            self._resolve_z_spacing(essential)

            # Load mass axis separately (still expensive operation)
            self._common_mass_axis = self.reader.get_common_mass_axis()
            if len(self._common_mass_axis) == 0:
                raise ConversionRefused(
                    "Common mass axis is empty. Cannot proceed with " "conversion."
                )

            # Only load comprehensive metadata if needed (lazy loading)
            self._metadata = None  # Will be loaded on demand

            logger.info(f"Dataset dimensions: {self._dimensions}")
            logger.info(f"Coordinate bounds: {self._coordinate_bounds}")
            logger.info(f"Total spectra: {self._n_spectra}")
            logger.info(f"Estimated memory: {self._estimated_memory_gb:.2f} GB")
            logger.info(f"Common mass axis length: {len(self._common_mass_axis)}")
        except ConversionRefused:
            # Said once, by whoever catches it. Re-prefixing a refusal
            # with the stage it came from adds nothing a user can act on.
            raise
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    @property
    def _is_volume(self) -> bool:
        """True when this conversion writes a real multi-slice volume.

        ``handle_3d`` alone is not enough: it says the caller asked for a
        volume, not that the source has one, and a single-slice
        acquisition converted with ``handle_3d=True`` still takes the 2D
        branch of ``_create_tic_image``. Only the combination has a z axis
        to space out. (The flag used to be forced on by
        ``SpatialData3DConverter.__init__``; that class went when the
        converters were folded into one, design decision D11, and the
        caller's own argument is now the only thing that sets it.)
        """
        return bool(self.handle_3d and self._dimensions and self._dimensions[2] > 1)

    def _adopt_detected_pixel_size(self, essential: "EssentialMetadata") -> None:
        """Take the source's pitch when nobody supplied one, on both axes.

        ``essential.pixel_size`` is an ``(x, y)`` pair and both halves are
        kept: taking only ``[0]`` is how an anisotropic raster came to be
        rendered square (issue #228). A pitch the caller stated applies to
        both axes and is not overridden here.
        """
        if self.pixel_size_source == PixelSizeSource.DEFAULT and essential.pixel_size:
            old_size = self.pixel_size_um
            self.pixel_size_um = essential.pixel_size[0]
            self.pixel_size_y_um = float(essential.pixel_size[1])
            self.pixel_size_source = PixelSizeSource.AUTO_DETECTED
            logger.info(
                f"Auto-detected pixel size: {self.pixel_size_um} um "
                f"(was default: {old_size} um)"
            )
        elif self.pixel_size_source == PixelSizeSource.USER_PROVIDED:
            logger.info(f"Using user-specified pixel size: {self.pixel_size_um} um")
        self._log_anisotropic_raster()

    def _log_anisotropic_raster(self) -> None:
        """Say so, once, when the two in-plane pitches differ.

        Only a genuine difference: the comparison is exact up to float
        noise, because a detector that computes the pitch by dividing an
        extent by a position count can land a few ULP apart on two axes
        of a square raster and that is not news. A real anisotropy is a
        percent or more -- a DESI method with ``DesiXStep != DesiYStep``.
        """
        if math.isclose(
            float(self.pixel_size_um), float(self.pixel_size_y_um), rel_tol=1e-9
        ):
            return
        logger.info(
            "Anisotropic raster: %g um in x, %g um in y. Both pitches are "
            "carried through the store -- the root attrs, the element "
            "transforms, the pixel footprints, obs['spatial_x']/['spatial_y'] "
            "and the msi_metadata block. Pass --pixel-size to declare the "
            "raster square instead.",
            self.pixel_size_um,
            self.pixel_size_y_um,
        )

    def _resolve_z_spacing(self, essential: "EssentialMetadata") -> None:
        """Settle the slice-to-slice spacing, and record where it came from.

        Precedence mirrors the in-plane pitch: an explicit argument wins,
        then whatever the source could report, then a fallback.

        The fallback is the in-plane pitch, which is what every 3D volume
        this project has written so far already used. What changes is that
        it is no longer *stated as a fact*: ``z_spacing_source`` marks it
        as an assumption, the store carries that marker, and a volume
        built on a guess is loud about it in the log.

        Reusing the in-plane pitch asserts that consecutive slices sit
        exactly one pixel-width apart. Sections are cut by a microtome and
        the raster is set by the stage, so the two agree only by
        coincidence -- for 3D MSI usually not at all, and often by an
        order of magnitude. A consumer reading the volume in micrometres
        renders the stack at the wrong depth.

        On an anisotropic raster there is no single in-plane pitch to
        reuse, and the x one is taken. That choice is arbitrary, which is
        the point: nothing about the raster predicts the section
        thickness either way, and ``z_spacing_source`` already marks the
        number as assumed rather than measured. Supply ``--z-spacing``
        and the question does not arise.

        Args:
            essential: Metadata for the dataset being converted.
        """
        if self._z_spacing_um_arg is not None:
            self.z_spacing_um = self._z_spacing_um_arg
            self.z_spacing_source = ZSpacingSource.USER_PROVIDED
        elif getattr(essential, "z_spacing_um", None) is not None:
            self.z_spacing_um = float(essential.z_spacing_um)
            self.z_spacing_source = ZSpacingSource.AUTO_DETECTED
        else:
            self.z_spacing_um = float(self.pixel_size_um)
            self.z_spacing_source = ZSpacingSource.ASSUMED_ISOTROPIC

        self._log_z_spacing()

    def _log_z_spacing(self) -> None:
        """Report the resolved z spacing, warning when it was assumed.

        Quiet for a 2D conversion nobody asked to make 3D: it has no
        slice-to-slice distance to get wrong, and warning about one on
        every ordinary dataset would train people to ignore the message
        on the datasets where it matters.

        A single-slice acquisition converted *with* ``--handle-3d`` is
        the exception. It is not a volume either -- one plane has no
        spacing -- but the flags were given, and docs/cli.md promises
        that ``--handle-3d`` says what it recorded and that a
        ``--z-spacing`` doing nothing "is logged as ignored rather than
        silently accepted". Both promises were skipped here, because
        ``_is_volume`` is false for ``n_z == 1`` and this method returned
        at DEBUG (issue #256).
        """
        if not self._is_volume:
            if self.handle_3d:
                self._log_single_plane_volume()
            else:
                logger.debug(
                    "Not a multi-slice volume; z spacing (%g um, %s) is unused.",
                    self.z_spacing_um,
                    self.z_spacing_source.value,
                )
            return

        if self.z_spacing_source is ZSpacingSource.ASSUMED_ISOTROPIC:
            logger.warning(
                "No z spacing was supplied, so this volume assumes slices sit "
                "one in-plane pixel apart (%g um). That is a guess, not a "
                "measurement: section thickness is set by the microtome, not "
                "by the raster, so any consumer reading this volume in "
                "micrometres will render the stack at the wrong depth unless "
                "the two happen to coincide. Pass --z-spacing (CLI) or "
                "z_spacing_um= (API) to set it. The store records this as "
                "z_spacing_source='%s' so the assumption stays visible.",
                self.z_spacing_um,
                ZSpacingSource.ASSUMED_ISOTROPIC.value,
            )
        else:
            logger.info(
                "Using %s z spacing: %g um (in-plane pitch %g um)",
                self.z_spacing_source.value,
                self.z_spacing_um,
                self.pixel_size_um,
            )

    def _log_single_plane_volume(self) -> None:
        """Say that 3D handling found one plane, and what that costs.

        The conversion is not refused: a one-plane volume is a legal
        store and the flag is often part of a batch script that also
        converts real stacks. What changes is that it says so, and names
        the ``--z-spacing`` it is dropping, instead of writing a store
        whose keys differ from the 2D run's with nothing in the log.
        """
        planes = self._dimensions[2] if self._dimensions else 0
        if self._z_spacing_um_arg is not None:
            logger.warning(
                "3D handling was asked for and this acquisition has %d plane, "
                "so the volume has no z extent and the z spacing of %g um is "
                "ignored: it is neither applied nor recorded. The store is "
                "written as a volume of one plane, which changes the element "
                "keys (no _z0 suffix) but not the values.",
                planes,
                self._z_spacing_um_arg,
            )
        else:
            logger.info(
                "3D handling was asked for and this acquisition has %d plane. "
                "One plane has no slice-to-slice distance, so no z spacing is "
                "recorded. The store is written as a volume of one plane, "
                "which changes the element keys (no _z0 suffix) but not the "
                "values.",
                planes,
            )

    @abstractmethod
    def _create_data_structures(self) -> Any:
        """Create format-specific data structures.

        Returns:
        --------
        Any: Format-specific data structures to be used in subsequent steps.
        """
        pass

    def _process_spectra(self, data_structures: Any) -> None:
        """Process all spectra from the reader and integrate into data structures.

        Parameters:
        -----------
        data_structures: Format-specific data containers created by
            _create_data_structures.
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized.")

        total_spectra = self._get_total_spectra_count()
        logger.info(
            f"Converting {total_spectra} spectra to "
            f"{self.__class__.__name__.replace('Converter', '')} format..."
        )

        setattr(self.reader, "_quiet_mode", True)

        # Process spectra with unified progress tracking
        with tqdm(
            total=total_spectra, desc="Converting spectra", unit="spectrum"
        ) as pbar:
            for coords, mzs, intensities in self.reader.iter_spectra(
                batch_size=self._buffer_size
            ):
                self._process_single_spectrum(data_structures, coords, mzs, intensities)
                pbar.update(1)

    def _get_total_spectra_count(self) -> int:
        """Get the total number of spectra for progress tracking.

        Uses cached essential metadata for efficient access.
        """
        # Use cached spectra count from essential metadata
        if self._n_spectra is not None:
            return self._n_spectra

        # Fallback: try reader-specific methods
        if hasattr(self.reader, "n_spectra"):
            return int(self.reader.n_spectra)

        # For ImzML readers, count coordinates
        if hasattr(self.reader, "parser") and self.reader.parser is not None:
            if hasattr(self.reader.parser, "coordinates"):
                return len(self.reader.parser.coordinates)

        # For Bruker readers, try frame count methods
        if hasattr(self.reader, "_get_frame_count"):
            return int(self.reader._get_frame_count())

        # Final fallback: calculate from dimensions
        if self._dimensions is not None:
            total_pixels = (
                self._dimensions[0] * self._dimensions[1] * self._dimensions[2]
            )
            logger.warning(
                f"Could not determine exact spectra count, estimating "
                f"{total_pixels} from dimensions"
            )
            return total_pixels

        # Should not reach here if initialization was successful
        raise ValueError(
            "Cannot determine spectra count - conversion not properly " "initialized"
        )

    def _process_single_spectrum(
        self,
        data_structures: Any,
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        """Process a single spectrum.

        Args:
            data_structures: Format-specific data containers
            coords: (x, y, z) coordinates
            mzs: m/z values
            intensities: Intensity values
        """
        # Default implementation - to be overridden by subclasses if needed
        pass

    def _finalize_data(self, data_structures: Any) -> None:
        """Perform any final processing on the data structures before saving.

        Args:
            data_structures: Format-specific data containers
        """
        # Default implementation - to be overridden by subclasses if needed
        pass

    def _get_comprehensive_metadata(self) -> Dict[str, Any]:
        """Lazy load comprehensive metadata when needed."""
        if self._metadata is None:
            logger.info("Loading comprehensive metadata...")
            comprehensive = self.reader.get_comprehensive_metadata()
            self._metadata = comprehensive.raw_metadata
        return self._metadata

    @abstractmethod
    def _save_output(self, data_structures: Any) -> bool:
        """Save the processed data to the output format.

        Args:
            data_structures: Format-specific data containers

        Returns:
            True if saving was successful, False otherwise
        """
        pass

    def add_metadata(self, metadata: Any) -> None:
        """Add comprehensive metadata to the output.

        Base implementation provides common metadata structure.
        Subclasses should override to add format-specific metadata storage.

        Args:
            metadata: Any object that can store metadata
        """
        # Get comprehensive metadata for complete information
        comprehensive_metadata = self.reader.get_comprehensive_metadata()

        # Create structured metadata dict that subclasses can use
        self._structured_metadata = {
            # Conversion metadata
            "conversion_info": {
                "dataset_id": self.dataset_id,
                "pixel_size_um": self.pixel_size_um,
                "pixel_size_y_um": self.pixel_size_y_um,
                "handle_3d": self.handle_3d,
                "compression_level": self.compression_level,
                "converter_class": self.__class__.__name__,
                "conversion_timestamp": pd.Timestamp.now().isoformat(),
            },
            # Essential metadata for quick access
            "essential_metadata": {
                "dimensions": comprehensive_metadata.essential.dimensions,
                "coordinate_bounds": (
                    comprehensive_metadata.essential.coordinate_bounds
                ),
                "mass_range": comprehensive_metadata.essential.mass_range,
                "pixel_size": comprehensive_metadata.essential.pixel_size,
                "n_spectra": comprehensive_metadata.essential.n_spectra,
                "estimated_memory_gb": (
                    comprehensive_metadata.essential.estimated_memory_gb
                ),
                "source_path": comprehensive_metadata.essential.source_path,
                "is_3d": comprehensive_metadata.essential.is_3d,
                "has_pixel_size": (comprehensive_metadata.essential.has_pixel_size),
            },
            # Format-specific metadata from source
            "format_specific_metadata": comprehensive_metadata.format_specific,
            "acquisition_parameters": (comprehensive_metadata.acquisition_params),
            "instrument_information": comprehensive_metadata.instrument_info,
            "raw_metadata": comprehensive_metadata.raw_metadata,
            # Processing statistics
            "processing_stats": {
                "total_grid_pixels": (
                    self._dimensions[0] * self._dimensions[1] * self._dimensions[2]
                    if self._dimensions
                    else 0
                ),
                "coordinate_bounds": self._coordinate_bounds,
                "estimated_memory_gb": self._estimated_memory_gb,
            },
        }

        # Subclasses should override to add this structured metadata to
        # their outputs
        logger.info(f"Base metadata structure prepared for {self.__class__.__name__}")

        # Default implementation does nothing - subclasses should override
        pass

    # --- Common Utility Methods ---

    def _create_coordinates_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame containing pixel coordinates.

        Returns:
        --------
        pd.DataFrame: DataFrame with pixel coordinates
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized.")
        n_x, n_y, n_z = self._dimensions

        coords = []
        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    pixel_idx = z * (n_y * n_x) + y * n_x + x
                    coords.append(
                        {
                            "z": z,
                            "y": y,
                            "x": x,
                            "pixel_id": str(
                                pixel_idx
                            ),  # Convert to string for compatibility
                        }
                    )

        coords_df: pd.DataFrame = pd.DataFrame(coords)
        coords_df.set_index("pixel_id", inplace=True)

        # Add spatial coordinates
        coords_df["spatial_x"] = coords_df["x"] * self.pixel_size_um
        coords_df["spatial_y"] = coords_df["y"] * self.pixel_size_y_um
        coords_df["spatial_z"] = coords_df["z"] * self.z_spacing_um

        return coords_df

    def _create_mass_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame containing mass values.

        Returns:
        --------
        pd.DataFrame: DataFrame with mass values
        """
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized.")
        var_df: DataFrame = pd.DataFrame({"mz": self._common_mass_axis})
        # Convert to string index for compatibility
        var_df["mz_str"] = var_df["mz"].astype(str)
        var_df.set_index("mz_str", inplace=True)

        return var_df

    def _get_pixel_index(self, x: int, y: int, z: int) -> int:
        """Convert 3D coordinates to a flat array index.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            Flat index
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized.")
        n_x, n_y, _ = self._dimensions
        return z * (n_y * n_x) + y * n_x + x

    def _map_mass_to_indices(self, mzs: NDArray[np.float64]) -> NDArray[np.int_]:
        """Map m/z values to indices in the common mass axis with high accuracy.

        The returned array is parallel to ``mzs`` (same length, same order),
        so callers may pair it element-for-element with the spectrum's
        intensity array.

        Args:
            mzs: Array of m/z values

        Returns:
            Array of indices in common mass axis, parallel to ``mzs``

        Raises:
            ConversionRefused: If any m/z value has no common-axis entry
                within tolerance. Without resampling the axis is built from
                the spectra themselves, so every value must match exactly; a
                near-miss means the axis and the data have diverged, and
                dropping the value silently would desync the index and
                intensity arrays downstream.
            ValueError: If the common mass axis has not been built yet. This
                one stays a plain ValueError on purpose: it is an internal
                invariant about call order, not a statement about the data,
                so it is a bug in Thyra rather than something the caller
                could have supplied differently.
        """
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized.")

        if mzs.size == 0:
            return np.array([], dtype=int)

        axis = self._common_mass_axis

        # Identity fast path. On a shared-axis reader (continuous imzML,
        # Rapiflex, Waters, PHI) every spectrum's m/z array is the very
        # array the common axis was built from, so the exact-match search
        # below returns 0..n-1 -- yet used to be re-derived by binary
        # search per spectrum, twice per streaming conversion. Equality is
        # proven, not assumed: an O(1) size check gates a full array
        # comparison, and the arange shortcut is only valid when the axis
        # is strictly increasing (searchsorted maps duplicates to their
        # first occurrence, which is not the identity).
        if mzs.size == axis.size and (mzs is axis or bool(np.array_equal(mzs, axis))):
            if self._axis_strictly_increasing is None:
                self._axis_strictly_increasing = bool(np.all(np.diff(axis) > 0))
            if self._axis_strictly_increasing:
                cached = self._identity_mass_indices
                if cached is None or cached.size != axis.size:
                    cached = np.arange(axis.size, dtype=np.intp)
                    self._identity_mass_indices = cached
                return cached

        # searchsorted returns the right-hand neighbor; the nearest axis
        # entry may sit on either side, so compare both before validating.
        right = np.clip(np.searchsorted(axis, mzs), 0, len(axis) - 1)
        left = np.maximum(right - 1, 0)
        indices = np.where(
            np.abs(axis[right] - mzs) <= np.abs(axis[left] - mzs), right, left
        )

        # Very small tolerance threshold for floating point differences
        max_diff = 1e-6
        diffs = np.abs(axis[indices] - mzs)
        bad = diffs > max_diff
        if np.any(bad):
            worst = int(np.argmax(diffs))
            raise ConversionRefused(
                f"{int(np.count_nonzero(bad))} of {mzs.size} m/z values have "
                f"no common mass axis entry within {max_diff:g} (worst: m/z "
                f"{mzs[worst]!r} is {diffs[worst]:.6g} from nearest axis "
                f"entry {axis[indices[worst]]!r}). The common mass axis does "
                f"not cover this spectrum; refusing to drop values silently "
                f"because callers pair these indices with the full intensity "
                f"array."
            )

        return indices
