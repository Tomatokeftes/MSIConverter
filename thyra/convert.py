# thyra/convert.py
import logging
import traceback
import warnings
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .core.base_converter import PixelSizeSource
from .core.registry import detect_format, get_converter_class, get_reader_class
from .errors import ConversionRefused
from .utils.windows_paths import prepare_zarr_output_path

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r"Accession IMS:1000046.*",  # ignore UserWarning
    category=UserWarning,
    module=r"pyimzml.ontology.ontology",
)


def _validate_paths_parameters(
    input_path: Union[str, Path], output_path: Union[str, Path]
) -> bool:
    """Validate path parameters."""
    if not input_path or not isinstance(input_path, (str, Path)):
        logger.error("Input path must be a valid string or Path object")
        return False

    if not output_path or not isinstance(output_path, (str, Path)):
        logger.error("Output path must be a valid string or Path object")
        return False

    return True


#: Characters SpatialData allows in an element name, besides letters and
#: digits. Its rule (``spatialdata._core.validation.check_valid_name``) is
#: not part of the public API, so it is restated here rather than imported:
#: a private import that moves would take the front-door check down with
#: it, and the only cost of the two drifting is that SpatialData refuses a
#: name Thyra let through -- which is exactly today's behaviour.
_ELEMENT_NAME_EXTRA_CHARS = "_-."


def dataset_id_problem(dataset_id: Any) -> Optional[str]:
    """Why ``dataset_id`` cannot name SpatialData elements, or ``None``.

    Every element key in the store is built from this id -- ``<id>_z0``,
    ``<id>_z0_tic``, ``<id>_pixels`` -- so SpatialData's naming rule
    applies to it. It used to be checked for emptiness only, and an id
    with a space or a slash was refused by SpatialData at ``_save_output``,
    after both passes over the source had run (issue #250). The suffixes
    Thyra appends are all legal characters, so an id that passes here
    yields keys that pass there.
    """
    if not isinstance(dataset_id, str):
        return f"Dataset ID must be a string, not {type(dataset_id).__name__}"
    if not dataset_id.strip():
        return "Dataset ID must be a non-empty string"
    if dataset_id in (".", ".."):
        return f"Dataset ID cannot be {dataset_id!r}"
    if dataset_id.startswith("__"):
        return "Dataset ID cannot start with '__'"
    bad = sorted(
        {c for c in dataset_id if not (c.isalnum() or c in _ELEMENT_NAME_EXTRA_CHARS)}
    )
    if bad:
        return (
            "Dataset ID names every element in the store, so it may hold only "
            "letters, digits, underscores, dots and hyphens; "
            f"{dataset_id!r} also holds " + ", ".join(repr(c) for c in bad)
        )
    return None


def _validate_string_parameters(format_type: str, dataset_id: str) -> bool:
    """Validate string parameters."""
    if not isinstance(format_type, str) or not format_type.strip():
        logger.error("Format type must be a non-empty string")
        return False

    problem = dataset_id_problem(dataset_id)
    if problem is not None:
        logger.error(problem)
        return False

    return True


def _validate_numeric_parameters(
    pixel_size_um: Optional[float], z_spacing_um: Optional[float] = None
) -> bool:
    """Validate numeric parameters."""
    # `<= 0` is False for NaN and for +infinity, so both used to pass every
    # one of these guards and reach the store as the dataset's pixel size or
    # z spacing. The API is validated as well as the CLI because a caller
    # can hand these in directly (issue #231).
    if pixel_size_um is not None and (
        not isinstance(pixel_size_um, (int, float))
        or not isfinite(pixel_size_um)
        or pixel_size_um <= 0
    ):
        logger.error("Pixel size must be a finite positive number")
        return False

    if z_spacing_um is not None and (
        not isinstance(z_spacing_um, (int, float))
        or not isfinite(z_spacing_um)
        or z_spacing_um <= 0
    ):
        logger.error("Z spacing must be a finite positive number")
        return False

    return True


def _validate_input_parameters(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    format_type: str,
    dataset_id: str,
    pixel_size_um: Optional[float],
    z_spacing_um: Optional[float] = None,
) -> bool:
    """Validate all input parameters for convert_msi function."""
    return (
        _validate_paths_parameters(input_path, output_path)
        and _validate_string_parameters(format_type, dataset_id)
        and _validate_numeric_parameters(pixel_size_um, z_spacing_um)
    )


def _validate_paths(input_path: Path, output_path: Path) -> bool:
    """Validate that input exists and output doesn't exist."""
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return False

    if output_path.exists():
        logger.error(f"Destination {output_path} already exists.")
        return False

    return True


def _create_reader(
    input_path: Path,
    reader_options: Optional[Dict[str, Any]] = None,
    lossless_tables: Optional[List[str]] = None,
) -> Tuple[Any, str]:
    """Create and return a reader for the input format.

    Args:
        input_path: Path to the input MSI data
        reader_options: Optional format-specific reader options (e.g., calibration settings)
        lossless_tables: Names the sibling tables being written that need
            the summed spectrum to keep all of the ion current
            (``"mobility grid"``, ``"demultiplexed MS/MS"``), so each of
            them adds back up to the summed one exactly; empty when none.

    Returns:
        Tuple of (reader instance, detected format string)
    """
    input_format = detect_format(input_path)
    logger.info(f"Detected format: {input_format}")
    reader_class = get_reader_class(input_format)
    logger.info(f"Using reader: {reader_class.__name__}")

    # Pass reader options to the reader if provided
    options = dict(reader_options or {})
    if lossless_tables:
        _force_scan_sum(input_format, options, lossless_tables)
    return reader_class(input_path, **options), input_format


def _force_scan_sum(
    input_format: str, options: Dict[str, Any], wanted: List[str]
) -> None:
    """Say out loud when a sibling table will not add up to the summed one.

    A sibling table -- the mobility grid, the demultiplexed MS/MS table --
    is built from the raw scans, so it reproduces the summed table (the
    grid's marginal over channels, the split's blocks added back up) only
    when that table was built from the same scans. The default
    ``scan_sum`` is built from them, so nothing needs switching; the
    vendor centroid is a peak-picked spectrum over the same ramp that
    keeps only the current inside the peaks it picks (87-96% on measured
    acquisitions), so an explicit ``--tdf-spectrum vendor_centroid`` is
    the caller saying they want the mismatch, which the sibling's ``uns``
    block then records. It is said at WARNING rather than overridden.

    Said before the reader is opened, because the spectrum mode binds
    into the vendor handle -- which is also why it cannot yet know
    whether the source is a PASEF acquisition at all. So it names the
    tables that *would* carry the mismatch; whether each is written is
    then said by name when the converter plans it.
    """
    mode = options.get("tdf_spectrum")
    if mode is None or mode == "scan_sum":
        return
    if input_format != "bruker":
        # The option reaches no other reader, so nothing about the summed
        # spectrum changed and there is no mismatch to announce. The CLI
        # has already said the flag is ignored here (issue #260).
        return
    subject = " and ".join(f"a {name} table" for name in wanted)
    logger.warning(
        "%s %s written with --tdf-spectrum %s. Such a table reads raw "
        "scans, so it will not add back up to the summed table; its uns "
        "block records by how much.",
        subject[0].upper() + subject[1:],
        "is" if len(wanted) == 1 else "are",
        mode,
    )


def _lossless_spectrum_for(kwargs: Dict[str, Any]) -> List[str]:
    """Which sibling tables, if any, need the lossless summed spectrum.

    ``msms_table`` defaults to *on* in the converter (design decision D2),
    and the CLI forwards the keyword only when the flag was given, so
    reading it as off-by-default meant the common path -- the default-on
    table -- never reached :func:`_force_scan_sum` and the mismatch it
    warns about went unsaid (issue #253). The default here is the
    converter's own.
    """
    wanted = []
    if kwargs.get("mobility_grid", False):
        wanted.append("mobility grid")
    if kwargs.get("msms_table", True):
        wanted.append("demultiplexed MS/MS")
    return wanted


def _determine_pixel_size(
    reader: Any, pixel_size_um: Optional[float], input_format: str
) -> Tuple[float, PixelSizeSource, Dict[str, Any]]:
    """Determine pixel size either from metadata or user input."""
    if pixel_size_um is not None:
        # Manual pixel size was provided
        pixel_size_detection_info = {
            "method": "manual",
            "source_format": input_format,
            "detection_successful": False,
            "note": "Pixel size manually specified via --pixel-size parameter",
        }
        return pixel_size_um, PixelSizeSource.USER_PROVIDED, pixel_size_detection_info

    # Attempt automatic detection
    logger.info("Attempting automatic pixel size detection...")
    essential_metadata = reader.get_essential_metadata()

    if essential_metadata.pixel_size is None:
        logger.error("Pixel size not found in metadata")
        logger.error("Use --pixel-size parameter (e.g., --pixel-size 25)")
        raise ConversionRefused("Pixel size not found in metadata")

    final_pixel_size = essential_metadata.pixel_size[0]  # Use X size
    logger.info(f"Detected pixel size: {final_pixel_size:.1f} um")

    pixel_size_detection_info = {
        "method": "automatic",
        "detected_x_um": float(essential_metadata.pixel_size[0]),
        "detected_y_um": float(essential_metadata.pixel_size[1]),
        "source_format": input_format,
        "detection_successful": True,
        "note": "Pixel size automatically detected from source metadata",
    }

    return final_pixel_size, PixelSizeSource.AUTO_DETECTED, pixel_size_detection_info


def _create_converter(
    format_type: str,
    reader: Any,
    output_path: Path,
    dataset_id: str,
    pixel_size_um: float,
    pixel_size_source: PixelSizeSource,
    handle_3d: bool,
    pixel_size_detection_info: Dict[str, Any],
    resampling_config: Optional[Dict[str, Any]] = None,
    include_optical: bool = True,
    apply_optical_alignment: bool = True,
    streaming: Union[bool, Literal["auto"]] = "auto",
    z_spacing_um: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """Create and return a converter for the specified format.

    ``streaming`` used to choose between an in-memory converter and a
    streaming one, on request or on an estimated size. There is one
    converter now and it streams (design decision D11), so the argument
    selects nothing; it is accepted so existing calls keep working, and
    ``False`` is answered with a note rather than silently.
    """
    if streaming is False:
        logger.warning(
            "streaming=False asks for the in-memory converter, which was folded "
            "into the streaming one in v3.23: every conversion makes two passes "
            "over the source and never holds the matrix in memory. The output "
            "is the same; the argument no longer selects anything."
        )

    # Deliberately outside any try. This is often the first call that
    # builds the reader's parser, so it is the call a refused file fails on
    # -- and swallowing it once logged the reason at DEBUG, returned False,
    # and let the converter parse the whole file a second time before
    # failing the same way. On a 2.1 GB imzML that is about two minutes of
    # apparent progress with the real reason invisible. The converter's
    # own constructor extracts this metadata inside a try that logs at
    # DEBUG, which is exactly the swallow.
    reader.get_essential_metadata()

    converter_kwargs = {
        "dataset_id": dataset_id,
        "pixel_size_um": pixel_size_um,
        "pixel_size_source": pixel_size_source,
        "handle_3d": handle_3d,
        "z_spacing_um": z_spacing_um,
        "pixel_size_detection_info": pixel_size_detection_info,
        "resampling_config": resampling_config,
        "include_optical": include_optical,
        "apply_optical_alignment": apply_optical_alignment,
        **kwargs,
    }
    converter_class = _resolve_converter_class(format_type)
    return converter_class(reader, output_path, **converter_kwargs)


def _resolve_converter_class(format_type: str) -> Any:
    """The converter class for ``format_type``, with the SpatialData hint."""
    try:
        converter_class = get_converter_class(format_type.lower())
        logger.info(f"Using converter: {converter_class.__name__}")
        return converter_class
    except ValueError as e:
        if "spatialdata" in format_type.lower():
            logger.error(
                "SpatialData converter is not available due to dependency issues."
            )
            logger.error("This is commonly caused by zarr version incompatibility.")
            logger.error("Try upgrading your dependencies:")
            logger.error("  pip install --upgrade anndata spatialdata zarr")
            logger.error("Or create a fresh environment with compatible versions.")
            raise ConversionRefused("SpatialData converter unavailable") from e
        else:
            raise e


def _perform_conversion_with_cleanup(converter: Any, reader: Any) -> bool:
    """Perform the conversion and handle reader cleanup."""
    try:
        logger.info("Starting conversion...")
        result = converter.convert()
        logger.info(f"Conversion {'completed successfully' if result else 'failed'}")
        return bool(result)
    finally:
        if hasattr(reader, "close"):
            reader.close()


def convert_msi(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    format_type: str = "spatialdata",
    dataset_id: str = "msi_dataset",
    pixel_size_um: Optional[float] = None,
    handle_3d: bool = False,
    z_spacing_um: Optional[float] = None,
    resampling_config: Optional[Dict[str, Any]] = None,
    reader_options: Optional[Dict[str, Any]] = None,
    include_optical: bool = True,
    apply_optical_alignment: bool = True,
    streaming: Union[bool, Literal["auto"]] = "auto",
    region: Optional[Union[int, str]] = None,
    **kwargs: Any,
) -> bool:
    """Convert MSI data to the specified format.

    Provides automatic pixel size detection from metadata or
    accepts user-specified values.

    Args:
        input_path: Path to input MSI data file or directory
        output_path: Path for output file
        format_type: Output format type (default: "spatialdata")
        dataset_id: Identifier for the dataset
        pixel_size_um: In-plane pixel size in micrometers (None for auto)
        handle_3d: Whether to process as 3D data (default: False)
        z_spacing_um: Distance between consecutive slices in micrometers.
            Only meaningful together with ``handle_3d=True``; ignored
            (with a warning) otherwise. ``None`` (default) means nothing
            supplied one, in which case the in-plane pitch is reused and
            the store records ``z_spacing_source="assumed_isotropic"``
            so the guess is not mistaken for a measurement. Supply it
            whenever the section thickness is known -- the in-plane
            pitch matches it only by coincidence.
        resampling_config: Optional resampling configuration
        reader_options: Optional format-specific reader options:
            - intensity_threshold: float - Minimum intensity
              to include. Default: None (no filtering).
            - use_recalibrated_state: bool - For Bruker data,
              use active/recalibrated calibration (default True).
            - tdf_spectrum: "scan_sum" | "vendor_centroid" - For Bruker
              TDF (TIMS) data, how a frame's mobility scans are collapsed
              into one spectrum per pixel (default "scan_sum").
            - use_centroid: bool - For Waters .raw, whether MassLynx
              hands back the vendor centroid (True) or the profile
              trace (False). Default None: the profile trace on a
              SELECT SERIES MRT, the vendor centroid on every other
              Waters instrument. The CLI spells it --waters-spectrum.
        **kwargs: Forwarded to the converter. ``write_mobility_table``
            (default True) writes the mobility-resolved sibling table when
            the source shares one set of (m/z, mobility) features across
            pixels, e.g. an imzML export with a mobility array.
            ``mobility_heatmap`` (default True) stores the mean
            mass-mobility frame on the summed table as
            ``uns["mobility_heatmap"]`` whenever the source has an ion
            mobility dimension (Bruker TDF, imzML with a mobility array).
            ``mobility_grid`` (default False) fills that same sibling for
            a source that carries mobility per pixel rather than as a
            shared feature axis (Bruker TDF) by binning the point cloud
            onto a common mobility grid; ``mobility_bins`` (default 256,
            the heatmap's own channel count), ``mobility_min`` and
            ``mobility_max`` size that grid. Its marginal reproduces the
            summed table exactly under the default
            ``tdf_spectrum="scan_sum"``; an explicit ``"vendor_centroid"``
            is kept, with a warning, and the mismatch recorded.
            ``msms_table`` (default True) writes the demultiplexed MS/MS
            sibling table when the source isolates several precursors per
            pixel in disjoint mobility slices (Bruker PASEF); refused with
            a reason, and nothing written, on any other source.
            - max_mass_axis_length: int - For processed-mode imzML
              converted with --no-resample, give up once the raw
              mass axis exceeds this many unique m/z values. This
              matters when the peak lists share no m/z values,
              where the raw axis grows to roughly one column per
              peak in the whole dataset. Default: 10,000,000,
              which is the limit SCiLS Lab places on the same
              quantity (2026b User Guide, p.76). Pass None for
              no limit.
            - spectrum_type: str - For imzML, declare the spectrum
              representation explicitly: 'profile' or 'centroid'
              (the full CV names are accepted too). Outranks the
              file's own MS:1000127/MS:1000128, so it can correct a
              file that declares the wrong thing; contradicting a
              declaration is logged as a warning. SCiLS Lab spells
              this --rep_type (2026b User Guide, p.81). Default:
              None, meaning detect. **Changes stored values** for
              files where it disagrees with what detection would
              have chosen, because the representation feeds
              instrument and axis-type selection.
        include_optical: Include optical images (default: True)
        apply_optical_alignment: If True (default) and the MSI source
            carries FlexImaging Area metadata, MSI elements are placed
            in optical-image pixel space at ``"global"``.  Set to
            False to keep MSI in pure micrometer coordinates -- needed
            when a downstream tool (e.g. Ousia's wizard) computes its
            own MSI-to-target registration.
        streaming: Kept for compatibility; selects nothing. Every
            conversion streams -- two passes over the source into
            memory-mapped arrays, the matrix never held in RAM -- since
            the in-memory converter was folded in (v3.23). ``False`` is
            accepted with a warning.
        region: For multi-region datasets (e.g. Bruker timsTOF),
            select a specific region. Accepts an int (DB
            RegionNumber) or a str (matched against .mis Area
            Name, falling back to integer parse). None (default)
            converts all regions. Passed to the reader as
            reader_options["region"].
        **kwargs: Additional keyword arguments

    Returns:
        True if conversion was successful, False otherwise
    """
    # Validate input parameters
    if not _validate_input_parameters(
        input_path,
        output_path,
        format_type,
        dataset_id,
        pixel_size_um,
        z_spacing_um,
    ):
        return False

    # A z spacing without --handle-3d changes nothing: the 2D route
    # writes one image per slice and never builds a z axis. Say so rather
    # than accepting it silently -- a caller who set it believes their
    # volume is calibrated.
    if z_spacing_um is not None and not handle_3d:
        logger.warning(
            "z_spacing_um=%g was given without handle_3d=True and will be "
            "ignored: without 3D handling each slice is written as its own "
            "2D image and there is no z axis to space out. Pass "
            "handle_3d=True (--handle-3d) to build a volume.",
            z_spacing_um,
        )

    # Convert to Path objects and validate
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    logger.info(f"Processing input file: {input_path}")

    if not _validate_paths(input_path, output_path):
        return False

    # A Zarr store's deepest key sits far below the path the caller named,
    # so a legal-looking output path can still blow the Windows 260
    # character limit part-way through the write. No-op elsewhere.
    output_path = prepare_zarr_output_path(output_path, dataset_id)

    # Merge region into reader_options if provided
    if region is not None:
        reader_options = dict(reader_options or {})
        reader_options["region"] = region

    reader = None
    try:
        # Create reader with format-specific options. A mobility grid
        # table decides the summed spectrum's semantics, so it has to be
        # known before the reader is opened -- the choice is bound into
        # the SDK handle.
        reader, input_format = _create_reader(
            input_path,
            reader_options,
            lossless_tables=_lossless_spectrum_for(kwargs),
        )

        # Determine pixel size
        final_pixel_size, pixel_size_source, pixel_size_detection_info = (
            _determine_pixel_size(reader, pixel_size_um, input_format)
        )

        # Create converter
        converter = _create_converter(
            format_type,
            reader,
            output_path,
            dataset_id,
            final_pixel_size,
            pixel_size_source,
            handle_3d,
            pixel_size_detection_info,
            resampling_config,
            include_optical=include_optical,
            apply_optical_alignment=apply_optical_alignment,
            streaming=streaming,
            z_spacing_um=z_spacing_um,
            **kwargs,
        )

        # Perform conversion with cleanup
        return _perform_conversion_with_cleanup(converter, reader)

    except ConversionRefused as e:
        # A refusal Thyra wrote: the message is the whole explanation, and
        # a traceback in front of it reads like a crash the code did not
        # plan for. The traceback is still there under --log-level DEBUG,
        # for the cases where the refusal itself is the surprise (#234).
        #
        # ``str(e)``, never ``e``: a log handler that retains records --
        # pytest's capture, Ousia's per-session log -- would otherwise hold
        # the exception, its traceback and every frame in it, which pins
        # the reader and leaves its file open (see issue #249).
        logger.error("%s", str(e))
        logger.debug("Refusal raised at:\n%s", traceback.format_exc())
        return False

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        logger.error(f"Detailed traceback:\n{traceback.format_exc()}")
        return False

    finally:
        # Only the conversion itself closed the reader, so anything that
        # failed before it -- "Pixel size not found in metadata" is the
        # common one -- left the source open until the garbage collector
        # happened to reach it. On Windows that holds a lock on the file
        # the user is about to retry with. Every reader's close() is
        # idempotent, so the conversion's own close is not disturbed.
        if reader is not None and hasattr(reader, "close"):
            try:
                reader.close()
            except Exception as close_error:  # pragma: no cover - defensive
                logger.debug("Could not close the reader: %s", str(close_error))
