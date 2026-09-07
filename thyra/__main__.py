# thyra/__main__.py

# Configure dependencies to suppress warnings BEFORE any imports
import logging  # noqa: E402
import os  # noqa: E402
import sqlite3  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Literal, Optional  # noqa: E402

import click  # noqa: E402

from thyra import __version__  # noqa: E402
from thyra.convert import convert_msi  # noqa: E402
from thyra.core.registry import detect_format  # noqa: E402
from thyra.resampling.mobility_grid import MOBILITY_CHANNELS  # noqa: E402
from thyra.utils.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Configure Dask to use new query planning (silences legacy DataFrame warning)
os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "True"

# Suppress dependency warnings at the earliest possible moment
warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="The legacy Dask DataFrame implementation is deprecated",
    category=FutureWarning,
)


def _get_calibration_states(bruker_path: Path) -> list[dict]:
    """Read calibration states from calibration.sqlite.

    Args:
        bruker_path: Path to Bruker .d directory

    Returns:
        List of calibration state dictionaries with id, datetime, and version info
    """
    cal_file = bruker_path / "calibration.sqlite"
    if not cal_file.exists():
        return []

    try:
        conn = sqlite3.connect(str(cal_file))
        cursor = conn.cursor()

        # Query calibration states
        cursor.execute(
            """
            SELECT cs.Id, ci.DateTime
            FROM CalibrationState cs
            LEFT JOIN CalibrationInfo ci ON cs.Id = ci.StateId
            ORDER BY cs.Id
            """
        )

        states = []
        for row in cursor.fetchall():
            state_id, datetime_str = row
            states.append(
                {
                    "id": state_id,
                    "datetime": datetime_str or "Unknown",
                    "version": state_id,
                }
            )

        conn.close()
        return states

    except Exception:
        return []


def _validate_basic_params(pixel_size: Optional[float], dataset_id: str) -> None:
    """Validate basic conversion parameters."""
    if pixel_size is not None and pixel_size <= 0:
        raise click.BadParameter("Pixel size must be positive", param_hint="pixel_size")
    if not dataset_id.strip():
        raise click.BadParameter("Dataset ID cannot be empty", param_hint="dataset_id")


def _validate_positive_int(value: Optional[int], param_name: str, label: str) -> None:
    """Validate that an optional int parameter is positive if provided."""
    if value is not None and value <= 0:
        raise click.BadParameter(f"{label} must be positive", param_hint=param_name)


def _validate_positive_float(
    value: Optional[float], param_name: str, label: str
) -> None:
    """Validate that an optional float parameter is positive if provided."""
    if value is not None and value <= 0:
        raise click.BadParameter(f"{label} must be positive", param_hint=param_name)


def _validate_mz_range(min_mz: Optional[float], max_mz: Optional[float]) -> None:
    """Validate that min_mz is less than max_mz when both are provided."""
    if min_mz is not None and max_mz is not None and min_mz >= max_mz:
        raise click.BadParameter("Minimum m/z must be less than maximum m/z")


def _validate_resampling_params(
    resample_bins: Optional[int],
    resample_min_mz: Optional[float],
    resample_max_mz: Optional[float],
    resample_width_at_mz: Optional[float],
    resample_reference_mz: float,
) -> None:
    """Validate resampling parameters."""
    _validate_positive_int(resample_bins, "resample_bins", "Number of resampling bins")
    _validate_positive_float(resample_min_mz, "resample_min_mz", "Minimum m/z")
    _validate_positive_float(resample_max_mz, "resample_max_mz", "Maximum m/z")
    _validate_mz_range(resample_min_mz, resample_max_mz)

    if resample_bins is not None and resample_width_at_mz is not None:
        raise click.BadParameter(
            "--resample-bins and --resample-width-at-mz are mutually exclusive"
        )

    _validate_positive_float(resample_width_at_mz, "resample_width_at_mz", "Mass width")

    if resample_reference_mz <= 0:
        raise click.BadParameter(
            "Reference m/z must be positive", param_hint="resample_reference_mz"
        )


def _validate_input_path(input: Path) -> None:
    """Validate input path and format requirements."""
    if not input.exists():
        raise click.BadParameter(f"Input path does not exist: {input}")

    if input.is_file() and input.suffix.lower() == ".imzml":
        ibd_path = input.with_suffix(".ibd")
        if not ibd_path.exists():
            raise click.BadParameter(
                "ImzML file requires corresponding .ibd file, but not "
                f"found: {ibd_path}"
            )
    elif input.is_dir() and input.suffix.lower() == ".d":
        # Several Bruker formats share the .d extension (timsTOF via
        # analysis.tsf/.tdf, solariX via peaks.sqlite), so delegate to the
        # registry's detection rather than duplicating the marker files
        # here -- its errors also carry the format-specific guidance
        # (e.g. the imzML-export fallback for a peaks-less solariX .d).
        try:
            detect_format(input)
        except ValueError as e:
            raise click.BadParameter(str(e)) from e


def _validate_output_path(output: Path) -> None:
    """Validate output path."""
    if output.exists():
        raise click.BadParameter(f"Output path already exists: {output}")


def _display_calibration_info(input: Path, use_recalibrated: bool) -> None:
    """Display calibration information for Bruker datasets.

    Note: This is informational only. Full interactive selection
    will be implemented in the future (see GitHub issue #54).
    """
    states = _get_calibration_states(input)
    if not states:
        return

    click.echo("\n" + "=" * 60)
    click.echo("Calibration Information (Display Only)")
    click.echo("=" * 60)
    for state in states:
        is_active = state["id"] == max(s["id"] for s in states)
        active_marker = " (active/will be used)" if is_active else ""
        recal_info = (
            f" - recalibrated {state['version'] - 1} times"
            if state["version"] > 1
            else ""
        )
        click.echo(
            f"  State {state['id']}: {state['datetime']}{recal_info}{active_marker}"
        )

    if use_recalibrated:
        click.echo(
            f"\nUsing active calibration state (State {max(s['id'] for s in states)})"
        )
    else:
        click.echo("\nUsing original calibration (--no-recalibrated flag set)")

    click.echo("\nNote: Interactive selection not yet available. See GitHub issue #54.")
    click.echo("=" * 60 + "\n")


def _select_bruker_dataset(input_path: Path) -> Path:
    """Prompt user to select a dataset when multiple .d folders exist.

    If the input directory contains multiple Bruker .d folders, displays
    them as a numbered list and lets the user pick one interactively.

    Args:
        input_path: The user-provided input path

    Returns:
        The selected .d folder path, or the original path if no
        selection is needed
    """
    if input_path.suffix.lower() == ".d":
        return input_path

    if not input_path.is_dir():
        return input_path

    d_folders = sorted(
        f for f in input_path.iterdir() if f.is_dir() and f.suffix.lower() == ".d"
    )

    if len(d_folders) <= 1:
        return input_path

    click.echo(f"\nFound {len(d_folders)} datasets in {input_path.name}:")
    for i, d_folder in enumerate(d_folders, 1):
        click.echo(f"  [{i}] {d_folder.name}")

    choice: int = click.prompt(
        "\nSelect dataset to convert",
        type=click.IntRange(1, len(d_folders)),
    )

    selected: Path = d_folders[choice - 1]
    click.echo(f"  -> {selected.name}\n")
    return selected


def _build_resampling_config(
    resample_method: str,
    mass_axis_type: str,
    resample_bins: Optional[int],
    resample_min_mz: Optional[float],
    resample_max_mz: Optional[float],
    resample_width_at_mz: Optional[float],
    resample_reference_mz: float,
    resample_gap_tolerance: Optional[float] = None,
) -> dict:
    """Build resampling configuration dictionary."""
    return {
        "method": resample_method,
        "axis_type": mass_axis_type,
        "target_bins": resample_bins,
        "min_mz": resample_min_mz,
        "max_mz": resample_max_mz,
        "width_at_mz": resample_width_at_mz,
        "reference_mz": resample_reference_mz,
        "gap_tolerance_da": resample_gap_tolerance,
    }


def _build_reader_options(
    use_recalibrated: bool,
    intensity_threshold: Optional[float],
    spectrum_type: str = "auto",
    tdf_spectrum: Optional[str] = None,
) -> dict[str, bool | float | str]:
    """Build reader options dictionary from CLI parameters.

    ``spectrum_type="auto"`` is the CLI's way of saying "no override", so it is
    omitted entirely rather than forwarded -- readers take ``None``/absent to
    mean detect, and ``"auto"`` is not a representation. ``tdf_spectrum`` is
    likewise only forwarded when given: it is a Bruker TDF option, and a
    reader for any other format would reject an unexpected keyword.
    """
    options: dict[str, bool | float | str] = {
        "use_recalibrated_state": use_recalibrated
    }
    if intensity_threshold is not None:
        options["intensity_threshold"] = intensity_threshold
    if spectrum_type != "auto":
        options["spectrum_type"] = spectrum_type
    if tdf_spectrum is not None:
        options["tdf_spectrum"] = tdf_spectrum
    return options


def _parse_streaming_option(streaming: str) -> bool | Literal["auto"]:
    """Convert the streaming CLI string to a typed value."""
    if streaming == "true":
        return True
    if streaming == "false":
        return False
    return "auto"


def _quarantine_partial_output(output: Path) -> None:
    """Move a partially written store aside after a failed conversion.

    A conversion that fails part-way through writing leaves an
    incomplete ``.zarr`` at the destination. That store cannot be opened
    (``spatialdata.read_zarr()`` raises), but it looks like a plausible
    artifact, and it also blocks a retry because the CLI refuses to write
    to an existing path. Rename it to a sibling ``.failed`` path so the
    destination is clear while the partial store remains available for
    diagnosis.

    The CLI validates that the output path does not exist before
    converting, so anything present at this point was written by this
    run and is safe to move.
    """
    if not output.exists():
        return

    quarantine = output.with_name(f"{output.name}.failed")
    attempt = 1
    while quarantine.exists():
        attempt += 1
        quarantine = output.with_name(f"{output.name}.failed{attempt}")

    try:
        output.rename(quarantine)
    except OSError as e:
        logger.error(
            "The incomplete output was left at %s because it could not be "
            "moved aside (%s). It will not open with "
            "spatialdata.read_zarr(); delete it before retrying.",
            output,
            e,
        )
        return

    logger.error(
        "The incomplete output was moved to %s. It will not open with "
        "spatialdata.read_zarr(); delete it once you no longer need it.",
        quarantine,
    )


def _handle_post_conversion(success: bool, output: Path) -> bool:
    """Report the conversion result, quarantining a partial store on failure.

    Returns whether the run succeeded. There are no fallible post-conversion
    steps left, so today this is just ``success`` -- but the caller keys the
    process exit status off the return value rather than off ``convert_msi``'s
    bool, so the next step added here has to report its failure instead of
    dropping it on the floor the way ``--optimize-chunks`` did.
    """
    if success:
        logger.info(f"Conversion completed successfully. Output stored at {output}")
        return True

    logger.error("Conversion failed.")
    _quarantine_partial_output(output)
    return False


class GroupedCommand(click.Command):
    """Command that groups options into sections in --help output.

    Every visible option must appear in exactly one ``GROUPS`` entry.
    Anything missing still shows up -- ``format_help`` sweeps the remainder
    into a trailing ``Options:`` section rather than dropping it -- but that
    section is a defect signal, not a home. It is where
    ``--resample-gap-tolerance`` and ``--spectrum-type`` sat for two
    releases, next to ``--version`` and ``--help``, while docs/cli.md told
    readers the help was organised by category.

    ``tests/unit/test_cli_help_grouping.py`` fails if the trailing section is
    ever non-empty again, so adding an option without classifying it here is
    caught before release rather than by a reader.

    Section order is mirrored by docs/cli.md; keep the two in step.
    """

    GROUPS = {
        "Conversion": [
            "--format",
            "--pixel-size",
            "--region",
            "--no-resample",
            "--resample",
            "--include-optical",
            "--no-optical",
            "--mobility-table",
            "--no-mobility-table",
            "--mobility-heatmap",
            "--no-mobility-heatmap",
            "--mobility-grid",
            "--no-mobility-grid",
            "--msms-table",
            "--no-msms-table",
        ],
        "Logging": ["--log-level", "-v", "--log-file"],
        "Ion mobility grid (advanced)": [
            "--mobility-bins",
            "--mobility-min",
            "--mobility-max",
        ],
        "Resampling (advanced)": [
            "--resample-method",
            "--mass-axis-type",
            "--resample-bins",
            "--resample-min-mz",
            "--resample-max-mz",
            "--resample-width-at-mz",
            "--resample-reference-mz",
            "--resample-gap-tolerance",
        ],
        "Performance": ["--streaming", "--sparse-format"],
        "imzML-specific": ["--spectrum-type"],
        "Bruker-specific": [
            "--use-recalibrated",
            "--no-recalibrated",
            "--interactive-calibration",
            "--intensity-threshold",
            "--tdf-spectrum",
        ],
        "Other": ["--dataset-id", "--handle-3d", "--z-spacing"],
        "General": ["--version", "--help"],
    }

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Custom help that groups options into sections."""
        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)

        opts, opt_map = self._collect_options(ctx)
        if not opts:
            return

        used: set = set()
        for group_name, group_opts in self.GROUPS.items():
            records = self._records_for_group(group_opts, opt_map, used)
            if records:
                with formatter.section(group_name):
                    formatter.write_dl(records)

        remaining = [rv for param, rv in opts if id(param) not in used]
        if remaining:
            with formatter.section("Options"):
                formatter.write_dl(remaining)

    @staticmethod
    def _collect_options(ctx: click.Context) -> tuple:
        """Collect options and build name-to-record mapping."""
        opts = []
        for param in ctx.command.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is not None:
                opts.append((param, rv))

        opt_map: dict = {}
        for param, rv in opts:
            for name in getattr(param, "opts", []) + getattr(
                param, "secondary_opts", []
            ):
                opt_map[name] = (param, rv)

        return opts, opt_map

    @staticmethod
    def _records_for_group(group_opts: list, opt_map: dict, used: set) -> list:
        """Extract help records for a group, tracking used params."""
        records = []
        for opt_name in group_opts:
            if opt_name in opt_map and id(opt_map[opt_name][0]) not in used:
                param, rv = opt_map[opt_name]
                records.append(rv)
                used.add(id(param))
        return records


@click.command(cls=GroupedCommand)
@click.version_option(version=__version__, prog_name="thyra")
@click.argument("input", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
# -- Conversion --
@click.option(
    "--format",
    type=click.Choice(["spatialdata"]),
    default="spatialdata",
    help="Output format (default: spatialdata)",
)
@click.option(
    "--pixel-size",
    type=float,
    default=None,
    help="Pixel size in um (default: auto-detect from metadata)",
)
@click.option(
    "--region",
    type=str,
    default=None,
    help=(
        "Convert specific region (default: all regions). Accepts a "
        ".mis Area Name (e.g. '03') or an integer DB RegionNumber. The "
        "DB-to-name mapping is logged at startup for multi-region datasets."
    ),
)
@click.option(
    "--resample/--no-resample",
    default=True,
    help="Mass axis resampling (default: enabled)",
)
@click.option(
    "--mobility-table/--no-mobility-table",
    default=True,
    help=(
        "When the source shares one set of (m/z, ion mobility) features across "
        "pixels (an imzML export with a mobility array), also write them as a "
        "mobility-resolved sibling table next to the summed MSI table "
        "(default: enabled)"
    ),
)
@click.option(
    "--mobility-heatmap/--no-mobility-heatmap",
    default=True,
    help=(
        "When the source has an ion mobility dimension (Bruker TDF with TIMS "
        "engaged, an imzML export with a mobility array), store the mean "
        "mass-mobility frame on the summed table as uns['mobility_heatmap'] "
        "(default: enabled; one extra pass over the source)"
    ),
)
@click.option(
    "--mobility-grid/--no-mobility-grid",
    default=False,
    help=(
        "When the source carries ion mobility per pixel rather than as a "
        "shared feature list (Bruker TDF), bin the point cloud onto a common "
        "mobility grid and write the same mobility-resolved sibling table "
        "(default: disabled; one extra pass over the source and a much larger "
        "table). Also switches the summed spectrum to --tdf-spectrum scan_sum "
        "unless that was given explicitly, which moves the stored TIC"
    ),
)
@click.option(
    "--msms-table/--no-msms-table",
    default=False,
    help=(
        "When the source isolates several precursors per pixel in disjoint "
        "mobility slices (Bruker PASEF), also write them split apart as a "
        "demultiplexed sibling table next to the summed MSI table "
        "(default: disabled; one extra pass over the source)"
    ),
)
@click.option(
    "--include-optical/--no-optical",
    default=True,
    help="Include optical images in output (default: True)",
)
# -- Ion mobility grid (advanced) --
@click.option(
    "--mobility-bins",
    type=int,
    default=MOBILITY_CHANNELS,
    show_default=True,
    help=(
        "Mobility channels the grid divides the range into, for "
        "--mobility-grid. The default is the mass-mobility heatmap's own "
        "channel count over the same edges, so a box drawn on the heatmap "
        "indexes grid channels directly; another value gives that up"
    ),
)
@click.option(
    "--mobility-min",
    type=float,
    default=None,
    help=(
        "Lower edge of the mobility grid, in the axis unit (1/K0 for TIMS). "
        "Default: the smallest value the source's mobility axis holds, which "
        "is where the heatmap starts"
    ),
)
@click.option(
    "--mobility-max",
    type=float,
    default=None,
    help=(
        "Upper edge of the mobility grid, in the axis unit. Default: the "
        "largest value the source's mobility axis holds"
    ),
)
# -- Logging --
@click.option(
    "-v",
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="Logging level (default: INFO)",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Write logs to file",
)
# -- Performance --
@click.option(
    "--streaming",
    type=click.Choice(["auto", "true", "false"]),
    default="auto",
    help="Streaming mode for large datasets (default: auto)",
)
@click.option(
    "--optimize-chunks",
    is_flag=True,
    hidden=True,
    help="Deprecated no-op, accepted so existing scripts keep running.",
)
@click.option(
    "--sparse-format",
    type=click.Choice(["csc", "csr"]),
    default="csc",
    help="Sparse matrix format: csc or csr (default: csc)",
)
# -- Resampling (advanced) --
@click.option(
    "--resample-method",
    type=click.Choice(["auto", "nearest_neighbor", "tic_preserving"]),
    default="auto",
    help="Resampling method (default: auto-detect)",
)
@click.option(
    "--mass-axis-type",
    type=click.Choice(
        ["auto", "constant", "linear_tof", "reflector_tof", "orbitrap", "fticr"]
    ),
    default="auto",
    help="Mass axis spacing type (default: auto-detect)",
)
@click.option(
    "--resample-bins",
    type=int,
    default=None,
    help="Number of bins (mutually exclusive with --resample-width-at-mz)",
)
@click.option(
    "--resample-min-mz",
    type=float,
    default=None,
    help="Minimum m/z (default: auto-detect)",
)
@click.option(
    "--resample-max-mz",
    type=float,
    default=None,
    help="Maximum m/z (default: auto-detect)",
)
@click.option(
    "--resample-width-at-mz",
    type=float,
    default=None,
    help="Mass width in Da at reference m/z for physics-based binning",
)
@click.option(
    "--resample-reference-mz",
    type=float,
    default=1000.0,
    help="Reference m/z for width specification (default: 1000.0)",
)
@click.option(
    "--resample-gap-tolerance",
    type=float,
    default=None,
    help=(
        "For tic_preserving only: discard target bins farther than this many "
        "Da from any measured m/z, instead of interpolating across the gap "
        "(default: no limit)"
    ),
)
@click.option(
    "--spectrum-type",
    type=click.Choice(["auto", "profile", "centroid"]),
    default="auto",
    help=(
        "Spectrum representation (default: auto-detect from the file). "
        "Overrides what the imzML declares via MS:1000127/MS:1000128; "
        "contradicting a declaration is logged as a warning. SCiLS Lab spells "
        "this --rep_type. imzML only."
    ),
)
# -- Bruker-specific --
@click.option(
    "--use-recalibrated/--no-recalibrated",
    default=True,
    help="Use recalibrated state (default: True)",
)
@click.option(
    "--interactive-calibration",
    is_flag=True,
    help="Display Bruker calibration states",
)
@click.option(
    "--intensity-threshold",
    type=float,
    default=None,
    help="Minimum intensity filter (useful for continuous mode data)",
)
@click.option(
    "--tdf-spectrum",
    type=click.Choice(["vendor_centroid", "scan_sum"]),
    default=None,
    help=(
        "How a TDF (TIMS) frame's mobility scans are collapsed into one "
        "spectrum per pixel: vendor_centroid (default) is Bruker's frame-level "
        "peak picker over the full ramp, matching TSF line spectra; scan_sum "
        "sums every scan and keeps all of the ion current. TDF only."
    ),
)
# -- Other --
@click.option(
    "--dataset-id",
    default="msi_dataset",
    help="Dataset identifier (default: msi_dataset)",
)
@click.option(
    "--handle-3d",
    is_flag=True,
    help="Process as 3D data instead of 2D slices (see --z-spacing)",
)
@click.option(
    "--z-spacing",
    type=float,
    default=None,
    help=(
        "Distance between consecutive slices in um, for --handle-3d. "
        "Default: reuse the in-plane pixel size and record it as an "
        "assumption -- correct only if sections happen to be one pixel "
        "width apart. Set this whenever the section thickness is known."
    ),
)
def main(
    input: Path,
    output: Path,
    format: str,
    dataset_id: str,
    pixel_size: Optional[float],
    handle_3d: bool,
    z_spacing: Optional[float],
    optimize_chunks: bool,
    log_level: str,
    log_file: Optional[Path],
    use_recalibrated: bool,
    interactive_calibration: bool,
    resample: bool,
    resample_method: str,
    resample_bins: Optional[int],
    resample_min_mz: Optional[float],
    resample_max_mz: Optional[float],
    resample_width_at_mz: Optional[float],
    resample_reference_mz: float,
    resample_gap_tolerance: Optional[float],
    mass_axis_type: str,
    spectrum_type: str,
    sparse_format: str,
    include_optical: bool,
    mobility_table: bool,
    mobility_heatmap: bool,
    mobility_grid: bool,
    mobility_bins: int,
    mobility_min: Optional[float],
    mobility_max: Optional[float],
    msms_table: bool,
    intensity_threshold: Optional[float],
    tdf_spectrum: Optional[str],
    streaming: str,
    region: Optional[str],
):
    """Convert MSI data to SpatialData format.

    INPUT: Path to input MSI file or directory
    OUTPUT: Path for output file

    Subcommands (each has its own --help): 'thyra validate PATH'
    validates MSI metadata against the schema; 'thyra export-metaspace
    PATH' writes the METASPACE submission JSON.
    """
    # Validate all parameters
    _validate_basic_params(pixel_size, dataset_id)
    _validate_resampling_params(
        resample_bins,
        resample_min_mz,
        resample_max_mz,
        resample_width_at_mz,
        resample_reference_mz,
    )
    _validate_positive_float(
        resample_gap_tolerance, "resample_gap_tolerance", "Gap tolerance"
    )
    _validate_positive_float(
        intensity_threshold, "intensity_threshold", "Intensity threshold"
    )
    _validate_positive_float(z_spacing, "z_spacing", "Z spacing")
    _validate_input_path(input)
    _validate_output_path(output)

    # Configure logging
    setup_logging(log_level=getattr(logging, log_level), log_file=log_file)

    # Warn before doing any work: on a multi-hour conversion, telling the user
    # at the end that the flag was dead is too late to be useful.
    if optimize_chunks:
        logger.warning(
            "--optimize-chunks is deprecated and does nothing. Chunk sizes for "
            "the converted store are chosen at write time; the post-hoc pass "
            "this flag used to invoke never worked and has been removed. The "
            "flag is still accepted so existing scripts keep running, and will "
            "be dropped in a future release."
        )

    # If input folder has multiple .d datasets, let the user choose
    input = _select_bruker_dataset(input)

    # Display calibration info if requested (Bruker datasets only)
    if interactive_calibration and input.is_dir() and input.suffix.lower() == ".d":
        _display_calibration_info(input, use_recalibrated)

    # Build resampling config if enabled
    resampling_config = (
        _build_resampling_config(
            resample_method,
            mass_axis_type,
            resample_bins,
            resample_min_mz,
            resample_max_mz,
            resample_width_at_mz,
            resample_reference_mz,
            resample_gap_tolerance,
        )
        if resample
        else None
    )

    # Build reader options for format-specific settings
    reader_options = _build_reader_options(
        use_recalibrated, intensity_threshold, spectrum_type, tdf_spectrum
    )

    # Perform conversion
    success = convert_msi(
        str(input),
        str(output),
        format_type=format,
        dataset_id=dataset_id,
        pixel_size_um=pixel_size,
        handle_3d=handle_3d,
        z_spacing_um=z_spacing,
        resampling_config=resampling_config,
        reader_options=reader_options,
        sparse_format=sparse_format,
        include_optical=include_optical,
        streaming=_parse_streaming_option(streaming),
        region=region,
        write_mobility_table=mobility_table,
        mobility_heatmap=mobility_heatmap,
        mobility_grid=mobility_grid,
        mobility_bins=mobility_bins,
        mobility_min=mobility_min,
        mobility_max=mobility_max,
        msms_table=msms_table,
    )

    ok = _handle_post_conversion(success, output)

    # Surface failure to the shell: without this a script or CI wrapper
    # calling `thyra` sees success even though nothing usable was written.
    if not ok:
        raise SystemExit(1)


def cli() -> None:
    """Console-script entry point: dispatch subcommands, else convert.

    ``thyra INPUT OUTPUT`` converts exactly as it always has; the
    metadata subcommands (``thyra validate``, ``thyra
    export-metaspace``) are picked off by their first argument before
    click sees it.  Dispatch is hand-rolled rather than a
    ``click.Group`` because a group cannot carry the two positional
    arguments the conversion interface is documented with.
    """
    import sys

    from thyra.metadata.schema.cli import METADATA_SUBCOMMANDS

    args = sys.argv[1:]
    if args and args[0] in METADATA_SUBCOMMANDS:
        command = METADATA_SUBCOMMANDS[args[0]]
        command.main(args[1:], prog_name=f"thyra {args[0]}")
        return
    main()


if __name__ == "__main__":
    cli()
