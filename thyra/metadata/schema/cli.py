# thyra/metadata/schema/cli.py
"""The ``thyra validate`` and ``thyra export-metaspace`` subcommands.

Both accept either a converted SpatialData ``.zarr`` store (only the
metadata block is read -- validating a 100 GB store is instant) or a
standalone metadata ``.json`` document, and both take ``--merge`` to
overlay user-supplied fields (organism, condition, matrix, ...) that
raw files cannot provide.

Exit status follows the conversion CLI's convention: 0 success,
1 validation errors / no metadata found, 2 usage errors (click).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from ...utils.windows_paths import prepare_zarr_read_path
from .metaspace import to_metaspace
from .models import MSI_METADATA_UNS_KEY, MSIMetadata
from .store_io import deep_merge, read_msi_metadata_blocks
from .validate import ValidationIssue, check_store_var_conventions, validate_document

logger = logging.getLogger(__name__)


def _resolve_store_path(path: Path) -> Path:
    r"""The path to read ``path`` through, refusing a missing one first.

    ``click.Path(exists=True)`` cannot be used for these arguments. The
    converter writes a store to a deep Windows path through an
    extended-length (``\\?\``) path, and click's existence check runs
    before anything can prepare the path the same way -- so the CLI
    refused to validate a store the CLI had just written, with
    ``Path '...deep.zarr' does not exist`` while the API validated it
    fine (issue #257). Existence is checked here instead, on the
    prepared path, and the same preparation is what the read then uses.
    """
    prepared = prepare_zarr_read_path(path)
    if not prepared.exists():
        raise click.BadParameter(
            f"Path {str(path)!r} does not exist.", param_hint="PATH"
        )
    return prepared


def _load_json(path: Path) -> Dict[str, Any]:
    """Parse a JSON file into a mapping, with a CLI-friendly error."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Could not read {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise click.ClickException(f"{path} does not contain a JSON object")
    return document


def _load_documents(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load metadata documents from a store or a JSON file.

    Returns a mapping of label (table name, or the file name for JSON
    input) to document.
    """
    if path.is_file() and path.suffix.lower() == ".json":
        return {path.name: _load_json(path)}

    try:
        blocks = read_msi_metadata_blocks(path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not blocks:
        raise click.ClickException(
            f"No {MSI_METADATA_UNS_KEY} block found in {path}. Stores "
            "written by Thyra versions without the metadata schema do not "
            "carry one; reconvert with a current version."
        )
    return blocks


def _apply_merge(
    documents: Dict[str, Dict[str, Any]], merge_path: Optional[Path]
) -> Dict[str, Dict[str, Any]]:
    """Overlay a user-supplied JSON document onto every loaded document."""
    if merge_path is None:
        return documents
    overlay = _load_json(merge_path)
    return {label: deep_merge(doc, overlay) for label, doc in documents.items()}


def _store_var_issues(path: Path) -> Dict[str, List[ValidationIssue]]:
    """Per-table var-contract issues; empty for JSON document input."""
    if path.is_file() and path.suffix.lower() == ".json":
        return {}
    try:
        return check_store_var_conventions(path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _echo_issues(label: str, issues: List[ValidationIssue]) -> None:
    """Print one document's findings in a stable, greppable layout."""
    for issue in sorted(issues, key=lambda i: (i.severity != "error", i.location)):
        location = issue.location or "(document)"
        click.echo(f"  {issue.severity.upper()} {location}: {issue.message}")


@click.command("validate")
@click.argument("path", type=click.Path(exists=False, path_type=Path))
@click.option(
    "--merge",
    "merge_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="JSON file overlaid onto the metadata before validation.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit a machine-readable report on stdout instead of text.",
)
def validate_command(path: Path, merge_path: Optional[Path], as_json: bool) -> None:
    """Validate MSI metadata against the Thyra metadata schema.

    PATH is a converted SpatialData .zarr store or a metadata JSON
    document. For a store, the var column contract is checked too
    ('mz' present, numeric, finite, strictly increasing). Exit status
    is 0 when every document conforms (warnings allowed), 1 otherwise,
    so it can gate CI.
    """
    path = _resolve_store_path(path)
    documents = _apply_merge(_load_documents(path), merge_path)
    var_issues = _store_var_issues(path)

    report: Dict[str, Any] = {}
    failed = False
    for label in sorted(set(documents) | set(var_issues)):
        if label in documents:
            _, issues = validate_document(documents[label])
        else:
            # The table exists in the store but carries no metadata
            # block; every table a converter writes gets one.
            issues = [
                ValidationIssue(
                    "error",
                    MSI_METADATA_UNS_KEY,
                    "table carries no msi_metadata block",
                )
            ]
        issues = list(issues) + var_issues.get(label, [])
        errors = [issue for issue in issues if issue.severity == "error"]
        failed = failed or bool(errors)
        report[label] = {
            "valid": not errors,
            "issues": [
                {
                    "severity": issue.severity,
                    "location": issue.location,
                    "message": issue.message,
                }
                for issue in issues
            ],
        }
        if not as_json:
            status = "FAILED" if errors else "OK"
            suffix = ""
            warning_count = len(issues) - len(errors)
            if warning_count:
                suffix = f" ({warning_count} warning(s))"
            click.echo(f"{label}: {status}{suffix}")
            _echo_issues(label, issues)

    if as_json:
        click.echo(json.dumps(report, indent=2))

    if failed:
        raise SystemExit(1)


def _select_document(
    documents: Dict[str, Dict[str, Any]], table: Optional[str]
) -> Tuple[str, Dict[str, Any]]:
    """Pick the document to export when a store holds several tables."""
    if table is not None:
        if table not in documents:
            raise click.ClickException(
                f"Table {table!r} not found; available: " + ", ".join(sorted(documents))
            )
        return table, documents[table]
    if len(documents) == 1:
        return next(iter(documents.items()))
    raise click.ClickException(
        "The store has metadata for several tables; pick one with --table. "
        "Available: " + ", ".join(sorted(documents))
    )


def _validated_model(label: str, document: Dict[str, Any]) -> MSIMetadata:
    """Validate before export; refuse to export a non-conforming document."""
    meta, issues = validate_document(document)
    if meta is None:
        click.echo(f"{label}: metadata does not conform to the schema:", err=True)
        _echo_issues(label, [i for i in issues if i.severity == "error"])
        raise SystemExit(1)
    return meta


@click.command("export-metaspace")
@click.argument("path", type=click.Path(exists=False, path_type=Path))
@click.option(
    "--merge",
    "merge_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="JSON file overlaid onto the metadata before export.",
)
@click.option(
    "--table",
    default=None,
    help="Table to export when the store has several (default: the only one).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file (default: <input>.metaspace.json next to the input; "
    "'-' for stdout).",
)
def export_metaspace_command(
    path: Path,
    merge_path: Optional[Path],
    table: Optional[str],
    output_path: Optional[Path],
) -> None:
    """Write the METASPACE submission metadata JSON for a dataset.

    PATH is a converted SpatialData .zarr store or a metadata JSON
    document. Required fields the metadata does not carry are emitted
    empty and reported as warnings on stderr; fill them via --merge or
    on the METASPACE submission form.
    """
    path = _resolve_store_path(path)
    documents = _apply_merge(_load_documents(path), merge_path)
    label, document = _select_document(documents, table)
    meta = _validated_model(label, document)

    metaspace_document, warnings = to_metaspace(meta)
    for warning in warnings:
        click.echo(f"warning: {warning}", err=True)

    rendered = json.dumps(metaspace_document, indent=2)
    if output_path is not None and str(output_path) == "-":
        click.echo(rendered)
        return

    if output_path is None:
        output_path = path.with_name(path.stem + ".metaspace.json")
    output_path.write_text(rendered + "\n", encoding="utf-8")
    click.echo(f"Wrote METASPACE metadata for {label} to {output_path}")
    if warnings:
        click.echo(
            f"{len(warnings)} required field(s) are still empty; complete "
            "them before submitting.",
            err=True,
        )


# Consumed by thyra.__main__ to dispatch `thyra <subcommand>` without
# disturbing the positional `thyra INPUT OUTPUT` conversion interface.
METADATA_SUBCOMMANDS: Dict[str, click.Command] = {
    "validate": validate_command,
    "export-metaspace": export_metaspace_command,
}
