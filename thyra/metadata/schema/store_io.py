# thyra/metadata/schema/store_io.py
"""Read the ``msi_metadata`` block back out of a converted store.

Deliberately memory-bounded: only the ``uns/msi_metadata`` group of
each table is read, never the intensity matrix, so validating or
exporting metadata from a 100+ GB store costs nothing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

from ...utils.windows_paths import prepare_zarr_read_path
from .models import MSI_METADATA_UNS_KEY

logger = logging.getLogger(__name__)


def read_msi_metadata_blocks(store_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Read every table's ``msi_metadata`` block from a SpatialData store.

    Args:
        store_path: Path to a converted ``.zarr`` store.

    Returns:
        Mapping of table name to the block as a plain dict.  Tables
        without a block are skipped; a store written by a Thyra version
        that predates the schema therefore returns an empty mapping.

    Raises:
        ValueError: If the path is not a SpatialData store (no
            ``tables`` group).
    """
    import anndata as ad
    import zarr

    # A store under a long Windows path reads back wrong rather than
    # failing: zarr hands out fill values for the keys it cannot open, so
    # every ontology term would come back as {"accession": "", "name": ""}.
    root = zarr.open_group(str(prepare_zarr_read_path(Path(store_path))), mode="r")
    if "tables" not in root:
        raise ValueError(
            f"{store_path} does not look like a SpatialData store: "
            "it has no 'tables' group"
        )

    tables = root["tables"]
    blocks: Dict[str, Dict[str, Any]] = {}
    for name in sorted(tables.keys()):
        try:
            uns = tables[name]["uns"]
        except KeyError:
            logger.debug("Table %s has no uns group", name)
            continue
        if MSI_METADATA_UNS_KEY not in uns:
            logger.debug("Table %s has no %s block", name, MSI_METADATA_UNS_KEY)
            continue
        block = dict(ad.io.read_elem(uns[MSI_METADATA_UNS_KEY]))
        # `processing` is stored as JSON (a list of objects cannot
        # round-trip through AnnData/zarr); hand callers the parsed form.
        if isinstance(block.get("processing"), str):
            try:
                block["processing"] = json.loads(block["processing"])
            except json.JSONDecodeError:
                if block["processing"] == "":
                    # Converters write at least "[]". An empty string is the
                    # fill value of a string array: the chunk exists but
                    # could not be read, which on Windows means a key past
                    # the path limit (see thyra.utils.windows_paths).
                    logger.warning(
                        "Table %s has an empty processing section, which no "
                        "converter writes; the store's deep keys may be "
                        "unreadable at this path",
                        name,
                    )
                else:
                    logger.warning(
                        "Table %s has an unparseable processing section", name
                    )
        _decode_isolation_windows(block, name)
        blocks[name] = block

    return blocks


def _decode_isolation_windows(block: Dict[str, Any], name: str) -> None:
    """Parse ``ms_analysis.fragmentation.windows`` back from its JSON string.

    Stored as JSON for the same reason ``processing`` is -- a list of
    objects does not round-trip through AnnData/zarr. Decoded in place so
    callers see the parsed form either way.
    """
    fragmentation = block.get("ms_analysis")
    if not isinstance(fragmentation, dict):
        return
    fragmentation = fragmentation.get("fragmentation")
    if not isinstance(fragmentation, dict):
        return
    windows = fragmentation.get("windows")
    if not isinstance(windows, str):
        return
    try:
        fragmentation["windows"] = json.loads(windows)
    except json.JSONDecodeError:
        logger.warning(
            "Table %s has an unparseable fragmentation.windows section", name
        )


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base`` with ``overlay`` merged over it, recursively.

    Nested dicts merge key-by-key; any other overlay value replaces the
    base value.  Neither input is modified.  Used to overlay
    user-supplied metadata (organism, condition, matrix, ...) onto the
    auto-populated block at validation or export time.
    """
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
