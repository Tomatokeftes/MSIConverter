r"""Extended-length path support for Zarr stores on Windows.

Windows caps a normal path at 260 characters including the terminating
NUL, so 259 usable. A Zarr store is a directory tree and the limit applies
to each key inside it, not to the path the user typed. Thyra's deepest key
is a metadata document about 134 characters below the store root::

    <out>.zarr\tables\<dataset_id>_z0_mobility\uns\msi_metadata\ms_analysis
        \ion_mobility\separation_term\accession\zarr.<32-hex>.partial

so an output path of roughly 120 characters is already enough to push the
deepest key past the limit. It surfaces as::

    Error saving SpatialData: [Errno 2] No such file or directory:
      '...\imzml_version\zarr.<hash>.partial'

which points at a file the user never named and gives no hint that path
length is the problem.

Prefixing the store path with ``\\?\`` opts that path out of the 260
character limit entirely. Verified here on Windows 11 with
``LongPathsEnabled=0``: a 207 character output path fails, and the same
path with the prefix converts cleanly.

The prefix is applied only when the projected deepest key would not fit
otherwise, so ordinary paths are handled exactly as before. It is not
applied to the *source data* path: readers reach vendor SDKs that may not
accept extended-length syntax.

A *relative* store path is a second, sharper trap, so both helpers below
resolve one before doing anything else and hand back the resolved path.
Windows measures a relative path as the raw ``<cwd> + "\" + <spelling>``
concatenation, *before* the ``..`` segments are collapsed, so a path whose
absolute form sits comfortably inside the limit can still be refused::

    cwd                                                    175
    ..\out.zarr\tables\<id>\uns\<block>\mobility_edges\zarr.json    +85
                                                           ---
                                                           260  refused
    C:\...\out.zarr\tables\<id>\uns\<block>\mobility_edges\zarr.json
                                                           252  fine

Measured here on a real store: the sibling key one character shorter
(``current_ratio``, 259) opened, ``mobility_edges`` at 260 did not, and the
absolute spelling of the refused one is 252. Nothing about the array was
different -- same shape, dtype, codecs and shards as its siblings. Windows
long-path support does not rescue a relative path either, since it applies
only to fully qualified paths, so the resolution is unconditional rather
than gated on the registry check.

Reading a converted store back is subject to the same limit, and a read
past it does not fail. Windows reports an over-long key as missing, and
Zarr treats a missing key as one that was never written: a whole-store
``spatialdata.read_zarr`` finds the deep groups absent and calls the store
structurally invalid, while ``zarr.open_group`` on the metadata alone
quietly returns fill values, so every ontology term reads back as
``{"accession": "", "name": ""}``. See :func:`prepare_zarr_read_path`,
which Thyra's own read paths go through.
"""

import logging
import ntpath
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: Longest usable normal Windows path, excluding the terminating NUL.
WINDOWS_MAX_PATH = 259

#: Characters Thyra needs below the store root for its deepest key, not
#: counting the dataset id. Measured at 134 for the deepest key Thyra emits
#: today, on the mobility-resolved sibling table::
#:
#:     \tables\<id>_z0_mobility\uns\msi_metadata\ms_analysis\ion_mobility
#:         \separation_term\accession\zarr.<32-hex>.partial
#:
#: (the ``_mobility`` suffix and the ``ion_mobility`` block each added a
#: level over the 95 this was first measured at, on
#: ``raw_metadata\\max count of pixels x``). Metadata key names come from
#: the source file and can be longer, so this carries headroom above the
#: observed worst case.
#:
#: The hazard on the read side is that a key past the limit is not an
#: error. Windows reports it as missing, and Zarr treats a missing chunk as
#: the array's fill value by design (``""`` for strings, ``0`` for
#: numbers), so a metadata block read from a plain path comes back with
#: empty ontology terms and a blank processing history instead of an
#: exception, and validation then blames the document. Zarr has no strict
#: mode that could tell "never written" from "cannot open", and its own
#: key listing cannot enter an over-long directory either, so nothing
#: inside Zarr can detect it. The only detection is an independent,
#: extended-length walk of the store, which :func:`prepare_zarr_read_path`
#: performs; every read entry point must go through it.
DEEPEST_KEY_RESERVE = 160

_EXTENDED_PREFIX = "\\\\?\\"


def _long_paths_enabled() -> bool:
    """Whether Windows long-path support is switched on system-wide.

    When it is, normal paths already work past 260 characters and there is
    nothing to work around.
    """
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        return bool(value)
    except (ImportError, OSError):
        # Missing key, no permission, or not Windows: assume the limit
        # applies, which only ever means using the prefix when it was not
        # strictly needed.
        return False


def _absolute(path: Path) -> Path:
    r"""The absolute spelling of ``path``.

    Absoluteness is judged by :mod:`ntpath` *or* ``Path.is_absolute``, and
    the two disagree only when the platform is faked. ``ntpath`` catches a
    Windows-shaped path such as ``C:\...\out.zarr`` while the suite is
    running on Linux; ``Path.is_absolute`` catches the POSIX path a
    ``tmp_path`` fixture hands back there, which ``ntpath`` reads as merely
    drive-relative. On Windows both agree and either alone would do.
    """
    if ntpath.isabs(str(path)) or path.is_absolute():
        return path
    return path.resolve()


def to_extended_length_path(path: Path) -> Path:
    r"""Rewrite an absolute Windows path into extended-length form.

    ``\\?\`` disables all path normalisation, so the input must already be
    absolute and free of ``..`` and forward slashes. Callers should pass a
    ``Path.resolve()`` result.

    Already-extended paths are returned unchanged, and UNC shares get the
    ``\\?\UNC\`` spelling rather than an invalid double prefix.
    """
    text = str(path)

    if text.startswith(_EXTENDED_PREFIX):
        return path

    if text.startswith("\\\\"):
        # \\server\share -> \\?\UNC\server\share
        return Path(f"{_EXTENDED_PREFIX}UNC\\{text.lstrip(chr(92))}")

    return Path(f"{_EXTENDED_PREFIX}{text}")


def projected_deepest_key_length(output_path: Path, dataset_id: str) -> int:
    """Length of the longest path Thyra expects to write inside the store."""
    return len(str(output_path)) + 1 + len(dataset_id) + DEEPEST_KEY_RESERVE


def prepare_zarr_read_path(store_path: Path) -> Path:
    r"""Return a path that can actually open a store written to a long path.

    The write side is protected by :func:`prepare_zarr_output_path`, but the
    limit applies just as much to reading it back, and a read past it does
    not fail. Windows reports an over-long key as missing and Zarr treats a
    missing key as never written: ``spatialdata.read_zarr(path)`` finds the
    deep groups absent and calls the store structurally invalid, while
    ``zarr.open_group(path)`` on an intact metadata block returns the fill
    value for every chunk it cannot open, so ontology terms read back as
    ``{"accession": "", "name": ""}`` and validation blames the document
    rather than the path. Plain ``os.path.exists`` on those keys returns
    ``False`` too, which makes the store look corrupt when it is intact.

    Unlike the output helper this cannot project the deepest key, because the
    keys already exist and their length depends on what was written. It walks
    the store to find the longest key instead, which is cheap next to the
    read that follows, and stops at the first key past the limit. The walk
    itself goes through an extended-length path: a plain ``os.walk`` cannot
    list a directory past the limit and skips it silently, so it would miss
    exactly the keys this is looking for and could pass a store whose deepest
    keys do not fit.

    A relative path is resolved first and the resolved path is what comes
    back, even when the store is shallow enough to need no prefix at all.
    Handing the caller's own relative spelling back is not safe: Windows
    measures it as ``<cwd> + "\" + <spelling>`` before collapsing the ``..``
    segments, so a store whose keys all fit in absolute terms can still lose
    an array through that spelling, with only a ``UserWarning`` from Zarr
    about an object it does not recognise. A path that already carries the
    prefix is returned as it is.

    Args:
        store_path: Path to an existing Zarr store.

    Returns:
        The path to hand to the reader, extended if the store needs it.
    """
    if sys.platform != "win32":
        return store_path

    if str(store_path).startswith(_EXTENDED_PREFIX):
        return store_path

    absolute = _absolute(store_path)
    extended = to_extended_length_path(absolute)
    # Keys are measured as the plain path would spell them.
    prefix_length = len(str(extended)) - len(str(absolute))

    longest = len(str(absolute))
    for root, _dirs, files in os.walk(extended):
        for name in files:
            longest = max(longest, len(root) + 1 + len(name) - prefix_length)
        if longest > WINDOWS_MAX_PATH:
            break

    # ``absolute``, never ``store_path``: the caller's relative spelling is
    # measured against the limit uncollapsed, so it can be refused where the
    # absolute form fits. Long-path support does not cover it either, which
    # is why that branch resolves too.
    if longest <= WINDOWS_MAX_PATH:
        return absolute

    if _long_paths_enabled():
        return absolute

    logger.info(
        "Store contains a key %d characters long, past the %d character "
        "Windows limit. Reading through an extended-length path.",
        longest,
        WINDOWS_MAX_PATH,
    )
    return extended


def prepare_zarr_output_path(output_path: Path, dataset_id: str) -> Path:
    """Return the path to hand to Zarr, extended if the store needs it.

    A no-op off Windows, and a no-op on Windows when long-path support is
    enabled or the projected deepest key fits inside the normal limit.

    A relative path is resolved first and the resolved path is what comes
    back. ``len(str(output_path))`` on a relative path measures the spelling
    rather than the path, which understates the projection and would wave
    through an output location that cannot hold the store.

    Args:
        output_path: The output store path. Resolved here if it is relative.
        dataset_id: The dataset identifier, which appears in the deepest key.
    """
    if sys.platform != "win32":
        return output_path

    if str(output_path).startswith(_EXTENDED_PREFIX):
        return output_path

    output_path = _absolute(output_path)

    projected = projected_deepest_key_length(output_path, dataset_id)
    if projected <= WINDOWS_MAX_PATH:
        return output_path

    if _long_paths_enabled():
        logger.debug(
            "Output path is long (projected deepest key %d characters) but "
            "Windows long-path support is enabled; using it as given",
            projected,
        )
        return output_path

    extended = to_extended_length_path(output_path)
    logger.warning(
        "Output path is long enough that the deepest Zarr key would exceed "
        "the %d character Windows limit (projected %d). Writing through an "
        "extended-length path so the conversion is not truncated. Note that "
        "reading the store back is subject to the same limit: pass it "
        "through thyra.utils.windows_paths.prepare_zarr_read_path, or move "
        "the store somewhere shallower, or enable Windows long-path support.",
        WINDOWS_MAX_PATH,
        projected,
    )
    return extended
