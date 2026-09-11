# thyra/readers/imzml/imzml_reader.py
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union, cast

# The stdlib XML parser, used to re-read one element of a document pyimzml has
# already parsed with the same stdlib parser -- see
# _first_spectrum_array_lengths. Thyra does depend on defusedxml, hard, and the
# Bruker `.mis` parser imports it unconditionally; it is deliberately not used
# here, because it would not change what has already been read. pyimzml parses
# the whole document with xml.etree first, this re-read only revisits one
# element of what it accepted, and defusedxml cannot retract that.
from xml.etree import ElementTree  # nosec B405

import numpy as np
from numpy.typing import NDArray
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm

from ...core.base_extractor import MetadataExtractor
from ...core.base_reader import BaseMSIReader
from ...core.mobility import MobilityAxis, classify_mobility_array
from ...core.registry import register_reader
from ...errors import ConversionRefused
from ...metadata.extractors.imzml_extractor import ImzMLMetadataExtractor
from ...resampling.constants import ImzMLAccessions, normalize_spectrum_type
from ...utils.imzml_coordinate_base import coordinate_bases
from ...utils.pyimzml_direct import read_spectrum_mzs_only
from ._pyimzml_compat import ensure_lenient_cv_param_values
from .mobility_array import (
    MobilityArraySpec,
    collect_array_offsets,
    detect_mobility_array,
)

logger = logging.getLogger(__name__)

# Scratch buffer floor for the processed-mode mass axis build, in ELEMENTS.
# 1Mi elements is 8 MiB at float64. Measured at a 512 MiB m/z payload on a
# shared grid (peak extra RSS / wall clock): 2^14 -> 7.14 MiB / 3.04 s,
# 2^17 -> 6.90 / 3.06, 2^20 -> 15.30 / 2.30, 2^22 -> 40.29 / 2.11,
# 2^24 -> 148.38 / 2.06. A 1024x sweep of the knob moves wall clock 1.5x with
# no cliff anywhere, so this is a knee rather than a fragile constant.
_MASS_AXIS_BATCH_VALUES = 1 << 20

# Block size for the searchsorted probe in ``_filter_absent``, in ELEMENTS.
# Bounds that function's temporaries to O(block) rather than O(batch).
_MASS_AXIS_PROBE_BLOCK = 1 << 20

# Default cap on the number of unique m/z values the processed-mode raw mass
# axis may reach, in ELEMENTS. Taken from SCiLS Lab, which applies the same
# limit to the same quantity: "Data sets in SCiLS Lab are limited to a maximum
# of 10 million bins on the common mass axis" (SCiLS Lab 2026b User Guide,
# p.76). The cap previously defaulted to None because no defensible number was
# available; this is one, from the tool Thyra's resampling was designed
# against. Override with ``reader_options={"max_mass_axis_length": N}``, or
# pass None for the old unlimited behaviour.
DEFAULT_MAX_MASS_AXIS_LENGTH = 10_000_000

# mzML's XML namespace, spelled the way pyimzml spells it.
_MZML_NS = "{http://psi.hupo.org/ms/mzml}"

# Number of values in a binary array, and the number of bytes those values
# occupy once encoded. pyimzml keeps the former and discards the latter.
_ARRAY_LENGTH_ACCESSION = "IMS:1000103"
_ENCODED_LENGTH_ACCESSION = "IMS:1000104"

# zlib-compressed binary arrays. The strings "zlib", "decompress", "1000574"
# and "1000104" appear nowhere in pyimzml 1.5.5's parser, so a compressed array
# is read as ``IMS:1000103 * itemsize`` raw deflate bytes and handed to
# ``np.frombuffer``, which returns numbers of the correct count.
_ZLIB_ACCESSION = "MS:1000574"
_ZLIB_NAME = "zlib compression"

# Precision characters whose numpy itemsize is not the itemsize pyimzml seeks
# with. ``SIZE_DICT['l']`` is 8 on every platform, but ``np.dtype('l').itemsize``
# is 4 on Windows and 8 on Linux, so pyimzml reads N*8 bytes and decodes 2N
# int32 values here where it decodes N int64 values there -- one file, two
# answers. '32-bit integer' ('i') is deliberately NOT in this set: MS:1000519 is
# spec-legal, pyimzml's own writer emits it for ``intensity_dtype=np.int32``,
# and it converts correctly today.
_PLATFORM_DEPENDENT_PRECISIONS = frozenset({"l"})


# A coalescing block reader for the .ibd was built and benchmarked here on
# 2026-08-26 and then removed: with the OS page cache warm, a 120k-spectrum
# iteration pass costs ~0.75 s whether reads are per-spectrum seek+read
# pairs or 8 MiB windows -- small cached reads are a few microseconds, so
# there was nothing to coalesce away. Do not re-add one without a
# measurement on storage where it wins (cold cache, network shares).


class _OffsetArrays(NamedTuple):
    """The four per-spectrum offset and length arrays, as int64."""

    mz_offsets: NDArray[np.int64]
    mz_lengths: NDArray[np.int64]
    int_offsets: NDArray[np.int64]
    int_lengths: NDArray[np.int64]


def _offset_arrays(parser: ImzMLParser) -> _OffsetArrays:
    """Materialise pyimzml's four offset/length lists as int64 arrays.

    ``np.fromiter(..., count=n)`` rather than ``np.asarray(..., dtype=...)``:
    measured 54.6 ms against 73.5 ms over xenium's 918,855 spectra, and these
    four conversions are most of what validation costs.

    Args:
        parser: An initialized ImzML parser.

    Returns:
        The parser's m/z and intensity offsets and lengths.
    """
    n = len(parser.mzOffsets)
    return _OffsetArrays(
        np.fromiter(parser.mzOffsets, dtype=np.int64, count=n),
        np.fromiter(parser.mzLengths, dtype=np.int64, count=n),
        np.fromiter(parser.intensityOffsets, dtype=np.int64, count=n),
        np.fromiter(parser.intensityLengths, dtype=np.int64, count=n),
    )


def _binary_array_specs(
    parser: ImzMLParser,
) -> Tuple[Tuple[str, Any, Optional[str]], ...]:
    """Return ``(label, param group id, precision char)`` for both arrays.

    Args:
        parser: An initialized ImzML parser.

    Returns:
        One tuple for the m/z array and one for the intensity array.
    """
    return (
        ("m/z", parser.mzGroupId, parser.mzPrecision),
        ("intensity", parser.intGroupId, parser.intensityPrecision),
    )


def _cv_param_int(elem: Any, accession: str) -> Optional[int]:
    """Read one cvParam's value off an element as an int.

    Args:
        elem: The XML element to search directly beneath.
        accession: The cvParam accession to look for.

    Returns:
        The integer value, or None if the param is absent or not an integer.
    """
    node = elem.find(f'{_MZML_NS}cvParam[@accession="{accession}"]')
    if node is None:
        return None
    try:
        return int(node.attrib["value"])
    except (KeyError, ValueError):
        return None


def _first_spectrum_array_lengths(
    imzml_path: Path,
) -> Dict[Any, Tuple[Optional[int], Optional[int]]]:
    """Read spectrum 0's declared array and encoded lengths, per param group.

    ``ImzMLParser`` prunes every ``<spectrum>`` out of the tree as it streams
    and keeps only ``IMS:1000102`` and ``IMS:1000103``, so ``IMS:1000104`` --
    the *encoded* byte length, and the only independent witness to the decode
    width -- is gone by the time the parser returns. Re-reading it for the
    first spectrum costs the header plus one spectrum element however large the
    file is.

    Args:
        imzml_path: Path to the imzML file.

    Returns:
        A mapping of ``referenceableParamGroupRef`` to
        ``(array_length, encoded_length)``, either of which is None when the
        file does not declare it. Empty if the file declares no spectra.
    """
    # Same document, same stdlib parser pyimzml itself used a moment ago; see
    # the note on the import.
    events = ElementTree.iterparse(str(imzml_path), events=("end",))  # nosec B314
    for _event, elem in events:
        if elem.tag != _MZML_NS + "spectrum":
            continue
        arrays: Dict[Any, Tuple[Optional[int], Optional[int]]] = {}
        for node in elem.findall(
            f"{_MZML_NS}binaryDataArrayList/{_MZML_NS}binaryDataArray"
        ):
            ref_node = node.find(f"{_MZML_NS}referenceableParamGroupRef")
            if ref_node is None:
                continue
            arrays[ref_node.attrib.get("ref")] = (
                _cv_param_int(node, _ARRAY_LENGTH_ACCESSION),
                _cv_param_int(node, _ENCODED_LENGTH_ACCESSION),
            )
        return arrays
    return {}


def _dedupe_sorted(a: NDArray[Any]) -> NDArray[Any]:
    """Deduplicate an ALREADY-SORTED array, as ``np.unique`` would.

    Reproduces numpy's ``_unique1d`` mask including its ``equal_nan=True``
    behaviour, which collapses a trailing run of NaNs to its first element.
    Unlike ``np.unique`` it does not make the extra full-size ``flatten()``
    copy, which is one of the three payload-sized allocations the old
    collect-everything-then-unique build paid for.

    Args:
        a: A sorted array.

    Returns:
        The distinct values of ``a``, in order.
    """
    n = a.size
    if n == 0:
        return a[:0].copy()

    mask = np.empty(n, dtype=bool)
    mask[0] = True

    if a.dtype.kind == "f" and bool(np.isnan(a[-1])):
        first_nan = int(np.searchsorted(a, a[-1], side="left"))
        # The ``first_nan > 0`` guard is load-bearing: an all-NaN input makes
        # the slices empty and ``np.not_equal`` raises on the shape mismatch.
        if first_nan > 0:
            np.not_equal(a[1:first_nan], a[: first_nan - 1], out=mask[1:first_nan])
        mask[first_nan] = True
        mask[first_nan + 1 :] = False
    else:
        np.not_equal(a[1:], a[:-1], out=mask[1:])

    return a[mask]


def _filter_absent(
    acc: NDArray[Any],
    s: NDArray[Any],
    block: int = _MASS_AXIS_PROBE_BLOCK,
) -> NDArray[Any]:
    """Return the elements of ``s`` that are not already in ``acc``.

    Both arrays must be sorted and free of duplicates. ``s`` is walked in
    blocks so the search temporaries stay O(block) rather than O(len(s)).
    Returns ``s`` itself, uncopied, when none of its values are present --
    the all-distinct fast path, worth half a payload at peak.

    Args:
        acc: The running axis, sorted and unique.
        s: A candidate batch, sorted and unique.
        block: Probe block size in elements.

    Returns:
        The subset of ``s`` absent from ``acc``.
    """
    n = s.size
    # ``acc.size == 0`` must be handled here: the ``np.minimum`` below would
    # otherwise index ``acc[-1]`` on an empty array and raise IndexError.
    if n == 0 or acc.size == 0:
        return s

    keep = np.empty(n, dtype=bool)
    n_keep = 0
    is_float = acc.dtype.kind == "f"
    a_len = acc.size

    for lo in range(0, n, block):
        hi = min(lo + block, n)
        sb = s[lo:hi]
        pos = np.searchsorted(acc, sb, side="left")
        out_of_range = pos >= a_len
        np.minimum(pos, a_len - 1, out=pos)
        va = acc[pos]
        eq = va == sb
        if is_float:
            # searchsorted routes a NaN key onto acc's first NaN, but NaN does
            # not compare equal to itself, so without this the same NaN would
            # be re-inserted on every fold.
            np.logical_or(eq, np.isnan(va) & np.isnan(sb), out=eq)
        np.logical_and(eq, ~out_of_range, out=eq)
        np.logical_not(eq, out=eq)
        keep[lo:hi] = eq
        n_keep += int(np.count_nonzero(eq))
        del pos, out_of_range, va, eq, sb

    if n_keep == n:
        return s
    if n_keep == 0:
        return s[:0]
    return s[keep]


def _merge_disjoint(box_a: List[Any], box_b: List[Any]) -> NDArray[Any]:
    """Merge two DISJOINT sorted arrays, releasing each once it is copied.

    The inputs arrive boxed in one-element lists so this function can drop the
    caller's reference as soon as each side has been copied. Passing them as
    plain parameters would keep both alive for the whole call and add a full
    copy of the larger side to the peak.

    The result needs no second deduplication because ``_filter_absent`` has
    already removed the overlap, which saves another full-size copy.

    Args:
        box_a: One-element list holding the first sorted array.
        box_b: One-element list holding the second sorted array.

    Returns:
        The sorted union of the two inputs.
    """
    a, b = box_a[0], box_b[0]
    out = np.empty(a.size + b.size, dtype=a.dtype)

    n_a = a.size
    out[:n_a] = a
    box_a[0] = None
    del a

    out[n_a:] = b
    box_b[0] = None
    del b

    # Exactly two ascending runs, which is the case quicksort handles best.
    out.sort(kind="quicksort")
    return out


def _read_spectrum_mzs(parser: Any, idx: int) -> Optional[NDArray[Any]]:
    """Read one spectrum's m/z values, or None if missing or unreadable.

    Reads and decodes the m/z binary array only, with the same
    seek/read/frombuffer sequence pyimzml's ``getspectrum`` performs --
    that call also fetches the intensity array, which this caller (the
    processed-mode mass axis build) immediately discards, doubling the
    bytes read per spectrum for nothing. Duck-typed parsers without
    pyimzml's offset tables fall back to ``getspectrum``.

    Args:
        parser: An initialized ImzML parser.
        idx: Spectrum index.

    Returns:
        The m/z array, or None when the spectrum is empty or failed to read.
    """
    try:
        mzs = read_spectrum_mzs_only(parser, idx)
    except Exception as e:
        logger.warning(f"Error getting spectrum {idx}: {e}")
        return None

    if mzs is None:
        return None
    return mzs if mzs.size else None


class _MassAxisAccumulator:
    """Builds a sorted, unique m/z axis without holding every spectrum.

    Values are copied into a scratch buffer; when that buffer fills, it is
    sorted, deduplicated, and merged into the running axis. The buffer's
    capacity tracks the axis length, so the number of merges is logarithmic in
    the input rather than linear -- a fixed batch size would re-copy the
    accumulator once per batch, which is the quadratic behaviour that makes a
    naive ``np.union1d`` fold far slower than the code it replaces.

    Peak memory therefore depends on the number of *unique* m/z values rather
    than on the size of the file.
    """

    def __init__(self, total_spectra: int, max_length: Optional[int] = None) -> None:
        """Initialize the accumulator.

        Args:
            total_spectra: Spectrum count, used only in the error message.
            max_length: Cap on unique m/z values, or None for unlimited.
        """
        self._total_spectra = total_spectra
        self._max_length = max_length
        self._acc: Optional[NDArray[Any]] = None
        self._buf: Optional[NDArray[Any]] = None
        self._cap = 0
        self._cap_target = _MASS_AXIS_BATCH_VALUES
        self._n = 0
        self.saw_any = False

    def add(self, mzs: NDArray[Any], index: int) -> None:
        """Buffer one spectrum's m/z values, folding first if the buffer is full.

        Args:
            mzs: The spectrum's m/z values. Not modified.
            index: Spectrum index, used only in the error message.
        """
        m = mzs.size
        if not m:
            return
        self.saw_any = True

        if self._buf is None:
            self._allocate(m, mzs.dtype)
        elif self._n + m > self._cap:
            self._fold(index)
            if self._buf is None:
                self._allocate(m, mzs.dtype)
            elif m > self._cap:
                self._buf = None
                self._allocate(m, mzs.dtype, exact=True)

        assert self._buf is not None
        self._buf[self._n : self._n + m] = mzs
        self._n += m

    def finish(self, index: int) -> NDArray[Any]:
        """Fold whatever is buffered and return the axis.

        Args:
            index: Last spectrum index, used only in the error message.

        Returns:
            Sorted, unique m/z values.

        Raises:
            ConversionRefused: If no spectrum yielded any m/z values.
        """
        self._fold(index)
        self._buf = None

        if not self.saw_any or self._acc is None:
            raise ConversionRefused("No spectra found to build common mass axis")
        if self._acc.size == 0:
            raise ConversionRefused("Failed to extract any m/z values")
        return self._acc

    def _allocate(self, m: int, dtype: Any, exact: bool = False) -> None:
        """Allocate the scratch buffer, at least large enough for one spectrum."""
        self._cap = m if exact else max(self._cap_target, m)
        self._buf = np.empty(self._cap, dtype=dtype)

    def _fold(self, index: int) -> None:
        """Sort, deduplicate and merge the buffered batch into the axis."""
        if self._n == 0 or self._buf is None:
            return

        view = self._buf[: self._n]
        view.sort(kind="quicksort")
        run = _dedupe_sorted(view)
        self._n = 0

        if self._cap > _MASS_AXIS_BATCH_VALUES:
            # The scratch has grown to axis size, so release it before the
            # merge allocates. Gating on the batch floor keeps this from firing
            # on every one of the thousands of folds a shared-grid dataset
            # performs, where re-faulting the buffer each time costs more than
            # the memory it frees.
            view = None
            self._buf = None
            self._cap = 0

        if self._acc is None:
            self._acc = run
        else:
            new = _filter_absent(self._acc, run)
            run = None  # drop the batch before the merge allocates
            if new.size:
                box_a, box_b = [self._acc], [new]
                self._acc = None
                new = None
                self._acc = _merge_disjoint(box_a, box_b)

        self._check_limit(index)

        # Record the next capacity but do NOT allocate it here. On the final
        # fold the stream is already exhausted, so allocating would commit a
        # whole axis-sized buffer that is never written.
        self._cap_target = max(_MASS_AXIS_BATCH_VALUES, self._acc.size)
        if self._cap and self._cap != self._cap_target:
            self._buf = None
            self._cap = 0

    def _check_limit(self, index: int) -> None:
        """Stop if the axis has outgrown ``max_length``."""
        if self._max_length is None or self._acc is None:
            return
        if self._acc.size <= self._max_length:
            return
        raise ConversionRefused(
            f"Common mass axis exceeded {self._max_length:,} unique m/z values "
            f"after {index + 1:,} of {self._total_spectra:,} spectra "
            f"({self._acc.size:,} so far). The peak lists in this dataset do "
            "not share m/z values, so a raw axis grows to roughly one column "
            "per peak, which is not usable downstream. Convert with resampling "
            "instead (it is the default; --no-resample disables it), or raise "
            "max_mass_axis_length."
        )


def _validate_max_mass_axis_length(value: Any) -> Optional[int]:
    """Check the raw-axis cap, which is a count of unique m/z values.

    Refused here rather than at extraction time, so a bad value fails while
    the caller is still looking at their own arguments -- the same reason
    ``spectrum_type`` is normalised in the constructor. Until then ``-1``,
    ``0`` and ``2.5`` were all accepted and then compared against a growing
    axis, so every file was refused for "exceeding" a cap it could not
    possibly satisfy, in a message that named the nonsense value back at the
    person who set it (issue #261).

    ``bool`` is rejected explicitly: it is an ``int`` subclass, so
    ``max_mass_axis_length=True`` would otherwise mean a cap of one.

    Args:
        value: What the caller passed, or the default.

    Returns:
        The value unchanged, once it is ``None`` or a positive int.

    Raises:
        ConversionRefused: On anything else.
    """
    if value is None or (
        isinstance(value, int) and not isinstance(value, bool) and value > 0
    ):
        return value

    raise ConversionRefused(
        f"max_mass_axis_length must be a positive integer or None (no limit), "
        f"got {value!r}. It caps how many unique m/z values a processed-mode "
        f"raw axis may reach before the build gives up; the default is "
        f"{DEFAULT_MAX_MASS_AXIS_LENGTH:,}."
    )


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
        """Initialize an ImzML reader.

        Args:
            data_path: Path to the imzML file
            batch_size: Default batch size for spectrum iteration
            cache_coordinates: Whether to cache coordinates upfront
            **kwargs: Additional arguments. ``max_mass_axis_length`` caps the
                number of unique m/z values the processed-mode raw axis may
                reach before the build gives up. Defaults to
                ``DEFAULT_MAX_MASS_AXIS_LENGTH`` (10 million, SCiLS Lab's own
                limit); pass ``None`` explicitly for unlimited. See
                ``_extract_continuous_mass_axis``.

                ``spectrum_type`` declares the spectrum representation
                explicitly -- ``"profile"`` or ``"centroid"`` -- instead of
                letting Thyra read or guess it. SCiLS Lab spells the same thing
                ``--rep_type``. Use it when a file declares the wrong
                representation; it outranks the file's own ``MS:1000127`` /
                ``MS:1000128``, and contradicting a declaration is logged as a
                warning. Defaults to ``None`` (detect).
        """
        super().__init__(data_path, **kwargs)
        self.filepath: Optional[Union[str, Path]] = data_path
        self.batch_size: int = batch_size
        self.cache_coordinates: bool = cache_coordinates
        # Absent means "use the default"; an explicit None means "unlimited",
        # so this cannot be a bare ``.get(...)`` with a None fallback.
        self.max_mass_axis_length: Optional[int] = _validate_max_mass_axis_length(
            kwargs.get("max_mass_axis_length", DEFAULT_MAX_MASS_AXIS_LENGTH)
        )
        # Validated here rather than at extraction time, so a bad value fails
        # while the caller is still looking at its own arguments.
        self.spectrum_type: Optional[str] = normalize_spectrum_type(
            kwargs.get("spectrum_type")
        )
        self.parser: Optional[ImzMLParser] = None
        self.ibd_file: Optional[Any] = None
        self.imzml_path: Optional[Path] = None
        self.ibd_path: Optional[Path] = None
        self.is_continuous: bool = False
        self.is_processed: bool = False

        # Parser initialization flag for lazy loading
        self._parser_initialized: bool = False
        # And the failure, if it failed. Initialization is expensive -- 63
        # seconds of XML on a 2.1 GB imzML -- and every public entry point
        # calls _ensure_parser_initialized, so a refused file would otherwise
        # be parsed again for each of them before failing the same way.
        self._parser_init_error: Optional[Exception] = None

        # Cached properties
        self._common_mass_axis: Optional[NDArray[np.float64]] = None
        self._coordinates_array: Optional[NDArray[np.int32]] = (
            None  # Fast numpy array cache
        )
        # See _coordinate_bases(): one memoised pass gives all three.
        self._coordinate_bases_value: Optional[Tuple[int, int, int]] = None

        # Continuous-mode fast read: the shared m/z block, and whether the
        # file really does point every spectrum at one block -- verified
        # from the offsets, not assumed from the declared mode. See
        # _read_spectrum_arrays.
        self._continuous_mzs: Optional[NDArray[np.float64]] = None
        self._continuous_uniform: bool = False

        # The third binary array, when the file has one (see mobility_array.py).
        # Offsets are collected on first use; the shared array is decoded once
        # when every spectrum carries the same one.
        self._mobility: Optional[MobilityArraySpec] = None
        self._mobility_offsets: Optional[
            Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]
        ] = None
        self._mobility_shared: Optional[NDArray[np.float64]] = None
        self._mobility_uniform: Optional[bool] = None

        # Store path but don't initialize parser yet - wait for first use
        if data_path is not None:
            self.filepath = data_path

    def _ensure_parser_initialized(self) -> None:
        """Guarantee parser is initialized exactly once, success or failure.

        Raises:
            ValueError: If no file path was given.
            Exception: Whatever the first initialization attempt raised. A file
                that has been refused once is refused from the memo rather than
                parsed again.
        """
        if self._parser_initialized:
            return
        # The stored exception is re-raised as itself rather than rebuilt as
        # ``type(e)(str(e))``, which would lose both the type and the message
        # for any exception whose constructor takes something other than a
        # single string, and would drop the traceback of the attempt that
        # actually failed.
        if self._parser_init_error is not None:
            raise self._parser_init_error
        if self.filepath is None:
            # Not memoized: nothing was parsed, and the caller can still fix it
            # by setting a path.
            raise ValueError("No file path provided for parser initialization")
        try:
            self._initialize_parser(self.filepath)
        except Exception as e:
            self._parser_init_error = e
            raise
        self._parser_initialized = True

    def _initialize_parser(self, imzml_path: Union[str, Path]) -> None:
        """Initialize the ImzML parser with the given path.

        Args:
            imzml_path: Path to the imzML file to parse

        Raises:
            ConversionRefused: If the corresponding .ibd file is not found or
                metadata parsing fails
            Exception: If parser initialization fails
        """
        if isinstance(imzml_path, str):
            imzml_path = Path(imzml_path)

        self.imzml_path = imzml_path
        self.ibd_path = imzml_path.with_suffix(".ibd")

        if not self.ibd_path.exists():
            raise ConversionRefused(
                f"Corresponding .ibd file not found for {imzml_path}"
            )

        # Open the .ibd file for reading
        self.ibd_file = open(self.ibd_path, mode="rb")

        # Initialize the parser
        logger.info(f"Initializing ImzML parser for {imzml_path}")
        try:
            # ElementTree, NOT lxml. pyimzml prunes each <spectrum> out of the
            # tree (`slist.remove(elem)`) while iterparse is still streaming the
            # document, which invalidates libxml2's text-node coalescing
            # accelerator: ctxt->nodelen/nodemem go on describing a text node
            # inside the subtree that was just removed, so later character data
            # is appended at a stale offset and the buffer is doubled on every
            # miss until xmlRealloc fails. lxml surfaces libxml2's
            # XML_ERR_NO_MEMORY as a *syntax* error, so it reads as a corrupt
            # file when nothing is wrong with it:
            #     XMLSyntaxError: xmlSAX2Characters, line 212575, column 1
            # It only bites when the text between </spectrum> and <spectrum> is
            # exactly "\r\n" -- CRLF with no indentation, how IONTOF SurfaceLab
            # writes imzML. Indented or LF-only files coalesce differently and
            # survive, which is why most files never hit it. ElementTree builds
            # the tree in Python, so there is no C parser state to invalidate;
            # it is also pyimzml's own default, parses byte-identically, and is
            # faster here -- 67s vs 118s on a 2.1 GB imzML at the same peak RSS.
            #
            # Two more pyimzml constraints live on this call and neither is
            # visible from it. The parser holds one unsynchronised file handle:
            # get_spectrum_as_string does two seek-then-read pairs on self.m, so
            # sharing a parser across threads returns another pixel's masses at
            # the correct declared length -- 5.4% of reads over 8 threads on
            # real pea, 84 of 108 undetectable. Harmless today because these
            # reads are strictly serial; a trap directly under the obvious
            # parallelisation. And ibd_file= below is not a convenience: without
            # it pyimzml resolves the .ibd itself, by a rule that is worse than
            # this one on case, on directories and on partial downloads.
            # docs/imzml-parser-notes.md has both, with the measurements.
            # A third hazard, and the reason the ion mobility exports of
            # TIMSCONVERT and TIMSImaging could not be opened at all: see
            # _pyimzml_compat.
            ensure_lenient_cv_param_values()
            self.parser = ImzMLParser(
                filename=str(imzml_path),
                parse_lib="ElementTree",
                ibd_file=self.ibd_file,
            )
        except Exception as e:
            if self.ibd_file:
                self.ibd_file.close()
            logger.error(f"Failed to initialize ImzML parser: {e}")
            raise

        if self.parser.metadata is None:
            raise ConversionRefused("Failed to parse metadata from imzML file.")

        # Determine file mode
        # Determine if file is continuous mode
        self.is_continuous = (
            "continuous" in self.parser.metadata.file_description.param_by_name
        )
        # Determine if file is processed mode
        self.is_processed = (
            "processed" in self.parser.metadata.file_description.param_by_name
        )

        if self.is_continuous == self.is_processed:
            raise ConversionRefused(
                "Invalid file mode, expected either 'continuous' or " "'processed'."
            )

        # Cache coordinates if requested
        if self.cache_coordinates:
            self._cache_all_coordinates()

        try:
            # The mobility array is found first so the .ibd extent check
            # below can account for it like the other two arrays.
            self._mobility = detect_mobility_array(self.parser.metadata)
            if self._mobility is not None:
                if self._mobility.compressed:
                    raise ConversionRefused(
                        f"{imzml_path} declares zlib compression on its ion "
                        f"mobility array ({self._mobility.array_accession}); "
                        "compressed binary arrays are not supported"
                    )
                logger.info(
                    "imzML carries an ion mobility array: %s (%s), unit %s",
                    self._mobility.array_name,
                    self._mobility.array_accession,
                    self._mobility.unit_accession or "undeclared",
                )
            self._validate_parser_state()
        except Exception:
            self.close()
            raise

    def _validate_parser_state(self) -> None:
        """Check pyimzml's parser state against the .ibd before anything reads.

        pyimzml seeks and reads unconditionally, and ``np.frombuffer`` objects
        only when the byte count is not a whole multiple of the item size -- so
        an offset pointing past the end of the ``.ibd`` yields an *empty* array
        rather than an error, and the affected pixels leave the store without a
        word. Nothing else in Thyra reads ``mzOffsets``, ``intensityOffsets``,
        ``mzLengths`` or ``intensityLengths``, and nothing else stats the
        ``.ibd``; the only check that exists today is that the file is there.

        What is refused, in order:

        1. ``MS:1000574 zlib compression`` on either binary array. pyimzml
           1.5.5 has no decompression path at all, so it would hand raw deflate
           bytes to ``np.frombuffer`` and get numbers of the declared length.
        2. A param group declaring no precision term, more than one, or one
           that disagrees with the precision pyimzml resolved -- pyimzml breaks
           ties by dictionary order rather than by the document. Also
           ``64-bit integer``, whose width is platform-dependent.
        3. Spectrum 0's ``IMS:1000104`` encoded byte length disagreeing with
           ``IMS:1000103 x itemsize``. This is the only check that catches a
           correct accession carrying a wrong *name*, which makes pyimzml
           decode float64 bytes as float32 at exactly the right length.
        4. A negative offset or length, or a spectrum whose m/z and intensity
           arrays declare different numbers of values.
        5. A spectrum whose array ends past the end of the ``.ibd``.

        Warned about but allowed:

        - non-monotonic offsets, and a maximum end byte short of the file
          size. Both are legal.
        - more than one ``<scanSettings>`` block. Also legal, and mishandled
          downstream -- pyimzml resolves each scan-settings accession by first
          match anywhere in the list, so a two-block file yields a
          per-accession chimera -- but refusing it belongs with the pixel-size
          unit work rather than here.
        - an ``IMS:1000080`` UUID the ``.ibd`` header does not carry. A
          warning rather than a refusal on measured evidence; see
          ``_check_ibd_uuid``.

        Limitation: this runs *after* ``ImzMLParser.__fix_offsets``, which
        silently adds 2**32 to every offset from the first positive-to-negative
        transition onward. Seeing the raw offsets would need an override of the
        name-mangled ``_ImzMLParser__fix_offsets``; that buys exactly one case
        these checks miss -- a spurious negative on the very last spectrum,
        where the repair leaves no later read to push past the end of the file
        -- and roughly doubles what validation costs. The repair is dormant on
        every file measured (0 negatives and 0 inversions in 1,896,000 offsets
        across bellini, pea and xenium), so it is left alone.

        Raises:
            ValueError: If the parser's state cannot produce correct reads.
        """
        parser = cast(ImzMLParser, self.parser)

        for label, group_id, precision in _binary_array_specs(parser):
            self._validate_binary_group(parser, label, group_id, precision)

        self._validate_encoded_lengths(parser)

        arrays = _offset_arrays(parser)
        self._validate_offset_arrays(arrays)
        self._validate_ibd_extent(parser, arrays)
        self._check_ibd_uuid(parser)

        # Does every spectrum point at one shared m/z block? True for a
        # well-formed continuous file, and the licence for the fast read in
        # _read_spectrum_arrays -- measured off the offsets rather than
        # trusted from the declared mode, so a file that declares
        # continuous while writing per-spectrum m/z blocks is read the
        # slow, correct way instead of being handed spectrum 0's axis.
        if self.is_continuous and arrays.mz_offsets.size:
            self._continuous_uniform = bool(
                (arrays.mz_offsets == arrays.mz_offsets[0]).all()
                and (arrays.mz_lengths == arrays.mz_lengths[0]).all()
            )

        n_scan_settings = len(parser.metadata.scan_settings)
        if n_scan_settings != 1:
            logger.warning(
                f"imzML declares {n_scan_settings} <scanSettings> blocks. "
                "pyimzml resolves each scan-settings accession by first match "
                "anywhere in the list, so the pixel size and pixel counts "
                "Thyra reads may come from different blocks and describe no "
                "single region."
            )

    def _validate_binary_group(
        self,
        parser: ImzMLParser,
        label: str,
        group_id: Any,
        precision: Optional[str],
    ) -> None:
        """Check the referenceable param group behind one binary array.

        Args:
            parser: An initialized ImzML parser.
            label: ``"m/z"`` or ``"intensity"``, for the messages.
            group_id: The param group id pyimzml resolved for this array.
            precision: The precision character pyimzml resolved for it.

        Raises:
            ConversionRefused: If the group is missing, declares zlib
                compression, declares no precision term or more than one,
                disagrees with the precision pyimzml resolved, or names a type
                whose width is platform-dependent.
        """
        groups = parser.metadata.referenceable_param_groups
        group = groups.get(group_id)
        if group is None or precision is None:
            raise ConversionRefused(
                f"imzML declares no usable referenceable param group for the "
                f"{label} array, so pyimzml cannot know how to decode it "
                f"(looked for {group_id!r} among {sorted(map(str, groups))})."
            )

        if _ZLIB_ACCESSION in group or _ZLIB_NAME in group:
            raise ConversionRefused(
                f"imzML declares zlib compression ({_ZLIB_ACCESSION}) on its "
                f"{label} array. pyimzml 1.5.5 has no decompression path: it "
                f"reads IMS:1000103 x itemsize raw deflate bytes and decodes "
                f"them as numbers, which succeeds silently at the declared "
                f"length. Re-export with MS:1000576 no compression."
            )

        # param_by_name, not the group's own cv_params: declaring the precision
        # in a param group this one *references* is legal, and reading
        # cv_params would refuse that file.
        declared = [
            name for name in parser.precisionDict if name in group.param_by_name
        ]
        if len(declared) != 1:
            raise ConversionRefused(
                f"imzML param group {group_id!r} declares {len(declared)} "
                f"precision terms for the {label} array "
                f"({', '.join(declared) if declared else 'none'}); exactly one "
                f"is required. pyimzml breaks ties by dictionary order rather "
                f"than by the document, so the decode width would be arbitrary."
            )

        resolved = parser.precisionDict[declared[0]]
        if resolved != precision:
            raise ConversionRefused(
                f"imzML param group {group_id!r} declares {declared[0]!r} for "
                f"the {label} array, which is {resolved!r}, but pyimzml "
                f"resolved {precision!r} and will decode at that width."
            )

        if precision in _PLATFORM_DEPENDENT_PRECISIONS:
            raise ConversionRefused(
                f"imzML declares {declared[0]!r} for its {label} array. "
                f"pyimzml reads {parser.sizeDict[precision]} bytes per value "
                f"but decodes them as numpy's 'l', whose itemsize is "
                f"{np.dtype(precision).itemsize} on this platform, so the same "
                f"file reads differently on Windows and on Linux. Re-export "
                f"the array as 32-bit or 64-bit float."
            )

    def _validate_encoded_lengths(self, parser: ImzMLParser) -> None:
        """Cross-check spectrum 0's IMS:1000104 against IMS:1000103 x itemsize.

        pyimzml's ``ACCESSION_FIX_MAPPING`` rewrites the *accession* of a
        mis-declared 32/64-bit float term and keeps the raw *name*, and the
        precision is then derived from the name -- so a correct ``MS:1000523``
        carrying the name ``64-bit float`` makes every m/z value in the file
        decode at the wrong width, at exactly the declared length, announced
        only by a ``UserWarning`` worded as a successful repair. ``IMS:1000104``
        is the independent witness: it equalled ``IMS:1000103 x itemsize`` for
        all ~1.9M arrays across bellini, pea and xenium, 0 violations.

        Checked on spectrum 0 only. Every spectrum would mean re-reading the
        whole document.

        Args:
            parser: An initialized ImzML parser.

        Raises:
            ConversionRefused: If a declared encoded length contradicts the
                precision pyimzml resolved.
        """
        if self.imzml_path is None:
            return
        try:
            arrays = _first_spectrum_array_lengths(self.imzml_path)
        except ElementTree.ParseError as e:
            # pyimzml has already parsed this document successfully, so a
            # failure here is this function's problem and must not condemn the
            # file.
            logger.debug(
                f"Could not re-read spectrum 0 for the "
                f"{_ENCODED_LENGTH_ACCESSION} cross-check: {e}"
            )
            return

        for label, group_id, precision in _binary_array_specs(parser):
            declared = arrays.get(group_id)
            if declared is None:
                continue
            array_length, encoded_length = declared
            if array_length is None or encoded_length is None:
                continue
            itemsize = parser.sizeDict[precision]
            expected = array_length * itemsize
            if encoded_length != expected:
                raise ConversionRefused(
                    f"imzML spectrum 0 declares {encoded_length:,} encoded "
                    f"bytes ({_ENCODED_LENGTH_ACCESSION}) for its {label} "
                    f"array, but its {array_length:,} values at the resolved "
                    f"precision {precision!r} occupy {expected:,} bytes "
                    f"({_ARRAY_LENGTH_ACCESSION} x {itemsize}). pyimzml reads "
                    f"{expected:,} bytes, so the array decodes at the wrong "
                    f"width or the wrong length."
                )

    def _validate_offset_arrays(self, arrays: _OffsetArrays) -> None:
        """Check the offset and length arrays for internally impossible values.

        Note that ``mz_len == int_len`` is not a corruption guard: the lengths
        come from ``IMS:1000103`` while pyimzml's offset repair touches only
        ``IMS:1000102``, so a re-pointed read agrees with itself. It is here
        because it is nearly free and it does catch truncation faults.

        Args:
            arrays: The parser's offsets and lengths, as int64.

        Raises:
            ConversionRefused: If any value is negative, or if a spectrum's m/z
                and intensity arrays declare different numbers of values.
        """
        for label, values in (
            ("m/z offset", arrays.mz_offsets),
            ("m/z length", arrays.mz_lengths),
            ("intensity offset", arrays.int_offsets),
            ("intensity length", arrays.int_lengths),
        ):
            negative = np.flatnonzero(values < 0)
            if negative.size:
                idx = int(negative[0])
                raise ConversionRefused(
                    f"imzML spectrum {idx} declares a negative {label} of "
                    f"{int(values[idx]):,} ({negative.size:,} spectra "
                    f"affected). Offsets and lengths are byte and element "
                    f"counts and cannot be negative; pyimzml's signed-32-bit "
                    f"offset repair did not remove this one."
                )

        mismatch = np.flatnonzero(arrays.mz_lengths != arrays.int_lengths)
        if mismatch.size:
            idx = int(mismatch[0])
            raise ConversionRefused(
                f"imzML spectrum {idx} declares {int(arrays.mz_lengths[idx]):,} "
                f"m/z values but {int(arrays.int_lengths[idx]):,} intensity "
                f"values ({mismatch.size:,} spectra disagree). A spectrum's two "
                f"arrays describe the same peaks and must be the same length."
            )

    def _validate_ibd_extent(self, parser: ImzMLParser, arrays: _OffsetArrays) -> None:
        """Check that every declared array lies inside the .ibd.

        Args:
            parser: An initialized ImzML parser.
            arrays: The parser's offsets and lengths, as int64.

        Raises:
            ConversionRefused: If any spectrum's array ends past the end of the
                ``.ibd``.
        """
        if self.ibd_path is None or arrays.mz_offsets.size == 0:
            return
        ibd_size = self.ibd_path.stat().st_size

        declared = [
            (
                "m/z",
                arrays.mz_offsets,
                arrays.mz_lengths,
                parser.sizeDict[parser.mzPrecision],
            ),
            (
                "intensity",
                arrays.int_offsets,
                arrays.int_lengths,
                parser.sizeDict[parser.intensityPrecision],
            ),
        ]
        if self._mobility is not None:
            # The third array is subject to the same check; without it a
            # mobility export always looked like it had trailing bytes.
            mob_offsets, mob_lengths, _ = self._ensure_mobility_offsets()
            declared.append(
                (
                    "ion mobility",
                    mob_offsets,
                    mob_lengths,
                    self._mobility.dtype.itemsize,
                )
            )

        max_end = 0
        for label, offsets, lengths, itemsize in declared:
            end = offsets + lengths * itemsize
            max_end = max(max_end, int(end.max()))
            past = np.flatnonzero(end > ibd_size)
            if past.size:
                idx = int(past[0])
                raise ConversionRefused(
                    f"imzML spectrum {idx} declares a {label} array ending at "
                    f"byte {int(end[idx]):,}, but {self.ibd_path.name} is "
                    f"{ibd_size:,} bytes ({past.size:,} spectra are affected; "
                    f"the furthest ends at {int(end.max()):,}). The .ibd is "
                    f"truncated or its offsets are wrong -- pyimzml would "
                    f"return empty arrays for these spectra without raising, "
                    f"and they would simply be missing from the output."
                )

        if max_end != ibd_size:
            logger.warning(
                f"{self.ibd_path.name} is {ibd_size:,} bytes but the last byte "
                f"any spectrum declares is {max_end:,}, leaving "
                f"{ibd_size - max_end:,} unaccounted for. Trailing bytes are "
                f"legal, but this is also what a partially-copied .ibd looks "
                f"like."
            )

        if np.any(np.diff(arrays.mz_offsets) < 0) or np.any(
            np.diff(arrays.int_offsets) < 0
        ):
            logger.warning(
                "imzML offsets are not monotonically non-decreasing. This is "
                "legal, but pyimzml's signed-32-bit offset repair assumes "
                "document order matches byte order and silently rewrites every "
                "offset after a sign flip when it does not."
            )

    def _check_ibd_uuid(self, parser: ImzMLParser) -> None:
        """Say so when the ``.ibd`` header does not carry the declared UUID.

        The imzML specification puts the binary file's UUID in the first 16
        bytes of the ``.ibd`` and the same value in the XML as
        ``IMS:1000080``. Matching them is the only check that tells an
        ``.imzML`` apart from a *different* acquisition's ``.ibd`` sitting
        beside it under the right name -- every other check here reads the
        XML's own numbers against the ``.ibd``'s size, which a wrong-but-
        plausible pairing can satisfy. Thyra read the term for the metadata
        store and never compared it (issue #261).

        **A warning, not a refusal**, and that is measured rather than
        cautious. Of the three real files in the corpus, ``pea`` and the
        Xenium export match byte for byte; ``bellini``, an IONTOF SurfaceLab
        export, declares ``{FC37F303-A9C0-4CD3-A28E-1D18E523C269}`` while its
        ``.ibd`` header reads ``3ad1bacd-dcc3-4f7b-aea7-f9b375dbf731``. Its
        first spectrum starts at byte 16, so the header slot is there and
        populated -- the writer simply put two different values in the two
        places. That file converts correctly under every other check, so a
        refusal here would reject data Thyra reads right today, for a
        disagreement between two copies of an identifier neither of which is
        used to locate a byte.

        Silent when either side is absent: a file that declares no UUID, or
        an ``.ibd`` shorter than its header, has nothing to compare and the
        extent checks above already speak for the truncated case.
        """
        if self.ibd_file is None:
            return

        declared = None
        try:
            param_by_name = parser.metadata.file_description.param_by_name
            value = param_by_name.get(ImzMLAccessions.UUID_NAME)
            if isinstance(value, str):
                # Vendors differ on the registry-format braces (IONTOF writes
                # them, SCiLS does not) and on case; the hyphens are
                # positional, not data. Compare the 32 hex digits alone.
                declared = value.strip().strip("{}").replace("-", "").lower()
        except AttributeError:
            return
        if not declared:
            return

        try:
            here = self.ibd_file.tell()
            self.ibd_file.seek(0)
            header = self.ibd_file.read(16)
            self.ibd_file.seek(here)
        except OSError as e:
            logger.debug("Could not read the .ibd UUID header: %s", str(e))
            return
        if len(header) != 16:
            return

        found = header.hex()
        if found == declared:
            return

        name = self.ibd_path.name if self.ibd_path else ".ibd"
        logger.warning(
            "imzML declares binary-file UUID %s (IMS:1000080) but %s begins "
            "with %s. The two are meant to be the same value, so either the "
            "writer filled the two places independently, or %s is paired "
            "with another acquisition's binary file. Reading continues -- the "
            "offsets, not the UUID, locate the data -- but check the pair if "
            "the spectra look wrong.",
            declared,
            name,
            found,
            self.imzml_path.name if self.imzml_path else "this .imzML",
        )

    def _cache_all_coordinates(self) -> None:
        """Cache all coordinates for faster access.

        Converts the file's coordinates to 0-based coordinates for internal
        use. Uses vectorized numpy operations for speed. Stores as numpy
        array for O(1) index lookup without dict overhead.

        No axis gets a constant base. x and y are 1-based by the imzML
        specification, but 0-based exports exist and subtracting a constant
        1 cost them their first row and column (issue #244); z is not
        reliably either base. All three come from
        :meth:`_coordinate_bases`.
        """
        # Parser should already be initialized when this is called from
        # _initialize_parser

        if self.parser is None:
            raise RuntimeError("Parser is not initialized")
        n_coords = len(self.parser.coordinates)
        logger.info(f"Caching {n_coords:,} coordinates...")

        x_base, y_base, z_base = self._coordinate_bases()

        # Vectorized conversion using numpy (much faster than Python loop)
        # np.array() on the coordinates list is the main cost here
        self._coordinates_array = np.array(self.parser.coordinates, dtype=np.int32)

        # Convert to 0-based in place
        self._coordinates_array[:, 0] -= x_base
        self._coordinates_array[:, 1] -= y_base
        self._coordinates_array[:, 2] -= z_base

        logger.info(f"Cached {n_coords:,} coordinates as numpy array")

    def _create_metadata_extractor(self) -> MetadataExtractor:
        """Create ImzML metadata extractor."""
        self._ensure_parser_initialized()

        if not self.imzml_path:
            raise ValueError("ImzML path not available")

        return ImzMLMetadataExtractor(
            self.parser, self.imzml_path, spectrum_type=self.spectrum_type
        )

    @property
    def has_shared_mass_axis(self) -> bool:
        """Check if all spectra share the same m/z axis.

        Returns True for continuous ImzML (all pixels have same m/z values),
        False for processed ImzML (each pixel has different m/z values).
        """
        return self.is_continuous

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """Return the common mass axis composed of all unique m/z values.

        For continuous mode, returns the m/z values from the first spectrum.
        For processed mode, collects all unique m/z values across spectra.

        Returns:
            NDArray[np.float64]: Array of m/z values in ascending order

        Raises:
            ValueError: If the common mass axis cannot be created
        """
        self._ensure_parser_initialized()

        if self._common_mass_axis is None:
            # We know parser is not None at this point
            parser = cast(ImzMLParser, self.parser)

            if self.is_continuous:
                logger.info("Using m/z values from first spectrum (continuous mode)")
                if self._continuous_mzs is not None:
                    # Iteration already decoded the shared block; reuse the
                    # object so axis and yielded m/z stay identity-equal
                    # whichever is asked for first (unless the block has
                    # duplicates, in which case the axis is its unique form).
                    self._common_mass_axis = self._deduplicated_shared_axis(
                        self._continuous_mzs
                    )
                else:
                    spectrum_data = parser.getspectrum(0)
                    if spectrum_data is None or len(spectrum_data) < 1:
                        raise ValueError("Could not get first spectrum")

                    mzs = spectrum_data[0]
                    if mzs.size == 0:
                        raise ConversionRefused("First spectrum contains no m/z values")

                    self._common_mass_axis = self._deduplicated_shared_axis(mzs)
                    if self._continuous_uniform:
                        self._continuous_mzs = mzs
            else:
                self._common_mass_axis = self._extract_continuous_mass_axis(parser)

        # Return the common mass axis
        return self._common_mass_axis

    def _extract_continuous_mass_axis(self, parser: ImzMLParser) -> NDArray[np.float64]:
        """Build the common mass axis for processed-mode data.

        Returns exactly what ``np.unique(np.concatenate(all_mzs))`` returned:
        sorted, deduplicated, with the dtype following the file's mzPrecision.

        This streams the file rather than collecting it. The previous
        implementation held every spectrum's m/z array in a list, concatenated
        that into one array, and handed the result to ``np.unique`` -- three
        live copies of the whole m/z payload P, plus the output. Measured peak
        was 3.40-3.51x P on data whose spectra share m/z values and 4.32-4.39x
        P when every value is distinct, which put a hard ceiling around a
        40 GB input on a 128 GB machine.

        Here each spectrum is copied into a scratch buffer, and when that
        buffer fills it is sorted, deduplicated, and merged into the running
        axis. Peak memory becomes a function of the number of *unique* values
        rather than the payload: measured 15.2-15.5 MiB, flat, for payloads
        from 32 MiB to 16 GiB of shared-grid data, and 2.00x P in the
        all-distinct case, which is the structural floor for an out-of-place
        merge.

        Args:
            parser: An initialized ImzML parser.

        Returns:
            Sorted, unique m/z values across all spectra.

        Raises:
            ValueError: If no spectrum yielded any m/z values, or if
                ``max_mass_axis_length`` is set and the axis outgrows it.
        """
        logger.info(
            "Building common mass axis from all unique m/z values " "(processed mode)"
        )

        total_spectra = len(parser.coordinates)
        accumulator = _MassAxisAccumulator(
            total_spectra, getattr(self, "max_mass_axis_length", None)
        )
        idx = -1

        with tqdm(
            total=total_spectra,
            desc="Building common mass axis",
            unit="spectrum",
        ) as pbar:
            for idx in range(total_spectra):
                mzs = _read_spectrum_mzs(parser, idx)
                if mzs is not None:
                    # Deliberately outside the read's try/except: a
                    # max_mass_axis_length failure must not be swallowed and
                    # logged as a per-spectrum warning.
                    accumulator.add(mzs, idx)
                mzs = None
                pbar.update(1)

        axis = accumulator.finish(idx)
        logger.info(f"Created common mass axis with {axis.size} unique m/z values")
        return axis

    def _coordinate_bases(self) -> Tuple[int, int, int]:
        """The ``(x, y, z)`` values in this file that map onto index 0.

        Measured off the file rather than assumed, because imzML gives no
        guarantee about the base and pyimzml is inconsistent about it:
        when ``IMS:1000052`` is absent it synthesises ``z = 1``, but an
        explicit ``z = 0`` is passed through verbatim -- contradicting its
        own docstring, which promises zero.

        Both z conventions were previously folded onto plane 0 by
        ``np.maximum(z - 1, 0)``. On a file written 0-based that merges
        planes 0 and 1: a two-plane acquisition converts to a two-row table
        whose rows are the union of both planes, with nothing logged.

        x and y are **not** rebased the same way, and
        :mod:`thyra.utils.imzml_coordinate_base` holds the reasoning: they
        have a physical origin that z does not, so only a 0 is folded down
        (issue #244).

        Neither ``bellini``, ``pea`` nor ``xenium`` declares ``IMS:1000052``,
        so all three go through pyimzml's synthesised ``z = 1`` and this
        returns 1 for z -- the same answer the clamp gave. The whole z path
        is exercised only in the shape real data does not have, which is why
        this went unnoticed.

        Memoised: the scan is one pass over ``parser.coordinates``
        (~60 ms at 900k spectra, against a 63 s parse), and the callers
        below would otherwise repeat it per spectrum.

        Returns:
            ``(x_base, y_base, z_base)``.

        Raises:
            RuntimeError: If the parser is not initialised.
        """
        if self._coordinate_bases_value is None:
            if self.parser is None:
                raise RuntimeError("Parser is not initialized")
            self._coordinate_bases_value = coordinate_bases(self.parser.coordinates)
        return self._coordinate_bases_value

    def _get_spectrum_coordinates(
        self, parser: ImzMLParser, idx: int
    ) -> Tuple[int, int, int]:
        """Get 0-based coordinates for a spectrum."""
        if self._coordinates_array is not None:
            # Fast O(1) numpy array lookup
            row = self._coordinates_array[idx]
            return (int(row[0]), int(row[1]), int(row[2]))

        # Fallback: compute on the fly. Every axis is rebased on what the
        # file actually contains -- see _coordinate_bases().
        x, y, z = parser.coordinates[idx]
        x_base, y_base, z_base = self._coordinate_bases()
        return cast(
            Tuple[int, int, int],
            (x - x_base, y - y_base, z - z_base),
        )

    def _read_spectrum_arrays(
        self, parser: ImzMLParser, idx: int
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Read one spectrum's arrays, reusing the shared m/z block when legal.

        pyimzml's ``getspectrum`` seeks and decodes both binary arrays on
        every call. In continuous mode every spectrum references the same
        m/z block (verified against the offsets at init -- see
        ``_continuous_uniform``), so re-reading it per spectrum roughly
        triples the decode work: measured ~1.6 ms per 331k-point spectrum
        for both arrays against ~0.5 ms for the intensities alone, paid
        twice per streaming conversion. Here the m/z block is decoded once
        and only the intensity bytes are read per spectrum, with the same
        seek/read/frombuffer sequence pyimzml itself performs.

        Args:
            parser: An initialized ImzML parser.
            idx: Spectrum index.

        Returns:
            The (m/z, intensity) arrays, exactly as ``getspectrum`` would.
        """
        if not self._continuous_uniform:
            return cast(
                Tuple[NDArray[np.float64], NDArray[np.float64]],
                parser.getspectrum(idx),
            )

        mzs = self._continuous_mzs
        if mzs is None:
            # Prefer the axis object get_common_mass_axis cached, so the
            # arrays this yields are identity-equal to the converter's
            # common axis and its equality checks cost O(1).
            if self._common_mass_axis is not None:
                mzs = self._continuous_mzs = self._common_mass_axis
            else:
                mzs, intensities = parser.getspectrum(idx)
                self._continuous_mzs = mzs
                return mzs, intensities

        m = parser.m
        m.seek(parser.intensityOffsets[idx])
        data = m.read(
            parser.intensityLengths[idx] * parser.sizeDict[parser.intensityPrecision]
        )
        return mzs, np.frombuffer(data, dtype=parser.intensityPrecision)

    def _process_single_spectrum(
        self, parser: ImzMLParser, idx: int, pbar
    ) -> Optional[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]]
    ]:
        """Process a single spectrum and return its data."""
        try:
            coords = self._get_spectrum_coordinates(parser, idx)
            mzs, intensities = self._read_spectrum_arrays(parser, idx)

            # Apply intensity threshold filtering if configured
            mzs, intensities = self._apply_intensity_filter(mzs, intensities)

            if mzs.size > 0 and intensities.size > 0:
                pbar.update(1)
                return coords, mzs, intensities

            pbar.update(1)
            return None
        except Exception as e:
            logger.warning(f"Error processing spectrum {idx}: {e}")
            pbar.update(1)
            return None

    def _iter_spectra_single(
        self, parser: ImzMLParser, total_spectra: int, pbar
    ) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """Process spectra one at a time."""
        for idx in range(total_spectra):
            result = self._process_single_spectrum(parser, idx, pbar)
            if result is not None:
                yield result

    def _iter_spectra_batch(
        self, parser: ImzMLParser, total_spectra: int, batch_size: int, pbar
    ) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """Process spectra in batches."""
        for batch_start in range(0, total_spectra, batch_size):
            batch_end = min(batch_start + batch_size, total_spectra)
            batch_size_actual = batch_end - batch_start

            for offset in range(batch_size_actual):
                idx = batch_start + offset
                result = self._process_single_spectrum(parser, idx, pbar)
                if result is not None:
                    yield result

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """Iterate through spectra with progress monitoring and batch processing.

        Maps m/z values to the common mass axis using searchsorted for
        accurate representation in the output data structures.

        Args:
            batch_size: Number of spectra to process in each batch (None for
                default)

        Yields:
            Tuple containing:
                - Tuple[int, int, int]: Coordinates (x, y, z) - 0-based
                - NDArray[np.float64]: m/z values array
                - NDArray[np.float64]: Intensity values array

        Raises:
            ValueError: If parser is not initialized and no filepath is
                available
        """
        self._ensure_parser_initialized()

        if batch_size is None:
            batch_size = self.batch_size

        parser = cast(ImzMLParser, self.parser)
        total_spectra = len(parser.coordinates)
        dimensions = self.get_essential_metadata().dimensions
        total_pixels = dimensions[0] * dimensions[1] * dimensions[2]

        logger.info(
            f"Processing {total_spectra} spectra in a grid of " f"{total_pixels} pixels"
        )

        with tqdm(
            total=total_spectra,
            desc="Reading spectra",
            unit="spectrum",
            disable=getattr(self, "_quiet_mode", False),
        ) as pbar:
            if batch_size <= 1:
                yield from self._iter_spectra_single(parser, total_spectra, pbar)
            else:
                yield from self._iter_spectra_batch(
                    parser, total_spectra, batch_size, pbar
                )

    def _deduplicated_shared_axis(
        self, mzs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The common axis a shared m/z block defines: its sorted unique values.

        A continuous file may legally repeat an m/z value -- an export with
        ion mobility lists a feature once per mobility at which it was
        found, so two isomers at one m/z are two entries. The MSI table's
        axis is strictly increasing by contract, so the duplicates collapse
        into one column there and their intensities are summed over
        mobility; the mobility-resolved table keeps them apart.
        """
        if mzs.size < 2 or bool(np.all(np.diff(mzs) > 0)):
            return mzs
        axis = np.unique(mzs)
        n_dup = int(mzs.size - axis.size)
        if self._mobility is not None:
            logger.info(
                "Shared m/z block repeats %d value(s), split by ion mobility; the "
                "MSI table sums them per m/z and the mobility table keeps them apart",
                n_dup,
            )
        else:
            logger.warning(
                "Shared m/z block is not strictly increasing (%d repeated or "
                "unordered value(s)); the common axis is its sorted unique form "
                "and coincident intensities are summed",
                n_dup,
            )
        return axis

    # ------------------------------------------------------------------
    # Ion mobility: the third binary array
    # ------------------------------------------------------------------

    @property
    def has_ion_mobility(self) -> bool:
        """Whether the file declares a mobility array beside m/z and intensity."""
        self._ensure_parser_initialized()
        return self._mobility is not None

    @property
    def has_shared_mobility_axis(self) -> bool:
        """Whether every spectrum carries the same (m/z, mobility) pairs.

        Requires continuous mode (one m/z block for the file) and a mobility
        array that is the same for every spectrum -- checked on the array
        contents, not assumed from the mode, because the writer both TIMS
        tools use stores a mobility copy per spectrum even in continuous mode.
        """
        if not self.has_ion_mobility or not self.is_continuous:
            return False
        return self._mobility_axis_is_shared()

    def _ensure_mobility_offsets(
        self,
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """Collect the mobility array's per-spectrum offsets once."""
        if self._mobility_offsets is not None:
            return self._mobility_offsets
        if self._mobility is None or self.imzml_path is None:
            raise ConversionRefused("This imzML declares no ion mobility array")
        parser = cast(ImzMLParser, self.parser)
        n_spectra = len(parser.coordinates)
        offsets, lengths, encoded = collect_array_offsets(
            self.imzml_path, self._mobility.group_id, n_spectra
        )
        missing = int(np.count_nonzero(offsets < 0))
        if missing:
            raise ConversionRefused(
                f"{missing} of {n_spectra} spectra reference no ion mobility array "
                f"although the file declares one ({self._mobility.group_id})"
            )
        mz_lengths = np.fromiter(parser.mzLengths, dtype=np.int64, count=n_spectra)
        mismatch = np.flatnonzero(lengths != mz_lengths)
        if mismatch.size:
            first = int(mismatch[0])
            raise ConversionRefused(
                f"Spectrum {first} declares {int(lengths[first])} ion mobility "
                f"values for {int(mz_lengths[first])} m/z values; the arrays "
                "must be parallel"
            )
        self._mobility_offsets = (offsets, lengths, encoded)
        return self._mobility_offsets

    def _read_mobility_array(self, idx: int) -> NDArray[np.float64]:
        """Decode one spectrum's mobility array from the .ibd."""
        offsets, lengths, _ = self._ensure_mobility_offsets()
        spec = cast(MobilityArraySpec, self._mobility)
        parser = cast(ImzMLParser, self.parser)
        m = parser.m
        m.seek(int(offsets[idx]))
        data = m.read(int(lengths[idx]) * spec.dtype.itemsize)
        values = np.frombuffer(data, dtype=spec.dtype)
        if values.size != int(lengths[idx]):
            raise ConversionRefused(
                f"Spectrum {idx}: ion mobility array truncated in the .ibd "
                f"({values.size} of {int(lengths[idx])} values)"
            )
        return values.astype(np.float64, copy=False)

    def _mobility_axis_is_shared(self) -> bool:
        """Decide once whether one mobility array describes every spectrum.

        Every spectrum must declare the same length, and the arrays of a
        spread of spectra (first, last and up to 64 in between) must be
        byte-identical to the first. Cheap on the feature-list files this
        exists for, and honest: a mismatch means per-pixel mobility.
        """
        if self._mobility_uniform is not None:
            return self._mobility_uniform
        offsets, lengths, _ = self._ensure_mobility_offsets()
        n = int(lengths.size)
        uniform = bool(n > 0 and np.all(lengths == lengths[0]))
        if uniform:
            first = self._read_mobility_array(0)
            probes = np.unique(
                np.concatenate(
                    [
                        np.linspace(0, n - 1, num=min(n, 64), dtype=np.int64),
                        [n - 1],
                    ]
                )
            )
            for idx in probes.tolist():
                if idx == 0 or int(offsets[idx]) == int(offsets[0]):
                    continue
                if not np.array_equal(self._read_mobility_array(idx), first):
                    uniform = False
                    break
            if uniform:
                self._mobility_shared = first
        self._mobility_uniform = uniform
        if not uniform:
            logger.info(
                "Ion mobility values differ between spectra; the mobility "
                "dimension is per pixel and no shared feature axis exists"
            )
        return uniform

    def get_mobility_axis(self) -> Optional[MobilityAxis]:
        """Describe the mobility array, with the shared values when shared."""
        if not self.has_ion_mobility:
            return None
        spec = cast(MobilityArraySpec, self._mobility)
        kind_accession, kind_name = classify_mobility_array(
            spec.array_accession, spec.unit_accession
        )
        values = None
        if self.is_continuous and self._mobility_axis_is_shared():
            values = self._mobility_shared
        return MobilityAxis(
            kind_accession=kind_accession,
            kind_name=kind_name,
            unit_accession=spec.unit_accession,
            unit_name=spec.unit_name,
            array_accession=spec.array_accession,
            values=values,
            source="imzml",
        )

    def get_shared_mobility_features(
        self,
    ) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """The shared ``(mz, mobility)`` pairs of a continuous mobility export."""
        if not self.has_shared_mobility_axis:
            return None
        parser = cast(ImzMLParser, self.parser)
        if self._continuous_mzs is not None:
            mzs = self._continuous_mzs
        else:
            mzs = cast(NDArray[np.float64], parser.getspectrum(0)[0])
            if self._continuous_uniform:
                self._continuous_mzs = mzs
        mobility = cast(NDArray[np.float64], self._mobility_shared)
        if mzs.size != mobility.size:
            raise ConversionRefused(
                f"Shared m/z block has {mzs.size} values but the shared mobility "
                f"array has {mobility.size}"
            )
        return np.asarray(mzs, dtype=np.float64), mobility

    def iter_mobility_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[
            Tuple[int, int, int],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        None,
        None,
    ]:
        """Iterate spectra as ``(coords, mz, mobility, intensity)`` point clouds.

        The three arrays are parallel; the intensity threshold, when set,
        masks all three together so they stay parallel.
        """
        self._ensure_parser_initialized()
        if self._mobility is None:
            raise NotImplementedError(
                f"{self.imzml_path} declares no ion mobility array"
            )
        parser = cast(ImzMLParser, self.parser)
        total_spectra = len(parser.coordinates)
        shared = (
            self._mobility_shared
            if self.is_continuous and self._mobility_axis_is_shared()
            else None
        )
        with tqdm(
            total=total_spectra,
            desc="Reading mobility spectra",
            unit="spectrum",
            disable=getattr(self, "_quiet_mode", False),
        ) as pbar:
            for idx in range(total_spectra):
                try:
                    coords = self._get_spectrum_coordinates(parser, idx)
                    mzs, intensities = self._read_spectrum_arrays(parser, idx)
                    mobility = (
                        shared if shared is not None else self._read_mobility_array(idx)
                    )
                    if mobility.size != mzs.size:
                        raise ConversionRefused(
                            f"{mobility.size} mobility values for {mzs.size} m/z values"
                        )
                    if self._intensity_threshold is not None:
                        keep = intensities >= self._intensity_threshold
                        mzs, mobility, intensities = (
                            mzs[keep],
                            mobility[keep],
                            intensities[keep],
                        )
                    if mzs.size > 0:
                        yield coords, mzs, mobility, intensities
                except Exception as e:
                    logger.warning(f"Error processing mobility spectrum {idx}: {e}")
                finally:
                    pbar.update(1)

    def read(self) -> Dict[str, Any]:
        """Read the entire imzML file and return a structured data dictionary.

        Returns:
            Dict containing:
                - mzs: NDArray[np.float64] - common m/z values array
                - intensities: NDArray[np.float64] - array of intensity arrays
                - coordinates: List[Tuple[int, int, int]] - list of (x,y,z)
                  coordinates
                - width: int - number of pixels in x dimension
                - height: int - number of pixels in y dimension
                - depth: int - number of pixels in z dimension

        Raises:
            ValueError: If parser is not initialized and no filepath is
                available
        """
        self._ensure_parser_initialized()

        # Get common mass axis
        mzs = self.get_common_mass_axis()

        # Get dimensions
        width, height, depth = self.get_essential_metadata().dimensions

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
                self.parser.m.close()
            self.parser = None

    @property
    def n_spectra(self) -> int:
        """Return the total number of spectra in the dataset.

        Returns:
            Total number of spectra (efficient implementation using parser)
        """
        self._ensure_parser_initialized()

        # Use parser coordinates which is efficient
        parser = cast(ImzMLParser, self.parser)
        return len(parser.coordinates)

    def get_total_peak_count(self) -> int:
        """Get total number of peaks across all spectra.

        The imzML header already declares every spectrum's array length
        (``IMS:1000103``, held in ``parser.mzLengths``), so this is a sum
        over an in-memory list. It used to read and decode every binary
        array in the file to count values whose count was already known.

        Returns:
            Total number of peaks across all spectra
        """
        self._ensure_parser_initialized()
        parser = cast(ImzMLParser, self.parser)

        n = len(parser.mzLengths)
        total_peaks = int(
            np.sum(np.fromiter(parser.mzLengths, dtype=np.int64, count=n))
        )
        logger.info(f"Total peak count: {total_peaks:,}")
        return total_peaks

    @property
    def mass_range(self) -> Tuple[float, float]:
        """Return the mass range (min_mz, max_mz) of the dataset.

        Returns:
            Tuple of (min_mz, max_mz) values
        """
        # Get mass range from essential metadata
        essential_metadata = self.get_essential_metadata()
        return essential_metadata.mass_range

    def get_peak_counts_per_pixel(self) -> Optional[NDArray[np.int32]]:
        """Get per-pixel peak counts for CSR indptr construction.

        Returns peak counts collected during metadata extraction.
        This enables optimized streaming conversion without a separate
        counting pass.

        Returns:
            Array of size n_pixels where arr[pixel_idx] = peak_count.
            pixel_idx = z * (n_x * n_y) + y * n_x + x
            Returns None if not available.
        """
        essential_metadata = self.get_essential_metadata()
        return essential_metadata.peak_counts_per_pixel
