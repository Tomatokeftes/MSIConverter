# imzML Parser Notes

Thyra reads imzML through [pyimzML](https://github.com/alexandrovteam/pyimzML)
1.5.5. This page is the written-down half of a survey of everything Thyra relies
on from that library: hazards that are real and verified, but that correctly get
**no code today** — because they are unreachable from Thyra's current call
graph, because the safe thing is already what Thyra does, or because the surface
in question is one Thyra deliberately does not use.

Nothing here is a bug report against the shipped converter. It is the set of
things that would cost somebody a day if they had to rediscover them, listed
where a future reader will look before making the change that reaches them.

---

## `ImzMLParser` is not thread-safe

`ImzMLParser.get_spectrum_as_string` does two unsynchronised seek-then-read
pairs on a single shared file handle:

```python
self.m.seek(offsets[0])
mz_string = self.m.read(lengths[0])
self.m.seek(offsets[1])
intensity_string = self.m.read(lengths[1])
```

Two threads interleaving between a `seek` and its `read` get each other's bytes.
Measured over real `pea.imzML`, eight threads sharing one parser, 2000
`getspectrum` calls:

| | |
|---|---|
| corrupt reads | 108 (5.4%) |
| exceptions raised | 0 |
| corrupt **at the correct declared length** | 84 of 108 |
| corrupt with `mz.size == 0` | 24 |
| corrupt reads with a `threading.Lock` added | 0 of 2000 |

The 84 are the dangerous ones: the array comes back the length the header
promised, holding another pixel's masses. No length check, emptiness check or
dtype check can see it.

**This is not a live defect.** Thyra's imzML reads are strictly serial — a grep
for concurrency primitives across `thyra/` finds three hits, all of them on the
class registry and the Bruker buffer pool, none on the imzML read path, and a
non-blocking-lock proxy over two real conversions confirmed no contention ever
occurs.

It is written down because **parallelising the imzML read is an obvious future
optimisation, and this trap is laid directly underneath it.** The sanctioned
route is `parser.portable_spectrum_reader()` with **one open file object per
worker**, never a shared `ImzMLParser`.

---

## Do not "fix" the `.ibd` resolution to match pyimzml

There are two rules for finding the `.ibd` beside an imzML, and **Thyra's is the
more robust of the two.** This section exists because the asymmetry looks, at a
glance, like Thyra being sloppy.

=== "Thyra (production)"

    ```python
    self.ibd_path = imzml_path.with_suffix(".ibd")
    if not self.ibd_path.exists():
        raise ValueError(...)
    self.ibd_file = open(self.ibd_path, mode="rb")
    ```

=== "pyimzml (`_infer_bin_filename`)"

    ```python
    ibd_path = [f for f in imzml_path.parent.glob('*')
                if re.match(r'.+\.ibd', str(f), re.IGNORECASE)
                and f.stem == imzml_path.stem][0]
    ```

pyimzml's `re.IGNORECASE` **is decorative**: the `f.stem == imzml_path.stem`
comparison sitting beside it is a plain case-sensitive `==`. So the flag buys
nothing, and three ordinary situations behave worse under pyimzml's rule:

| Situation | pyimzml | Thyra |
|---|---|---|
| `SAMPLE.IBD` beside `sample.imzML` | `IndexError: list index out of range` | resolves — `Path.exists()` is case-insensitive on Windows |
| A **parent directory** named `proj.ibd` | the unanchored regex matches the full path, so every same-stem sibling qualifies — *including the imzML itself*, which is what gets returned | unaffected; only the `.ibd` suffix is ever constructed |
| A half-finished download `sample.ibdtmp` | `.+\.ibd` matches it, so it competes with the real file | unaffected |

!!! warning "Do not change `with_suffix('.ibd').exists()` to match pyimzml"
    It would be a regression on all three rows. If the resolution rule ever does
    need to change, it should get *stricter*, not more like `_infer_bin_filename`.

A related consequence worth knowing when writing tests: production always passes
`ibd_file=` explicitly, so it never reaches `_infer_bin_filename` at all. Tests
that construct a bare `ImzMLParser(path)` are exercising a rule the shipped code
does not use, which means they cannot catch a resolution bug in it. Use
`tests.fixtures.imzml_parser.production_parser` instead; it mirrors
`ImzMLReader._initialize_parser` exactly.

Two smaller traps on the same handle:

- **`ibd_file=None` is not the same as omitting the argument.** Omitting it
  infers the path; passing `None` gives a parser that raises
  `AttributeError: 'NoneType' object has no attribute 'seek'` on the first read.
  A caller writing `ibd_file=maybe_handle` gets the unreadable parser whenever
  the handle is `None`.
- **`__enter__`/`__exit__` close a handle the parser does not own.** With
  `ibd_file=f` supplied, `parser.m is f`, and leaving a `with` block closes the
  caller's file. Thyra does not use `with` on a parser; `ImzMLReader.close()`
  already closes the same aliased handle twice, which is idempotent in CPython
  and load-bearing on nothing.

---

## Dead pyimzml surface, for anyone tempted to adopt it

None of the following is reachable from Thyra today. All of it looks useful, and
all of it is broken in a way that would not be obvious on first use.

### `_bisect_spectrum` returns the first peak for any query below the range

```python
ix_l, ix_u = bisect_left(mzs, mz_value - tol), bisect_right(mzs, mz_value + tol) - 1
if ix_l == len(mzs):
    return len(mzs), len(mzs)
if ix_u < 1:
    return 0, 0          # <-- "nothing matched" and "peak 0 matched" are the same answer
```

Query below the mass range and `bisect_right` returns 0, so `ix_u` is -1, so the
function reports the half-open range `(0, 0)` — which the caller reads as *peak
zero matched*. Measured: `getionimage(bellini, mz=0.5, tol=0.05)` returns a
128x128 image with **16384 of 16384 pixels non-zero**, where the truth is a
blank image.

`getionimage` also constructs a `UserWarning` about `z=0` and never raises it —
`UserWarning("...")` on a line by itself is just an allocated object.

### `browse()` hands back bare `map` objects

`_SpectrumMetaDataBrowser._find_referenceable_param_groups` ends with
`ids = map(...); return ids`. Anything that indexes the result, checks its
length, or iterates it a second time sees an exhausted iterator rather than an
error.

### `check_peaks_overlap` is not a centroid/profile tiebreaker

Two independent reasons not to adopt it:

1. It calls **`random.seed(42)` on the global `random` module**, clobbering the
   caller's RNG stream as a side effect of asking a question about a file.
   `calc_statistics` does the same.
2. On real `bellini.imzML` — which *declares* `MS:1000128 profile spectrum` —
   it returns 0.01% overlap, i.e. it says **centroid**. Adopting it would put
   Thyra in disagreement with a shipped file's own declaration on day one.

### `include_spectra_metadata='full'` is infeasible, not merely expensive

It appends a `SpectrumData` object per spectrum. Measured at **+147 MiB per
12,737 spectra**, which extrapolates to roughly **10.6 GiB on the 918,855-spectrum
xenium file** — on top of the parse. The `list` mode is cheaper and does not work
either: it never follows `referenceableParamGroupRef`, so it returns `None` for
every spectrum, with no signal that the lookup was structurally incapable of
succeeding. That is exactly where SCiLS puts `MS:1000127`.

---

## Two more constraints worth not rediscovering

- **`choose_iterparse` validates nothing.** `"LXML"`, `"elementtree"`, `"expat"`
  and `""` all fall through to ElementTree without complaint. Thyra's
  `parse_lib="ElementTree"` pin is therefore protected by exact string equality
  and nothing else — a typo would be silent, and *here* it would be harmless
  because the fallback happens to be the pinned choice. Do not rely on that
  staying true. The reason for the pin is in the comment block at
  `ImzMLReader._initialize_parser`.
- **`SIZE_DICT` and numpy disagree about `'l'`.** `PRECISION_DICT['64-bit integer']`
  is `'l'` and `SIZE_DICT['l']` is 8, but `np.dtype('l').itemsize` is 4 on Windows
  and 8 on Linux, so the same file decodes differently on the two platforms.
  If this ever needs fixing, **patch the module globals** — set
  `PRECISION_DICT['64-bit integer'] = 'q'` *and* `SIZE_DICT['q'] = 8`. Patching
  the per-instance `parser.sizeDict` is not enough:
  `PortableSpectrumReader` reads the module-level dict, so the same parser would
  report one size from `getspectrum` and another from its own
  `portable_spectrum_reader`.

---

## The coordinate base is not a constant

The specification numbers x and y from 1, and pyimzml passes through whatever
the document holds. Exports numbered from **0** exist, so subtracting a
constant 1 -- which Thyra did until v3.24.0 -- produced `x = -1` for the first
column of one. A negative index is a legal *negative* numpy index, so the
converter's grid guard dropped those spectra rather than crashing: a 3x3
acquisition at coordinates 0..2 previewed as `grid (2, 2)`, warned that "5
spectra sat outside the declared 2x2x1 grid", stored 4 rows and exited 0.

The base is now measured, by `thyra.utils.imzml_coordinate_base`, and the rule
is deliberately **not** the one z uses:

| axis | rule | why |
|---|---|---|
| x, y | `min(observed_minimum, 1)` | 0 folds down; 1 and anything above it keep the spec base |
| z | `observed_minimum` | z has no physical origin to preserve |

Rebasing x and y on their observed minimum would move a legitimately **cropped**
acquisition. A file whose leftmost occupied column is `x = 5` is a region of a
larger slide, not a 0-based export, and nothing in the document distinguishes
the two; sliding it to the origin would change its grid width, every
`obs["spatial_x"]`, the TIC image extent and its pixel footprint. Folding only
a 0 down leaves every such file exactly where it was.

The declared `IMS:1000042` / `IMS:1000043` pixel counts were the third
candidate and are still unread here. `mzpeak_extractor` is the only place in
Thyra that reads them at all, and it carries a comment about a declared extent
disagreeing with the coordinates shipped beside it.

Whatever is subtracted is reported as
`EssentialMetadata.coordinate_offsets` and written to
`coordinate_systems.global.coordinate_offsets_px`, so a store can say where its
origin came from. Before this the imzML extractor set no offsets at all.

Pinned by `tests/unit/readers/test_imzml_zero_based.py`, which converts a
0-based file, a 1-based one and a cropped 1-based one -- the last being the
file the rejected alternative would have moved.

## Fixed in Thyra, still true of pyimzml

One finding was real enough to grow code: **`imzmldict` discards
`unitAccession`.** `convert_cv_param(accession, value)` takes no unit argument,
so `__readimzmlmeta` cannot carry one even though `ParamGroup.cv_params`
preserves it. `imzmldict['pixel size x']` is a bare number in whatever unit the
vendor declared, and Thyra used to label it micrometres. A file declaring its
pixel as `4406.25` nanometre (`UO:0000018`) therefore landed on disk with
`obs/spatial_x` a thousand times too large, and `convert_msi` returned `True`.

The information is never lost from the document, only from the dict, so
`ImzMLMetadataExtractor` now reads the unit-bearing path — each `cv_params`
tuple is `(name, accession, value, raw_name, raw_value, unit_name,
unit_accession)` — and converts: `UO:0000016` mm x1000, `UO:0000017` um x1,
`UO:0000018` nm /1000. Any other declared unit is refused with a `ValueError`,
which fails the conversion; a cvParam with no unit at all keeps the historical
micrometre reading, because real vendor files (the IONTOF class among them)
write `IMS:1000046` unitless and were being read correctly.

Corroboration that the ambiguity was genuine rather than theoretical: pyimzml's
own `get_physical_coordinates` docstring says it returns **nanometers** while
multiplying by the same unlabelled `pixel size x` that Thyra treats as
micrometres. The library's author disagreed with Thyra about the unit.

The behaviour is pinned by
`tests/unit/readers/test_hand_authored_fixtures.py::TestUnitNanometre`, whose
assertions now state the correct conversion — including the refusal — rather
than characterise the error.

### A typed cvParam with no `value` aborts the metadata parse

`pyimzml.ontology.ontology.convert_xml_value` converts each cvParam value to
the type its ontology entry declares. It catches `ValueError` (an empty
`value=""` on a float-typed term becomes `None`) but not `TypeError`, so a
term with **no `value` attribute at all** raises `float(None)` out of
`ImzMLParser.__init__`, before any spectrum is read.

That is what the pyimzML fork behind TIMSCONVERT and TIMSImaging writes for
the ion mobility array it adds: a `mobilityArray` param group whose
`MS:1003006 mean inverse reduced ion mobility array` term carries a unit and
no value, and pyimzml types `MS:1003006` as `xsd:float`. Upstream pyimzml
therefore cannot open any of those exports:

```
TypeError: float() argument must be a string or a real number, not 'NoneType'
```

Thyra installs a patch as narrow as the defect before building a parser
(`thyra/readers/imzml/_pyimzml_compat.py`): a missing value on a typed term
converts to `None`, the same result an empty value already produces. Nothing
else changes. `tests/data/fixtures/mobility_continuous.imzML` reproduces the
layout, and the reader then goes on to read the third array itself, since
`__iter_read_spectrum_meta` matches only the m/z and intensity group ids and
ignores every other `binaryDataArray` -- see
[Output Format: Ion mobility](output-format.md#ion-mobility).

---

## The fixtures behind this page

Several of the shapes described here are only reachable from
[`tests/data/fixtures/`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/tree/main/tests/data/fixtures),
a small corpus of hand-authored imzML/`.ibd` pairs. They exist because every
other imzML in the test suite was written by pyimzml's own `ImzMLWriter`, and a
parser and a writer from one codebase agree on each other's mistakes.

Read that directory's README before editing any of them; two of the ways they
break are invisible to the test suite.
