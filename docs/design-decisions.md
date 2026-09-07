# Design Decisions

This page records the decisions behind Thyra's defaults where a reasonable
person could have chosen otherwise. Each entry gives the decision, the
reason, the strongest objection that was raised against it and why it did
not win, and the known limit. The goal is that someone who disagrees can see
exactly which premise to argue with, and open an issue against that premise
rather than against a number in a table.

Every measurement quoted here was taken on real acquisitions on the date
given, so a future maintainer can repeat it.

**Status vocabulary.** *Implemented* means the code on `main` does this.
*Accepted* means the decision is made and recorded but the change has not
shipped yet. *Deferred* means no change, with the condition that would reopen
it stated.

---

## D1. Which spectrum a reader takes

**Decision.** Thyra reads the instrument's sparse record when the file holds
one, otherwise the vendor's picked spectrum. It applies no peak picking of
its own.

**Status:** Implemented 2026-09-07. The Bruker TDF default changed from
`vendor_centroid` to `scan_sum`; the centroid stays available through
`--tdf-spectrum vendor_centroid`. Every other row of the table below was
already what the code does.

| Source | What the file holds | What Thyra reads | Why |
|---|---|---|---|
| Bruker TDF (TIMS on) | per-scan digitizer-index counts for every mobility scan, plus a vendor centroid on request | the scans, summed per index (`scan_sum`) | the sparse record; equals Bruker's own frame total |
| Bruker TSF (TIMS off) | a vendor line spectrum, plus a raw digitizer trace | the line spectrum | the trace is a continuum with a baseline, not a peak list |
| Bruker solariX | centroided peak lists in `peaks.sqlite`; raw transients | the peak lists | the only sparse record; transients need an FT |
| Bruker rapifleX | profile spectra | the profile, resampled | nothing else exists; the resampler conserves current |
| Waters SELECT SERIES MRT | a zero-suppressed digitiser trace, plus a vendor centroid on demand | the trace, onto a digitiser-matched axis | the picker merges 8 to 9 mDa doublets the trace resolves; see [Supported Formats](supported-formats.md#waters-masslynx) |
| every other Waters instrument | the same trace, plus a vendor centroid on demand | the centroid (profile on request) | measured on a Synapt G2-Si, the picker keeps everything the trace resolves, so the trace would cost 2 to 3 times the store for nothing |
| imzML, mzPeak, PHI | whatever was exported | as exported | the export already made the choice |

The line between TDF and TSF is the whole decision, so it was measured
rather than assumed.

**Measured on TDF** (2026-09-07; 30 frames of a 26k-pixel MALDI-2 slide and
10 frames of a 20 um biofilm set):

| Quantity | Slide | Biofilm |
|---|---|---|
| `scan_sum` total against Bruker's quasi-profile export (`tims_extract_profile_for_frame`) | identical | identical |
| `scan_sum` total against Bruker's per-frame TIC column (`Frames.SummedIntensities`) | 1.0000 | 0.9992 |
| vendor centroid total against the same TIC column | 0.8748 | 0.9644 |
| centroid intensity of a strong peak against the summed indices under it | 0.97 to 0.99 | 0.98 to 0.99 |
| points per frame, `scan_sum` over centroid | 2.9x | 1.5x |
| raw pair intensities equal to 1 | none | none |

Three things are therefore true at once. The vendor centroid reports peak
*areas* and conserves the current inside each peak it picks. What it drops
is every index bin that its picker assigned to no peak: 12.5 percent of the
slide's current and 3.6 percent of the biofilm's. And Bruker's own
definition of a frame's total ion current is the scan sum, not the centroid.
The earlier description of the loss as "single-count noise" was wrong: these
files contain no single counts, and the dropped current is sub-threshold
signal.

**Measured on TSF** (2026-09-07; two files from different acquisitions): the
line spectrum's intensity equals the maximum of the digitizer trace under
the peak exactly (ratio 1.000 on every peak checked), so TSF intensities are
peak *heights*, and `Frames.SummedIntensities` is the sum of those heights.
The trace has a baseline of 18 on every sample, peaks 15 to 150 samples
wide, and a total 3.4 to 3.7 times the line sum. Summing it would import a
baseline and change the meaning of intensity from height to area. That is a
different kind of data, so TSF stays on the vendor's line spectrum. The
Waters trace is not the same kind: it is zero-suppressed, so it carries no
baseline, and it is the digitiser's record. Which side of the rule a Waters
instrument falls on is therefore decided by whether its peak picker has been
shown to lose what the trace keeps, which is the MRT measurement in the
next row of the table.

**Why `scan_sum` is the TDF default.** Thyra's one invariant everywhere
else is that ion current is conserved: the resampler is TIC-preserving and
every sibling table is checked against an exact identity. The vendor
centroid was the single place where a closed, unversioned algorithm removed
current before the store was written. Under `scan_sum` every identity holds
by construction: the mobility heatmap's marginal is the mean spectrum, the
mobility grid's marginal is the summed table, and the MS/MS blocks add back
up to the summed table. And the number a reader calls TIC is the number the
vendor calls TIC.

**Objections considered.**

- *The dropped current is noise, and keeping it raises the noise floor.*
  Some of it is. Removing it is analysis, reproducible downstream by any
  method the user chooses and documented in their own pipeline. Removing it
  at conversion is irreversible and undocumented.
- *Users comparing with SCiLS Lab or TSF exports will see different
  numbers.* True. The mode is recorded in
  `msi_metadata.processing[0].parameters.tdf_spectrum`, and
  `--tdf-spectrum vendor_centroid` restores the vendor numbers exactly.
- *It changes the numbers of every existing TDF pipeline.* True, and the
  real cost. It ships as a minor release with a changelog entry, and an old
  store can be told from a new one by the provenance field.
- *Then TSF should sum its profile for consistency.* No. The TSF profile
  is a continuum with a baseline, and its line intensities are heights by
  Bruker's own definition. The rule is about sparse records, not about
  reading the largest array available.
- *Then Waters should stay on its centroid, as this page first said.* The
  first version of this table put every Waters instrument on the centroid
  and called its profile "the same kind of data" as the TSF trace. That was
  wrong on both counts, and the correction came from the measurements
  behind the MRT profile default rather than from this page: the Waters
  trace has no baseline, and on the MRT the picker merges near-isobars the
  trace resolves. The rule did not change; the row did.

**Known limits.** The summed table holds about three times the points per
pixel on a short-ramp slide, half again on a long-ramp one. Measured on
the whole 26,087-pixel slide (2026-09-07, default mass axis, optical image
left out, same machine):

| | `vendor_centroid` | `scan_sum` | ratio |
|---|---|---|---|
| non-zeros in the summed table | 282,223,487 | 893,289,573 | 3.2x |
| store on disk | 1.36 GB | 2.50 GB | 1.8x |
| conversion, warm | 438 s | 528 s | 1.2x |

The store grows less than the point count because the sharded chunks
compress the low counts well. That is the price of the default, and the
flag buys the old size back.

---

## D2. The MS/MS table is written by default when the schedule qualifies

**Status:** Implemented 2026-09-07. `--msms-table` is the default; the
table is written for a Bruker PASEF acquisition with a constant,
non-overlapping schedule of at least two precursors and refused, with the
reason logged, on everything else. `--no-msms-table` opts out.

**Reason.** A scheduled PASEF acquisition fragments each precursor in turn
at every pixel. The one-spectrum-per-pixel summed table of such a pixel is
therefore a mixture of unrelated fragment spectra. It is not a spectrum of
anything, and writing only that mixture is the inaccurate representation.
The split is parameter-free, since the schedule is the instrument's own
list, and lossless, since every recorded point falls in exactly one
isolation window; under D1 its blocks add back to the summed table exactly.
The refusals for a schedule that varies across pixels, overlapping windows,
or a single precursor already fall back silently to the summed table, so the
default never breaks a conversion. The prior-art survey in
[Output Format](output-format.md#fragmentation-msms) found no open
analysis-layer convention to defer to.

**Objections considered.**

- *It has run on one dataset: 713 pixels, 15 precursors, one schedule
  shape.* True, and stated as a limit. The assembly engine underneath is
  the one verified on the 26k-pixel mobility grid, and the MS/MS count span
  is tiny, so scale is not where it would fail. A different schedule shape
  is, and no such file exists to test on. That is a reason to document the
  limit, not to ship the mixture by default.
- *A default-on writes an element the user did not ask for and consumers
  must recognise.* The sibling is discriminated by its `uns` block exactly
  as the mobility sibling already is, and the summed table remains the
  primary element.

**Known limits.** Tested to 713 pixels, 15 precursors, one schedule shape.

---

## D3. The mobility grid stays opt in

**Status:** Implemented. `--mobility-grid` is off by default.

**Reason.** This is the opposite case to D2. Summing a TIMS pixel's
mobility scans gives an ordinary MS1 spectrum; nothing about it is
inaccurate. The grid is the same data unfolded along a second axis with
choices the summed table does not need: how many mobility channels, and a
channel width that then depends on each acquisition's ramp. Anything with
free parameters is a derived product, and derived products are written when
asked, with their parameters recorded, so that nobody later finds a 256 in a
store and wonders who chose it. The lossless summary of the mobility
dimension, the mass-mobility heatmap plus the mobility axis metadata, is
written by default, and under D1 the heatmap's marginal equals the stored
mean spectrum exactly.

The sentence that decides D2 and D3 together: the default store contains
what is accurate without any parameter beyond the mass axis, plus a
lossless summary of every extra dimension. Grids over an extra dimension
are opt in and carry their parameters.

**Objections considered.**

- *The mass axis is also a parameterised, derived choice, and it is
  default.* A mass axis is necessary to build any table at all; the grid is
  optional on top of a complete table. This is the thinnest argument on the
  page and is recorded as such.
- *Users will not discover it.* The conversion log says the source has
  mobility, and this page exists.
- *Practical costs are secondary but real.* On 400 frames the store is six
  times larger and the conversion about five times slower warm; at the
  default mass axis the whole 26k-pixel slide is refused (see D4). A
  default that refuses the flagship dataset is not a default anyone can
  rely on.

**Known limits.** None beyond D4.

---

## D4. The grid's feature ceiling is a memory guard, not a format limit

**Status:** Implemented 2026-09-07. The fixed ceiling of 20,000,000 occupied
(m/z bin, mobility channel) pairs became a projection of the `var` frame's
memory (330 bytes per feature, measured) against the machine's free memory,
warning past a quarter of it and refusing past half, with an absolute cap of
100,000,000 kept as a statement of what downstream tools can be expected to
open rather than as the operative guard.

**Reason.** Both sibling tables are built out of core, so the remaining
peak is the `var` frame, measured at roughly 330 bytes per feature including
the AnnData copies. The ceiling exists to protect that build. Raising the
constant from 20M to 25M because the flagship slide occupies 21.1M pairs at
the default axis would be fitting a constant to one dataset, and the
objection to that is immediate. The honest guard is the one the grid used
once before and that was accepted then: project bytes from the count and
compare with what the machine has. Fractions rather than sizes, because a
table that is routine on a workstation is fatal on a laptop.

**Objections considered.**

- *The per-feature constant is version and machine dependent.* It is
  measured and documented, and a guess wrong by a factor of two still
  refuses at half of free memory.

**Known limits.** The whole slide at the default axis lands 5.5 percent over
the present fixed ceiling and is refused after one fused read; with
`--resample-bins 40000` it converts.

---

## D5. One raw read per frame per pass serves every table

**Status:** Implemented 2026-09-07, the same day it was deferred. It was
deferred on a 400-frame warm measurement that put the grid's extra read at
about 4 percent of wall time; the whole-slide logs from the D1 measurement
then showed the mobility heatmap's own pass at 201 s of a 528 s conversion
under `scan_sum` and 272 s of 438 s under the vendor centroid. The extra
reads were 38 to 62 percent of a default conversion, not 4, and the reopen
condition was met on the same afternoon.

**Decision.** A Bruker TDF reader hands each frame over once, as a record
of its raw scan read, from which the summed spectrum, the mobility point
cloud and the fragment spectrum of each precursor are all derived. On the
streaming route the two passes the summed table already takes, count then
scatter, feed the heatmap, the mobility grid and the MS/MS table from that
same read. Two raw reads of the source per conversion, whatever is written;
a default conversion used to take three, a grid four, a grid with the MS/MS
split six. The engine's two passes are still inherent, so "one pass" was
never the right name; "one read per pass" is.

**What makes it safe.** Every derivation goes through the very helpers the
reader's three iterators use, and every sink is the same accumulator the
standalone passes feed, so the fused route cannot see different numbers.
That was checked rather than assumed: stores written by the committed code
and by the fused code were compared table by table on five configurations
(the 400-frame slide with and without the grid on both write routes, and
the 713-pixel PASEF set with its split on both routes) and were identical
in every array, every `var` and `obs` frame and every `uns` block, index
dtypes included. A stub source that answers both the iterators and the
records pins the same identity in the unit tests, and pins that the fused
route reads it exactly twice and never through the iterators.

**Measured** (2026-09-07, warm, optical image left out):

| conversion | before | after |
|---|---|---|
| 400 frames, default (heatmap) | 13 s | 11 s |
| 400 frames, `--mobility-grid` | 32 s | 21 s |
| 713-pixel PASEF, MS/MS split | 12 s | 9 s |
| whole 26,087-pixel slide, default | 528 s | 454 s |
| whole slide, `--mobility-grid --resample-bins 40000` | 1,400 s | 858 s |

The grid row's "before" was measured on an earlier commit, before the
engine's column sort was made six times faster, so part of that gain is
the sort's; the 400-frame row above is the clean comparison for the grid.
On the whole slide the heatmap's own pass (201 s) became part of the count
pass, which grew from about 120 s to 266 s: what was saved is the read and
the index-to-m/z conversion of every frame, about 55 s, plus the second
scatter-side read. What remains is the mapping of every raw point onto the
mass axis, 51,000 points per frame on this slide, which the heatmap and
the grid need and the summed spectrum does not. That mapping is the next
lever, not another read: the points of a frame share about 30,000 unique
digitizer indices, and mapping those once and gathering would give the
same bins for two fifths of the work. Not done here.

**Objections considered.**

- *The in-memory route still runs the standalone passes.* True, and left
  so: that route holds the whole matrix in RAM and is taken by small
  files, where the passes cost seconds. The streaming route is where a
  file large enough for the passes to matter goes.
- *Under `vendor_centroid` the summed spectrum cannot be derived from the
  raw scans.* Correct; in that mode the record asks the library for the
  centroid as a second call per frame, and the raw read still serves the
  sinks. That mode is the opt-in.

**Known limits.** Only the Bruker TDF reader hands frames over as records;
every other source keeps its iterators and its standalone passes, which for
an imzML mobility export is one pass over an already sparse file.

---

## D6. The MS/MS fragment axis is the MS1 mass axis

**Status:** Implemented. A condition attached to it on 2026-09-07, to refuse
the table on an unresampled axis, was withdrawn the same day; see below.

**Reason.** Fragments and precursors pass through the same TOF and have the
same resolving power, so one mass axis per store is the accurate
representation. A separate fragment axis would invent a second resolution
for the same analyser. The coupling is also what makes `mz_index`
meaningful and the conservation check exact.

**Objections considered.**

- *On a raw, unresampled axis fragments snap to MS1-only m/z values, and
  any fragment outside the axis range is dropped.* This objection was
  raised, accepted, implemented as a refusal, and then found to be wrong
  by the existing PASEF test, which relies on the raw axis for an exact
  equality. The premise fails because on a scheduled MS/MS acquisition the
  summed table holds no intact ions: its spectra are the same fragment
  peaks the split re-reads, so the raw axis is the union of the fragment
  m/z values themselves and the mapping is exact. The refusal was removed
  and a test now pins the raw axis as exact on both write routes.

**Known limits.** The resampling grid chosen for the summed table sets
fragment resolution. That is a consequence of the decision, not an
accident.

---

## D7. Waters: which functions hold the image

**Status:** Implemented 2026-09-07, then **restated the same day against
real files, which contradicted its premise.** The first version is kept
below because the correction is the point.

### What the first version decided, and why it was wrong

The Waters reader classifies acquisition functions and then yields every
scan of every MS-classified function as a pixel spectrum, keyed by laser
coordinate. A two-function acquisition -- MSe low and high energy, or a
data-dependent run -- would therefore emit two spectra at the same
coordinate, one MS1 and one MS2, and the per-scan MS level MassLynx reports
was parsed and then ignored. Summing an intact-ion and a fragment spectrum
into one pixel makes a spectrum of nothing, so the fix was to convert the
functions MassLynx labelled MS level 1 and record the rest under
`excluded_functions`. No Waters MS/MS imaging file was available; the
objection *no test data* was answered with "the filter is on a field already
parsed and is testable with a synthetic scan record".

That answer was wrong, and the entry said so itself without noticing: a
synthetic record can only confirm that the code reads the field it was
written to read. It cannot say what the field **means**.

### What the real files say

7,486 Waters `.raw` directories were found on the lab share (the six Waters
instrument folders under `V:\Instruments`, plus `V:\Users\Cuypers_Eva`);
1,396 hold more than one `_FUNC*.DAT`, and the multi-function ones were
opened. **Every multi-function MALDI imaging run is a single-function raster
that MassLynx split across functions**, because it caps a `_FUNC*.DAT` file
at about 1.6 GB and opens a new *function* when a long run reaches it. In
each of those files:

- `_extern.inf` declares exactly **one** acquisition function -- "MALDI TOF
  MS FUNCTION", or "MALDI MOBILITY TOF MS FUNCTION" on the G2-Si -- however
  many functions the file holds. `_FUNCTNS.INF` carries one 416-byte record
  per stored chunk (10,400 bytes for the 25-function file).
- The chunks **tile** the stage and the run: consecutive, non-overlapping y
  bands and retention-time ranges, and **zero** shared pixels between any
  two functions. Their positioned scans sum exactly to the file's distinct
  laser positions (7,682 on `180814_EVO_Fresh_image.raw`, which is also
  exactly its 167 x 46 grid).
- MassLynx reports MS level 1 for the first chunk, 2 for the middle ones and
  **0** for the last, and `getLockmassFunction` names that last chunk as the
  file's lockmass function (it returns -1 only when the file has one
  function). `isMsFunction` is 1 for every chunk, including the one it calls
  lockmass, and **no chunk carries a precursor m/z**.

So on these files the level filter dropped every chunk after the first, and
the lockmass classification -- which predates this decision -- dropped the
last one on top of that. Pixels converted, against the pixels the file
holds:

| File | Instrument | Functions | v3.19.0 | Now (centroid) | In the file |
|---|---|---|---|---|---|
| `20140509_ZF_No13.raw` | Synapt G1 | 25 | 1,657 | 36,633 | 37,033 |
| `20140513_ZF_No16.raw` | Synapt G1 | 11 | 2,496 | 20,438 | 21,788 |
| `20141003 MTB_04.raw` | Synapt G1 | 8 | 1,700 | 12,110 | 12,972 |
| `140903_trypsin_vs_no.raw` | Synapt G1 | 5 | 2,284 | 8,843 | 9,221 |
| `180814_EVO_Fresh_image.raw` | Synapt G1 | 3 | 3,200 | 6,408 | 7,682 |
| `20201209_BCtumor_left Analyte 2.raw` | Synapt G2-Si | 4 | 15,790 | 49,096 | 61,108 |
| `20201223_BCTumor_Eva MDA_468.raw` | Synapt G2-Si | 3 | 5,837 | 11,413 | 17,550 |
| `20201218_MDA468_slide20201208 Analyte 5.raw` | Synapt G2-Si | 2 | 7,355 | 7,355 | 8,547 |

The two-function G2-Si run is the mildest case and still lost 14 percent of
the image; the 25-function one kept 4.5 percent of it. Nothing in the store
said so: the reader logged a warning about "MS level 2" functions and the
conversion looked healthy. The middle column is what the rule below converts
by default; the gap that remains is one chunk per file, and the next section
is why.

**One real multi-function acquisition was found**, and it is what makes the
rule below decidable rather than a guess:
`Xevo DESI\Pierre\DESI_PIMAX.PRO\Data\20191107_fastDDA_neg_002.raw`. Its
`_extern.inf` declares **16** functions -- one "TOF FAST DDA FUNCTION" and
15 "TOF SURVEY FUNCTION"s -- against the one function the chunked files
declare. Function 0 is MS1 with no precursor; functions 1 to 15 are level 2
and report a *different* precursor per scan, 28 distinct values each. And
every one of the 16 lands on the **same** position. So the two cases
separate cleanly on the positions: a chunked raster tiles them, a real
parallel acquisition repeats them.

(That file grids to 1x1, because the reader builds the grid from laser
coordinates and a DESI stage records none. That is a separate gap, tracked
outside this entry; it is not what D7 is about.)

### Decision

Decide from the **laser positions**, the same measurement the pixel grid is
already built from.

- A function landing on pixels no earlier function covers **extends the
  raster** and is converted, whatever level MassLynx reports for it and
  whether or not MassLynx calls it the lockmass function. Only ever widens a
  file that already has an MS-classified function, so a run with no MS
  function is still refused.
- Functions **competing for the same pixels** were acquired in parallel.
  Only one of them can be the pixel's spectrum, so among those the MS1 ones
  win when the group has any, and the rest are listed with their level,
  precursor m/z, scan count and the reason they stayed out under
  `excluded_functions`. That is the original decision, kept -- scoped to the
  functions it was actually about.

`format_specific.function_types` keeps MassLynx's own classification next to
`format_specific.ms_functions`, so a rescued chunk is visible in the store.

### The catch: the tail chunk cannot be centroided

The library will not centroid the function `getLockmassFunction` names.
Measured on `180814_EVO_Fresh_image.raw`, sampling scans from each chunk:

| Function | `setCentroid(1)` | `setCentroid(0)` | `isRawSpectrumContinuum` |
|---|---|---|---|
| 0 (MS) | 7,408 points, TIC 3.35e4 | 82,876 points | continuum |
| 1 (MS, "level 2") | 8,508 points, TIC 3.60e4 | 91,274 points | continuum |
| 2 (named lockmass) | **112,594 points, TIC 1.11e5** | 112,594 points | continuum |

All three are acquired as continuum, and the request is honoured for the
first two and ignored for the third: it returns the same profile trace
either way. `ScanInfo.isProfile` reports this faithfully (0, 0, 1), since it
is read after `setCentroid`. There is no per-function centroid entry point
in the library to work around it.

Converting that chunk into a store of centroids therefore lays a band of
profile rows across the top of the image. It was converted that way once, by
accident, and the TIC image shows it: rows 0 to 37 average 3.1e4 to 3.7e4 and
rows 39 to 45 -- exactly the rescued chunk -- average 8.6e4 to 9.3e4, a
sharp 2.3x step at the chunk boundary and not a feature of the sample.

So the rule takes the representation into account: while the run is read as
centroids, a chunk the library will not centroid **stays out**, and
`excluded_functions` records it with its scan count, the pixels it would
have added and the reason. Reading the run as the profile trace
(`--waters-spectrum profile`, and the default on an MRT) selects every
chunk, because then all of them come back the same way. The warning names
the cost and the flags:

    Function(s) 2 hold 1274 pixels (16.6% of the image) that no other
    function covers, but MassLynx names them the lockmass function and will
    not centroid them. They stay out rather than put profile rows in a table
    of centroids: pass --waters-spectrum profile --streaming true to convert
    the whole image.

`--streaming true` is in that sentence because the profile store is large
and `--streaming auto` does not notice: its estimate assumes 10,000 peaks
per spectrum whatever the source, which is 0.57 GB for this run, while the
streaming converter's own estimate once running is **74.1 GB** (7,682
pixels x 2,590,447 bins). Left on `auto` the conversion stays in memory and
dies at 72 percent asking for a 24.5 GiB array on a 128 GB machine. That
estimate is a general defect, tracked separately.

The alternative -- forcing the whole run to the profile trace whenever a
chunk cannot be centroided -- would keep the image whole automatically, but
it silently overrides D1's measured choice for a whole vendor and turns a
241 MB store into a 74 GB one, on every chunked file, of which this share
holds 1,396. Better to convert what is consistent, say what is missing, and
leave the trade to the flags that already exist.

The `fragmentation` block follows the **precursor**, not the level: a
converted function holds fragment spectra when MassLynx reports a precursor
m/z for it. A reported level with no precursor behind it is the chunk
artefact above, and reads as MS1.

**Verified on real data** (2026-09-07):

| Check | Result |
|---|---|
| `180814_EVO_Fresh_image.raw` converted, default centroid | 6,408 pixels against v3.19.0's 3,200, no duplicate coordinates, no empty pixel, `ms_functions [0, 1]`, `excluded_functions["2"]` carrying `n_unique_pixels 1274` and its reason, `fragmentation` MS1. The TIC image is continuous across the chunk boundary at row 19 |
| the same run with `--waters-spectrum profile` | all three chunks selected: 7,683 spectra on all 7,682 pixels, one representation. Needs `--streaming true`; see above |
| `20170818_08.raw`, a real single-precursor MS/MS run | `fragmentation` MS level 2, one window at m/z 377.4, collision energy 35.0, CID; the converter declines a demultiplexed table because one precursor needs none |
| `20191107_fastDDA_neg_002.raw`, the real DDA run | functions 1 to 15 recorded under `excluded_functions` with their per-scan precursors, function 0 converted, `fragmentation` MS1 |
| every file in the table above | no two converted functions share a pixel, and converted plus recorded pixels equal the file's distinct laser positions |

### Objections considered

- *The level is what MassLynx says; the chunking is its bug to report, not
  ours to work around.* It is not a bug that can be worked around later:
  there is no other field that separates a chunk from a high-energy
  function, and the level is wrong in both directions at once (0 for a
  chunk that holds image data, 2 for one that holds MS1 data). The
  positions are a direct measurement of the thing the rule is about --
  whether two spectra land on the same pixel -- so they are the better
  signal even if MassLynx were fixed tomorrow.
- *A lockmass function could legitimately carry laser positions, and would
  now be converted.* A reference function is acquired **alongside** the
  image, so its scans land on pixels the MS function already covers and it
  is excluded by the same rule. The one real parallel lockmass function
  found (`050517_BILE ACIDS 01.raw`) sits at a single constant position for
  the whole run, which no raster chunk does.
- *Two functions covering complementary mass ranges at one pixel should be
  summed, not filtered.* They still are: functions competing for a pixel and
  reporting the same level are all kept and summed, exactly as before. Only
  a mixed-level group is filtered.

**Known limit.** A raster chunk is recognised by covering new pixels, so a
chunked acquisition whose stage revisits a position -- a re-scan of the same
area in a later function -- would have that function read as a parallel one
and excluded. No such file was found, and the acquisition would be
ambiguous anyway: the store holds one spectrum per pixel.

### No Waters demultiplexed table yet

The second half of the original entry -- "a demultiplexed Waters table waits
until a scheduled Waters acquisition exists to look at" -- still waits. Of
the 506 methods whose `_extern.inf` was read, **113 declare a "MALDI TOF
MSMS FUNCTION"**, and every one of them has exactly **one** function isolating
**one** precursor, 224 KB to 3.9 MB, 9 to 60 scans on five or fewer distinct
positions: spot acquisitions, not images. A Waters demultiplexer needs a
file whose MS/MS functions each isolate a *different* constant precursor
over one raster, and no such file exists here, so it was not built.

What those files do establish, which the synthetic tests could not: MassLynx
reports the precursor faithfully when there is one (377.4, 482.0, 476.16,
each matching `Set Mass` in `_extern.inf`) together with a real collision
energy (35.0, and a 6 -> 30 eV ramp on one), so `precursor_mz` is the field
to trust. `quadIsolationStart`/`End` stay 0.0 even on these, so the
isolation window has no offsets from this API and only a target.

---

## D8. The `var` frame is the memory peak, and stays so

**Status:** Accepted limit.

With both sibling tables out of core, memory is proportional to the number
of features, not to the number of non-zeros. That is the intended shape;
D4 is the guard on it.
