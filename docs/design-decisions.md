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

**Measured on the whole slide** (2026-09-07, 26,087 pixels at the default
138,629-bin axis, `--mobility-grid --no-optical`, warm, no optical image).
The conversion the fixed ceiling refused now runs: 21,101,151 occupied
`(m/z bin, mobility channel)` pairs, 1,627,659,585 non-zeros through 18.19 GB
of scratch, 239 of 256 channels occupied, a marginal of exactly 1.0 in every
pixel, an 8.0 GB store (2.5 GB summed, 5.6 GB grid). The guard neither warned
nor refused, which is what a projection of 6.5 GB against 90 GB free should
do.

The constant held: peak private bytes were 6.38 GB, which over 21,101,151
features is **302 bytes per feature**, 8 percent under the 330 the guard
projects with and measured the way that number was, as the whole process's
private peak over the feature count. The `var` frame's own step is narrower
-- private sat at 2.49 GB through both passes and both column sorts and rose
to 6.38 GB over three seconds while the frame was built, so 184 bytes per
feature is what the frame alone costs -- and the guard is conservative under
either reading, on a grid 1.6 times the one the constant was measured on. It
stays at 330.

**Known limits.** The `var` frame is still the peak (D8), and the grid at the
default axis is 5.6 GB of store for a slide whose summed table is 2.5 GB.

---

## D5. One raw read per frame per pass serves every table

**Status:** Implemented 2026-09-07, the same day it was deferred. It was
deferred on a 400-frame warm measurement that put the grid's extra read at
about 4 percent of wall time; the whole-slide logs from the D1 measurement
then showed the mobility heatmap's own pass at 201 s of a 528 s conversion
under `scan_sum` and 272 s of 438 s under the vendor centroid. The extra
reads were 38 to 62 percent of a default conversion, not 4, and the reopen
condition was met on the same afternoon. The mapping this left as the next
lever was taken the same week, by the frame's own unique indices; see below.

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
the grid need and the summed spectrum does not. That mapping was the next
lever, not another read, and it was taken the same week.

**The mapping, by the frame's unique indices** (implemented 2026-09-07). A
TDF frame is read as digitizer indices, and the reader already converts the
unique ones: measured on this slide, 28,560 unique of 49,070 points, 58
percent. Both halves of the mapping -- nearest bin, and in range or not --
are elementwise in the m/z, so mapping the unique values and gathering by
the frame's own inverse gives the same bins, the same mask and the same drop
count for 58 percent of the binary searches. The record offers that view as
`mobility_points_indexed`, the fused passes prefer it, and a record that
does not offer it is mapped point by point as before; the iterators and the
standalone passes are untouched.

**Measured** (2026-09-07, the same slide, warm, local disk, no optical
image; each pair one session, v3.19.0 against the branch):

| whole 26,087-pixel slide | v3.19.0 | indexed mapping |
|---|---|---|
| default, wall | 420 s | 343 s |
| default, pass 1 (maps every point, for the heatmap) | 250 s | 171 s |
| default, pass 2 (no mobility sink) | 137 s | 139 s |
| `--mobility-grid` at the default axis, wall | 1,201 s | 724 s |
| grid, pass 1 (heatmap and discovery) | 401 s | 237 s |
| grid, pass 2 (grid scatter) | 627 s | 357 s |

The pass that maps points is a third faster, and where both passes map, the
conversion is 40 percent faster; the default's second pass, which has no
mobility sink to map for, is unchanged, which is the control. Taking 42
percent of the searches out of the default's mapping pass took 79 s off it,
which puts the searches at roughly 190 s of that pass's 250 and roughly 110 s
of its 171 now -- still the largest single item in it, and the gather and
the mask that remain are proportional to the points however few searches
they cost.

Identity was checked the way the fused passes were: stores written by
v3.19.0 and by the branch, compared array by array (X data, indices and
indptr with dtypes, `var`, `obs` and `uns`), on the 400-frame slice with and
without the grid and on the 713-pixel PASEF set, and then on the whole slide
in both configurations by hashing every array of every table in chunks.
Identical throughout, the heatmap's `counts` included.

**Cold cache on the lab share** (2026-09-07, the deferral D5 left open).
The same slide copied to the SMB share, default conversion, branch code, the
file pushed out of the machine's standby list before the first run so the
read is genuinely cold: 365 s cold against 342 s warm, pass 1 178 s against
170 s, pass 2 143 s against 138 s. The first read of a never-read file costs
about 5 percent of the pass; a conversion's second pass is warm either way,
since the first pulls the acquisition into the cache. The share is not the
bottleneck the deferral suspected -- on this machine a cold network read is
within seconds of a warm local one (171 s for the same pass), so the mapping
and not the read is what a conversion of this shape spends its time on,
wherever the file lives.

**Objections considered.**

- *The in-memory route still runs the standalone passes.* True, and left
  so: that route holds the whole matrix in RAM and is taken by small
  files, where the passes cost seconds. The streaming route is where a
  file large enough for the passes to matter goes.
- *Under `vendor_centroid` the summed spectrum cannot be derived from the
  raw scans.* Correct; in that mode the record asks the library for the
  centroid as a second call per frame, and the raw read still serves the
  sinks. That mode is the opt-in.
- *The record contract grew a method for one reader's convenience.* The
  indexed view is reached with `getattr` and a record that does not offer
  it is mapped point by point, which the stub records in the tests pin. It
  is a view of the read every record already has, not a new obligation.
- *Mapping every unique value maps ones that are then dropped as out of
  range.* It does, and it costs a few searches more on a frame whose points
  overhang the axis -- 8,328 points of 1.65 billion on this slide. The
  alternative, masking before mapping, would need the mask expanded to the
  points first, which is the work the factoring exists to avoid.

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

## D7. Waters: the summed table takes MS level 1 only

**Status:** Implemented 2026-09-07, as a correctness fix.

**What was found.** The Waters reader classifies acquisition functions and
then yields every scan of every MS-classified function as a pixel spectrum,
keyed by laser coordinate. A two-function acquisition, MSe low and high
energy or a data-dependent run, therefore emits two spectra at the same
coordinate, one MS1 and one MS2. The per-scan MS level and precursor m/z
that MassLynx reports are parsed and then ignored. What the converter does
with the duplicate coordinate is not verified, and no Waters MS/MS imaging
file was available to test on; either outcome, overwrite or sum, is wrong.

**Decision.** Convert the MS level 1 functions only, and list the others,
with their level, precursor m/z and scan count, under `excluded_functions`
in the Waters-specific metadata block. The versioned `fragmentation` block
describes the spectra in the store, so it says MS1 for such a file; a file
with MS/MS functions only converts them and reports their precursors there,
as the Bruker path does. A demultiplexed Waters table waits until a
scheduled Waters acquisition exists to look at.

**Objections considered.**

- *No test data.* The filter is on a field already parsed and is testable
  with a synthetic scan record. The fix does not depend on the MS/MS table
  work.

---

## D8. The `var` frame is the memory peak, and stays so

**Status:** Accepted limit.

With both sibling tables out of core, memory is proportional to the number
of features, not to the number of non-zeros. That is the intended shape;
D4 is the guard on it.
