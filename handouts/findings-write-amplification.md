# Findings: Write Amplification and Resampling Memory

This reports the outcome of the investigation described in
[handout C](write-amplification.md): four leads suggesting that Thyra's
conversion path does substantially more work than it needs to.

Each lead was measured independently, and each measurement was then re-run
adversarially by a second engineer who did not see the first engineer's
scripts. Two of the four verification passes agreed with the original
conclusion; **two disagreed**, and in both cases the disagreement changed the
recommendation. Those disagreements are reported below as disagreements, not
smoothed over.

The handout is treated here as a document to be corrected by evidence.
Several of its claims did not survive measurement, including two of the six
per-key numbers it opens with, and they are corrected explicitly.

---

## Summary

| Lead | Verdict | The number that decides it |
|---|---|---|
| **1. Metadata rewritten dozens of times per conversion** | Real, **not worth fixing** | 413 metadata writes at 4 pixels and 413 at 50,000 pixels -- bit-identical across a 12,500x pixel range, for a fixed ~0.24 s |
| **2. Default bin counts upsample heavily** | **Split.** Not worth changing for `nearest_neighbor`; a real cost for `tic_preserving` -- **owner's decision, proposal below** | `tic_preserving` at the default axis stores **1.436 MB per pixel**, of which **0 bytes** are removable zeros (0 explicit zeros out of 75,998,341 stored values) |
| **3. Processed-mode imzML scans every spectrum twice** | Real (it is a **triple** scan), **modestly worth fixing** -- but the time saving is not the reason | `get_common_mass_axis()` peaks at **3.30x the file's entire m/z payload** in RAM, extrapolating to a hard OOM at roughly a 40 GB input |
| **4. Dense resampling path allocates per pixel** | **Real and worth fixing**, on more counts than the handout claims | `tic_preserving` stores **189,997 non-zeros per pixel** versus 3,984 for `nearest_neighbor` (47.7x), at a measured 44 bytes of peak RSS per stored non-zero |

Two findings fall outside the four leads and are larger than three of them:

- `_tic_preserving_resample` does not preserve TIC. Measured output/input TIC
  ratio at the default axis: **47.5** for a 4,000-point profile spectrum,
  **1275** for a 150-peak centroid spectrum. This contradicts the repository's
  own `ResamplingMethod` docstring, a second implementation in
  `thyra/resampling/strategies/tic_preserving.py` that *does* renormalise, and
  [the Resampling page](../docs/resampling.md#tic_preserving).
  **Fixed in [PR #112](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/112)**,
  which also moves the rule into `thyra/resampling/tic.py` so the converter and
  the strategy share one definition rather than two that can drift.
- With `streaming=True`, a processed-mode conversion issues **4N**
  `getspectrum` calls rather than 3N -- the streaming converter reads every
  spectrum twice. Confirmed two independent ways: call counting, and counting
  bytes read off the `.ibd` handle (4.00x the file size).

---

## How the numbers were produced

All work was done in a separate worktree, driven with the main virtualenv's
interpreter by absolute path and `PYTHONPATH` pinned to the worktree, with an
assertion on `thyra.__file__` at the top of every script. This matters: the
virtualenv carries an editable install pointing at a *different* checkout, so
a naive `poetry run python` silently measures the wrong tree. Both engineers
independently confirmed that for the files in question the two checkouts are
byte-identical, so the trap could not have changed any number reported here --
but the guard was in place regardless.

Datasets are synthetic throughout: `generate_example_imzml` for
continuous/profile data (4,000 m/z points over 250-1200 Da), and a
hand-written `pyimzml.ImzMLWriter(mode="processed")` generator for
processed/centroid data. Largest inputs were a 921 MB imzML/ibd pair
(50,000 pixels) and a 768 MB `.ibd` (128,000 spectra). **Nothing was measured
at 100 GB.** Every 100 GB figure below is an explicitly labelled linear
extrapolation from a measured scaling series.

Per the handout's warning, write counts and `WinError 5` race frequency were
never drawn from the same run. Race frequency was not measured at all.

Nothing in the source tree was modified. Prototypes referenced below exist
only as measurement scripts.

---

## Lead 1: metadata rewritten dozens of times per conversion

**Verdict: real, and not worth fixing. The handout's own escape hatch applies.**

### What the handout claimed

40 distinct Zarr keys written during a 4-spectrum conversion, 22 of them more
than once, roughly 430 metadata renames in total, with the store root
`zarr.json` written 28 times. It asserted this is "the direct cause of the
Windows `WinError 5` race", and asked whether the rewriting scales with
element count rather than pixel count -- noting that if it does, "the finding
is keep the retry, close the issue".

### What was measured

The rewriting is real, and worse than reported. It is also completely flat.

| Pixels | Metadata writes | Redundant | Root `zarr.json` rewrites | Data chunk writes |
|---|---|---|---|---|
| 4 | 413 | 354 | 63 | 35 |
| 100 | 413 | 354 | 63 | 45 |
| 500 | 413 | 354 | 63 | 81 |
| 2,000 | 413 | 354 | 63 | 129 |
| 10,000 | 413 | 354 | 63 | 417 |
| 50,000 | 413 | 354 | 63 | 1,059 |

"Redundant" is `sum(writes_per_key - 1)` over keys written more than once --
the theoretical ceiling on what a perfect write-once store would avoid.

The same flatness holds against m/z axis width, which rules out the other
obvious candidate:

| m/z bins (100 pixels fixed) | Metadata writes | Data chunk writes |
|---|---|---|
| 1,000 | 413 | 39 |
| 10,000 | 413 | 45 |
| 100,000 | 413 | 52 |
| ~190,000 (default) | 413 | 45 |

What it *does* scale with is Zarr group and array count. In an isolated
pure-`spatialdata` experiment with no Thyra involved:

| Elements written | Total atomic writes | Root `zarr.json` rewrites | Fit |
|---|---|---|---|
| k image elements, k = 1/2/4/8/16 | 19 / 33 / 61 / 117 / 229 | 6 / 8 / 12 / 20 / 36 | `14k + 5`, `2k + 4` |
| k table elements plus 1 image, k = 1/2/4/8 | 182 / 343 / 665 / 1309 | 29 / 51 / 95 / 183 | `~161k + 21`, `~22k + 7` |

The store root is re-serialised roughly twice per Zarr group or array created
anywhere in the store. An image costs 2 root rewrites; an AnnData table costs
about 22, because each `obs` column, `var`, `uns` subgroup and `X` is its own
group or array. Thyra emits a fixed three elements for a 2D single-region
dataset, so the whole thing is constant.

Cost of that constant, corrected for concurrency (see below):

| Pixels | Clean wall clock | Union-of-intervals cost of all 413 metadata writes | Redundant share of wall clock |
|---|---|---|---|
| 100 | 0.83 s | 0.247 s | 25.5% |
| 2,000 | 1.60 s | 0.245 s | 13.2% |
| 10,000 | 4.95 s | 0.244 s | 4.2% |
| 50,000 | 22.60 s | 0.266 s | 1.01% |

Extrapolated (linear fit `wall = 0.6 s + 4.45e-4 s/pixel`, fitted on
100-50,000 px, predicting 1.48 s at 2,000 px against 1.58 s measured): a
100 GB dataset at 32 KB per spectrum is roughly 3.3M pixels, about 24 minutes,
against which a fixed 0.22 s is **~0.03%**.

Alternatives that were supposed to help do not:

| Write path | Metadata writes | Root rewrites | Verdict |
|---|---|---|---|
| In-memory converter, `SpatialData.write()` | 413 | 63 | baseline |
| Streaming converter, non-PCS | 434 | 68 | worse: +21 / +5 from its own `zarr.open_group` |
| Streaming PCS path, `write_element` | 297 | 51 | 28% better, but gated behind a 30 GB threshold and still flat |
| Isolated incremental `write_element`, 16 images | 245 atomic | 52 | **7% worse** than one-shot `write()` at 229 / 36 |

### What the verification pass found

The verifier **agreed**, with high confidence, and reproduced the load-bearing
table exactly -- all six pixel counts, bit-identical -- using a *different*
instrumentation hook (`zarr.storage._local._put` rather than `_atomic_write`).
Distinct-key counts, the per-key distribution, and the clean wall clocks all
matched within noise.

The verifier found one real benchmarking error, and it runs in the direction
that made the lead look **worse** than it is:

> The original report's "~0.4 s wall-clock cost of the redundant metadata
> writes" sums per-call durations. But zarr's `_put` runs under
> `asyncio.to_thread`, so those calls genuinely overlap -- 32 distinct threads
> were observed issuing metadata writes, with up to 6 in flight
> simultaneously, an overlap factor of 2.46-2.59x. Taking the union of
> intervals instead of the sum gives ~0.24-0.27 s for all 413 metadata writes
> and ~0.21-0.23 s for the redundant share.

Every percentage in the first report was therefore roughly double the truth.
The table above uses the corrected figures. The union-of-intervals metric is
itself flat across a 500x pixel range (0.247 / 0.245 / 0.244 / 0.266 s), which
independently confirms the "fixed cost" conclusion through a metric free of
the double-counting bias.

The verifier also flagged the claimed "instrumentation overhead of -18 us per
call, i.e. zero" as false precision -- a negative overhead is physically
impossible; the honest statement is "below a ~5% run-to-run noise floor". The
operative conclusion, that instrumented timings are usable directly, stands.

### Corrections to the handout

| Handout claim | Status | Correction |
|---|---|---|
| ~430 metadata renames for a 4-spectrum dataset | Confirmed | 448 total atomic writes: 413 metadata plus 35 data |
| 40 distinct keys, 22 written more than once | **Understated** | 94 distinct keys at 4 px, 104 at 100 px, 55 written more than once |
| Store root `zarr.json` written 28 times | **Wrong** | 63 times. The document is ~50 KB, so ~3.1 MB of root-metadata rewriting per conversion |
| `tables/zarr.json` 21, `tables/ds_z0/zarr.json` 20 | **Wrong** | 54 and 60 |
| `obs` 15, `images/.../tic` 8, `X` 7 | Confirmed | Exact matches. Only the top-of-tree keys diverge |
| Scales with element count, not pixel count | **Confirmed** | This is the central result and it closes the lead |
| `SpatialData.write()` re-serialises the root once per element | Partly | *Twice* per element, and the real driver is per Zarr group or array, not per element |
| Compare against `write_element` used incrementally | **Refuted** | Incremental `write_element` is 7% worse in isolation |
| The streaming converter's own calls add to it | Confirmed | +21 metadata writes, +5 root rewrites |
| This is the direct cause of the `WinError 5` race | **Untested** | Deliberately not measured, per the handout's own warning that instrumenting `_atomic_write` changes the failure rate. Nothing here confirms or refutes it |

### Conclusion

Keep the bounded retry in `thyra/utils/zarr_atomic_write.py` and close the
issue. The handout's own criterion is met: this is fixed overhead of a few
hundred renames, it matters for robustness and not for throughput, and both
candidate alternatives are worse or barely better.

One free cleanup falls out.
`thyra/converters/spatialdata/base_spatialdata_converter.py:1922` calls
`zarr.consolidate_metadata(str(self.output_path))` immediately after
`sdata.write(...)` on line 1921 -- but `SpatialData.write()` already
consolidates, since its `consolidate_metadata` parameter defaults to `True`.
Exactly two consolidate calls per conversion were measured, the duplicate
costing 35-44 ms. Deleting line 1922 is a one-line, behaviour-preserving
saving. Not worth a commit on its own; worth folding into any other change in
that function. The streaming converter's consolidate at
`streaming_converter.py:1560` is on a different path and is not a duplicate.

!!! note "Robustness, if it ever becomes the goal"
    The highest-leverage single target is the store root `zarr.json`: 63
    rewrites of a ~50 KB document per conversion, all inside
    `spatialdata` and `zarr` rather than Thyra, and it is exactly the file the
    `WinError 5` race lands on. That is another argument for keeping the retry
    rather than engineering the rewrites away.

---

## Lead 2: default bin counts upsample heavily

**Verdict: split, and the two engineers disagreed. The recommendation below is
a proposal for the owner, not a decision.**

### What the handout claimed

The default is 5 mDa at m/z 1000 for every axis type except `linear_tof`,
producing 190,000 bins from 4,000 source m/z values on the tutorial dataset --
a "45x upsample". The store stays small because it is sparse, "so this is not
a disk problem", but "it costs every consumer that materialises the table".
The handout told us to treat the lead sceptically, because wide `var` axes are
the design target of the lazy read path.

### What was measured

First, the bin counts themselves, recomputed by calling
`_calculate_bins_from_width` directly rather than trusting the handout. The
handout cites only the `constant` axis; the others are far wider.

| Mass range | `linear_tof` | `constant` | `reflector_tof` | `orbitrap` | `fticr` |
|---|---|---|---|---|---|
| 250-1200 Da | 38,369 | **190,000** | 313,723 | 434,851 | 633,333 |
| 400-1000 Da | -- | 120,000 | 183,258 | 232,455 | 300,000 |
| 100-2000 Da | -- | 380,000 | 599,146 | 982,068 | 1,900,000 |
| 2000-20000 Da | -- | 3,600,000 | 460,517 | 193,399 | 90,000 |

`reflector_tof` is what the detector chain picks for unknown-vendor centroid
and timsTOF centroid data -- the most common real centroid paths -- so the
typical default is closer to 314,000 bins than to 190,000.

#### On the nearest_neighbor path the width is nearly free

10,000 pixels, source axis 4,000 points, only `target_bins` varied:

| Bins | Store | nnz | Eager RSS | Eager open | Lazy 1 Da ion image | Lazy single-pixel spectrum | Conversion |
|---|---|---|---|---|---|---|---|
| 4,000 | 171.93 MB | 39,850,401 | 486.9 MB | 0.34 s | 70.5 ms | 0.273 s | 4.85 s |
| 19,000 | 172.03 MB | 39,850,401 | -- | -- | 19.9 ms | 0.303 s | 4.50 s |
| 47,500 | 172.27 MB | 39,850,401 | -- | -- | 11.0 ms | 0.360 s | 4.94 s |
| **190,000 (default)** | **173.35 MB** | **39,850,401** | **508.1 MB** | **0.34 s** | **8.2 ms** | **0.704 s** | **9.19 s** |
| 633,333 | 176.64 MB | 39,850,401 | 562 MB | 0.71 s | 5.3 ms | 1.560 s | 19.44 s |

Eager figures are `anndata.read_zarr` in a fresh process. The non-zero count
is *identical* at every width -- `nearest_neighbor` maps each source point to
exactly one bin, so width only lengthens `indptr` (0.76 MB versus 0.016 MB).
The default costs **+4.4% eager memory, +0.8% disk, and no extra open time**.

Correctness was gated throughout: the ion-image checksum
(`93552464.50942102`) and single-pixel spectrum checksum (`16368.925...`) were
byte-identical across every bin count and both access modes.

The lazy result inverts the lead's premise, for the reason that
`anndata.experimental.read_lazy` chunks CSC at a fixed 1000 columns:

| Bins | Chunk mass span | nnz decompressed for one 1 Da ion image | Useful nnz | Read amplification |
|---|---|---|---|---|
| 4,000 | 237.5 Da | 9,964,039 | 39,938 | 249.5x |
| 19,000 | 50 Da | 2,103,672 | 39,938 | 52.7x |
| 47,500 | 20 Da | 837,168 | 39,938 | 21.0x |
| 190,000 | 5.0 Da | 209,480 | 39,938 | 5.2x |
| 633,333 | 1.5 Da | 69,899 | 39,938 | 1.8x |

A wider axis gives finer mass locality inside a fixed-size chunk. The
advantage decays and eventually reverses as the window widens:

| m/z window | 4,000 bins | 190,000 bins | Wide axis is |
|---|---|---|---|
| 0.5 Da | 60.7 ms | 7.0 ms | 8.7x faster |
| 1 Da | 62.6 ms | 7.1 ms | 8.8x faster |
| 10 Da | 64.7 ms | 15.1 ms | 4.3x faster |
| 50 Da | 79.9 ms | 44.9 ms | 1.8x faster |
| 200 Da | 183.3 ms | 162.8 ms | 1.13x faster |
| 950 Da, full range | 391.1 ms | 716.5 ms | **1.83x slower** |

#### On the tic_preserving path the width is not free at all

400 pixels, same 4,000-point source:

| Bins | Store | nnz | nnz per pixel | MB per pixel | Peak RSS |
|---|---|---|---|---|---|
| 4,000 | 6.93 MB | 1,593,947 | 3,985 | 0.0173 | ~350 MB |
| 19,000 | -- | -- | 19,000 | 0.1439 | -- |
| 47,500 | -- | -- | 47,499 | 0.3593 | -- |
| **190,000 (default)** | **574.33 MB** | **75,998,341** | **189,996** | **1.4358** | **3,377 MB** |

The matrix is 99.998% dense. Store size on this path equals
`n_pixels x target_bins x ~7.5 bytes` and depends on the source density **not
at all**. It is linear in pixels too (1.436 / 1.431 / 1.429 MB per pixel at
400 / 900 / 1600 pixels), which supports this extrapolation:

| Pixels | Store, extrapolated at 190,000 bins | Conversion time |
|---|---|---|
| 10,000 | 14.3 GB | ~1.9 min |
| 100,000 | **142.9 GB** | ~18.7 min |
| 1,000,000 | 1.43 TB | -- |

This is a live default path, not a hypothetical. The detector chain routes
profile data with more than 5,000 points per spectrum to
`tic_preserving` plus `constant`, and `RapiflexDetector.matches` returns
`True` for any Bruker MALDI-TOF **with no density condition at all**.

### What the verification pass found, and where it disagreed

The verifier reproduced every number: all 24 cells of the bin-count table
exactly, store sizes to six significant figures (171.926 / 173.346 / 6.930 /
574.330 MB), non-zero counts exactly, MB per pixel to five decimals, both
checksums, the +4.35% eager RSS, the 8.6x lazy ion-image speedup with the
1000-column chunk mechanism confirmed directly, and the detector routing
table. An end-to-end run with method, axis and bins all on auto was confirmed
to produce `nearest_neighbor` plus `constant` plus exactly 190,000 bins.

The verifier nevertheless **disagreed with the verdict**, on one disprovable
point. The first report concluded "no change", and handed the
`tic_preserving` cost to lead 4 on the grounds that fixing lead 4 "changes
only which zeros get stored, not the `var` axis".

> There are no stored zeros to remove. Explicit zeros in the `tic_preserving`
> 190,000-bin store: **0 out of 75,998,341**.
> `base_spatialdata_converter.py:1270` already filters `intensities != 0.0`
> before accumulation, and `np.interp` onto a 190,000-point axis yields
> 190,000 non-zero values out of 190,000. The 574 MB is 574 MB of real
> interpolated numbers.

That matters, because it means lead 4 structurally cannot absorb this cost.
Making `tic_preserving` output sparse by thresholding would drop real values
-- a lossy data change, not a storage-representation change. Since store size
on that path is exactly `n_pixels x target_bins`, the bin count is the only
**non-lossy** lever available, and the bin count is lead 2.

The verifier also showed the first report's softening caveat was optimistic:
it argued that real 100,000-point-per-spectrum profile data would only be a
1.9x upsample. The density threshold is 5,000 points per spectrum, a 38x
upsample at the boundary, and a Bruker MALDI-TOF profile file with only 2,000
points per spectrum was confirmed to route to `tic_preserving` -- a 95x
upsample. **The live range is roughly 1.9x to 95x.**

Finally, the verifier noted that the 8.6-9.3x lazy speedup is an artifact of
AnnData's hard-coded 1000-column chunk. Identical mass locality could be had
at 4,000 bins with a ~20-column chunk. It is evidence that shrinking is not
urgent; it is not evidence that shrinking is harmful.

!!! note "One verifier-flagged item that does not hold up"
    The first report also called the "narrow the mass range before increasing
    bin counts" bullet in
    [Practical guidance](../docs/resampling.md#practical-guidance) "exactly backwards
    in cost terms for the tic path". It is not: at a fixed target width,
    narrowing the range reduces the bin count and therefore reduces
    `tic_preserving` cost proportionally. The guidance is directionally right.
    What it does not do is say how steep the cost curve is.

### Corrections to the handout

| Handout claim | Status | Correction |
|---|---|---|
| 190,000 bins from 4,000 source points, a 45x upsample | Confirmed, slightly understated | 47.5x. 190,000 bins verified by direct call, realized width a flat 5.0000 mDa |
| The store stays small (~29 MB) because it is sparse, so this is not a disk problem | **Half wrong** | True for `nearest_neighbor` at +0.8% for the default. **False for `tic_preserving`: 574.33 MB versus 6.93 MB for 400 pixels, an 82.9x blowup.** [The Resampling page](../docs/resampling.md#bin-count) repeats the unqualified claim and should be scoped |
| It costs every consumer that materialises the table | **Refuted** | +21.2 MB RSS (+4.35%) and no measurable time difference at 190,000 versus 4,000 bins. Ion-image and spectrum times unchanged |
| Wide `var` axes are the design target of the lazy path | **Confirmed, and stronger than claimed** | Narrow lazy ion images are 8.6x *faster* on the wide axis. The effect reverses only for full-range column reads |
| Is 45x upsampling inventing resolution the instrument never had? | Depends entirely on the method | `nearest_neighbor` invents nothing: only 2.11% of bins are ever populated, and the fine grid adds +/-2.50 mDa of quantisation on top of the +/-118.78 mDa the source sampling already imposes. `tic_preserving` interpolates 99.95% of stored values -- which is what it is for, so the objection is cost, not integrity |
| `peak density` is already collected in `DataCharacteristics`, so a source-derived default is feasible | Partly | `avg_peaks_per_spectrum` is a peak *count*, not a spacing. For centroid data, dividing the mass range by it would badly underestimate the resolution needed. It is a sound proxy for profile data only |
| The handout cites only the `constant` axis | Incomplete | On the same range, `reflector_tof` gives 313,723 bins, `orbitrap` 434,851, `fticr` 633,333. On 100-2000 Da, `fticr` reaches 1,900,000 |

### Proposal for the owner

**Nothing here should be changed quietly.** Three options, with figures.

**Option A -- leave the defaults alone.**
On `nearest_neighbor`, which covers centroid data, unknown instruments and the
tutorial dataset, the current default costs +4.4% consumer memory, +0.8% disk
and zero extra open time, and makes narrow lazy ion images 8.6x faster. The
one real cost is producer time: 4.85 s to 9.19 s at 10,000 pixels, a ~1.9x
conversion-time tax for bins that are 97.9% permanently empty. On
`tic_preserving`, this option accepts an extrapolated 142.9 GB store for a
100,000-pixel acquisition.

**Option B -- derive the default bin count from observed source spacing.**
Before and after, at 10,000 pixels for `nearest_neighbor` and 400 pixels for
`tic_preserving`, moving from 190,000 bins to a source-matched 4,000:

| Metric | Now, 190,000 bins | Proposed, 4,000 bins | Change |
|---|---|---|---|
| `nearest_neighbor` store | 173.35 MB | 171.93 MB | -0.8% |
| `nearest_neighbor` nnz | 39,850,401 | 39,850,401 | unchanged |
| Eager RSS | 508.1 MB | 486.9 MB | -4.2% |
| Eager open time | 0.34 s | 0.34 s | unchanged |
| Lazy 1 Da ion image | 8.2 ms | 70.5 ms | **8.6x slower** |
| Lazy full-range column read | 716.5 ms | 391.1 ms | 1.83x faster |
| Lazy single-pixel spectrum | 0.704 s | 0.273 s | 2.6x faster |
| Conversion time, 10,000 px | 9.19 s | 4.85 s | 1.9x faster |
| **`tic_preserving` store, 400 px** | **574.33 MB** | **6.93 MB** | **-98.8%** |
| **`tic_preserving` peak RSS** | **3,377 MB** | **~350 MB** | **-90%** |
| **`tic_preserving`, extrapolated 100,000 px** | **142.9 GB** | **~1.7 GB** | **-98.8%** |

**Compatibility impact on Ousia.** This changes the `var` axis of every future
conversion -- a different number of columns and different m/z bin centres.
Specifically:

- Ion images extracted by **summing over an m/z window remain correct**. The
  checksum `93552464.50942102` was identical at every bin count tested. This
  is the access pattern
  [the Resampling page already recommends](../docs/resampling.md#nearest_neighbor).
- Anything that selects a **single nearest bin**, caches bin indices, or
  persists an m/z-window selection as column indices will silently disagree
  between stores written by different Thyra versions.
- Narrow-window lazy ion images -- the primary Ousia access pattern -- would
  get **8.6x slower** on the `nearest_neighbor` path, from 8.2 ms to 70.5 ms.
  Still fast in absolute terms, but it is a regression.
- Stores mixed across Thyra versions in one analysis would not be
  column-comparable.

This needs a deprecation path and a version bump, not a default flip.

**Option C -- fix the resampler instead, and the bin-count pressure
disappears.**
Replace `tic_preserving`'s linear point-sampling with conservative,
intensity-splitting rebinning. A prototype measured output non-zeros
**bounded by 2x the source point count and independent of bin count**: 2,988
versus 71,314 for a profile spectrum at 190,000 bins, 300 versus 188,159 for a
centroid spectrum. It is exactly TIC-preserving (ratio `1.00000000`), never
allocates an array of the target width, and is 4.4x faster at 190,000 bins and
14.8x faster at 1,000,000 bins.

Option C is also a behaviour change, but a different one: it changes the
*values* written on the `tic_preserving` path without touching the `var` axis,
so Ousia's column indices, cached selections and cross-version comparability
all survive. Option B changes the axis but not the interpolation semantics.
Option C additionally fixes the TIC correctness bug documented under lead 4.

The two engineers disagreed on A versus B. Option C dissolves the
disagreement, at the price of a change to interpolated intensities that needs
scientific sign-off.

---

## Lead 3: processed-mode imzML scans every spectrum twice

**Verdict: real -- it is a triple scan -- and modestly worth fixing. The
verifier disagreed with the original "not worth fixing" label, and the
disagreement holds.**

### What the handout claimed

With `--no-resample` on processed data, the raw-axis path iterates all spectra
to collect unique m/z values, and there is a separate "Scanning mass range and
counting peaks" pass before it. It asked whether they can be merged, and
whether the unique-m/z collection "can use something cheaper than a Python set
over every value".

### What was measured

It is not a double scan. Counting `getspectrum` calls per phase, and
separately counting bytes read off the `.ibd` file handle:

| Configuration | Pass A, metadata scan | Pass B, unique m/z | Pass C, read loop | Bytes read / file size |
|---|---|---|---|---|
| Default, `--no-resample` | N | N | N | 3.00x |
| `pixel_size_um` given, `streaming=False` | N | N | N | 3.00x |
| **`streaming=True`** | N | N | **2N** | **4.00x** |
| Resampling on, control | N | -- | N | 2.00x |

The byte-ratio channel is independent of `getspectrum` instrumentation
entirely and confirms three genuine full passes over the `.ibd`. The
`streaming=True` row is a separate bug: the streaming converter's own read
loop reads every spectrum twice, a 100% redundant read on exactly the path
meant for the biggest datasets.

Cost of the two redundant pre-passes as a share of total conversion wall
clock, on shared-pool processed data with 500 peaks per spectrum:

| Spectra | `.ibd` | Pass A | Pass B1 | Pass B2 | A+B1+B2 share |
|---|---|---|---|---|---|
| 500 | 3 MB | 1.0% | 0.5% | 0.2% | 1.7% |
| 2,000 | 12 MB | 2.5% | 1.3% | 0.4% | 4.2% |
| 8,000 | 48 MB | 4.2% | 2.2% | 0.7% | 7.0% |
| 32,000 | 192 MB | 4.8% | 2.7% | 0.9% | 8.4% |
| 128,000 | 768 MB | 5.5% | 2.8% | 0.9% | 9.2% |

Where a large processed-mode conversion actually spends its time, at 128,000
spectra:

| Phase | Time | Share |
|---|---|---|
| `ImzMLParser.__init__`, XML parse | 31.8 s | 49% (verifier measured 35.9%) |
| `_process_spectra`, pass C | 16.4 s | 25% |
| `_finalize_data` | 6.8 s | 11% |
| A + B1 + B2, the lead | 5.9 s | 9% |
| `_save_output` | 2.8 s | 4% |

Cheaper implementations of pass A, at 32,000 spectra and 16M peaks:

| Implementation | Time | Change |
|---|---|---|
| As written: `getspectrum`, then `del intensities` | 194 ms | baseline |
| m/z bytes only, skipping the intensity array | 155 ms | -20% |
| First and last m/z value only, 16 B per spectrum | 94 ms | -51% |
| Peak counts from `parser.mzLengths`, zero `.ibd` I/O | 0.6 ms | **-99.7%** |

**The memory finding, which the handout does not mention at all.** Peak extra
process RSS of `get_common_mass_axis()` as a multiple of the file's m/z
payload:

| m/z payload | Peak RSS | Ratio |
|---|---|---|
| 8 MB | 24.5 MB | 3.06x |
| 32 MB | 101 MB | 3.16x |
| 128 MB | 411 MB | 3.21x |
| 512 MB | 1,691 MB | 3.30x |
| 1,049 MB | 3,470 MB | 3.31x |
| 128 MB, all-distinct m/z | 532 MB | 4.15x |

This is analytically derivable -- the `all_mzs` list, plus `concatenate`, plus
`np.unique`'s internal copy, is 3P, and the output adds a fourth P when all
values are distinct -- and both engineers reproduced it to within 2%. It does
not amortise. **Extrapolated:** a 100 GB `.ibd` with float64 m/z and float32
intensity carries a 66.7 GB m/z payload, which at 3.30x needs ~220 GB of peak
RSS to build the common mass axis. The test machine has 137 GB, so
`get_common_mass_axis()` hard-OOMs somewhere around a 40 GB input -- and the
streaming converter does not escape it, because
`StreamingSpatialDataConverter` inherits `_initialize_conversion` and calls
`_setup_mass_axis` unconditionally.

### What the verification pass found, and where it disagreed

The verifier reproduced the pass counts exactly, through the independent
byte-counting channel, the memory ratios to within 2%, and the A+B1+B2
fractions in magnitude -- measuring them slightly **higher** than the first
report, at 3.2% / 9.8% / 11.4% for 2,000 / 32,000 / 128,000 spectra.

Two disagreements, both material.

> **The headline is wrong by ~2.5-3x, from a unit mismatch.** The claim
> "merging would save only ~1.9% of total conversion time" divides a saving
> measured in *isolated* micro-benchmarks (2,320 ms to 1,111 ms) by a total
> measured in a *conversion run* (64.45 s). In that same conversion run,
> A+B1+B2 was measured at 5.92 s, not 2.32 s. The report contains the
> inconsistency in plain sight: pass A at 32,000 spectra appears as both
> 194 ms (isolated) and 0.859 s (in-conversion). Applying the report's own
> 52% merge ratio to its own in-conversion cost gives **4.8%**; independent
> data gives **5.9%**.

The verifier's own isolated (246 ms) and in-conversion (234 ms) numbers for
pass A agree within 5%, and their absolute times are ~3.3x faster throughout,
which points at CPU contention during the first engineer's conversion runs
rather than a methodology break. The corrected figure is **~5%, not ~2%**.

> **The top proposed fix contains a correctness bug as stated.** The report
> asserts that per-pixel `peak_counts` is exactly `parser.mzLengths`.
> `total_peaks` is; `peak_counts` is not. `_store_pixel_peak_count` writes
> `peak_counts[pixel_idx]` where `pixel_idx` is derived from the spectrum's
> *coordinate*, not its index -- a scatter through the coordinate map with a
> bounds filter, not a positional copy. The two coincide only when spectra are
> stored in dense raster order, which is what both engineers' generators
> happen to produce.

Tested on a file with shuffled acquisition order and variable peaks per pixel:
the naive `np.asarray(parser.mzLengths)` reconstruction **does not match**; a
correct coordinate scatter does. Non-raster acquisition order and irregular
ROIs are common in real MSI, so the fix as written would silently corrupt
per-pixel peak counts. The performance conclusion survives -- a vectorised
scatter is still 44-49x faster than the real scan, 246 ms to 5.1 ms -- but the
implementation must be the scatter, not the copy.

The verifier also noted that the synthetic data is atypically XML-heavy, a
314 MB `.imzML` against a 768 MB `.ibd`, which inflates the
`ImzMLParser.__init__` share and *deflates* the pre-pass share being reported.
On an 8,000 x 2,000-peak file with the identical `.ibd` byte count, A+B1+B2 is
**12.5%** rather than 9.8%. So ~9% is a floor, not a converged value.

### Corrections to the handout

| Handout claim | Status | Correction |
|---|---|---|
| Processed-mode imzML scans every spectrum **twice** | **Understated** | Three times: 3N `getspectrum` calls. Four times with `streaming=True` |
| The raw-axis path iterates all spectra to collect unique m/z | Confirmed | Fires in every `--no-resample` processed configuration, including `streaming=True`. Disappears when resampling is enabled |
| There is a separate "Scanning mass range and counting peaks" pass before it | Confirmed | It runs first, is unconditional -- supplying `--pixel-size` does not skip it -- and is cached so it runs exactly once |
| The code's own warning, "This is slow for large datasets!" | Partly | The warned-about pass is ~3.7% of wall clock. It is not the slow part. It **is** the memory part, which the warning does not mention |
| The unique-m/z collection can use something cheaper than a Python set over every value | **Refuted for imzML** | There is no Python set in the imzML path -- it uses `np.unique(np.concatenate(...))`, which beat a Python set by 5.3-39.8x and a chunked `union1d` fold by 30x. The live Python set is at `thyra/readers/bruker/timstof/timstof_reader.py:78`, on the Bruker timsTOF raw-axis path, which was not measured end to end |
| The two passes can be merged | Confirmed | A merged pass produces byte-identical `total_peaks` and mass range, saving 52-59% of the two-pass cost |

### Conclusion

The merge is worth doing, but as a side effect rather than a goal. The reason
to touch this code path is the memory ceiling: `_collect_processed_mzs` and
`_finalize_mass_axis` materialise every m/z value from every spectrum in a
Python list and then make two more full copies. The right shape is a k-way
merge of the already-ascending per-spectrum arrays into a preallocated buffer
or to disk, which is O(1) extra memory and preserves `np.unique`'s
asymptotics. **Do not** fix it with a chunked `union1d` fold -- that was
measured at 30x slower on exactly the all-distinct data that motivates the
fix. Folding B1 and B2 together and dropping pass A's `.ibd` I/O removes the
triple scan as a by-product.

---

## Lead 4: the dense resampling path allocates per pixel

**Verdict: real and worth fixing. The verifier agreed with the label and
disagreed with the report's headline and one of its recommendations.**

**2026-09-11: `_add_to_sparse_matrix` no longer exists**, so the two places in
this lead that name it in the present tense -- the per-pixel phase table and
the second conclusion -- describe a code path that has since been deleted, not
one you can profile today. A third names it outside this lead, in the
`## What to do` summary table, and carries the same note there. It and its partner `BaseMSIConverter._create_sparse_matrix`
filled and built a `scipy.sparse.lil_matrix`; design decisions D10 (CSC is the
only stored layout) and D11 (the in-memory conversion routes folded into the
streaming one) left both unreachable, and they were removed as
[issue #317](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/issues/317)
item 2. Every table is now assembled by
`thyra/converters/spatialdata/csc_assembly.py`, a memmapped two-pass CSC engine
that never materialises a row-wise sparse object at all. **The measurements
below are left exactly as taken** -- they are the record of what the LIL path
cost, and the storage law they establish is the bar its replacement had to
clear.

### What the handout claimed

`_resample_spectrum_to_indices` returns `np.arange(len(common_mass_axis))` for
every spectrum; the non-nearest-neighbor branch builds a dense full-width array
per pixel; `nearest_neighbor` has a sparse fast path and `tic_preserving` does
not; ~1.5 MB per pixel at 190,000 bins, so a 10,000-pixel `tic_preserving`
conversion does 15 GB of allocation churn. It noted the constraint that TIC
preservation needs the full interpolated spectrum's sum.

### What was measured

The `np.arange`-per-spectrum claim is **wrong for the in-memory converter**:
zero calls to `_resample_spectrum_to_indices` across eight instrumented
conversions. Statically, its only caller is reached inside the branch taken
when resampling is *disabled*; with resampling on, the dense branch uses
`self._cached_mass_axis_indices` at line 1002, built once at line 798.

The pattern is real, but it lives in the **streaming** converter at
`streaming_converter.py:683` -- a fresh `np.arange(len(self._common_mass_axis))`
per spectrum, 265 us and 1.52 MB per call at 190,000 bins, 200 calls for a
100-pixel conversion because that converter makes two passes.

The dense per-pixel array is real and exactly as claimed: 1,484.4 KB per call
at 190,000 bins. The 15 GB of churn is arithmetically confirmed at 15.20 GB,
and its cost is contested -- see the disagreement below.

What actually dominates, per pixel, at 190,000 bins:

| Phase | `tic_preserving` | `nearest_neighbor` |
|---|---|---|
| Resample | 632.5 us | 514.4 us |
| TIC accumulation | 242.9 us | 16.3 us |
| `_add_to_sparse_matrix` | **3,124.2 us** | 49.7 us |
| Loop total | 4,170.9 us | 635.8 us |

Storing the output is 75% of the loop, and the resample is 15%. The reason is
what gets stored:

| Data and bins | `tic_preserving` nnz/pixel | `nearest_neighbor` nnz/pixel | Amplification |
|---|---|---|---|
| Profile, 190,000 bins | 189,997 | 3,984 | 47.7x |
| Profile, 50,000 bins | 49,999 | 3,984 | 12.5x |
| Profile, 10,000 bins | 9,999 | 3,984 | 2.5x |
| Centroid, 150 peaks, 50,000 bins | 49,998 | 149.8 | **334x** |

Memory follows stored non-zeros with a strikingly stable constant:

| Pixels, 10,000 bins | `tic_preserving` peak RSS | `nearest_neighbor` peak RSS | Bytes of peak RSS per stored nnz |
|---|---|---|---|
| 100 | 68.6 MB | 53.0 MB | -- |
| 400 | 185.7 MB | 102.9 MB | -- |
| 900 | 399.6 MB | 180.2 MB | 44.4 |
| 1,600 | 705.7 MB | 302.4 MB | 44.1 |
| 3,600 | 1,582.2 MB | 673.9 MB | 44.0 |
| 10,000 | 4,385.6 MB | 1,862.6 MB | 43.9 |

That law predicted 4.39 GB for the 10,000-pixel run before it was executed;
measured 4.386 GB. **Extrapolated** to 10,000 pixels at the auto-selected
190,000 bins: ~1.9e9 stored non-zeros, ~84 GB peak RSS and ~14 GB on disk with
`tic_preserving`, versus ~40M non-zeros, ~1.9 GB and ~0.2 GB with
`nearest_neighbor`. The in-memory converter would not complete that job on an
ordinary machine.

**The correctness finding.** `_tic_preserving_resample`
(`base_spatialdata_converter.py:1091-1122`) is a bare `np.interp` with
`left=0` and `right=0` and no renormalisation. Measured output TIC divided by
input TIC:

| Bins | 4,000-point profile | 150-peak centroid |
|---|---|---|
| 1,000 | 0.2494 | 6.71 |
| 4,000 | 1.0000 | -- |
| 10,000 | 2.5004 | 67.10 |
| 50,000 | 12.5029 | -- |
| **190,000, auto default** | **47.5116** | **1274.99** |

`_nearest_neighbor_resample`, which is not named for the property, preserves
TIC exactly (`1.000000`) at every bin count and both spectrum types.

This is not a naming quibble. The repository contradicts itself in three
places:

- `thyra/resampling/types.py` documents `TIC_PRESERVING` as "Redistribute
  intensity so the total ion count is preserved after rebinning (recommended
  for quantitative work)."
- `thyra/resampling/strategies/tic_preserving.py:93-98` contains a second
  implementation that **does** renormalise
  (`scaling_factor = original_tic / new_tic`).
- [The Resampling page](../docs/resampling.md#tic_preserving) says "followed by
  rescaling so the spectrum's total ion current matches the original".

The converter's inline implementation matches none of them.

### What the verification pass found, and where it disagreed

The verifier reproduced twelve measurements to within 2%: the densification
(189,996.9 versus 3,984.9 nnz per pixel), the TIC inflation (47.5056 against
47.5116, and `nearest_neighbor` at exactly 1.000000), the RSS scaling series
point by point, the 44 bytes per nnz constant at 44.42 / 44.14 / 43.95, the
phase breakdown, the zero calls to `_resample_spectrum_to_indices`, and the
churn probe itself (0.062 s against 0.061 s). Instrumenting
`_process_resampled_spectrum` to record the converter class, resampling method
and actual axis length confirmed on every run that the intended code path was
exercised with no silent fallback.

The disagreement is about what that churn probe measures.

> **The headline is wrong by roughly 40x.** The claim that the 15 GB of churn
> "costs 0.061 s (0.9% of the resample)" rests on a probe that never touches
> the buffer: `a = np.empty(190_000); del a`. On Windows an allocation this
> size goes to `VirtualAlloc`, and pages are committed only on **first write**.
> The loop reserves and releases address space. The real `np.interp` writes all
> 190,000 values and therefore takes **373.0 soft page faults per call, every
> call**, deterministically across five repeats with zero variance -- and
> 371.1 is exactly 1,520,000 bytes / 4096.

Direct A/B on `np.multiply`, fresh output versus `out=`, identical arithmetic:

| Elements | Bytes | Fresh output | Page faults | With `out=` | Allocation penalty |
|---|---|---|---|---|---|
| 10,000 | 78 KB | 1.85 us | 0.0 | 1.61 us | none |
| 100,000 | 781 KB | 24.5 us | 0.0 | 21.4 us | none |
| **190,000** | **1.45 MB** | **312.5 us** | **372.7** | **68.8 us** | **243.7 us** |
| 500,000 | 3.8 MB | 771.4 us | 978.5 | 166.6 us | 604.8 us |

That prices the fresh per-pixel allocation at roughly **240 us of the ~660 us
resample -- about 36%, not 0.9%**. The effect has a hard threshold: zero
faults were measured at every size up to 100,000 float64 elements (781 KB),
and 372 at 190,000 (1,484 KB). The first report's entire end-to-end
pixel-scaling series ran at 10,000 bins, **below the threshold** -- so those
numbers are correct and structurally incapable of detecting the effect they
were used to dismiss.

The consequence is that the first report's explicit "NOT proposed: any change
aimed at reducing allocation count" is refuted by measurement. A bit-exact
alternative to `_nearest_neighbor_resample` that allocates the dense
accumulator once and clears only touched entries -- verified as an identical
index array with `max|delta| = 0.000e+00` -- measured:

| Bins | Current | Reused accumulator | Change |
|---|---|---|---|
| 10,000 | -- | -- | **94% slower**: no faults occur, and `np.add.at` loses to `np.bincount` |
| 50,000 | -- | -- | **28% slower** |
| 190,000 | 661.2 us | 375.3 us | **43.2% faster**, 371 faults to 0 |
| 500,000 | 1,517.7 us | 434.0 us | **71.4% faster** |

The verifier also measured 372.6 page faults per pixel inside
`_nearest_neighbor_resample` as well -- `np.bincount` allocates its own dense
accumulator spanning the axis. So on the path the first report recommends
migrating toward, the fresh allocation is ~240 us of a 579.6 us per-pixel
loop, about 42% of the entire per-pixel loop.

Finally, the verifier noted that the first report's supporting evidence for
"the churn is free" -- `tracemalloc` peak flat at 1.55 MB over 2,000 resamples,
RSS delta +0.0 MB -- reproduces exactly and is true, but measures memory
*footprint*, not allocation *time*. A flat peak is precisely what you expect
when the allocator returns pages to the OS and re-faults them, which is the
expensive case.

### Corrections to the handout

| Handout claim | Status | Correction |
|---|---|---|
| `_resample_spectrum_to_indices` returns `np.arange(...)` for every spectrum | **Refuted for the in-memory converter** | Zero calls in eight instrumented conversions. The pattern is real but lives at `streaming_converter.py:683` |
| "roughly lines 890-970" | Wrong location | The dense branch is `_tic_preserving_resample` at lines 1091-1122; the index array comes from a cache built once at line 798 |
| ~1.5 MB allocated per pixel at 190k bins | Confirmed | 1,484.4 KB |
| A 10,000-pixel `tic_preserving` dataset does 15 GB of allocation churn | Confirmed arithmetically at 15.20 GB | Its cost is ~36% of the resample, not the 0.9% first reported. See the disagreement above |
| `nearest_neighbor` has a sparse fast path; `tic_preserving` does not | **Partly** | True of the *output*, false of transient memory. `np.bincount` allocates a dense accumulator, and `nearest_neighbor` peaks **higher** at 1,900.2 KB versus 1,516.3 KB. Its advantage is entirely downstream in what gets stored -- and, per the page-fault finding, not in allocation time either |
| TIC preservation needs the full interpolated spectrum's sum, which makes sparse output non-trivial | **Premise is wrong** | There is no rescaling factor to compute -- `_tic_preserving_resample` never renormalises. Separately, the sum *is* computable analytically without materialising the dense array, with a prototype exact to 1e-13 and 6.8x faster at 190k bins, 284.6x at 1M bins for centroid data -- but the sum was never the constraint, because the dense *values* are what gets written and `np.interp` output is 37-99% non-zero, so there is no sparsity to recover by thresholding |

### Conclusion

Worth fixing, for three separate reasons, in this order.

1. **Correctness.** `_tic_preserving_resample` inflates TIC by up to 1275x at
   the auto-selected axis, on a code path the detector chain routes real
   profile MALDI-TOF data onto by default. Either rename it to reflect that it
   is linear point-sampling, or make it preserve TIC.
2. **The storage blow-up, which the same fix removes.** Conservative,
   intensity-splitting rebinning -- for each source point, find the straddling
   target bins and deposit `I*(1-w)` and `I*w`, then accumulate duplicates --
   measured exactly TIC-preserving, with output non-zeros bounded by 2x the
   source point count and independent of bin count, and 4.4x faster at 190,000
   bins. That one change removes the 44 bytes per nnz memory law, the
   4.4x-189x store inflation, and the 3,124 us per pixel spent in
   `_add_to_sparse_matrix`. Caveat: at small bin counts with dense profile
   input the prototype was slower than `np.interp`, 132 versus 39.8 us,
   because the argsort dominates.
3. **Allocation reuse above the page-fault threshold.** Contrary to the first
   report, this is worth ~43% of the resample at 190,000 bins and ~71% at
   500,000, on both methods. It must be gated on axis width, because below
   roughly 1 MB of axis it is a regression.

Two smaller items: `streaming_converter.py:683` should use the caching pattern
already present at `base_spatialdata_converter.py:798`; and
`spatialdata_2d_converter.py:220` calls
`np.add.at(total_intensity, arange(190k), intensities)` at 242.9 us per pixel
at 190,000 bins, where a plain `+=` would do, since the indices are known to
be the full arange.

---

## What to do

**2026-09-11: item 3 below can no longer offer "the 3,124 us per pixel store
cost" as a saving.** That cost was `_add_to_sparse_matrix`, which has since
been deleted -- see the note at the head of lead 4 for what replaced it and
why -- so a caller who lands on that row today would be told to buy back a
cost that is already gone, and would have no way to reproduce the 3,124 us.
What the CSC assembler that replaced it costs per pixel was not measured here,
so read that clause as withdrawn rather than as zero. The rest of the row is
untouched by the deletion and stands as taken: the nnz figures, the exact-1.0
TIC ratio, the 4.4x and 14.8x speedups, and the 142.9 GB extrapolated store.

### (a) Changes the numbers justify

| # | Change | Evidence | Size |
|---|---|---|---|
| 1 | **Done -- [PR #112](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/112).** Make `_tic_preserving_resample` (`base_spatialdata_converter.py:1091-1122`) either preserve TIC or stop claiming to | TIC ratio 47.5 for profile to 1275 for centroid at the auto default; contradicts `resampling/types.py`, `resampling/strategies/tic_preserving.py:93-98`, and `docs/resampling.md` | Correctness bug on a live default path |
| 2 | Bound the memory of `ImzMLReader._collect_processed_mzs` and `_finalize_mass_axis` with a k-way merge of the already-ascending per-spectrum arrays | Peak RSS 3.30x the m/z payload, measured across a 131x payload range; extrapolated hard OOM at ~40 GB input; the streaming path does not escape it | Blocker for the stated 100+ GB use case |
| 3 | Replace `tic_preserving`'s point-sampling with conservative rebinning, which subsumes item 1 | Output nnz 2,988 versus 71,314 at 190k bins, independent of bin count; TIC ratio exactly 1.0; 4.4x faster at 190k bins, 14.8x at 1M | Removes the 142.9 GB extrapolated store and the 3,124 us per pixel store cost |
| 4 | Fix the `streaming=True` double read of every spectrum, 4N versus 3N `getspectrum` calls | Confirmed two ways: call counting, and 4.00x file-size byte counting | 100% redundant read on the large-dataset path |
| 5 | Drop pass A's `.ibd` I/O in `ImzMLMetadataExtractor`, using a **vectorised coordinate scatter** for `peak_counts` | 246 ms to 5.1 ms at 32,000 spectra, 44-49x. Must not use `np.asarray(parser.mzLengths)` directly -- proven to corrupt per-pixel counts on non-raster acquisition order | ~5% of wall clock, a few lines |
| 6 | Reuse the dense accumulator instead of allocating fresh per pixel, **gated on axis width** | Bit-exact; 661.2 to 375.3 us at 190k bins (-43%), 1517.7 to 434.0 us at 500k (-71%); a regression below ~100,000 float64 elements | ~36% of the resample above the threshold |
| 7 | Use the cached index array at `streaming_converter.py:683`, as `base_spatialdata_converter.py:798` already does | 265 us and 1.52 MB per call at 190k bins, 200 calls per 100-pixel conversion | ~5% of that path; disappears if item 3 lands |
| 8 | Delete the duplicate `zarr.consolidate_metadata` at `base_spatialdata_converter.py:1922` | Exactly two consolidate calls per conversion; the duplicate costs 35-44 ms | One line; fold into another change |
| 9 | Correct `docs/resampling.md`: the `tic_preserving` description, which claims a rescaling step that does not exist, and the scope of the "the store stays small -- roughly 29 MB here" claim | The same dataset with `tic_preserving` is 82.9x larger | Documentation only |

### (b) Changes the numbers do not justify

| Change | Why not |
|---|---|
| Restructuring the Zarr write path to reduce metadata rewrites | Fixed ~0.24 s regardless of dataset size: 25.5% of a 100-pixel toy conversion, 1.01% at 50,000 pixels, ~0.03% extrapolated at 100 GB |
| Removing the `WinError 5` retry | The race needs only one contended rename, and there are 413 per conversion. Keep the retry |
| Using `write_element` incrementally instead of one-shot `write()` | 7% *worse* in isolation: 245 versus 229 atomic writes for 16 images. The PCS path's 297 versus 413 is a smaller fixed cost, not a scaling fix |
| Replacing `np.unique(np.concatenate(...))` in the imzML path with a Python set or a chunked `union1d` fold | Python set 5.3-39.8x slower with 9x the memory; chunked `union1d` 30x slower on exactly the all-distinct data that motivates the change |
| Merging imzML passes A and B purely as a time optimisation | ~5% of total conversion, corrected up from a mis-stated 1.9%. Worth doing as a side effect of the memory fix, not as a goal |
| Reducing allocation *count* for its own sake on the `tic_preserving` path | `tracemalloc` peak is flat at 1.55 MB over 2,000 consecutive resamples; numpy recycles one buffer. What costs is page-fault commit on first write, which is fixed by reuse -- item (a) 6 -- not by allocating less often |
| Shrinking the default bin count on the `nearest_neighbor` path | +4.4% consumer memory, +0.8% disk, no open-time cost, and narrow lazy ion images 8.6x faster on the wide axis. See the lead 2 proposal -- this is the owner's call, and the case against is stronger than the case for |

### (c) Open questions

1. **The `WinError 5` race frequency was never measured.** Deliberately:
   instrumenting `_atomic_write` perturbs the failure rate, so write counts and
   race frequency must not come from the same run. Nothing here confirms or
   refutes the handout's claim that metadata rewriting causes it.
2. **All data is synthetic.** The severity of every `tic_preserving` finding is
   governed by `target_bins / source_points_per_spectrum`, measured at 47.5 and
   1275. A real Rapiflex profile spectrum with ~190,000 native points would
   make both the TIC error and the densification largely vanish for that
   instrument. No vendor data was available.
3. **Nothing was tested against Bruker `.d` input**, and the Bruker timsTOF
   raw-axis path (`timstof_reader.py:78`, which does use a live Python set) was
   never run end to end.
4. **Nothing crossed 100 GB, or even 10 GB.** Largest inputs were 921 MB for
   the imzML pair and 768 MB for the `.ibd`; largest store ~2.3 GB. Every
   100 GB figure is a labelled linear extrapolation, and real 100 GB
   conversions cross the `streaming="auto"` threshold, derived at 134,218
   pixels, onto a code path measured only at small scale.
5. **All timings are warm-page-cache on local NVMe.** The Windows page cache
   could not be dropped. This biases *against* the imzML pass-merge: the
   20-22% saving from m/z-only reads is a warm number, and on a cold cache or
   network share the saving should approach the byte ratio -- 33% fewer bytes
   for skipping intensities, 44% fewer total bytes for merging the passes.
6. **The lazy-path result depends on an AnnData implementation detail.**
   `spatialdata` 0.7.3 in this environment has no `read_zarr(..., lazy=True)`,
   so the lazy path was measured through `anndata.experimental.read_lazy`
   directly. Its fixed 1000-column CSC chunk is what produces the 8.6x
   narrow-window speedup; a different chunk size would move every lazy number.
7. **3D and multi-region datasets were not tested.** Each additional table
   element costs ~161 atomic writes and ~22 root rewrites, so a 50-slice
   dataset lands near 20,000 metadata writes -- still small against data time,
   but no longer trivially negligible, and a proportionally larger exposure to
   the `WinError 5` race.
8. **The conservative-rebin prototype was validated for exact TIC preservation
   and output sparsity, but not for spectral fidelity** against any reference
   implementation, and it clips out-of-range source points rather than dropping
   them as `np.interp`'s `left=0` and `right=0` do.

---

## See also

- **[Resampling](../docs/resampling.md)** -- the methods, axis types and bin-count
  defaults this report measures. Two statements on that page are corrected
  above
- **[Output Format](../docs/output-format.md)** -- what the sparse matrix and `var`
  axis look like in the store
