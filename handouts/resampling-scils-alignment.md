# G. Align resampling with SCiLS, and stop leaking modalities

**Branch:** `fix/resampling-scils-alignment` (not created)
**Worktree:** `../Thyra-resampling`
**Priority:** the modality leak is the real bug. The rest is alignment.

```bash
git fetch origin && git worktree add -b fix/resampling-scils-alignment ../Thyra-resampling origin/main
```

Thyra's resampling was designed against **SCiLS Lab**, and the SCiLS Lab 2026b
User Guide is the baseline it should be judged against. This handout records
what the manual actually says, what Thyra does differently, and which of those
differences are bugs.

Everything below was measured or sourced. **Do not re-derive it.** Where a
claim is inference rather than measurement, it says so.

---

## Framing

Thyra's auto-selection sends **any** high-density profile data to a detector
named `RapiflexDetector`, which returns `tic_preserving` + `constant` -- a
strategy and an axis derived from MALDI-TOF assumptions. There is no check
that the data *is* MALDI-TOF.

That is the bug. `bellini.imzML`, one of the project's own test files, is
**TOF-SIMS** (IONTOF SurfaceLab) -- a different modality entirely, one SCiLS
does not even ingest. It escapes the leak today only because it averages 2,236
points/spectrum, under the 5,000 threshold that gates the branch. A denser
SIMS acquisition would be resampled by MALDI-TOF logic onto a MALDI-shaped
axis, silently.

SCiLS does not guess modality. Its own command-line importer takes it as an
argument (p.81):

```
--project arg    forced project type (TIMSTOF, TOF, FT)
--rep_type arg   spectra representation (PROFILE, CENTROID)
```

---

## What the SCiLS manual says

Page numbers are from `UG_SCiLS_Lab_2026b_en_A.pdf`, revision A.

### The axis types are Thyra's, under their older names (p.75)

| Thyra | SCiLS 2026b | SCiLS legacy name | Law |
|---|---|---|---|
| `constant` | Constant (equidistant) | same | equal spacing |
| `linear_tof` | Axial TOF | **"Linear TOF" in earlier versions** | bin ~ sqrt(m) |
| `reflector_tof` | Orthogonal TOF | **"Reflector TOF" in earlier versions** | bin ~ m |
| `orbitrap` | Orbitrap | same | inverse-sqrt resolution law |
| `fticr` | MRMS (Fourier-transform) | -- | inverse resolution law |

Thyra's names are SCiLS's *legacy* names. The scaling laws match Thyra's
implementations. This finally sources a claim `docs/resampling.md` has been
making without a citation.

### The decision tree (p.79 Figure 46, p.80 text)

```
Single mass axis                     -> No resampling
Multiple mass axes - centroid data   -> Nearest neighbor        (always)
                   - profile data - Identical axis types -> TIC preserving
                                  - Different axis types -> Linear interpolation
```

Verbatim from p.80:

> For centroid spectra SCiLS Lab always uses nearest neighbor resampling.
> [...] In the case of profile data, the resampling method depends on whether
> multiple mass axis types are combined. If all axis types are identical, a
> TIC preserving resampling is applied, otherwise a linear interpolation is
> performed.

**Two things follow, and they are the heart of this handout.**

1. **SCiLS treats "TIC preserving" and "linear interpolation" as different
   methods.** Thyra's `tic_preserving` *is* linear interpolation followed by a
   single global rescale. Thyra has merged two operations SCiLS deliberately
   keeps apart, and kept the name of the one it does not implement.

2. **SCiLS gates TIC-preserving on "identical axis types."** That is exactly
   the condition under which the operator is mathematically exact. Derived
   independently and confirmed by measurement: the composite operator is exact
   **iff the source grid law matches the target axis law**, and off-diagonal
   the peak-ratio error is exactly `(m_hi/m_lo)^(p_target - p_source)` with
   `p = 0, 0.5, 1, 1.5, 2` for constant / linear_tof / reflector_tof /
   orbitrap / fticr. Thyra has no such gate.

### Other sourced facts

- **10 million bin cap** (p.76): "Data sets in SCiLS Lab are limited to a
  maximum of 10 million bins on the common mass axis." Direct precedent for
  `max_mass_axis_length`, which Thyra added with a default of `None`.
- **Interval width is data-derived** (p.126): "filled automatically during
  import with a value fitting best to the given data." Thyra uses a fixed
  5 mDa at m/z 1000. Related but not the same quantity -- SCiLS's interval
  width is the ion-image extraction window, not the axis bin width.
- **Baseline removal is profile-only** (p.76): "not available for any data
  with centroid spectrum representation." SCiLS treats profile/centroid as a
  first-class distinction throughout.
- **Sampling-rate warning** (p.75): "If the sampling rate of the data sets
  varies widely, the resampling might cause a loss of data quality."

---

## Settled questions -- do not reopen these

### Intensity is a COUNT, not a per-Da density

This decides which operator is correct, and it is settled by three independent
lines of evidence measured on the project's own files:

1. **Declared units.** `pea.imzML` and `20240826_xenium_0041899.imzML` both
   attach `MS:1000131 "number of detector counts"` to the intensity array,
   verbatim:
   ```xml
   <cvParam accession="MS:1000515" cvRef="MS" name="intensity array"
            unitAccession="MS:1000131" unitCvRef="MS"
            unitName="number of detector counts"/>
   ```
   `bellini.imzML` declares no unit at all.
2. **The vendors' own TIC is a sum.** Every pea/xenium spectrum carries
   `MS:1000285 total ion current`. It equals `sum(I)` in **800/800** spectra
   to 1e-6 relative, and equals the integral `sum(I dm)` in **0/800**.
3. **The intensity quantum does not track bin width.** `bellini`'s axis is
   uniform in flight time, so its m/z bin width grows **12.71x** across the
   range. Across that whole sweep the smallest positive stored intensity is
   exactly 1 and the value quantum is exactly 1 -- spread 1.00x. A density
   would have a floor of 1/w varying by 12.71x. Reproduced on pea (2.83x width
   variation) and xenium (2x).

**Consequences.** `nearest_neighbor` (accumulate into the nearest bin) is the
correct operator. The TIC rescale added in PR #112 is correct and **must not
be reverted** -- without it `tic_preserving` loses 81% of TIC at a 5x-coarser
target and 98% at 50x. But fixing the total leaves the distribution wrong:
relative L1 against a true area-conserving rebin runs 0.23 -> 1.27 as the
target coarsens.

### The default paths are correct today

Measured on real data, through the real decision tree:

| Path | Result |
|---|---|
| Rapiflex `tic_preserving` + `constant` | 0.00000 error on a normalised ion image, TIC 1.000000 |
| Centroid route on `pea.imzML` | 0 points lost, TIC 1.000000, max snap error 2.4987 mDa (2.5 ppm) |

All five local datasets route to `nearest_neighbor`. **Zero** currently reach
`tic_preserving`. Do not "fix" what is measuring exact.

### The 47.5x upsample claim is an artefact

Earlier material (including `handouts/write-amplification.md` lead 2 and the
findings report) describes the default as a 47.5x upsample. On real data it is
locally a **downsample**: 46% of `bellini`'s source gaps are finer than one
5 mDa bin and 33% of its source points collide into shared bins. The 47.5x
figure comes from the tutorial's synthetic generator (4,000 evenly spaced
points over 250-1200 Da), which is not representative. **Any proposal to cut
the bin count based on that figure is wrong** -- one such proposal (bins = 3x
source point count) would give 479 mDa bins on `bellini` and destroy it.

---

## The work

### 1. Close the modality leak (the actual bug)

`thyra/resampling/instrument_detectors.py`: `RapiflexDetector` currently
matches unknown-vendor high-density profile data and hands it
`tic_preserving` + `constant`. Any modality that is dense and profile gets
MALDI-TOF treatment.

Send unknown-provenance profile data to `nearest_neighbor` instead. It moves
counts into bins, which is correct under the settled extensive reading, and it
is safe for **any** modality because it makes no assumption about the axis law.

**This is a behaviour change** for anyone currently hitting that route: their
intensities will change. Flag it in the PR with before/after numbers.

### 2. Gate `tic_preserving` on matching axis laws, as SCiLS does

Only use it when the source grid law matches the target axis law. In practice
that is the flexImaging/Rapiflex path, where `RapiflexReader` builds a uniform
axis (`np.linspace`, `rapiflex_reader.py:562`) over a source that is uniform
in m/z. Everywhere else, `nearest_neighbor`.

Note this makes item 3's underlying bug harmless, which dissolves an ordering
trap: with the gate in place, nothing unknown reaches the interpolating path,
so the missing gap tolerance (item 6) stops being urgent.

### 3. Stop inferring representation from storage mode

`thyra/metadata/extractors/imzml_extractor.py`, `_detect_centroid_spectrum`
tries `_check_parser_metadata_for_centroid()` **first**, which returns
`"centroid spectrum"` for any *processed*-mode file on the reasoning "If it's
processed data, it's likely centroided" -- and only falls through to the real
`MS:1000127`/`MS:1000128` accession scan if that returns nothing.

Processed and centroid are orthogonal. Processed means each spectrum carries
its own m/z array; it says nothing about peak representation. `bellini`
declares `MS:1000128 profile spectrum` and Thyra calls it centroid.

Read the declared accession first and fall back to the heuristic only when no
accession is present. **Do this after items 1 and 2**, or it opens the
interpolating path for processed-profile files.

Consider also taking modality explicitly, as SCiLS does, rather than
inferring it -- an option mirroring `--rep_type PROFILE|CENTROID`.

### 4. Adopt the 10M bin cap as the default

`max_mass_axis_length` currently defaults to `None` (unlimited) because no
defensible number was available. SCiLS uses 10 million (p.76). Adopt it, cite
the manual, and keep the override.

### 5. Consider implementing real TIC-preserving

SCiLS's "TIC preserving" is a distinct method from linear interpolation.
Thyra's is interpolation wearing that name. A conservative, intensity-splitting
rebin -- treat each source sample as carrying its intensity across its own
width, redistribute onto target bins by overlap -- is exact on every axis type
and every source grid, preserves TIC by construction, is fully local, and
produces no comb artefact. A prototype measured peak-ratio 1.0000, TIC
1.0000 and locality 1.0000 in all 10 configurations tested.

**This changes stored values on the `tic_preserving` path**, so it needs
sign-off, not just a green test run. It is the honest fix; item 2 is the safe
one. Doing item 2 first means this can be taken at leisure.

Note this also corrects `handouts/interpolation-resampling.md` (handout F),
which describes `tic_preserving` as redistributing intensity. That is what
SCiLS means by the name and what item 5 would build -- it is not what Thyra
currently does.

### 6. Add a gap tolerance to `_tic_preserving_resample`

`np.interp` has no gap tolerance, so on sparse/thresholded m/z arrays it draws
straight lines across empty regions. Forcing `tic_preserving` on `bellini`
puts **80.47%** of output TIC into m/z regions where nothing was measured
(`nearest_neighbor`: far lower). Cardinal (`tolerance`) and matter
(`approx1(..., tol=)`) both have this parameter.

Zero out any output bin farther than a tolerance from the nearest source m/z.
On dense profile data (max gap 0.0834 Da on the one continuous profile file
available) any tolerance above ~0.1 Da is a no-op.

Lower priority once items 1-2 land, because nothing unknown will reach this
code. Still worth having.

### 7. Documentation corrections

- `docs/resampling.md` claims the 17 mDa / m/z 300 `linear_tof` default
  "reproduces the axis SCiLS Lab produces for FlexImaging data." **That figure
  does not appear anywhere in the 2026b manual.** Either source it from a
  SCiLS version that does state it, or soften the claim.
- Update the axis-type table to note the current SCiLS names (Axial TOF,
  Orthogonal TOF, MRMS), since Thyra uses the legacy ones.
- The `FTICRDetector` and `OrbitrapDetector` dead-code warning added in
  PR #124 stays accurate until item 3 lands; revisit it then.

---

## Constraints

**`read_lazy` must keep working.** `anndata.experimental.read_lazy()` on Thyra
output is what Ousia depends on. Guarded by
`tests/unit/test_read_lazy_contract.py` and
`tests/unit/converters/test_lazy_loading_encoding.py`. Assert it still works,
not merely that files exist.

**Do not revert the TIC rescale** (`thyra/resampling/tic.py`, PR #112). It is
correct under the settled extensive reading. If item 5 lands, the rescale
becomes redundant rather than wrong.

**Do not touch the Rapiflex default path** without a measurement showing it
improves. It currently measures exact (0.00000 ion-image error).

**Do not coarsen the centroid path.** Measured clean on real `pea` data:
0 points lost, TIC exactly 1.000000.

**Coordinate systems are contract-tested.**
`tests/unit/converters/test_coordinate_systems.py` pins that the TIC image and
pixel polygons agree at `"global"`.

**Behaviour changes need before/after numbers in the PR.** Items 1, 3 and 5
all change output for some inputs. Thyra is a library with an external
consumer (Ousia); changes that alter stored values are the owner's call, not
the implementer's.

---

## Environment

**The venv trap (historical, now gone).** This work was done under Poetry, where
the venv was shared across worktrees and held `thyra` as an editable install
pointing at the *main* checkout, so `poetry run python` from a worktree imported
the wrong tree. `PYTHONPATH` alone was not enough -- `sys.path[0]`, the script's
own directory, beat it -- and `poetry run` from a worktree resolved to a
different, empty venv with no numpy. The workaround was to call the venv
interpreter by absolute path and assert on `thyra.__file__` in every script.

uv removed the trap rather than renaming it. It resolves the project root by
walking up from the working directory, and a worktree has its own
`pyproject.toml`, so each worktree gets its own `.venv` holding its own source.
`uv run` syncs that environment before running:

```bash
uv run pytest -q
uv run python <script.py>
```

**Real data lives outside the worktree**, at
`<thyra-checkout>/test_data`. A previous investigation missed
it entirely by searching only the worktree, and reached the wrong conclusion
as a result. Inventory, with what Thyra's real decision tree selects:

| Dataset | Declared | Mean pts/spec | Margin to 5,000 | Routes to |
|---|---|---|---|---|
| `bellini.imzML` (IONTOF **TOF-SIMS**) | profile (overridden to centroid) | 2,220 | +55.6% | nearest_neighbor |
| `pea.imzML` (SCiLS/Bruker flex) | centroid | 4,722 | **+5.6%** | nearest_neighbor |
| `20240826_xenium_0041899.imzML` | centroid | 1,273 | +74.5% | nearest_neighbor |
| `20231109_PEA_NEDC.d` (timsTOF) | -- | 2,314 | +53.7% | nearest_neighbor |
| `20240826_Xenium_0041899.d` (timsTOF) | -- | 1,273 | +74.5% | nearest_neighbor |

`pea` has only 5.6% headroom, and 46% of its individual spectra already exceed
5,000 -- but the threshold compares the **mean** (`total_peaks / n_spectra`),
so only that scalar matters. Worth a regression test pinning its routing.

**Line endings.** Every commit touching a file fails the `mixed-line-ending`
hook once, rewrites to LF, and needs a second `git add` + `git commit`. Do not
run `pre-commit run --all-files`; it rewrites unrelated files.

---

## Verification

```bash
uv run pytest -q
uv run black . && uv run isort . && uv run flake8
uv run --group docs mkdocs build --strict
```

Beyond the suite, for anything touching the operators:

- Convert `pea` and `bellini` end to end and assert `read_lazy` opens the
  result.
- Measure TIC in/out per pixel for every method x axis-type pairing you touch.
- For a behaviour change, report the ion-image error against the previous
  implementation on a real file, not a synthetic one.

## Deliverable

Items 1-2 as one PR -- they are one decision, and together they close the leak.
Items 3, 4, 6, 7 separately. Item 5 only with sign-off.

State plainly in each PR whether stored values change, and for which inputs.
