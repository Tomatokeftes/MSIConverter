# Resampling

Mass axis resampling maps every spectrum in a dataset onto one shared m/z axis.
It is **enabled by default**, and by default every parameter is chosen
automatically from the acquisition metadata.

This page explains what that automatic choice is, how it is made, and how to
override it. For the list of flags alone, see the
[CLI Reference](cli.md#resampling-advanced).

---

## Why resampling exists

An imzML file comes in one of two storage modes:

| Mode | m/z arrays | Consequence |
|---|---|---|
| **continuous** | one shared array for the whole dataset | pixels are already comparable bin-for-bin |
| **processed** | a separate array per spectrum | pixel 1's column 500 and pixel 2's column 500 are *different masses* |

For processed data -- which includes most centroided vendor exports -- you
cannot build a pixel x m/z matrix at all without first agreeing on a common
axis. Thyra reports this as `needs_resampling`, and it is the reason resampling
defaults to on rather than off.

Even for continuous data, resampling is often still what you want: it lets you
impose a physically sensible bin spacing, restrict the mass range, or match the
axis convention of another tool.

To keep the original per-spectrum axes untouched, disable it:

```bash
thyra input.imzML output.zarr --no-resample
```

Note that with `--no-resample` on processed data, each pixel keeps its own m/z
values, and most downstream tools will not be able to treat the table as a
matrix of comparable features.

---

## How the automatic choice is made

Three things are decided: the **method** (how intensity moves onto the new
axis), the **axis type** (how bin widths scale with m/z), and the **bin count**.

```
  acquisition metadata
          │
          ▼
  DataCharacteristics        storage mode, spectrum type, instrument
          │                  name/type/manufacturer, peak density
          ▼
  InstrumentDetectorChain    first matching detector wins
          │
          ├──────────────► resampling method   (nearest_neighbor | tic_preserving)
          ├──────────────► axis type           (constant | *_tof | orbitrap | fticr)
          └──────────────► bin count           from a target width at a reference m/z
```

### What Thyra looks at

`DataCharacteristics` collects, from the metadata:

- whether there is a shared mass axis (continuous) or not (processed)
- spectrum type -- `centroid spectrum` (`MS:1000127`) or `profile spectrum` (`MS:1000128`)
- instrument name, instrument type, and manufacturer
- average peaks per spectrum; above **5000** the data is recorded as high-density profile
- format flags for Rapiflex, timsTOF, PHI, and Waters

Spectrum type is whatever the file **declares** -- `MS:1000127` or
`MS:1000128` -- read from the file description first and then from the rest of
the document. Only when a file declares neither does Thyra fall back to
guessing centroid from processed storage mode, and it says so in a warning
when it does. Processed and centroid are independent: processed means each
spectrum carries its own m/z array, which says nothing about whether its peaks
are centroided.

Peak density is reported but does **not** select a method or an axis type. It
describes how finely the spectra were sampled, not what acquired them, and a
dense profile spectrum can come from a MALDI-TOF, a TOF-SIMS, or an Orbitrap
run in profile mode. SCiLS Lab does not guess modality either -- its importer
takes it as an argument (`--project TIMSTOF|TOF|FT`).

The timsTOF flag is a case-insensitive substring match on the instrument name,
because Bruker names the family many ways (`timsTOF fleX MALDI-2`,
`timsTOF Pro 2`, `timsTOF SCP`, and so on). The name comes from the Bruker
`.d`'s `GlobalMetadata` when there is one, and otherwise from the instrument
model the imzML itself declares -- so a timsTOF exported to imzML rides the
same match, as long as the export names the instrument (see the info box
below for the one common exporter that does not).

### Which detector wins

Detectors are tried in a fixed priority order and the first match wins:
timsTOF, Rapiflex, FT-ICR, Orbitrap, PHI, Waters, generic centroid, then a
catch-all default. This table is the actual observed behaviour of that chain:

| Metadata | Detector | Method | Axis type |
|---|---|---|---|
| timsTOF, centroid | timsTOF | `nearest_neighbor` | `reflector_tof` |
| timsTOF, profile high-density | timsTOF | `nearest_neighbor` | `reflector_tof` |
| Rapiflex, profile | Rapiflex MALDI-TOF | `tic_preserving` | `constant` |
| Bruker MALDI-TOF | Rapiflex MALDI-TOF | `tic_preserving` | `constant` |
| imzML declaring an FT-ICR analyzer or model | FT-ICR | `nearest_neighbor` | `fticr` |
| solariX `.d` (native, peaks.sqlite) | FT-ICR | `nearest_neighbor` | `fticr` |
| imzML declaring an Orbitrap analyzer or model | Orbitrap | `nearest_neighbor` | `orbitrap` |
| PHI SmartSoft-TOF `.raw` | PHI SmartSoft-TOF (ToF-SIMS) | `nearest_neighbor` | `linear_tof` |
| Waters MassLynx `.raw`, profile trace (the SELECT SERIES MRT default) | Waters MassLynx (profile trace) | `tic_preserving` | `linear_tof` |
| Waters MassLynx `.raw`, SELECT SERIES MRT vendor centroid | Waters SELECT SERIES MRT (vendor centroid) | `nearest_neighbor` | `tof` |
| Waters MassLynx `.raw`, vendor centroid or undeclared, other instruments | Waters MassLynx | `nearest_neighbor` | `reflector_tof` |
| unknown vendor, profile (any density) | Unknown (default) | `nearest_neighbor` | `constant` |
| unknown, centroid | ImzML Centroid | `nearest_neighbor` | `reflector_tof` |
| no usable metadata | Unknown (default) | `nearest_neighbor` | `constant` |

!!! note "Why PHI needs its own row"
    Without it PHI reaches the catch-all default and is reported as
    `constant`. Thyra's own answer would still be `nearest_neighbor`, so
    nothing breaks inside Thyra -- but `preview_msi` hands that axis type
    to callers, and a caller that maps `constant` to profile-MALDI
    conventions arrives at `tic_preserving` onto an equidistant axis. That
    combination is destructive here: a PHI pixel holds a median of 44
    measured points across m/z 0.5--1850, and interpolating between them
    fabricates intensity in every bin in the gaps, with the TIC rescale
    hiding it behind a total that still balances. The detector reports the
    law the data actually follows -- `PhiMassAxis` steps at a constant
    flight time, so spacing goes as `sqrt(m/z)` -- which is `linear_tof`.

!!! info "`tic_preserving` is gated on the source and target axis laws matching"
    A detector may only ask for `tic_preserving` if it knows the spacing law
    of the grid the spectra *arrive* on, and that law is the one it is asking
    the target axis to use. Two routes qualify. `RapiflexReader` lays every
    spectrum out with `np.linspace`, so the source is uniform in m/z, and the
    axis it requests is `constant` -- the same law. The Waters profile trace
    is the digitiser's own record, sampled at a fixed clock rate and so
    spaced as `sqrt(m/z)` (measured `(m/z)^0.494` on a SELECT SERIES MRT and
    `(m/z)^0.498` on a Synapt G2-Si), and the axis it requests is
    `linear_tof` -- again the same law.

    Anything else is refused and gets `nearest_neighbor` instead, with a log
    line saying so. This is the rule SCiLS Lab applies: TIC-preserving
    resampling "if all axis types are identical", otherwise interpolation
    (2026b User Guide, p.80). It is also exactly when Thyra's
    interpolate-then-rescale operator is exact -- see the `!!! danger` box
    below for what it costs off the diagonal.

!!! info "How an imzML reaches the Orbitrap and FT-ICR rows"
    Both detectors match on an `instrument_type` of `"Orbitrap"` or
    `"FT-ICR"`, which the imzML metadata extractor resolves from the file's
    own instrumentConfiguration, in declaration-strength order: the
    `<analyzer>` component cvParam (`MS:1000484` orbitrap, `MS:1000079`
    FT-ICR) first, then a recognised instrument-model term (the solariX,
    apex, LTQ FT, and Orbitrap/Exactive/Exploris families), then a
    family-identifying product name in free-form model text. A file that
    declares none of the three stays untyped and falls through to the
    generic rows below -- an unstated analyzer stays unstated.

    The native solariX `.d` route needs none of this: its extractor stamps
    `instrument_type = "FT-ICR"` directly from the instrument identity in
    `peaks.sqlite`, so the same FT-ICR row is reached without an export
    step (see [Supported Formats](supported-formats.md#bruker-solarix)).

    One known gap: SCiLS Lab-lineage imzML exports of timsTOF data stamp the
    generic `MS:1001534 Bruker Daltonics flex series` term rather than a
    timsTOF model term. That term genuinely covers the axial flex series
    too, so Thyra does not attribute it to a timsTOF; a centroid export
    still lands on the right pair via the centroid row, but a profile export
    falls to the default. `--mass-axis-type` remains the explicit override
    for files whose metadata is silent or too generic.

!!! note "`tic_preserving` is selected for profile MALDI-TOF, not for high resolution"
    It is easy to assume the high-resolution analysers get the more elaborate
    method. They do not. `tic_preserving` is chosen for **profile** data, where
    a peak is spread over many points and rebinning would otherwise change the
    total ion count. Orbitrap and FT-ICR data is normally centroided, so it gets
    `nearest_neighbor`, which is the correct choice for discrete peaks. If you
    have *profile* Orbitrap or FT-ICR data, set
    `--resample-method tic_preserving` yourself.

!!! danger "Do not combine `tic_preserving` with a non-uniform axis type"
    `tic_preserving` interpolates onto the target axis and then applies a
    single scaling factor to the whole spectrum. A single factor cannot
    account for bin widths that vary across the mass range, so pairing it with
    `linear_tof`, `reflector_tof`, `orbitrap` or `fticr` suppresses high-m/z
    ions relative to low-m/z ones. Measured across 300-1100 m/z, two ions of
    equal abundance come back with their ratio distorted by 1.9x on
    `linear_tof`, 3.7x on `reflector_tof`, 7.0x on `orbitrap` and 13.4x on
    `fticr`.

    Auto-selection cannot produce these pairings. `tic_preserving` is only
    ever chosen alongside an axis whose law the source grid itself follows --
    `constant` for the Rapiflex, `linear_tof` for the Waters profile trace --
    which is what makes it exact; the detector chain enforces that. You have
    to ask for a mismatched combination with two explicit flags, and Thyra
    takes you at your word.

    If you want a non-uniform axis, use `nearest_neighbor`, which moves each
    peak into a single bin and is unaffected by bin width.

The chosen detector, method, and axis type are all logged:

```
INFO - Detected instrument type: timsTOF
INFO - Selected resampling method: NEAREST_NEIGHBOR
INFO - Selected axis type: REFLECTOR_TOF
```

### Overriding the detector

`--resample-method` overrules the detector, and the detector is still asked
what it would have picked. When the two disagree, the conversion says so and
carries on:

```
WARNING - Resampling method TIC_PRESERVING was given explicitly, but this
source's detector chose NEAREST_NEIGHBOR for it. Interpolating a source the
detector reads as sparse fills the whole axis: [...] Pass
--resample-gap-tolerance to discard bins no measured m/z vouches for, or drop
the override.
```

Nothing about the output changes -- the method you asked for is the method
used. The warning exists because the failure it points at is invisible in the
totals: on a 713-frame PASEF acquisition, `--resample-method tic_preserving`
stored **423,386,757** non-zeros against 302,106 for the detector's choice, a
583 MB table against 7.6 MB, and per-pixel TIC identical to the last digit.
Adding `--resample-gap-tolerance 0.01` to the same command brings it down to
4,801,946. See [Gaps in the source m/z values](#gaps-in-the-source-mz-values).

---

## Methods

### `nearest_neighbor`

Each target bin takes the intensity of the nearest original m/z value. Peaks
stay sharp and stay put, and no intensity is invented between them. This is the
right choice for **centroid** data, where a peak is a single discrete mass and
smearing it across neighbouring bins would be wrong.

Consequence worth knowing: on a target axis finer than the source spacing, most
bins receive nothing. A dataset resampled from 4,000 source points onto 190,000
bins has exactly 4,000 populated bins per spectrum, and the rest are zero. This
is why you should extract ion images by **summing over an m/z window** rather
than picking the single nearest bin -- see
[the tutorial](tutorial.md#step-7-ion-images).

### `tic_preserving`

Linear interpolation onto the target axis, followed by rescaling so the
spectrum's total ion current matches the original. This is the right choice for
**profile** data: a peak spans many points, interpolation reconstructs its shape
on the new grid, and the rescaling step stops rebinning from quietly changing
quantitation.

Use it whenever the total ion count per pixel has to stay comparable before and
after conversion.

The total that is preserved is the share of the spectrum inside the axis range.
If you crop with `--resample-min-mz` or `--resample-max-mz`, intensity outside
the window is dropped rather than redistributed over the bins you kept, so the
per-pixel total falls by whatever you cropped out -- a cropped window should not
claim ions from the parts that were cut away. That share is measured by area, so
a window narrower than the spacing between source points still keeps a
proportionate amount instead of collapsing to zero.

With the default axis, which spans the dataset's own mass range, nothing is
dropped and the total is preserved exactly.

#### Gaps in the source m/z values

Interpolation has no notion of a gap. Between the last source point before an
empty stretch and the first one after it, `np.interp` draws a straight line,
and every target bin in between gets an intensity that was never measured. On
sparse or thresholded m/z arrays that is most of the axis: forcing
`tic_preserving` onto `bellini.imzML` -- 2,222 points per spectrum, median
source spacing 0.0072 Da, largest gap 1,269 Da -- puts **80.6%** of the output
total ion current more than 0.5 Da from any measured point.

`--resample-gap-tolerance` sets how far a target bin may sit from the nearest
source m/z before its interpolated value is discarded instead of trusted. It is
the same parameter Cardinal calls `tolerance` and matter calls `tol`.

```bash
thyra data.imzML out.zarr \
  --resample-method tic_preserving \
  --resample-gap-tolerance 0.1
```

Masking happens **before** the TIC rescale, so the intensity returns to the
bins that do have measurements behind them rather than being deleted; the
per-pixel total is still preserved.

Sizing it: on a source grid of uniform step `s`, no target bin is ever more
than `s/2` from a source point, so **any tolerance above half the widest gap in
your source m/z values changes nothing**. Continuous profile data is dense and
evenly sampled, which is why the default is *no* limit -- there is nothing to
refuse. Set it when your m/z arrays are sparse or peak-picked and you are
asking for `tic_preserving` anyway.

`nearest_neighbor` needs no such parameter: it only ever fills a bin some peak
was snapped into.

---

## What "in range" means

Both methods keep a peak when it lies inside the **declared** mass range --
`--resample-min-mz` and `--resample-max-mz`, or the source's own range when you
set neither. Peaks outside it are **discarded**, not folded into the first or
last bin.

The declared range is not quite the same as the span of the axis points. Every
physics axis type lays its bins as `target_bins + 1` edges across the range and
stores the **centres**, so the first and last centre sit half a bin inside the
range you asked for: a source declaring 50-1000 m/z builds an axis running
`50.0001` to `999.9975`. A peak at exactly 50.0 belongs in the first bin, and
until v3.24.0 it was thrown away instead. That mattered most for sources whose
declared range *is* their first and last sample -- PHI ToF-SIMS takes its mass
range from the first and last detector channel, so both channels were lost in
every pixel.

`constant` axes are unaffected either way: they are laid out with
`np.linspace(min_mz, max_mz, n)`, whose end points already are the declared
bounds. `--no-resample` is unaffected too -- the axis there is the source's own
values.

Dropping is deliberate and is reported once per conversion:

```
WARNING - Dropping peaks that fall outside the target mass range
[250.0000, 1200.0000] m/z -- 2 of 20 in the first spectrum affected. They are
discarded, not folded into the edge bins. Widen the resampling range to keep
them.
```

Folding them in instead would be worse, and used to happen: clamping every
out-of-range peak onto the nearest edge bin put 654,158 counts in bin 0 of
`pea.imzML` cropped to 400-800 m/z, where a real peak there is around 80. The
total was conserved exactly, so no TIC check could see it.

---

## Axis types

The axis type sets how bin width grows with m/z. Each corresponds to the
physics of a mass analyser, so that bins track the instrument's real resolving
power instead of over-sampling the low end and under-sampling the high end.

| Axis type | In SCiLS Lab | Bin width scales as | Rationale |
|---|---|---|---|
| `constant` | Constant (equidistant) | constant Da | Equidistant bins |
| `linear_tof` | **Axial TOF** | `sqrt(m/z)` | Linear TOF: flight time `t ∝ sqrt(m/z)`, so equal time bins give `sqrt(m/z)` mass bins |
| `reflector_tof` | **Orthogonal TOF** | `m/z` | Constant *relative* resolution `R = m/Δm`; bins are uniform in `ln(m/z)` |
| `tof` | -- | `sqrt(A m + B m^2)` | A measured TOF peak width; `linear_tof` and `reflector_tof` are its two limits. See [The two-term TOF law](#the-two-term-tof-law) |
| `orbitrap` | Orbitrap | `m/z^1.5` | Orbitrap frequency `f ∝ 1/sqrt(m/z)`, so equal frequency bins give `m/z^1.5` mass bins |
| `fticr` | **MRMS** (Fourier-transform) | `m/z^2` | Cyclotron frequency `f ∝ 1/(m/z)`, so equal frequency bins give `m/z^2` mass bins |

The five types, and their scaling laws, are the ones SCiLS Lab offers for the
common mass axis (2026b User Guide, p.75). **Thyra's names are SCiLS's older
ones**: `linear_tof` and `reflector_tof` were "Linear TOF" and "Reflector TOF"
in earlier SCiLS versions and are now Axial TOF and Orthogonal TOF, and the
FT-ICR type is now called MRMS. The laws are unchanged; only the labels moved.

`reflector_tof` is the most broadly useful of these: constant relative
resolution means constant relative mass accuracy across the whole range, which
is what most MS workflows assume.

### The two-term TOF law

A time-of-flight peak's width in flight time has a constant part (detector
and digitiser response, pusher timing) and a part proportional to the flight
time (energy spread, turnaround time), added in quadrature. In m/z:

```
FWHM(m) = sqrt(A * m + B * m^2)        m in Da, FWHM in mDa
```

`A` is in mDa<sup>2</sup>/Da and `B` is dimensionless. `linear_tof` is the
`B = 0` limit (width grows as `sqrt(m)`), `reflector_tof` the `A = 0` limit
(width grows as `m`, constant ppm, with `1/sqrt(B)` the resolving power). Real
instruments sit between, and where they sit is measurable from a few hundred
isolated peaks by least squares on `FWHM^2 = A m + B m^2`:

| Instrument | `A` | `B` | `1/sqrt(B)` | Fitted from |
|---|---|---|---|---|
| SELECT SERIES MRT | 0.0185 | 9.1e-6 | 331,000 | 229 peaks, m/z 300-1000 (R<sup>2</sup> 0.36 on FWHM<sup>2</sup>: the peaks scatter, the trend does not) |
| timsTOF fleX | 0.0877 | 8.74e-4 | 34,000 | 180 peaks, m/z 300-1000 (R<sup>2</sup> 0.89) |

The MRT pair reproduces the measured 2.97 / 3.79 / 4.54 mDa at m/z 400 / 600 /
800; a log-log fit of the same peaks gives an exponent of 0.67, between the
0.5 and 1.0 the two single-term laws allow, which is why neither of them fits
an MRT centroid list exactly. The timsTOF pair is within 10% of the
`reflector_tof` shape over m/z 400-1000 (7% at 400, 3% at 600; the constant
term shows below that and the gap reaches 10% at m/z 300), so nothing changes
for timsTOF by default and the pair is opt-in.

The axis lays bins at `FWHM(m) / k` for `k` bins per peak width (default 3).
The cumulative bin count has a closed form, `(2/sqrt(B)) asinh(sqrt(B m / A))`,
so the axis is a uniform grid in that variable and the count is exact.

```bash
# An MRT centroid conversion: this is the default, no flags needed
thyra mrt_run.raw out.zarr --waters-spectrum centroid

# A timsTOF opting in to its measured pair (the default stays reflector_tof)
thyra run.d out.zarr --mass-axis-type tof

# A TOF instrument Thyra has no pair for, fitted from its own peaks
thyra run.imzML out.zarr --mass-axis-type tof --tof-law 0.0185 9.1e-6

# Finer bins: the width at the reference m/z sets k, as on any other axis
thyra run.d out.zarr --mass-axis-type tof \
    --resample-width-at-mz 0.005 --resample-reference-mz 1000
```

`--mass-axis-type tof` takes the pair the detected instrument declares (MRT
centroid, timsTOF); `--tof-law A B` supplies one for an instrument that has
none, and it is an error to have neither. There is no separate flag for the
bins per peak width: `--resample-width-at-mz` at `--resample-reference-mz`
fixes it, since `k` is whatever puts a bin of that width at the reference m/z.
The Python API's `ResamplingConfig` also accepts `bins_per_fwhm` directly.

**Centroid axes follow peak width; profile axes follow the sample grid.** The
law describes how wide a peak is, which is what the bins of a *centroid* list
should track. A *profile* trace is a set of samples on the digitiser's own
grid, and its bins should track that grid instead -- the Waters profile
default uses `linear_tof` because that is how its samples are spaced (see
[Supported Formats](supported-formats.md#waters-masslynx)), not because of how
wide its peaks are. The two-term law is never applied to profile data, nor to
FT-ICR or Orbitrap data, whose widths are not time-of-flight quantities.

---

## Bin count

You can set the bin count directly, or specify a target bin **width at a
reference m/z** and let Thyra derive the count from the axis physics. The two
are mutually exclusive.

When you specify neither, the detected instrument may declare a width; failing
that, the axis type's default applies:

| Source | Default width | At reference m/z |
|---|---|---|
| Waters SELECT SERIES MRT vendor centroid | `tof` law at 3 bins per FWHM (1.75 mDa) | 1000 |
| Waters vendor centroid, other instruments | 2 mDa | 1000 |
| Waters SELECT SERIES MRT profile trace | 1.3 mDa | 1000 |
| Waters profile trace, other instruments | 1.14 x the run's predicted sample spacing | 1000 |
| any other source, `linear_tof` | 17 mDa | 300 |
| any other source, everything else | 5 mDa | 1000 |

The Waters figures are measured, not conventional. 2 mDa is the coarsest
centroid setting that clears two bins per measured peak width at m/z 800 on an
MRT (2.8 bins per FWHM of 4.48 mDa; the earlier 5 mDa gave 1.1). 1.3 mDa is
about 1.14 times the MRT digitiser's sample spacing at m/z 1000, pinned rather
than derived so every MRT run with the same mass range shares one axis; see
[Supported Formats](supported-formats.md#waters-masslynx). An instrument's
width applies only on the fully automatic path -- set `--mass-axis-type` and
the width defaults revert to the axis type's, since a width tuned for one law
is not a sensible default for another.

The 17 mDa / m/z 300 pairing for `linear_tof` was chosen to be close to the
axis SCiLS Lab produces for FlexImaging data. Treat it as a working default
rather than a reproduction: **the figure does not appear in the SCiLS Lab
2026b User Guide**, and no version that states it has been identified, so
nothing here is a claim of exact equivalence. The code comment implementing it
says "SCiLS-like ~17 mDa", which is the accurate description. If you need to
match a particular SCiLS project, set `--resample-width-at-mz` and
`--resample-reference-mz` from that project's own axis.

The count is then derived per axis type:

| Axis type | Bin count |
|---|---|
| `reflector_tof` | `ln(max/min) * (ref_mz / width)` |
| `linear_tof` | `(2/k) * (sqrt(max) - sqrt(min))`, `k = width / sqrt(ref_mz)` |
| `orbitrap` | `2 * (1/sqrt(min) - 1/sqrt(max)) * (ref_mz^1.5 / width)` |
| `fticr` | `(1/min - 1/max) * (ref_mz^2 / width)` |
| `constant` | `(max - min) / width` |

Each is the integral of `1 / width(m)` across the mass range, so the bin width
the axis actually realizes at your reference m/z is the width you asked for.
The result is floored at 100 bins.

A worked example, from the [tutorial](tutorial.md)'s synthetic dataset --
250-1200 Da, `constant` axis, default 5 mDa width:

```
(1200 - 250) / 0.005 = 190000 bins
```

which is what the log reports:

```
INFO - Calculating bins for 5.0 mDa width at m/z 1000.0
INFO - Calculated 190000 bins for constant axis type
INFO - Building resampled mass axis: 250.00 - 1200.00 m/z, 190000 bins
```

190,000 bins from 4,000 source points sounds extravagant, and on the
`nearest_neighbor` path -- which is what `auto` picks for this dataset -- it
very nearly is free. Each source point lands in exactly one bin, so the
non-zero count does not depend on the bin width at all and only the index
arrays lengthen: the tutorial store comes out at roughly 31 MB, and going from
4,000 bins to 190,000 costs about +0.8% on disk.

!!! warning "That only holds for `nearest_neighbor`, and for zero-suppressed profiles"
    `tic_preserving` interpolates, so on a continuous profile it populates
    essentially every bin. The matrix comes out 99.998% dense and the store
    becomes `n_pixels x target_bins x ~7.5 bytes` -- independent of how sparse
    the source was. Measured on this same 4,000-point source: at 190,000 bins
    it costs 1.44 MB per pixel, so 400 pixels take 574 MB against 6.93 MB at
    4,000 bins, an 82.9x blowup, and the 1,728-pixel tutorial dataset would
    take roughly 2.5 GB.

    The exception is a source that stores explicit zeros around its peaks and
    nothing in between, like the Waters profile trace. Interpolation between
    two zeros is zero, so only the bins under the stored clusters are
    populated -- and only those are evaluated, which is what keeps the MRT
    default's conversion time in line with nearest-neighbour binning.

    On that path the default bin count is a decision rather than a detail.

Pass `--resample-bins` or `--no-resample` if you would rather not upsample.

### The 10 million bin cap

`--no-resample` on a processed-mode imzML does not build a physics axis at all.
It builds the **raw** axis: every distinct m/z value in the whole dataset, one
column each. When the peak lists genuinely share m/z values that is small. When
they do not -- which is the normal case for centroided peak picking, where two
pixels almost never report a peak at the same double -- it grows to roughly one
column per peak in the entire dataset, and there is no natural stopping point.

Thyra gives up once that axis passes **10 million** unique m/z values, and says
what to do instead:

```
ValueError: Common mass axis exceeded 10,000,000 unique m/z values after
12,400 of 918,855 spectra (10,004,112 so far). The peak lists in this dataset
do not share m/z values, so a raw axis grows to roughly one column per peak,
which is not usable downstream. Convert with resampling instead (it is the
default; --no-resample disables it), or raise max_mass_axis_length.
```

10 million is SCiLS Lab's own limit on the same quantity: "Data sets in SCiLS
Lab are limited to a maximum of 10 million bins on the common mass axis" (2026b
User Guide, p.76).

The cap is Python-API only, through `reader_options`, and applies to the imzML
reader:

```python
from thyra import convert_msi

convert_msi(
    "data.imzML",
    "out.zarr",
    reader_options={"max_mass_axis_length": 50_000_000},  # or None for no cap
)
```

It has no effect when resampling is on, which is the default: a resampled axis
has the bin count you asked for.

---

## Overriding the automatic choice

Every part is independently overridable; anything you leave alone stays
automatic.

=== "CLI"

    ```bash
    # Method and axis type. nearest_neighbor is the pairing to use with a
    # non-uniform axis -- see the warning under "Selection" above.
    thyra input.imzML output.zarr \
        --resample-method nearest_neighbor \
        --mass-axis-type orbitrap

    # Fixed bin count
    thyra input.imzML output.zarr --resample-bins 50000

    # Target width at a reference m/z instead of a bin count
    thyra input.imzML output.zarr \
        --resample-width-at-mz 0.01 \
        --resample-reference-mz 500

    # Restrict the mass range
    thyra input.imzML output.zarr \
        --resample-min-mz 400 --resample-max-mz 1000

    # Off entirely
    thyra input.imzML output.zarr --no-resample
    ```

=== "Python"

    ```python
    from thyra import convert_msi

    convert_msi(
        "input.imzML",
        "output.zarr",
        resampling_config={
            "method": "nearest_neighbor",
            "axis_type": "orbitrap",
            "width_at_mz": 0.01,
            "reference_mz": 500.0,
        },
    )
    ```

!!! warning "The Python API does not resample by default"
    `convert_msi()` called without `resampling_config` keeps the original mass
    axis, whereas the `thyra` command-line tool resamples by default. Pass a
    `resampling_config` explicitly if you want the CLI's behaviour from Python.

---

## Checking what was used

The decision is recorded in the output store, so a converted dataset is
self-describing:

```python
import spatialdata as sd

sdata = sd.read_zarr("output.zarr")
table = sdata.tables["msi_dataset_z0"]

mz = table.var["mz"].values
print(f"{len(mz)} bins, {mz.min():.2f} -- {mz.max():.2f} m/z")

# Bin width across the range tells you the axis type in practice:
# flat -> constant, growing linearly -> reflector_tof, and so on.
import numpy as np
d = np.diff(mz)
print(f"bin width: {d.min()*1000:.3f} -- {d.max()*1000:.3f} mDa")
```

For the tutorial dataset this prints a flat 5.000 mDa across the range,
confirming a `constant` axis at the default width.

---

## Practical guidance

- **Leave it alone** unless you have a reason. The detected settings are right for the common Bruker and imzML cases.
- **Disable it** (`--no-resample`) when you want the untouched vendor axis -- your own peak picking, centroiding, or calibration downstream.
- **Set `tic_preserving`** if quantitation matters and your data is profile, especially profile data from a high-resolution analyser, which the automatic choice will not pick it for.
- **Set `--resample-bins`** any time you need to match another tool's axis exactly.
- **Narrow the mass range** (`--resample-min-mz` / `--resample-max-mz`) before increasing bin counts; it is usually the cheaper way to get the resolution you need where you need it.

---

## See also

- **[CLI Reference](cli.md#resampling-advanced)** -- the flags
- **[Tutorial](tutorial.md)** -- resampling in the context of a full conversion
- **[Output Format](output-format.md)** -- where the m/z axis lives in the store
- **[API Reference](api.md)** -- `ResamplingConfig`, `ResamplingDecisionTree`
