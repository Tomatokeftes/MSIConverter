# PHI ToF-SIMS Notes

Thyra reads PHI (Physical Electronics) SmartSoft-TOF `.raw` files from nanoTOF
instruments directly -- there is no vendor SDK for this format, so the layout
below was established by reverse engineering and verified against the
instrument software's own exports.

Everything on this page was measured on a 512x512 negative-mode acquisition,
148 MB, 18.4 million ion events. The verification is not incidental: PHI's
software can export a total-ion image and peak images as `.bif6` files, and
reconstructing the total ion image from Thyra's parse reproduces the vendor's
export **bit-exactly** -- 262,144 of 262,144 pixels identical.

---

## File layout

```
[0, HeaderSize)   ASCII key/value header, CRLF lines, SOFH ... EOFH,
                  zero-padded out to the size it declares itself
[HeaderSize, EOF) block-structured stream
```

The header is plain text. Top-level entries are `Key: value`; entries under a
`[Section]` marker are `Key=value`. `HeaderSize` (20000 in the reference file)
says where the binary begins.

### Blocks

Everything after the header is a chain of blocks, each introduced by a 16-byte
header:

| offset | type | meaning |
|---|---|---|
| 0 | uint16 | block id |
| 2 | 10 bytes | not interpreted |
| 12 | uint32 | record size |

| id | meaning |
|---|---|
| 1 | ion events (8 bytes each); `8` instead when MS/MS is active |
| 2 | end of frame, no payload |
| 6 | mosaic tile origin: two uint32 tile indices |
| 14 | appended ASCII info block, e.g. a post-hoc recalibration |
| 0 | end of stream |

Any other id is skipped using its record size. The chain tiles the payload
exactly, which is what makes it trustworthy: a parse that ends anywhere other
than end-of-file has gone wrong. In the reference file the walk covers 32,179
blocks and lands precisely on the final byte, and the 100 end-of-frame blocks
match the header's `NoFrames: 100`.

### Events

Each event is a little-endian uint64:

| bits | meaning |
|---|---|
| 31 | set on a valid event |
| 0-30 | flight time in picoseconds (27 bits used) |
| 32-42 | X pixel index |
| 43-53 | Y pixel index |
| 58-63 | clear on a valid event |

Records failing that test are dropped. The reference file carries 64,493 of
them among 18.5 million (0.35%), many holding the `0xADADADAD` uninitialised
-heap fill. Excluding them is what makes the reconstruction exact.

!!! warning "Do not parse the payload as a flat event array"
    Because block headers happen to be 16 bytes -- exactly two event slots --
    and fail the validity test, a naive flat parse *appears* to work on
    single-tile data. It silently collapses every mosaic tile onto the same
    grid, because it never sees the block 6 tile origins.

---

## Mass calibration

The relation is the standard time-of-flight one:

```
sqrt(m/z) = (Mass/Time) * t_us + MassOffset
```

### The header calibration may be stale

The acquisition header carries `Mass/Time` and `MassOffset`, and often
`Calibrated: no` alongside them. In the reference file those coefficients are
left over from a positive-mode session -- its `Calibration:` line lists C+H,
C₂, C₃, C₄ although the acquisition is negative-mode -- and they place known
ions about **+3 mDa** high:

| ion | theoretical | header calibration | error |
|---|---|---|---|
| CH₂⁻ | 14.01565 | 14.01851 | +2.86 mDa |
| CN⁻ | 26.00307 | 26.00676 | +3.69 mDa |
| CNO⁻ | 41.99799 | 42.00144 | +3.45 mDa |
| PO₂⁻ | 62.96343 | 62.96569 | +2.26 mDa |
| PO₃⁻ | 78.95834 | 78.96093 | +2.59 mDa |

Mean +3.02 mDa, standard deviation 0.46 -- a consistent bias, not scatter.

### The real calibration is appended to the file

A recalibration performed after acquisition is **not** written back into the
header. It is appended near the end of the file as a block 14
`AsciiInfoBlock01`:

```
AppendedBlockType: AsciiInfoBlock01
BlockAppendedDate: 07/27/2026 17:29:40
Calibration: 4  (26.003016, C+N, 26.003099)  (41.998143, C+N+O, 41.998001) ...
Mass/Time: 0.382674
MassOffset: -1.598302
```

Thyra prefers it automatically. The difference is not cosmetic -- measured
against the instrument's own peak-image exports:

| calibration | peak count recovery | mean correlation |
|---|---|---|
| header | 83.4% | 0.852 |
| appended | **99.7%** | **0.985** |

Pass `use_appended_calibration=False` to reproduce a legacy conversion that
ignored it. Whichever is used, `uns["raw_metadata"]["calibration"]` records
both sets of coefficients, which one was applied, and the appended block
verbatim.

---

## The time axis, and why it is binned

The instrument measures **flight time**; m/z is derived. Thyra therefore keeps
the flight time in `var["tof_us"]` next to `var["mz"]`, so the calibration is
reversible without re-reading the raw file:

```python
import spatialdata as sd

table = next(iter(sd.read_zarr("out.zarr").tables.values()))
cal = table.uns["raw_metadata"]["calibration"]

tof = table.var["tof_us"].to_numpy()
mz = (cal["mass_slope_used"] * tof + cal["mass_offset_used"]) ** 2   # == var["mz"]

# a corrected calibration needs only the stored times
better = (0.382674 * tof - 1.598302) ** 2
```

### What binning costs

Events are binned onto the detector's time-channel grid, whose width the header
gives as `SpecBinSize` (0.128 ns = 128 ps). Individual flight times are
recorded more finely than that -- `SpecBinIncr` is 1 ps -- and in a
mass-shift-corrected file they are scattered off the native grid entirely (only
0.77% of times are multiples of 128 ps).

So it is fair to ask what binning to 128 ps throws away. Measured across bin
widths on the reference file, against the vendor's peak images:

| bin (ps) | channels | non-zeros | axis size | peak recovery | correlation |
|---|---|---|---|---|---|
| 1 | 116,578,841 | 16,712,840 | 932.6 MB | 99.7% | 0.9846 |
| 8 | 14,572,356 | 16,712,840 | 116.6 MB | 99.7% | 0.9851 |
| 32 | 3,643,089 | 16,712,840 | 29.1 MB | 99.8% | 0.9869 |
| **128** | **910,773** | **16,712,608** | **7.3 MB** | **100.4%** | **0.9937** |
| 512 | 227,694 | 14,453,296 | 1.8 MB | 100.8% | 0.9820 |
| 2048 | 56,924 | 11,935,480 | 0.5 MB | 76.3% | -- |

Going from 1 ps to 128 ps merges **232 cells out of 16.7 million** and shrinks
the mass axis 128-fold. The data is so sparse that events essentially never
share a bin: 18.4 million events occupy 16.7 million distinct (pixel, channel)
cells regardless of how fine the bins are.

Agreement with the vendor does not merely survive the coarsening, it *improves*
-- 0.9846 to 0.9937 -- because 128 ps is the grid the instrument software
itself works on. Finer bins place events at positions that then fall
differently against peak-window edges.

Below 128 ps you pay 128x the storage for nothing. Above it you start
destroying data: at 512 ps a fifth of the cells merge, and at 2048 ps a quarter
of all counts fall outside their peak windows and some peaks vanish entirely.

The rounding itself is small against any real peak. Events sit a mean 32 ps
from their bin centre, worst case 64 ps, which at m/z 79 is 4.7 ppm or
0.37 mDa -- against a vendor integration window for PO₃⁻ that is 49.3 mDa wide.

!!! note "What is not stored"
    The output holds the binned axis, not the 18.4 million individual events.
    Per-event data -- for instance the exact arrival time of one ion, or
    re-binning at a width Thyra did not choose -- requires the original `.raw`.
    Keep it.

---

## Resampling onto a common axis

Converting a single `.raw` needs no resampling -- the reader's own axis is the
output axis. Resampling only comes up when several samples have to share one
mass axis, and PHI is an awkward case for it.

`InstrumentDetectorChain` recognises the format (via the
`format_specific["format"]` stamp) and answers **`nearest_neighbor` on a
`linear_tof` axis**. Both halves matter.

**Why `linear_tof`.** Channels are laid out at a constant flight-time step, and
m/z goes as the square of time, so the m/z spacing grows as `sqrt(m/z)` -- the
`linear_tof` law exactly. Measured on the reference file: 9.8e-5 u per channel
at m/z 1 against 5.0e-4 at m/z 26, a ratio of 5.1 where `sqrt(26)` is 5.099.

**Why not `tic_preserving`.** The method interpolates onto the target axis and
rescales the result back to the source TIC. A PHI pixel is not a profile
spectrum -- it holds only the channels that happened to fire, a median of 44
points spread across m/z 0.5--1850. `np.interp` draws a straight line between
consecutive points, so every bin in those gaps comes back with an intensity
nothing measured, and the rescale then normalises the total to the pixel's true
TIC. **The fabrication is invisible to any total-ion check**: on the reference
file the total came out at 70.3704 against a true 70.3707, while the average
spectrum went from 56% hard zeros to 0.1% -- empty regions such as m/z 200
sitting at 44% of the base peak. Use `nearest_neighbor`, or if the method is
forced, set `ResamplingConfig.gap_tolerance_da` (see
[Resampling](resampling.md) and `thyra.resampling.gaps`).

Forcing it no longer happens silently: since v3.24.0 an explicit
`--resample-method` that contradicts the detector is warned about, and the
warning names `--resample-gap-tolerance`. The detector alone was never enough,
because a detector only steers `auto`.

**The first and last channel.** `mass_range` here is literally
`(axis.mz[0], axis.mz[-1])` -- the first and last detector channel, not an
acquisition setting. A physics axis stores bin centres, which stop half a bin
short of the range they were built across, so testing membership against the
axis points discarded both channels in every pixel: the mock fixture stored 12
counts against the source's 14, on all six resampled variants, where
`--no-resample` stored all 14. Since v3.24.0 the test is the declared range, so
a peak on either bound lands in the edge bin. See
[What "in range" means](resampling.md#what-in-range-means).

### Choosing a bin width

The instrument's measured resolving power on the reference file is **R ~ 4,000**
(median), which is normal for a nanoTOF:

| ion | m/z | FWHM | R |
|---|---|---|---|
| C⁻ | 12.000 | 5.02 mDa | 2,393 |
| CN⁻ | 26.003 | 5.35 mDa | 4,861 |
| CNO⁻ | 41.998 | 8.90 mDa | 4,717 |
| PO₃⁻ | 78.958 | 17.12 mDa | 4,613 |

Note that this is resolving power, not axis spacing -- the 128 ps grid samples
each peak 11 to 20 times. A target axis only has to keep a few samples per
FWHM. A `linear_tof` axis of 0.01 Da at a reference of m/z 500 gives 189,191
bins over the full range and 2.3 to 4.3 samples per FWHM, which keeps CN⁻ and
C₂H₂⁻ -- 12.6 mDa apart, or 1.94 FWHM, genuinely resolved by this instrument --
5.5 bins apart.

!!! danger "A constant 0.1 Da axis destroys this data"
    It is roughly 19x the peak width at m/z 26, so CN⁻ and C₂H₂⁻ land in the
    same bin along with everything else between them. Constant-width axes suit
    profile MALDI-TOF; they do not suit ToF-SIMS.

## Previewing without decoding events

`preview_msi` promises that no spectra are decoded, and passes
`metadata_only=True` to the reader to keep that promise. `PhiReader` used to
swallow the kwarg, and the extractor called `get_peak_counts_per_pixel()`,
which walks every 8-byte event in the file: 0.16 s for a 16 MB acquisition with
2.02 M events, 0.14 s for a 14 MB one -- linear, so previewing a multi-gigabyte
SmartSoft file cost what converting it costs.

`PhiReader(path, metadata_only=True)` now answers from the acquisition header
and the block chain alone. Everything header-derived is still exact --
dimensions, coordinate bounds, m/z range, pixel size, and both detector
verdicts. What it cannot answer is which pixels carry an event, because that is
a property of the stream rather than the header, so `n_spectra` comes back as 0
with `n_spectra_counted=False` and `preview_msi` reports `n_pixels` as `None`.

It is reported as unknown rather than as `n_x * n_y` on purpose. Every other
reader's `n_spectra` counts spectra *present*, cheaply -- imzML the length of
its coordinate list, Bruker a SQL count -- and filling this one in with the
raster size would make the same field mean two different things. The raster is
already reported, as `grid_dims`.

A reader built for conversion is unaffected and still counts every pixel.

## Pixel size

Derived as `Raster Size (um) / ImagePixels`, which for the reference file is
512 µm over 512 pixels: **1.00 µm**, matching what the operator selected.

The header also carries `Raster Size Calibration` (1.550 here). Thyra
deliberately does **not** apply it: it trims the scan generator so the raster
achieves the requested pitch, rather than scaling the field of view.
Multiplying by it would inflate every coordinate by 55%. Override with
`pixel_size_um=` if a future instrument configuration needs it; the source is
recorded in `uns["format_specific"]["pixel_size_source"]`.

---

## Acquisition characteristics

Some properties of this format are worth knowing before analysis:

- **Extremely sparse.** 18.4 million events over 262,144 pixels and 863,670
  mass channels is 64 occupied channels per pixel on average -- 0.007% density.
  Counts are low: the mean pixel holds 70 ions, the maximum 491.
- **Frames are summed.** `NoFrames: 100` means 100 passes over the raster,
  concatenated in the stream and summed into one image. Thyra does not separate
  them; there is no per-frame counter in the event record.
- **Scatter raster.** `Raster Pattern=Scatter` means pixels are not acquired in
  raster order, which is why every event carries its own coordinates.
- **Empty pixels are possible.** The reference file has 2 pixels that recorded
  no ions at all. These are skipped by `iter_spectra` rather than emitted as
  empty spectra.

## Not yet exercised on real data

These paths are implemented from the format and covered by tests, but those
tests build **synthetic** `.raw` files. No real acquisition using them has been
through the reader, so treat them as untested rather than working:

- **Mosaic acquisitions** (block 6 tile origins). The reference file declares
  `Number Of Tiles X=8, Y=2` but has mosaic mode off and contains no block 6,
  so the grid is sized from the tile origins actually encountered, not from the
  declared counts.
- **MS/MS acquisitions**, where events move to block id 8.
- **Depth profiling / 3D.** The reference file has `NoSputCycles: 0`. Thyra
  treats PHI data as 2D.

The gap is data, not effort: each was straightforward to write and is
impossible to confirm without a file that exercises it. Everything Thyra *is*
verified on was verified the same way — against the instrument's own exports —
and the same would be done here.

!!! tip "Have one of these acquisitions?"
    Please [open an issue](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/issues)
    if you need one of these supported. A single representative file is enough
    to turn any of them from untested into verified, and a `.bif6` export
    alongside it makes the check exact. Say which acquisition mode you used and
    roughly how large the file is; the data itself does not have to be attached
    to the issue.

## Related exports

PHI's software writes `.bif6` sidecar files -- a total-ion image, and a peak
list of operator-selected mass windows. Thyra does **not** read these, because
they are derived products: the total ion image is reproducible bit-exactly from
the `.raw`, and the peak images to 99.7%. They remain useful as ground truth
when validating a change to the reader.
