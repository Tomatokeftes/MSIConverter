# Coordinate Systems

Every SpatialData zarr Thyra writes carries a small but crucial promise:
**all elements within the zarr resolve to the same physical frame at the
``"global"`` coordinate system, and that frame is documented in the
zarr metadata.** This page explains the contract so consumers (Ousia,
registration tooling, custom analysis scripts) can render and compute
without guessing what the numbers mean.

---

## The contract in one sentence

In every Thyra-produced zarr,

> the ``"global"`` coordinate system is one self-consistent frame, its
> unit and pixel-size calibration are recorded under
> ``zarr.attrs["coordinate_systems"]["global"]``, and every element
> registers a transform that lands in that frame.

If you only remember one thing, remember that.

One thing the contract deliberately does **not** cover: ion mobility. A
TIMS acquisition has a third measured dimension, but it is a coordinate on a
*feature* -- like m/z, it says what was measured, not where. Thyra never
writes mobility into a coordinate system, a transform, `obs`, or a z axis; it
lives in `var` (on a mobility-resolved table) and in the summed table's
`uns["mobility_axis"]` / `uns["mobility_heatmap"]`, see
[Output Format](output-format.md#ion-mobility). Anything that reads
`coordinate_systems` can ignore mobility entirely.

---

## The schema

At the zarr top level Thyra writes:

```python
zarr.attrs["coordinate_systems"] = {
    "global": {
        "unit": "micrometer" | "pixel",
        "pixel_size_um_x": float | None,
        "pixel_size_um_y": float | None,
        "reference_element": str | None,
        "convention_version": 1,
        "produced_by": "thyra/<version>",
        "raster_to_global_affine": [[a, b, tx], [c, d, ty], [0, 0, 1]],

        # Only when the reader reports raw coordinate offsets
        "coordinate_offsets_px": [x, y, z],
        "stage_offset_um": [x_um, y_um],  # micrometer variant only

        # Multi-slice volumes only -- see "The z axis" below
        "z_spacing_um": float,
        "z_spacing_source": "manual" | "automatic" | "assumed_isotropic",
    }
}
```

| Field | Meaning |
|-------|---------|
| ``unit`` | Unit of one step in ``"global"`` -- either ``"micrometer"`` or ``"pixel"``. |
| ``pixel_size_um_x``, ``pixel_size_um_y`` | Conversion factor: micrometers per one ``"global"`` unit. ``1.0`` when ``unit="micrometer"``; physical pixel size when ``unit="pixel"``. ``None`` if the producer cannot calibrate (e.g. an uncalibrated optical photo). |
| ``reference_element`` | Name of the canonical raster element that defines pixel space, when ``unit="pixel"``. ``None`` otherwise. |
| ``convention_version`` | Schema version; bump when the shape of this attr changes. Currently ``1``. |
| ``produced_by`` | ``"thyra/<version>"`` for Thyra-produced zarrs. |
| ``raster_to_global_affine`` | Explicit 3x3 row-major affine from TIC raster indices to ``"global"`` -- the same mapping the TIC element's transform expresses, duplicated here so a consumer that reads only attrs still gets the full placement. A pure pixel-size scale in the micrometer variant; the optical alignment matrix in the pixel variant. Purely additive (``convention_version`` stays 1, same reasoning as the z fields below). |
| ``coordinate_offsets_px`` | The source's raw acquisition-index offsets ``[x, y, z]``, which 0-based normalisation otherwise erases. **Only present when the reader reports them.** |
| ``stage_offset_um`` | ``coordinate_offsets_px`` times the pixel size: where the raster origin sat, in micrometers, relative to the source's index origin. **Only written when ``unit="micrometer"``**, so it cannot be misread in the optical-pixel variant. |
| ``z_spacing_um`` | Micrometers between consecutive slices. **Only present on multi-slice volumes.** Always an absolute micrometer distance, even when ``unit="pixel"``: the optical affine governs only x and y, while z is always scaled directly. |
| ``z_spacing_source`` | Where ``z_spacing_um`` came from. **Only present on multi-slice volumes.** |

---

## The z axis

A 2D store has no slice-to-slice distance, so it carries neither z field. Their
**absence is the signal** that there is no z axis — which is also why adding
them did not move ``convention_version``: they are purely additive, and a
consumer that never reads volumes sees exactly the schema it already knew.

``z_spacing_source`` is the field that matters:

| Value | Meaning |
|-------|---------|
| ``"manual"`` | The caller supplied it (``--z-spacing`` / ``z_spacing_um=``). Trust it. |
| ``"automatic"`` | The source metadata reported it. Trust it. |
| ``"assumed_isotropic"`` | **Nobody supplied one.** The in-plane pixel pitch was reused, which asserts that slices sit exactly one pixel width apart. Treat the depth as unknown. |

Section thickness is set by the microtome that cut the sections; the in-plane
pitch is set by the stage that rastered them. The two agree only by coincidence,
and for 3D MSI usually do not — often by an order of magnitude. A volume whose
``z_spacing_source`` is ``"assumed_isotropic"`` has correct voxel *values* and an
unreliable *depth*, so rendering it in micrometers stretches or squashes the
stack along z.

Thyra cannot detect this: imzML has no standard term for slice spacing, and 3D
acquisitions are frequently non-consecutive sections, so the value has to be
supplied by whoever prepared them.

Consumers that don't know the schema can still open the zarr; they
just won't be able to render distances in micrometers without
making assumptions. Consumers that do know it (Ousia, the EscDat
registration script, this project's tests) read this attr first and
trust it.

---

## What ``"global"`` is in each Thyra mode

Thyra operates in two modes depending on whether the input came with
FlexImaging optical alignment data. The two modes pick different
``"global"`` conventions because they have different *canonical
references*.

### Mode A: standalone (no optical alignment)

When there is no optical photo to align against, the only naturally
meaningful frame is **physical micrometers of the imaged tissue**.

- TIC / ion images: stored intrinsically in raster pixel indices,
  with a transform ``Scale([pixel_size_x_um, pixel_size_y_um])`` to
  ``"global"``.
- Pixel-polygon shapes: stored intrinsically in micrometers
  (``spatial_x = x * pixel_size_x_um``, ``spatial_y = y *
  pixel_size_y_um``), with ``Identity`` to ``"global"``.
- ``zarr.attrs["coordinate_systems"]["global"]`` declares
  ``unit="micrometer"`` and fills ``pixel_size_um_x/y`` with the
  MSI grid pixel size.

The two pitches are equal on a square raster and are *not* assumed to be:
an anisotropic acquisition (a DESI method with ``DesiXStep !=
DesiYStep``) scales each axis by its own, here and in every other block
of the store that states a pitch.

Both elements resolve to the same micrometer extent at ``"global"``.

For a **multi-slice volume** (``--handle-3d`` over more than one plane) the
promise holds in x and y, and is **deliberately silent in z for the shapes
element**. The TIC volume carries the depth on its ``Scale``, and every table
row carries it in ``obs["spatial_z"]``; the pixel polygons stay
two-dimensional and make no claim about depth at all.

So a volume's elements agree like this:

| Element | x, y at ``"global"`` | z at ``"global"`` |
|---------|---------------------|-------------------|
| TIC volume | ``Scale([pixel_size_x_um, pixel_size_y_um])`` | ``Scale([z_spacing_um])`` |
| Table | ``spatial_x``, ``spatial_y`` | ``spatial_z`` |
| Pixel shapes | micrometres, ``Identity`` | **absent** — join via ``spatial_z`` |

!!! note "Why the footprints carry no depth"
    They briefly did. v3.2.0 made them ``POLYGON Z`` at the depth of their
    slice, which is the geometrically honest representation, and it broke
    ``spatialdata``'s spatial queries: a bounding box enclosing an entire test
    volume returned 26 of 30 footprints and 26 of 30 table rows, silently. A
    z-restricted query returned the same rows whether z was inside or far
    outside the data.

    Upstream asks for 2D here. ``ShapesModel.validate`` warns that a
    3-dimensional geometry column "could led to unexpected behaviors" and
    names ``force_2d()`` as the remedy — the query result above is that
    behaviour. 3D shapes are not on the spatialdata roadmap
    ([#109](https://github.com/scverse/spatialdata/issues/109), idle since
    June 2023, covers images, labels and transformations), and the live 2.5D
    discussion ([#961](https://github.com/scverse/spatialdata/issues/961))
    scopes itself to points, images and labels. Serial-section MSI is 2.5D in
    that taxonomy.

    Expressing the depth as a ``Translation`` on flat geometry — the obvious
    remaining alternative — does **not** work either: the transform silently
    drops z, leaving a depth that exists only in metadata.

    Both negative results are pinned by
    ``tests/unit/converters/test_3d_pixel_shapes_z.py``. If either starts
    passing, upstream has changed and this decision should be reopened.

### Mode B: with FlexImaging optical alignment

When the input includes a ``.mis`` ``ImageFile`` and registration
landmarks, Thyra picks a different canonical reference: **the
optical photo itself**. The natural frame is then *that image's
pixel grid*, because the photo can be drawn with no transform.

- TIC / ion images: stored intrinsically in raster pixel indices,
  with an ``Affine`` transform mapping into the optical pixel
  grid.
- Pixel-polygon shapes: stored directly in optical pixels, with
  ``Identity`` to ``"global"``.
- Primary optical image: ``Identity`` to ``"global"``.
- Other optical images (overview, etc.): a ``Scale`` to align with
  the primary image's pixel grid.
- ``zarr.attrs["coordinate_systems"]["global"]`` declares
  ``unit="pixel"`` with ``reference_element`` set to the primary
  optical filename; ``pixel_size_um_x/y`` is typically ``None``
  because FlexImaging photos do not generally carry a um-per-pixel
  calibration.

Note that the two modes pick the right convention for what is
actually known about the data; consumers should look at
``unit`` rather than assuming Thyra always uses one or the other.

---

## Reading the coordinate-system metadata

```python
import json
from pathlib import Path

zarr_path = Path("output.zarr")
with open(zarr_path / "zarr.json") as f:
    attrs = json.load(f)["attributes"]

cs = attrs["coordinate_systems"]["global"]
print(f"global unit:        {cs['unit']}")
print(f"pixel size (um):    {cs['pixel_size_um_x']} x {cs['pixel_size_um_y']}")
print(f"reference element:  {cs['reference_element']}")
print(f"schema version:     {cs['convention_version']}")
```

To resolve a ``"global"`` coordinate to micrometers, multiply by the
``pixel_size_um_x/y`` factors. When ``unit="micrometer"`` those
factors are ``1.0`` and the multiplication is a no-op; when
``unit="pixel"`` they perform the px-to-um conversion.

---

## Why two conventions?

Picking ``unit="micrometer"`` for everything would be simpler, but
also wrong:

- In **Mode B** there is a canonical raster image (the optical
  photo). Putting ``"global"`` in micrometers would force that
  image to carry a transform, since its data is intrinsically in
  pixels. With ``unit="pixel"``, the canonical image carries
  ``Identity`` and downstream tools can blit it without thinking.
- In **Mode A** there is no canonical raster image - the MSI
  itself is the data. There's no privileged pixel grid to use as
  ``"global"``, so ``unit="micrometer"`` is the natural and
  unambiguous choice.

The rule generalises: **``"global"`` is whichever frame is most
natural for that dataset's canonical reference, and the metadata
declares which one was picked.**

This matches how the broader spatial-omics ecosystem works:
``spatialdata-io``'s Xenium reader, for example, uses
``unit="pixel"`` because Xenium has a canonical morphology image;
its MERFISH-style readers use ``unit="micrometer"`` because they
do not.

---

## Verifying the contract

Because the contract is just metadata, it can drift if a producer
has a bug. Every Thyra release has a unit-test guard that:

1. Runs a representative conversion end-to-end.
2. Reads the resulting zarr.
3. Asserts the bbox of the TIC image and the pixel-polygon shapes
   resolve to the same ``"global"`` extent within a half-pixel
   tolerance.
4. Asserts the ``coordinate_systems.global`` attr is present and
   self-consistent.

See ``tests/unit/converters/test_coordinate_systems.py``.

A consumer can run the same check at load time. The recommended
pattern is to compute the bbox at ``"global"`` for every element
and warn if any pair differs by more than ~10x; that catches the
``Scale(1/pixel_size)`` vs identity unit-confusion class of bug
without flagging legitimate small-vs-large element pairs (e.g.
a small ROI within a full-tissue dataset).

---

## Cross-modality registration

When a Thyra zarr is later registered against another modality
(e.g. Xenium via the EscDat registration pipeline), the
registration step rewrites every Thyra element's transform to
``"global"`` so that the merged zarr's ``"global"`` matches the
other modality's convention. The merged zarr's
``coordinate_systems.global`` attr is rewritten to declare the new
unit and reference element.

That is, **the coordinate-system contract is per-zarr, not
per-element-or-modality**. A standalone Thyra zarr can have
``unit="micrometer"`` while a merged Thyra+Xenium zarr produced
from the same data will have ``unit="pixel"``. Both are correct
for their respective zarrs.

---

## What if I open a zarr that doesn't have this attr?

Two kinds of Thyra output lack the ``coordinate_systems`` attr, and
the second is more recent than it looks:

- **Anything written before v1.22.0**, which introduced the schema.
- **Anything written between v1.22.0 and v2.3.1 by the streaming
  route**, which in those releases was one of four converters and
  hand-wrote its own Zarr root attributes. Its copy was short: 7
  attrs against the 10 the other three produced, missing
  ``coordinate_systems``, ``format_specific_metadata`` and
  ``msi_dataset_info``. Fixed in v3.0.0. This is the awkward one,
  because that route was chosen by estimated size -- so the biggest
  datasets are exactly the ones that lost it, and nothing in the
  store records which route wrote it. There is no live setting to
  check: the four converters became one in v3.23 (design decision
  D11) and every store written since v3.0.0 carries the attr.

Either way the store is still readable, but you have to infer the
convention from element layouts and accept some risk that two
elements disagree silently.

If you control the data: **regenerate**. The schema is cheap to
write and the resulting zarrs are self-describing forever after.
If you do not control the data, the best fallback is to write a
producer-specific heuristic (see Ousia's coordinate-system loader
for an example: it recognises Xenium-from-spatialdata-io zarrs by
their ``morphology_focus`` image and assumes Xenium's documented
pixel size).
