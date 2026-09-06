"""Fragmentation as a property of the whole acquisition.

An MS/MS imaging run measures fragments, not intact ions, so the m/z axis
of the table Thyra writes means something different from an MS1 run's. The
axis alone does not say so: a converted MS/MS store is otherwise shaped
exactly like an MS1 one. This module holds the vocabulary the readers, the
converters and the metadata schema share so that a store can say which it
is, in the PSI-MS terms mzPeak and mzML also use.

The field names follow mzPeak's, checked against the reference archive at
HUPO-PSI/mzPeak ``502c3a4`` (``spectra_metadata_precursors.parquet``): an
``isolation_window`` of ``target`` plus a ``lower_offset`` and an
``upper_offset`` -- not a single width -- and activation as CV parameters,
``MS:1000133`` with ``MS:1000045`` collision energy in electronvolts.
``ms_level`` is ``MS:1000511`` there as here. Naming the same quantities
the same way is the point: mzPeak is the raw/archival layer and Thyra the
bridge to the analysis layer, so a term that means one thing in an archive
must not mean another in a store built from it.

What it deliberately does **not** do is change the data. A frame that
isolated several precursors is still summed into one spectrum per pixel;
the schedule recorded here is what makes that visible rather than silent.
Splitting such a frame into one spectrum per precursor is a separate,
larger piece of work -- it needs a feature axis of (precursor, fragment)
pairs, which is a data-model change, not a metadata one.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

#: PSI-MS terms for what an MS/MS acquisition records, spelled as mzPeak
#: spells them.
MS_LEVEL_ACCESSION = "MS:1000511"
SELECTED_ION_MZ_ACCESSION = "MS:1000744"
ISOLATION_WINDOW_TARGET_ACCESSION = "MS:1000827"
ISOLATION_WINDOW_LOWER_OFFSET_ACCESSION = "MS:1000828"
ISOLATION_WINDOW_UPPER_OFFSET_ACCESSION = "MS:1000829"
COLLISION_ENERGY_ACCESSION = "MS:1000045"

#: Unit of the collision energy, as mzPeak's activation parameters carry it.
COLLISION_ENERGY_UNIT_ACCESSION = "UO:0000266"
COLLISION_ENERGY_UNIT_NAME = "electronvolt"

#: Dissociation methods, as PSI-MS terms. Bruker's TDF schema records the
#: collision energy but not the method; a reader that knows the method from
#: elsewhere names it here rather than inventing a free-text string.
COLLISION_INDUCED_DISSOCIATION_ACCESSION = "MS:1000133"

DISSOCIATION_METHOD_NAMES: Dict[str, str] = {
    COLLISION_INDUCED_DISSOCIATION_ACCESSION: "collision-induced dissociation",
}


@dataclass(frozen=True)
class IsolationWindow:
    """One precursor isolation, as the source scheduled it.

    ``target`` with ``lower_offset`` and ``upper_offset`` is mzPeak's
    shape for the quadrupole selection, and mzML's before it: the window
    spans ``[target - lower_offset, target + upper_offset]``. A source
    that reports a single full width -- Bruker's ``IsolationWidth`` does
    -- is halved into two equal offsets by whoever builds this, which is
    the same reading every mzML writer for those instruments takes.

    The mobility scan range is Bruker PASEF-specific and optional
    everywhere else: it is what makes several windows fit inside one frame
    without overlapping, and therefore what a future demultiplexer would
    slice on.
    """

    target: float
    lower_offset: Optional[float] = None
    upper_offset: Optional[float] = None
    collision_energy: Optional[float] = None
    scan_begin: Optional[int] = None
    scan_end: Optional[int] = None

    @classmethod
    def from_full_width(
        cls, target: float, width: Optional[float], **kwargs: Any
    ) -> "IsolationWindow":
        """Build from a source that reports one full width, not two offsets."""
        half = None if width is None else float(width) / 2.0
        return cls(target=target, lower_offset=half, upper_offset=half, **kwargs)

    @property
    def is_mobility_resolved(self) -> bool:
        """Whether this window occupies its own slice of the mobility ramp."""
        return self.scan_begin is not None and self.scan_end is not None


@dataclass(frozen=True)
class FragmentationSchedule:
    """What a reader knows about the fragmentation of its acquisition.

    ``windows`` is the precursor schedule. It is empty for an MS1 run, one
    entry for a single-precursor MS/MS image (every pixel fragments the
    same thing), and several when the source isolates more than one
    precursor per pixel -- the PASEF MALDI case, where each window owns a
    disjoint slice of the mobility ramp.

    ``constant_across_pixels`` says whether that schedule is the same
    everywhere. A scheduled method makes it so; a data-dependent one would
    not, and a consumer must not assume a global precursor axis then.
    """

    ms_level: int
    windows: Tuple[IsolationWindow, ...] = ()
    constant_across_pixels: bool = True
    dissociation_accession: Optional[str] = None
    source: Optional[str] = None

    @property
    def is_msms(self) -> bool:
        """Whether the spectra are fragment spectra."""
        return self.ms_level > 1

    @property
    def merges_precursors(self) -> bool:
        """Whether one stored spectrum sums fragments of several precursors.

        The property that makes a summed MS/MS spectrum a chimera: more
        than one precursor was isolated per pixel, so the spectrum holds
        fragments of all of them with nothing distinguishing which came
        from which.
        """
        return len(self.windows) > 1

    def to_uns(self) -> Dict[str, Any]:
        """The schedule as a plain ``uns`` block: arrays, no colons in keys.

        Zarr writes a dict key as a directory name and a colon is not a
        legal Windows path character, so CV accessions are values here,
        never keys. The per-window quantities are parallel arrays rather
        than a list of dicts, which is also what survives the zarr writer
        without being stringified (see ``thyra.metadata.uns_compat``).
        """
        block: Dict[str, Any] = {
            "ms_level": int(self.ms_level),
            "ms_level_accession": MS_LEVEL_ACCESSION,
            "n_windows": len(self.windows),
            "constant_across_pixels": bool(self.constant_across_pixels),
            "merges_precursors": bool(self.merges_precursors),
        }
        if self.dissociation_accession is not None:
            block["dissociation_accession"] = self.dissociation_accession
            name = DISSOCIATION_METHOD_NAMES.get(self.dissociation_accession)
            if name is not None:
                block["dissociation_name"] = name
        if self.source is not None:
            block["source"] = self.source
        if not self.windows:
            return block

        block["isolation_window_target"] = np.array(
            [w.target for w in self.windows], dtype=np.float64
        )
        block["isolation_window_target_accession"] = ISOLATION_WINDOW_TARGET_ACCESSION
        if any(w.lower_offset is not None for w in self.windows):
            block["isolation_window_lower_offset"] = _optional_array(
                w.lower_offset for w in self.windows
            )
            block["isolation_window_lower_offset_accession"] = (
                ISOLATION_WINDOW_LOWER_OFFSET_ACCESSION
            )
        if any(w.upper_offset is not None for w in self.windows):
            block["isolation_window_upper_offset"] = _optional_array(
                w.upper_offset for w in self.windows
            )
            block["isolation_window_upper_offset_accession"] = (
                ISOLATION_WINDOW_UPPER_OFFSET_ACCESSION
            )
        if any(w.collision_energy is not None for w in self.windows):
            block["collision_energy"] = _optional_array(
                w.collision_energy for w in self.windows
            )
            block["collision_energy_accession"] = COLLISION_ENERGY_ACCESSION
            block["collision_energy_unit_accession"] = COLLISION_ENERGY_UNIT_ACCESSION
            block["collision_energy_unit_name"] = COLLISION_ENERGY_UNIT_NAME
        if all(w.is_mobility_resolved for w in self.windows):
            block["scan_begin"] = np.array(
                [w.scan_begin for w in self.windows], dtype=np.int64
            )
            block["scan_end"] = np.array(
                [w.scan_end for w in self.windows], dtype=np.int64
            )
        return block

    def to_extractor_report(self) -> Dict[str, Any]:
        """The shape :mod:`thyra.metadata.schema.builder` reads from ``format_specific``."""
        report: Dict[str, Any] = {
            "present": self.is_msms,
            "ms_level": int(self.ms_level),
            "constant_across_pixels": bool(self.constant_across_pixels),
        }
        if self.dissociation_accession is not None:
            report["dissociation_accession"] = self.dissociation_accession
        if self.source is not None:
            report["source"] = self.source
        report["windows"] = [
            {
                key: value
                for key, value in (
                    ("isolation_window_target", window.target),
                    ("isolation_window_lower_offset", window.lower_offset),
                    ("isolation_window_upper_offset", window.upper_offset),
                    ("collision_energy", window.collision_energy),
                    ("scan_begin", window.scan_begin),
                    ("scan_end", window.scan_end),
                )
                if value is not None
            }
            for window in self.windows
        ]
        return report


def _optional_array(values: Any) -> Any:
    """Per-window floats as an array, with the unreported ones as NaN."""
    return np.array(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float64,
    )


def windows_overlap(windows: Sequence[IsolationWindow]) -> bool:
    """Whether any two mobility-resolved windows share scans.

    Disjoint windows are what makes a frame separable by mobility alone.
    Windows that overlap, or that carry no scan range at all, are not --
    which is a fact about the acquisition worth knowing before anyone
    tries to demultiplex it.
    """
    ranges = sorted(
        (w.scan_begin, w.scan_end) for w in windows if w.is_mobility_resolved
    )
    if len(ranges) < len(windows):
        return True
    return any(
        current[0] < previous[1] for previous, current in zip(ranges, ranges[1:])
    )
