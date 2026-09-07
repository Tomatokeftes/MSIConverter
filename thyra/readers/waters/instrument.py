"""Which Waters instrument wrote a ``.raw`` directory, from its own metadata.

MassLynx leaves two text files beside the ``_FUNC*.DAT`` data: ``_extern.inf``
with the instrument configuration and per-function parameters, and
``_header.txt`` with the acquisition header. Neither passes through the
native library, so they are read here directly.

The one distinction Thyra acts on is whether the analyser is a SELECT SERIES
MRT. Its multi-reflecting flight path gives R ~ 170,000 (measured 130,000 at
m/z 300 rising to 190,000 at m/z 1000 on a MALDI brain section), and at that
resolving power the vendor peak picker is the limit rather than the analyser:
it reports one centroid where the sampled trace has two maxima 8-9 mDa apart,
in 96-100% of the pixels that resolve them. On a Synapt G2-Si (R ~ 26,000)
the same test finds the picker merging nothing the profile resolves. So MRT
runs default to the profile trace and every other Waters instrument keeps the
vendor centroid; see :mod:`thyra.readers.waters.waters_reader`.

Fields checked, in order, with what each run actually carries:

======================  ==========  ===============
field                   MRT run     Synapt G2-Si run
======================  ==========  ===============
``OpticMode``           ``MRT``     absent
``$$ Instrument`` (hdr) ``MRT#``    absent
``Resolution``          225909.053  10000
======================  ==========  ===============

``OpticMode`` is the instrument's own word for it and decides outright. The
header line is the acquisition's instrument label. ``Resolution`` is the
fallback: no other Waters TOF is configured anywhere near 100,000, so a
declared resolution above that threshold is taken as MRT when the other two
fields are missing. The decision and the field it rested on are logged, and
returned so the metadata extractor can record them in the store.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

#: ``Resolution`` in ``_extern.inf`` at or above which a run is taken as MRT
#: when neither ``OpticMode`` nor the header instrument label says so. The
#: MRT declares ~225,000; a Synapt G2-Si declares 10,000, Xevo-class QTofs
#: are in the 20,000-40,000 range. Nothing but an MRT reaches six figures.
MRT_RESOLUTION_THRESHOLD = 100_000.0

# Physical constants for the digitiser sample spacing estimate below.
_ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
_ELEMENTARY_CHARGE_C = 1.602176634e-19


@dataclass(frozen=True)
class WatersInstrument:
    """What ``_extern.inf`` and ``_header.txt`` say about the analyser.

    Attributes:
        is_mrt: Whether the run was acquired on a SELECT SERIES MRT.
        decided_by: The field the decision rested on, as ``"field=value"``,
            or ``"no instrument field found"`` when none of the three was
            present and ``is_mrt`` is ``False`` by default.
        optic_mode: ``OpticMode`` from ``_extern.inf``, if present.
        instrument_label: ``$$ Instrument`` from ``_header.txt``, if present.
        resolution: ``Resolution`` from ``_extern.inf``, if present and
            numeric.
        flight_path_mm: ``Lteff`` from ``_extern.inf`` (effective flight
            path, mm), if present.
        effective_voltage_v: ``Veff`` from ``_extern.inf``, if present.
        adc_sample_frequency_ghz: ``ADC Sample Frequency (GHz)`` from
            ``_extern.inf``, if present.
    """

    is_mrt: bool
    decided_by: str
    optic_mode: Optional[str] = None
    instrument_label: Optional[str] = None
    resolution: Optional[float] = None
    flight_path_mm: Optional[float] = None
    effective_voltage_v: Optional[float] = None
    adc_sample_frequency_ghz: Optional[float] = None

    @property
    def name(self) -> str:
        """A short label for logs and stored metadata."""
        return "SELECT SERIES MRT" if self.is_mrt else "Waters (not MRT)"

    def profile_sample_spacing_da(self, mz: float) -> Optional[float]:
        """Predicted spacing of the profile trace's samples at ``mz``, in Da.

        The trace is sampled at the ADC's fixed rate, so consecutive samples
        are one clock period apart in flight time. With ``t = L * sqrt(m /
        (2 e V))`` for a singly charged ion of mass ``m`` over flight path
        ``L`` at effective voltage ``V``, one period ``dt`` spans ``dm = 2 m
        dt / t``. On the MRT reference run this predicts 1.143 mDa at m/z
        1000 against 1.16 mDa measured (1.5% off); on a Synapt G2-Si, 13.8
        mDa against 13.4 measured.

        Returns:
            The spacing in Da, or ``None`` when ``Lteff``, ``Veff`` or the
            ADC frequency is missing from ``_extern.inf``.
        """
        if (
            self.flight_path_mm is None
            or self.effective_voltage_v is None
            or self.adc_sample_frequency_ghz is None
            or self.flight_path_mm <= 0
            or self.effective_voltage_v <= 0
            or self.adc_sample_frequency_ghz <= 0
        ):
            return None
        dt = 1.0 / (self.adc_sample_frequency_ghz * 1e9)
        velocity_term = math.sqrt(
            2.0
            * _ELEMENTARY_CHARGE_C
            * self.effective_voltage_v
            * mz
            / _ATOMIC_MASS_UNIT_KG
        )
        return 2.0 * dt * velocity_term / (self.flight_path_mm * 1e-3)


def _read_text(path: Path) -> Optional[str]:
    """Read a MassLynx side file, tolerating its Latin-1 degree signs."""
    try:
        return path.read_text(encoding="latin-1")
    except OSError:
        return None


def parse_extern_inf(text: str) -> Dict[str, str]:
    """Parse ``_extern.inf`` into ``{key: value}``, first occurrence wins.

    Lines are ``key<tabs>value``; the MRT pads with spaces then one tab, the
    Synapt with runs of tabs. Section headers carry no tab and are skipped.
    ``ADC Sample Frequency (GHz)`` appears under both the instrument
    configuration and each function's parameters with the same value, so
    keeping the first occurrence is safe; ``OpticMode`` appears once per
    function.
    """
    fields: Dict[str, str] = {}
    for line in text.splitlines():
        if "\t" not in line:
            continue
        key, _, value = line.partition("\t")
        key = key.strip()
        value = value.strip()
        if key and value and key not in fields:
            fields[key] = value
    return fields


def parse_header_txt(text: str) -> Dict[str, str]:
    """Parse ``_header.txt`` (``$$ Key: value`` lines) into ``{key: value}``."""
    fields: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("$$"):
            continue
        key, sep, value = line[2:].partition(":")
        if sep and key.strip():
            fields[key.strip()] = value.strip()
    return fields


def _float_or_none(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def identify_waters_instrument(raw_path: Path) -> WatersInstrument:
    """Decide whether ``raw_path`` was acquired on a SELECT SERIES MRT.

    Args:
        raw_path: The ``.raw`` directory.

    Returns:
        The decision, the field it rested on, and the raw fields consulted.
        A directory with none of the fields is *not* MRT; that is the safe
        answer, since it keeps the vendor centroid the reader always
        delivered.
    """
    extern = parse_extern_inf(_read_text(raw_path / "_extern.inf") or "")
    header = parse_header_txt(_read_text(raw_path / "_header.txt") or "")

    optic_mode = extern.get("OpticMode")
    instrument_label = header.get("Instrument")
    resolution = _float_or_none(extern.get("Resolution"))
    if optic_mode is not None:
        is_mrt = optic_mode.upper() == "MRT"
        decided_by = f"OpticMode={optic_mode}"
    elif instrument_label is not None:
        is_mrt = instrument_label.upper().startswith("MRT")
        decided_by = f"$$ Instrument={instrument_label}"
    elif resolution is not None:
        is_mrt = resolution >= MRT_RESOLUTION_THRESHOLD
        decided_by = f"Resolution={resolution:g}"
    else:
        is_mrt = False
        decided_by = "no instrument field found"

    decision = WatersInstrument(
        is_mrt=is_mrt,
        decided_by=decided_by,
        optic_mode=optic_mode,
        instrument_label=instrument_label,
        resolution=resolution,
        flight_path_mm=_float_or_none(extern.get("Lteff")),
        effective_voltage_v=_float_or_none(extern.get("Veff")),
        adc_sample_frequency_ghz=_float_or_none(
            extern.get("ADC Sample Frequency (GHz)")
        ),
    )

    logger.info(
        "%s identified as %s (%s)",
        raw_path.name,
        decision.name,
        decision.decided_by,
    )
    return decision
