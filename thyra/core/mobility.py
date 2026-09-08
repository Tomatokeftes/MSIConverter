"""Ion mobility as a spectral coordinate.

Mobility is a second coordinate on a *feature*, next to m/z. It never
enters ``obs``, a coordinate system or a transform: those say where a
pixel is, mobility says what was measured there. The reader contract in
:class:`thyra.core.base_reader.BaseMSIReader` carries it as an optional
fourth array per pixel; this module holds the vocabulary the readers, the
converters and the metadata schema share so they name the axis the same
way -- with the PSI-MS terms mzPeak also uses.

It also holds :func:`ccs_from_one_over_k0`, the one derived quantity of
the axis. A collision cross section needs a charge state, which no
converter can know, so it is computed on request and never written as a
``var`` column.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..errors import ConversionRefused

#: The two mobility quantities an instrument records, as PSI-MS terms.
INVERSE_REDUCED_MOBILITY_ACCESSION = "MS:1002815"
DRIFT_TIME_ACCESSION = "MS:1002476"

MOBILITY_KIND_NAMES: Dict[str, str] = {
    INVERSE_REDUCED_MOBILITY_ACCESSION: "inverse reduced ion mobility",
    DRIFT_TIME_ACCESSION: "ion mobility drift time",
}

#: PSI-MS array terms a source may bind a per-point mobility array to
#: (imzML ``referenceableParamGroup``, mzPeak column metadata), mapped to
#: the quantity they carry -- ``None`` when the term itself does not say
#: and the unit has to decide.
MOBILITY_ARRAY_TERMS: Dict[str, Tuple[str, Optional[str]]] = {
    "MS:1002893": ("ion mobility array", None),
    "MS:1003007": ("raw ion mobility array", None),
    "MS:1002816": ("mean ion mobility array", None),
    "MS:1003154": ("deconvoluted ion mobility array", None),
    "MS:1003006": (
        "mean inverse reduced ion mobility array",
        INVERSE_REDUCED_MOBILITY_ACCESSION,
    ),
    "MS:1003008": (
        "raw inverse reduced ion mobility array",
        INVERSE_REDUCED_MOBILITY_ACCESSION,
    ),
    "MS:1003155": (
        "deconvoluted inverse reduced ion mobility array",
        INVERSE_REDUCED_MOBILITY_ACCESSION,
    ),
    "MS:1002477": ("mean ion mobility drift time array", DRIFT_TIME_ACCESSION),
    "MS:1003153": ("raw ion mobility drift time array", DRIFT_TIME_ACCESSION),
    "MS:1003156": (
        "deconvoluted ion mobility drift time array",
        DRIFT_TIME_ACCESSION,
    ),
}

#: Units that identify the quantity when the array term does not.
_UNIT_TO_KIND: Dict[str, str] = {
    "MS:1002814": INVERSE_REDUCED_MOBILITY_ACCESSION,  # volt-second per square centimeter
    "UO:0000028": DRIFT_TIME_ACCESSION,  # millisecond
    "UO:0000010": DRIFT_TIME_ACCESSION,  # second
}


def classify_mobility_array(
    array_accession: Optional[str], unit_accession: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """The quantity a mobility array carries, as ``(accession, name)``.

    The array term decides when it can (``MS:1003006`` is 1/K0 by
    definition); a generic ``ion mobility array`` is disambiguated by its
    unit; anything else stays ``(None, None)`` rather than guessed.
    """
    kind = None
    if array_accession in MOBILITY_ARRAY_TERMS:
        kind = MOBILITY_ARRAY_TERMS[array_accession][1]
    if kind is None and unit_accession is not None:
        kind = _UNIT_TO_KIND.get(unit_accession)
    if kind is None:
        return None, None
    return kind, MOBILITY_KIND_NAMES[kind]


@dataclass(frozen=True)
class MobilityAxis:
    """What a reader knows about its mobility dimension.

    ``values`` is the native axis when the source shares one across pixels:
    the 1/K0 of every feature of a continuous imzML export, or the 1/K0 of
    every TIMS scan of a Bruker frame. It is ``None`` when each pixel
    carries its own mobility values (processed imzML), in which case only
    :meth:`BaseMSIReader.iter_mobility_spectra` has them.

    The quantity and unit are PSI-MS terms (``MS:1002815`` inverse reduced
    ion mobility, ``MS:1002476`` drift time; ``MS:1002814`` volt-second per
    square centimeter, ``UO:0000028`` millisecond) so a store, an mzPeak
    archive and an imzML export describe the same axis in the same words.
    """

    kind_accession: Optional[str]
    kind_name: Optional[str]
    unit_accession: Optional[str]
    unit_name: Optional[str]
    array_accession: Optional[str] = None
    values: Optional[NDArray[np.float64]] = None
    acq_range: Optional[Tuple[float, float]] = None
    calibration: Optional[Dict[str, Any]] = None
    source: Optional[str] = None

    @property
    def is_shared(self) -> bool:
        """Whether every pixel is described by the same ``values`` array."""
        return self.values is not None

    def to_uns(self) -> Dict[str, Any]:
        """The axis as a plain ``uns`` block: plain-name keys, no colons.

        Zarr writes a dict key as a directory name and a colon is not a
        legal Windows path character, so CV accessions are values here,
        never keys.
        """
        block: Dict[str, Any] = {"present": True}
        if self.kind_accession is not None:
            block["type_accession"] = self.kind_accession
            block["type_name"] = self.kind_name
        if self.unit_accession is not None:
            block["unit_accession"] = self.unit_accession
        if self.unit_name is not None:
            block["unit_name"] = self.unit_name
        if self.array_accession is not None:
            block["array_accession"] = self.array_accession
        if self.values is not None:
            block["values"] = np.asarray(self.values, dtype=np.float64)
            # ``n_scans`` is the name the store contract uses for the
            # length of ``values`` (one entry per TIMS scan on a Bruker
            # source); ``n_values`` is the older spelling and stays so
            # stores written before it are read the same way.
            block["n_scans"] = int(self.values.size)
            block["n_values"] = int(self.values.size)
        lo_hi = self.acq_range
        if lo_hi is None and self.values is not None and self.values.size:
            lo_hi = (float(np.min(self.values)), float(np.max(self.values)))
        if lo_hi is not None:
            block["acq_range"] = np.array(
                [float(lo_hi[0]), float(lo_hi[1])], dtype=np.float64
            )
            block["range_lower"] = float(lo_hi[0])
            block["range_upper"] = float(lo_hi[1])
        if self.calibration:
            block["calibration"] = dict(self.calibration)
        if self.source is not None:
            block["source"] = self.source
        return block

    def to_extractor_report(self) -> Dict[str, Any]:
        """The shape :func:`thyra.metadata.schema.builder` reads from ``format_specific``."""
        report: Dict[str, Any] = {"present": True}
        if self.kind_name is not None:
            report["separation"] = self.kind_name
            report["separation_accession"] = self.kind_accession
        if self.unit_name is not None:
            report["unit"] = self.unit_name
        if self.unit_accession is not None:
            report["unit_accession"] = self.unit_accession
        if self.array_accession is not None:
            report["array_accession"] = self.array_accession
        lo_hi = self.acq_range
        if lo_hi is None and self.values is not None and self.values.size:
            lo_hi = (float(np.min(self.values)), float(np.max(self.values)))
        if lo_hi is not None:
            report["range"] = [float(lo_hi[0]), float(lo_hi[1])]
        if self.values is not None:
            report["n_values"] = int(self.values.size)
        report["shared_axis"] = self.is_shared
        return report


# ----------------------------------------------------------------------
# Collision cross section
# ----------------------------------------------------------------------

#: Drift gas and temperature of the Mason-Schamp convention Bruker's
#: ``tims_oneoverk0_to_ccs_for_mz`` uses: nitrogen (2 x 14.0067 Da) at
#: 305 K (31.85 C). Measured against the bundled library on 2026-09-05:
#: these two values reproduce it to 8e-8 relative; 28.013 Da or 304.85 K
#: do not (7e-6 and 2.5e-4).
CCS_DRIFT_GAS_MASS_DA = 28.0134
CCS_TEMPERATURE_K = 305.0

# CODATA 2018 constants the Mason-Schamp prefactor is built from.
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_BOLTZMANN_J_PER_K = 1.380649e-23
_LOSCHMIDT_PER_M3 = 2.686780111e25  # number density at 273.15 K, 101325 Pa
_DALTON_KG = 1.66053906660e-27


def _mason_schamp_prefactor() -> float:
    """The constant in ``CCS[A^2] = C * z * (1/K0) / sqrt(mu[Da] * T)``.

    Mason-Schamp: ``CCS = 3 z e / (16 N) * sqrt(2 pi / (mu k T)) / K0``
    with ``N`` the gas number density at standard conditions. The unit
    conversions -- 1/K0 in V s cm^-2, the reduced mass in daltons, the
    result in square angstroms -- are folded in here, so the function
    below is a plain evaluation.
    """
    si = (3.0 * _ELEMENTARY_CHARGE_C / (16.0 * _LOSCHMIDT_PER_M3)) * math.sqrt(
        2.0 * math.pi / _BOLTZMANN_J_PER_K
    )
    # 1/K0: V s cm^-2 -> V s m^-2 (x 1e4); mu: Da -> kg (sqrt divides);
    # m^2 -> A^2 (x 1e20).
    return si * 1e4 / math.sqrt(_DALTON_KG) * 1e20


#: Evaluates to 18509.86 A^2 sqrt(Da K) cm^2 V^-1 s^-1, the number the
#: vendor's own documentation and the open reimplementations carry.
MASON_SCHAMP_PREFACTOR = _mason_schamp_prefactor()


def mason_schamp_ccs(one_over_k0: Any, mz: Any, charge: int) -> NDArray[np.float64]:
    """Collision cross section (A^2) by the Mason-Schamp equation, no SDK.

    Nitrogen drift gas at 305 K, the convention behind Bruker's own
    conversion (:data:`CCS_DRIFT_GAS_MASS_DA`, :data:`CCS_TEMPERATURE_K`).
    ``one_over_k0`` (V s cm^-2) and ``mz`` broadcast against each other;
    ``charge`` is the absolute charge state and is the caller's to know.
    """
    z = _checked_charge(charge)
    ook0 = np.asarray(one_over_k0, dtype=np.float64)
    mz_arr = np.asarray(mz, dtype=np.float64)
    ion_mass = mz_arr * z
    reduced_mass = ion_mass * CCS_DRIFT_GAS_MASS_DA / (ion_mass + CCS_DRIFT_GAS_MASS_DA)
    return MASON_SCHAMP_PREFACTOR * z * ook0 / np.sqrt(reduced_mass * CCS_TEMPERATURE_K)


def ccs_from_one_over_k0(
    one_over_k0: Any, mz: Any, charge: int, sdk: Optional[Any] = None
) -> NDArray[np.float64]:
    """Collision cross section (A^2) from 1/K0, m/z and a charge state.

    Goes through the vendor library when ``sdk`` (an
    :class:`~thyra.readers.bruker.timstof.sdk.sdk_functions.SDKFunctions`
    opened on a TDF file) is given and exports the conversion, and
    through :func:`mason_schamp_ccs` otherwise; the two agree to better
    than 1e-6 relative and a test pins them to each other.

    The charge is an input on purpose. MALDI imaging is mostly singly
    charged, but that is an assumption an annotation step has to make
    and record; the converter never writes a ``ccs`` column.
    """
    _checked_charge(charge)
    if sdk is not None and getattr(sdk, "file_type", None) == "tdf":
        bound = getattr(sdk, "_bound_conversions", {})
        if bound.get("tims_oneoverk0_to_ccs_for_mz", False):
            return np.asarray(
                sdk.oneoverk0_to_ccs(
                    np.asarray(one_over_k0, dtype=np.float64),
                    int(charge),
                    np.asarray(mz, dtype=np.float64),
                ),
                dtype=np.float64,
            )
    return mason_schamp_ccs(one_over_k0, mz, charge)


def _checked_charge(charge: Any) -> int:
    """A charge state usable by both CCS routes: a positive integer."""
    if isinstance(charge, bool) or not isinstance(charge, (int, np.integer)):
        raise TypeError(f"charge must be an integer charge state, got {charge!r}")
    if int(charge) < 1:
        raise ConversionRefused(
            f"charge must be a positive charge state (the absolute value), "
            f"got {int(charge)}"
        )
    return int(charge)
