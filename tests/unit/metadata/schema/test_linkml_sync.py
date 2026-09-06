"""The LinkML rendering must not drift from the pydantic models.

The YAML is the discussion artifact for the PSI/mzPeak imaging work and
the planned spec repository; the pydantic models are what Thyra actually
writes and validates.  This test pins the two together on everything
that matters for convergence: class and slot inventory, required flags,
CV bindings, and the polarity enum's term meanings.
"""

import pytest

yaml = pytest.importorskip("yaml")

from pathlib import Path  # noqa: E402

from thyra.metadata.schema import models  # noqa: E402
from thyra.metadata.schema.vocab import POLARITY_ACCESSIONS  # noqa: E402

_LINKML_PATH = Path(models.__file__).resolve().parent / "msi_metadata.linkml.yaml"

# LinkML records the on-disk requirement; pydantic gives some of those
# fields construction-time defaults. These are the deliberate deltas.
_REQUIRED_DESPITE_DEFAULT = {("MSIMetadata", "schema_version")}

_MODEL_CLASSES = {
    "OntologyTerm": models.OntologyTerm,
    "PixelSizeUm": models.PixelSizeUm,
    "ResolvingPower": models.ResolvingPower,
    "SampleInformation": models.SampleInformation,
    "SamplePreparation": models.SamplePreparation,
    "MSAnalysis": models.MSAnalysis,
    "IonMobility": models.IonMobility,
    "MobilityGrid": models.MobilityGrid,
    "Fragmentation": models.Fragmentation,
    "IsolationWindow": models.IsolationWindow,
    "SoftwareRef": models.SoftwareRef,
    "ProcessingStep": models.ProcessingStep,
    "Provenance": models.Provenance,
    "MSIMetadata": models.MSIMetadata,
}


@pytest.fixture(scope="module")
def linkml_schema():
    return yaml.safe_load(_LINKML_PATH.read_text(encoding="utf-8"))


def test_versions_agree(linkml_schema):
    assert linkml_schema["version"] == models.MSI_METADATA_SCHEMA_VERSION


def test_every_model_class_is_rendered(linkml_schema):
    assert set(linkml_schema["classes"]) == set(_MODEL_CLASSES)


@pytest.mark.parametrize("class_name", sorted(_MODEL_CLASSES))
def test_slots_and_required_flags_match(linkml_schema, class_name):
    model = _MODEL_CLASSES[class_name]
    attributes = linkml_schema["classes"][class_name]["attributes"]

    assert set(attributes) == set(model.model_fields), class_name

    for field_name, field in model.model_fields.items():
        rendered = attributes[field_name] or {}
        expected = field.is_required() or (
            (class_name, field_name) in _REQUIRED_DESPITE_DEFAULT
        )
        assert (
            bool(rendered.get("required")) == expected
        ), f"{class_name}.{field_name} required flag drifted"


@pytest.mark.parametrize("class_name", sorted(_MODEL_CLASSES))
def test_cv_bindings_match(linkml_schema, class_name):
    """Every `_cv` binding on a pydantic field is the slot's slot_uri."""
    model = _MODEL_CLASSES[class_name]
    attributes = linkml_schema["classes"][class_name]["attributes"]

    for field_name, field in model.model_fields.items():
        extra = field.json_schema_extra
        bound = (
            extra.get("cv", {}).get("accession") if isinstance(extra, dict) else None
        )
        rendered = (attributes[field_name] or {}).get("slot_uri")
        assert rendered == bound, (
            f"{class_name}.{field_name}: LinkML slot_uri {rendered!r} vs "
            f"pydantic cv binding {bound!r}"
        )


def test_polarity_enum_meanings_match(linkml_schema):
    values = linkml_schema["enums"]["PolarityEnum"]["permissible_values"]
    assert {
        name: entry.get("meaning") for name, entry in values.items()
    } == POLARITY_ACCESSIONS


def test_tree_root_is_the_document(linkml_schema):
    assert linkml_schema["classes"]["MSIMetadata"].get("tree_root") is True
