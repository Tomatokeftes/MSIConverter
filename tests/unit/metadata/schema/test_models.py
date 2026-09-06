"""Structural validation: the pydantic models enforce the schema's shape."""

import pytest
from pydantic import ValidationError

from thyra.metadata.schema import (
    MSI_METADATA_SCHEMA_VERSION,
    MSAnalysis,
    MSIMetadata,
    OntologyTerm,
    PixelSizeUm,
    Provenance,
)


def _minimal() -> MSIMetadata:
    return MSIMetadata(
        ms_analysis=MSAnalysis(pixel_size_um=PixelSizeUm(x=20.0, y=20.0)),
        provenance=Provenance(thyra_version="1.0.0"),
    )


class TestMSIMetadata:
    def test_minimal_document_is_valid(self):
        meta = _minimal()
        assert meta.schema_version == MSI_METADATA_SCHEMA_VERSION

    def test_round_trips_through_dump_and_validate(self):
        meta = _minimal()
        assert MSIMetadata.model_validate(meta.model_dump()) == meta

    def test_pixel_size_is_required(self):
        with pytest.raises(ValidationError):
            MSAnalysis()

    def test_pixel_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            PixelSizeUm(x=0.0, y=20.0)

    def test_unknown_fields_are_rejected(self):
        doc = _minimal().model_dump()
        doc["ms_analysis"]["lazer_power"] = 3.0
        with pytest.raises(ValidationError):
            MSIMetadata.model_validate(doc)

    def test_accession_must_be_a_curie(self):
        with pytest.raises(ValidationError):
            OntologyTerm(accession="not a curie", name="x")

    def test_polarity_vocabulary_is_closed(self):
        doc = _minimal().model_dump()
        doc["ms_analysis"]["polarity"] = "both"
        with pytest.raises(ValidationError):
            MSIMetadata.model_validate(doc)

    def test_contradicting_polarity_term_is_rejected(self):
        with pytest.raises(ValidationError, match="contradicts"):
            MSAnalysis(
                pixel_size_um=PixelSizeUm(x=20.0, y=20.0),
                polarity="positive",
                polarity_term=OntologyTerm(
                    accession="MS:1000129", name="negative scan"
                ),
            )

    def test_agreeing_polarity_term_is_accepted(self):
        analysis = MSAnalysis(
            pixel_size_um=PixelSizeUm(x=20.0, y=20.0),
            polarity="negative",
            polarity_term=OntologyTerm(accession="MS:1000129", name="negative scan"),
        )
        assert analysis.polarity == "negative"


class TestCvBindings:
    def test_bound_fields_are_exactly_the_expected_set(self):
        from thyra.metadata.schema.models import field_cv_bindings

        accessions = {cv["accession"] for cv in field_cv_bindings().values()}
        assert accessions == {
            "IMS:1000046",  # pixel size (x)
            "IMS:1000047",  # pixel size y
            "MS:1000465",  # scan polarity
            "MS:1000008",  # ionization type
            "MS:1000443",  # mass analyzer type
            "MS:1000031",  # instrument model
            "MS:1000800",  # mass resolving power
            "MS:1002892",  # ion mobility attribute
            "MS:1000511",  # ms level
            "MS:1000827",  # isolation window target m/z
            "MS:1000828",  # isolation window lower offset
            "MS:1000829",  # isolation window upper offset
            "MS:1000045",  # collision energy
        }

    def test_every_binding_resolves_in_the_local_tables(self):
        from thyra.metadata.ontology.cache import ONTOLOGY
        from thyra.metadata.schema.models import field_cv_bindings

        for path, cv in field_cv_bindings().items():
            entry = ONTOLOGY.terms.get(cv["accession"])
            assert entry is not None, f"{path}: unknown {cv['accession']}"
            assert (
                entry[0] == cv["name"]
            ), f"{path}: bound name {cv['name']!r} vs CV label {entry[0]!r}"

    def test_bindings_land_in_the_committed_json_schema(self):
        # The claim must be checkable from the artifact alone, without
        # importing Thyra.
        import json
        from pathlib import Path

        from thyra.metadata.schema import models

        artifact = (
            Path(models.__file__).parent / models.SCHEMA_JSON_FILENAME
        ).read_text(encoding="utf-8")
        rendered = json.dumps(json.loads(artifact))
        assert '"IMS:1000046"' in rendered
        assert '"MS:1000443"' in rendered

    def test_candidate_concepts_are_declared(self):
        from thyra.metadata.schema import models

        concepts = [c for c, _ in models.CANDIDATE_CV_CONCEPTS]
        assert any("resampling" in c for c in concepts)
        assert any("stage offset" in c for c in concepts)


class TestProcessing:
    def test_processing_steps_round_trip(self):
        from thyra.metadata.schema import ProcessingStep, SoftwareRef

        meta = _minimal().model_copy(deep=True)
        meta.processing = [
            ProcessingStep(
                name="mass axis resampling",
                software=SoftwareRef(name="thyra", version="1.0.0"),
                parameters={"target_bins": 50000, "method": "nearest_neighbor"},
            )
        ]
        restored = MSIMetadata.model_validate(meta.model_dump())
        assert restored.processing[0].parameters["target_bins"] == 50000

    def test_empty_processing_is_omitted_from_uns(self):
        assert "processing" not in _minimal().to_uns_dict()

    def test_step_requires_software(self):
        from thyra.metadata.schema import ProcessingStep

        with pytest.raises(ValidationError):
            ProcessingStep(name="normalisation")


class TestToUnsDict:
    def test_none_fields_are_dropped(self):
        data = _minimal().to_uns_dict()
        assert "polarity" not in data["ms_analysis"]
        assert "source_format" not in data["provenance"]

    def test_empty_user_sections_are_omitted_not_written_empty(self):
        data = _minimal().to_uns_dict()
        assert "sample" not in data
        assert "preparation" not in data

    def test_populated_user_sections_are_kept(self):
        meta = _minimal().model_copy(deep=True)
        meta.sample.organism = "Mus musculus"
        data = meta.to_uns_dict()
        assert data["sample"] == {"organism": "Mus musculus"}

    def test_uns_dict_validates_back(self):
        data = _minimal().to_uns_dict()
        assert MSIMetadata.model_validate(data) is not None
