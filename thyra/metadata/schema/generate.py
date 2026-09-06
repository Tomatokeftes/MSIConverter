# thyra/metadata/schema/generate.py
"""Regenerate the committed JSON Schema artifact.

The JSON Schema rendering of the pydantic models is committed next to
the models (``msi_metadata_schema_v0_5.json``) so that non-Python
consumers can validate documents without importing Thyra.  A unit test
asserts the committed file matches the models; when it fails, rerun::

    python -m thyra.metadata.schema.generate

and commit the result together with the model change (and the version
bump in ``MSI_METADATA_SCHEMA_VERSION`` that the change warrants).
"""

import json
from pathlib import Path

from .models import SCHEMA_JSON_FILENAME, MSIMetadata


def main() -> None:
    """Write the JSON Schema for the current models next to this module."""
    schema = MSIMetadata.model_json_schema()
    target = Path(__file__).with_name(SCHEMA_JSON_FILENAME)
    target.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
