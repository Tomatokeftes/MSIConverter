"""A store under a deep path must read back exactly, not as fill values.

Windows caps a normal path at 260 characters and the limit applies to every
key inside the store. Past it, Windows reports a key as missing and Zarr
returns the array's fill value instead of raising, so a metadata block read
from a plain path came back with every ontology term as
``{"accession": "", "name": ""}`` and ``thyra validate`` blamed the document.
Reproduced on 2026-09-05 with a TDF conversion into a 178-character scratch
directory. The mobility fixture writes Thyra's deepest key
(``..._mobility/uns/msi_metadata/ms_analysis/ion_mobility/...``), which is
the one that broke.

The deep path is built by hand rather than under ``tmp_path`` because the
store has to be removed through an extended-length path afterwards, and
pytest's own cleanup would trip the same limit. Off Windows the path is
simply deep and the test is a plain round trip.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

spatialdata = pytest.importorskip("spatialdata")

from thyra.convert import convert_msi  # noqa: E402
from thyra.metadata.schema import (  # noqa: E402
    check_store_var_conventions,
    read_msi_metadata_blocks,
)
from thyra.metadata.schema.cli import validate_command  # noqa: E402
from thyra.utils.windows_paths import (  # noqa: E402
    WINDOWS_MAX_PATH,
    to_extended_length_path,
)

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "fixtures"
    / "mobility_continuous.imzML"
)

#: Long enough for the deepest metadata key, about 100 characters below the
#: store root, to land well past the limit.
DEEP_ROOT_LENGTH = 180


def _walkable(path: Path) -> Path:
    """The spelling under which every key of a deep store is reachable."""
    return to_extended_length_path(path) if sys.platform == "win32" else path


def _longest_key(store: Path) -> int:
    """Longest key in the store as a plain path would spell it."""
    root = _walkable(store)
    prefix = len(str(root)) - len(str(store))
    return max(
        len(parent) + 1 + len(name) - prefix
        for parent, _dirs, files in os.walk(root)
        for name in files
    )


@pytest.fixture(scope="module")
def stores():
    """The same store twice: converted at a deep path, then copied short."""
    base = Path(tempfile.mkdtemp(prefix="thyra_deep_"))
    try:
        deep_dir = base
        while len(str(deep_dir)) < DEEP_ROOT_LENGTH:
            deep_dir = deep_dir / "nested_directory_segment"
            deep_dir.mkdir()
        deep = deep_dir / "x.zarr"
        assert convert_msi(
            str(FIXTURE), str(deep), dataset_id="mob", pixel_size_um=10.0
        ), "conversion reported failure"
        assert _longest_key(deep) > WINDOWS_MAX_PATH, "path is not deep enough"

        short = base / "short.zarr"
        shutil.copytree(_walkable(deep), short)
        yield deep, short
    finally:
        shutil.rmtree(_walkable(base), ignore_errors=True)


class TestDeepStoreReadsBack:
    def test_metadata_block_is_identical_to_the_short_copy(self, stores):
        deep, short = stores

        deep_blocks = read_msi_metadata_blocks(deep)

        assert set(deep_blocks) == {"mob_z0", "mob_z0_mobility"}
        assert deep_blocks == read_msi_metadata_blocks(short)

    def test_the_deepest_terms_are_not_fill_values(self, stores):
        """The exact symptom: the ion mobility terms read back empty."""
        deep, _ = stores

        block = read_msi_metadata_blocks(deep)["mob_z0_mobility"]
        mobility = block["ms_analysis"]["ion_mobility"]

        assert mobility["unit_term"]["accession"].startswith("MS:")
        assert mobility["separation_term"]["accession"].startswith("MS:")

    def test_processing_history_survives(self, stores):
        """The processing JSON must come back parsed, not left as a string."""
        deep, _ = stores

        for block in read_msi_metadata_blocks(deep).values():
            assert isinstance(block["processing"], list)
            assert block["processing"][0]["name"] == "conversion"

    def test_var_conventions_agree_with_the_short_copy(self, stores):
        deep, short = stores

        issues = check_store_var_conventions(deep)

        assert issues == check_store_var_conventions(short)
        assert all(not table_issues for table_issues in issues.values())

    def test_validate_command_passes(self, stores):
        deep, _ = stores

        result = CliRunner().invoke(validate_command, [str(deep)])

        assert result.exit_code == 0, result.output
        assert "FAILED" not in result.output

    def test_validate_command_accepts_a_relative_path(self, stores, monkeypatch):
        """The length that matters is the absolute one, whatever was typed."""
        deep, _ = stores
        monkeypatch.chdir(deep.parent)

        result = CliRunner().invoke(validate_command, [deep.name])

        assert result.exit_code == 0, result.output
