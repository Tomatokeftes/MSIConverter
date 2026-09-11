"""spatialdata is a hard dependency: its absence must abort ``import thyra``.

``pyproject.toml`` declares ``spatialdata`` in ``[project] dependencies``
with no ``[project.optional-dependencies]`` table anywhere, so an install
without it is broken, not a supported configuration. It used to be treated
as optional in three independent places, and the one that fired swallowed
the cause: ``import thyra`` succeeded, every SpatialData name in the base
converter was rebound to ``None``, the converter was never registered, and
the first thing the user saw was a registry miss naming no package and no
reason. The real ``ImportError`` sat in a module-level string one module
away and reached the user on no path (issue #310).

What a broken install is entitled to is the exception and its traceback,
pointing at the import that failed.
"""

import subprocess
import sys

# The child installs this finder ahead of every other one and refuses the
# spatialdata root package only -- ``fullname.split(".")[0]`` keeps anndata,
# geopandas, shapely and thyra's own ``thyra.converters.spatialdata``
# subpackage importable, so what is measured is the one dependency.
_BLOCK_SPATIALDATA = """
import importlib.abc
import sys


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "spatialdata":
            raise ImportError("blocked by tests/unit/test_hard_dependency.py")
        return None


sys.meta_path.insert(0, _Blocker())

import thyra

print("import thyra SUCCEEDED", thyra.SpatialDataConverter)
"""


def _run_child_without_spatialdata() -> "subprocess.CompletedProcess[str]":
    """Import thyra in a fresh interpreter that cannot import spatialdata.

    A child is required: ``thyra`` and ``spatialdata`` are both already in
    this process's ``sys.modules``, so a finder installed here would never
    be consulted for either.
    """
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_SPATIALDATA],
        capture_output=True,
        text=True,
    )


def test_import_thyra_fails_when_spatialdata_is_missing():
    """``import thyra`` must raise, not warn and carry on.

    The exit status is the load-bearing assertion. stderr alone does not
    discriminate: before the fix the child also wrote "SpatialData
    dependencies not available: ..." to stderr, because the swallowing
    handler logged a warning and logging's lastResort handler printed it --
    so the substring "spatialdata" was present either way. What changed is
    that the process now dies.
    """
    proc = _run_child_without_spatialdata()

    assert proc.returncode != 0, (
        "import thyra succeeded without spatialdata; the absence of a hard "
        f"dependency was swallowed.\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "import thyra SUCCEEDED" not in proc.stdout


def test_the_failure_names_the_import_that_failed():
    """The user gets the real exception, not a paraphrase of it.

    "ImportError" is the discriminator here: the message the swallowing
    path produced ("SpatialData dependencies not available", later
    "SpatialData converter unavailable") named neither the exception type
    nor the import site.
    """
    proc = _run_child_without_spatialdata()

    assert "ImportError" in proc.stderr, proc.stderr
    assert "spatialdata" in proc.stderr, proc.stderr
    # The traceback points at the import statement, which is the part of a
    # broken install that is actually actionable.
    assert "Traceback" in proc.stderr, proc.stderr
