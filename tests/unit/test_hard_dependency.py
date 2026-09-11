"""A declared dependency's absence must abort ``import thyra``.

``pyproject.toml`` declares ``spatialdata`` and ``defusedxml`` in
``[project] dependencies`` with no ``[project.optional-dependencies]``
table anywhere, so an install without either is broken, not a supported
configuration.

spatialdata used to be treated as optional in three independent places,
and the one that fired swallowed the cause: ``import thyra`` succeeded,
every SpatialData name in the base converter was rebound to ``None``, the
converter was never registered, and the first thing the user saw was a
registry miss naming no package and no reason. The real ``ImportError``
sat in a module-level string one module away and reached the user on no
path (issue #310).

defusedxml is the same shape of mistake with a sharper edge: the fallback
was to ``xml.etree``, which is the parser the defusedxml swap was made to
get away from, so a warning nobody reads was all that separated a
hardened parse from an entity-expanding one.

What a broken install is entitled to, in both cases, is the exception and
its traceback, pointing at the import that failed.
"""

import subprocess
import sys

# The child installs this finder ahead of every other one and refuses one
# root package -- ``fullname.split(".")[0]`` keeps anndata, geopandas,
# shapely and thyra's own ``thyra.converters.spatialdata`` subpackage
# importable, so what is measured is the one dependency.
_BLOCK_TEMPLATE = """
import importlib.abc
import sys


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "{package}":
            raise ImportError("blocked by tests/unit/test_hard_dependency.py")
        return None


sys.meta_path.insert(0, _Blocker())

import thyra

print("import thyra SUCCEEDED", {witness})
"""

_BLOCK_SPATIALDATA = _BLOCK_TEMPLATE.format(
    package="spatialdata", witness="thyra.SpatialDataConverter"
)

# The witness for defusedxml is the module object the parser binds ``ET``
# to: on the fallback the name existed and was ``xml.etree.ElementTree``,
# so printing it is what distinguishes "imported defusedxml" from
# "imported something".
_BLOCK_DEFUSEDXML = _BLOCK_TEMPLATE.format(
    package="defusedxml", witness="thyra.readers.bruker.mis_parser.ET"
)


def _run_child(source: str) -> "subprocess.CompletedProcess[str]":
    """Import thyra in a fresh interpreter that cannot import one package.

    A child is required: ``thyra`` and its dependencies are all already in
    this process's ``sys.modules``, so a finder installed here would never
    be consulted for any of them.
    """
    return subprocess.run(
        [sys.executable, "-c", source],
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
    proc = _run_child(_BLOCK_SPATIALDATA)

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
    proc = _run_child(_BLOCK_SPATIALDATA)

    assert "ImportError" in proc.stderr, proc.stderr
    assert "spatialdata" in proc.stderr, proc.stderr
    # The traceback points at the import statement, which is the part of a
    # broken install that is actually actionable.
    assert "Traceback" in proc.stderr, proc.stderr


def test_import_thyra_fails_when_defusedxml_is_missing():
    """No silent downgrade to the parser that expands entities.

    ``mis_parser`` carried ``try: import defusedxml ... except ImportError:
    import xml.etree.ElementTree``, which logged a warning and carried on.
    Measured on a ``.mis`` holding ``<!ENTITY r "5,5">``: the stdlib parser
    returns the expansion and reports a 5x5 raster, defusedxml raises
    EntitiesForbidden. So the fallback was not a reduced-function parse, it
    was the unhardened one -- reachable on any install where the wheel
    happened not to be there, announced by a single WARNING line during
    ``import thyra`` that nothing is watching.

    Every Bruker path goes through this parser (the Rapiflex, timsTOF and
    solariX readers, and BrukerMetadataExtractor), and ``import thyra``
    reaches it through ``thyra.readers``, so the abort is at import of the
    package rather than at first use.
    """
    proc = _run_child(_BLOCK_DEFUSEDXML)

    assert proc.returncode != 0, (
        "import thyra succeeded without defusedxml; the .mis parser fell "
        "back to the entity-expanding stdlib parser.\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "import thyra SUCCEEDED" not in proc.stdout
    assert "ImportError" in proc.stderr, proc.stderr
    assert "defusedxml" in proc.stderr, proc.stderr
    assert "mis_parser" in proc.stderr, proc.stderr
    assert "Traceback" in proc.stderr, proc.stderr
