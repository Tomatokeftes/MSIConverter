"""A reader's own progress bar must be silenceable by the converter.

Every conversion reads the source twice (issue #226), and the converter
sets ``reader._quiet_mode`` for both passes because it draws its own
Pre-scan and Scatter bars. A reader whose bar ignores that flag prints it
twice per conversion, interleaved with the converter's bars and their
cursor-control codes:

    grep -c "Reading Rapiflex spectra: 100%" rapi_default.log  ->  2

imzML and timsTOF honoured it; solariX, Rapiflex and Waters did not
(issue #238). Rather than pin those three call sites, this states the
rule for any reader added later: a progress bar drawn inside a reader's
own iteration is a bar the converter can turn off.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, Tuple

import pytest

READERS = Path(__file__).resolve().parents[3] / "thyra" / "readers"

#: Functions whose bars are drawn once per pass over the source.
ITERATORS = ("iter_spectra", "iter_frame_scans")


def _bars() -> Iterator[Tuple[str, str, ast.Call]]:
    """Every ``tqdm(...)`` a reader draws from one of its iteration methods."""
    for path in sorted(READERS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(name in function.name for name in ITERATORS):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and (
                    getattr(node.func, "id", None) == "tqdm"
                ):
                    module = path.relative_to(READERS.parent.parent).as_posix()
                    yield module, function.name, node


CASES = list(_bars())


def test_the_readers_that_draw_bars_are_the_ones_expected():
    """A reader dropping its bar entirely would make the rule below vacuous."""
    modules = {module for module, _fn, _call in CASES}
    assert modules >= {
        "thyra/readers/bruker/rapiflex/rapiflex_reader.py",
        "thyra/readers/bruker/solarix/solarix_reader.py",
        "thyra/readers/imzml/imzml_reader.py",
        "thyra/readers/waters/waters_reader.py",
    }


@pytest.mark.parametrize(
    "module,function,call",
    CASES,
    ids=[f"{module.split('/')[-1]}::{fn}" for module, fn, _call in CASES],
)
def test_iteration_bars_can_be_disabled(module, function, call):
    """``disable=`` is what lets ``_quiet_mode`` reach the bar."""
    keywords = {keyword.arg for keyword in call.keywords}
    assert "disable" in keywords, (
        f"{module}:{call.lineno} draws a progress bar in {function}() without "
        "disable=; the converter's _quiet_mode cannot silence it, so it "
        "prints once per pass."
    )
