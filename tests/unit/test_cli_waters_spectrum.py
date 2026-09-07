"""``--waters-spectrum`` is a transport for ``WatersReader(use_centroid=...)``.

Omitted, nothing is forwarded, and the reader decides from the instrument
(profile on a SELECT SERIES MRT, centroid elsewhere). Given, it is
forwarded as the reader keyword -- the negation, since the flag names the
representation and the keyword names the peak picker.
"""

from __future__ import annotations

import pytest

from thyra.__main__ import _build_reader_options


class TestBuildReaderOptions:
    def test_omitted_is_not_forwarded(self):
        """``use_centroid=None`` is the reader's "decide from the instrument"."""
        assert "use_centroid" not in _build_reader_options(True, None)

    @pytest.mark.parametrize(
        "value, use_centroid", [("centroid", True), ("profile", False)]
    )
    def test_an_explicit_choice_is_forwarded_as_the_reader_keyword(
        self, value, use_centroid
    ):
        options = _build_reader_options(True, None, waters_spectrum=value)
        assert options["use_centroid"] is use_centroid
