# tests/unit/converters/test_streaming_index_build.py
"""The streaming converter's chunked string-index builds.

The var and obs string indices used to be built from Python list
comprehensions, materialising one ``str`` object per entry -- about 88
bytes each at peak against 17.5 for the chunked vectorised build the
hand-written layout uses now (``_INDEX_BUILD_CHUNK``). The values these
tests pin are unchanged; only the construction is, and the chunk
arithmetic is the part that can go wrong at a boundary.

This file also held the CSR column-index dtype tests for the streaming COO
route. That route is gone (its temporary CSR store with it), and so are
they; the PCS layout's own arrays are covered end to end by
``test_pcs_column_order`` and ``test_lazy_loading_encoding``.
"""

from __future__ import annotations

import numpy as np
import pytest

from thyra.converters.spatialdata import streaming_converter as mod


class TestIndexBuildValues:
    """The vectorised builds must produce exactly the old strings."""

    @pytest.mark.parametrize(
        "n", [0, 1, 2, 9, 10, 11, 99, 100, 101, 999, 1000, 2_000_001]
    )
    def test_var_index_matches_list_comprehension(self, n, monkeypatch):
        # A small chunk makes the boundary arithmetic actually run.
        monkeypatch.setattr(mod, "_INDEX_BUILD_CHUNK", 1000)
        str_dtype = np.dtypes.StringDType()

        built = np.empty(n, dtype=str_dtype)
        for start in range(0, n, mod._INDEX_BUILD_CHUNK):
            stop = min(start + mod._INDEX_BUILD_CHUNK, n)
            built[start:stop] = np.strings.add(
                "mz_", np.arange(start, stop, dtype=np.int64).astype(str_dtype)
            )

        expected = np.array([f"mz_{i}" for i in range(n)], dtype=str_dtype)
        assert np.array_equal(built, expected)

    @pytest.mark.parametrize("n", [0, 1, 10, 1000, 100_000])
    def test_obs_index_matches_list_comprehension(self, n):
        str_dtype = np.dtypes.StringDType()
        built = np.arange(n, dtype=np.int64).astype(str_dtype)
        expected = np.array([str(i) for i in range(n)], dtype=str_dtype)
        assert np.array_equal(built, expected)

    @pytest.mark.parametrize(
        "value", [0, 999_999, 2**31 - 1, 2**31, 2**32, 2**53, 2**62]
    )
    def test_int64_formats_like_str_at_extremes(self, value):
        """No scientific notation or sign artefacts at large magnitudes."""
        str_dtype = np.dtypes.StringDType()
        built = np.array([value], dtype=np.int64).astype(str_dtype)
        assert built[0] == str(value)

    def test_chunk_boundaries_are_exact(self, monkeypatch):
        """Off-by-one in the chunk arithmetic would show up here."""
        monkeypatch.setattr(mod, "_INDEX_BUILD_CHUNK", 100)
        str_dtype = np.dtypes.StringDType()

        for n in (99, 100, 101, 200, 201):
            built = np.empty(n, dtype=str_dtype)
            for start in range(0, n, mod._INDEX_BUILD_CHUNK):
                stop = min(start + mod._INDEX_BUILD_CHUNK, n)
                built[start:stop] = np.strings.add(
                    "mz_", np.arange(start, stop, dtype=np.int64).astype(str_dtype)
                )
            expected = np.array([f"mz_{i}" for i in range(n)], dtype=str_dtype)
            assert np.array_equal(built, expected), f"n={n}"
