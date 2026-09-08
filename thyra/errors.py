"""The exception a refusal is spelled with.

Thyra refuses a great deal on purpose: a ``.d`` directory with no
analysis files, a Waters raster whose stage never moved, a ``tof`` axis
with no width law, an output path that already holds a store. Each of
those refusals is a sentence someone wrote for the person who hit it,
and each used to reach that person as a thirty-line traceback --
``convert_msi`` caught every exception alike and logged the message
*and* the traceback at ERROR, so a refusal Thyra planned for was
presented exactly like a crash it did not (issue #234).

:class:`ConversionRefused` is how the two are told apart. Raising it
says "this is the whole explanation": the CLI prints the message once at
ERROR and keeps the traceback for ``--log-level DEBUG``. Anything else
reaching the same handler is unexpected, and keeps its traceback at
ERROR, because for those the traceback *is* the explanation.

It subclasses :class:`ValueError` because every one of these sites
raised a ``ValueError`` before, and callers -- Thyra's own tests
included -- catch that. Nothing that used to work stops working.

The distinction is about who the message is for, not about where it is
raised. An internal invariant ("Common mass axis is not initialized")
stays a plain ``ValueError``: nobody can act on it, and its traceback is
the only useful part.
"""


class ConversionRefused(ValueError):
    """Thyra declined to do something, and the message says why.

    The message is addressed to the person who ran the conversion and is
    expected to name what to do about it. No traceback is shown for it
    at the default log level.
    """


__all__ = ["ConversionRefused"]
