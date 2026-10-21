"""Exceptions used by the symbolic primitive layer.

A primitive is a single, well-defined mathematical operation.  When the
caller invokes one whose preconditions don't match the input, the
primitive raises `PrimitiveDoesNotMatch` rather than silently
no-op'ing or — worse — guessing.  This is the explicit failure mode
the redesign is built around: the caller must compose primitives with
matching preconditions, and any mistake becomes a loud exception
rather than a quiet wrong-form result.
"""


class PrimitiveDoesNotMatch(Exception):
    """The primitive's structural pattern did not match the input.

    Carries the primitive name and the (sympy) atom that failed the
    match so the caller can localise the error.
    """

    def __init__(self, primitive: str, atom, reason: str = ""):
        self.primitive = primitive
        self.atom = atom
        self.reason = reason
        message = f"{primitive} did not match: {atom!r}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)
