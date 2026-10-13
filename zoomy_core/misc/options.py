"""Typed options: every choice is an `Option` (a str-Enum), held on a `param.Parameterized` by an
`OptionParam`.  A plain string where a member belongs raises `TypeError` naming the member, and
`Configurable` answers a misspelt argument name with "did you mean".
"""
import difflib
from enum import Enum

import param


def did_you_mean(word, candidates):
    """`"; did you mean 'x'?"` for the closest candidate, or `""`."""
    hit = difflib.get_close_matches(str(word), [str(c) for c in candidates], n=1, cutoff=0.5)
    return f"; did you mean {hit[0]!r}?" if hit else ""


class Option(str, Enum):
    """A typed choice; members compare equal to their value.  `Imex("SSP2")` converts a string
    explicitly (member value or name, any case); a typo raises `ValueError` naming the closest
    member.
    """

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"{type(self).__name__}.{self.name}"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            key = value.lower()
            for m in cls:
                if key in (m.value.lower(), m.name.lower()):
                    return m
        values = [m.value for m in cls]
        raise ValueError(f"{value!r} is not a valid {cls.__name__}{did_you_mean(value, values)} "
                         f"(one of: {', '.join(values)})")


def typed(enum, value, name, allowed=None):
    """`value`, unless it is a plain string where a member of `enum` belongs: then `TypeError`
    naming the member (`projection='increment': use Projection.increment`), or for a typo the
    closest of `allowed` (default: all members) and the list.
    """
    if not isinstance(value, str) or isinstance(value, enum):
        return value
    allowed = list(enum) if allowed is None else list(allowed)
    listing = ", ".join(repr(a) for a in allowed)
    key = value.lower()
    hit = next((m for m in enum if key in (m.value.lower(), m.name.lower())), None)
    if hit is not None and hit in allowed:
        raise TypeError(f"{name}={value!r}: use {hit!r}")
    if hit is not None:
        raise TypeError(f"{name}={value!r}: {hit!r} is not supported here (one of: {listing})")
    members = [a for a in allowed if isinstance(a, enum)]
    words = {w: m for m in members for w in (m.value, m.name)}
    close = difflib.get_close_matches(value, list(words), n=1, cutoff=0.5)
    if close:
        raise TypeError(f"{name}={value!r}: did you mean {words[close[0]]!r}? (one of: {listing})")
    raise TypeError(f"{name}={value!r}: use one of {listing}")


class OptionParam(param.Selector):
    """A `param.Selector` over the members of one `Option` enum (or the subset `members`), plus
    `extra` plain values such as `False`; a plain string raises `TypeError` (`typed`).
    """

    __slots__ = ["enum"]

    def __init__(self, enum=None, default=None, *, members=None, extra=(), **params):
        self.enum = enum
        objects = [*extra, *(members if members is not None else (enum or ()))]
        super().__init__(default=default, objects=objects, **params)

    def _validate_value(self, val, *args, **kwargs):
        if self.enum is not None:
            typed(self.enum, val, self.name, self.objects)
            if isinstance(val, self.enum) and val not in self.objects:
                supported = ", ".join(repr(o) for o in self.objects)
                raise ValueError(f"{self.name}={val!r} is not supported here "
                                 f"(one of: {supported})")
        super()._validate_value(val, *args, **kwargs)


class Configurable(param.Parameterized):
    """A `param.Parameterized` whose constructor answers an unknown argument with "did you mean"."""

    def __init__(self, **kwargs):
        known = [n for n in type(self).param if n != "name"]
        for key in kwargs:
            if key != "name" and key not in known:
                raise TypeError(f"{type(self).__name__} has no setting {key!r}"
                                f"{did_you_mean(key, known)}")
        super().__init__(**kwargs)
