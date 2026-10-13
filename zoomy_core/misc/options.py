"""Typed options and the settings hierarchy: class defaults < settings-file section < arguments.

Every choice-valued setting is an `Option` (a str-Enum: the IDE completes it, a typo is answered
with "did you mean"); every configurable component is a `Configurable`, which reads its section of
a settings file and rejects unknown keys with a suggestion.
"""
import difflib
import json
import os
from enum import Enum

import param


def did_you_mean(word, candidates):
    """`"; did you mean 'x'?"` for the closest candidate, or `""`."""
    hit = difflib.get_close_matches(str(word), [str(c) for c in candidates], n=1, cutoff=0.5)
    return f"; did you mean {hit[0]!r}?" if hit else ""


class Option(str, Enum):
    """A typed choice: `Imex("SSP2")` is `Imex.ssp2` (member value or name, any case); a typo
    raises `ValueError` naming the closest member.  Members compare equal to their value.
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


class OptionParam(param.Selector):
    """A `param.Selector` over the members of one `Option` enum (or the subset `members`), plus
    `extra` plain values such as `False`; strings are coerced to members.
    """

    __slots__ = ["enum"]

    def __init__(self, enum=None, default=None, *, members=None, extra=(), **params):
        self.enum = enum
        objects = [*extra, *(members if members is not None else (enum or ()))]
        super().__init__(default=default, objects=objects, **params)

    def _coerce(self, val):
        if self.enum is not None and isinstance(val, str) and not isinstance(val, self.enum):
            return self.enum(val)
        return val

    def __set__(self, obj, val):
        super().__set__(obj, self._coerce(val))

    def _validate_value(self, val, *args, **kwargs):
        val = self._coerce(val)
        if self.enum is not None and isinstance(val, self.enum) and val not in self.objects:
            supported = [str(o) for o in self.objects]
            raise ValueError(f"{self.owner.__name__ if self.owner else ''}.{self.name}: "
                             f"{val.value!r} is not supported here (one of: {', '.join(supported)})")
        super()._validate_value(val, *args, **kwargs)


def load_settings(path):
    """Read a settings file (`.json`, `.yml` / `.yaml`) into a dict of sections."""
    with open(path) as f:
        if os.path.splitext(str(path))[1] in (".yml", ".yaml"):
            import yaml
            return yaml.safe_load(f) or {}
        return json.load(f)


def settings_section(settings, name):
    """Section `name` of `settings` (a dict, a Zstruct / `Settings`, or a file path) as a dict."""
    if settings is None or name is None:
        return {}
    if isinstance(settings, (str, os.PathLike)):
        settings = load_settings(settings)
    sec = settings.get(name) if isinstance(settings, dict) else getattr(settings, name, None)
    if sec is None:
        return {}
    return dict(sec.as_dict() if hasattr(sec, "as_dict") else sec)


class Configurable(param.Parameterized):
    """A component configured by class defaults < `settings[section]` < keyword arguments;
    an unknown key raises `TypeError` naming the closest setting.
    """

    section = None  # this component's section of a settings file, e.g. "solver"

    def __init__(self, settings=None, **kwargs):
        values = {**settings_section(settings, type(self).section), **kwargs}
        known = [n for n in type(self).param if n != "name"]
        for key in values:
            if key != "name" and key not in known:
                raise TypeError(f"{type(self).__name__} has no setting {key!r}"
                                f"{did_you_mean(key, known)}")
        super().__init__(**values)
