"""Model tags (kind of system) from which the numerics pick defaults, through `TAG_RULES`.
Sits between `SystemModel` (carries the tags in `history`) and `NumericalSystemModel`
(applies the rules).
"""
from __future__ import annotations

from enum import Enum

TAG_RECORD = "declare_tags"


class ModelTag(str, Enum):
    """Kinds of systems the numerics distinguish; a `str` enum, so `"free_surface"` works too."""
    #: depth-averaged free-surface flow: a depth that must stay >= 0, a bed,
    #: wet/dry fronts (hydrostatic reconstruction, positivity)
    free_surface = "free_surface"


def declare_tags(sm, tags, description=None):
    """Record `tags` on `sm.history` and return `sm`."""
    tags = sorted({ModelTag(t).value if t in ModelTag._value2member_map_
                   else str(t) for t in tags})
    sm.history.append({"name": TAG_RECORD, "tags": tags,
                       "description": description or "model tags"})
    return sm


def declared_tags(sm) -> frozenset:
    """All tags recorded on `sm` (union over its history)."""
    out = set()
    for rec in getattr(sm, "history", None) or []:
        if isinstance(rec, dict) and rec.get("name") == TAG_RECORD:
            out |= set(rec["tags"])
    return frozenset(out)


def _rules():
    """Default tag -> numerics table; built lazily because the flux classes import late."""
    from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
    return {ModelTag.free_surface.value: {"riemann": PositiveNonconservativeRusanov}}


#: tag -> numerics defaults; filled lazily (the flux classes import late)
TAG_RULES: dict = {}


def tag_rules() -> dict:
    """The tag -> numerics-defaults table `TAG_RULES`, filled on first use."""
    if not TAG_RULES:
        TAG_RULES.update(_rules())
    return TAG_RULES


def register_tag_rule(tag, **defaults):
    """Add or replace the numerics defaults of `tag`."""
    tag_rules()[getattr(tag, "value", str(tag))] = dict(defaults)


def tag_defaults(sm, rules=None) -> dict:
    """Numerics defaults implied by the tags of `sm`; `rules` overrides `TAG_RULES` per tag.
    Raises `ValueError` when two tags imply different values for one key.
    """
    rules = {**tag_rules(), **{getattr(k, "value", str(k)): v for k, v in (rules or {}).items()}}
    out, src = {}, {}
    for tag in sorted(declared_tags(sm)):
        for key, val in rules.get(tag, {}).items():
            if key in out and out[key] is not val:
                raise ValueError(f"tags {src[key]!r} and {tag!r} imply "
                                 f"different {key!r}: pass it explicitly")
            out[key], src[key] = val, tag
    return out


__all__ = ["ModelTag", "declare_tags", "declared_tags", "register_tag_rule",
           "tag_rules", "tag_defaults", "TAG_RULES"]
