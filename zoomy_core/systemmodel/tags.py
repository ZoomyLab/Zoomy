"""Model TAGS: what kind of system a model is, so the numerics can pick
defaults from it instead of from one hard-wired rule.

A tag is a :class:`ModelTag` member.  A model states its tags (class or
instance attribute ``tags``), or a user declares them on a SystemModel
(:func:`declare_tags`); declaring free-surface roles implies
``FREE_SURFACE``.  Tags live in ``sm.history`` like the roles, so every copy
and every Chorin split carries them.

The NumericalSystemModel turns tags into defaults through RULES
(:data:`TAG_RULES`, :func:`register_tag_rule`, or ``tag_rules=`` per call):
``{tag: {"riemann": cls, ...}}`` -- applied only to what the caller left
unset.  A new kind of system gets a new tag and a rule, nothing else.
"""
from __future__ import annotations

from enum import Enum

TAG_RECORD = "declare_tags"


class ModelTag(Enum):
    """Kinds of systems the numerics distinguish."""
    #: depth-averaged free-surface flow: a depth that must stay >= 0, a bed,
    #: wet/dry fronts (hydrostatic reconstruction, positivity)
    FREE_SURFACE = "free_surface"


def declare_tags(sm, tags, description=None):
    """Record ``tags`` (ModelTag members) on ``sm``.  Returns ``sm``."""
    tags = frozenset(tags)
    bad = [t for t in tags if not isinstance(t, ModelTag)]
    if bad:
        raise TypeError(f"tags must be ModelTag members, got {bad}")
    sm.history.append({"name": TAG_RECORD, "tags": sorted(t.name for t in tags),
                       "description": description or "model tags"})
    return sm


def declared_tags(sm) -> frozenset:
    """All tags recorded on ``sm`` (union over its history)."""
    out = set()
    for rec in getattr(sm, "history", None) or []:
        if isinstance(rec, dict) and rec.get("name") == TAG_RECORD:
            out |= {ModelTag[n] for n in rec["tags"]}
    return frozenset(out)


def _rules():
    from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
    return {ModelTag.FREE_SURFACE: {"riemann": PositiveNonconservativeRusanov}}


#: tag -> numerics defaults; filled lazily (the flux classes import late)
TAG_RULES: dict = {}


def register_tag_rule(tag, **defaults):
    """Add or replace the numerics defaults of ``tag``."""
    if not isinstance(tag, ModelTag):
        raise TypeError(f"{tag!r} is not a ModelTag")
    tag_rules()[tag] = dict(defaults)


def tag_rules() -> dict:
    if not TAG_RULES:
        TAG_RULES.update(_rules())
    return TAG_RULES


def tag_defaults(sm, rules=None) -> dict:
    """The numerics defaults the tags of ``sm`` imply (``rules`` overrides
    :data:`TAG_RULES` per tag).  Conflicting defaults are an error."""
    rules = {**tag_rules(), **(rules or {})}
    out, src = {}, {}
    for tag in sorted(declared_tags(sm), key=lambda t: t.name):
        for key, val in rules.get(tag, {}).items():
            if key in out and out[key] is not val:
                raise ValueError(f"tags {src[key].name} and {tag.name} imply "
                                 f"different {key!r}: pass it explicitly")
            out[key], src[key] = val, tag
    return out


__all__ = ["ModelTag", "declare_tags", "declared_tags", "register_tag_rule",
           "tag_rules", "tag_defaults", "TAG_RULES"]
