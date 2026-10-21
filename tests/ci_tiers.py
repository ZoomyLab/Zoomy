"""CI tiers: the markers `large`, `legacy` and `broken`, applied by test id.

The ids live in one table, `tests/ci_tiers.tsv` (tier, test id, note), so no test file is edited.
Every conftest that can be a pytest root (the superrepo, `library/zoomy_core/tests`,
`library/zoomy_jax/tests`) registers this file as a plugin in `pytest_configure`, so the markers
are the same whichever directory pytest starts in. Ids are relative to the repository root.

    large   passing, over the push time budget: weekly tier only
    legacy  exercises superseded code: weekly tier only
    broken  fails today, `reason=` is the first error line: in no tier

The tiers select with `-m`, see `docs/book/developer/testing.md`.
"""
import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parent
TABLE = _HERE / "ci_tiers.tsv"


def load(table=TABLE):
    """`{test id: [(tier, note), ...]}` from the table."""
    by_id = {}
    if not table.is_file():
        return by_id
    for line in table.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tier, test_id, *note = line.split("\t")
        by_id.setdefault(test_id, []).append((tier, note[0] if note else ""))
    return by_id


def repo_id(item):
    """The node id of `item` relative to the repository root."""
    try:
        rel = pathlib.Path(str(item.path)).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return None
    return "::".join([rel, *item.nodeid.split("::")[1:]])


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    table = load()
    if not table:
        return
    for item in items:
        for tier, note in table.get(repo_id(item), ()):
            if tier == "broken":
                item.add_marker(pytest.mark.broken(reason=note))
            elif tier in ("large", "legacy"):
                item.add_marker(getattr(pytest.mark, tier))
