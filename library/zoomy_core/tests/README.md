# zoomy_core test suite (approved refactor, 2026-07-19 final v3)

Two layers guard the core:

1. **Goldens** — 25 checked-in text snapshots under `tests/goldens/`
   (16 model + 3 systemmodel + 3 NSM + 2 printer + 1 numpy solver), verbatim
   normalized `describe()`/emit of the DERIVED systems (SWE = SME(level=0)
   composition, never hand-built; model goldens derive **no-cache**).
   Regenerate with `python scripts/regen_goldens.py`; review = `git diff`.
   A regen touching more than one family requires the rederive tier green
   first (**re-bless protocol** — a golden detects CHANGE, not WRONGNESS;
   `tests/model/test_model_references.py` is the truth anchor).
2. **Exceptions** — 8 runtime-semantic files + 1 IMEX smoke pinning what
   goldens are structurally blind to (inverse maps, cache staleness, dry-state
   meaning, runtime BCs, MOOD positivity, kernel values, VAM→SME limit).

## Markers

| marker | meaning |
|---|---|
| `gate` | T1 membership: 6 model goldens + 7 structure/emit/solver + 24 runtime smalls (~37 fns) |
| `small` / `large` | size tags; `large` = real marches, deselected by default |
| `rederive` | cold-cache / fresh-derivation truth checks; deselected by default |
| `model` `systemmodel` `nsm` `printer` `solver` | area tags (verify.py scoping) |
| `fusion_wip` | transitional REQ-188 fusion seam (test_resolve_opaque) |
| `study` | parked REQ-194 study scaffolding — excluded from EVERY tier, only `-m study` runs it |

All markers are registered in `pyproject.toml`; nothing may warn.

## Tiers / commands

```bash
# T1 gate, scoped to the areas your diff touches (fallback: all areas)
micromamba run -n zoomy python scripts/verify.py
micromamba run -n zoomy python scripts/verify.py --all-areas   # full gate, < 5 min

# T2 feature tier: ALL small + large + rederive (golden re-bless validator)
micromamba run -n zoomy python scripts/verify.py --tier feature

# raw pytest equivalents
pytest tests/ -q -m gate                          # full gate
pytest tests/ -q -m "gate and (model or nsm)"     # scoped gate
pytest tests/ -q                                  # all smalls (default tiering)
pytest tests/ -q --run-large --run-rederive       # T2 by hand
pytest tests/ -q -m rederive                      # rederive tier alone
```

T3 (full) = T2 + backend regression suites + container integration — lives
outside this repo; run only after discussing with the user.

## Cache expectations

Non-golden tests assume a WARM derivation cache (shipped `_prebuilt` +
REQ-163 `sm_cache`).  Model goldens always derive fresh (no-cache) by spec.
After a `CACHE_VERSION` bump or derivation/builder edit:

```bash
python -m zoomy_core.systemmodel.build_prebuilt_cache
```

## Time canary

`tests/numerics/test_nsm_goldens.py::test_n02_time_canary` compares ONE
CPU-time run of the N02 derive+build+lower seam against
`tests/goldens/n02_time_baseline.json` (threshold 1.5x; catches the ~7x
factor_terms blowup class).  Re-baseline via
`python scripts/regen_goldens.py --baseline` (auto-refreshed on a full regen);
the canary skips on hosts other than the one that recorded the baseline.
