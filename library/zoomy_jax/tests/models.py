"""The MODEL layer, cached. Tests do their own SystemModel and
NumericalSystemModel steps explicitly — the derivation chain must be visible
in the test, per user direction (v2 correction 3: there is NO ``swe_nsm``
convenience wrapper here, and there must never be one again).

Caching: the zoomy_core derivation cache makes ``SystemModel.from_model``
cheap on repeat (measured 0.12 s cold / 0.07 s warm for SME(0, dim=2));
``lru_cache`` here avoids rebuilding the Model object. BCs are declared ON
THE MODEL, so they are part of the cache key — hence the string ``bc``
argument (hashable).

NEVER ``no_cache()``: model correctness is owned by the core goldens
m01/m02/m10/m12/s02/n01/n02, which re-derive cache-free.

CACHE-SHARING (the trap the proposal flags, RESOLVED BY MEASUREMENT):
``SystemModel.from_model`` does NOT hand back a shared instance, so tests may
mutate ``initial_conditions`` freely and test order cannot leak.  Proof, not
inference — ``zoomy_core/systemmodel/sm_cache.py:146`` caches the pickled
BYTES and ``fetch`` returns ``pickle.loads(blob)``, a fresh object every call;
verified at runtime with ``sm1 is sm2 -> False`` and a mutation on ``sm1``
that did NOT appear on a subsequently built ``sm3``.  No ``copy.deepcopy`` is
therefore needed (and adding one would silently cost a full re-pickle per
test).

Closure-set note (REPORTED, not silently accepted): the proposal specifies
``[Newtonian(), StressFree()]`` for VAM and MLSME, matching the Escalante case
``run_derived.py``.  The core goldens m10/m12 use ``_std_closures()`` =
``[Newtonian(), NavierSlip(), StressFree()]``.  The two differ by NavierSlip,
so the exact VAM/MLSME closure compositions run here are NOT bit-for-bit the
ones the goldens pin.  Kept as specified (the Escalante physics is the point);
flagged so the coverage claim is not overstated.

Import style: bare (``from models import swe``), not ``from tests.models``.
The top-level package name ``tests`` is already taken by the superrepo
(``~/git/Zoomy/tests/__init__.py`` exists; this directory has no
``__init__.py``), so a ``tests`` package here would collide under a
superrepo-wide run.  ``conftest.py`` injects this directory onto
``sys.path``.  Module and symbol names are exactly the proposal's.
"""
from functools import lru_cache


@lru_cache(maxsize=None)
def swe(dimension: int = 2, bc: str = "extrapolation"):
    """SME(level=0) — the derived shallow-water Model.
    dimension=2 -> 1-D horizontal, dimension=3 -> 2-D horizontal."""
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import (
        ManningFriction, Newtonian, StressFree)
    from cases import bcs_for
    return SME(level=0, dimension=dimension,
               closures=[Newtonian(), ManningFriction(), StressFree()],
               boundary_conditions=bcs_for(bc, dimension))


@lru_cache(maxsize=None)
def vam(level: int = 1, dimension: int = 2, bc: str = "bump"):
    """VAM — the Escalante bump Model (core golden m12)."""
    from zoomy_core.model.models import VAM
    from zoomy_core.model.models.closures import Newtonian, StressFree
    from cases import bcs_for
    return VAM(level=level, dimension=dimension,
               closures=[Newtonian(), StressFree()],
               boundary_conditions=bcs_for(bc, dimension))


@lru_cache(maxsize=None)
def mlsme(n_layers: int = 2, level: int = 1, bc: str = "periodic"):
    """ML-SME — the AHS26 multilayer Model (core golden m10)."""
    from zoomy_core.model.models import MLSME
    from zoomy_core.model.models.closures import Newtonian, StressFree
    from cases import bcs_for
    return MLSME(level=level, n_layers=n_layers, dimension=2,
                 closures=[Newtonian(), StressFree()],
                 boundary_conditions=bcs_for(bc, 2))


def state_index(nsm, name: str) -> int:
    """Row index of a state BY NAME.  Never index a state row positionally:
    an unmapped row must RAISE, not silently resolve to something else (a
    positional default once made 12 of 16 amrex checks evaluate at zero
    momentum and still pass)."""
    names = [str(s) for s in nsm.state]
    if name not in names:
        raise KeyError(f"state {name!r} not in {names}")
    return names.index(name)
