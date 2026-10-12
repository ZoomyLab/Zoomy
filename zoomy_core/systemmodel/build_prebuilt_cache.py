"""Regenerate the shipped ``SystemModel`` cache (``python -m
zoomy_core.systemmodel.build_prebuilt_cache``).
Writes the pickled default models into ``zoomy_core/systemmodel/_prebuilt/`` so a fresh install
builds instantly.
"""
from __future__ import annotations

import os
import pickle
import time


def default_models():
    """Yield ``(label, factory)`` for the default model configurations that ship prebuilt."""
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.model.models.sme import SME
    from zoomy_core.model.models.vam import VAM
    from zoomy_core.model.models.ml_swe import MLSWE
    from zoomy_core.model.models.ml_sme import MLSME
    from zoomy_core.model.models.ml_vam import MLVAM
    from zoomy_core.model.models.closures import (
        Newtonian, NavierSlip, StressFree)
    clo = [Newtonian(), NavierSlip(), StressFree()]
    yield "swe-1d", lambda: SWE(dimension=1)
    yield "swe-2d", lambda: SWE(dimension=2)
    for lvl in (0, 1, 2):
        yield f"sme-l{lvl}-2d", lambda lvl=lvl: SME(
            level=lvl, dimension=2, closures=list(clo))
    for lvl in (1, 2):
        yield f"vam-l{lvl}-2d", lambda lvl=lvl: VAM(closures=list(clo), level=lvl, dimension=2)
    # 3-D (two-horizontal, t,x,y,z) structural specs.  These are built COLD by
    # default-tier structural tests (``test_sme_2d`` SME(dim=3); ``test_vam_2d``
    # + the ``fvm`` elliptic-BC / wet-dry Chorin tests VAM(dim=3)) — ~10-20 s
    # each uncached, which is the residual default-tier floor.  Ship them so the
    # tests hit the warm cache.  Closures/parameters MUST match the test spec
    # verbatim (they are part of ``model_spec_key``; parameter VALUES are not).
    for lvl in (1, 2):
        yield f"sme-l{lvl}-3d", lambda lvl=lvl: SME(
            level=lvl, dimension=3, parameters={"nu": 0.1, "lambda_s": 0.5},
            closures=[Newtonian(), NavierSlip(), StressFree()])
    yield "vam-l1-3d-navierslip", lambda: VAM(
        level=1, dimension=3,
        closures=[Newtonian(), NavierSlip(), StressFree()])
    yield "vam-l1-3d-newtonian", lambda: VAM(
        level=1, dimension=3, closures=[Newtonian(), StressFree()])
    yield "mlswe-2d", lambda: MLSWE(dimension=2, closures=list(clo))
    yield "mlsme-2d", lambda: MLSME(dimension=2, closures=list(clo))
    yield "mlvam-2d", lambda: MLVAM(dimension=2, closures=list(clo))
    # Sigma3D is NOT cacheable: its resolved bed/surface BCs hold a
    # ``_lambdifygenerated`` closure, which pickle cannot address by name, so
    # every attempt raises PicklingError.  Left out deliberately rather than
    # failing this build on every run.
    # KESME / QRKESME are left out for now: neither can currently be built
    # CLOSED.  KESME still reaches the SystemModel with sigma_1..3 unbound
    # even under KEpsilonViscosity, and QRKESME raises "no attribute
    # 'derivation'" once closures are passed.  Both are defects in those two
    # classes rather than in this file (coordd cid 207); caching them is
    # blocked until they are fixed.


def main() -> int:
    """Build every default model from scratch (cache reads off) and pickle it to the prebuilt
    directory; returns 0 on full success.
    """
    os.environ["ZOOMY_DERIVATION_REBUILD"] = "1"     # always build fresh
    from zoomy_core.systemmodel import sm_cache
    from zoomy_core.systemmodel.model_builders import _BUILDERS, build_system_model

    out = sm_cache._prebuilt_dir()
    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    from zoomy_core.systemmodel.system_model import allow_unclosed
    import contextlib
    import warnings
    for label, make in default_models():
        try:
            t0 = time.time()
            model = make()
            # Sigma3D is deliberately open (see default_models); everything
            # else must be closed, and a build that trips the guard is a spec
            # bug in this file, not something to wave through.
            with (allow_unclosed() if label == "sigma3d"
                  else contextlib.nullcontext()):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*unclosed term")
                    sm = build_system_model(model)
            key = sm_cache.cache_key(model, _BUILDERS[model._system_model_kind])
            (out / f"{key}.pkl").write_bytes(pickle.dumps(sm))
            print(f"  {label:14s} -> {key[:12]}…  ({time.time()-t0:.1f}s)")
            n_ok += 1
        except Exception as exc:
            print(f"  {label:14s} FAILED: {type(exc).__name__}: {str(exc)[:100]}")
            n_fail += 1
    print(f"prebuilt cache: {n_ok} built, {n_fail} failed -> {out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
