# 0008 — Land the projected_bernoulli + Gauss WB kernel (branch → default)

**What:** the production moving-WB kernel (audusse / bernoulli / projected_bernoulli
+ Gauss/exact quadrature) lives on a feature branch, not the `zoomy_core` default
line, so the committed thesis cases are not reproducible against default `zoomy_core`.

**Where:** `library/zoomy_core` branch `cstrong-opaque-derivative` (WB commits
`057508e`, `136bf92`, `762b443`, `37cdeb8`); then the Zoomy superproject
`zoomy_core` submodule pointer.

**How:** merge/rebase `cstrong-opaque-derivative` onto the `zoomy_core` default
branch, then bump the superproject submodule pointer on `develop`.
**CAUTION:** that branch also carries unrelated work from other threads
(`riemann_solvers.py`, `model/derivation/*`, `systemmodel/system_model.py` lazy
operators, `qr_kesme.py`) — do **not** sweep those in blind; coordinate with whoever
owns the C-strong/derivative thread before merging.

**Why:** the thesis `cases/hoern` scripts import `equilibrium_reconstruction=
'projected_bernoulli'`; without the kernel on the default line + the submodule
pointer bumped, a fresh checkout of the thesis + zoomy_core won't have it.

**Learned:** I staged only the WB files into my commits; the branch's working tree
had heavy unrelated dirty state that I deliberately left unstaged. The WB pieces
are self-contained (`fvm/bernoulli_wb.py`, `fvm/solver_numpy.py` hook,
`model/basemodel.py` Selector, `systemmodel` field, `postprocessing/{panels,style}`)
and could be cherry-picked if a clean WB-only merge is preferred over merging the
whole branch.
