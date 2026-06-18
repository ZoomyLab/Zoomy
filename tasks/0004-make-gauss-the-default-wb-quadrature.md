# 0004 — Make Gauss the default WB reconstruction quadrature + regenerate figures

**What:** flip the `projected_bernoulli` (and `bernoulli`) reconstruction quadrature
from `trapezoid` to `gauss` by default, and regenerate the L0–8 thesis figures so
the committed figures match the default.

**Where:**
- `library/zoomy_core/zoomy_core/fvm/bernoulli_wb.py` — `build_bernoulli_config(...,
  quadrature="trapezoid", ...)` → default `"gauss"`.
- `thesis/cases/hoern/bbsm13_nobackflow_figure.py` — re-run to refresh
  `bbsm13_nobackflow_figure.png` / `bbsm13_recirc_figure.png` (the Verification
  chapter embeds them).

**How:** one-line default change. Then a full re-run (~88 min, detached `setsid
nohup`, 48 workers) — `--replot` will NOT do, the underlying data changes. The
no-backflow convergence panel will then ride the projection floor through ~L6
instead of flattening at 1e-6; recirc is unchanged (Gibbs ceiling). Confirm
`tests/fvm/test_equilibrium_wb.py` still passes (it asserts the *audusse* hook,
not the quadrature).

**Why:** Gauss is strictly better — `n=2(L+1)` (2–18) nodes, spectral, tracks the
floor through L6, 15–37× cheaper than the 200-pt trapezoid. Kept opt-in only to
keep the already-committed figures byte-identical at commit time.

**Learned:** Gauss uses a spectral cumulative-weight matrix for the σ*(σ) remap
(O(n) integrand evals). A secondary ~1e-9 floor (NOT quadrature) then caps L7–8
(see task 0006). The kernel change is committed on `zoomy_core` branch
`cstrong-opaque-derivative` (37cdeb8); see task 0008 about landing it.
