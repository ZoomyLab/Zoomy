# 0009 — VAM Chorin-split predictor drops the non-x conservative flux (dim≥3 mass leak / over-fill)

**What:** The VAM dim=3 bend over-fills — the outflow settles at h≈0.093 m
instead of the prescribed 0.061 m — because the Chorin-split **predictor** is
not mass-conservative on the 2-D mesh. Its mass equation keeps only the
x-direction flux; the y-direction flux is mis-routed into the *source* term and
therefore does not telescope. Closed-domain (all-wall) dam break leaks ~1.9 %
of total mass over t≈10 s and growing (should be round-off).

**Where:**
- `library/zoomy_core/zoomy_core/model/derivation/tag_extraction.py` —
  `_classify_term` (≈ line 360–378) and its caller `auto_solver_tag`. **The
  classifier is hard-coded to a single spatial coordinate `x`:** line ~365,
  `if len(deriv.variables) != 1 or deriv.variables[0] != x: return
  "implicit_source"`. So `∂_y(q_y_0)` (any non-`x` derivative) is tagged as a
  source, not a flux.
- `library/zoomy_core/zoomy_core/model/splitter.py` —
  `split_for_pressure_structural` builds the predictor `SM_pred` via
  `_build_subsystem(..., source_only=False)` (≈ line 835), which **re-derives**
  flux/NCP/source from the pressure-free residuals by calling `auto_solver_tag`
  from scratch (≈ line 393). That re-derivation is where the loss happens.

**How (two options; (b) preferred — it is what the user asked for):**
- **(a) Make the tagger N-D.** `auto_solver_tag`/`_classify_term` take a single
  `x`; generalise to the full `coords` list and treat a derivative w.r.t. *any*
  coord as flux/NCP. NOTE the *extractors* are already N-D —
  `collect_solver_tag` → `_extract_conservative`/`_extract_nc` use
  `_first_order_direction(deriv, coords)` and place `F_i` in the right column —
  so only the classifier is 1-D. Also touches `basemodel._auto_tag_equations`
  (passes `x_sym = coords[0]`), the `from_model` path for *untagged* equations.
- **(b) Don't re-classify in the predictor at all.** The parent `SystemModel`
  already has correct, all-direction `flux`/`nonconservative_matrix`/`source`
  (the unsplit VAM mass row is `F=[q_x_0, q_y_0]`, `B=0`). Have `_build_subsystem`
  (predictor branch) **inherit** the parent operators for the evolution rows and
  merely remove the pressure contribution (substitute pressure modes → 0 in
  F/B/S and zero the pressure columns of B), instead of re-tagging from scratch.
  This is structurally faithful and removes the fragile heuristic from the hot
  path. The corrector already does NOT re-classify (it is built from
  `state_update`), so this only changes the predictor (and, analogously, the
  `source_only=True` pressure stage already bypasses the tagger).

**Why:** This is THE root cause of the long-standing VAM outflow over-fill and
the reason adding moments never reproduced the secondary circulation cleanly on
the bend — the predictor was leaking mass every step at the open outflow. SME
does not show it because SME runs the unsplit `from_model` SystemModel through
the `HyperbolicSolver` (explicit, correct tags), never the split predictor.

**Learned (this session):**
- **Last finding / smoking gun:** with the partial fix in place, the dim=3
  predictor mass row is `pred_h`: `flux F = [q_x_0, 0]`, `B = []`,
  `source = -q_y_0_y`. The y-flux `∂_y(q_y_0)` is sitting in the SOURCE. Verify
  with: `make_model(model='vam', level=1, dimension=3).chorin_split(...)` then
  inspect `SM_pred.flux` / `.source` for the `pred_h` row.
- A **partial fix already landed** — zoomy_core `b95d6dd` "tag_extraction: bare
  d_x(state) with state-free coeff is a conservative flux, not NCP". It corrects
  the *first* bug (constant-coeff bare-state `∂_x(q_0)` was being tagged
  `nonconservative_flux` instead of `flux`; `_classify_term` now splits on
  whether the coefficient references state). That fixes the **x-direction** only.
  The N-D loss above is still open and is what (a)/(b) must address.
- The bug is **invisible in 1-D / structured grids**: a constant-coefficient NCP
  product still telescopes there. The 1-D closed-box gate
  (`library/zoomy_core/_mass_gate.py`) gives machine-zero drift **with or
  without** the fix — it does NOT discriminate. Use the **dim=3 unstructured**
  gate (`thesis/notebooks/steffler_jax/_mass_gate_dim3.py`) — multi-directional
  NCP does not telescope there, so the leak shows.
- Only the **predictor** (and the elliptic/pressure stage) re-tag; the
  **corrector** does not (it is `state_update`). So the user's instinct ("the
  splitter should not re-classify") is correct and points straight at (b).

**Acceptance gate:**
1. `thesis/notebooks/steffler_jax/_mass_gate_dim3.py` — all-wall closed-mesh dam
   break: relative mass drift → round-off (currently 1.9e-2 and growing).
2. Then the open steffler VAM run (`steffler_vam_jax.py`) pins the outflow depth
   at ≈0.061 m instead of 0.093 m (→ unblocks task 0011).
3. Re-run the SME + VAM model test suite (`tests/model/test_vam_*`,
   `test_sme_*`) — all green (38 tests passed with the partial fix).
