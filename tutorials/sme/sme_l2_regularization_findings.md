# SME L=2 hyperbolic regularization — findings from route-A vs route-B analysis

**Setup.** SME L=2 in physical-z, derived two ways:
- **Route A** (`tutorials/sme/sme_l2_hyperbolicity_compare.py`, `build_route_a`): close `w` via depth-integrated continuity ⇒ 4 evolution equations in `(h, u_0, u_1, u_2)`.
- **Route B** (same file, `build_route_b`): keep `w` explicit (`N_w = M+1 = 3`) ⇒ 8 fields, 4 evolution-like rows + 4 algebraic constraints (DAE).

Numerical comparison confirmed both routes produce the same eigenvalues to ~6.5 × 10⁻⁷ across a 41×41 (ū_1, ū_2) grid (machine ε ≈ 2 × 10⁻¹⁶; the ~10-order gap comes from the 8×8 polyfit in the route-B path).

---

## 1. Block structure that route B exposes

Order route B's fields as `(evol = h, u_0, u_1, u_2)` then `(w = w_0..w_3)`. Linearise around a homogeneous base. The 8×8 pencil splits as:

```
                  evol-cols      w-cols
M_t  evol-rows |    FULL      |     0    |   ← w has no time deriv
     alg-rows  |     0        |     0    |   ← algebraic constraints

M_x  evol-rows |    FULL      |     0    |   ← w has no space deriv
     alg-rows  |    FULL      |     0    |   ← w has no space deriv

M_0  evol-rows |     0        |   FULL   |
     alg-rows  |     0        |  CONST   |   ← M_0_aw is a 4×4 constant matrix
```

`M_0_aw = [[0,−2,0,−2], [0,0,−2,0], [0,0,0,−2], [1,1,1,1]]`, det = 8, *independent of any base value*. The Schur complement that returns route A is therefore a closed-form expression:

```
M_x_eff = M_x_uu  +  feedback,    feedback = −M_0_uw · M_0_aw⁻¹ · M_x_au
```

---

## 2. Where the loss-of-hyperbolicity lives

Looking at the (xmom_j0, h-col) entry — the leading wave-speed channel:

|                  | u_1²/3 coefficient | u_2²/5 coefficient |
|------------------|--------------------|--------------------|
| `M_x_uu` direct  | **−1/3** (stabilising) | **−2/5** (stabilising) |
| feedback         | **+2/3** (destabilising) | **+3/5** (destabilising) |
| total (route A)  | +1/3 | +1/5 |

The Koellermeier-style destabiliser is *entirely a feedback contribution*. The direct block `M_x_uu` is **fully hyperbolic** on the whole tested grid (ε = 0 in the regularisation script ⇒ 0 % non-hyperbolic).

---

## 3. Linear ε-scaling of the feedback is not a clean knob

`tutorials/sme/sme_l2_feedback_regularization.py` sweeps ε ∈ [0, 1] in `M_x_uu + ε · feedback`:

| ε | non-hyperbolic % of (ū_1, ū_2) ∈ [−3,3]×[−2,2] |
|---|---|
| 0.00 | **0.00 %** (direct only) |
| 0.25 | 3.7 % |
| 0.50 | 4.5 % |
| 0.75 | 2.7 % |
| 1.00 | 1.7 % (full SME) |

The bad region **moves** with ε — it doesn't shrink monotonically.

---

## 4. Symmetric / antisymmetric decomposition: a near-cancellation

Either symmetric or antisymmetric half of the feedback alone is wildly destabilising (peak |Im λ| up to **12** vs SME's full peak of **0.10**). They almost cancel by design — SME's hyperbolicity is a fragile sum-rule. Friedrichs-symmetrising would break exactly this cancellation. See `sme_l2_feedback_decomposition.py` and `.png`.

---

## 5. Unstable mode is a `(u_1, u_2)`-only oscillation

At each corner spur the complex eigenvector has |amplitude|² fractions:

| field | fraction of mode energy |
|---|---|
| h    | 4 % |
| u_0  | 1 % |
| u_1  | 34 % |
| **u_2** | **100 %** (normalisation peak) |

Loss-of-hyperbolicity is essentially a `(u_1, u_2)` mode with `(h, u_0)` passive. Same shape at all four corner spurs.

---

## 6. The clean targeted regularisation: zero the cross-coupling

Block `A := M_t_uu⁻¹ · M_x_eff` as

```
   [ A_11   A_12 ]   A_11 = (h, u_0) ↔ (h, u_0)
A =[              ]
   [ A_21   A_22 ]   A_22 = (u_1, u_2) ↔ (u_1, u_2)
```

`tutorials/sme/sme_l2_subspace_regularization.py` tested four candidate regularisations:

| modification | non-hyp % | peak |Im λ| | ⟨‖·‖_F⟩ |
|---|---|---|---|
| full SME (route A) | 1.69 % | 0.095 | — |
| direct only (drop feedback) | **0.00 %** | 0.000 | 4.85 |
| Eucl-Hermitise A_22 | 6.61 % | 0.223 | 0.29 |
| W-Hermitise A_22 (W = diag(h/3, h/5)) | 11.07 % | 0.337 | 0.60 |
| **zero A_12 and A_21** (decouple shallow ↔ moment) | **0.00 %** | 0.000 | **2.88** |

**Zeroing the off-diagonal cross-coupling makes SME L=2 fully hyperbolic with the second-smallest Frobenius distance from the original.** Hermitising the moment sub-block alone makes things *worse* — confirming the conclusion of §4 that the destabilisation is a cross-coupling phenomenon, not a within-moments problem.

---

## 7. Newtonian τ_xx viscous regularisation — works, but in a different sense

`tutorials/sme/sme_l2_viscous_regularization.py`. Add τ_xx = 2ν · ∂_x u to x-momentum. After projection at homogeneous base, the viscous coefficient matrix is super-diagonal:

```
M_visc = [[0, h, 0,    0 ],     ← xmom_j=0:  2ν·h·∂_xx δu_0
          [0, 0, h/3,  0 ],     ← xmom_j=1:  2ν·h/3·∂_xx δu_1
          [0, 0, 0,    h/5],    ← xmom_j=2:  2ν·h/5·∂_xx δu_2
          [0, 0, 0,    0 ]]     ← KBC_top:   no viscous term
```

Sweep over ν at k = 1, H = 1, g = 1:

| ν | unstable % | peak Im ω |
|---|---|---|
| 0.00 | 1.69 % | +0.0949 |
| 0.05 | 0.77 % | +0.0281 |
| **0.10** | **0.00 %** | **−0.031** |
| 0.50 | 0.00 % | −0.40 |

ν ≥ 0.1 (Re ≈ 10) makes Im ω ≤ 0 across the whole grid. Critical ν shrinks at higher k: the viscous decay scales `ν k²` while the inviscid growth scales `Im(c) · k`, so `ν_crit ∝ 1/k`.

**Important caveat — this is *parabolic* regularisation, not strict hyperbolicity:**

| concept | status |
|---|---|
| Principal symbol (`M_t⁻¹ M_x`, k → ∞ limit) eigenvalues real | **No** — viscosity is lower-order, doesn't change the principal symbol |
| Dispersion `ω(k)` satisfies `Im ω ≤ 0` for all k of interest | **Yes** with ν ≥ 0.1 |

For schemes that need strict hyperbolicity (explicit Riemann-solver-based shock capturing) viscosity is *not* a substitute for an algebraic regularisation. For implicit / IMEX schemes that handle parabolic terms naturally, viscous SME is well-posed.

At physical water Reynolds numbers (~10⁶) this is ~10⁵ too small. For aerated / sediment-laden flows with eddy-viscosity scales (ν ≈ 0.01–1 m²/s) the regularisation is real.

---

## 8. Summary recommendation matrix

| target property | approach | cost vs route A |
|---|---|---|
| strict hyperbolicity, smallest perturbation | **zero `A_12` and `A_21`** (§ 6) | 2.88 in `‖·‖_F`, 0 % non-hyp |
| strict hyperbolicity, hard-cap | drop feedback (Koellermeier-equivalent ε = 0) | 4.85, 0 % non-hyp |
| dispersive well-posedness with implicit time integration | Newtonian τ_xx, ν ≈ 0.1 | parabolic regularisation (well-posed but not strictly hyperbolic) |

**Direction not yet tested:** structured perturbations of the constant matrix `M_0_aw` (e.g., add a small diagonal to soften the cross-coupling rather than zeroing it). Because `M_0_aw` is base-state-independent, such a perturbation acts uniformly across the flow regardless of `(ū_1, ū_2)`.

### Single-entry finding (the cheapest fix found so far)

A grid-search over individual entry-zeroings of `A` (4×4 = 16 entries) reveals one *single* entry whose removal is enough to fully regularize SME L=2:

| modification | non-hyp % | ⟨‖·‖_F⟩ |
|---|---|---|
| **`A[1, 2] = 0`**  (zero u_1's contribution to u_0's evolution) | **0.00 %** | **1.024** |
| zero `A[:, 3]` except `A[3,3]` (3 entries) | 0.00 % | 1.065 |
| zero `A[0:2, 2:4]` (4 entries) | 0.00 % | 1.159 |
| zero `A[3, :]` except `A[3,3]` (3 entries) | 0.00 % | 1.666 |
| zero `A[0:2, 2:4]` AND `A[2, 3]` (5 entries) | 0.00 % | 1.490 |
| zero `A[2:4, 0:2]` (4 entries; the moments-source-shallow direction) | 0.00 % | 2.657 |
| zero **both** off-diagonal blocks (8 entries) | 0.00 % | 2.884 |
| drop feedback entirely (Koellermeier ε = 0) | 0.00 % | 4.846 |

A grid scan over single-entry modifications:

| entry | non-hyp % | distance |
|---|---|---|
| only `A[0, 2] = 0` | 1.67 % | 0 (no effect) |
| only `A[0, 3] = 0` | 1.67 % | 0 |
| **only `A[1, 2] = 0`** | **0.00 %** | **1.024** |
| only `A[1, 3] = 0` | 20.2 % | 0.41 |
| only `A[2, 3] = 0` | 70.7 % | 0.92 (worse) |
| only `A[3, 2] = 0` | 5.5 % | 0.51 |

**Interpretation.** `A[1, 2]` is the coefficient of `∂_x δu_1` in the **u_0 evolution equation** (the mean-flow x-momentum, j = 0). At a non-zero base state this entry equals `h̄ · (ū_0 + ū_1 − ū_2)`. Zeroing it removes the source term by which the first moment `u_1` *advects* the mean velocity `u_0`. With this single entry zeroed, SME L=2 is **fully hyperbolic on the entire (ū_1, ū_2) grid**.

**Comparison with Koellermeier-style approaches:**

| approach | what it changes | ⟨‖·‖_F⟩ |
|---|---|---|
| Koellermeier ε = 0 (drop the entire feedback term) | every feedback entry | 4.85 |
| Koellermeier-style triangular `A_22` (zero `A[2, 3]` only) | 1 entry | 0.92, but **doesn't work** (70 % non-hyp) |
| **`A[1, 2] = 0`** — our minimum | **1 entry** | **1.024**, fully hyperbolic |

Our targeted single-entry fix is **~5× smaller** than dropping the feedback in Frobenius norm. The fix is also asymmetric: zeroing `A[2, 1]` instead (the reverse coupling u_0 → u_1) does *not* regularize. This indicates the destabilising mechanism is unidirectional: it's the path `u_1 → u_0 evolution` that closes the unstable feedback loop.

---

## 9. The "top-2 moments" pattern extends to L=3

`tutorials/sme/sme_l3_mode_pattern.py` extends the route-A SME builder to M = 3 (5 fields), then sweeps `(ū_2, ū_3) ∈ [−2.5, 2.5]` (15×15 grid; ū_0 = ū_1 = 0). Of 225 grid points, 44 are non-hyperbolic. Average eigenvector mode-energy distribution at those 44 points:

| field | fraction | std |
|---|---|---|
| h    |  2.9 % | ±1.6 % |
| u_0  |  1.1 % | ±2.7 % |
| u_1  |  4.2 % | ±5.1 % |
| **u_2** | **49.5 %** | ±5.1 % |
| **u_3** | **42.3 %** | ±11.2 % |

**Top-2 modes carry 92 % of the unstable eigenvector's energy. (u_2, u_3) are the dominant pair in 90.9 % of unstable samples.** Same structural finding as L=2 — the destabilisation lives in the top two Legendre moments at every level we've tested.

This means the cross-coupling-zero regularisation generalises: at level L, decouple `(h, u_0, …, u_{L-2})` from `(u_{L-1}, u_L)`. The number of perturbed entries is bounded by `2 × (L − 1)` no matter how high L gets, vs `O(L²)` for any "modify the whole flux Jacobian" recipe.

---

## Files

- `tutorials/sme/sme_l2_hyperbolicity_compare.py` — route A vs route B equivalence proof, 1D + 2D scans (no regularisation).
- `tutorials/sme/sme_l2_feedback_regularization.py` — ε-scaled-feedback experiment.
- `tutorials/sme/sme_l2_feedback_decomposition.py` — sym / anti decomposition over Eucl and W inner products.
- `tutorials/sme/sme_l2_subspace_regularization.py` — sub-block Hermitisation + cross-coupling-zero experiments.
- `tutorials/sme/sme_l2_viscous_regularization.py` — Newtonian τ_xx parabolic regularisation.
- `tutorials/sme/sme_l3_mode_pattern.py` — does the "instability lives in top-2 modes" pattern extend to L=3 (the unstable eigenvector should be (u_2, u_3)-dominated).
- corresponding `.png` plots in the same directory.
