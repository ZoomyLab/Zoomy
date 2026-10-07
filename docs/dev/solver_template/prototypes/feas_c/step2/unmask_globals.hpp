// Pass-2 helper: give every undeclared call result a global definition so the
// errors MASKED behind "not declared" (arity, pointer arithmetic) surface.
#pragma once
static double Q0[8], bf_values[8], Q_L[8], Q_R[8], lim_grad[8], cell_term[8],
              Fface[8], Dp[8], Dm[8], Q_cand[8], lam_lo_f[8], lam_hi_f[8];
static int troubled[8], troubled_stage[8];
static double dt; static int do_write; static int n_faces;
