# dev/wavespeed-frame (2026-10-21)

- Item A: default wave speed = closed form, else Gelfand bound of A_n in the face normal frame,
  every consumer; `SpectralRadius.eig` opt-in kept (JAX device eigensolver kept). ANALYSIS 61.
- Item B: shore rule already role-based; test added, no code change. ANALYSIS 62.
- Not done: full matrix (main runs it after merge). Partial store comparison in ANALYSIS 61.
- Beyond scope (main may drop): Perron bound / SME cone / WAVE_SPEED removed; `bound_wave_speed`
  op for no-gate transport systems; `symbolic_spectrum()` re-gates; C38 adaptive rows; gate 49
  checks the cells slot; legacy JAX refs re-blessed.
