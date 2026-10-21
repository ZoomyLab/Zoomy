# zoomy_jax regression baselines (blessed only)

`candidate_baselines.json` holds the BLESSED scalar baselines the
`tests/regression/` tier asserts against (metric keys → measured values).
It ships EMPTY: a missing key makes its test SKIP with an
"awaiting user blessing" message — never a fake pass, never a WARN.

Bless / re-bless protocol (mirrors `zoomy_core/tests/goldens`):

1. Generate candidates (scratchpad, never straight into the repo):

       ZOOMY_JAX_CANDIDATE_BASELINES=/path/to/scratch/candidates.json \
           micromamba run -n zoomy pytest library/zoomy_jax/tests -q \
           --run-large -m "regression and jax"

2. Show the candidate numbers to the user. Only after explicit approval,
   merge them into this `candidate_baselines.json`, fill `_meta`
   (recording host — the persistent XLA compilation cache is host-keyed,
   so same-host reruns are effectively deterministic while cross-host
   runs are compared with the tests' 10% tolerances — plus the zoomy_core
   and zoomy_jax shas and the x64 flag), commit with explicit paths, push.

3. A later intentional numerics change that moves a metric = re-run step 1,
   get approval, update the entry (a baseline detects CHANGE, not
   WRONGNESS).

Field-array baselines (`.npz`), if a future test needs them, follow the same
protocol and live next to this file.
