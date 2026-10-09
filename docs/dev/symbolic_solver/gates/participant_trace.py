"""participant.py with every adapter call printed (debugging only)."""
import sys
from zoomy_jax.symsolver import runtime as R
_orig = R.PreciceAdapter
class Traced(_orig):
    def max_dt(self):
        r = super().max_dt(); print("max_dt", r, flush=True); return r
    def read(self):
        r = super().read(); print("read", r.ravel(), flush=True); return r
    def write(self, q):
        print("write", __import__("numpy").asarray(q).ravel(), flush=True); return super().write(q)
    def advance(self, dt):
        r = super().advance(dt); print("advance", float(dt), r, flush=True); return r
R.PreciceAdapter = Traced
sys.argv = ["participant.py"] + sys.argv[1:]
exec(open(__file__.replace("participant_trace.py", "participant.py")).read())
