"""WeightedBasis: the metric-weighted moments must match my hand values."""
import sympy as sp
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted, WeightedBasis
eta = sp.Symbol("eta"); kap, W = sp.symbols("kappa W", positive=True)
m = 1 - kap*W*(eta - sp.Rational(1,2))
inner = Legendre_shifted(level=3)
wb = WeightedBasis(inner, m, level=3)
psi = [sp.Integer(1), 2*eta-1, 6*eta**2-6*eta+1]
want = {(0,0): sp.Integer(1), (0,1): -kap*W/6, (1,1): sp.Rational(1,3)}
ok = lambda c: "PASS" if c else "**FAIL**"
print("metric-weighted moments  <psi_i, m psi_j>")
for (i,j), w in want.items():
    got = sp.expand(wb.analytical_weighted_integral(sp.expand(psi[i]*psi[j]), eta))
    print(f"   <psi_{i}, m psi_{j}> = {got}      expect {w}   {ok(sp.simplify(got-w)==0)}")
print(f"\nunit-weight control (kappa=0) must recover Legendre orthogonality")
wb0 = WeightedBasis(inner, sp.Integer(1), level=3)
for (i,j),w in {(0,0):sp.Integer(1),(0,1):sp.Integer(0),(1,1):sp.Rational(1,3)}.items():
    got = sp.expand(wb0.analytical_weighted_integral(sp.expand(psi[i]*psi[j]), eta))
    print(f"   <psi_{i}, psi_{j}> = {got}      expect {w}   {ok(sp.simplify(got-w)==0)}")
print(f"\nclosed_form_bracket must refuse delta forms: {wb.closed_form_bracket('Gram',(0,1))}   "
      f"{ok(wb.closed_form_bracket('Gram',(0,1)) is None)}")
print(f"unwrapped Legendre still gives its delta form: {inner.closed_form_bracket('Gram',(1,1))}")
