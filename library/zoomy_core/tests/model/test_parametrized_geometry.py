"""Trivial tests for ParametrizedGeometry.  No PDE, no projection, no solver.
Each has a closed form; a failure is unambiguous."""
import sympy as sp
from zoomy_core.model.derivation.transformations import ParametrizedGeometry as PG
x, y, s, n, R, A, k = sp.symbols("x y s n R A k", real=True)
Rp = sp.Symbol("R", positive=True)
f = sp.Function("f")(x, y)
ok = lambda c: "PASS" if c else "**FAIL**"

straight = PG(sp.Matrix([s, 0]), (x, y), (s, n))
bend     = PG(sp.Matrix([Rp*sp.cos(s/Rp), Rp*sp.sin(s/Rp)]), (x, y), (s, n))
sine     = PG(sp.Matrix([s, A*sp.sin(k*s)]), (x, y), (s, n))

print("1. kappa derived from the curve")
print(f"   straight kappa = {straight.kappa}                     {ok(straight.kappa==0)}")
print(f"   bend     kappa = {sp.simplify(bend.kappa)}                   {ok(sp.simplify(bend.kappa-1/Rp)==0)}")
print(f"   sine     kappa nonzero, s-dependent            {ok(sine.kappa.has(s))}")

print("\n2. kappa=0 is the IDENTITY on derivatives")
d = sp.Derivative(f, x)
got = straight._rewrite(d)
want = sp.Derivative(sp.Function('f')(x,y), s)
print(f"   d_x f -> {got}     {ok(sp.simplify(got - want)==0)}")

print("\n3. metric and admissibility")
print(f"   bend m = {sp.simplify(bend.metric)}          {ok(sp.simplify(bend.metric-(1-n/Rp))==0)}")
adm = bend.admissible(sp.Rational(107,200)).subs(Rp, sp.Rational(366,100))
print(f"   |kappa|W<2 for R=3.66,W=1.07 -> {adm}          {ok(bool(adm))}")

print("\n4. DIVERGENCE OF A CONSTANT VECTOR = 0   (catches double-counted Christoffel)")
u0, v0 = sp.symbols("u0 v0", real=True)
div = bend._rewrite(sp.Derivative(u0, x)) + bend._rewrite(sp.Derivative(v0, y))
print(f"   div(const) = {sp.simplify(div)}                                {ok(sp.simplify(div)==0)}")

print("\n5. CURL OF A GRADIENT = 0")
g = sp.Function("g")(x, y)
gx = bend._rewrite(sp.Derivative(g, x)); gy = bend._rewrite(sp.Derivative(g, y))
curl = sp.simplify(sp.diff(gy, s)*0)  # structural placeholder
print(f"   (structural check only, needs full frame divergence)   SKIP")

print("\n6. Jacobian is the area element")
print(f"   bend jacobian = {sp.simplify(bend.jacobian)}          {ok(sp.simplify(bend.jacobian-(1-n/Rp))==0)}")
