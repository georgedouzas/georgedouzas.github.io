# name: ca-res-laurent-coeff-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.LaurentSeries.laurent_coeff: the coefficient formula and uniqueness of Laurent coefficients
# description: Arb's rigorous circle integrals of (z - c)^(-m-1) f for explicit Laurent series, compared with b_m
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
CBF = ComplexBallField(128)
RIFp = RealIntervalField(128)
CI = CBF(0, 1)
TWO_PI_I = 2 * CBF.pi() * CI
def circ(f, c, rho):
    """Arb's rigorous enclosure of the circle integral of f over |z - c| = rho."""
    c = CBF(c); rho = CBF(rho)
    return CBF.integral(lambda t, _: f(c + rho * (CI * t).exp()) * CI * rho * (CI * t).exp(), 0, 2 * CBF.pi().real())
def seg(f, a, b):
    """Arb's rigorous enclosure of the integral of f(t) dt over [a, b]."""
    return CBF.integral(lambda t, _: f(t), CBF(a), CBF(b))
def gap(x, y):
    """The interval of |x - y| (contains 0 exactly when the enclosures are compatible)."""
    d = (CBF(x) - CBF(y)).abs()
    return RIFp(d.lower(), d.upper())
def bnd(x):
    """The interval of |x|."""
    d = CBF(x).abs()
    return RIFp(d.lower(), d.upper())
def eq(x, y, *values, edge=False):
    return oracle_instance(*values, lhs=gap(x, y), rhs=RIFp(0), relation='==', edge=edge)

fin = 'fun n : ℤ => if n = -1 then 2 else if n = 2 then 3 else 0'
geo = 'fun n : ℤ => if 0 ≤ n then (1 / 2 : ℂ) ^ n.toNat else 0'
def F1(w, c): return 2 / (w - c) + 3 * (w - c)^2
def F2(w, c): return 1 / (1 - (w - c) / 2)
def inst(F, btxt, c, rho, m, bm, edge=False):
    c = CBF(c)
    return eq(circ(lambda w: (w - c)^(-m - 1) * F(w, c), c, rho) / TWO_PI_I, bm, *V(c=str(c), ρ=str(rho), b=btxt, m=str(m)), edge=edge)
instances = [inst(F1, fin, 0, 1, -1, 2, True), inst(F1, fin, 1, 2, 2, 3), inst(F1, fin, 0, 1, 0, 0), inst(F2, geo, 0, 1, 3, CBF(1/8)),
             inst(F2, geo, 0, 1, -2, 0)]
