# name: ca-res-coeff-indep-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.LaurentSeries.coeff_indep_radius: the coefficient integrals do not depend on the radius in an annulus
# description: Arb's rigorous circle integrals of f (z - c)^(-n-1) on two radii of the annulus 1 < |z| < 2 of 1/((z - 1)(z - 2))
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

f = lambda w: 1 / ((w - 1) * (w - 2))
def inst(r, R, k, edge=False):
    g = lambda w: f(w) * w^(-k - 1)
    return eq(circ(g, 0, R), circ(g, 0, r), *V(f='fun w => 1 / ((w - 1) * (w - 2))', c='0', r=str(r), R=str(R), n=str(k)), edge=edge)
instances = [inst(6/5, 6/5, 0, True), inst(6/5, 9/5, -1), inst(11/10, 19/10, 2), inst(3/2, 7/4, -3)]
