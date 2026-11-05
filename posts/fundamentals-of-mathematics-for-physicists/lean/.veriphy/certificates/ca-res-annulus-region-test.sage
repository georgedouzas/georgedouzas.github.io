# name: ca-res-annulus-region-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.LaurentSeries.annulus_region: the expansion of 1/((z - 1)(z - 2)) on 1 < |z| < 2
# description: Sage sums the series exactly; for z = 1.5i, Arb bounds the partial sum to N = 200 against the tail
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
n = var('n'); z = var('z'); x = var('x')
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

inst = [oracle_instance(*V(z=str(p)), lhs=-sum(p^n / 2^(n + 1), n, 0, oo) - sum(p^(-(n + 1)), n, 0, oo), rhs=1/((p - 1)*(p - 2)),
        edge=(p == 3/2)) for p in [3/2, -3/2, 6/5]]
p = CBF(0, 3/2)
S = sum(-(p^k / 2^(k + 1)) - p^(-(k + 1)) for k in range(200))
tail = RIFp(3/4)^200 * 2 + RIFp(2/3)^200 * 3
inst.append(oracle_instance(*V(z='3 / 2 * Complex.I'), lhs=RIFp(gap(S, 1 / ((p - 1) * (p - 2))).upper()), rhs=tail, relation='<='))
instances = inst
