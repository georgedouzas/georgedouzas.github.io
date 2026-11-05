# name: ca-res-one-sub-cos-series-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.one_sub_cos_series: (1 - cos z)/z^4 = 1/(2z^2) - sum (-1)^n z^(2n)/(2n+4)!
# description: Arb bounds the partial sum to N = 40 against the closed form, within the tail bound 2|z|^80/84! plus 1e-30 for rounding
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

def inst(p, ptxt, edge=False):
    P = CBF(p)
    S = sum((-1)^k * P^(2*k) / factorial(2*k + 4) for k in range(40))
    tail = 2 * RIFp(abs(CBF(p)).upper())^80 / factorial(84)
    return oracle_instance(*V(z=ptxt), lhs=RIFp(gap(S, 1/(2*P^2) - (1 - P.cos())/P^4).upper()), rhs=tail + RIFp(10)^-30, relation='<=', edge=edge)
instances = [inst(1, '1', True), inst(2, '2'), inst(-1/2, '-1 / 2'), inst(CBF(0, 1), 'Complex.I')]
