# name: ca-res-coeff-circle-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.coeff_eq_circleIntegral: Taylor coefficients as circle integrals
# description: Arb's rigorous circle integrals compared with the exact coefficients f^(n)(c)/n!
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

def inst(f, ftxt, c, R, k, exact, edge=False):
    return eq(circ(lambda w: f(w) / (w - CBF(c))^(k + 1), c, R) / TWO_PI_I, exact, *V(f=ftxt, c=str(c), R=str(R), n=str(k)), edge=edge)
instances = [inst(lambda w: CBF(5), 'fun _ => (5 : ℂ)', 0, 1, 0, 5, True), inst(lambda w: w.exp(), 'Complex.exp', 0, 1, 3, CBF(1/6)),
             inst(lambda w: w.exp(), 'Complex.exp', 1, 2, 2, CBF(1).exp() / 2), inst(lambda w: 1 / (3 - w), 'fun w => 1 / (3 - w)', 0, 2, 4, CBF(1/3^5))]
