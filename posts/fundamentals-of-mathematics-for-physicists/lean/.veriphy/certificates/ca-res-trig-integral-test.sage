# name: ca-res-trig-integral-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueWorked.trig_integral: ∫_0^(2π) dθ/(a + cos θ) = 2π/√(a^2 - 1)
# description: Arb's rigorous enclosure of the real integral and of the circle integral (the exercise: a = 2, 2π/√3)
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

def inst(a, edge=False):
    A = CBF(a)
    real = seg(lambda t: 1/(A + t.cos()), 0, 2*CBF.pi())
    circle = circ(lambda w: 2/(CI*(w^2 + 2*A*w + 1)), 0, 1)
    exact = 2*CBF.pi()/(A^2 - 1).sqrt()
    return oracle_instance(*V(a=str(a)), lhs=gap(real, exact) + gap(circle, exact), rhs=RIFp(0), relation='==', edge=edge)
instances = [inst(2, True), inst(3), inst(5/4)]
