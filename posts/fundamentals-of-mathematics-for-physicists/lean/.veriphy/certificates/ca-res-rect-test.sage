# name: ca-res-rect-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueTheorem.rect_both: the rectangle x₀ ± ih, x₁ ± ih enclosing both poles gives 0
# description: Arb's rigorous enclosure of the four side integrals of the rectangle (the exercise: 0.5 ± i, 2.5 ± i)
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

f = lambda w: 1/((w - 1)*(w - 2))
def inst(x0, x1, h, x0t, x1t, ht, edge=False):
    X0, X1, H = CBF(x0), CBF(x1), CBF(h)
    total = (seg(lambda t: f(t - H*CI), x0, x1) - seg(lambda t: f(t + H*CI), x0, x1)) + CI*seg(lambda t: f(X1 + t*CI), -h, h) - CI*seg(lambda t: f(X0 + t*CI), -h, h)
    return eq(total, 0, *V({'x₀': x0t, 'x₁': x1t}, h=ht), edge=edge)
instances = [inst(1/2, 5/2, 1, '1 / 2', '5 / 2', '1', True), inst(0, 3, 2, '0', '3', '2'), inst(-1, 4, 1/2, '-1', '4', '1 / 2')]
