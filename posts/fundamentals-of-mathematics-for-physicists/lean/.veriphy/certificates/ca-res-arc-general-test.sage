# name: ca-res-arc-general-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueWorked.arc_bound_general: |f| <= C/R^2 on the arc gives an arc integral at most πC/R
# description: Arb's rigorous enclosure of the arc integral
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

def arc(f, R):
    R = CBF(R)
    return seg(lambda t: f(R*(CI*t).exp()) * CI*R*(CI*t).exp(), 0, CBF.pi())
cases = [(lambda w: 1/(1 + w^2), 'fun w => 1 / (1 + w ^ 2)', 2, 4/3, False), (lambda w: CBF(0), 'fun _ => 0', 1, 0, True),
         (lambda w: 1/w^2, 'fun w => 1 / w ^ 2', 3, 1, False), (lambda w: w^2/(1 + w^4), 'fun w => w ^ 2 / (1 + w ^ 4)', 2, 64/15, False)]
instances = [oracle_instance(*V(f=ft, R=str(R), C=str(C)), lhs=RIFp(bnd(arc(f, R)).upper()), rhs=RIFp(pi)*C/R, relation='<=', edge=e)
             for f, ft, R, C, e in cases]
