# name: ca-res-rational-residues-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.Residues.rational_residues: the residues 1/a^2 at 0 and 1 - 1/a^2 at a
# description: Arb's rigorous small circle integrals compared with 1/a^2 and 1 - 1/a^2
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

def inst(a, at, rho, edge=False):
    A = CBF(a); f = lambda w: (w^2 + 1)/(w*(w - A)^2)
    d = gap(circ(f, 0, rho) / TWO_PI_I, 1/A^2) + gap(circ(f, a, rho) / TWO_PI_I, 1 - 1/A^2)
    return oracle_instance(*V(a=at, ρ=str(rho)), lhs=d, rhs=RIFp(0), relation='==', edge=edge)
instances = [inst(1, '1', 1/2, True), inst(2, '2', 1), inst(CBF(0, 1), 'Complex.I', 1/2)]
