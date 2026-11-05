# name: ca-res-simple-pole-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.Residues.simple_pole: the residue at a simple pole is r = lim (z - z0) f
# description: Arb's rigorous circle integrals of r/(z - z0) + g compared with r
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

def inst(r, rt, g, gt, z0, rho, edge=False):
    ft = 'fun w => (%s) / (w - %s) + (%s) w' % (rt, z0, gt)
    return eq(circ(lambda w: CBF(r)/(w - CBF(z0)) + g(w), z0, rho) / TWO_PI_I, r, *V({'z₀': str(z0)}, f=ft, g=gt, r=rt, ρ=str(rho)), edge=edge)
instances = [inst(2, '2', lambda w: w.exp(), 'Complex.exp', 0, 1), inst(CBF(0, 1), 'Complex.I', lambda w: CBF(0), 'fun _ => 0', 0, 1, True),
             inst(-1, '-1', lambda w: 1/(w - 2), 'fun w => 1 / (w - 2)', 1, 1/2)]
