# name: ca-res-radius-taylor-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.le_radius_taylor: the radius is at least the radius of the disc of holomorphy
# description: Sage computes the radius lim |c_n/c_(n+1)| of the Taylor coefficients and compares
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

cases = [('Complex.exp', 0, 10, 1/factorial(n), False), ('fun w => 1 / (1 - w)', 0, 1, SR(1) + 0*n, True),
         ('fun w => 1 / (1 - w)', 0, 1/2, SR(1) + 0*n, False), ('fun w => 1 / (3 - w)', 1, 2, 1/2^(n + 1), False)]
instances = [oracle_instance(*V(f=f, c=str(c), r=str(r)), lhs=r, rhs=limit(abs(t / t.subs(n=n + 1)), n=oo), relation='<=', edge=e)
             for f, c, r, t, e in cases]
