# name: ca-res-taylor-diverges-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.LaurentSeries.taylor_diverges: the Taylor series diverges for |z| >= 1
# description: Sage finds the terms do not tend to 0
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

instances = [oracle_instance(*V(z=str(p)), lhs=limit(abs((1 - (1/2)^(n + 1)) * p^n), n=oo), rhs=0, relation='>', edge=(p == 1))
             for p in [1, 3, -2, -1]]
