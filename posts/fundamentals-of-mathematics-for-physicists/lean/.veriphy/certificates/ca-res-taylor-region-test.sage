# name: ca-res-taylor-region-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.LaurentSeries.taylor_region: the Taylor expansion of 1/((z - 1)(z - 2)) on |z| < 1
# description: Sage sums the series exactly
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

instances = [oracle_instance(*V(z=str(p)), lhs=sum(p^n, n, 0, oo) - sum((1/2)^(n + 1) * p^n, n, 0, oo), rhs=1/((p - 1)*(p - 2)),
             edge=(p == 9/10)) for p in [9/10, 1/2, -3/4, 1/3]]
