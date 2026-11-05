# name: ca-res-log-series-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_log: Log(1 + z) = sum (-1)^(n+1) z^n/n
# description: Sage sums the logarithmic series exactly
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

instances = [oracle_instance(*V(z=str(p)), lhs=sum((-1)^(n + 1) * p^n / n, n, 1, oo), rhs=log(1 + p), edge=(p == 9/10)) for p in [9/10, 1/2, -1/2, 1/7]]
