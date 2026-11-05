# name: ca-res-sinc-series-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.sinc_series: sin z/z = sum (-1)^n z^(2n)/(2n+1)!
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

instances = [oracle_instance(*V(z=str(p)), lhs=X(sum((-1)^n * p^(2*n) / factorial(2*n + 1), n, 0, oo)), rhs=X(sin(p)/p), edge=(p == 1)) for p in [1, 2, -1/2, 5]]
