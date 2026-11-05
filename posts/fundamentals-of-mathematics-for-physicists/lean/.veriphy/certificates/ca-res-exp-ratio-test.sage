# name: ca-res-exp-ratio-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.exp_ratio: the ratio of successive terms of the exponential series tends to 0
# description: Sage computes the limit of the ratio
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

instances = [oracle_instance(*V(z=str(p)), lhs=limit(abs(p)^(n + 1) / factorial(n + 1) / (abs(p)^n / factorial(n)), n=oo), rhs=0, edge=(p == 1))
             for p in [1, -5, 1/3, 100]]
