# name: ca-res-inv-sq-add-four-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_inv_sq_add_four: 1/(z^2 + 4) = sum (-1)^n z^(2n)/4^(n+1)
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

instances = [oracle_instance(*V(z=str(p)), lhs=sum((-1)^n * p^(2*n) / 4^(n + 1), n, 0, oo), rhs=1/(p^2 + 4), edge=(p == 19/10)) for p in [19/10, 1, -3/2, 1/2]]
