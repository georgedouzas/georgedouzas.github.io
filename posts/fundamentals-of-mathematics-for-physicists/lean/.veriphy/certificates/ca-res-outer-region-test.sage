# name: ca-res-outer-region-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.LaurentSeries.outer_region: the expansion of 1/((z - 1)(z - 2)) on |z| > 2
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

instances = [oracle_instance(*V(z=str(p)), lhs=sum(2^(n + 1) * p^(-(n + 2)), n, 0, oo) - sum(p^(-(n + 2)), n, 0, oo),
             rhs=1/((p - 1)*(p - 2)), edge=(p == 3)) for p in [3, -5/2, 10, -4]]
