# name: ca-res-one-sub-cos-order-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.one_sub_cos_order_two: z^2 (1 - cos(az))/z^4 -> a^2/2
# description: Sage computes the limit symbolically
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

instances = [oracle_instance(*V(a=str(a)), lhs=limit(z^2*(1 - cos(a*z))/z^4, z=0), rhs=a^2/2, edge=(a == 0)) for a in [0, 1, 2, I]]
