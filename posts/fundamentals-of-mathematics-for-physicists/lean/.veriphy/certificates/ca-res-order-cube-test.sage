# name: ca-res-order-cube-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.order_cube_at_I: 1/(z^2 + b^2)^3 has a pole of order 3 at ib
# description: Sage computes lim (z - ib)^3 f = 1/(2ib)^3, finite and non-zero
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

instances = [oracle_instance(*V(b=str(b)), lhs=limit((z - I*b)^3/(z^2 + b^2)^3, z=I*b), rhs=1/(2*I*b)^3, edge=(b == 1)) for b in [1, 2, -1/2]]
