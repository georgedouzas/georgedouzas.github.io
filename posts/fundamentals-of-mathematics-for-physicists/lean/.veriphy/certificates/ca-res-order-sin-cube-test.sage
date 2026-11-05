# name: ca-res-order-sin-cube-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.order_sin_cube: sin(az)/z^3 has a pole of order 2
# description: Sage computes lim z^2 f = a, finite and non-zero
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

instances = [oracle_instance(*V(a=str(a)), lhs=limit(z^2*sin(a*z)/z^3, z=0), rhs=a, edge=(a == 1)) for a in [1, 2, -1/3, I]]
