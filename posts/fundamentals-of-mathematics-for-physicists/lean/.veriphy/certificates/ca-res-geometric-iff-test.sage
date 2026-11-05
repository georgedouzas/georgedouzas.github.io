# name: ca-res-geometric-iff-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.summable_geometric_iff: the geometric series converges exactly on |z| < 1
# description: Sage decides convergence (finite sum, or terms not tending to 0) and compares with |z| < 1, as 1/0 flags
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

def conv(p):
    return 1 if bool(limit(abs(p)^n, n=oo) == 0) else 0
instances = [oracle_instance(*V(z=str(p)), lhs=conv(p), rhs=(1 if abs(p) < 1 else 0), edge=(p == 1)) for p in [1, 0, 1/2, -2, 3/2]]
