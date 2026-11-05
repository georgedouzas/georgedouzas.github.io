# name: ca-res-csc-pole-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.csc_simple_pole: 1/sin z has a simple pole at every kπ
# description: Sage computes lim (z - kπ)/sin z, finite and non-zero
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

instances = [oracle_instance(*V(k=str(k)), lhs=abs(limit((z - k*pi)/sin(z), z=k*pi)), rhs=1, edge=(k == 0)) for k in [0, 1, -1, 3]]
