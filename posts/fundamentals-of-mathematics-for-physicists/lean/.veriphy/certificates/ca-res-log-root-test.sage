# name: ca-res-log-root-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.log_root: the n-th root test for the logarithmic series
# description: Sage computes lim (|z|^n/n)^(1/n)
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

instances = [oracle_instance(*V(z=str(p)), lhs=limit((abs(p)^n / n)^(1/n), n=oo), rhs=abs(p), edge=(p == 0)) for p in [0, 1/2, 2, -3]]
