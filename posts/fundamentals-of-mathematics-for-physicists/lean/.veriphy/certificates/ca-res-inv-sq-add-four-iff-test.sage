# name: ca-res-inv-sq-add-four-iff-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.summable_inv_sq_add_four_iff: the series of 1/(z^2 + 4) converges exactly on |z| < 2
# description: Sage decides convergence of the terms and compares with |z| < 2, as 1/0 flags
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
    return 1 if bool(limit(abs(p)^(2*n) / 4^(n + 1), n=oo) == 0) else 0
instances = [oracle_instance(*V(z=str(p)), lhs=conv(p), rhs=(1 if abs(p) < 2 else 0), edge=(p == 2)) for p in [2, 0, 1, -3, 5/2]]
