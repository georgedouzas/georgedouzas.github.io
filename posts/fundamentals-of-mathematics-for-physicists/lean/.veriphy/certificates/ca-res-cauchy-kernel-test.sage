# name: ca-res-cauchy-kernel-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_cauchy_kernel: the geometric expansion of the Cauchy kernel
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

cases = [(0, 1, 9/10, True), (0, 2, 1, False), (1, 3, 2, False), (0, -1, 1/2, False)]
instances = [oracle_instance(*V(c=str(c), w=str(w), z=str(p)), lhs=sum((p - c)^n / (w - c)^(n + 1), n, 0, oo), rhs=1/(w - p), edge=e)
             for c, w, p, e in cases]
