# name: ca-res-taylor-error-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.LaurentSeries.taylor_error: the error of the Taylor partial sums
# description: Sage evaluates the partial sum to N exactly (the exercise: N = 20 at z = 0.5)
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

def inst(p, N, edge=False):
    S = sum((1 - (1/2)^(k + 1)) * p^k for k in range(N))
    return oracle_instance(*V(z=str(p), N=str(N)), lhs=1/((p - 1)*(p - 2)) - S, rhs=p^N/(1 - p) - (p/2)^N/(2 - p), edge=edge)
instances = [inst(1/2, 20), inst(1/2, 0, True), inst(-3/4, 7), inst(3, 5)]
