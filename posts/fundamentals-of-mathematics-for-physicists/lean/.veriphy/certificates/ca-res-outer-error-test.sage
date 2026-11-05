# name: ca-res-outer-error-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.LaurentSeries.outer_error: the error of the outer partial sums
# description: Sage evaluates the partial sum to N exactly (the exercise: N = 20 at z = 3)
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
def inst(p, ptxt, N, edge=False):
    S = sum((2^(k + 1) - 1) * p^(-(k + 2)) for k in range(N))
    return oracle_instance(*V(z=ptxt, N=str(N)), lhs=X(1/((p - 1)*(p - 2)) - S), rhs=X(2*(2/p)^N/(p*(p - 2)) - p^(-N)/(p*(p - 1))), edge=edge)
instances = [inst(SR(3), '3', 20), inst(SR(3), '3', 0, True), inst(SR(-5/2), '-5 / 2', 6), inst(3*I, '3 * Complex.I', 3)]
