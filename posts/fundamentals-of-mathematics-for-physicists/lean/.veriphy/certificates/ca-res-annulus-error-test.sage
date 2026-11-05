# name: ca-res-annulus-error-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.LaurentSeries.annulus_error: the error of the annulus partial sums
# description: Sage evaluates the partial sum to N exactly (the exercise: N = 20 at z = 1.5i)
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
    S = sum(-(p^k / 2^(k + 1)) - p^(-(k + 1)) for k in range(N))
    return oracle_instance(*V(z=ptxt, N=str(N)), lhs=X(1/((p - 1)*(p - 2)) - S), rhs=X(-(p/2)^N/(2 - p) + p^(-N)/(1 - p)), edge=edge)
instances = [inst(3/2*I, '3 / 2 * Complex.I', 20), inst(SR(3/2), '3 / 2', 0, True), inst(SR(-6/5), '-6 / 5', 9), inst(1 + I, '1 + Complex.I', 4)]
