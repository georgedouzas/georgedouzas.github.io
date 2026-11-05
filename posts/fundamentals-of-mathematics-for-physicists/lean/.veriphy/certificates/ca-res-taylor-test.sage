# name: ca-res-taylor-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_taylor: a holomorphic function equals its Taylor series on the disc
# description: Sage checks the coefficient formula against the derivatives f^(k)(c)/k!, k <= 8, and sums the series exactly
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

def inst(f, ftxt, c, r, p, coeff, edge=False):
    for k in range(9):
        assert bool(diff(f, z, k).subs(z=c) / factorial(k) == coeff.subs(n=k))
    return oracle_instance(*V(f=ftxt, c=str(c), r=str(r), z=str(p)), lhs=sum(coeff * (p - c)^n, n, 0, oo), rhs=f.subs(z=p), edge=edge)
instances = [inst(exp(z), 'Complex.exp', 0, 10, 1, 1/factorial(n)), inst(exp(z), 'Complex.exp', 2, 10, -3, exp(SR(2))/factorial(n), True),
             inst(1/(1 - z), 'fun w => 1 / (1 - w)', 0, 1, 1/2, SR(1) + 0*n),
             inst(1/(3 - z), 'fun w => 1 / (3 - w)', 1, 2, 2, 1/2^(n + 1))]
