# name: ca-res-cauchy-estimate-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.cauchy_estimate: the Cauchy estimate |a_n| <= M/r^n
# description: Sage computes |f^(n)(c)/n!| exactly and compares with M/r^n, M the maximum of |f| on the circle
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

cases = [(exp(z), 'Complex.exp', 0, 1, exp(SR(1)), 'Real.exp 1', 0, False), (exp(z), 'Complex.exp', 0, 1, exp(SR(1)), 'Real.exp 1', 2, False),
         (z^3, 'fun w => w ^ 3', 0, 2, 8, '8', 3, True), (1/(3 - z), 'fun w => 1 / (3 - w)', 0, 1, 1/2, '1 / 2', 2, False)]
instances = [oracle_instance(*V(f=ft, c=str(c), r=str(r), M=Mt, n=str(k)), lhs=abs(diff(f, z, k).subs(z=c) / factorial(k)),
             rhs=M / r^k, relation='<=', edge=ed) for f, ft, c, r, M, Mt, k, ed in cases]
