# name: ca-res-holo-sum-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.differentiableOn_sum: the sum is holomorphic inside the radius
# description: Sage differentiates the closed form of the sum at an interior point and compares with the sum of the derivatives
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

cases = [('fun _ => (0 : ℂ)', 0, 0, 1/2, True), ('fun _ => (1 : ℂ)', 1, 1/(1 - z), 1/2, False),
         ('fun n => 1 / (n.factorial : ℂ)', 1/factorial(n), exp(z), 2, False), ('fun n => (-1 : ℂ) ^ n', (-1)^n, 1/(1 + z), 1/3, False)]
instances = [oracle_instance(*V(a=a), lhs=diff(SR(F), z).subs(z=p), rhs=sum(t * n * p^(n - 1), n, 1, oo), edge=e) for a, t, F, p, e in cases]
