# name: ca-res-termwise-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_deriv: term-by-term differentiation
# description: Sage sums the differentiated series exactly and differentiates the closed form
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

cases = [('fun _ => (0 : ℂ)', '1 / 2', 0, 0, 1/2, True), ('fun _ => (1 : ℂ)', '1 / 2', 1, 1/(1 - z), 1/2, False),
         ('fun n => 1 / (n.factorial : ℂ)', '2', 1/factorial(n), exp(z), 2, False), ('fun n => (-1 : ℂ) ^ n', '1 / 3', (-1)^n, 1/(1 + z), 1/3, False)]
instances = [oracle_instance(*V(a=a, z=zz), lhs=sum(t * n * p^(n - 1), n, 1, oo), rhs=diff(SR(F), z).subs(z=p), edge=e)
             for a, zz, t, F, p, e in cases]
