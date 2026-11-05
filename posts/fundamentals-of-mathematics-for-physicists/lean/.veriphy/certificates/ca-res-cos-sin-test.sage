# name: ca-res-cos-sin-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_cos_sin: the cosine and sine series
# description: Sage sums both series exactly; the instance compares the pair through (cos series - cos z) + i (sin series - sin z)
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

def d(p):
    a = sum((-1)^n * p^(2*n) / factorial(2*n), n, 0, oo) - cos(p)
    b = sum((-1)^n * p^(2*n + 1) / factorial(2*n + 1), n, 0, oo) - sin(p)
    return (a^2 + b^2).simplify_full()
instances = [oracle_instance(*V(z=pt), lhs=d(SR(p)), rhs=0, edge=(pt == 'Real.pi')) for pt, p in
             [('Real.pi', pi), ('1', 1), ('2', 2), ('1 / 2', 1/2)]]
