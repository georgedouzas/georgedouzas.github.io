# name: ca-res-divergent-radius-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.not_summable_of_radius_lt: divergence outside the radius
# description: Sage finds |a_n z^n| does not tend to 0 (limit +oo or positive)
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

cases = [('fun n => (n.factorial : ℂ)', '1 / 10', factorial(n) / 10^n, True), ('fun _ => (1 : ℂ)', '2', 2^n, False),
         ('fun n => (2 : ℂ) ^ n', '1', 2^n, False), ('fun n => (1 / 2 : ℂ) ^ n', '3', (3/2)^n, False)]
instances = [oracle_instance(*V(a=a, z=zz), lhs=limit(t, n=oo), rhs=0, relation='>', edge=e) for a, zz, t, e in cases]
