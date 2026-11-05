# name: ca-res-summable-radius-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.summable_norm_of_lt_radius: absolute convergence inside the radius
# description: Sage sums |a_n z^n| exactly and finds a finite value
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

cases = [('fun _ => (0 : ℂ)', '5', 0, True), ('fun n => (2 : ℂ) ^ n', '1 / 4', 2^n * (1/4)^n, False),
         ('fun n => 1 / (n.factorial : ℂ)', '3', 3^n / factorial(n), False), ('fun n => (n : ℂ)', '1 / 2', n / 2^n, False)]
instances = [oracle_instance(*V(a=a, z=zz), lhs=sum(t, n, 0, oo), rhs=oo, relation='<', edge=e) for a, zz, t, e in cases]
