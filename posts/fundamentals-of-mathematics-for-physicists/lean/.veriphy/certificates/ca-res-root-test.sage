# name: ca-res-root-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.radius_eq_liminf: the root test for the radius
# description: Sage compares the ratio-test radius lim |a_n/a_(n+1)| with lim |a_n|^(-1/n)
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

cases = [('fun _ => (1 : ℂ)', 1, True), ('fun n => (2 : ℂ) ^ n', 2^n, False), ('fun n => (3 : ℂ) ^ n / (n + 1)', 3^n/(n + 1), False),
         ('fun n => (n + 1 : ℂ)', n + 1, False)]
instances = [oracle_instance(*V(a=a), lhs=limit(SR(t / t.subs(n=n + 1)).simplify_full(), n=oo), rhs=limit(SR(t)^(-1/n), n=oo), edge=e)
             for a, t, e in cases]
