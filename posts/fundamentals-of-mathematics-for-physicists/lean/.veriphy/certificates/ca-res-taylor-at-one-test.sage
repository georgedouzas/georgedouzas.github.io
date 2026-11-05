# name: ca-res-taylor-at-one-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_taylor_at_one: the Taylor series of 1/(z^2 + 4) about 1 on |z - 1| < sqrt 5
# description: Sage checks the coefficient formula against taylor() to order 8 and sums the series exactly
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

f = 1/(z^2 + 4)
coeff = (-1/(2*I - 1)^(n + 1) + 1/(-2*I - 1)^(n + 1)) / (4*I)
for k in range(9):
    assert bool((diff(f, z, k).subs(z=1) / factorial(k) - coeff.subs(n=k)).simplify_full() == 0)
def S(p):
    a, b = 1/(2*I - 1), 1/(-2*I - 1)
    return (-a / (1 - a*(p - 1)) + b / (1 - b*(p - 1))) / (4*I)
for p in [1, 2, 0, 3]:
    assert abs((p - 1) / (2*I - 1)) < 1
instances = [oracle_instance(*V(z=str(p)), lhs=X(S(p)), rhs=X(f.subs(z=p)), edge=(p == 1)) for p in [1, 2, 0, 3]]
