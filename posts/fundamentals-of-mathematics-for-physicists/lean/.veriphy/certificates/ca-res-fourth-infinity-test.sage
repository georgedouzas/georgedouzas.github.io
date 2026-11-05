# name: ca-res-fourth-infinity-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueTheorem.inv_one_add_fourth_infinity: 1/(1 + z^4) = sum (-1)^n z^(-4n-4) on |z| > 1
# description: Sage sums the series exactly
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

instances = [oracle_instance(*V(z=str(p)), lhs=sum((-1)^n * p^(-4*n - 4), n, 0, oo), rhs=1/(1 + p^4), edge=(p == 2)) for p in [2, -3/2, 10]]
