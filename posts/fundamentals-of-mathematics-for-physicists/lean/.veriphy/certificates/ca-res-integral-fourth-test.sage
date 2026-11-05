# name: ca-res-integral-fourth-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueWorked.integral_inv_fourth: ∫ dx/(a^4 + x^4) = π/(√2 a^3)
# description: Sage integrates exactly
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

instances = [oracle_instance(*V(a=str(a)), lhs=integrate(1/(a^4 + x^4), x, -oo, oo), rhs=pi/(sqrt(2)*a^3), edge=(a == 1)) for a in [1, 2, 1/2]]
