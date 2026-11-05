# name: ca-res-arctan-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueTheorem.integral_symm_arctan: ∫_(-R)^R dx/(1 + x^2) = 2 arctan R
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

instances = [oracle_instance(*V(R=str(R)), lhs=integrate(1/(1 + x^2), x, -R, R), rhs=2*arctan(R), edge=(R == 0)) for R in [0, 1, 5, -2]]
