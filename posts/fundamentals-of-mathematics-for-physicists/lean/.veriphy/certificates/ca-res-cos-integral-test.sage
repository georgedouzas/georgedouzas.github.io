# name: ca-res-cos-integral-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueTheorem.integral_cos_div: ∫ cos(kx)/(1 + x^2) dx = π e^(-|k|)
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

instances = [oracle_instance(*V(k=str(k)), lhs=integrate(cos(k*x)/(1 + x^2), x, -oo, oo), rhs=pi*exp(-abs(k)), edge=(k == 0)) for k in [0, 1, 1/2, -2]]
