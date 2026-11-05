# name: ca-res-quotient-fourth-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.Residues.quotient_fourth: lim (z - w)/(1 + z^4) = 1/(4w^3) at w^4 = -1
# description: Sage computes the limit symbolically at the four roots
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

instances = [oracle_instance(*V(w='Complex.exp (((2 * %d + 1) * Real.pi / 4 : ℝ) * Complex.I)' % k),
             lhs=limit((z - exp(I*(2*k + 1)*pi/4))/(1 + z^4), z=exp(I*(2*k + 1)*pi/4)).simplify_full(),
             rhs=(1/(4*exp(I*(2*k + 1)*pi/4)^3)).simplify_full(), edge=(k == 0)) for k in [0, 1, 2, 3]]
