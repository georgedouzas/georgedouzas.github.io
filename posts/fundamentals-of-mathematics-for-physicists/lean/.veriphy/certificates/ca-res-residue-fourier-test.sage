# name: ca-res-residue-fourier-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueTheorem.residue_fourier: Res_i e^(ikz)/(1 + z^2) = e^(-k)/(2i)
# description: Sage computes the limit symbolically
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

instances = [oracle_instance(*V(k=str(k)), lhs=limit((z - I)*exp(I*k*z)/(1 + z^2), z=I), rhs=exp(-k)/(2*I), edge=(k == 0)) for k in [0, 1, 2, -1]]
