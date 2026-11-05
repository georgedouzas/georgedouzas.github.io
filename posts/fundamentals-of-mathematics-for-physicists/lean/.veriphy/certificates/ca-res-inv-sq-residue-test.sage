# name: ca-res-inv-sq-residue-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.Residues.residue_inv_sq_add_sq: Res_(ic) 1/(z^2 + c^2) = 1/(2ic)
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

instances = [oracle_instance(*V(c=str(c)), lhs=limit((z - I*c)/(z^2 + c^2), z=I*c), rhs=1/(2*I*c), edge=(c == 1)) for c in [1, 2, -1/3]]
