# name: ca-res-lower-half-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueWorked.lower_half_residue: Res_(-ic) 1/(z^2 + c^2) and -2πi times it is π/c
# description: Sage computes the limit symbolically and the product exactly
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

def inst(c, edge=False):
    L = limit((z + I*c)/(z^2 + c^2), z=-I*c)
    return oracle_instance(*V(c=str(c)), lhs=abs(L - 1/(2*(-I*c))) + abs(-2*pi*I*L - pi/c), rhs=0, edge=edge)
instances = [inst(1, True), inst(2), inst(-1/3)]
