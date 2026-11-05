# name: ca-res-rational-deriv-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.Residues.rational_deriv: d/dz (z^2 + 1)/z at a is 1 - 1/a^2
# description: Sage differentiates symbolically
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

instances = [oracle_instance(*V(a=str(a)), lhs=diff((z^2 + 1)/z, z).subs(z=a), rhs=1 - 1/a^2, edge=(a == 1)) for a in [1, 2, -1/2, I]]
