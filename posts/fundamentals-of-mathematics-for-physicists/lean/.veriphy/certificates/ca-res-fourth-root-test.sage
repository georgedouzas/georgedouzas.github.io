# name: ca-res-fourth-root-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.Residues.fourth_root: z_k^4 = -1 for z_k = e^(iπ(2k+1)/4)
# description: Sage evaluates z_k^4 exactly
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
instances = [oracle_instance(*V(k=str(k)), lhs=(exp(I*(2*k + 1)*pi/4)^4).simplify_full(), rhs=-1, edge=(k == 0)) for k in [0, 1, 2, 3, 7]]
