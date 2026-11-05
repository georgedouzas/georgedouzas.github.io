# name: ca-res-fourth-rotate-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueTheorem.fourth_roots_rotate: z_k = z_0 i^k
# description: Sage evaluates both sides exactly
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
instances = [oracle_instance(*V(k=str(k)), lhs=exp(I*(2*k + 1)*pi/4).simplify_full(), rhs=(exp(I*pi/4)*I^k).simplify_full(), edge=(k == 0))
             for k in [0, 1, 2, 3, 6]]
