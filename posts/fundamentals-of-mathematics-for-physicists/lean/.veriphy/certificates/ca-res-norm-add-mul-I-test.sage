# name: ca-res-norm-add-mul-I-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.norm_add_mul_I: |a + bi| = sqrt(a^2 + b^2): distances 2 and sqrt 5 to the poles ±2i
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
instances = [oracle_instance(*V(a=str(a), b=str(b)), lhs=abs(a + b*I), rhs=sqrt(a^2 + b^2), edge=(a == 0 and b == 0))
             for a, b in [(0, 0), (0, 2), (1, -2), (1, 2), (-1, 2)]]
