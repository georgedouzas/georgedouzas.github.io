# name: ca-res-inside-root-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueWorked.inside_root: only -a + √(a^2 - 1) lies inside the unit circle
# description: Sage evaluates max(|p|, 1/|q|) exactly and compares with 1 (the exercise: a = 2)
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
instances = [oracle_instance(*V(a=str(a)), lhs=max_symbolic(abs(-a + sqrt(a^2 - 1)), 1/abs(-a - sqrt(a^2 - 1))), rhs=1, relation='<', edge=(a == 2))
             for a in [2, 5/4, 10]]
