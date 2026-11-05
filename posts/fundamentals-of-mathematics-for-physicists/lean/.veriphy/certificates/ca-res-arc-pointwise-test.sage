# name: ca-res-arc-pointwise-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueTheorem.arc_pointwise: |1/(1 + z^2)| <= 1/(R^2 - 1) on |z| = R
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
cases = [(2, 2*I, '2 * Complex.I', True), (2, 2, '2', False), (3/2, -3/2, '-3 / 2', False), (5, 3 + 4*I, '3 + 4 * Complex.I', False)]
instances = [oracle_instance(*V(R=str(R), z=zt), lhs=abs(1/(1 + p^2)), rhs=1/(R^2 - 1), relation='<=', edge=e) for R, p, zt, e in cases]
