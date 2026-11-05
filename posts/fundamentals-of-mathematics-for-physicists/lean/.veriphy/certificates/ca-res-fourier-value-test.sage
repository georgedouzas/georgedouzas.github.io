# name: ca-res-fourier-value-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueTheorem.fourier_contour_value: 2πi e^(-k)/(2i) = π e^(-k)
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
instances = [oracle_instance(*V(k=str(k)), lhs=2*pi*I*exp(I*k*I)/(2*I), rhs=pi*exp(-k), edge=(k == 0)) for k in [0, 1, -1/2]]
