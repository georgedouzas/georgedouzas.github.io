# name: ca-res-exp-imag-axis-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.exp_inv_imag_axis: |e^(1/(iy))| = 1
# description: Sage evaluates the modulus exactly; y = 0 is kernel-checked (1/0 = 0 in Lean)
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
inst = [kernel_instance('0', proof='simp', edge=True)]
inst += [oracle_instance(*V(y=str(y)), lhs=abs(exp(1/(y*I))), rhs=1) for y in [1, -2, 1/3]]
instances = inst
