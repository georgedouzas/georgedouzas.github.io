# name: ca-res-norm-exp-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueTheorem.norm_exp_I_mul: |e^(iz)| = e^(-Im z) <= 1 on the upper half-plane
# description: Sage evaluates the modulus exactly
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
instances = [oracle_instance(*V(z=zt), lhs=abs(exp(I*p)), rhs=exp(-imag(p)), edge=(p == 0)) for p, zt in
             [(0, '0'), (1 + I, '1 + Complex.I'), (-2*I, '-2 * Complex.I'), (3, '3')]]
