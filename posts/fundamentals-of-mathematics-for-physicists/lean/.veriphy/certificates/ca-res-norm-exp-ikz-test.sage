# name: ca-res-norm-exp-ikz-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueWorked.norm_exp_ikz: |e^(ikz)| = e^(-k Im z)
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
instances = [oracle_instance(*V(k=str(k), z=zt), lhs=abs(exp(I*k*p)), rhs=exp(-k*imag(p)), edge=(k == 0))
             for k, p, zt in [(0, I, 'Complex.I'), (1, 2*I, '2 * Complex.I'), (-1, -I, '-Complex.I'), (2, 1 + I, '1 + Complex.I')]]
