# name: ca-res-two-pi-residue-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueTheorem.two_pi_I_residue_I: 2πi (-ic/2) = πc (the text: c = 1, 2πi (-i/2) = π)
# description: kernel-checked instances, each closed by linear_combination with I^2 = -1
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
def A(kw, k):
    return SR(sage_eval(kw[k].replace('Complex.I', 'I'), locals={'I': I}))
z = var('z')
instances = [kernel_instance(c, proof='linear_combination (-(Real.pi : ℂ) * (%s)) * Complex.I_sq' % c, edge=(c == '0'))
             for c in ['0', '1', 'Complex.I', '2 - 3 * Complex.I']]
