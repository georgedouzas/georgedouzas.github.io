# name: ca-res-pole-norm-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.pole_norm: the zeros of c^2 + z^2 lie at distance |c|
# description: kernel-checked instances at c = 1 (the poles ±i of 1/(1 + z^2)), c = 2 and c = 0
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
P = 'simp'
instances = [kernel_instance('1', 'Complex.I', '(by simp)', proof='simp', edge=False),
             kernel_instance('1', '-Complex.I', '(by simp)', proof='simp'),
             kernel_instance('2', '2 * Complex.I', '(by rw [mul_pow, Complex.I_sq]; norm_num)', proof='simp'),
             kernel_instance('0', '0', '(by simp)', proof='simp', edge=True)]
