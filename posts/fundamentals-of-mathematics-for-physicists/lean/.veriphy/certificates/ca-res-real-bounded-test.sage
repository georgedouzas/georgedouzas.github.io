# name: ca-res-real-bounded-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.real_smooth_bounded: 1/(1 + x^2) is smooth, positive and at most 1 on the real line
# description: kernel-checked instances
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
P = 'exact ⟨ContDiffAt.div (by fun_prop) (by fun_prop) (by positivity), by positivity, by norm_num⟩'
instances = [kernel_instance(p, proof=P, edge=(p == '0')) for p in ['0', '1', '-3', '1 / 2']]
