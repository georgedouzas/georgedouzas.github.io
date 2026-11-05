# name: ca-res-entire-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.entire: exp, cos and sin are differentiable at every point
# description: kernel-checked instances, each proved by fun_prop at the point
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
P = 'exact ⟨by fun_prop, by fun_prop, by fun_prop⟩'
instances = [kernel_instance(p, proof=P, edge=(p == '0')) for p in ['0', '1', 'Complex.I', '-3 + 2 * Complex.I']]
