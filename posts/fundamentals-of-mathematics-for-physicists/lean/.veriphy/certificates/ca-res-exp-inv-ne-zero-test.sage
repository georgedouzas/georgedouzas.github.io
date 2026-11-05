# name: ca-res-exp-inv-ne-zero-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.exp_inv_ne_zero: e^(1/z) ≠ 0
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
instances = [kernel_instance(p, proof='exact Complex.exp_ne_zero _', edge=(p == '0')) for p in ['0', '1', 'Complex.I']]
