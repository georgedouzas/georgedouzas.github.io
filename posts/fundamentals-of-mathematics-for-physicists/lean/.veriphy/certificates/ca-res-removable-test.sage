# name: ca-res-removable-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.removable_of_bounded: Riemann: a bounded singularity is removable
# description: kernel-checked instances (constant functions on the unit disc), each proved at the instance by the removable singularity theorem
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
instances = []
for f, M, edge in [('fun _ => (0 : ℂ)', '0', True), ('fun _ => (1 : ℂ)', '1', False), ('fun _ => Complex.I', '1', False)]:
    hd = '(differentiableOn_const _)'
    hb = '(⟨%s, by rintro _ ⟨x, -, rfl⟩; simp⟩)' % M
    hs = '(Metric.ball_mem_nhds 0 one_pos)'
    instances.append(kernel_instance(f, 'Metric.ball 0 1', '0', hs, hd, hb,
        proof='exact Complex.differentiableOn_update_limUnder_of_bddAbove %s %s %s' % (hs, hd, hb), edge=edge))
