# name: ca-res-sinc-entire-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.sinc_entire: sin z/z with value 1 at 0 is entire
# description: kernel-checked instances, proved at the point by the removable singularity theorem (dslope)
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
P = """have h : Function.update (fun z : ℂ => Complex.sin z / z) 0 1 = dslope Complex.sin 0 := by
  funext w
  by_cases hw : w = 0
  · subst hw; simp [dslope_same]
  · rw [Function.update_of_ne hw, dslope_of_ne _ hw, slope_def_field]; simp
rw [h]
have := (Complex.differentiableOn_dslope (s := Set.univ) (f := Complex.sin) (c := 0) Filter.univ_mem).2
  Complex.differentiable_sin.differentiableOn
exact (this _ (Set.mem_univ _)).differentiableAt Filter.univ_mem"""
instances = [kernel_instance(p, proof=P, edge=(p == '0')) for p in ['0', '1', 'Complex.I']]
