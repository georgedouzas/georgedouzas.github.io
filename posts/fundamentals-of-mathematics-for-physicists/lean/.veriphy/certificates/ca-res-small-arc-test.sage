# name: ca-res-small-arc-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueWorked.small_arc: the small semicircle contributes -iπ in the limit
# description: kernel-checked instances, each proved at the instance by continuity of the parametric integral
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
P = """have hc : Continuous (fun ε : ℝ => ∫ θ in Real.pi..0, Complex.exp (Complex.I * (A : ℝ) * (ε * Complex.exp (θ * Complex.I))) * Complex.I) :=
  intervalIntegral.continuous_parametric_intervalIntegral_of_continuous' (by fun_prop) Real.pi 0
have := hc.tendsto 0
simp only [Complex.ofReal_zero, zero_mul, mul_zero, Complex.exp_zero, one_mul, intervalIntegral.integral_const, smul_eq_mul, zero_sub] at this
convert this using 2
first
  | (rw [Complex.real_smul]; push_cast; ring)
  | (push_cast; ring)
  | simp"""
# a = 1 is the edge case (the text's value); at a = 0 the integrand is constant and the block's `convert` does not apply
instances = [kernel_instance(a, proof=P.replace('A', a), edge=(a == '1')) for a in ['1', '-2', '1 / 2', '3']]
