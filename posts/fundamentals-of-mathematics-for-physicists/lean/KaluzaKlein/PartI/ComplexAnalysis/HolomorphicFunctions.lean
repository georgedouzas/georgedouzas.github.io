import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.ComplexDerivative
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Holomorphic functions"

The claims of the section "Holomorphic functions" of `tutorial/part-i-04-complex-analysis.qmd`: holomorphy on an
open set (mathlib's `DifferentiableOn ℂ`) and at a point (`HolomorphicAt`, differentiable on a neighbourhood); the
closure under sums, products, quotients and compositions; polynomials, `eᶻ`, `cos`, `sin`, `cosh`, `sinh` are
entire; `(z - 1)/(z² + 1)` is holomorphic off `±i`; the principal logarithm is holomorphic on the slit plane with
`Log' z = 1/z` and discontinuous on the cut; `z̄`, `Re z`, `|z|`, `|z|²` are nowhere holomorphic; the domains of
`(z² + 1)/(z⁴ - 1)` and `tan z`; holomorphic functions of constant modulus are constant.
-/

namespace KaluzaKlein.ComplexAnalysis.HolomorphicFunctions

open Complex Filter Topology ComplexConjugate
open scoped Real

/-! ## def-holomorphic -/

/-- A function is holomorphic at `z` if it is complex differentiable at every point of a neighbourhood of `z`. -/
def HolomorphicAt (f : ℂ → ℂ) (z : ℂ) : Prop := ∀ᶠ w in 𝓝 z, DifferentiableAt ℂ f w

/-- Sums and products of holomorphic functions are holomorphic, and so are quotients off the zeros of the
denominator. -/
theorem closure_ops (f g : ℂ → ℂ) (U : Set ℂ) (hf : DifferentiableOn ℂ f U) (hg : DifferentiableOn ℂ g U) :
    DifferentiableOn ℂ (fun z => f z + g z) U ∧ DifferentiableOn ℂ (fun z => f z * g z) U ∧
      ((∀ z ∈ U, g z ≠ 0) → DifferentiableOn ℂ (fun z => f z / g z) U) :=
  ⟨hf.add hg, hf.mul hg, fun h => hf.div hg h⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.closure_ops ((fun w : ℂ => w)) ((fun _ : ℂ => (1 : ℂ))) (Set.univ) ((by fun_prop)) ((by fun_prop)) := by
  beta_reduce
  refine ⟨by fun_prop, by fun_prop, fun h => ?_⟩
  exact DifferentiableOn.div (by fun_prop) (by fun_prop) h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.closure_ops ((fun z : ℂ => z ^ 2)) ((fun z : ℂ => z + 3)) ((Metric.ball 0 1)) ((by fun_prop)) ((by fun_prop)) := by
  beta_reduce
  refine ⟨by fun_prop, by fun_prop, fun h => ?_⟩
  exact DifferentiableOn.div (by fun_prop) (by fun_prop) h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.closure_ops ((fun z : ℂ => z * Complex.I)) ((fun z : ℂ => z ^ 2 + 1)) (Set.univ) ((by fun_prop)) ((by fun_prop)) := by
  beta_reduce
  refine ⟨by fun_prop, by fun_prop, fun h => ?_⟩
  exact DifferentiableOn.div (by fun_prop) (by fun_prop) h

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.closure_ops
-- test: ca-holo-closure-test.sage

/-- Compositions of holomorphic functions are holomorphic. -/
theorem comp_holo (f g : ℂ → ℂ) (U V : Set ℂ) (hf : DifferentiableOn ℂ f U) (hg : DifferentiableOn ℂ g V)
    (hUV : Set.MapsTo f U V) : DifferentiableOn ℂ (fun z => g (f z)) U :=
  hg.comp hf hUV

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.comp_holo ((fun w : ℂ => w)) ((fun z : ℂ => z ^ 2)) (Set.univ) (Set.univ) ((by fun_prop)) ((by fun_prop)) ((Set.mapsTo_univ _ _)) := by
  beta_reduce
  fun_prop

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.comp_holo ((fun z : ℂ => z + 1)) ((fun z : ℂ => z ^ 3)) (Set.univ) (Set.univ) ((by fun_prop)) ((by fun_prop)) ((Set.mapsTo_univ _ _)) := by
  beta_reduce
  fun_prop

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.comp_holo ((fun z : ℂ => 2 * z)) ((fun z : ℂ => z * z + Complex.I)) ((Metric.ball 0 1)) (Set.univ) ((by fun_prop)) ((by fun_prop)) ((Set.mapsTo_univ _ _)) := by
  beta_reduce
  fun_prop

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.comp_holo
-- test: ca-holo-comp-test.sage

/-! ## exm-holomorphic-polynomials -/

/-- Every polynomial is entire. -/
theorem polynomial_entire (p : Polynomial ℂ) : Differentiable ℂ (fun z => p.eval z) :=
  p.differentiable

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.polynomial_entire ((0 : Polynomial ℂ)) := by
  simp only [Polynomial.eval_add, Polynomial.eval_pow, Polynomial.eval_X, Polynomial.eval_one, Polynomial.eval_zero]
  fun_prop

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.polynomial_entire ((Polynomial.X : Polynomial ℂ)) := by
  simp only [Polynomial.eval_add, Polynomial.eval_pow, Polynomial.eval_X, Polynomial.eval_one, Polynomial.eval_zero]
  fun_prop

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.polynomial_entire ((Polynomial.X ^ 2 + 1 : Polynomial ℂ)) := by
  simp only [Polynomial.eval_add, Polynomial.eval_pow, Polynomial.eval_X, Polynomial.eval_one, Polynomial.eval_zero]
  fun_prop

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.polynomial_entire
-- test: ca-holo-polynomial-test.sage

/-- `e^{az}`, `cos az`, `sin az`, `cosh az`, `sinh az` are entire, for every `a` (`a = 1`: the functions of the
text). -/
theorem exp_family_entire (a : ℂ) :
    Differentiable ℂ (fun z => Complex.exp (a * z)) ∧ Differentiable ℂ (fun z => Complex.cos (a * z)) ∧
      Differentiable ℂ (fun z => Complex.sin (a * z)) ∧ Differentiable ℂ (fun z => Complex.cosh (a * z)) ∧
      Differentiable ℂ (fun z => Complex.sinh (a * z)) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> fun_prop

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.exp_family_entire (0) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> fun_prop

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.exp_family_entire (1) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> fun_prop

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.exp_family_entire (Complex.I) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> fun_prop

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.exp_family_entire
-- test: ca-holo-exp-family-test.sage

/-- `z² + 1 = (z - i)(z + i)`. -/
theorem sq_add_one_factor (z : ℂ) : z ^ 2 + 1 = (z - I) * (z + I) := by
  have hI : I ^ 2 = -1 := I_sq
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination hI)
  -- certificate: ca-holo-ring.sage

/-- The zeros of `z² + 1` are `i` and `-i`. -/
theorem sq_add_one_eq_zero_iff (z : ℂ) : z ^ 2 + 1 = 0 ↔ z = I ∨ z = -I := by
  rw [sq_add_one_factor, mul_eq_zero, sub_eq_zero, add_eq_zero_iff_eq_neg]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.sq_add_one_eq_zero_iff (0) := by
  norm_num [Complex.ext_iff, pow_two]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.sq_add_one_eq_zero_iff (Complex.I) := by
  norm_num [Complex.ext_iff, pow_two]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.sq_add_one_eq_zero_iff ((-Complex.I)) := by
  norm_num [Complex.ext_iff, pow_two]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.sq_add_one_eq_zero_iff
-- test: ca-holo-sq-add-one-test.sage

/-- `(z - 1)/(z² + 1)` is holomorphic at every `z ≠ ±i`. -/
theorem rational_differentiableAt (z : ℂ) (h1 : z ≠ I) (h2 : z ≠ -I) :
    DifferentiableAt ℂ (fun w : ℂ => (w - 1) / (w ^ 2 + 1)) z := by
  refine DifferentiableAt.div (by fun_prop) (by fun_prop) ?_
  rw [Ne, sq_add_one_eq_zero_iff]
  tauto

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.rational_differentiableAt (0) ((by norm_num [Complex.ext_iff])) ((by norm_num [Complex.ext_iff])) := by
  exact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff, pow_two])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.rational_differentiableAt (1) ((by norm_num [Complex.ext_iff])) ((by norm_num [Complex.ext_iff])) := by
  exact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff, pow_two])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.rational_differentiableAt ((2 * Complex.I)) ((by norm_num [Complex.ext_iff])) ((by norm_num [Complex.ext_iff])) := by
  exact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff, pow_two])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.rational_differentiableAt
-- test: ca-holo-rational-test.sage

/-! ## exm-holomorphic-log -/

/-- `Log` is holomorphic on the slit plane with `Log' z = 1/z`. -/
theorem log_hasDerivAt (z : ℂ) (hz : z ∈ slitPlane) : HasDerivAt Complex.log z⁻¹ z :=
  Complex.hasDerivAt_log hz

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_hasDerivAt (1) ((by norm_num [Complex.mem_slitPlane_iff])) := by
  exact Complex.hasDerivAt_log (by norm_num [Complex.mem_slitPlane_iff])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_hasDerivAt (Complex.I) ((by norm_num [Complex.mem_slitPlane_iff])) := by
  exact Complex.hasDerivAt_log (by norm_num [Complex.mem_slitPlane_iff])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_hasDerivAt ((2 + Complex.I)) ((by norm_num [Complex.mem_slitPlane_iff])) := by
  exact Complex.hasDerivAt_log (by norm_num [Complex.mem_slitPlane_iff])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_hasDerivAt
-- test: ca-holo-log-deriv-test.sage

/-- The other branches `Log z + 2πik` have the same derivative `1/z`. -/
theorem branch_hasDerivAt (k : ℤ) (z : ℂ) (hz : z ∈ slitPlane) :
    HasDerivAt (fun w => Complex.log w + 2 * π * I * k) z⁻¹ z :=
  (Complex.hasDerivAt_log hz).add_const _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.branch_hasDerivAt (0) (1) ((by norm_num [Complex.mem_slitPlane_iff])) := by
  exact (Complex.hasDerivAt_log (by norm_num [Complex.mem_slitPlane_iff])).add_const _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.branch_hasDerivAt (1) (Complex.I) ((by norm_num [Complex.mem_slitPlane_iff])) := by
  exact (Complex.hasDerivAt_log (by norm_num [Complex.mem_slitPlane_iff])).add_const _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.branch_hasDerivAt ((-2)) ((2 + Complex.I)) ((by norm_num [Complex.mem_slitPlane_iff])) := by
  exact (Complex.hasDerivAt_log (by norm_num [Complex.mem_slitPlane_iff])).add_const _

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.branch_hasDerivAt
-- test: ca-holo-branch-test.sage

/-- At a point `x < 0` of the cut, `Log` tends to `log |x| - πi` from below, while `Log x = log |x| + πi`. -/
theorem log_limit_below (x : ℝ) (hx : x < 0) :
    Tendsto Complex.log (𝓝[{z : ℂ | z.im < 0}] (x : ℂ)) (𝓝 (Real.log ‖(x : ℂ)‖ - π * I)) ∧
      (Complex.log x).im = π :=
  ⟨tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (by simpa using hx) (by simp),
    by rw [log_im, arg_ofReal_of_neg hx]⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_limit_below ((-1)) ((by norm_num)) := by
  exact ⟨Complex.tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by norm_num), by rw [Complex.log_im, Complex.arg_ofReal_of_neg (by norm_num)]⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_limit_below ((-2)) ((by norm_num)) := by
  exact ⟨Complex.tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by norm_num), by rw [Complex.log_im, Complex.arg_ofReal_of_neg (by norm_num)]⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_limit_below ((-1 / 2)) ((by norm_num)) := by
  exact ⟨Complex.tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by norm_num), by rw [Complex.log_im, Complex.arg_ofReal_of_neg (by norm_num)]⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_limit_below
-- test: ca-holo-log-limit-test.sage

/-- `Log` is not continuous at any point of the cut `(-∞, 0)`, hence not holomorphic there. -/
theorem log_not_continuousAt (x : ℝ) (hx : x < 0) : ¬ ContinuousAt Complex.log (x : ℂ) := by
  intro h
  obtain ⟨h1, h3⟩ := log_limit_below x hx
  have hcl : (x : ℂ) ∈ closure {z : ℂ | z.im < 0} := by
    rw [closure_setOfPred_im_lt]
    simp
  haveI : (𝓝[{z : ℂ | z.im < 0}] (x : ℂ)).NeBot := mem_closure_iff_nhdsWithin_neBot.mp hcl
  have h2 := tendsto_nhds_unique (h.tendsto.mono_left nhdsWithin_le_nhds) h1
  rw [h2] at h3
  simp only [Complex.sub_im, Complex.ofReal_im, Complex.mul_im, Complex.ofReal_re, Complex.I_re, Complex.I_im,
    mul_zero, mul_one, zero_add, zero_sub] at h3
  linarith [Real.pi_pos]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_not_continuousAt ((-1)) ((by norm_num)) := by
  have hx := (by norm_num : (-1 : ℝ) < 0)
  intro h
  obtain ⟨h1, h3⟩ := log_limit_below (-1 : ℝ) hx
  have hcl : ((-1 : ℝ) : ℂ) ∈ closure {z : ℂ | z.im < 0} := by
    rw [closure_setOfPred_im_lt]
    simp
  haveI : (𝓝[{z : ℂ | z.im < 0}] ((-1 : ℝ) : ℂ)).NeBot := mem_closure_iff_nhdsWithin_neBot.mp hcl
  have h2 := tendsto_nhds_unique (h.tendsto.mono_left nhdsWithin_le_nhds) h1
  rw [h2] at h3
  simp only [Complex.sub_im, Complex.ofReal_im, Complex.mul_im, Complex.ofReal_re, Complex.I_re, Complex.I_im,
    mul_zero, mul_one, zero_add, zero_sub] at h3
  linarith [Real.pi_pos]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_not_continuousAt ((-2)) ((by norm_num)) := by
  have hx := (by norm_num : (-2 : ℝ) < 0)
  intro h
  obtain ⟨h1, h3⟩ := log_limit_below (-2 : ℝ) hx
  have hcl : ((-2 : ℝ) : ℂ) ∈ closure {z : ℂ | z.im < 0} := by
    rw [closure_setOfPred_im_lt]
    simp
  haveI : (𝓝[{z : ℂ | z.im < 0}] ((-2 : ℝ) : ℂ)).NeBot := mem_closure_iff_nhdsWithin_neBot.mp hcl
  have h2 := tendsto_nhds_unique (h.tendsto.mono_left nhdsWithin_le_nhds) h1
  rw [h2] at h3
  simp only [Complex.sub_im, Complex.ofReal_im, Complex.mul_im, Complex.ofReal_re, Complex.I_re, Complex.I_im,
    mul_zero, mul_one, zero_add, zero_sub] at h3
  linarith [Real.pi_pos]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_not_continuousAt ((-1 / 2)) ((by norm_num)) := by
  have hx := (by norm_num : (-1 / 2 : ℝ) < 0)
  intro h
  obtain ⟨h1, h3⟩ := log_limit_below (-1 / 2 : ℝ) hx
  have hcl : ((-1 / 2 : ℝ) : ℂ) ∈ closure {z : ℂ | z.im < 0} := by
    rw [closure_setOfPred_im_lt]
    simp
  haveI : (𝓝[{z : ℂ | z.im < 0}] ((-1 / 2 : ℝ) : ℂ)).NeBot := mem_closure_iff_nhdsWithin_neBot.mp hcl
  have h2 := tendsto_nhds_unique (h.tendsto.mono_left nhdsWithin_le_nhds) h1
  rw [h2] at h3
  simp only [Complex.sub_im, Complex.ofReal_im, Complex.mul_im, Complex.ofReal_re, Complex.I_re, Complex.I_im,
    mul_zero, mul_one, zero_add, zero_sub] at h3
  linarith [Real.pi_pos]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_not_continuousAt
-- test: ca-holo-log-cut-test.sage

/-! ## exm-holomorphic-nowhere -/

/-- `|z|` is not complex differentiable at any `z ≠ 0` (else `|z|² = |z| |z|` would be). -/
theorem abs_not_differentiableAt (z : ℂ) (hz : z ≠ 0) : ¬ DifferentiableAt ℂ (fun w : ℂ => (‖w‖ : ℂ)) z := by
  intro h
  have h2 : DifferentiableAt ℂ (fun w : ℂ => w * conj w) z := by
    have e : (fun w : ℂ => w * conj w) = fun w => (‖w‖ : ℂ) * (‖w‖ : ℂ) := by
      funext w
      rw [mul_conj, normSq_eq_norm_sq]
      push_cast
      ring
    rw [e]
    exact h.mul h
  exact hz (((ComplexDerivative.normSq_differentiableAt_iff z).1).1 h2)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.abs_not_differentiableAt (1) ((by norm_num [Complex.ext_iff])) := by
  intro h
  have h2 : DifferentiableAt ℂ (fun w : ℂ => w * (starRingEnd ℂ) w) (1) := by
    have e : (fun w : ℂ => w * (starRingEnd ℂ) w) = fun w => (‖w‖ : ℂ) * (‖w‖ : ℂ) := by
      funext w
      rw [Complex.mul_conj, Complex.normSq_eq_norm_sq]
      push_cast
      ring
    rw [e]
    exact h.mul h
  exact (by norm_num [Complex.ext_iff] : (1 : ℂ) ≠ 0) (((KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff (1)).1).1 h2)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.abs_not_differentiableAt (Complex.I) ((by norm_num [Complex.ext_iff])) := by
  intro h
  have h2 : DifferentiableAt ℂ (fun w : ℂ => w * (starRingEnd ℂ) w) (Complex.I) := by
    have e : (fun w : ℂ => w * (starRingEnd ℂ) w) = fun w => (‖w‖ : ℂ) * (‖w‖ : ℂ) := by
      funext w
      rw [Complex.mul_conj, Complex.normSq_eq_norm_sq]
      push_cast
      ring
    rw [e]
    exact h.mul h
  exact (by norm_num [Complex.ext_iff] : (Complex.I : ℂ) ≠ 0) (((KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff (Complex.I)).1).1 h2)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.abs_not_differentiableAt ((1 + Complex.I)) ((by norm_num [Complex.ext_iff])) := by
  intro h
  have h2 : DifferentiableAt ℂ (fun w : ℂ => w * (starRingEnd ℂ) w) (1 + Complex.I) := by
    have e : (fun w : ℂ => w * (starRingEnd ℂ) w) = fun w => (‖w‖ : ℂ) * (‖w‖ : ℂ) := by
      funext w
      rw [Complex.mul_conj, Complex.normSq_eq_norm_sq]
      push_cast
      ring
    rw [e]
    exact h.mul h
  exact (by norm_num [Complex.ext_iff] : (1 + Complex.I : ℂ) ≠ 0) (((KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff (1 + Complex.I)).1).1 h2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.abs_not_differentiableAt
-- test: ca-holo-abs-test.sage

/-- A neighbourhood of any point contains a non-zero point. -/
theorem exists_ne_zero_of_eventually {P : ℂ → Prop} {z : ℂ} (h : ∀ᶠ w in 𝓝 z, P w) : ∃ w, w ≠ 0 ∧ P w := by
  have h0 : ∀ᶠ w in 𝓝[≠] z, w ≠ 0 := by
    by_cases hz : z = 0
    · subst hz
      exact self_mem_nhdsWithin
    · exact nhdsWithin_le_nhds (eventually_ne_nhds hz)
  obtain ⟨w, hw1, hw2⟩ := (h0.and (nhdsWithin_le_nhds h)).exists
  exact ⟨w, hw1, hw2⟩

/-- `z̄`, `Re z`, `|z|` and `|z|²` are holomorphic at no point. -/
theorem nowhere_holomorphic (z : ℂ) :
    ¬ HolomorphicAt (fun w => conj w) z ∧ ¬ HolomorphicAt (fun w => (w.re : ℂ)) z ∧
      ¬ HolomorphicAt (fun w => (‖w‖ : ℂ)) z ∧ ¬ HolomorphicAt (fun w => w * conj w) z := by
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_⟩
  · exact (ComplexDerivative.conj_not_differentiableAt _).2 (Filter.Eventually.self_of_nhds h)
  · exact ComplexDerivative.re_not_differentiableAt _ (Filter.Eventually.self_of_nhds h)
  · obtain ⟨w, hw, hd⟩ := exists_ne_zero_of_eventually h
    exact abs_not_differentiableAt w hw hd
  · obtain ⟨w, hw, hd⟩ := exists_ne_zero_of_eventually h
    exact hw (((ComplexDerivative.normSq_differentiableAt_iff w).1).1 hd)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.nowhere_holomorphic (0) := by
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_⟩
  · exact (KaluzaKlein.ComplexAnalysis.ComplexDerivative.conj_not_differentiableAt _).2 (Filter.Eventually.self_of_nhds h)
  · exact KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_not_differentiableAt _ (Filter.Eventually.self_of_nhds h)
  · obtain ⟨w, hw, hd⟩ := KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.exists_ne_zero_of_eventually h
    exact KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.abs_not_differentiableAt w hw hd
  · obtain ⟨w, hw, hd⟩ := KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.exists_ne_zero_of_eventually h
    exact hw (((KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff w).1).1 hd)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.nowhere_holomorphic (1) := by
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_⟩
  · exact (KaluzaKlein.ComplexAnalysis.ComplexDerivative.conj_not_differentiableAt _).2 (Filter.Eventually.self_of_nhds h)
  · exact KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_not_differentiableAt _ (Filter.Eventually.self_of_nhds h)
  · obtain ⟨w, hw, hd⟩ := KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.exists_ne_zero_of_eventually h
    exact KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.abs_not_differentiableAt w hw hd
  · obtain ⟨w, hw, hd⟩ := KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.exists_ne_zero_of_eventually h
    exact hw (((KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff w).1).1 hd)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.nowhere_holomorphic (Complex.I) := by
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_⟩
  · exact (KaluzaKlein.ComplexAnalysis.ComplexDerivative.conj_not_differentiableAt _).2 (Filter.Eventually.self_of_nhds h)
  · exact KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_not_differentiableAt _ (Filter.Eventually.self_of_nhds h)
  · obtain ⟨w, hw, hd⟩ := KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.exists_ne_zero_of_eventually h
    exact KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.abs_not_differentiableAt w hw hd
  · obtain ⟨w, hw, hd⟩ := KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.exists_ne_zero_of_eventually h
    exact hw (((KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff w).1).1 hd)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.nowhere_holomorphic
-- test: ca-holo-nowhere-test.sage

/-! ## exr-holomorphic-polynomials -/

/-- `f(z + h) g(z + h) - f(z) g(z) = (f(z + h) - f(z)) g(z + h) + f(z) (g(z + h) - g(z))`. -/
theorem product_split (a b c d : ℂ) : a * b - c * d = (a - c) * b + c * (b - d) := by
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- Differentiable implies continuous: `f(z + h) - f(z) → 0` as `h → 0`. -/
theorem tendsto_of_hasDerivAt (f : ℂ → ℂ) (f' z : ℂ) (hf : HasDerivAt f f' z) :
    Tendsto (fun h => f (z + h) - f z) (𝓝 0) (𝓝 0) := by
  have h1 : Tendsto (fun h : ℂ => z + h) (𝓝 0) (𝓝 z) := by
    simpa using (tendsto_const_nhds (x := z)).add (tendsto_id (x := 𝓝 (0 : ℂ)))
  simpa using (hf.continuousAt.tendsto.comp h1).sub_const (f z)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.tendsto_of_hasDerivAt ((fun w : ℂ => w)) (1) (0) ((hasDerivAt_id' 0)) := by
  have h1 : Filter.Tendsto (fun h : ℂ => (0 : ℂ) + h) (nhds 0) (nhds (0 : ℂ)) := by
    simpa using (tendsto_const_nhds (x := (0 : ℂ))).add (Filter.tendsto_id (x := nhds (0 : ℂ)))
  simpa using ((hasDerivAt_id' (0 : ℂ)).continuousAt.tendsto.comp h1).sub_const ((fun w : ℂ => w) (0 : ℂ))

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.tendsto_of_hasDerivAt ((fun w : ℂ => w)) (1) (1) ((hasDerivAt_id' 1)) := by
  have h1 : Filter.Tendsto (fun h : ℂ => (1 : ℂ) + h) (nhds 0) (nhds (1 : ℂ)) := by
    simpa using (tendsto_const_nhds (x := (1 : ℂ))).add (Filter.tendsto_id (x := nhds (0 : ℂ)))
  simpa using ((hasDerivAt_id' (1 : ℂ)).continuousAt.tendsto.comp h1).sub_const ((fun w : ℂ => w) (1 : ℂ))

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.tendsto_of_hasDerivAt ((fun w : ℂ => 3 * w)) ((3 * 1)) (Complex.I) (((hasDerivAt_id' Complex.I).const_mul 3)) := by
  have h1 : Filter.Tendsto (fun h : ℂ => Complex.I + h) (nhds 0) (nhds Complex.I) := by
    simpa using (tendsto_const_nhds (x := Complex.I)).add (Filter.tendsto_id (x := nhds (0 : ℂ)))
  simpa using (((hasDerivAt_id' Complex.I).const_mul (3 : ℂ)).continuousAt.tendsto.comp h1).sub_const ((fun w : ℂ => 3 * w) Complex.I)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.tendsto_of_hasDerivAt
-- test: ca-holo-continuity-test.sage

/-- If `f`, `g` are holomorphic on an open `D`, so are `f + g` and `fg`, with `(fg)' = f'g + fg'`. -/
theorem holo_add_mul (f g : ℂ → ℂ) (U : Set ℂ) (hU : IsOpen U) (hf : DifferentiableOn ℂ f U)
    (hg : DifferentiableOn ℂ g U) :
    DifferentiableOn ℂ (fun z => f z + g z) U ∧ DifferentiableOn ℂ (fun z => f z * g z) U ∧
      ∀ z ∈ U, deriv (fun w => f w * g w) z = deriv f z * g z + f z * deriv g z := by
  refine ⟨hf.add hg, hf.mul hg, fun z hz => ?_⟩
  exact deriv_mul (hf.differentiableAt (hU.mem_nhds hz)) (hg.differentiableAt (hU.mem_nhds hz))

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.holo_add_mul ((fun w : ℂ => w)) ((fun _ : ℂ => (1 : ℂ))) (Set.univ) (isOpen_univ) ((by fun_prop)) ((by fun_prop)) := by
  beta_reduce
  refine ⟨by fun_prop, by fun_prop, fun z _ => ?_⟩
  exact deriv_mul (by fun_prop) (by fun_prop)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.holo_add_mul ((fun z : ℂ => z ^ 2)) ((fun z : ℂ => z + 3)) ((Metric.ball 0 1)) (Metric.isOpen_ball) ((by fun_prop)) ((by fun_prop)) := by
  beta_reduce
  refine ⟨by fun_prop, by fun_prop, fun z _ => ?_⟩
  exact deriv_mul (by fun_prop) (by fun_prop)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.holo_add_mul ((fun z : ℂ => z * Complex.I)) ((fun z : ℂ => z ^ 3)) (Set.univ) (isOpen_univ) ((by fun_prop)) ((by fun_prop)) := by
  beta_reduce
  refine ⟨by fun_prop, by fun_prop, fun z _ => ?_⟩
  exact deriv_mul (by fun_prop) (by fun_prop)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.holo_add_mul
-- test: ca-holo-add-mul-test.sage

/-! ## exr-holomorphic-rational-domain -/

/-- `z⁴ - 1 = (z² + 1)(z² - 1)`. -/
theorem quartic_factor (z : ℂ) : z ^ 4 - 1 = (z ^ 2 + 1) * (z ^ 2 - 1) := by
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- After cancelling, `(z² + 1)/(z⁴ - 1) = 1/(z² - 1)`. -/
theorem rational_cancel (z : ℂ) (h : z ^ 4 - 1 ≠ 0) : (z ^ 2 + 1) / (z ^ 4 - 1) = 1 / (z ^ 2 - 1) := by
  have h2 : z ^ 2 - 1 ≠ 0 := fun h0 => h (by rw [quartic_factor, h0, mul_zero])
  rw [div_eq_div_iff h h2]
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `z² - 1 = (z - 1)(z + 1)`. -/
theorem sq_sub_one_factor (z : ℂ) : z ^ 2 - 1 = (z - 1) * (z + 1) := by
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- The zeros of `z² - 1` are `1` and `-1`. -/
theorem sq_sub_one_eq_zero_iff (z : ℂ) : z ^ 2 - 1 = 0 ↔ z = 1 ∨ z = -1 := by
  rw [sq_sub_one_factor, mul_eq_zero, sub_eq_zero, add_eq_zero_iff_eq_neg]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.sq_sub_one_eq_zero_iff (0) := by
  norm_num [Complex.ext_iff, pow_two]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.sq_sub_one_eq_zero_iff (1) := by
  norm_num [Complex.ext_iff, pow_two]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.sq_sub_one_eq_zero_iff ((-1)) := by
  norm_num [Complex.ext_iff, pow_two]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.sq_sub_one_eq_zero_iff
-- test: ca-holo-sq-sub-one-test.sage

/-- `1/(z² - 1)` is holomorphic on `ℂ ∖ {1, -1}`. -/
theorem cancelled_differentiableAt (z : ℂ) (h1 : z ≠ 1) (h2 : z ≠ -1) :
    DifferentiableAt ℂ (fun w : ℂ => 1 / (w ^ 2 - 1)) z := by
  refine DifferentiableAt.div (by fun_prop) (by fun_prop) ?_
  rw [Ne, sq_sub_one_eq_zero_iff]
  tauto

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cancelled_differentiableAt (0) ((by norm_num [Complex.ext_iff])) ((by norm_num [Complex.ext_iff])) := by
  exact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff, pow_two])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cancelled_differentiableAt (Complex.I) ((by norm_num [Complex.ext_iff])) ((by norm_num [Complex.ext_iff])) := by
  exact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff, pow_two])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cancelled_differentiableAt (2) ((by norm_num [Complex.ext_iff])) ((by norm_num [Complex.ext_iff])) := by
  exact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff, pow_two])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cancelled_differentiableAt
-- test: ca-holo-cancelled-test.sage

/-- `cos z = 0` exactly at `z = π/2 + kπ`. -/
theorem cos_eq_zero_iff_claim (z : ℂ) : Complex.cos z = 0 ↔ ∃ k : ℤ, z = π / 2 + k * π := by
  rw [Complex.cos_eq_zero_iff]
  constructor <;> rintro ⟨k, hk⟩ <;> exact ⟨k, by linear_combination hk⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cos_eq_zero_iff_claim (0) := by
  rw [Complex.cos_eq_zero_iff]
  constructor <;> rintro ⟨k, hk⟩ <;> exact ⟨k, by linear_combination hk⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cos_eq_zero_iff_claim (((Real.pi : ℂ) / 2)) := by
  rw [Complex.cos_eq_zero_iff]
  constructor <;> rintro ⟨k, hk⟩ <;> exact ⟨k, by linear_combination hk⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cos_eq_zero_iff_claim ((3 * (Real.pi : ℂ) / 2)) := by
  rw [Complex.cos_eq_zero_iff]
  constructor <;> rintro ⟨k, hk⟩ <;> exact ⟨k, by linear_combination hk⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cos_eq_zero_iff_claim
-- test: ca-holo-cos-zero-test.sage

/-- `cos z = 0` forces `e^{2iz} = -1`. -/
theorem cos_zero_exp (z : ℂ) (h : Complex.cos z = 0) : Complex.exp (2 * I * z) = -1 := by
  have h1 : Complex.exp (z * I) + Complex.exp (-z * I) = 0 := by rw [← two_cos, h, mul_zero]
  have h2 : Complex.exp (z * I) * Complex.exp (-z * I) = 1 := by rw [← Complex.exp_add]; simp
  have h3 : Complex.exp (2 * I * z) = Complex.exp (z * I) ^ 2 := by rw [sq, ← Complex.exp_add]; ring_nf
  generalize Complex.exp (2 * I * z) = E2 at *
  generalize Complex.exp (z * I) = E at *
  generalize Complex.exp (-z * I) = F at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination h3 + (E) * h1 + (-1) * h2)
  -- certificate: ca-holo-ring.sage

/-- The zeros of `cos` are real. -/
theorem cos_zero_real (z : ℂ) (h : Complex.cos z = 0) : z.im = 0 := by
  obtain ⟨k, rfl⟩ := (cos_eq_zero_iff_claim z).1 h
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cos_zero_real (((Real.pi : ℂ) / 2 + ((0 : ℤ) : ℂ) * (Real.pi : ℂ))) ((by rw [Complex.cos_eq_zero_iff]; exact ⟨0, by push_cast; ring⟩)) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cos_zero_real (((Real.pi : ℂ) / 2 + ((1 : ℤ) : ℂ) * (Real.pi : ℂ))) ((by rw [Complex.cos_eq_zero_iff]; exact ⟨1, by push_cast; ring⟩)) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cos_zero_real (((Real.pi : ℂ) / 2 + ((-1 : ℤ) : ℂ) * (Real.pi : ℂ))) ((by rw [Complex.cos_eq_zero_iff]; exact ⟨-1, by push_cast; ring⟩)) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cos_zero_real
-- test: ca-holo-cos-real-test.sage

/-- `tan z = sin z/cos z` is holomorphic where `cos z ≠ 0`. -/
theorem tan_differentiableAt (z : ℂ) (h : Complex.cos z ≠ 0) : DifferentiableAt ℂ Complex.tan z :=
  Complex.differentiableAt_tan.mpr h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.tan_differentiableAt (0) ((by simp)) := by
  exact Complex.differentiableAt_tan.mpr (by simp)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.tan_differentiableAt ((Real.pi : ℂ)) ((by simp)) := by
  exact Complex.differentiableAt_tan.mpr (by simp)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.tan_differentiableAt ((2 * (Real.pi : ℂ))) ((by simp)) := by
  exact Complex.differentiableAt_tan.mpr (by simp)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.tan_differentiableAt
-- test: ca-holo-tan-test.sage

/-! ## exr-holomorphic-abs -/

/-- The linear system `u u_x + v v_x = 0`, `v u_x - u v_x = 0` has determinant `u² + v²`: `(u² + v²) u_x = 0` and
`(u² + v²) v_x = 0`. -/
theorem cr_system (u v a b : ℝ) (h1 : u * a + v * b = 0) (h2 : v * a - u * b = 0) :
    (u ^ 2 + v ^ 2) * a = 0 ∧ (u ^ 2 + v ^ 2) * b = 0 := by
  refine ⟨?_, ?_⟩
  · rw [← sub_eq_zero]
    veriphy_certified (linear_combination (u) * h1 + (v) * h2)
    -- certificate: ca-holo-ring.sage
  · rw [← sub_eq_zero]
    veriphy_certified (linear_combination (v) * h1 + (-u) * h2)
    -- certificate: ca-holo-ring.sage

/-- A holomorphic function of constant modulus on a connected open set is constant. -/
theorem const_of_norm_const (f : ℂ → ℂ) (U : Set ℂ) (hU : IsOpen U) (hc : IsPreconnected U)
    (hf : DifferentiableOn ℂ f U) (c : ℝ) (h : ∀ z ∈ U, ‖f z‖ = c) : ∃ w, ∀ z ∈ U, f z = w := by
  rcases U.eq_empty_or_nonempty with rfl | ⟨z0, hz0⟩
  · exact ⟨0, by simp⟩
  · refine ⟨f z0, fun z hz => ?_⟩
    have hmax : IsMaxOn (norm ∘ f) U z0 := fun w hw => by simp [Function.comp, h w hw, h z0 hz0]
    exact Complex.eqOn_of_isPreconnected_of_isMaxOn_norm hc hU hf hz0 hmax hz

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.const_of_norm_const ((fun _ : ℂ => (0 : ℂ))) (Set.univ) (isOpen_univ) (isPreconnected_univ) ((differentiableOn_const 0)) (0) ((by simp)) := by
  exact ⟨_, fun _ _ => rfl⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.const_of_norm_const ((fun _ : ℂ => (1 : ℂ))) ((Metric.ball 0 1)) (Metric.isOpen_ball) ((convex_ball (0 : ℂ) 1).isPreconnected) ((differentiableOn_const 1)) (1) ((by simp)) := by
  exact ⟨_, fun _ _ => rfl⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.const_of_norm_const ((fun _ : ℂ => Complex.I)) (Set.univ) (isOpen_univ) (isPreconnected_univ) ((differentiableOn_const Complex.I)) (1) ((by simp)) := by
  exact ⟨_, fun _ _ => rfl⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.const_of_norm_const
-- test: ca-holo-const-norm-test.sage

end KaluzaKlein.ComplexAnalysis.HolomorphicFunctions
