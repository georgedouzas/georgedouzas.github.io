import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Analytic continuation"

The claims of the section "Analytic continuation" of `tutorial/part-i-04-complex-analysis.qmd`: the identity
theorem and the uniqueness of analytic continuation, the geometric series and its continuation `1/(1 - z)` (and
`1/(1 - z)²` for `∑ (n + 1) zⁿ`), identities between entire functions continued from the real axis, the logarithm
and the square root continued along the unit circle, and the impossibility of an entire extension of `|x|`.
-/

namespace KaluzaKlein.ComplexAnalysis.AnalyticContinuation

open Complex Filter Topology Set
open scoped Real

/-! ## def-analytic-continuation -/

/-- The identity theorem: two functions holomorphic on a connected open set `U` that agree on a set accumulating at
a point `z₀ ∈ U` agree on `U`. -/
theorem identity_theorem (f g : ℂ → ℂ) (U : Set ℂ) (hU : IsOpen U) (hc : IsPreconnected U)
    (hf : DifferentiableOn ℂ f U) (hg : DifferentiableOn ℂ g U) (z₀ : ℂ) (hz₀ : z₀ ∈ U)
    (h : ∃ᶠ z in 𝓝[≠] z₀, f z = g z) : EqOn f g U :=
  (hf.analyticOnNhd hU).eqOn_of_preconnected_of_frequently_eq (hg.analyticOnNhd hU) hc hz₀ h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.identity_theorem (Complex.exp) (Complex.exp) (Set.univ) (isOpen_univ) (isPreconnected_univ) (Complex.differentiable_exp.differentiableOn) (Complex.differentiable_exp.differentiableOn) (0) (Set.mem_univ 0) (Filter.Frequently.of_forall (fun _ => rfl)) := by
  exact fun _ _ => rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.identity_theorem (fun z : ℂ => (z + 1) ^ 2) (fun z : ℂ => z ^ 2 + 2 * z + 1) (Metric.ball (0 : ℂ) 1) (Metric.isOpen_ball) ((convex_ball (0 : ℂ) 1).isPreconnected) (by fun_prop) (by fun_prop) (0) (Metric.mem_ball_self one_pos) (Filter.Frequently.of_forall (fun z => by beta_reduce; ring)) := by
  intro z _
  beta_reduce
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.identity_theorem (Complex.cos) (fun z : ℂ => Complex.cos (-z)) (Set.univ) (isOpen_univ) (isPreconnected_univ) (Complex.differentiable_cos.differentiableOn) (by fun_prop) (0) (Set.mem_univ 0) (Filter.Frequently.of_forall (fun z => by simp)) := by
  intro z _
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.identity_theorem
-- test: ca-spec-identity-test.sage

/-- Uniqueness of analytic continuation: two holomorphic functions on a connected open `D₂` that both agree with `f`
on a nonempty open `D₁ ⊆ D₂` agree on `D₂`. -/
theorem continuation_unique (D₁ D₂ : Set ℂ) (hD₂ : IsOpen D₂) (hc : IsPreconnected D₂) (hD₁ : IsOpen D₁)
    (hne : D₁.Nonempty) (hsub : D₁ ⊆ D₂) (f F G : ℂ → ℂ) (hF : DifferentiableOn ℂ F D₂)
    (hG : DifferentiableOn ℂ G D₂) (hFf : EqOn F f D₁) (hGf : EqOn G f D₁) : EqOn F G D₂ := by
  obtain ⟨z₀, hz₀⟩ := hne
  have hev : F =ᶠ[𝓝 z₀] G := by
    filter_upwards [hD₁.mem_nhds hz₀] with z hz
    rw [hFf hz, hGf hz]
  exact (hF.analyticOnNhd hD₂).eqOn_of_preconnected_of_eventuallyEq (hG.analyticOnNhd hD₂) hc (hsub hz₀) hev

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.continuation_unique (Metric.ball 0 1) (Set.univ) (isOpen_univ) (isPreconnected_univ) (Metric.isOpen_ball) (⟨0, Metric.mem_ball_self one_pos⟩) (Set.subset_univ _) (Complex.exp) (Complex.exp) (Complex.exp) (Complex.differentiable_exp.differentiableOn) (Complex.differentiable_exp.differentiableOn) (fun _ _ => rfl) (fun _ _ => rfl) := by
  exact fun _ _ => rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.continuation_unique (Metric.ball 0 1) (Set.univ) (isOpen_univ) (isPreconnected_univ) (Metric.isOpen_ball) (⟨0, Metric.mem_ball_self one_pos⟩) (Set.subset_univ _) (fun z : ℂ => (z + 1) ^ 2) (fun z : ℂ => (z + 1) ^ 2) (fun z : ℂ => z ^ 2 + 2 * z + 1) (by fun_prop) (by fun_prop) (fun _ _ => rfl) (fun z _ => by beta_reduce; ring) := by
  intro z _
  beta_reduce
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.continuation_unique (Set.univ) (Set.univ) (isOpen_univ) (isPreconnected_univ) (isOpen_univ) (⟨0, Set.mem_univ 0⟩) (subset_refl _) (Complex.sin) (Complex.sin) (fun z : ℂ => -Complex.sin (-z)) (Complex.differentiable_sin.differentiableOn) (by fun_prop) (fun _ _ => rfl) (fun z _ => by simp) := by
  intro z _
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.continuation_unique
-- test: ca-spec-unique-test.sage

/-! ## exm-continuation-geometric, exr-continuation-geometric -/

/-- `∑ zⁿ = 1/(1 - z)` on the unit disc. -/
theorem geometric_hasSum (z : ℂ) (hz : ‖z‖ < 1) : HasSum (fun n : ℕ => z ^ n) (1 / (1 - z)) := by
  simpa [one_div] using hasSum_geometric_of_norm_lt_one hz

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_hasSum
-- test: ca-spec-geom-test.sage

/-- `∑ (n + 1) zⁿ = 1/(1 - z)²` on the unit disc. -/
theorem geometric_succ_hasSum (z : ℂ) (hz : ‖z‖ < 1) :
    HasSum (fun n : ℕ => ((n : ℂ) + 1) * z ^ n) (1 / (1 - z) ^ 2) := by
  have h1 := hasSum_coe_mul_geometric_of_norm_lt_one hz
  have h2 := hasSum_geometric_of_norm_lt_one hz
  have hz1 : (1 : ℂ) - z ≠ 0 := by
    intro h
    have : z = 1 := by linear_combination -h
    rw [this, norm_one] at hz
    exact lt_irrefl _ hz
  convert h1.add h2 using 1
  · funext n
    ring
  · field_simp
    ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_succ_hasSum
-- test: ca-spec-geom-succ-test.sage

/-- The geometric series converges only on the unit disc: it diverges for `|z| ≥ 1` (`1 + 2 + 4 + …` at `z = 2`). -/
theorem geometric_not_summable (z : ℂ) (hz : 1 ≤ ‖z‖) : ¬ Summable (fun n : ℕ => z ^ n) := by
  rw [summable_geometric_iff_norm_lt_one]
  exact not_lt.mpr hz

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_not_summable (1) ((by norm_num)) := by
  rw [summable_geometric_iff_norm_lt_one]
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_not_summable (2) ((by norm_num)) := by
  rw [summable_geometric_iff_norm_lt_one]
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_not_summable (-3) ((by norm_num)) := by
  rw [summable_geometric_iff_norm_lt_one]
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_not_summable (Complex.I) ((by simp)) := by
  rw [summable_geometric_iff_norm_lt_one]
  simp

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_not_summable
-- test: ca-spec-geom-div-test.sage

/-- The closed forms `1/(a - z)` and `1/(a - z)²` are holomorphic on `ℂ ∖ {a}` (for `a = 1`: the continuations of
the two series). -/
theorem closed_forms_differentiableOn (a : ℂ) :
    DifferentiableOn ℂ (fun z => 1 / (a - z)) {a}ᶜ ∧ DifferentiableOn ℂ (fun z => 1 / (a - z) ^ 2) {a}ᶜ := by
  have hne : ∀ z ∈ ({a}ᶜ : Set ℂ), a - z ≠ 0 := fun z hz => sub_ne_zero.mpr (Ne.symm hz)
  refine ⟨?_, ?_⟩
  · exact (differentiableOn_const 1).div ((differentiableOn_const a).sub differentiableOn_id) hne
  · exact (differentiableOn_const 1).div (((differentiableOn_const a).sub differentiableOn_id).pow 2)
      (fun z hz => pow_ne_zero 2 (hne z hz))

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.closed_forms_differentiableOn (0) := by
  refine ⟨(differentiableOn_const 1).div ((differentiableOn_const _).sub differentiableOn_id) (fun z hz => sub_ne_zero.mpr (Ne.symm hz)), (differentiableOn_const 1).div (((differentiableOn_const _).sub differentiableOn_id).pow 2) (fun z hz => pow_ne_zero 2 (sub_ne_zero.mpr (Ne.symm hz)))⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.closed_forms_differentiableOn (1) := by
  refine ⟨(differentiableOn_const 1).div ((differentiableOn_const _).sub differentiableOn_id) (fun z hz => sub_ne_zero.mpr (Ne.symm hz)), (differentiableOn_const 1).div (((differentiableOn_const _).sub differentiableOn_id).pow 2) (fun z hz => pow_ne_zero 2 (sub_ne_zero.mpr (Ne.symm hz)))⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.closed_forms_differentiableOn (Complex.I) := by
  refine ⟨(differentiableOn_const 1).div ((differentiableOn_const _).sub differentiableOn_id) (fun z hz => sub_ne_zero.mpr (Ne.symm hz)), (differentiableOn_const 1).div (((differentiableOn_const _).sub differentiableOn_id).pow 2) (fun z hz => pow_ne_zero 2 (sub_ne_zero.mpr (Ne.symm hz)))⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.closed_forms_differentiableOn
-- test: ca-spec-closed-forms-test.sage

/-- `1/(1 - z)` is the unique analytic continuation of `∑ zⁿ`: a function holomorphic on `ℂ ∖ {1}` that equals the
sum of the series on the unit disc is `1/(1 - z)`. -/
theorem geometric_continuation_unique (F : ℂ → ℂ) (hF : DifferentiableOn ℂ F {1}ᶜ)
    (h : ∀ z : ℂ, ‖z‖ < 1 → HasSum (fun n : ℕ => z ^ n) (F z)) (z : ℂ) (hz : z ≠ 1) : F z * (1 - z) = 1 := by
  have hU : IsOpen ({1}ᶜ : Set ℂ) := isOpen_compl_singleton
  have hc : IsPreconnected ({1}ᶜ : Set ℂ) :=
    (isConnected_compl_singleton_of_one_lt_rank (by simp) 1).isPreconnected
  have hG := (closed_forms_differentiableOn 1).1
  have hev : F =ᶠ[𝓝 0] fun z => 1 / (1 - z) := by
    filter_upwards [Metric.ball_mem_nhds (0 : ℂ) one_pos] with z hz
    have hz' : ‖z‖ < 1 := by simpa using hz
    exact (h z hz').unique (geometric_hasSum z hz')
  have hFz : F z = 1 / (1 - z) :=
    (hF.analyticOnNhd hU).eqOn_of_preconnected_of_eventuallyEq (hG.analyticOnNhd hU) hc (by simp) hev hz
  have hz1 : (1 : ℂ) - z ≠ 0 := sub_ne_zero.mpr (Ne.symm hz)
  rw [hFz, one_div, inv_mul_cancel₀ hz1]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_continuation_unique (fun w : ℂ => 1 / (1 - w)) ((KaluzaKlein.ComplexAnalysis.AnalyticContinuation.closed_forms_differentiableOn 1).1) (fun w hw => KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_hasSum w hw) (0) ((by norm_num)) := by
  beta_reduce
  rw [one_div, inv_mul_cancel₀ (by norm_num [Complex.ext_iff])]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_continuation_unique (fun w : ℂ => 1 / (1 - w)) ((KaluzaKlein.ComplexAnalysis.AnalyticContinuation.closed_forms_differentiableOn 1).1) (fun w hw => KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_hasSum w hw) (2) ((by norm_num)) := by
  beta_reduce
  rw [one_div, inv_mul_cancel₀ (by norm_num [Complex.ext_iff])]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_continuation_unique (fun w : ℂ => 1 / (1 - w)) ((KaluzaKlein.ComplexAnalysis.AnalyticContinuation.closed_forms_differentiableOn 1).1) (fun w hw => KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_hasSum w hw) (-3) ((by norm_num)) := by
  beta_reduce
  rw [one_div, inv_mul_cancel₀ (by norm_num [Complex.ext_iff])]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_continuation_unique (fun w : ℂ => 1 / (1 - w)) ((KaluzaKlein.ComplexAnalysis.AnalyticContinuation.closed_forms_differentiableOn 1).1) (fun w hw => KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_hasSum w hw) (Complex.I) ((by simp [Complex.ext_iff])) := by
  beta_reduce
  rw [one_div, inv_mul_cancel₀ (by norm_num [Complex.ext_iff])]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_continuation_unique
-- test: ca-spec-cont-unique-test.sage

/-- The values `-1` at `z = 2` (and `1/4` at `z = -3`) belong to the continuation `F`, characterised by
`F z (1 - z) = 1`; the closed form of `∑ (n + 1) zⁿ`, characterised by `G z (1 - z)² = 1`, takes `1` and `1/16`. -/
theorem continuation_values (F G : ℂ → ℂ) (hF : ∀ z : ℂ, z ≠ 1 → F z * (1 - z) = 1)
    (hG : ∀ z : ℂ, z ≠ 1 → G z * (1 - z) ^ 2 = 1) :
    F 2 = -1 ∧ G 2 = 1 ∧ F (-3) = 1 / 4 ∧ G (-3) = 1 / 16 := by
  have h1 : F 2 * (1 - 2) = 1 := hF 2 (by norm_num)
  have h2 : G 2 * (1 - 2) ^ 2 = 1 := hG 2 (by norm_num)
  have h3 : F (-3) * (1 - -3) = 1 := hF (-3) (by norm_num)
  have h4 : G (-3) * (1 - -3) ^ 2 = 1 := hG (-3) (by norm_num)
  refine ⟨?_, ?_, ?_, ?_⟩
  · veriphy_certified (linear_combination (-1) * h1)
    -- certificate: ideal_membership.sage
  · veriphy_certified (linear_combination h2)
    -- certificate: ideal_membership.sage
  · veriphy_certified (linear_combination (1 / 4) * h3)
    -- certificate: ideal_membership.sage
  · veriphy_certified (linear_combination (1 / 16) * h4)
    -- certificate: ideal_membership.sage

/-! ## exm-continuation-identities, exr-continuation-identity -/

/-- Real points accumulate at `0` in `ℂ`. -/
theorem frequently_real (P : ℂ → Prop) (h : ∀ x : ℝ, P x) : ∃ᶠ z in 𝓝[≠] (0 : ℂ), P z := by
  have ht : Tendsto (fun n : ℕ => ((1 / ((n : ℝ) + 1) : ℝ) : ℂ)) atTop (𝓝[≠] (0 : ℂ)) := by
    refine tendsto_nhdsWithin_iff.mpr ⟨?_, Eventually.of_forall fun n => ?_⟩
    · have h0 : Tendsto (fun n : ℕ => 1 / ((n : ℝ) + 1)) atTop (𝓝 0) := tendsto_one_div_add_atTop_nhds_zero_nat
      have := (Complex.continuous_ofReal.tendsto 0).comp h0
      rw [Complex.ofReal_zero] at this
      exact this
    · simp only [mem_compl_iff, mem_singleton_iff, Complex.ofReal_eq_zero]
      positivity
  exact ht.frequently (Frequently.of_forall fun n => h _)

/-- An identity between entire functions that holds on the real axis holds on `ℂ`. -/
theorem entire_eq_of_eq_on_real (f g : ℂ → ℂ) (hf : Differentiable ℂ f) (hg : Differentiable ℂ g)
    (h : ∀ x : ℝ, f x = g x) : f = g := by
  funext z
  exact identity_theorem f g univ isOpen_univ isPreconnected_univ hf.differentiableOn hg.differentiableOn 0
    (mem_univ _) (frequently_real (fun z => f z = g z) h) (mem_univ z)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.entire_eq_of_eq_on_real (Complex.exp) (Complex.exp) (Complex.differentiable_exp) (Complex.differentiable_exp) (fun _ => rfl) := by
  rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.entire_eq_of_eq_on_real (fun z : ℂ => Complex.cos z ^ 2 + Complex.sin z ^ 2) (fun _ : ℂ => (1 : ℂ)) (by fun_prop) (differentiable_const 1) (fun x => Complex.cos_sq_add_sin_sq _) := by
  funext z
  exact Complex.cos_sq_add_sin_sq z

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.entire_eq_of_eq_on_real (fun z : ℂ => Complex.cos (-z)) (Complex.cos) (by fun_prop) (Complex.differentiable_cos) (fun x => by simp) := by
  funext z
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.entire_eq_of_eq_on_real
-- test: ca-spec-real-axis-test.sage

/-- `cos² z + sin² z = 1` for every complex `z`, continued from the real identity. -/
theorem cos_sq_add_sin_sq (z : ℂ) : Complex.cos z ^ 2 + Complex.sin z ^ 2 = 1 := by
  have := entire_eq_of_eq_on_real (fun z => Complex.cos z ^ 2 + Complex.sin z ^ 2) (fun _ => 1)
    (by fun_prop) (differentiable_const 1) (fun x => by
      beta_reduce
      rw [← Complex.ofReal_cos, ← Complex.ofReal_sin]
      exact_mod_cast Real.cos_sq_add_sin_sq x)
  exact congrFun this z

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.cos_sq_add_sin_sq
-- test: ca-spec-cos-sin-test.sage

/-- The functional equation `e^{z + w} = e^z e^w` holds for all complex `z, w`. -/
theorem exp_add (z w : ℂ) : Complex.exp (z + w) = Complex.exp z * Complex.exp w :=
  Complex.exp_add z w

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.exp_add
-- test: ca-spec-exp-add-test.sage

/-- An entire function equal to `sin` on `ℝ` is `sin`. -/
theorem eq_sin_of_real (f : ℂ → ℂ) (hf : Differentiable ℂ f) (h : ∀ x : ℝ, f x = Real.sin x) :
    f = Complex.sin :=
  entire_eq_of_eq_on_real f Complex.sin hf Complex.differentiable_sin
    (fun x => by rw [h x, Complex.ofReal_sin])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.eq_sin_of_real (Complex.sin) (Complex.differentiable_sin) (fun x => (Complex.ofReal_sin x).symm) := by
  rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.eq_sin_of_real (fun z : ℂ => -Complex.sin (-z)) (by fun_prop) (fun x => by simp) := by
  funext z
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.eq_sin_of_real (fun z : ℂ => Complex.sin (z + 2 * Real.pi)) (by fun_prop) (fun x => by simp [Complex.sin_add_two_pi]) := by
  funext z
  simp [Complex.sin_add_two_pi]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.eq_sin_of_real
-- test: ca-spec-sin-ext-test.sage

/-- No entire function restricts to `|x - a|` on `ℝ` (for `a = 0`: to `|x|`): its restriction would be
differentiable at `a`. -/
theorem no_entire_abs_sub (a : ℝ) : ¬ ∃ g : ℂ → ℂ, Differentiable ℂ g ∧ ∀ x : ℝ, g x = |x - a| := by
  rintro ⟨g, hg, h⟩
  have hshift : DifferentiableAt ℝ (fun t : ℝ => (((t + a : ℝ)) : ℂ)) 0 :=
    Complex.ofRealCLM.differentiableAt.comp 0 (differentiableAt_id.add_const a)
  have h1 : DifferentiableAt ℝ (fun t : ℝ => g ((t + a : ℝ) : ℂ)) 0 :=
    ((hg _).restrictScalars ℝ).comp 0 hshift
  have hd : DifferentiableAt ℝ (fun t : ℝ => (g ((t + a : ℝ) : ℂ)).re) 0 :=
    Complex.reCLM.differentiableAt.comp 0 h1
  have heq : (fun t : ℝ => (g ((t + a : ℝ) : ℂ)).re) = abs := by
    funext t
    rw [h]
    simp
  rw [heq] at hd
  exact not_differentiableAt_abs_zero hd

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.no_entire_abs_sub
-- test: ca-spec-abs-test.sage

/-! ## exm-continuation-log -/

/-- Along the arc `e^{iθ}`, `-π < θ < π`, the continuation `iθ` of `Log` from `z = 1` agrees with `Log`. -/
theorem log_exp_I (θ : ℝ) (h₁ : -π < θ) (h₂ : θ < π) : Complex.log (Complex.exp (θ * I)) = θ * I := by
  apply Complex.log_exp <;> simp <;> linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.log_exp_I
-- test: ca-spec-log-arc-test.sage

/-- After `n` counter-clockwise turns the continued logarithm `iθ` at `θ = 2πn` is back over `z = 1`, with the
value `Log 1 + 2πin`: the continuation depends on the winding. -/
theorem log_continuation_turns (n : ℕ) :
    Complex.exp (((2 * π * n : ℝ)) * I) = 1 ∧ ((2 * π * n : ℝ) : ℂ) * I = Complex.log 1 + 2 * π * I * n := by
  refine ⟨?_, ?_⟩
  · have := Complex.exp_nat_mul_two_pi_mul_I n
    rw [← this]
    congr 1
    push_cast <;> ring
  · rw [Complex.log_one]
    push_cast <;> ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.log_continuation_turns (0) := by
  refine ⟨?_, ?_⟩
  · have := Complex.exp_nat_mul_two_pi_mul_I 0
    rw [← this]
    congr 1
    push_cast
    ring
  · rw [Complex.log_one]
    push_cast
    ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.log_continuation_turns (1) := by
  refine ⟨?_, ?_⟩
  · have := Complex.exp_nat_mul_two_pi_mul_I 1
    rw [← this]
    congr 1
    push_cast
    ring
  · rw [Complex.log_one]
    push_cast
    ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.log_continuation_turns (2) := by
  refine ⟨?_, ?_⟩
  · have := Complex.exp_nat_mul_two_pi_mul_I 2
    rw [← this]
    congr 1
    push_cast
    ring
  · rw [Complex.log_one]
    push_cast
    ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.AnalyticContinuation.log_continuation_turns (3) := by
  refine ⟨?_, ?_⟩
  · have := Complex.exp_nat_mul_two_pi_mul_I 3
    rw [← this]
    congr 1
    push_cast
    ring
  · rw [Complex.log_one]
    push_cast
    ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.log_continuation_turns
-- test: ca-spec-log-turns-test.sage

/-- Every `Log z + 2πik` is a logarithm of `z`: the continued logarithm is many-valued. -/
theorem exp_log_add (z : ℂ) (hz : z ≠ 0) (k : ℤ) : Complex.exp (Complex.log z + 2 * π * I * k) = z := by
  rw [Complex.exp_add, Complex.exp_log hz]
  have := Complex.exp_int_mul_two_pi_mul_I k
  rw [show (2 * π * I * k : ℂ) = k * (2 * π * I) by ring, this, mul_one]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.exp_log_add
-- test: ca-spec-log-branches-test.sage

/-! ## exr-continuation-monodromy -/

/-- `e^{iθ/2}` is a continuous square root of `e^{iθ}` along the unit circle. -/
theorem sqrt_lift_sq (θ : ℝ) : Complex.exp (θ * I / 2) ^ 2 = Complex.exp (θ * I) := by
  rw [← Complex.exp_nat_mul]
  congr 1
  push_cast <;> ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.sqrt_lift_sq
-- test: ca-spec-sqrt-sq-test.sage

/-- For `-π < θ < π` the lift `e^{iθ/2}` is the principal square root of `e^{iθ}`. -/
theorem sqrt_lift_principal (θ : ℝ) (h₁ : -π < θ) (h₂ : θ < π) :
    Complex.exp (θ * I) ^ (1 / 2 : ℂ) = Complex.exp (θ * I / 2) := by
  rw [Complex.cpow_def_of_ne_zero (Complex.exp_ne_zero _), log_exp_I θ h₁ h₂]
  congr 1
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.sqrt_lift_principal
-- test: ca-spec-sqrt-principal-test.sage

/-- After `n` turns the continued square root at `z = 1` is `e^{iπn} = (-1)ⁿ`: `-1` after one turn, `1` after two. -/
theorem sqrt_lift_turns (n : ℕ) : Complex.exp (((2 * π * n : ℝ)) * I / 2) = (-1) ^ n := by
  rw [show (((2 * π * n : ℝ)) * I / 2 : ℂ) = n * (π * I) by push_cast <;> ring, Complex.exp_nat_mul,
    Complex.exp_pi_mul_I]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.AnalyticContinuation.sqrt_lift_turns
-- test: ca-spec-sqrt-turns-test.sage

end KaluzaKlein.ComplexAnalysis.AnalyticContinuation
