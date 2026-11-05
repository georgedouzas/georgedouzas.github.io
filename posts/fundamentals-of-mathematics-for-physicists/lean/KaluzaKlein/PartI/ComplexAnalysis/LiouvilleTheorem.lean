import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Liouville's theorem"

The claims of the section "Liouville's theorem" of `tutorial/part-i-04-complex-analysis.qmd` (part "Contour integration"): Liouville's theorem, its polynomial-growth form (proved here from the Cauchy estimates and the Taylor series), the unboundedness of `exp`, `cos`, `sin`, the fundamental theorem of algebra by Liouville, and `|sin(x + iy)|² = sin² x + sinh² y`.
-/

open Complex MeasureTheory Metric Filter Topology
open scoped Real

noncomputable section

namespace KaluzaKlein.ComplexAnalysis.LiouvilleTheorem

open Polynomial

theorem liouville (f : ℂ → ℂ) (hf : Differentiable ℂ f) (hb : ∃ M, ∀ z, ‖f z‖ ≤ M) (z w : ℂ) : f z = f w := by
  obtain ⟨M, hM⟩ := hb
  exact hf.apply_eq_apply_of_bounded (isBounded_iff_forall_norm_le.2 ⟨M, by rintro _ ⟨x, rfl⟩; exact hM x⟩) z w

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.liouville ((fun _ : ℂ => (0 : ℂ))) ((differentiable_const _)) (⟨0, fun _ => by simp⟩) (0) (1) := by
  rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.liouville ((fun _ : ℂ => (5 : ℂ))) ((differentiable_const _)) (⟨5, fun _ => by norm_num⟩) (Complex.I) ((2 + Complex.I)) := by
  rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.liouville ((fun _ : ℂ => Complex.I)) ((differentiable_const _)) (⟨1, fun _ => by simp⟩) ((-3)) ((7 * Complex.I)) := by
  rfl

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.liouville
-- test: ca-contour-liouville-test.sage

theorem tendsto_growth (n m : ℕ) : Tendsto (fun r : ℝ => (1 + r) ^ n / r ^ (n + m + 1)) atTop (𝓝 0) := by
  have h1 : Tendsto (fun r : ℝ => (r⁻¹ + 1) ^ n * (r⁻¹) ^ (m + 1)) atTop (𝓝 ((0 + 1) ^ n * 0 ^ (m + 1))) :=
    ((tendsto_inv_atTop_zero.add_const 1).pow n).mul (tendsto_inv_atTop_zero.pow (m + 1))
  rw [zero_pow (Nat.succ_ne_zero m), mul_zero] at h1
  refine h1.congr' ?_
  filter_upwards [eventually_gt_atTop 0] with r hr
  have hr' : r ≠ 0 := hr.ne'
  have e : r⁻¹ + 1 = (1 + r) * r⁻¹ := by rw [add_mul, one_mul, mul_inv_cancel₀ hr']
  rw [e, mul_pow, mul_assoc, ← pow_add, inv_pow, ← div_eq_mul_inv, ← add_assoc]

theorem growth_estimate (f : ℂ → ℂ) (hf : Differentiable ℂ f) (n : ℕ) (C : ℝ)
    (hb : ∀ z, ‖f z‖ ≤ C * (1 + ‖z‖) ^ n) (k : ℕ) (r : ℝ) (hr : 0 < r) :
    ‖iteratedDeriv k f 0‖ ≤ k.factorial * (C * (1 + r) ^ n) / r ^ k :=
  Complex.norm_iteratedDeriv_le_of_forall_mem_sphere_norm_le (c := 0) k hr hf.diffContOnCl (fun z hz => by
    have : ‖z‖ = r := by simpa using hz
    simpa [this] using hb z)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.growth_estimate
-- test: ca-contour-growth-estimate-test.sage

theorem iteratedDeriv_eq_zero_of_growth (f : ℂ → ℂ) (hf : Differentiable ℂ f) (n : ℕ) (C : ℝ)
    (hb : ∀ z, ‖f z‖ ≤ C * (1 + ‖z‖) ^ n) (k : ℕ) (hk : n < k) : iteratedDeriv k f 0 = 0 := by
  obtain ⟨m, rfl⟩ : ∃ m, k = n + m + 1 := ⟨k - n - 1, by omega⟩
  have hlim : Tendsto (fun r : ℝ => ((n + m + 1).factorial : ℝ) * (C * ((1 + r) ^ n / r ^ (n + m + 1))))
      atTop (𝓝 0) := by
    have := ((tendsto_growth n m).const_mul C).const_mul ((n + m + 1).factorial : ℝ)
    rwa [mul_zero, mul_zero] at this
  have hle : ∀ᶠ r in atTop, ‖iteratedDeriv (n + m + 1) f 0‖ ≤
      ((n + m + 1).factorial : ℝ) * (C * ((1 + r) ^ n / r ^ (n + m + 1))) := by
    filter_upwards [eventually_gt_atTop 0] with r hr
    calc _ ≤ _ := growth_estimate f hf n C hb (n + m + 1) r hr
      _ = _ := by ring
  exact norm_le_zero_iff.1 (ge_of_tendsto hlim hle)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.iteratedDeriv_eq_zero_of_growth
-- test: ca-contour-iteratedderiv-eq-zero-of-growth-test.sage

theorem liouville_poly (f : ℂ → ℂ) (hf : Differentiable ℂ f) (n : ℕ) (C : ℝ)
    (hb : ∀ z, ‖f z‖ ≤ C * (1 + ‖z‖) ^ n) :
    ∃ p : ℂ[X], p.natDegree ≤ n ∧ ∀ z, f z = p.eval z := by
  refine ⟨∑ k ∈ Finset.range (n + 1), Polynomial.C ((k.factorial : ℂ)⁻¹ * iteratedDeriv k f 0) * X ^ k, ?_, ?_⟩
  · apply Polynomial.natDegree_sum_le_of_forall_le
    intro k hk
    exact (natDegree_C_mul_X_pow_le _ _).trans (Nat.lt_succ_iff.1 (Finset.mem_range.1 hk))
  · intro z
    rw [← Complex.taylorSeries_eq_of_entire' 0 z hf, tsum_eq_sum (s := Finset.range (n + 1))]
    · simp [Polynomial.eval_finset_sum]
    · intro k hk
      rw [iteratedDeriv_eq_zero_of_growth f hf n C hb k (by simp at hk; omega)]
      simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.liouville_poly
-- test: ca-contour-liouville-poly-test.sage

theorem exp_unbounded (M : ℝ) : ∃ z : ℂ, M < ‖exp z‖ := by
  refine ⟨((|M| + 1 : ℝ) : ℂ), ?_⟩
  rw [norm_exp_ofReal]
  have := Real.add_one_le_exp (|M| + 1)
  linarith [le_abs_self M]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.exp_unbounded
-- test: ca-contour-exp-unbounded-test.sage

theorem cos_imag_axis (y : ℝ) : cos (y * I) = Real.cosh y := by
  rw [cos_mul_I, ofReal_cosh]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.cos_imag_axis
-- test: ca-contour-cos-imag-axis-test.sage

theorem cos_unbounded (M : ℝ) : ∃ z : ℂ, M < ‖cos z‖ := by
  refine ⟨((2 * |M| + 2 : ℝ) : ℂ) * I, ?_⟩
  rw [cos_imag_axis, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (Real.cosh_pos _), Real.cosh_eq]
  have h1 := Real.add_one_le_exp (2 * |M| + 2)
  have h2 := Real.exp_pos (-(2 * |M| + 2))
  linarith [le_abs_self M]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.cos_unbounded
-- test: ca-contour-cos-unbounded-test.sage

theorem sin_unbounded (M : ℝ) : ∃ z : ℂ, M < ‖sin z‖ := by
  refine ⟨((|M| + 1 : ℝ) : ℂ) * I, ?_⟩
  rw [sin_mul_I, norm_mul, norm_I, mul_one, ← ofReal_sinh, Complex.norm_real, Real.norm_eq_abs]
  have h := Real.self_le_sinh_iff.2 (by positivity : (0 : ℝ) ≤ |M| + 1)
  rw [abs_of_nonneg (by linarith [abs_nonneg M])]
  linarith [le_abs_self M]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.sin_unbounded
-- test: ca-contour-sin-unbounded-test.sage

theorem inv_poly_tendsto_zero (p : ℂ[X]) (hp : 0 < p.natDegree) :
    Tendsto (fun z => (p.eval z)⁻¹) (cocompact ℂ) (𝓝 0) := by
  have hd : 0 < p.degree := natDegree_pos_iff_degree_pos.1 hp
  rw [tendsto_zero_iff_norm_tendsto_zero]
  simpa [norm_inv, Function.comp_def] using tendsto_inv_atTop_zero.comp (p.tendsto_norm_atTop hd tendsto_norm_cocompact_atTop)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.inv_poly_tendsto_zero ((Polynomial.X : Polynomial ℂ)) ((by rw [Polynomial.natDegree_X]; norm_num)) := by
  have hd : 0 < ((Polynomial.X : Polynomial ℂ)).degree := Polynomial.natDegree_pos_iff_degree_pos.1 (by rw [Polynomial.natDegree_X]; norm_num)
  rw [tendsto_zero_iff_norm_tendsto_zero]
  simpa [norm_inv, Function.comp_def] using tendsto_inv_atTop_zero.comp (Polynomial.tendsto_norm_atTop (Polynomial.X : Polynomial ℂ) hd tendsto_norm_cocompact_atTop)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.inv_poly_tendsto_zero ((Polynomial.X ^ 2 : Polynomial ℂ)) ((by rw [Polynomial.natDegree_X_pow]; norm_num)) := by
  have hd : 0 < ((Polynomial.X ^ 2 : Polynomial ℂ)).degree := Polynomial.natDegree_pos_iff_degree_pos.1 (by rw [Polynomial.natDegree_X_pow]; norm_num)
  rw [tendsto_zero_iff_norm_tendsto_zero]
  simpa [norm_inv, Function.comp_def] using tendsto_inv_atTop_zero.comp (Polynomial.tendsto_norm_atTop (Polynomial.X ^ 2 : Polynomial ℂ) hd tendsto_norm_cocompact_atTop)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.inv_poly_tendsto_zero ((Polynomial.X ^ 3 : Polynomial ℂ)) ((by rw [Polynomial.natDegree_X_pow]; norm_num)) := by
  have hd : 0 < ((Polynomial.X ^ 3 : Polynomial ℂ)).degree := Polynomial.natDegree_pos_iff_degree_pos.1 (by rw [Polynomial.natDegree_X_pow]; norm_num)
  rw [tendsto_zero_iff_norm_tendsto_zero]
  simpa [norm_inv, Function.comp_def] using tendsto_inv_atTop_zero.comp (Polynomial.tendsto_norm_atTop (Polynomial.X ^ 3 : Polynomial ℂ) hd tendsto_norm_cocompact_atTop)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.inv_poly_tendsto_zero
-- test: ca-contour-inv-poly-tendsto-zero-test.sage

theorem fta (p : ℂ[X]) (hp : 0 < p.natDegree) : ∃ z, p.eval z = 0 := by
  by_contra h
  push_neg at h
  have hd : Differentiable ℂ (fun z => (p.eval z)⁻¹) := p.differentiable.inv h
  have hc := hd.eq_const_of_tendsto_cocompact (inv_poly_tendsto_zero p hp)
  have := congrFun hc 0
  simp only [Function.const_apply, inv_eq_zero] at this
  exact h 0 this

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.fta ((Polynomial.X : Polynomial ℂ)) ((by rw [Polynomial.natDegree_X]; norm_num)) := by
  exact ⟨0, by simp⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.fta ((Polynomial.X ^ 2 + 1 : Polynomial ℂ)) ((by rw [show (Polynomial.X ^ 2 + 1 : Polynomial ℂ).natDegree = 2 from by compute_degree!]; norm_num)) := by
  exact ⟨Complex.I, by simp⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.fta ((Polynomial.X - 1 : Polynomial ℂ)) ((by rw [show (Polynomial.X - 1 : Polynomial ℂ).natDegree = 1 from by compute_degree!]; norm_num)) := by
  exact ⟨1, by simp⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.fta
-- test: ca-contour-fta-test.sage

theorem poly_factors (p : ℂ[X]) : C p.leadingCoeff * (p.roots.map (fun a => X - C a)).prod = p :=
  C_leadingCoeff_mul_prod_multiset_X_sub_C IsAlgClosed.card_roots_eq_natDegree

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.poly_factors
-- test: ca-contour-poly-factors-test.sage

theorem card_roots (p : ℂ[X]) : p.roots.card = p.natDegree := IsAlgClosed.card_roots_eq_natDegree

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.card_roots
-- test: ca-contour-card-roots-test.sage

theorem factor_root (p : ℂ[X]) (z : ℂ) (hz : p.IsRoot z) :
    (X - C z) * (p /ₘ (X - C z)) = p ∧ (p /ₘ (X - C z)).natDegree = p.natDegree - 1 := by
  refine ⟨mul_divByMonic_eq_iff_isRoot.2 hz, ?_⟩
  rw [natDegree_divByMonic p (monic_X_sub_C z), natDegree_X_sub_C]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.factor_root
-- test: ca-contour-factor-root-test.sage

theorem growth_eventually (f : ℂ → ℂ) (hf : Differentiable ℂ f) (n : ℕ) (C R : ℝ)
    (hb : ∀ z, R ≤ ‖z‖ → ‖f z‖ ≤ C * ‖z‖ ^ n) :
    ∃ p : ℂ[X], p.natDegree ≤ n ∧ ∀ z, f z = p.eval z := by
  obtain ⟨B, hB⟩ := (isCompact_closedBall (0 : ℂ) R).exists_bound_of_continuousOn hf.continuous.continuousOn
  apply liouville_poly f hf n (max B 0 + |C|)
  intro z
  have h1 : 1 ≤ (1 + ‖z‖) ^ n := one_le_pow₀ (by linarith [norm_nonneg z])
  by_cases hz : R ≤ ‖z‖
  · calc ‖f z‖ ≤ C * ‖z‖ ^ n := hb z hz
      _ ≤ |C| * (1 + ‖z‖) ^ n :=
          mul_le_mul (le_abs_self C) (pow_le_pow_left₀ (norm_nonneg z) (by linarith) n) (by positivity)
            (abs_nonneg C)
      _ ≤ (max B 0 + |C|) * (1 + ‖z‖) ^ n := by
          have := le_max_right B 0
          nlinarith
  · have hzB : ‖f z‖ ≤ B := hB z (by rw [mem_closedBall, dist_zero_right]; linarith)
    have := le_max_left B 0
    have := le_max_right B 0
    nlinarith [abs_nonneg C]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.growth_eventually
-- test: ca-contour-growth-eventually-test.sage

theorem const_of_sqrt_growth (f : ℂ → ℂ) (hf : Differentiable ℂ f) (C : ℝ)
    (hb : ∀ z, ‖f z‖ ≤ C * Real.sqrt ‖z‖) (z w : ℂ) : f z = f w := by
  have hC : 0 ≤ C := by
    have h := hb 1
    simp at h
    linarith [norm_nonneg (f 1)]
  have hgrowth : ∀ z, ‖f z‖ ≤ C * (1 + ‖z‖) ^ 1 := by
    intro z
    have hs : Real.sqrt ‖z‖ ≤ 1 + ‖z‖ := by
      nlinarith [Real.sq_sqrt (norm_nonneg z), Real.sqrt_nonneg ‖z‖]
    calc ‖f z‖ ≤ C * Real.sqrt ‖z‖ := hb z
      _ ≤ C * (1 + ‖z‖) ^ 1 := by rw [pow_one]; exact mul_le_mul_of_nonneg_left hs hC
  have hd1 : iteratedDeriv 1 f 0 = 0 := by
    have hlim : Tendsto (fun r : ℝ => C / Real.sqrt r) atTop (𝓝 0) :=
      tendsto_const_nhds.div_atTop (Real.tendsto_sqrt_atTop)
    have hle : ∀ᶠ r in atTop, ‖iteratedDeriv 1 f 0‖ ≤ C / Real.sqrt r := by
      filter_upwards [eventually_gt_atTop 0] with r hr
      have := Complex.norm_iteratedDeriv_le_of_forall_mem_sphere_norm_le (c := 0) 1 hr hf.diffContOnCl
        (C := C * Real.sqrt r) (fun z hz => by
          have : ‖z‖ = r := by simpa using hz
          simpa [this] using hb z)
      calc _ ≤ _ := this
        _ = C / Real.sqrt r := by
          have hs : Real.sqrt r ≠ 0 := (Real.sqrt_pos.2 hr).ne'
          field_simp
          rw [Real.sq_sqrt hr.le]
          simp
    exact norm_le_zero_iff.1 (ge_of_tendsto hlim hle)
  have hall : ∀ k, 1 ≤ k → iteratedDeriv k f 0 = 0 := by
    intro k hk
    rcases Nat.lt_or_ge 1 k with h | h
    · exact iteratedDeriv_eq_zero_of_growth f hf 1 C hgrowth k h
    · have : k = 1 := by omega
      rw [this, hd1]
  have hval : ∀ u, f u = f 0 := by
    intro u
    rw [← Complex.taylorSeries_eq_of_entire' 0 u hf, tsum_eq_single 0]
    · simp
    · intro k hk
      rw [hall k (Nat.one_le_iff_ne_zero.2 hk)]
      simp
  rw [hval z, hval w]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.const_of_sqrt_growth ((fun _ : ℂ => (0 : ℂ))) ((differentiable_const _)) (0) ((fun _ => by simp only [norm_zero]; positivity)) (0) (1) := by
  rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.const_of_sqrt_growth ((fun _ : ℂ => (0 : ℂ))) ((differentiable_const _)) (1) ((fun _ => by simp only [norm_zero]; positivity)) (Complex.I) ((2 + Complex.I)) := by
  rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.const_of_sqrt_growth ((fun _ : ℂ => (0 : ℂ))) ((differentiable_const _)) (5) ((fun _ => by simp only [norm_zero]; positivity)) ((-3)) ((7 * Complex.I)) := by
  rfl

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.const_of_sqrt_growth
-- test: ca-contour-const-of-sqrt-growth-test.sage

theorem sq_identity (s c S C : ℝ) (h1 : c ^ 2 + s ^ 2 = 1) (h2 : C ^ 2 - S ^ 2 = 1) :
    (s * C) ^ 2 + (c * S) ^ 2 = s ^ 2 + S ^ 2 := by
  veriphy_certified (linear_combination (S ^ 2) * h1 + (s ^ 2) * h2)
  -- certificate: ca-contour-ideal.sage

theorem norm_sin_sq (x y : ℝ) : ‖sin (x + y * I)‖ ^ 2 = Real.sin x ^ 2 + Real.sinh y ^ 2 := by
  have e : sin (x + y * I) = ((Real.sin x * Real.cosh y : ℝ) : ℂ) + ((Real.cos x * Real.sinh y : ℝ) : ℂ) * I := by
    rw [sin_add_mul_I]; push_cast; ring
  rw [e, Complex.sq_norm, normSq_add_mul_I]
  exact sq_identity _ _ _ _ (Real.cos_sq_add_sin_sq x) (Real.cosh_sq_sub_sinh_sq y)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.norm_sin_sq
-- test: ca-contour-norm-sin-sq-test.sage

theorem sin_lines (x y : ℝ) :
    Real.sinh y ^ 2 ≤ ‖sin (x + y * I)‖ ^ 2 ∧ ‖sin (x + y * I)‖ ^ 2 ≤ 1 + Real.sinh y ^ 2 := by
  rw [norm_sin_sq]
  constructor
  · nlinarith [sq_nonneg (Real.sin x)]
  · nlinarith [Real.sin_sq_le_one x]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.sin_lines
-- test: ca-contour-sin-lines-test.sage

theorem sin_vertical_unbounded (x M : ℝ) : ∃ y : ℝ, M < ‖sin (x + y * I)‖ := by
  refine ⟨|M| + 1, ?_⟩
  have h1 := (sin_lines x (|M| + 1)).1
  have h2 := Real.self_le_sinh_iff.2 (by positivity : (0 : ℝ) ≤ |M| + 1)
  have h3 : 0 ≤ ‖sin (x + ((|M| + 1 : ℝ) : ℂ) * I)‖ := norm_nonneg _
  by_contra h
  push_neg at h
  have hM := le_abs_self M
  nlinarith [abs_nonneg M]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.sin_vertical_unbounded
-- test: ca-contour-sin-vertical-unbounded-test.sage

theorem poly_lower_bound (p : ℂ[X]) (hp : 0 < p.natDegree) (z : ℂ)
    (hz : max 1 (2 * (∑ k ∈ Finset.range p.natDegree, ‖p.coeff k‖) / ‖p.leadingCoeff‖) ≤ ‖z‖) :
    ‖p.leadingCoeff‖ * ‖z‖ ^ p.natDegree / 2 ≤ ‖p.eval z‖ := by
  have ha : 0 < ‖p.leadingCoeff‖ := norm_pos_iff.2 (leadingCoeff_ne_zero.2 (ne_zero_of_natDegree_gt hp))
  have hz1 : 1 ≤ ‖z‖ := le_trans (le_max_left _ _) hz
  have hz2 := le_trans (le_max_right _ _) hz
  have hS : 2 * (∑ k ∈ Finset.range p.natDegree, ‖p.coeff k‖) ≤ ‖p.leadingCoeff‖ * ‖z‖ := by
    rw [div_le_iff₀ ha] at hz2; linarith
  have he : p.eval z = p.leadingCoeff * z ^ p.natDegree + ∑ k ∈ Finset.range p.natDegree, p.coeff k * z ^ k := by
    rw [eval_eq_sum_range, Finset.sum_range_succ, add_comm]; rfl
  have hq : ‖∑ k ∈ Finset.range p.natDegree, p.coeff k * z ^ k‖ ≤
      (∑ k ∈ Finset.range p.natDegree, ‖p.coeff k‖) * ‖z‖ ^ (p.natDegree - 1) := by
    calc _ ≤ ∑ k ∈ Finset.range p.natDegree, ‖p.coeff k * z ^ k‖ := norm_sum_le _ _
      _ ≤ ∑ k ∈ Finset.range p.natDegree, ‖p.coeff k‖ * ‖z‖ ^ (p.natDegree - 1) := by
          apply Finset.sum_le_sum
          intro k hk
          rw [norm_mul, norm_pow]
          have := Finset.mem_range.1 hk
          gcongr
          all_goals first | exact hz1 | omega
      _ = _ := by rw [Finset.sum_mul]
  have hzn : ‖z‖ ^ p.natDegree = ‖z‖ * ‖z‖ ^ (p.natDegree - 1) := by
    rw [← pow_succ']; congr 1; omega
  have hpos : 0 ≤ ‖z‖ ^ (p.natDegree - 1) := by positivity
  have htri := norm_sub_norm_le (p.leadingCoeff * z ^ p.natDegree) (-(∑ k ∈ Finset.range p.natDegree, p.coeff k * z ^ k))
  rw [sub_neg_eq_add, norm_neg] at htri
  rw [he]
  have hm := mul_le_mul_of_nonneg_right hS hpos
  rw [norm_mul, norm_pow] at htri
  rw [hzn] at htri ⊢
  nlinarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.poly_lower_bound
-- test: ca-contour-poly-lower-bound-test.sage

end KaluzaKlein.ComplexAnalysis.LiouvilleTheorem
