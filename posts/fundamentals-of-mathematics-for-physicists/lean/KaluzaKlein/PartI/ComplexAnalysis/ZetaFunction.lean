import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "The zeta function"

The claims of the section "The zeta function" of `tutorial/part-i-04-complex-analysis.qmd`: the Dirichlet series of
`ζ` (absolutely convergent for `Re s > 1`), holomorphy off `s = 1`, the simple pole of residue `1`, the Euler
product (finite products over the primes below `N` as sums over `N`-smooth numbers), the functional equation, the
values `ζ(2n)` (Bernoulli formula), `ζ(-k)` and the trivial zeros, the alternating series `(1 - 2^{1-s}) ζ(s)`, and
the regularised sums `∑ n^k e^{-εn}` and their finite parts through the Bernoulli generating function
(`1/(e^ε - 1) = B(ε)/ε`, mathlib's `bernoulliPowerSeries_mul_exp_sub_one`): `ζ(-k) = (-1)^k k! [ε^{k+1}] B(ε)`. The
analytic asymptotic expansion as `ε → 0⁺` is not formalised.
-/

namespace KaluzaKlein.ComplexAnalysis.ZetaFunction

open Complex Filter Topology Set
open scoped Nat Real

/-! ## def-zeta-function -/

/-- For `Re s > 1` the series `∑ 1/(n + 1)^s` converges absolutely, with sum `ζ(s)`. -/
theorem zeta_series (s : ℂ) (hs : 1 < s.re) :
    Summable (fun n : ℕ => ‖1 / ((n : ℂ) + 1) ^ s‖) ∧ HasSum (fun n : ℕ => 1 / ((n : ℂ) + 1) ^ s) (riemannZeta s) := by
  have h1 : Summable (fun n : ℕ => 1 / (n : ℂ) ^ s) := Complex.summable_one_div_nat_cpow.mpr hs
  have h2 : Summable (fun n : ℕ => 1 / ((n : ℂ) + 1) ^ s) := by
    have := (summable_nat_add_iff 1).mpr h1
    simpa using this
  refine ⟨summable_norm_iff.mpr h2, ?_⟩
  rw [zeta_eq_tsum_one_div_nat_add_one_cpow hs]
  exact h2.hasSum

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_series
-- test: ca-spec-zeta-series-test.sage

/-- `ζ` is holomorphic on `ℂ ∖ {1}`. -/
theorem zeta_differentiableAt (s : ℂ) (hs : s ≠ 1) : DifferentiableAt ℂ riemannZeta s :=
  differentiableAt_riemannZeta hs

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_differentiableAt (0) ((by norm_num)) := by
  exact differentiableAt_riemannZeta (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_differentiableAt (2) ((by norm_num)) := by
  exact differentiableAt_riemannZeta (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_differentiableAt (1 / 2 + Complex.I) ((by simp [Complex.ext_iff])) := by
  exact differentiableAt_riemannZeta (by simp [Complex.ext_iff])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_differentiableAt
-- test: ca-spec-zeta-hol-test.sage

/-- The simple pole at `s = 1` with residue `1`, approached along any direction `c ≠ 0`:
`(ct) ζ(1 + ct) → 1` as `t → 0`. -/
theorem zeta_residue_direction (c : ℂ) (hc : c ≠ 0) :
    Tendsto (fun t : ℝ => (c * t) * riemannZeta (1 + c * t)) (𝓝[≠] 0) (𝓝 1) := by
  have hφ : Tendsto (fun t : ℝ => 1 + c * t) (𝓝[≠] 0) (𝓝[≠] 1) := by
    refine tendsto_nhdsWithin_iff.mpr ⟨?_, ?_⟩
    · have : Tendsto (fun t : ℝ => 1 + c * (t : ℂ)) (𝓝 0) (𝓝 (1 + c * ((0 : ℝ) : ℂ))) :=
        (continuous_const.add (continuous_const.mul Complex.continuous_ofReal)).tendsto 0
      simp only [Complex.ofReal_zero, mul_zero, add_zero] at this
      exact this.mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin] with t ht
      simp only [mem_compl_iff, mem_singleton_iff] at ht ⊢
      intro h
      have h' : c * t = 0 := by linear_combination h
      rcases mul_eq_zero.mp h' with h'' | h''
      · exact hc h''
      · exact ht (Complex.ofReal_eq_zero.mp h'')
  refine (riemannZeta_residue_one.comp hφ).congr (fun t => ?_)
  simp only [Function.comp, add_sub_cancel_left]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_residue_direction
-- test: ca-spec-zeta-res-test.sage

/-- The Euler product `ζ(s) = ∏_p (1 - p^{-s})⁻¹` for `Re s > 1`. -/
theorem zeta_euler_product (s : ℂ) (hs : 1 < s.re) :
    ∏' p : Nat.Primes, (1 - (p : ℂ) ^ (-s))⁻¹ = riemannZeta s :=
  riemannZeta_eulerProduct_tprod hs

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_euler_product (2) ((by norm_num)) := by
  exact riemannZeta_eulerProduct_tprod (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_euler_product (3) ((by norm_num)) := by
  exact riemannZeta_eulerProduct_tprod (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_euler_product (2 + Complex.I) ((by simp)) := by
  exact riemannZeta_eulerProduct_tprod (by simp)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_euler_product
-- test: ca-spec-euler-test.sage

/-- The functional equation `ζ(s) = 2^s π^{s-1} sin(πs/2) Γ(1 - s) ζ(1 - s)` (`s ≠ 0, 1, 2, …`). -/
theorem zeta_functional_equation (s : ℂ) (hs0 : s ≠ 0) (hs : ∀ n : ℕ, s ≠ n + 1) :
    riemannZeta s =
      2 ^ s * (π : ℂ) ^ (s - 1) * Complex.sin (π * s / 2) * Complex.Gamma (1 - s) * riemannZeta (1 - s) := by
  have h := riemannZeta_one_sub (s := 1 - s) (fun n h => hs n (by linear_combination -h))
    (fun h => hs0 (by linear_combination -h))
  rw [sub_sub_cancel] at h
  rw [h]
  have hcos : Complex.cos (π * (1 - s) / 2) = Complex.sin (π * s / 2) := by
    rw [show (π : ℂ) * (1 - s) / 2 = π / 2 - π * s / 2 by ring, Complex.cos_pi_div_two_sub]
  have hpow : (2 * (π : ℂ)) ^ (-(1 - s)) = 2 ^ (s - 1) * (π : ℂ) ^ (s - 1) := by
    rw [neg_sub, show (2 : ℂ) * π = ((2 : ℝ) : ℂ) * ((π : ℝ) : ℂ) by push_cast <;> ring,
      Complex.mul_cpow_ofReal_nonneg (by norm_num) Real.pi_pos.le]
    push_cast
    try ring
  rw [hcos, hpow, Complex.cpow_sub (x := 2) s 1 two_ne_zero, Complex.cpow_one]
  field_simp
  try ring

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_functional_equation
-- test: ca-spec-zeta-fe-test.sage

/-- `ζ(2n) = (-1)^{n+1} B_{2n} (2π)^{2n}/(2 (2n)!)` for `n ≥ 1` (`ζ(2) = π²/6`, `ζ(4) = π⁴/90`, `ζ(6) = π⁶/945`;
`B₂ = 1/6`, `B₄ = -1/30`), stated as `2 (2n)! ζ(2n) = (-1)^{n+1} B_{2n} (2π)^{2n}`. -/
theorem zeta_two_mul_nat (n : ℕ) (hn : n ≠ 0) :
    2 * (2 * n)! * riemannZeta (2 * n) = (-1) ^ (n + 1) * bernoulli (2 * n) * (2 * π) ^ (2 * n) := by
  rw [riemannZeta_two_mul_nat hn]
  have h2 : (2 : ℂ) ^ (2 * n) = 2 * 2 ^ (2 * n - 1) := by
    rw [← pow_succ']
    congr 1
    omega
  rw [mul_pow, h2]
  have hf : ((2 * n)! : ℂ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero _
  field_simp <;> ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_two_mul_nat
-- test: ca-spec-zeta-even-test.sage

/-- `ζ(-k) = (-1)^k B_{k+1}/(k + 1)`, i.e. `(k + 1) ζ(-k) = (-1)^k B_{k+1}` (`ζ(0) = -1/2`, `ζ(-1) = -1/12`,
`ζ(-3) = 1/120`). -/
theorem zeta_neg_nat (k : ℕ) : (k + 1) * riemannZeta (-k) = (-1) ^ k * bernoulli (k + 1) := by
  rw [riemannZeta_neg_nat_eq_bernoulli k]
  have hk : (k : ℂ) + 1 ≠ 0 := by exact_mod_cast Nat.succ_ne_zero k
  field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_neg_nat
-- test: ca-spec-zeta-neg-test.sage

/-- The trivial zeros `ζ(-2n) = 0`, `n ≥ 1`. -/
theorem zeta_neg_even (n : ℕ) : riemannZeta (-2 * ((n : ℂ) + 1)) = 0 :=
  riemannZeta_neg_two_mul_nat_add_one n

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ZetaFunction.zeta_neg_even
-- test: ca-spec-zeta-zeros-test.sage

/-! ## exm-zeta-continuation -/

/-- For `Re s > 1`: `(1 - 2^{1-s}) ζ(s) = ∑_{n ≥ 1} (-1)^{n+1} n^{-s}` (subtracting twice the even terms). -/
theorem eta_relation (s : ℂ) (hs : 1 < s.re) :
    HasSum (fun n : ℕ => (-1) ^ n / ((n : ℂ) + 1) ^ s) ((1 - 2 ^ (1 - s)) * riemannZeta s) := by
  set a : ℕ → ℂ := fun n => 1 / ((n : ℂ) + 1) ^ s with ha
  have hA : HasSum a (riemannZeta s) := (zeta_series s hs).2
  have hnorm : Summable (fun n => ‖a n‖) := (zeta_series s hs).1
  have h2 : (2 : ℂ) ^ s ≠ 0 := by simp
  have hodd_eq : ∀ k : ℕ, a (2 * k + 1) = (2 : ℂ) ^ (-s) * a k := by
    intro k
    simp only [ha]
    rw [show ((2 * k + 1 : ℕ) : ℂ) + 1 = ((2 : ℝ) : ℂ) * (((k : ℝ) + 1 : ℝ) : ℂ) by push_cast <;> ring,
      Complex.mul_cpow_ofReal_nonneg (by norm_num) (by positivity), Complex.cpow_neg]
    push_cast <;> ring
  have hO : HasSum (fun k => a (2 * k + 1)) ((2 : ℂ) ^ (-s) * riemannZeta s) := by
    simp_rw [hodd_eq]
    exact hA.mul_left _
  have hEs : Summable (fun k => a (2 * k)) :=
    (hnorm.comp_injective (mul_right_injective₀ two_ne_zero)).of_norm
  have hE := hEs.hasSum
  have hsplit := hE.even_add_odd hO
  have he : ∑' k, a (2 * k) = riemannZeta s - (2 : ℂ) ^ (-s) * riemannZeta s := by
    have := hsplit.unique hA
    linear_combination this
  have hf : HasSum (fun n : ℕ => (-1) ^ n / ((n : ℂ) + 1) ^ s)
      (∑' k, a (2 * k) + -((2 : ℂ) ^ (-s) * riemannZeta s)) := by
    refine HasSum.even_add_odd ?_ ?_
    · convert hE using 1
      funext k
      simp [ha, pow_mul, div_eq_mul_inv]
    · convert hO.neg using 1
      funext k
      simp [ha, pow_succ, pow_mul, div_eq_mul_inv]
  convert hf using 1
  rw [he, Complex.cpow_sub (x := 2) 1 s two_ne_zero, Complex.cpow_one, Complex.cpow_neg]
  field_simp <;> ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ZetaFunction.eta_relation
-- test: ca-spec-eta-test.sage

/-- For real `s > 0` the alternating series `∑ (-1)ⁿ/(n + 1)^s` converges (its partial sums have a limit). -/
theorem alternating_converges (s : ℝ) (hs : 0 < s) :
    ∃ l : ℝ, Tendsto (fun N => ∑ n ∈ Finset.range N, (-1) ^ n * (1 / ((n : ℝ) + 1) ^ s)) atTop (𝓝 l) := by
  refine Antitone.tendsto_alternating_series_of_tendsto_zero ?_ ?_
  · intro m n hmn
    have hm : (0 : ℝ) < (m : ℝ) + 1 := by positivity
    exact one_div_le_one_div_of_le (Real.rpow_pos_of_pos hm s)
      (Real.rpow_le_rpow hm.le (by exact_mod_cast Nat.add_le_add_right hmn 1) hs.le)
  · have h : Tendsto (fun n : ℕ => ((n : ℝ) + 1) ^ s) atTop atTop :=
      (tendsto_rpow_atTop hs).comp (tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop)
    refine h.inv_tendsto_atTop.congr fun n => ?_
    simp [one_div]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ZetaFunction.alternating_converges (1) ((by norm_num)) := by
  refine Antitone.tendsto_alternating_series_of_tendsto_zero ?_ ?_
  · intro m n hmn
    have hm : (0 : ℝ) < (m : ℝ) + 1 := by positivity
    exact one_div_le_one_div_of_le (Real.rpow_pos_of_pos hm _) (Real.rpow_le_rpow hm.le (by exact_mod_cast Nat.add_le_add_right hmn 1) (by norm_num))
  · have h : Filter.Tendsto (fun n : ℕ => ((n : ℝ) + 1) ^ (1 : ℝ)) Filter.atTop Filter.atTop :=
      (tendsto_rpow_atTop (by norm_num)).comp (Filter.tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop)
    refine h.inv_tendsto_atTop.congr fun n => ?_
    simp [one_div]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ZetaFunction.alternating_converges (1 / 2) ((by norm_num)) := by
  refine Antitone.tendsto_alternating_series_of_tendsto_zero ?_ ?_
  · intro m n hmn
    have hm : (0 : ℝ) < (m : ℝ) + 1 := by positivity
    exact one_div_le_one_div_of_le (Real.rpow_pos_of_pos hm _) (Real.rpow_le_rpow hm.le (by exact_mod_cast Nat.add_le_add_right hmn 1) (by norm_num))
  · have h : Filter.Tendsto (fun n : ℕ => ((n : ℝ) + 1) ^ (1 / 2 : ℝ)) Filter.atTop Filter.atTop :=
      (tendsto_rpow_atTop (by norm_num)).comp (Filter.tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop)
    refine h.inv_tendsto_atTop.congr fun n => ?_
    simp [one_div]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ZetaFunction.alternating_converges (2) ((by norm_num)) := by
  refine Antitone.tendsto_alternating_series_of_tendsto_zero ?_ ?_
  · intro m n hmn
    have hm : (0 : ℝ) < (m : ℝ) + 1 := by positivity
    exact one_div_le_one_div_of_le (Real.rpow_pos_of_pos hm _) (Real.rpow_le_rpow hm.le (by exact_mod_cast Nat.add_le_add_right hmn 1) (by norm_num))
  · have h : Filter.Tendsto (fun n : ℕ => ((n : ℝ) + 1) ^ (2 : ℝ)) Filter.atTop Filter.atTop :=
      (tendsto_rpow_atTop (by norm_num)).comp (Filter.tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop)
    refine h.inv_tendsto_atTop.congr fun n => ?_
    simp [one_div]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ZetaFunction.alternating_converges
-- test: ca-spec-alternating-test.sage

/-- The prefactor `1 - 2^{1-s}` vanishes at `s = 1 + 2πik/log 2` (`k = 0`: the pole of `ζ` at `s = 1`). -/
theorem prefactor_zero (k : ℤ) : (1 : ℂ) - 2 ^ (1 - (1 + 2 * π * I * k * ((Real.log 2)⁻¹ : ℝ))) = 0 := by
  have hl : (Real.log 2 : ℂ) ≠ 0 := by
    exact_mod_cast (Real.log_pos (by norm_num : (1 : ℝ) < 2)).ne'
  rw [Complex.cpow_def_of_ne_zero two_ne_zero,
    show Complex.log 2 = (Real.log 2 : ℂ) by rw [← Complex.ofReal_ofNat, Complex.ofReal_log (by norm_num)]]
  have : (Real.log 2 : ℂ) * (1 - (1 + 2 * π * I * k * ((Real.log 2)⁻¹ : ℝ))) = -(k * (2 * π * I)) := by
    push_cast
    field_simp <;> ring
  rw [this, Complex.exp_neg, Complex.exp_int_mul_two_pi_mul_I]
  simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ZetaFunction.prefactor_zero
-- test: ca-spec-prefactor-test.sage

/-- At `s = 0`: `(1 - 2^{1-0}) ζ(0) = 1/2`, the value assigned to `1 - 1 + 1 - …`, so `ζ(0) = -1/2`. -/
theorem eta_zero : (1 - (2 : ℂ) ^ (1 - 0 : ℂ)) * riemannZeta 0 = 1 / 2 := by
  have h0 : riemannZeta 0 = -1 / 2 := riemannZeta_zero
  have h2 : (2 : ℂ) ^ (1 - 0 : ℂ) = 2 := by simp
  generalize riemannZeta 0 = Z at h0 ⊢
  generalize (2 : ℂ) ^ (1 - 0 : ℂ) = T at h2 ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-T + 1) * h0 + (1 / 2) * h2)
  -- certificate: ideal_membership.sage

/-! ## exm-zeta-minus-one, exr-zeta-regularised-sum -/

/-- `∑_{n ≥ 1} n^k` diverges for every `k` (`k = 1`: `1 + 2 + 3 + …`). -/
theorem not_summable_pow (k : ℕ) : ¬ Summable (fun n : ℕ => ((n : ℝ) + 1) ^ k) := by
  intro h
  have ht := h.tendsto_atTop_zero
  have : (1 : ℝ) ≤ 0 :=
    ge_of_tendsto ht (Eventually.of_forall fun n => one_le_pow₀ (by linarith [(n.cast_nonneg : (0 : ℝ) ≤ n)]))
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ZetaFunction.not_summable_pow
-- test: ca-spec-divergent-test.sage

/-- `∑_{n ≥ 1} n e^{-εn} = e^{-ε}/(1 - e^{-ε})²` for `ε > 0`. -/
theorem regularised_sum_one (ε : ℝ) (hε : 0 < ε) :
    HasSum (fun n : ℕ => (n : ℝ) * Real.exp (-ε * n)) (Real.exp (-ε) / (1 - Real.exp (-ε)) ^ 2) := by
  have hr : ‖Real.exp (-ε)‖ < 1 := by
    rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    exact Real.exp_lt_one_iff.mpr (by linarith)
  convert hasSum_coe_mul_geometric_of_norm_lt_one hr using 1
  funext n
  rw [← Real.exp_nat_mul]
  congr 2
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ZetaFunction.regularised_sum_one
-- test: ca-spec-reg-one-test.sage

/-- `∑_{n ≥ 1} e^{-εn} = 1/(e^ε - 1)` for `ε > 0`. -/
theorem regularised_sum_zero (ε : ℝ) (hε : 0 < ε) :
    HasSum (fun n : ℕ => Real.exp (-ε * ((n : ℝ) + 1))) (1 / (Real.exp ε - 1)) := by
  have hr : ‖Real.exp (-ε)‖ < 1 := by
    rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    exact Real.exp_lt_one_iff.mpr (by linarith)
  have h := (hasSum_geometric_of_norm_lt_one hr).mul_left (Real.exp (-ε))
  convert h using 1
  · funext n
    rw [← Real.exp_nat_mul, ← Real.exp_add]
    congr 1
    ring
  · have hpos : Real.exp ε ≠ 0 := (Real.exp_pos ε).ne'
    have he : Real.exp ε - 1 ≠ 0 := (sub_pos.mpr (Real.one_lt_exp_iff.mpr hε)).ne'
    rw [Real.exp_neg, show 1 - (Real.exp ε)⁻¹ = (Real.exp ε - 1) / Real.exp ε by field_simp]
    field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ZetaFunction.regularised_sum_zero
-- test: ca-spec-reg-zero-test.sage

/-- `∑_{n ≥ 1} n e^{-εn} = e^ε/(e^ε - 1)²` for `ε > 0`. -/
theorem regularised_sum_one_claim (ε : ℝ) (hε : 0 < ε) :
    HasSum (fun n : ℕ => (n : ℝ) * Real.exp (-ε * n)) (Real.exp ε / (Real.exp ε - 1) ^ 2) := by
  convert regularised_sum_one ε hε using 1
  have hpos : Real.exp ε ≠ 0 := (Real.exp_pos ε).ne'
  have he : Real.exp ε - 1 ≠ 0 := (sub_pos.mpr (Real.one_lt_exp_iff.mpr hε)).ne'
  rw [Real.exp_neg]
  field_simp <;> ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ZetaFunction.regularised_sum_one_claim
-- test: ca-spec-reg-one-b-test.sage

/-- Zeta regularisation as a formal statement. `∑_{n≥1} nᵏ e^{-εn} = (-d/dε)ᵏ (1/(e^ε - 1))` and `1/(e^ε - 1) = B(ε)/ε`
with `B(ε) = ∑ Bₙ εⁿ/n!` the Bernoulli generating function (mathlib: `bernoulliPowerSeries_mul_exp_sub_one`), so the
`ε⁰` coefficient of the `k`-th regularised sum, its finite part, is `(-1)ᵏ k! · B_{k+1}/(k + 1)!` (the `k`-th
derivative at `0` of the regular part `∑_{n≥1} Bₙ εⁿ⁻¹/n!`, with the sign `(-1)ᵏ`). The claim: this finite part is
`ζ(-k)` (`k = 0`: `-1/2`, `k = 1`: `-1/12`). -/
theorem finite_part_zeta (k : ℕ) :
    riemannZeta (-k) = (((-1) ^ k * k ! * PowerSeries.coeff (k + 1) (bernoulliPowerSeries ℚ) : ℚ) : ℂ) := by
  have hc : PowerSeries.coeff (k + 1) (bernoulliPowerSeries ℚ) = bernoulli (k + 1) / ((k + 1) * k !) := by
    simp [bernoulliPowerSeries, Nat.factorial_succ]
  have hk : ((k : ℂ) + 1) ≠ 0 := by exact_mod_cast Nat.succ_ne_zero k
  have hf : ((k ! : ℕ) : ℂ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero k
  rw [riemannZeta_neg_nat_eq_bernoulli, hc]
  push_cast
  first | (field_simp; done) | (field_simp; ring)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ZetaFunction.finite_part_zeta
-- test: ca-spec-finite-part-test.sage

/-! ## exr-zeta-euler-product -/

/-- `(1 - p^{-s})⁻¹ = ∑_k p^{-ks}` for a prime `p` and `Re s > 0`. -/
theorem prime_geometric (p : ℕ) (hp : p.Prime) (s : ℂ) (hs : 0 < s.re) :
    HasSum (fun k : ℕ => ((p : ℂ) ^ (-s)) ^ k) (1 - (p : ℂ) ^ (-s))⁻¹ := by
  apply hasSum_geometric_of_norm_lt_one
  rw [Complex.norm_natCast_cpow_of_pos hp.pos, Complex.neg_re]
  exact Real.rpow_lt_one_of_one_lt_of_neg (by exact_mod_cast hp.one_lt) (by linarith)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ZetaFunction.prime_geometric
-- test: ca-spec-prime-geom-test.sage

/-- `∏_{p < N} (1 - p^{-s})⁻¹ = ∑ n^{-s}` over the `N`-smooth numbers `n`, for `Re s > 1`. -/
theorem euler_product_finite (N : ℕ) (s : ℂ) (hs : 1 < s.re) :
    ∏ p ∈ N.primesBelow, (1 - (p : ℂ) ^ (-s))⁻¹ = ∑' m : N.smoothNumbers, ((m : ℕ) : ℂ) ^ (-s) := by
  have h := EulerProduct.prod_primesBelow_geometric_eq_tsum_smoothNumbers
    (f := (riemannZetaSummandHom (ne_zero_of_one_lt_re hs) : ℕ →* ℂ))
    (summable_riemannZetaSummand hs).of_norm N
  simpa [riemannZetaSummandHom] using h

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ZetaFunction.euler_product_finite
-- test: ca-spec-euler-finite-test.sage

end KaluzaKlein.ComplexAnalysis.ZetaFunction
