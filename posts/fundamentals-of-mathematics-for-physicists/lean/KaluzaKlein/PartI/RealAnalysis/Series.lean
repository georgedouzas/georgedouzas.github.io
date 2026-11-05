import Mathlib
import Veriphy.Kit.Levels

/-!
# Real analysis, part "Series"

The claims of the section "Series" of `tutorial/part-i-03-real-analysis.qmd`: partial sums and the geometric,
harmonic and telescoping series, absolute and conditional convergence with the alternating harmonic series and its
rearrangement, the comparison and ratio tests, power series and their radii, and the sums of the running series.

A closed example is stated through the general claim it instantiates (a family with the example's numbers as
parameters, or the section's theorem), so that it can be tested at the example's values; a partial-sum closed form
is certified by induction (`sum_induction`) inside the proof of the series value it gives.
-/

namespace KaluzaKlein.RealAnalysis.Series

open Filter Topology Real Set Finset
open scoped Nat

/-- The partial sums `s_m = ∑_{k ≤ m} 1/k!` of the series of `e`. -/
noncomputable def expPartial (m : ℕ) : ℝ := ∑ k ∈ Finset.range (m + 1), 1 / (k ! : ℝ)

/-! ## def-series -/

/-- If a series converges, its terms tend to `0`. -/
theorem terms_tendsto_zero (a : ℕ → ℝ) (h : Summable a) : Tendsto a atTop (𝓝 0) :=
  h.tendsto_atTop_zero

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Series.terms_tendsto_zero
-- test: ra-terms-tendsto-zero-test.sage

/-- The `p`-series `∑ 1/n^p` converges exactly when `p > 1`; at `p = 1` (the harmonic series) the terms tend to
`0` but the series diverges. -/
theorem p_series (p : ℝ) : Summable (fun n : ℕ => 1 / (n : ℝ) ^ p) ↔ 1 < p :=
  Real.summable_one_div_nat_rpow

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.p_series
-- test: ra-p-series-test.sage

/-! ## exm-series-geometric -/

/-- The partial sums of the geometric series: `∑_{k ≤ n} r^k = (1 - r^(n+1))/(1 - r)` for `r ≠ 1`. -/
theorem geom_partial (r : ℝ) (hr : r ≠ 1) (n : ℕ) :
    ∑ k ∈ range (n + 1), r ^ k = (1 - r ^ (n + 1)) / (1 - r) := by
  have h : ∀ N : ℕ, ∑ k ∈ range N, r ^ k * (1 - r) = 1 - r ^ N := by
    intro N
    veriphy_certified
      (induction N with
        | zero => rw [Finset.sum_range_zero]; (try push_cast); ring1
        | succ N ih => rw [Finset.sum_range_succ, ih]; (try push_cast); ring1)
    -- certificate: ra-geom-partial.sage
  have h1 : 1 - r ≠ 0 := sub_ne_zero.2 (Ne.symm hr)
  rw [eq_div_iff h1, Finset.sum_mul]
  exact h (n + 1)

/-- The geometric series converges to `1/(1 - r)` for `|r| < 1`, and `∑_{k ≥ 1} r^k = r/(1 - r)`
(`∑ 2^(-k) = 2`, `∑_{k ≥ 1} 2^(-k) = 1`, `∑ (-1/3)^k = 3/4`). -/
theorem geometric_sums (r : ℝ) (hr : |r| < 1) :
    HasSum (fun k : ℕ => r ^ k) (1 / (1 - r)) ∧ HasSum (fun k : ℕ => r ^ (k + 1)) (r / (1 - r)) := by
  have hr1 : r ≠ 1 := by intro h; rw [h] at hr; norm_num at hr
  have h1 : 1 - r ≠ 0 := sub_ne_zero.2 (Ne.symm hr1)
  have hcore : ∀ N : ℕ, ∑ k ∈ range N, r ^ k * (1 - r) = 1 - r ^ N := by
    intro N
    veriphy_certified
      (induction N with
        | zero => rw [Finset.sum_range_zero]; (try push_cast); ring1
        | succ N ih => rw [Finset.sum_range_succ, ih]; (try push_cast); ring1)
    -- certificate: ra-geom-partial.sage
  have hpart : ∀ N : ℕ, ∑ k ∈ range N, r ^ k = (1 - r ^ N) / (1 - r) := fun N => by
    rw [eq_div_iff h1, Finset.sum_mul]; exact hcore N
  have hpow : Tendsto (fun N : ℕ => r ^ N) atTop (𝓝 0) := tendsto_pow_atTop_nhds_zero_of_abs_lt_one hr
  have hsum : HasSum (fun k : ℕ => r ^ k) (1 / (1 - r)) := by
    have hs : Summable (fun k : ℕ => r ^ k) := summable_geometric_of_abs_lt_one hr
    rw [hs.hasSum_iff_tendsto_nat]
    have := ((tendsto_const_nhds (x := (1 : ℝ))).sub hpow).div_const (1 - r)
    rw [sub_zero] at this
    simpa [hpart] using this
  refine ⟨hsum, ?_⟩
  have := hsum.mul_left r
  simp only [← pow_succ'] at this
  convert this using 1
  field_simp

/-- The geometric series diverges for `|r| ≥ 1`. -/
theorem geometric_diverges (r : ℝ) (hr : 1 ≤ |r|) : ¬ Summable (fun k : ℕ => r ^ k) := by
  rw [summable_geometric_iff_norm_lt_one, Real.norm_eq_abs, not_lt]
  exact hr

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.geometric_diverges
-- test: ra-geometric-diverges-test.sage

/-! ## exm-series-harmonic -/

/-- Oresme's grouping: `H_(2^m) ≥ 1 + m/2`. -/
theorem harmonic_two_pow (m : ℕ) : 1 + (m : ℝ) / 2 ≤ (harmonic (2 ^ m) : ℝ) := by
  suffices h : (1 : ℚ) + m / 2 ≤ harmonic (2 ^ m) by
    have := (Rat.cast_le (K := ℝ)).2 h
    push_cast at this
    exact this
  induction m with
  | zero => simp [harmonic]
  | succ k ih =>
    -- H_(2^(k+1)) - H_(2^k) = ∑_{i = 2^k}^{2^(k+1) - 1} 1/(i+1) ≥ 2^k · 1/2^(k+1) = 1/2
    have hle : 2 ^ k ≤ 2 ^ (k + 1) := Nat.pow_le_pow_right (by norm_num) (Nat.le_succ k)
    have hsplit : harmonic (2 ^ (k + 1)) =
        harmonic (2 ^ k) + ∑ i ∈ Finset.Ico (2 ^ k) (2 ^ (k + 1)), (((i + 1 : ℕ) : ℚ))⁻¹ := by
      unfold harmonic
      rw [← Finset.sum_range_add_sum_Ico _ hle]
    have hterm : ∀ i ∈ Finset.Ico (2 ^ k) (2 ^ (k + 1)),
        (((2 ^ (k + 1) : ℕ) : ℚ))⁻¹ ≤ (((i + 1 : ℕ) : ℚ))⁻¹ := by
      intro i hi
      rw [Finset.mem_Ico] at hi
      apply inv_anti₀ (by positivity)
      exact_mod_cast Nat.succ_le_of_lt hi.2
    have hsum := Finset.card_nsmul_le_sum _ _ _ hterm
    rw [Nat.card_Ico, nsmul_eq_mul, Nat.cast_sub hle] at hsum
    have hc : (((2 ^ (k + 1) : ℕ) : ℚ) - ((2 ^ k : ℕ) : ℚ)) * (((2 ^ (k + 1) : ℕ) : ℚ))⁻¹ = 1 / 2 := by
      push_cast
      rw [pow_succ]
      field_simp
      ring
    rw [hsplit, Nat.cast_succ]
    linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.harmonic_two_pow (0) := by
  norm_num [harmonic, Finset.sum_range_succ]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.harmonic_two_pow (1) := by
  norm_num [harmonic, Finset.sum_range_succ]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.harmonic_two_pow (2) := by
  norm_num [harmonic, Finset.sum_range_succ]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.harmonic_two_pow (3) := by
  norm_num [harmonic, Finset.sum_range_succ]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.harmonic_two_pow (5) := by
  norm_num [harmonic, Finset.sum_range_succ]

attribute [veriphy_tested kernel_test 5 0] KaluzaKlein.RealAnalysis.Series.harmonic_two_pow
-- test: ra-harmonic-two-pow-test.sage

/-- The partial sums of the harmonic series grow like `log n`: `log(n + 1) ≤ H_n ≤ 1 + log n`. -/
theorem harmonic_log_bounds (n : ℕ) :
    Real.log ((n : ℝ) + 1) ≤ harmonic n ∧ (harmonic n : ℝ) ≤ 1 + Real.log n := by
  refine ⟨?_, harmonic_le_one_add_log n⟩
  have := log_add_one_le_harmonic n
  simpa using this

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.harmonic_log_bounds
-- test: ra-harmonic-log-bounds-test.sage

/-- The telescoping term `1/((k + a)(k + a + 1))`, named so that its partial sums can be certified (a sum whose
summand divides by the bound variable is not translated). -/
noncomputable def telescopingTerm (a : ℝ) (k : ℕ) : ℝ := 1 / (((k : ℝ) + a) * ((k : ℝ) + a + 1))

/-- `∑_{k ≥ 1} 1/((k + a)(k + a + 1))`-type telescoping: `∑_{k ≥ 0} 1/((k + a)(k + a + 1)) = 1/a` for `a > 0`; at
`a = 1`, `∑_{k ≥ 1} 1/(k(k+1)) = 1`. -/
theorem telescoping_sum (a : ℝ) (ha : 0 < a) :
    HasSum (fun k : ℕ => 1 / (((k : ℝ) + a) * ((k : ℝ) + a + 1))) (1 / a) := by
  have hcore : ∀ N : ℕ, ∑ k ∈ range N, telescopingTerm a k = 1 / a - 1 / ((N : ℝ) + a) := by
    intro N
    veriphy_certified
      (induction N with
        | zero => simp [telescopingTerm]
        | succ N ih =>
          rw [Finset.sum_range_succ, ih]
          simp only [telescopingTerm]
          push_cast
          field_simp
          ring)
    -- certificate: ra-telescoping-a.sage
  have hs : Tendsto (fun N : ℕ => 1 / a - 1 / ((N : ℝ) + a)) atTop (𝓝 (1 / a - 0)) := by
    apply tendsto_const_nhds.sub
    apply Tendsto.div_atTop tendsto_const_nhds
    exact tendsto_atTop_add_const_right _ a tendsto_natCast_atTop_atTop
  rw [sub_zero] at hs
  rw [hasSum_iff_tendsto_nat_of_nonneg (fun k => by have : (0 : ℝ) ≤ k := k.cast_nonneg; positivity)]
  exact hs.congr (fun N => (hcore N).symm)

/-! ## exm-series-basel -/

/-- The term `1/(k + 1)²` of the Basel series. -/
noncomputable def baselTerm (k : ℕ) : ℝ := 1 / ((k : ℝ) + 1) ^ 2

/-- The partial sums of `∑ 1/k²` are bounded by `2`: `∑_{k=1}^N 1/k² ≤ 2 - 1/N` (telescoping comparison). -/
theorem basel_partial_bound (N : ℕ) (hN : 1 ≤ N) :
    ∑ k ∈ Finset.range N, baselTerm k ≤ 2 - 1 / (N : ℝ) := by
  simp only [baselTerm]
  induction N with
  | zero => omega
  | succ n ih =>
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · norm_num
    · rw [Finset.sum_range_succ]
      have ih' := ih hn
      have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
      have key : 1 / ((n : ℝ) + 1) ^ 2 ≤ 1 / (n : ℝ) - 1 / ((n + 1 : ℕ) : ℝ) := by
        push_cast
        rw [div_sub_div _ _ hn0.ne' (by positivity), div_le_div_iff₀ (by positivity) (by positivity)]
        nlinarith
      push_cast at key ⊢
      linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.basel_partial_bound (1) ((by norm_num)) := by
  norm_num [baselTerm, Finset.sum_range_succ]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.basel_partial_bound (2) ((by norm_num)) := by
  norm_num [baselTerm, Finset.sum_range_succ]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.basel_partial_bound (5) ((by norm_num)) := by
  norm_num [baselTerm, Finset.sum_range_succ]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.basel_partial_bound (10) ((by norm_num)) := by
  norm_num [baselTerm, Finset.sum_range_succ]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Series.basel_partial_bound
-- test: ra-basel-partial-bound-test.sage

/-- Euler's formula `ζ(2k) = (-1)^(k+1) 2^(2k-1) π^(2k) B_(2k)/(2k)!`: `∑ 1/n² = π²/6` (`k = 1`),
`∑ 1/n⁴ = π⁴/90`, `∑ 1/n⁶ = π⁶/945`. -/
theorem zeta_even (k : ℕ) (hk : 1 ≤ k) :
    HasSum (fun n : ℕ => 1 / (n : ℝ) ^ (2 * k))
      ((-1 : ℝ) ^ (k + 1) * (2 : ℝ) ^ (2 * k - 1) * π ^ (2 * k) * bernoulli (2 * k) / (2 * k)!) :=
  hasSum_zeta_nat (by omega)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Series.zeta_even
-- test: ra-zeta-even-test.sage

/-- `∑ 1/n² = π²/6`, and the tail after `N` terms lies between `1/(N + 1)` and `1/N`. -/
theorem basel_tail (N : ℕ) (hN : 1 ≤ N) :
    1 / ((N : ℝ) + 1) ≤ π ^ 2 / 6 - ∑ k ∈ Finset.range N, 1 / ((k : ℝ) + 1) ^ 2 ∧
      π ^ 2 / 6 - ∑ k ∈ Finset.range N, 1 / ((k : ℝ) + 1) ^ 2 ≤ 1 / N := by
  have hg : HasSum (fun n : ℕ => 1 / ((n : ℝ) + 1) ^ 2) (π ^ 2 / 6) := by
    have := (hasSum_nat_add_iff' 1).2 hasSum_zeta_two
    simpa using this
  have htail := (hasSum_nat_add_iff' N).2 hg
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hlow := telescoping_sum ((N : ℝ) + 1) (by positivity)
  have hhigh := telescoping_sum (N : ℝ) hN0
  constructor
  · apply hasSum_le (fun j => ?_) hlow htail
    push_cast
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith
  · apply hasSum_le (fun j => ?_) htail hhigh
    push_cast
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.RealAnalysis.Series.basel_tail
-- test: ra-basel-tail-test.sage

/-! ## exr-geometric-partial-sums -/

/-- The partial sums for `r = 2` are `2^(n+1) - 1`, unbounded. -/
theorem geom_two_partial (n : ℕ) : ∑ k ∈ range n, (2 : ℝ) ^ k = 2 ^ n - 1 := by
  veriphy_certified
    (induction n with
      | zero => rw [Finset.sum_range_zero]; (try push_cast); ring1
      | succ n ih => rw [Finset.sum_range_succ, ih]; (try push_cast); ring1)
  -- certificate: ra-geom-two.sage

/-- The partial sums for `r = -1` alternate between `1` and `0`: `∑_{k < n} (-1)^k = (1 - (-1)^n)/2`. -/
theorem geom_neg_one_partial (n : ℕ) : ∑ k ∈ range n, (-1 : ℝ) ^ k = (1 - (-1) ^ n) / 2 := by
  veriphy_certified
    (induction n with
      | zero => rw [Finset.sum_range_zero]; (try push_cast); ring1
      | succ n ih => rw [Finset.sum_range_succ, ih]; (try push_cast); ring1)
  -- certificate: ra-geom-neg-one.sage

/-! ## exr-telescoping -/

/-- `∑_{k=1}^n 1/(k(k+1)) = 1 - 1/(n + 1)`: with `k` from `0`, the terms are `telescopingTerm 1 k = 1/((k+1)(k+2))`. -/
theorem telescoping_partial (n : ℕ) :
    ∑ k ∈ range n, telescopingTerm 1 k = 1 - 1 / ((n : ℝ) + 1) := by
  veriphy_certified
    (induction n with
      | zero => simp [telescopingTerm]
      | succ n ih =>
        rw [Finset.sum_range_succ, ih]
        simp only [telescopingTerm]
        push_cast
        field_simp
        ring)
  -- certificate: ra-telescoping-partial.sage

/-- The term `√(k+2) - √(k+1)` of the telescoping square-root series. -/
noncomputable def sqrtStep (k : ℕ) : ℝ := √((k : ℝ) + 2) - √((k : ℝ) + 1)

/-- `∑ (√(k+1) - √k)` has partial sums `√(n+1) - 1 → ∞` and diverges, although its terms tend to `0`. -/
theorem sqrt_telescoping :
    (∀ n : ℕ, ∑ k ∈ range n, sqrtStep k = √((n : ℝ) + 1) - 1) ∧
      ¬ Summable sqrtStep ∧ Tendsto sqrtStep atTop (𝓝 0) := by
  have hpart : ∀ n : ℕ, ∑ k ∈ range n, sqrtStep k = √((n : ℝ) + 1) - 1 := by
    intro n
    veriphy_certified
      (induction n with
        | zero => simp [sqrtStep]
        | succ n ih =>
          rw [Finset.sum_range_succ, ih]
          simp only [sqrtStep]
          push_cast
          ring_nf)
    -- certificate: ra-sqrt-telescoping.sage
  refine ⟨hpart, ?_, ?_⟩
  · intro hs
    have hb := hs.hasSum.tendsto_sum_nat
    simp_rw [hpart] at hb
    have hinf : Tendsto (fun n : ℕ => √((n : ℝ) + 1) - 1) atTop atTop := by
      apply tendsto_atTop_add_const_right
      exact Real.tendsto_sqrt_atTop.comp (tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop)
    exact not_tendsto_nhds_of_tendsto_atTop hinf _ hb
  · show Tendsto (fun k : ℕ => √((k : ℝ) + 2) - √((k : ℝ) + 1)) atTop (𝓝 0)
    have e : ∀ k : ℕ, √((k : ℝ) + 2) - √((k : ℝ) + 1) = 1 / (√((k : ℝ) + 2) + √((k : ℝ) + 1)) := by
      intro k
      have h1 : 0 < √((k : ℝ) + 2) + √((k : ℝ) + 1) := by positivity
      rw [eq_div_iff h1.ne']
      have := Real.sq_sqrt (show (0 : ℝ) ≤ (k : ℝ) + 2 by positivity)
      have := Real.sq_sqrt (show (0 : ℝ) ≤ (k : ℝ) + 1 by positivity)
      nlinarith
    simp_rw [e]
    apply Tendsto.div_atTop tendsto_const_nhds
    exact tendsto_atTop_mono (fun k => le_add_of_nonneg_right (Real.sqrt_nonneg _))
      (Real.tendsto_sqrt_atTop.comp (tendsto_atTop_add_const_right _ 2 tendsto_natCast_atTop_atTop))

/-! ## exr-series-necessary-condition -/

/-- `H_(2n) - H_n ≥ 1/2`: the partial sums of the harmonic series are not Cauchy. -/
theorem harmonic_double (n : ℕ) (hn : 1 ≤ n) : (1 : ℝ) / 2 ≤ harmonic (2 * n) - harmonic n := by
  have hsplit : (harmonic (2 * n) : ℝ) = harmonic n + ∑ i ∈ Finset.Ico n (2 * n), (1 : ℝ) / ((i : ℝ) + 1) := by
    simp only [harmonic, Rat.cast_sum, Rat.cast_inv, Rat.cast_natCast]
    rw [← Finset.sum_range_add_sum_Ico _ (by omega : n ≤ 2 * n)]
    congr 1
    refine Finset.sum_congr rfl (fun i _ => ?_)
    push_cast; ring
  have hcard : ((Finset.Ico n (2 * n)).card : ℝ) = n := by
    rw [Nat.card_Ico]; push_cast [Nat.cast_sub (by omega : n ≤ 2 * n)]; ring
  have hterm : ∀ i ∈ Finset.Ico n (2 * n), (1 : ℝ) / (2 * n) ≤ 1 / ((i : ℝ) + 1) := by
    intro i hi
    rw [Finset.mem_Ico] at hi
    have : (i : ℝ) + 1 ≤ 2 * n := by exact_mod_cast hi.2
    exact one_div_le_one_div_of_le (by positivity) this
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  have hb : (1 : ℝ) / 2 ≤ ∑ i ∈ Finset.Ico n (2 * n), (1 : ℝ) / ((i : ℝ) + 1) := by
    calc (1 : ℝ) / 2 = ((Finset.Ico n (2 * n)).card : ℝ) * (1 / (2 * n)) := by rw [hcard]; field_simp
      _ = ∑ i ∈ Finset.Ico n (2 * n), (1 : ℝ) / (2 * n) := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ _ := Finset.sum_le_sum hterm
  rw [hsplit]
  push_cast
  linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.harmonic_double (1) ((by norm_num)) := by
  norm_num [harmonic, Finset.sum_range_succ]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.harmonic_double (2) ((by norm_num)) := by
  norm_num [harmonic, Finset.sum_range_succ]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.harmonic_double (3) ((by norm_num)) := by
  norm_num [harmonic, Finset.sum_range_succ]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.harmonic_double (10) ((by norm_num)) := by
  norm_num [harmonic, Finset.sum_range_succ]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Series.harmonic_double
-- test: ra-harmonic-double-test.sage

/-- `1/√k ≥ 1/k` for `k ≥ 1`, so `∑ 1/√k` diverges by comparison with the harmonic series. -/
theorem inv_sqrt_ge_inv (k : ℕ) (hk : 1 ≤ k) : 1 / (k : ℝ) ≤ 1 / √(k : ℝ) := by
  have hk1 : (1 : ℝ) ≤ k := by exact_mod_cast hk
  apply one_div_le_one_div_of_le (Real.sqrt_pos.2 (by linarith))
  calc √(k : ℝ) ≤ √(k : ℝ) * √(k : ℝ) := by
        have : 1 ≤ √(k : ℝ) := by rw [show (1 : ℝ) = √1 by simp]; exact Real.sqrt_le_sqrt hk1
        nlinarith [Real.sqrt_nonneg (k : ℝ)]
    _ = k := Real.mul_self_sqrt (by linarith)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.inv_sqrt_ge_inv
-- test: ra-inv-sqrt-ge-inv-test.sage

/-! ## def-absolute-convergence -/

/-- Absolute convergence implies convergence. -/
theorem abs_conv_conv (a : ℕ → ℝ) (h : Summable (fun k => |a k|)) : Summable a :=
  h.of_abs

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.abs_conv_conv ((fun _ : ℕ => (0 : ℝ))) ((summable_zero.congr (fun k => by simp))) := by
  exact summable_zero

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.abs_conv_conv ((fun k : ℕ => (1 / 2 : ℝ) ^ k)) ((summable_geometric_two.congr (fun k => (abs_of_nonneg (by positivity)).symm))) := by
  exact summable_geometric_two

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.abs_conv_conv ((fun k : ℕ => (1 / 3 : ℝ) ^ k)) (((summable_geometric_of_lt_one (by norm_num) (by norm_num)).congr (fun k => (abs_of_nonneg (by positivity)).symm))) := by
  exact summable_geometric_of_lt_one (by norm_num) (by norm_num)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Series.abs_conv_conv
-- test: ra-abs-conv-conv-test.sage

/-- The sum of an absolutely convergent series does not depend on the order of its terms. -/
theorem rearrangement_invariant (a : ℕ → ℝ) (h : Summable (fun k => |a k|)) (e : ℕ ≃ ℕ) :
    ∑' n, a (e n) = ∑' n, a n :=
  e.tsum_eq a

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Series.rearrangement_invariant
-- test: ra-rearrangement-invariant-test.sage

/-! ## exm-absolute-geometric -/

/-- `∑ |r^k|` converges for `|r| < 1` (`r = -1/2`). -/
theorem geometric_abs (r : ℝ) (hr : |r| < 1) : Summable (fun k : ℕ => |r ^ k|) := by
  simp_rw [abs_pow]
  exact summable_geometric_of_lt_one (abs_nonneg r) hr

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.geometric_abs
-- test: ra-geometric-abs-test.sage

/-- `∑ |(-1)^k / k^p|` converges for `p > 1` (`p = 2`), by comparison with the `p`-series. -/
theorem alt_p_abs (p : ℝ) (hp : 1 < p) : Summable (fun k : ℕ => |(-1 : ℝ) ^ k / (k : ℝ) ^ p|) := by
  have h := (p_series p).2 hp
  refine h.congr (fun k => ?_)
  rw [abs_div, abs_pow, abs_neg, abs_one, one_pow, abs_of_nonneg (Real.rpow_nonneg (Nat.cast_nonneg k) p)]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Series.alt_p_abs
-- test: ra-alt-p-abs-test.sage

/-! ## exm-absolute-alternating-harmonic -/

/-- The alternating harmonic series converges to `log 2`: `1 - 1/2 + 1/3 - … = log 2` (as a limit of partial sums; the
convergence is conditional, so mathlib's unconditional `HasSum` does not hold). -/
theorem alt_harmonic :
    Tendsto (fun n : ℕ => ∑ k ∈ range n, (-1 : ℝ) ^ k / ((k : ℝ) + 1)) atTop (𝓝 (Real.log 2)) := by
  have hanti : Antitone (fun k : ℕ => 1 / ((k : ℝ) + 1)) := fun a b hab =>
    one_div_le_one_div_of_le (by positivity) (by linarith [(Nat.cast_le (α := ℝ)).2 hab])
  have h0 : Tendsto (fun k : ℕ => 1 / ((k : ℝ) + 1)) atTop (𝓝 0) := tendsto_one_div_add_atTop_nhds_zero_nat
  obtain ⟨l, hl⟩ := hanti.tendsto_alternating_series_of_tendsto_zero h0
  have habel := Real.tendsto_tsum_powerSeries_nhdsWithin_lt hl
  have hval : ∀ x : ℝ, 0 < x → x < 1 →
      ∑' n : ℕ, (-1 : ℝ) ^ n * (1 / ((n : ℝ) + 1)) * x ^ n = Real.log (1 + x) / x := by
    intro x hx0 hx1
    have habs : |(-x)| < 1 := by rw [abs_neg, abs_lt]; constructor <;> linarith
    have h := ((Real.hasSum_pow_div_log_of_abs_lt_one habs).neg).div_const x
    refine HasSum.tsum_eq ?_
    have hxne : x ≠ 0 := hx0.ne'
    convert h using 1
    · funext n
      rw [neg_pow x (n + 1), pow_succ (-1 : ℝ) n, pow_succ x n]
      field_simp
    · rw [sub_neg_eq_add, neg_neg]
  have hcont : Tendsto (fun x : ℝ => Real.log (1 + x) / x) (𝓝[<] 1) (𝓝 (Real.log 2)) := by
    have hc : ContinuousAt (fun x : ℝ => Real.log (1 + x) / x) 1 := by
      apply ContinuousAt.div _ continuousAt_id one_ne_zero
      exact (Real.continuousAt_log (by norm_num)).comp (continuousAt_const.add continuousAt_id)
    have h1 := hc.tendsto.mono_left (nhdsWithin_le_nhds (s := Iio 1))
    rw [show Real.log (1 + 1) / 1 = Real.log 2 by norm_num] at h1
    exact h1
  have hl2 : l = Real.log 2 := by
    apply tendsto_nhds_unique habel
    apply hcont.congr'
    filter_upwards [Ioo_mem_nhdsLT (show (0 : ℝ) < 1 by norm_num)] with x hx
    exact (hval x hx.1 hx.2).symm
  rw [← hl2]
  refine hl.congr (fun n => Finset.sum_congr rfl (fun k _ => ?_))
  rw [mul_one_div]

/-- The series of the logarithm: `log(1 + x) = ∑_{k ≥ 1} (-1)^(k+1) x^k / k` for `-1 < x ≤ 1` (the partial sums
converge); at `x = 1` it is the alternating harmonic series with sum `log 2`. -/
theorem log_series (x : ℝ) (hx : -1 < x) (hx1 : x ≤ 1) :
    Tendsto (fun n : ℕ => ∑ k ∈ range n, (-1 : ℝ) ^ k * x ^ (k + 1) / ((k : ℝ) + 1)) atTop
      (𝓝 (Real.log (1 + x))) := by
  rcases hx1.lt_or_eq with hlt | rfl
  · have habs : |(-x)| < 1 := by rw [abs_neg, abs_lt]; exact ⟨hx, hlt⟩
    have h := (Real.hasSum_pow_div_log_of_abs_lt_one habs).neg
    rw [neg_neg, sub_neg_eq_add] at h
    refine (h.tendsto_sum_nat).congr (fun n => Finset.sum_congr rfl (fun k _ => ?_))
    rw [neg_pow x (k + 1), pow_succ (-1 : ℝ) k]
    push_cast
    ring
  · simpa [one_pow, one_add_one_eq_two] using alt_harmonic

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.log_series
-- test: ra-log-series-test.sage

/-! ## exm-absolute-rearrangement -/

/-- One block of the rearranged alternating harmonic series: a positive term and two negative ones. -/
noncomputable def blockTerm (j : ℕ) : ℝ := 1 / (2 * (j : ℝ) + 1) - 1 / (4 * (j : ℝ) + 2) - 1 / (4 * (j : ℝ) + 4)

/-- The term `(-1)^k/(k + 1)` of the alternating harmonic series. -/
noncomputable def altTerm (k : ℕ) : ℝ := (-1 : ℝ) ^ k / ((k : ℝ) + 1)

/-- After `b` blocks (one positive, two negative terms) the rearranged series has partial sum
`∑_{j<b} (1/(2j+1) - 1/(4j+2) - 1/(4j+4)) = ½ ∑_{k < 2b} (-1)^k/(k+1)`. -/
theorem rearranged_blocks (b : ℕ) :
    ∑ j ∈ range b, blockTerm j = 1 / 2 * ∑ k ∈ range (2 * b), altTerm k := by
  veriphy_certified
    (induction b with
      | zero => simp [blockTerm, altTerm]
      | succ b
        ih =>
        rw [Finset.sum_range_succ, ih, show 2 * (b + 1) = 2 * b + 1 + 1 by ring, Finset.sum_range_succ,
          Finset.sum_range_succ]
        simp only [blockTerm, altTerm]
        have e : (-1 : ℝ) ^ (2 * b) = 1 := by rw [pow_mul]; norm_num
        rw [pow_succ, e]
        push_cast
        field_simp
        ring)
  -- certificate: ra-rearranged-blocks.sage

/-- The rearranged alternating harmonic series has the sum `½ log 2`. -/
theorem rearranged_limit :
    Tendsto (fun b : ℕ => ∑ j ∈ range b, (1 / (2 * (j : ℝ) + 1) - 1 / (4 * (j : ℝ) + 2) - 1 / (4 * (j : ℝ) + 4)))
      atTop (𝓝 (Real.log 2 / 2)) := by
  have hcore : ∀ b : ℕ, ∑ j ∈ range b, blockTerm j = 1 / 2 * ∑ k ∈ range (2 * b), altTerm k := by
    intro b
    veriphy_certified
      (induction b with
        | zero => simp [blockTerm, altTerm]
        | succ b
          ih =>
          rw [Finset.sum_range_succ, ih, show 2 * (b + 1) = 2 * b + 1 + 1 by ring, Finset.sum_range_succ,
            Finset.sum_range_succ]
          simp only [blockTerm, altTerm]
          have e : (-1 : ℝ) ^ (2 * b) = 1 := by rw [pow_mul]; norm_num
          rw [pow_succ, e]
          push_cast
          field_simp
          ring)
    -- certificate: ra-rearranged-blocks.sage
  show Tendsto (fun b : ℕ => ∑ j ∈ range b, blockTerm j) atTop (𝓝 (Real.log 2 / 2))
  simp_rw [hcore]
  show Tendsto (fun b : ℕ => 1 / 2 * ∑ k ∈ range (2 * b), (-1 : ℝ) ^ k / ((k : ℝ) + 1)) atTop (𝓝 (Real.log 2 / 2))
  have h := alt_harmonic.comp (tendsto_id.const_mul_atTop' (by norm_num : 0 < 2))
  have := h.const_mul (1 / 2)
  rw [show 1 / 2 * Real.log 2 = Real.log 2 / 2 by ring] at this
  exact this

/-! ## exr-absolute-alternating -/

/-- Leibniz's error bound: the partial sums `s_m` of the alternating harmonic series alternate around `log 2`,
`(-1)^m (log 2 - s_m) ≥ 0`, with `|s_m - log 2| < 1/(m + 1)`. -/
theorem alt_harmonic_error (m : ℕ) :
    0 ≤ (-1 : ℝ) ^ m * (Real.log 2 - ∑ k ∈ range m, (-1 : ℝ) ^ k / ((k : ℝ) + 1)) ∧
      |∑ k ∈ range m, (-1 : ℝ) ^ k / ((k : ℝ) + 1) - Real.log 2| < 1 / ((m : ℝ) + 1) := by
  set f : ℕ → ℝ := fun k => 1 / ((k : ℝ) + 1) with hf
  have hanti : Antitone f := fun a b hab =>
    one_div_le_one_div_of_le (by positivity) (by linarith [(Nat.cast_le (α := ℝ)).2 hab])
  have hS : ∀ n, ∑ k ∈ range n, (-1 : ℝ) ^ k / ((k : ℝ) + 1) = ∑ i ∈ range n, (-1) ^ i * f i := fun n =>
    Finset.sum_congr rfl (fun k _ => by rw [hf, mul_one_div])
  have hl : Tendsto (fun n => ∑ i ∈ range n, (-1 : ℝ) ^ i * f i) atTop (𝓝 (Real.log 2)) :=
    alt_harmonic.congr (fun n => hS n)
  have lo := hanti.alternating_series_le_tendsto hl
  have hi := hanti.tendsto_le_alternating_series hl
  have hfs : ∀ j : ℕ, f (j + 1) < f j := fun j => by
    simp only [hf]
    push_cast
    exact one_div_lt_one_div_of_lt (by positivity) (by linarith)
  have e1 : ∀ k : ℕ, (-1 : ℝ) ^ (2 * k) = 1 := fun k => by rw [pow_mul]; norm_num
  have e2 : ∀ k : ℕ, (-1 : ℝ) ^ (2 * k + 1) = -1 := fun k => by rw [pow_succ, e1]; norm_num
  have e3 : ∀ k : ℕ, (-1 : ℝ) ^ (2 * k + 1 + 1) = 1 := fun k => by rw [pow_succ, e2]; norm_num
  have e4 : ∀ k : ℕ, (-1 : ℝ) ^ (2 * k + 1 + 1 + 1) = -1 := fun k => by rw [pow_succ, e3]; norm_num
  rw [hS]
  rcases Nat.even_or_odd m with ⟨k, rfl⟩ | ⟨k, rfl⟩
  · -- m = 2k: s_2k ≤ log 2 ≤ s_(2k+3) < s_(2k+1) = s_2k + f(2k)
    rw [← two_mul]
    have h1 := lo k
    have h2 := hi (k + 1)
    rw [show 2 * (k + 1) + 1 = 2 * k + 1 + 1 + 1 by ring, Finset.sum_range_succ, Finset.sum_range_succ,
      Finset.sum_range_succ, e1, e2, e3] at h2
    have hlt := hfs (2 * k + 1)
    have hf0 : f (2 * k) = 1 / (((2 * k : ℕ) : ℝ) + 1) := rfl
    refine ⟨by rw [e1, one_mul]; linarith, ?_⟩
    rw [abs_sub_comm, abs_of_nonneg (by linarith), ← hf0]
    linarith
  · -- m = 2k+1: s_(2k+4) ≤ log 2 ≤ s_(2k+1), and s_(2k+1) - f(2k+1) < s_(2k+4)
    have h1 := hi k
    have h3 := lo (k + 2)
    rw [show 2 * (k + 2) = 2 * k + 1 + 1 + 1 + 1 by ring, Finset.sum_range_succ, Finset.sum_range_succ,
      Finset.sum_range_succ, e2, e3, e4] at h3
    have hlt := hfs (2 * k + 1 + 1)
    have hf0 : f (2 * k + 1) = 1 / (((2 * k + 1 : ℕ) : ℝ) + 1) := rfl
    refine ⟨by rw [e2]; linarith, ?_⟩
    rw [abs_of_nonneg (by linarith), ← hf0]
    linarith

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.RealAnalysis.Series.alt_harmonic_error
-- test: ra-alt-harmonic-error-test.sage

/-! ## exr-absolute-conditional-proof -/

/-- The triangle inequality for an absolutely convergent series: `|∑ a_k| ≤ ∑ |a_k|`. -/
theorem abs_tsum_le (a : ℕ → ℝ) (h : Summable (fun k => |a k|)) : |∑' k, a k| ≤ ∑' k, |a k| := by
  have := norm_tsum_le_tsum_norm (f := a) (by simpa [Real.norm_eq_abs] using h)
  simpa [Real.norm_eq_abs] using this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Series.abs_tsum_le
-- test: ra-abs-tsum-le-test.sage

/-! ## thm-convergence-tests -/

/-- The comparison test: `0 ≤ a_k ≤ b_k` and `∑ b_k` convergent imply `∑ a_k` convergent. -/
theorem comparison_test (a b : ℕ → ℝ) (h0 : ∀ k, 0 ≤ a k) (hab : ∀ k, a k ≤ b k) (hb : Summable b) :
    Summable a :=
  hb.of_nonneg_of_le h0 hab

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.comparison_test ((fun _ : ℕ => (0 : ℝ))) ((fun k : ℕ => (1 / 2 : ℝ) ^ k)) ((fun _ => le_rfl)) ((fun k => by positivity)) (summable_geometric_two) := by
  exact summable_zero

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.comparison_test ((fun k : ℕ => (1 / 3 : ℝ) ^ k)) ((fun k : ℕ => (1 / 2 : ℝ) ^ k)) ((fun k => by positivity)) ((fun k => pow_le_pow_left₀ (by norm_num) (by norm_num) k)) (summable_geometric_two) := by
  exact summable_geometric_of_lt_one (by norm_num) (by norm_num)

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.comparison_test ((fun k : ℕ => (1 / 4 : ℝ) ^ k)) ((fun k : ℕ => (1 / 2 : ℝ) ^ k)) ((fun k => by positivity)) ((fun k => pow_le_pow_left₀ (by norm_num) (by norm_num) k)) (summable_geometric_two) := by
  exact summable_geometric_of_lt_one (by norm_num) (by norm_num)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Series.comparison_test
-- test: ra-comparison-test-test.sage

/-- The comparison test for divergence: if `∑ a_k` diverges, so does `∑ b_k`. -/
theorem comparison_test_diverges (a b : ℕ → ℝ) (h0 : ∀ k, 0 ≤ a k) (hab : ∀ k, a k ≤ b k)
    (ha : ¬ Summable a) : ¬ Summable b :=
  fun hb => ha (comparison_test a b h0 hab hb)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Series.comparison_test_diverges
-- test: ra-comparison-test-diverges-test.sage

/-- The ratio test, convergent case: `|a_(k+1)/a_k| → L < 1` gives absolute convergence. -/
theorem ratio_test (a : ℕ → ℝ) (L : ℝ) (hL : L < 1) (h0 : ∀ k, a k ≠ 0)
    (h : Tendsto (fun k => |a (k + 1)| / |a k|) atTop (𝓝 L)) : Summable (fun k => |a k|) := by
  have : Summable a := summable_of_ratio_test_tendsto_lt_one hL (Eventually.of_forall h0)
    (by simpa [Real.norm_eq_abs] using h)
  have h' : Summable (fun k => ‖a k‖) := by
    refine summable_of_ratio_test_tendsto_lt_one hL (Eventually.of_forall (fun k => by simpa using h0 k)) ?_
    simpa [Real.norm_eq_abs] using h
  simpa [Real.norm_eq_abs] using h'

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.ratio_test
-- test: ra-ratio-test-test.sage

/-- The ratio test, divergent case: `|a_(k+1)/a_k| → L > 1` gives divergence. -/
theorem ratio_test_diverges (a : ℕ → ℝ) (L : ℝ) (hL : 1 < L)
    (h : Tendsto (fun k => |a (k + 1)| / |a k|) atTop (𝓝 L)) : ¬ Summable a :=
  not_summable_of_ratio_test_tendsto_gt_one hL (by simpa [Real.norm_eq_abs] using h)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Series.ratio_test_diverges
-- test: ra-ratio-test-diverges-test.sage

/-- The ratio test is inconclusive at `L = 1`: for every `p` the ratio of `1/n^p` tends to `1`, and the
`p`-series converges for `p = 2` and diverges for `p = 1`. -/
theorem ratio_p_series (p : ℝ) :
    Tendsto (fun n : ℕ => (1 / ((n : ℝ) + 2) ^ p) / (1 / ((n : ℝ) + 1) ^ p)) atTop (𝓝 1) := by
  have h : Tendsto (fun n : ℕ => (((n : ℝ) + 1) / ((n : ℝ) + 2)) ^ p) atTop (𝓝 (1 ^ p)) := by
    apply Tendsto.rpow_const _ (Or.inl one_ne_zero)
    have : Tendsto (fun n : ℕ => 1 - 1 / ((n : ℝ) + 2)) atTop (𝓝 (1 - 0)) := by
      apply tendsto_const_nhds.sub
      apply Tendsto.div_atTop tendsto_const_nhds
      exact tendsto_atTop_add_const_right _ 2 tendsto_natCast_atTop_atTop
    rw [sub_zero] at this
    refine this.congr (fun n => ?_)
    field_simp
    ring
  rw [Real.one_rpow] at h
  refine h.congr (fun n => ?_)
  rw [Real.div_rpow (by positivity) (by positivity)]
  field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.ratio_p_series
-- test: ra-ratio-p-series-test.sage

/-! ## exm-tests-exponential -/

/-- The ratio of consecutive terms of `∑ x^k/k!` is `|x|/(k + 1)`, which tends to `0`. -/
theorem exp_ratio (x : ℝ) (hx : x ≠ 0) (k : ℕ) :
    |x ^ (k + 1) / ((k + 1)! : ℝ)| / |x ^ k / (k ! : ℝ)| = |x| / ((k : ℝ) + 1) := by
  rw [Nat.factorial_succ, abs_div, abs_div, abs_pow, abs_pow, pow_succ]
  push_cast
  have h1 : |x| ^ k ≠ 0 := pow_ne_zero _ (abs_ne_zero.2 hx)
  have h2 : (k ! : ℝ) ≠ 0 := by positivity
  rw [abs_of_pos (by positivity : (0 : ℝ) < ((k : ℝ) + 1) * k !), abs_of_pos (by positivity : (0 : ℝ) < k !)]
  field_simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_ratio (1) ((by norm_num)) (0) := by
  norm_num [Nat.factorial, abs_div]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_ratio (2) ((by norm_num)) (3) := by
  norm_num [Nat.factorial, abs_div]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_ratio (-3) ((by norm_num)) (5) := by
  norm_num [Nat.factorial, abs_div]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_ratio (1 / 2) ((by norm_num)) (1) := by
  norm_num [Nat.factorial, abs_div]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Series.exp_ratio
-- test: ra-exp-ratio-test.sage

/-! ## exm-tests-factorial -/

/-- The ratio of consecutive terms of `n!/n^n` is `(n/(n + 1))^n`. -/
theorem factorial_power_ratio (n : ℕ) (hn : 1 ≤ n) :
    (((n + 1)! : ℝ) / ((n : ℝ) + 1) ^ (n + 1)) / ((n ! : ℝ) / (n : ℝ) ^ n) = ((n : ℝ) / ((n : ℝ) + 1)) ^ n := by
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  rw [Nat.factorial_succ]
  push_cast
  have h2 : (n ! : ℝ) ≠ 0 := by positivity
  rw [div_pow, pow_succ]
  field_simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.factorial_power_ratio (1) ((by norm_num)) := by
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.factorial_power_ratio (2) ((by norm_num)) := by
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.factorial_power_ratio (3) ((by norm_num)) := by
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.factorial_power_ratio (6) ((by norm_num)) := by
  norm_num [Nat.factorial]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Series.factorial_power_ratio
-- test: ra-factorial-power-ratio-test.sage

/-- `1/(1 + x/n)^n → e^(-x)`; at `x = 1` this is `(n/(n + 1))^n → 1/e`. -/
theorem inv_one_add_div_pow_limit (x : ℝ) :
    Tendsto (fun n : ℕ => 1 / (1 + x / n) ^ n) atTop (𝓝 (Real.exp (-x))) := by
  have h := (Real.tendsto_one_add_div_pow_exp x).inv₀ (Real.exp_pos x).ne'
  rw [← Real.exp_neg] at h
  simpa [one_div] using h

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.inv_one_add_div_pow_limit
-- test: ra-inv-one-add-div-pow-limit-test.sage

/-- `∑ n² x^n = x(1 + x)/(1 - x)³` for `|x| < 1`; at `x = 1/2` the sum `∑ n²/2^n` is `6`. -/
theorem sum_sq_mul_geometric (x : ℝ) (hx : |x| < 1) :
    HasSum (fun n : ℕ => (n : ℝ) ^ 2 * x ^ n) (x * (1 + x) / (1 - x) ^ 3) := by
  have hx1 : (1 - x) ≠ 0 := by
    intro h; have : x = 1 := by linarith
    rw [this] at hx; norm_num at hx
  have hn : ‖x‖ < 1 := by rwa [Real.norm_eq_abs]
  have h1 := hasSum_coe_mul_geometric_of_norm_lt_one hn
  have h2 := hasSum_descFactorial_mul_geometric_of_norm_lt_one 2 hn
  have h := h2.add h1
  convert h using 1
  · funext n
    rw [Nat.cast_descFactorial_two] <;> ring
  · field_simp
    ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.sum_sq_mul_geometric
-- test: ra-sum-sq-mul-geometric-test.sage

/-! ## exm-tests-p-series -/

/-- Cauchy condensation: the block of `2^m` terms of `∑ 1/k^p` gives the geometric term `2^m · 2^(-mp) = (2^(1-p))^m`,
so the condensed series is geometric with ratio `2^(1-p)`, which is below `1` exactly when `p > 1`. -/
theorem condensed_term (p : ℝ) (m : ℕ) :
    (2 : ℝ) ^ m * (1 / ((2 : ℝ) ^ m) ^ p) = ((2 : ℝ) ^ (1 - p)) ^ m ∧ ((2 : ℝ) ^ (1 - p) < 1 ↔ 1 < p) := by
  constructor
  · rw [← Real.rpow_natCast, ← Real.rpow_natCast, ← Real.rpow_mul (by norm_num), ← Real.rpow_mul (by norm_num),
      one_div, ← Real.rpow_neg (by norm_num), ← Real.rpow_add (by norm_num)]
    congr 1
    ring
  · rw [Real.rpow_lt_one_iff_of_pos (by norm_num)]
    constructor
    · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩)
      · linarith
      · norm_num at h1
    · intro h; left; exact ⟨by norm_num, by linarith⟩

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.condensed_term
-- test: ra-condensed-term-test.sage

/-! ## def-power-series -/

/-- Term-by-term differentiation of a power series inside its interval of convergence (stated, tested at the
geometric and exponential series, not proved here): if `∑ c_k r^k` converges absolutely for every `r < R`, then
`f(x) = ∑ c_k x^k` has derivative `∑ k c_k x^(k-1)` on `(-R, R)`. -/
def TermwiseDerivative : Prop :=
  ∀ (c : ℕ → ℝ) (R : ℝ), (∀ r, 0 ≤ r → r < R → Summable (fun k => |c k| * r ^ k)) →
    ∀ x, |x| < R → HasDerivAt (fun y => ∑' k, c k * y ^ k) (∑' k : ℕ, ((k : ℝ) + 1) * c (k + 1) * x ^ k) x

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Series.TermwiseDerivative
-- test: ra-termwisederivative-test.sage

/-! ## exm-power-geometric -/

/-- Differentiating the geometric series term by term: `∑ (k + 1) x^k = 1/(1 - x)²` for `|x| < 1`. -/
theorem geometric_derivative (x : ℝ) (hx : |x| < 1) :
    HasSum (fun k : ℕ => ((k : ℝ) + 1) * x ^ k) (1 / (1 - x) ^ 2) := by
  have hn : ‖x‖ < 1 := by rwa [Real.norm_eq_abs]
  have hx1 : (1 - x) ≠ 0 := by
    intro h; have : x = 1 := by linarith
    rw [this] at hx; norm_num at hx
  have h1 := hasSum_coe_mul_geometric_of_norm_lt_one hn
  have h2 := hasSum_geometric_of_norm_lt_one hn
  convert h1.add h2 using 1
  · funext k; ring
  · field_simp; ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.geometric_derivative
-- test: ra-geometric-derivative-test.sage

/-- Integrating the geometric series term by term: `∑ x^(k+1)/(k + 1) = -log(1 - x)` for `|x| < 1`. -/
theorem geometric_integral (x : ℝ) (hx : |x| < 1) :
    HasSum (fun k : ℕ => x ^ (k + 1) / ((k : ℝ) + 1)) (-Real.log (1 - x)) :=
  Real.hasSum_pow_div_log_of_abs_lt_one hx

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.geometric_integral
-- test: ra-geometric-integral-test.sage

/-! ## exm-power-exponential -/

/-- The power series of `sin` and `cos`, convergent for every `x`. -/
theorem sin_cos_series (x : ℝ) :
    HasSum (fun k : ℕ => (-1 : ℝ) ^ k * x ^ (2 * k + 1) / ((2 * k + 1)! : ℝ)) (Real.sin x) ∧
      HasSum (fun k : ℕ => (-1 : ℝ) ^ k * x ^ (2 * k) / ((2 * k)! : ℝ)) (Real.cos x) :=
  ⟨Real.hasSum_sin x, Real.hasSum_cos x⟩

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.RealAnalysis.Series.sin_cos_series
-- test: ra-sin-cos-series-test.sage

/-- Term-by-term differentiation of the exponential series: `(x^(k+1)/(k+1)!)' = x^k/k!`. -/
theorem exp_term_deriv (k : ℕ) (x : ℝ) :
    HasDerivAt (fun y : ℝ => y ^ (k + 1) / ((k + 1)! : ℝ)) (x ^ k / (k ! : ℝ)) x := by
  have h := (hasDerivAt_pow (k + 1) x).div_const ((k + 1)! : ℝ)
  convert h using 1
  rw [Nat.factorial_succ]
  push_cast
  have : (k ! : ℝ) ≠ 0 := by positivity
  field_simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_term_deriv (0) (0) := by
  have h := (hasDerivAt_pow (0 + 1) (0 : ℝ)).div_const (((0 + 1 : ℕ)) ! : ℝ)
  convert h using 1
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_term_deriv (2) (3) := by
  have h := (hasDerivAt_pow (2 + 1) (3 : ℝ)).div_const (((2 + 1 : ℕ)) ! : ℝ)
  convert h using 1
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_term_deriv (4) (-1) := by
  have h := (hasDerivAt_pow (4 + 1) (-1 : ℝ)).div_const (((4 + 1 : ℕ)) ! : ℝ)
  convert h using 1
  norm_num [Nat.factorial]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Series.exp_term_deriv
-- test: ra-exp-term-deriv-test.sage

/-! ## exm-power-binomial -/

/-- The binomial series: `(1 + x)^α = ∑ C(α, k) x^k` for `|x| < 1` and every real `α`. -/
theorem binomial_series (a x : ℝ) (hx : |x| < 1) :
    HasSum (fun k : ℕ => Ring.choose a k * x ^ k) ((1 + x) ^ a) := by
  have h := Real.one_add_rpow_hasFPowerSeriesOnBall_zero (a := a)
  have hx' : x ∈ Metric.eball (0 : ℝ) 1 := by
    rw [Metric.mem_eball, edist_dist, Real.dist_eq, sub_zero]
    exact ENNReal.ofReal_lt_one.2 hx
  have := h.hasSum hx'
  rw [zero_add] at this
  convert this using 2 with k
  rw [binomialSeries, FormalMultilinearSeries.ofScalars_apply_eq, smul_eq_mul]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.binomial_series
-- test: ra-binomial-series-test.sage

/-- The first binomial coefficients: `C(-1, k) = (-1)^k` (the geometric series in `-x`) and `C(1/2, 1) = 1/2`,
`C(1/2, 2) = -1/8` (`√(1 + x) = 1 + x/2 - x²/8 + …`). -/
theorem binomial_coefficients (k : ℕ) :
    Ring.choose (-1 : ℝ) k = (-1) ^ k ∧ Ring.choose (1 / 2 : ℝ) 1 = 1 / 2 ∧ Ring.choose (1 / 2 : ℝ) 2 = -1 / 8 := by
  refine ⟨?_, Ring.choose_one_right _, ?_⟩
  · induction k with
    | zero => simp [Ring.choose_zero_right]
    | succ k ih =>
      have h := Ring.choose_succ_succ (-1 : ℝ) k
      rw [neg_add_cancel, show (0 : ℝ) = ((0 : ℕ) : ℝ) by simp, Ring.choose_natCast, Nat.choose_zero_succ] at h
      push_cast at h
      rw [pow_succ]
      linarith
  · rw [Ring.choose_eq_smul]
    simp [descPochhammer_succ_right, Polynomial.smeval_mul, Polynomial.smeval_X, Polynomial.smeval_sub,
      Polynomial.smeval_natCast, Nat.factorial]
    norm_num

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.binomial_coefficients
-- test: ra-binomial-coefficients-test.sage

/-- The non-relativistic limit: `√(m² + p²) ≤ m + p²/(2m)`, with equality to second order. -/
theorem sqrt_energy (m p : ℝ) (hm : 0 < m) :
    √(m ^ 2 + p ^ 2) ≤ m + p ^ 2 / (2 * m) ∧ m + p ^ 2 / (2 * m) - p ^ 4 / (8 * m ^ 3) ≤ √(m ^ 2 + p ^ 2) := by
  have hq : 0 ≤ p ^ 2 / (2 * m) := by positivity
  constructor
  · rw [Real.sqrt_le_left (by positivity)]
    have : (m + p ^ 2 / (2 * m)) ^ 2 = m ^ 2 + p ^ 2 + (p ^ 2 / (2 * m)) ^ 2 := by field_simp; ring
    nlinarith [sq_nonneg (p ^ 2 / (2 * m))]
  · rcases le_or_gt (m + p ^ 2 / (2 * m) - p ^ 4 / (8 * m ^ 3)) 0 with h | h
    · exact h.trans (Real.sqrt_nonneg _)
    · rw [Real.le_sqrt h.le (by positivity)]
      set u := p ^ 2 / (2 * m ^ 2) with hu
      have hu0 : 0 ≤ u := by positivity
      have e1 : m + p ^ 2 / (2 * m) - p ^ 4 / (8 * m ^ 3) = m * (1 + u - u ^ 2 / 2) := by
        rw [hu]; field_simp <;> ring
      have e2 : m ^ 2 + p ^ 2 = m ^ 2 * (1 + 2 * u) := by rw [hu]; field_simp <;> ring
      rw [e1] at h ⊢
      rw [e2]
      have hpos : 0 < 1 + u - u ^ 2 / 2 := by
        by_contra hc
        push_neg at hc
        have := mul_nonpos_of_nonneg_of_nonpos hm.le hc
        linarith
      have hu4 : u ≤ 4 := by
        by_contra hc
        push_neg at hc
        nlinarith
      have key : (1 + u - u ^ 2 / 2) ^ 2 ≤ 1 + 2 * u := by
        nlinarith [mul_nonneg (pow_nonneg hu0 3) (by linarith : (0 : ℝ) ≤ 4 - u)]
      calc (m * (1 + u - u ^ 2 / 2)) ^ 2 = m ^ 2 * (1 + u - u ^ 2 / 2) ^ 2 := by ring
        _ ≤ m ^ 2 * (1 + 2 * u) := by gcongr

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.sqrt_energy
-- test: ra-sqrt-energy-test.sage

/-! ## exr-power-series-exp -/

/-- The remainder of the exponential series at `x = 1`: `1/(m+1)! ≤ e - ∑_{k ≤ m} 1/k! ≤ (m+2)/((m+1)!(m+1)) < 1/m!`
(checked at `m = 3, 6, 9`; at `m = 12` the gap is `1.7 · 10⁻¹⁰`). -/
theorem exp_remainder (m : ℕ) (hm : 1 ≤ m) :
    1 / ((m + 1)! : ℝ) ≤ Real.exp 1 - expPartial m ∧
      Real.exp 1 - expPartial m ≤ ((m : ℝ) + 2) / (((m + 1)! : ℝ) * ((m : ℝ) + 1)) ∧
      ((m : ℝ) + 2) / (((m + 1)! : ℝ) * ((m : ℝ) + 1)) < 1 / (m ! : ℝ) := by
  have hle : expPartial (m + 1) ≤ Real.exp 1 := by
    have := Real.sum_le_exp_of_nonneg (zero_le_one' ℝ) (m + 2)
    simpa [expPartial] using this
  have hb := Real.exp_bound (x := 1) (by norm_num) (n := m + 1) (by omega)
  simp only [abs_one, one_pow, one_mul] at hb
  refine ⟨?_, ?_, ?_⟩
  · have : expPartial (m + 1) = expPartial m + 1 / ((m + 1)! : ℝ) := by
      simp [expPartial, Finset.sum_range_succ _ (m + 1)]
    linarith
  · have := (abs_le.1 hb).2
    have e : (∑ i ∈ range (m + 1), (1 : ℝ) / (i ! : ℝ)) = expPartial m := rfl
    rw [e] at this
    push_cast at this ⊢
    rw [show (m : ℝ) + 1 + 1 = m + 2 by ring] at this
    linarith
  · rw [Nat.factorial_succ]
    push_cast
    have hf : (0 : ℝ) < m ! := by positivity
    have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
    rw [div_lt_div_iff₀ (by positivity) hf]
    have h1 : (m : ℝ) + 2 < ((m : ℝ) + 1) * ((m : ℝ) + 1) := by nlinarith
    nlinarith [mul_lt_mul_of_pos_right h1 hf]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_remainder (1) ((by norm_num)) := by
  have h1 := Real.exp_one_gt_d9
  have h2 := Real.exp_one_lt_d9
  simp only [KaluzaKlein.RealAnalysis.Series.expPartial, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]
  norm_num
  refine ⟨by linarith, by linarith⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_remainder (3) ((by norm_num)) := by
  have h1 := Real.exp_one_gt_d9
  have h2 := Real.exp_one_lt_d9
  simp only [KaluzaKlein.RealAnalysis.Series.expPartial, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]
  norm_num
  refine ⟨by linarith, by linarith⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_remainder (6) ((by norm_num)) := by
  have h1 := Real.exp_one_gt_d9
  have h2 := Real.exp_one_lt_d9
  simp only [KaluzaKlein.RealAnalysis.Series.expPartial, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]
  norm_num
  refine ⟨by linarith, by linarith⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Series.exp_remainder
-- test: ra-exp-remainder-test.sage

/-! ## exr-power-series-geometric-derivative -/

/-- `∑ k x^k = x/(1 - x)²` and `∑ k(k - 1) x^k = 2x²/(1 - x)³` for `|x| < 1`; at `x = 1/2`, `∑ k/2^k = 2` and
`∑ k²/2^k = 6`. -/
theorem geometric_moments (x : ℝ) (hx : |x| < 1) :
    HasSum (fun k : ℕ => (k : ℝ) * x ^ k) (x / (1 - x) ^ 2) ∧
      HasSum (fun k : ℕ => (k : ℝ) * ((k : ℝ) - 1) * x ^ k) (2 * x ^ 2 / (1 - x) ^ 3) := by
  have hn : ‖x‖ < 1 := by rwa [Real.norm_eq_abs]
  refine ⟨hasSum_coe_mul_geometric_of_norm_lt_one hn, ?_⟩
  have h2 := hasSum_descFactorial_mul_geometric_of_norm_lt_one 2 hn
  convert h2 using 1
  · funext k
    rw [Nat.cast_descFactorial_two] <;> ring
  · norm_num [Nat.factorial]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Series.geometric_moments
-- test: ra-geometric-moments-test.sage

/-! ## exr-power-series-log -/

/-- `1/(1 + x) = ∑ (-1)^k x^k` for `|x| < 1`, the geometric series in `-x`. -/
theorem geometric_neg (x : ℝ) (hx : |x| < 1) : HasSum (fun k : ℕ => (-1 : ℝ) ^ k * x ^ k) (1 / (1 + x)) := by
  have h := (geometric_sums (-x) (by rwa [abs_neg])).1
  rw [sub_neg_eq_add] at h
  convert h using 2 with k
  exact (neg_pow x k).symm

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Series.geometric_neg
-- test: ra-geometric-neg-test.sage

/-! ## def-radius-of-convergence -/

/-- The ratio formula for the radius: if `|c_k / c_(k+1)| → R > 0`, then `∑ c_k x^k` converges absolutely for
`|x| < R` and diverges for `|x| > R`. -/
theorem radius_ratio (c : ℕ → ℝ) (R : ℝ) (hR : 0 < R) (hc : ∀ k, c k ≠ 0)
    (h : Tendsto (fun k => |c k / c (k + 1)|) atTop (𝓝 R)) (x : ℝ) :
    (|x| < R → Summable (fun k => |c k * x ^ k|)) ∧ (R < |x| → ¬ Summable (fun k => c k * x ^ k)) := by
  have hinv : Tendsto (fun k => |c (k + 1)| / |c k|) atTop (𝓝 R⁻¹) := by
    refine (h.inv₀ hR.ne').congr (fun k => ?_)
    rw [abs_div, inv_div]
  have hrat : Tendsto (fun k => ‖c (k + 1) * x ^ (k + 1)‖ / ‖c k * x ^ k‖) atTop (𝓝 (R⁻¹ * |x|)) := by
    refine (hinv.mul_const |x|).congr (fun k => ?_)
    rw [Real.norm_eq_abs, Real.norm_eq_abs]
    by_cases hx : x = 0
    · subst hx; simp
    · have h1 : |c k| ≠ 0 := abs_ne_zero.2 (hc k)
      have h2 : |x| ^ k ≠ 0 := pow_ne_zero _ (abs_ne_zero.2 hx)
      rw [abs_mul, abs_mul, abs_pow, abs_pow, pow_succ, div_mul_eq_mul_div,
        div_eq_div_iff h1 (mul_ne_zero h1 h2)]
      ring
  constructor
  · intro hx
    by_cases hx0 : x = 0
    · subst hx0
      refine (hasSum_single 0 (fun k hk => ?_)).summable
      simp [zero_pow hk]
    · have hlt : R⁻¹ * |x| < 1 := by
        rw [← div_eq_inv_mul, div_lt_one hR]; exact hx
      have hs := summable_of_ratio_test_tendsto_lt_one (f := fun k => c k * x ^ k) hlt
        (Eventually.of_forall (fun k => mul_ne_zero (hc k) (pow_ne_zero _ hx0))) hrat
      exact hs.abs
  · intro hx
    have hgt : 1 < R⁻¹ * |x| := by
      rw [← div_eq_inv_mul, one_lt_div hR]; exact hx
    exact not_summable_of_ratio_test_tendsto_gt_one (f := fun k => c k * x ^ k) hgt hrat

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.radius_ratio
-- test: ra-radius-ratio-test.sage

/-- Cauchy-Hadamard (stated, tested at the standard series, not proved here): the radius is
`1/limsup |c_k|^(1/k)`; for coefficients whose `k`-th roots converge to `ρ > 0`, the series converges absolutely for
`|x| < 1/ρ` and diverges for `|x| > 1/ρ`. -/
def CauchyHadamard : Prop :=
  ∀ (c : ℕ → ℝ) (ρ : ℝ), 0 < ρ → Tendsto (fun k : ℕ => |c k| ^ ((k : ℝ)⁻¹)) atTop (𝓝 ρ) →
    ∀ x : ℝ, (|x| < 1 / ρ → Summable (fun k => |c k * x ^ k|)) ∧ (1 / ρ < |x| → ¬ Summable (fun k => c k * x ^ k))

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.CauchyHadamard
-- test: ra-cauchyhadamard-test.sage

/-! ## exm-radius-singularity -/

/-- `1/(1 + x²) = ∑ (-1)^k x^(2k)` for `|x| < 1`, and the series diverges for `|x| ≥ 1`: the radius is `1`. -/
theorem inv_one_add_sq_series (x : ℝ) :
    (|x| < 1 → HasSum (fun k : ℕ => (-1 : ℝ) ^ k * x ^ (2 * k)) (1 / (1 + x ^ 2))) ∧
      (1 ≤ |x| → ¬ Summable (fun k : ℕ => (-1 : ℝ) ^ k * x ^ (2 * k))) := by
  constructor
  · intro hx
    have h2 : |x ^ 2| < 1 := by
      rw [abs_pow]; nlinarith [abs_nonneg x]
    have := geometric_neg (x ^ 2) h2
    simpa [pow_mul] using this
  · intro hx hs
    have ht := hs.tendsto_atTop_zero
    have hge : ∀ k : ℕ, 1 ≤ |(-1 : ℝ) ^ k * x ^ (2 * k)| := by
      intro k
      rw [abs_mul, abs_pow, abs_neg, abs_one, one_pow, one_mul, abs_pow]
      exact one_le_pow₀ hx
    have h0 := (continuous_abs.tendsto 0).comp ht
    rw [abs_zero] at h0
    have := ge_of_tendsto h0 (Eventually.of_forall hge)
    norm_num at this

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.inv_one_add_sq_series
-- test: ra-inv-one-add-sq-series-test.sage

/-- The radius is the distance to the nearest complex singularity (stated, tested, not proved here: a theorem of
complex analysis): at `x₀ = 0`, `1/(1 + x²)` has radius `1`, `1/(x - 2)` radius `2`, `1/((x - 2)(x² + 9))` radius
`2`, and at `x₀ = 1` the last one has radius `1` (distances `1` and `√10`). -/
def RadiusSingularity : Prop :=
  ∀ (P : Polynomial ℂ) (x₀ : ℝ), P.eval (x₀ : ℂ) ≠ 0 →
    ∀ x : ℝ, (∀ z : ℂ, P.eval z = 0 → |x - x₀| < ‖z - x₀‖) →
      ∃ c : ℕ → ℂ, HasSum (fun k => c k * ((x : ℂ) - x₀) ^ k) (1 / P.eval (x : ℂ))

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.RadiusSingularity
-- test: ra-radiussingularity-test.sage

/-! ## exr-radius-boundary -/

/-- `∑ x^k/k` converges exactly on `[-1, 1)` (its partial sums converge; at `-1` conditionally). -/
theorem log_series_set (x : ℝ) :
    (∃ L, Tendsto (fun n : ℕ => ∑ k ∈ range n, x ^ (k + 1) / ((k : ℝ) + 1)) atTop (𝓝 L)) ↔ -1 ≤ x ∧ x < 1 := by
  constructor
  · rintro ⟨L, hL⟩
    -- the terms tend to 0
    have hterm : Tendsto (fun n : ℕ => x ^ (n + 1) / ((n : ℝ) + 1)) atTop (𝓝 0) := by
      have h2 := (hL.comp (tendsto_add_atTop_nat 1)).sub hL
      rw [sub_self] at h2
      refine h2.congr (fun n => ?_)
      simp [Finset.sum_range_succ]
    refine ⟨?_, ?_⟩
    · by_contra hlt
      push_neg at hlt
      -- |x| > 1: the terms are at least |x| - 1 in absolute value (Bernoulli)
      have hb : ∀ n : ℕ, |x| - 1 ≤ |x ^ (n + 1) / ((n : ℝ) + 1)| := by
        intro n
        have hx1 : 1 < |x| := by rw [abs_of_neg (by linarith)]; linarith
        have hber := one_add_mul_le_pow (show (-2 : ℝ) ≤ |x| - 1 by linarith) (n + 1)
        rw [add_sub_cancel] at hber
        rw [abs_div, abs_pow, abs_of_pos (by positivity : (0 : ℝ) < (n : ℝ) + 1), le_div_iff₀ (by positivity)]
        push_cast at hber
        nlinarith
      have h0 := (continuous_abs.tendsto 0).comp hterm
      rw [abs_zero] at h0
      have := ge_of_tendsto h0 (Eventually.of_forall hb)
      have hx1 : 1 < |x| := by rw [abs_of_neg (by linarith)]; linarith
      linarith
    · by_contra hge
      push_neg at hge
      -- x ≥ 1: the partial sums dominate the harmonic numbers, which are unbounded
      have hdom : ∀ n : ℕ, Real.log ((n : ℝ) + 1) ≤ ∑ k ∈ range n, x ^ (k + 1) / ((k : ℝ) + 1) := by
        intro n
        have hH := log_add_one_le_harmonic n
        simp only [harmonic, Rat.cast_sum, Rat.cast_inv] at hH
        push_cast at hH
        calc Real.log ((n : ℝ) + 1) ≤ ∑ i ∈ range n, ((i : ℝ) + 1)⁻¹ := by simpa using hH
          _ ≤ _ := Finset.sum_le_sum (fun k _ => by
              rw [← one_div]
              exact div_le_div_of_nonneg_right (one_le_pow₀ hge) (by positivity))
      have hinf : Tendsto (fun n : ℕ => Real.log ((n : ℝ) + 1)) atTop atTop :=
        Real.tendsto_log_atTop.comp (tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop)
      exact not_tendsto_nhds_of_tendsto_atTop (tendsto_atTop_mono hdom hinf) L hL
  · rintro ⟨h1, h2⟩
    rcases h1.lt_or_eq with hlt | rfl
    · have habs : |x| < 1 := abs_lt.2 ⟨hlt, h2⟩
      exact ⟨_, (Real.hasSum_pow_div_log_of_abs_lt_one habs).tendsto_sum_nat⟩
    · refine ⟨-Real.log 2, (alt_harmonic.neg).congr (fun n => ?_)⟩
      rw [← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl (fun k _ => ?_)
      rw [pow_succ]
      ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.log_series_set
-- test: ra-log-series-set-test.sage

/-- `∑ x^k/k²` converges exactly on `[-1, 1]` (absolutely). -/
theorem sq_series_set (x : ℝ) : Summable (fun k : ℕ => x ^ (k + 1) / ((k : ℝ) + 1) ^ 2) ↔ |x| ≤ 1 := by
  constructor
  · intro hs
    by_contra hgt
    push_neg at hgt
    set d := |x| - 1 with hd
    have hd0 : 0 < d := by rw [hd]; linarith
    -- (1 + d)^(n+1) ≥ C(n+1, 2) d², so the terms are at least d²/4 in absolute value for n ≥ 1
    have hb : ∀ n : ℕ, 1 ≤ n → d ^ 2 / 4 ≤ |x ^ (n + 1) / ((n : ℝ) + 1) ^ 2| := by
      intro n hn
      have hbin : (((n + 1).choose 2 : ℕ) : ℝ) * d ^ 2 ≤ (1 + d) ^ (n + 1) := by
        rw [add_comm 1 d, add_pow]
        have hmem : 2 ∈ Finset.range (n + 1 + 1) := Finset.mem_range.2 (by omega)
        refine le_trans (le_of_eq ?_) (Finset.single_le_sum (f := fun k => d ^ k * 1 ^ (n + 1 - k) * ((n + 1).choose k : ℝ))
          (fun k _ => by positivity) hmem)
        simp only [one_pow, mul_one]
        ring
      rw [Nat.cast_choose_two] at hbin
      have hxd : |x| = 1 + d := by rw [hd]; ring
      rw [abs_div, abs_pow, hxd, abs_of_pos (by positivity : (0 : ℝ) < ((n : ℝ) + 1) ^ 2), le_div_iff₀ (by positivity)]
      have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
      push_cast at hbin
      have hq : 0 ≤ d ^ 2 * ((n : ℝ) + 1) * ((n : ℝ) - 1) :=
        mul_nonneg (mul_nonneg (sq_nonneg d) (by positivity)) (by linarith)
      nlinarith [sq_nonneg d]
    have h0 := (continuous_abs.tendsto 0).comp hs.tendsto_atTop_zero
    rw [abs_zero] at h0
    have := ge_of_tendsto h0 (eventually_atTop.2 ⟨1, fun n hn => hb n hn⟩)
    nlinarith [sq_pos_of_pos hd0]
  · intro hx
    have hsum : Summable (fun k : ℕ => 1 / ((k : ℝ) + 1) ^ 2) := by
      have := (summable_nat_add_iff 1).2 (Real.summable_one_div_nat_pow.2 (by norm_num : 1 < 2))
      simpa using this
    refine Summable.of_norm_bounded hsum (fun k => ?_)
    rw [Real.norm_eq_abs, abs_div, abs_pow, abs_of_pos (by positivity : (0 : ℝ) < ((k : ℝ) + 1) ^ 2)]
    exact div_le_div_of_nonneg_right (pow_le_one₀ (abs_nonneg x) hx) (by positivity)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.sq_series_set
-- test: ra-sq-series-set-test.sage

/-- `∑ (-1)^k x^(2k)` converges exactly on `(-1, 1)`. -/
theorem alt_sq_set (x : ℝ) : Summable (fun k : ℕ => (-1 : ℝ) ^ k * x ^ (2 * k)) ↔ |x| < 1 := by
  constructor
  · intro hs
    by_contra h
    push_neg at h
    exact (inv_one_add_sq_series x).2 h hs
  · intro hx
    exact ((inv_one_add_sq_series x).1 hx).summable

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Series.alt_sq_set
-- test: ra-alt-sq-set-test.sage

/-! ## exm-values-geometric -/

/-- `∑_{n=1}^N 2^(-n) = 1 - 2^(-N)` exactly. -/
theorem half_partial (N : ℕ) : ∑ n ∈ Finset.Icc 1 N, (1 / 2 : ℝ) ^ n = 1 - (1 / 2) ^ N := by
  veriphy_certified
    (induction N with
      | zero => simp
      | succ N ih => rw [Finset.sum_Icc_succ_top (by omega), ih]; (try push_cast); ring1)
  -- certificate: ra-half-partial.sage

/-! ## exr-zeta-values -/

/-- A telescoping series with decreasing terms tending to `0`: `∑ (u_k - u_(k+1)) = u_0`. -/
theorem hasSum_telescope (u : ℕ → ℝ) (hu : Antitone u) (h0 : Tendsto u atTop (𝓝 0)) :
    HasSum (fun k => u k - u (k + 1)) (u 0) := by
  have hnn : ∀ k, 0 ≤ u k - u (k + 1) := fun k => sub_nonneg.2 (hu (Nat.le_succ k))
  rw [hasSum_iff_tendsto_nat_of_nonneg hnn]
  simp_rw [Finset.sum_range_sub']
  simpa using (tendsto_const_nhds (x := u 0)).sub h0

/-- `ζ(3) = 1.2020…`: the partial sums `S_N = ∑_{n ≤ N} 1/n³` satisfy `S_N ≤ ζ(3) ≤ S_N + 1/(2N²)`. -/
theorem zeta_three_bounds (N : ℕ) (hN : 1 ≤ N) :
    ∑ k ∈ range N, 1 / ((k : ℝ) + 1) ^ 3 ≤ ∑' k : ℕ, 1 / ((k : ℝ) + 1) ^ 3 ∧
      ∑' k : ℕ, 1 / ((k : ℝ) + 1) ^ 3 ≤ ∑ k ∈ range N, 1 / ((k : ℝ) + 1) ^ 3 + 1 / (2 * (N : ℝ) ^ 2) := by
  have hs : Summable (fun k : ℕ => 1 / ((k : ℝ) + 1) ^ 3) := by
    have := (summable_nat_add_iff 1).2 (Real.summable_one_div_nat_pow.2 (by norm_num : 1 < 3))
    simpa using this
  have hsplit := (hs.sum_add_tsum_nat_add N).symm
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  -- the tail is at most ∑_j ½ (u_j - u_(j+1)) = ½ u_0 with u_j = 1/((j+N)(j+N+1))
  set u : ℕ → ℝ := fun j => 1 / (((j : ℝ) + N) * ((j : ℝ) + N + 1)) with hu
  have hanti : Antitone u := fun a b hab => by
    simp only [hu]
    have : (a : ℝ) ≤ b := by exact_mod_cast hab
    apply one_div_le_one_div_of_le (by positivity)
    nlinarith
  have hu0 : Tendsto u atTop (𝓝 0) := by
    simp only [hu]
    apply Tendsto.div_atTop tendsto_const_nhds
    apply Tendsto.atTop_mul_atTop₀
    · exact tendsto_atTop_add_const_right _ _ tendsto_natCast_atTop_atTop
    · exact tendsto_atTop_add_const_right _ _ (tendsto_atTop_add_const_right _ _ tendsto_natCast_atTop_atTop)
  have htel := (hasSum_telescope u hanti hu0).mul_left (1 / 2)
  have htail := (hs.comp_injective (add_left_injective N)).hasSum
  have hle : ∑' j : ℕ, 1 / (((j + N : ℕ) : ℝ) + 1) ^ 3 ≤ 1 / 2 * u 0 := by
    refine hasSum_le (fun j => ?_) htail htel
    simp only [hu, Function.comp_apply]
    push_cast
    have hj : (0 : ℝ) < (j : ℝ) + N := by positivity
    rw [div_sub_div _ _ (by positivity) (by positivity)]
    rw [mul_div_assoc', div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith [sq_nonneg ((j : ℝ) + N), mul_pos hj hj]
  have htail0 : 0 ≤ ∑' j : ℕ, 1 / (((j + N : ℕ) : ℝ) + 1) ^ 3 := tsum_nonneg (fun j => by positivity)
  have hu0v : 1 / 2 * u 0 ≤ 1 / (2 * (N : ℝ) ^ 2) := by
    simp only [hu]
    push_cast
    rw [zero_add, one_div_mul_one_div, div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith
  simp only [Function.comp_def] at htail
  constructor
  · rw [hsplit]; linarith
  · rw [hsplit]; linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Series.zeta_three_bounds
-- test: ra-zeta-three-bounds-test.sage

/-! ## exr-exp-series-estimate -/

/-- `1/m! < 10⁻¹⁰` exactly when `m ≥ 14` (`1/13! = 1.6 · 10⁻¹⁰`, `1/14! = 1.1 · 10⁻¹¹`). -/
theorem exp_terms_needed (m : ℕ) : (1 : ℝ) / (m ! : ℝ) < 1 / 10 ^ 10 ↔ 14 ≤ m := by
  have hf : (0 : ℝ) < m ! := by positivity
  rw [div_lt_div_iff₀ hf (by positivity), one_mul, one_mul]
  constructor
  · intro h
    by_contra hm
    push_neg at hm
    have : m ! ≤ 13 ! := Nat.factorial_le (by omega)
    have h13 : (13 ! : ℝ) < 10 ^ 10 := by norm_num [Nat.factorial]
    have : (m ! : ℝ) ≤ 13 ! := by exact_mod_cast this
    linarith
  · intro hm
    have : 14 ! ≤ m ! := Nat.factorial_le hm
    have h14 : (10 : ℝ) ^ 10 < 14 ! := by norm_num [Nat.factorial]
    have : (14 ! : ℝ) ≤ m ! := by exact_mod_cast this
    linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_terms_needed (0) := by
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_terms_needed (13) := by
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_terms_needed (14) := by
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Series.exp_terms_needed (20) := by
  norm_num [Nat.factorial]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Series.exp_terms_needed
-- test: ra-exp-terms-needed-test.sage

end KaluzaKlein.RealAnalysis.Series
