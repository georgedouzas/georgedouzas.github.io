import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.PartialSums

/-- Closes the goals left by `convert` on a composed `HasDerivAt`: function equality (pointwise) and the derivative. -/
macro "fa_close1b" : tactic => `(tactic| first
  | rfl
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; done)
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; ring1)
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; field_simp; ring1)
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; field_simp; done)
  | (simp only [Function.comp_apply, id_eq, Pi.mul_apply, mul_one, one_mul]; ring1)
  | (simp only [Function.comp_apply, id_eq, Pi.mul_apply, mul_one, one_mul]; field_simp; ring1)
  | (simp only [Function.comp_apply, id_eq, Pi.mul_apply, mul_one, one_mul]; field_simp; done)
  | ring1
  | (field_simp; done)
  | (field_simp; ring1)
  | (norm_num; done)
  | (norm_num; ring1)
  | (norm_num; field_simp; ring1)
  | (simp; done)
  | (simp; ring1)
  | (ring_nf; done)
  | (field_simp; ring_nf; done))

/-! # Fourier analysis: convergence (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.Convergence

open Real Filter Topology

/-- The partial sum `S_N` of the square wave (as in `PartialSums`). -/
noncomputable def squarePartial (N : ℕ) (x : ℝ) : ℝ :=
  ∑ n ∈ Finset.range N, 2 * (1 - (-1) ^ (n + 1)) / (π * (n + 1)) * Real.sin ((n + 1) * x)

/-- The Dirichlet kernel `D_N` (as in `PartialSums`). -/
noncomputable def dirichlet (N : ℕ) (t : ℝ) : ℝ := 1 + 2 * ∑ n ∈ Finset.range N, Real.cos ((n + 1) * t)

/-! ## thm-fourier-convergence -/

/-- (ii) An absolutely summable Fourier series converges (uniformly, hence at every point) to the function. -/
theorem pointwise_of_summable {T : ℝ} [Fact (0 < T)] (f : C(AddCircle T, ℂ)) (h : Summable (fourierCoeff f))
    (x : AddCircle T) : HasSum (fun i => fourierCoeff f i • fourier i x) (f x) :=
  has_pointwise_sum_fourier_series_of_summable h x

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Convergence.pointwise_of_summable
-- test: fa-pointwise-of-summable-test.sage

/-- (iii) The Fourier series of a square-integrable function converges to it in `L²`. -/
theorem l2_convergence {T : ℝ} [hT : Fact (0 < T)] (f : MeasureTheory.Lp ℂ 2 (@AddCircle.haarAddCircle T hT)) :
    HasSum (fun i => fourierCoeff f i • fourierLp 2 i) f :=
  hasSum_fourier_series_L2 f

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Convergence.l2_convergence
-- test: fa-l2-convergence-test.sage

/-- (i) Dirichlet's theorem for the square wave, as stated (tested, not proved: Dirichlet's pointwise theorem for
piecewise `C¹` functions is not in mathlib): at every `x ∈ (0, π)` the partial sums tend to `sgn x = 1`. -/
def SquareWaveDirichlet : Prop :=
  ∀ x : ℝ, 0 < x → x < π → Tendsto (fun N => squarePartial N x) atTop (𝓝 1)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.Convergence.SquareWaveDirichlet
-- test: fa-square-wave-dirichlet-test.sage

/-! ## exm-fs-convergence-square -/

/-- At the jump `x = 0` every partial sum vanishes, the midpoint of the jump. -/
theorem square_partial_at_zero (N : ℕ) : squarePartial N 0 = 0 := by
  simp [squarePartial]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.Convergence.square_partial_at_zero (0) := by
  simp [KaluzaKlein.FourierAnalysis.Convergence.squarePartial]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.Convergence.square_partial_at_zero (4) := by
  simp [KaluzaKlein.FourierAnalysis.Convergence.squarePartial]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.Convergence.square_partial_at_zero (1) := by
  simp [KaluzaKlein.FourierAnalysis.Convergence.squarePartial]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.FourierAnalysis.Convergence.square_partial_at_zero
-- test: fa-square-partial-at-zero-test.sage

/-- At the jump `x = π` every partial sum vanishes. -/
theorem square_partial_at_pi (N : ℕ) : squarePartial N π = 0 := by
  simp only [squarePartial]
  apply Finset.sum_eq_zero
  intro n _
  rw [show ((n : ℝ) + 1) * π = ((n + 1 : ℕ) : ℝ) * π by push_cast; ring, Real.sin_nat_mul_pi, mul_zero]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Convergence.square_partial_at_pi
-- test: fa-square-partial-at-pi-test.sage

/-! ## exm-fs-convergence-abs, exr-fs-parseval-square -/

/-- `Σ_{k ≥ 0} a / (2k+1)² = a π² / 8`. -/
theorem odd_sq_sum (a : ℝ) : HasSum (fun k : ℕ => a / (2 * k + 1) ^ 2) (a * π ^ 2 / 8) := by
  set f : ℕ → ℝ := fun n => 1 / (n : ℝ) ^ 2 with hf
  have h : HasSum f (π ^ 2 / 6) := hasSum_zeta_two
  have he : HasSum (fun k : ℕ => f (2 * k)) (π ^ 2 / 24) := by
    have h4 := h.mul_left (1 / 4)
    have hfun : (fun k : ℕ => f (2 * k)) = fun k : ℕ => 1 / 4 * f k := by
      funext k
      simp only [hf]
      push_cast
      rw [mul_pow, one_div, one_div, one_div, mul_inv]
      norm_num
    rw [hfun]
    convert h4 using 1
    ring
  have hi : Function.Injective (fun k : ℕ => 2 * k + 1) := fun x y hxy => by
    simp only at hxy; omega
  have hs : Summable (fun k : ℕ => f (2 * k + 1)) := h.summable.comp_injective hi
  have hsum := HasSum.even_add_odd (f := f) he hs.hasSum
  have hu := h.unique hsum
  have hodd : HasSum (fun k : ℕ => f (2 * k + 1)) (π ^ 2 / 8) := by
    convert hs.hasSum using 1
    linarith
  convert hodd.mul_left a using 1
  · funext k
    simp only [hf]
    push_cast
    ring
  · ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Convergence.odd_sq_sum
-- test: fa-odd-sq-sum-test.sage

/-! ## exm-fs-convergence-du-bois -/

/-- The Lebesgue constants grow like `log N`, as stated (tested, not proved): `∫_{-π}^{π} |D_N| ≥ (8/π) log N`,
so the functionals `f ↦ S_N f(0)` are unbounded on continuous functions. -/
def LebesgueConstantGrowth : Prop :=
  ∀ N : ℕ, 1 ≤ N → 8 / π * Real.log N ≤ ∫ t in (-π)..π, |dirichlet N t|

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Convergence.LebesgueConstantGrowth
-- test: fa-lebesgue-constant-growth-test.sage

/-! ## exr-fs-leibniz -/

/-- Leibniz's series: `a Σ_{i<k} (-1)^i / (2i + 1) → a π/4`. -/
theorem leibniz (a : ℝ) :
    Tendsto (fun k => a * ∑ i ∈ Finset.range k, (-1 : ℝ) ^ i / (2 * i + 1)) atTop (𝓝 (a * π / 4)) := by
  have := Real.tendsto_sum_pi_div_four.const_mul a
  rwa [show a * (π / 4) = a * π / 4 by ring] at this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Convergence.leibniz
-- test: fa-leibniz-test.sage

/-- `sin ((2k+1) π / 2) = (-1)^k`. -/
theorem sin_odd_half_pi (k : ℕ) : Real.sin ((2 * k + 1) * π / 2) = (-1) ^ k := by
  rw [show (2 * (k : ℝ) + 1) * π / 2 = π / 2 + k * π by ring, Real.sin_add, Real.sin_pi_div_two,
    Real.cos_pi_div_two, Real.sin_nat_mul_pi, Real.cos_nat_mul_pi]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Convergence.sin_odd_half_pi
-- test: fa-sin-odd-half-pi-test.sage

/-- The Leibniz term `(-1)^i / (2i + 1)`. -/
noncomputable def leibnizTerm (i : ℕ) : ℝ := (-1) ^ i / (2 * (i : ℝ) + 1)

/-- At `x = π/2` the partial sums of the square wave are `4/π` times Leibniz's partial sums
`Σ_{i<K} (-1)^i/(2i+1)`. -/
theorem square_partial_half_pi (K : ℕ) :
    squarePartial (2 * K) (π / 2) = 4 / π * ∑ i ∈ Finset.range K, leibnizTerm i := by
  induction K with
  | zero => simp [squarePartial]
  | succ K ih =>
    have hπ : π ≠ 0 := Real.pi_ne_zero
    rw [show 2 * (K + 1) = 2 * K + 1 + 1 by ring]
    simp only [squarePartial] at ih ⊢
    rw [Finset.sum_range_succ, Finset.sum_range_succ, ih, Finset.sum_range_succ]
    have e1 : Real.sin ((((2 * K : ℕ) : ℝ) + 1) * (π / 2)) = (-1) ^ K := by
      rw [← sin_odd_half_pi K]; congr 1; push_cast; ring
    have e2 : (-1 : ℝ) ^ (2 * K + 1 + 1) = 1 := by
      rw [show 2 * K + 1 + 1 = 2 * (K + 1) by ring, pow_mul]; norm_num
    have e3 : (-1 : ℝ) ^ (2 * K + 1) = -1 := by rw [pow_succ, pow_mul]; norm_num
    rw [e1, e2, e3]
    simp only [leibnizTerm]
    push_cast
    field_simp
    ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Convergence.square_partial_half_pi
-- test: fa-square-partial-half-pi-test.sage

/-! ## exr-fs-riemann-lebesgue -/

/-- The integral of `sin λt` over `[c, d]`: the step-function case of the Riemann-Lebesgue lemma. -/
theorem step_integral (l c d : ℝ) (hl : l ≠ 0) :
    ∫ t in c..d, Real.sin (l * t) = (Real.cos (l * c) - Real.cos (l * d)) / l := by
  rw [intervalIntegral.integral_comp_mul_left (fun x => Real.sin x) hl, integral_sin, smul_eq_mul]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Convergence.step_integral
-- test: fa-step-integral-test.sage

/-- `|∫_c^d sin λt dt| ≤ 2/λ`, which tends to `0` as `λ → ∞`. -/
theorem step_bound (l : ℝ) (hl : 0 < l) (c d : ℝ) : |∫ t in c..d, Real.sin (l * t)| ≤ 2 / l := by
  rw [step_integral l c d hl.ne', abs_div, abs_of_pos hl]
  apply div_le_div_of_nonneg_right _ hl.le
  calc |Real.cos (l * c) - Real.cos (l * d)| ≤ |Real.cos (l * c)| + |Real.cos (l * d)| := abs_sub _ _
    _ ≤ 1 + 1 := add_le_add (Real.abs_cos_le_one _) (Real.abs_cos_le_one _)
    _ = 2 := by norm_num

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Convergence.step_bound
-- test: fa-step-bound-test.sage

/-- The Riemann-Lebesgue lemma on the line (mathlib): `∫ e^{-2πi v w} f(v) dv → 0` as `|w| → ∞`. -/
theorem riemann_lebesgue (f : ℝ → ℂ) :
    Tendsto (fun w : ℝ => ∫ v : ℝ, Real.fourierChar (-(w * v)) • f v) (cocompact ℝ) (𝓝 0) := by
  have := tendsto_integral_exp_inner_smul_cocompact (V := ℝ) (f := f)
  simpa using this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Convergence.riemann_lebesgue
-- test: fa-riemann-lebesgue-test.sage

/-! ## exr-fs-termwise-differentiation -/

/-- The derivative of a term of the series of `|x|` is the term of the square wave. -/
theorem abs_series_term_deriv (k : ℕ) (x : ℝ) :
    HasDerivAt (fun x => -(4 / π) * Real.cos ((2 * k + 1) * x) / (2 * k + 1) ^ 2)
      (4 / π * Real.sin ((2 * k + 1) * x) / (2 * k + 1)) x := by
  have hk : (2 * (k : ℝ) + 1) ≠ 0 := by positivity
  have hπ : π ≠ 0 := Real.pi_ne_zero
  have := ((((hasDerivAt_id' x).const_mul (2 * (k : ℝ) + 1)).cos).const_mul (-(4 / π))).div_const ((2 * (k : ℝ) + 1) ^ 2)
  convert this using 1
  all_goals fa_close1b

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Convergence.abs_series_term_deriv
-- test: fa-abs-series-term-deriv-test.sage

/-- The terms `cos n x` of the differentiated sawtooth series do not tend to `0`, at any `x`. -/
theorem cos_mul_not_tendsto_zero (x : ℝ) : ¬ Tendsto (fun n : ℕ => Real.cos (n * x)) atTop (𝓝 0) := by
  intro h
  have hm : Tendsto (fun n : ℕ => 2 * n) atTop atTop := Filter.tendsto_atTop_atTop.2 fun b => ⟨b, fun a ha => by omega⟩
  have h2 : Tendsto (fun n : ℕ => Real.cos (((2 * n : ℕ) : ℝ) * x)) atTop (𝓝 0) := h.comp hm
  have h3 : Tendsto (fun n : ℕ => 2 * Real.cos (n * x) ^ 2 - 1) atTop (𝓝 (2 * 0 ^ 2 - 1)) :=
    ((h.pow 2).const_mul 2).sub_const 1
  have heq : (fun n : ℕ => Real.cos (((2 * n : ℕ) : ℝ) * x)) = fun n : ℕ => 2 * Real.cos (n * x) ^ 2 - 1 := by
    funext n; push_cast; rw [show 2 * (n : ℝ) * x = 2 * (n * x) by ring, Real.cos_two_mul]
  rw [heq] at h2
  have := tendsto_nhds_unique h2 h3
  norm_num at this

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.Convergence.cos_mul_not_tendsto_zero
-- test: fa-cos-mul-not-tendsto-zero-test.sage

end KaluzaKlein.FourierAnalysis.Convergence
