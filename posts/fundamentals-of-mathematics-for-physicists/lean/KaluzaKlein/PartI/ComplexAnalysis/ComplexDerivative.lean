import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "The complex derivative"

The claims of the section "The complex derivative" of `tutorial/part-i-04-complex-analysis.qmd`: the sum, product,
quotient and chain rules, `(zⁿ)' = n zⁿ⁻¹`, `(eᶻ)' = eᶻ`, `(cos z)' = -sin z`, `(sin z)' = cos z`; the difference
quotients of `z²`, `z³`, `1/z`, `z̄`, `|z|² = z z̄` and `eᶻ`; the failure of complex differentiability of `z̄`
(everywhere), `Re z` (everywhere) and `|z|²` (away from `0`); and the derivative of `(z - 1)/(z² + 1)` with the
zeros `1 ± √2` of its numerator.

A function is complex differentiable at `z` exactly when its real derivative `L` satisfies `L i = i L 1`
(mathlib's `differentiableAt_complex_iff_differentiableAt_real`): this is the test used for `z̄`, `Re z`, `|z|²`.
-/

namespace KaluzaKlein.ComplexAnalysis.ComplexDerivative

open Complex Filter Topology ComplexConjugate

/-! ## The test for complex differentiability -/

/-- A function with real derivative `L` at `z` is complex differentiable there iff `L i = i · L 1`. -/
theorem differentiableAt_iff_of_hasFDerivAt {f : ℂ → ℂ} {z : ℂ} {L : ℂ →L[ℝ] ℂ} (hL : HasFDerivAt f L z) :
    DifferentiableAt ℂ f z ↔ L I = I * L 1 := by
  rw [differentiableAt_complex_iff_differentiableAt_real, hL.fderiv, smul_eq_mul]
  exact ⟨fun h => h.2, fun h => ⟨hL.differentiableAt, h⟩⟩

/-! ## def-complex-derivative -/

/-- The sum, product and quotient rules for the complex derivative. -/
theorem rules (f g : ℂ → ℂ) (f' g' z : ℂ) (hf : HasDerivAt f f' z) (hg : HasDerivAt g g' z) :
    HasDerivAt (fun w => f w + g w) (f' + g') z ∧
      HasDerivAt (fun w => f w * g w) (f' * g z + f z * g') z ∧
      (g z ≠ 0 → HasDerivAt (fun w => f w / g w) ((f' * g z - f z * g') / g z ^ 2) z) :=
  ⟨hf.add hg, hf.mul hg, fun h => hf.div hg h⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.rules ((fun w : ℂ => w)) ((fun _ : ℂ => (2 : ℂ))) (1) (0) (0) ((hasDerivAt_id' 0)) ((hasDerivAt_const 0 (2 : ℂ))) := by
  exact ⟨(hasDerivAt_id' _).add (hasDerivAt_const _ _), (hasDerivAt_id' _).mul (hasDerivAt_const _ _), fun h => (hasDerivAt_id' _).div (hasDerivAt_const _ _) h⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.rules ((fun w : ℂ => w)) ((fun _ : ℂ => (2 : ℂ))) (1) (0) (1) ((hasDerivAt_id' 1)) ((hasDerivAt_const 1 (2 : ℂ))) := by
  exact ⟨(hasDerivAt_id' _).add (hasDerivAt_const _ _), (hasDerivAt_id' _).mul (hasDerivAt_const _ _), fun h => (hasDerivAt_id' _).div (hasDerivAt_const _ _) h⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.rules ((fun w : ℂ => w)) ((fun w : ℂ => w)) (1) (1) ((1 + Complex.I)) ((hasDerivAt_id' (1 + Complex.I))) ((hasDerivAt_id' (1 + Complex.I))) := by
  exact ⟨(hasDerivAt_id' _).add (hasDerivAt_id' _), (hasDerivAt_id' _).mul (hasDerivAt_id' _), fun h => (hasDerivAt_id' _).div (hasDerivAt_id' _) h⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.rules
-- test: ca-holo-rules-test.sage

/-- The chain rule for the complex derivative. -/
theorem chain (f g : ℂ → ℂ) (f' g' z : ℂ) (hf : HasDerivAt f f' z) (hg : HasDerivAt g g' (f z)) :
    HasDerivAt (fun w => g (f w)) (g' * f') z :=
  hg.comp z hf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.chain ((fun w : ℂ => w)) ((fun w : ℂ => w)) ((1 : ℂ)) ((1 : ℂ)) ((0 : ℂ)) ((hasDerivAt_id' (0 : ℂ))) ((hasDerivAt_id' (0 : ℂ))) := by
  have h := HasDerivAt.comp (h₂ := fun w : ℂ => w) (h := fun w : ℂ => w) (0 : ℂ) (hasDerivAt_id' (0 : ℂ)) (hasDerivAt_id' (0 : ℂ))
  exact h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.chain ((fun w : ℂ => w + 1)) ((fun w : ℂ => 2 * w)) ((1 : ℂ)) ((2 * 1 : ℂ)) ((1 : ℂ)) (((hasDerivAt_id' (1 : ℂ)).add_const 1)) (((hasDerivAt_id' ((1 : ℂ) + 1)).const_mul 2)) := by
  have h := HasDerivAt.comp (h₂ := fun w : ℂ => 2 * w) (h := fun w : ℂ => w + 1) (1 : ℂ) ((hasDerivAt_id' ((1 : ℂ) + 1)).const_mul 2) ((hasDerivAt_id' (1 : ℂ)).add_const 1)
  exact h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.chain ((fun w : ℂ => w + 1)) ((fun w : ℂ => 2 * w)) ((1 : ℂ)) ((2 * 1 : ℂ)) (Complex.I) (((hasDerivAt_id' Complex.I).add_const 1)) (((hasDerivAt_id' (Complex.I + 1)).const_mul 2)) := by
  have h := HasDerivAt.comp (h₂ := fun w : ℂ => 2 * w) (h := fun w : ℂ => w + 1) Complex.I ((hasDerivAt_id' (Complex.I + 1)).const_mul 2) ((hasDerivAt_id' Complex.I).add_const 1)
  exact h

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.chain
-- test: ca-holo-chain-test.sage

/-- `(zⁿ)' = n zⁿ⁻¹`. -/
theorem deriv_pow (n : ℕ) (z : ℂ) : deriv (fun w : ℂ => w ^ n) z = n * z ^ (n - 1) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow (0) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow (2) ((1 + Complex.I)) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow (3) (Complex.I) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow
-- test: ca-holo-deriv-pow-test.sage

/-- `(eᶻ)' = eᶻ`. -/
theorem deriv_exp (z : ℂ) : deriv Complex.exp z = Complex.exp z := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_exp (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_exp (1) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_exp (Complex.I) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_exp
-- test: ca-holo-deriv-exp-test.sage

/-! ## exm-derivative-z2 -/

/-- `((z + h)² - z²)/h = 2z + h`. -/
theorem diffQuot_sq (z h : ℂ) (hh : h ≠ 0) : ((z + h) ^ 2 - z ^ 2) / h = 2 * z + h := by
  rw [div_eq_iff hh]
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `(z²)' = 2z`. -/
theorem deriv_sq (z : ℂ) : deriv (fun w : ℂ => w ^ 2) z = 2 * z := by
  rw [deriv_pow]
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_sq (0) := by
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow]
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_sq ((1 + Complex.I)) := by
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow]
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_sq (Complex.I) := by
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow]
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_sq
-- test: ca-holo-deriv-sq-test.sage

/-! ## exm-derivative-conjugate -/

/-- The difference quotient of `z̄` is `h̄/h`. -/
theorem conj_diffQuot (z h : ℂ) (hh : h ≠ 0) : (conj (z + h) - conj z) / h = conj h / h := by
  rw [map_add, div_eq_div_iff hh hh]
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `h̄/h = 1` for real `h = t ≠ 0` and `h̄/h = -1` for imaginary `h = t i`. -/
theorem conj_quot_axes (t : ℝ) (ht : t ≠ 0) : conj (t : ℂ) / t = 1 ∧ conj ((t : ℂ) * I) / (t * I) = -1 := by
  have ht' : (t : ℂ) ≠ 0 := ofReal_ne_zero.mpr ht
  have hI : I ^ 2 = -1 := I_sq
  refine ⟨?_, ?_⟩
  · rw [conj_ofReal, div_eq_iff ht']
    generalize ((t : ℝ) : ℂ) = tC at *
    generalize Complex.I = i at *
    rw [← sub_eq_zero]
    veriphy_certified (ring1)
    -- certificate: ca-holo-ring.sage
  · rw [map_mul, conj_ofReal, conj_I, div_eq_iff (mul_ne_zero ht' I_ne_zero)]
    generalize ((t : ℝ) : ℂ) = tC at *
    generalize Complex.I = i at *
    rw [← sub_eq_zero]
    veriphy_certified (ring1)
    -- certificate: ca-holo-ring.sage

/-- Conjugation is real differentiable (a reflection of `ℝ²`) but complex differentiable nowhere. -/
theorem conj_not_differentiableAt (z : ℂ) :
    DifferentiableAt ℝ (fun w : ℂ => conj w) z ∧ ¬ DifferentiableAt ℂ (fun w : ℂ => conj w) z := by
  have hL : HasFDerivAt (fun w : ℂ => conj w) (conjCLE : ℂ →L[ℝ] ℂ) z := conjCLE.hasFDerivAt
  refine ⟨hL.differentiableAt, ?_⟩
  rw [differentiableAt_iff_of_hasFDerivAt hL]
  norm_num [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.conj_not_differentiableAt (0) := by
  have hL : HasFDerivAt (fun w : ℂ => (starRingEnd ℂ) w) (Complex.conjCLE : ℂ →L[ℝ] ℂ) (0) := Complex.conjCLE.hasFDerivAt
  refine ⟨hL.differentiableAt, ?_⟩
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt hL]
  norm_num [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.conj_not_differentiableAt (1) := by
  have hL : HasFDerivAt (fun w : ℂ => (starRingEnd ℂ) w) (Complex.conjCLE : ℂ →L[ℝ] ℂ) (1) := Complex.conjCLE.hasFDerivAt
  refine ⟨hL.differentiableAt, ?_⟩
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt hL]
  norm_num [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.conj_not_differentiableAt (Complex.I) := by
  have hL : HasFDerivAt (fun w : ℂ => (starRingEnd ℂ) w) (Complex.conjCLE : ℂ →L[ℝ] ℂ) (Complex.I) := Complex.conjCLE.hasFDerivAt
  refine ⟨hL.differentiableAt, ?_⟩
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt hL]
  norm_num [Complex.ext_iff]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.conj_not_differentiableAt
-- test: ca-holo-conj-test.sage

/-! ## exm-derivative-exponential -/

/-- `e^{z + h} - e^z = e^z (e^h - 1)`. -/
theorem exp_diff (z h : ℂ) : Complex.exp (z + h) - Complex.exp z = Complex.exp z * (Complex.exp h - 1) := by
  rw [Complex.exp_add]
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `(cos z)' = -sin z` and `(sin z)' = cos z`. -/
theorem deriv_cos_sin (z : ℂ) : deriv Complex.cos z = -Complex.sin z ∧ deriv Complex.sin z = Complex.cos z := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_cos_sin (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_cos_sin (1) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_cos_sin (Complex.I) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_cos_sin
-- test: ca-holo-deriv-trig-test.sage

/-! ## exr-derivative-z3 -/

/-- `((z + h)³ - z³)/h = 3z² + 3zh + h²`. -/
theorem diffQuot_cube (z h : ℂ) (hh : h ≠ 0) : ((z + h) ^ 3 - z ^ 3) / h = 3 * z ^ 2 + 3 * z * h + h ^ 2 := by
  rw [div_eq_iff hh]
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `(1/(z + h) - 1/z)/h = -1/(z (z + h))`. -/
theorem diffQuot_inv (z h : ℂ) (hz : z ≠ 0) (hzh : z + h ≠ 0) (hh : h ≠ 0) :
    (1 / (z + h) - 1 / z) / h = -1 / (z * (z + h)) := by
  rw [div_sub_div _ _ hzh hz, div_div, div_eq_div_iff (mul_ne_zero (mul_ne_zero hzh hz) hh) (mul_ne_zero hz hzh)]
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `(z³)' = 3z²`. -/
theorem deriv_cube (z : ℂ) : deriv (fun w : ℂ => w ^ 3) z = 3 * z ^ 2 := by
  rw [deriv_pow]
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_cube (0) := by
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow]
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_cube ((1 + Complex.I)) := by
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow]
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_cube (Complex.I) := by
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_pow]
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_cube
-- test: ca-holo-deriv-cube-test.sage

/-- `w ↦ 1/w`. -/
noncomputable def recip (w : ℂ) : ℂ := 1 / w

/-- `(1/z)' = -1/z² = -(1/z)²` for `z ≠ 0`. -/
theorem deriv_inv (z : ℂ) (hz : z ≠ 0) : deriv recip z = -(recip z) ^ 2 := by
  show deriv (fun w : ℂ => 1 / w) z = -(1 / z) ^ 2
  simp only [one_div]
  rw [_root_.deriv_inv]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_inv (1) (one_ne_zero) := by
  have hz := (one_ne_zero : (1 : ℂ) ≠ 0)
  show deriv (fun w : ℂ => 1 / w) (1 : ℂ) = -(1 / (1 : ℂ)) ^ 2
  simp only [one_div]
  rw [_root_.deriv_inv]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_inv (Complex.I) (Complex.I_ne_zero) := by
  have hz := (Complex.I_ne_zero)
  show deriv (fun w : ℂ => 1 / w) (Complex.I) = -(1 / (Complex.I)) ^ 2
  simp only [one_div]
  rw [_root_.deriv_inv]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_inv ((-2)) ((by norm_num)) := by
  have hz := (by norm_num : (-2 : ℂ) ≠ 0)
  show deriv (fun w : ℂ => 1 / w) (-2 : ℂ) = -(1 / (-2 : ℂ)) ^ 2
  simp only [one_div]
  rw [_root_.deriv_inv]
  ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_inv
-- test: ca-holo-deriv-inv-test.sage

/-! ## exr-derivative-conj-fails -/

/-- The difference quotient of `|z|² = z̄ z` is `z̄ + h̄ + z h̄/h`. -/
theorem normSq_diffQuot (z h : ℂ) (hh : h ≠ 0) :
    (conj (z + h) * (z + h) - conj z * z) / h = conj z + conj h + z * conj h / h := by
  rw [map_add, add_div' _ _ _ hh, div_eq_div_iff hh hh]
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- The real derivative of `w ↦ w w̄` at `z` is `h ↦ z h̄ + z̄ h`. -/
theorem hasFDerivAt_normSq (z : ℂ) :
    HasFDerivAt (fun w : ℂ => w * conj w)
      (z • (conjCLE : ℂ →L[ℝ] ℂ) + conj z • ContinuousLinearMap.id ℝ ℂ) z :=
  (hasFDerivAt_id (𝕜 := ℝ) z).mul (conjCLE.hasFDerivAt (x := z))

/-- `|z|² = z z̄` has complex derivative `0` at `0`. -/
theorem normSq_hasDerivAt_zero : HasDerivAt (fun w : ℂ => w * conj w) 0 0 := by
  rw [hasDerivAt_iff_isLittleO_nhds_zero]
  simp only [zero_add, map_zero, mul_zero, sub_zero, smul_zero]
  refine (Asymptotics.IsBigO.of_bound 1 (Eventually.of_forall fun h => ?_)).trans_isLittleO
    (Asymptotics.isLittleO_pow_id (𝕜 := ℂ) one_lt_two)
  simp [norm_mul, sq]

/-- `|z|² = z z̄` is complex differentiable at `z` exactly when `z = 0`, with derivative `0` there. -/
theorem normSq_differentiableAt_iff (z : ℂ) :
    (DifferentiableAt ℂ (fun w : ℂ => w * conj w) z ↔ z = 0) ∧ HasDerivAt (fun w : ℂ => w * conj w) 0 0 := by
  refine ⟨?_, normSq_hasDerivAt_zero⟩
  rw [differentiableAt_iff_of_hasFDerivAt (hasFDerivAt_normSq z)]
  simp only [ContinuousLinearMap.add_apply, ContinuousLinearMap.coe_smul', Pi.smul_apply,
    ContinuousLinearEquiv.coe_coe, conjCLE_apply, ContinuousLinearMap.coe_id', id_eq, conj_I, map_one,
    smul_eq_mul]
  constructor
  · intro h
    have h2 : (2 * I) * z = 0 := by linear_combination -h
    simpa [I_ne_zero] using h2
  · rintro rfl
    simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff (0) := by
  refine ⟨?_, KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_hasDerivAt_zero⟩
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt (KaluzaKlein.ComplexAnalysis.ComplexDerivative.hasFDerivAt_normSq (0))]
  simp [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff (1) := by
  refine ⟨?_, KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_hasDerivAt_zero⟩
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt (KaluzaKlein.ComplexAnalysis.ComplexDerivative.hasFDerivAt_normSq (1))]
  simp [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff (Complex.I) := by
  refine ⟨?_, KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_hasDerivAt_zero⟩
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt (KaluzaKlein.ComplexAnalysis.ComplexDerivative.hasFDerivAt_normSq (Complex.I))]
  simp [Complex.ext_iff]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff
-- test: ca-holo-normsq-test.sage

/-- The quotient of `Re`: `Re(h)/h` is `1` along the real axis and `0` along the imaginary axis. -/
theorem re_quot_axes (t : ℝ) (ht : t ≠ 0) :
    ((t : ℂ).re : ℂ) / t = 1 ∧ (((t : ℂ) * I).re : ℂ) / (t * I) = 0 := by
  have ht' : (t : ℂ) ≠ 0 := ofReal_ne_zero.mpr ht
  simp [ht']

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_quot_axes (1) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_quot_axes (2) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_quot_axes ((-1 / 2)) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_quot_axes
-- test: ca-holo-re-quot-test.sage

/-- `Re z` is complex differentiable nowhere. -/
theorem re_not_differentiableAt (z : ℂ) : ¬ DifferentiableAt ℂ (fun w : ℂ => (w.re : ℂ)) z := by
  have hL : HasFDerivAt (fun w : ℂ => (w.re : ℂ)) (ofRealCLM.comp reCLM) z := (ofRealCLM.comp reCLM).hasFDerivAt
  rw [differentiableAt_iff_of_hasFDerivAt hL]
  simp [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_not_differentiableAt (0) := by
  have hL : HasFDerivAt (fun w : ℂ => (w.re : ℂ)) (Complex.ofRealCLM.comp Complex.reCLM) (0) := (Complex.ofRealCLM.comp Complex.reCLM).hasFDerivAt
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt hL]
  simp [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_not_differentiableAt (1) := by
  have hL : HasFDerivAt (fun w : ℂ => (w.re : ℂ)) (Complex.ofRealCLM.comp Complex.reCLM) (1) := (Complex.ofRealCLM.comp Complex.reCLM).hasFDerivAt
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt hL]
  simp [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_not_differentiableAt (Complex.I) := by
  have hL : HasFDerivAt (fun w : ℂ => (w.re : ℂ)) (Complex.ofRealCLM.comp Complex.reCLM) (Complex.I) := (Complex.ofRealCLM.comp Complex.reCLM).hasFDerivAt
  rw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt hL]
  simp [Complex.ext_iff]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_not_differentiableAt
-- test: ca-holo-re-test.sage

/-! ## exr-derivative-rules -/

/-- `f(z) = (z - 1)/(z² + 1)`, the function of the figure. -/
noncomputable def ratFun (w : ℂ) : ℂ := (w - 1) / (w ^ 2 + 1)

/-- `(-z² + 2z + 1)/(z² + 1)²`. -/
noncomputable def ratDeriv (w : ℂ) : ℂ := (-w ^ 2 + 2 * w + 1) / (w ^ 2 + 1) ^ 2

/-- The quotient rule: `((z - 1)/(z² + 1))' = (-z² + 2z + 1)/(z² + 1)²`. -/
theorem deriv_rational (z : ℂ) (hz : z ^ 2 + 1 ≠ 0) : deriv ratFun z = ratDeriv z := by
  show deriv (fun w : ℂ => (w - 1) / (w ^ 2 + 1)) z = (-z ^ 2 + 2 * z + 1) / (z ^ 2 + 1) ^ 2
  have h := ((hasDerivAt_id' z).sub_const 1).div ((hasDerivAt_pow 2 z).add_const 1) hz
  refine h.deriv.trans ?_
  simp only [show (2 : ℕ) - 1 = 1 from rfl, pow_one, Nat.cast_ofNat]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_rational (0) ((by norm_num [Complex.ext_iff, pow_two])) := by
  have hz := (by norm_num [Complex.ext_iff, pow_two] : (0 : ℂ) ^ 2 + 1 ≠ 0)
  show deriv (fun w : ℂ => (w - 1) / (w ^ 2 + 1)) (0 : ℂ) = (-(0 : ℂ) ^ 2 + 2 * (0 : ℂ) + 1) / ((0 : ℂ) ^ 2 + 1) ^ 2
  have h := ((hasDerivAt_id' (0 : ℂ)).sub_const 1).div ((hasDerivAt_pow 2 (0 : ℂ)).add_const 1) hz
  refine h.deriv.trans ?_
  simp only [show (2 : ℕ) - 1 = 1 from rfl, pow_one, Nat.cast_ofNat]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_rational (1) ((by norm_num [Complex.ext_iff, pow_two])) := by
  have hz := (by norm_num [Complex.ext_iff, pow_two] : (1 : ℂ) ^ 2 + 1 ≠ 0)
  show deriv (fun w : ℂ => (w - 1) / (w ^ 2 + 1)) (1 : ℂ) = (-(1 : ℂ) ^ 2 + 2 * (1 : ℂ) + 1) / ((1 : ℂ) ^ 2 + 1) ^ 2
  have h := ((hasDerivAt_id' (1 : ℂ)).sub_const 1).div ((hasDerivAt_pow 2 (1 : ℂ)).add_const 1) hz
  refine h.deriv.trans ?_
  simp only [show (2 : ℕ) - 1 = 1 from rfl, pow_one, Nat.cast_ofNat]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_rational ((2 * Complex.I)) ((by norm_num [Complex.ext_iff, pow_two])) := by
  have hz := (by norm_num [Complex.ext_iff, pow_two] : (2 * Complex.I : ℂ) ^ 2 + 1 ≠ 0)
  show deriv (fun w : ℂ => (w - 1) / (w ^ 2 + 1)) (2 * Complex.I) = (-(2 * Complex.I) ^ 2 + 2 * (2 * Complex.I) + 1) / ((2 * Complex.I) ^ 2 + 1) ^ 2
  have h := ((hasDerivAt_id' (2 * Complex.I)).sub_const 1).div ((hasDerivAt_pow 2 (2 * Complex.I)).add_const 1) hz
  refine h.deriv.trans ?_
  simp only [show (2 : ℕ) - 1 = 1 from rfl, pow_one, Nat.cast_ofNat]
  ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_rational
-- test: ca-holo-deriv-rational-test.sage

/-- The numerator `-z² + 2z + 1` vanishes at `1 + s` and `1 - s` for `s² = 2`, that is at `1 ± √2`. -/
theorem numerator_roots (s : ℂ) (hs : s ^ 2 = 2) :
    -(1 + s) ^ 2 + 2 * (1 + s) + 1 = 0 ∧ -(1 - s) ^ 2 + 2 * (1 - s) + 1 = 0 := by
  refine ⟨?_, ?_⟩
  · rw [← sub_eq_zero]
    veriphy_certified (linear_combination (-1) * hs)
    -- certificate: ca-holo-ring.sage
  · rw [← sub_eq_zero]
    veriphy_certified (linear_combination (-1) * hs)
    -- certificate: ca-holo-ring.sage

/-- The numerator factors as `-(z - (1 + s))(z - (1 - s))` with `s² = 2`: its only zeros are `1 ± √2`. -/
theorem numerator_factor (z s : ℂ) (hs : s ^ 2 = 2) :
    -z ^ 2 + 2 * z + 1 = -((z - (1 + s)) * (z - (1 - s))) := by
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-1) * hs)
  -- certificate: ca-holo-ring.sage

end KaluzaKlein.ComplexAnalysis.ComplexDerivative
