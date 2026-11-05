import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.ComplexDerivative
import KaluzaKlein.PartI.ComplexAnalysis.HolomorphicFunctions
import KaluzaKlein.PartI.ComplexAnalysis.CauchyRiemann
import KaluzaKlein.PartI.ComplexAnalysis.HarmonicFunctions
import KaluzaKlein.PartI.ComplexAnalysis.ConformalMaps
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Worked computations" (holomorphic functions)

The claims of the worked computations closing "Holomorphic functions" in `tutorial/part-i-04-complex-analysis.qmd`:
the zero and poles of `(z - 1)/(z² + 1)`; the Cauchy-Riemann check for `z³` (`f' = 3z²`) and for `|z|²` (only at
`0`); the columns of the Jacobian of `z²` (orthogonal, of equal length `2|z|`); the argument principle read off the
domain colouring (the logarithmic derivative `1/(z - 1) - 1/(z - i) - 1/(z + i)` winds `+1` around the zero and `-1`
around each pole); the Jacobian of `eᶻ` (`eˣ` times the rotation by `y`); the harmonicity of `uv` and `u² - v²`; and
the derivative of an inverse, applied to `Log` as the inverse of `eᶻ` on the strip `|Im z| < π`.
-/

namespace KaluzaKlein.ComplexAnalysis.HolomorphicWorked

open Complex Filter Topology ComplexConjugate InnerProductSpace Metric
open scoped Real

/-! ## prp-cauchy-riemann-examples -/

/-- `(z - 1)/(z² + 1)` vanishes at `z = 1`. -/
theorem rational_zero : ((1 : ℂ) - 1) / (1 ^ 2 + 1) = 0 := by
  rw [div_eq_zero_iff]
  left
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- Its only zero is `1`: off the poles `±i`, `(z - 1)/(z² + 1) = 0` iff `z = 1`. -/
theorem rational_eq_zero_iff (z : ℂ) (h : z ^ 2 + 1 ≠ 0) : (z - 1) / (z ^ 2 + 1) = 0 ↔ z = 1 := by
  rw [div_eq_zero_iff, sub_eq_zero]
  tauto

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.rational_eq_zero_iff (1) ((by norm_num)) := by
  have h : ((1 : ℂ)) ^ 2 + 1 ≠ 0 := by norm_num [Complex.ext_iff, pow_two]
  rw [div_eq_zero_iff, sub_eq_zero]
  tauto

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.rational_eq_zero_iff (0) ((by norm_num)) := by
  have h : ((0 : ℂ)) ^ 2 + 1 ≠ 0 := by norm_num [Complex.ext_iff, pow_two]
  rw [div_eq_zero_iff, sub_eq_zero]
  tauto

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.rational_eq_zero_iff (2) ((by norm_num)) := by
  have h : ((2 : ℂ)) ^ 2 + 1 ≠ 0 := by norm_num [Complex.ext_iff, pow_two]
  rw [div_eq_zero_iff, sub_eq_zero]
  tauto

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicWorked.rational_eq_zero_iff
-- test: ca-holo-rational-zero-test.sage

/-- The columns `(2x, 2y)` and `(-2y, 2x)` of the Jacobian of `z²` are orthogonal and of equal length. -/
theorem sq_jac_columns (x y : ℝ) :
    (2 * x) * (-2 * y) + (2 * y) * (2 * x) = 0 ∧ (2 * x) ^ 2 + (2 * y) ^ 2 = (-2 * y) ^ 2 + (2 * x) ^ 2 := by
  refine ⟨?_, ?_⟩
  · rw [← sub_eq_zero]
    veriphy_certified (ring1)
    -- certificate: ca-holo-ring.sage
  · rw [← sub_eq_zero]
    veriphy_certified (ring1)
    -- certificate: ca-holo-ring.sage

/-! ## exm-cr-check-method -/

/-- `u_x + i v_x = 3(x² - y²) + 6ixy = 3z²`. -/
theorem cube_deriv_parts (x y : ℝ) :
    ((3 * (x ^ 2 - y ^ 2) : ℝ) : ℂ) + ((6 * x * y : ℝ) : ℂ) * I = 3 * ((x : ℂ) + y * I) ^ 2 := by
  have hI : I ^ 2 = -1 := I_sq
  have e1 : ((3 * (x ^ 2 - y ^ 2) : ℝ) : ℂ) = 3 * ((x : ℂ) ^ 2 - (y : ℂ) ^ 2) := by push_cast <;> ring
  have e2 : ((6 * x * y : ℝ) : ℂ) = 6 * (x : ℂ) * (y : ℂ) := by push_cast <;> ring
  rw [e1, e2]
  clear e1 e2
  generalize ((x : ℝ) : ℂ) = xC at *
  generalize ((y : ℝ) : ℂ) = yC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-3 * yC ^ 2) * hI)
  -- certificate: ca-holo-ring.sage

/-- `(z³)' = 3z²`. -/
theorem cube_deriv (z : ℂ) : deriv (fun w : ℂ => w ^ 3) z = 3 * z ^ 2 :=
  ComplexDerivative.deriv_cube z

/-- For `|z|² = x² + y²` (`v = 0`): `v_x = v_y = 0`. -/
theorem zero_partial (y : ℝ) : deriv (fun _ : ℝ => (0 : ℝ)) y = 0 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => 0) ((0 : ℝ)) (y) :=
        by
        convert (hasDerivAt_const (y : ℝ) (0 : ℝ)) using 1
        all_goals rfl
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- For `|z|²` the equations `u_x = v_y`, `u_y = -v_x` read `2x = 0`, `2y = 0`: they hold only at `z = 0`. -/
theorem normSq_cr_iff (x y : ℝ) : (2 * x = 0 ∧ 2 * y = -0) ↔ (x = 0 ∧ y = 0) := by
  constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.normSq_cr_iff (0) (0) := by
  constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.normSq_cr_iff (1) (-5) := by
  constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.normSq_cr_iff (0) (1) := by
  constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.normSq_cr_iff (0) (0) := by
  constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.HolomorphicWorked.normSq_cr_iff
-- test: ca-holo-normsq-cr-test.sage

/-! ## exm-cr-jacobian-structure -/

/-- The common length of the columns is `2|z|`: `(2x)² + (2y)² = (2|z|)²`. -/
theorem sq_jac_column_length (x y : ℝ) : (2 * x) ^ 2 + (2 * y) ^ 2 = (2 * ‖(x : ℂ) + y * I‖) ^ 2 := by
  rw [mul_pow (2 : ℝ) ‖_‖, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
  simp
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.sq_jac_column_length (0) (0) := by
  rw [mul_pow (2 : ℝ) ‖_‖, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
  simp
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.sq_jac_column_length (1) (-5) := by
  rw [mul_pow (2 : ℝ) ‖_‖, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
  simp
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.sq_jac_column_length (0) (1) := by
  rw [mul_pow (2 : ℝ) ‖_‖, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
  simp
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.sq_jac_column_length (0) (0) := by
  rw [mul_pow (2 : ℝ) ‖_‖, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
  simp
  try ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.HolomorphicWorked.sq_jac_column_length
-- test: ca-holo-column-length-test.sage

/-! ## exm-cr-domain-colouring-reading -/

/-- `f'/f = 1/(z - 1) - 1/(z - i) - 1/(z + i)` for `f = (z - 1)/(z² + 1)` (cleared of denominators, with
`f' = (-z² + 2z + 1)/(z² + 1)²`): the logarithmic derivative has residue `+1` at the zero and `-1` at each pole. -/
theorem log_deriv_partial (z : ℂ) :
    (-z ^ 2 + 2 * z + 1) * ((z - 1) * (z - I) * (z + I)) =
      ((z - I) * (z + I) - (z - 1) * (z + I) - (z - 1) * (z - I)) * ((z ^ 2 + 1) * (z - 1)) := by
  have hI : I ^ 2 = -1 := I_sq
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (2 * z ^ 3 - 4 * z ^ 2 + 2 * z) * hI)
  -- certificate: ca-holo-ring.sage

/-- Auxiliary form of `circleIntegral_sub_inv` (its proof); the claim below restates it. -/
theorem circleIntegral_sub_inv_aux (a c : ℂ) (r : ℝ) (h0 : 0 < r) :
    (dist a c < r → (∮ z in C(c, r), (z - a)⁻¹) = 2 * π * I) ∧
      (r < dist a c → (∮ z in C(c, r), (z - a)⁻¹) = 0) := by
  refine ⟨fun h => circleIntegral.integral_sub_inv_of_mem_ball h, fun h => ?_⟩
  have hd : DiffContOnCl ℂ (fun z => (z - a)⁻¹) (ball c r) := by
    refine DifferentiableOn.diffContOnCl ?_
    intro z hz
    have hza : z ≠ a := by
      rintro rfl
      rw [closure_ball c h0.ne', mem_closedBall] at hz
      linarith
    exact DifferentiableAt.differentiableWithinAt (by fun_prop (disch := exact sub_ne_zero.mpr hza))
  exact hd.circleIntegral_eq_zero h0.le

/-- `∮ dz/(z - a)` over the circle `C(c, r)` (`r > 0`) is `2πi` if `a` is inside and `0` if `a` is outside the
closed disc. -/
theorem circleIntegral_sub_inv (a c : ℂ) (r : ℝ) (h0 : 0 < r) :
    (dist a c < r → (∮ z in C(c, r), (z - a)⁻¹) = 2 * π * I) ∧
      (r < dist a c → (∮ z in C(c, r), (z - a)⁻¹) = 0) :=
  circleIntegral_sub_inv_aux a c r h0

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.circleIntegral_sub_inv (0) (0) (1) (one_pos) := by
  exact KaluzaKlein.ComplexAnalysis.HolomorphicWorked.circleIntegral_sub_inv_aux 0 0 1 one_pos

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.circleIntegral_sub_inv (2) (0) (1) (one_pos) := by
  exact KaluzaKlein.ComplexAnalysis.HolomorphicWorked.circleIntegral_sub_inv_aux 2 0 1 one_pos

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.circleIntegral_sub_inv (Complex.I) (0) (2) (two_pos) := by
  exact KaluzaKlein.ComplexAnalysis.HolomorphicWorked.circleIntegral_sub_inv_aux Complex.I 0 2 two_pos

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicWorked.circleIntegral_sub_inv
-- test: ca-holo-circle-inv-test.sage

/-- `z ↦ 1/(z - a)` is integrable on a circle not passing through `a`. -/
theorem circleIntegrable_inv (a c : ℂ) (r : ℝ) (h0 : 0 < r) (h : dist a c ≠ r) :
    CircleIntegrable (fun z => (z - a)⁻¹) c r := by
  rw [circleIntegrable_sub_inv_iff]
  right
  intro ha
  rw [mem_sphere, abs_of_pos h0] at ha
  exact h ha

/-- Auxiliary form of `winding_numbers` (its proof); the claim below restates it. -/
theorem winding_numbers_aux (r : ℝ) (h0 : 0 < r) (h1 : r < 1) :
    (2 * π * I)⁻¹ * (∮ z in C(1, r), ((z - 1)⁻¹ - (z - I)⁻¹ - (z - -I)⁻¹)) = 1 ∧
      (2 * π * I)⁻¹ * (∮ z in C(I, r), ((z - 1)⁻¹ - (z - I)⁻¹ - (z - -I)⁻¹)) = -1 ∧
      (2 * π * I)⁻¹ * (∮ z in C(-I, r), ((z - 1)⁻¹ - (z - I)⁻¹ - (z - -I)⁻¹)) = -1 := by
  have hpi : (2 * π * I : ℂ) ≠ 0 := by simp [Real.pi_ne_zero, I_ne_zero]
  have key : ∀ a b : ℂ, 1 < Complex.normSq (a - b) → 1 < dist a b := fun a b h => by
    rw [Complex.dist_eq, Complex.norm_def]
    calc (1 : ℝ) = Real.sqrt 1 := Real.sqrt_one.symm
      _ < Real.sqrt (Complex.normSq (a - b)) := Real.sqrt_lt_sqrt (by norm_num) h
  have dI1 : 1 < dist I 1 := key _ _ (by norm_num [Complex.normSq_apply])
  have dnI1 : 1 < dist (-I) 1 := key _ _ (by norm_num [Complex.normSq_apply])
  have d1I : 1 < dist (1 : ℂ) I := key _ _ (by norm_num [Complex.normSq_apply])
  have dnII : 1 < dist (-I) I := key _ _ (by norm_num [Complex.normSq_apply])
  have d1nI : 1 < dist (1 : ℂ) (-I) := key _ _ (by norm_num [Complex.normSq_apply])
  have dInI : 1 < dist I (-I) := key _ _ (by norm_num [Complex.normSq_apply])
  have split : ∀ c : ℂ, dist 1 c ≠ r → dist I c ≠ r → dist (-I) c ≠ r →
      (∮ z in C(c, r), ((z - 1)⁻¹ - (z - I)⁻¹ - (z - -I)⁻¹)) =
        (∮ z in C(c, r), (z - 1)⁻¹) - (∮ z in C(c, r), (z - I)⁻¹) - (∮ z in C(c, r), (z - -I)⁻¹) := by
    intro c ha hb hc
    have i1 := circleIntegrable_inv 1 c r h0 ha
    have i2 := circleIntegrable_inv I c r h0 hb
    have i3 := circleIntegrable_inv (-I) c r h0 hc
    rw [circleIntegral.integral_sub (f := fun z => (z - 1)⁻¹ - (z - I)⁻¹) (g := fun z => (z - -I)⁻¹) (i1.sub i2) i3,
      circleIntegral.integral_sub i1 i2]
  refine ⟨?_, ?_, ?_⟩
  · rw [split 1 (by rw [dist_self]; exact h0.ne) (by linarith) (by linarith),
      (circleIntegral_sub_inv 1 1 r h0).1 (by rw [dist_self]; exact h0), (circleIntegral_sub_inv I 1 r h0).2 (by linarith),
      (circleIntegral_sub_inv (-I) 1 r h0).2 (by linarith)]
    field_simp
    ring
  · rw [split I (by linarith) (by rw [dist_self]; exact h0.ne) (by linarith),
      (circleIntegral_sub_inv 1 I r h0).2 (by linarith), (circleIntegral_sub_inv I I r h0).1 (by rw [dist_self]; exact h0),
      (circleIntegral_sub_inv (-I) I r h0).2 (by linarith)]
    field_simp
    ring
  · rw [split (-I) (by linarith) (by linarith) (by rw [dist_self]; exact h0.ne),
      (circleIntegral_sub_inv 1 (-I) r h0).2 (by linarith), (circleIntegral_sub_inv I (-I) r h0).2 (by linarith),
      (circleIntegral_sub_inv (-I) (-I) r h0).1 (by rw [dist_self]; exact h0)]
    field_simp
    ring

/-- The argument principle for `f = (z - 1)/(z² + 1)` on circles of radius `r < 1`: `(2πi)⁻¹ ∮ f'/f` is `+1` around
the zero `1` and `-1` around each pole `±i` (the hue winds once, in opposite senses). -/
theorem winding_numbers (r : ℝ) (h0 : 0 < r) (h1 : r < 1) :
    (2 * π * I)⁻¹ * (∮ z in C(1, r), ((z - 1)⁻¹ - (z - I)⁻¹ - (z - -I)⁻¹)) = 1 ∧
      (2 * π * I)⁻¹ * (∮ z in C(I, r), ((z - 1)⁻¹ - (z - I)⁻¹ - (z - -I)⁻¹)) = -1 ∧
      (2 * π * I)⁻¹ * (∮ z in C(-I, r), ((z - 1)⁻¹ - (z - I)⁻¹ - (z - -I)⁻¹)) = -1 :=
  winding_numbers_aux r h0 h1

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.winding_numbers ((1 / 2)) ((by norm_num)) ((by norm_num)) := by
  exact KaluzaKlein.ComplexAnalysis.HolomorphicWorked.winding_numbers_aux (1 / 2) (by norm_num) (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.winding_numbers ((1 / 4)) ((by norm_num)) ((by norm_num)) := by
  exact KaluzaKlein.ComplexAnalysis.HolomorphicWorked.winding_numbers_aux (1 / 4) (by norm_num) (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.winding_numbers ((9 / 10)) ((by norm_num)) ((by norm_num)) := by
  exact KaluzaKlein.ComplexAnalysis.HolomorphicWorked.winding_numbers_aux (9 / 10) (by norm_num) (by norm_num)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicWorked.winding_numbers
-- test: ca-holo-winding-test.sage

/-! ## exr-cr-exp-jacobian -/

/-- `J = eˣ (cos y, -sin y; sin y, cos y)`: the Jacobian of `eᶻ` is `eˣ` times the rotation by `y`. -/
theorem exp_jac_factor (x y : ℝ) :
    !![Real.exp x * Real.cos y, -(Real.exp x * Real.sin y); Real.exp x * Real.sin y, Real.exp x * Real.cos y] =
      !![Real.exp x, 0; 0, Real.exp x] * !![Real.cos y, -Real.sin y; Real.sin y, Real.cos y] := by
  generalize Real.exp x = E
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
        all_goals ring1))
  -- certificate: ca-holo-matrix.sage

/-- `det J = e^{2x}` (as `(eˣ)²`). -/
theorem exp_jac_det (x y : ℝ) :
    !![Real.exp x * Real.cos y, -(Real.exp x * Real.sin y); Real.exp x * Real.sin y, Real.exp x * Real.cos y].det =
      Real.exp x ^ 2 := by
  generalize Real.exp x = E
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
      all_goals linear_combination ((E ^ 2 : ℝ)) * (Real.cos_sq_add_sin_sq (y : ℝ)))
  -- certificate: ca-holo-matrix.sage

/-- `Jᵀ J = e^{2x} 1`. -/
theorem exp_jac_gram (x y : ℝ) :
    !![Real.exp x * Real.cos y, -(Real.exp x * Real.sin y); Real.exp x * Real.sin y, Real.exp x * Real.cos y].transpose *
        !![Real.exp x * Real.cos y, -(Real.exp x * Real.sin y); Real.exp x * Real.sin y, Real.exp x * Real.cos y] =
      !![Real.exp x ^ 2, 0; 0, Real.exp x ^ 2] := by
  generalize Real.exp x = E
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
        all_goals
          first
          | ring1
          | (norm_num; done)
          | linear_combination ((E ^ 2 : ℝ)) * (Real.cos_sq_add_sin_sq (y : ℝ))
          | linear_combination -(((E ^ 2 : ℝ)) * (Real.cos_sq_add_sin_sq (y : ℝ)))))
  -- certificate: ca-holo-matrix.sage

/-- `e^{2x} = (eˣ)²`. -/
theorem exp_two_mul (x : ℝ) : Real.exp (2 * x) = Real.exp x ^ 2 := by
  rw [sq, ← Real.exp_add]
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.exp_two_mul (0) := by
  rw [sq, ← Real.exp_add]
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.exp_two_mul (1) := by
  rw [sq, ← Real.exp_add]
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.exp_two_mul (1) := by
  rw [sq, ← Real.exp_add]
  ring_nf

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicWorked.exp_two_mul
-- test: ca-holo-exp-two-mul-test.sage

/-- `|eᶻ|² = e^{2x}`. -/
theorem exp_norm_sq (x y : ℝ) : ‖Complex.exp ((x : ℂ) + y * I)‖ ^ 2 = Real.exp (2 * x) := by
  rw [Complex.norm_exp]
  simp only [add_re, ofReal_re, mul_re, I_re, mul_zero, ofReal_im, I_im, mul_one, sub_self, add_zero]
  rw [sq, ← Real.exp_add]
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.exp_norm_sq (0) (0) := by
  rw [Complex.norm_exp]
  simp only [Complex.add_re, Complex.ofReal_re, Complex.mul_re, Complex.I_re, mul_zero, Complex.ofReal_im, Complex.I_im, mul_one, sub_self, add_zero]
  rw [sq, ← Real.exp_add]
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.exp_norm_sq (1) (-5) := by
  rw [Complex.norm_exp]
  simp only [Complex.add_re, Complex.ofReal_re, Complex.mul_re, Complex.I_re, mul_zero, Complex.ofReal_im, Complex.I_im, mul_one, sub_self, add_zero]
  rw [sq, ← Real.exp_add]
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.exp_norm_sq (0) (1) := by
  rw [Complex.norm_exp]
  simp only [Complex.add_re, Complex.ofReal_re, Complex.mul_re, Complex.I_re, mul_zero, Complex.ofReal_im, Complex.I_im, mul_one, sub_self, add_zero]
  rw [sq, ← Real.exp_add]
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.exp_norm_sq (0) (0) := by
  rw [Complex.norm_exp]
  simp only [Complex.add_re, Complex.ofReal_re, Complex.mul_re, Complex.I_re, mul_zero, Complex.ofReal_im, Complex.I_im, mul_one, sub_self, add_zero]
  rw [sq, ← Real.exp_add]
  ring_nf

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.HolomorphicWorked.exp_norm_sq
-- test: ca-holo-exp-norm-sq-test.sage

/-! ## exr-cr-harmonic-general -/

/-- `uv = ½ Im f²` and `u² - v² = Re f²` are harmonic, as parts of the holomorphic `f²`. -/
theorem products_harmonic (f : ℂ → ℂ) (U : Set ℂ) (hU : IsOpen U) (hf : DifferentiableOn ℂ f U) (z : ℂ) (hz : z ∈ U) :
    HarmonicAt (fun w => (f w).re * (f w).im) z ∧ HarmonicAt (fun w => (f w).re ^ 2 - (f w).im ^ 2) z := by
  have hA : AnalyticAt ℂ (fun w => f w ^ 2) z := (hf.analyticAt (hU.mem_nhds hz)).pow 2
  have e1 : (fun w => (f w).re * (f w).im) = (1 / 2 : ℝ) • (fun w => ((f w) ^ 2).im) := by
    funext w
    simp [sq]
    ring
  have e2 : (fun w => (f w).re ^ 2 - (f w).im ^ 2) = fun w => ((f w) ^ 2).re := by
    funext w
    simp [sq]
  rw [e1, e2]
  exact ⟨hA.harmonicAt_im.const_smul, hA.harmonicAt_re⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.products_harmonic ((fun w : ℂ => w)) (Set.univ) (isOpen_univ) ((by fun_prop)) (0) ((Set.mem_univ _)) := by
  have hA : AnalyticAt ℂ (fun w => (fun w : ℂ => w) w ^ 2) (0) := ((DifferentiableOn.analyticAt (s := Set.univ) (by fun_prop) (Filter.univ_mem))).pow 2
  have e1 : (fun w => ((fun w : ℂ => w) w).re * ((fun w : ℂ => w) w).im) = (1 / 2 : ℝ) • (fun w => (((fun w : ℂ => w) w) ^ 2).im) := by
    funext w
    simp [sq]
    ring
  have e2 : (fun w => ((fun w : ℂ => w) w).re ^ 2 - ((fun w : ℂ => w) w).im ^ 2) = fun w => (((fun w : ℂ => w) w) ^ 2).re := by
    funext w
    simp [sq]
  rw [e1, e2]
  exact ⟨hA.harmonicAt_im.const_smul, hA.harmonicAt_re⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.products_harmonic ((fun w : ℂ => w ^ 2)) (Set.univ) (isOpen_univ) ((by fun_prop)) ((1 + Complex.I)) ((Set.mem_univ _)) := by
  have hA : AnalyticAt ℂ (fun w => (fun w : ℂ => w ^ 2) w ^ 2) (1 + Complex.I) := ((DifferentiableOn.analyticAt (s := Set.univ) (by fun_prop) (Filter.univ_mem))).pow 2
  have e1 : (fun w => ((fun w : ℂ => w ^ 2) w).re * ((fun w : ℂ => w ^ 2) w).im) = (1 / 2 : ℝ) • (fun w => (((fun w : ℂ => w ^ 2) w) ^ 2).im) := by
    funext w
    simp [sq]
    ring
  have e2 : (fun w => ((fun w : ℂ => w ^ 2) w).re ^ 2 - ((fun w : ℂ => w ^ 2) w).im ^ 2) = fun w => (((fun w : ℂ => w ^ 2) w) ^ 2).re := by
    funext w
    simp [sq]
  rw [e1, e2]
  exact ⟨hA.harmonicAt_im.const_smul, hA.harmonicAt_re⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.products_harmonic ((fun w : ℂ => w ^ 3 + w)) ((Metric.ball 0 1)) (Metric.isOpen_ball) ((by fun_prop)) (0) ((Metric.mem_ball_self one_pos)) := by
  have hA : AnalyticAt ℂ (fun w => (fun w : ℂ => w ^ 3 + w) w ^ 2) (0) := ((DifferentiableOn.analyticAt (s := Metric.ball 0 1) (by fun_prop) (Metric.isOpen_ball.mem_nhds (Metric.mem_ball_self one_pos)))).pow 2
  have e1 : (fun w => ((fun w : ℂ => w ^ 3 + w) w).re * ((fun w : ℂ => w ^ 3 + w) w).im) = (1 / 2 : ℝ) • (fun w => (((fun w : ℂ => w ^ 3 + w) w) ^ 2).im) := by
    funext w
    simp [sq]
    ring
  have e2 : (fun w => ((fun w : ℂ => w ^ 3 + w) w).re ^ 2 - ((fun w : ℂ => w ^ 3 + w) w).im ^ 2) = fun w => (((fun w : ℂ => w ^ 3 + w) w) ^ 2).re := by
    funext w
    simp [sq]
  rw [e1, e2]
  exact ⟨hA.harmonicAt_im.const_smul, hA.harmonicAt_re⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicWorked.products_harmonic
-- test: ca-holo-products-harmonic-test.sage

/-! ## exr-cr-conformal-inverse -/

/-- The derivative of the inverse: if `f'(z₀) ≠ 0` and `g` is a continuous local inverse at `w₀ = f(z₀)`, then
`g'(w₀) = 1/f'(z₀)`. -/
theorem inverse_deriv (f g : ℂ → ℂ) (f' w₀ : ℂ) (hg : ContinuousAt g w₀) (hf : HasDerivAt f f' (g w₀))
    (hf' : f' ≠ 0) (hfg : ∀ᶠ w in 𝓝 w₀, f (g w) = w) : HasDerivAt g f'⁻¹ w₀ :=
  HasDerivAt.of_local_left_inverse hg hf hf' hfg

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.inverse_deriv ((fun w : ℂ => w)) ((fun w : ℂ => w)) (1) (0) ((continuousAt_id' _)) ((hasDerivAt_id' _)) (one_ne_zero) ((Filter.Eventually.of_forall fun _ => rfl)) := by
  exact HasDerivAt.of_local_left_inverse (continuousAt_id' _) (hasDerivAt_id' _) one_ne_zero (Filter.Eventually.of_forall fun _ => rfl)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.inverse_deriv ((fun w : ℂ => w)) ((fun w : ℂ => w)) (1) (1) ((continuousAt_id' _)) ((hasDerivAt_id' _)) (one_ne_zero) ((Filter.Eventually.of_forall fun _ => rfl)) := by
  exact HasDerivAt.of_local_left_inverse (continuousAt_id' _) (hasDerivAt_id' _) one_ne_zero (Filter.Eventually.of_forall fun _ => rfl)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.inverse_deriv ((fun w : ℂ => w)) ((fun w : ℂ => w)) (1) (Complex.I) ((continuousAt_id' _)) ((hasDerivAt_id' _)) (one_ne_zero) ((Filter.Eventually.of_forall fun _ => rfl)) := by
  exact HasDerivAt.of_local_left_inverse (continuousAt_id' _) (hasDerivAt_id' _) one_ne_zero (Filter.Eventually.of_forall fun _ => rfl)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicWorked.inverse_deriv
-- test: ca-holo-inverse-deriv-test.sage

/-- `Log` inverts `eᶻ` on the strip `-π < Im z < π`. -/
theorem log_exp_strip (z : ℂ) (h₁ : -π < z.im) (h₂ : z.im < π) : Complex.log (Complex.exp z) = z :=
  Complex.log_exp h₁ h₂.le

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.log_exp_strip (0) ((by simp; linarith [Real.pi_pos])) ((by simp; linarith [Real.pi_pos])) := by
  exact Complex.log_exp (by simp; linarith [Real.pi_pos]) (by simp; linarith [Real.pi_pos])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.log_exp_strip (Complex.I) ((by simp; linarith [Real.pi_gt_three])) ((by simp; linarith [Real.pi_gt_three])) := by
  exact Complex.log_exp (by simp; linarith [Real.pi_gt_three]) (by simp; linarith [Real.pi_gt_three])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.log_exp_strip ((1 + 2 * Complex.I)) ((by simp; linarith [Real.pi_gt_three])) ((by simp; linarith [Real.pi_gt_three])) := by
  exact Complex.log_exp (by simp; linarith [Real.pi_gt_three]) (by simp; linarith [Real.pi_gt_three])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicWorked.log_exp_strip
-- test: ca-holo-log-exp-test.sage

/-- By the inverse rule `Log' w = 1/e^{Log w} = 1/w` on the slit plane. -/
theorem log_deriv_by_inverse (w : ℂ) (hw : w ∈ slitPlane) : HasDerivAt Complex.log w⁻¹ w := by
  have h := inverse_deriv Complex.exp Complex.log (Complex.exp (Complex.log w)) w (_root_.continuousAt_clog hw)
    (Complex.hasDerivAt_exp _) (Complex.exp_ne_zero _)
    (Filter.eventually_of_mem (isOpen_ne.mem_nhds (slitPlane_ne_zero hw)) fun y hy => Complex.exp_log hy)
  rwa [Complex.exp_log (slitPlane_ne_zero hw)] at h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.log_deriv_by_inverse (1) ((by norm_num [Complex.mem_slitPlane_iff])) := by
  have hw := (by norm_num [Complex.mem_slitPlane_iff] : (1 : ℂ) ∈ Complex.slitPlane)
  have h := inverse_deriv Complex.exp Complex.log (Complex.exp (Complex.log (1))) (1) (_root_.continuousAt_clog hw)
    (Complex.hasDerivAt_exp _) (Complex.exp_ne_zero _)
    (Filter.eventually_of_mem (isOpen_ne.mem_nhds (slitPlane_ne_zero hw)) fun y hy => Complex.exp_log hy)
  rwa [Complex.exp_log (slitPlane_ne_zero hw)] at h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.log_deriv_by_inverse (Complex.I) ((by norm_num [Complex.mem_slitPlane_iff])) := by
  have hw := (by norm_num [Complex.mem_slitPlane_iff] : (Complex.I : ℂ) ∈ Complex.slitPlane)
  have h := inverse_deriv Complex.exp Complex.log (Complex.exp (Complex.log (Complex.I))) (Complex.I) (_root_.continuousAt_clog hw)
    (Complex.hasDerivAt_exp _) (Complex.exp_ne_zero _)
    (Filter.eventually_of_mem (isOpen_ne.mem_nhds (slitPlane_ne_zero hw)) fun y hy => Complex.exp_log hy)
  rwa [Complex.exp_log (slitPlane_ne_zero hw)] at h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HolomorphicWorked.log_deriv_by_inverse ((2 + Complex.I)) ((by norm_num [Complex.mem_slitPlane_iff])) := by
  have hw := (by norm_num [Complex.mem_slitPlane_iff] : (2 + Complex.I : ℂ) ∈ Complex.slitPlane)
  have h := inverse_deriv Complex.exp Complex.log (Complex.exp (Complex.log (2 + Complex.I))) (2 + Complex.I) (_root_.continuousAt_clog hw)
    (Complex.hasDerivAt_exp _) (Complex.exp_ne_zero _)
    (Filter.eventually_of_mem (isOpen_ne.mem_nhds (slitPlane_ne_zero hw)) fun y hy => Complex.exp_log hy)
  rwa [Complex.exp_log (slitPlane_ne_zero hw)] at h

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HolomorphicWorked.log_deriv_by_inverse
-- test: ca-holo-log-inverse-test.sage

end KaluzaKlein.ComplexAnalysis.HolomorphicWorked
