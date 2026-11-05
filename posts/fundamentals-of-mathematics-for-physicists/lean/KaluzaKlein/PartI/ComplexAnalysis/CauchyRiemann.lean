import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.ComplexDerivative
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "The Cauchy-Riemann equations"

The claims of the section "The Cauchy-Riemann equations" of `tutorial/part-i-04-complex-analysis.qmd`.

For `f : ℂ → ℂ` real differentiable at `z`, write `L = fderiv ℝ f z`; then `L 1 = u_x + i v_x` and `L i = u_y + i v_y`
are the partial derivatives of `f = u + iv`. The Cauchy-Riemann equations `u_x = v_y`, `u_y = -v_x` are equivalent to
complex differentiability (`cr_iff`, from mathlib's `differentiableAt_complex_iff_differentiableAt_real`), and then
`f' = u_x + i v_x`. Explicit parts `u`, `v` are real functions of `x, y`, and their partial derivatives are `deriv` of
the one-variable sections. The section checks the equations on `z²`, `eᶻ`, `z̄`, `1/z`, `cos z`; the Wirtinger form;
and the constancy of real-valued holomorphic functions and of functions with vanishing derivative.
-/

namespace KaluzaKlein.ComplexAnalysis.CauchyRiemann

open Complex Filter Topology ComplexConjugate

/-! ## thm-cauchy-riemann -/

/-- The Cauchy-Riemann equations: a real differentiable `f` is complex differentiable at `z` iff
`u_x = v_y` and `u_y = -v_x` there, with `u_x + i v_x = L 1`, `u_y + i v_y = L i`. -/
theorem cr_iff (f : ℂ → ℂ) (z : ℂ) (hf : DifferentiableAt ℝ f z) :
    DifferentiableAt ℂ f z ↔
      ((fderiv ℝ f z 1).re = (fderiv ℝ f z I).im ∧ (fderiv ℝ f z I).re = -(fderiv ℝ f z 1).im) := by
  rw [differentiableAt_complex_iff_differentiableAt_real, smul_eq_mul, Complex.ext_iff]
  simp only [hf, true_and, mul_re, I_re, zero_mul, I_im, one_mul, zero_sub, mul_im, zero_add]
  constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cr_iff ((fun w : ℂ => w ^ 2)) (0) ((by fun_prop)) := by
  have hf := (by fun_prop : DifferentiableAt ℝ (fun w : ℂ => w ^ 2) 0)
  rw [differentiableAt_complex_iff_differentiableAt_real, smul_eq_mul, Complex.ext_iff]
  simp only [hf, true_and, mul_re, I_re, zero_mul, I_im, one_mul, zero_sub, mul_im, zero_add]
  constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cr_iff ((fun w : ℂ => w ^ 2)) ((1 + Complex.I)) ((by fun_prop)) := by
  have hf := (by fun_prop : DifferentiableAt ℝ (fun w : ℂ => w ^ 2) (1 + Complex.I))
  rw [differentiableAt_complex_iff_differentiableAt_real, smul_eq_mul, Complex.ext_iff]
  simp only [hf, true_and, mul_re, I_re, zero_mul, I_im, one_mul, zero_sub, mul_im, zero_add]
  constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cr_iff ((fun w : ℂ => (starRingEnd ℂ) w)) (1) ((Complex.differentiable_conj 1)) := by
  have hf := (Complex.differentiable_conj 1)
  rw [differentiableAt_complex_iff_differentiableAt_real, smul_eq_mul, Complex.ext_iff]
  simp only [hf, true_and, mul_re, I_re, zero_mul, I_im, one_mul, zero_sub, mul_im, zero_add]
  constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.cr_iff
-- test: ca-holo-cr-iff-test.sage

/-- `f'(z) = u_x + i v_x`: the complex derivative is the real derivative in the direction `1`. -/
theorem deriv_eq_fderiv_one (f : ℂ → ℂ) (z : ℂ) (h : DifferentiableAt ℂ f z) :
    deriv f z = fderiv ℝ f z 1 := by
  rw [h.fderiv_restrictScalars ℝ]
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.deriv_eq_fderiv_one ((fun w : ℂ => w)) (0) ((by fun_prop)) := by
  have h := (by fun_prop : DifferentiableAt ℂ ((fun w : ℂ => w)) (0 : ℂ))
  rw [h.fderiv_restrictScalars ℝ]
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.deriv_eq_fderiv_one ((fun w : ℂ => w ^ 2)) (Complex.I) ((by fun_prop)) := by
  have h := (by fun_prop : DifferentiableAt ℂ (fun w : ℂ => w ^ 2) (Complex.I))
  rw [h.fderiv_restrictScalars ℝ]
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.deriv_eq_fderiv_one (Complex.exp) (1) ((by fun_prop)) := by
  have h := (by fun_prop : DifferentiableAt ℂ (Complex.exp) (1 : ℂ))
  rw [h.fderiv_restrictScalars ℝ]
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.deriv_eq_fderiv_one
-- test: ca-holo-deriv-fderiv-test.sage

/-- If `f` is complex differentiable at `z`, its partial derivatives exist: `f_x = f'` and `f_y = f' i` (that is
`u_y + i v_y = i(u_x + i v_x)`: `u_y = -v_x`, `v_y = u_x`). -/
theorem partials (f : ℂ → ℂ) (z : ℂ) (h : DifferentiableAt ℂ f z) :
    HasDerivAt (fun t : ℝ => f (z + t)) (deriv f z) 0 ∧
      HasDerivAt (fun t : ℝ => f (z + t * I)) (deriv f z * I) 0 := by
  constructor
  · have h1 : HasDerivAt (fun w : ℂ => f (z + w)) (deriv f z) ((0 : ℝ) : ℂ) :=
      HasDerivAt.comp_const_add z ((0 : ℝ) : ℂ) (by rw [Complex.ofReal_zero, add_zero]; exact h.hasDerivAt)
    exact h1.comp_ofReal
  · have hl : HasDerivAt (fun w : ℂ => z + w * I) I ((0 : ℝ) : ℂ) :=
      (((hasDerivAt_id' ((0 : ℝ) : ℂ)).mul_const I).const_add z).congr_deriv (one_mul I)
    have hf' : HasDerivAt f (deriv f z) (z + ((0 : ℝ) : ℂ) * I) := by
      rw [Complex.ofReal_zero, zero_mul, add_zero]; exact h.hasDerivAt
    exact (hf'.comp ((0 : ℝ) : ℂ) hl).comp_ofReal

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.partials ((fun w : ℂ => w)) (0) ((by fun_prop)) := by
  have h := (by fun_prop : DifferentiableAt ℂ ((fun w : ℂ => w)) (0 : ℂ))
  constructor
  · have h1 : HasDerivAt (fun w : ℂ => ((fun w : ℂ => w)) ((0 : ℂ) + w)) (deriv ((fun w : ℂ => w)) (0 : ℂ)) ((0 : ℝ) : ℂ) :=
      HasDerivAt.comp_const_add (0 : ℂ) ((0 : ℝ) : ℂ) (by rw [Complex.ofReal_zero, add_zero]; exact h.hasDerivAt)
    exact h1.comp_ofReal
  · have hl : HasDerivAt (fun w : ℂ => (0 : ℂ) + w * I) I ((0 : ℝ) : ℂ) :=
      (((hasDerivAt_id' ((0 : ℝ) : ℂ)).mul_const I).const_add (0 : ℂ)).congr_deriv (one_mul I)
    have hf' : HasDerivAt ((fun w : ℂ => w)) (deriv ((fun w : ℂ => w)) (0 : ℂ)) ((0 : ℂ) + ((0 : ℝ) : ℂ) * I) := by
      rw [Complex.ofReal_zero, zero_mul, add_zero]; exact h.hasDerivAt
    exact (hf'.comp ((0 : ℝ) : ℂ) hl).comp_ofReal

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.partials ((fun w : ℂ => w ^ 2)) ((1 + Complex.I)) ((by fun_prop)) := by
  have h := (by fun_prop : DifferentiableAt ℂ (fun w : ℂ => w ^ 2) (1 + Complex.I))
  constructor
  · have h1 : HasDerivAt (fun w : ℂ => (fun w : ℂ => w ^ 2) ((1 + Complex.I) + w)) (deriv (fun w : ℂ => w ^ 2) (1 + Complex.I)) ((0 : ℝ) : ℂ) :=
      HasDerivAt.comp_const_add (1 + Complex.I) ((0 : ℝ) : ℂ) (by rw [Complex.ofReal_zero, add_zero]; exact h.hasDerivAt)
    exact h1.comp_ofReal
  · have hl : HasDerivAt (fun w : ℂ => (1 + Complex.I) + w * I) I ((0 : ℝ) : ℂ) :=
      (((hasDerivAt_id' ((0 : ℝ) : ℂ)).mul_const I).const_add (1 + Complex.I)).congr_deriv (one_mul I)
    have hf' : HasDerivAt (fun w : ℂ => w ^ 2) (deriv (fun w : ℂ => w ^ 2) (1 + Complex.I)) ((1 + Complex.I) + ((0 : ℝ) : ℂ) * I) := by
      rw [Complex.ofReal_zero, zero_mul, add_zero]; exact h.hasDerivAt
    exact (hf'.comp ((0 : ℝ) : ℂ) hl).comp_ofReal

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.partials (Complex.exp) (Complex.I) ((by fun_prop)) := by
  have h := (by fun_prop : DifferentiableAt ℂ (Complex.exp) (Complex.I))
  constructor
  · have h1 : HasDerivAt (fun w : ℂ => (Complex.exp) ((Complex.I) + w)) (deriv (Complex.exp) (Complex.I)) ((0 : ℝ) : ℂ) :=
      HasDerivAt.comp_const_add (Complex.I) ((0 : ℝ) : ℂ) (by rw [Complex.ofReal_zero, add_zero]; exact h.hasDerivAt)
    exact h1.comp_ofReal
  · have hl : HasDerivAt (fun w : ℂ => (Complex.I) + w * I) I ((0 : ℝ) : ℂ) :=
      (((hasDerivAt_id' ((0 : ℝ) : ℂ)).mul_const I).const_add (Complex.I)).congr_deriv (one_mul I)
    have hf' : HasDerivAt (Complex.exp) (deriv (Complex.exp) (Complex.I)) ((Complex.I) + ((0 : ℝ) : ℂ) * I) := by
      rw [Complex.ofReal_zero, zero_mul, add_zero]; exact h.hasDerivAt
    exact (hf'.comp ((0 : ℝ) : ℂ) hl).comp_ofReal

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.partials
-- test: ca-holo-partials-test.sage

/-- Multiplication by `a + ib` acts on `(x, y)` by the matrix `(a, -b; b, a)`. -/
theorem mul_parts (a b x y : ℝ) :
    ((a : ℂ) + b * I) * ((x : ℂ) + y * I) = ((a * x - b * y : ℝ) : ℂ) + ((b * x + a * y : ℝ) : ℂ) * I := by
  have hI : I ^ 2 = -1 := I_sq
  push_cast
  generalize ((a : ℝ) : ℂ) = aC at *
  generalize ((b : ℝ) : ℂ) = bC at *
  generalize ((x : ℝ) : ℂ) = xC at *
  generalize ((y : ℝ) : ℂ) = yC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (bC * yC) * hI)
  -- certificate: ca-holo-ring.sage

/-! ## exm-cr-z2 -/

/-- `z² = (x² - y²) + i 2xy`. -/
theorem sq_parts (x y : ℝ) : ((x : ℂ) + y * I) ^ 2 = ((x ^ 2 - y ^ 2 : ℝ) : ℂ) + ((2 * x * y : ℝ) : ℂ) * I := by
  have hI : I ^ 2 = -1 := I_sq
  have e1 : ((x ^ 2 - y ^ 2 : ℝ) : ℂ) = (x : ℂ) ^ 2 - (y : ℂ) ^ 2 := by push_cast <;> ring
  have e2 : ((2 * x * y : ℝ) : ℂ) = 2 * (x : ℂ) * (y : ℂ) := by push_cast <;> ring
  rw [e1, e2]
  clear e1 e2
  generalize ((x : ℝ) : ℂ) = xC at *
  generalize ((y : ℝ) : ℂ) = yC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (yC ^ 2) * hI)
  -- certificate: ca-holo-ring.sage

/-- `u_x = 2x` for `u = x² - y²`. -/
theorem sq_ux (x y : ℝ) : deriv (fun x => x ^ 2 - y ^ 2) x = 2 * x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x ^ 2 - y ^ 2) ((2 * x : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-(y ^ 2) : ℝ))) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `u_y = -2y` for `u = x² - y²`. -/
theorem sq_uy (x y : ℝ) : deriv (fun y => x ^ 2 - y ^ 2) y = -2 * y := by
  rw [deriv_const_sub, neg_eq_iff_eq_neg]
  veriphy_certified
    ( have hd : HasDerivAt (fun y => y ^ 2) ((2 * y : ℝ)) (y) :=
        by
        convert ((hasDerivAt_id' (y : ℝ)).pow 2) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ca-holo-deriv.sage

/-- `v_x = 2y` for `v = 2xy`. -/
theorem sq_vx (x y : ℝ) : deriv (fun x => 2 * x * y) x = 2 * y := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => 2 * x * y) ((2 * y : ℝ)) (x) :=
        by
        convert ((hasDerivAt_id' (x : ℝ)).const_mul (2 * y : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `v_y = 2x` for `v = 2xy`. -/
theorem sq_vy (x y : ℝ) : deriv (fun y => 2 * x * y) y = 2 * x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun y => 2 * x * y) ((2 * x : ℝ)) (y) :=
        by
        convert ((hasDerivAt_id' (y : ℝ)).const_mul (2 * x : ℝ)) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- The Jacobian `(2x, -2y; 2y, 2x)` of `z²` has determinant `4(x² + y²)`. -/
theorem sq_jac_det (x y : ℝ) : !![2 * x, -2 * y; 2 * y, 2 * x].det = 4 * (x ^ 2 + y ^ 2) := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
      all_goals ring1)
  -- certificate: ca-holo-matrix.sage

/-- `4(x² + y²) = |2z|²`. -/
theorem sq_jac_det_norm (x y : ℝ) : 4 * (x ^ 2 + y ^ 2) = ‖2 * ((x : ℂ) + y * I)‖ ^ 2 := by
  rw [← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
  simp
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.sq_jac_det_norm (0) (0) := by
  rw [← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.sq_jac_det_norm (1) (-5) := by
  rw [← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.sq_jac_det_norm (0) (1) := by
  rw [← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.sq_jac_det_norm (0) (0) := by
  rw [← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
  norm_num
  try ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.sq_jac_det_norm
-- test: ca-holo-sq-jac-norm-test.sage

/-- `2z = 2x + i 2y`: the Jacobian of `z²` is the matrix of multiplication by `2z` (`mul_parts`). -/
theorem two_z_parts (x y : ℝ) : 2 * ((x : ℂ) + y * I) = ((2 * x : ℝ) : ℂ) + ((2 * y : ℝ) : ℂ) * I := by
  have e1 : ((2 * x : ℝ) : ℂ) = 2 * (x : ℂ) := by push_cast <;> ring
  have e2 : ((2 * y : ℝ) : ℂ) = 2 * (y : ℂ) := by push_cast <;> ring
  rw [e1, e2]
  clear e1 e2
  generalize ((x : ℝ) : ℂ) = xC at *
  generalize ((y : ℝ) : ℂ) = yC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-! ## exm-cr-exp -/

/-- `eᶻ = eˣ cos y + i eˣ sin y`. -/
theorem exp_parts (x y : ℝ) :
    Complex.exp ((x : ℂ) + y * I) = ((Real.exp x * Real.cos y : ℝ) : ℂ) + ((Real.exp x * Real.sin y : ℝ) : ℂ) * I := by
  push_cast
  rw [Complex.exp_add, Complex.exp_mul_I]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.exp_parts (0) (0) := by
  push_cast
  rw [Complex.exp_add, Complex.exp_mul_I]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.exp_parts (1) (-5) := by
  push_cast
  rw [Complex.exp_add, Complex.exp_mul_I]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.exp_parts (0) (1) := by
  push_cast
  rw [Complex.exp_add, Complex.exp_mul_I]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.exp_parts (0) (0) := by
  push_cast
  rw [Complex.exp_add, Complex.exp_mul_I]
  ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.exp_parts
-- test: ca-holo-exp-parts-test.sage

/-- `u_x = eˣ cos y`. -/
theorem exp_ux (x y : ℝ) : deriv (fun x => Real.exp x * Real.cos y) x = Real.exp x * Real.cos y := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => Real.exp x * Real.cos y) (((Real.cos y) * (Real.exp x) : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).exp).const_mul (Real.cos y : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ca-holo-deriv.sage

/-- `u_y = -eˣ sin y`. -/
theorem exp_uy (x y : ℝ) : deriv (fun y => Real.exp x * Real.cos y) y = -(Real.exp x * Real.sin y) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun y => Real.exp x * Real.cos y) ((-(Real.exp x) * (Real.sin y) : ℝ)) (y) :=
        by
        convert (((hasDerivAt_id' (y : ℝ)).cos).const_mul (Real.exp x : ℝ)) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ca-holo-deriv.sage

/-- `v_x = eˣ sin y`. -/
theorem exp_vx (x y : ℝ) : deriv (fun x => Real.exp x * Real.sin y) x = Real.exp x * Real.sin y := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => Real.exp x * Real.sin y) (((Real.exp x) * (Real.sin y) : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).exp).const_mul (Real.sin y : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `v_y = eˣ cos y`. -/
theorem exp_vy (x y : ℝ) : deriv (fun y => Real.exp x * Real.sin y) y = Real.exp x * Real.cos y := by
  veriphy_certified
    ( have hd : HasDerivAt (fun y => Real.exp x * Real.sin y) (((Real.cos y) * (Real.exp x) : ℝ)) (y) :=
        by
        convert (((hasDerivAt_id' (y : ℝ)).sin).const_mul (Real.exp x : ℝ)) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ca-holo-deriv.sage

/-! ## exm-cr-conjugate -/

/-- `z̄ = x - iy`. -/
theorem conj_parts (x y : ℝ) : conj ((x : ℂ) + y * I) = (x : ℂ) - y * I := by
  rw [map_add, map_mul, conj_ofReal, conj_ofReal, conj_I]
  generalize ((x : ℝ) : ℂ) = xC at *
  generalize ((y : ℝ) : ℂ) = yC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `u_x = 1` for `u = x`. -/
theorem conj_ux (x : ℝ) : deriv (fun x : ℝ => x) x = 1 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x) ((1 : ℝ)) (x) :=
        by
        convert (hasDerivAt_id' (x : ℝ)) using 1
        all_goals rfl
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `v_y = -1` for `v = -y`. -/
theorem conj_vy (y : ℝ) : deriv (fun y : ℝ => -y) y = -1 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun y => -y) ((-1 : ℝ)) (y) :=
        by
        convert ((hasDerivAt_id' (y : ℝ)).const_mul (-1 : ℝ)) using 1
        all_goals ((ring_nf; done))
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- The Jacobian `diag(1, -1)` of `z̄` has determinant `-1` (a reflection). -/
theorem conj_jac_det : !![(1 : ℝ), 0; 0, -1].det = -1 := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
      all_goals ring1)
  -- certificate: ca-holo-matrix.sage

/-- `diag(1, -1)` is not a rotation-scaling `(a, -b; b, a)`. -/
theorem conj_jac_not_rotation (a b : ℝ) : !![(1 : ℝ), 0; 0, -1] ≠ !![a, -b; b, a] := by
  intro h
  have h1 := congrFun (congrFun h 0) 0
  have h2 := congrFun (congrFun h 1) 1
  simp only [Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.empty_val', Matrix.cons_val_fin_one] at h1 h2
  linarith

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.conj_jac_not_rotation (0) (0) := by
  intro h
  have h1 := congrFun (congrFun h 0) 0
  have h2 := congrFun (congrFun h 1) 1
  simp only [Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.empty_val', Matrix.cons_val_fin_one] at h1 h2
  linarith

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.conj_jac_not_rotation (1) (-5) := by
  intro h
  have h1 := congrFun (congrFun h 0) 0
  have h2 := congrFun (congrFun h 1) 1
  simp only [Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.empty_val', Matrix.cons_val_fin_one] at h1 h2
  linarith

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.conj_jac_not_rotation (0) (1) := by
  intro h
  have h1 := congrFun (congrFun h 0) 0
  have h2 := congrFun (congrFun h 1) 1
  simp only [Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.empty_val', Matrix.cons_val_fin_one] at h1 h2
  linarith

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.conj_jac_not_rotation (0) (0) := by
  intro h
  have h1 := congrFun (congrFun h 0) 0
  have h2 := congrFun (congrFun h 1) 1
  simp only [Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.empty_val', Matrix.cons_val_fin_one] at h1 h2
  linarith

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.conj_jac_not_rotation
-- test: ca-holo-conj-jac-test.sage

/-! ## exr-cr-check -/

/-- `1/z = x/(x² + y²) - i y/(x² + y²)`: the product with `z` is `1`. -/
theorem inv_parts (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) :
    ((x : ℂ) + y * I) * (((x / (x ^ 2 + y ^ 2) : ℝ) : ℂ) + ((-y / (x ^ 2 + y ^ 2) : ℝ) : ℂ) * I) = 1 := by
  have h' : ((x : ℂ) ^ 2 + (y : ℂ) ^ 2) ≠ 0 := by exact_mod_cast h
  have hI : I ^ 2 = -1 := I_sq
  push_cast
  rw [div_eq_mul_inv, div_eq_mul_inv]
  have hinv : ((x : ℂ) ^ 2 + (y : ℂ) ^ 2) * ((x : ℂ) ^ 2 + (y : ℂ) ^ 2)⁻¹ = 1 := mul_inv_cancel₀ h'
  generalize ((x : ℂ) ^ 2 + (y : ℂ) ^ 2)⁻¹ = d at *
  generalize ((x : ℝ) : ℂ) = xC at *
  generalize ((y : ℝ) : ℂ) = yC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination hinv + (-yC ^ 2 * d) * hI)
  -- certificate: ca-holo-ring.sage

/-- `u_x = (y² - x²)/(x² + y²)²` for `u = x/(x² + y²)`. -/
theorem inv_ux (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) :
    deriv (fun x => x / (x ^ 2 + y ^ 2)) x = (y ^ 2 - x ^ 2) / (x ^ 2 + y ^ 2) ^ 2 := by
  veriphy_certified
    ( have hd :
        HasDerivAt (fun x => x / (x ^ 2 + y ^ 2))
          ((-2 * (x ^ 2) / ((x ^ 2 + (y ^ 2)) ^ 2) + (1 / (x ^ 2 + (y ^ 2)) ^ 1) : ℝ)) (x) :=
        by
        convert
          ((hasDerivAt_id' (x : ℝ)).div (((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (y ^ 2 : ℝ)))
            (by assumption)) using
          1
        all_goals (simp only [Pi.add_apply, Pi.pow_apply]; (field_simp; ring1))
      exact hd.deriv.trans (by (field_simp; ring1)))
  -- certificate: ca-holo-deriv.sage

/-- `u_y = -2xy/(x² + y²)²` for `u = x/(x² + y²)`. -/
theorem inv_uy (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) :
    deriv (fun y => x / (x ^ 2 + y ^ 2)) y = -2 * x * y / (x ^ 2 + y ^ 2) ^ 2 := by
  have hd := (hasDerivAt_const y x).fun_div ((hasDerivAt_pow 2 y).const_add (x ^ 2)) h
  refine hd.deriv.trans ?_
  rw [div_left_inj' (pow_ne_zero 2 h)]
  simp only [Nat.cast_ofNat, show (2 : ℕ) - 1 = 1 from rfl, pow_one]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.inv_uy (1) (0) ((by norm_num)) := by
  have h := (by norm_num : (1 : ℝ) ^ 2 + (0 : ℝ) ^ 2 ≠ 0)
  have hd := (hasDerivAt_const (0 : ℝ) (1 : ℝ)).fun_div ((hasDerivAt_pow 2 (0 : ℝ)).const_add ((1 : ℝ) ^ 2)) h
  refine hd.deriv.trans ?_
  rw [div_left_inj' (pow_ne_zero 2 h)]
  simp only [Nat.cast_ofNat, show (2 : ℕ) - 1 = 1 from rfl, pow_one]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.inv_uy (1) (2) ((by norm_num)) := by
  have h := (by norm_num : (1 : ℝ) ^ 2 + (2 : ℝ) ^ 2 ≠ 0)
  have hd := (hasDerivAt_const (2 : ℝ) (1 : ℝ)).fun_div ((hasDerivAt_pow 2 (2 : ℝ)).const_add ((1 : ℝ) ^ 2)) h
  refine hd.deriv.trans ?_
  rw [div_left_inj' (pow_ne_zero 2 h)]
  simp only [Nat.cast_ofNat, show (2 : ℕ) - 1 = 1 from rfl, pow_one]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.inv_uy ((-1 / 2)) (3) ((by norm_num)) := by
  have h := (by norm_num : (-1 / 2 : ℝ) ^ 2 + (3 : ℝ) ^ 2 ≠ 0)
  have hd := (hasDerivAt_const (3 : ℝ) (-1 / 2 : ℝ)).fun_div ((hasDerivAt_pow 2 (3 : ℝ)).const_add ((-1 / 2 : ℝ) ^ 2)) h
  refine hd.deriv.trans ?_
  rw [div_left_inj' (pow_ne_zero 2 h)]
  simp only [Nat.cast_ofNat, show (2 : ℕ) - 1 = 1 from rfl, pow_one]
  ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.inv_uy
-- test: ca-holo-inv-uy-test.sage

/-- `v_x = 2xy/(x² + y²)²` for `v = -y/(x² + y²)`. -/
theorem inv_vx (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) :
    deriv (fun x => -y / (x ^ 2 + y ^ 2)) x = 2 * x * y / (x ^ 2 + y ^ 2) ^ 2 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => -y / (x ^ 2 + y ^ 2)) ((2 * x * y / ((x ^ 2 + (y ^ 2)) ^ 2) : ℝ)) (x) :=
        by
        convert
          (((hasDerivAt_const (x : ℝ) (1 : ℝ)).div
                (((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (y ^ 2 : ℝ))) (by assumption)).const_mul
            (-y : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply, Pi.div_apply, Pi.pow_apply]; ring_nf; done)
          | (simp only [Pi.add_apply, Pi.pow_apply]; ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `v_y = (y² - x²)/(x² + y²)²` for `v = -y/(x² + y²)`. -/
theorem inv_vy (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) :
    deriv (fun y => -y / (x ^ 2 + y ^ 2)) y = (y ^ 2 - x ^ 2) / (x ^ 2 + y ^ 2) ^ 2 := by
  have hd := (hasDerivAt_neg' y : HasDerivAt (fun t : ℝ => -t) (-1) y).fun_div ((hasDerivAt_pow 2 y).const_add (x ^ 2)) h
  refine hd.deriv.trans ?_
  rw [div_left_inj' (pow_ne_zero 2 h)]
  simp only [Nat.cast_ofNat, show (2 : ℕ) - 1 = 1 from rfl, pow_one]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.inv_vy (1) (0) ((by norm_num)) := by
  have h := (by norm_num : (1 : ℝ) ^ 2 + (0 : ℝ) ^ 2 ≠ 0)
  have hd := (hasDerivAt_neg' (0 : ℝ) : HasDerivAt (fun t : ℝ => -t) (-1) (0 : ℝ)).fun_div ((hasDerivAt_pow 2 (0 : ℝ)).const_add ((1 : ℝ) ^ 2)) h
  refine hd.deriv.trans ?_
  rw [div_left_inj' (pow_ne_zero 2 h)]
  simp only [Nat.cast_ofNat, show (2 : ℕ) - 1 = 1 from rfl, pow_one]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.inv_vy (1) (2) ((by norm_num)) := by
  have h := (by norm_num : (1 : ℝ) ^ 2 + (2 : ℝ) ^ 2 ≠ 0)
  have hd := (hasDerivAt_neg' (2 : ℝ) : HasDerivAt (fun t : ℝ => -t) (-1) (2 : ℝ)).fun_div ((hasDerivAt_pow 2 (2 : ℝ)).const_add ((1 : ℝ) ^ 2)) h
  refine hd.deriv.trans ?_
  rw [div_left_inj' (pow_ne_zero 2 h)]
  simp only [Nat.cast_ofNat, show (2 : ℕ) - 1 = 1 from rfl, pow_one]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.inv_vy ((-1 / 2)) (3) ((by norm_num)) := by
  have h := (by norm_num : (-1 / 2 : ℝ) ^ 2 + (3 : ℝ) ^ 2 ≠ 0)
  have hd := (hasDerivAt_neg' (3 : ℝ) : HasDerivAt (fun t : ℝ => -t) (-1) (3 : ℝ)).fun_div ((hasDerivAt_pow 2 (3 : ℝ)).const_add ((-1 / 2 : ℝ) ^ 2)) h
  refine hd.deriv.trans ?_
  rw [div_left_inj' (pow_ne_zero 2 h)]
  simp only [Nat.cast_ofNat, show (2 : ℕ) - 1 = 1 from rfl, pow_one]
  ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.inv_vy
-- test: ca-holo-inv-vy-test.sage

/-- `u_x + i v_x = -1/z²`: its product with `z²` is `-1`. -/
theorem inv_deriv_parts (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) :
    ((x : ℂ) + y * I) ^ 2 *
        ((((y ^ 2 - x ^ 2) / (x ^ 2 + y ^ 2) ^ 2 : ℝ) : ℂ) + (((2 * x * y) / (x ^ 2 + y ^ 2) ^ 2 : ℝ) : ℂ) * I) = -1 := by
  have h' : ((x : ℂ) ^ 2 + (y : ℂ) ^ 2) ≠ 0 := by exact_mod_cast h
  have hI : I ^ 2 = -1 := I_sq
  have e1 : (((y ^ 2 - x ^ 2) / (x ^ 2 + y ^ 2) ^ 2 : ℝ) : ℂ) =
      ((y : ℂ) ^ 2 - (x : ℂ) ^ 2) * (((x : ℂ) ^ 2 + (y : ℂ) ^ 2) ^ 2)⁻¹ := by push_cast <;> ring
  have e2 : (((2 * x * y) / (x ^ 2 + y ^ 2) ^ 2 : ℝ) : ℂ) =
      2 * (x : ℂ) * (y : ℂ) * (((x : ℂ) ^ 2 + (y : ℂ) ^ 2) ^ 2)⁻¹ := by push_cast <;> ring
  rw [e1, e2]
  clear e1 e2
  have hinv : ((x : ℂ) ^ 2 + (y : ℂ) ^ 2) ^ 2 * (((x : ℂ) ^ 2 + (y : ℂ) ^ 2) ^ 2)⁻¹ = 1 :=
    mul_inv_cancel₀ (pow_ne_zero 2 h')
  generalize (((x : ℂ) ^ 2 + (y : ℂ) ^ 2) ^ 2)⁻¹ = d at *
  generalize ((x : ℝ) : ℂ) = xC at *
  generalize ((y : ℝ) : ℂ) = yC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified
    (linear_combination (-1) * hinv + (3 * xC ^ 2 * yC ^ 2 * d + 2 * xC * yC ^ 3 * i * d + yC ^ 4 * d) * hI)
  -- certificate: ca-holo-ring.sage

/-- `cos z = cos x cosh y - i sin x sinh y`. -/
theorem cos_parts (x y : ℝ) :
    Complex.cos ((x : ℂ) + y * I) =
      ((Real.cos x * Real.cosh y : ℝ) : ℂ) + ((-(Real.sin x * Real.sinh y) : ℝ) : ℂ) * I := by
  push_cast
  rw [Complex.cos_add_mul_I]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_parts (0) (0) := by
  push_cast
  rw [Complex.cos_add_mul_I]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_parts (1) (-5) := by
  push_cast
  rw [Complex.cos_add_mul_I]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_parts (0) (1) := by
  push_cast
  rw [Complex.cos_add_mul_I]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_parts (0) (0) := by
  push_cast
  rw [Complex.cos_add_mul_I]
  ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_parts
-- test: ca-holo-cos-parts-test.sage

/-- `u_x = -sin x cosh y` for `u = cos x cosh y`. -/
theorem cos_ux (x y : ℝ) : deriv (fun x => Real.cos x * Real.cosh y) x = -(Real.sin x * Real.cosh y) := by
  generalize Real.cosh y = C
  veriphy_certified
    ( have hd : HasDerivAt (fun x => Real.cos x * C) ((-C * (Real.sin x) : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).cos).const_mul (C : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ca-holo-deriv.sage

/-- `u_y = cos x sinh y` for `u = cos x cosh y`. -/
theorem cos_uy (x y : ℝ) : deriv (fun y => Real.cos x * Real.cosh y) y = Real.cos x * Real.sinh y := by
  exact ((Real.hasDerivAt_cosh y).const_mul (Real.cos x)).deriv

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_uy (0) (0) := by
  exact ((Real.hasDerivAt_cosh _).const_mul _).deriv

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_uy (1) (-5) := by
  exact ((Real.hasDerivAt_cosh _).const_mul _).deriv

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_uy (0) (1) := by
  exact ((Real.hasDerivAt_cosh _).const_mul _).deriv

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_uy (0) (0) := by
  exact ((Real.hasDerivAt_cosh _).const_mul _).deriv

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_uy
-- test: ca-holo-cos-uy-test.sage

/-- `v_x = -cos x sinh y` for `v = -sin x sinh y`. -/
theorem cos_vx (x y : ℝ) : deriv (fun x => -(Real.sin x * Real.sinh y)) x = -(Real.cos x * Real.sinh y) := by
  generalize Real.sinh y = S
  veriphy_certified
    ( have hd : HasDerivAt (fun x => -(Real.sin x * S)) ((-S * (Real.cos x) : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).sin).const_mul (-S : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ca-holo-deriv.sage

/-- `v_y = -sin x cosh y` for `v = -sin x sinh y`. -/
theorem cos_vy (x y : ℝ) : deriv (fun y => -(Real.sin x * Real.sinh y)) y = -(Real.sin x * Real.cosh y) := by
  exact ((Real.hasDerivAt_sinh y).const_mul (Real.sin x)).neg.deriv

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_vy (0) (0) := by
  exact ((Real.hasDerivAt_sinh _).const_mul _).neg.deriv

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_vy (1) (-5) := by
  exact ((Real.hasDerivAt_sinh _).const_mul _).neg.deriv

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_vy (0) (1) := by
  exact ((Real.hasDerivAt_sinh _).const_mul _).neg.deriv

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_vy (0) (0) := by
  exact ((Real.hasDerivAt_sinh _).const_mul _).neg.deriv

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_vy
-- test: ca-holo-cos-vy-test.sage

/-- `u_x + i v_x = -sin x cosh y - i cos x sinh y = -sin z`. -/
theorem cos_deriv_parts (x y : ℝ) :
    -Complex.sin ((x : ℂ) + y * I) =
      ((-(Real.sin x * Real.cosh y) : ℝ) : ℂ) + ((-(Real.cos x * Real.sinh y) : ℝ) : ℂ) * I := by
  push_cast
  rw [Complex.sin_add_mul_I]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_deriv_parts (0) (0) := by
  push_cast
  rw [Complex.sin_add_mul_I]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_deriv_parts (1) (-5) := by
  push_cast
  rw [Complex.sin_add_mul_I]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_deriv_parts (0) (1) := by
  push_cast
  rw [Complex.sin_add_mul_I]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_deriv_parts (0) (0) := by
  push_cast
  rw [Complex.sin_add_mul_I]
  ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_deriv_parts
-- test: ca-holo-cos-deriv-test.sage

/-! ## exr-cr-polar (the Wirtinger form)

With `f_x = u_x + i v_x = a + ic` and `f_y = u_y + i v_y = b + id`: `∂_z̄ f = ½(f_x + i f_y)`, `∂_z f = ½(f_x - i f_y)`. -/

/-- `∂_z̄ f = ½[(u_x - v_y) + i(u_y + v_x)]`. -/
theorem wirtinger_bar (a b c d : ℝ) :
    (1 / 2 : ℂ) * (((a : ℂ) + c * I) + I * ((b : ℂ) + d * I)) =
      (1 / 2) * (((a - d : ℝ) : ℂ) + ((b + c : ℝ) : ℂ) * I) := by
  have hI : I ^ 2 = -1 := I_sq
  push_cast
  generalize ((a : ℝ) : ℂ) = aC at *
  generalize ((b : ℝ) : ℂ) = bC at *
  generalize ((c : ℝ) : ℂ) = cC at *
  generalize ((d : ℝ) : ℂ) = dC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (1 / 2 * dC) * hI)
  -- certificate: ca-holo-ring.sage

/-- `∂_z̄ f = 0` exactly when both Cauchy-Riemann equations `u_x = v_y`, `u_y = -v_x` hold. -/
theorem wirtinger_bar_zero_iff (a b c d : ℝ) :
    (1 / 2 : ℂ) * (((a : ℂ) + c * I) + I * ((b : ℂ) + d * I)) = 0 ↔ a = d ∧ b = -c := by
  rw [wirtinger_bar, mul_eq_zero, or_iff_right (by norm_num), Complex.ext_iff]
  simp only [add_re, ofReal_re, mul_re, I_re, mul_zero, ofReal_im, I_im, mul_one, sub_zero, add_im, mul_im,
    zero_add, add_zero, zero_re, zero_im, zero_mul, sub_self]
  constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.wirtinger_bar_zero_iff (0) (0) (0) (0) := by
  norm_num [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.wirtinger_bar_zero_iff (1) (-5) ((-1 / 5)) (0) := by
  norm_num [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.wirtinger_bar_zero_iff (0) (0) (0) (1) := by
  norm_num [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.wirtinger_bar_zero_iff (0) (0) (0) (1) := by
  norm_num [Complex.ext_iff]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.wirtinger_bar_zero_iff
-- test: ca-holo-wirtinger-test.sage

/-- Under the Cauchy-Riemann equations `∂_z f = ½[(u_x + v_y) + i(v_x - u_y)] = u_x + i v_x = f'`. -/
theorem wirtinger_z (a b c d : ℝ) (h1 : a = d) (h2 : b = -c) :
    (1 / 2 : ℂ) * (((a : ℂ) + c * I) - I * ((b : ℂ) + d * I)) = (a : ℂ) + c * I := by
  have h1' : (a : ℂ) = d := by exact_mod_cast h1
  have h2' : (b : ℂ) = -c := by exact_mod_cast h2
  have hI : I ^ 2 = -1 := I_sq
  generalize ((a : ℝ) : ℂ) = aC at *
  generalize ((b : ℝ) : ℂ) = bC at *
  generalize ((c : ℝ) : ℂ) = cC at *
  generalize ((d : ℝ) : ℂ) = dC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-1 / 2 * i) * h2' + (-1 / 2) * h1' + (-1 / 2 * dC) * hI)
  -- certificate: ca-holo-ring.sage

/-- `∂_z̄(z²) = 0` (`u_x = 2x`, `u_y = -2y`, `v_x = 2y`, `v_y = 2x`). -/
theorem wirtinger_sq_bar (x y : ℝ) :
    (1 / 2 : ℂ) * ((((2 * x : ℝ) : ℂ) + ((2 * y : ℝ) : ℂ) * I) + I * (((-2 * y : ℝ) : ℂ) + ((2 * x : ℝ) : ℂ) * I)) = 0 := by
  have hI : I ^ 2 = -1 := I_sq
  have e1 : ((2 * x : ℝ) : ℂ) = 2 * (x : ℂ) := by push_cast <;> ring
  have e2 : ((2 * y : ℝ) : ℂ) = 2 * (y : ℂ) := by push_cast <;> ring
  have e3 : ((-2 * y : ℝ) : ℂ) = -2 * (y : ℂ) := by push_cast <;> ring
  rw [e1, e2, e3]
  clear e1 e2 e3
  generalize ((x : ℝ) : ℂ) = xC at *
  generalize ((y : ℝ) : ℂ) = yC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (xC) * hI)
  -- certificate: ca-holo-ring.sage

/-- `∂_z(z²) = 2z`. -/
theorem wirtinger_sq (x y : ℝ) :
    (1 / 2 : ℂ) * ((((2 * x : ℝ) : ℂ) + ((2 * y : ℝ) : ℂ) * I) - I * (((-2 * y : ℝ) : ℂ) + ((2 * x : ℝ) : ℂ) * I)) =
      2 * ((x : ℂ) + y * I) := by
  have hI : I ^ 2 = -1 := I_sq
  have e1 : ((2 * x : ℝ) : ℂ) = 2 * (x : ℂ) := by push_cast <;> ring
  have e2 : ((2 * y : ℝ) : ℂ) = 2 * (y : ℂ) := by push_cast <;> ring
  have e3 : ((-2 * y : ℝ) : ℂ) = -2 * (y : ℂ) := by push_cast <;> ring
  rw [e1, e2, e3]
  clear e1 e2 e3
  generalize ((x : ℝ) : ℂ) = xC at *
  generalize ((y : ℝ) : ℂ) = yC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-xC) * hI)
  -- certificate: ca-holo-ring.sage

/-- `∂_z̄(z z̄) = z` (`u = x² + y²`, `v = 0`: `u_x = 2x`, `u_y = 2y`, `v_x = v_y = 0`), non-zero off the origin. -/
theorem wirtinger_normSq_bar (x y : ℝ) :
    (1 / 2 : ℂ) * ((((2 * x : ℝ) : ℂ) + ((0 : ℝ) : ℂ) * I) + I * (((2 * y : ℝ) : ℂ) + ((0 : ℝ) : ℂ) * I)) =
      (x : ℂ) + y * I := by
  have hI : I ^ 2 = -1 := I_sq
  have e1 : ((2 * x : ℝ) : ℂ) = 2 * (x : ℂ) := by push_cast <;> ring
  have e2 : ((2 * y : ℝ) : ℂ) = 2 * (y : ℂ) := by push_cast <;> ring
  have e0 : ((0 : ℝ) : ℂ) = 0 := Complex.ofReal_zero
  rw [e1, e2, e0]
  clear e1 e2 e0
  generalize ((x : ℝ) : ℂ) = xC at *
  generalize ((y : ℝ) : ℂ) = yC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-! ## exr-cr-constant -/

/-- A real-valued function complex differentiable at an interior point `z` of the set where it is real has
`f'(z) = 0` (`v = 0` forces `u_x = v_y = 0`, `v_x = 0`). -/
theorem deriv_eq_zero_of_real (f : ℂ → ℂ) (U : Set ℂ) (hU : IsOpen U) (z : ℂ) (hz : z ∈ U)
    (hf : DifferentiableAt ℂ f z) (hr : ∀ w ∈ U, (f w).im = 0) : deriv f z = 0 := by
  have hL := hf.hasDerivAt.hasFDerivAt.restrictScalars ℝ
  have h1 := imCLM.hasFDerivAt.comp z hL
  have h2 : HasFDerivAt (fun w => (f w).im) (0 : ℂ →L[ℝ] ℝ) z :=
    (hasFDerivAt_const (0 : ℝ) z).congr_of_eventuallyEq
      (Filter.eventually_of_mem (hU.mem_nhds hz) fun w hw => hr w hw)
  have h3 := h1.unique h2
  have e1 := congrArg (fun L : ℂ →L[ℝ] ℝ => L 1) h3
  have e2 := congrArg (fun L : ℂ →L[ℝ] ℝ => L I) h3
  simp at e1 e2
  exact Complex.ext (by simpa using e2) (by simpa using e1)

/-- A holomorphic function with real values on a connected open set is constant. -/
theorem real_valued_const (f : ℂ → ℂ) (U : Set ℂ) (hU : IsOpen U) (hc : IsPreconnected U)
    (hf : DifferentiableOn ℂ f U) (hr : ∀ w ∈ U, (f w).im = 0) : ∀ z ∈ U, ∀ w ∈ U, f z = f w :=
  fun _ hz _ hw => hU.is_const_of_deriv_eq_zero hc hf
    (fun x hx => deriv_eq_zero_of_real f U hU x hx (hf.differentiableAt (hU.mem_nhds hx)) hr) hz hw

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.real_valued_const ((fun _ : ℂ => (0 : ℂ))) (Set.univ) (isOpen_univ) (isPreconnected_univ) ((differentiableOn_const 0)) ((by simp)) (0) ((Set.mem_univ _)) (1) ((Set.mem_univ _)) := by
  rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.real_valued_const ((fun _ : ℂ => (1 : ℂ))) ((Metric.ball 0 1)) (Metric.isOpen_ball) ((convex_ball (0 : ℂ) 1).isPreconnected) ((differentiableOn_const 1)) ((by simp)) (0) ((Metric.mem_ball_self one_pos)) ((1 / 2)) ((by simp; norm_num)) := by
  rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.real_valued_const ((fun _ : ℂ => (-2 : ℂ))) (Set.univ) (isOpen_univ) (isPreconnected_univ) ((differentiableOn_const (-2))) ((by simp)) (Complex.I) ((Set.mem_univ _)) (2) ((Set.mem_univ _)) := by
  rfl

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.real_valued_const
-- test: ca-holo-real-const-test.sage

/-- A holomorphic function with vanishing derivative on a connected open set is constant. -/
theorem deriv_zero_const (f : ℂ → ℂ) (U : Set ℂ) (hU : IsOpen U) (hc : IsPreconnected U)
    (hf : DifferentiableOn ℂ f U) (hd : ∀ z ∈ U, deriv f z = 0) : ∀ z ∈ U, ∀ w ∈ U, f z = f w :=
  fun _ hz _ hw => hU.is_const_of_deriv_eq_zero hc hf (fun x hx => hd x hx) hz hw

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.deriv_zero_const ((fun _ : ℂ => (0 : ℂ))) (Set.univ) (isOpen_univ) (isPreconnected_univ) ((differentiableOn_const 0)) ((by simp)) (0) ((Set.mem_univ _)) (1) ((Set.mem_univ _)) := by
  rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.deriv_zero_const ((fun _ : ℂ => (1 : ℂ))) ((Metric.ball 0 1)) (Metric.isOpen_ball) ((convex_ball (0 : ℂ) 1).isPreconnected) ((differentiableOn_const 1)) ((by simp)) (0) ((Metric.mem_ball_self one_pos)) ((1 / 2)) ((by simp; norm_num)) := by
  rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyRiemann.deriv_zero_const ((fun _ : ℂ => Complex.I)) (Set.univ) (isOpen_univ) (isPreconnected_univ) ((differentiableOn_const Complex.I)) ((by simp)) (Complex.I) ((Set.mem_univ _)) (2) ((Set.mem_univ _)) := by
  rfl

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.CauchyRiemann.deriv_zero_const
-- test: ca-holo-deriv-zero-test.sage

end KaluzaKlein.ComplexAnalysis.CauchyRiemann
