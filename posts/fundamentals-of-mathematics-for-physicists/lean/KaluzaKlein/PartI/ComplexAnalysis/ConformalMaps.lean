import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.ComplexDerivative
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Conformal maps"

The claims of the section "Conformal maps" of `tutorial/part-i-04-complex-analysis.qmd`. Conformality at a point is
mathlib's `ConformalAt` (the real derivative is a non-zero multiple of a linear isometry), angles are
`InnerProductGeometry.angle` in `ℂ ≅ ℝ²`. The section shows: a holomorphic map with `f' ≠ 0` is conformal, multiplies
displacements by `f' = |f'| e^{i arg f'}` and preserves angles; conformal maps are the holomorphic or antiholomorphic
ones; `z²` maps the grid to orthogonal parabolas and doubles angles at `0`; the Cayley map `(z - i)/(z + i)` maps the
real axis to the unit circle and the upper half-plane bijectively onto the disc, and Möbius maps send generalised
circles to generalised circles; `eᶻ` maps lines to rays and circles and the strip `0 < Im z < π` onto the upper
half-plane; the angle computation for `z²` at `1 + i`; angle multiplication by `n` at a zero of order `n` of `f'`.
-/

namespace KaluzaKlein.ComplexAnalysis.ConformalMaps

open Complex Filter Topology ComplexConjugate Metric
open scoped Real

/-! ## def-conformal -/

/-- A holomorphic map with `f'(z₀) ≠ 0` is conformal at `z₀`. -/
theorem conformal_of_deriv_ne_zero (f : ℂ → ℂ) (z : ℂ) (h : DifferentiableAt ℂ f z) (h' : deriv f z ≠ 0) :
    ConformalAt f z :=
  h.conformalAt h'

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.conformal_of_deriv_ne_zero ((fun w : ℂ => w)) (0) ((by fun_prop)) ((by simp)) := by
  exact (by fun_prop : DifferentiableAt ℂ ((fun w : ℂ => w)) (0 : ℂ)).conformalAt (by simp)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.conformal_of_deriv_ne_zero (Complex.exp) (1) ((by fun_prop)) ((by simp)) := by
  exact (by fun_prop : DifferentiableAt ℂ (Complex.exp) (1 : ℂ)).conformalAt (by simp)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.conformal_of_deriv_ne_zero ((fun w : ℂ => w ^ 2)) (1) ((by fun_prop)) ((by simp)) := by
  exact (by fun_prop : DifferentiableAt ℂ (fun w : ℂ => w ^ 2) (1 : ℂ)).conformalAt (by simp)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.conformal_of_deriv_ne_zero
-- test: ca-holo-conformal-test.sage

/-- Multiplication by `a ≠ 0` (the first-order action of a holomorphic map with `f' = a`) preserves the angle
between any two displacements. -/
theorem mul_preserves_angle (a : ℂ) (ha : a ≠ 0) (u v : ℂ) :
    InnerProductGeometry.angle (a * u) (a * v) = InnerProductGeometry.angle u v := by
  have hm : (ContinuousLinearMap.mul ℂ ℂ a) ≠ 0 := by
    intro h
    have := congrArg (fun L : ℂ →L[ℂ] ℂ => L 1) h
    simp at this <;> exact ha this
  have hc := InnerProductGeometry.IsConformalMap.preserves_angle (isConformalMap_complex_linear hm) u v
  first | exact hc | simpa using hc

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.mul_preserves_angle (1) (one_ne_zero) (1) (Complex.I) := by
  have ha := (one_ne_zero : (1 : ℂ) ≠ 0)
  have hm : (ContinuousLinearMap.mul ℂ ℂ (1 : ℂ)) ≠ 0 := by
    intro h
    have := congrArg (fun L : ℂ →L[ℂ] ℂ => L 1) h
    simp at this <;> exact ha this
  have hc := InnerProductGeometry.IsConformalMap.preserves_angle (isConformalMap_complex_linear hm) (1 : ℂ) (Complex.I)
  first | exact hc | simpa using hc

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.mul_preserves_angle (Complex.I) (Complex.I_ne_zero) (1) ((1 + Complex.I)) := by
  have ha := (Complex.I_ne_zero)
  have hm : (ContinuousLinearMap.mul ℂ ℂ (Complex.I)) ≠ 0 := by
    intro h
    have := congrArg (fun L : ℂ →L[ℂ] ℂ => L 1) h
    simp at this <;> exact ha this
  have hc := InnerProductGeometry.IsConformalMap.preserves_angle (isConformalMap_complex_linear hm) (1 : ℂ) (1 + Complex.I)
  first | exact hc | simpa using hc

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.mul_preserves_angle ((2 + Complex.I)) ((by norm_num [Complex.ext_iff])) (Complex.I) (3) := by
  have ha := (by norm_num [Complex.ext_iff] : (2 + Complex.I : ℂ) ≠ 0)
  have hm : (ContinuousLinearMap.mul ℂ ℂ (2 + Complex.I)) ≠ 0 := by
    intro h
    have := congrArg (fun L : ℂ →L[ℂ] ℂ => L 1) h
    simp at this <;> exact ha this
  have hc := InnerProductGeometry.IsConformalMap.preserves_angle (isConformalMap_complex_linear hm) (Complex.I) (3 : ℂ)
  first | exact hc | simpa using hc

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.mul_preserves_angle
-- test: ca-holo-angle-test.sage

/-- `f' h = |f'| (e^{i arg f'} h)`: a rotation by `arg f'` followed by a scaling by `|f'|`. -/
theorem rotation_scaling (a h : ℂ) : a * h = (‖a‖ : ℂ) * (Complex.exp (arg a * I) * h) := by
  rw [← mul_assoc, norm_mul_exp_arg_mul_I]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.rotation_scaling (0) (0) := by
  rw [← mul_assoc, Complex.norm_mul_exp_arg_mul_I]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.rotation_scaling ((1 + Complex.I)) (2) := by
  rw [← mul_assoc, Complex.norm_mul_exp_arg_mul_I]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.rotation_scaling (Complex.I) (Complex.I) := by
  rw [← mul_assoc, Complex.norm_mul_exp_arg_mul_I]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.rotation_scaling
-- test: ca-holo-rotation-scaling-test.sage

/-- Conversely, a conformal map is holomorphic or antiholomorphic (holomorphic after `z ↦ z̄`), with non-zero real
derivative; the antiholomorphic ones reverse orientation (the reflection `z̄` has determinant `-1`), so an
orientation-preserving conformal map is holomorphic. -/
theorem conformal_iff (f : ℂ → ℂ) (z : ℂ) :
    ConformalAt f z ↔
      (DifferentiableAt ℂ f z ∨ DifferentiableAt ℂ (f ∘ conj) (conj z)) ∧ fderiv ℝ f z ≠ 0 :=
  conformalAt_iff_differentiableAt_or_differentiableAt_comp_conj

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.conformal_iff ((fun w : ℂ => w)) (0) := by
  exact conformalAt_iff_differentiableAt_or_differentiableAt_comp_conj

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.conformal_iff ((fun w : ℂ => (starRingEnd ℂ) w)) (1) := by
  exact conformalAt_iff_differentiableAt_or_differentiableAt_comp_conj

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.conformal_iff (Complex.exp) (Complex.I) := by
  exact conformalAt_iff_differentiableAt_or_differentiableAt_comp_conj

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.conformal_iff
-- test: ca-holo-conformal-iff-test.sage

/-- The reflection `z ↦ z̄` has determinant `-1`. -/
theorem det_conj : LinearMap.det (conjLIE.toLinearEquiv : ℂ →ₗ[ℝ] ℂ) = -1 :=
  Complex.det_conjLIE

/-- At a zero of `f'` a holomorphic map is not conformal (its real derivative vanishes): `(z - 1)/(z² + 1)` at
`1 ± √2`, `z²` at `0`. -/
theorem not_conformalAt_of_deriv_eq_zero (f : ℂ → ℂ) (z : ℂ) (hf : DifferentiableAt ℂ f z) (h : deriv f z = 0) :
    ¬ ConformalAt f z := by
  rintro ⟨f', hd, hc⟩
  have h0 : HasFDerivAt f (0 : ℂ →L[ℝ] ℂ) z := by
    have := hf.hasDerivAt.hasFDerivAt.restrictScalars ℝ
    convert this using 1
    ext w
    simp [h]
  exact hc.ne_zero (hd.unique h0)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.not_conformalAt_of_deriv_eq_zero ((fun w : ℂ => w ^ 2)) (0) ((by fun_prop)) ((by simp)) := by
  rintro ⟨f', hd, hc⟩
  have hf : DifferentiableAt ℂ (fun w : ℂ => w ^ 2) (0 : ℂ) := by fun_prop
  have h : deriv (fun w : ℂ => w ^ 2) (0 : ℂ) = 0 := by simp
  have h0 : HasFDerivAt (fun w : ℂ => w ^ 2) (0 : ℂ →L[ℝ] ℂ) (0 : ℂ) := by
    have := hf.hasDerivAt.hasFDerivAt.restrictScalars ℝ
    convert this using 1
    ext w
    simp [h]
  exact hc.ne_zero (hd.unique h0)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.not_conformalAt_of_deriv_eq_zero ((fun w : ℂ => w ^ 3)) (0) ((by fun_prop)) ((by simp)) := by
  rintro ⟨f', hd, hc⟩
  have hf : DifferentiableAt ℂ (fun w : ℂ => w ^ 3) (0 : ℂ) := by fun_prop
  have h : deriv (fun w : ℂ => w ^ 3) (0 : ℂ) = 0 := by simp
  have h0 : HasFDerivAt (fun w : ℂ => w ^ 3) (0 : ℂ →L[ℝ] ℂ) (0 : ℂ) := by
    have := hf.hasDerivAt.hasFDerivAt.restrictScalars ℝ
    convert this using 1
    ext w
    simp [h]
  exact hc.ne_zero (hd.unique h0)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.not_conformalAt_of_deriv_eq_zero ((fun _ : ℂ => (1 : ℂ))) (Complex.I) ((differentiableAt_const _)) ((by simp)) := by
  rintro ⟨f', hd, hc⟩
  have hf : DifferentiableAt ℂ (fun _ : ℂ => (1 : ℂ)) (Complex.I) := differentiableAt_const _
  have h : deriv (fun _ : ℂ => (1 : ℂ)) (Complex.I) = 0 := by simp
  have h0 : HasFDerivAt (fun _ : ℂ => (1 : ℂ)) (0 : ℂ →L[ℝ] ℂ) (Complex.I) := by
    have := hf.hasDerivAt.hasFDerivAt.restrictScalars ℝ
    convert this using 1
    ext w
    simp [h]
  exact hc.ne_zero (hd.unique h0)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.not_conformalAt_of_deriv_eq_zero
-- test: ca-holo-not-conformal-test.sage

/-! ## exm-conformal-square -/

/-- The vertical line `Re z = c` goes to the parabola `u = c² - t²`, `v = 2ct`. -/
theorem image_vertical (c t : ℝ) : ((c : ℂ) + t * I) ^ 2 = ((c ^ 2 - t ^ 2 : ℝ) : ℂ) + ((2 * c * t : ℝ) : ℂ) * I := by
  have hI : I ^ 2 = -1 := I_sq
  have e1 : ((c ^ 2 - t ^ 2 : ℝ) : ℂ) = (c : ℂ) ^ 2 - (t : ℂ) ^ 2 := by push_cast <;> ring
  have e2 : ((2 * c * t : ℝ) : ℂ) = 2 * (c : ℂ) * (t : ℂ) := by push_cast <;> ring
  rw [e1, e2]
  clear e1 e2
  generalize ((c : ℝ) : ℂ) = cC at *
  generalize ((t : ℝ) : ℂ) = tC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (tC ^ 2) * hI)
  -- certificate: ca-holo-ring.sage

/-- The horizontal line `Im z = c` goes to the parabola `u = t² - c²`, `v = 2tc`. -/
theorem image_horizontal (t c : ℝ) :
    ((t : ℂ) + c * I) ^ 2 = ((t ^ 2 - c ^ 2 : ℝ) : ℂ) + ((2 * t * c : ℝ) : ℂ) * I := by
  have hI : I ^ 2 = -1 := I_sq
  have e1 : ((t ^ 2 - c ^ 2 : ℝ) : ℂ) = (t : ℂ) ^ 2 - (c : ℂ) ^ 2 := by push_cast <;> ring
  have e2 : ((2 * t * c : ℝ) : ℂ) = 2 * (t : ℂ) * (c : ℂ) := by push_cast <;> ring
  rw [e1, e2]
  clear e1 e2
  generalize ((t : ℝ) : ℂ) = tC at *
  generalize ((c : ℝ) : ℂ) = cC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (cC ^ 2) * hI)
  -- certificate: ca-holo-ring.sage

/-- At the image of `a + ib` the two parabolas have tangents `(-2b, 2a)` and `(2a, 2b)` (the derivatives
`CauchyRiemann.sq_uy`, `sq_vy`, `sq_ux`, `sq_vx`), which are orthogonal. -/
theorem parabolas_orthogonal (a b : ℝ) : (-2 * b) * (2 * a) + (2 * a) * (2 * b) = 0 := by
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- The tangents are non-zero away from `z = 0`. -/
theorem parabola_tangent_ne_zero (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : 0 < (-2 * b) ^ 2 + (2 * a) ^ 2 := by
  rcases h with h | h
  · have : 0 < (2 * a) ^ 2 := by positivity
    nlinarith [sq_nonneg (-2 * b)]
  · have : 0 < (-2 * b) ^ 2 := by positivity
    nlinarith [sq_nonneg (2 * a)]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.parabola_tangent_ne_zero (1) (0) ((Or.inl one_ne_zero)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.parabola_tangent_ne_zero (0) (2) ((Or.inr (by norm_num))) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.parabola_tangent_ne_zero ((-1)) (3) ((Or.inl (by norm_num))) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.parabola_tangent_ne_zero
-- test: ca-holo-tangent-ne-zero-test.sage

/-- At `0` the two half-axes `t > 0` and `-t < 0` (at angle `π`) are both sent to `t²` on the positive real axis. -/
theorem half_axes (t : ℝ) : (t : ℂ) ^ 2 = ((t ^ 2 : ℝ) : ℂ) ∧ (-(t : ℂ)) ^ 2 = ((t ^ 2 : ℝ) : ℂ) := by
  push_cast
  refine ⟨?_, ?_⟩
  · generalize ((t : ℝ) : ℂ) = tC at *
    generalize Complex.I = i at *
    rw [← sub_eq_zero]
    veriphy_certified (ring1)
    -- certificate: ca-holo-ring.sage
  · generalize ((t : ℝ) : ℂ) = tC at *
    generalize Complex.I = i at *
    rw [← sub_eq_zero]
    veriphy_certified (ring1)
    -- certificate: ca-holo-ring.sage

/-! ## exm-conformal-mobius -/

/-- The Cayley map `f(z) = (z - i)/(z + i)`. -/
noncomputable def cayley (w : ℂ) : ℂ := (w - I) / (w + I)

/-- `2i/(z + i)²`. -/
noncomputable def cayleyDeriv (w : ℂ) : ℂ := 2 * I / (w + I) ^ 2

/-- `f(z) = (z - i)/(z + i)` is holomorphic on `ℂ ∖ {-i}`. -/
theorem cayley_differentiableAt (z : ℂ) (hz : z + I ≠ 0) : DifferentiableAt ℂ cayley z := by
  show DifferentiableAt ℂ (fun w : ℂ => (w - I) / (w + I)) z
  exact DifferentiableAt.div (by fun_prop) (by fun_prop) hz

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_differentiableAt (0) ((by simp)) := by
  show DifferentiableAt ℂ (fun w : ℂ => (w - Complex.I) / (w + Complex.I)) (0 : ℂ)
  exact DifferentiableAt.div (by fun_prop) (by fun_prop) (by simp)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_differentiableAt (1) ((by norm_num [Complex.ext_iff])) := by
  show DifferentiableAt ℂ (fun w : ℂ => (w - Complex.I) / (w + Complex.I)) (1 : ℂ)
  exact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_differentiableAt (Complex.I) ((by norm_num [Complex.ext_iff])) := by
  show DifferentiableAt ℂ (fun w : ℂ => (w - Complex.I) / (w + Complex.I)) (Complex.I)
  exact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_differentiableAt
-- test: ca-holo-cayley-holo-test.sage

/-- `f(z) = (z - i)/(z + i)` has `f'(z) = 2i/(z + i)²`. -/
theorem mobius_deriv (z : ℂ) (hz : z + I ≠ 0) : deriv cayley z = cayleyDeriv z := by
  show deriv (fun w : ℂ => (w - I) / (w + I)) z = 2 * I / (z + I) ^ 2
  have h := ((hasDerivAt_id' z).sub_const I).div ((hasDerivAt_id' z).add_const I) hz
  refine h.deriv.trans ?_
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.mobius_deriv (0) ((by norm_num [Complex.ext_iff])) := by
  have hz := (by norm_num [Complex.ext_iff] : (0 : ℂ) + Complex.I ≠ 0)
  show deriv (fun w : ℂ => (w - I) / (w + I)) (0 : ℂ) = 2 * I / ((0 : ℂ) + I) ^ 2
  have h := ((hasDerivAt_id' (0 : ℂ)).sub_const I).div ((hasDerivAt_id' (0 : ℂ)).add_const I) hz
  refine h.deriv.trans ?_
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.mobius_deriv (1) ((by norm_num [Complex.ext_iff])) := by
  have hz := (by norm_num [Complex.ext_iff] : (1 : ℂ) + Complex.I ≠ 0)
  show deriv (fun w : ℂ => (w - I) / (w + I)) (1 : ℂ) = 2 * I / ((1 : ℂ) + I) ^ 2
  have h := ((hasDerivAt_id' (1 : ℂ)).sub_const I).div ((hasDerivAt_id' (1 : ℂ)).add_const I) hz
  refine h.deriv.trans ?_
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.mobius_deriv (Complex.I) ((by norm_num [Complex.ext_iff])) := by
  have hz := (by norm_num [Complex.ext_iff] : Complex.I + Complex.I ≠ 0)
  show deriv (fun w : ℂ => (w - I) / (w + I)) (Complex.I) = 2 * I / ((Complex.I) + I) ^ 2
  have h := ((hasDerivAt_id' (Complex.I)).sub_const I).div ((hasDerivAt_id' (Complex.I)).add_const I) hz
  refine h.deriv.trans ?_
  ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.mobius_deriv
-- test: ca-holo-mobius-deriv-test.sage

/-- `f'(z) = 2i/(z + i)² ≠ 0`. -/
theorem mobius_deriv_ne_zero (z : ℂ) (hz : z + I ≠ 0) : cayleyDeriv z ≠ 0 :=
  div_ne_zero (mul_ne_zero two_ne_zero I_ne_zero) (pow_ne_zero 2 hz)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.mobius_deriv_ne_zero (0) ((by norm_num [Complex.ext_iff])) := by
  exact div_ne_zero (mul_ne_zero two_ne_zero Complex.I_ne_zero) (pow_ne_zero 2 (by norm_num [Complex.ext_iff]))

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.mobius_deriv_ne_zero (1) ((by norm_num [Complex.ext_iff])) := by
  exact div_ne_zero (mul_ne_zero two_ne_zero Complex.I_ne_zero) (pow_ne_zero 2 (by norm_num [Complex.ext_iff]))

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.mobius_deriv_ne_zero (Complex.I) ((by norm_num [Complex.ext_iff])) := by
  exact div_ne_zero (mul_ne_zero two_ne_zero Complex.I_ne_zero) (pow_ne_zero 2 (by norm_num [Complex.ext_iff]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.mobius_deriv_ne_zero
-- test: ca-holo-mobius-ne-zero-test.sage

/-- For real `x`, `|x - i|² = x² + 1 = |x + i|²`. -/
theorem real_axis_equidistant (x : ℝ) :
    Complex.normSq ((x : ℂ) - I) = x ^ 2 + 1 ∧ Complex.normSq ((x : ℂ) + I) = x ^ 2 + 1 := by
  constructor <;> simp [Complex.normSq_apply] <;> ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (0) := by
  constructor <;> simp [Complex.normSq_apply] <;> ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (1) := by
  constructor <;> simp [Complex.normSq_apply] <;> ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (1) := by
  constructor <;> simp [Complex.normSq_apply] <;> ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant
-- test: ca-holo-equidistant-test.sage

/-- So `|f(x)| = 1`: the real axis goes into the unit circle. -/
theorem real_axis_to_circle (x : ℝ) : ‖((x : ℂ) - I) / ((x : ℂ) + I)‖ = 1 := by
  have h1 : ‖(x : ℂ) - I‖ = ‖(x : ℂ) + I‖ := by
    rw [Complex.norm_def, Complex.norm_def, (real_axis_equidistant x).1, (real_axis_equidistant x).2]
  have h2 : (x : ℂ) + I ≠ 0 := by
    intro h
    have := congrArg Complex.im h
    simp at this
  rw [norm_div, h1, div_self (norm_ne_zero_iff.mpr h2)]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_to_circle (0) := by
  have h1 : ‖((0 : ℝ) : ℂ) - Complex.I‖ = ‖((0 : ℝ) : ℂ) + Complex.I‖ := by
    rw [Complex.norm_def, Complex.norm_def, (KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (0)).1, (KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (0)).2]
  have h2 : ((0 : ℝ) : ℂ) + Complex.I ≠ 0 := by
    intro h
    have := congrArg Complex.im h
    simp at this
  rw [norm_div, h1, div_self (norm_ne_zero_iff.mpr h2)]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_to_circle (1) := by
  have h1 : ‖((1 : ℝ) : ℂ) - Complex.I‖ = ‖((1 : ℝ) : ℂ) + Complex.I‖ := by
    rw [Complex.norm_def, Complex.norm_def, (KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (1)).1, (KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (1)).2]
  have h2 : ((1 : ℝ) : ℂ) + Complex.I ≠ 0 := by
    intro h
    have := congrArg Complex.im h
    simp at this
  rw [norm_div, h1, div_self (norm_ne_zero_iff.mpr h2)]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_to_circle ((-3 / 2)) := by
  have h1 : ‖((-3 / 2 : ℝ) : ℂ) - Complex.I‖ = ‖((-3 / 2 : ℝ) : ℂ) + Complex.I‖ := by
    rw [Complex.norm_def, Complex.norm_def, (KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (-3 / 2)).1, (KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant (-3 / 2)).2]
  have h2 : ((-3 / 2 : ℝ) : ℂ) + Complex.I ≠ 0 := by
    intro h
    have := congrArg Complex.im h
    simp at this
  rw [norm_div, h1, div_self (norm_ne_zero_iff.mpr h2)]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_to_circle
-- test: ca-holo-real-axis-test.sage

/-- A Möbius map with `c ≠ 0` is a composition of affine maps and the inversion `w ↦ 1/w`:
`(az + b)/(cz + d) = a/c + (bc - ad)/(c (cz + d))`. -/
theorem mobius_decomposition (a b c d z : ℂ) (hc : c ≠ 0) (hz : c * z + d ≠ 0) :
    (a * z + b) / (c * z + d) = a / c + (b * c - a * d) / (c * (c * z + d)) := by
  rw [div_add_div _ _ hc (mul_ne_zero hc hz), div_eq_div_iff hz (mul_ne_zero hc (mul_ne_zero hc hz))]
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- The generalised circle `A|z|² + 2 Re(B z̄) + C = 0` (`A, C` real; a line when `A = 0`). -/
noncomputable def gcircle (A : ℝ) (B : ℂ) (C : ℝ) (z : ℂ) : ℝ := A * Complex.normSq z + 2 * (B * conj z).re + C

/-- The inversion maps generalised circles to generalised circles: `gcircle A B C (1/z) = gcircle C B̄ A z / |z|²`. -/
theorem inversion_gcircle (A : ℝ) (B : ℂ) (C : ℝ) (z : ℂ) (hz : z ≠ 0) :
    gcircle A B C z⁻¹ = gcircle C (conj B) A z / Complex.normSq z := by
  have hn : Complex.normSq z ≠ 0 := Complex.normSq_eq_zero.not.mpr hz
  simp only [gcircle, map_inv₀, mul_re, conj_re, conj_im, Complex.inv_re, Complex.inv_im, Complex.normSq_conj]
  field_simp
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.inversion_gcircle (1) (0) ((-1)) (1) (one_ne_zero) := by
  have hz := (one_ne_zero : (1 : ℂ) ≠ 0)
  have hn : Complex.normSq (1 : ℂ) ≠ 0 := Complex.normSq_eq_zero.not.mpr hz
  simp only [gcircle, map_inv₀, mul_re, conj_re, conj_im, Complex.inv_re, Complex.inv_im, Complex.normSq_conj]
  field_simp
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.inversion_gcircle (0) ((1 + Complex.I)) (2) (Complex.I) (Complex.I_ne_zero) := by
  have hz := (Complex.I_ne_zero)
  have hn : Complex.normSq (Complex.I) ≠ 0 := Complex.normSq_eq_zero.not.mpr hz
  simp only [gcircle, map_inv₀, mul_re, conj_re, conj_im, Complex.inv_re, Complex.inv_im, Complex.normSq_conj]
  field_simp
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.inversion_gcircle (2) (Complex.I) ((-3)) ((2 - Complex.I)) ((by norm_num [Complex.ext_iff])) := by
  have hz := (by norm_num [Complex.ext_iff] : (2 - Complex.I : ℂ) ≠ 0)
  have hn : Complex.normSq (2 - Complex.I) ≠ 0 := Complex.normSq_eq_zero.not.mpr hz
  simp only [gcircle, map_inv₀, mul_re, conj_re, conj_im, Complex.inv_re, Complex.inv_im, Complex.normSq_conj]
  field_simp
  ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.inversion_gcircle
-- test: ca-holo-inversion-test.sage

/-- Affine maps `z ↦ az + b` map generalised circles to generalised circles. -/
theorem affine_gcircle (A : ℝ) (B : ℂ) (C : ℝ) (a b z : ℂ) :
    gcircle A B C (a * z + b) =
      gcircle (A * Complex.normSq a) (A * conj a * b + B * conj a)
        (A * Complex.normSq b + 2 * (B * conj b).re + C) z := by
  simp only [gcircle, Complex.normSq_apply, map_add, map_mul, mul_re, mul_im, add_re, add_im, conj_re, conj_im,
    ofReal_re, ofReal_im]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.affine_gcircle (1) (0) ((-1)) (1) (0) (0) := by
  simp only [KaluzaKlein.ComplexAnalysis.ConformalMaps.gcircle, Complex.normSq_apply, map_add, map_mul, Complex.mul_re, Complex.mul_im, Complex.add_re, Complex.add_im,
    Complex.conj_re, Complex.conj_im, Complex.ofReal_re, Complex.ofReal_im]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.affine_gcircle (0) ((1 + Complex.I)) (2) (Complex.I) (1) ((2 - Complex.I)) := by
  simp only [KaluzaKlein.ComplexAnalysis.ConformalMaps.gcircle, Complex.normSq_apply, map_add, map_mul, Complex.mul_re, Complex.mul_im, Complex.add_re, Complex.add_im,
    Complex.conj_re, Complex.conj_im, Complex.ofReal_re, Complex.ofReal_im]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.affine_gcircle (2) (Complex.I) ((-3)) ((1 + Complex.I)) ((-Complex.I)) (3) := by
  simp only [KaluzaKlein.ComplexAnalysis.ConformalMaps.gcircle, Complex.normSq_apply, map_add, map_mul, Complex.mul_re, Complex.mul_im, Complex.add_re, Complex.add_im,
    Complex.conj_re, Complex.conj_im, Complex.ofReal_re, Complex.ofReal_im]
  ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.affine_gcircle
-- test: ca-holo-affine-test.sage

/-! ## exm-conformal-exp -/

/-- `(eᶻ)' = eᶻ ≠ 0`: the exponential is conformal everywhere. -/
theorem exp_conformal (z : ℂ) : deriv Complex.exp z = Complex.exp z ∧ Complex.exp z ≠ 0 ∧ ConformalAt Complex.exp z :=
  ⟨by simp, Complex.exp_ne_zero z,
    Complex.differentiableAt_exp.conformalAt (by simpa using Complex.exp_ne_zero z)⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_conformal (0) := by
  exact ⟨by simp, Complex.exp_ne_zero _, Complex.differentiableAt_exp.conformalAt (by simpa using Complex.exp_ne_zero _)⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_conformal (1) := by
  exact ⟨by simp, Complex.exp_ne_zero _, Complex.differentiableAt_exp.conformalAt (by simpa using Complex.exp_ne_zero _)⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_conformal (Complex.I) := by
  exact ⟨by simp, Complex.exp_ne_zero _, Complex.differentiableAt_exp.conformalAt (by simpa using Complex.exp_ne_zero _)⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_conformal
-- test: ca-holo-exp-conformal-test.sage

/-- The horizontal line `Im z = y₀` goes to the ray of argument `y₀` (`-π < y₀ ≤ π`). -/
theorem exp_horizontal_ray (x y₀ : ℝ) (h₁ : -π < y₀) (h₂ : y₀ ≤ π) : arg (Complex.exp ((x : ℂ) + y₀ * I)) = y₀ := by
  have e : Complex.exp ((x : ℂ) + y₀ * I) = (Real.exp x : ℂ) * (Complex.cos y₀ + Complex.sin y₀ * I) := by
    rw [Complex.exp_add, Complex.exp_mul_I, Complex.ofReal_exp]
  rw [e]
  exact arg_mul_cos_add_sin_mul_I (Real.exp_pos x) ⟨h₁, h₂⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_horizontal_ray (0) (0) ((by linarith [Real.pi_gt_three])) ((by linarith [Real.pi_gt_three])) := by
  have e : Complex.exp (((0 : ℝ) : ℂ) + ((0 : ℝ) : ℂ) * Complex.I) = (Real.exp (0) : ℂ) * (Complex.cos ((0 : ℝ) : ℂ) + Complex.sin ((0 : ℝ) : ℂ) * Complex.I) := by
    rw [Complex.exp_add, Complex.exp_mul_I, Complex.ofReal_exp]
  rw [e]
  exact Complex.arg_mul_cos_add_sin_mul_I (Real.exp_pos _) ⟨by linarith [Real.pi_gt_three], by linarith [Real.pi_gt_three]⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_horizontal_ray (1) (1) ((by linarith [Real.pi_gt_three])) ((by linarith [Real.pi_gt_three])) := by
  have e : Complex.exp (((1 : ℝ) : ℂ) + ((1 : ℝ) : ℂ) * Complex.I) = (Real.exp (1) : ℂ) * (Complex.cos ((1 : ℝ) : ℂ) + Complex.sin ((1 : ℝ) : ℂ) * Complex.I) := by
    rw [Complex.exp_add, Complex.exp_mul_I, Complex.ofReal_exp]
  rw [e]
  exact Complex.arg_mul_cos_add_sin_mul_I (Real.exp_pos _) ⟨by linarith [Real.pi_gt_three], by linarith [Real.pi_gt_three]⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_horizontal_ray ((-1)) (3) ((by linarith [Real.pi_gt_three])) ((by linarith [Real.pi_gt_three])) := by
  have e : Complex.exp (((-1 : ℝ) : ℂ) + ((3 : ℝ) : ℂ) * Complex.I) = (Real.exp (-1) : ℂ) * (Complex.cos ((3 : ℝ) : ℂ) + Complex.sin ((3 : ℝ) : ℂ) * Complex.I) := by
    rw [Complex.exp_add, Complex.exp_mul_I, Complex.ofReal_exp]
  rw [e]
  exact Complex.arg_mul_cos_add_sin_mul_I (Real.exp_pos _) ⟨by linarith [Real.pi_gt_three], by linarith [Real.pi_gt_three]⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_horizontal_ray
-- test: ca-holo-exp-ray-test.sage

/-- The vertical line `Re z = x₀` goes to the circle of radius `e^{x₀}`. -/
theorem exp_vertical_circle (x₀ y : ℝ) : ‖Complex.exp ((x₀ : ℂ) + y * I)‖ = Real.exp x₀ := by
  rw [Complex.norm_exp]
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_vertical_circle (0) (0) := by
  rw [Complex.norm_exp]
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_vertical_circle (1) (-5) := by
  rw [Complex.norm_exp]
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_vertical_circle (0) (1) := by
  rw [Complex.norm_exp]
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_vertical_circle (0) (0) := by
  rw [Complex.norm_exp]
  simp

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_vertical_circle
-- test: ca-holo-exp-circle-test.sage

/-- The strip `0 < Im z < π` is mapped onto the upper half-plane: `w` is an image iff `Im w > 0`. -/
theorem exp_strip_onto (w : ℂ) :
    (∃ z : ℂ, 0 < z.im ∧ z.im < π ∧ Complex.exp z = w) ↔ 0 < w.im := by
  constructor
  · rintro ⟨z, h1, h2, hz⟩
    rw [← hz, Complex.exp_im]
    exact mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi h1 h2)
  · intro hw
    have hw0 : w ≠ 0 := by
      intro h
      rw [h] at hw
      simp at hw
    refine ⟨Complex.log w, ?_, ?_, Complex.exp_log hw0⟩
    · rw [log_im]
      rcases (arg_nonneg_iff.mpr hw.le).lt_or_eq with h | h
      · exact h
      · exact absurd (arg_eq_zero_iff.mp h.symm).2 hw.ne'
    · rw [log_im]
      exact lt_of_le_of_ne (arg_le_pi w) (fun h => by linarith [(arg_eq_pi_iff.mp h).2])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_strip_onto (Complex.I) := by
  constructor
  · rintro ⟨z, h1, h2, hz⟩
    rw [← hz, Complex.exp_im]
    exact mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi h1 h2)
  · intro hw
    have hw0 : (Complex.I) ≠ 0 := Complex.I_ne_zero
    refine ⟨Complex.log (Complex.I), ?_, ?_, Complex.exp_log hw0⟩
    · rw [Complex.log_im]
      exact lt_of_le_of_ne (Complex.arg_nonneg_iff.mpr hw.le) fun h => hw.ne' (Complex.arg_eq_zero_iff.mp h.symm).2
    · rw [Complex.log_im]
      exact lt_of_le_of_ne (Complex.arg_le_pi _) fun h => by linarith [(Complex.arg_eq_pi_iff.mp h).2]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_strip_onto ((1 + Complex.I)) := by
  constructor
  · rintro ⟨z, h1, h2, hz⟩
    rw [← hz, Complex.exp_im]
    exact mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi h1 h2)
  · intro hw
    have hw0 : (1 + Complex.I) ≠ 0 := by norm_num [Complex.ext_iff]
    refine ⟨Complex.log (1 + Complex.I), ?_, ?_, Complex.exp_log hw0⟩
    · rw [Complex.log_im]
      exact lt_of_le_of_ne (Complex.arg_nonneg_iff.mpr hw.le) fun h => hw.ne' (Complex.arg_eq_zero_iff.mp h.symm).2
    · rw [Complex.log_im]
      exact lt_of_le_of_ne (Complex.arg_le_pi _) fun h => by linarith [(Complex.arg_eq_pi_iff.mp h).2]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_strip_onto ((-1 + 2 * Complex.I)) := by
  constructor
  · rintro ⟨z, h1, h2, hz⟩
    rw [← hz, Complex.exp_im]
    exact mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi h1 h2)
  · intro hw
    have hw0 : (-1 + 2 * Complex.I) ≠ 0 := by norm_num [Complex.ext_iff]
    refine ⟨Complex.log (-1 + 2 * Complex.I), ?_, ?_, Complex.exp_log hw0⟩
    · rw [Complex.log_im]
      exact lt_of_le_of_ne (Complex.arg_nonneg_iff.mpr hw.le) fun h => hw.ne' (Complex.arg_eq_zero_iff.mp h.symm).2
    · rw [Complex.log_im]
      exact lt_of_le_of_ne (Complex.arg_le_pi _) fun h => by linarith [(Complex.arg_eq_pi_iff.mp h).2]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_strip_onto
-- test: ca-holo-exp-strip-test.sage

/-! ## exr-conformal-angles -/

/-- `(f ∘ γ₁)'(0) = f'(z₀) γ₁'(0)` and `(f ∘ γ₂)'(0) = f'(z₀) γ₂'(0)` for `f = z²`, `γ₁ = z₀ + t`, `γ₂ = z₀ + it`. -/
theorem curve_tangents (z₀ : ℂ) :
    HasDerivAt (fun t : ℝ => ((z₀ : ℂ) + t) ^ 2) (2 * z₀ * 1) 0 ∧
      HasDerivAt (fun t : ℝ => ((z₀ : ℂ) + t * I) ^ 2) (2 * z₀ * I) 0 := by
  constructor
  · have h : HasDerivAt (fun w : ℂ => (z₀ + w) ^ 2) (2 * z₀ * 1) ((0 : ℝ) : ℂ) := by
      have := (hasDerivAt_pow 2 (z₀ + ((0 : ℝ) : ℂ))).comp ((0 : ℝ) : ℂ) ((hasDerivAt_id' ((0 : ℝ) : ℂ)).const_add z₀)
      convert this using 1
      · funext w
        simp [Function.comp_def]
      · norm_num
    exact h.comp_ofReal
  · have h : HasDerivAt (fun w : ℂ => (z₀ + w * I) ^ 2) (2 * z₀ * I) ((0 : ℝ) : ℂ) := by
      have := (hasDerivAt_pow 2 (z₀ + ((0 : ℝ) : ℂ) * I)).comp ((0 : ℝ) : ℂ)
        (((hasDerivAt_id' ((0 : ℝ) : ℂ)).mul_const I).const_add z₀)
      convert this using 1
      · funext w
        simp [Function.comp_def]
      · norm_num
    exact h.comp_ofReal

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.curve_tangents (0) := by
  constructor
  · have h : HasDerivAt (fun w : ℂ => ((0 : ℂ) + w) ^ 2) (2 * (0 : ℂ) * 1) ((0 : ℝ) : ℂ) := by
      have := (hasDerivAt_pow 2 ((0 : ℂ) + ((0 : ℝ) : ℂ))).comp ((0 : ℝ) : ℂ) ((hasDerivAt_id' ((0 : ℝ) : ℂ)).const_add (0 : ℂ))
      convert this using 1
      · funext w
        simp [Function.comp_def]
      · norm_num
    exact h.comp_ofReal
  · have h : HasDerivAt (fun w : ℂ => ((0 : ℂ) + w * I) ^ 2) (2 * (0 : ℂ) * I) ((0 : ℝ) : ℂ) := by
      have := (hasDerivAt_pow 2 ((0 : ℂ) + ((0 : ℝ) : ℂ) * I)).comp ((0 : ℝ) : ℂ)
        (((hasDerivAt_id' ((0 : ℝ) : ℂ)).mul_const I).const_add (0 : ℂ))
      convert this using 1
      · funext w
        simp [Function.comp_def]
      · norm_num
    exact h.comp_ofReal

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.curve_tangents ((1 + Complex.I)) := by
  constructor
  · have h : HasDerivAt (fun w : ℂ => ((1 + Complex.I) + w) ^ 2) (2 * (1 + Complex.I) * 1) ((0 : ℝ) : ℂ) := by
      have := (hasDerivAt_pow 2 ((1 + Complex.I) + ((0 : ℝ) : ℂ))).comp ((0 : ℝ) : ℂ) ((hasDerivAt_id' ((0 : ℝ) : ℂ)).const_add (1 + Complex.I))
      convert this using 1
      · funext w
        simp [Function.comp_def]
      · norm_num
    exact h.comp_ofReal
  · have h : HasDerivAt (fun w : ℂ => ((1 + Complex.I) + w * I) ^ 2) (2 * (1 + Complex.I) * I) ((0 : ℝ) : ℂ) := by
      have := (hasDerivAt_pow 2 ((1 + Complex.I) + ((0 : ℝ) : ℂ) * I)).comp ((0 : ℝ) : ℂ)
        (((hasDerivAt_id' ((0 : ℝ) : ℂ)).mul_const I).const_add (1 + Complex.I))
      convert this using 1
      · funext w
        simp [Function.comp_def]
      · norm_num
    exact h.comp_ofReal

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.curve_tangents (Complex.I) := by
  constructor
  · have h : HasDerivAt (fun w : ℂ => ((Complex.I) + w) ^ 2) (2 * (Complex.I) * 1) ((0 : ℝ) : ℂ) := by
      have := (hasDerivAt_pow 2 ((Complex.I) + ((0 : ℝ) : ℂ))).comp ((0 : ℝ) : ℂ) ((hasDerivAt_id' ((0 : ℝ) : ℂ)).const_add (Complex.I))
      convert this using 1
      · funext w
        simp [Function.comp_def]
      · norm_num
    exact h.comp_ofReal
  · have h : HasDerivAt (fun w : ℂ => ((Complex.I) + w * I) ^ 2) (2 * (Complex.I) * I) ((0 : ℝ) : ℂ) := by
      have := (hasDerivAt_pow 2 ((Complex.I) + ((0 : ℝ) : ℂ) * I)).comp ((0 : ℝ) : ℂ)
        (((hasDerivAt_id' ((0 : ℝ) : ℂ)).mul_const I).const_add (Complex.I))
      convert this using 1
      · funext w
        simp [Function.comp_def]
      · norm_num
    exact h.comp_ofReal

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.curve_tangents
-- test: ca-holo-curve-tangents-test.sage

/-- At `z₀ = 1 + i`: `f'(z₀) = 2(1 + i)` sends the tangents `1` and `i` to `2 + 2i` and `-2 + 2i`. -/
theorem tangent_images : 2 * (1 + I) * 1 = 2 + 2 * I ∧ 2 * (1 + I) * I = -2 + 2 * I := by
  have hI : I ^ 2 = -1 := I_sq
  refine ⟨?_, ?_⟩
  · generalize Complex.I = i at *
    rw [← sub_eq_zero]
    veriphy_certified (ring1)
    -- certificate: ca-holo-ring.sage
  · generalize Complex.I = i at *
    rw [← sub_eq_zero]
    veriphy_certified (linear_combination (2) * hI)
    -- certificate: ca-holo-ring.sage

/-- The images `(2, 2)` and `(-2, 2)` are orthogonal: `2·(-2) + 2·2 = 0`. -/
theorem tangent_images_orthogonal : (2 : ℝ) * (-2) + 2 * 2 = 0 := by
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `2(1 + i) = 2√2 e^{iπ/4}`. -/
theorem polar_form : 2 * (1 + I) = ((2 * Real.sqrt 2 : ℝ) : ℂ) * Complex.exp (↑(π / 4) * I) := by
  rw [Complex.exp_mul_I, ← ofReal_cos, ← ofReal_sin, Real.cos_pi_div_four, Real.sin_pi_div_four]
  have hs : ((Real.sqrt 2 : ℝ) : ℂ) ^ 2 = 2 := by
    rw [← ofReal_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
    norm_num
  have hI : I ^ 2 = -1 := I_sq
  have e1 : ((2 * Real.sqrt 2 : ℝ) : ℂ) = 2 * ((Real.sqrt 2 : ℝ) : ℂ) := by push_cast <;> ring
  have e2 : ((Real.sqrt 2 / 2 : ℝ) : ℂ) = ((Real.sqrt 2 : ℝ) : ℂ) * (1 / 2) := by push_cast <;> ring
  rw [e1, e2]
  clear e1 e2
  generalize ((Real.sqrt 2 : ℝ) : ℂ) = s at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-i - 1) * hs)
  -- certificate: ca-holo-ring.sage

/-- Auxiliary form of `norm_arg_diag` (its proof); the claim below restates it. -/
theorem norm_arg_diag_aux (a : ℝ) (ha : 0 < a) :
    ‖(a : ℂ) * (1 + I)‖ = a * Real.sqrt 2 ∧ arg ((a : ℂ) * (1 + I)) = π / 4 := by
  have e : (a : ℂ) * (1 + I) = ((a * Real.sqrt 2 : ℝ) : ℂ) * (Complex.cos ↑(π / 4) + Complex.sin ↑(π / 4) * I) := by
    rw [← ofReal_cos, ← ofReal_sin, Real.cos_pi_div_four, Real.sin_pi_div_four]
    have hs : ((Real.sqrt 2 : ℝ) : ℂ) * ((Real.sqrt 2 : ℝ) : ℂ) = 2 := by
      rw [← ofReal_mul, Real.mul_self_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
      norm_num
    simp only [Complex.ofReal_mul, Complex.ofReal_div, Complex.ofReal_ofNat]
    linear_combination (-(a : ℂ) / 2 * (1 + I)) * hs
  have hr : 0 < a * Real.sqrt 2 := by positivity
  constructor
  · rw [e, norm_mul, ← Complex.exp_mul_I, Complex.norm_exp_ofReal_mul_I, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos hr, mul_one]
  · rw [e]
    exact arg_mul_cos_add_sin_mul_I hr ⟨by linarith [Real.pi_pos], by linarith [Real.pi_pos]⟩

/-- For `a > 0`: `|a(1 + i)| = a√2` and `arg(a(1 + i)) = π/4` (at `a = 2`: `|f'(z₀)| = 2√2`, `arg f'(z₀) = π/4`). -/
theorem norm_arg_diag (a : ℝ) (ha : 0 < a) :
    ‖(a : ℂ) * (1 + I)‖ = a * Real.sqrt 2 ∧ arg ((a : ℂ) * (1 + I)) = π / 4 :=
  norm_arg_diag_aux a ha

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.norm_arg_diag (1) (one_pos) := by
  exact KaluzaKlein.ComplexAnalysis.ConformalMaps.norm_arg_diag_aux 1 one_pos

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.norm_arg_diag (2) (two_pos) := by
  exact KaluzaKlein.ComplexAnalysis.ConformalMaps.norm_arg_diag_aux 2 two_pos

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.norm_arg_diag ((1 / 2)) ((by norm_num)) := by
  exact KaluzaKlein.ComplexAnalysis.ConformalMaps.norm_arg_diag_aux (1 / 2) (by norm_num)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.norm_arg_diag
-- test: ca-holo-norm-arg-test.sage

/-! ## exr-conformal-mobius -/

/-- `|z + i|² - |z - i|² = 4 Im z`, so `|z - i| < |z + i|` exactly when `Im z > 0`. -/
theorem cayley_normSq_diff (x y : ℝ) : (x ^ 2 + (y + 1) ^ 2) - (x ^ 2 + (y - 1) ^ 2) = 4 * y := by
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- The Cayley map sends the upper half-plane into the unit disc. -/
theorem cayley_disc (z : ℂ) (hz : 0 < z.im) : ‖(z - I) / (z + I)‖ < 1 := by
  have h2 : z + I ≠ 0 := by
    intro h
    have := congrArg Complex.im h
    norm_num at this <;> linarith
  rw [norm_div, div_lt_one (norm_pos_iff.mpr h2), ← sq_lt_sq₀ (norm_nonneg _) (norm_nonneg _),
    ← Complex.normSq_eq_norm_sq, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply, Complex.normSq_apply]
  simp <;> nlinarith

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_disc (Complex.I) ((by simp)) := by
  have h2 : (Complex.I) + Complex.I ≠ 0 := by norm_num [Complex.ext_iff]
  rw [norm_div, div_lt_one (norm_pos_iff.mpr h2), ← sq_lt_sq₀ (norm_nonneg _) (norm_nonneg _),
    ← Complex.normSq_eq_norm_sq, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply, Complex.normSq_apply]
  simp <;> nlinarith

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_disc ((1 + Complex.I)) ((by simp)) := by
  have h2 : (1 + Complex.I) + Complex.I ≠ 0 := by norm_num [Complex.ext_iff]
  rw [norm_div, div_lt_one (norm_pos_iff.mpr h2), ← sq_lt_sq₀ (norm_nonneg _) (norm_nonneg _),
    ← Complex.normSq_eq_norm_sq, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply, Complex.normSq_apply]
  simp <;> nlinarith

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_disc ((-3 + 2 * Complex.I)) ((by simp)) := by
  have h2 : (-3 + 2 * Complex.I) + Complex.I ≠ 0 := by norm_num [Complex.ext_iff]
  rw [norm_div, div_lt_one (norm_pos_iff.mpr h2), ← sq_lt_sq₀ (norm_nonneg _) (norm_nonneg _),
    ← Complex.normSq_eq_norm_sq, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply, Complex.normSq_apply]
  simp <;> nlinarith

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_disc
-- test: ca-holo-cayley-disc-test.sage

/-- With `z = i(1 + w)/(1 - w)`: `z - i = 2iw/(1 - w)`. -/
theorem cayley_inv_num (w : ℂ) (h1 : 1 - w ≠ 0) : I * (1 + w) / (1 - w) - I = 2 * I * w / (1 - w) := by
  rw [div_sub' h1, div_left_inj' h1]
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- With `z = i(1 + w)/(1 - w)`: `z + i = 2i/(1 - w)`. -/
theorem cayley_inv_den (w : ℂ) (h1 : 1 - w ≠ 0) : I * (1 + w) / (1 - w) + I = 2 * I / (1 - w) := by
  rw [div_add' _ _ _ h1, div_left_inj' h1]
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- The inverse `z = i(1 + w)/(1 - w)`: `f(z(w)) = w`. -/
theorem cayley_right_inv (w : ℂ) (hw : w ≠ 1) : (I * (1 + w) / (1 - w) - I) / (I * (1 + w) / (1 - w) + I) = w := by
  have h1 : 1 - w ≠ 0 := sub_ne_zero.mpr (Ne.symm hw)
  have h3 : (2 : ℂ) * I ≠ 0 := mul_ne_zero two_ne_zero I_ne_zero
  rw [cayley_inv_num w h1, cayley_inv_den w h1, div_div_div_cancel_right₀ h1, div_eq_iff h3]
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `1 + f(z) = 2z/(z + i)`. -/
theorem cayley_one_add (z : ℂ) (hz : z + I ≠ 0) : 1 + (z - I) / (z + I) = 2 * z / (z + I) := by
  rw [add_div' _ _ _ hz, div_left_inj' hz]
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `1 - f(z) = 2i/(z + i)`. -/
theorem cayley_one_sub (z : ℂ) (hz : z + I ≠ 0) : 1 - (z - I) / (z + I) = 2 * I / (z + I) := by
  rw [sub_div' hz, div_left_inj' hz]
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `z(f(z)) = z`. -/
theorem cayley_left_inv (z : ℂ) (hz : z + I ≠ 0) : I * (1 + (z - I) / (z + I)) / (1 - (z - I) / (z + I)) = z := by
  have h3 : (2 : ℂ) * I ≠ 0 := mul_ne_zero two_ne_zero I_ne_zero
  rw [cayley_one_add z hz, cayley_one_sub z hz, mul_div_assoc', div_div_div_cancel_right₀ hz, div_eq_iff h3]
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- Auxiliary form of `cayley_inv_im` (its proof); the claim below restates it. -/
theorem cayley_inv_im_aux (w : ℂ) (hw : ‖w‖ < 1) : 0 < (I * (1 + w) / (1 - w)).im := by
  have h1 : 1 - w ≠ 0 := by
    intro h
    have : w = 1 := by linear_combination -h
    rw [this, norm_one] at hw
    exact lt_irrefl _ hw
  have hn : 0 < Complex.normSq (1 - w) := Complex.normSq_pos.mpr h1
  have hw2 : Complex.normSq w < 1 := by
    rw [Complex.normSq_eq_norm_sq]
    nlinarith [norm_nonneg w]
  rw [Complex.div_im]
  simp only [Complex.normSq_apply] at hn hw2 ⊢
  simp
  have hn' : 0 < (1 - w.re) * (1 - w.re) + w.im * w.im := by simpa using hn
  rw [div_lt_div_iff₀ hn' hn']
  nlinarith

/-- `Im z = (1 - |w|²)/|1 - w|² > 0` for `|w| < 1`: the inverse maps the disc into the half-plane. -/
theorem cayley_inv_im (w : ℂ) (hw : ‖w‖ < 1) : 0 < (I * (1 + w) / (1 - w)).im :=
  cayley_inv_im_aux w hw

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_inv_im (0) ((by simp)) := by
  exact KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_inv_im_aux 0 (by simp)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_inv_im ((1 / 2)) ((by norm_num)) := by
  exact KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_inv_im_aux (1 / 2) (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_inv_im ((Complex.I / 2)) ((by simp; norm_num)) := by
  exact KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_inv_im_aux (Complex.I / 2) (by simp; norm_num)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_inv_im
-- test: ca-holo-cayley-im-test.sage

/-- `f(i) = 0`, `f(0) = -1`, `f(1) = (1 - i)/(1 + i) = -i`. -/
theorem cayley_values : (I - I) / (I + I) = 0 ∧ (0 - I) / (0 + I) = -1 ∧ (1 - I) / (1 + I) = -I := by
  have hI : I ^ 2 = -1 := I_sq
  have h1 : I + I ≠ 0 := by rw [← two_mul]; exact mul_ne_zero two_ne_zero I_ne_zero
  have h2 : (0 : ℂ) + I ≠ 0 := by rw [zero_add]; exact I_ne_zero
  have h3 : (1 : ℂ) + I ≠ 0 := by
    intro h
    have := congrArg Complex.re h
    simp at this
  refine ⟨?_, ?_, ?_⟩
  · rw [div_eq_iff h1]
    generalize Complex.I = i at *
    rw [← sub_eq_zero]
    veriphy_certified (ring1)
    -- certificate: ca-holo-ring.sage
  · rw [div_eq_iff h2]
    generalize Complex.I = i at *
    rw [← sub_eq_zero]
    veriphy_certified (ring1)
    -- certificate: ca-holo-ring.sage
  · rw [div_eq_iff h3]
    generalize Complex.I = i at *
    rw [← sub_eq_zero]
    veriphy_certified (linear_combination hI)
    -- certificate: ca-holo-ring.sage

/-- `(z - a)/(z + b) → 1` as `|z| → ∞` (for the Cayley map `a = b = i`). -/
theorem tendsto_mobius_infty (a b : ℂ) :
    Tendsto (fun z : ℂ => (z - a) / (z + b)) (Bornology.cobounded ℂ) (𝓝 1) := by
  have hadd : Tendsto (fun z : ℂ => z + b) (Bornology.cobounded ℂ) (Bornology.cobounded ℂ) := by
    rw [← tendsto_norm_atTop_iff_cobounded]
    refine tendsto_atTop_mono (fun z => ?_)
      ((tendsto_norm_cobounded_atTop (E := ℂ)).atTop_add (tendsto_const_nhds (x := -‖b‖)))
    have := norm_sub_le (z + b) b
    simp only [add_sub_cancel_right] at this
    linarith
  have hinv : Tendsto (fun z : ℂ => (z + b)⁻¹) (Bornology.cobounded ℂ) (𝓝 0) :=
    tendsto_inv₀_cobounded.comp hadd
  have hlim : Tendsto (fun z : ℂ => 1 - (a + b) * (z + b)⁻¹) (Bornology.cobounded ℂ) (𝓝 1) := by
    simpa using (tendsto_const_nhds (x := (1 : ℂ))).sub (hinv.const_mul (a + b))
  refine hlim.congr' ?_
  have hev : ∀ᶠ z in Bornology.cobounded ℂ, z + b ≠ 0 :=
    hadd.eventually_ne_cobounded 0
  filter_upwards [hev] with z hz
  field_simp
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.tendsto_mobius_infty (0) (0) := by
  have hadd : Tendsto (fun z : ℂ => z + (0 : ℂ)) (Bornology.cobounded ℂ) (Bornology.cobounded ℂ) := by
    rw [← tendsto_norm_atTop_iff_cobounded]
    refine tendsto_atTop_mono (fun z => ?_)
      ((tendsto_norm_cobounded_atTop (E := ℂ)).atTop_add (tendsto_const_nhds (x := -‖(0 : ℂ)‖)))
    have := norm_sub_le (z + (0 : ℂ)) (0 : ℂ)
    simp only [add_sub_cancel_right] at this
    linarith
  have hinv : Tendsto (fun z : ℂ => (z + (0 : ℂ))⁻¹) (Bornology.cobounded ℂ) (𝓝 0) :=
    tendsto_inv₀_cobounded.comp hadd
  have hlim : Tendsto (fun z : ℂ => 1 - ((0 : ℂ) + (0 : ℂ)) * (z + (0 : ℂ))⁻¹) (Bornology.cobounded ℂ) (𝓝 1) := by
    simpa using (tendsto_const_nhds (x := (1 : ℂ))).sub (hinv.const_mul ((0 : ℂ) + (0 : ℂ)))
  refine hlim.congr' ?_
  have hev : ∀ᶠ z in Bornology.cobounded ℂ, z + (0 : ℂ) ≠ 0 :=
    hadd.eventually_ne_cobounded 0
  filter_upwards [hev] with z hz
  field_simp
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.tendsto_mobius_infty (Complex.I) (Complex.I) := by
  have hadd : Tendsto (fun z : ℂ => z + (Complex.I)) (Bornology.cobounded ℂ) (Bornology.cobounded ℂ) := by
    rw [← tendsto_norm_atTop_iff_cobounded]
    refine tendsto_atTop_mono (fun z => ?_)
      ((tendsto_norm_cobounded_atTop (E := ℂ)).atTop_add (tendsto_const_nhds (x := -‖(Complex.I)‖)))
    have := norm_sub_le (z + (Complex.I)) (Complex.I)
    simp only [add_sub_cancel_right] at this
    linarith
  have hinv : Tendsto (fun z : ℂ => (z + (Complex.I))⁻¹) (Bornology.cobounded ℂ) (𝓝 0) :=
    tendsto_inv₀_cobounded.comp hadd
  have hlim : Tendsto (fun z : ℂ => 1 - ((Complex.I) + (Complex.I)) * (z + (Complex.I))⁻¹) (Bornology.cobounded ℂ) (𝓝 1) := by
    simpa using (tendsto_const_nhds (x := (1 : ℂ))).sub (hinv.const_mul ((Complex.I) + (Complex.I)))
  refine hlim.congr' ?_
  have hev : ∀ᶠ z in Bornology.cobounded ℂ, z + (Complex.I) ≠ 0 :=
    hadd.eventually_ne_cobounded 0
  filter_upwards [hev] with z hz
  field_simp
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.tendsto_mobius_infty (1) (2) := by
  have hadd : Tendsto (fun z : ℂ => z + (2 : ℂ)) (Bornology.cobounded ℂ) (Bornology.cobounded ℂ) := by
    rw [← tendsto_norm_atTop_iff_cobounded]
    refine tendsto_atTop_mono (fun z => ?_)
      ((tendsto_norm_cobounded_atTop (E := ℂ)).atTop_add (tendsto_const_nhds (x := -‖(2 : ℂ)‖)))
    have := norm_sub_le (z + (2 : ℂ)) (2 : ℂ)
    simp only [add_sub_cancel_right] at this
    linarith
  have hinv : Tendsto (fun z : ℂ => (z + (2 : ℂ))⁻¹) (Bornology.cobounded ℂ) (𝓝 0) :=
    tendsto_inv₀_cobounded.comp hadd
  have hlim : Tendsto (fun z : ℂ => 1 - ((1 : ℂ) + (2 : ℂ)) * (z + (2 : ℂ))⁻¹) (Bornology.cobounded ℂ) (𝓝 1) := by
    simpa using (tendsto_const_nhds (x := (1 : ℂ))).sub (hinv.const_mul ((1 : ℂ) + (2 : ℂ)))
  refine hlim.congr' ?_
  have hev : ∀ᶠ z in Bornology.cobounded ℂ, z + (2 : ℂ) ≠ 0 :=
    hadd.eventually_ne_cobounded 0
  filter_upwards [hev] with z hz
  field_simp
  ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.tendsto_mobius_infty
-- test: ca-holo-mobius-infty-test.sage

/-! ## exr-conformal-critical -/

/-- Along the ray `z₀ + t e^{iα}`, `(z - z₀)ⁿ = tⁿ e^{inα}`. -/
theorem ray_power (n : ℕ) (t α : ℝ) :
    ((t : ℂ) * Complex.exp (α * I)) ^ n = ((t ^ n : ℝ) : ℂ) * Complex.exp (((n * α : ℝ) : ℂ) * I) := by
  rw [mul_pow, ← Complex.exp_nat_mul]
  push_cast
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.ray_power (0) (0) (0) := by
  rw [mul_pow, ← Complex.exp_nat_mul]
  push_cast
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.ray_power (4) (-5) ((-5 / 2)) := by
  rw [mul_pow, ← Complex.exp_nat_mul]
  push_cast
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.ray_power (0) (0) (1) := by
  rw [mul_pow, ← Complex.exp_nat_mul]
  push_cast
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.ray_power (0) (0) (-5) := by
  rw [mul_pow, ← Complex.exp_nat_mul]
  push_cast
  ring_nf

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.ray_power
-- test: ca-holo-ray-power-test.sage

/-- If `f(z) - f(z₀) = (z - z₀)ⁿ g(z)` with `g` continuous at `z₀` (the Taylor expansion, `g(z₀) = f⁽ⁿ⁾(z₀)/n!`),
then along the ray `z₀ + t e^{iα}` the displacement `(f(z) - f(z₀))/tⁿ` tends to `g(z₀) e^{inα}`: the image curve
leaves `f(z₀)` in the direction `arg g(z₀) + nα`, so angles between curves are multiplied by `n`. -/
theorem critical_direction (f g : ℂ → ℂ) (z₀ : ℂ) (n : ℕ) (α : ℝ)
    (hfg : ∀ z, f z - f z₀ = (z - z₀) ^ n * g z) (hg : ContinuousAt g z₀) :
    Tendsto (fun t : ℝ => (f (z₀ + t * Complex.exp (α * I)) - f z₀) / (t : ℂ) ^ n) (𝓝[≠] 0)
      (𝓝 (g z₀ * Complex.exp (((n * α : ℝ) : ℂ) * I))) := by
  have hpath : Tendsto (fun t : ℝ => z₀ + t * Complex.exp (α * I)) (𝓝 0) (𝓝 z₀) := by
    have := ((continuous_ofReal.tendsto (0 : ℝ)).mul_const (Complex.exp (α * I))).const_add z₀
    simpa using this
  have hlim : Tendsto (fun t : ℝ => g (z₀ + t * Complex.exp (α * I)) * Complex.exp (((n * α : ℝ) : ℂ) * I))
      (𝓝[≠] 0) (𝓝 (g z₀ * Complex.exp (((n * α : ℝ) : ℂ) * I))) :=
    ((hg.tendsto.comp hpath).mono_left nhdsWithin_le_nhds).mul_const _
  refine hlim.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with t (ht : t ≠ 0)
  have ht' : (t : ℂ) ^ n ≠ 0 := pow_ne_zero n (ofReal_ne_zero.mpr ht)
  rw [hfg, add_sub_cancel_left, ray_power, eq_div_iff ht']
  push_cast
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.critical_direction ((fun z : ℂ => z ^ 2)) ((fun _ : ℂ => (1 : ℂ))) (0) (2) (0) ((by intro z; ring)) (continuousAt_const) := by
  have hpath : Filter.Tendsto (fun t : ℝ => (0 : ℂ) + t * Complex.exp ((0 : ℝ) * Complex.I)) (nhds 0) (nhds (0 : ℂ)) := by
    have := ((Complex.continuous_ofReal.tendsto (0 : ℝ)).mul_const (Complex.exp ((0 : ℝ) * Complex.I))).const_add (0 : ℂ)
    simpa using this
  have hlim : Filter.Tendsto (fun t : ℝ => (fun _ : ℂ => (1 : ℂ)) ((0 : ℂ) + t * Complex.exp ((0 : ℝ) * Complex.I)) * Complex.exp (((((2 : ℕ) : ℝ) * (0) : ℝ) : ℂ) * Complex.I))
      (nhdsWithin 0 {0}ᶜ) (nhds ((fun _ : ℂ => (1 : ℂ)) (0) * Complex.exp (((((2 : ℕ) : ℝ) * (0) : ℝ) : ℂ) * Complex.I))) :=
    (((continuousAt_const).tendsto.comp hpath).mono_left nhdsWithin_le_nhds).mul_const _
  refine hlim.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with t (ht : t ≠ 0)
  have ht' : (t : ℂ) ^ (2 : ℕ) ≠ 0 := pow_ne_zero _ (Complex.ofReal_ne_zero.mpr ht)
  rw [((by intro z; ring : ∀ z : ℂ, (fun z : ℂ => z ^ 2) z - (fun z : ℂ => z ^ 2) 0 = (z - 0) ^ 2 * (fun _ : ℂ => (1 : ℂ)) z)), add_sub_cancel_left, KaluzaKlein.ComplexAnalysis.ConformalMaps.ray_power, eq_div_iff ht']
  push_cast
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.critical_direction ((fun z : ℂ => z ^ 3)) ((fun _ : ℂ => (1 : ℂ))) (0) (3) (1) ((by intro z; ring)) (continuousAt_const) := by
  have hpath : Filter.Tendsto (fun t : ℝ => (0 : ℂ) + t * Complex.exp ((1 : ℝ) * Complex.I)) (nhds 0) (nhds (0 : ℂ)) := by
    have := ((Complex.continuous_ofReal.tendsto (0 : ℝ)).mul_const (Complex.exp ((1 : ℝ) * Complex.I))).const_add (0 : ℂ)
    simpa using this
  have hlim : Filter.Tendsto (fun t : ℝ => (fun _ : ℂ => (1 : ℂ)) ((0 : ℂ) + t * Complex.exp ((1 : ℝ) * Complex.I)) * Complex.exp (((((3 : ℕ) : ℝ) * (1) : ℝ) : ℂ) * Complex.I))
      (nhdsWithin 0 {0}ᶜ) (nhds ((fun _ : ℂ => (1 : ℂ)) (0) * Complex.exp (((((3 : ℕ) : ℝ) * (1) : ℝ) : ℂ) * Complex.I))) :=
    (((continuousAt_const).tendsto.comp hpath).mono_left nhdsWithin_le_nhds).mul_const _
  refine hlim.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with t (ht : t ≠ 0)
  have ht' : (t : ℂ) ^ (3 : ℕ) ≠ 0 := pow_ne_zero _ (Complex.ofReal_ne_zero.mpr ht)
  rw [((by intro z; ring : ∀ z : ℂ, (fun z : ℂ => z ^ 3) z - (fun z : ℂ => z ^ 3) 0 = (z - 0) ^ 3 * (fun _ : ℂ => (1 : ℂ)) z)), add_sub_cancel_left, KaluzaKlein.ComplexAnalysis.ConformalMaps.ray_power, eq_div_iff ht']
  push_cast
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.critical_direction ((fun z : ℂ => 2 * (z - 1) ^ 2)) ((fun _ : ℂ => (2 : ℂ))) (1) (2) ((-1)) ((by intro z; ring)) (continuousAt_const) := by
  have hpath : Filter.Tendsto (fun t : ℝ => (1 : ℂ) + t * Complex.exp ((-1 : ℝ) * Complex.I)) (nhds 0) (nhds (1 : ℂ)) := by
    have := ((Complex.continuous_ofReal.tendsto (0 : ℝ)).mul_const (Complex.exp ((-1 : ℝ) * Complex.I))).const_add (1 : ℂ)
    simpa using this
  have hlim : Filter.Tendsto (fun t : ℝ => (fun _ : ℂ => (2 : ℂ)) ((1 : ℂ) + t * Complex.exp ((-1 : ℝ) * Complex.I)) * Complex.exp (((((2 : ℕ) : ℝ) * (-1) : ℝ) : ℂ) * Complex.I))
      (nhdsWithin 0 {0}ᶜ) (nhds ((fun _ : ℂ => (2 : ℂ)) (1) * Complex.exp (((((2 : ℕ) : ℝ) * (-1) : ℝ) : ℂ) * Complex.I))) :=
    (((continuousAt_const).tendsto.comp hpath).mono_left nhdsWithin_le_nhds).mul_const _
  refine hlim.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with t (ht : t ≠ 0)
  have ht' : (t : ℂ) ^ (2 : ℕ) ≠ 0 := pow_ne_zero _ (Complex.ofReal_ne_zero.mpr ht)
  rw [((by intro z; ring : ∀ z : ℂ, (fun z : ℂ => 2 * (z - 1) ^ 2) z - (fun z : ℂ => 2 * (z - 1) ^ 2) 1 = (z - 1) ^ 2 * (fun _ : ℂ => (2 : ℂ)) z)), add_sub_cancel_left, KaluzaKlein.ComplexAnalysis.ConformalMaps.ray_power, eq_div_iff ht']
  push_cast
  ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.critical_direction
-- test: ca-holo-critical-test.sage

/-- Two directions `α`, `β` are sent to `nα`, `nβ`: `e^{inβ} = e^{in(β - α)} e^{inα}`, the angle `n(β - α)`. -/
theorem angle_multiplied (n : ℕ) (α β : ℝ) :
    Complex.exp (((n * β : ℝ) : ℂ) * I) = Complex.exp (((n * (β - α) : ℝ) : ℂ) * I) * Complex.exp (((n * α : ℝ) : ℂ) * I) := by
  rw [← Complex.exp_add]
  push_cast
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.angle_multiplied (0) (0) (0) := by
  rw [← Complex.exp_add]
  push_cast
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.angle_multiplied (4) (-5) ((-5 / 2)) := by
  rw [← Complex.exp_add]
  push_cast
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.angle_multiplied (0) (0) (1) := by
  rw [← Complex.exp_add]
  push_cast
  ring_nf

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ConformalMaps.angle_multiplied (0) (0) (-5) := by
  rw [← Complex.exp_add]
  push_cast
  ring_nf

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.ConformalMaps.angle_multiplied
-- test: ca-holo-angle-mult-test.sage

end KaluzaKlein.ComplexAnalysis.ConformalMaps
