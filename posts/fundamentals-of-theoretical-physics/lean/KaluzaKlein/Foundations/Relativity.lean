import KaluzaKlein.Geometry.Curvature

/-!
# General relativity on explicit metrics

* `sphere_einstein_zero`: in two dimensions the Einstein tensor `G_ij = R_ij − ½ R g_ij` vanishes identically; on the
  unit sphere this follows from `R_ij = g_ij` and `R = 2`.
* Three-dimensional de Sitter space in flat slicing, `ds² = dt² − e^{2Ht}(dx² + dy²)` (signature `(+, −, −)`): its
  Christoffel symbols, the Riemann components that feed the Ricci tensor, `R_ij = −2H² g_ij`, `R = −6H²`, and the
  Einstein equation with cosmological constant `G_ij − Λ g_ij = 0` for `Λ = H²` (in the mostly-plus metric the same
  space has `R_ij = 2H² g_ij`, `R = 6H²`: the Ricci tensor is unchanged by `g → −g`, the metric and `R` change sign).
  The closed forms hold on the whole chart (`e^{2Ht} ≠ 0`),
  so the derivatives of the Christoffel symbols are computed from function equalities, without a chart hypothesis.
-/

namespace KaluzaKlein.Relativity

open CoordinateGeometry

/-- `G_ij = R_ij − ½ R g_ij = 0` on the unit sphere (the Einstein tensor of any two-dimensional metric vanishes). -/
theorem sphere_einstein_zero (i j : Fin 2) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    ricci Geometry.sphere i j x - 1 / 2 * scalarCurvature Geometry.sphere x * Geometry.sphere i j x = 0 := by
  have e00 := Geometry.sphere_ricci_0_0 x h; have e01 := Geometry.sphere_ricci_0_1 x h
  have e10 := Geometry.sphere_ricci_1_0 x h; have e11 := Geometry.sphere_ricci_1_1 x h
  have eR := Geometry.sphere_scalarCurvature x h
  fin_cases i <;> fin_cases j <;> simp only [Geometry.sphere] at e00 e01 e10 e11 eR ⊢ <;>
    simp [e00, e01, e10, e11, eR] <;> ring

/-- `e^{−a} (e^{a} c) = c` and `e^{a} (e^{−a} c) = c`, for the simplifier. -/
theorem exp_neg_mul (a c : ℝ) : Real.exp (-a) * (Real.exp a * c) = c := by
  rw [← mul_assoc, ← Real.exp_add]; simp
theorem exp_mul_neg (a c : ℝ) : Real.exp a * (Real.exp (-a) * c) = c := by
  rw [← mul_assoc, ← Real.exp_add]; simp

/-- Three-dimensional de Sitter space in flat slicing, `ds² = dt² − e^{2Ht}(dx² + dy²)`, coordinates `(t, x, y)`. -/
noncomputable def dS3 (H : ℝ) : Fin 3 → Fin 3 → (Fin 3 → ℝ) → ℝ :=
  ![![fun _ => 1, fun _ => 0, fun _ => 0],
    ![fun _ => 0, fun x => -Real.exp (2 * H * x 0), fun _ => 0],
    ![fun _ => 0, fun _ => 0, fun x => -Real.exp (2 * H * x 0)]]

theorem dS3_symm (H : ℝ) (i j : Fin 3) : dS3 H i j = dS3 H j i := by fin_cases i <;> fin_cases j <;> rfl

/-- The inverse metric `diag(1, −e^{−2Ht}, −e^{−2Ht})`. -/
theorem dS3_invMetric (H : ℝ) (x : Fin 3 → ℝ) :
    invMetric (dS3 H) x = !![(1 : ℝ), 0, 0; 0, -Real.exp (-(2 * H * x 0)), 0; 0, 0, -Real.exp (-(2 * H * x 0))] := by
  unfold invMetric
  refine Matrix.inv_eq_right_inv ?_
  ext i j; fin_cases i <;> fin_cases j <;>
    simp [dS3, metricAt, Matrix.mul_apply, Fin.sum_univ_succ, ← Real.exp_add]

/-- The Christoffel symbols: `Γ^t_xx = Γ^t_yy = H e^{2Ht}`, `Γ^x_tx = Γ^y_ty = H`, all others zero. -/
theorem dS3_christoffel_fun (H : ℝ) (k i j : Fin 3) :
    christoffel (dS3 H) k i j =
      (![![![fun _ => (0:ℝ), fun _ => 0, fun _ => 0], ![fun _ => 0, fun x => H * Real.exp (2 * H * x 0), fun _ => 0],
            ![fun _ => 0, fun _ => 0, fun x => H * Real.exp (2 * H * x 0)]],
         ![![fun _ => 0, fun _ => H, fun _ => 0], ![fun _ => H, fun _ => 0, fun _ => 0], ![fun _ => 0, fun _ => 0, fun _ => 0]],
         ![![fun _ => 0, fun _ => 0, fun _ => H], ![fun _ => 0, fun _ => 0, fun _ => 0], ![fun _ => H, fun _ => 0, fun _ => 0]]]
        : Fin 3 → Fin 3 → Fin 3 → (Fin 3 → ℝ) → ℝ) k i j := by
  funext x
  rw [christoffel, dS3_invMetric]
  fin_cases k <;> fin_cases i <;> fin_cases j <;>
    simp (disch := fun_prop) [pd, dS3, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
      Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1)

theorem dS3_riemann_0_0_0_1 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 0 0 0 1 x = 0 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_0_0_0_2 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 0 0 0 2 x = 0 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_0_1_0_1 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 0 1 0 1 x = H ^ 2 * Real.exp (2 * H * x 0) := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_0_1_0_2 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 0 1 0 2 x = 0 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_0_2_0_1 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 0 2 0 1 x = 0 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_0_2_0_2 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 0 2 0 2 x = H ^ 2 * Real.exp (2 * H * x 0) := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_1_0_1_0 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 1 0 1 0 x = -H ^ 2 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_1_0_1_2 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 1 0 1 2 x = 0 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_1_1_1_0 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 1 1 1 0 x = 0 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_1_1_1_2 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 1 1 1 2 x = 0 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_1_2_1_0 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 1 2 1 0 x = 0 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_1_2_1_2 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 1 2 1 2 x = H ^ 2 * Real.exp (2 * H * x 0) := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_2_0_2_0 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 2 0 2 0 x = -H ^ 2 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_2_0_2_1 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 2 0 2 1 x = 0 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_2_1_2_0 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 2 1 2 0 x = 0 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_2_1_2_1 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 2 1 2 1 x = H ^ 2 * Real.exp (2 * H * x 0) := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_2_2_2_0 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 2 2 2 0 x = 0 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_riemann_2_2_2_1 (H : ℝ) (x : Fin 3 → ℝ) : riemann (dS3 H) 2 2 2 1 x = 0 := by
  unfold riemann
  simp (disch := fun_prop) [dS3_christoffel_fun, pd, Fin.sum_univ_succ, Function.update_apply, deriv_exp, deriv_const_mul,
    deriv_mul_const, Real.deriv_exp, exp_neg_mul, exp_mul_neg]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp only [← Real.exp_add]; ring1) | (field_simp; ring1)

theorem dS3_ricci_0_0 (H : ℝ) (x : Fin 3 → ℝ) : ricci (dS3 H) 0 0 x = -2 * H ^ 2 := by
  simp [ricci, Fin.sum_univ_succ, Geometry.riemann_diag, dS3_riemann_1_0_1_0 H x, dS3_riemann_2_0_2_0 H x]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp [Real.exp_neg]; field_simp; ring1)

theorem dS3_ricci_0_1 (H : ℝ) (x : Fin 3 → ℝ) : ricci (dS3 H) 0 1 x = 0 := by
  simp [ricci, Fin.sum_univ_succ, Geometry.riemann_diag, dS3_riemann_0_0_0_1 H x, dS3_riemann_2_0_2_1 H x]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp [Real.exp_neg]; field_simp; ring1)

theorem dS3_ricci_0_2 (H : ℝ) (x : Fin 3 → ℝ) : ricci (dS3 H) 0 2 x = 0 := by
  simp [ricci, Fin.sum_univ_succ, Geometry.riemann_diag, dS3_riemann_0_0_0_2 H x, dS3_riemann_1_0_1_2 H x]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp [Real.exp_neg]; field_simp; ring1)

theorem dS3_ricci_1_0 (H : ℝ) (x : Fin 3 → ℝ) : ricci (dS3 H) 1 0 x = 0 := by
  simp [ricci, Fin.sum_univ_succ, Geometry.riemann_diag, dS3_riemann_1_1_1_0 H x, dS3_riemann_2_1_2_0 H x]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp [Real.exp_neg]; field_simp; ring1)

theorem dS3_ricci_1_1 (H : ℝ) (x : Fin 3 → ℝ) : ricci (dS3 H) 1 1 x = 2 * H ^ 2 * Real.exp (2 * H * x 0) := by
  simp [ricci, Fin.sum_univ_succ, Geometry.riemann_diag, dS3_riemann_0_1_0_1 H x, dS3_riemann_2_1_2_1 H x]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp [Real.exp_neg]; field_simp; ring1)

theorem dS3_ricci_1_2 (H : ℝ) (x : Fin 3 → ℝ) : ricci (dS3 H) 1 2 x = 0 := by
  simp [ricci, Fin.sum_univ_succ, Geometry.riemann_diag, dS3_riemann_0_1_0_2 H x, dS3_riemann_1_1_1_2 H x]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp [Real.exp_neg]; field_simp; ring1)

theorem dS3_ricci_2_0 (H : ℝ) (x : Fin 3 → ℝ) : ricci (dS3 H) 2 0 x = 0 := by
  simp [ricci, Fin.sum_univ_succ, Geometry.riemann_diag, dS3_riemann_1_2_1_0 H x, dS3_riemann_2_2_2_0 H x]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp [Real.exp_neg]; field_simp; ring1)

theorem dS3_ricci_2_1 (H : ℝ) (x : Fin 3 → ℝ) : ricci (dS3 H) 2 1 x = 0 := by
  simp [ricci, Fin.sum_univ_succ, Geometry.riemann_diag, dS3_riemann_0_2_0_1 H x, dS3_riemann_2_2_2_1 H x]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp [Real.exp_neg]; field_simp; ring1)

theorem dS3_ricci_2_2 (H : ℝ) (x : Fin 3 → ℝ) : ricci (dS3 H) 2 2 x = 2 * H ^ 2 * Real.exp (2 * H * x 0) := by
  simp [ricci, Fin.sum_univ_succ, Geometry.riemann_diag, dS3_riemann_0_2_0_2 H x, dS3_riemann_1_2_1_2 H x]
  all_goals first | ring1 | (simp only [exp_neg_mul, exp_mul_neg]; ring1) | (simp [Real.exp_neg]; field_simp; ring1)

/-- `R_ij = −2H² g_ij`: de Sitter space is an Einstein space (the Ricci tensor is the same as in the mostly-plus
metric, the metric has the opposite sign). -/
theorem dS3_ricci_eq (H : ℝ) (i j : Fin 3) (x : Fin 3 → ℝ) : ricci (dS3 H) i j x = -2 * H ^ 2 * dS3 H i j x := by
  have e := fun (s n : Fin 3) => (⟨dS3_ricci_0_0 H x, dS3_ricci_0_1 H x, dS3_ricci_0_2 H x, dS3_ricci_1_0 H x, dS3_ricci_1_1 H x, dS3_ricci_1_2 H x, dS3_ricci_2_0 H x, dS3_ricci_2_1 H x, dS3_ricci_2_2 H x⟩ :
    _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _)
  obtain ⟨e00, e01, e02, e10, e11, e12, e20, e21, e22⟩ := e 0 0
  fin_cases i <;> fin_cases j <;> simp only [dS3] at e00 e01 e02 e10 e11 e12 e20 e21 e22 ⊢ <;>
    simp [e00, e01, e02, e10, e11, e12, e20, e21, e22] <;> ring

/-- `R = −6H²` in the signature `(+, −, −)`. -/
theorem dS3_scalarCurvature (H : ℝ) (x : Fin 3 → ℝ) : scalarCurvature (dS3 H) x = -6 * H ^ 2 := by
  simp only [scalarCurvature, Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.succ_zero_eq_one, Fin.succ_one_eq_two, dS3_invMetric H x, dS3_ricci_0_0 H x, dS3_ricci_0_1 H x, dS3_ricci_0_2 H x, dS3_ricci_1_0 H x, dS3_ricci_1_1 H x, dS3_ricci_1_2 H x, dS3_ricci_2_0 H x, dS3_ricci_2_1 H x, dS3_ricci_2_2 H x]
  simp [Real.exp_neg]
  field_simp
  ring

/-- The Einstein equation with cosmological constant `Λ = H²` in the signature `(+, −, −)`:
`G_ij − Λ g_ij = R_ij − ½ R g_ij − H² g_ij = 0`. -/
theorem dS3_einstein (H : ℝ) (i j : Fin 3) (x : Fin 3 → ℝ) :
    ricci (dS3 H) i j x - 1 / 2 * scalarCurvature (dS3 H) x * dS3 H i j x - H ^ 2 * dS3 H i j x = 0 := by
  rw [dS3_ricci_eq, dS3_scalarCurvature]; ring

end KaluzaKlein.Relativity
