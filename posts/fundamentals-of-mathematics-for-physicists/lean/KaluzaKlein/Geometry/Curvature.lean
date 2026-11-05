import KaluzaKlein.Geometry.Christoffel

/-!
# Curvature of the running examples

Riemann components, Ricci tensor and scalar curvature of the round sphere and of `SU(2)`, proved from the
Christoffel lemmas of `Christoffel.lean`. The derivative of a Christoffel symbol is computed through its closed form,
which holds on the chart domain `Real.sin (x 0) ≠ 0`: the two functions agree near the point (the domain is open),
so their derivatives agree (`Filter.EventuallyEq.deriv_eq`). Veriphy has no curvature builder, so these proofs are
written by hand following that pattern.
-/

namespace KaluzaKlein.Geometry

open CoordinateGeometry Filter Topology

/-! ## Derivatives of functions known on the chart domain -/

/-- `Real.sin (y 0)` depends continuously on the `i`-th coordinate of `y = Function.update x i t`. -/
lemma continuousAt_sin_update {n : ℕ} (x : Fin (n + 1) → ℝ) (i : Fin (n + 1)) :
    ContinuousAt (fun t : ℝ => Real.sin (Function.update x i t 0)) (x i) := by
  by_cases hi : (0 : Fin (n + 1)) = i
  · subst hi; simp only [Function.update_self]; fun_prop
  · simp only [Function.update_of_ne hi]; fun_prop

/-- Two functions that agree where `Real.sin (y 0) ≠ 0` have the same partial derivatives there. -/
lemma pd_congr_chart {n : ℕ} {f g : (Fin (n + 1) → ℝ) → ℝ} (i : Fin (n + 1)) (x : Fin (n + 1) → ℝ)
    (h : Real.sin (x 0) ≠ 0) (hfg : ∀ y, Real.sin (y 0) ≠ 0 → f y = g y) : pd i f x = pd i g x := by
  unfold pd
  apply Filter.EventuallyEq.deriv_eq
  have hne : Real.sin (Function.update x i (x i) 0) ≠ 0 := by simpa using h
  filter_upwards [(continuousAt_sin_update x i).eventually_ne hne] with t ht using hfg _ ht

/-! ## The sphere -/

theorem sphere_symm (i j : Fin 2) : sphere i j = sphere j i := by
  fin_cases i <;> fin_cases j <;> rfl

/-- The closed forms of all four `Γᵏᵢⱼ` on the chart domain, as functions, for the derivative lemma. -/
theorem sphere_christoffel_fun (k i j : Fin 2) (y : Fin 2 → ℝ) (h : Real.sin (y 0) ≠ 0) :
    christoffel sphere k i j y =
      (![![![fun _ => (0 : ℝ), fun _ => 0], ![fun _ => 0, fun y => -(Real.sin (y 0) * Real.cos (y 0))]],
         ![![fun _ => 0, fun y => Real.cos (y 0) / Real.sin (y 0)],
           ![fun y => Real.cos (y 0) / Real.sin (y 0), fun _ => 0]]] : Fin 2 → Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ)
        k i j y := by
  fin_cases k <;> fin_cases i <;> fin_cases j <;>
    simp [sphere_christoffel_0_0_0 y h, sphere_christoffel_0_0_1 y h, sphere_christoffel_0_1_1 y h,
      sphere_christoffel_1_0_0 y h, sphere_christoffel_1_0_1 y h, sphere_christoffel_1_1_1 y h,
      christoffel_symm sphere sphere_symm 0 1 0, christoffel_symm sphere sphere_symm 1 1 0]

/-- `R^θ_{φθφ} = sin²θ`. -/
theorem sphere_riemann_0_1_0_1 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann sphere 0 1 0 1 x = Real.sin (x 0) ^ 2 := by
  unfold riemann
  rw [pd_congr_chart 0 x h (fun y hy => sphere_christoffel_fun 0 1 1 y hy),
      pd_congr_chart 1 x h (fun y hy => sphere_christoffel_fun 0 0 1 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_inv'', Real.deriv_sin, Real.deriv_cos, sphere_christoffel_0_0_0 x h, sphere_christoffel_0_0_1 x h,
    sphere_christoffel_0_1_1 x h, sphere_christoffel_1_0_0 x h, sphere_christoffel_1_0_1 x h,
    sphere_christoffel_1_1_1 x h, christoffel_symm sphere sphere_symm 0 1 0, christoffel_symm sphere sphere_symm 1 1 0]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

/-- `R^φ_{θθφ} = −1`. -/
theorem sphere_riemann_1_0_0_1 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann sphere 1 0 0 1 x = -1 := by
  unfold riemann
  rw [pd_congr_chart 0 x h (fun y hy => sphere_christoffel_fun 1 1 0 y hy),
      pd_congr_chart 1 x h (fun y hy => sphere_christoffel_fun 1 0 0 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', Real.deriv_sin, Real.deriv_cos, sphere_christoffel_0_0_0 x h, sphere_christoffel_0_0_1 x h,
    sphere_christoffel_0_1_1 x h, sphere_christoffel_1_0_0 x h, sphere_christoffel_1_0_1 x h,
    sphere_christoffel_1_1_1 x h, christoffel_symm sphere sphere_symm 0 1 0, christoffel_symm sphere sphere_symm 1 1 0]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

/-! ## General identities, Ricci tensor and scalar curvature of the sphere -/

/-- The Riemann tensor is antisymmetric in its last two indices, directly from the definition. -/
theorem riemann_antisymm {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (ρ σ μ ν : Fin n) (x : Fin n → ℝ) :
    riemann g ρ σ ν μ x = -riemann g ρ σ μ ν x := by
  unfold riemann
  have hs : ∑ l, (christoffel g ρ ν l x * christoffel g l μ σ x - christoffel g ρ μ l x * christoffel g l ν σ x)
      = -∑ l, (christoffel g ρ μ l x * christoffel g l ν σ x - christoffel g ρ ν l x * christoffel g l μ σ x) := by
    rw [← Finset.sum_neg_distrib]; exact Finset.sum_congr rfl fun l _ => by ring
  rw [hs]; ring

/-- A Riemann component with equal last indices vanishes. -/
theorem riemann_diag {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (ρ σ μ : Fin n) (x : Fin n → ℝ) :
    riemann g ρ σ μ μ x = 0 := by
  have := riemann_antisymm g ρ σ μ μ x; linarith

/-- `R^θ_{θθφ} = 0`. -/
theorem sphere_riemann_0_0_0_1 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) : riemann sphere 0 0 0 1 x = 0 := by
  unfold riemann
  rw [pd_congr_chart 0 x h (fun y hy => sphere_christoffel_fun 0 1 0 y hy),
      pd_congr_chart 1 x h (fun y hy => sphere_christoffel_fun 0 0 0 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_inv'', Real.deriv_sin, Real.deriv_cos, sphere_christoffel_0_0_0 x h, sphere_christoffel_0_0_1 x h,
    sphere_christoffel_0_1_1 x h, sphere_christoffel_1_0_0 x h, sphere_christoffel_1_0_1 x h,
    sphere_christoffel_1_1_1 x h, christoffel_symm sphere sphere_symm 0 1 0, christoffel_symm sphere sphere_symm 1 1 0]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

/-- `R^φ_{φθφ} = 0`. -/
theorem sphere_riemann_1_1_0_1 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) : riemann sphere 1 1 0 1 x = 0 := by
  unfold riemann
  rw [pd_congr_chart 0 x h (fun y hy => sphere_christoffel_fun 1 1 1 y hy),
      pd_congr_chart 1 x h (fun y hy => sphere_christoffel_fun 1 0 1 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', Real.deriv_sin, Real.deriv_cos, sphere_christoffel_0_0_0 x h, sphere_christoffel_0_0_1 x h,
    sphere_christoffel_0_1_1 x h, sphere_christoffel_1_0_0 x h, sphere_christoffel_1_0_1 x h,
    sphere_christoffel_1_1_1 x h, christoffel_symm sphere sphere_symm 0 1 0, christoffel_symm sphere sphere_symm 1 1 0]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

/-- `R_θθ = 1`. -/
theorem sphere_ricci_0_0 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci sphere 0 0 x = 1 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, riemann_antisymm sphere 1 0 0 1, sphere_riemann_1_0_0_1 x h]

/-- `R_θφ = 0`. -/
theorem sphere_ricci_0_1 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci sphere 0 1 x = 0 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, sphere_riemann_0_0_0_1 x h]

/-- `R_φθ = 0`. -/
theorem sphere_ricci_1_0 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci sphere 1 0 x = 0 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, riemann_antisymm sphere 1 1 0 1, sphere_riemann_1_1_0_1 x h]

/-- `R_φφ = sin²θ`: the Ricci tensor of the unit sphere is the metric. -/
theorem sphere_ricci_1_1 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci sphere 1 1 x = Real.sin (x 0) ^ 2 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, sphere_riemann_0_1_0_1 x h]

/-- The scalar curvature of the unit sphere is `2`. -/
theorem sphere_scalarCurvature (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) : scalarCurvature sphere x = 2 := by
  simp [scalarCurvature, Fin.sum_univ_succ, sphere_invMetric x h, sphere_ricci_0_0 x h, sphere_ricci_0_1 x h,
    sphere_ricci_1_0 x h, sphere_ricci_1_1 x h]
  all_goals first | (field_simp; ring1) | (field_simp; done) | norm_num

/-! ## `SU(2)`: Ricci tensor `R_ij = 2 g_ij` and scalar curvature `6` -/

theorem su2_symm (i j : Fin 3) : su2 i j = su2 j i := by
  fin_cases i <;> fin_cases j <;> rfl

/-- The closed forms of all `Γᵏᵢⱼ` of `SU(2)` on the chart domain, as functions, for the derivative lemma. -/
theorem su2_christoffel_fun (k i j : Fin 3) (y : Fin 3 → ℝ) (h : Real.sin (y 0) ≠ 0) :
    christoffel su2 k i j y =
      (![![![fun _ => (0 : ℝ), fun _ => (0 : ℝ), fun _ => (0 : ℝ)], ![fun _ => (0 : ℝ), fun _ => (0 : ℝ), fun y => Real.sin (y 0) * (1 / 2)], ![fun _ => (0 : ℝ), fun y => Real.sin (y 0) * (1 / 2), fun _ => (0 : ℝ)]], ![![fun _ => (0 : ℝ), fun y => Real.cos (y 0) * (Real.sin (y 0))⁻¹ * (1 / 2), fun y => -(Real.sin (y 0))⁻¹ * (1 / 2)], ![fun y => Real.cos (y 0) * (Real.sin (y 0))⁻¹ * (1 / 2), fun _ => (0 : ℝ), fun _ => (0 : ℝ)], ![fun y => -(Real.sin (y 0))⁻¹ * (1 / 2), fun _ => (0 : ℝ), fun _ => (0 : ℝ)]], ![![fun _ => (0 : ℝ), fun y => -(Real.sin (y 0))⁻¹ * (1 / 2), fun y => Real.cos (y 0) * (Real.sin (y 0))⁻¹ * (1 / 2)], ![fun y => -(Real.sin (y 0))⁻¹ * (1 / 2), fun _ => (0 : ℝ), fun _ => (0 : ℝ)], ![fun y => Real.cos (y 0) * (Real.sin (y 0))⁻¹ * (1 / 2), fun _ => (0 : ℝ), fun _ => (0 : ℝ)]]] : Fin 3 → Fin 3 → Fin 3 → (Fin 3 → ℝ) → ℝ) k i j y := by
  fin_cases k <;> fin_cases i <;> fin_cases j <;>
    simp [su2_christoffel_0_0_0 y h, su2_christoffel_0_0_1 y h, su2_christoffel_0_0_2 y h, su2_christoffel_0_1_1 y h, su2_christoffel_0_1_2 y h, su2_christoffel_0_2_2 y h, su2_christoffel_1_0_0 y h, su2_christoffel_1_0_1 y h, su2_christoffel_1_0_2 y h, su2_christoffel_1_1_1 y h, su2_christoffel_1_1_2 y h, su2_christoffel_1_2_2 y h, su2_christoffel_2_0_0 y h, su2_christoffel_2_0_1 y h, su2_christoffel_2_0_2 y h, su2_christoffel_2_1_1 y h, su2_christoffel_2_1_2 y h, su2_christoffel_2_2_2 y h, christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | rfl | (field_simp; ring1) | ring1

theorem su2_riemann_0_0_0_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 0 0 0 1 x = 0 := by
  unfold riemann
  rw [pd_congr_chart 0 x h (fun y hy => su2_christoffel_fun 0 1 0 y hy),
      pd_congr_chart 1 x h (fun y hy => su2_christoffel_fun 0 0 0 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_0_0_0_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 0 0 0 2 x = 0 := by
  unfold riemann
  rw [pd_congr_chart 0 x h (fun y hy => su2_christoffel_fun 0 2 0 y hy),
      pd_congr_chart 2 x h (fun y hy => su2_christoffel_fun 0 0 0 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_0_1_0_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 0 1 0 1 x = 1/4 := by
  unfold riemann
  rw [pd_congr_chart 0 x h (fun y hy => su2_christoffel_fun 0 1 1 y hy),
      pd_congr_chart 1 x h (fun y hy => su2_christoffel_fun 0 0 1 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_0_1_0_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 0 1 0 2 x = Real.cos (x 0) / 4 := by
  unfold riemann
  rw [pd_congr_chart 0 x h (fun y hy => su2_christoffel_fun 0 2 1 y hy),
      pd_congr_chart 2 x h (fun y hy => su2_christoffel_fun 0 0 1 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_0_2_0_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 0 2 0 1 x = Real.cos (x 0) / 4 := by
  unfold riemann
  rw [pd_congr_chart 0 x h (fun y hy => su2_christoffel_fun 0 1 2 y hy),
      pd_congr_chart 1 x h (fun y hy => su2_christoffel_fun 0 0 2 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_0_2_0_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 0 2 0 2 x = 1/4 := by
  unfold riemann
  rw [pd_congr_chart 0 x h (fun y hy => su2_christoffel_fun 0 2 2 y hy),
      pd_congr_chart 2 x h (fun y hy => su2_christoffel_fun 0 0 2 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_1_0_1_0 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 1 0 1 0 x = -(-1/4) := by
  unfold riemann
  rw [pd_congr_chart 1 x h (fun y hy => su2_christoffel_fun 1 0 0 y hy),
      pd_congr_chart 0 x h (fun y hy => su2_christoffel_fun 1 1 0 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_1_0_1_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 1 0 1 2 x = 0 := by
  unfold riemann
  rw [pd_congr_chart 1 x h (fun y hy => su2_christoffel_fun 1 2 0 y hy),
      pd_congr_chart 2 x h (fun y hy => su2_christoffel_fun 1 1 0 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_1_1_1_0 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 1 1 1 0 x = 0 := by
  unfold riemann
  rw [pd_congr_chart 1 x h (fun y hy => su2_christoffel_fun 1 0 1 y hy),
      pd_congr_chart 0 x h (fun y hy => su2_christoffel_fun 1 1 1 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_1_1_1_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 1 1 1 2 x = Real.cos (x 0) / 4 := by
  unfold riemann
  rw [pd_congr_chart 1 x h (fun y hy => su2_christoffel_fun 1 2 1 y hy),
      pd_congr_chart 2 x h (fun y hy => su2_christoffel_fun 1 1 1 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_1_2_1_0 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 1 2 1 0 x = 0 := by
  unfold riemann
  rw [pd_congr_chart 1 x h (fun y hy => su2_christoffel_fun 1 0 2 y hy),
      pd_congr_chart 0 x h (fun y hy => su2_christoffel_fun 1 1 2 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_1_2_1_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 1 2 1 2 x = 1/4 := by
  unfold riemann
  rw [pd_congr_chart 1 x h (fun y hy => su2_christoffel_fun 1 2 2 y hy),
      pd_congr_chart 2 x h (fun y hy => su2_christoffel_fun 1 1 2 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_2_0_2_0 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 2 0 2 0 x = -(-1/4) := by
  unfold riemann
  rw [pd_congr_chart 2 x h (fun y hy => su2_christoffel_fun 2 0 0 y hy),
      pd_congr_chart 0 x h (fun y hy => su2_christoffel_fun 2 2 0 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_2_0_2_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 2 0 2 1 x = 0 := by
  unfold riemann
  rw [pd_congr_chart 2 x h (fun y hy => su2_christoffel_fun 2 1 0 y hy),
      pd_congr_chart 1 x h (fun y hy => su2_christoffel_fun 2 2 0 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_2_1_2_0 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 2 1 2 0 x = 0 := by
  unfold riemann
  rw [pd_congr_chart 2 x h (fun y hy => su2_christoffel_fun 2 0 1 y hy),
      pd_congr_chart 0 x h (fun y hy => su2_christoffel_fun 2 2 1 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_2_1_2_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 2 1 2 1 x = -(-1/4) := by
  unfold riemann
  rw [pd_congr_chart 2 x h (fun y hy => su2_christoffel_fun 2 1 1 y hy),
      pd_congr_chart 1 x h (fun y hy => su2_christoffel_fun 2 2 1 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_2_2_2_0 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 2 2 2 0 x = 0 := by
  unfold riemann
  rw [pd_congr_chart 2 x h (fun y hy => su2_christoffel_fun 2 0 2 y hy),
      pd_congr_chart 0 x h (fun y hy => su2_christoffel_fun 2 2 2 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_riemann_2_2_2_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann su2 2 2 2 1 x = -(-(Real.cos (x 0) / 4)) := by
  unfold riemann
  rw [pd_congr_chart 2 x h (fun y hy => su2_christoffel_fun 2 1 2 y hy),
      pd_congr_chart 1 x h (fun y hy => su2_christoffel_fun 2 2 2 y hy)]
  simp (disch := first | assumption | fun_prop (disch := assumption)) [pd, Fin.sum_univ_succ, Function.update_apply, deriv_fun_mul, deriv.fun_neg,
    deriv_fun_div, deriv_fun_inv'', deriv_div_const, deriv_mul_const, deriv_const_mul, Real.deriv_sin, Real.deriv_cos, su2_christoffel_0_0_0 x h, su2_christoffel_0_0_1 x h, su2_christoffel_0_0_2 x h, su2_christoffel_0_1_1 x h, su2_christoffel_0_1_2 x h, su2_christoffel_0_2_2 x h, su2_christoffel_1_0_0 x h, su2_christoffel_1_0_1 x h, su2_christoffel_1_0_2 x h, su2_christoffel_1_1_1 x h, su2_christoffel_1_1_2 x h, su2_christoffel_1_2_2 x h, su2_christoffel_2_0_0 x h, su2_christoffel_2_0_1 x h, su2_christoffel_2_0_2 x h, su2_christoffel_2_1_1 x h, su2_christoffel_2_1_2 x h, su2_christoffel_2_2_2 x h,
    christoffel_symm su2 su2_symm 0 1 0, christoffel_symm su2 su2_symm 0 2 0, christoffel_symm su2 su2_symm 0 2 1, christoffel_symm su2 su2_symm 1 1 0, christoffel_symm su2 su2_symm 1 2 0, christoffel_symm su2 su2_symm 1 2 1, christoffel_symm su2 su2_symm 2 1 0, christoffel_symm su2 su2_symm 2 2 0, christoffel_symm su2 su2_symm 2 2 1]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_ricci_0_0 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci su2 0 0 x = 1/2 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, su2_riemann_1_0_1_0 x h, su2_riemann_2_0_2_0 x h]
  all_goals first | ring1 | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_ricci_0_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci su2 0 1 x = 0 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, su2_riemann_0_0_0_1 x h, su2_riemann_2_0_2_1 x h]
  all_goals first | ring1 | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_ricci_0_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci su2 0 2 x = 0 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, su2_riemann_0_0_0_2 x h, su2_riemann_1_0_1_2 x h]
  all_goals first | ring1 | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_ricci_1_0 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci su2 1 0 x = 0 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, su2_riemann_1_1_1_0 x h, su2_riemann_2_1_2_0 x h]
  all_goals first | ring1 | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_ricci_1_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci su2 1 1 x = 1/2 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, su2_riemann_0_1_0_1 x h, su2_riemann_2_1_2_1 x h]
  all_goals first | ring1 | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_ricci_1_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci su2 1 2 x = Real.cos (x 0) / 2 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, su2_riemann_0_1_0_2 x h, su2_riemann_1_1_1_2 x h]
  all_goals first | ring1 | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_ricci_2_0 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci su2 2 0 x = 0 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, su2_riemann_1_2_1_0 x h, su2_riemann_2_2_2_0 x h]
  all_goals first | ring1 | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_ricci_2_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci su2 2 1 x = Real.cos (x 0) / 2 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, su2_riemann_0_2_0_1 x h, su2_riemann_2_2_2_1 x h]
  all_goals first | ring1 | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_ricci_2_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) : ricci su2 2 2 x = 1/2 := by
  simp [ricci, Fin.sum_univ_succ, riemann_diag, su2_riemann_0_2_0_2 x h, su2_riemann_1_2_1_2 x h]
  all_goals first | ring1 | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

theorem su2_scalarCurvature (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) : scalarCurvature su2 x = 6 := by
  simp [scalarCurvature, Fin.sum_univ_succ, su2_invMetric x h, su2_ricci_0_0 x h, su2_ricci_0_1 x h, su2_ricci_0_2 x h, su2_ricci_1_0 x h, su2_ricci_1_1 x h, su2_ricci_1_2 x h, su2_ricci_2_0 x h, su2_ricci_2_1 x h, su2_ricci_2_2 x h]
  field_simp
  ring_nf
  simp only [Real.cos_sq']
  ring

end KaluzaKlein.Geometry
