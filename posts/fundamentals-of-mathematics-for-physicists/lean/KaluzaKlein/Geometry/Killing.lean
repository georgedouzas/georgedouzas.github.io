import KaluzaKlein.Conventions
import KaluzaKlein.Geometry.Metrics

/-!
# Killing vectors of the round sphere

The three rotation generators `ξ_a = ε_abc x^b ∂_c` of `ℝ³`, restricted to the unit sphere and written in spherical
coordinates, are Killing vector fields of the round metric, and their Lie brackets close into `so(3)` with the
opposite sign to the matrices `L_a` of `Conventions.lean`: `[K_a, K_b] = −ε_abc K_c` (the vector fields of a left
action form an anti-representation). The Killing property is a `coordinate_geometry` certificate (`killing_proof`);
the brackets are proved by expanding the definition of `lieBracket`.
-/

namespace KaluzaKlein.Geometry

open CoordinateGeometry

/-- `K₁ = −sin φ ∂_θ − cot θ cos φ ∂_φ`, the generator of rotations about the `X`-axis. -/
noncomputable def killingX : Fin 2 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => -Real.sin (x 1), fun x => -(Real.cos (x 0) / Real.sin (x 0)) * Real.cos (x 1)]

/-- `K₂ = cos φ ∂_θ − cot θ sin φ ∂_φ`, the generator of rotations about the `Y`-axis. -/
noncomputable def killingY : Fin 2 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => Real.cos (x 1), fun x => -(Real.cos (x 0) / Real.sin (x 0)) * Real.sin (x 1)]

/-- `K₃ = ∂_φ`, the generator of rotations about the `Z`-axis. -/
noncomputable def killingZ : Fin 2 → (Fin 2 → ℝ) → ℝ :=
  ![fun _ => 0, fun _ => 1]

/-- `∂_φ` is a Killing vector of the round sphere. -/
theorem killingZ_isKilling : IsKilling sphere killingZ := by
  intro a b x
  fin_cases a <;> fin_cases b <;>
    simp (disch := fun_prop) [CoordinateGeometry.lieMetric, CoordinateGeometry.pd, KaluzaKlein.Geometry.sphere,
      KaluzaKlein.Geometry.killingZ, Fin.sum_univ_succ, Function.update_apply, deriv_fun_pow, deriv_fun_mul,
      deriv_fun_add, deriv_fun_sub, Real.deriv_sin, Real.deriv_cos, Real.deriv_exp]
  all_goals ring1
  -- certificate: sphere_killing.sage

/-- The Lie derivative of the round metric along `K₁` vanishes on the chart domain. -/
theorem killingX_lieMetric (i j : Fin 2) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lieMetric sphere killingX i j x = 0 := by
  fin_cases i <;> fin_cases j <;>
    simp (disch :=
      first
      | assumption
      | fun_prop) [CoordinateGeometry.lieMetric,
      CoordinateGeometry.pd, KaluzaKlein.Geometry.sphere, KaluzaKlein.Geometry.killingX,
      KaluzaKlein.Geometry.killingY, Fin.sum_univ_succ, Function.update_apply, deriv_fun_pow, deriv_fun_mul,
      deriv.fun_neg, deriv_fun_div, deriv_fun_inv'', Real.deriv_sin, Real.deriv_cos]
  all_goals
    first
    | ring1
    | (field_simp; done)
    | (field_simp; ring1)
    | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)
  -- certificate: sphere_killing_chart.sage

/-- The Lie derivative of the round metric along `K₂` vanishes on the chart domain. -/
theorem killingY_lieMetric (i j : Fin 2) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lieMetric sphere killingY i j x = 0 := by
  fin_cases i <;> fin_cases j <;>
    simp (disch :=
      first
      | assumption
      | fun_prop) [CoordinateGeometry.lieMetric,
      CoordinateGeometry.pd, KaluzaKlein.Geometry.sphere, KaluzaKlein.Geometry.killingX,
      KaluzaKlein.Geometry.killingY, Fin.sum_univ_succ, Function.update_apply, deriv_fun_pow, deriv_fun_mul,
      deriv.fun_neg, deriv_fun_div, deriv_fun_inv'', Real.deriv_sin, Real.deriv_cos]
  all_goals
    first
    | ring1
    | (field_simp; done)
    | (field_simp; ring1)
    | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)
  -- certificate: sphere_killing_chart.sage

/-! ## The brackets close into `so(3)`, with the sign of an anti-representation -/

/-- `[K₁, K₂] = −K₃`. -/
theorem bracket_killingX_killingY (k : Fin 2) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lieBracket killingX killingY k x = -killingZ k x := by
  fin_cases k <;>
    simp (disch := first | assumption | fun_prop) [lieBracket, pd, killingX, killingY, killingZ, Fin.sum_univ_succ,
      Function.update_apply, deriv_fun_mul, deriv.fun_neg, deriv_fun_div, deriv_fun_inv'', Real.deriv_sin, Real.deriv_cos]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

/-- `[K₂, K₃] = −K₁`. -/
theorem bracket_killingY_killingZ (k : Fin 2) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lieBracket killingY killingZ k x = -killingX k x := by
  fin_cases k <;>
    simp (disch := first | assumption | fun_prop) [lieBracket, pd, killingX, killingY, killingZ, Fin.sum_univ_succ,
      Function.update_apply, deriv_fun_mul, deriv.fun_neg, deriv_fun_div, deriv_fun_inv'', Real.deriv_sin, Real.deriv_cos]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

/-- `[K₃, K₁] = −K₂`. -/
theorem bracket_killingZ_killingX (k : Fin 2) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lieBracket killingZ killingX k x = -killingY k x := by
  fin_cases k <;>
    simp (disch := first | assumption | fun_prop) [lieBracket, pd, killingX, killingY, killingZ, Fin.sum_univ_succ,
      Function.update_apply, deriv_fun_mul, deriv.fun_neg, deriv_fun_div, deriv_fun_inv'', Real.deriv_sin, Real.deriv_cos]
  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)

end KaluzaKlein.Geometry
