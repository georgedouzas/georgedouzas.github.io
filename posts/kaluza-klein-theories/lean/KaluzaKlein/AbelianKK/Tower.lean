import Mathlib

/-!
# The Kaluza-Klein tower (design D4)

Fourier modes on the circle of radius `R`: `e^{i n y / R}` is an eigenfunction of `∂_y²` with eigenvalue `-(n/R)²`,
the five-dimensional Klein-Gordon operator `□₅ = □₄ - ∂_y²` (signature `(+,-,-,-,-)`, `□₄ = ∂_t² - ∇²`) on a mode
gives the four-dimensional one with mass `|n|/R`, the
periodicity `y ~ y + 2πR` forces `n ∈ ℤ`, and the charge and the gauge coupling are algebraic consequences of the
normalisation of the reduced action.
-/

namespace KaluzaKlein.AbelianKK.Tower

open Complex

/-- The mode function `e^{i n y / R}` on the circle. -/
noncomputable def mode (n R y : ℝ) : ℂ := exp (I * n / R * y)

/-- `∂_y e^{i n y/R} = (i n/R) e^{i n y/R}`. -/
theorem hasDerivAt_mode (n R y : ℝ) : HasDerivAt (mode n R) (I * n / R * mode n R y) y := by
  have h : HasDerivAt (fun x : ℝ => (I * n / R) * (x : ℂ)) (I * n / R) y := by
    simpa using (Complex.ofRealCLM.hasDerivAt (x := y)).const_mul (I * n / R)
  have h2 := h.cexp
  unfold mode
  convert h2 using 1
  ring

/-- `∂_y² e^{i n y/R} = -(n/R)² e^{i n y/R}`: the mode is an eigenfunction of `∂_y²`. -/
theorem deriv2_mode (n R y : ℝ) :
    deriv (deriv (mode n R)) y = -((n / R : ℂ) ^ 2) * mode n R y := by
  have hd : deriv (mode n R) = fun y => I * n / R * mode n R y := by
    funext y; exact (hasDerivAt_mode n R y).deriv
  rw [hd]
  have h2 : HasDerivAt (fun y => I * n / R * mode n R y) (I * n / R * (I * n / R * mode n R y)) y :=
    (hasDerivAt_mode n R y).const_mul _
  rw [h2.deriv]
  linear_combination ((n : ℂ) / R) ^ 2 * mode n R y * I_sq

/-- The massless five-dimensional Klein-Gordon operator `□₅ = □₄ - ∂_y²` (mostly-minus signature, the circle
direction spacelike) on a product mode `φ(x) e^{i n y/R}` gives the four-dimensional operator with mass squared
`(n/R)²`: if `□₄ φ = c` then `(□₄ - ∂_y²)(φ e^{iny/R}) = (c + (n/R)² φ) e^{iny/R}`, so `□₅ Φ = 0` is
`(□₄ + (n/R)²) φ = 0` mode by mode. -/
theorem klein_gordon_mode (n R y : ℝ) (φ c : ℂ) :
    c * mode n R y - φ * deriv (deriv (mode n R)) y = (c + (n / R : ℂ) ^ 2 * φ) * mode n R y := by
  rw [deriv2_mode]; ring

/-- Periodicity: `e^{i n (y + 2πR)/R} = e^{i n y/R}` holds for every `y` exactly when `n` is an integer
(`R ≠ 0`): Klein's quantisation of the fifth-momentum. -/
theorem periodic_iff_int (n R y : ℝ) (hR : R ≠ 0) :
    mode n R (y + 2 * Real.pi * R) = mode n R y ↔ ∃ k : ℤ, (n : ℂ) = k := by
  unfold mode
  rw [Complex.exp_eq_exp_iff_exists_int]
  have hR' : (R : ℂ) ≠ 0 := by exact_mod_cast hR
  have hπ : (Real.pi : ℂ) ≠ 0 := by exact_mod_cast Real.pi_ne_zero
  have e1 : (I * n / R) * ((y : ℂ) + 2 * Real.pi * R) = I * n / R * y + I * n * (2 * Real.pi) := by
    field_simp
    all_goals ring
  constructor
  · rintro ⟨k, hk⟩
    refine ⟨k, ?_⟩
    push_cast at hk
    have h2 : (2 * Real.pi * I) * ((n : ℂ) - k) = 0 := by linear_combination hk - e1
    have hne : (2 * Real.pi * I : ℂ) ≠ 0 := mul_ne_zero (mul_ne_zero two_ne_zero hπ) I_ne_zero
    exact sub_eq_zero.mp ((mul_eq_zero.mp h2).resolve_left hne)
  · rintro ⟨k, hk⟩
    refine ⟨k, ?_⟩
    push_cast
    all_goals (rw [e1, hk]; ring)

/-- Masses of the tower: the mode `n` has mass `|n|/R`, the square root of the eigenvalue. -/
theorem mass_of_mode (n R : ℝ) (hR : 0 < R) : Real.sqrt ((n / R) ^ 2) = |n| / R := by
  rw [Real.sqrt_sq_eq_abs, abs_div, abs_of_pos hR]

/-- The gauge coupling of the reduced theory: with `2π κ² = m κ̂²` (`m = 1/R`) and charge unit `e = √2 κ m`
(Duff, hep-th/9410046, eqs. 8 and 16), the radius is fixed by the coupling, `R = √2 κ / e`, and the charge of the
mode `n` is `n e`. -/
theorem radius_of_coupling (κ e R : ℝ) (hR : R ≠ 0) (he : e = Real.sqrt 2 * κ / R) (hκ : κ ≠ 0) :
    R = Real.sqrt 2 * κ / e := by
  have h2 : Real.sqrt 2 ≠ 0 := by positivity
  rw [he]; field_simp

end KaluzaKlein.AbelianKK.Tower
