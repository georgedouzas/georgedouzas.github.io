import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.GammaFunction
import KaluzaKlein.PartI.ComplexAnalysis.BetaFunction
import KaluzaKlein.PartI.ComplexAnalysis.WickRotation
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Worked computations" of "Analytic continuation and special functions"

The claims of the two propositions of the section and of their examples and exercises in
`tutorial/part-i-04-complex-analysis.qmd`. Values of the special functions: `Γ(5/2) = 3√π/4`, `Γ(-1/2) = -2√π`,
`Γ(1/4)Γ(3/4) = π√2` (reflection, and duplication), `Γ(n + 1/2)` as a product, `Γ'(n + 1) = n!(H_n - γ)`
(`Γ'(1) = -γ`), the areas of spheres `S_{n-1} = 2π^{n/2}/Γ(n/2)` and their derivative in the dimension, the step of
the half-integer induction, the product `(-n)⋯(-1) = (-1)ⁿ n!` and `1/Γ`. The propagator: the residue as a limit and
as a partial-fraction coefficient, closing below, the decay `O(1/k²)`, the Minkowski integral with `iε` in closed form
and its numerical bounds, the Lorentzian weight `∫ ε/(x² + ε²) = π`, and the opposite prescription.
-/

namespace KaluzaKlein.ComplexAnalysis.SpecialWorked

open Complex Filter Topology Set MeasureTheory
open scoped Nat Real

/-! ## prp-gamma-values, exm-gamma-check-method -/

/-- `Γ(5/2) = (3/2)(1/2)√π = 3√π/4`. -/
theorem gamma_five_halves : Real.Gamma (5 / 2) = 3 / 4 * √π := by
  have h1 : Real.Gamma (3 / 2 + 1) = 3 / 2 * Real.Gamma (3 / 2) := Real.Gamma_add_one (by norm_num)
  have h2 : Real.Gamma (1 / 2 + 1) = 1 / 2 * Real.Gamma (1 / 2) := Real.Gamma_add_one (by norm_num)
  have h3 : Real.Gamma (1 / 2) = √π := Real.Gamma_one_half_eq
  rw [show (3 / 2 : ℝ) + 1 = 5 / 2 by norm_num] at h1
  rw [show (1 / 2 : ℝ) + 1 = 3 / 2 by norm_num] at h2
  generalize Real.Gamma (5 / 2) = A at h1 ⊢
  generalize Real.Gamma (3 / 2) = B at h1 h2
  generalize Real.Gamma (1 / 2) = C at h2 h3
  generalize √π = r at h3 ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination h1 + (3 / 2) * h2 + (3 / 4) * h3)
  -- certificate: ideal_membership.sage

/-- `Γ(-1/2) = Γ(1/2)/(-1/2) = -2√π`. -/
theorem gamma_neg_half : Real.Gamma (-1 / 2) = -2 * √π := by
  have h1 : Real.Gamma (-1 / 2 + 1) = (-1 / 2) * Real.Gamma (-1 / 2) := Real.Gamma_add_one (by norm_num)
  have h3 : Real.Gamma (1 / 2) = √π := Real.Gamma_one_half_eq
  rw [show (-1 / 2 : ℝ) + 1 = 1 / 2 by norm_num] at h1
  generalize Real.Gamma (-1 / 2) = A at h1 ⊢
  generalize Real.Gamma (1 / 2) = C at h1 h3
  generalize √π = r at h3 ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (2) * h1 + (-2) * h3)
  -- certificate: ideal_membership.sage

/-- `Γ(1/4)Γ(3/4) = π/sin(π/4) = π√2`, by the reflection formula. -/
theorem gamma_quarter_product : Real.Gamma (1 / 4) * Real.Gamma (3 / 4) = π * √2 := by
  have h := Real.Gamma_mul_Gamma_one_sub (1 / 4)
  have hs : Real.sin (π * (1 / 4)) = √2 / 2 := by
    rw [show π * (1 / 4) = π / 4 by ring]
    exact Real.sin_pi_div_four
  have h2 : √2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have h2' : √2 ≠ 0 := by positivity
  rw [hs, show (1 : ℝ) - 1 / 4 = 3 / 4 by norm_num] at h
  field_simp at h
  generalize Real.Gamma (1 / 4) = A at h ⊢
  generalize Real.Gamma (3 / 4) = B at h ⊢
  generalize √2 = r at h h2 ⊢
  generalize π = q at h ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-1 / 2 * A * B) * h2 + (1 / 2 * r) * h)
  -- certificate: ideal_membership.sage

/-! ## exm-gamma-half-integer-formula, exr-gamma-half-integers -/

/-- `Γ(n + 1/2) = (n - 1/2)(n - 3/2)⋯(1/2) √π`. -/
theorem gamma_half_integer_prod (n : ℕ) :
    Real.Gamma (n + 1 / 2) = (∏ k ∈ Finset.range n, ((k : ℝ) + 1 / 2)) * √π := by
  induction n with
  | zero =>
    rw [Nat.cast_zero, zero_add, Real.Gamma_one_half_eq]
    simp
  | succ n ih =>
    rw [show ((n + 1 : ℕ) : ℝ) + 1 / 2 = ((n : ℝ) + 1 / 2) + 1 by push_cast <;> ring,
      Real.Gamma_add_one (by positivity), ih, Finset.prod_range_succ]
    ring

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.SpecialWorked.gamma_half_integer_prod
-- test: ca-spec-half-prod-test.sage

/-- The induction step: `(2n + 1)/2 · (2n)!/(4ⁿ n!) = (2n + 2)!/(4^{n+1}(n + 1)!)`. -/
theorem half_integer_step (n : ℕ) :
    ((2 * n + 1 : ℝ) / 2) * ((2 * n)! / (4 ^ n * n !)) = (2 * n + 2)! / (4 ^ (n + 1) * (n + 1)!) := by
  have h1 : ((2 * n + 2)! : ℝ) = (2 * n + 2) * (2 * n + 1) * (2 * n)! := by
    rw [show 2 * n + 2 = (2 * n + 1) + 1 by ring, Nat.factorial_succ, Nat.factorial_succ]
    push_cast <;> ring
  have h2 : ((n + 1)! : ℝ) = (n + 1) * n ! := by
    rw [Nat.factorial_succ]
    push_cast <;> ring
  have hf : (n ! : ℝ) ≠ 0 := by positivity
  have h4 : (4 : ℝ) ^ n ≠ 0 := by positivity
  have hn : (n : ℝ) + 1 ≠ 0 := by positivity
  rw [h1, h2, pow_succ, div_mul_div_comm, div_eq_div_iff (by positivity) (by positivity)]
  generalize (((2 * n)! : ℕ) : ℝ) = A
  generalize ((n ! : ℕ) : ℝ) = B
  generalize (4 : ℝ) ^ n = C
  generalize (n : ℝ) = x
  clear h1 h2 hf h4 hn
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-spec-ring.sage

/-! ## exm-gamma-sphere-areas -/

/-- `Γ'(n + 1) = n!(H_n - γ)` (`n = 0`: `γ = -Γ'(1)`). -/
theorem gamma_deriv_nat (n : ℕ) :
    deriv Complex.Gamma (n + 1) = n ! * (((harmonic n : ℚ) : ℂ) - (Real.eulerMascheroniConstant : ℂ)) := by
  have h := Complex.digamma_nat_add_one n
  rw [Complex.digamma_def, logDeriv_apply, Complex.Gamma_nat_eq_factorial] at h
  have hf : (n ! : ℂ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero n
  rw [div_eq_iff hf] at h
  rw [h]
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.SpecialWorked.gamma_deriv_nat
-- test: ca-spec-gamma-deriv-test.sage

/-- The area of the unit sphere `S_{n-1} = n V_n = 2π^{n/2}/Γ(n/2)` (`S₁ = 2π`, `S₂ = 4π`, `S₃ = 2π²`). -/
theorem sphere_area (n : ℕ) (hn : 0 < n) :
    ((volume : Measure (EuclideanSpace ℝ (Fin n))).toSphere Set.univ).toReal =
      2 * π ^ ((n : ℝ) / 2) / Real.Gamma ((n : ℝ) / 2) := by
  rw [Measure.toSphere_apply_univ, finrank_euclideanSpace, Fintype.card_fin, ENNReal.toReal_mul,
    ENNReal.toReal_natCast, BetaFunction.unit_ball_volume n hn, ← BetaFunction.ball_volume_from_sphere n hn]
  have : (n : ℝ) ≠ 0 := by positivity
  field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.SpecialWorked.sphere_area
-- test: ca-spec-sphere-test.sage

/-- `S₁ = 2π¹/Γ(1) = 2π`, `S₂ = 2π^{3/2}/Γ(3/2) = 4π`, `S₃ = 2π²/Γ(2) = 2π²`. -/
theorem sphere_areas :
    2 * π ^ (1 : ℝ) / Real.Gamma 1 = 2 * π ∧ 2 * π ^ (3 / 2 : ℝ) / Real.Gamma (3 / 2) = 4 * π ∧
      2 * π ^ (2 : ℝ) / Real.Gamma 2 = 2 * π ^ 2 := by
  have hp1 : π ^ (1 : ℝ) = π := Real.rpow_one π
  have hp3 : π ^ (3 / 2 : ℝ) = π * √π := by
    rw [show (3 / 2 : ℝ) = 1 + 1 / 2 by norm_num, Real.rpow_add Real.pi_pos, Real.rpow_one, Real.sqrt_eq_rpow]
  have hp2 : π ^ (2 : ℝ) = π ^ 2 := Real.rpow_two π
  have hg1 : Real.Gamma 1 = 1 := Real.Gamma_one
  have hg3 : Real.Gamma (3 / 2) = 1 / 2 * √π := by
    have h := Real.Gamma_add_one (s := 1 / 2) (by norm_num)
    rw [show (1 / 2 : ℝ) + 1 = 3 / 2 by norm_num, Real.Gamma_one_half_eq] at h
    exact h
  have hg2 : Real.Gamma 2 = 1 := by
    rw [show (2 : ℝ) = (1 : ℕ) + 1 by norm_num, Real.Gamma_nat_eq_factorial]
    simp
  have hs : √π ≠ 0 := by positivity
  refine ⟨?_, ?_, ?_⟩
  · rw [div_eq_iff (by rw [hg1]; norm_num)]
    generalize π ^ (1 : ℝ) = P at hp1 ⊢
    generalize Real.Gamma 1 = G at hg1 ⊢
    generalize π = q at hp1 ⊢
    veriphy_certified (linear_combination (-2 * q) * hg1 + (2) * hp1)
    -- certificate: ideal_membership.sage
  · rw [div_eq_iff (by rw [hg3]; positivity)]
    generalize π ^ (3 / 2 : ℝ) = P at hp3 ⊢
    generalize Real.Gamma (3 / 2) = G at hg3 ⊢
    generalize √π = r at hp3 hg3
    generalize π = q at hp3 ⊢
    veriphy_certified (linear_combination (-4 * q) * hg3 + (2) * hp3)
    -- certificate: ideal_membership.sage
  · rw [div_eq_iff (by rw [hg2]; norm_num)]
    generalize π ^ (2 : ℝ) = P at hp2 ⊢
    generalize Real.Gamma 2 = G at hg2 ⊢
    generalize π = q at hp2 ⊢
    veriphy_certified (linear_combination (-2 * q ^ 2) * hg2 + (2) * hp2)
    -- certificate: ideal_membership.sage

/-- The derivative in the dimension: `d/dε [2π^{n/2-ε}/Γ(n/2 - ε)]` at `ε = 0` is
`2π^{n/2}/Γ(n/2) (ψ(n/2) - log π)` (`n = 4`: `2π²(1 - γ - log π)`, since `ψ(2) = 1 - γ`). -/
theorem sphere_area_deriv (n : ℕ) (hn : 0 < n) :
    HasDerivAt (fun e : ℂ => 2 * (π : ℂ) ^ ((n : ℂ) / 2 - e) / Complex.Gamma ((n : ℂ) / 2 - e))
      (2 * (π : ℂ) ^ ((n : ℂ) / 2) / Complex.Gamma ((n : ℂ) / 2) *
        (Complex.digamma ((n : ℂ) / 2) - Complex.log π)) 0 := by
  have hc0 : ∀ m : ℕ, (n : ℂ) / 2 ≠ -m := fun m h => by
    have := congrArg Complex.re h
    simp at this
    have hpos : (0 : ℝ) < n / 2 := by positivity
    linarith [(m.cast_nonneg : (0 : ℝ) ≤ m)]
  have hΓ : Complex.Gamma ((n : ℂ) / 2) ≠ 0 := Complex.Gamma_ne_zero hc0
  have hπ : (π : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr Real.pi_ne_zero
  have hlin : HasDerivAt (fun e : ℂ => (n : ℂ) / 2 - e) (-1) 0 := (hasDerivAt_id (0 : ℂ)).const_sub _
  have hpow := (hlin.const_cpow (c := (π : ℂ)) (Or.inl hπ)).const_mul 2
  have hG : HasDerivAt (fun e : ℂ => Complex.Gamma ((n : ℂ) / 2 - e))
      (deriv Complex.Gamma ((n : ℂ) / 2 - 0) * (-1)) 0 := by
    have := (Complex.differentiableAt_Gamma ((n : ℂ) / 2 - 0) (by simpa using hc0)).hasDerivAt
    exact this.comp 0 hlin
  have h := hpow.div hG (by simpa using hΓ)
  convert h using 1
  simp only [sub_zero]
  rw [Complex.digamma_def, logDeriv_apply]
  field_simp <;> ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.SpecialWorked.sphere_area_deriv
-- test: ca-spec-sphere-deriv-test.sage

/-! ## exr-gamma-duplication -/

/-- Legendre's duplication formula `Γ(z)Γ(z + 1/2) = 2^{1-2z}√π Γ(2z)`. -/
theorem gamma_duplication (z : ℂ) :
    Complex.Gamma z * Complex.Gamma (z + 1 / 2) = 2 ^ (1 - 2 * z) * (√π : ℝ) * Complex.Gamma (2 * z) := by
  rw [Complex.Gamma_mul_Gamma_add_half]
  ring

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.SpecialWorked.gamma_duplication
-- test: ca-spec-duplication-test.sage

/-- Duplication at `z = 1/4`: `Γ(1/4)Γ(3/4) = 2^{1/2}√π Γ(1/2) = √2 π`. -/
theorem gamma_quarter_duplication : Real.Gamma (1 / 4) * Real.Gamma (3 / 4) = √2 * π := by
  have h := Real.Gamma_mul_Gamma_add_half (1 / 4)
  rw [show (1 / 4 : ℝ) + 1 / 2 = 3 / 4 by norm_num, show 2 * (1 / 4 : ℝ) = 1 / 2 by norm_num,
    show (1 : ℝ) - 1 / 2 = 1 / 2 by norm_num, ← Real.sqrt_eq_rpow, Real.Gamma_one_half_eq] at h
  have hs : √π ^ 2 = π := Real.sq_sqrt Real.pi_pos.le
  generalize Real.Gamma (1 / 4) = A at h ⊢
  generalize Real.Gamma (3 / 4) = B at h ⊢
  generalize √2 = r at h ⊢
  generalize √π = t at h hs
  generalize π = q at hs ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination h + (r) * hs)
  -- certificate: ideal_membership.sage

/-! ## exr-gamma-residues -/

/-- `(-n)(-n + 1)⋯(-1) = (-1)ⁿ n!`. -/
theorem prod_neg_range (n : ℕ) : ∏ k ∈ Finset.range n, ((k : ℤ) - n) = (-1) ^ n * n ! := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.prod_range_succ']
    have e : ∏ k ∈ Finset.range n, (((k + 1 : ℕ) : ℤ) - ((n + 1 : ℕ) : ℤ)) = ∏ k ∈ Finset.range n, ((k : ℤ) - n) := by
      refine Finset.prod_congr rfl fun k _ => ?_
      push_cast <;> ring
    rw [e, ih, Nat.factorial_succ]
    push_cast <;> ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.SpecialWorked.prod_neg_range (0) := by
  norm_num [Finset.prod_range_succ, Nat.factorial]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.SpecialWorked.prod_neg_range (1) := by
  norm_num [Finset.prod_range_succ, Nat.factorial]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.SpecialWorked.prod_neg_range (2) := by
  norm_num [Finset.prod_range_succ, Nat.factorial]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.SpecialWorked.prod_neg_range (3) := by
  norm_num [Finset.prod_range_succ, Nat.factorial]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.SpecialWorked.prod_neg_range (4) := by
  norm_num [Finset.prod_range_succ, Nat.factorial]

attribute [veriphy_tested kernel_test 5 0] KaluzaKlein.ComplexAnalysis.SpecialWorked.prod_neg_range
-- test: ca-spec-neg-prod-test.sage

/-- `1/Γ(z) = Γ(1 - z) sin(πz)/π` for `z ∉ ℤ`, from the reflection formula. -/
theorem inv_gamma_reflection' (z : ℂ) (hz : ∀ m : ℤ, z ≠ m) :
    (Complex.Gamma z)⁻¹ = Complex.Gamma (1 - z) * Complex.sin (π * z) / π := by
  have h := Complex.Gamma_mul_Gamma_one_sub z
  have hπ : (π : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr Real.pi_ne_zero
  have hΓ : Complex.Gamma z ≠ 0 := Complex.Gamma_ne_zero (fun m hm => hz (-m) (by rw [hm]; simp))
  have hs : Complex.sin (π * z) ≠ 0 := by
    rw [Ne, Complex.sin_eq_zero_iff]
    rintro ⟨k, hk⟩
    exact hz k (mul_left_cancel₀ hπ (hk.trans (mul_comm _ _)))
  have h' : Complex.Gamma z * Complex.Gamma (1 - z) * Complex.sin (π * z) = π := by
    rw [h]
    field_simp
  rw [eq_div_iff hπ]
  calc (Complex.Gamma z)⁻¹ * π
      = (Complex.Gamma z)⁻¹ * (Complex.Gamma z * Complex.Gamma (1 - z) * Complex.sin (π * z)) := by rw [h']
    _ = Complex.Gamma (1 - z) * Complex.sin (π * z) := by field_simp

/-- `1/Γ(z) = Γ(1 - z) sin(πz)/π`, i.e. `Γ(z) · (Γ(1 - z) sin πz) = π`, for `z ∉ ℤ`. -/
theorem inv_gamma_reflection (z : ℂ) (hz : ∀ m : ℤ, z ≠ m) :
    Complex.Gamma z * (Complex.Gamma (1 - z) * Complex.sin (π * z)) = π := by
  have hπ : (π : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr Real.pi_ne_zero
  have hΓ : Complex.Gamma z ≠ 0 := Complex.Gamma_ne_zero (fun m hm => hz (-m) (by rw [hm]; simp))
  rw [← div_mul_cancel₀ (Complex.Gamma (1 - z) * Complex.sin (π * z)) hπ, ← inv_gamma_reflection' z hz,
    ← mul_assoc, mul_inv_cancel₀ hΓ, one_mul]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.SpecialWorked.inv_gamma_reflection
-- test: ca-spec-inv-gamma-refl-test.sage

/-- The zeros of `1/Γ` are exactly `0, -1, -2, …`. -/
theorem inv_gamma_eq_zero_iff (z : ℂ) : (Complex.Gamma z)⁻¹ = 0 ↔ ∃ n : ℕ, z = -n := by
  rw [inv_eq_zero, Complex.Gamma_eq_zero_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.SpecialWorked.inv_gamma_eq_zero_iff (0) := by
  rw [inv_eq_zero, Complex.Gamma_eq_zero_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.SpecialWorked.inv_gamma_eq_zero_iff (-2) := by
  rw [inv_eq_zero, Complex.Gamma_eq_zero_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.SpecialWorked.inv_gamma_eq_zero_iff (1) := by
  rw [inv_eq_zero, Complex.Gamma_eq_zero_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.SpecialWorked.inv_gamma_eq_zero_iff (1 / 2) := by
  rw [inv_eq_zero, Complex.Gamma_eq_zero_iff]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.SpecialWorked.inv_gamma_eq_zero_iff
-- test: ca-spec-inv-gamma-zeros-test.sage

/-- The zeros of `1/Γ` are simple: `(1/Γ)'(-n) = (-1)ⁿ n!`. -/
theorem inv_gamma_hasDerivAt (n : ℕ) :
    HasDerivAt (fun z => (Complex.Gamma z)⁻¹) ((-1) ^ n * n !) (-(n : ℂ)) := by
  have hres := GammaFunction.gamma_residue n
  have hne : ((-1 : ℂ) ^ n / n !) ≠ 0 := by
    have : (n ! : ℂ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero n
    exact div_ne_zero (pow_ne_zero _ (by norm_num)) this
  rw [hasDerivAt_iff_tendsto_slope]
  have h0 : (Complex.Gamma (-(n : ℂ)))⁻¹ = 0 := by
    rw [inv_eq_zero, Complex.Gamma_eq_zero_iff]
    exact ⟨n, rfl⟩
  have hlim := hres.inv₀ hne
  have hval : ((-1 : ℂ) ^ n / n !)⁻¹ = (-1) ^ n * n ! := by
    have hp : ((-1 : ℂ) ^ n) ≠ 0 := pow_ne_zero _ (by norm_num)
    have hsq : ((-1 : ℂ) ^ n) ^ 2 = 1 := by
      rw [← pow_mul, mul_comm, pow_mul]
      norm_num
    rw [inv_div, div_eq_iff hp]
    linear_combination (-(n ! : ℂ)) * hsq
  rw [hval] at hlim
  refine hlim.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with z hz
  rw [slope_def_field, h0, sub_zero, sub_neg_eq_add, mul_inv, div_eq_mul_inv, mul_comm]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.SpecialWorked.inv_gamma_hasDerivAt
-- test: ca-spec-inv-gamma-simple-test.sage

/-! ## prp-wick-rotation, exm-wick-by-residues -/

/-- The residue at `-E + iδ` as a limit: `(k - (-E + iδ)) h(k) → 1/((-E + iδ) - (E - iδ))`. -/
theorem propagator_residue (E δ : ℝ) (hE : 0 < E) :
    Tendsto (fun k : ℂ => (k - (-E + δ * I)) * (1 / ((k - (E - δ * I)) * (k - (-E + δ * I)))))
      (𝓝[≠] (-E + δ * I)) (𝓝 (1 / ((-E + δ * I) - (E - δ * I)))) := by
  have hne : (-(E : ℂ) + δ * I) - (E - δ * I) ≠ 0 := fun h => by
    have := congrArg Complex.re h
    simp at this
    linarith
  have hc : ContinuousAt (fun k : ℂ => 1 / (k - (E - δ * I))) (-E + δ * I) :=
    continuousAt_const.div (continuousAt_id.sub continuousAt_const) hne
  refine (hc.tendsto.mono_left nhdsWithin_le_nhds).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with k hk
  have hk' : k - (-(E : ℂ) + δ * I) ≠ 0 := sub_ne_zero.mpr hk
  rw [mul_one_div, mul_comm (k - ((E : ℂ) - δ * I)) _, ← div_div, div_self hk']

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.SpecialWorked.propagator_residue
-- test: ca-spec-prop-res-test.sage

/-- The residue at `-E + iδ` is `1/((-E + iδ) - (E - iδ))` with `(-E + iδ) - (E - iδ) = -2E + 2iδ`, so
`2πi Res_{-E+iδ} h = 2πi/(-2E + 2iδ)`. -/
theorem residue_value (E δ : ℝ) : (-(E : ℂ) + δ * I) - (E - δ * I) = -2 * E + 2 * δ * I := by
  generalize I = j
  veriphy_certified (ring1)
  -- certificate: ca-spec-ring.sage

/-- Partial fractions: `1/((k - p₁)(k - p₂)) = (1/(p₂ - p₁))/(k - p₂) - (1/(p₂ - p₁))/(k - p₁)`; the residue at `p₂`
is the coefficient `1/(p₂ - p₁)`. -/
theorem partial_fractions (k p₁ p₂ : ℂ) (h1 : k - p₁ ≠ 0) (h2 : k - p₂ ≠ 0) (h12 : p₂ - p₁ ≠ 0) :
    1 / ((k - p₁) * (k - p₂)) = 1 / (p₂ - p₁) / (k - p₂) - 1 / (p₂ - p₁) / (k - p₁) := by
  rw [div_div, div_div, div_sub_div _ _ (mul_ne_zero h12 h2) (mul_ne_zero h12 h1),
    div_eq_div_iff (mul_ne_zero h1 h2) (mul_ne_zero (mul_ne_zero h12 h2) (mul_ne_zero h12 h1))]
  veriphy_certified (ring1)
  -- certificate: ca-spec-ring.sage

/-- Closing below picks `-2πi Res_{E-iδ} h = -2πi/(2E - 2iδ)`, the same value. -/
theorem lower_closing (E δ : ℝ) (hE : 0 < E) :
    -(2 * π * I) * (1 / ((E - δ * I) - (-E + δ * I))) = 2 * π * I / (-2 * E + 2 * δ * I) := by
  have h1 : ((E : ℂ) - δ * I) - (-E + δ * I) ≠ 0 := fun h => by
    have := congrArg Complex.re h
    simp at this
    linarith
  have h2 : (-2 * (E : ℂ) + 2 * δ * I) ≠ 0 := fun h => by
    have := congrArg Complex.re h
    simp at this
    linarith
  rw [mul_one_div, div_eq_div_iff h1 h2]
  veriphy_certified (ring1)
  -- certificate: ca-spec-ring.sage

/-- `h` decays like `1/k²`: `|1/(k² - c²)| ≤ 4/(3k²)` for `|k| ≥ 2|c|`. -/
theorem propagator_decay (c : ℂ) (k : ℝ) (hk : 2 * ‖c‖ ≤ |k|) : ‖1 / ((k : ℂ) ^ 2 - c ^ 2)‖ ≤ 4 / (3 * k ^ 2) := by
  rcases eq_or_ne k 0 with h0 | h0
  · subst h0
    have hk0 : 2 * ‖c‖ ≤ 0 := by simpa using hk
    have hc : c = 0 := norm_le_zero_iff.mp (by linarith [norm_nonneg c])
    subst hc
    simp
  have hk2 : 0 < k ^ 2 := by positivity
  have hlow : 3 * k ^ 2 / 4 ≤ ‖(k : ℂ) ^ 2 - c ^ 2‖ := by
    have h1 : ‖((k : ℂ) ^ 2)‖ - ‖c ^ 2‖ ≤ ‖(k : ℂ) ^ 2 - c ^ 2‖ := norm_sub_norm_le _ _
    rw [norm_pow, norm_pow, Complex.norm_real, Real.norm_eq_abs, sq_abs] at h1
    have h2 : ‖c‖ ^ 2 ≤ k ^ 2 / 4 := by
      have : ‖c‖ ≤ |k| / 2 := by linarith
      have hc0 := norm_nonneg c
      calc ‖c‖ ^ 2 ≤ (|k| / 2) ^ 2 := pow_le_pow_left₀ hc0 this 2
        _ = k ^ 2 / 4 := by rw [div_pow, sq_abs]; ring
    linarith
  have hpos : 0 < ‖(k : ℂ) ^ 2 - c ^ 2‖ := by linarith
  rw [norm_div, norm_one, div_le_div_iff₀ hpos (by positivity)]
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.SpecialWorked.propagator_decay
-- test: ca-spec-decay-test.sage

/-- The Minkowski integral in closed form: with `c² = E² - iε` and `Im c < 0`, `∫ dk/(k² - E² + iε) = -πi/c`. -/
theorem minkowski_integral' (E ε : ℝ) (c : ℂ) (hc : c ^ 2 = E ^ 2 - ε * I) (him : c.im < 0) :
    ∫ k : ℝ, 1 / ((k : ℂ) ^ 2 - E ^ 2 + ε * I) = -(π * I / c) := by
  have h := WickRotation.integral_inv_sub_mul_add' (-c) (by simpa using him)
  have e : ∀ k : ℝ, (k : ℂ) ^ 2 - E ^ 2 + ε * I = ((k : ℂ) - -c) * ((k : ℂ) + -c) := fun k => by
    linear_combination hc
  simp_rw [e]
  rw [h, div_neg]

/-- The Minkowski integral in closed form: with `c² = E² - iε` and `Im c < 0`, `∫ dk/(k² - E² + iε) = -πi/c`, i.e.
`c ∫ dk/(k² - E² + iε) = -πi`. -/
theorem minkowski_integral (E ε : ℝ) (c : ℂ) (hc : c ^ 2 = E ^ 2 - ε * I) (him : c.im < 0) :
    c * ∫ k : ℝ, 1 / ((k : ℂ) ^ 2 - E ^ 2 + ε * I) = -(π * I) := by
  have h0 : c ≠ 0 := fun h => by rw [h] at him; simp at him
  rw [minkowski_integral' E ε c hc him, mul_neg, mul_div_cancel₀ _ h0]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.SpecialWorked.minkowski_integral
-- test: ca-spec-mink-int-test.sage

/-- With `c² = E² - iε`, `Re c > 0` (the pole in the fourth quadrant): the imaginary part of the Minkowski integral
is `-π/E` up to `πε²/(2E⁵)` and its real part is at most `πε/(2E³)` (`E = 1`, `ε = 10⁻⁴`: `-π` to `10⁻³`). -/
theorem minkowski_numeric (E ε : ℝ) (hE : 0 < E) (hε : 0 < ε) (c : ℂ) (hc : c ^ 2 = E ^ 2 - ε * I)
    (hre : 0 < c.re) :
    |(∫ k : ℝ, 1 / ((k : ℂ) ^ 2 - E ^ 2 + ε * I)).im + π / E| ≤ π * ε ^ 2 / (2 * E ^ 5) ∧
      |(∫ k : ℝ, 1 / ((k : ℂ) ^ 2 - E ^ 2 + ε * I)).re| ≤ π * ε / (2 * E ^ 3) := by
  obtain ⟨x, y, rfl⟩ : ∃ x y : ℝ, c = x + y * I := ⟨c.re, c.im, (Complex.re_add_im c).symm⟩
  have hx : 0 < x := by simpa using hre
  have h1 : x ^ 2 - y ^ 2 = E ^ 2 := by
    have := congrArg Complex.re hc
    simp [sq] at this
    linarith
  have h2 : 2 * x * y = -ε := by
    have := congrArg Complex.im hc
    simp [sq] at this
    linarith
  have hy : y < 0 := by nlinarith
  rw [minkowski_integral' E ε _ hc (by simpa using hy)]
  have hN : 0 < x ^ 2 + y ^ 2 := by positivity
  have hre' : (-(π * I / ((x : ℂ) + y * I))).re = -π * y / (x ^ 2 + y ^ 2) := by
    simp [Complex.div_re, Complex.normSq_apply]
    field_simp <;> ring
  have him' : (-(π * I / ((x : ℂ) + y * I))).im = -π * x / (x ^ 2 + y ^ 2) := by
    simp [Complex.div_im, Complex.normSq_apply]
    field_simp <;> ring
  rw [hre', him']
  have hxE : E ≤ x := by nlinarith
  have hNE : E ^ 2 ≤ x ^ 2 + y ^ 2 := by nlinarith
  have hε2 : ε ^ 2 = 4 * x ^ 2 * y ^ 2 := by rw [← neg_sq, ← h2]; ring
  constructor
  · have hV : -π * x / (x ^ 2 + y ^ 2) + π / E = π * ((x ^ 2 + y ^ 2) - E * x) / (E * (x ^ 2 + y ^ 2)) := by
      field_simp <;> ring
    rw [hV]
    have hA : 0 ≤ (x ^ 2 + y ^ 2) - E * x := by nlinarith
    have hA' : (x ^ 2 + y ^ 2) - E * x ≤ 2 * y ^ 2 := by nlinarith
    have hB : E ^ 4 ≤ x ^ 2 * (x ^ 2 + y ^ 2) := by nlinarith
    rw [abs_of_nonneg (div_nonneg (mul_nonneg Real.pi_pos.le hA) (by positivity)),
      div_le_div_iff₀ (by positivity) (by positivity), hε2]
    nlinarith [mul_le_mul_of_nonneg_left hA' (by positivity : (0 : ℝ) ≤ π * (2 * E ^ 5)),
      mul_le_mul_of_nonneg_left hB (by positivity : (0 : ℝ) ≤ 4 * π * y ^ 2 * E), Real.pi_pos]
  · have hxN : E ^ 3 ≤ x * (x ^ 2 + y ^ 2) := by nlinarith
    rw [abs_of_nonneg (div_nonneg (by nlinarith [Real.pi_pos]) hN.le), div_le_div_iff₀ hN (by positivity),
      show ε = -(2 * x * y) by linarith]
    nlinarith [mul_nonneg (mul_nonneg Real.pi_pos.le (neg_nonneg.mpr hy.le)) (sub_nonneg.mpr hxN)]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.SpecialWorked.minkowski_numeric
-- test: ca-spec-mink-num-test.sage

/-! ## exm-wick-numerical-check -/

/-- The weight of `-iπδ(x)`: `∫ ε/(x² + ε²) dx = π` for every `ε > 0`. -/
theorem lorentzian_mass (ε : ℝ) (hε : 0 < ε) : ∫ x : ℝ, ε / (x ^ 2 + ε ^ 2) = π := by
  simp_rw [div_eq_mul_one_div ε]
  rw [MeasureTheory.integral_const_mul, WickRotation.euclidean_integral ε hε]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.SpecialWorked.lorentzian_mass
-- test: ca-spec-lorentz-test.sage

/-! ## exr-wick-sign -/

/-- The opposite prescription: poles `E + iδ`, `-E - iδ`; `∫ dk/((k - (E + iδ))(k - (-E - iδ))) = 2πi/(2E + 2iδ)`. -/
theorem propagator_opposite' (E δ : ℝ) (hE : 0 < E) (hδ : 0 < δ) :
    ∫ k : ℝ, 1 / (((k : ℂ) - (E + δ * I)) * ((k : ℂ) - (-E - δ * I))) = 2 * π * I / (2 * E + 2 * δ * I) := by
  have h := WickRotation.integral_inv_sub_mul_add' (E + δ * I) (by simpa using hδ)
  have e : ∀ k : ℝ, ((k : ℂ) - (E + δ * I)) * ((k : ℂ) - (-E - δ * I)) =
      ((k : ℂ) - (E + δ * I)) * ((k : ℂ) + (E + δ * I)) := fun k => by ring
  simp_rw [e]
  rw [h]
  have h0 : ((E : ℂ) + δ * I) ≠ 0 := fun h0 => by
    have := congrArg Complex.re h0
    simp at this
    linarith
  have h0' : (2 * (E : ℂ) + 2 * δ * I) ≠ 0 := fun h1 => h0 (by linear_combination h1 / 2)
  field_simp <;> ring

/-- The opposite prescription, multiplied out: `(2E + 2iδ) ∫ dk/((k - (E + iδ))(k - (-E - iδ))) = 2πi`. -/
theorem propagator_opposite (E δ : ℝ) (hE : 0 < E) (hδ : 0 < δ) :
    (2 * E + 2 * δ * I) * ∫ k : ℝ, 1 / (((k : ℂ) - (E + δ * I)) * ((k : ℂ) - (-E - δ * I))) = 2 * π * I := by
  have h0 : (2 * (E : ℂ) + 2 * δ * I) ≠ 0 := fun h1 => by
    have := congrArg Complex.re h1
    simp at this
    linarith
  rw [propagator_opposite' E δ hE hδ, mul_div_cancel₀ _ h0]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.SpecialWorked.propagator_opposite
-- test: ca-spec-prop-opp-test.sage

/-- As `δ → 0⁺` the opposite prescription gives `2πi/(2E) = iπ/E`. -/
theorem propagator_opposite_limit (E : ℝ) (hE : 0 < E) :
    Tendsto (fun δ : ℝ => 2 * π * I / (2 * E + 2 * δ * I)) (𝓝[>] 0) (𝓝 (π * I / E)) := by
  have hE' : (E : ℂ) ≠ 0 := by exact_mod_cast hE.ne'
  have hc : ContinuousAt (fun δ : ℝ => 2 * π * I / (2 * E + 2 * δ * I)) 0 := by
    apply ContinuousAt.div (by fun_prop) (by fun_prop)
    simp [hE']
  have h : Tendsto (fun δ : ℝ => 2 * π * I / (2 * E + 2 * δ * I)) (𝓝[>] (0 : ℝ))
      (𝓝 (2 * π * I / (2 * E + 2 * ((0 : ℝ) : ℂ) * I))) := hc.tendsto.mono_left nhdsWithin_le_nhds
  have hv : (2 * π * I / (2 * E + 2 * ((0 : ℝ) : ℂ) * I) : ℂ) = π * I / E := by
    simp only [Complex.ofReal_zero, mul_zero, zero_mul, add_zero]
    field_simp
  rw [hv] at h
  exact h

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.SpecialWorked.propagator_opposite_limit
-- test: ca-spec-prop-opp-limit-test.sage

/-- The clockwise rotation `k⁰ = -ik⁴`: `-i ∫ dk⁴/((-ik⁴)² - E²) = iπ/E`. -/
theorem euclidean_side_clockwise' (E : ℝ) (hE : 0 < E) :
    -I * ∫ k : ℝ, 1 / ((-I * k) ^ 2 - (E : ℂ) ^ 2) = π * I / E := by
  have e : ∀ k : ℝ, 1 / ((-I * k) ^ 2 - (E : ℂ) ^ 2) = -(((1 / (k ^ 2 + E ^ 2) : ℝ)) : ℂ) := fun k => by
    have : (k : ℂ) ^ 2 + (E : ℂ) ^ 2 ≠ 0 := by exact_mod_cast (by positivity : k ^ 2 + E ^ 2 ≠ 0)
    push_cast
    rw [← one_div_neg_eq_neg_one_div]
    congr 1
    linear_combination (k : ℂ) ^ 2 * Complex.I_sq
  simp_rw [e]
  rw [MeasureTheory.integral_neg, integral_complex_ofReal, WickRotation.euclidean_integral E hE]
  push_cast <;> ring

/-- The clockwise rotation, multiplied out: `E · (-i ∫ dk⁴/((-ik⁴)² - E²)) = iπ`. -/
theorem euclidean_side_clockwise (E : ℝ) (hE : 0 < E) :
    (E : ℂ) * (-I * ∫ k : ℝ, 1 / ((-I * k) ^ 2 - (E : ℂ) ^ 2)) = π * I := by
  have h0 : (E : ℂ) ≠ 0 := by exact_mod_cast hE.ne'
  rw [euclidean_side_clockwise' E hE, mul_div_cancel₀ _ h0]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.SpecialWorked.euclidean_side_clockwise
-- test: ca-spec-eucl-cw-test.sage

/-- The two prescriptions are complex conjugates: `∫ dk/(k² - E² - iε) = conj ∫ dk/(k² - E² + iε)`. -/
theorem prescriptions_conj (E ε : ℝ) :
    ∫ k : ℝ, 1 / ((k : ℂ) ^ 2 - E ^ 2 - ε * I) = (starRingEnd ℂ) (∫ k : ℝ, 1 / ((k : ℂ) ^ 2 - E ^ 2 + ε * I)) := by
  rw [← integral_conj]
  congr 1
  funext k
  simp [map_div₀, sub_eq_add_neg]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.SpecialWorked.prescriptions_conj
-- test: ca-spec-conj-test.sage

end KaluzaKlein.ComplexAnalysis.SpecialWorked
