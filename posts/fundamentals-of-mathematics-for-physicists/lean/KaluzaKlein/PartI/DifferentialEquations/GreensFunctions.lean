import Mathlib
import Veriphy.Kit.Levels

/-!
# Differential equations, part "Green's functions"

The claims of the part "Green's functions" of `tutorial/part-i-05-differential-equations.qmd`, one namespace per
section. The Dirac delta is modelled in two ways, both as statements about ordinary functions: as a limit of
Gaussians `g_ε` of unit integral (their integrals against test functions), and in weak form, `LG = δ` meaning
`∫ G (L*f) = f(0)` for test functions `f` vanishing (with their derivative) at the ends of the interval. Fundamental
solutions are checked in radial form (`∇²G = G'' + (n-1)G'/r`) and by their fluxes; the string's Green's function is
the tent `x_<(1 - x_>)`; the retarded and Feynman prescriptions are checked on the one-dimensional `k₀` integrals and
the oscillator Green's function.
-/

open Real Filter Topology MeasureTheory
open scoped ComplexOrder

namespace KaluzaKlein.DifferentialEquations

/-! Helper lemmas of the hand proofs below (private: each module carries its own copy). -/

private lemma deriv_eventually_pos {g g' : ℝ → ℝ} {r : ℝ} (hr : 0 < r)
    (h : ∀ s, 0 < s → HasDerivAt g (g' s) s) : (fun s => deriv g s) =ᶠ[𝓝 r] g' := by
  filter_upwards [Ioi_mem_nhds hr] with s hs
  exact (h s hs).deriv

private lemma coulomb_slice (a b s : ℝ) (hq : 0 < s ^ 2 + a ^ 2 + b ^ 2) :
    HasDerivAt (fun s => s / (s ^ 2 + a ^ 2 + b ^ 2) ^ (3 / 2 : ℝ))
      ((s ^ 2 + a ^ 2 + b ^ 2 - 3 * s ^ 2) / ((s ^ 2 + a ^ 2 + b ^ 2) ^ 2 * Real.sqrt (s ^ 2 + a ^ 2 + b ^ 2))) s := by
  have h1 : HasDerivAt (fun s => s ^ 2 + a ^ 2 + b ^ 2) (2 * s) s := by
    have := ((hasDerivAt_pow 2 s).add_const (a ^ 2)).add_const (b ^ 2)
    convert this using 1; norm_num
  have h2 := h1.rpow_const (p := (3 / 2 : ℝ)) (Or.inr (by norm_num))
  have hq32 : (s ^ 2 + a ^ 2 + b ^ 2) ^ (3 / 2 : ℝ) ≠ 0 := (Real.rpow_pos_of_pos hq _).ne'
  have h3 : HasDerivAt (fun s => s / (s ^ 2 + a ^ 2 + b ^ 2) ^ (3 / 2 : ℝ))
      ((1 * (s ^ 2 + a ^ 2 + b ^ 2) ^ (3 / 2 : ℝ) -
        s * (2 * s * (3 / 2) * (s ^ 2 + a ^ 2 + b ^ 2) ^ ((3 / 2 : ℝ) - 1))) /
        ((s ^ 2 + a ^ 2 + b ^ 2) ^ (3 / 2 : ℝ)) ^ 2) s := (hasDerivAt_id' s).div h2 hq32
  have e1 : (s ^ 2 + a ^ 2 + b ^ 2) ^ (3 / 2 : ℝ) = (s ^ 2 + a ^ 2 + b ^ 2) * Real.sqrt (s ^ 2 + a ^ 2 + b ^ 2) := by
    rw [show (3 / 2 : ℝ) = 1 + 1 / 2 by norm_num, Real.rpow_add hq, Real.rpow_one, Real.sqrt_eq_rpow]
  have e2 : (s ^ 2 + a ^ 2 + b ^ 2) ^ ((3 / 2 : ℝ) - 1) = Real.sqrt (s ^ 2 + a ^ 2 + b ^ 2) := by
    rw [show (3 / 2 : ℝ) - 1 = 1 / 2 by norm_num, Real.sqrt_eq_rpow]
  rw [e2, e1] at h3
  refine h3.congr_deriv ?_
  have hs : Real.sqrt (s ^ 2 + a ^ 2 + b ^ 2) ≠ 0 := (Real.sqrt_pos.mpr hq).ne'
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)

private lemma int_x_sin (n ξ : ℝ) (hn : n ≠ 0) :
    ∫ x in (0 : ℝ)..ξ, x * Real.sin (n * x) = Real.sin (n * ξ) / n ^ 2 - ξ * Real.cos (n * ξ) / n := by
  have hF : ∀ x : ℝ, HasDerivAt (fun x => -x * Real.cos (n * x) / n + Real.sin (n * x) / n ^ 2) (x * Real.sin (n * x)) x := by
    intro x
    have hneg : HasDerivAt (fun x : ℝ => -x) (-1) x := hasDerivAt_neg x
    have h1' : HasDerivAt (fun x => -x * Real.cos (n * x)) _ x := hneg.mul ((hasDerivAt_id' x).const_mul n).cos
    have h1 : HasDerivAt (fun x => -x * Real.cos (n * x) / n) _ x := h1'.div_const n
    have h2 : HasDerivAt (fun x => Real.sin (n * x) / n ^ 2) _ x := (((hasDerivAt_id' x).const_mul n).sin).div_const (n ^ 2)
    refine (h1.add h2).congr_deriv ?_
    first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hF x) (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [mul_zero, neg_zero, zero_mul, zero_div, Real.sin_zero, add_zero, sub_zero]
  ring

end KaluzaKlein.DifferentialEquations

namespace KaluzaKlein.DifferentialEquations.DeltaFunction

/-! ## def-dirac-delta and exm-delta-gaussian -/

/-- The Gaussians `g_ε(x) = e^{-x²/ε²}/(√π ε)` have unit integral for every `ε > 0`. -/
theorem gaussian_unit_integral (ε : ℝ) (hε : 0 < ε) :
    ∫ x, Real.exp (-x ^ 2 / ε ^ 2) / (Real.sqrt Real.pi * ε) = 1 := by
  have e : (fun x : ℝ => Real.exp (-x ^ 2 / ε ^ 2) / (Real.sqrt Real.pi * ε)) =
      fun x => (Real.sqrt Real.pi * ε)⁻¹ * Real.exp (-(1 / ε ^ 2) * x ^ 2) := by
    funext x; rw [div_eq_inv_mul]; congr 2; ring
  rw [e, integral_const_mul, integral_gaussian, show Real.pi / (1 / ε ^ 2) = Real.pi * ε ^ 2 by field_simp,
    Real.sqrt_mul Real.pi_pos.le, Real.sqrt_sq hε.le]
  have : Real.sqrt Real.pi ≠ 0 := (Real.sqrt_pos.mpr Real.pi_pos).ne'
  field_simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.DeltaFunction.gaussian_unit_integral
-- test: de-gaussian-unit-test.sage

/-- The step's derivative is the delta, in weak form: `∫θ'f = -∫₀^∞ f' = f(0)` for `f` vanishing at infinity. -/
theorem step_derivative (f f' : ℝ → ℝ) (hf : ∀ x, HasDerivAt f (f' x) x) (hi : IntegrableOn f' (Set.Ioi 0))
    (hlim : Tendsto f atTop (𝓝 0)) : -∫ x in Set.Ioi (0 : ℝ), f' x = f 0 := by
  rw [integral_Ioi_of_hasDerivAt_of_tendsto (continuous_iff_continuousAt.mpr fun x => (hf x).continuousAt).continuousWithinAt
    (fun x _ => hf x) hi hlim]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.DeltaFunction.step_derivative
-- test: de-step-derivative-test.sage

/-- The heat kernel is `g_ε` with `ε² = 4κt`: `√π·√(4κt) = √(4πκt)`. -/
theorem kernel_is_gaussian (κ t : ℝ) (h : 0 ≤ 4 * κ * t) :
    Real.sqrt Real.pi * Real.sqrt (4 * κ * t) = Real.sqrt (4 * Real.pi * κ * t) := by
  veriphy_certified
    ( rw [← Real.sqrt_mul Real.pi_pos.le]
      ring_nf)
  -- certificate: de-rewrite.sage

/-- Against `f = eˣ`: `∫ g_ε(x) eˣ dx = e^{ε²/4}` (completing the square). -/
theorem gaussian_against_exp (ε : ℝ) (hε : 0 < ε) :
    ∫ x, Real.exp (-x ^ 2 / ε ^ 2) / (Real.sqrt Real.pi * ε) * Real.exp x = Real.exp (ε ^ 2 / 4) := by
  have e : (fun x : ℝ => Real.exp (-x ^ 2 / ε ^ 2) / (Real.sqrt Real.pi * ε) * Real.exp x) =
      fun x => (Real.exp (ε ^ 2 / 4) / (Real.sqrt Real.pi * ε)) * Real.exp (-(1 / ε ^ 2) * (x - ε ^ 2 / 2) ^ 2) := by
    funext x
    rw [div_mul_eq_mul_div, mul_comm (Real.exp (ε ^ 2 / 4) / _), ← mul_div_assoc, ← Real.exp_add, ← Real.exp_add]
    congr 2
    field_simp
    ring
  rw [e, integral_const_mul, integral_sub_right_eq_self (fun x => Real.exp (-(1 / ε ^ 2) * x ^ 2)) (ε ^ 2 / 2),
    integral_gaussian, show Real.pi / (1 / ε ^ 2) = Real.pi * ε ^ 2 by field_simp,
    Real.sqrt_mul Real.pi_pos.le, Real.sqrt_sq hε.le]
  have : Real.sqrt Real.pi ≠ 0 := (Real.sqrt_pos.mpr Real.pi_pos).ne'
  field_simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.DeltaFunction.gaussian_against_exp
-- test: de-gaussian-exp-test.sage

/-- Against `f = cos x`: `∫ g_ε(x) cos x dx = e^{-ε²/4}`. -/
theorem gaussian_against_cos (ε : ℝ) (hε : 0 < ε) :
    ∫ x, Real.exp (-x ^ 2 / ε ^ 2) / (Real.sqrt Real.pi * ε) * Real.cos x = Real.exp (-ε ^ 2 / 4) := by
  set b : ℂ := ((1 / ε ^ 2 : ℝ) : ℂ) with hbdef
  have hb : 0 < b.re := by rw [hbdef, Complex.ofReal_re]; positivity
  have hF := fourierIntegral_gaussian hb 1
  have hpt : ∀ x : ℝ, (Complex.exp (Complex.I * 1 * x) * Complex.exp (-b * x ^ 2)).re =
      Real.cos x * Real.exp (-x ^ 2 / ε ^ 2) := by
    intro x
    have e1 : Complex.exp (-b * x ^ 2) = ((Real.exp (-x ^ 2 / ε ^ 2) : ℝ) : ℂ) := by
      rw [Complex.ofReal_exp]; congr 1; rw [hbdef]; push_cast; ring
    rw [e1, Complex.re_mul_ofReal, show Complex.I * 1 * (x : ℂ) = (x : ℂ) * Complex.I by ring, Complex.exp_mul_I]
    simp [Complex.cos_ofReal_re]
  have hint : MeasureTheory.Integrable fun x : ℝ => Complex.exp (Complex.I * 1 * x) * Complex.exp (-b * x ^ 2) := by
    have := integrable_cexp_quadratic hb (Complex.I * 1) 0
    refine this.congr (Eventually.of_forall fun x => ?_)
    simp only [add_zero]
    rw [← Complex.exp_add]
    ring_nf
  have hre := congrArg Complex.re hF
  have hir := integral_re hint
  simp only [RCLike.re_to_complex] at hir
  rw [← hir] at hre
  simp only [hpt] at hre
  have hπb : (Real.pi : ℂ) / b = ((Real.pi * ε ^ 2 : ℝ) : ℂ) := by rw [hbdef]; push_cast; field_simp
  have hexp : -(1 : ℂ) ^ 2 / (4 * b) = ((-ε ^ 2 / 4 : ℝ) : ℂ) := by rw [hbdef]; push_cast; field_simp
  have hhalf : (1 / 2 : ℂ) = ((1 / 2 : ℝ) : ℂ) := by push_cast; ring
  rw [hπb, hexp, hhalf, ← Complex.ofReal_cpow (by positivity), ← Complex.ofReal_exp, ← Complex.ofReal_mul,
    Complex.ofReal_re, ← Real.sqrt_eq_rpow, Real.sqrt_mul Real.pi_pos.le, Real.sqrt_sq hε.le] at hre
  have e : (fun x : ℝ => Real.exp (-x ^ 2 / ε ^ 2) / (Real.sqrt Real.pi * ε) * Real.cos x) =
      fun x => (Real.sqrt Real.pi * ε)⁻¹ * (Real.cos x * Real.exp (-x ^ 2 / ε ^ 2)) := by
    funext x; field_simp
  rw [e, integral_const_mul, hre]
  have : Real.sqrt Real.pi * ε ≠ 0 := by positivity
  field_simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.DeltaFunction.gaussian_against_cos
-- test: de-gaussian-cos-test.sage

/-- … and the values tend to `f(0)`: `e^{cε²} → 1` as `ε → 0` (`c = -1/4` for `cos`, `c = 1/4` for `eˣ`). -/
theorem gaussian_limits (c : ℝ) : Tendsto (fun ε : ℝ => Real.exp (c * ε ^ 2)) (𝓝 0) (𝓝 1) := by
  have : Continuous fun ε : ℝ => Real.exp (c * ε ^ 2) := by fun_prop
  simpa using this.tendsto 0
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.DeltaFunction.gaussian_limits
-- test: de-gaussian-limits-test.sage

/-! ## exm-delta-step (with exm-green-string, prp-greens-laplacian): a kink gives a delta -/

/-- The string's Green's function in weak form: for `f` with `f(0) = f(1) = 0`,
`-∫₀¹ G(x, ξ) f''(x) dx = f(ξ)` with `G = x(1 - ξ)` on `[0, ξ]` and `ξ(1 - x)` on `[ξ, 1]`: the slope jump `-1` of `G`
at `ξ` is `-G'' = δ(x - ξ)`. -/
theorem tent_weak (ξ : ℝ) (f f' f'' : ℝ → ℝ) (hf : ∀ x, HasDerivAt f (f' x) x) (hf' : ∀ x, HasDerivAt f' (f'' x) x)
    (hc : Continuous f'') (h0 : f 0 = 0) (h1 : f 1 = 0) :
    -((∫ x in (0 : ℝ)..ξ, x * (1 - ξ) * f'' x) + ∫ x in ξ..1, ξ * (1 - x) * f'' x) = f ξ := by
  have hF1 : ∀ x, HasDerivAt (fun x => (1 - ξ) * (x * f' x - f x)) (x * (1 - ξ) * f'' x) x := by
    intro x
    convert (((hasDerivAt_id' x).mul (hf' x)).sub (hf x)).const_mul (1 - ξ) using 1
    simp; ring
  have hF2 : ∀ x, HasDerivAt (fun x => ξ * ((1 - x) * f' x + f x)) (ξ * (1 - x) * f'' x) x := by
    intro x
    convert ((((hasDerivAt_id' x).const_sub 1).mul (hf' x)).add (hf x)).const_mul ξ using 1
    simp; ring
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hF1 x) (by apply Continuous.intervalIntegrable; fun_prop),
    intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hF2 x) (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [h0, h1]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.DeltaFunction.tent_weak
-- test: de-tent-weak-test.sage

/-! ## exm-delta-flux (with exm-fund-laplacian-3d, prp-greens-laplacian) -/

/-- `F = x/(4πr³)` is divergence free away from the origin: `∂ₓ(x/r³) + ∂_y(y/r³) + ∂_z(z/r³) = 0`. -/
theorem coulomb_field_divergence (x y z : ℝ) (h : 0 < x ^ 2 + y ^ 2 + z ^ 2) :
    deriv (fun s => s / (s ^ 2 + y ^ 2 + z ^ 2) ^ (3 / 2 : ℝ)) x +
      deriv (fun s => s / (x ^ 2 + s ^ 2 + z ^ 2) ^ (3 / 2 : ℝ)) y +
      deriv (fun s => s / (x ^ 2 + y ^ 2 + s ^ 2) ^ (3 / 2 : ℝ)) z = 0 := by
  have e2 : (fun s => s / (x ^ 2 + s ^ 2 + z ^ 2) ^ (3 / 2 : ℝ)) = fun s => s / (s ^ 2 + x ^ 2 + z ^ 2) ^ (3 / 2 : ℝ) := by
    funext s; rw [show x ^ 2 + s ^ 2 + z ^ 2 = s ^ 2 + x ^ 2 + z ^ 2 by ring]
  have e3 : (fun s => s / (x ^ 2 + y ^ 2 + s ^ 2) ^ (3 / 2 : ℝ)) = fun s => s / (s ^ 2 + x ^ 2 + y ^ 2) ^ (3 / 2 : ℝ) := by
    funext s; rw [show x ^ 2 + y ^ 2 + s ^ 2 = s ^ 2 + x ^ 2 + y ^ 2 by ring]
  rw [e2, e3, (coulomb_slice y z x h).deriv, (coulomb_slice x z y (by linarith)).deriv,
    (coulomb_slice x y z (by linarith)).deriv]
  have q1 : y ^ 2 + x ^ 2 + z ^ 2 = x ^ 2 + y ^ 2 + z ^ 2 := by ring
  have q2 : z ^ 2 + x ^ 2 + y ^ 2 = x ^ 2 + y ^ 2 + z ^ 2 := by ring
  rw [q1, q2]
  have hs : Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2) ≠ 0 := (Real.sqrt_pos.mpr h).ne'
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.DeltaFunction.coulomb_field_divergence
-- test: de-coulomb-field-divergence-test.sage

/-- Its flux through the sphere of radius `R` is `1`: `∫₀^{2π}∫₀^π (1/(4πR²)) R² sin θ dθ dφ = 1`. -/
theorem coulomb_flux (R : ℝ) (hR : R ≠ 0) :
    ∫ φ in (0 : ℝ)..(2 * Real.pi), ∫ θ in (0 : ℝ)..Real.pi, 1 / (4 * Real.pi * R ^ 2) * (R ^ 2 * Real.sin θ) = 1 := by
  have hpi : Real.pi ≠ 0 := Real.pi_ne_zero
  simp only [intervalIntegral.integral_const_mul, integral_sin, Real.cos_pi, Real.cos_zero,
    intervalIntegral.integral_const, smul_eq_mul]
  field_simp
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.DeltaFunction.coulomb_flux
-- test: de-coulomb-flux-test.sage

/-! ## exr-delta-scaling -/

/-- `δ(ax) = δ(x)/|a|`, in integral form: `∫ g(ax) f(x) dx = |a|⁻¹ ∫ g(y) f(y/a) dy` (substitution `y = ax`). -/
theorem delta_scaling (g f : ℝ → ℝ) (a : ℝ) (ha : a ≠ 0) :
    ∫ x, g (a * x) * f x = |a⁻¹| * ∫ y, g y * f (y / a) := by
  have := Measure.integral_comp_mul_left (fun y => g y * f (y / a)) a
  simp only [smul_eq_mul] at this
  rw [← this]
  congr 1; funext x
  rw [mul_div_cancel_left₀ x ha]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.DeltaFunction.delta_scaling
-- test: de-delta-scaling-test.sage

/-- The zeros `±a` of `h(x) = x² - a²` have `|h'(±a)| = 2|a|`, giving `δ(x² - a²) = [δ(x - a) + δ(x + a)]/(2|a|)`. -/
theorem delta_square_derivative (a : ℝ) : deriv (fun s => s ^ 2 - a ^ 2) a = 2 * a := by
  veriphy_certified
    ( have hd : HasDerivAt (fun s => s ^ 2 - a ^ 2) ((2 * a : ℝ)) (a) :=
        by
        refine
          (((((hasDerivAt_id' (a : ℝ)).pow 2).add (hasDerivAt_const (a : ℝ) (-(a ^ 2) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      exact hd.deriv.trans (by rfl))
  -- certificate: de-derivative.sage

/-! ## exr-delta-eigenfunction-expansion -/

/-- Integrating `(2/π)Σₙ sin nξ sin nx` against `sin mx` term by term leaves `sin mξ` (orthogonality), for a finite
sum containing `m`. -/
theorem delta_series_against_sine (N m : ℕ) (hm : m ∈ Finset.Icc 1 N) (ξ : ℝ) :
    ∑ n ∈ Finset.Icc 1 N, 2 / Real.pi * Real.sin (n * ξ) *
      ∫ x in (0 : ℝ)..Real.pi, Real.sin (n * x) * Real.sin (m * x) = Real.sin (m * ξ) := by
  have hsn : ∀ n : ℕ, n ≠ 0 → ∫ x in (0 : ℝ)..Real.pi, Real.sin (n * x) ^ 2 = Real.pi / 2 := by
    intro n hn
    have hrw : (fun x : ℝ => Real.sin (n * x) ^ 2) = fun x => (1 - Real.cos (((2 * n : ℕ) : ℤ) * x)) / 2 := by
      funext x; push_cast
      rw [show 2 * (n : ℝ) * x = 2 * (n * x) by ring, Real.cos_two_mul, Real.sin_sq]; ring
    have h2 : (((2 * n : ℕ) : ℤ)) ≠ 0 := by omega
    rw [hrw, intervalIntegral.integral_div, intervalIntegral.integral_sub
      (by apply Continuous.intervalIntegrable; fun_prop) (by apply Continuous.intervalIntegrable; fun_prop),
      intervalIntegral.integral_comp_mul_left (fun x => Real.cos x) (by exact_mod_cast h2), integral_cos]
    have h2π : Real.sin (2 * (n : ℝ) * Real.pi) = 0 := by exact_mod_cast Real.sin_nat_mul_pi (2 * n)
    simp [h2π]
  have hcos : ∀ k : ℤ, k ≠ 0 → ∫ x in (0 : ℝ)..Real.pi, Real.cos (k * x) = 0 := by
    intro k hk
    rw [intervalIntegral.integral_comp_mul_left (fun x => Real.cos x) (by exact_mod_cast hk), integral_cos]
    simp [Real.sin_int_mul_pi]
  have horth : ∀ n : ℕ, n ≠ m → ∫ x in (0 : ℝ)..Real.pi, Real.sin (n * x) * Real.sin (m * x) = 0 := by
    intro n h
    have hrw : (fun x : ℝ => Real.sin (n * x) * Real.sin (m * x)) =
        fun x => (Real.cos (((n : ℤ) - m : ℤ) * x) - Real.cos (((n : ℤ) + m : ℤ) * x)) / 2 := by
      funext x; push_cast; rw [sub_mul, add_mul, Real.cos_sub, Real.cos_add]; ring
    rw [hrw, intervalIntegral.integral_div, intervalIntegral.integral_sub
      (by apply Continuous.intervalIntegrable; fun_prop) (by apply Continuous.intervalIntegrable; fun_prop),
      hcos _ (by omega), hcos _ (by omega)]
    simp
  rw [Finset.sum_eq_single m]
  · rw [show (fun x : ℝ => Real.sin (m * x) * Real.sin (m * x)) = fun x => Real.sin (m * x) ^ 2 by
      funext x; ring, hsn m (by simp at hm; omega)]
    field_simp
  · intro n _ hn; rw [horth n hn, mul_zero]
  · intro h; exact absurd hm h
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.DeltaFunction.delta_series_against_sine
-- test: de-delta-series-test.sage

end KaluzaKlein.DifferentialEquations.DeltaFunction

namespace KaluzaKlein.DifferentialEquations.FundamentalSolutions

/-! ## def-fundamental-solution and exr-fund-1d: `G = |x|/2` and `G'' = δ` in weak form -/

/-- The one-dimensional fundamental solution in weak form: for `f` with `f(±L) = f'(±L) = 0`,
`∫_{-L}^{L} (|x|/2) f''(x) dx = f(0)`; equivalently `u = ∫ G(x - y) f(y) dy` solves `u'' = f`. -/
theorem abs_weak (L : ℝ) (hL : 0 ≤ L) (f f' f'' : ℝ → ℝ) (hf : ∀ x, HasDerivAt f (f' x) x)
    (hf' : ∀ x, HasDerivAt f' (f'' x) x) (hc : Continuous f'') (h₁ : f L = 0) (h₂ : f (-L) = 0) (h₃ : f' L = 0)
    (h₄ : f' (-L) = 0) :
    (∫ x in (-L)..0, -x / 2 * f'' x) + ∫ x in (0 : ℝ)..L, x / 2 * f'' x = f 0 := by
  have hF1 : ∀ x, HasDerivAt (fun x => -(x * f' x - f x) / 2) (-x / 2 * f'' x) x := by
    intro x
    convert (((hasDerivAt_id' x).mul (hf' x)).sub (hf x)).neg.div_const 2 using 1
    simp; ring
  have hF2 : ∀ x, HasDerivAt (fun x => (x * f' x - f x) / 2) (x / 2 * f'' x) x := by
    intro x
    convert (((hasDerivAt_id' x).mul (hf' x)).sub (hf x)).div_const 2 using 1
    simp; ring
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hF1 x) (by apply Continuous.intervalIntegrable; fun_prop),
    intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hF2 x) (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [h₁, h₂, h₃, h₄]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.FundamentalSolutions.abs_weak
-- test: de-abs-weak-test.sage

/-- `G'' = 0` away from the origin: `G = x/2` for `x > 0` … -/
theorem abs_right (x : ℝ) (hx : 0 < x) : deriv (fun s => |s| / 2) x = 1 / 2 := by
  have : (fun s => |s| / 2) =ᶠ[𝓝 x] fun s => s / 2 := by
    filter_upwards [Ioi_mem_nhds hx] with s hs
    rw [abs_of_pos hs]
  rw [this.deriv_eq]
  simpa using ((hasDerivAt_id' x).div_const 2).deriv
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.FundamentalSolutions.abs_right
-- test: de-abs-right-test.sage

/-- … and `-x/2` for `x < 0`: the slope jumps by `1/2 - (-1/2) = 1` at `0`. -/
theorem abs_left (x : ℝ) (hx : x < 0) : deriv (fun s => |s| / 2) x = -1 / 2 := by
  have : (fun s => |s| / 2) =ᶠ[𝓝 x] fun s => -s / 2 := by
    filter_upwards [Iio_mem_nhds hx] with s hs
    rw [abs_of_neg hs]
  rw [this.deriv_eq]
  simpa [neg_div] using ((hasDerivAt_id' x).neg.div_const 2).deriv
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.FundamentalSolutions.abs_left
-- test: de-abs-left-test.sage

/-! ## exm-fund-laplacian-3d and exr-fund-gauss -/

/-- `-1/(4πr)` is harmonic for `r > 0` (radial Laplacian `G'' + (2/r)G'`). -/
theorem coulomb_harmonic (r : ℝ) (hr : 0 < r) :
    deriv (fun s => deriv (fun ρ => -1 / (4 * Real.pi * ρ)) s) r + 2 / r * deriv (fun ρ => -1 / (4 * Real.pi * ρ)) r = 0 := by
  have hin : ∀ s : ℝ, 0 < s → HasDerivAt (fun ρ => -1 / (4 * Real.pi * ρ)) (1 / (4 * Real.pi * s ^ 2)) s := by
    intro s hs
    have h : HasDerivAt (fun ρ => -1 / (4 * Real.pi * ρ)) ((0 * (4 * Real.pi * s) - -1 * (4 * Real.pi * 1)) / (4 * Real.pi * s) ^ 2) s :=
      (hasDerivAt_const s (-1 : ℝ)).div ((hasDerivAt_id' s).const_mul (4 * Real.pi)) (mul_ne_zero (by positivity) hs.ne')
    exact h.congr_deriv (by field_simp; ring)
  rw [(deriv_eventually_pos hr hin).deriv_eq, (hin r hr).deriv]
  have hp : HasDerivAt (fun s : ℝ => s ^ 2) (2 * r) r := by simpa using hasDerivAt_pow 2 r
  have h2 : HasDerivAt (fun s => 1 / (4 * Real.pi * s ^ 2)) _ r :=
    (hasDerivAt_const r (1 : ℝ)).div (hp.const_mul (4 * Real.pi)) (by positivity)
  rw [h2.deriv]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.FundamentalSolutions.coulomb_harmonic
-- test: de-coulomb-harmonic-test.sage

/-- Its radial derivative is `G' = 1/(4πr²)` … -/
theorem coulomb_gradient (r : ℝ) (hr : r ≠ 0) : deriv (fun ρ => -1 / (4 * Real.pi * ρ)) r = 1 / (4 * Real.pi * r ^ 2) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun ρ => -1 / (4 * π * ρ)) ((1 / (4 * Real.pi * (r ^ 2)) : ℝ)) (r) :=
        by
        convert
          (((hasDerivAt_const (r : ℝ) (1 : ℝ)).div (hasDerivAt_id' (r : ℝ)) hr).const_mul
            (-1 / (4 * Real.pi) : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.div_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: de-derivative.sage

/-- … so the flux through the sphere of radius `r` is `4πr² G'(r) = 1`. -/
theorem coulomb_unit_flux (r : ℝ) (hr : r ≠ 0) : 4 * Real.pi * r ^ 2 * (1 / (4 * Real.pi * r ^ 2)) = 1 := by
  veriphy_certified (field_simp)
  -- certificate: de-rewrite.sage

/-- Radial and harmonic: `(r²G')' = 0` gives `r²G' = A`, and `G = -A/r + C` indeed has `r²G' = A`. -/
theorem radial_general (A C r : ℝ) (hr : r ≠ 0) : r ^ 2 * deriv (fun ρ => -A / ρ + C) r = A := by
  veriphy_certified
    ( have h0 : HasDerivAt (fun ρ => -A / ρ + C) ((A / (r ^ 2) : ℝ)) (r) :=
        by
        refine
          ((((hasDerivAt_const (r : ℝ) (C : ℝ)).add
                    (((hasDerivAt_const (r : ℝ) (1 : ℝ)).div (hasDerivAt_id' (r : ℝ)) hr).const_mul
                      (-A : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h0.deriv]
      try beta_reduce
      all_goals ((field_simp; done)))
  -- certificate: de-partials.sage

/-! ## exm-fund-laplacian-2d (with exr-pgl-two-dimensions) -/

/-- `G = log r/(2π)` has `G' = 1/(2πr)` … -/
theorem log_gradient (r : ℝ) (hr : 0 < r) : deriv (fun ρ => Real.log ρ / (2 * Real.pi)) r = 1 / (2 * Real.pi * r) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun ρ => log ρ / (2 * π)) ((1 / (2 * Real.pi * r) : ℝ)) (r) :=
        by
        refine
          (((((hasDerivAt_id' (r : ℝ)).log (by exact hr.ne')).const_mul (1 / (2 * Real.pi) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (ring_nf; done)
        · ring1
      exact hd.deriv.trans (by rfl))
  -- certificate: de-derivative.sage

/-- … and unit flux `G'(r)·2πr = 1` through every circle … -/
theorem log_unit_flux (r : ℝ) (hr : r ≠ 0) : 1 / (2 * Real.pi * r) * (2 * Real.pi * r) = 1 := by
  veriphy_certified (field_simp)
  -- certificate: de-rewrite.sage

/-- … and is harmonic: `G'' + G'/r = 0` for `r > 0`. -/
theorem log_harmonic (r : ℝ) (hr : 0 < r) :
    deriv (fun s => deriv (fun ρ => Real.log ρ / (2 * Real.pi)) s) r + 1 / r * deriv (fun ρ => Real.log ρ / (2 * Real.pi)) r = 0 := by
  have hin : ∀ s : ℝ, 0 < s → HasDerivAt (fun ρ => Real.log ρ / (2 * Real.pi)) (s⁻¹ / (2 * Real.pi)) s :=
    fun s hs => (Real.hasDerivAt_log hs.ne').div_const _
  rw [(deriv_eventually_pos hr hin).deriv_eq, (hin r hr).deriv]
  have h2 : HasDerivAt (fun s : ℝ => s⁻¹ / (2 * Real.pi)) (-(r ^ 2)⁻¹ / (2 * Real.pi)) r :=
    (hasDerivAt_inv hr.ne').div_const _
  rw [h2.deriv]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.FundamentalSolutions.log_harmonic
-- test: de-log-harmonic-gf-test.sage

/-! ## exm-fund-yukawa (with prp-greens-laplacian, exr-pgl-yukawa-radial) -/

/-- `G = e^{-mr}/(4πr)` solves `-(G'' + (2/r)G') + m²G = 0` for `r > 0`. -/
theorem yukawa_solves (m r : ℝ) (hr : 0 < r) :
    -(deriv (fun s => deriv (fun ρ => Real.exp (-m * ρ) / (4 * Real.pi * ρ)) s) r +
      2 / r * deriv (fun ρ => Real.exp (-m * ρ) / (4 * Real.pi * ρ)) r) +
      m ^ 2 * (Real.exp (-m * r) / (4 * Real.pi * r)) = 0 := by
  have hin : ∀ s : ℝ, 0 < s → HasDerivAt (fun ρ => Real.exp (-m * ρ) / (4 * Real.pi * ρ))
      ((Real.exp (-m * s) * (-m * 1) * (4 * Real.pi * s) - Real.exp (-m * s) * (4 * Real.pi * 1)) / (4 * Real.pi * s) ^ 2) s :=
    fun s hs => (((hasDerivAt_id' s).const_mul (-m)).exp).div ((hasDerivAt_id' s).const_mul (4 * Real.pi))
      (mul_ne_zero (by positivity) hs.ne')
  rw [(deriv_eventually_pos hr hin).deriv_eq, (hin r hr).deriv]
  have hE : HasDerivAt (fun s => Real.exp (-m * s)) (Real.exp (-m * r) * (-m * 1)) r :=
    ((hasDerivAt_id' r).const_mul (-m)).exp
  have hlin : HasDerivAt (fun s => 4 * Real.pi * s) (4 * Real.pi * 1) r := (hasDerivAt_id' r).const_mul (4 * Real.pi)
  have hsq : HasDerivAt (fun s => (4 * Real.pi * s) ^ 2) (2 * (4 * Real.pi * r) * (4 * Real.pi * 1)) r := by
    have := hlin.pow 2
    convert this using 1; norm_num
  have hN1 : HasDerivAt (fun s => Real.exp (-m * s) * (-m * 1)) _ r := hE.mul_const (-m * 1)
  have hN2 : HasDerivAt (fun s => Real.exp (-m * s) * (-m * 1) * (4 * Real.pi * s)) _ r := hN1.mul hlin
  have hN3 : HasDerivAt (fun s => Real.exp (-m * s) * (4 * Real.pi * 1)) _ r := hE.mul_const (4 * Real.pi * 1)
  have hN : HasDerivAt (fun s => Real.exp (-m * s) * (-m * 1) * (4 * Real.pi * s) - Real.exp (-m * s) * (4 * Real.pi * 1)) _ r :=
    hN2.sub hN3
  have h2 : HasDerivAt (fun s => (Real.exp (-m * s) * (-m * 1) * (4 * Real.pi * s) - Real.exp (-m * s) * (4 * Real.pi * 1)) /
      (4 * Real.pi * s) ^ 2) _ r :=
    hN.div hsq (pow_ne_zero 2 (mul_ne_zero (by positivity) hr.ne'))
  rw [h2.deriv]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.FundamentalSolutions.yukawa_solves
-- test: de-yukawa-solves-test.sage

/-- The inward flux of `∇G` through the sphere of radius `R` is `e^{-mR}(1 + mR)`. -/
theorem yukawa_flux (m R : ℝ) (hR : R ≠ 0) :
    -deriv (fun ρ => Real.exp (-m * ρ) / (4 * Real.pi * ρ)) R * (4 * Real.pi * R ^ 2) = Real.exp (-m * R) * (1 + m * R) := by
  veriphy_certified
    ( have h0 :
        HasDerivAt (fun ρ => rexp (-m * ρ) / (4 * π * ρ))
          ((-m * (Real.exp (-R * m)) / (4 * Real.pi * R) - ((Real.exp (-R * m)) / (4 * Real.pi * (R ^ 2))) : ℝ)) (R) :=
        by
        convert
          (((((hasDerivAt_id' (R : ℝ)).const_mul (-m : ℝ)).exp).div (hasDerivAt_id' (R : ℝ)) hR).const_mul
            (1 / (4 * Real.pi) : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.div_apply]; ring_nf; done)
          | ((field_simp; done))
      rw [h0.deriv]
      (field_simp; ring1))
  -- certificate: de-partials.sage

/-- … which tends to `1` as `R → 0`. -/
theorem yukawa_flux_limit (m : ℝ) : Tendsto (fun R => Real.exp (-m * R) * (1 + m * R)) (𝓝 0) (𝓝 1) := by
  have : Continuous fun R => Real.exp (-m * R) * (1 + m * R) := by fun_prop
  simpa using this.tendsto 0
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.FundamentalSolutions.yukawa_flux_limit
-- test: de-yukawa-flux-limit-test.sage

/-- For `m → 0` it becomes `1/(4πr)`. -/
theorem yukawa_massless (r : ℝ) :
    Tendsto (fun m => Real.exp (-m * r) / (4 * Real.pi * r)) (𝓝 0) (𝓝 (1 / (4 * Real.pi * r))) := by
  have : Continuous fun m => Real.exp (-m * r) / (4 * Real.pi * r) := by fun_prop
  simpa using this.tendsto 0
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.FundamentalSolutions.yukawa_massless
-- test: de-yukawa-massless-test.sage

/-! ## exr-fund-n-dim -/

/-- In `n` dimensions `G' = A r^{1-n}` makes the radial Laplacian `r^{1-n}(r^{n-1}G')'` vanish (`r > 0`). -/
theorem n_dim_harmonic (n A r : ℝ) (hr : 0 < r) :
    deriv (fun s => s ^ (n - 1) * (A * s ^ (1 - n))) r = 0 := by
  have h1 : HasDerivAt (fun s : ℝ => s ^ (n - 1)) ((n - 1) * r ^ (n - 1 - 1)) r :=
    Real.hasDerivAt_rpow_const (Or.inl hr.ne')
  have h2 : HasDerivAt (fun s : ℝ => A * s ^ (1 - n)) (A * ((1 - n) * r ^ (1 - n - 1))) r :=
    (Real.hasDerivAt_rpow_const (Or.inl hr.ne')).const_mul A
  have h : HasDerivAt (fun s => s ^ (n - 1) * (A * s ^ (1 - n))) _ r := h1.mul h2
  rw [h.deriv]
  have e1 : r ^ (n - 1 - 1) = r ^ n / r / r := by rw [Real.rpow_sub hr, Real.rpow_sub hr, Real.rpow_one]
  have e2 : r ^ (1 - n - 1) = r / r ^ n / r := by rw [Real.rpow_sub hr, Real.rpow_sub hr, Real.rpow_one]
  have e3 : r ^ (n - 1) = r ^ n / r := by rw [Real.rpow_sub hr, Real.rpow_one]
  have e4 : r ^ (1 - n) = r / r ^ n := by rw [Real.rpow_sub hr, Real.rpow_one]
  rw [e1, e2, e3, e4]
  have hn : r ^ n ≠ 0 := (Real.rpow_pos_of_pos hr n).ne'
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.FundamentalSolutions.n_dim_harmonic
-- test: de-n-dim-harmonic-test.sage

/-- `G = -r^{2-n}/((n - 2)ω)` has `G' = r^{1-n}/ω` (`n ≠ 2`, `r > 0`): unit flux `ω r^{n-1} G' = 1`. -/
theorem n_dim_gradient (n ω r : ℝ) (hn : n - 2 ≠ 0) (hω : ω ≠ 0) (hr : 0 < r) :
    deriv (fun s => -s ^ (2 - n) / ((n - 2) * ω)) r = r ^ (1 - n) / ω := by
  veriphy_certified
    ( have h0 : HasDerivAt (fun s => -s ^ (2 - n) / ((n - 2) * ω)) (((r ^ (-n + 1)) / ω : ℝ)) (r) :=
        by
        refine
          (((((hasDerivAt_id' (r : ℝ)).rpow_const (p := ((-n + 2 : ℝ) : ℝ)) (Or.inl (by exact hr.ne'))).const_mul
                    (-1 / ((n - 2) * ω) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (ring_nf; done)
        · (field_simp; ring_nf; done)
      try rw [h0.deriv]
      try beta_reduce
      all_goals ((field_simp; ring_nf; done)))
  -- certificate: de-partials-pos.sage

/-- For `n = 3`, `ω₂ = 4π`: `-r^{-1}/((3 - 2)·4π) = -1/(4πr)`. -/
theorem n_dim_three (r : ℝ) (hr : 0 < r) : -r ^ ((2 : ℝ) - 3) / ((3 - 2) * (4 * Real.pi)) = -1 / (4 * Real.pi * r) := by
  veriphy_certified
    ( rw [show (2 : ℝ) - 3 = -1 by norm_num, Real.rpow_neg_one]
      field_simp
      ring)
  -- certificate: de-rewrite.sage

end KaluzaKlein.DifferentialEquations.FundamentalSolutions

namespace KaluzaKlein.DifferentialEquations.GreensBoundary

/-- The string's Green's function, the tent `G(x, ξ) = x_<(1 - x_>)` on `[0, 1]`. -/
noncomputable def tent (x ξ : ℝ) : ℝ := min x ξ * (1 - max x ξ)

/-! ## def-greens-function and exm-green-string (with prp-greens-laplacian, exm-pgl-tent) -/

/-- The tent is symmetric … -/
theorem tent_symm (x ξ : ℝ) : tent x ξ = tent ξ x := by
  simp [tent, min_comm, max_comm]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.GreensBoundary.tent_symm
-- test: de-tent-symm-test.sage

/-- … vanishes at the ends `x = 0`, `x = 1` (for `ξ ∈ [0, 1]`) … -/
theorem tent_boundary (ξ : ℝ) (h0 : 0 ≤ ξ) (h1 : ξ ≤ 1) : tent 0 ξ = 0 ∧ tent 1 ξ = 0 := by
  simp [tent, min_eq_left h0, max_eq_right h0, min_eq_right h1, max_eq_left h1]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.GreensBoundary.tent_boundary
-- test: de-tent-boundary-test.sage

/-- … is continuous at `x = ξ`: both pieces take the value `ξ(1 - ξ)` there … -/
theorem tent_diagonal (ξ : ℝ) : tent ξ ξ = ξ * (1 - ξ) := by
  simp [tent]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.GreensBoundary.tent_diagonal
-- test: de-tent-diagonal-test.sage

/-- … and its slope jumps by `-1`: `∂ₓ[ξ(1 - x)] - ∂ₓ[x(1 - ξ)] = -ξ - (1 - ξ) = -1`. -/
theorem tent_slope_jump (ξ x : ℝ) : deriv (fun s => ξ * (1 - s)) x - deriv (fun s => s * (1 - ξ)) x = -1 := by
  veriphy_certified
    ( have h0 : HasDerivAt (fun s => ξ * (1 - s)) ((-ξ : ℝ)) (x) :=
        by
        refine
          (((((hasDerivAt_id' (x : ℝ)).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).const_mul
                    (-ξ : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h0.deriv]
      have h1 : HasDerivAt (fun s => s * (1 - ξ)) ((-ξ + 1 : ℝ)) (x) :=
        by
        refine
          ((((hasDerivAt_id' (x : ℝ)).const_mul (-ξ + 1 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (ring_nf; done)
        · ring1
      try rw [h1.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

/-- For `f = 1`: `u(x) = ∫₀ˣ ξ(1 - x) dξ + ∫ₓ¹ x(1 - ξ) dξ`, with `∫₀ˣ ξ(1 - x) dξ = x²(1 - x)/2` … -/
theorem tent_load_left (x : ℝ) : ∫ ξ in (0 : ℝ)..x, ξ * (1 - x) = x ^ 2 * (1 - x) / 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (x : ℝ),
          HasDerivAt (fun x_veriphy => (-(x - 1) * (x_veriphy ^ 2) / 2 : ℝ)) ((fun ξ => ξ * (1 - x)) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (-x / 2 + (1 / 2) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-(x - 1) * (x_veriphy ^ 2) / 2 : ℝ))
          (f' := fun ξ => ξ * (1 - x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: de-integral.sage

/-- … and `∫ₓ¹ x(1 - ξ) dξ = x(1 - x)²/2` … -/
theorem tent_load_right (x : ℝ) : ∫ ξ in x..1, x * (1 - ξ) = x * (1 - x) ^ 2 / 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (x : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (-(x_veriphy ^ 2 - (2 * x_veriphy)) * x / 2 : ℝ))
            ((fun ξ => x * (1 - ξ)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).add
                        ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (-2 : ℝ))).const_mul
                    (-x / 2 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-(x_veriphy ^ 2 - (2 * x_veriphy)) * x / 2 : ℝ)) (f' := fun ξ => x * (1 - ξ)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: de-integral.sage

/-- … adding to the parabola `x(1 - x)/2` of the loaded string. -/
theorem tent_load_sum (x : ℝ) : x ^ 2 * (1 - x) / 2 + x * (1 - x) ^ 2 / 2 = x * (1 - x) / 2 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-! ## exm-green-images -/

/-- On the plane `z = 0` the distances to `y` and to its mirror image `ȳ = (y₁, y₂, -y₃)` agree, so `G = 0` there. -/
theorem images_vanish (x₁ x₂ y₁ y₂ y₃ : ℝ) :
    (x₁ - y₁) ^ 2 + (x₂ - y₂) ^ 2 + (0 - y₃) ^ 2 = (x₁ - y₁) ^ 2 + (x₂ - y₂) ^ 2 + (0 - -y₃) ^ 2 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-! ## exm-green-eigenfunction-series (with exr-de-string-greens) -/

/-- Applying `L` to `Σ φₙ(x)φₙ(ξ)/λₙ` term by term: for the string `φₙ = √(2/π) sin nx`, `λₙ = n²`,
`-∂ₓ²[sin nx sin nξ/n²] = sin nx sin nξ`. -/
theorem eigen_series_term (n ξ x : ℝ) (hn : n ≠ 0) :
    -deriv (fun s => deriv (fun r => Real.sin (n * r) * Real.sin (n * ξ) / n ^ 2) s) x = Real.sin (n * x) * Real.sin (n * ξ) := by
  veriphy_certified
    ( have i0 :
        deriv (fun (r : ℝ) => (sin (n * r) * sin (n * ξ) / n ^ 2 : ℝ)) = fun (d1 : ℝ) =>
          ((Real.cos (d1 * n)) * (Real.sin (n * ξ)) / n : ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun (r : ℝ) => (sin (n * r) * sin (n * ξ) / n ^ 2 : ℝ))
            (((Real.cos (d1 * n)) * (Real.sin (n * ξ)) / n : ℝ)) (d1) :=
          by
          refine
            ((((((hasDerivAt_id' (d1 : ℝ)).const_mul (n : ℝ)).sin).const_mul
                      ((Real.sin (n * ξ)) / (n ^ 2) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (field_simp; done)
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt (fun s => (fun (d1 : ℝ) => ((Real.cos (d1 * n)) * (Real.sin (n * ξ)) / n : ℝ)) s)
          ((-(Real.sin (n * ξ)) * (Real.sin (n * x)) : ℝ)) (x) :=
        by
        refine
          ((((((hasDerivAt_id' (x : ℝ)).const_mul (n : ℝ)).cos).const_mul
                    ((Real.sin (n * ξ)) / n : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (ring_nf; done)
        · (field_simp; done)
      try rw [h1.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

/-! ## exr-green-1d-sine -/

/-- `u(x) = ∫₀¹ G(x, ξ) sin πξ dξ`: the piece `∫₀ˣ ξ(1 - x) sin πξ dξ` … -/
theorem green_sine_left (x : ℝ) :
    ∫ ξ in (0 : ℝ)..x, ξ * (1 - x) * Real.sin (Real.pi * ξ) =
      (1 - x) * (Real.sin (Real.pi * x) / Real.pi ^ 2 - x * Real.cos (Real.pi * x) / Real.pi) := by
  rw [show (fun ξ => ξ * (1 - x) * Real.sin (Real.pi * ξ)) = fun ξ => (1 - x) * (ξ * Real.sin (Real.pi * ξ)) from
    funext fun ξ => by ring, intervalIntegral.integral_const_mul, int_x_sin Real.pi x Real.pi_ne_zero]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.GreensBoundary.green_sine_left
-- test: de-green-sine-left-test.sage

/-- … and `∫ₓ¹ x(1 - ξ) sin πξ dξ` … -/
theorem green_sine_right (x : ℝ) :
    ∫ ξ in x..1, x * (1 - ξ) * Real.sin (Real.pi * ξ) =
      x * ((1 - x) * Real.cos (Real.pi * x) / Real.pi + Real.sin (Real.pi * x) / Real.pi ^ 2) := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (x : ℝ) (1 : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (x *
                    ((Real.pi * x_veriphy * (Real.cos (Real.pi * x_veriphy)) - (Real.sin (Real.pi * x_veriphy))) /
                        Real.pi -
                      (Real.cos (Real.pi * x_veriphy))) /
                  Real.pi :
                ℝ))
            ((fun ξ => x * (1 - ξ) * sin (π * ξ)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                        (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (Real.pi : ℝ)).cos)).const_mul
                                    (Real.pi : ℝ)).add
                                ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (Real.pi : ℝ)).sin).const_mul
                                  (-1 : ℝ))).const_mul
                            (1 / Real.pi ^ 1 : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (Real.pi : ℝ)).cos).const_mul (-1 : ℝ))).const_mul
                    (x / Real.pi : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (field_simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (x *
                ((Real.pi * x_veriphy * (Real.cos (Real.pi * x_veriphy)) - (Real.sin (Real.pi * x_veriphy))) / Real.pi -
                  (Real.cos (Real.pi * x_veriphy))) /
              Real.pi :
            ℝ))
          (f' := fun ξ => x * (1 - ξ) * sin (π * ξ)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos ((Real.pi * (1) : ℝ)) = -1 := by
        rw [show ((Real.pi * (1) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he1 : Real.sin ((Real.pi * (1) : ℝ)) = 0 := by
        rw [show ((Real.pi * (1) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1]
      all_goals (field_simp; ring1))
  -- certificate: de-integral.sage

/-- … add to `sin(πx)/π²`, the solution of `exr-bvp-solve`. -/
theorem green_sine_sum (x : ℝ) :
    (1 - x) * (Real.sin (Real.pi * x) / Real.pi ^ 2 - x * Real.cos (Real.pi * x) / Real.pi) +
      x * ((1 - x) * Real.cos (Real.pi * x) / Real.pi + Real.sin (Real.pi * x) / Real.pi ^ 2) =
      Real.sin (Real.pi * x) / Real.pi ^ 2 := by
  veriphy_certified
    ( field_simp
      ring)
  -- certificate: de-rewrite.sage

/-- For `L = -d²/dx² - 1`: the one-sided solution `sin x` (vanishing at `0`) solves `u'' + u = 0` … -/
theorem helmholtz_left (u : ℝ → ℝ) (hu : u = fun s => Real.sin s) (x : ℝ) : deriv (deriv u) x + u x = 0 := by
  veriphy_certified
    ( subst hu
      have d1 : deriv (fun s => sin s) = fun s => Real.cos s :=
        by
        funext s
        have hd : HasDerivAt (fun s => sin s) ((Real.cos s : ℝ)) (s) :=
          by
          convert ((hasDerivAt_id' (s : ℝ)).sin) using 1
          all_goals (ring1)
        exact hd.deriv
      have d2 : deriv (fun s => Real.cos s) = fun s => -(Real.sin s) :=
        by
        funext s
        have hd : HasDerivAt (fun s => Real.cos s) ((-(Real.sin s) : ℝ)) (s) :=
          by
          convert ((hasDerivAt_id' (s : ℝ)).cos) using 1
          all_goals (ring1)
        exact hd.deriv
      simp only [d1, d2]
      ring1)
  -- certificate: de-ode.sage

/-- … and so does `sin(1 - x)` (vanishing at `1`) … -/
theorem helmholtz_right (u : ℝ → ℝ) (hu : u = fun s => Real.sin (1 - s)) (x : ℝ) : deriv (deriv u) x + u x = 0 := by
  veriphy_certified
    ( subst hu
      have d1 : deriv (fun s => sin (1 - s)) = fun s => -(Real.cos (-s + 1)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => sin (1 - s)) ((-(Real.cos (-s + 1)) : ℝ)) (s) :=
          by
          convert ((((hasDerivAt_id' (s : ℝ)).const_mul (-1 : ℝ)).add (hasDerivAt_const (s : ℝ) (1 : ℝ))).sin) using 1
          all_goals (simp only [Pi.add_apply]; (ring_nf; done))
        exact hd.deriv
      have d2 : deriv (fun s => -(Real.cos (-s + 1))) = fun s => -(Real.sin (-s + 1)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => -(Real.cos (-s + 1))) ((-(Real.sin (-s + 1)) : ℝ)) (s) :=
          by
          convert
            (((((hasDerivAt_id' (s : ℝ)).const_mul (-1 : ℝ)).add (hasDerivAt_const (s : ℝ) (1 : ℝ))).cos).const_mul
              (-1 : ℝ)) using
            1
          all_goals (simp only [Pi.add_apply]; (ring_nf; done))
        exact hd.deriv
      simp only [d1, d2]
      (ring_nf; done))
  -- certificate: de-ode.sage

/-- … their Wronskian is `sin x · (-cos(1 - x)) - cos x · sin(1 - x) = -sin 1` … -/
theorem helmholtz_wronskian (x : ℝ) :
    Real.sin x * (-Real.cos (1 - x)) - Real.cos x * Real.sin (1 - x) = -Real.sin 1 := by
  veriphy_certified
    ( have h := Real.sin_add x (1 - x)
      rw [show x + (1 - x) = 1 by ring] at h
      linarith)
  -- certificate: de-rewrite.sage

/-- … and `G = sin x sin(1 - ξ)/sin 1` (for `x ≤ ξ`) has slope jump `-1` at `x = ξ`:
`[sin ξ·(-cos(1 - ξ)) - cos ξ·sin(1 - ξ)]/sin 1 = -1`. -/
theorem helmholtz_slope_jump (ξ : ℝ) (h : Real.sin 1 ≠ 0) :
    (Real.sin ξ * (-Real.cos (1 - ξ)) - Real.cos ξ * Real.sin (1 - ξ)) / Real.sin 1 = -1 := by
  veriphy_certified
    ( rw [show Real.sin ξ * -Real.cos (1 - ξ) - Real.cos ξ * Real.sin (1 - ξ) = -Real.sin 1 by
          rw [Real.cos_sub, Real.sin_sub]; linear_combination (-Real.sin 1) * Real.sin_sq_add_cos_sq ξ]
      field_simp)
  -- certificate: de-rewrite.sage

/-! ## exr-green-symmetry -/

/-- Reciprocity from Lagrange's identity: `∫(vLu - uLv) = [p(uv' - u'v)]ₐᵇ = 0`, and with `Lu = δ_ξ`, `Lv = δ_η` the
left side is `G(ξ, η) - G(η, ξ)`: in weak form, `G(ξ, η) - G(η, ξ) = 0` follows from the vanishing boundary term. -/
theorem reciprocity (Gξη Gηξ boundary : ℝ) (h : Gξη - Gηξ = boundary) (hb : boundary = 0) : Gξη = Gηξ := by
  veriphy_certified (linear_combination h + hb)
  -- certificate: de-poly.sage

/-! ## exr-green-construction -/

/-- The Wronskian `W = p(u_a u_b' - u_a' u_b)` is constant when `(pu_a')' = qu_a`, `(pu_b')' = qu_b`:
`W' = u_a(pu_b')' - u_b(pu_a')' = 0`. -/
theorem wronskian_constant (q ua ub Pa' Pb' : ℝ) (ha : Pa' = q * ua) (hb : Pb' = q * ub) : ua * Pb' - ub * Pa' = 0 := by
  veriphy_certified (linear_combination (-ub) * ha + (ua) * hb)
  -- certificate: de-poly.sage

/-- The slope jump of `G = u_a(x_<)u_b(x_>)/(-W)` at `x = ξ` is `[u_a u_b' - u_a' u_b]/(-W) = -1/p`. -/
theorem construction_slope_jump (p ua ub ua' ub' : ℝ) (hp : p ≠ 0) (hW : p * (ua * ub' - ua' * ub) ≠ 0) :
    (ua * ub' - ua' * ub) / (-(p * (ua * ub' - ua' * ub))) = -1 / p := by
  veriphy_certified
    ( rw [div_eq_div_iff (neg_ne_zero.mpr hW) hp]
      ring)
  -- certificate: de-rewrite.sage

/-- For `-d²/dx²` on `[0, 1]`: `u_a = x`, `u_b = 1 - x`, `W = x·(-1) - 1·(1 - x) = -1`. -/
theorem construction_string (x : ℝ) : x * (-1) - 1 * (1 - x) = -1 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

end KaluzaKlein.DifferentialEquations.GreensBoundary

namespace KaluzaKlein.DifferentialEquations.RetardedGreens

/-! ## def-retarded-greens-function, exm-ret-1d, exr-ret-1d -/

/-- The one-dimensional retarded Green's function `½θ(t - |x|)` from d'Alembert's formula: for an initial velocity
`ψ` concentrated in `(-ε, ε)` with unit integral (a delta of width `ε`), `u = ½∫_{x-t}^{x+t} ψ` is `½` inside the
light cone (`|x| + ε < t`) … -/
theorem retarded_1d_inside (ψ : ℝ → ℝ) (ε t x : ℝ) (hψ : Continuous ψ) (hsupp : ∀ y, ε ≤ |y| → ψ y = 0)
    (hunit : ∫ y in (-ε)..ε, ψ y = 1) (h : |x| + ε < t) : 1 / 2 * ∫ y in (x - t)..(x + t), ψ y = 1 / 2 := by
  have hx1 : x - t ≤ -ε := by linarith [le_abs_self x]
  have hx2 : ε ≤ x + t := by linarith [neg_abs_le x]
  have hi : ∀ a b, IntervalIntegrable ψ MeasureTheory.volume a b := fun a b => hψ.intervalIntegrable a b
  rw [← intervalIntegral.integral_add_adjacent_intervals (hi _ (-ε)) (hi (-ε) _),
    ← intervalIntegral.integral_add_adjacent_intervals (hi (-ε) ε) (hi ε _), hunit]
  have z1 : ∫ y in (x - t)..(-ε), ψ y = 0 := by
    rw [intervalIntegral.integral_congr (g := fun _ => 0) fun y hy => ?_]
    · simp
    · rw [Set.uIcc_of_le hx1] at hy
      exact hsupp y (le_abs.mpr (Or.inr (by linarith [hy.2])))
  have z2 : ∫ y in ε..(x + t), ψ y = 0 := by
    rw [intervalIntegral.integral_congr (g := fun _ => 0) fun y hy => ?_]
    · simp
    · rw [Set.uIcc_of_le hx2] at hy
      exact hsupp y (le_abs.mpr (Or.inl hy.1))
  rw [z1, z2]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.RetardedGreens.retarded_1d_inside
-- test: de-retarded-inside-test.sage

/-- … and `0` outside it (`t + ε < |x|`): the signal fills the forward cone. -/
theorem retarded_1d_outside (ψ : ℝ → ℝ) (ε t x : ℝ) (hε : 0 ≤ ε) (ht : 0 ≤ t) (hsupp : ∀ y, ε ≤ |y| → ψ y = 0)
    (h : t + ε < |x|) : 1 / 2 * ∫ y in (x - t)..(x + t), ψ y = 0 := by
  rw [intervalIntegral.integral_congr (g := fun _ => 0) fun y hy => ?_]
  · simp
  · rw [Set.uIcc_of_le (by linarith)] at hy
    apply hsupp
    rcases le_or_gt 0 x with hx | hx
    · rw [abs_of_nonneg hx] at h
      rw [abs_of_nonneg (by linarith [hy.1])]; linarith [hy.1]
    · rw [abs_of_neg hx] at h
      rw [abs_of_neg (by linarith [hy.2])]; linarith [hy.2]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.RetardedGreens.retarded_1d_outside
-- test: de-retarded-outside-test.sage

/-! ## exm-ret-moving-source -/

/-- The retarded-time rule: with `R(τ) = |x - y₀(τ)|`, `d/dτ (τ + R(τ)) = 1 - n·v` (here along one axis, with
`R = |x - y₀(τ)|`, `x > y₀(τ)`): the factor `1/(1 - n·v)` of the Liénard-Wiechert potential. -/
theorem retarded_time_derivative (x : ℝ) (y₀ : ℝ → ℝ) (v τ : ℝ) (hy : HasDerivAt y₀ v τ) (hx : y₀ τ < x) :
    HasDerivAt (fun s => s + |x - y₀ s|) (1 - v) τ := by
  have hev : (fun s => s + |x - y₀ s|) =ᶠ[𝓝 τ] fun s => s + (x - y₀ s) := by
    have hc : ContinuousAt y₀ τ := hy.continuousAt
    filter_upwards [hc.eventually (Iio_mem_nhds hx)] with s hs
    have hs' : y₀ s < x := hs
    rw [abs_of_pos (by linarith)]
  refine HasDerivAt.congr_of_eventuallyEq ?_ hev
  convert (hasDerivAt_id' τ).add (hy.const_sub x) using 1 <;> ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.RetardedGreens.retarded_time_derivative
-- test: de-retarded-time-test.sage

end KaluzaKlein.DifferentialEquations.RetardedGreens

namespace KaluzaKlein.DifferentialEquations.FeynmanPrescription

/-! ## def-feynman-propagator and exm-feyn-momentum-space -/

/-- `(□ + m²)e^{-ik·x} = -(k² - m²)e^{-ik·x}` with `k² = k₀² - |k|²`: the factor is `-k₀² + |k|² + m²`. -/
theorem momentum_factor (k₀ K m : ℝ) : -k₀ ^ 2 + K + m ^ 2 = -((k₀ ^ 2 - K) - m ^ 2) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- The poles: `k₀² - ω² = (k₀ - ω)(k₀ + ω)` with `ω² = |k|² + m²`. -/
theorem momentum_poles (k₀ K m ω : ℝ) (hω : ω ^ 2 = K + m ^ 2) : (k₀ ^ 2 - K) - m ^ 2 = (k₀ - ω) * (k₀ + ω) := by
  veriphy_certified (linear_combination hω)
  -- certificate: de-poly.sage

/-! ## exm-feyn-euclidean -/

/-- Wick rotation `k₀ = ik₄`: `k² - m² = -(k₄² + |k|² + m²)`. -/
theorem wick_rotation (i k₄ K m : ℂ) (hi : i ^ 2 = -1) : ((i * k₄) ^ 2 - K) - m ^ 2 = -(k₄ ^ 2 + K + m ^ 2) := by
  veriphy_certified (linear_combination (k₄ ^ 2) * hi)
  -- certificate: de-poly.sage

/-! ## exr-feyn-poles -/

/-- The displaced poles: `(ω - iε/(2ω))² = ω² - iε - ε²/(4ω²)`, so `±(ω - iε/(2ω))` solve `k₀² = ω² - iε` to first
order in `ε`. -/
theorem displaced_pole (i ω ε : ℂ) (hi : i ^ 2 = -1) (hω : ω ≠ 0) :
    (ω - i * ε / (2 * ω)) ^ 2 = ω ^ 2 - i * ε - ε ^ 2 / (4 * ω ^ 2) := by
  field_simp
  veriphy_certified (linear_combination (4 * ε ^ 2) * hi)
  -- certificate: de-poly.sage

/-- The residue at `k₀ = ω`: `-e^{-ik₀t}/(k₀ + ω)` there is `-e^{-iωt}/(2ω)`; `-2πi/(2π)` times it is
`(i/2ω)e^{-iωt}`. -/
theorem feynman_residue (E i ω : ℂ) (hω : ω ≠ 0) : -i * (-E / (ω + ω)) = i / (2 * ω) * E := by
  veriphy_certified ((field_simp; ring))
  -- certificate: de-rewrite.sage

/-- In real form `(i/2ω)e^{-iω|t|} = sin(ω|t|)/(2ω) + i cos(ωt)/(2ω)`: the real part solves `g̈ + ω²g = 0` for `t > 0` … -/
theorem feynman_real_part_right (ω : ℝ) (hω : ω ≠ 0) (g : ℝ → ℝ) (hg : g = fun s => Real.sin (ω * s) / (2 * ω)) (t : ℝ) :
    deriv (deriv g) t + ω ^ 2 * g t = 0 := by
  veriphy_certified
    ( subst hg
      have d1 : deriv (fun (s : ℝ) => (sin (ω * s) / (2 * ω) : ℝ)) = fun (s : ℝ) => ((Real.cos (s * ω)) / 2 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (sin (ω * s) / (2 * ω) : ℝ)) (((Real.cos (s * ω)) / 2 : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (ω : ℝ)).sin).const_mul (1 / (2 * ω) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (field_simp; done)
        exact hd.deriv
      have d2 :
        deriv (fun (s : ℝ) => ((Real.cos (s * ω)) / 2 : ℝ)) = fun (s : ℝ) => (-ω * (Real.sin (s * ω)) / 2 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => ((Real.cos (s * ω)) / 2 : ℝ)) ((-ω * (Real.sin (s * ω)) / 2 : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (ω : ℝ)).cos).const_mul (1 / 2 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (ring_nf; done)
        exact hd.deriv
      simp only [d1, d2]
      all_goals ((field_simp; ring1)))
  -- certificate: de-ode.sage

/-- … and for `t < 0` (where it is `-sin ωt/(2ω)`) … -/
theorem feynman_real_part_left (ω : ℝ) (hω : ω ≠ 0) (g : ℝ → ℝ) (hg : g = fun s => -Real.sin (ω * s) / (2 * ω)) (t : ℝ) :
    deriv (deriv g) t + ω ^ 2 * g t = 0 := by
  veriphy_certified
    ( subst hg
      have d1 : deriv (fun (s : ℝ) => (-sin (ω * s) / (2 * ω) : ℝ)) = fun (s : ℝ) => (-(Real.cos (s * ω)) / 2 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (-sin (ω * s) / (2 * ω) : ℝ)) ((-(Real.cos (s * ω)) / 2 : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (ω : ℝ)).sin).const_mul (-1 / (2 * ω) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (field_simp; done)
        exact hd.deriv
      have d2 :
        deriv (fun (s : ℝ) => (-(Real.cos (s * ω)) / 2 : ℝ)) = fun (s : ℝ) => (ω * (Real.sin (s * ω)) / 2 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (-(Real.cos (s * ω)) / 2 : ℝ)) ((ω * (Real.sin (s * ω)) / 2 : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (ω : ℝ)).cos).const_mul (-(1 / 2) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (ring_nf; done)
        exact hd.deriv
      simp only [d1, d2]
      all_goals ((field_simp; ring1)))
  -- certificate: de-ode.sage

/-- … with slopes `½` and `-½` at `0`: a unit jump in `ġ`, which is `δ(t)` (`ω ≠ 0`). -/
theorem feynman_jump (ω : ℝ) (hω : ω ≠ 0) :
    deriv (fun s => Real.sin (ω * s) / (2 * ω)) 0 - deriv (fun s => -Real.sin (ω * s) / (2 * ω)) 0 = 1 := by
  have h1 : HasDerivAt (fun s => Real.sin (ω * s) / (2 * ω)) _ 0 :=
    ((hasDerivAt_id' 0).const_mul ω).sin.div_const (2 * ω)
  have h2 : HasDerivAt (fun s => -Real.sin (ω * s) / (2 * ω)) _ 0 :=
    ((hasDerivAt_id' 0).const_mul ω).sin.neg.div_const (2 * ω)
  rw [h1.deriv, h2.deriv, mul_zero, Real.cos_zero]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.FeynmanPrescription.feynman_jump
-- test: de-feynman-jump-test.sage

/-! ## exr-feyn-retarded-compare -/

/-- The retarded residues sum: `-i(-e^{-iωt}/(2ω) + e^{iωt}/(2ω)) = sin ωt/ω` (with `e^{±iωt} = c ± is`). -/
theorem retarded_residues (i c s ω : ℂ) (hi : i ^ 2 = -1) (hω : ω ≠ 0) :
    -i * (-(c - i * s) / (2 * ω) + (c + i * s) / (2 * ω)) = s / ω := by
  field_simp
  veriphy_certified (linear_combination (-2 * s) * hi)
  -- certificate: de-poly.sage

/-- `g = θ(t) sin ωt/ω`: for `t > 0` it solves `g̈ + ω²g = 0` … -/
theorem oscillator_green_solves (ω : ℝ) (hω : ω ≠ 0) (g : ℝ → ℝ) (hg : g = fun s => Real.sin (ω * s) / ω) (t : ℝ) :
    deriv (deriv g) t + ω ^ 2 * g t = 0 := by
  veriphy_certified
    ( subst hg
      have d1 : deriv (fun (s : ℝ) => (sin (ω * s) / ω : ℝ)) = fun (s : ℝ) => (Real.cos (s * ω) : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (sin (ω * s) / ω : ℝ)) ((Real.cos (s * ω) : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (ω : ℝ)).sin).const_mul (1 / ω ^ 1 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (field_simp; done)
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => (Real.cos (s * ω) : ℝ)) = fun (s : ℝ) => (-ω * (Real.sin (s * ω)) : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (Real.cos (s * ω) : ℝ)) ((-ω * (Real.sin (s * ω)) : ℝ)) (s) :=
          by
          refine
            (((((hasDerivAt_id' (s : ℝ)).const_mul (ω : ℝ)).cos)).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (ring_nf; done)
        exact hd.deriv
      simp only [d1, d2]
      all_goals ((field_simp; ring1)))
  -- certificate: de-ode.sage

/-- … with `g(0⁺) = 0` … -/
theorem oscillator_green_zero (ω : ℝ) (g : ℝ → ℝ) (hg : g = fun s => Real.sin (ω * s) / ω) : g 0 = 0 := by
  veriphy_certified
    ( subst hg
      (norm_num; done))
  -- certificate: de-ode.sage

/-- … and `ġ(0⁺) = 1` (`ω ≠ 0`): the unit jump that makes `g̈ + ω²g = δ(t)`. -/
theorem oscillator_green_jump (ω : ℝ) (hω : ω ≠ 0) (g : ℝ → ℝ) (hg : g = fun s => Real.sin (ω * s) / ω) :
    deriv g 0 = 1 := by
  veriphy_certified
    ( subst hg
      have d1 : deriv (fun (s : ℝ) => (sin (ω * s) / ω : ℝ)) = fun (s : ℝ) => (Real.cos (s * ω) : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (sin (ω * s) / ω : ℝ)) ((Real.cos (s * ω) : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (ω : ℝ)).sin).const_mul (1 / ω ^ 1 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (field_simp; done)
        exact hd.deriv
      simp only [d1]
      all_goals ((norm_num; done)))
  -- certificate: de-ode.sage

/-- `G_ret - G_F = -(i/2ω)e^{iωt}` for `t > 0`: `s/ω - (s/(2ω) + ic/(2ω)) = -(i/(2ω))(c + is)` … -/
theorem difference_positive (i c s ω : ℂ) (hi : i ^ 2 = -1) (hω : ω ≠ 0) :
    s / ω - (s / (2 * ω) + i * c / (2 * ω)) = -(i / (2 * ω)) * (c + i * s) := by
  field_simp
  veriphy_certified (linear_combination (s) * hi)
  -- certificate: de-poly.sage

/-- … and for `t < 0` (`G_ret = 0`, `sin ω|t| = -s`): `0 - (-s/(2ω) + ic/(2ω)) = -(i/(2ω))(c + is)`. -/
theorem difference_negative (i c s ω : ℂ) (hi : i ^ 2 = -1) (hω : ω ≠ 0) :
    0 - (-s / (2 * ω) + i * c / (2 * ω)) = -(i / (2 * ω)) * (c + i * s) := by
  field_simp
  veriphy_certified (linear_combination (s) * hi)
  -- certificate: de-poly.sage

/-- The difference `-(i/2ω)e^{iωt}` solves the homogeneous equation: its real and imaginary parts are
`sin ωt/(2ω)` and `-cos ωt/(2ω)`. -/
theorem difference_homogeneous (ω : ℝ) (hω : ω ≠ 0) (g : ℝ → ℝ) (hg : g = fun s => -Real.cos (ω * s) / (2 * ω)) (t : ℝ) :
    deriv (deriv g) t + ω ^ 2 * g t = 0 := by
  veriphy_certified
    ( subst hg
      have d1 : deriv (fun (s : ℝ) => (-cos (ω * s) / (2 * ω) : ℝ)) = fun (s : ℝ) => ((Real.sin (s * ω)) / 2 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (-cos (ω * s) / (2 * ω) : ℝ)) (((Real.sin (s * ω)) / 2 : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (ω : ℝ)).cos).const_mul (-1 / (2 * ω) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (field_simp; done)
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => ((Real.sin (s * ω)) / 2 : ℝ)) = fun (s : ℝ) => (ω * (Real.cos (s * ω)) / 2 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => ((Real.sin (s * ω)) / 2 : ℝ)) ((ω * (Real.cos (s * ω)) / 2 : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (ω : ℝ)).sin).const_mul (1 / 2 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (ring_nf; done)
        exact hd.deriv
      simp only [d1, d2]
      all_goals ((field_simp; ring1)))
  -- certificate: de-ode.sage

/-! ## exr-feyn-wick -/

/-- The residue at `k₄ = iω`: `e^{ik₄τ}/(k₄ + iω)` there is `e^{-ωτ}/(2iω)`; `2πi/(2π)` times it is `e^{-ωτ}/(2ω)`. -/
theorem wick_residue (i E ω : ℂ) (hi : i ^ 2 = -1) (hω : ω ≠ 0) : i * (E / (i * ω + i * ω)) = E / (2 * ω) := by
  veriphy_certified
    ( have hi0 : i ≠ 0 := by rintro rfl; norm_num at hi
      field_simp
      ring)
  -- certificate: de-rewrite.sage

/-- At the pole `e^{i(iω)τ} = e^{-ωτ}`. -/
theorem wick_exponent (ω τ : ℂ) : Complex.exp (Complex.I * (Complex.I * ω) * τ) = Complex.exp (-(ω * τ)) := by
  congr 1
  ring_nf
  rw [Complex.I_sq]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.FeynmanPrescription.wick_exponent
-- test: de-wick-exponent-test.sage

/-- Continuing `τ → it`: `e^{-ω|τ|}` at `|τ| = i|t|` is `e^{-iω|t|}`, and `dk₀ = i dk₄` supplies the factor `i`:
`i · e^{-iω|t|}/(2ω)` is the Feynman result. -/
theorem wick_continuation (ω t : ℂ) (hω : ω ≠ 0) :
    Complex.I * (Complex.exp (-ω * (Complex.I * t)) / (2 * ω)) = Complex.I / (2 * ω) * Complex.exp (-Complex.I * ω * t) := by
  rw [show -ω * (Complex.I * t) = -Complex.I * ω * t by ring]
  field_simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.FeynmanPrescription.wick_continuation
-- test: de-wick-continuation-test.sage

end KaluzaKlein.DifferentialEquations.FeynmanPrescription

namespace KaluzaKlein.DifferentialEquations.GreensComputations

/-! ## exr-pgl-flux-numbers -/

/-- The missing flux is the screening charge: `∫₀ᴿ m²G·4πr² dr = m²∫₀ᴿ re^{-mr} dr = 1 - e^{-mR}(1 + mR)`. -/
theorem screening_charge (m R : ℝ) (hm : m ≠ 0) :
    ∫ r in (0 : ℝ)..R, m ^ 2 * r * Real.exp (-m * r) = 1 - Real.exp (-m * R) * (1 + m * R) := by
  have hF : ∀ r : ℝ, HasDerivAt (fun r => -(1 + m * r) * Real.exp (-m * r)) (m ^ 2 * r * Real.exp (-m * r)) r := by
    intro r
    have hn : HasDerivAt (fun r => -(1 + m * r)) _ r := (((hasDerivAt_id' r).const_mul m).const_add 1).neg
    have h : HasDerivAt (fun r => -(1 + m * r) * Real.exp (-m * r)) _ r := hn.mul (((hasDerivAt_id' r).const_mul (-m)).exp)
    refine h.congr_deriv ?_
    ring
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun r _ => hF r) (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [mul_zero, add_zero, Real.exp_zero]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.GreensComputations.screening_charge
-- test: de-screening-charge-test.sage

/-- For `m = 1`, `R = 1`: the flux `e^{-1}(1 + 1) = 2/e ≈ 0.736`. -/
theorem flux_one : |Real.exp (-1) * (1 + 1) - 92 / 125| < (1 / 2000 : ℝ) := by
  veriphy_certified
    ( have h1 := Real.exp_one_gt_d9
      have h2 := Real.exp_one_lt_d9
      have he := Real.exp_pos 1
      have hi : Real.exp 1 * (Real.exp 1)⁻¹ = 1 := mul_inv_cancel₀ he.ne'
      rw [Real.exp_neg, abs_lt]
      constructor <;> nlinarith [inv_pos.mpr he])
  -- certificate: de-rewrite.sage

/-- For `R = 5`: `6e^{-5} ≈ 0.040`. -/
theorem flux_five : |Real.exp (-5) * (1 + 5) - 1 / 25| < (1 / 2000 : ℝ) := by
  veriphy_certified
    ( have h1 := Real.exp_one_gt_d9
      have h2 := Real.exp_one_lt_d9
      have e5 : Real.exp 5 = Real.exp 1 ^ 5 := by rw [← Real.exp_nat_mul]; norm_num
      have hl : (2.7182818283 : ℝ) ^ 5 < Real.exp 1 ^ 5 := pow_lt_pow_left₀ h1 (by norm_num) (by norm_num)
      have hu : Real.exp 1 ^ 5 < (2.7182818286 : ℝ) ^ 5 := pow_lt_pow_left₀ h2 (by positivity) (by norm_num)
      norm_num at hl hu
      have he := Real.exp_pos 5
      have hi : Real.exp 5 * (Real.exp 5)⁻¹ = 1 := mul_inv_cancel₀ he.ne'
      rw [Real.exp_neg, abs_lt, e5] at *
      constructor <;> nlinarith [inv_pos.mpr he])
  -- certificate: de-rewrite.sage

/-- For `R = 0.1`: `1.1e^{-0.1} ≈ 0.995`. -/
theorem flux_tenth : |Real.exp (-(1 / 10)) * (1 + 1 / 10) - 199 / 200| < (1 / 2000 : ℝ) := by
  veriphy_certified
    ( have hb := Real.exp_bound (x := -(1 / 10)) (by norm_num) (n := 4) (by norm_num)
      norm_num [Finset.sum_range_succ, Nat.factorial] at hb
      rw [abs_le] at hb
      rw [abs_lt]
      constructor <;> nlinarith [hb.1, hb.2])
  -- certificate: de-rewrite.sage

/-! ## exr-pgl-yukawa-radial -/

/-- With `G = v/r`: `(1/r²)(r²G')' = v''/r` (from `r²G' = rv' - v`, whose derivative is `rv''`). -/
theorem yukawa_substitution (r v v' v'' : ℝ) (hr : r ≠ 0) : 1 / r ^ 2 * (r * v'' + v' - v') = v'' / r := by
  veriphy_certified ((field_simp; ring))
  -- certificate: de-rewrite.sage

/-- `r²G' = rv' - v` for `G = v/r`. -/
theorem yukawa_substitution_derivative (v : ℝ → ℝ) (v' r : ℝ) (hr : r ≠ 0) (hv : HasDerivAt v v' r) :
    HasDerivAt (fun s => v s / s) ((r * v' - v r) / r ^ 2) r := by
  convert hv.div (hasDerivAt_id' r) hr using 1
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.GreensComputations.yukawa_substitution_derivative
-- test: de-yukawa-subst-test.sage

/-- `v = Ae^{-mr} + Be^{mr}` solves `v'' = m²v`. -/
theorem yukawa_radial_general (A B m : ℝ) (v : ℝ → ℝ) (hv : v = fun s => A * Real.exp (-m * s) + B * Real.exp (m * s))
    (r : ℝ) : deriv (deriv v) r = m ^ 2 * v r := by
  veriphy_certified
    ( subst hv
      have d1 :
        deriv (fun (s : ℝ) => (A * rexp (-m * s) + B * rexp (m * s) : ℝ)) = fun (s : ℝ) =>
          (m * B * (Real.exp (s * m)) - (m * A * (Real.exp (-s * m))) : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => (A * rexp (-m * s) + B * rexp (m * s) : ℝ))
            ((m * B * (Real.exp (s * m)) - (m * A * (Real.exp (-s * m))) : ℝ)) (s) :=
          by
          refine
            (((((((hasDerivAt_id' (s : ℝ)).const_mul (m : ℝ)).exp).const_mul (B : ℝ)).add
                      ((((hasDerivAt_id' (s : ℝ)).const_mul (-m : ℝ)).exp).const_mul (A : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · (ring_nf; done)
        exact hd.deriv
      have d2 :
        deriv (fun (s : ℝ) => (m * B * (Real.exp (s * m)) - (m * A * (Real.exp (-s * m))) : ℝ)) = fun (s : ℝ) =>
          ((m ^ 2) * B * (Real.exp (s * m)) + ((m ^ 2) * A * (Real.exp (-s * m))) : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => (m * B * (Real.exp (s * m)) - (m * A * (Real.exp (-s * m))) : ℝ))
            (((m ^ 2) * B * (Real.exp (s * m)) + ((m ^ 2) * A * (Real.exp (-s * m))) : ℝ)) (s) :=
          by
          refine
            (((((((hasDerivAt_id' (s : ℝ)).const_mul (m : ℝ)).exp).const_mul (m * B : ℝ)).add
                      ((((hasDerivAt_id' (s : ℝ)).const_mul (-m : ℝ)).exp).const_mul
                        (-m * A : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · (ring_nf; done)
        exact hd.deriv
      simp only [d1, d2]
      all_goals ((ring_nf; done)))
  -- certificate: de-ode.sage

/-- Decay at infinity forces `B = 0`: `e^{-mr} → 0` while `e^{mr} → ∞` (`m > 0`). -/
theorem yukawa_decay (A B m : ℝ) (hm : 0 < m) (hB : B ≠ 0) :
    ¬ Tendsto (fun r => A * Real.exp (-m * r) + B * Real.exp (m * r)) atTop (𝓝 0) := by
  intro h
  have h1 : Tendsto (fun r => A * Real.exp (-m * r)) atTop (𝓝 0) := by
    simpa using (Real.tendsto_exp_atBot.comp (tendsto_id.const_mul_atTop_of_neg (show -m < 0 by linarith))).const_mul A
  have h2 : Tendsto (fun r => B * Real.exp (m * r)) atTop (𝓝 0) := by
    simpa using h.sub h1
  have h3 : Tendsto (fun r => Real.exp (m * r)) atTop (𝓝 0) := by
    simpa [hB] using h2.const_mul B⁻¹
  have h4 : Tendsto (fun r => Real.exp (m * r)) atTop atTop :=
    Real.tendsto_exp_atTop.comp (tendsto_id.const_mul_atTop hm)
  exact not_tendsto_nhds_of_tendsto_atTop h4 0 h3
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.GreensComputations.yukawa_decay
-- test: de-yukawa-decay-test.sage

/-- Normalisation: near `0`, `G ≈ A/r` has inward flux `4πA`, equal to `1` for `A = 1/(4π)`. -/
theorem yukawa_normalisation (r : ℝ) (hr : r ≠ 0) :
    -deriv (fun s => 1 / (4 * Real.pi) / s) r * (4 * Real.pi * r ^ 2) = 1 := by
  veriphy_certified
    ( have h0 : HasDerivAt (fun s => 1 / (4 * π) / s) ((-1 / (4 * Real.pi * (r ^ 2)) : ℝ)) (r) :=
        by
        refine
          (((((hasDerivAt_const (r : ℝ) (1 : ℝ)).div (hasDerivAt_id' (r : ℝ)) hr).const_mul
                    (1 / (4 * Real.pi) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h0.deriv]
      try beta_reduce
      all_goals ((field_simp; done)))
  -- certificate: de-partials.sage

end KaluzaKlein.DifferentialEquations.GreensComputations

namespace KaluzaKlein.DifferentialEquations.GreensComputations

/-! ## exm-pgl-fourier -/

/-- Sage's value `(π/2)e^{-mr}` of `∫₀^∞ k sin kr/(k² + m²) dk`, divided by `2π²r`, is `e^{-mr}/(4πr)`. -/
theorem yukawa_fourier_value (m r : ℝ) (hr : r ≠ 0) :
    1 / (2 * Real.pi ^ 2 * r) * (Real.pi / 2 * Real.exp (-m * r)) = Real.exp (-m * r) / (4 * Real.pi * r) := by
  veriphy_certified
    ( field_simp
      ring)
  -- certificate: de-rewrite.sage

end KaluzaKlein.DifferentialEquations.GreensComputations
