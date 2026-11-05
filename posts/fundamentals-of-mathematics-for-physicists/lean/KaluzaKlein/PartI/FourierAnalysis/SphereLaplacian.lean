import Mathlib
import Veriphy.Kit.Linear
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.DistributionComputations

macro "fa_close4" : tactic => `(tactic| first
  | rfl
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; done)
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; ring1)
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; field_simp; ring1)
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; field_simp; done)
  | (simp only [Function.comp_apply, id_eq, Pi.mul_apply, mul_one, one_mul]; ring1)
  | (simp only [Function.comp_apply, id_eq, Pi.mul_apply, mul_one, one_mul]; field_simp; ring1)
  | (simp only [Function.comp_apply, id_eq, Pi.mul_apply, mul_one, one_mul]; field_simp; done)
  | ring1
  | (field_simp; done)
  | (field_simp; ring1)
  | (norm_num; done)
  | (norm_num; ring1)
  | (norm_num; field_simp; ring1)
  | (simp; done)
  | (simp; ring1)
  | (ring_nf; done)
  | (field_simp; ring_nf; done))

/-! # Fourier analysis: sphere laplacian (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.SphereLaplacian

open Real MeasureTheory Filter Topology

/-! ## def-spherical-laplacian -/

/-- The `φ`-dependence of a mode: `∂²_φ cos(mφ) = -m² cos(mφ)` (and likewise `sin`), so `Δ` acts on
`Θ(θ) cos(mφ)` by `(sin θ)⁻¹ (sin θ Θ')' - m²Θ/sin² θ`. -/
theorem phi_mode (m φ : ℝ) :
    deriv (deriv fun p : ℝ => Real.cos (m * p)) φ = -m ^ 2 * Real.cos (m * φ) := by
  have h1 : deriv (fun p : ℝ => Real.cos (m * p)) = fun p => -(m * Real.sin (m * p)) := by
    funext p
    exact (((hasDerivAt_id' p).const_mul m).cos).deriv.trans (by ring)
  rw [h1]
  exact ((((hasDerivAt_id' φ).const_mul m).sin).const_mul m).neg.deriv.trans (by ring)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SphereLaplacian.phi_mode
-- test: fa-phi-mode-test.sage

/-- The radial part: `(∂_r² + (2/r) ∂_r) r^l = l(l+1) r^{l-2}`, so `Δ Y = -l(l+1) Y` for `P = r^l Y` harmonic. -/
theorem radial_power (l : ℕ) (r : ℝ) (hr : r ≠ 0) (hl : 2 ≤ l) :
    deriv (deriv fun r : ℝ => r ^ l) r + 2 / r * deriv (fun r : ℝ => r ^ l) r = l * (l + 1) * r ^ (l - 2) := by
  have h1 : deriv (fun r : ℝ => r ^ l) = fun r => (l : ℝ) * r ^ (l - 1) := by
    funext r; exact (hasDerivAt_pow l r).deriv
  rw [h1]
  have := ((hasDerivAt_pow (l - 1) r).const_mul (l : ℝ))
  rw [this.deriv]
  obtain ⟨k, rfl⟩ : ∃ k, l = k + 2 := ⟨l - 2, by omega⟩
  simp only [show k + 2 - 1 = k + 1 by omega, show k + 2 - 2 = k by omega, show k + 1 - 1 = k by omega]
  push_cast
  field_simp
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SphereLaplacian.radial_power
-- test: fa-radial-power-test.sage

/-- The symmetry of `Δ` in `θ`: `∫₀^π g (sin θ f')' dθ = -∫₀^π g' sin θ f' dθ`, the boundary terms vanishing since
`sin 0 = sin π = 0`; so the eigenvalues are real and non-positive. -/
theorem theta_symmetric (f' g g' F : ℝ → ℝ) (hg : ∀ x, HasDerivAt g (g' x) x)
    (hF : ∀ x, HasDerivAt (fun t => Real.sin t * f' t) (F x) x) (hg' : Continuous g') (hFc : Continuous F)
    (hf' : Continuous f') :
    ∫ θ in (0 : ℝ)..π, g θ * F θ = -∫ θ in (0 : ℝ)..π, g' θ * (Real.sin θ * f' θ) := by
  have := intervalIntegral.integral_mul_deriv_eq_deriv_mul (a := 0) (b := π) (fun x _ => hg x) (fun x _ => hF x)
    (hg'.intervalIntegrable _ _) (hFc.intervalIntegrable _ _)
  rw [this]; simp [Real.sin_pi]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SphereLaplacian.theta_symmetric
-- test: fa-theta-symmetric-test.sage

/-- The radial operator in `r`: `r⁻² (r² f')' = f'' + (2/r) f'`, the radial part of `∇²` in spherical coordinates. -/
theorem radial_operator (f f' f'' : ℝ → ℝ) (r : ℝ) (hr : r ≠ 0) (hf : HasDerivAt f' (f'' r) r) :
    deriv (fun r => r ^ 2 * f' r) r / r ^ 2 = f'' r + 2 / r * f' r := by
  have h : HasDerivAt (fun r => r ^ 2 * f' r) (2 * r * f' r + r ^ 2 * f'' r) r := by
    have := (hasDerivAt_pow 2 r).mul hf
    convert this using 1
    all_goals first | rfl | (funext y; simp [Pi.mul_apply]) | (norm_num; done) | (simp; ring) | ring
  rw [h.deriv]; field_simp; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SphereLaplacian.radial_operator
-- test: fa-radial-operator-test.sage

/-- The volume factor of the spherical metric `dr² + r²(dθ² + sin²θ dφ²)`: `det g = r⁴ sin² θ`, so `√g = r² sin θ`. -/
theorem spherical_metric_det (r θ : ℝ) :
    Matrix.det !![1, 0, 0; 0, r ^ 2, 0; 0, 0, r ^ 2 * Real.sin θ ^ 2] = (r ^ 2 * Real.sin θ) ^ 2 := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
      all_goals ring1)
  -- certificate: fa-matrix.sage

/-- `Θ_{10}(θ) = Real.cos θ` solves the associated Legendre equation with `l = 1`, `m = 0`:
`sin θ (sin θ Θ')' - m² Θ = -l(l+1) sin² θ Θ`, so `Θ e^{imφ}` has `Δ = -2`. -/
theorem legendre_10 (θ : ℝ) :
    Real.sin θ * deriv (fun t : ℝ => Real.sin t * deriv (fun t : ℝ => Real.cos t) t) θ - 0 * (Real.cos θ) +
      2 * Real.sin θ ^ 2 * (Real.cos θ) = 0 := by
  have h1 : ∀ t : ℝ, deriv (fun t : ℝ => Real.cos t) t = -Real.sin t := by
    intro t
    veriphy_certified
      ( have hd : HasDerivAt (fun t => cos t) ((-(Real.sin t) : ℝ)) (t) :=
          by
          convert ((hasDerivAt_id' (t : ℝ)).cos) using 1
          all_goals ring1
        exact hd.deriv.trans (by rfl))
    -- certificate: fa-deriv2.sage
  have h2 : deriv (fun t : ℝ => Real.sin t * (-Real.sin t)) θ = -2 * Real.sin θ * Real.cos θ := by
    veriphy_certified
      ( have hd : HasDerivAt (fun t => sin t * -sin t) ((-2 * (Real.cos θ) * (Real.sin θ) : ℝ)) (θ) :=
          by
          convert ((((hasDerivAt_id' (θ : ℝ)).sin).pow 2).const_mul (-1 : ℝ)) using 1
          all_goals
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; ring1)
            | (funext y; ring_nf; done)
            | (simp only [Pi.pow_apply]; (ring_nf; done))
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  simp only [show (deriv fun t : ℝ => Real.cos t) = fun t : ℝ => -Real.sin t from funext h1]
  rw [h2]
  linear_combination (0) * Real.sin_sq_add_cos_sq θ

/-- `Θ_{11}(θ) = Real.sin θ` solves the associated Legendre equation with `l = 1`, `m = 1`:
`sin θ (sin θ Θ')' - m² Θ = -l(l+1) sin² θ Θ`, so `Θ e^{imφ}` has `Δ = -2`. -/
theorem legendre_11 (θ : ℝ) :
    Real.sin θ * deriv (fun t : ℝ => Real.sin t * deriv (fun t : ℝ => Real.sin t) t) θ - 1 * (Real.sin θ) +
      2 * Real.sin θ ^ 2 * (Real.sin θ) = 0 := by
  have h1 : ∀ t : ℝ, deriv (fun t : ℝ => Real.sin t) t = Real.cos t := by
    intro t
    veriphy_certified
      ( have hd : HasDerivAt (fun t => sin t) ((Real.cos t : ℝ)) (t) :=
          by
          convert ((hasDerivAt_id' (t : ℝ)).sin) using 1
          all_goals ring1
        exact hd.deriv.trans (by rfl))
    -- certificate: fa-deriv2.sage
  have h2 : deriv (fun t : ℝ => Real.sin t * (Real.cos t)) θ = Real.cos θ ^ 2 - Real.sin θ ^ 2 := by
    veriphy_certified
      ( have hd : HasDerivAt (fun t => sin t * cos t) (((Real.cos θ) ^ 2 - ((Real.sin θ) ^ 2) : ℝ)) (θ) :=
          by
          convert (((hasDerivAt_id' (θ : ℝ)).cos).mul ((hasDerivAt_id' (θ : ℝ)).sin)) using 1
          all_goals
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; ring1)
            | (funext y; simp only [Pi.mul_apply]; ring_nf; done)
            | (ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: fa-deriv2.sage
  simp only [show (deriv fun t : ℝ => Real.sin t) = fun t : ℝ => Real.cos t from funext h1]
  rw [h2]
  linear_combination (Real.sin θ) * Real.sin_sq_add_cos_sq θ

/-- `Θ_{20}(θ) = 3 / 2 * Real.cos θ ^ 2 - 1 / 2` solves the associated Legendre equation with `l = 2`, `m = 0`:
`sin θ (sin θ Θ')' - m² Θ = -l(l+1) sin² θ Θ`, so `Θ e^{imφ}` has `Δ = -6`. -/
theorem legendre_20 (θ : ℝ) :
    Real.sin θ * deriv (fun t : ℝ => Real.sin t * deriv (fun t : ℝ => 3 / 2 * Real.cos t ^ 2 - 1 / 2) t) θ - 0 * (3 / 2 * Real.cos θ ^ 2 - 1 / 2) +
      6 * Real.sin θ ^ 2 * (3 / 2 * Real.cos θ ^ 2 - 1 / 2) = 0 := by
  have h1 : ∀ t : ℝ, deriv (fun t : ℝ => 3 / 2 * Real.cos t ^ 2 - 1 / 2) t = -3 * Real.sin t * Real.cos t := by
    intro t
    veriphy_certified
      ( have hd : HasDerivAt (fun t => 3 / 2 * cos t ^ 2 - 1 / 2) ((-3 * (Real.cos t) * (Real.sin t) : ℝ)) (t) :=
          by
          convert
            (((((hasDerivAt_id' (t : ℝ)).cos).pow 2).const_mul (3 / 2 : ℝ)).add
              (hasDerivAt_const (t : ℝ) (-(1 / 2) : ℝ))) using
            1
          all_goals ring1
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  have h2 : deriv (fun t : ℝ => Real.sin t * (-3 * Real.sin t * Real.cos t)) θ = -6 * Real.sin θ * Real.cos θ ^ 2 + 3 * Real.sin θ ^ 3 := by
    veriphy_certified
      ( have hd :
          HasDerivAt (fun t => sin t * (-3 * sin t * cos t))
            ((-6 * ((Real.cos θ) ^ 2) * (Real.sin θ) + (3 * ((Real.sin θ) ^ 3)) : ℝ)) (θ) :=
          by
          convert ((((hasDerivAt_id' (θ : ℝ)).cos).mul (((hasDerivAt_id' (θ : ℝ)).sin).pow 2)).const_mul (-3 : ℝ)) using 1
          all_goals
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; ring1)
            | (funext y; simp only [Pi.mul_apply, Pi.pow_apply]; ring_nf; done)
            | (simp only [Pi.pow_apply]; ring1)
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  simp only [show (deriv fun t : ℝ => 3 / 2 * Real.cos t ^ 2 - 1 / 2) = fun t : ℝ => -3 * Real.sin t * Real.cos t from funext h1]
  rw [h2]
  linear_combination (3 * Real.sin θ ^ 2) * Real.sin_sq_add_cos_sq θ

/-- `Θ_{21}(θ) = 3 * Real.sin θ * Real.cos θ` solves the associated Legendre equation with `l = 2`, `m = 1`:
`sin θ (sin θ Θ')' - m² Θ = -l(l+1) sin² θ Θ`, so `Θ e^{imφ}` has `Δ = -6`. -/
theorem legendre_21 (θ : ℝ) :
    Real.sin θ * deriv (fun t : ℝ => Real.sin t * deriv (fun t : ℝ => 3 * Real.sin t * Real.cos t) t) θ - 1 * (3 * Real.sin θ * Real.cos θ) +
      6 * Real.sin θ ^ 2 * (3 * Real.sin θ * Real.cos θ) = 0 := by
  have h1 : ∀ t : ℝ, deriv (fun t : ℝ => 3 * Real.sin t * Real.cos t) t = 3 * Real.cos t ^ 2 - 3 * Real.sin t ^ 2 := by
    intro t
    veriphy_certified
      ( have hd : HasDerivAt (fun t => 3 * sin t * cos t) ((3 * ((Real.cos t) ^ 2) - (3 * ((Real.sin t) ^ 2)) : ℝ)) (t) :=
          by
          convert ((((hasDerivAt_id' (t : ℝ)).cos).mul ((hasDerivAt_id' (t : ℝ)).sin)).const_mul (3 : ℝ)) using 1
          all_goals
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; ring1)
            | (funext y; simp only [Pi.mul_apply]; ring_nf; done)
            | (ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: fa-deriv2.sage
  have h2 : deriv (fun t : ℝ => Real.sin t * (3 * Real.cos t ^ 2 - 3 * Real.sin t ^ 2)) θ = 3 * Real.cos θ ^ 3 - 15 * Real.sin θ ^ 2 * Real.cos θ := by
    veriphy_certified
      ( have hd :
          HasDerivAt (fun t => sin t * (3 * cos t ^ 2 - 3 * sin t ^ 2))
            ((-12 * (Real.cos θ) * ((Real.sin θ) ^ 2) + (3 * ((Real.cos θ) ^ 2 - ((Real.sin θ) ^ 2)) * (Real.cos θ)) : ℝ))
            (θ) :=
          by
          convert
            ((((((hasDerivAt_id' (θ : ℝ)).cos).pow 2).add ((((hasDerivAt_id' (θ : ℝ)).sin).pow 2).const_mul (-1 : ℝ))).mul
                  ((hasDerivAt_id' (θ : ℝ)).sin)).const_mul
              (3 : ℝ)) using
            1
          all_goals
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; ring1)
            | (funext y; simp only [Pi.add_apply, Pi.mul_apply, Pi.pow_apply]; ring_nf; done)
            | (simp only [Pi.add_apply, Pi.pow_apply]; ring1)
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  simp only [show (deriv fun t : ℝ => 3 * Real.sin t * Real.cos t) = fun t : ℝ => 3 * Real.cos t ^ 2 - 3 * Real.sin t ^ 2 from funext h1]
  rw [h2]
  linear_combination (3 * Real.sin θ * Real.cos θ) * Real.sin_sq_add_cos_sq θ

/-- `Θ_{22}(θ) = 3 * Real.sin θ ^ 2` solves the associated Legendre equation with `l = 2`, `m = 2`:
`sin θ (sin θ Θ')' - m² Θ = -l(l+1) sin² θ Θ`, so `Θ e^{imφ}` has `Δ = -6`. -/
theorem legendre_22 (θ : ℝ) :
    Real.sin θ * deriv (fun t : ℝ => Real.sin t * deriv (fun t : ℝ => 3 * Real.sin t ^ 2) t) θ - 4 * (3 * Real.sin θ ^ 2) +
      6 * Real.sin θ ^ 2 * (3 * Real.sin θ ^ 2) = 0 := by
  have h1 : ∀ t : ℝ, deriv (fun t : ℝ => 3 * Real.sin t ^ 2) t = 6 * Real.sin t * Real.cos t := by
    intro t
    veriphy_certified
      ( have hd : HasDerivAt (fun t => 3 * sin t ^ 2) ((6 * (Real.cos t) * (Real.sin t) : ℝ)) (t) :=
          by
          convert ((((hasDerivAt_id' (t : ℝ)).sin).pow 2).const_mul (3 : ℝ)) using 1
          all_goals ring1
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  have h2 : deriv (fun t : ℝ => Real.sin t * (6 * Real.sin t * Real.cos t)) θ = 12 * Real.sin θ * Real.cos θ ^ 2 - 6 * Real.sin θ ^ 3 := by
    veriphy_certified
      ( have hd :
          HasDerivAt (fun t => sin t * (6 * sin t * cos t))
            ((12 * ((Real.cos θ) ^ 2) * (Real.sin θ) - (6 * ((Real.sin θ) ^ 3)) : ℝ)) (θ) :=
          by
          convert ((((hasDerivAt_id' (θ : ℝ)).cos).mul (((hasDerivAt_id' (θ : ℝ)).sin).pow 2)).const_mul (6 : ℝ)) using 1
          all_goals
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; ring1)
            | (funext y; simp only [Pi.mul_apply, Pi.pow_apply]; ring_nf; done)
            | (simp only [Pi.pow_apply]; ring1)
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  simp only [show (deriv fun t : ℝ => 3 * Real.sin t ^ 2) = fun t : ℝ => 6 * Real.sin t * Real.cos t from funext h1]
  rw [h2]
  linear_combination (12 * Real.sin θ ^ 2) * Real.sin_sq_add_cos_sq θ

/-- `Θ_{30}(θ) = 5 / 2 * Real.cos θ ^ 3 - 3 / 2 * Real.cos θ` solves the associated Legendre equation with `l = 3`, `m = 0`:
`sin θ (sin θ Θ')' - m² Θ = -l(l+1) sin² θ Θ`, so `Θ e^{imφ}` has `Δ = -12`. -/
theorem legendre_30 (θ : ℝ) :
    Real.sin θ * deriv (fun t : ℝ => Real.sin t * deriv (fun t : ℝ => 5 / 2 * Real.cos t ^ 3 - 3 / 2 * Real.cos t) t) θ - 0 * (5 / 2 * Real.cos θ ^ 3 - 3 / 2 * Real.cos θ) +
      12 * Real.sin θ ^ 2 * (5 / 2 * Real.cos θ ^ 3 - 3 / 2 * Real.cos θ) = 0 := by
  have h1 : ∀ t : ℝ, deriv (fun t : ℝ => 5 / 2 * Real.cos t ^ 3 - 3 / 2 * Real.cos t) t = -15 / 2 * Real.sin t * Real.cos t ^ 2 + 3 / 2 * Real.sin t := by
    intro t
    have := ((((hasDerivAt_id' t).cos).pow 3).const_mul (5 / 2 : ℝ)).sub (((hasDerivAt_id' t).cos).const_mul (3 / 2 : ℝ))
    exact this.deriv.trans (by first | ring | (norm_num; ring) | (simp; ring) | norm_num)
  have h2 : deriv (fun t : ℝ => Real.sin t * (-15 / 2 * Real.sin t * Real.cos t ^ 2 + 3 / 2 * Real.sin t)) θ = -15 * Real.sin θ * Real.cos θ ^ 3 + 15 * Real.sin θ ^ 3 * Real.cos θ + 3 * Real.sin θ * Real.cos θ := by
    veriphy_certified
      ( have hd :
          HasDerivAt (fun t => sin t * (-15 / 2 * sin t * cos t ^ 2 + 3 / 2 * sin t))
            ((-3 * (5 * ((Real.cos θ) ^ 2) * (Real.sin θ) - (Real.sin θ)) * (Real.cos θ) / 2 -
                (3 * (5 * ((Real.cos θ) ^ 3) - (10 * (Real.cos θ) * ((Real.sin θ) ^ 2)) - (Real.cos θ)) * (Real.sin θ) /
                  2) :
              ℝ))
            (θ) :=
          by
          convert
            ((((((((hasDerivAt_id' (θ : ℝ)).cos).pow 2).mul ((hasDerivAt_id' (θ : ℝ)).sin)).const_mul (5 : ℝ)).add
                      (((hasDerivAt_id' (θ : ℝ)).sin).const_mul (-1 : ℝ))).mul
                  ((hasDerivAt_id' (θ : ℝ)).sin)).const_mul
              (-(3 / 2) : ℝ)) using
            1
          all_goals (simp only [Pi.add_apply, Pi.mul_apply, Pi.pow_apply]; (field_simp; ring_nf; done))
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  simp only [show (deriv fun t : ℝ => 5 / 2 * Real.cos t ^ 3 - 3 / 2 * Real.cos t) = fun t : ℝ => -15 / 2 * Real.sin t * Real.cos t ^ 2 + 3 / 2 * Real.sin t from funext h1]
  rw [h2]
  linear_combination (15 * Real.sin θ ^ 2 * Real.cos θ) * Real.sin_sq_add_cos_sq θ

/-- `Θ_{31}(θ) = 15 / 2 * Real.sin θ * Real.cos θ ^ 2 - 3 / 2 * Real.sin θ` solves the associated Legendre equation with `l = 3`, `m = 1`:
`sin θ (sin θ Θ')' - m² Θ = -l(l+1) sin² θ Θ`, so `Θ e^{imφ}` has `Δ = -12`. -/
theorem legendre_31 (θ : ℝ) :
    Real.sin θ * deriv (fun t : ℝ => Real.sin t * deriv (fun t : ℝ => 15 / 2 * Real.sin t * Real.cos t ^ 2 - 3 / 2 * Real.sin t) t) θ - 1 * (15 / 2 * Real.sin θ * Real.cos θ ^ 2 - 3 / 2 * Real.sin θ) +
      12 * Real.sin θ ^ 2 * (15 / 2 * Real.sin θ * Real.cos θ ^ 2 - 3 / 2 * Real.sin θ) = 0 := by
  have h1 : ∀ t : ℝ, deriv (fun t : ℝ => 15 / 2 * Real.sin t * Real.cos t ^ 2 - 3 / 2 * Real.sin t) t = 15 / 2 * Real.cos t ^ 3 - 15 * Real.sin t ^ 2 * Real.cos t - 3 / 2 * Real.cos t := by
    intro t
    veriphy_certified
      ( have hd :
          HasDerivAt (fun t => 15 / 2 * sin t * cos t ^ 2 - 3 / 2 * sin t)
            ((15 * ((Real.cos t) ^ 3) / 2 - (15 * (Real.cos t) * ((Real.sin t) ^ 2)) - (3 * (Real.cos t) / 2) : ℝ)) (t) :=
          by
          convert
            ((((((hasDerivAt_id' (t : ℝ)).cos).pow 2).mul ((hasDerivAt_id' (t : ℝ)).sin)).const_mul (15 / 2 : ℝ)).add
              (((hasDerivAt_id' (t : ℝ)).sin).const_mul (-(3 / 2) : ℝ))) using
            1
          all_goals
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; ring1)
            | (funext y; simp only [Pi.add_apply, Pi.mul_apply, Pi.pow_apply]; ring_nf; done)
            | (simp only [Pi.pow_apply]; ring1)
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  have h2 : deriv (fun t : ℝ => Real.sin t * (15 / 2 * Real.cos t ^ 3 - 15 * Real.sin t ^ 2 * Real.cos t - 3 / 2 * Real.cos t)) θ = 15 / 2 * Real.cos θ ^ 4 - 135 / 2 * Real.sin θ ^ 2 * Real.cos θ ^ 2 + 15 * Real.sin θ ^ 4 - 3 / 2 * Real.cos θ ^ 2 + 3 / 2 * Real.sin θ ^ 2 := by
    veriphy_certified
      ( have hd :
          HasDerivAt (fun t => sin t * (15 / 2 * cos t ^ 3 - 15 * sin t ^ 2 * cos t - 3 / 2 * cos t))
            ((3 * (5 * ((Real.cos θ) ^ 3) - (10 * (Real.cos θ) * ((Real.sin θ) ^ 2)) - (Real.cos θ)) * (Real.cos θ) / 2 -
                (3 * (35 * ((Real.cos θ) ^ 2) * (Real.sin θ) - (10 * ((Real.sin θ) ^ 3)) - (Real.sin θ)) * (Real.sin θ) /
                  2) :
              ℝ))
            (θ) :=
          by
          convert
            ((((((((hasDerivAt_id' (θ : ℝ)).cos).pow 3).const_mul (5 : ℝ)).add
                          ((((hasDerivAt_id' (θ : ℝ)).cos).mul (((hasDerivAt_id' (θ : ℝ)).sin).pow 2)).const_mul
                            (-10 : ℝ))).add
                      (((hasDerivAt_id' (θ : ℝ)).cos).const_mul (-1 : ℝ))).mul
                  ((hasDerivAt_id' (θ : ℝ)).sin)).const_mul
              (3 / 2 : ℝ)) using
            1
          all_goals
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; ring1)
            | (funext y; simp only [Pi.add_apply, Pi.mul_apply, Pi.pow_apply]; ring_nf; done)
            | (simp only [Pi.add_apply, Pi.mul_apply, Pi.pow_apply]; ring1)
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  simp only [show (deriv fun t : ℝ => 15 / 2 * Real.sin t * Real.cos t ^ 2 - 3 / 2 * Real.sin t) = fun t : ℝ => 15 / 2 * Real.cos t ^ 3 - 15 * Real.sin t ^ 2 * Real.cos t - 3 / 2 * Real.cos t from funext h1]
  rw [h2]
  linear_combination (15 / 2 * Real.sin θ * Real.cos θ ^ 2 + 15 * Real.sin θ ^ 3 - 3 / 2 * Real.sin θ) * Real.sin_sq_add_cos_sq θ

/-- `Θ_{32}(θ) = 15 * Real.sin θ ^ 2 * Real.cos θ` solves the associated Legendre equation with `l = 3`, `m = 2`:
`sin θ (sin θ Θ')' - m² Θ = -l(l+1) sin² θ Θ`, so `Θ e^{imφ}` has `Δ = -12`. -/
theorem legendre_32 (θ : ℝ) :
    Real.sin θ * deriv (fun t : ℝ => Real.sin t * deriv (fun t : ℝ => 15 * Real.sin t ^ 2 * Real.cos t) t) θ - 4 * (15 * Real.sin θ ^ 2 * Real.cos θ) +
      12 * Real.sin θ ^ 2 * (15 * Real.sin θ ^ 2 * Real.cos θ) = 0 := by
  have h1 : ∀ t : ℝ, deriv (fun t : ℝ => 15 * Real.sin t ^ 2 * Real.cos t) t = 30 * Real.sin t * Real.cos t ^ 2 - 15 * Real.sin t ^ 3 := by
    intro t
    veriphy_certified
      ( have hd :
          HasDerivAt (fun t => 15 * sin t ^ 2 * cos t)
            ((30 * ((Real.cos t) ^ 2) * (Real.sin t) - (15 * ((Real.sin t) ^ 3)) : ℝ)) (t) :=
          by
          convert ((((hasDerivAt_id' (t : ℝ)).cos).mul (((hasDerivAt_id' (t : ℝ)).sin).pow 2)).const_mul (15 : ℝ)) using 1
          all_goals
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; ring1)
            | (funext y; simp only [Pi.mul_apply, Pi.pow_apply]; ring_nf; done)
            | (simp only [Pi.pow_apply]; ring1)
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  have h2 : deriv (fun t : ℝ => Real.sin t * (30 * Real.sin t * Real.cos t ^ 2 - 15 * Real.sin t ^ 3)) θ = 60 * Real.sin θ * Real.cos θ ^ 3 - 120 * Real.sin θ ^ 3 * Real.cos θ := by
    veriphy_certified
      ( have hd :
          HasDerivAt (fun t => sin t * (30 * sin t * cos t ^ 2 - 15 * sin t ^ 3))
            ((15 * (2 * ((Real.cos θ) ^ 2) * (Real.sin θ) - ((Real.sin θ) ^ 3)) * (Real.cos θ) +
                (15 * (2 * ((Real.cos θ) ^ 3) - (7 * (Real.cos θ) * ((Real.sin θ) ^ 2))) * (Real.sin θ)) :
              ℝ))
            (θ) :=
          by
          convert
            ((((((((hasDerivAt_id' (θ : ℝ)).cos).pow 2).mul ((hasDerivAt_id' (θ : ℝ)).sin)).const_mul (2 : ℝ)).add
                      ((((hasDerivAt_id' (θ : ℝ)).sin).pow 3).const_mul (-1 : ℝ))).mul
                  ((hasDerivAt_id' (θ : ℝ)).sin)).const_mul
              (15 : ℝ)) using
            1
          all_goals (simp only [Pi.add_apply, Pi.mul_apply, Pi.pow_apply]; (field_simp; ring_nf; done))
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  simp only [show (deriv fun t : ℝ => 15 * Real.sin t ^ 2 * Real.cos t) = fun t : ℝ => 30 * Real.sin t * Real.cos t ^ 2 - 15 * Real.sin t ^ 3 from funext h1]
  rw [h2]
  linear_combination (60 * Real.sin θ ^ 2 * Real.cos θ) * Real.sin_sq_add_cos_sq θ

/-- `Θ_{33}(θ) = 15 * Real.sin θ ^ 3` solves the associated Legendre equation with `l = 3`, `m = 3`:
`sin θ (sin θ Θ')' - m² Θ = -l(l+1) sin² θ Θ`, so `Θ e^{imφ}` has `Δ = -12`. -/
theorem legendre_33 (θ : ℝ) :
    Real.sin θ * deriv (fun t : ℝ => Real.sin t * deriv (fun t : ℝ => 15 * Real.sin t ^ 3) t) θ - 9 * (15 * Real.sin θ ^ 3) +
      12 * Real.sin θ ^ 2 * (15 * Real.sin θ ^ 3) = 0 := by
  have h1 : ∀ t : ℝ, deriv (fun t : ℝ => 15 * Real.sin t ^ 3) t = 45 * Real.sin t ^ 2 * Real.cos t := by
    intro t
    veriphy_certified
      ( have hd : HasDerivAt (fun t => 15 * sin t ^ 3) ((45 * (Real.cos t) * ((Real.sin t) ^ 2) : ℝ)) (t) :=
          by
          convert ((((hasDerivAt_id' (t : ℝ)).sin).pow 3).const_mul (15 : ℝ)) using 1
          all_goals ring1
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  have h2 : deriv (fun t : ℝ => Real.sin t * (45 * Real.sin t ^ 2 * Real.cos t)) θ = 135 * Real.sin θ ^ 2 * Real.cos θ ^ 2 - 45 * Real.sin θ ^ 4 := by
    veriphy_certified
      ( have hd :
          HasDerivAt (fun t => sin t * (45 * sin t ^ 2 * cos t))
            ((135 * ((Real.cos θ) ^ 2) * ((Real.sin θ) ^ 2) - (45 * ((Real.sin θ) ^ 4)) : ℝ)) (θ) :=
          by
          convert ((((hasDerivAt_id' (θ : ℝ)).cos).mul (((hasDerivAt_id' (θ : ℝ)).sin).pow 3)).const_mul (45 : ℝ)) using 1
          all_goals
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; ring1)
            | (funext y; simp only [Pi.mul_apply, Pi.pow_apply]; ring_nf; done)
            | (simp only [Pi.pow_apply]; ring1)
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  simp only [show (deriv fun t : ℝ => 15 * Real.sin t ^ 3) = fun t : ℝ => 45 * Real.sin t ^ 2 * Real.cos t from funext h1]
  rw [h2]
  linear_combination (135 * Real.sin θ ^ 3) * Real.sin_sq_add_cos_sq θ

/-- `Δ cos² θ = 2 - 6 cos² θ`: `cos² θ` is not an eigenfunction; `cos² θ = 1/3 + (cos² θ - 1/3)` splits it into the
eigenvalues `0` and `-6`. -/
theorem lap_cos_sq (θ : ℝ) (hs : Real.sin θ ≠ 0) :
    deriv (fun t : ℝ => Real.sin t * deriv (fun t : ℝ => Real.cos t ^ 2) t) θ / Real.sin θ = 2 - 6 * Real.cos θ ^ 2 := by
  have h1 : ∀ t : ℝ, deriv (fun t : ℝ => Real.cos t ^ 2) t = -2 * Real.sin t * Real.cos t := by
    intro t
    veriphy_certified
      ( have hd : HasDerivAt (fun t => cos t ^ 2) ((-2 * (Real.cos t) * (Real.sin t) : ℝ)) (t) :=
          by
          convert (((hasDerivAt_id' (t : ℝ)).cos).pow 2) using 1
          all_goals ring1
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  have h2 : deriv (fun t : ℝ => Real.sin t * (-2 * Real.sin t * Real.cos t)) θ =
      -4 * Real.sin θ * Real.cos θ ^ 2 + 2 * Real.sin θ ^ 3 := by
    veriphy_certified
      ( have hd :
          HasDerivAt (fun t => sin t * (-2 * sin t * cos t))
            ((-4 * ((Real.cos θ) ^ 2) * (Real.sin θ) + (2 * ((Real.sin θ) ^ 3)) : ℝ)) (θ) :=
          by
          convert ((((hasDerivAt_id' (θ : ℝ)).cos).mul (((hasDerivAt_id' (θ : ℝ)).sin).pow 2)).const_mul (-2 : ℝ)) using 1
          all_goals
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; ring1)
            | (funext y; simp only [Pi.mul_apply, Pi.pow_apply]; ring_nf; done)
            | (simp only [Pi.pow_apply]; ring1)
        exact hd.deriv.trans (by ring1))
    -- certificate: fa-deriv2.sage
  simp only [show (deriv fun t : ℝ => Real.cos t ^ 2) = fun t : ℝ => -2 * Real.sin t * Real.cos t from funext h1]
  rw [h2, div_eq_iff hs]
  linear_combination (2 * Real.sin θ) * Real.sin_sq_add_cos_sq θ

/-- On the circle, `∂²_φ e^{inφ} = -n² e^{inφ}`: the Fourier modes with eigenvalues `-n²`. -/
theorem circle_modes (n φ : ℝ) :
    deriv (deriv fun p : ℝ => Real.sin (n * p)) φ = -n ^ 2 * Real.sin (n * φ) := by
  have h1 : deriv (fun p : ℝ => Real.sin (n * p)) = fun p => n * Real.cos (n * p) := by
    funext p
    exact (((hasDerivAt_id' p).const_mul n).sin).deriv.trans (by ring)
  rw [h1]
  exact ((((hasDerivAt_id' φ).const_mul n).cos).const_mul n).deriv.trans (by ring)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SphereLaplacian.circle_modes
-- test: fa-circle-modes-test.sage

end KaluzaKlein.FourierAnalysis.SphereLaplacian
