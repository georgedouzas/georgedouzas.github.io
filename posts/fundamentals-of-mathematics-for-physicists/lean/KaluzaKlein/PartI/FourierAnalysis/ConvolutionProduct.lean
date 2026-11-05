import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.Inversion

/-! # Fourier analysis: convolution product (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.ConvolutionProduct

open Real MeasureTheory Filter Topology Complex
open scoped FourierTransform Convolution

/-! ## def-convolution, exr-ft-convolution-theorem -/

/-- The convolution theorem (mathlib, normalisation `𝓕`): `𝓕(f ⋆ g) = 𝓕 f · 𝓕 g` for integrable `f, g`. -/
theorem convolution_theorem (f g : ℝ → ℂ) (hf : Integrable f) (hg : Integrable g) (ξ : ℝ) :
    𝓕 (f ⋆[ContinuousLinearMap.mul ℂ ℂ] g) ξ = 𝓕 f ξ * 𝓕 g ξ :=
  Real.fourier_mul_convolution_eq hf hg ξ

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ConvolutionProduct.convolution_theorem
-- test: fa-convolution-theorem-test.sage

/-- Convolution is commutative. -/
theorem convolution_comm_lsmul (f g : ℝ → ℝ) : f ⋆[ContinuousLinearMap.lsmul ℝ ℝ] g = g ⋆[ContinuousLinearMap.lsmul ℝ ℝ] f := by
  rw [← MeasureTheory.convolution_flip]
  congr 1
  ext x y
  simp [mul_comm]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ConvolutionProduct.convolution_comm_lsmul
-- test: fa-convolution-comm-test.sage

/-! ## exm-ft-box-box, exm-ft-sage-convolution -/

/-- For `0 ≤ x ≤ 2a` the boxes `[-a, a]` and `[x - a, x + a]` overlap in `[x - a, a]`, of length `2a - x`. -/
theorem box_overlap (a x : ℝ) (hx0 : 0 ≤ x) (hx : x ≤ 2 * a) :
    Set.Icc (-a) a ∩ Set.Icc (x - a) (x + a) = Set.Icc (x - a) a ∧ ∫ y in (x - a)..a, (1 : ℝ) = 2 * a - x := by
  constructor
  · ext y; simp only [Set.mem_inter_iff, Set.mem_Icc]; constructor
    · rintro ⟨⟨h1, h2⟩, h3, h4⟩; exact ⟨h3, h2⟩
    · rintro ⟨h1, h2⟩; exact ⟨⟨by linarith, h2⟩, h1, by linarith⟩
  · simp; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ConvolutionProduct.box_overlap
-- test: fa-box-overlap-test.sage

/-- The triangle's transform (cosine part): `2 ∫₀^{2a} (2a - x) cos kx dx = (2 - 2 cos 2ak)/k²`. -/
theorem triangle_transform (a k : ℝ) (hk : k ≠ 0) :
    2 * ∫ x in (0 : ℝ)..(2 * a), (2 * a - x) * Real.cos (k * x) = (2 - 2 * Real.cos (2 * a * k)) / k ^ 2 := by
  have hF : ∀ x ∈ Set.uIcc 0 (2 * a), HasDerivAt
      (fun x => (2 * a - x) * Real.sin (k * x) / k - Real.cos (k * x) / k ^ 2) ((2 * a - x) * Real.cos (k * x)) x := by
    intro x _
    have hs : HasDerivAt (fun x => Real.sin (k * x)) (Real.cos (k * x) * k) x := by
      simpa using ((hasDerivAt_id x).const_mul k).sin
    have hc : HasDerivAt (fun x => Real.cos (k * x)) (-Real.sin (k * x) * k) x := by
      simpa using ((hasDerivAt_id x).const_mul k).cos
    have e1 := ((((hasDerivAt_id' x).const_sub (2 * a)).mul hs).div_const k)
    have e2 := hc.div_const (k ^ 2)
    convert e1.sub e2 using 1
    field_simp; ring
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [mul_zero, Real.sin_zero, Real.cos_zero, sub_self, zero_mul, zero_div]
  rw [show k * (2 * a) = 2 * a * k by ring]
  field_simp
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ConvolutionProduct.triangle_transform
-- test: fa-triangle-transform-test.sage

/-- `2 - 2 cos 2ak = 4 sin² ak`: the triangle's transform is the square of the box's. -/
theorem two_sub_two_cos (a k : ℝ) : 2 - 2 * Real.cos (2 * (a * k)) = 4 * Real.sin (a * k) ^ 2 := by
  veriphy_certified
    ( try (rw [show ((2 * a * k : ℝ) : ℝ) = 2 * ((a * k : ℝ)) by ring])
      have hmc2 : Real.cos (2 * ((a * k : ℝ))) = ((Real.cos (a * k)) ^ 2 - ((Real.sin (a * k)) ^ 2) : ℝ) := by
        rw [show (2 : ℝ) * ((a * k : ℝ)) = ((a * k : ℝ)) + ((a * k : ℝ)) by ring, Real.cos_add]; ring
      have hms2 : Real.sin (2 * ((a * k : ℝ))) = (2 * (Real.cos (a * k)) * (Real.sin (a * k)) : ℝ) := by
        rw [show (2 : ℝ) * ((a * k : ℝ)) = ((a * k : ℝ)) + ((a * k : ℝ)) by ring, Real.sin_add]; ring
      simp only [hmc2, hms2]
      linear_combination ((-2 : ℝ)) * Real.sin_sq_add_cos_sq ((a * k : ℝ)))
  -- certificate: fa-trig-poly.sage

/-- `(χ₁ * χ₁)(1/2) = 3/2`, the length of `[-1/2, 1]`. -/
theorem box_conv_half : ∫ y in (-1 / 2 : ℝ)..1, (1 : ℝ) = 3 / 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 / 2 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (x_veriphy : ℝ)) ((fun y => 1) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((hasDerivAt_id' (x_veriphy : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (x_veriphy : ℝ)) (f' := fun y => 1) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: fa-integral.sage

/-! ## exm-ft-gaussian-gaussian -/

/-- `ĝ_σ ĝ_τ` is the transform of `(στ√(2π)/√(σ² + τ²)) g_{√(σ²+τ²)}`: the widths add in quadrature. -/
theorem gauss_product (σ τ k : ℝ) (hσ : 0 < σ) (hτ : 0 < τ) :
    (σ * Real.sqrt (2 * π) * Real.exp (-σ ^ 2 * k ^ 2 / 2)) * (τ * Real.sqrt (2 * π) * Real.exp (-τ ^ 2 * k ^ 2 / 2)) =
      (σ * τ * Real.sqrt (2 * π) / Real.sqrt (σ ^ 2 + τ ^ 2)) *
        (Real.sqrt (σ ^ 2 + τ ^ 2) * Real.sqrt (2 * π) * Real.exp (-(σ ^ 2 + τ ^ 2) * k ^ 2 / 2)) := by
  have h1 : 0 < Real.sqrt (σ ^ 2 + τ ^ 2) := Real.sqrt_pos.2 (by positivity)
  rw [show -(σ ^ 2 + τ ^ 2) * k ^ 2 / 2 = -σ ^ 2 * k ^ 2 / 2 + -τ ^ 2 * k ^ 2 / 2 by ring, Real.exp_add]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ConvolutionProduct.gauss_product
-- test: fa-gauss-product-test.sage

/-- The heat semigroup: `G_ε * G_η = G_{ε+η}` for `G_ε(x) = e^{-x²/2ε}/√(2πε)`. -/
theorem heat_semigroup (ε η x : ℝ) (hε : 0 < ε) (hη : 0 < η) :
    ∫ y : ℝ, (Real.exp (-(x - y) ^ 2 / (2 * ε)) / Real.sqrt (2 * π * ε)) * (Real.exp (-y ^ 2 / (2 * η)) / Real.sqrt (2 * π * η)) =
      Real.exp (-x ^ 2 / (2 * (ε + η))) / Real.sqrt (2 * π * (ε + η)) := by
  set b := (ε + η) / (2 * ε * η) with hb
  set c := η * x / (ε + η) with hc
  have hb0 : 0 < b := by positivity
  have hpt : ∀ y : ℝ, (Real.exp (-(x - y) ^ 2 / (2 * ε)) / Real.sqrt (2 * π * ε)) *
      (Real.exp (-y ^ 2 / (2 * η)) / Real.sqrt (2 * π * η)) =
      (Real.exp (-x ^ 2 / (2 * (ε + η))) / (Real.sqrt (2 * π * ε) * Real.sqrt (2 * π * η))) *
        Real.exp (-b * (y - c) ^ 2) := by
    intro y
    rw [div_mul_div_comm, div_mul_eq_mul_div, ← Real.exp_add, ← Real.exp_add]
    congr 2
    rw [hb, hc]; field_simp
    try ring
  simp_rw [hpt]
  rw [integral_const_mul]
  have hshift : ∫ y : ℝ, Real.exp (-b * (y - c) ^ 2) = ∫ y : ℝ, Real.exp (-b * y ^ 2) :=
    integral_sub_right_eq_self (fun y => Real.exp (-b * y ^ 2)) c
  rw [hshift, integral_gaussian]
  have e1 : Real.sqrt (π / b) = Real.sqrt (2 * π * ε) * Real.sqrt (2 * π * η) / Real.sqrt (2 * π * (ε + η)) := by
    rw [← Real.sqrt_mul (show (0 : ℝ) ≤ 2 * π * ε by positivity), eq_div_iff (Real.sqrt_pos.2 (by positivity)).ne',
      ← Real.sqrt_mul (show (0 : ℝ) ≤ π / b by positivity)]
    congr 1
    rw [hb]; field_simp
  rw [e1]
  have h1 : 0 < Real.sqrt (2 * π * ε) := Real.sqrt_pos.2 (by positivity)
  have h2 : 0 < Real.sqrt (2 * π * η) := Real.sqrt_pos.2 (by positivity)
  have h3 : 0 < Real.sqrt (2 * π * (ε + η)) := Real.sqrt_pos.2 (by positivity)
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ConvolutionProduct.heat_semigroup
-- test: fa-heat-semigroup-test.sage

/-! ## exm-ft-convolution-heat, exr-ft-heat-kernel -/

/-- The heat kernel `K_t(x) = e^{-x²/4κt} / √(4πκt)`. -/
noncomputable def heatKernel (κ t x : ℝ) : ℝ := Real.exp (-x ^ 2 / (4 * κ * t)) / Real.sqrt (4 * π * κ * t)

/-- `K_t` has unit integral. -/
theorem heat_kernel_unit (κ t : ℝ) (hκ : 0 < κ) (ht : 0 < t) : ∫ x : ℝ, heatKernel κ t x = 1 := by
  have := KaluzaKlein.FourierAnalysis.Inversion.heat_gauss_unit (2 * κ * t) (by positivity)
  simp only [heatKernel]
  rw [← this]
  congr 1; funext x
  congr 1
  · congr 1; ring
  · congr 1; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ConvolutionProduct.heat_kernel_unit
-- test: fa-heat-kernel-unit-test.sage

/-- `K_t` solves the heat equation `∂_t K = κ ∂²_x K`, from its certified partial derivatives. -/
theorem heat_kernel_pde (κ t x : ℝ) (hκ : 0 < κ) (ht : 0 < t) :
    deriv (fun s => Real.exp (-x ^ 2 / (4 * κ * s)) / Real.sqrt (4 * π * κ * s)) t =
      κ * ((x ^ 2 / (4 * κ ^ 2 * t ^ 2) - 1 / (2 * κ * t)) * (Real.exp (-x ^ 2 / (4 * κ * t)) / Real.sqrt (4 * π * κ * t))) ∧
    deriv (fun y => -y / (2 * κ * t) * (Real.exp (-y ^ 2 / (4 * κ * t)) / Real.sqrt (4 * π * κ * t))) x =
      (x ^ 2 / (4 * κ ^ 2 * t ^ 2) - 1 / (2 * κ * t)) * (Real.exp (-x ^ 2 / (4 * κ * t)) / Real.sqrt (4 * π * κ * t)) := by
  have hs : 0 < Real.sqrt (4 * π * κ * t) := Real.sqrt_pos.2 (by positivity)
  have hs2 : Real.sqrt (4 * π * κ * t) ^ 2 = 4 * π * κ * t := Real.sq_sqrt (by positivity)
  constructor
  · have hd : HasDerivAt (fun s => Real.sqrt (4 * π * κ * s)) (4 * π * κ * 1 / (2 * Real.sqrt (4 * π * κ * t))) t :=
      ((hasDerivAt_id' t).const_mul (4 * π * κ)).sqrt (by positivity)
    have he : HasDerivAt (fun s => Real.exp (-x ^ 2 / (4 * κ * s)))
        (Real.exp (-x ^ 2 / (4 * κ * t)) * (x ^ 2 / (4 * κ * t ^ 2))) t := by
      have h1 : HasDerivAt (fun s => -x ^ 2 / (4 * κ * s)) (x ^ 2 / (4 * κ * t ^ 2)) t := by
        have := (hasDerivAt_inv ht.ne').const_mul (-x ^ 2 / (4 * κ))
        convert this using 1
        · funext s; ring
        · ring
      exact h1.exp
    refine (he.div hd hs.ne').deriv.trans ?_
    generalize Real.sqrt (4 * π * κ * t) = S at hs hs2 ⊢
    have hπ : π = S ^ 2 / (4 * κ * t) := by rw [hs2]; field_simp
    rw [hπ]
    field_simp
    try ring
  · have h1 : HasDerivAt (fun y => -y / (2 * κ * t)) (-1 / (2 * κ * t)) x :=
      ((hasDerivAt_id' x).neg).div_const (2 * κ * t)
    have h2 := (((hasDerivAt_pow 2 x).neg.div_const (4 * κ * t)).exp).div_const (Real.sqrt (4 * π * κ * t))
    refine (h1.mul h2).deriv.trans ?_
    generalize Real.sqrt (4 * π * κ * t) = S at hs hs2 ⊢
    norm_num
    field_simp
    ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ConvolutionProduct.heat_kernel_pde
-- test: fa-heat-kernel-pde-test.sage

/-- The inverse transform of `e^{-κk²t}` is `K_t`: `K̂_t(k) = e^{-κk²t}`. -/
theorem heat_kernel_transform (κ t : ℝ) (hκ : 0 < κ) (ht : 0 < t) (k : ℝ) :
    ∫ x : ℝ, (heatKernel κ t x : ℂ) * Complex.exp (-(k * x) * Complex.I) = Complex.exp (-(κ * k ^ 2 * t)) := by
  set σ := Real.sqrt (2 * κ * t) with hσdef
  have hσ : 0 < σ := Real.sqrt_pos.2 (by positivity)
  have hσ2 : σ ^ 2 = 2 * κ * t := Real.sq_sqrt (by positivity)
  have hσ2C : ((σ : ℂ)) ^ 2 = 2 * κ * t := by exact_mod_cast hσ2
  have hG := KaluzaKlein.FourierAnalysis.TheTransform.gaussian_transform σ hσ k
  have hs : Real.sqrt (4 * π * κ * t) = σ * Real.sqrt (2 * π) := by
    rw [hσdef, ← Real.sqrt_mul (by positivity)]; congr 1; ring
  have hpt : ∀ x : ℝ, (heatKernel κ t x : ℂ) * Complex.exp (-(k * x) * Complex.I) =
      (1 / (σ * Real.sqrt (2 * π)) : ℂ) * (Complex.exp (-(x : ℂ) ^ 2 / (2 * σ ^ 2)) * Complex.exp (-((k : ℂ) * x) * Complex.I)) := by
    intro x
    have hx : ((-x ^ 2 / (4 * κ * t) : ℝ) : ℂ) = -(x : ℂ) ^ 2 / (2 * (σ : ℂ) ^ 2) := by
      rw [hσ2C]; push_cast; ring
    simp only [heatKernel, hs]
    rw [Complex.ofReal_div, Complex.ofReal_exp, hx]
    push_cast
    ring
  simp_rw [hpt]
  rw [integral_const_mul, hG]
  have hσ' : (σ : ℂ) ≠ 0 := by exact_mod_cast hσ.ne'
  have hsq : (Real.sqrt (2 * π) : ℂ) ≠ 0 := by exact_mod_cast (Real.sqrt_pos.2 (by positivity)).ne'
  have hk : -(σ : ℂ) ^ 2 * (k : ℂ) ^ 2 / 2 = -((κ : ℂ) * (k : ℂ) ^ 2 * t) := by rw [hσ2C]; ring
  rw [hk, one_div, inv_mul_cancel_left₀ (mul_ne_zero hσ' hsq)]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ConvolutionProduct.heat_kernel_transform
-- test: fa-heat-kernel-transform-test.sage

/-- The solution with `u₀ = e^{-x²/2}`: `u(t, x) = e^{-x²/2(1+2κt)} / √(1 + 2κt)` starts at `u₀` and solves the heat
equation (from its certified derivatives in `t` and `x`). -/
theorem heat_gaussian_solution (κ t x : ℝ) (hκ : 0 < κ) (ht : 0 ≤ t) :
    Real.exp (-x ^ 2 / (2 * (1 + 2 * κ * 0))) / Real.sqrt (1 + 2 * κ * 0) = Real.exp (-x ^ 2 / 2) ∧
    deriv (fun s => Real.exp (-x ^ 2 / (2 * (1 + 2 * κ * s))) / Real.sqrt (1 + 2 * κ * s)) t =
      κ * ((x ^ 2 / (1 + 2 * κ * t) ^ 2 - 1 / (1 + 2 * κ * t)) *
        (Real.exp (-x ^ 2 / (2 * (1 + 2 * κ * t))) / Real.sqrt (1 + 2 * κ * t))) := by
  constructor
  · simp
  have hp : 0 < 1 + 2 * κ * t := by positivity
  have hs : 0 < Real.sqrt (1 + 2 * κ * t) := Real.sqrt_pos.2 hp
  have hs2 : Real.sqrt (1 + 2 * κ * t) ^ 2 = 1 + 2 * κ * t := Real.sq_sqrt hp.le
  have hd := (((hasDerivAt_id' t).const_mul (2 * κ)).const_add 1).sqrt hp.ne'
  have he : HasDerivAt (fun s => Real.exp (-x ^ 2 / (2 * (1 + 2 * κ * s))))
      (Real.exp (-x ^ 2 / (2 * (1 + 2 * κ * t))) * (x ^ 2 * κ / (1 + 2 * κ * t) ^ 2)) t := by
    have h1 : HasDerivAt (fun s => -x ^ 2 / (2 * (1 + 2 * κ * s))) (x ^ 2 * κ / (1 + 2 * κ * t) ^ 2) t := by
      have := ((((hasDerivAt_id' t).const_mul (2 * κ)).const_add 1).inv hp.ne').const_mul (-x ^ 2 / 2)
      convert this using 1
      · funext s; simp only [Pi.inv_apply]; rw [div_mul_eq_div_div]; ring1
      · ring
    exact h1.exp
  refine (he.div hd hs.ne').deriv.trans ?_
  generalize Real.sqrt (1 + 2 * κ * t) = S at hs hs2 ⊢
  rw [← hs2]
  field_simp
  try ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ConvolutionProduct.heat_gaussian_solution
-- test: fa-heat-gaussian-solution-test.sage

end KaluzaKlein.FourierAnalysis.ConvolutionProduct
