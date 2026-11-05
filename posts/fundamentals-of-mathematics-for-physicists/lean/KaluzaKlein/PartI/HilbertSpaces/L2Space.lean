import KaluzaKlein.PartI.HilbertSpaces.InnerProducts

/-!
# Hilbert spaces and operator theory, section "The space L²"

The claims of the section "The space `L²`" of `tutorial/part-i-07-hilbert-spaces-operators.qmd`: null sets and
the identification of functions agreeing almost everywhere, `L²` as a Hilbert space with the continuous functions
dense (Riesz-Fischer), membership of bounded functions, of `x^{-1/2}` and `x^{-1/4}` in `L²(0, 2π)`, the Gaussian,
the constant and the plane wave on the line, the sphere with its area element, the four functions of the membership
exercise, the Dirichlet function, and the normalisation of the `l = 1` spherical harmonics.
-/

namespace KaluzaKlein.HilbertSpaces.L2Space

noncomputable section

open Matrix Basic ComplexConjugate MeasureTheory

/-! ## def-l2-space -/

/-- A countable set (the rationals, a finite set, a point) is null for Lebesgue measure. -/
theorem countable_null (s : Set ℝ) (hs : s.Countable) : volume s = 0 := hs.measure_zero volume

/-- Functions agreeing outside a null set are identified in `L²`: they define the same element. -/
theorem ae_eq_identified (f g : ℝ → ℂ) (hf : MemLp f 2 volume) (hg : MemLp g 2 volume) (h : f =ᵐ[volume] g) :
    hf.toLp f = hg.toLp g := (MemLp.toLp_eq_toLp_iff hf hg).mpr h

/-- `L²(X)` is a Hilbert space (the Riesz-Fischer theorem): an inner product space, complete. -/
theorem l2_complete (X : Type) [MeasurableSpace X] (μ : Measure X) :
    CompleteSpace (Lp ℂ 2 μ) ∧ Nonempty (InnerProductSpace ℂ (Lp ℂ 2 μ)) := ⟨inferInstance, ⟨inferInstance⟩⟩

/-- The continuous functions are dense in `L²` of a compact interval (here `L²` of a compact metric space with a
finite regular measure). -/
theorem continuous_dense (X : Type) [MetricSpace X] [CompactSpace X] [MeasurableSpace X] [BorelSpace X]
    (μ : Measure X) [μ.WeaklyRegular] [IsFiniteMeasure μ] :
    DenseRange (ContinuousMap.toLp (E := ℂ) 2 μ ℂ : C(X, ℂ) →L[ℂ] Lp ℂ 2 μ) :=
  ContinuousMap.toLp_denseRange (E := ℂ) (𝕜 := ℂ) (μ := μ) (by norm_num)

/-! ## exm-l2-circle -/

/-- A bounded measurable function on `(0, 2π)` (a step function, `sign (x - π)`, every bounded Riemann integrable
function) is in `L²(0, 2π)`. -/
theorem bounded_memLp (f : ℝ → ℂ) (hf : Measurable f) (C : ℝ) (hC : ∀ x, ‖f x‖ ≤ C) :
    MemLp f 2 (volume.restrict (Set.Ioo 0 (2 * Real.pi))) :=
  MemLp.of_bound hf.aestronglyMeasurable C (Filter.Eventually.of_forall hC)

/-- `∫₀^{2π} x^s dx` converges exactly when `s > -1`: `1/x` is not integrable on `(0, 2π)`, so `1/√x ∉ L²`,
while `x^{-1/2}` is integrable, so `x^{-1/4} ∈ L²`. -/
theorem integrableOn_rpow_iff (s : ℝ) : IntegrableOn (fun x : ℝ => x ^ s) (Set.Ioo 0 (2 * Real.pi)) ↔ -1 < s :=
  integrableOn_Ioo_rpow_iff (by positivity)

/-- `∫₀^{2π} x^{-1/2} dx = 2√(2π)`, the squared norm of `x^{-1/4}`. -/
theorem integral_rpow_neg_half : ∫ x in (0 : ℝ)..(2 * Real.pi), x ^ (-1 / 2 : ℝ) = 2 * Real.sqrt (2 * Real.pi) := by
  rw [integral_rpow (Or.inl (by norm_num))]
  have h : (2 * Real.pi) ^ (-1 / 2 + 1 : ℝ) = Real.sqrt (2 * Real.pi) := by
    rw [Real.sqrt_eq_rpow]; norm_num
  rw [h, Real.zero_rpow (by norm_num)]
  ring

/-! ## exm-l2-real-line -/

/-- The Gaussian integral `∫ e^{-x²} dx = √π`: `‖e^{-x²/2}‖² = √π`. -/
theorem gaussian_norm_sq : ∫ x : ℝ, Real.exp (-x ^ 2) = Real.sqrt Real.pi := by
  have := integral_gaussian 1
  simpa using this

/-- `∫ (c e^{-x²/2})² dx = c² √π`; for `c = π^{-1/4}` this is `1`: `π^{-1/4} e^{-x²/2}` is a unit vector. -/
theorem gaussian_scaled_norm_sq (c : ℝ) : ∫ x : ℝ, (c * Real.exp (-x ^ 2 / 2)) ^ 2 = c ^ 2 * Real.sqrt Real.pi := by
  have h : ∀ x : ℝ, (c * Real.exp (-x ^ 2 / 2)) ^ 2 = c ^ 2 * Real.exp (-x ^ 2) := by
    intro x
    rw [mul_pow, ← Real.exp_nat_mul]
    push_cast
    ring_nf
  simp_rw [h]
  rw [integral_const_mul, gaussian_norm_sq]

/-- The constant `1` and the plane wave `e^{ikx}` have `|f|² = 1`, which is not integrable on `ℝ`: they are not in
`L²(ℝ)`. -/
theorem const_not_integrable (k : ℝ) : ¬ Integrable (fun _ : ℝ => (1 : ℝ)) ∧
    ¬ Integrable (fun x : ℝ => ‖Complex.exp (Complex.I * k * x)‖ ^ 2) := by
  have h : (fun x : ℝ => ‖Complex.exp (Complex.I * k * x)‖ ^ 2) = fun _ => (1 : ℝ) := by
    ext x
    rw [show Complex.I * k * x = ((k * x : ℝ) : ℂ) * Complex.I by push_cast; ring, Complex.norm_exp_ofReal_mul_I]
    norm_num
  rw [h]
  have : ¬ Integrable (fun _ : ℝ => (1 : ℝ)) := by
    rw [integrable_const_iff]
    simp
  exact ⟨this, this⟩

/-- `∫ (1 + x²)⁻¹ dx = π` and `∫ (1 + x²)⁻² dx = π/2`: `1/√(1 + x²)` and `1/(1 + x²)` are in `L²(ℝ)` (stated; the
second value is evaluated by the oracle, the first is mathlib's `integral_univ_inv_one_add_sq`). -/
def inv_one_add_sq_norms : Prop := ∀ k : Fin 2, ∫ x : ℝ, (1 / (1 + x ^ 2)) ^ ((k : ℕ) + 1) = ![Real.pi, Real.pi / 2] k

/-! ## exm-l2-sphere -/

/-- `∫₀^π sin θ dθ = 2`, so that `‖1‖² = ∫₀^{2π} 2 dφ = 4π`, the area of the sphere. -/
theorem integral_sin_pi : ∫ θ in (0 : ℝ)..Real.pi, Real.sin θ = 2 := by
  VERIPHY[hs-integral]

/-- `‖1‖² = ∫₀^{2π} (∫₀^π sin θ dθ) dφ = ∫₀^{2π} 2 dφ = 4π`. -/
theorem sphere_area : ∫ φ in (0 : ℝ)..(2 * Real.pi), (2 : ℝ) = 4 * Real.pi := by
  VERIPHY[hs-integral]

/-- `∫₀^π cos²θ sin θ dθ = 2/3`, so `‖cos θ‖² = 2π · 2/3 = 4π/3`. -/
theorem integral_cos_sq_sin : ∫ θ in (0 : ℝ)..Real.pi, Real.cos θ ^ 2 * Real.sin θ = 2 / 3 := by
  VERIPHY[hs-integral]

/-- `‖cos θ‖² = ∫₀^{2π} (2/3) dφ = 4π/3`. -/
theorem cos_norm_sq : ∫ φ in (0 : ℝ)..(2 * Real.pi), (2 / 3 : ℝ) = 4 * Real.pi / 3 := by
  VERIPHY[hs-integral]

/-- `⟪1, cos θ⟫ = 2π ∫₀^π cos θ sin θ dθ = 0`. -/
theorem integral_cos_sin : ∫ θ in (0 : ℝ)..Real.pi, Real.cos θ * Real.sin θ = 0 := by
  VERIPHY[hs-integral]

/-- Normalising: `(√(1/c))² c = 1` for `c > 0`; with `c = 4π` this is `Y₀₀ = 1/√(4π)` and with `c = 4π/3`,
`Y₁₀ = √(3/(4π)) cos θ`. -/
theorem normalise_sq (c : ℝ) (hc : 0 < c) : (Real.sqrt (1 / c)) ^ 2 * c = 1 := by
  rw [Real.sq_sqrt (by positivity)]
  field_simp

/-! ## exr-l2-membership -/

/-- `∫ (1 + x²)⁻¹ dx = π` on `ℝ`: `1/√(1 + x²)` is in `L²(ℝ)`. -/
theorem integral_inv_one_add_sq_line : ∫ x : ℝ, (1 + x ^ 2)⁻¹ = Real.pi := integral_univ_inv_one_add_sq

/-- `∫ sin x / x dx = π` and `∫ sin²x / x² dx = π` on `ℝ`: `sin x / x` is in `L²(ℝ)`, although not absolutely
integrable (stated; the oracle evaluates the integrals). -/
def sinc_integrals : Prop := ∀ k : Fin 2, ∫ x : ℝ, (Real.sin x / x) ^ ((k : ℕ) + 1) = Real.pi

/-- `∫₀¹ x^{-2/3} dx = 3`: `x^{-1/3}` is in `L²(0, 1)`. -/
theorem integral_rpow_neg_two_thirds : ∫ x in (0 : ℝ)..1, x ^ (-2 / 3 : ℝ) = 3 := by
  rw [integral_rpow (Or.inl (by norm_num))]
  norm_num

/-- `∫₀¹ x^s dx` diverges for `s ≤ -1`: `x^{-1/2} ∉ L²(0, 1)` since `∫₀¹ x⁻¹ dx` diverges. -/
theorem integrableOn_rpow_unit_iff (s : ℝ) : IntegrableOn (fun x : ℝ => x ^ s) (Set.Ioo 0 1) ↔ -1 < s :=
  integrableOn_Ioo_rpow_iff one_pos

/-! ## exr-l2-dirichlet-function -/

/-- The indicator of a countable set (the Dirichlet function `D = 1_ℚ`) vanishes almost everywhere: the set is
null. -/
theorem indicator_countable_ae_zero (s : Set ℝ) (hs : s.Countable) :
    s.indicator (fun _ => (1 : ℝ)) =ᵐ[volume] 0 :=
  (indicator_ae_eq_zero (f := fun _ => (1 : ℝ))).mpr (by simpa using hs.measure_zero volume)

/-- `∫₀¹ D² = 0`: the indicator of a countable set (the Dirichlet function) is the zero element of `L²(0, 1)`. -/
theorem indicator_countable_integral_zero (s : Set ℝ) (hs : s.Countable) :
    ∫ x in (0 : ℝ)..1, (s.indicator (fun _ => (1 : ℝ)) x) ^ 2 = 0 := by
  have h := indicator_countable_ae_zero s hs
  apply intervalIntegral.integral_congr_ae ?_ |>.trans intervalIntegral.integral_zero
  filter_upwards [h] with x hx _
  simp [hx]

/-- Positive definiteness holds for classes: `‖f‖ = 0` in `L²` exactly when `f = 0` as a class. -/
theorem norm_zero_iff_class_zero (X : Type) [MeasurableSpace X] (μ : Measure X) (f : Lp ℂ 2 μ) : ‖f‖ = 0 ↔ f = 0 :=
  Lp.norm_eq_zero_iff (by norm_num)

/-- A point is a null set, so changing a function at one point does not change its class: `Function.update f x₀ c`
agrees with `f` almost everywhere. -/
theorem update_ae_eq (f : ℝ → ℂ) (x₀ : ℝ) (c : ℂ) : Function.update f x₀ c =ᵐ[volume] f := by
  have h : volume ({x₀} : Set ℝ) = 0 := measure_singleton x₀
  rw [Filter.EventuallyEq, ae_iff]
  refine measure_mono_null (fun x hx => ?_) h
  by_contra hne
  exact hx (Function.update_of_ne hne c f)

/-! ## exr-l2-sphere-norm -/

/-- `∫₀^π sin³θ dθ = 4/3`, so `‖sin θ e^{iφ}‖² = 2π · 4/3 = 8π/3`. -/
theorem integral_sin_cube : ∫ θ in (0 : ℝ)..Real.pi, Real.sin θ ^ 3 = 4 / 3 := by
  VERIPHY[hs-integral]

/-- `‖sin θ e^{iφ}‖² = ∫₀^{2π} (4/3) dφ = 8π/3`. -/
theorem sin_exp_norm_sq : ∫ φ in (0 : ℝ)..(2 * Real.pi), (4 / 3 : ℝ) = 8 * Real.pi / 3 := by
  VERIPHY[hs-integral]

/-- `∫₀^{2π} e^{inφ} dφ = 0` for `n ≠ 0`: the cross inner product `⟪cos θ, sin θ e^{iφ}⟫` vanishes. -/
theorem integral_exp_phase (n : ℤ) (hn : n ≠ 0) : ∫ φ in (0 : ℝ)..(2 * Real.pi), Complex.exp (Complex.I * n * φ) = 0 := by
  have := InnerProducts.exp_orthogonality 0 n
  simp only [Int.cast_zero, mul_zero, zero_mul, Complex.exp_zero, map_one, one_mul] at this
  rw [this, if_neg (Ne.symm hn)]

end

end KaluzaKlein.HilbertSpaces.L2Space
