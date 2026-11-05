import KaluzaKlein.PartI.HilbertSpaces.InnerProducts

/-!
# Hilbert spaces and operator theory, section "Orthonormal systems"

The claims of the section "Orthonormal systems" of `tutorial/part-i-07-hilbert-spaces-operators.qmd`: the
Gram-Schmidt process (orthonormal output, the same spans), the orthogonal projection onto the span of a finite
orthonormal system (`u - Pu` orthogonal to the span, Pythagoras, Bessel's inequality, best approximation), the pair
`(1, ±i)/√2` in `ℂ²`, Gram-Schmidt on the monomials in `L²(-1, 1)` (the Legendre polynomials) and on three vectors
of `ℝ³`, and the exponentials `e^{inx}/√(2π)`.
-/

namespace KaluzaKlein.HilbertSpaces.OrthonormalSystems

noncomputable section

open Matrix Basic ComplexConjugate

/-! ## def-orthonormal-system -/

/-- The Gram-Schmidt process turns a linearly independent family into an orthonormal system with the same spans. -/
theorem gramSchmidt_orthonormal_span (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (n : ℕ)
    (f : Fin n → E) (hf : LinearIndependent ℂ f) :
    Orthonormal ℂ (gramSchmidtNormed ℂ f) ∧
      ∀ k, Submodule.span ℂ (gramSchmidtNormed ℂ f '' Set.Iic k) = Submodule.span ℂ (f '' Set.Iic k) := by
  refine ⟨gramSchmidt_orthonormal hf, fun k => ?_⟩
  rw [span_gramSchmidtNormed, span_gramSchmidt_Iic]
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.gramSchmidt_orthonormal_span, hs-on-gram-schmidt-test]

/-- The orthogonal projection onto the span of a finite system `e`: `P u = ∑ₖ ⟪eₖ, u⟫ eₖ`. -/
noncomputable def proj {E : Type} [NormedAddCommGroup E] [InnerProductSpace ℂ E] {n : ℕ} (e : Fin n → E) (u : E) :
    E := ∑ k, inner ℂ (e k) u • e k

/-- `u - P u` is orthogonal to every `eⱼ`, hence to the span. -/
theorem inner_sub_proj (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (n : ℕ) (e : Fin n → E)
    (he : Orthonormal ℂ e) (u : E) (j : Fin n) : inner ℂ (e j) (u - proj e u) = 0 := by
  rw [inner_sub_right, proj, inner_sum]
  simp [inner_smul_right, orthonormal_iff_ite.mp he]
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.inner_sub_proj, hs-on-inner-sub-proj-test]

/-- `‖P u‖² = ∑ₖ |⟪eₖ, u⟫|²`. -/
theorem norm_proj_sq (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (n : ℕ) (e : Fin n → E)
    (he : Orthonormal ℂ e) (u : E) : ‖proj e u‖ ^ 2 = ∑ k, ‖inner ℂ (e k) u‖ ^ 2 := by
  rw [← inner_self_eq_norm_sq (𝕜 := ℂ), proj, inner_sum]
  simp only [inner_smul_right, he.inner_left_fintype, Complex.mul_conj']
  push_cast
  simp [Complex.re_sum]
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.norm_proj_sq, hs-on-norm-proj-test]

/-- Bessel's inequality `∑ₖ |⟪eₖ, u⟫|² ≤ ‖u‖²`. -/
theorem bessel (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (n : ℕ) (e : Fin n → E)
    (he : Orthonormal ℂ e) (u : E) : ∑ k, ‖inner ℂ (e k) u‖ ^ 2 ≤ ‖u‖ ^ 2 := he.sum_inner_products_le u
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.bessel, hs-on-bessel-test]

/-- `u - P u` is orthogonal to every vector of the span. -/
theorem inner_sub_proj_span (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (n : ℕ) (e : Fin n → E)
    (he : Orthonormal ℂ e) (u w : E) (hw : w ∈ Submodule.span ℂ (Set.range e)) :
    inner ℂ w (u - proj e u) = 0 := by
  obtain ⟨c, rfl⟩ := Submodule.mem_span_range_iff_exists_fun.mp hw
  rw [sum_inner]
  simp [inner_smul_left, inner_sub_proj E n e he u]
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.inner_sub_proj_span, hs-on-orthogonal-span-test]

/-- Pythagoras: `‖u‖² = ‖P u‖² + ‖u - P u‖²`. -/
theorem norm_sq_proj_add (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (n : ℕ) (e : Fin n → E)
    (he : Orthonormal ℂ e) (u : E) : ‖u‖ ^ 2 = ‖proj e u‖ ^ 2 + ‖u - proj e u‖ ^ 2 := by
  have h : inner ℂ (proj e u) (u - proj e u) = 0 :=
    inner_sub_proj_span E n e he u _ (Submodule.sum_mem _ fun k _ => Submodule.smul_mem _ _
      (Submodule.subset_span ⟨k, rfl⟩))
  have := norm_add_sq_eq_norm_sq_add_norm_sq_of_inner_eq_zero (proj e u) (u - proj e u) h
  rwa [add_sub_cancel] at this
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.norm_sq_proj_add, hs-on-pythagoras-test]

/-- `P u` is the point of the span closest to `u`: `‖u - P u‖ ≤ ‖u - w‖` for every `w` in the span, with
equality only for `w = P u`. -/
theorem proj_closest (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (n : ℕ) (e : Fin n → E)
    (he : Orthonormal ℂ e) (u w : E) (hw : w ∈ Submodule.span ℂ (Set.range e)) :
    ‖u - proj e u‖ ≤ ‖u - w‖ ∧ (‖u - w‖ = ‖u - proj e u‖ → w = proj e u) := by
  have hmem : proj e u - w ∈ Submodule.span ℂ (Set.range e) :=
    Submodule.sub_mem _ (Submodule.sum_mem _ fun k _ => Submodule.smul_mem _ _ (Submodule.subset_span ⟨k, rfl⟩)) hw
  have h0 : inner ℂ (proj e u - w) (u - proj e u) = 0 := inner_sub_proj_span E n e he u _ hmem
  have hpy := norm_add_sq_eq_norm_sq_add_norm_sq_of_inner_eq_zero (u - proj e u) (proj e u - w)
    (by rw [← inner_conj_symm, h0, map_zero])
  rw [sub_add_sub_cancel] at hpy
  constructor
  · have : ‖u - proj e u‖ ^ 2 ≤ ‖u - w‖ ^ 2 := by rw [hpy]; nlinarith [norm_nonneg (proj e u - w)]
    exact abs_le_abs_of_sq_le_sq' (by positivity) this |>.2 |> fun h => by
      simpa [abs_of_nonneg (norm_nonneg _)] using h
  · intro heq
    rw [heq] at hpy
    have : ‖proj e u - w‖ ^ 2 = 0 := by linarith
    have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp this
    exact (sub_eq_zero.mp (norm_eq_zero.mp this)).symm
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.proj_closest, hs-on-best-approximation-test]

/-! ## exm-on-rotated-basis -/

/-- `⟪e₁, e₁⟫ = (1 + 1)/2 = 1` for `e₁ = (1, i)/√2` (the two factors `1/√2` combined into the row). -/
theorem e1_norm : !![(1 : ℂ) / 2, -Complex.I / 2] * !![1; Complex.I] = 1 := by
  VERIPHY[hs-matrix]

/-- `⟪e₁, e₂⟫ = (1 + conj i · (-i))/2 = (1 - 1)/2 = 0` for `e₂ = (1, -i)/√2`. -/
theorem e1_e2_orthogonal : !![(1 : ℂ) / 2, -Complex.I / 2] * !![1; -Complex.I] = 0 := by
  VERIPHY[hs-matrix]

/-- `(1, i)` and `(1, -i)` are eigenvectors of `σ₂`, for the eigenvalues `1` and `-1`. -/
theorem sigma2_eigenvectors : σ2 * !![1; Complex.I] = !![1; Complex.I] ∧ σ2 * !![1; -Complex.I] = !![-1; Complex.I] := by
  constructor
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- The components of `u = (a, b)`: `√2 ⟪e₁, u⟫ = a - i b` and `√2 ⟪e₂, u⟫ = a + i b`. -/
theorem rotated_components (a b : ℂ) :
    !![(1 : ℂ), -Complex.I] * !![a; b] = !![a - Complex.I * b] ∧ !![(1 : ℂ), Complex.I] * !![a; b] = !![a + Complex.I * b] := by
  constructor
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- `|⟪e₁, u⟫|² + |⟪e₂, u⟫|² = (|a - ib|² + |a + ib|²)/2 = |a|² + |b|² = ‖u‖²`. -/
theorem rotated_parseval (a b : ℂ) : (‖a - Complex.I * b‖ ^ 2 + ‖a + Complex.I * b‖ ^ 2) / 2 = ‖a‖ ^ 2 + ‖b‖ ^ 2 := by
  simp only [Complex.sq_norm, Complex.normSq_apply, Complex.sub_re, Complex.sub_im, Complex.add_re, Complex.add_im,
    Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im]
  ring
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.rotated_parseval, hs-on-rotated-parseval-test]

/-! ## exm-on-legendre -/

/-- `⟪1, x⟫ = ∫₋₁¹ x dx = 0`. -/
theorem integral_x : ∫ x in (-1 : ℝ)..1, x = 0 := by
  VERIPHY[hs-integral]

/-- `⟪1, 1⟫ = ∫₋₁¹ 1 dx = 2`. -/
theorem integral_one : ∫ x in (-1 : ℝ)..1, (1 : ℝ) = 2 := by
  VERIPHY[hs-integral]

/-- `⟪1, x²⟫ = ⟪x, x⟫ = ∫₋₁¹ x² dx = 2/3`. -/
theorem integral_x_sq : ∫ x in (-1 : ℝ)..1, x ^ 2 = 2 / 3 := by
  VERIPHY[hs-integral]

/-- `⟪x, x²⟫ = ∫₋₁¹ x³ dx = 0`. -/
theorem integral_x_cube : ∫ x in (-1 : ℝ)..1, x ^ 3 = 0 := by
  VERIPHY[hs-integral]

/-- `w₃ = x² - ⟪1, x²⟫/⟪1, 1⟫ - ⟪x, x²⟫/⟪x, x⟫ x = x² - (2/3)/2 - 0 · x = x² - 1/3`, and `x² - 1/3 = (2/3) P₂`
with `P₂ = (3x² - 1)/2`. -/
theorem w3_legendre (x : ℝ) :
    x ^ 2 - (2 / 3) / 2 - (0 / (2 / 3)) * x = x ^ 2 - 1 / 3 ∧ x ^ 2 - 1 / 3 = (2 / 3) * ((3 * x ^ 2 - 1) / 2) := by
  constructor <;> ring
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.w3_legendre, hs-on-w3-test]

/-- The normalisation `‖P₂‖² = ∫₋₁¹ ((3x² - 1)/2)² dx = 2/5`, so that `e₂ = √(5/2) P₂` is a unit vector. -/
theorem integral_P2_sq : ∫ x in (-1 : ℝ)..1, ((3 * x ^ 2 - 1) / 2) ^ 2 = 2 / 5 := by
  VERIPHY[hs-integral]

/-- `‖P₀‖² = 2 = 2/1` and `‖P₁‖² = 2/3`: the normalisation `eₖ = √((2k + 1)/2) Pₖ` for `k = 0, 1` (with
`integral_one`, `integral_x_sq`); the general pattern `∫₋₁¹ Pₖ² = 2/(2k + 1)` for `k = 0, 1, 2`. -/
theorem legendre_norms (k : Fin 3) :
    ∫ x in (-1 : ℝ)..1, (![fun _ : ℝ => (1 : ℝ), fun x => x, fun x => (3 * x ^ 2 - 1) / 2] k x) ^ 2 = 2 / (2 * (k : ℝ) + 1) := by
  fin_cases k
  · simp only [Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero, one_pow, Fin.val_zero, Nat.cast_zero, mul_zero,
      zero_add, div_one]
    exact integral_one
  · simp only [Fin.mk_one, Fin.isValue, Matrix.cons_val_one, Matrix.head_cons, Fin.val_one, Nat.cast_one, mul_one]
    exact integral_x_sq
  · simp only [Fin.reduceFinMk, Matrix.cons_val_two, Matrix.tail_cons, Matrix.head_cons, Fin.val_two, Nat.cast_ofNat]
    rw [integral_P2_sq]; norm_num
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.legendre_norms, hs-on-legendre-norms-test]

/-! ## exm-on-exponentials -/

/-- `e_n(x) = e^{inx}/√(2π)` is an orthonormal system: `(1/(2π)) ∫₀^{2π} conj e^{imx} e^{inx} dx = δ_mn`. -/
theorem exp_normalised_orthonormal (m n : ℤ) :
    (1 / (2 * Real.pi) : ℂ) * ∫ x in (0 : ℝ)..(2 * Real.pi),
      conj (Complex.exp (Complex.I * m * x)) * Complex.exp (Complex.I * n * x) = if m = n then 1 else 0 := by
  rw [InnerProducts.exp_orthogonality]
  split_ifs
  · push_cast; field_simp
  · simp
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.exp_normalised_orthonormal, hs-on-exp-orthonormal-test]

/-- `⟪e_n, f⟫ e_n = c_n e^{inx}` with `c_n = (1/2π) ∫ e^{-inx} f`: the two factors `1/√(2π)` combine into `1/(2π)`
(`J` stands for the integral `∫₀^{2π} e^{-inx} f`). -/
theorem proj_coefficient (J z : ℂ) :
    (1 / (Real.sqrt (2 * Real.pi) : ℂ)) * J * (z / (Real.sqrt (2 * Real.pi) : ℂ)) = (1 / (2 * Real.pi) : ℂ) * J * z := by
  have h : ((Real.sqrt (2 * Real.pi) : ℝ) : ℂ) * (Real.sqrt (2 * Real.pi) : ℝ) = (2 * Real.pi : ℝ) := by
    rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by positivity)]
  have hs : ((Real.sqrt (2 * Real.pi) : ℝ) : ℂ) ≠ 0 := by
    exact_mod_cast (Real.sqrt_pos.mpr (by positivity)).ne'
  have hpi : ((2 * Real.pi : ℝ) : ℂ) ≠ 0 := by exact_mod_cast (by positivity : (2 * Real.pi : ℝ) ≠ 0)
  field_simp
  push_cast at h ⊢
  linear_combination J * z * h
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.proj_coefficient, hs-on-proj-coefficient-test]

/-- Bessel's inequality for the Fourier system, `2π ∑_{|n| ≤ N} |c_n|² ≤ ∫₀^{2π} |f|²` with
`c_n = (1/2π) ∫₀^{2π} e^{-inx} f(x) dx`, for continuous `f` (stated; the transfer from mathlib's `AddCircle`
Parseval identity to interval integrals is not carried out). -/
def fourierBessel : Prop := ∀ (f : ℝ → ℂ), Continuous f → ∀ N : ℕ,
    2 * Real.pi * ∑ n ∈ Finset.Icc (-(N : ℤ)) N,
      ‖(1 / (2 * Real.pi) : ℂ) * ∫ x in (0 : ℝ)..(2 * Real.pi), Complex.exp (-(Complex.I * n * x)) * f x‖ ^ 2 ≤
      ∫ x in (0 : ℝ)..(2 * Real.pi), ‖f x‖ ^ 2
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.fourierBessel, hs-on-fourier-bessel-test]

/-! ## exr-on-gram-schmidt-r3 -/

/-- `e₁ = (1, 1, 0)/√2`: `⟪v₁, v₁⟫ = 2`. -/
theorem gs_v1_norm : !![(1 : ℚ), 1, 0] * !![1; 1; 0] = !![2] := by
  VERIPHY[hs-matrix]

/-- `w₂ = v₂ - ½ v₁ = (1, 0, 1) - (½, ½, 0) = (½, -½, 1)`, with `⟪v₁, v₂⟫/⟪v₁, v₁⟫ = 1/2`. -/
theorem gs_w2 : !![(1 : ℚ), 1, 0] * !![1; 0; 1] = !![1] ∧ !![(1 : ℚ); 0; 1] - !![1 / 2; 1 / 2; 0] = !![1 / 2; -1 / 2; 1] := by
  constructor
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- `w₂ ⊥ v₁`, `⟪w₂, w₂⟫ = 3/2` (so `e₂ = (1, -1, 2)/√6`). -/
theorem gs_w2_props : !![(1 : ℚ), 1, 0] * !![1 / 2; -1 / 2; 1] = !![0] ∧ !![(1 : ℚ) / 2, -1 / 2, 1] * !![1 / 2; -1 / 2; 1] = !![3 / 2] := by
  constructor
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- `w₃ = v₃ - ½ v₁ - ⅙ (1, -1, 2) = (-2/3, 2/3, 2/3)`, the coefficients being `⟪v₁, v₃⟫/2 = 1/2` and
`⟪(1, -1, 2), v₃⟫/6 = 1/6`. -/
theorem gs_w3 : !![(0 : ℚ); 1; 1] - !![1 / 2; 1 / 2; 0] - !![1 / 6; -1 / 6; 1 / 3] = !![-2 / 3; 2 / 3; 2 / 3] ∧
    !![(1 : ℚ), 1, 0] * !![0; 1; 1] = !![1] ∧ !![(1 : ℚ), -1, 2] * !![0; 1; 1] = !![1] := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- `e₃ = (-1, 1, 1)/√3`: `w₃` is orthogonal to `v₁` and `w₂`, and `(-1, 1, 1)` has squared length `3`. -/
theorem gs_w3_props : !![(1 : ℚ), 1, 0] * !![-1; 1; 1] = !![0] ∧ !![(1 : ℚ), -1, 2] * !![-1; 1; 1] = !![0] ∧
    !![(-1 : ℚ), 1, 1] * !![-1; 1; 1] = !![3] := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- The orthogonal vectors span the same space as `v₁, v₂, v₃`. -/
theorem gs_same_span : Submodule.span ℚ (Set.range ![![(1 : ℚ), 1, 0], ![1, 0, 1], ![0, 1, 1]]) =
    Submodule.span ℚ (Set.range ![![(1 : ℚ), 1, 0], ![1, -1, 2], ![-1, 1, 1]]) := by
  VERIPHY[hs-gs-span]

/-! ## exr-on-legendre-p3 -/

/-- `⟪x, x³⟫ = ∫₋₁¹ x⁴ dx = 2/5`, so `w₄ = x³ - (2/5)/(2/3) x = x³ - 3x/5`. -/
theorem integral_x_four : ∫ x in (-1 : ℝ)..1, x ^ 4 = 2 / 5 := by
  VERIPHY[hs-integral]

/-- `w₄ = x³ - (3/5) x = (2/5) P₃` with `P₃ = (5x³ - 3x)/2`. -/
theorem w4_legendre (x : ℝ) : x ^ 3 - ((2 / 5) / (2 / 3)) * x = x ^ 3 - (3 / 5) * x ∧
    x ^ 3 - (3 / 5) * x = (2 / 5) * ((5 * x ^ 3 - 3 * x) / 2) := by
  constructor <;> ring
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.OrthonormalSystems.w4_legendre, hs-on-w4-test]

/-- `‖P₃‖² = ∫₋₁¹ ((5x³ - 3x)/2)² dx = 2/7`: `e₃ = √(7/2) P₃`. -/
theorem integral_P3_sq : ∫ x in (-1 : ℝ)..1, ((5 * x ^ 3 - 3 * x) / 2) ^ 2 = 2 / 7 := by
  VERIPHY[hs-integral]

/-- `P₃` is orthogonal to `1` and to `x²` (odd integrand) and to `x`: `∫₋₁¹ x P₃ = 0`. -/
theorem integral_x_P3 : ∫ x in (-1 : ℝ)..1, x * ((5 * x ^ 3 - 3 * x) / 2) = 0 := by
  VERIPHY[hs-integral]

end
end KaluzaKlein.HilbertSpaces.OrthonormalSystems
