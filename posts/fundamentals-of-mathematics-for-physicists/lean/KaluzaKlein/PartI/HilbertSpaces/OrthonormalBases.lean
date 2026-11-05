import KaluzaKlein.PartI.HilbertSpaces.InnerProducts

/-!
# Hilbert spaces and operator theory, section "Orthonormal bases"

The claims of the section "Orthonormal bases" of `tutorial/part-i-07-hilbert-spaces-operators.qmd`: the three
characterisations of an orthonormal basis (completeness, the expansion, Parseval) and the isomorphism with `ℓ²`,
the standard basis of `ℓ²`, the Fourier basis with the coefficients of `x` and `∑ 1/n² = π²/6`, the Legendre basis
and the expansion of `eˣ`, Parseval for `x` and `x²` (`∑ 1/n⁴ = π⁴/90`), and separability.
-/

namespace KaluzaKlein.HilbertSpaces.OrthonormalBases

noncomputable section

open Matrix Basic ComplexConjugate Filter Topology

/-! ## def-orthonormal-basis -/

/-- For a Hilbert basis every vector is the norm-convergent sum of its expansion `u = ∑ ⟪eₖ, u⟫ eₖ`. -/
theorem hilbertBasis_expansion (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]
    (ι : Type) (b : HilbertBasis ι ℂ E) (u : E) : HasSum (fun k => inner ℂ (b k) u • b k) u := b.hasSum_repr u

/-- Parseval's identity and its polarised form: `⟪u, v⟫ = ∑ conj ⟪eₖ, u⟫ ⟪eₖ, v⟫` and `‖u‖² = ∑ |⟪eₖ, u⟫|²`. -/
theorem hilbertBasis_parseval (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]
    (ι : Type) (b : HilbertBasis ι ℂ E) (u v : E) :
    HasSum (fun k => conj (inner ℂ (b k) u) * inner ℂ (b k) v) (inner ℂ u v) ∧
      HasSum (fun k => ‖inner ℂ (b k) u‖ ^ 2) (‖u‖ ^ 2) := by
  constructor
  · have := b.hasSum_inner_mul_inner u v
    convert this using 2 with k
    rw [← inner_conj_symm (b k) u]
  · have h := b.hasSum_inner_mul_inner u u
    have h2 : (fun k => inner ℂ u (b k) * inner ℂ (b k) u) = fun k => ((‖inner ℂ (b k) u‖ ^ 2 : ℝ) : ℂ) := by
      ext k
      rw [← inner_conj_symm (b k) u, Complex.conj_mul']
      push_cast; ring
    rw [h2, ← inner_self_eq_norm_sq_to_K] at h
    have h3 := (Complex.hasSum_re h)
    simpa using h3

/-- A complete orthonormal system (the only vector orthogonal to every `eₖ` is `0`) is a Hilbert basis. -/
theorem hilbertBasis_of_complete (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]
    (ι : Type) (e : ι → E) (he : Orthonormal ℂ e) (hc : ∀ u, (∀ k, inner ℂ (e k) u = 0) → u = 0) :
    ∃ b : HilbertBasis ι ℂ E, ⇑b = e := by
  refine ⟨HilbertBasis.mk he ?_, HilbertBasis.coe_mk he _⟩
  rw [← Submodule.orthogonal_eq_bot_iff, Submodule.eq_bot_iff]
  intro u hu
  refine hc u fun k => ?_
  have := (Submodule.mem_orthogonal _ u).mp hu (e k) (Submodule.subset_span ⟨k, rfl⟩)
  rwa [← inner_conj_symm, map_eq_zero] at this

/-- The coefficient map `u ↦ (⟪eₖ, u⟫)ₖ` is a linear isometric isomorphism of `H` onto `ℓ²(ι)` (which preserves
inner products). -/
theorem hilbertBasis_repr_isometry (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]
    (ι : Type) (b : HilbertBasis ι ℂ E) (u v : E) :
    (∀ k, b.repr u k = inner ℂ (b k) u) ∧ ‖b.repr u‖ = ‖u‖ ∧ inner ℂ (b.repr u) (b.repr v) = inner ℂ u v :=
  ⟨fun k => b.repr_apply_apply u k, b.repr.norm_map u, b.repr.inner_map_map u v⟩

/-- An orthonormal sequence `e₀, e₁, …` is linearly independent, so a space containing one is infinite-dimensional:
an orthonormal basis in the Hilbert sense is then not a basis in the linear-algebra sense (finite combinations do not
reach every vector). -/
theorem orthonormal_seq_infinite_dim (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (e : ℕ → E)
    (he : Orthonormal ℂ e) : LinearIndependent ℂ e ∧ ¬ FiniteDimensional ℂ E :=
  ⟨he.linearIndependent, fun _ => Module.Finite.not_linearIndependent_of_infinite e he.linearIndependent⟩

/-! ## exm-onb-l2-sequences -/

/-- The sequences `δ⁽ᵏ⁾` form an orthonormal system in `ℓ²` with `⟪δ⁽ᵏ⁾, a⟫ = aₖ`. -/
theorem l2_single_orthonormal (k : ℕ) (a : lp (fun _ : ℕ => ℂ) 2) :
    inner ℂ (lp.single 2 k (1 : ℂ)) a = a k ∧ ‖(lp.single 2 k (1 : ℂ) : lp (fun _ : ℕ => ℂ) 2)‖ = 1 := by
  constructor
  · rw [lp.inner_eq_tsum]
    rw [tsum_eq_single k]
    · simp [lp.single_apply]
    · intro j hj; simp [lp.single_apply, hj]
  · rw [lp.norm_single (by norm_num)]; simp

/-- Completeness: a sequence is orthogonal to every `δ⁽ᵏ⁾` exactly when every coordinate vanishes, that is, when it
is `0`. -/
theorem l2_single_complete (a : lp (fun _ : ℕ => ℂ) 2) : (∀ k, inner ℂ (lp.single 2 k (1 : ℂ)) a = 0) ↔ a = 0 := by
  constructor
  · intro h
    ext k
    have := (l2_single_orthonormal k a).1
    rw [h k] at this
    simp [← this]
  · intro h k
    simp [h]

/-- Parseval's identity is the definition of the norm, `‖a‖² = ∑ |aₖ|²`, and the tail `∑_{k > N} |aₖ|²` tends to
`0`, so the expansion converges in norm. -/
theorem l2_norm_sq_tsum (a : lp (fun _ : ℕ => ℂ) 2) : ‖a‖ ^ 2 = ∑' k, ‖a k‖ ^ 2 ∧
    Tendsto (fun N => ∑' k, ‖a (k + N)‖ ^ 2) atTop (𝓝 0) := by
  have hs : Summable fun k => ‖a k‖ ^ 2 := by
    have := lp.memℓp a
    rw [memℓp_gen_iff (by norm_num)] at this
    simpa using this
  constructor
  · rw [lp.norm_eq_tsum_rpow (by norm_num) a]
    simp only [ENNReal.toReal_ofNat, Real.rpow_two]
    rw [← Real.rpow_natCast, ← Real.rpow_mul (tsum_nonneg fun k => by positivity)]
    norm_num
  · exact hs.tendsto_atTop_zero_tail?

/-! ## exm-onb-fourier-parseval -/

/-- The Fourier system `e^{2πinx/T}/√T` is a Hilbert basis of `L²` of the circle of length `T` (`T = 2π` for
`L²(-π, π)`): mathlib's `fourierBasis`. -/
theorem fourier_hilbertBasis (T : ℝ) [hT : Fact (0 < T)] :
    ∃ b : HilbertBasis ℤ ℂ (MeasureTheory.Lp ℂ 2 (AddCircle.haarAddCircle (T := T))), ∀ n, b n = fourierLp 2 n :=
  ⟨fourierBasis, fun n => coe_fourierBasis n⟩

/-- The Fourier coefficients of `f(x) = x` on `(-π, π)`: `∫₋π^π x e^{-inx} dx = 2π i (-1)ⁿ / n` for `n ≠ 0` and
`0` for `n = 0` (stated; the oracle integrates exactly). -/
def fourier_coeff_x : Prop := ∀ n : ℤ, ∫ x in (-Real.pi)..Real.pi, (x : ℂ) * Complex.exp (-(Complex.I * n * x)) =
    if n = 0 then 0 else 2 * Real.pi * Complex.I * (-1) ^ n / n

/-- `‖f‖² = ∫₋π^π x² dx = 2π³/3`. -/
theorem norm_sq_x : ∫ x in (-Real.pi)..Real.pi, x ^ 2 = 2 * Real.pi ^ 3 / 3 := by
  VERIPHY[hs-integral]

/-- Parseval for `x`: `c ∑_{n ≥ 1} 1/n² = c π²/6` for every `c`; with `c = 4π` this is `2π³/3 = 4π ∑ 1/n²`, that
is, `∑ 1/n² = π²/6`. -/
theorem parseval_x_zeta (c : ℝ) : HasSum (fun n : ℕ => c * (1 / ((n + 1 : ℕ) : ℝ) ^ 2)) (c * (Real.pi ^ 2 / 6)) := by
  have h := (hasSum_nat_add_iff' 1).mpr hasSum_zeta_two
  simp only [Finset.range_one, Finset.sum_singleton, Nat.cast_zero, zero_pow two_ne_zero, div_zero, sub_zero] at h
  exact h.mul_left c

/-- `4π · π²/6 = 2π³/3 = ‖x‖²`: the constant of Parseval for `x` (and the same arithmetic for any `c`). -/
theorem parseval_x_constant (c : ℝ) : c * (Real.pi ^ 2 / 6) = c * Real.pi ^ 2 / 6 := by ring

/-! ## exm-onb-legendre -/

/-- `∫₋₁¹ eˣ dx = 2 sinh 1`: the `P₀` coefficient of `eˣ` is `sinh 1`. -/
theorem integral_exp_legendre_zero : ∫ x in (-1 : ℝ)..1, Real.exp x = Real.exp 1 - Real.exp (-1) := by
  VERIPHY[hs-integral]

/-- `∫₋₁¹ x eˣ dx = 2 e⁻¹`: the `P₁` coefficient is `(3/2)(2e⁻¹) = 3 e⁻¹`. -/
theorem integral_exp_legendre_one : ∫ x in (-1 : ℝ)..1, x * Real.exp x = 2 * Real.exp (-1) := by
  VERIPHY[hs-integral]

/-- `∫₋₁¹ P₂ eˣ dx = e - 7 e⁻¹`: the `P₂` coefficient is `(5/2)(e - 7e⁻¹)`. -/
theorem integral_exp_legendre_two :
    ∫ x in (-1 : ℝ)..1, ((3 * x ^ 2 - 1) / 2) * Real.exp x = Real.exp 1 - 7 * Real.exp (-1) := by
  VERIPHY[hs-integral]

/-- Every polynomial of degree `≤ 2` is a combination of `P₀, P₁, P₂`:
`a + b x + c x² = (a + c/3) · 1 + b · x + (2c/3) · (3x² - 1)/2`. -/
theorem degree_two_legendre_expansion (a b c x : ℝ) :
    a + b * x + c * x ^ 2 = (a + c / 3) * 1 + b * x + (2 * c / 3) * ((3 * x ^ 2 - 1) / 2) := by
  VERIPHY[hs-ideal]

/-- Polynomials are dense in the continuous functions on `[a, b]` (Weierstrass). -/
theorem polynomials_dense (a b : ℝ) : (polynomialFunctions (Set.Icc a b)).topologicalClosure = ⊤ :=
  polynomialFunctions_closure_eq_top a b

/-! ## exr-onb-parseval-x -/

/-- `|⟪eₙ, x⟫|² = 2π/n²` for `n ≠ 0`, and for `g = x²`: `⟪e₀, g⟫ = √(2π) π²/3`, `|⟪eₙ, g⟫|² = 8π/n⁴` (stated;
oracle). -/
def fourier_coeff_x_sq : Prop := ∀ n : ℤ, n ≠ 0 →
    ∫ x in (-Real.pi)..Real.pi, ((x ^ 2 : ℝ) : ℂ) * Complex.exp (-(Complex.I * n * x)) = 4 * Real.pi * (-1) ^ n / (n : ℂ) ^ 2

/-- `∑ 1/n⁴ = π⁴/90`, Parseval for `x²`; `‖g‖² = ∫₋π^π x⁴ dx = 2π⁵/5`. -/
theorem norm_sq_x_sq : ∫ x in (-Real.pi)..Real.pi, x ^ 4 = 2 * Real.pi ^ 5 / 5 := by
  VERIPHY[hs-integral]

/-- `∑ 1/n² = π²/6` and `∑ 1/n⁴ = π⁴/90` (`ζ(2)` and `ζ(4)`). -/
theorem zeta_values (k : Fin 2) :
    HasSum (fun n : ℕ => 1 / (n : ℝ) ^ (![2, 4] k : ℕ)) (![Real.pi ^ 2 / 6, Real.pi ^ 4 / 90] k) := by
  fin_cases k
  · simpa using hasSum_zeta_two
  · simpa using hasSum_zeta_four

/-- The partial sums to `N ≥ 10` terms: `∑_{n ≤ N} 1/n²` is within `10 %` of `π²/6` and `∑_{n ≤ N} 1/n⁴` within
`0.1 %` of `π⁴/90`. -/
theorem partial_sums (N : ℕ) (hN : 10 ≤ N) :
    Real.pi ^ 2 / 6 - ∑ n ∈ Finset.range N, (1 : ℝ) / (n + 1) ^ 2 < Real.pi ^ 2 / 6 / 10 ∧
      Real.pi ^ 4 / 90 - ∑ n ∈ Finset.range N, (1 : ℝ) / (n + 1) ^ 4 < Real.pi ^ 4 / 90 / 1000 := by
  have h2 : Real.pi ^ 2 < 3.141593 ^ 2 := pow_lt_pow_left Real.pi_lt_d6 Real.pi_pos.le (by norm_num)
  have h4 : Real.pi ^ 4 < 3.141593 ^ 4 := pow_lt_pow_left Real.pi_lt_d6 Real.pi_pos.le (by norm_num)
  have m2 : ∑ n ∈ Finset.range 10, (1 : ℝ) / (n + 1) ^ 2 ≤ ∑ n ∈ Finset.range N, (1 : ℝ) / (n + 1) ^ 2 :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono hN) fun _ _ _ => by positivity
  have m4 : ∑ n ∈ Finset.range 10, (1 : ℝ) / (n + 1) ^ 4 ≤ ∑ n ∈ Finset.range N, (1 : ℝ) / (n + 1) ^ 4 :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono hN) fun _ _ _ => by positivity
  simp only [Finset.sum_range_succ, Finset.sum_range_zero] at m2 m4
  norm_num at m2 m4
  constructor <;> linarith

end

end KaluzaKlein.HilbertSpaces.OrthonormalBases
