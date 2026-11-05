import KaluzaKlein.PartI.HilbertSpaces.InnerProducts

/-!
# Hilbert spaces and operator theory, section "The Riesz representation theorem"

The claims of the section "The Riesz representation theorem" of `tutorial/part-i-07-hilbert-spaces-operators.qmd`:
the theorem itself (existence, uniqueness, the norm of the representing vector, the converse), functionals on `ℂⁿ`
and on `ℓ²`, the unbounded functional `∑ aₙ`, the evaluation functional on `L²(-1, 1)`, the norm of a functional, the
proof through the orthogonal complement of the kernel, and the representation of `f ↦ ∫ x³ f` on the quadratic
polynomials.
-/

namespace KaluzaKlein.HilbertSpaces.Riesz

noncomputable section

open Matrix Basic ComplexConjugate Filter Topology

/-! ## thm-riesz-representation -/

/-- Riesz representation: every bounded linear functional `φ` on a Hilbert space is `φ(u) = ⟪v, u⟫` for a unique
`v`, and `‖φ‖ = ‖v‖`. -/
theorem riesz_representation (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]
    (φ : E →L[ℂ] ℂ) : (∃! v : E, ∀ u, φ u = inner ℂ v u) ∧ ‖(InnerProductSpace.toDual ℂ E).symm φ‖ = ‖φ‖ := by
  refine ⟨⟨(InnerProductSpace.toDual ℂ E).symm φ, fun u => ?_, fun v hv => ?_⟩, (InnerProductSpace.toDual ℂ E).symm.norm_map φ⟩
  · rw [← InnerProductSpace.toDual_apply, LinearIsometryEquiv.apply_symm_apply]
  · apply (InnerProductSpace.toDual ℂ E).injective
    rw [LinearIsometryEquiv.apply_symm_apply]
    ext u
    rw [InnerProductSpace.toDual_apply, hv u]

/-- Conversely `u ↦ ⟪v, u⟫` is bounded by `‖v‖` (Cauchy-Schwarz), and `‖v‖` is the smallest constant. -/
theorem inner_functional_bounded (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]
    (v : E) : (∀ u, ‖inner ℂ v u‖ ≤ ‖v‖ * ‖u‖) ∧ ‖InnerProductSpace.toDual ℂ E v‖ = ‖v‖ :=
  ⟨fun u => norm_inner_le_norm v u, (InnerProductSpace.toDual ℂ E).norm_map v⟩

/-- The kernel of a bounded functional is a closed subspace, and `H = K ⊕ K^⊥`. -/
theorem ker_closed_compl (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]
    (φ : E →L[ℂ] ℂ) : IsClosed (LinearMap.ker φ : Set E) ∧
      (LinearMap.ker φ ⊔ (LinearMap.ker φ)ᗮ = ⊤ ∧ LinearMap.ker φ ⊓ (LinearMap.ker φ)ᗮ = ⊥) := by
  refine ⟨φ.isClosed_ker, ?_, Submodule.inf_orthogonal_eq_bot _⟩
  exact Submodule.sup_orthogonal_of_completeSpace (K := LinearMap.ker φ)

/-! ## exm-riesz-cn -/

/-- A linear functional on `ℂⁿ` is `φ(u) = ∑ cₖ uₖ` with `cₖ = φ(eₖ)`, and `φ(u) = ⟪v, u⟫` with `vₖ = conj cₖ`. -/
theorem functional_cn (n : ℕ) (φ : EuclideanSpace ℂ (Fin n) →ₗ[ℂ] ℂ) (u : EuclideanSpace ℂ (Fin n)) :
    φ u = ∑ k, φ (EuclideanSpace.single k 1) * u.ofLp k ∧
      φ u = inner ℂ (WithLp.toLp 2 fun k => conj (φ (EuclideanSpace.single k 1))) u := by
  have h : φ u = ∑ k, φ (EuclideanSpace.single k 1) * u.ofLp k := by
    conv_lhs => rw [← (EuclideanSpace.basisFun (Fin n) ℂ).sum_repr u]
    simp only [map_sum, map_smul, EuclideanSpace.basisFun_repr, EuclideanSpace.basisFun_apply, smul_eq_mul]
    exact Finset.sum_congr rfl fun k _ => mul_comm _ _
  refine ⟨h, ?_⟩
  rw [h, EuclideanSpace.inner_eq_star_dotProduct]
  simp [dotProduct, Pi.star_apply, Complex.star_def, Complex.conj_conj, WithLp.ofLp_toLp]

/-- For `φ(u) = u₁ + i u₂` on `ℂ²` the representing vector is `v = (1, -i)`: `⟪v, u⟫ = conj 1 · u₁ + conj (-i) u₂`,
written as the row of conjugates `(1, i)` times the column `(a, b)`. -/
theorem functional_c2 (a b : ℂ) : !![(1 : ℂ), Complex.I] * !![a; b] = !![a + Complex.I * b] := by
  VERIPHY[hs-matrix]

/-- `‖v‖² = 1 + 1 = 2` for `v = (1, -i)`: the best constant is `√2`, attained at `u = v` where `φ(v) = 2`. -/
theorem functional_c2_norm : !![(1 : ℂ), Complex.I] * !![1; -Complex.I] = !![2] := by
  VERIPHY[hs-matrix]

/-! ## exm-riesz-l2 -/

/-- `φ_c(a) = ⟪c, a⟫ = ∑ conj cₙ aₙ` is bounded by `‖c‖` on `ℓ²`. -/
theorem l2_functional_bounded (c a : lp (fun _ : ℕ => ℂ) 2) :
    inner ℂ c a = ∑' n, conj (c n) * a n ∧ ‖inner ℂ c a‖ ≤ ‖c‖ * ‖a‖ :=
  ⟨by rw [lp.inner_eq_tsum]; rfl, norm_inner_le_norm c a⟩

/-- The partial sums of a bounded functional's coefficients: `∑_{n ≤ N} |cₙ|² ≤ C² ` (from
`∑_{n ≤ N} |cₙ|² = φ(∑ cₙ δ⁽ⁿ⁾) ≤ C (∑ |cₙ|²)^{1/2}`), so `c ∈ ℓ²`. -/
theorem partial_sums_bounded (C : ℝ) (hC : 0 ≤ C) (N : ℕ) (c : Fin N → ℂ)
    (h : ∑ n, ‖c n‖ ^ 2 ≤ C * Real.sqrt (∑ n, ‖c n‖ ^ 2)) : ∑ n, ‖c n‖ ^ 2 ≤ C ^ 2 := by
  set S := ∑ n, ‖c n‖ ^ 2 with hS
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun n _ => by positivity
  rcases eq_or_lt_of_le hS0 with h0 | h0
  · rw [← h0]; positivity
  · have hsq : Real.sqrt S * Real.sqrt S = S := Real.mul_self_sqrt hS0
    have hpos : 0 < Real.sqrt S := Real.sqrt_pos.mpr h0
    have : Real.sqrt S ≤ C := by
      by_contra hc
      push_neg at hc
      have := mul_lt_mul_of_pos_right hc hpos
      rw [hsq] at this
      linarith
    calc S = Real.sqrt S * Real.sqrt S := hsq.symm
      _ ≤ C * C := mul_le_mul this this hpos.le hC
      _ = C ^ 2 := by ring

/-- On `a = (1, 1/2, …, 1/N, 0, …)` the norm stays bounded: `‖a‖² = ∑_{n ≤ N} 1/n² ≤ π²/6`, while `∑ aₙ = H_N`. -/
theorem harmonic_vector_bounded (N : ℕ) : ∑ n ∈ Finset.range N, (1 : ℝ) / (n + 1) ^ 2 ≤ Real.pi ^ 2 / 6 := by
  have h := (hasSum_nat_add_iff' 1).mpr hasSum_zeta_two
  simp only [Finset.range_one, Finset.sum_singleton, Nat.cast_zero, zero_pow two_ne_zero, div_zero, sub_zero] at h
  have := sum_le_hasSum (Finset.range N) (fun n _ => by positivity) h
  simpa using this

/-- The harmonic numbers are unbounded: `H_N → ∞`. -/
theorem harmonic_tendsto_atTop (c : ℝ) :
    Tendsto (fun N => c + ∑ n ∈ Finset.range N, (1 : ℝ) / (n + 1)) atTop atTop :=
  tendsto_atTop_add_const_left _ c Real.tendsto_sum_range_one_div_nat_succ_atTop

/-! ## exm-riesz-delta -/

/-- The hat functions `fₙ(x) = max(0, 1 - n|x|)`: `fₙ(0) = 1` while `‖fₙ‖² = ∫₋₁¹ fₙ² = 2/(3n)` (stated for the
integral; the oracle integrates piecewise), so evaluation at `0` is not bounded in the `L²` norm. -/
def hat_norm_sq : Prop := ∀ n : ℕ, 0 < n →
    (max 0 (1 - n * |(0 : ℝ)|) = 1) ∧ ∫ x in (-1 : ℝ)..1, (max 0 (1 - n * |x|)) ^ 2 = 2 / (3 * n)

/-- `2/(3n) → 0`: no `C` with `|f(0)| ≤ C ‖f‖` for all continuous `f`. -/
theorem hat_norm_tendsto_zero : Tendsto (fun n : ℕ => (2 / (3 * n) : ℝ)) atTop (𝓝 0) := by
  have : (fun n : ℕ => (2 / (3 * n) : ℝ)) = fun n : ℕ => (2 / 3) * (1 / (n : ℝ)) := by ext n; field_simp
  rw [this, ← mul_zero (2 / 3 : ℝ)]
  exact tendsto_const_nhds.mul tendsto_one_div_atTop_nhds_zero_nat

/-- A functional with `|φ(fₙ)| = 1` and `‖fₙ‖ → 0` is not bounded: no `C` with `|φ f| ≤ C ‖f‖` for all `f`. -/
theorem not_bounded_of_unit_values (F : Type) [NormedAddCommGroup F] (φ : F → ℝ) (f : ℕ → F)
    (hφ : ∀ n, |φ (f n)| = 1) (hf : Tendsto (fun n => ‖f n‖) atTop (𝓝 0)) : ¬ ∃ C, ∀ g, |φ g| ≤ C * ‖g‖ := by
  rintro ⟨C, hC⟩
  have h1 : Tendsto (fun n => C * ‖f n‖) atTop (𝓝 0) := by simpa using hf.const_mul C
  have h2 : ∀ n, (1 : ℝ) ≤ C * ‖f n‖ := fun n => (hφ n) ▸ hC (f n)
  have := ge_of_tendsto h1 (Eventually.of_forall h2)
  linarith

/-- The evaluation `f ↦ f(0)` is not well defined on `L²`: changing `f` at `0` keeps the class (the point is null). -/
theorem eval_not_well_defined (f : ℝ → ℂ) (c : ℂ) :
    Function.update f 0 c =ᵐ[MeasureTheory.volume] f ∧ Function.update f 0 c 0 = c := by
  refine ⟨?_, Function.update_self 0 c f⟩
  have h : MeasureTheory.volume ({0} : Set ℝ) = 0 := MeasureTheory.measure_singleton 0
  rw [Filter.EventuallyEq, MeasureTheory.ae_iff]
  refine MeasureTheory.measure_mono_null (fun x hx => ?_) h
  by_contra hne
  exact hx (Function.update_of_ne hne c f)

/-- `f ↦ ∫₋₁⁰ f dx` is represented by the indicator function of `[-1, 0]`: `∫₋₁¹ 1_{[-1,0]} f = ∫₋₁⁰ f`. -/
theorem indicator_represents (f : ℝ → ℂ) :
    ∫ x in Set.Icc (-1 : ℝ) 1, (Set.Icc (-1 : ℝ) 0).indicator f x = ∫ x in Set.Icc (-1 : ℝ) 0, f x := by
  rw [MeasureTheory.setIntegral_indicator measurableSet_Icc, Set.Icc_inter_Icc]
  norm_num

/-! ## exr-riesz-norm -/

/-- `|⟪v, u⟫| ≤ ‖v‖ ‖u‖` with equality at `u = v`: the norm of `u ↦ ⟪v, u⟫` is exactly `‖v‖`. -/
theorem functional_norm (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (v : E) :
    (∀ u, ‖inner ℂ v u‖ ≤ ‖v‖ * ‖u‖) ∧ ‖inner ℂ v v‖ = ‖v‖ * ‖v‖ := by
  refine ⟨fun u => norm_inner_le_norm v u, ?_⟩
  rw [inner_self_eq_norm_sq_to_K, norm_pow, Complex.norm_real, Real.norm_eq_abs, abs_norm]
  ring

/-- `φ(a) = ∑ aₙ/n` is represented by `v = (1/n)` with `‖v‖² = ∑ 1/n² = π²/6`: the norm of `φ` is `π/√6`. -/
theorem harmonic_vector_norm : HasSum (fun n : ℕ => ((1 : ℝ) / (n + 1)) ^ 2) ((Real.pi / Real.sqrt 6) ^ 2) := by
  have h := (hasSum_nat_add_iff' 1).mpr hasSum_zeta_two
  simp only [Finset.range_one, Finset.sum_singleton, Nat.cast_zero, zero_pow two_ne_zero, div_zero, sub_zero] at h
  convert h using 1
  · ext n; push_cast; ring
  · rw [div_pow, Real.sq_sqrt (by norm_num)]

/-! ## exr-riesz-proof-outline -/

/-- For `w` a unit vector orthogonal to `K = ker φ` and any `u`: `φ(w) u - φ(u) w ∈ K`, hence it is orthogonal to
`w`, which gives `φ(u) = φ(w) ⟪w, u⟫ = ⟪conj φ(w) w, u⟫`. -/
theorem riesz_outline (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (φ : E →ₗ[ℂ] ℂ) (w : E)
    (hw : ‖w‖ = 1) (hK : ∀ k, φ k = 0 → inner ℂ w k = 0) (u : E) :
    φ (φ w • u - φ u • w) = 0 ∧ φ u = inner ℂ (conj (φ w) • w) u := by
  have h1 : φ (φ w • u - φ u • w) = 0 := by simp [map_sub, map_smul, mul_comm]
  refine ⟨h1, ?_⟩
  have h2 := hK _ h1
  rw [inner_sub_right, inner_smul_right, inner_smul_right, inner_self_eq_norm_sq_to_K, hw] at h2
  rw [inner_smul_left, Complex.conj_conj]
  simp at h2
  linear_combination -h2

/-- `K^⊥` is one-dimensional: every `u ∈ K^⊥` is `⟪w, u⟫ w` (`u - (φ u/φ w) w` lies in `K ∩ K^⊥ = 0`). -/
theorem ker_compl_one_dim (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (φ : E →ₗ[ℂ] ℂ) (w : E)
    (hw : ‖w‖ = 1) (hφw : φ w ≠ 0) (hwK : ∀ k, φ k = 0 → inner ℂ w k = 0) (u : E)
    (hu : ∀ k, φ k = 0 → inner ℂ u k = 0) : u = inner ℂ w u • w := by
  have hk : φ (u - (φ u / φ w) • w) = 0 := by
    simp only [map_sub, map_smul, smul_eq_mul]
    field_simp
  have h1 := hu _ hk
  have h2 := hwK _ hk
  have h3 : inner ℂ (u - (φ u / φ w) • w) (u - (φ u / φ w) • w) = 0 := by
    rw [inner_sub_left, inner_smul_left, h1, h2]; ring
  have h4 : u = (φ u / φ w) • w := sub_eq_zero.mp (inner_self_eq_zero.mp h3)
  have h5 : φ u = φ w * inner ℂ w u := by
    have := (riesz_outline E φ w hw hwK u).2
    rwa [inner_smul_left, Complex.conj_conj] at this
  rw [h5, mul_div_cancel_left₀ _ hφw] at h4
  exact h4

/-- Uniqueness: if `⟪v - v', u⟫ = 0` for all `u` then `v = v'` (take `u = v - v'`). -/
theorem representing_vector_unique (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (v v' : E)
    (h : ∀ u, inner ℂ v u = inner ℂ v' u) : v = v' := by
  have := h (v - v')
  rw [← sub_eq_zero, ← inner_self_eq_zero (𝕜 := ℂ), inner_sub_left, this, sub_self]

/-! ## exr-riesz-sage-coefficients -/

/-- `⟪x, x³⟫ = ∫₋₁¹ x⁴ dx = 2/5` and `⟪x, x⟫ = 2/3`: the projection of `x³` onto the quadratics is `(2/5)/(2/3) x =
(3/5) x`. -/
theorem x_cubed_projection_coefficients : ∫ x in (-1 : ℝ)..1, x * x ^ 3 = 2 / 5 := by
  VERIPHY[hs-integral]

/-- `⟪1, x³⟫ = 0` and `⟪x², x³⟫ = 0`: the even basis polynomials do not see `x³`. -/
theorem x_cubed_even_orthogonal : ∫ x in (-1 : ℝ)..1, x ^ 2 * x ^ 3 = 0 := by
  VERIPHY[hs-integral]

/-- The values `φ(1) = ∫ x³ = 0`, `φ(x) = ∫ x⁴ = 2/5`, `φ(x²) = ∫ x⁵ = 0` agree with `⟪(3/5) x, f⟫`:
`∫ (3/5) x · 1 = 0`, `∫ (3/5) x · x = 2/5`, `∫ (3/5) x · x² = 0`. -/
theorem representing_values : ∫ x in (-1 : ℝ)..1, (3 / 5) * x * x = 2 / 5 := by
  VERIPHY[hs-integral]

/-- `∫₋₁¹ (3/5) x · 1 dx = 0 = ∫ x³` and `∫₋₁¹ (3/5) x · x² dx = 0 = ∫ x⁵`. -/
theorem representing_values_odd : ∫ x in (-1 : ℝ)..1, (3 / 5) * x * x ^ 2 = 0 := by
  VERIPHY[hs-integral]

end

end KaluzaKlein.HilbertSpaces.Riesz
