import KaluzaKlein.PartI.HilbertSpaces.Basic

/-!
# Hilbert spaces and operator theory, section "Hilbert spaces"

The claims of the section "Hilbert spaces" of `tutorial/part-i-07-hilbert-spaces-operators.qmd`: closed subspaces
and completions, the completeness of `ℂⁿ`, of finite-dimensional spaces and of `ℓ²`, membership of `1/n` and
`1/√n` in `ℓ²`, the incomplete space of continuous functions with the `L²` norm (the sequence `fₙ = max(-1, min(1,
nx))`), the sup and `ℓ¹` norms violating the parallelogram law, and norm versus uniform convergence.
-/

namespace KaluzaKlein.HilbertSpaces.HilbertSpaces

noncomputable section

open Matrix Basic ComplexConjugate Filter Topology

/-! ## def-hilbert-space -/

/-- A closed subspace of a Hilbert space is complete, hence a Hilbert space. -/
theorem closed_subspace_complete (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]
    (K : Submodule ℂ E) (hK : IsClosed (K : Set E)) : CompleteSpace K := hK.completeSpace_coe

/-- Every inner product space has a completion: a complete inner product space in which it is dense. -/
theorem completion_exists (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] :
    CompleteSpace (UniformSpace.Completion E) ∧ DenseRange ((↑) : E → UniformSpace.Completion E) ∧
      Nonempty (InnerProductSpace ℂ (UniformSpace.Completion E)) :=
  ⟨inferInstance, UniformSpace.Completion.denseRange_coe, ⟨inferInstance⟩⟩

/-! ## exm-hs-finite -/

/-- Each coordinate is controlled by the norm: `|uₖ - vₖ| ≤ ‖u - v‖` in `ℂⁿ`. -/
theorem coord_le_norm (n : ℕ) (u v : EuclideanSpace ℂ (Fin n)) (k : Fin n) : ‖u.ofLp k - v.ofLp k‖ ≤ ‖u - v‖ := by
  have := PiLp.norm_apply_le (u - v) k
  simpa using this

/-- `ℂⁿ` is complete, and so is every finite-dimensional inner product space. -/
theorem finite_dimensional_complete (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E]
    [FiniteDimensional ℂ E] : CompleteSpace E := FiniteDimensional.complete ℂ E

/-- The `n × n` complex matrices with the trace (Frobenius) norm form a complete space. -/
theorem matrix_frobenius_complete (n : ℕ) :
    letI := Matrix.frobeniusNormedAddCommGroup (Fin n) (Fin n) ℂ
    letI := Matrix.frobeniusNormedSpace (Fin n) (Fin n) ℂ
    CompleteSpace (Matrix (Fin n) (Fin n) ℂ) := by
  letI := Matrix.frobeniusNormedAddCommGroup (Fin n) (Fin n) ℂ
  letI := Matrix.frobeniusNormedSpace (Fin n) (Fin n) ℂ
  exact FiniteDimensional.complete ℂ _

/-! ## exm-hs-l2-sequences -/

/-- `ℓ²` is a Hilbert space: an inner product space, complete, with `⟪a, b⟫ = ∑ conj aₙ bₙ` a convergent series. -/
theorem l2_hilbert (a b : lp (fun _ : ℕ => ℂ) 2) :
    CompleteSpace (lp (fun _ : ℕ => ℂ) 2) ∧ inner ℂ a b = ∑' n, conj (a n) * b n ∧
      Summable fun n => conj (a n) * b n := by
  refine ⟨inferInstance, ?_, ?_⟩
  · rw [lp.inner_eq_tsum]; rfl
  · exact lp.summable_inner a b

/-- `(1/(n + 1)^s)` is in `ℓ²` exactly when `2s > 1`: `1/n` is in `ℓ²`, `1/√n` is not. -/
theorem memℓp_one_div_rpow (s : ℝ) :
    Memℓp (fun n : ℕ => ((1 / ((n : ℝ) + 1) ^ s : ℝ) : ℂ)) 2 ↔ 1 < 2 * s := by
  rw [memℓp_gen_iff (by norm_num)]
  have h : ∀ n : ℕ, ‖((1 / ((n : ℝ) + 1) ^ s : ℝ) : ℂ)‖ ^ (2 : ℝ≥0∞).toReal = 1 / ((n + 1 : ℕ) : ℝ) ^ (2 * s) := by
    intro n
    rw [Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    simp only [ENNReal.toReal_ofNat]
    rw [Real.rpow_two, div_pow, one_pow, ← Real.rpow_natCast, ← Real.rpow_mul (by positivity)]
    push_cast
    ring_nf
  simp_rw [h]
  rw [summable_nat_add_iff 1]
  exact Real.summable_one_div_nat_rpow

/-- `∑ 1/n² = π²/6` and `∑ 1/n⁴ = π⁴/90` (`ζ(2)` and `ζ(4)`). -/
theorem zeta_two_four (k : Fin 2) :
    HasSum (fun n : ℕ => 1 / (n : ℝ) ^ (![2, 4] k : ℕ)) (![Real.pi ^ 2 / 6, Real.pi ^ 4 / 90] k) := by
  fin_cases k
  · simpa using hasSum_zeta_two
  · simpa using hasSum_zeta_four

/-! ## exm-hs-incomplete -/

/-- The continuous functions `fₙ(x) = max(-1, min(1, n x))`. -/
noncomputable def f (n : ℕ) (x : ℝ) : ℝ := max (-1) (min 1 (n * x))

/-- `fₙ` is continuous, equals `sign x` for `|x| ≥ 1/n`, and `|f_m x - f_n x| ≤ 2`. -/
theorem f_props (n : ℕ) (hn : 0 < n) (x : ℝ) :
    Continuous (f n) ∧ (1 / n ≤ |x| → f n x = Real.sign x) ∧ ∀ m, |f m x - f n x| ≤ 2 := by
  refine ⟨by unfold f; fun_prop, fun hx => ?_, fun m => ?_⟩
  · have hn' : (0 : ℝ) < n := by exact_mod_cast hn
    rcases le_or_lt 0 x with h0 | h0
    · have hx1 : 1 ≤ n * x := by
        rw [abs_of_nonneg h0] at hx
        rw [div_le_iff₀ hn'] at hx; linarith
      have hpos : 0 < x := by
        rcases eq_or_lt_of_le h0 with h | h
        · subst h; simp at hx1; linarith
        · exact h
      simp [f, Real.sign_of_pos hpos, min_eq_left hx1]
    · have hx1 : n * x ≤ -1 := by
        rw [abs_of_neg h0] at hx
        rw [div_le_iff₀ hn'] at hx; linarith
      simp [f, Real.sign_of_neg h0, min_eq_right (by linarith : n * x ≤ 1), max_eq_left hx1]
  · have h1 : ∀ k : ℕ, -1 ≤ f k x ∧ f k x ≤ 1 := fun k => ⟨le_max_left _ _, max_le (by norm_num) (min_le_left _ _)⟩
    rw [abs_le]; constructor <;> linarith [(h1 m).1, (h1 m).2, (h1 n).1, (h1 n).2]

/-- `‖f_m - fₙ‖² = ∫₋₁¹ (f_m - fₙ)² dx ≤ 8/n` for `m ≥ n ≥ 1`: a Cauchy sequence in the `L²` norm (stated; the
integrand vanishes outside `[-1/n, 1/n]` and is at most `4` there). -/
def cauchy_bound : Prop := ∀ m n : ℕ, 1 ≤ n → n ≤ m → ∫ x in (-1 : ℝ)..1, (f m x - f n x) ^ 2 ≤ 8 / n

/-- `fₙ(x) → sign x` for every `x ≠ 0`: the only candidate limit is the sign function. -/
theorem f_tendsto_sign (x : ℝ) (hx : x ≠ 0) : Tendsto (fun n : ℕ => f n x) atTop (𝓝 (Real.sign x)) := by
  apply tendsto_atTop_of_eventually_const (i₀ := Nat.ceil (1 / |x|) + 1)
  intro n hn
  have hpos : 0 < n := by omega
  refine (f_props n hpos x).2.1 ?_
  have h1 : (1 / |x| : ℝ) ≤ n := by
    calc (1 / |x| : ℝ) ≤ Nat.ceil (1 / |x|) := Nat.le_ceil _
      _ ≤ n := by exact_mod_cast (by omega : Nat.ceil (1 / |x|) ≤ n)
  have hx' : 0 < |x| := abs_pos.mpr hx
  rw [div_le_iff₀ hx'] at h1
  rw [div_le_iff₀ (by exact_mod_cast hpos)]; linarith

/-- The sign function is not continuous at `0`: for every `ε > 0` some `x` with `|x| < ε` has
`|sign x - sign 0| ≥ 1`; the limit of the `fₙ` is not in the space of continuous functions. -/
theorem sign_discontinuous (ε : ℝ) (hε : 0 < ε) : ∃ x : ℝ, |x| < ε ∧ 1 ≤ |Real.sign x - Real.sign 0| := by
  refine ⟨ε / 2, by rw [abs_of_pos (by positivity)]; linarith, ?_⟩
  rw [Real.sign_of_pos (by positivity), Real.sign_zero]
  norm_num

/-! ## exr-hs-l2-cauchy -/

/-- `|aₙ - bₙ| ≤ ‖a - b‖` in `ℓ²`: the coordinates of a Cauchy sequence are Cauchy. -/
theorem l2_coord_le (a b : lp (fun _ : ℕ => ℂ) 2) (n : ℕ) : ‖a n - b n‖ ≤ ‖a - b‖ := by
  have := lp.norm_apply_le_norm (by norm_num : (2 : ℝ≥0∞) ≠ 0) (a - b) n
  simpa using this

/-- A Cauchy sequence in `ℓ²` has Cauchy coordinate sequences. -/
theorem l2_cauchy_coords (a : ℕ → lp (fun _ : ℕ => ℂ) 2) (ha : CauchySeq a) (n : ℕ) :
    CauchySeq fun m => a m n := by
  have hl : LipschitzWith 1 fun v : lp (fun _ : ℕ => ℂ) 2 => v n := by
    apply LipschitzWith.of_dist_le_mul
    intro v w
    simp only [NNReal.coe_one, one_mul, dist_eq_norm]
    exact l2_coord_le v w n
  exact hl.uniformContinuous.comp_cauchySeq ha

/-- The limit lies in `ℓ²` and the convergence holds in norm: `ℓ²` of any family of Hilbert spaces (`ℂ` for the
sequence space) is complete. -/
theorem l2_complete (G : ℕ → Type) [∀ i, NormedAddCommGroup (G i)] [∀ i, InnerProductSpace ℂ (G i)]
    [∀ i, CompleteSpace (G i)] : CompleteSpace (lp G 2) := inferInstance

/-! ## exr-hs-sup-norm-not-hilbert -/

/-- Sup norms on `[0, b]` of `1 + x` and `1 - x` for `0 < b ≤ 2`: `‖1 + x‖_∞ = 1 + b`, `‖1 - x‖_∞ = 1`,
`‖1‖_∞ = 1`, `‖x‖_∞ = b`. -/
theorem sup_norms (b : ℝ) (hb : 0 < b) (hb2 : b ≤ 2) :
    IsGreatest ((fun x => |1 + x|) '' Set.Icc 0 b) (1 + b) ∧ IsGreatest ((fun x => |1 - x|) '' Set.Icc 0 b) 1 ∧
      IsGreatest ((fun x => |(1 : ℝ)|) '' Set.Icc 0 b) 1 ∧ IsGreatest ((fun x => |x|) '' Set.Icc 0 b) b := by
  refine ⟨⟨⟨b, ⟨hb.le, le_rfl⟩, abs_of_pos (by linarith)⟩, ?_⟩, ⟨⟨0, ⟨le_rfl, hb.le⟩, by simp⟩, ?_⟩,
    ⟨⟨0, ⟨le_rfl, hb.le⟩, by simp⟩, ?_⟩, ⟨⟨b, ⟨hb.le, le_rfl⟩, abs_of_pos hb⟩, ?_⟩⟩
  · rintro _ ⟨x, ⟨hx0, hxb⟩, rfl⟩; rw [abs_of_nonneg (by linarith)]; linarith
  · rintro _ ⟨x, ⟨hx0, hxb⟩, rfl⟩; rw [abs_le]; constructor <;> linarith
  · rintro _ ⟨x, _, rfl⟩; simp
  · rintro _ ⟨x, ⟨hx0, hxb⟩, rfl⟩; rw [abs_of_nonneg hx0]; exact hxb

/-- With these norms the parallelogram law fails for `0 < b < 2`: `(1 + b)² + 1 ≠ 2(1 + b²)` (`5 ≠ 4` for `b = 1`). -/
theorem sup_norm_parallelogram_fails (b : ℝ) (hb : 0 < b) (hb2 : b < 2) : (1 + b) ^ 2 + 1 ^ 2 ≠ 2 * (1 ^ 2 + b ^ 2) := by
  intro h; nlinarith

/-- The `ℓ¹` norm `∑ |vₖ|` on `a = (s, 0)`, `b = (0, s)`, `s ≠ 0`: `‖a + b‖₁² + ‖a - b‖₁² = 8 s² ≠ 4 s² = 2(‖a‖₁² + ‖b‖₁²)`. -/
theorem l1_parallelogram_fails (s : ℝ) (hs : s ≠ 0) :
    (∑ k, |(![s, 0] + ![0, s] : Fin 2 → ℝ) k|) ^ 2 + (∑ k, |(![s, 0] - ![0, s] : Fin 2 → ℝ) k|) ^ 2 ≠
      2 * ((∑ k, |(![s, 0] : Fin 2 → ℝ) k|) ^ 2 + (∑ k, |(![0, s] : Fin 2 → ℝ) k|) ^ 2) := by
  simp only [Fin.sum_univ_two, Pi.add_apply, Pi.sub_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, add_zero, zero_add, sub_zero, zero_sub, abs_neg, abs_zero]
  have : 0 < |s| := abs_pos.mpr hs
  intro h; nlinarith [sq_abs s]

/-- Every inner-product norm satisfies the parallelogram law; a norm violating it is not induced by an inner product
(restated here from the parallelogram law). -/
theorem inner_norm_parallelogram (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (u v : E) :
    ‖u + v‖ ^ 2 + ‖u - v‖ ^ 2 = 2 * (‖u‖ ^ 2 + ‖v‖ ^ 2) := parallelogram_law_with_norm ℂ u v

/-! ## exr-hs-l2-completeness-proof -/

/-- `∫₀^{1/n} (n x - 1)² dx = 1/(3n)` for `n ≥ 1`. -/
theorem integral_nx_sub_one_sq (n : ℕ) (hn : 0 < n) : ∫ x in (0 : ℝ)..(1 / n), (n * x - 1) ^ 2 = 1 / (3 * n) := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hF : ∀ x ∈ Set.uIcc (0 : ℝ) (1 / n), HasDerivAt (fun x : ℝ => (n * x - 1) ^ 3 / (3 * n)) ((n * x - 1) ^ 2) x := by
    intro x _
    have := ((hasDerivAt_id x).const_mul (n : ℝ)).sub_const 1 |>.pow 3 |>.div_const (3 * n)
    convert this using 1
    field_simp
    ring
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
  field_simp
  ring

/-- `‖fₙ - sign‖² = 2/(3n) → 0`: `fₙ → sign` in the `L²` norm (the value `2/(3n)` is twice the integral above, by
the symmetry `fₙ(-x) = -fₙ(x)`; stated). -/
def norm_sq_f_sub_sign : Prop := ∀ n : ℕ, 0 < n → ∫ x in (-1 : ℝ)..1, (f n x - Real.sign x) ^ 2 = 2 / (3 * n)

/-- `c/(3n) → 0` for every constant `c`, in particular `2/(3n) → 0`. -/
theorem const_div_three_n_tendsto (c : ℝ) : Tendsto (fun n : ℕ => (c / (3 * n) : ℝ)) atTop (𝓝 0) := by
  have : (fun n : ℕ => (c / (3 * n) : ℝ)) = fun n : ℕ => (c / 3) * (1 / (n : ℝ)) := by ext n; field_simp
  rw [this, ← mul_zero (c / 3 : ℝ)]
  exact tendsto_const_nhds.mul tendsto_one_div_atTop_nhds_zero_nat

/-- The convergence is not uniform: `|fₙ(x) - 1| = max 0 (1 - n x)` for `x > 0`, which tends to `1` as `x → 0⁺`, so
`sup |fₙ - sign| = 1` for every `n`. -/
theorem f_sub_sign_near_zero (n : ℕ) : (∀ x, 0 < x → |f n x - Real.sign x| = max 0 (1 - n * x)) ∧
    Tendsto (fun x : ℝ => max 0 (1 - n * x)) (𝓝[>] 0) (𝓝 1) := by
  constructor
  · intro x hx
    have hnx : 0 ≤ (n : ℝ) * x := by positivity
    rw [Real.sign_of_pos hx]
    simp only [f]
    rw [max_eq_right (by linarith [min_le_right 1 ((n : ℝ) * x), (le_min (by norm_num : (-1 : ℝ) ≤ 1) hnx : -1 ≤ min 1 (n * x))] : (-1 : ℝ) ≤ min 1 (n * x))]
    rcases le_or_lt 1 ((n : ℝ) * x) with h | h
    · rw [min_eq_left h, max_eq_left (by linarith)]; simp
    · rw [min_eq_right h.le, max_eq_right (by linarith), abs_of_nonpos (by linarith)]; ring
  · have : Tendsto (fun x : ℝ => max 0 (1 - n * x)) (𝓝 0) (𝓝 (max 0 (1 - n * 0))) :=
      (continuous_const.max (continuous_const.sub (continuous_const.mul continuous_id))).tendsto 0
    simpa using this.mono_left nhdsWithin_le_nhds

/-- A uniform limit of continuous functions is continuous; `sign` is not, so the convergence `fₙ → sign` cannot be
uniform. -/
theorem uniform_limit_continuous (F : ℕ → ℝ → ℝ) (g : ℝ → ℝ) (hF : ∀ n, Continuous (F n))
    (h : TendstoUniformly F g atTop) : Continuous g := h.continuous (Eventually.of_forall hF)

end
end KaluzaKlein.HilbertSpaces.HilbertSpaces
