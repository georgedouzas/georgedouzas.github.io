import KaluzaKlein.PartI.HilbertSpaces.OrthonormalSystems
import KaluzaKlein.PartI.HilbertSpaces.OrthonormalBases

/-!
# Hilbert spaces and operator theory, section "Worked computations" of the Hilbert-space part

The claims of `prp-hilbert-space-computations` and its six blocks: Gram-Schmidt on monomials (the Legendre
polynomials, from the section "Orthonormal systems"), the projection in the plane, the Legendre projection of `eˣ`
against its Taylor polynomial, the partial Parseval sums for `f(x) = x`, the projection onto a plane of `ℝ³` with two
bases, the error formula of a projection and the square wave.
-/

namespace KaluzaKlein.HilbertSpaces.HilbertWorked

noncomputable section

open Matrix Basic ComplexConjugate Filter Topology

/-! ## prp-hilbert-space-computations, exm-hsc-gram-schmidt-sage -/

/-- The projection of `(1, 2)` onto the line of `(2, 1)`: `⟪v₁, v₂⟫ = 4`, `⟪v₁, v₁⟫ = 5`, so the projection is
`(4/5)(2, 1) = (8/5, 4/5)`. -/
theorem plane_projection : !![(2 : ℚ), 1] * !![1; 2] = !![4] ∧ !![(2 : ℚ), 1] * !![2; 1] = !![5] ∧
    !![(4 / 5 : ℚ) * 2; (4 / 5) * 1] = !![8 / 5; 4 / 5] := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- The residual `w₂ = (1, 2) - (8/5, 4/5) = (-3/5, 6/5)` is orthogonal to the line: `⟪v₁, w₂⟫ = 0`. -/
theorem plane_residual : !![(1 : ℚ); 2] - !![8 / 5; 4 / 5] = !![-3 / 5; 6 / 5] ∧ !![(2 : ℚ), 1] * !![-3 / 5; 6 / 5] = !![0] := by
  constructor
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- `‖w₂‖² = 9/5`, and `(-1, 2)` has squared length `5`: `e₂ = w₂/‖w₂‖ = (-1, 2)/√5`, and `e₁ = (2, 1)/√5`. -/
theorem plane_normalisation : !![(-3 / 5 : ℚ), 6 / 5] * !![-3 / 5; 6 / 5] = !![9 / 5] ∧ !![(-1 : ℚ), 2] * !![-1; 2] = !![5] ∧
    !![(-3 / 5 : ℚ) * 5; (6 / 5) * 5] = !![-3; 6] := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-! ## exm-hsc-projection -/

/-- The Taylor polynomial `T₂ = 1 + x + x²/2` and the Legendre projection `P e^x = sinh 1 + 3e⁻¹ x +
(5/2)(e - 7e⁻¹)(3x² - 1)/2` of `eˣ` on `[-1, 1]`. -/
def T2 (x : ℝ) : ℝ := 1 + x + x ^ 2 / 2

/-- The Legendre projection of `eˣ` onto the quadratics. -/
def P2exp (x : ℝ) : ℝ := Real.sinh 1 + 3 * Real.exp (-1) * x + (5 / 2) * (Real.exp 1 - 7 * Real.exp (-1)) * ((3 * x ^ 2 - 1) / 2)

/-- `‖eˣ - T₂‖² ≈ 0.00899` and `‖eˣ - P eˣ‖² ≈ 0.00144`, the second about six times smaller (stated; the oracle
evaluates the integrals exactly and compares in intervals). -/
def projection_errors : Prop := ∀ k : Fin 2,
    |(∫ x in (-1 : ℝ)..1, (Real.exp x - ![T2, P2exp] k x) ^ 2) - ![0.00899, 0.00144] k| < 0.00001

/-- The coefficients of `P eˣ ≈ 0.9963 + 1.1036 x + 0.5367 x²` (stated; oracle). -/
def projection_coefficients : Prop := ∀ k : Fin 3,
    |![Real.sinh 1 - (5 / 4) * (Real.exp 1 - 7 * Real.exp (-1)), 3 * Real.exp (-1), (15 / 4) * (Real.exp 1 - 7 * Real.exp (-1))] k -
      ![0.9963, 1.1036, 0.5367] k| < 0.0001

/-- Near `x = 0` the Taylor polynomial is closer: `|eˣ - T₂(x)| ≤ |eˣ - P eˣ(x)|` for `|x| ≤ 1/10` (stated; oracle). -/
def taylor_closer_near_zero : Prop := ∀ x : ℝ, |x| ≤ 1 / 10 → |Real.exp x - T2 x| ≤ |Real.exp x - P2exp x|

/-! ## exm-hsc-parseval-rate -/

/-- The partial Parseval sums `S_N = 4π ∑_{n ≤ N} 1/n²` for `f(x) = x`. -/
def S (N : ℕ) : ℝ := 4 * Real.pi * ∑ n ∈ Finset.range N, (1 : ℝ) / (n + 1) ^ 2

/-- `S_N` increases with `N` and never exceeds `‖f‖² = 2π³/3`. -/
theorem partial_parseval_mono (N M : ℕ) (h : N ≤ M) : S N ≤ S M ∧ S M ≤ 2 * Real.pi ^ 3 / 3 := by
  constructor
  · unfold S
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono h) fun _ _ _ => by positivity
  · unfold S
    have hs := (hasSum_nat_add_iff' 1).mpr hasSum_zeta_two
    simp only [Finset.range_one, Finset.sum_singleton, Nat.cast_zero, zero_pow two_ne_zero, div_zero, sub_zero] at hs
    have := sum_le_hasSum (Finset.range M) (fun n _ => by positivity) hs
    have h2 : 4 * Real.pi * ∑ n ∈ Finset.range M, (1 : ℝ) / (n + 1) ^ 2 ≤ 4 * Real.pi * (Real.pi ^ 2 / 6) := by
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      simpa using this
    linarith

/-- `S_N → 2π³/3`: Parseval's identity for `f(x) = x`. -/
theorem partial_parseval_tendsto : Tendsto S atTop (𝓝 (2 * Real.pi ^ 3 / 3)) := by
  have h := (OrthonormalBases.parseval_x_zeta (4 * Real.pi)).tendsto_sum_nat
  have : 4 * Real.pi * (Real.pi ^ 2 / 6) = 2 * Real.pi ^ 3 / 3 := by ring
  rw [this] at h
  convert h using 1
  ext N
  simp [S, Finset.mul_sum]

/-- The values `S₁ ≈ 12.57`, `S₅ ≈ 18.39`, `S₁₀ ≈ 19.47` against `2π³/3 ≈ 20.67` (stated; oracle in intervals). -/
def partial_parseval_values : Prop := ∀ k : Fin 4,
    |![S 1, S 5, S 10, 2 * Real.pi ^ 3 / 3] k - ![12.57, 18.39, 19.47, 20.67] k| < 0.01

/-- The tail decays like `1/N`: `∑_{N < n ≤ M} 1/n² ≤ 1/N - 1/M ≤ 1/N`. -/
theorem tail_bound (N M : ℕ) (hN : 0 < N) (h : N ≤ M) : ∑ n ∈ Finset.Ioc N M, (1 : ℝ) / (n : ℝ) ^ 2 ≤ 1 / N := by
  have := sum_Ioc_inv_sq_le_sub (K := ℝ) hN.ne' h
  simp only [one_div] at this ⊢
  calc ∑ n ∈ Finset.Ioc N M, ((n : ℝ) ^ 2)⁻¹ ≤ (N : ℝ)⁻¹ - (M : ℝ)⁻¹ := this
    _ ≤ (N : ℝ)⁻¹ := by linarith [inv_nonneg.mpr (Nat.cast_nonneg (α := ℝ) M)]

/-! ## exr-hsc-projection-plane -/

/-- First basis of the plane `x + y + z = 0`: `w₁ = (1, -1, 0)`, `w₂ = (1, 0, -1) - ½ w₁ = (½, ½, -1) ∝ (1, 1, -2)`;
`⟪w₁, w₁⟫ = 2`, `⟪(1,0,-1), w₁⟫ = 1`, and `(1, 1, -2) ⊥ w₁` with `⟪(1,1,-2), (1,1,-2)⟫ = 6`. -/
theorem plane_basis_one : !![(1 : ℚ), -1, 0] * !![1; -1; 0] = !![2] ∧ !![(1 : ℚ), -1, 0] * !![1; 0; -1] = !![1] ∧
    !![(1 : ℚ); 0; -1] - !![1 / 2; -1 / 2; 0] = !![1 / 2; 1 / 2; -1] ∧ !![(1 : ℚ), -1, 0] * !![1; 1; -2] = !![0] ∧
    !![(1 : ℚ), 1, -2] * !![1; 1; -2] = !![6] := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- The projection of `u = (1, 2, 3)` with the first basis: `⟪w₁, u⟫ = -1`, `⟪(1,1,-2), u⟫ = -3`, so
`P u = (-1/2) w₁ + (-3/6)(1, 1, -2) = (-1, 0, 1)`. -/
theorem plane_projection_one : !![(1 : ℚ), -1, 0] * !![1; 2; 3] = !![-1] ∧ !![(1 : ℚ), 1, -2] * !![1; 2; 3] = !![-3] ∧
    !![(-1 / 2 : ℚ) * 1 + (-3 / 6) * 1; (-1 / 2) * (-1) + (-3 / 6) * 1; (-1 / 2) * 0 + (-3 / 6) * (-2)] = !![-1; 0; 1] := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- Second basis: `w₁' = (0, 1, -1)`, `w₂' = (1, -1, 0) + ½ w₁' = (1, -½, -½) ∝ (2, -1, -1)`: `⟪w₁', w₁'⟫ = 2`,
`⟪(1,-1,0), w₁'⟫ = -1`, `(2, -1, -1) ⊥ w₁'`, `⟪(2,-1,-1), (2,-1,-1)⟫ = 6`. -/
theorem plane_basis_two : !![(0 : ℚ), 1, -1] * !![0; 1; -1] = !![2] ∧ !![(0 : ℚ), 1, -1] * !![1; -1; 0] = !![-1] ∧
    !![(1 : ℚ); -1; 0] + !![0; 1 / 2; -1 / 2] = !![1; -1 / 2; -1 / 2] ∧ !![(0 : ℚ), 1, -1] * !![2; -1; -1] = !![0] ∧
    !![(2 : ℚ), -1, -1] * !![2; -1; -1] = !![6] := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- With the second basis: `⟪w₁', u⟫ = -1`, `⟪(2,-1,-1), u⟫ = -3`, `P u = (-1/2) w₁' + (-3/6)(2, -1, -1) = (-1, 0, 1)`:
the same vector. -/
theorem plane_projection_two : !![(0 : ℚ), 1, -1] * !![1; 2; 3] = !![-1] ∧ !![(2 : ℚ), -1, -1] * !![1; 2; 3] = !![-3] ∧
    !![(-1 / 2 : ℚ) * 0 + (-3 / 6) * 2; (-1 / 2) * 1 + (-3 / 6) * (-1); (-1 / 2) * (-1) + (-3 / 6) * (-1)] = !![-1; 0; 1] := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- Removing the normal component: `⟪(1,1,1), u⟫ = 6`, `n = (1,1,1)/√3`, `⟪n, u⟫ n = (6/3)(1, 1, 1) = (2, 2, 2)`, and
`u - (2, 2, 2) = (-1, 0, 1)`, which lies in the plane. -/
theorem plane_projection_normal : !![(1 : ℚ), 1, 1] * !![1; 2; 3] = !![6] ∧ !![(1 : ℚ); 2; 3] - !![6 / 3; 6 / 3; 6 / 3] = !![-1; 0; 1] ∧
    !![(1 : ℚ), 1, 1] * !![-1; 0; 1] = !![0] := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]
  · VERIPHY[hs-matrix]

/-- The projection onto a subspace does not depend on the orthonormal basis used: `∑ ⟪eₖ, u⟫ eₖ = ∑ ⟪fₖ, u⟫ fₖ`
for two orthonormal bases of the same span (both are the unique closest point). -/
theorem proj_basis_independent (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (n : ℕ) (e f : Fin n → E)
    (he : Orthonormal ℂ e) (hf : Orthonormal ℂ f) (hspan : Submodule.span ℂ (Set.range e) = Submodule.span ℂ (Set.range f))
    (u : E) : OrthonormalSystems.proj e u = OrthonormalSystems.proj f u := by
  have hmem : OrthonormalSystems.proj f u ∈ Submodule.span ℂ (Set.range e) := by
    rw [hspan]
    exact Submodule.sum_mem _ fun k _ => Submodule.smul_mem _ _ (Submodule.subset_span ⟨k, rfl⟩)
  have hmem' : OrthonormalSystems.proj e u ∈ Submodule.span ℂ (Set.range f) := by
    rw [← hspan]
    exact Submodule.sum_mem _ fun k _ => Submodule.smul_mem _ _ (Submodule.subset_span ⟨k, rfl⟩)
  have h1 := (OrthonormalSystems.proj_closest E n e he u _ hmem).1
  have h2 := (OrthonormalSystems.proj_closest E n f hf u _ hmem').1
  exact ((OrthonormalSystems.proj_closest E n e he u _ hmem).2 (le_antisymm h2 h1)).symm

/-! ## exr-hsc-best-approximation -/

/-- `‖u - Pₙ u‖² = ‖u‖² - ∑_{k ≤ n} |⟪eₖ, u⟫|²`. -/
theorem projection_error (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (n : ℕ) (e : Fin n → E)
    (he : Orthonormal ℂ e) (u : E) :
    ‖u - OrthonormalSystems.proj e u‖ ^ 2 = ‖u‖ ^ 2 - ∑ k, ‖inner ℂ (e k) u‖ ^ 2 := by
  have h1 := OrthonormalSystems.norm_sq_proj_add E n e he u
  have h2 := OrthonormalSystems.norm_proj_sq E n e he u
  linarith

/-- `‖eˣ‖² = ∫₋₁¹ e^{2x} dx = (e² - e⁻²)/2 = sinh 2`. -/
theorem norm_sq_exp : ∫ x in (-1 : ℝ)..1, (Real.exp x) ^ 2 = (Real.exp 2 - Real.exp (-2)) / 2 := by
  VERIPHY[hs-integral]

/-- `∫₋₁¹ P₃ eˣ dx = 37 e⁻¹ - 5 e` (with `∫ P₀ eˣ = e - e⁻¹`, `∫ P₁ eˣ = 2e⁻¹`, `∫ P₂ eˣ = e - 7e⁻¹` from the
section "Orthonormal bases"). -/
theorem integral_exp_legendre_three :
    ∫ x in (-1 : ℝ)..1, ((5 * x ^ 3 - 3 * x) / 2) * Real.exp x = 37 * Real.exp (-1) - 5 * Real.exp 1 := by
  VERIPHY[hs-integral]

/-- The numbers: `|⟪eₖ, eˣ⟫|² ≈ 2.7622, 0.8120, 0.0512, 0.00142`, `‖eˣ‖² ≈ 3.6269`, `‖eˣ - P₂eˣ‖² ≈ 0.00144`,
`‖eˣ - P₃eˣ‖² ≈ 0.0000223`, a reduction by a factor of about `65` (stated; oracle in intervals). -/
def best_approximation_values : Prop :=
  let c : Fin 4 → ℝ := ![Real.sqrt (1 / 2) * (Real.exp 1 - Real.exp (-1)), Real.sqrt (3 / 2) * (2 * Real.exp (-1)),
    Real.sqrt (5 / 2) * (Real.exp 1 - 7 * Real.exp (-1)), Real.sqrt (7 / 2) * (37 * Real.exp (-1) - 5 * Real.exp 1)]
  let N : ℝ := (Real.exp 2 - Real.exp (-2)) / 2
  (∀ k : Fin 4, |c k ^ 2 - ![2.7622, 0.8120, 0.0512, 0.00142] k| < 0.0001) ∧ |N - 3.6269| < 0.0001 ∧
    |N - (c 0 ^ 2 + c 1 ^ 2 + c 2 ^ 2) - 0.00144| < 0.00001 ∧ |N - (c 0 ^ 2 + c 1 ^ 2 + c 2 ^ 2 + c 3 ^ 2) - 0.0000223| < 0.0000005 ∧
    60 < (N - (c 0 ^ 2 + c 1 ^ 2 + c 2 ^ 2)) / (N - (c 0 ^ 2 + c 1 ^ 2 + c 2 ^ 2 + c 3 ^ 2)) ∧
    (N - (c 0 ^ 2 + c 1 ^ 2 + c 2 ^ 2)) / (N - (c 0 ^ 2 + c 1 ^ 2 + c 2 ^ 2 + c 3 ^ 2)) < 70

/-! ## exr-hsc-trig-projection -/

/-- `sign² = 1` away from `0`: `‖sign‖² = ∫₋π^π 1 dx = 2π`. -/
theorem sign_sq (x : ℝ) (hx : x ≠ 0) : Real.sign x ^ 2 = 1 := by
  rcases lt_or_gt_of_ne hx with h | h
  · rw [Real.sign_of_neg h]; norm_num
  · rw [Real.sign_of_pos h]; norm_num

/-- `∫₋π^π 1 dx = 2π`. -/
theorem integral_one_pi : ∫ x in (-Real.pi)..Real.pi, (1 : ℝ) = 2 * Real.pi := by
  VERIPHY[hs-integral]

/-- The Fourier coefficients of the square wave: `⟪eₙ, sign⟫ = (1/√(2π)) ∫₋π^π sign x e^{-inx} dx = -2i√(2/π)/n`
for odd `n` and `0` for even `n` (stated; the oracle integrates on the two half-intervals). -/
def square_wave_coefficients : Prop := ∀ n : ℤ, n ≠ 0 →
    (1 / Real.sqrt (2 * Real.pi) : ℂ) * ((∫ x in (0 : ℝ)..Real.pi, Complex.exp (-(Complex.I * n * x))) -
      ∫ x in (-Real.pi)..0, Complex.exp (-(Complex.I * n * x))) =
      if Odd n then -2 * Complex.I * Real.sqrt (2 / Real.pi) / n else 0

/-- `P_N sign = (4/π) ∑_{odd n ≤ N} sin(nx)/n` and `‖sign - P_N sign‖² = 2π - (16/π) ∑_{odd n ≤ N} 1/n²`, equal to
`1.190, 0.624, 0.421` for `N = 1, 3, 5` (stated; oracle). -/
def square_wave_errors : Prop := ∀ k : Fin 3,
    |2 * Real.pi - (16 / Real.pi) * ∑ n ∈ (Finset.range (![1, 3, 5] k + 1)).filter Odd, (1 : ℝ) / (n : ℝ) ^ 2 -
      ![1.190, 0.624, 0.421] k| < 0.001

end

end KaluzaKlein.HilbertSpaces.HilbertWorked
