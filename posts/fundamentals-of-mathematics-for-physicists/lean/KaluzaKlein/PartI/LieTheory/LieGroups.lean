import KaluzaKlein.PartI.LieTheory.Generators
import Veriphy.Kit.Levels

/-!
# Lie groups and Lie algebras, section "Lie groups": groups, matrix groups and subgroups

The claims of the blocks `def-lie-group` … `exr-lg-maximal-torus` of `tutorial/part-i-11-lie-groups-algebras.qmd`:
smoothness of the group operations of the circle group and of products, the Heisenberg group law, the twisted
product on a half-line, closedness of the classical matrix groups, their dimensions (as the dimensions of their Lie
algebras of matrices), the symplectic group, closed subgroups, the centre of `SU(2)` and the diagonal torus.
-/

namespace KaluzaKlein.LieTheory.LieGroups

open Matrix KaluzaKlein.LieTheory.Generators Module

/-! ## def-lie-group, exm-lie-group-u1 -/

/-- Multiplication of the circle group is smooth (of every order `n`). -/
theorem circle_mul_contMDiff (n : ℕ∞ω) :
    ContMDiff ((𝓡 1).prod (𝓡 1)) (𝓡 1) n fun p : Circle × Circle => p.1 * p.2 := contMDiff_mul

/-- Inversion of the circle group is smooth. -/
theorem circle_inv_contMDiff (n : ℕ∞ω) : ContMDiff (𝓡 1) (𝓡 1) n fun z : Circle => z⁻¹ := contMDiff_inv

/-- `e^{iα} e^{iβ} = e^{i(α + β)}`. -/
theorem circle_exp_add (a b : ℝ) : Circle.exp (a + b) = Circle.exp a * Circle.exp b := Circle.exp_add a b

/-- The circle group is abelian. -/
theorem circle_comm (z w : Circle) : z * w = w * z := mul_comm z w

/-- The tangent line `ℝ x` of a non-zero vector is one-dimensional: `u(1) = iℝ` has dimension `1`. -/
theorem tangent_line_dim (x : ℂ) (hx : x ≠ 0) : finrank ℝ (ℝ ∙ x) = 1 := finrank_span_singleton hx

/-- `SO(2)` is the same Lie group: plane rotations compose by adding angles. -/
theorem rot2_mul (s t : ℝ) : rot2 s * rot2 t = rot2 (s + t) := by
  rw [show rot2 (s + t) = !![Real.cos s * Real.cos t - Real.sin s * Real.sin t,
      -(Real.sin s * Real.cos t + Real.cos s * Real.sin t);
      Real.sin s * Real.cos t + Real.cos s * Real.sin t, Real.cos s * Real.cos t - Real.sin s * Real.sin t] by
    ext i j; fin_cases i <;> fin_cases j <;> simp [rot2, Real.cos_add, Real.sin_add]]
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ,
            KaluzaKlein.LieTheory.Generators.s1, KaluzaKlein.LieTheory.Generators.s2,
            KaluzaKlein.LieTheory.Generators.s3, KaluzaKlein.LieTheory.Generators.T0,
            KaluzaKlein.LieTheory.Generators.T1, KaluzaKlein.LieTheory.Generators.T2,
            KaluzaKlein.LieTheory.Generators.Tv, KaluzaKlein.LieTheory.Generators.qi,
            KaluzaKlein.LieTheory.Generators.qj, KaluzaKlein.LieTheory.Generators.qk,
            KaluzaKlein.LieTheory.Generators.S1, KaluzaKlein.LieTheory.Generators.S2,
            KaluzaKlein.LieTheory.Generators.S3, KaluzaKlein.LieTheory.Generators.su2El,
            KaluzaKlein.LieTheory.Generators.L0R, KaluzaKlein.LieTheory.Generators.L1R,
            KaluzaKlein.LieTheory.Generators.nL, KaluzaKlein.LieTheory.Generators.J3v,
            KaluzaKlein.LieTheory.Generators.Jpv, KaluzaKlein.LieTheory.Generators.Jmv,
            KaluzaKlein.LieTheory.Generators.J1v, KaluzaKlein.LieTheory.Generators.J2v,
            KaluzaKlein.LieTheory.Generators.rotX, KaluzaKlein.LieTheory.Generators.rotY,
            KaluzaKlein.LieTheory.Generators.rot2, KaluzaKlein.LieTheory.Generators.H,
            KaluzaKlein.LieTheory.Generators.E, KaluzaKlein.LieTheory.Generators.F, KaluzaKlein.LieTheory.Generators.HR,
            KaluzaKlein.LieTheory.Generators.ER, KaluzaKlein.LieTheory.Generators.FR,
            KaluzaKlein.LieTheory.Generators.E12, KaluzaKlein.LieTheory.Generators.E23,
            KaluzaKlein.LieTheory.Generators.E13, KaluzaKlein.LieTheory.Generators.heis,
            KaluzaKlein.LieTheory.Generators.heisAlg, KaluzaKlein.LieTheory.Generators.affX,
            KaluzaKlein.LieTheory.Generators.affY, KaluzaKlein.LieTheory.Generators.aff,
            KaluzaKlein.LieTheory.Generators.affAlg, KaluzaKlein.LieTheory.Generators.boost,
            KaluzaKlein.LieTheory.Generators.eta2, KaluzaKlein.LieTheory.Generators.Jsym,
            KaluzaKlein.LieTheory.Generators.expT3, KaluzaKlein.LieTheory.Generators.expT1,
            KaluzaKlein.LieTheory.Generators.expT2, KaluzaKlein.LieTheory.Generators.euler,
            KaluzaKlein.LieTheory.Generators.su2Q, KaluzaKlein.LieTheory.Generators.gm1,
            KaluzaKlein.LieTheory.Generators.gm2, KaluzaKlein.LieTheory.Generators.gm3,
            KaluzaKlein.LieTheory.Generators.gm4, KaluzaKlein.LieTheory.Generators.gm5,
            KaluzaKlein.LieTheory.Generators.gm6, KaluzaKlein.LieTheory.Generators.gm7,
            KaluzaKlein.LieTheory.Generators.gm8s, KaluzaKlein.LieTheory.Generators.E12c,
            KaluzaKlein.LieTheory.Generators.E13c, KaluzaKlein.LieTheory.Generators.E23c, KaluzaKlein.rotation,
            KaluzaKlein.L2R, KaluzaKlein.L0, KaluzaKlein.L1, KaluzaKlein.L2, KaluzaKlein.L,
            KaluzaKlein.Representations.J, KaluzaKlein.Representations.K, KaluzaKlein.M4]
        all_goals ring1))
  -- certificate: lt-matrix.sage

/-! ## exm-lie-group-su2 -/

/-- Inversion in `SU(2)` is the adjoint: `g g† = (a₀² + a₁² + a₂² + a₃²) 1` for `g = a₀ + 2 a·T`. -/
theorem su2Q_mul_adj (a0 a1 a2 a3 : ℂ) :
    su2Q a0 a1 a2 a3 * su2Q a0 (-a1) (-a2) (-a3) = (a0 ^ 2 + a1 ^ 2 + a2 ^ 2 + a3 ^ 2) • (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  VERIPHY[lt-matrix-c]

/-- `det (a₀ + 2 a·T) = a₀² + a₁² + a₂² + a₃²`: `SU(2)` is the unit sphere `S³`. -/
theorem su2Q_det (a0 a1 a2 a3 : ℂ) : (su2Q a0 a1 a2 a3).det = a0 ^ 2 + a1 ^ 2 + a2 ^ 2 + a3 ^ 2 := by
  VERIPHY[lt-matrix-c]

/-- The abelian group `ℝⁿ` has dimension `n` (as does the torus `Tⁿ`, with Lie algebra `ℝⁿ`). -/
theorem finrank_Rn (n : ℕ) : finrank ℝ (Fin n → ℝ) = n := by simp

/-- The torus `T² = U(1) × U(1)` is abelian. -/
theorem torus_comm (z w : Circle × Circle) : z * w = w * z := mul_comm z w

/-! ## exm-lie-group-so3 -/

/-- `Ad_{-g} = Ad_g`: `±g` act alike, so `SO(3) = SU(2)/{±1}` identifies antipodal points of `S³`. -/
theorem conj_neg (g X : Matrix (Fin 2) (Fin 2) ℂ) (hg : IsUnit g.det) : (-g) * X * (-g)⁻¹ = g * X * g⁻¹ := by
  have h : (-g)⁻¹ = -g⁻¹ := Matrix.inv_eq_right_inv (by rw [Matrix.neg_mul, Matrix.mul_neg, neg_neg, Matrix.mul_nonsing_inv g hg])
  rw [h, Matrix.neg_mul, Matrix.mul_neg, neg_neg]

/-- Finite groups are zero-dimensional Lie groups: `S₃` has `3! = 6` elements. -/
theorem card_perm (n : ℕ) : Fintype.card (Equiv.Perm (Fin n)) = n.factorial := by rw [Fintype.card_perm, Fintype.card_fin]

/-! ## exr-lg-heisenberg-group -/

/-- `h(x₁, y₁, z₁) h(x₂, y₂, z₂) = h(x₁ + x₂, y₁ + y₂, z₁ + z₂ + x₁ y₂)`. -/
theorem heis_mul (x1 y1 z1 x2 y2 z2 : ℝ) :
    heis x1 y1 z1 * heis x2 y2 z2 = heis (x1 + x2) (y1 + y2) (z1 + z2 + x1 * y2) := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ,
            KaluzaKlein.LieTheory.Generators.s1, KaluzaKlein.LieTheory.Generators.s2,
            KaluzaKlein.LieTheory.Generators.s3, KaluzaKlein.LieTheory.Generators.T0,
            KaluzaKlein.LieTheory.Generators.T1, KaluzaKlein.LieTheory.Generators.T2,
            KaluzaKlein.LieTheory.Generators.Tv, KaluzaKlein.LieTheory.Generators.qi,
            KaluzaKlein.LieTheory.Generators.qj, KaluzaKlein.LieTheory.Generators.qk,
            KaluzaKlein.LieTheory.Generators.S1, KaluzaKlein.LieTheory.Generators.S2,
            KaluzaKlein.LieTheory.Generators.S3, KaluzaKlein.LieTheory.Generators.su2El,
            KaluzaKlein.LieTheory.Generators.L0R, KaluzaKlein.LieTheory.Generators.L1R,
            KaluzaKlein.LieTheory.Generators.nL, KaluzaKlein.LieTheory.Generators.J3v,
            KaluzaKlein.LieTheory.Generators.Jpv, KaluzaKlein.LieTheory.Generators.Jmv,
            KaluzaKlein.LieTheory.Generators.J1v, KaluzaKlein.LieTheory.Generators.J2v,
            KaluzaKlein.LieTheory.Generators.rotX, KaluzaKlein.LieTheory.Generators.rotY,
            KaluzaKlein.LieTheory.Generators.rot2, KaluzaKlein.LieTheory.Generators.H,
            KaluzaKlein.LieTheory.Generators.E, KaluzaKlein.LieTheory.Generators.F, KaluzaKlein.LieTheory.Generators.HR,
            KaluzaKlein.LieTheory.Generators.ER, KaluzaKlein.LieTheory.Generators.FR,
            KaluzaKlein.LieTheory.Generators.E12, KaluzaKlein.LieTheory.Generators.E23,
            KaluzaKlein.LieTheory.Generators.E13, KaluzaKlein.LieTheory.Generators.heis,
            KaluzaKlein.LieTheory.Generators.heisAlg, KaluzaKlein.LieTheory.Generators.affX,
            KaluzaKlein.LieTheory.Generators.affY, KaluzaKlein.LieTheory.Generators.aff,
            KaluzaKlein.LieTheory.Generators.affAlg, KaluzaKlein.LieTheory.Generators.boost,
            KaluzaKlein.LieTheory.Generators.eta2, KaluzaKlein.LieTheory.Generators.Jsym,
            KaluzaKlein.LieTheory.Generators.expT3, KaluzaKlein.LieTheory.Generators.expT1,
            KaluzaKlein.LieTheory.Generators.expT2, KaluzaKlein.LieTheory.Generators.euler,
            KaluzaKlein.LieTheory.Generators.su2Q, KaluzaKlein.LieTheory.Generators.gm1,
            KaluzaKlein.LieTheory.Generators.gm2, KaluzaKlein.LieTheory.Generators.gm3,
            KaluzaKlein.LieTheory.Generators.gm4, KaluzaKlein.LieTheory.Generators.gm5,
            KaluzaKlein.LieTheory.Generators.gm6, KaluzaKlein.LieTheory.Generators.gm7,
            KaluzaKlein.LieTheory.Generators.gm8s, KaluzaKlein.LieTheory.Generators.E12c,
            KaluzaKlein.LieTheory.Generators.E13c, KaluzaKlein.LieTheory.Generators.E23c, KaluzaKlein.rotation,
            KaluzaKlein.L2R, KaluzaKlein.L0, KaluzaKlein.L1, KaluzaKlein.L2, KaluzaKlein.L,
            KaluzaKlein.Representations.J, KaluzaKlein.Representations.K, KaluzaKlein.M4]
        all_goals ring1))
  -- certificate: lt-matrix.sage

/-- `h(x, y, z)⁻¹ = h(−x, −y, xy − z)`. -/
theorem heis_inv (x y z : ℝ) : heis x y z * heis (-x) (-y) (x * y - z) = 1 := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ,
            KaluzaKlein.LieTheory.Generators.s1, KaluzaKlein.LieTheory.Generators.s2,
            KaluzaKlein.LieTheory.Generators.s3, KaluzaKlein.LieTheory.Generators.T0,
            KaluzaKlein.LieTheory.Generators.T1, KaluzaKlein.LieTheory.Generators.T2,
            KaluzaKlein.LieTheory.Generators.Tv, KaluzaKlein.LieTheory.Generators.qi,
            KaluzaKlein.LieTheory.Generators.qj, KaluzaKlein.LieTheory.Generators.qk,
            KaluzaKlein.LieTheory.Generators.S1, KaluzaKlein.LieTheory.Generators.S2,
            KaluzaKlein.LieTheory.Generators.S3, KaluzaKlein.LieTheory.Generators.su2El,
            KaluzaKlein.LieTheory.Generators.L0R, KaluzaKlein.LieTheory.Generators.L1R,
            KaluzaKlein.LieTheory.Generators.nL, KaluzaKlein.LieTheory.Generators.J3v,
            KaluzaKlein.LieTheory.Generators.Jpv, KaluzaKlein.LieTheory.Generators.Jmv,
            KaluzaKlein.LieTheory.Generators.J1v, KaluzaKlein.LieTheory.Generators.J2v,
            KaluzaKlein.LieTheory.Generators.rotX, KaluzaKlein.LieTheory.Generators.rotY,
            KaluzaKlein.LieTheory.Generators.rot2, KaluzaKlein.LieTheory.Generators.H,
            KaluzaKlein.LieTheory.Generators.E, KaluzaKlein.LieTheory.Generators.F, KaluzaKlein.LieTheory.Generators.HR,
            KaluzaKlein.LieTheory.Generators.ER, KaluzaKlein.LieTheory.Generators.FR,
            KaluzaKlein.LieTheory.Generators.E12, KaluzaKlein.LieTheory.Generators.E23,
            KaluzaKlein.LieTheory.Generators.E13, KaluzaKlein.LieTheory.Generators.heis,
            KaluzaKlein.LieTheory.Generators.heisAlg, KaluzaKlein.LieTheory.Generators.affX,
            KaluzaKlein.LieTheory.Generators.affY, KaluzaKlein.LieTheory.Generators.aff,
            KaluzaKlein.LieTheory.Generators.affAlg, KaluzaKlein.LieTheory.Generators.boost,
            KaluzaKlein.LieTheory.Generators.eta2, KaluzaKlein.LieTheory.Generators.Jsym,
            KaluzaKlein.LieTheory.Generators.expT3, KaluzaKlein.LieTheory.Generators.expT1,
            KaluzaKlein.LieTheory.Generators.expT2, KaluzaKlein.LieTheory.Generators.euler,
            KaluzaKlein.LieTheory.Generators.su2Q, KaluzaKlein.LieTheory.Generators.gm1,
            KaluzaKlein.LieTheory.Generators.gm2, KaluzaKlein.LieTheory.Generators.gm3,
            KaluzaKlein.LieTheory.Generators.gm4, KaluzaKlein.LieTheory.Generators.gm5,
            KaluzaKlein.LieTheory.Generators.gm6, KaluzaKlein.LieTheory.Generators.gm7,
            KaluzaKlein.LieTheory.Generators.gm8s, KaluzaKlein.LieTheory.Generators.E12c,
            KaluzaKlein.LieTheory.Generators.E13c, KaluzaKlein.LieTheory.Generators.E23c, KaluzaKlein.rotation,
            KaluzaKlein.L2R, KaluzaKlein.L0, KaluzaKlein.L1, KaluzaKlein.L2, KaluzaKlein.L,
            KaluzaKlein.Representations.J, KaluzaKlein.Representations.K, KaluzaKlein.M4]
        all_goals ring1))
  -- certificate: lt-matrix.sage

/-- The commutator `g₁ g₂ − g₂ g₁` has the single entry `x₁ y₂ − x₂ y₁` in the corner. -/
theorem heis_commutator (x1 y1 z1 x2 y2 z2 : ℝ) :
    heis x1 y1 z1 * heis x2 y2 z2 - heis x2 y2 z2 * heis x1 y1 z1 = !![0, 0, x1 * y2 - x2 * y1; 0, 0, 0; 0, 0, 0] := by
  VERIPHY[lt-matrix]

/-- The Heisenberg group is not abelian: `h(1, 0, 0)` and `h(0, 1, 0)` do not commute. -/
theorem heis_not_comm (x1 y1 x2 y2 : ℝ) (h : x1 * y2 ≠ x2 * y1) :
    heis x1 y1 0 * heis x2 y2 0 ≠ heis x2 y2 0 * heis x1 y1 0 := by
  intro he
  have := congrFun (congrFun he 0) 2
  simp [heis, Matrix.mul_apply, Fin.sum_univ_succ] at this
  exact h (by linarith)

/-- The coordinates `(x, y, z)` form a global chart: `h` is injective. -/
theorem heis_injective (x y z x' y' z' : ℝ) (h : heis x y z = heis x' y' z') : x = x' ∧ y = y' ∧ z = z' := by
  refine ⟨?_, ?_, ?_⟩
  · simpa [heis] using congrFun (congrFun h 0) 1
  · simpa [heis] using congrFun (congrFun h 1) 2
  · simpa [heis] using congrFun (congrFun h 0) 2

/-! ## exr-lg-product-group -/

/-- `G × H` is a Lie group when `G` and `H` are: the torus `U(1) × U(1)`. -/
theorem torus_lieGroup (n : ℕ∞ω) : LieGroup ((𝓡 1).prod (𝓡 1)) n (Circle × Circle) := inferInstance

/-- `dim (G × H) = dim G + dim H` for the model spaces `ℝᵐ × ℝⁿ`: `SU(2) × U(1)` has dimension `3 + 1 = 4`. -/
theorem finrank_prod_fin (m n : ℕ) : finrank ℝ ((Fin m → ℝ) × (Fin n → ℝ)) = m + n := by
  simp [Module.finrank_prod]

/-- The map `(g, e^{iα}) ↦ e^{iα} g` onto `U(2)` is two-to-one: `(−1, −1)` also goes to `1`. -/
theorem u2_cover_kernel : (-1 : ℂ) • (-(1 : Matrix (Fin 2) (Fin 2) ℂ)) = 1 := by
  veriphy_certified
    ( norm_num [KaluzaKlein.LieTheory.Generators.s1, KaluzaKlein.LieTheory.Generators.s2,
        KaluzaKlein.LieTheory.Generators.s3, KaluzaKlein.LieTheory.Generators.T0, KaluzaKlein.LieTheory.Generators.T1,
        KaluzaKlein.LieTheory.Generators.T2, KaluzaKlein.LieTheory.Generators.Tv, KaluzaKlein.LieTheory.Generators.qi,
        KaluzaKlein.LieTheory.Generators.qj, KaluzaKlein.LieTheory.Generators.qk, KaluzaKlein.LieTheory.Generators.S1,
        KaluzaKlein.LieTheory.Generators.S2, KaluzaKlein.LieTheory.Generators.S3,
        KaluzaKlein.LieTheory.Generators.su2El, KaluzaKlein.LieTheory.Generators.L0R,
        KaluzaKlein.LieTheory.Generators.L1R, KaluzaKlein.LieTheory.Generators.nL, KaluzaKlein.LieTheory.Generators.J3v,
        KaluzaKlein.LieTheory.Generators.Jpv, KaluzaKlein.LieTheory.Generators.Jmv,
        KaluzaKlein.LieTheory.Generators.J1v, KaluzaKlein.LieTheory.Generators.J2v,
        KaluzaKlein.LieTheory.Generators.rotX, KaluzaKlein.LieTheory.Generators.rotY,
        KaluzaKlein.LieTheory.Generators.rot2, KaluzaKlein.LieTheory.Generators.H, KaluzaKlein.LieTheory.Generators.E,
        KaluzaKlein.LieTheory.Generators.F, KaluzaKlein.LieTheory.Generators.HR, KaluzaKlein.LieTheory.Generators.ER,
        KaluzaKlein.LieTheory.Generators.FR, KaluzaKlein.LieTheory.Generators.E12, KaluzaKlein.LieTheory.Generators.E23,
        KaluzaKlein.LieTheory.Generators.E13, KaluzaKlein.LieTheory.Generators.heis,
        KaluzaKlein.LieTheory.Generators.heisAlg, KaluzaKlein.LieTheory.Generators.affX,
        KaluzaKlein.LieTheory.Generators.affY, KaluzaKlein.LieTheory.Generators.aff,
        KaluzaKlein.LieTheory.Generators.affAlg, KaluzaKlein.LieTheory.Generators.boost,
        KaluzaKlein.LieTheory.Generators.eta2, KaluzaKlein.LieTheory.Generators.Jsym,
        KaluzaKlein.LieTheory.Generators.expT3, KaluzaKlein.LieTheory.Generators.expT1,
        KaluzaKlein.LieTheory.Generators.expT2, KaluzaKlein.LieTheory.Generators.euler,
        KaluzaKlein.LieTheory.Generators.su2Q, KaluzaKlein.LieTheory.Generators.gm1,
        KaluzaKlein.LieTheory.Generators.gm2, KaluzaKlein.LieTheory.Generators.gm3,
        KaluzaKlein.LieTheory.Generators.gm4, KaluzaKlein.LieTheory.Generators.gm5,
        KaluzaKlein.LieTheory.Generators.gm6, KaluzaKlein.LieTheory.Generators.gm7,
        KaluzaKlein.LieTheory.Generators.gm8s, KaluzaKlein.LieTheory.Generators.E12c,
        KaluzaKlein.LieTheory.Generators.E13c, KaluzaKlein.LieTheory.Generators.E23c, KaluzaKlein.rotation,
        KaluzaKlein.L2R, KaluzaKlein.L0, KaluzaKlein.L1, KaluzaKlein.L2, KaluzaKlein.L, KaluzaKlein.Representations.J,
        KaluzaKlein.Representations.K, KaluzaKlein.M4, Matrix.det_fin_two_of, Matrix.det_fin_three,
        Matrix.trace_fin_two, Matrix.trace_fin_three, Matrix.mul_apply, Fin.sum_univ_succ, Complex.ext_iff]
      all_goals ring1)
  -- certificate: lt-matrix-c.sage

/-! ## exr-lg-twisted-line -/

/-- The twisted product `x * y = x + y + xy` on `(−1, ∞)`. -/
def twist (x y : ℝ) : ℝ := x + y + x * y

/-- `φ(x) = 1 + x` turns the twisted product into the ordinary one. -/
theorem twist_phi (x y : ℝ) : 1 + twist x y = (1 + x) * (1 + y) := by
  unfold twist
  veriphy_certified (ring1)
  -- certificate: lt-ring.sage

/-- The twisted product is associative. -/
theorem twist_assoc (x y z : ℝ) : twist (twist x y) z = twist x (twist y z) := by
  unfold twist
  veriphy_certified (ring1)
  -- certificate: lt-ring.sage

/-- `0` is the identity. -/
theorem twist_zero_right (x : ℝ) : twist x 0 = x := by
  unfold twist
  veriphy_certified (ring1)
  -- certificate: lt-ring.sage
theorem twist_zero_left (x : ℝ) : twist 0 x = x := by
  unfold twist
  VERIPHY[lt-ring]

/-- The inverse of `x` is `−x / (1 + x)`. -/
theorem twist_inv (x : ℝ) (hx : 1 + x ≠ 0) : twist x (-x / (1 + x)) = 0 := by
  unfold twist; field_simp; ring

/-- `1 + x > 0` is preserved: the half-line is closed under the product. -/
theorem twist_pos (x y : ℝ) (hx : -1 < x) (hy : -1 < y) : -1 < twist x y := by
  unfold twist; nlinarith

/-- `log (1 + x)` turns the twisted product into addition: `(M, *) ≅ (ℝ, +)`. -/
theorem twist_log (x y : ℝ) (hx : -1 < x) (hy : -1 < y) :
    Real.log (1 + twist x y) = Real.log (1 + x) + Real.log (1 + y) := by
  rw [twist_phi]
  exact Real.log_mul (by linarith) (by linarith)

/-! ## def-matrix-lie-group, exm-matrix-gl, exm-matrix-orthogonal, exm-matrix-unitary -/

/-- `O(n) = {R : Rᵀ R = 1}` is a closed subset of the matrices. -/
theorem orthogonal_isClosed (n : ℕ) : IsClosed {R : Matrix (Fin n) (Fin n) ℝ | Rᵀ * R = 1} :=
  isClosed_eq (continuous_id.matrix_transpose.matrix_mul continuous_id) continuous_const

/-- `U(n) = {U : U† U = 1}` is closed. -/
theorem unitary_isClosed (n : ℕ) : IsClosed {U : Matrix (Fin n) (Fin n) ℂ | Uᴴ * U = 1} :=
  isClosed_eq (continuous_id.matrix_conjTranspose.matrix_mul continuous_id) continuous_const

/-- `SL(n) = {det = 1}` is closed. -/
theorem specialLinear_isClosed (n : ℕ) : IsClosed {M : Matrix (Fin n) (Fin n) ℝ | M.det = 1} :=
  isClosed_eq continuous_id.matrix_det continuous_const

/-- `O(n)` is a subgroup: closed under products. -/
theorem orthogonal_mul (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ) (hA : Aᵀ * A = 1) (hB : Bᵀ * B = 1) :
    (A * B)ᵀ * (A * B) = 1 := by
  rw [Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Aᵀ, hA, Matrix.one_mul, hB]

/-- `O(n)` is closed under inversion: `Rᵀ` is the inverse, and it is again orthogonal. -/
theorem orthogonal_inv (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (hA : Aᵀ * A = 1) : A * Aᵀ = 1 ∧ (Aᵀ)ᵀ * Aᵀ = 1 := by
  have h : A * Aᵀ = 1 := Matrix.mul_eq_one_comm.mp hA
  exact ⟨h, by simpa using h⟩

/-- `GL(n, ℝ)` is open: the complement of the closed set `det = 0`. -/
theorem generalLinear_isOpen (n : ℕ) : IsOpen {M : Matrix (Fin n) (Fin n) ℝ | M.det ≠ 0} :=
  isOpen_ne_fun continuous_id.matrix_det continuous_const

/-- `dim GL(n, ℝ) = n²`: the matrices form a space of dimension `n²`. -/
theorem finrank_matrices (n : ℕ) : finrank ℝ (Matrix (Fin n) (Fin n) ℝ) = n ^ 2 := by
  simp [Module.finrank_matrix, sq]

/-- `GL(n, ℂ)` has real dimension `2 n²`. -/
theorem finrank_complex_matrices (n : ℕ) : finrank ℝ (Matrix (Fin n) (Fin n) ℂ) = 2 * n ^ 2 := by
  rw [Module.finrank_matrix, Complex.finrank_real_complex]; simp [sq]; ring

/-- An orthogonal matrix has `det R = ±1`: `SO(n)` is the component `det = 1`. -/
theorem orthogonal_det_sq (n : ℕ) (R : Matrix (Fin n) (Fin n) ℝ) (hR : Rᵀ * R = 1) : R.det ^ 2 = 1 := by
  have := congrArg Matrix.det hR
  rwa [Matrix.det_mul, Matrix.det_transpose, ← sq, Matrix.det_one] at this

/-- The rotations about the third axis have determinant one. -/
theorem rotation_det (t : ℝ) : (rotation t).det = 1 := by
  VERIPHY[lt-matrix-trig]

/-- The skew-Hermitian matrices, the Lie algebra `u(n)`, as a real subspace. -/
def skewHermitian (n : ℕ) : Submodule ℝ (Matrix (Fin n) (Fin n) ℂ) where
  carrier := {X | Xᴴ = -X}
  add_mem' {X Y} hX hY := by
    simp only [Set.mem_setOf_eq] at *
    rw [Matrix.conjTranspose_add, hX, hY, neg_add]
  zero_mem' := by simp
  smul_mem' c {X} hX := by
    simp only [Set.mem_setOf_eq] at *
    rw [Matrix.conjTranspose_smul, hX, smul_neg]
    congr 1
    ext i j
    simp [Matrix.smul_apply]

/-- The traceless skew-Hermitian matrices, the Lie algebra `su(n)`. -/
def skewHermitianTraceless (n : ℕ) : Submodule ℝ (Matrix (Fin n) (Fin n) ℂ) :=
  skewHermitian n ⊓ LinearMap.ker ((Matrix.traceLinearMap (Fin n) ℂ ℂ).restrictScalars ℝ)

/-- `dim U(n) = n²`: the dimension formula for the skew-Hermitian matrices. -/
def uDimFormula (n : ℕ) : Prop := finrank ℝ (skewHermitian n) = n ^ 2

/-- `dim SU(n) = n² − 1`: the dimension formula for the traceless skew-Hermitian matrices. -/
def suDimFormula (n : ℕ) : Prop := finrank ℝ (skewHermitianTraceless n) = n ^ 2 - 1

/-! ## exr-dimension-count -/

/-- `Rᵀ R = 1` is a symmetric equation: `n (n + 1) / 2` conditions on `n²` entries leave `n (n − 1) / 2`. -/
theorem orthogonal_count (n : ℕ) :
    finrank ℝ (KaluzaKlein.LinearAlgebra.VectorSpaces.symMatrices n) +
      finrank ℝ (KaluzaKlein.LinearAlgebra.VectorSpaces.antisymMatrices n) = n ^ 2 :=
  KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym_add_antisym n

/-! ## exr-lg-sl2-dimension -/

/-- `det` is multiplicative, so `SL(2, ℝ)` is a subgroup. -/
theorem det_mul_sl (A B : Matrix (Fin 2) (Fin 2) ℝ) (hA : A.det = 1) (hB : B.det = 1) : (A * B).det = 1 := by
  rw [Matrix.det_mul, hA, hB, one_mul]

/-- The gradient of `det = ad − bc` is `(d, −c, −b, a)`: the four partial derivatives. -/
theorem det_grad_a (a b c d : ℝ) : deriv (fun x : ℝ => x * d - b * c) a = d := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x * d - b * c) ((d : ℝ)) (a) :=
        by
        convert (((hasDerivAt_id' (a : ℝ)).const_mul (d : ℝ)).add (hasDerivAt_const (a : ℝ) (-b * c : ℝ))) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: lt-deriv.sage
theorem det_grad_b (a b c d : ℝ) : deriv (fun x : ℝ => a * d - x * c) b = -c := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => a * d - x * c) ((-c : ℝ)) (b) :=
        by
        convert (((hasDerivAt_id' (b : ℝ)).const_mul (-c : ℝ)).add (hasDerivAt_const (b : ℝ) (a * d : ℝ))) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: lt-deriv.sage
theorem det_grad_c (a b c d : ℝ) : deriv (fun x : ℝ => a * d - b * x) c = -b := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => a * d - b * x) ((-b : ℝ)) (c) :=
        by
        convert (((hasDerivAt_id' (c : ℝ)).const_mul (-b : ℝ)).add (hasDerivAt_const (c : ℝ) (a * d : ℝ))) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: lt-deriv.sage
theorem det_grad_d (a b c d : ℝ) : deriv (fun x : ℝ => a * x - b * c) d = a := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => a * x - b * c) ((a : ℝ)) (d) :=
        by
        convert (((hasDerivAt_id' (d : ℝ)).const_mul (a : ℝ)).add (hasDerivAt_const (d : ℝ) (-b * c : ℝ))) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: lt-deriv.sage

/-- The gradient vanishes only at `M = 0`, where `det = 0`: `1` is a regular value. -/
theorem det_grad_zero (a b c d : ℝ) (h : d = 0 ∧ -c = 0 ∧ -b = 0 ∧ a = 0) : a * d - b * c = 0 ∧ a * d - b * c ≠ 1 := by
  obtain ⟨rfl, hc, hb, rfl⟩ := h
  have hc' : c = 0 := by linarith
  have hb' : b = 0 := by linarith
  subst hc' hb'; norm_num

/-- At the identity the gradient `(1, 0, 0, 1)` pairs with `X` to `X₁₁ + X₂₂ = tr X`. -/
theorem det_grad_identity (X : Matrix (Fin 2) (Fin 2) ℝ) : 1 * X 0 0 + 0 * X 0 1 + 0 * X 1 0 + 1 * X 1 1 = X.trace := by
  simp [Matrix.trace_fin_two]

/-- The tangent space `{tr X = 0}` of `SL(2, ℝ)` has dimension `4 − 1 = 3`. -/
theorem finrank_sl2 : finrank ℝ (LinearMap.ker (Matrix.traceLinearMap (Fin 2) ℝ ℝ)) = 3 := by
  veriphy_certified
    ( have h :=
        Veriphy.Linear.finrank_ker_add_one (Matrix.traceLinearMap (Fin 2) ℝ ℝ) (!![1, 0; 0, 0])
          (by simp [Matrix.trace, Fin.sum_univ_succ])
      simp [Module.finrank_matrix, Module.finrank_fin_fun] at h
      omega)
  -- certificate: lt-dim-ker.sage

/-- The traceless `3 × 3` matrices, `sl(3)`, have dimension `8`. -/
theorem finrank_sl3 : finrank ℝ (LinearMap.ker (Matrix.traceLinearMap (Fin 3) ℝ ℝ)) = 8 := by
  veriphy_certified
    ( have h :=
        Veriphy.Linear.finrank_ker_add_one (Matrix.traceLinearMap (Fin 3) ℝ ℝ) (!![1, 0, 0; 0, 0, 0; 0, 0, 0])
          (by simp [Matrix.trace, Fin.sum_univ_succ])
      simp [Module.finrank_matrix, Module.finrank_fin_fun] at h
      omega)
  -- certificate: lt-dim-ker.sage

/-- The curve `M(t) = (1 + t E) R(t)` lies in `SL(2, ℝ)`. -/
theorem sl2_curve_det (t : ℝ) : (!![1, t; 0, 1] * rot2 t).det = 1 := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply, KaluzaKlein.LieTheory.Generators.s1,
        KaluzaKlein.LieTheory.Generators.s2, KaluzaKlein.LieTheory.Generators.s3, KaluzaKlein.LieTheory.Generators.T0,
        KaluzaKlein.LieTheory.Generators.T1, KaluzaKlein.LieTheory.Generators.T2, KaluzaKlein.LieTheory.Generators.Tv,
        KaluzaKlein.LieTheory.Generators.qi, KaluzaKlein.LieTheory.Generators.qj, KaluzaKlein.LieTheory.Generators.qk,
        KaluzaKlein.LieTheory.Generators.S1, KaluzaKlein.LieTheory.Generators.S2, KaluzaKlein.LieTheory.Generators.S3,
        KaluzaKlein.LieTheory.Generators.su2El, KaluzaKlein.LieTheory.Generators.L0R,
        KaluzaKlein.LieTheory.Generators.L1R, KaluzaKlein.LieTheory.Generators.nL, KaluzaKlein.LieTheory.Generators.J3v,
        KaluzaKlein.LieTheory.Generators.Jpv, KaluzaKlein.LieTheory.Generators.Jmv,
        KaluzaKlein.LieTheory.Generators.J1v, KaluzaKlein.LieTheory.Generators.J2v,
        KaluzaKlein.LieTheory.Generators.rotX, KaluzaKlein.LieTheory.Generators.rotY,
        KaluzaKlein.LieTheory.Generators.rot2, KaluzaKlein.LieTheory.Generators.H, KaluzaKlein.LieTheory.Generators.E,
        KaluzaKlein.LieTheory.Generators.F, KaluzaKlein.LieTheory.Generators.HR, KaluzaKlein.LieTheory.Generators.ER,
        KaluzaKlein.LieTheory.Generators.FR, KaluzaKlein.LieTheory.Generators.E12, KaluzaKlein.LieTheory.Generators.E23,
        KaluzaKlein.LieTheory.Generators.E13, KaluzaKlein.LieTheory.Generators.heis,
        KaluzaKlein.LieTheory.Generators.heisAlg, KaluzaKlein.LieTheory.Generators.affX,
        KaluzaKlein.LieTheory.Generators.affY, KaluzaKlein.LieTheory.Generators.aff,
        KaluzaKlein.LieTheory.Generators.affAlg, KaluzaKlein.LieTheory.Generators.boost,
        KaluzaKlein.LieTheory.Generators.eta2, KaluzaKlein.LieTheory.Generators.Jsym,
        KaluzaKlein.LieTheory.Generators.expT3, KaluzaKlein.LieTheory.Generators.expT1,
        KaluzaKlein.LieTheory.Generators.expT2, KaluzaKlein.LieTheory.Generators.euler,
        KaluzaKlein.LieTheory.Generators.su2Q, KaluzaKlein.LieTheory.Generators.gm1,
        KaluzaKlein.LieTheory.Generators.gm2, KaluzaKlein.LieTheory.Generators.gm3,
        KaluzaKlein.LieTheory.Generators.gm4, KaluzaKlein.LieTheory.Generators.gm5,
        KaluzaKlein.LieTheory.Generators.gm6, KaluzaKlein.LieTheory.Generators.gm7,
        KaluzaKlein.LieTheory.Generators.gm8s, KaluzaKlein.LieTheory.Generators.E12c,
        KaluzaKlein.LieTheory.Generators.E13c, KaluzaKlein.LieTheory.Generators.E23c, KaluzaKlein.rotation,
        KaluzaKlein.L2R, KaluzaKlein.L0, KaluzaKlein.L1, KaluzaKlein.L2, KaluzaKlein.L, KaluzaKlein.Representations.J,
        KaluzaKlein.Representations.K, KaluzaKlein.M4]
      all_goals linear_combination ((1 : ℝ)) * (Real.sin_sq_add_cos_sq (t : ℝ)))
  -- certificate: lt-matrix-trig.sage

/-- Its velocity at `t = 0` is traceless: the diagonal entries `t sin t + cos t` and `cos t` have zero derivative at `0`. -/
theorem sl2_curve_velocity_11 : deriv (fun t : ℝ => t * Real.sin t + Real.cos t) 0 = 0 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun t => t * Real.sin t + Real.cos t) ((0 : ℝ)) (0) :=
        by
        convert (((hasDerivAt_id' (0 : ℝ)).mul ((hasDerivAt_id' (0 : ℝ)).sin)).add ((hasDerivAt_id' (0 : ℝ)).cos)) using
          1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: lt-deriv.sage
theorem sl2_curve_velocity_22 : deriv (fun t : ℝ => Real.cos t) 0 = 0 := by
  VERIPHY[lt-deriv]

/-! ## exr-lg-symplectic -/

/-- `Sp(2n)` is closed under products. -/
theorem symplectic_mul {m : Type*} [Fintype m] [DecidableEq m] (J M N : Matrix m m ℝ)
    (hM : Mᵀ * J * M = J) (hN : Nᵀ * J * N = J) : (M * N)ᵀ * J * (M * N) = J := by
  rw [Matrix.transpose_mul]
  calc Nᵀ * Mᵀ * J * (M * N) = Nᵀ * (Mᵀ * J * M) * N := by simp only [Matrix.mul_assoc]
    _ = J := by rw [hM, hN]

/-- Taking determinants of `Mᵀ J M = J` gives `(det M)² = 1` when `det J ≠ 0`. -/
theorem symplectic_det_sq {m : Type*} [Fintype m] [DecidableEq m] (J M : Matrix m m ℝ) (hJ : J.det ≠ 0)
    (hM : Mᵀ * J * M = J) : M.det ^ 2 = 1 := by
  have h := congrArg Matrix.det hM
  rw [Matrix.det_mul, Matrix.det_mul, Matrix.det_transpose] at h
  have : M.det ^ 2 * J.det = 1 * J.det := by rw [one_mul, ← h]; ring
  exact mul_right_cancel₀ hJ this

/-- The condition is polynomial, so `Sp(2n)` is closed. -/
theorem symplectic_isClosed (n : ℕ) (J : Matrix (Fin n) (Fin n) ℝ) : IsClosed {M : Matrix (Fin n) (Fin n) ℝ | Mᵀ * J * M = J} :=
  isClosed_eq ((continuous_id.matrix_transpose.matrix_mul continuous_const).matrix_mul continuous_id) continuous_const

/-- For `n = 1`, `Mᵀ J M = (det M) J`: `Sp(2, ℝ) = SL(2, ℝ)`. -/
theorem symplectic_two (a b c d : ℝ) : (!![a, b; c, d])ᵀ * Jsym * !![a, b; c, d] = (a * d - b * c) • Jsym := by
  VERIPHY[lt-matrix]

/-- The inverse of a symplectic matrix is `J⁻¹ Mᵀ J`. -/
theorem symplectic_inv {m : Type*} [Fintype m] [DecidableEq m] (J M : Matrix m m ℝ) (hJ : IsUnit J.det)
    (hM : Mᵀ * J * M = J) : (J⁻¹ * Mᵀ * J) * M = 1 := by
  rw [Matrix.mul_assoc, Matrix.mul_assoc, ← Matrix.mul_assoc J, hM, Matrix.nonsing_inv_mul J hJ]

/-- The symplectic Lie algebra `{X : Xᵀ J + J X = 0}`. -/
def spAlgebra (n : ℕ) (J : Matrix (Fin n) (Fin n) ℝ) : Submodule ℝ (Matrix (Fin n) (Fin n) ℝ) where
  carrier := {X | Xᵀ * J + J * X = 0}
  add_mem' {X Y} hX hY := by
    simp only [Set.mem_setOf_eq] at *
    rw [Matrix.transpose_add, Matrix.add_mul, Matrix.mul_add]
    calc Xᵀ * J + Yᵀ * J + (J * X + J * Y) = (Xᵀ * J + J * X) + (Yᵀ * J + J * Y) := by abel
      _ = 0 := by rw [hX, hY, add_zero]
  zero_mem' := by simp
  smul_mem' c {X} hX := by
    simp only [Set.mem_setOf_eq] at *
    rw [Matrix.transpose_smul, Matrix.smul_mul, Matrix.mul_smul, ← smul_add, hX, smul_zero]

/-- The standard symplectic form `J` on `ℝ²ⁿ`, indexed by `Fin n ⊕ Fin n`. -/
def Jstd (n : ℕ) : Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ := Matrix.fromBlocks 0 1 (-1) 0

/-- `dim sp(2n) = n (2n + 1)`: the dimension formula for the symplectic Lie algebra of `Jₙ` (reindexed to `Fin (2n)`). -/
def spDimFormula (n : ℕ) : Prop :=
  finrank ℝ (spAlgebra (n + n) (Matrix.reindex finSumFinEquiv finSumFinEquiv (Jstd n))) = n * (2 * n + 1)

/-! ## def-lie-subgroup, exm-lie-subgroup-so2 -/

/-- The stabiliser of a vector is closed: `SO(2) = {R : R e₃ = e₃}` is a closed subgroup of `SO(3)`. -/
theorem stabiliser_isClosed (n : ℕ) (v : Fin n → ℝ) : IsClosed {R : Matrix (Fin n) (Fin n) ℝ | R *ᵥ v = v} :=
  isClosed_eq (continuous_id.matrix_mulVec continuous_const) continuous_const

/-- `R(t) R(−t) = 1`: the rotations about the third axis form a subgroup. -/
theorem rotation_mul_neg (t : ℝ) : rotation t * rotation (-t) = 1 := by
  VERIPHY[lt-matrix-trig]

/-- The rotation subgroup is a one-parameter subgroup: `R(s) R(t) = R(s + t)`. -/
theorem rotation_group_law (s t : ℝ) : rotation s * rotation t = rotation (s + t) := by
  rw [show rotation (s + t) = !![Real.cos s * Real.cos t - Real.sin s * Real.sin t,
      -(Real.sin s * Real.cos t + Real.cos s * Real.sin t), 0;
      Real.sin s * Real.cos t + Real.cos s * Real.sin t, Real.cos s * Real.cos t - Real.sin s * Real.sin t, 0;
      0, 0, 1] by ext i j; fin_cases i <;> fin_cases j <;> simp [rotation, Real.cos_add, Real.sin_add]]
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ,
            KaluzaKlein.LieTheory.Generators.s1, KaluzaKlein.LieTheory.Generators.s2,
            KaluzaKlein.LieTheory.Generators.s3, KaluzaKlein.LieTheory.Generators.T0,
            KaluzaKlein.LieTheory.Generators.T1, KaluzaKlein.LieTheory.Generators.T2,
            KaluzaKlein.LieTheory.Generators.Tv, KaluzaKlein.LieTheory.Generators.qi,
            KaluzaKlein.LieTheory.Generators.qj, KaluzaKlein.LieTheory.Generators.qk,
            KaluzaKlein.LieTheory.Generators.S1, KaluzaKlein.LieTheory.Generators.S2,
            KaluzaKlein.LieTheory.Generators.S3, KaluzaKlein.LieTheory.Generators.su2El,
            KaluzaKlein.LieTheory.Generators.L0R, KaluzaKlein.LieTheory.Generators.L1R,
            KaluzaKlein.LieTheory.Generators.nL, KaluzaKlein.LieTheory.Generators.J3v,
            KaluzaKlein.LieTheory.Generators.Jpv, KaluzaKlein.LieTheory.Generators.Jmv,
            KaluzaKlein.LieTheory.Generators.J1v, KaluzaKlein.LieTheory.Generators.J2v,
            KaluzaKlein.LieTheory.Generators.rotX, KaluzaKlein.LieTheory.Generators.rotY,
            KaluzaKlein.LieTheory.Generators.rot2, KaluzaKlein.LieTheory.Generators.H,
            KaluzaKlein.LieTheory.Generators.E, KaluzaKlein.LieTheory.Generators.F, KaluzaKlein.LieTheory.Generators.HR,
            KaluzaKlein.LieTheory.Generators.ER, KaluzaKlein.LieTheory.Generators.FR,
            KaluzaKlein.LieTheory.Generators.E12, KaluzaKlein.LieTheory.Generators.E23,
            KaluzaKlein.LieTheory.Generators.E13, KaluzaKlein.LieTheory.Generators.heis,
            KaluzaKlein.LieTheory.Generators.heisAlg, KaluzaKlein.LieTheory.Generators.affX,
            KaluzaKlein.LieTheory.Generators.affY, KaluzaKlein.LieTheory.Generators.aff,
            KaluzaKlein.LieTheory.Generators.affAlg, KaluzaKlein.LieTheory.Generators.boost,
            KaluzaKlein.LieTheory.Generators.eta2, KaluzaKlein.LieTheory.Generators.Jsym,
            KaluzaKlein.LieTheory.Generators.expT3, KaluzaKlein.LieTheory.Generators.expT1,
            KaluzaKlein.LieTheory.Generators.expT2, KaluzaKlein.LieTheory.Generators.euler,
            KaluzaKlein.LieTheory.Generators.su2Q, KaluzaKlein.LieTheory.Generators.gm1,
            KaluzaKlein.LieTheory.Generators.gm2, KaluzaKlein.LieTheory.Generators.gm3,
            KaluzaKlein.LieTheory.Generators.gm4, KaluzaKlein.LieTheory.Generators.gm5,
            KaluzaKlein.LieTheory.Generators.gm6, KaluzaKlein.LieTheory.Generators.gm7,
            KaluzaKlein.LieTheory.Generators.gm8s, KaluzaKlein.LieTheory.Generators.E12c,
            KaluzaKlein.LieTheory.Generators.E13c, KaluzaKlein.LieTheory.Generators.E23c, KaluzaKlein.rotation,
            KaluzaKlein.L2R, KaluzaKlein.L0, KaluzaKlein.L1, KaluzaKlein.L2, KaluzaKlein.L,
            KaluzaKlein.Representations.J, KaluzaKlein.Representations.K, KaluzaKlein.M4]
        all_goals ring1))
  -- certificate: lt-matrix.sage

/-- The diagonal matrices `e^{t T₃}` form a one-parameter subgroup of `SU(2)`. -/
theorem expT3_mul (s t : ℂ) : expT3 s * expT3 t = expT3 (s + t) := by
  rw [show expT3 (s + t) = !![Complex.exp (-(Complex.I * s / 2)) * Complex.exp (-(Complex.I * t / 2)), 0;
      0, Complex.exp (Complex.I * s / 2) * Complex.exp (Complex.I * t / 2)] by
    ext i j; fin_cases i <;> fin_cases j <;> simp [expT3, ← Complex.exp_add] <;> ring_nf]
  VERIPHY[lt-matrix-c]

/-- `e^{t T₃} e^{−t T₃} = 1`: the inverse is the adjoint, and the circle `e^{t T₃}` lies on `S³`. -/
theorem expT3_mul_neg (t : ℂ) : expT3 t * expT3 (-t) = 1 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [expT3, ← Complex.exp_add] <;> ring_nf <;> simp

/-! ## exm-lie-subgroup-torus -/

/-- The diagonal `{(z, z)}` of the torus is closed. -/
theorem diagonal_isClosed {X : Type*} [TopologicalSpace X] [T2Space X] : IsClosed {p : X × X | p.1 = p.2} :=
  isClosed_eq continuous_fst continuous_snd

/-- The multiples of an irrational angle are dense on the circle: for irrational `α` the points
`e^{2πi α k}`, `k ∈ ℤ`, are dense in `U(1)`, so the line `(e^{it}, e^{iαt})` winds densely around the torus. -/
theorem irrational_multiples_dense (α : ℝ) (hα : Irrational α) :
    DenseRange fun k : ℤ => k • ((2 * Real.pi * α : ℝ) : AddCircle (2 * Real.pi)) := by
  haveI : Fact (0 < 2 * Real.pi) := ⟨by positivity⟩
  rw [AddCircle.denseRange_zsmul_coe_iff]
  have h2 : (2 * Real.pi) ≠ 0 := by positivity
  rwa [show 2 * Real.pi * α / (2 * Real.pi) = α by field_simp]

/-- For rational slope `p/q` the line closes up: `t = 2πq` returns both coordinates to `1`. -/
theorem rational_slope_closes (p : ℤ) (q : ℕ) (hq : q ≠ 0) :
    Complex.exp ((2 * Real.pi * q : ℝ) * Complex.I) = 1 ∧
      Complex.exp (((p : ℝ) / q * (2 * Real.pi * q) : ℝ) * Complex.I) = 1 := by
  constructor
  · have := Complex.exp_nat_mul_two_pi_mul_I q
    convert this using 2; push_cast; ring
  · have := Complex.exp_int_mul_two_pi_mul_I p
    convert this using 2
    have hq' : (q : ℝ) ≠ 0 := by exact_mod_cast hq
    push_cast; field_simp; ring

/-! ## exm-lie-subgroup-finite -/

/-- `{±1}` is a closed subgroup of `SU(2)`: `(−1)² = 1`, and a finite set is closed. -/
theorem neg_one_sq : (-1 : Matrix (Fin 2) (Fin 2) ℂ) * (-1) = 1 := by
  VERIPHY[lt-matrix-c]

theorem finite_isClosed {X : Type*} [TopologicalSpace X] [T1Space X] (s : Set X) (hs : s.Finite) : IsClosed s :=
  hs.isClosed

/-- The `n`-th roots of unity in `ℂ` form a subgroup of exactly `n` elements, `ℤₙ ⊂ U(1)`. -/
theorem card_roots_of_unity (n : ℕ) [NeZero n] : Fintype.card (rootsOfUnity n ℂ) = n :=
  Complex.card_rootsOfUnity n

/-- `2πℤ` is the kernel of `t ↦ e^{it}`: `ℝ / 2πℤ = U(1)`. -/
theorem exp_eq_one_iff_kernel (t : ℝ) : Complex.exp (t * Complex.I) = 1 ↔ ∃ k : ℤ, t = k * (2 * Real.pi) := by
  rw [Complex.exp_eq_one_iff]
  constructor
  · rintro ⟨k, hk⟩
    refine ⟨k, ?_⟩
    have := congrArg Complex.im hk
    simpa using this
  · rintro ⟨k, rfl⟩
    exact ⟨k, by push_cast; ring⟩

/-- The power map `z ↦ zⁿ` identifies `U(1)/ℤₙ` with `U(1)`: its kernel is the `n`-th roots of unity. -/
theorem pow_kernel (n : ℕ) (z : ℂˣ) : z ^ n = 1 ↔ z ∈ rootsOfUnity n ℂ := (mem_rootsOfUnity n z).symm

/-! ## exr-closed-subgroup -/

/-- The diagonal matrices form a closed subset. -/
theorem diagonalMatrices_isClosed (n : ℕ) : IsClosed {M : Matrix (Fin n) (Fin n) ℝ | ∀ i j, i ≠ j → M i j = 0} := by
  have : {M : Matrix (Fin n) (Fin n) ℝ | ∀ i j, i ≠ j → M i j = 0} =
      ⋂ i, ⋂ j, ⋂ (_ : i ≠ j), {M : Matrix (Fin n) (Fin n) ℝ | M i j = 0} := by
    ext M; simp
  rw [this]
  exact isClosed_iInter fun i => isClosed_iInter fun j => isClosed_iInter fun _ =>
    isClosed_eq (continuous_id.matrix_elem i j) continuous_const

/-- The diagonal matrices as a subspace. -/
def diagMatrices (n : ℕ) : Submodule ℝ (Matrix (Fin n) (Fin n) ℝ) where
  carrier := {M | ∀ i j, i ≠ j → M i j = 0}
  add_mem' {M N} hM hN i j hij := by simp [hM i j hij, hN i j hij]
  zero_mem' i j _ := rfl
  smul_mem' c {M} hM i j hij := by simp [hM i j hij]

/-- The diagonal matrices are `ℝⁿ`: their diagonal entries. -/
def diagEquiv (n : ℕ) : diagMatrices n ≃ₗ[ℝ] (Fin n → ℝ) where
  toFun M i := (M : Matrix (Fin n) (Fin n) ℝ) i i
  invFun d := ⟨Matrix.diagonal d, fun i j hij => Matrix.diagonal_apply_ne d hij⟩
  map_add' M N := rfl
  map_smul' c M := rfl
  left_inv M := by
    ext i j
    by_cases hij : i = j
    · subst hij; simp
    · simp [Matrix.diagonal_apply_ne _ hij, M.2 i j hij]
  right_inv d := by ext i; simp

/-- The diagonal matrices in `GL(n, ℝ)` form a Lie group of dimension `n`. -/
theorem finrank_diag (n : ℕ) : finrank ℝ (diagMatrices n) = n := by
  rw [(diagEquiv n).finrank_eq, Module.finrank_fin_fun]

/-- `ℚ` is dense in `ℝ`: every real is within `ε` of a rational, so `ℚ` is not closed. -/
theorem rat_dense (x : ℝ) (ε : ℝ) (hε : 0 < ε) : ∃ q : ℚ, |x - q| < ε := by
  obtain ⟨q, hq1, hq2⟩ := exists_rat_btwn (show x - ε < x + ε by linarith)
  exact ⟨q, abs_sub_lt_iff.mpr ⟨by linarith, by linarith⟩⟩

/-- `ℚ` is not discrete: every rational has other rationals arbitrarily close, so it is no Lie subgroup. -/
theorem rat_not_discrete (q : ℚ) (ε : ℚ) (hε : 0 < ε) : ∃ q' : ℚ, q' ≠ q ∧ |q' - q| < ε :=
  ⟨q + ε / 2, by linarith, by rw [add_sub_cancel_left, abs_of_pos (by linarith)]; linarith⟩

/-! ## exr-lg-centre-su2 -/

/-- Every element of `SU(2)` is a product of exponentials of the generators (the Euler angles). -/
theorem euler_eq_product (θ φ ψ : ℂ) : expT3 φ * expT1 θ * expT3 ψ = euler θ φ ψ := by
  rw [show euler θ φ ψ = !![Complex.cos (θ / 2) * (Complex.exp (-(Complex.I * φ / 2)) * Complex.exp (-(Complex.I * ψ / 2))),
      -(Complex.I * Complex.sin (θ / 2) * (Complex.exp (-(Complex.I * φ / 2)) * Complex.exp (Complex.I * ψ / 2)));
      -(Complex.I * Complex.sin (θ / 2) * (Complex.exp (Complex.I * φ / 2) * Complex.exp (-(Complex.I * ψ / 2)))),
      Complex.cos (θ / 2) * (Complex.exp (Complex.I * φ / 2) * Complex.exp (Complex.I * ψ / 2))] by
    ext i j; fin_cases i <;> fin_cases j <;> simp [euler, ← Complex.exp_add] <;> ring_nf]
  VERIPHY[lt-matrix-c]

/-- An element commuting with the three one-parameter subgroups commutes with every Euler-angle element. -/
theorem comm_generators_comm_all (g : Matrix (Fin 2) (Fin 2) ℂ) (θ φ ψ : ℂ)
    (h3 : ∀ t, g * expT3 t = expT3 t * g) (h1 : ∀ t, g * expT1 t = expT1 t * g) :
    g * euler θ φ ψ = euler θ φ ψ * g := by
  rw [← euler_eq_product, ← Matrix.mul_assoc, h3, Matrix.mul_assoc, Matrix.mul_assoc, ← Matrix.mul_assoc (expT1 θ), ← h1,
    Matrix.mul_assoc, h3, Matrix.mul_assoc, Matrix.mul_assoc]

/-- A `2 × 2` matrix commuting with `σ₃` is diagonal: `[M, σ₃]` has the off-diagonal entries `−2b, 2c`. -/
theorem comm_sigma3 (a b c d : ℂ) : !![a, b; c, d] * s3 - s3 * !![a, b; c, d] = !![0, -2 * b; 2 * c, 0] := by
  VERIPHY[lt-matrix-c]

/-- A diagonal matrix commuting with `σ₁` has equal entries: `[diag(a, d), σ₁] = (a − d) (E₁₂ − E₂₁)`. -/
theorem comm_sigma1 (a d : ℂ) : !![a, 0; 0, d] * s1 - s1 * !![a, 0; 0, d] = !![0, a - d; d - a, 0] := by
  veriphy_certified
    ( ext i j
      fin_cases i <;> fin_cases j <;>
        norm_num [KaluzaKlein.LieTheory.Generators.s1, KaluzaKlein.LieTheory.Generators.s2,
          KaluzaKlein.LieTheory.Generators.s3, KaluzaKlein.LieTheory.Generators.T0, KaluzaKlein.LieTheory.Generators.T1,
          KaluzaKlein.LieTheory.Generators.T2, KaluzaKlein.LieTheory.Generators.Tv, KaluzaKlein.LieTheory.Generators.qi,
          KaluzaKlein.LieTheory.Generators.qj, KaluzaKlein.LieTheory.Generators.qk, KaluzaKlein.LieTheory.Generators.S1,
          KaluzaKlein.LieTheory.Generators.S2, KaluzaKlein.LieTheory.Generators.S3,
          KaluzaKlein.LieTheory.Generators.su2El, KaluzaKlein.LieTheory.Generators.L0R,
          KaluzaKlein.LieTheory.Generators.L1R, KaluzaKlein.LieTheory.Generators.nL,
          KaluzaKlein.LieTheory.Generators.J3v, KaluzaKlein.LieTheory.Generators.Jpv,
          KaluzaKlein.LieTheory.Generators.Jmv, KaluzaKlein.LieTheory.Generators.J1v,
          KaluzaKlein.LieTheory.Generators.J2v, KaluzaKlein.LieTheory.Generators.rotX,
          KaluzaKlein.LieTheory.Generators.rotY, KaluzaKlein.LieTheory.Generators.rot2,
          KaluzaKlein.LieTheory.Generators.H, KaluzaKlein.LieTheory.Generators.E, KaluzaKlein.LieTheory.Generators.F,
          KaluzaKlein.LieTheory.Generators.HR, KaluzaKlein.LieTheory.Generators.ER, KaluzaKlein.LieTheory.Generators.FR,
          KaluzaKlein.LieTheory.Generators.E12, KaluzaKlein.LieTheory.Generators.E23,
          KaluzaKlein.LieTheory.Generators.E13, KaluzaKlein.LieTheory.Generators.heis,
          KaluzaKlein.LieTheory.Generators.heisAlg, KaluzaKlein.LieTheory.Generators.affX,
          KaluzaKlein.LieTheory.Generators.affY, KaluzaKlein.LieTheory.Generators.aff,
          KaluzaKlein.LieTheory.Generators.affAlg, KaluzaKlein.LieTheory.Generators.boost,
          KaluzaKlein.LieTheory.Generators.eta2, KaluzaKlein.LieTheory.Generators.Jsym,
          KaluzaKlein.LieTheory.Generators.expT3, KaluzaKlein.LieTheory.Generators.expT1,
          KaluzaKlein.LieTheory.Generators.expT2, KaluzaKlein.LieTheory.Generators.euler,
          KaluzaKlein.LieTheory.Generators.su2Q, KaluzaKlein.LieTheory.Generators.gm1,
          KaluzaKlein.LieTheory.Generators.gm2, KaluzaKlein.LieTheory.Generators.gm3,
          KaluzaKlein.LieTheory.Generators.gm4, KaluzaKlein.LieTheory.Generators.gm5,
          KaluzaKlein.LieTheory.Generators.gm6, KaluzaKlein.LieTheory.Generators.gm7,
          KaluzaKlein.LieTheory.Generators.gm8s, KaluzaKlein.LieTheory.Generators.E12c,
          KaluzaKlein.LieTheory.Generators.E13c, KaluzaKlein.LieTheory.Generators.E23c, KaluzaKlein.rotation,
          KaluzaKlein.L2R, KaluzaKlein.L0, KaluzaKlein.L1, KaluzaKlein.L2, KaluzaKlein.L, KaluzaKlein.Representations.J,
          KaluzaKlein.Representations.K, KaluzaKlein.M4, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply,
          Matrix.one_apply, Complex.ext_iff]
      all_goals ring1)
  -- certificate: lt-matrix-c.sage

/-- `det (λ 1) = λ² = 1` forces `λ = ±1`: the centre of `SU(2)` is `{±1}`. -/
theorem scalar_in_su2 (l : ℂ) (h : (l • (1 : Matrix (Fin 2) (Fin 2) ℂ)).det = 1) : l = 1 ∨ l = -1 := by
  rw [Matrix.det_smul, Matrix.det_one, mul_one, Fintype.card_fin] at h
  exact sq_eq_one_iff.mp h

/-- `Ad_g = 1` exactly when `g` commutes with every matrix: the centre is the kernel of `Ad`. -/
theorem ad_trivial_iff_comm (g : Matrix (Fin 2) (Fin 2) ℂ) (hg : IsUnit g.det) (X : Matrix (Fin 2) (Fin 2) ℂ) :
    g * X * g⁻¹ = X ↔ g * X = X * g := by
  constructor
  · intro h
    calc g * X = g * X * g⁻¹ * g := by rw [Matrix.mul_assoc, Matrix.nonsing_inv_mul g hg, Matrix.mul_one]
      _ = X * g := by rw [h]
  · intro h
    rw [h, Matrix.mul_assoc, Matrix.mul_nonsing_inv g hg, Matrix.mul_one]

/-! ## exr-lg-maximal-torus -/

/-- A diagonal matrix is unitary exactly when its entries have modulus one. -/
theorem diagonal_unitary_iff (n : ℕ) (d : Fin n → ℂ) :
    (Matrix.diagonal d)ᴴ * Matrix.diagonal d = 1 ↔ ∀ i, star (d i) * d i = 1 := by
  rw [Matrix.diagonal_conjTranspose, Matrix.diagonal_mul_diagonal, ← Matrix.diagonal_one, Matrix.diagonal_eq_diagonal_iff]
  simp [Pi.star_apply]

/-- Products of diagonal matrices are diagonal: the torus is a subgroup. -/
theorem diagonal_mul (n : ℕ) (d e : Fin n → ℂ) : Matrix.diagonal d * Matrix.diagonal e = Matrix.diagonal (d * e) :=
  Matrix.diagonal_mul_diagonal d e

/-- `det diag(e^{iα₁}, …, e^{iαₙ}) = e^{i Σ αₖ}`: the determinant condition is `Σ αₖ ∈ 2πℤ`. -/
theorem det_diagonal_exp (n : ℕ) (α : Fin n → ℝ) :
    (Matrix.diagonal fun i => Complex.exp (α i * Complex.I)).det = Complex.exp ((∑ i, α i : ℝ) * Complex.I) := by
  rw [Matrix.det_diagonal, ← Complex.exp_sum]
  congr 1; push_cast; rw [Finset.sum_mul]

/-- For `n = 2` the diagonal torus is the circle `diag(e^{iα}, e^{−iα}) = e^{−2α T₃}`. -/
theorem diagonal_torus_two (α : ℂ) : expT3 (-2 * α) = !![Complex.exp (Complex.I * α), 0; 0, Complex.exp (-(Complex.I * α))] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [expT3] <;> ring_nf

end KaluzaKlein.LieTheory.LieGroups
