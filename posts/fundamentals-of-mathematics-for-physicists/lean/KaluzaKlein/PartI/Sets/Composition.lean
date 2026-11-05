import Mathlib
import KaluzaKlein.PartI.Sets.Counting
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Composition"

The claims of the section "Composition" of `tutorial/part-i-01-sets-relations-topology.qmd`: a bijection composed with
its inverse, the swap and the cycle of three elements, the transition map of the two stereographic charts of the
sphere, rotations about one axis and about two axes, and associativity.
-/

namespace KaluzaKlein.Sets.Composition

open Matrix KaluzaKlein.Sets.Surjective

/-- The swap `s` of `1` and `2` (as `0` and `1`). -/
def swapS : Equiv.Perm (Fin 3) := Equiv.swap 0 1

/-- The cycle `t : 1 ↦ 2 ↦ 3 ↦ 1` (as `0 ↦ 1 ↦ 2 ↦ 0`). -/
def cycleT : Equiv.Perm (Fin 3) := finRotate 3

/-- The rotation by `β` about the `X`-axis. -/
noncomputable def Rx (β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![1, 0, 0; 0, Real.cos β, -Real.sin β; 0, Real.sin β, Real.cos β]

/-! ## def-composition -/

/-- A bijection and its inverse compose to the identities: `f⁻¹ ∘ f = id_X`, `f ∘ f⁻¹ = id_Y`. -/
theorem inverse_comp {X Y : Type} (e : X ≃ Y) : e.symm ∘ e = id ∧ e ∘ e.symm = id :=
  ⟨e.symm_comp_self, e.self_comp_symm⟩

example : veriphy_instance% KaluzaKlein.Sets.Composition.inverse_comp (Fin 3) (Fin 3) (Equiv.refl _) := by
  have key : ∀ {X Y : Type} (e : X ≃ Y), e.symm ∘ e = id ∧ e ∘ e.symm = id := by
    intro X Y e
    exact (⟨e.symm_comp_self, e.self_comp_symm⟩)
  exact @key (Fin 3) (Fin 3) (Equiv.refl _)

example : veriphy_instance% KaluzaKlein.Sets.Composition.inverse_comp (Fin 3) (Fin 3) (finRotate 3) := by
  have key : ∀ {X Y : Type} (e : X ≃ Y), e.symm ∘ e = id ∧ e ∘ e.symm = id := by
    intro X Y e
    exact (⟨e.symm_comp_self, e.self_comp_symm⟩)
  exact @key (Fin 3) (Fin 3) (finRotate 3)

example : veriphy_instance% KaluzaKlein.Sets.Composition.inverse_comp (ℤ) (ℤ) (Equiv.addRight 1) := by
  have key : ∀ {X Y : Type} (e : X ≃ Y), e.symm ∘ e = id ∧ e ∘ e.symm = id := by
    intro X Y e
    exact (⟨e.symm_comp_self, e.self_comp_symm⟩)
  exact @key (ℤ) (ℤ) (Equiv.addRight 1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Composition.inverse_comp
-- test: set-inverse-comp-test.sage

/-! ## exm-composition-permutations -/

/-- `t ∘ s ≠ s ∘ t`: a point where the two compositions differ (Sage finds it). -/
theorem swap_cycle_noncommute : ∃ x : Fin 3, cycleT (swapS x) ≠ swapS (cycleT x) := by
  veriphy_certified (exact ⟨0, by decide⟩)
  -- certificate: set-perm-noncommute.sage

/-! ## exm-composition-transition

With `w = 1/(u² + v²)` and `d = 1/(u² + v² + 1)` as variables with their defining equations. -/

/-- The transition map `x_S ∘ x_N⁻¹` of the stereographic charts from the north and south poles is the inversion
`(u, v) ↦ (u, v)/(u² + v²)`: with `(X, Y, Z) = x_N⁻¹(u, v) = (2u, 2v, u² + v² - 1) d`, `X = u w (1 + Z)` and
`Y = v w (1 + Z)`, that is `x_S = (X, Y)/(1 + Z) = (u w, v w)`. -/
theorem transition_inversion (u v w d : ℝ) (hw : (u ^ 2 + v ^ 2) * w = 1) (hd : (u ^ 2 + v ^ 2 + 1) * d = 1) :
    2 * u * d = u * w * (1 + (u ^ 2 + v ^ 2 - 1) * d) ∧ 2 * v * d = v * w * (1 + (u ^ 2 + v ^ 2 - 1) * d) := by
  refine ⟨?_, ?_⟩
  · veriphy_certified (linear_combination (-2 * u * d) * hw + (u * w) * hd)
    -- certificate: ideal_membership.sage
  · veriphy_certified (linear_combination (-2 * v * d) * hw + (v * w) * hd)
    -- certificate: ideal_membership.sage

/-- The inversion is its own inverse: `|(u w, v w)|² = w`, and `(u w, v w)/w = (u, v)`. -/
theorem inversion_involutive (u v w : ℝ) (hw : (u ^ 2 + v ^ 2) * w = 1) :
    (u * w) ^ 2 + (v * w) ^ 2 = w := by
  veriphy_certified (linear_combination (w) * hw)
  -- certificate: ideal_membership.sage

/-! ## exm-composition-rotations -/

/-- About a common axis the angles add: `R_z(α) R_z(β) = R_z(α + β)` (the right side expanded by the addition
formulas). -/
theorem Rz_mul (α β : ℝ) : Rz α * Rz β = Rz (α + β) := by
  unfold Rz
  rw [Real.cos_add, Real.sin_add]
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ, Matrix.mul_apply,
            Fin.sum_univ_succ, Matrix.transpose_apply]
        all_goals ring1))
  -- certificate: set-matrix-trig.sage

/-! ## exr-compose-permutations -/

/-- The inverse of `t` (Sage: `(1 3 2)`, as `0 ↦ 2 ↦ 1 ↦ 0`): `t ∘ t⁻¹ = t⁻¹ ∘ t = id`. -/
theorem cycle_inverse : ∃ u : Equiv.Perm (Fin 3), cycleT * u = 1 ∧ u * cycleT = 1 ∧ u 0 = 2 ∧ u 2 = 1 ∧ u 1 = 0 := by
  veriphy_certified (exact ⟨⟨![2, 0, 1], ![1, 2, 0], by decide, by decide⟩, by decide⟩)
  -- certificate: set-perm-inverse.sage

/-! ## exr-compose-associative -/

/-- Composition is associative: `(h ∘ g) ∘ f = h ∘ (g ∘ f)`. -/
theorem comp_assoc {W X Y Z : Type} (f : W → X) (g : X → Y) (h : Y → Z) : (h ∘ g) ∘ f = h ∘ (g ∘ f) :=
  Function.comp_assoc h g f

example : veriphy_instance% KaluzaKlein.Sets.Composition.comp_assoc (Fin 5) (Fin 5) (Fin 5) (Fin 5) (![1, 2, 3, 4, 0]) (![0, 2, 1, 4, 3]) (![4, 3, 2, 1, 0]) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Composition.comp_assoc (Fin 5) (Fin 5) (Fin 5) (Fin 5) (id) (id) (id) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Composition.comp_assoc (Fin 5) (Fin 5) (Fin 5) (Fin 5) (![2, 0, 4, 1, 3]) (![3, 4, 0, 2, 1]) (![1, 3, 0, 4, 2]) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Composition.comp_assoc
-- test: set-comp-assoc-test.sage

/-! ## exr-compose-rotations-noncommute -/

/-- `R_z(α) R_x(β)`, entry by entry. -/
theorem Rz_mul_Rx (α β : ℝ) : Rz α * Rx β =
    !![Real.cos α, -(Real.sin α * Real.cos β), Real.sin α * Real.sin β;
       Real.sin α, Real.cos α * Real.cos β, -(Real.cos α * Real.sin β);
       0, Real.sin β, Real.cos β] := by
  unfold Rz Rx
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ, Matrix.mul_apply,
            Fin.sum_univ_succ, Matrix.transpose_apply]
        all_goals ring1))
  -- certificate: set-matrix-trig.sage

/-- `R_x(β) R_z(α)`, entry by entry. -/
theorem Rx_mul_Rz (α β : ℝ) : Rx β * Rz α =
    !![Real.cos α, -Real.sin α, 0;
       Real.cos β * Real.sin α, Real.cos β * Real.cos α, -Real.sin β;
       Real.sin β * Real.sin α, Real.sin β * Real.cos α, Real.cos β] := by
  unfold Rz Rx
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ, Matrix.mul_apply,
            Fin.sum_univ_succ, Matrix.transpose_apply]
        all_goals ring1))
  -- certificate: set-matrix-trig.sage

/-- The `(1, 3)` entries are `sin α sin β` and `0`, so the products differ whenever `sin α sin β ≠ 0`. -/
theorem rotations_noncommute (α β : ℝ) (h : Real.sin α * Real.sin β ≠ 0) : Rz α * Rx β ≠ Rx β * Rz α := by
  intro hc
  have := congrFun (congrFun hc 0) 2
  rw [Rz_mul_Rx, Rx_mul_Rz] at this
  simp at this
  exact h (mul_eq_zero.2 this)

example : veriphy_instance% KaluzaKlein.Sets.Composition.rotations_noncommute (Real.pi / 2) (Real.pi / 2) ((by simp)) := by
  have key : ∀ (α β : ℝ) (h : Real.sin α * Real.sin β ≠ 0), Rz α * Rx β ≠ Rx β * Rz α := by
    intro α β h
    intro hc
    have := congrFun (congrFun hc 0) 2
    rw [Rz_mul_Rx, Rx_mul_Rz] at this
    simp at this
    exact h (mul_eq_zero.2 this)
  exact @key (Real.pi / 2) (Real.pi / 2) ((by simp))

example : veriphy_instance% KaluzaKlein.Sets.Composition.rotations_noncommute (Real.pi / 2) (-(Real.pi / 2)) ((by simp)) := by
  have key : ∀ (α β : ℝ) (h : Real.sin α * Real.sin β ≠ 0), Rz α * Rx β ≠ Rx β * Rz α := by
    intro α β h
    intro hc
    have := congrFun (congrFun hc 0) 2
    rw [Rz_mul_Rx, Rx_mul_Rz] at this
    simp at this
    exact h (mul_eq_zero.2 this)
  exact @key (Real.pi / 2) (-(Real.pi / 2)) ((by simp))

example : veriphy_instance% KaluzaKlein.Sets.Composition.rotations_noncommute (Real.pi / 6) (Real.pi / 2) ((by simp)) := by
  have key : ∀ (α β : ℝ) (h : Real.sin α * Real.sin β ≠ 0), Rz α * Rx β ≠ Rx β * Rz α := by
    intro α β h
    intro hc
    have := congrFun (congrFun hc 0) 2
    rw [Rz_mul_Rx, Rx_mul_Rz] at this
    simp at this
    exact h (mul_eq_zero.2 this)
  exact @key (Real.pi / 6) (Real.pi / 2) ((by simp))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Composition.rotations_noncommute
-- test: set-rotations-noncommute-test.sage

end KaluzaKlein.Sets.Composition
