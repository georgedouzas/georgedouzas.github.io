import Mathlib
import KaluzaKlein.PartI.Sets.FiniteTopologies
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Hausdorff spaces"

The claims of the section "Hausdorff spaces" of `tutorial/part-i-01-sets-relations-topology.qmd` not counted in
`FiniteTopologies`: unique limits and closed points, disjoint balls in `ℝⁿ` and Hausdorff subspaces (the circle, the
sphere, the torus, `SO(3)`), discrete and indiscrete topologies, the quotient `ℝ/cℚ` and the quotient
`SO(3)/SO(2)`, the cofinite topology, the line with a doubled point, finite Hausdorff spaces, and two Hausdorff
topologies on `ℝⁿ`.
-/

namespace KaluzaKlein.Sets.Hausdorff

open Set Metric Filter Topology Matrix KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Quotients

/-! ## def-hausdorff -/

/-- In a Hausdorff space limits are unique and single points are closed. -/
theorem hausdorff_consequences (X : Type) [TopologicalSpace X] [T2Space X] (u : ℕ → X) (a b : X)
    (ha : Tendsto u atTop (𝓝 a)) (hb : Tendsto u atTop (𝓝 b)) : a = b ∧ IsClosed ({a} : Set X) :=
  ⟨tendsto_nhds_unique ha hb, isClosed_singleton⟩

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.hausdorff_consequences (ℝ) (inferInstance) (inferInstance) (fun _ => 0) (0) (0) (tendsto_const_nhds) (tendsto_const_nhds) := by
  have key : ∀ (X : Type) [inst1 : TopologicalSpace X] [inst2 : T2Space X] (u : ℕ → X) (a b : X) (ha : Tendsto u atTop (𝓝 a)) (hb : Tendsto u atTop (𝓝 b)), a = b ∧ IsClosed ({a} : Set X) := by
    intro X inst1 inst2 u a b ha hb
    exact (⟨tendsto_nhds_unique ha hb, isClosed_singleton⟩)
  exact @key (ℝ) (inferInstance) (inferInstance) (fun _ => 0) (0) (0) (tendsto_const_nhds) (tendsto_const_nhds)

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.hausdorff_consequences (ℝ) (inferInstance) (inferInstance) (fun n => 1 / ((n : ℝ) + 1)) (0) (0) (tendsto_one_div_add_atTop_nhds_zero_nat) (tendsto_one_div_add_atTop_nhds_zero_nat) := by
  have key : ∀ (X : Type) [inst1 : TopologicalSpace X] [inst2 : T2Space X] (u : ℕ → X) (a b : X) (ha : Tendsto u atTop (𝓝 a)) (hb : Tendsto u atTop (𝓝 b)), a = b ∧ IsClosed ({a} : Set X) := by
    intro X inst1 inst2 u a b ha hb
    exact (⟨tendsto_nhds_unique ha hb, isClosed_singleton⟩)
  exact @key (ℝ) (inferInstance) (inferInstance) (fun n => 1 / ((n : ℝ) + 1)) (0) (0) (tendsto_one_div_add_atTop_nhds_zero_nat) (tendsto_one_div_add_atTop_nhds_zero_nat)

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.hausdorff_consequences (ℝ × ℝ) (inferInstance) (inferInstance) (fun _ => (1, 2)) ((1, 2)) ((1, 2)) (tendsto_const_nhds) (tendsto_const_nhds) := by
  have key : ∀ (X : Type) [inst1 : TopologicalSpace X] [inst2 : T2Space X] (u : ℕ → X) (a b : X) (ha : Tendsto u atTop (𝓝 a)) (hb : Tendsto u atTop (𝓝 b)), a = b ∧ IsClosed ({a} : Set X) := by
    intro X inst1 inst2 u a b ha hb
    exact (⟨tendsto_nhds_unique ha hb, isClosed_singleton⟩)
  exact @key (ℝ × ℝ) (inferInstance) (inferInstance) (fun _ => (1, 2)) ((1, 2)) ((1, 2)) (tendsto_const_nhds) (tendsto_const_nhds)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Hausdorff.hausdorff_consequences
-- test: set-hausdorff-consequences-test.sage

/-! ## exm-hausdorff-rn -/

/-- Two distinct points of `ℝⁿ` at distance `d` have disjoint balls of radius `d/2`. -/
theorem disjoint_balls (n : ℕ) (x y : EuclideanSpace ℝ (Fin n)) :
    Disjoint (ball x (dist x y / 2)) (ball y (dist x y / 2)) :=
  ball_disjoint_ball (by linarith)

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.disjoint_balls (0) (0) (0) := by
  have key : ∀ (n : ℕ) (x y : EuclideanSpace ℝ (Fin n)), Disjoint (ball x (dist x y / 2)) (ball y (dist x y / 2)) := by
    intro n x y
    exact (ball_disjoint_ball (by linarith))
  exact @key (0) (0) (0)

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.disjoint_balls (2) (0) (EuclideanSpace.single 0 1) := by
  have key : ∀ (n : ℕ) (x y : EuclideanSpace ℝ (Fin n)), Disjoint (ball x (dist x y / 2)) (ball y (dist x y / 2)) := by
    intro n x y
    exact (ball_disjoint_ball (by linarith))
  exact @key (2) (0) (EuclideanSpace.single 0 1)

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.disjoint_balls (3) (EuclideanSpace.single 1 2) (0) := by
  have key : ∀ (n : ℕ) (x y : EuclideanSpace ℝ (Fin n)), Disjoint (ball x (dist x y / 2)) (ball y (dist x y / 2)) := by
    intro n x y
    exact (ball_disjoint_ball (by linarith))
  exact @key (3) (EuclideanSpace.single 1 2) (0)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Hausdorff.disjoint_balls
-- test: set-disjoint-balls-test.sage

/-- Every subspace of a Hausdorff space is Hausdorff: the circle, the sphere, the torus and `SO(3)`. -/
theorem subspace_hausdorff (X : Type) [TopologicalSpace X] [T2Space X] (S : Set X) : T2Space S :=
  inferInstance

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.subspace_hausdorff (ℝ × ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.circle) := by
  infer_instance

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.subspace_hausdorff (Fin 3 → ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.sphere) := by
  infer_instance

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.subspace_hausdorff ((ℝ × ℝ) × (ℝ × ℝ)) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.circle ×ˢ KaluzaKlein.Sets.Sets.circle) := by
  infer_instance

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.subspace_hausdorff (Matrix (Fin 3) (Fin 3) ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.SO3) := by
  infer_instance

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.subspace_hausdorff (ℝ) (inferInstance) (inferInstance) (∅) := by
  infer_instance

attribute [veriphy_tested kernel_test 5 0] KaluzaKlein.Sets.Hausdorff.subspace_hausdorff
-- test: set-subspace-hausdorff-test.sage

/-! ## exm-hausdorff-finite -/

/-- The discrete topology is Hausdorff; the indiscrete topology on a set with two distinct points is not. -/
theorem discrete_indiscrete_hausdorff (X : Type) (x y : X) (hxy : x ≠ y) :
    @T2Space X ⊥ ∧ ¬ @T2Space X ⊤ := by
  refine ⟨by letI : TopologicalSpace X := ⊥; haveI : DiscreteTopology X := ⟨rfl⟩; infer_instance, fun h => ?_⟩
  obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := @t2_separation X ⊤ h x y hxy
  rw [@TopologicalSpace.isOpen_top_iff] at hU hV
  rcases hU with rfl | rfl
  · exact hxU
  · rcases hV with rfl | rfl
    · exact hyV
    · exact (Set.nonempty_of_mem hxU).ne_empty (Set.disjoint_univ.1 hUV)

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.discrete_indiscrete_hausdorff (Fin 2) (0) (1) ((by decide)) := by
  have key : ∀ (X : Type) (x y : X) (hxy : x ≠ y), @T2Space X ⊥ ∧ ¬ @T2Space X ⊤ := by
    intro X x y hxy
    refine ⟨by letI : TopologicalSpace X := ⊥; haveI : DiscreteTopology X := ⟨rfl⟩; infer_instance, fun h => ?_⟩
    obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := @t2_separation X ⊤ h x y hxy
    rw [@TopologicalSpace.isOpen_top_iff] at hU hV
    rcases hU with rfl | rfl
    · exact hxU
    · rcases hV with rfl | rfl
      · exact hyV
      · exact (Set.nonempty_of_mem hxU).ne_empty (Set.disjoint_univ.1 hUV)
  exact @key (Fin 2) (0) (1) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.discrete_indiscrete_hausdorff (Fin 3) (0) (2) ((by decide)) := by
  have key : ∀ (X : Type) (x y : X) (hxy : x ≠ y), @T2Space X ⊥ ∧ ¬ @T2Space X ⊤ := by
    intro X x y hxy
    refine ⟨by letI : TopologicalSpace X := ⊥; haveI : DiscreteTopology X := ⟨rfl⟩; infer_instance, fun h => ?_⟩
    obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := @t2_separation X ⊤ h x y hxy
    rw [@TopologicalSpace.isOpen_top_iff] at hU hV
    rcases hU with rfl | rfl
    · exact hxU
    · rcases hV with rfl | rfl
      · exact hyV
      · exact (Set.nonempty_of_mem hxU).ne_empty (Set.disjoint_univ.1 hUV)
  exact @key (Fin 3) (0) (2) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.discrete_indiscrete_hausdorff (ℕ) (0) (1) ((by decide)) := by
  have key : ∀ (X : Type) (x y : X) (hxy : x ≠ y), @T2Space X ⊥ ∧ ¬ @T2Space X ⊤ := by
    intro X x y hxy
    refine ⟨by letI : TopologicalSpace X := ⊥; haveI : DiscreteTopology X := ⟨rfl⟩; infer_instance, fun h => ?_⟩
    obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := @t2_separation X ⊤ h x y hxy
    rw [@TopologicalSpace.isOpen_top_iff] at hU hV
    rcases hU with rfl | rfl
    · exact hxU
    · rcases hV with rfl | rfl
      · exact hyV
      · exact (Set.nonempty_of_mem hxU).ne_empty (Set.disjoint_univ.1 hUV)
  exact @key (ℕ) (0) (1) ((by decide))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Hausdorff.discrete_indiscrete_hausdorff
-- test: set-discrete-hausdorff-test.sage

/-! ## exm-hausdorff-quotient -/

/-- `x ∼ y` iff `y - x ∈ cℚ`. -/
def ratSetoid (c : ℝ) : Setoid ℝ where
  r x y := ∃ q : ℚ, y - x = c * q
  iseqv := ⟨fun x => ⟨0, by simp⟩, fun ⟨q, hq⟩ => ⟨-q, by push_cast; linarith⟩,
    fun ⟨q, hq⟩ ⟨r, hr⟩ => ⟨q + r, by push_cast; linarith⟩⟩

/-- The quotient of `ℝ` by `x ∼ y` iff `x - y ∈ cℚ` (`c > 0`) is not Hausdorff: every open set is empty or everything,
since every open interval meets every class (`c = 1`: the quotient `ℝ/ℚ`). -/
theorem rat_quotient_not_hausdorff (c : ℝ) (hc : 0 < c) :
    (∀ V : Set (Quotient (ratSetoid c)), IsOpen V → V = ∅ ∨ V = univ) ∧ ¬ T2Space (Quotient (ratSetoid c)) := by
  have key : ∀ V : Set (Quotient (ratSetoid c)), IsOpen V → V = ∅ ∨ V = univ := by
    intro V hV
    rcases eq_empty_or_nonempty V with h | ⟨v, hv⟩
    · left; exact h
    · right
      refine eq_univ_of_forall fun w => ?_
      induction v using Quotient.ind
      induction w using Quotient.ind
      rename_i a b
      have hU : IsOpen (Quotient.mk (ratSetoid c) ⁻¹' V) := isQuotientMap_quotient_mk'.isOpen_preimage.2 hV
      obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 hU a hv
      obtain ⟨q, hq1, hq2⟩ := exists_rat_btwn (show (a - b) / c < (a - b + ε) / c by
        apply div_lt_div_of_pos_right _ hc; linarith)
      have hmem : b + c * q ∈ Quotient.mk (ratSetoid c) ⁻¹' V := by
        apply hball
        rw [mem_ball, Real.dist_eq, abs_lt]
        rw [div_lt_iff₀ hc] at hq1
        rw [lt_div_iff₀ hc] at hq2
        constructor <;> nlinarith
      have : Quotient.mk (ratSetoid c) (b + c * q) = Quotient.mk (ratSetoid c) b :=
        Quotient.sound ⟨-q, by push_cast; ring⟩
      simpa [this] using hmem
  refine ⟨key, fun h => ?_⟩
  -- two distinct classes cannot be separated
  have hne : Quotient.mk (ratSetoid c) 0 ≠ Quotient.mk (ratSetoid c) (c * Real.sqrt 2) := by
    intro h'
    obtain ⟨q, hq⟩ := Quotient.exact h'
    have : Real.sqrt 2 = q := by
      have := mul_left_cancel₀ hc.ne' (show c * Real.sqrt 2 = c * q by linarith)
      exact this
    exact irrational_sqrt_two ⟨q, this.symm⟩
  obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := t2_separation hne
  rcases key U hU with rfl | rfl
  · exact hxU
  · rcases key V hV with rfl | rfl
    · exact hyV
    · exact (Set.nonempty_of_mem hxU).ne_empty (Set.disjoint_univ.1 hUV)

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.rat_quotient_not_hausdorff (1) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), (∀ V : Set (Quotient (ratSetoid c)), IsOpen V → V = ∅ ∨ V = univ) ∧ ¬ T2Space (Quotient (ratSetoid c)) := by
    intro c hc
    have key : ∀ V : Set (Quotient (ratSetoid c)), IsOpen V → V = ∅ ∨ V = univ := by
      intro V hV
      rcases eq_empty_or_nonempty V with h | ⟨v, hv⟩
      · left; exact h
      · right
        refine eq_univ_of_forall fun w => ?_
        induction v using Quotient.ind
        induction w using Quotient.ind
        rename_i a b
        have hU : IsOpen (Quotient.mk (ratSetoid c) ⁻¹' V) := isQuotientMap_quotient_mk'.isOpen_preimage.2 hV
        obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 hU a hv
        obtain ⟨q, hq1, hq2⟩ := exists_rat_btwn (show (a - b) / c < (a - b + ε) / c by
          apply div_lt_div_of_pos_right _ hc; linarith)
        have hmem : b + c * q ∈ Quotient.mk (ratSetoid c) ⁻¹' V := by
          apply hball
          rw [mem_ball, Real.dist_eq, abs_lt]
          rw [div_lt_iff₀ hc] at hq1
          rw [lt_div_iff₀ hc] at hq2
          constructor <;> nlinarith
        have : Quotient.mk (ratSetoid c) (b + c * q) = Quotient.mk (ratSetoid c) b :=
          Quotient.sound ⟨-q, by push_cast; ring⟩
        simpa [this] using hmem
    refine ⟨key, fun h => ?_⟩
    -- two distinct classes cannot be separated
    have hne : Quotient.mk (ratSetoid c) 0 ≠ Quotient.mk (ratSetoid c) (c * Real.sqrt 2) := by
      intro h'
      obtain ⟨q, hq⟩ := Quotient.exact h'
      have : Real.sqrt 2 = q := by
        have := mul_left_cancel₀ hc.ne' (show c * Real.sqrt 2 = c * q by linarith)
        exact this
      exact irrational_sqrt_two ⟨q, this.symm⟩
    obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := t2_separation hne
    rcases key U hU with rfl | rfl
    · exact hxU
    · rcases key V hV with rfl | rfl
      · exact hyV
      · exact (Set.nonempty_of_mem hxU).ne_empty (Set.disjoint_univ.1 hUV)
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.rat_quotient_not_hausdorff (2) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), (∀ V : Set (Quotient (ratSetoid c)), IsOpen V → V = ∅ ∨ V = univ) ∧ ¬ T2Space (Quotient (ratSetoid c)) := by
    intro c hc
    have key : ∀ V : Set (Quotient (ratSetoid c)), IsOpen V → V = ∅ ∨ V = univ := by
      intro V hV
      rcases eq_empty_or_nonempty V with h | ⟨v, hv⟩
      · left; exact h
      · right
        refine eq_univ_of_forall fun w => ?_
        induction v using Quotient.ind
        induction w using Quotient.ind
        rename_i a b
        have hU : IsOpen (Quotient.mk (ratSetoid c) ⁻¹' V) := isQuotientMap_quotient_mk'.isOpen_preimage.2 hV
        obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 hU a hv
        obtain ⟨q, hq1, hq2⟩ := exists_rat_btwn (show (a - b) / c < (a - b + ε) / c by
          apply div_lt_div_of_pos_right _ hc; linarith)
        have hmem : b + c * q ∈ Quotient.mk (ratSetoid c) ⁻¹' V := by
          apply hball
          rw [mem_ball, Real.dist_eq, abs_lt]
          rw [div_lt_iff₀ hc] at hq1
          rw [lt_div_iff₀ hc] at hq2
          constructor <;> nlinarith
        have : Quotient.mk (ratSetoid c) (b + c * q) = Quotient.mk (ratSetoid c) b :=
          Quotient.sound ⟨-q, by push_cast; ring⟩
        simpa [this] using hmem
    refine ⟨key, fun h => ?_⟩
    -- two distinct classes cannot be separated
    have hne : Quotient.mk (ratSetoid c) 0 ≠ Quotient.mk (ratSetoid c) (c * Real.sqrt 2) := by
      intro h'
      obtain ⟨q, hq⟩ := Quotient.exact h'
      have : Real.sqrt 2 = q := by
        have := mul_left_cancel₀ hc.ne' (show c * Real.sqrt 2 = c * q by linarith)
        exact this
      exact irrational_sqrt_two ⟨q, this.symm⟩
    obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := t2_separation hne
    rcases key U hU with rfl | rfl
    · exact hxU
    · rcases key V hV with rfl | rfl
      · exact hyV
      · exact (Set.nonempty_of_mem hxU).ne_empty (Set.disjoint_univ.1 hUV)
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.rat_quotient_not_hausdorff (1/2) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), (∀ V : Set (Quotient (ratSetoid c)), IsOpen V → V = ∅ ∨ V = univ) ∧ ¬ T2Space (Quotient (ratSetoid c)) := by
    intro c hc
    have key : ∀ V : Set (Quotient (ratSetoid c)), IsOpen V → V = ∅ ∨ V = univ := by
      intro V hV
      rcases eq_empty_or_nonempty V with h | ⟨v, hv⟩
      · left; exact h
      · right
        refine eq_univ_of_forall fun w => ?_
        induction v using Quotient.ind
        induction w using Quotient.ind
        rename_i a b
        have hU : IsOpen (Quotient.mk (ratSetoid c) ⁻¹' V) := isQuotientMap_quotient_mk'.isOpen_preimage.2 hV
        obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 hU a hv
        obtain ⟨q, hq1, hq2⟩ := exists_rat_btwn (show (a - b) / c < (a - b + ε) / c by
          apply div_lt_div_of_pos_right _ hc; linarith)
        have hmem : b + c * q ∈ Quotient.mk (ratSetoid c) ⁻¹' V := by
          apply hball
          rw [mem_ball, Real.dist_eq, abs_lt]
          rw [div_lt_iff₀ hc] at hq1
          rw [lt_div_iff₀ hc] at hq2
          constructor <;> nlinarith
        have : Quotient.mk (ratSetoid c) (b + c * q) = Quotient.mk (ratSetoid c) b :=
          Quotient.sound ⟨-q, by push_cast; ring⟩
        simpa [this] using hmem
    refine ⟨key, fun h => ?_⟩
    -- two distinct classes cannot be separated
    have hne : Quotient.mk (ratSetoid c) 0 ≠ Quotient.mk (ratSetoid c) (c * Real.sqrt 2) := by
      intro h'
      obtain ⟨q, hq⟩ := Quotient.exact h'
      have : Real.sqrt 2 = q := by
        have := mul_left_cancel₀ hc.ne' (show c * Real.sqrt 2 = c * q by linarith)
        exact this
      exact irrational_sqrt_two ⟨q, this.symm⟩
    obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := t2_separation hne
    rcases key U hU with rfl | rfl
    · exact hxU
    · rcases key V hV with rfl | rfl
      · exact hyV
      · exact (Set.nonempty_of_mem hxU).ne_empty (Set.disjoint_univ.1 hUV)
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Hausdorff.rat_quotient_not_hausdorff
-- test: set-rat-quotient-test.sage

/-- The quotient of `SO(3)` by "same image of `p₀`" is Hausdorff, being homeomorphic to the sphere (`SO(3)/SO(2)`). -/
theorem coset_space_hausdorff (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere) : T2Space (Quotient (orbitSetoid p₀)) := by
  obtain ⟨e, -⟩ := QuotientTopology.sphere_quotient_homeomorph p₀ hp₀
  exact e.isEmbedding.t2Space

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.coset_space_hausdorff (KaluzaKlein.Sets.Sets.northPole) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere), T2Space (Quotient (orbitSetoid p₀)) := by
    intro p₀ hp₀
    obtain ⟨e, -⟩ := QuotientTopology.sphere_quotient_homeomorph p₀ hp₀
    exact e.isEmbedding.t2Space
  exact @key (KaluzaKlein.Sets.Sets.northPole) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.coset_space_hausdorff (![1, 0, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere), T2Space (Quotient (orbitSetoid p₀)) := by
    intro p₀ hp₀
    obtain ⟨e, -⟩ := QuotientTopology.sphere_quotient_homeomorph p₀ hp₀
    exact e.isEmbedding.t2Space
  exact @key (![1, 0, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.coset_space_hausdorff (![0, 1, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere), T2Space (Quotient (orbitSetoid p₀)) := by
    intro p₀ hp₀
    obtain ⟨e, -⟩ := QuotientTopology.sphere_quotient_homeomorph p₀ hp₀
    exact e.isEmbedding.t2Space
  exact @key (![0, 1, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Hausdorff.coset_space_hausdorff
-- test: set-coset-hausdorff-test.sage

/-! ## exr-hausdorff-points-closed -/

/-- In a Hausdorff space every point is closed. -/
theorem points_closed (X : Type) [TopologicalSpace X] [T2Space X] (p : X) : IsClosed ({p} : Set X) :=
  isClosed_singleton

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.points_closed (ℝ) (inferInstance) (inferInstance) (0) := by
  exact isClosed_singleton

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.points_closed (ℝ × ℝ) (inferInstance) (inferInstance) ((1, 2)) := by
  exact isClosed_singleton

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.points_closed (ℕ) (inferInstance) (inferInstance) (5) := by
  exact isClosed_singleton

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Hausdorff.points_closed
-- test: set-points-closed-test.sage

/-- The converse fails: on an infinite set the cofinite topology has closed points but is not Hausdorff (two non-empty
open sets always meet). -/
theorem cofinite_not_hausdorff (X : Type) [Infinite X] :
    T1Space (CofiniteTopology X) ∧ ¬ T2Space (CofiniteTopology X) := by
  refine ⟨inferInstance, fun h => ?_⟩
  obtain ⟨x, y, hxy⟩ := exists_pair_ne X
  obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := @t2_separation (CofiniteTopology X) _ h
    (CofiniteTopology.of x) (CofiniteTopology.of y) (CofiniteTopology.of.injective.ne hxy)
  rw [CofiniteTopology.isOpen_iff'] at hU hV
  rcases hU with rfl | hU
  · exact hxU
  rcases hV with rfl | hV
  · exact hyV
  have : (Uᶜ ∪ Vᶜ).Finite := hU.union hV
  rw [← compl_inter, Set.disjoint_iff_inter_eq_empty.1 hUV, compl_empty] at this
  exact Set.infinite_univ this

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.cofinite_not_hausdorff (ℕ) (inferInstance) := by
  have key : ∀ (X : Type) [inst1 : Infinite X], T1Space (CofiniteTopology X) ∧ ¬ T2Space (CofiniteTopology X) := by
    intro X inst1
    refine ⟨inferInstance, fun h => ?_⟩
    obtain ⟨x, y, hxy⟩ := exists_pair_ne X
    obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := @t2_separation (CofiniteTopology X) _ h
      (CofiniteTopology.of x) (CofiniteTopology.of y) (CofiniteTopology.of.injective.ne hxy)
    rw [CofiniteTopology.isOpen_iff'] at hU hV
    rcases hU with rfl | hU
    · exact hxU
    rcases hV with rfl | hV
    · exact hyV
    have : (Uᶜ ∪ Vᶜ).Finite := hU.union hV
    rw [← compl_inter, Set.disjoint_iff_inter_eq_empty.1 hUV, compl_empty] at this
    exact Set.infinite_univ this
  exact @key (ℕ) (inferInstance)

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.cofinite_not_hausdorff (ℤ) (inferInstance) := by
  have key : ∀ (X : Type) [inst1 : Infinite X], T1Space (CofiniteTopology X) ∧ ¬ T2Space (CofiniteTopology X) := by
    intro X inst1
    refine ⟨inferInstance, fun h => ?_⟩
    obtain ⟨x, y, hxy⟩ := exists_pair_ne X
    obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := @t2_separation (CofiniteTopology X) _ h
      (CofiniteTopology.of x) (CofiniteTopology.of y) (CofiniteTopology.of.injective.ne hxy)
    rw [CofiniteTopology.isOpen_iff'] at hU hV
    rcases hU with rfl | hU
    · exact hxU
    rcases hV with rfl | hV
    · exact hyV
    have : (Uᶜ ∪ Vᶜ).Finite := hU.union hV
    rw [← compl_inter, Set.disjoint_iff_inter_eq_empty.1 hUV, compl_empty] at this
    exact Set.infinite_univ this
  exact @key (ℤ) (inferInstance)

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.cofinite_not_hausdorff (ℚ) (inferInstance) := by
  have key : ∀ (X : Type) [inst1 : Infinite X], T1Space (CofiniteTopology X) ∧ ¬ T2Space (CofiniteTopology X) := by
    intro X inst1
    refine ⟨inferInstance, fun h => ?_⟩
    obtain ⟨x, y, hxy⟩ := exists_pair_ne X
    obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := @t2_separation (CofiniteTopology X) _ h
      (CofiniteTopology.of x) (CofiniteTopology.of y) (CofiniteTopology.of.injective.ne hxy)
    rw [CofiniteTopology.isOpen_iff'] at hU hV
    rcases hU with rfl | hU
    · exact hxU
    rcases hV with rfl | hV
    · exact hyV
    have : (Uᶜ ∪ Vᶜ).Finite := hU.union hV
    rw [← compl_inter, Set.disjoint_iff_inter_eq_empty.1 hUV, compl_empty] at this
    exact Set.infinite_univ this
  exact @key (ℚ) (inferInstance)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Hausdorff.cofinite_not_hausdorff
-- test: set-cofinite-test.sage

/-! ## exr-hausdorff-two-origins -/

/-- Two copies of `ℝ` glued along `x ≠ c`: `(x, a) ∼ (y, b)` iff `x = y` and (`a = b` or `x ≠ c`). -/
def doubledSetoid (c : ℝ) : Setoid (ℝ × Bool) where
  r p q := p.1 = q.1 ∧ (p.2 = q.2 ∨ p.1 ≠ c)
  iseqv := ⟨fun p => ⟨rfl, Or.inl rfl⟩, fun ⟨h1, h2⟩ => ⟨h1.symm, h2.elim (fun h => Or.inl h.symm)
      (fun h => Or.inr (h1 ▸ h))⟩,
    fun ⟨h1, h2⟩ ⟨h3, h4⟩ => ⟨h1.trans h3, by
      rcases h2 with h2 | h2
      · rcases h4 with h4 | h4
        · exact Or.inl (h2.trans h4)
        · exact Or.inr (h1 ▸ h4)
      · exact Or.inr h2⟩⟩

/-- The line with the point `c` doubled is not Hausdorff: the two copies of `c` are distinct, but any neighbourhoods of
them meet (the line with two origins, `c = 0`). -/
theorem doubled_point_not_hausdorff (c : ℝ) :
    Quotient.mk (doubledSetoid c) (c, false) ≠ Quotient.mk (doubledSetoid c) (c, true) ∧
      ¬ T2Space (Quotient (doubledSetoid c)) := by
  have hne : Quotient.mk (doubledSetoid c) (c, false) ≠ Quotient.mk (doubledSetoid c) (c, true) := by
    intro h
    obtain ⟨-, h2⟩ := Quotient.exact h
    rcases h2 with h2 | h2
    · exact Bool.false_ne_true h2
    · exact h2 rfl
  refine ⟨hne, fun hT => ?_⟩
  obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := t2_separation hne
  have hU' : IsOpen (Quotient.mk (doubledSetoid c) ⁻¹' U) := isQuotientMap_quotient_mk'.isOpen_preimage.2 hU
  have hV' : IsOpen (Quotient.mk (doubledSetoid c) ⁻¹' V) := isQuotientMap_quotient_mk'.isOpen_preimage.2 hV
  -- the open sets contain the points `(c + t, false)` and `(c + t, true)` for small `t > 0`
  have h1 : ∀ᶠ t in 𝓝[>] (0 : ℝ), ((c + t, false) : ℝ × Bool) ∈ Quotient.mk (doubledSetoid c) ⁻¹' U := by
    have hcont : Continuous (fun t : ℝ => ((c + t, false) : ℝ × Bool)) := by fun_prop
    exact (hcont.tendsto' 0 (c, false) (by simp)).eventually (hU'.mem_nhds hxU) |>.filter_mono nhdsWithin_le_nhds
  have h2 : ∀ᶠ t in 𝓝[>] (0 : ℝ), ((c + t, true) : ℝ × Bool) ∈ Quotient.mk (doubledSetoid c) ⁻¹' V := by
    have hcont : Continuous (fun t : ℝ => ((c + t, true) : ℝ × Bool)) := by fun_prop
    exact (hcont.tendsto' 0 (c, true) (by simp)).eventually (hV'.mem_nhds hyV) |>.filter_mono nhdsWithin_le_nhds
  obtain ⟨t, ⟨ht1, ht2⟩, ht0⟩ := ((h1.and h2).and self_mem_nhdsWithin).exists
  simp only [mem_preimage, mem_Ioi] at ht1 ht2 ht0
  have heq : Quotient.mk (doubledSetoid c) (c + t, false) = Quotient.mk (doubledSetoid c) (c + t, true) :=
    Quotient.sound ⟨rfl, Or.inr (by simp only; linarith)⟩
  rw [heq] at ht1
  exact Set.disjoint_left.1 hUV ht1 ht2

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.doubled_point_not_hausdorff (0) := by
  have key : ∀ (c : ℝ), Quotient.mk (doubledSetoid c) (c, false) ≠ Quotient.mk (doubledSetoid c) (c, true) ∧ ¬ T2Space (Quotient (doubledSetoid c)) := by
    intro c
    have hne : Quotient.mk (doubledSetoid c) (c, false) ≠ Quotient.mk (doubledSetoid c) (c, true) := by
      intro h
      obtain ⟨-, h2⟩ := Quotient.exact h
      rcases h2 with h2 | h2
      · exact Bool.false_ne_true h2
      · exact h2 rfl
    refine ⟨hne, fun hT => ?_⟩
    obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := t2_separation hne
    have hU' : IsOpen (Quotient.mk (doubledSetoid c) ⁻¹' U) := isQuotientMap_quotient_mk'.isOpen_preimage.2 hU
    have hV' : IsOpen (Quotient.mk (doubledSetoid c) ⁻¹' V) := isQuotientMap_quotient_mk'.isOpen_preimage.2 hV
    -- the open sets contain the points `(c + t, false)` and `(c + t, true)` for small `t > 0`
    have h1 : ∀ᶠ t in 𝓝[>] (0 : ℝ), ((c + t, false) : ℝ × Bool) ∈ Quotient.mk (doubledSetoid c) ⁻¹' U := by
      have hcont : Continuous (fun t : ℝ => ((c + t, false) : ℝ × Bool)) := by fun_prop
      exact (hcont.tendsto' 0 (c, false) (by simp)).eventually (hU'.mem_nhds hxU) |>.filter_mono nhdsWithin_le_nhds
    have h2 : ∀ᶠ t in 𝓝[>] (0 : ℝ), ((c + t, true) : ℝ × Bool) ∈ Quotient.mk (doubledSetoid c) ⁻¹' V := by
      have hcont : Continuous (fun t : ℝ => ((c + t, true) : ℝ × Bool)) := by fun_prop
      exact (hcont.tendsto' 0 (c, true) (by simp)).eventually (hV'.mem_nhds hyV) |>.filter_mono nhdsWithin_le_nhds
    obtain ⟨t, ⟨ht1, ht2⟩, ht0⟩ := ((h1.and h2).and self_mem_nhdsWithin).exists
    simp only [mem_preimage, mem_Ioi] at ht1 ht2 ht0
    have heq : Quotient.mk (doubledSetoid c) (c + t, false) = Quotient.mk (doubledSetoid c) (c + t, true) :=
      Quotient.sound ⟨rfl, Or.inr (by simp only; linarith)⟩
    rw [heq] at ht1
    exact Set.disjoint_left.1 hUV ht1 ht2
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.doubled_point_not_hausdorff (1) := by
  have key : ∀ (c : ℝ), Quotient.mk (doubledSetoid c) (c, false) ≠ Quotient.mk (doubledSetoid c) (c, true) ∧ ¬ T2Space (Quotient (doubledSetoid c)) := by
    intro c
    have hne : Quotient.mk (doubledSetoid c) (c, false) ≠ Quotient.mk (doubledSetoid c) (c, true) := by
      intro h
      obtain ⟨-, h2⟩ := Quotient.exact h
      rcases h2 with h2 | h2
      · exact Bool.false_ne_true h2
      · exact h2 rfl
    refine ⟨hne, fun hT => ?_⟩
    obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := t2_separation hne
    have hU' : IsOpen (Quotient.mk (doubledSetoid c) ⁻¹' U) := isQuotientMap_quotient_mk'.isOpen_preimage.2 hU
    have hV' : IsOpen (Quotient.mk (doubledSetoid c) ⁻¹' V) := isQuotientMap_quotient_mk'.isOpen_preimage.2 hV
    -- the open sets contain the points `(c + t, false)` and `(c + t, true)` for small `t > 0`
    have h1 : ∀ᶠ t in 𝓝[>] (0 : ℝ), ((c + t, false) : ℝ × Bool) ∈ Quotient.mk (doubledSetoid c) ⁻¹' U := by
      have hcont : Continuous (fun t : ℝ => ((c + t, false) : ℝ × Bool)) := by fun_prop
      exact (hcont.tendsto' 0 (c, false) (by simp)).eventually (hU'.mem_nhds hxU) |>.filter_mono nhdsWithin_le_nhds
    have h2 : ∀ᶠ t in 𝓝[>] (0 : ℝ), ((c + t, true) : ℝ × Bool) ∈ Quotient.mk (doubledSetoid c) ⁻¹' V := by
      have hcont : Continuous (fun t : ℝ => ((c + t, true) : ℝ × Bool)) := by fun_prop
      exact (hcont.tendsto' 0 (c, true) (by simp)).eventually (hV'.mem_nhds hyV) |>.filter_mono nhdsWithin_le_nhds
    obtain ⟨t, ⟨ht1, ht2⟩, ht0⟩ := ((h1.and h2).and self_mem_nhdsWithin).exists
    simp only [mem_preimage, mem_Ioi] at ht1 ht2 ht0
    have heq : Quotient.mk (doubledSetoid c) (c + t, false) = Quotient.mk (doubledSetoid c) (c + t, true) :=
      Quotient.sound ⟨rfl, Or.inr (by simp only; linarith)⟩
    rw [heq] at ht1
    exact Set.disjoint_left.1 hUV ht1 ht2
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.doubled_point_not_hausdorff (-1) := by
  have key : ∀ (c : ℝ), Quotient.mk (doubledSetoid c) (c, false) ≠ Quotient.mk (doubledSetoid c) (c, true) ∧ ¬ T2Space (Quotient (doubledSetoid c)) := by
    intro c
    have hne : Quotient.mk (doubledSetoid c) (c, false) ≠ Quotient.mk (doubledSetoid c) (c, true) := by
      intro h
      obtain ⟨-, h2⟩ := Quotient.exact h
      rcases h2 with h2 | h2
      · exact Bool.false_ne_true h2
      · exact h2 rfl
    refine ⟨hne, fun hT => ?_⟩
    obtain ⟨U, V, hU, hV, hxU, hyV, hUV⟩ := t2_separation hne
    have hU' : IsOpen (Quotient.mk (doubledSetoid c) ⁻¹' U) := isQuotientMap_quotient_mk'.isOpen_preimage.2 hU
    have hV' : IsOpen (Quotient.mk (doubledSetoid c) ⁻¹' V) := isQuotientMap_quotient_mk'.isOpen_preimage.2 hV
    -- the open sets contain the points `(c + t, false)` and `(c + t, true)` for small `t > 0`
    have h1 : ∀ᶠ t in 𝓝[>] (0 : ℝ), ((c + t, false) : ℝ × Bool) ∈ Quotient.mk (doubledSetoid c) ⁻¹' U := by
      have hcont : Continuous (fun t : ℝ => ((c + t, false) : ℝ × Bool)) := by fun_prop
      exact (hcont.tendsto' 0 (c, false) (by simp)).eventually (hU'.mem_nhds hxU) |>.filter_mono nhdsWithin_le_nhds
    have h2 : ∀ᶠ t in 𝓝[>] (0 : ℝ), ((c + t, true) : ℝ × Bool) ∈ Quotient.mk (doubledSetoid c) ⁻¹' V := by
      have hcont : Continuous (fun t : ℝ => ((c + t, true) : ℝ × Bool)) := by fun_prop
      exact (hcont.tendsto' 0 (c, true) (by simp)).eventually (hV'.mem_nhds hyV) |>.filter_mono nhdsWithin_le_nhds
    obtain ⟨t, ⟨ht1, ht2⟩, ht0⟩ := ((h1.and h2).and self_mem_nhdsWithin).exists
    simp only [mem_preimage, mem_Ioi] at ht1 ht2 ht0
    have heq : Quotient.mk (doubledSetoid c) (c + t, false) = Quotient.mk (doubledSetoid c) (c + t, true) :=
      Quotient.sound ⟨rfl, Or.inr (by simp only; linarith)⟩
    rw [heq] at ht1
    exact Set.disjoint_left.1 hUV ht1 ht2
  exact @key (-1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Hausdorff.doubled_point_not_hausdorff
-- test: set-doubled-point-test.sage

/-! ## exm-hausdorff-discrete-general -/

/-- A finite Hausdorff space is discrete: the discrete topology is the only Hausdorff topology on a finite set. -/
theorem finite_hausdorff_discrete (X : Type) [Finite X] [TopologicalSpace X] [T2Space X] : DiscreteTopology X :=
  inferInstance

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.finite_hausdorff_discrete (Fin 3) (inferInstance) (⊥) ((@DiscreteTopology.toT2Space _ ⊥ ⟨rfl⟩)) := by
  exact ⟨rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.finite_hausdorff_discrete (Fin 1) (inferInstance) (⊥) ((@DiscreteTopology.toT2Space _ ⊥ ⟨rfl⟩)) := by
  exact ⟨rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.finite_hausdorff_discrete (Bool) (inferInstance) (⊥) ((@DiscreteTopology.toT2Space _ ⊥ ⟨rfl⟩)) := by
  exact ⟨rfl⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Hausdorff.finite_hausdorff_discrete
-- test: set-finite-hausdorff-test.sage

/-! ## exm-hausdorff-infinite -/

/-- On `ℝⁿ` (`n ≥ 1`) the usual topology and the discrete one are both Hausdorff, and they differ (the usual one
is not discrete). -/
theorem two_hausdorff_topologies (n : ℕ) (hn : 1 ≤ n) :
    T2Space (EuclideanSpace ℝ (Fin n)) ∧ @T2Space (EuclideanSpace ℝ (Fin n)) ⊥ ∧
      ¬ DiscreteTopology (EuclideanSpace ℝ (Fin n)) := by
  refine ⟨inferInstance, by letI : TopologicalSpace (EuclideanSpace ℝ (Fin n)) := ⊥; haveI : DiscreteTopology (EuclideanSpace ℝ (Fin n)) := ⟨rfl⟩; infer_instance, fun h => ?_⟩
  haveI : Nontrivial (EuclideanSpace ℝ (Fin n)) := by
    haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
    infer_instance
  exact not_isOpen_singleton (0 : EuclideanSpace ℝ (Fin n)) (isOpen_discrete _)

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.two_hausdorff_topologies (1) ((by norm_num)) := by
  have key : ∀ (n : ℕ) (hn : 1 ≤ n), T2Space (EuclideanSpace ℝ (Fin n)) ∧ @T2Space (EuclideanSpace ℝ (Fin n)) ⊥ ∧ ¬ DiscreteTopology (EuclideanSpace ℝ (Fin n)) := by
    intro n hn
    refine ⟨inferInstance, by letI : TopologicalSpace (EuclideanSpace ℝ (Fin n)) := ⊥; haveI : DiscreteTopology (EuclideanSpace ℝ (Fin n)) := ⟨rfl⟩; infer_instance, fun h => ?_⟩
    haveI : Nontrivial (EuclideanSpace ℝ (Fin n)) := by
      haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
      infer_instance
    exact not_isOpen_singleton (0 : EuclideanSpace ℝ (Fin n)) (isOpen_discrete _)
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.two_hausdorff_topologies (2) ((by norm_num)) := by
  have key : ∀ (n : ℕ) (hn : 1 ≤ n), T2Space (EuclideanSpace ℝ (Fin n)) ∧ @T2Space (EuclideanSpace ℝ (Fin n)) ⊥ ∧ ¬ DiscreteTopology (EuclideanSpace ℝ (Fin n)) := by
    intro n hn
    refine ⟨inferInstance, by letI : TopologicalSpace (EuclideanSpace ℝ (Fin n)) := ⊥; haveI : DiscreteTopology (EuclideanSpace ℝ (Fin n)) := ⟨rfl⟩; infer_instance, fun h => ?_⟩
    haveI : Nontrivial (EuclideanSpace ℝ (Fin n)) := by
      haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
      infer_instance
    exact not_isOpen_singleton (0 : EuclideanSpace ℝ (Fin n)) (isOpen_discrete _)
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Hausdorff.two_hausdorff_topologies (3) ((by norm_num)) := by
  have key : ∀ (n : ℕ) (hn : 1 ≤ n), T2Space (EuclideanSpace ℝ (Fin n)) ∧ @T2Space (EuclideanSpace ℝ (Fin n)) ⊥ ∧ ¬ DiscreteTopology (EuclideanSpace ℝ (Fin n)) := by
    intro n hn
    refine ⟨inferInstance, by letI : TopologicalSpace (EuclideanSpace ℝ (Fin n)) := ⊥; haveI : DiscreteTopology (EuclideanSpace ℝ (Fin n)) := ⟨rfl⟩; infer_instance, fun h => ?_⟩
    haveI : Nontrivial (EuclideanSpace ℝ (Fin n)) := by
      haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
      infer_instance
    exact not_isOpen_singleton (0 : EuclideanSpace ℝ (Fin n)) (isOpen_discrete _)
  exact @key (3) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Hausdorff.two_hausdorff_topologies
-- test: set-two-hausdorff-test.sage

end KaluzaKlein.Sets.Hausdorff
