import Mathlib
import KaluzaKlein.PartI.Sets.QuotientTopology
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, the topologies on a finite set

The claims of the blocks `prp-finite-topologies` to `exr-topology-four-points` and the finite counts of the
Hausdorff blocks (`exm-hausdorff-finite`, `exr-hausdorff-check-finite`, `prp-hausdorff-topologies`,
`exm-hausdorff-two-points`, `exr-hausdorff-count-t1`, `exr-hausdorff-sierpinski`) of
`tutorial/part-i-01-sets-relations-topology.qmd`.

A topology on a finite type `α` is the same as a family `F : Finset (Finset α)` with `IsFinTop F`
(`topEquiv`), so the topologies on `n` points are counted by enumerating such families: Sage lists them, and the
kernel checks by evaluation that the list is complete and without repetition.
-/

namespace KaluzaKlein.Sets.FiniteTopologies

open Set Topology KaluzaKlein.Sets.TopologicalSpaces

/-- The topologies on a finite type are the families satisfying the finite axioms: `t ↦ {U | U open}`. -/
noncomputable def topEquiv (α : Type) [Fintype α] [DecidableEq α] :
    TopologicalSpace α ≃ {F : Finset (Finset α) // IsFinTop F} := by
  classical
  exact {
    toFun := fun t => ⟨Finset.univ.filter (fun U : Finset α => @IsOpen α t (U : Set α)), by
      letI := t
      refine ⟨by simp, by simp, fun U hU V hV => ?_⟩
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.coe_union, Finset.coe_inter] at hU hV ⊢
      exact ⟨hU.union hV, hU.inter hV⟩⟩
    invFun := fun F => {
      IsOpen := fun s => ∃ U ∈ F.1, (U : Set α) = s
      isOpen_univ := ⟨Finset.univ, F.2.2.1, by simp⟩
      isOpen_inter := by
        rintro _ _ ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
        exact ⟨U ∩ V, (F.2.2.2 U hU V hV).2, by simp⟩
      isOpen_sUnion := fun S hS => by
        have h := ((isFinTop_iff F.1).1 F.2).2.2.2 S (fun s hs => by
          obtain ⟨U, hU, rfl⟩ := hS s hs; exact ⟨U, hU, rfl⟩)
        obtain ⟨U, hU, hU'⟩ := h
        exact ⟨U, hU, hU'⟩ }
    left_inv := fun t => by
      ext s
      constructor
      · rintro ⟨U, hU, rfl⟩
        exact (Finset.mem_filter.1 hU).2
      · intro hs
        exact ⟨s.toFinset, Finset.mem_filter.2 ⟨Finset.mem_univ _, by simpa using hs⟩, by simp⟩
    right_inv := fun F => by
      apply Subtype.ext
      ext U
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      constructor
      · rintro ⟨V, hV, hVU⟩
        rwa [Finset.coe_inj.1 hVU] at hV
      · intro hU
        exact ⟨U, hU, rfl⟩ }

/-! ## prp-finite-topologies -/

/-- The number of topologies on a finite type is the number of families satisfying the finite axioms. -/
theorem card_topologies (α : Type) [Fintype α] [DecidableEq α] :
    Nat.card (TopologicalSpace α) = (Finset.univ.filter fun F : Finset (Finset α) => IsFinTop F).card := by
  rw [Nat.card_congr (topEquiv α), Nat.card_eq_fintype_card, Fintype.card_subtype]

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.card_topologies (Fin 0) (inferInstance) (inferInstance) := by
  have key : ∀ (α : Type) [inst1 : Fintype α] [inst2 : DecidableEq α], Nat.card (TopologicalSpace α) = (Finset.univ.filter fun F : Finset (Finset α) => IsFinTop F).card := by
    intro α inst1 inst2
    rw [Nat.card_congr (topEquiv α), Nat.card_eq_fintype_card, Fintype.card_subtype]
  exact @key (Fin 0) (inferInstance) (inferInstance)

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.card_topologies (Fin 1) (inferInstance) (inferInstance) := by
  have key : ∀ (α : Type) [inst1 : Fintype α] [inst2 : DecidableEq α], Nat.card (TopologicalSpace α) = (Finset.univ.filter fun F : Finset (Finset α) => IsFinTop F).card := by
    intro α inst1 inst2
    rw [Nat.card_congr (topEquiv α), Nat.card_eq_fintype_card, Fintype.card_subtype]
  exact @key (Fin 1) (inferInstance) (inferInstance)

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.card_topologies (Fin 2) (inferInstance) (inferInstance) := by
  have key : ∀ (α : Type) [inst1 : Fintype α] [inst2 : DecidableEq α], Nat.card (TopologicalSpace α) = (Finset.univ.filter fun F : Finset (Finset α) => IsFinTop F).card := by
    intro α inst1 inst2
    rw [Nat.card_congr (topEquiv α), Nat.card_eq_fintype_card, Fintype.card_subtype]
  exact @key (Fin 2) (inferInstance) (inferInstance)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.FiniteTopologies.card_topologies
-- test: set-card-topologies-test.sage

/-- The topologies on three points: Sage lists `29` families, without repetition, and they are exactly the families
satisfying the axioms. -/
theorem topologies_three : ∃ L : List (Finset (Finset (Fin 3))), L.Nodup ∧ L.length = 29 ∧
    ∀ F : Finset (Finset (Fin 3)), F ∈ L ↔ IsFinTop F := by
  veriphy_certified
    (set_option maxRecDepth 100000 in
        exact
          ⟨[{∅, {0, 1, 2}}, {∅, {0}, {0, 1, 2}}, {∅, { 1 }, {0, 1, 2}}, {∅, { 2 }, {0, 1, 2}}, {∅, {0, 1}, {0, 1, 2}},
              {∅, {0, 2}, {0, 1, 2}}, {∅, { 1, 2 }, {0, 1, 2}}, {∅, {0}, {0, 1}, {0, 1, 2}},
              {∅, {0}, {0, 2}, {0, 1, 2}}, {∅, {0}, { 1, 2 }, {0, 1, 2}}, {∅, { 1 }, {0, 1}, {0, 1, 2}},
              {∅, { 1 }, {0, 2}, {0, 1, 2}}, {∅, { 1 }, { 1, 2 }, {0, 1, 2}}, {∅, { 2 }, {0, 1}, {0, 1, 2}},
              {∅, { 2 }, {0, 2}, {0, 1, 2}}, {∅, { 2 }, { 1, 2 }, {0, 1, 2}}, {∅, {0}, { 1 }, {0, 1}, {0, 1, 2}},
              {∅, {0}, { 2 }, {0, 2}, {0, 1, 2}}, {∅, {0}, {0, 1}, {0, 2}, {0, 1, 2}},
              {∅, { 1 }, { 2 }, { 1, 2 }, {0, 1, 2}}, {∅, { 1 }, {0, 1}, { 1, 2 }, {0, 1, 2}},
              {∅, { 2 }, {0, 2}, { 1, 2 }, {0, 1, 2}}, {∅, {0}, { 1 }, {0, 1}, {0, 2}, {0, 1, 2}},
              {∅, {0}, { 1 }, {0, 1}, { 1, 2 }, {0, 1, 2}}, {∅, {0}, { 2 }, {0, 1}, {0, 2}, {0, 1, 2}},
              {∅, {0}, { 2 }, {0, 2}, { 1, 2 }, {0, 1, 2}}, {∅, { 1 }, { 2 }, {0, 1}, { 1, 2 }, {0, 1, 2}},
              {∅, { 1 }, { 2 }, {0, 2}, { 1, 2 }, {0, 1, 2}},
              {∅, {0}, { 1 }, { 2 }, {0, 1}, {0, 2}, { 1, 2 }, {0, 1, 2}}],
            by decide, rfl, by decide⟩)
  -- certificate: set-topologies-three.sage

/-! ## exm-topologies-lattice -/

/-- Every topology on a finite set lies between the indiscrete one `{∅, X}` and the discrete one (all subsets). -/
theorem between_indiscrete_discrete {α : Type} [Fintype α] [DecidableEq α] (F : Finset (Finset α))
    (hF : IsFinTop F) : ({∅, Finset.univ} : Finset (Finset α)) ⊆ F ∧ F ⊆ Finset.univ := by
  refine ⟨?_, Finset.subset_univ _⟩
  intro U hU
  simp only [Finset.mem_insert, Finset.mem_singleton] at hU
  rcases hU with rfl | rfl
  · exact hF.1
  · exact hF.2.1

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.between_indiscrete_discrete (Fin 3) (inferInstance) (inferInstance) ({∅, {0}, {0, 1}, Finset.univ}) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.between_indiscrete_discrete (Fin 0) (inferInstance) (inferInstance) ({∅}) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.between_indiscrete_discrete (Fin 2) (inferInstance) (inferInstance) (Finset.univ) ((by decide)) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.FiniteTopologies.between_indiscrete_discrete
-- test: set-between-test.sage

/-- A finer topology on the domain keeps every continuous map continuous; a finer topology on the codomain can only
lose continuous maps (a map continuous into the finer one is continuous into the coarser one). -/
theorem finer_coarser (X Y : Type) (t₁ t₂ : TopologicalSpace X) (s₁ s₂ : TopologicalSpace Y) (f : X → Y)
    (ht : t₂ ≤ t₁) (hs : s₂ ≤ s₁) :
    (Continuous[t₁, s₁] f → Continuous[t₂, s₁] f) ∧ (Continuous[t₁, s₂] f → Continuous[t₁, s₁] f) :=
  ⟨fun h => continuous_le_dom ht h, fun h => continuous_le_rng hs h⟩

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.finer_coarser (ℝ) (ℝ) (⊤) (⊥) (inferInstance) (⊥) (id) (bot_le) (bot_le) := by
  have key : ∀ (X Y : Type) (t₁ t₂ : TopologicalSpace X) (s₁ s₂ : TopologicalSpace Y) (f : X → Y) (ht : t₂ ≤ t₁) (hs : s₂ ≤ s₁), (Continuous[t₁, s₁] f → Continuous[t₂, s₁] f) ∧ (Continuous[t₁, s₂] f → Continuous[t₁, s₁] f) := by
    intro X Y t₁ t₂ s₁ s₂ f ht hs
    exact (⟨fun h => continuous_le_dom ht h, fun h => continuous_le_rng hs h⟩)
  exact @key (ℝ) (ℝ) (⊤) (⊥) (inferInstance) (⊥) (id) (bot_le) (bot_le)

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.finer_coarser (Unit) (Unit) (⊥) (⊥) (⊥) (⊥) (id) (le_rfl) (le_rfl) := by
  have key : ∀ (X Y : Type) (t₁ t₂ : TopologicalSpace X) (s₁ s₂ : TopologicalSpace Y) (f : X → Y) (ht : t₂ ≤ t₁) (hs : s₂ ≤ s₁), (Continuous[t₁, s₁] f → Continuous[t₂, s₁] f) ∧ (Continuous[t₁, s₂] f → Continuous[t₁, s₁] f) := by
    intro X Y t₁ t₂ s₁ s₂ f ht hs
    exact (⟨fun h => continuous_le_dom ht h, fun h => continuous_le_rng hs h⟩)
  exact @key (Unit) (Unit) (⊥) (⊥) (⊥) (⊥) (id) (le_rfl) (le_rfl)

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.finer_coarser (Bool) (Bool) (⊤) (⊤) (⊤) (⊥) (not) (le_rfl) (bot_le) := by
  have key : ∀ (X Y : Type) (t₁ t₂ : TopologicalSpace X) (s₁ s₂ : TopologicalSpace Y) (f : X → Y) (ht : t₂ ≤ t₁) (hs : s₂ ≤ s₁), (Continuous[t₁, s₁] f → Continuous[t₂, s₁] f) ∧ (Continuous[t₁, s₂] f → Continuous[t₁, s₁] f) := by
    intro X Y t₁ t₂ s₁ s₂ f ht hs
    exact (⟨fun h => continuous_le_dom ht h, fun h => continuous_le_rng hs h⟩)
  exact @key (Bool) (Bool) (⊤) (⊤) (⊤) (⊥) (not) (le_rfl) (bot_le)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.FiniteTopologies.finer_coarser
-- test: set-finer-coarser-test.sage

/-! ## exm-topologies-count-grows

The counts on four and five points are beyond evaluation by the kernel (`2¹⁶` and `2³²` families); the claim is
stated and tested by Sage at every size up to five, the sizes `0, …, 3` also by the kernel. -/

/-- The number of topologies on `n` points, for `n ≤ 5`, is `1, 1, 4, 29, 355, 6942`. -/
def topologies_count_claim : Prop := ∀ n : Fin 6,
  (Finset.univ.filter fun F : Finset (Finset (Fin n)) => IsFinTop F).card = ![1, 1, 4, 29, 355, 6942] n

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.topologies_count_claim (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.topologies_count_claim (1) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.topologies_count_claim (2) := by
  decide

attribute [veriphy_tested oracle_test 3 3] KaluzaKlein.Sets.FiniteTopologies.topologies_count_claim
-- test: set-topologies-count-test.sage

/-! ## exr-finite-topologies -/

/-- There are `2^(2ⁿ)` families of subsets of an `n`-element set: `2⁸ = 256` for three points, `2¹⁶ = 65536` for
four. -/
theorem card_families (n : ℕ) : Fintype.card (Finset (Finset (Fin n))) = 2 ^ 2 ^ n := by
  simp [Fintype.card_finset]

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.card_families (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.card_families (1) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.card_families (2) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.card_families (3) := by
  simp [Fintype.card_finset]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.Sets.FiniteTopologies.card_families
-- test: set-card-families-test.sage

/-! ## exr-topology-t0-count -/

/-- A family separates points (`T₀`): any two distinct points have an open set containing one and not the other. -/
def IsT0Fin {α : Type} [Fintype α] [DecidableEq α] (F : Finset (Finset α)) : Prop :=
  ∀ x y : α, x ≠ y → ∃ U ∈ F, (x ∈ U ∧ y ∉ U) ∨ (y ∈ U ∧ x ∉ U)

instance {α : Type} [Fintype α] [DecidableEq α] (F : Finset (Finset α)) : Decidable (IsT0Fin F) := by
  unfold IsT0Fin; infer_instance

/-- The `T₀` topologies on three points: Sage lists `19`, and they are exactly the topologies that separate points. -/
theorem t0_topologies_three : ∃ L : List (Finset (Finset (Fin 3))), L.Nodup ∧ L.length = 19 ∧
    ∀ F : Finset (Finset (Fin 3)), F ∈ L ↔ IsFinTop F ∧ IsT0Fin F := by
  veriphy_certified
    (set_option maxRecDepth 100000 in
        exact
          ⟨[{∅, {0}, {0, 1}, {0, 1, 2}}, {∅, {0}, {0, 2}, {0, 1, 2}}, {∅, { 1 }, {0, 1}, {0, 1, 2}},
              {∅, { 1 }, { 1, 2 }, {0, 1, 2}}, {∅, { 2 }, {0, 2}, {0, 1, 2}}, {∅, { 2 }, { 1, 2 }, {0, 1, 2}},
              {∅, {0}, { 1 }, {0, 1}, {0, 1, 2}}, {∅, {0}, { 2 }, {0, 2}, {0, 1, 2}},
              {∅, {0}, {0, 1}, {0, 2}, {0, 1, 2}}, {∅, { 1 }, { 2 }, { 1, 2 }, {0, 1, 2}},
              {∅, { 1 }, {0, 1}, { 1, 2 }, {0, 1, 2}}, {∅, { 2 }, {0, 2}, { 1, 2 }, {0, 1, 2}},
              {∅, {0}, { 1 }, {0, 1}, {0, 2}, {0, 1, 2}}, {∅, {0}, { 1 }, {0, 1}, { 1, 2 }, {0, 1, 2}},
              {∅, {0}, { 2 }, {0, 1}, {0, 2}, {0, 1, 2}}, {∅, {0}, { 2 }, {0, 2}, { 1, 2 }, {0, 1, 2}},
              {∅, { 1 }, { 2 }, {0, 1}, { 1, 2 }, {0, 1, 2}}, {∅, { 1 }, { 2 }, {0, 2}, { 1, 2 }, {0, 1, 2}},
              {∅, {0}, { 1 }, { 2 }, {0, 1}, {0, 2}, { 1, 2 }, {0, 1, 2}}],
            by decide, rfl, by decide⟩)
  -- certificate: set-t0-three.sage

/-! ## Hausdorff topologies on two and three points -/

/-- A family is Hausdorff: any two distinct points have disjoint open sets containing them. -/
def IsHausdorffFin {α : Type} [Fintype α] [DecidableEq α] (F : Finset (Finset α)) : Prop :=
  ∀ x y : α, x ≠ y → ∃ U ∈ F, ∃ V ∈ F, x ∈ U ∧ y ∈ V ∧ Disjoint U V

instance {α : Type} [Fintype α] [DecidableEq α] (F : Finset (Finset α)) : Decidable (IsHausdorffFin F) := by
  unfold IsHausdorffFin; infer_instance

set_option maxRecDepth 100000 in
/-- On three points a family is a Hausdorff topology exactly when it is the discrete topology: one of the `29`. -/
theorem hausdorff_three (F : Finset (Finset (Fin 3))) : IsFinTop F ∧ IsHausdorffFin F ↔ F = Finset.univ := by
  revert F
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.hausdorff_three (Finset.univ) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.hausdorff_three ({∅, Finset.univ}) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.hausdorff_three ({∅, {0}, Finset.univ}) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.FiniteTopologies.hausdorff_three
-- test: set-hausdorff-three-test.sage

set_option maxRecDepth 100000 in
/-- On two points a family is a Hausdorff topology exactly when it is the discrete topology: one of the `4`. -/
theorem hausdorff_two (F : Finset (Finset (Fin 2))) : IsFinTop F ∧ IsHausdorffFin F ↔ F = Finset.univ := by
  revert F
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.hausdorff_two (Finset.univ) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.hausdorff_two ({∅, Finset.univ}) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.hausdorff_two ({∅, {0}, Finset.univ}) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.FiniteTopologies.hausdorff_two
-- test: set-hausdorff-two-test.sage

set_option maxRecDepth 100000 in
/-- On three points a topology in which every point is closed (the complement of each point open) is the discrete one,
as for the Hausdorff condition. -/
theorem t1_three (F : Finset (Finset (Fin 3))) :
    IsFinTop F ∧ (∀ x : Fin 3, ({x}ᶜ : Finset (Fin 3)) ∈ F) ↔ F = Finset.univ := by
  revert F
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.t1_three (Finset.univ) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.t1_three ({∅, Finset.univ}) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.t1_three ({∅, {0}, {0, 1}, Finset.univ}) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.FiniteTopologies.t1_three
-- test: set-t1-three-test.sage

/-! ## exr-hausdorff-sierpinski -/

/-- The Sierpiński topology `{∅, {1}, {1, 2}}` on `{1, 2}` (as `{0, 1}`). -/
def sierpinski : Finset (Finset (Fin 2)) := {∅, {0}, Finset.univ}

set_option maxRecDepth 100000 in
/-- In the Sierpiński space: it is a topology; every neighbourhood of `y` contains `x` exactly when `x = 1` or
`x = y` (the neighbourhoods of `1` are `{1}` and `{1, 2}`, the only one of `2` is `{1, 2}`; so the constant sequence
`1, 1, …` converges to both points); a point `x` is closed exactly when `x = 2`. -/
theorem sierpinski_facts (x y : Fin 2) :
    IsFinTop sierpinski ∧ ((∀ U ∈ sierpinski, y ∈ U → x ∈ U) ↔ (x = 0 ∨ x = y)) ∧
      (({x}ᶜ : Finset (Fin 2)) ∈ sierpinski ↔ x = 1) := by
  revert x y
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.sierpinski_facts (0) (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.sierpinski_facts (0) (1) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.sierpinski_facts (1) (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.FiniteTopologies.sierpinski_facts (1) (1) := by
  decide

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.Sets.FiniteTopologies.sierpinski_facts
-- test: set-sierpinski-test.sage

end KaluzaKlein.Sets.FiniteTopologies
