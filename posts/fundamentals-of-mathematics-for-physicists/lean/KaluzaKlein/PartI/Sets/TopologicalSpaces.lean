import Mathlib
import KaluzaKlein.PartI.Sets.OpenSets
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Topological spaces" (the axioms)

The claims of the blocks `def-topological-space` to `exr-topology-intersection` of
`tutorial/part-i-01-sets-relations-topology.qmd`: the topology of `ℝⁿ`, discrete and indiscrete topologies, families
of subsets of a finite set that are or are not topologies, and intersections and unions of topologies.

On a finite set a topology is a finite family of subsets, and arbitrary unions reduce to unions of two members:
`IsFinTop F` (for a `Finset` of `Finset`s) is the decidable form of the axioms, equivalent to `IsTopologyOn` by
`isFinTop_iff`.
-/

namespace KaluzaKlein.Sets.TopologicalSpaces

open Set Metric

/-- A topology on `X`: a family of subsets containing `∅` and `X`, closed under finite intersections and arbitrary
unions. -/
def IsTopologyOn (X : Type) (opens : Set (Set X)) : Prop :=
  ∅ ∈ opens ∧ univ ∈ opens ∧ (∀ U V, U ∈ opens → V ∈ opens → U ∩ V ∈ opens) ∧
    (∀ S : Set (Set X), S ⊆ opens → ⋃₀ S ∈ opens)

/-- The axioms of a topology for a family of subsets of a finite set: `∅` and the whole set, and pairwise unions and
intersections. -/
def IsFinTop {α : Type} [Fintype α] [DecidableEq α] (F : Finset (Finset α)) : Prop :=
  ∅ ∈ F ∧ Finset.univ ∈ F ∧ ∀ U ∈ F, ∀ V ∈ F, U ∪ V ∈ F ∧ U ∩ V ∈ F

instance {α : Type} [Fintype α] [DecidableEq α] (F : Finset (Finset α)) : Decidable (IsFinTop F) := by
  unfold IsFinTop; infer_instance

/-- A family of sets closed under binary unions and containing `∅` contains the union of each of its finite
subfamilies. -/
theorem sUnion_mem_of_finite {X : Type} (O : Set (Set X)) (h0 : ∅ ∈ O) (hu : ∀ U V, U ∈ O → V ∈ O → U ∪ V ∈ O)
    (S : Set (Set X)) (hS : S.Finite) (hSO : S ⊆ O) : ⋃₀ S ∈ O := by
  induction S, hS using Set.Finite.induction_on with
  | empty => simpa using h0
  | @insert a s _ _ ih =>
    rw [sUnion_insert]
    exact hu _ _ (hSO (mem_insert _ _)) (ih (fun x hx => hSO (mem_insert_of_mem _ hx)))

/-- On a finite set, the decidable axioms `IsFinTop` are the axioms of a topology. -/
theorem isFinTop_iff {α : Type} [Fintype α] [DecidableEq α] (F : Finset (Finset α)) :
    IsFinTop F ↔ IsTopologyOn α ((fun V : Finset α => (V : Set α)) '' (F : Set (Finset α))) := by
  constructor
  · rintro ⟨h0, h1, h2⟩
    have hu : ∀ U V, U ∈ (fun V : Finset α => (V : Set α)) '' (F : Set (Finset α)) →
        V ∈ (fun V : Finset α => (V : Set α)) '' (F : Set (Finset α)) →
        U ∪ V ∈ (fun V : Finset α => (V : Set α)) '' (F : Set (Finset α)) := by
      rintro _ _ ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
      exact ⟨U ∪ V, (h2 U hU V hV).1, by simp⟩
    refine ⟨⟨∅, h0, by simp⟩, ⟨Finset.univ, h1, by simp⟩, ?_, fun S hS => ?_⟩
    · rintro _ _ ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
      exact ⟨U ∩ V, (h2 U hU V hV).2, by simp⟩
    · have hfin : S.Finite := Set.Finite.subset ((F.finite_toSet).image _) hS
      exact sUnion_mem_of_finite ((fun V : Finset α => (V : Set α)) '' (F : Set (Finset α))) ⟨∅, h0, by simp⟩
        hu S hfin hS
  · rintro ⟨⟨E, hE, hE'⟩, ⟨T, hT, hT'⟩, hi, hs⟩
    have hE'' : E = ∅ := Finset.coe_eq_empty.1 hE'
    have hT'' : T = Finset.univ := Finset.coe_eq_univ.1 hT'
    refine ⟨hE'' ▸ hE, hT'' ▸ hT, fun U hU V hV => ⟨?_, ?_⟩⟩
    · obtain ⟨W, hW, hW'⟩ := hs {(U : Set α), (V : Set α)} (by
        rintro _ (rfl | rfl); exacts [⟨U, hU, rfl⟩, ⟨V, hV, rfl⟩])
      have : W = U ∪ V := Finset.coe_inj.1 (by simpa using hW')
      exact this ▸ hW
    · obtain ⟨W, hW, hW'⟩ := hi U V ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
      have : W = U ∩ V := Finset.coe_inj.1 (by simpa using hW')
      exact this ▸ hW

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.isFinTop_iff (Fin 2) (inferInstance) (inferInstance) ({∅, {0}, Finset.univ}) := by
  have key : ∀ {α : Type} [inst1 : Fintype α] [inst2 : DecidableEq α] (F : Finset (Finset α)), IsFinTop F ↔ IsTopologyOn α ((fun V : Finset α => (V : Set α)) '' (F : Set (Finset α))) := by
    intro α inst1 inst2 F
    constructor
    · rintro ⟨h0, h1, h2⟩
      have hu : ∀ U V, U ∈ (fun V : Finset α => (V : Set α)) '' (F : Set (Finset α)) →
          V ∈ (fun V : Finset α => (V : Set α)) '' (F : Set (Finset α)) →
          U ∪ V ∈ (fun V : Finset α => (V : Set α)) '' (F : Set (Finset α)) := by
        rintro _ _ ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
        exact ⟨U ∪ V, (h2 U hU V hV).1, by simp⟩
      refine ⟨⟨∅, h0, by simp⟩, ⟨Finset.univ, h1, by simp⟩, ?_, fun S hS => ?_⟩
      · rintro _ _ ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
        exact ⟨U ∩ V, (h2 U hU V hV).2, by simp⟩
      · have hfin : S.Finite := Set.Finite.subset ((F.finite_toSet).image _) hS
        exact sUnion_mem_of_finite ((fun V : Finset α => (V : Set α)) '' (F : Set (Finset α))) ⟨∅, h0, by simp⟩
          hu S hfin hS
    · rintro ⟨⟨E, hE, hE'⟩, ⟨T, hT, hT'⟩, hi, hs⟩
      have hE'' : E = ∅ := Finset.coe_eq_empty.1 hE'
      have hT'' : T = Finset.univ := Finset.coe_eq_univ.1 hT'
      refine ⟨hE'' ▸ hE, hT'' ▸ hT, fun U hU V hV => ⟨?_, ?_⟩⟩
      · obtain ⟨W, hW, hW'⟩ := hs {(U : Set α), (V : Set α)} (by
          rintro _ (rfl | rfl); exacts [⟨U, hU, rfl⟩, ⟨V, hV, rfl⟩])
        have : W = U ∪ V := Finset.coe_inj.1 (by simpa using hW')
        exact this ▸ hW
      · obtain ⟨W, hW, hW'⟩ := hi U V ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
        have : W = U ∩ V := Finset.coe_inj.1 (by simpa using hW')
        exact this ▸ hW
  exact @key (Fin 2) (inferInstance) (inferInstance) ({∅, {0}, Finset.univ})

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.isFinTop_iff (Fin 0) (inferInstance) (inferInstance) (∅) := by
  have key : ∀ {α : Type} [inst1 : Fintype α] [inst2 : DecidableEq α] (F : Finset (Finset α)), IsFinTop F ↔ IsTopologyOn α ((fun V : Finset α => (V : Set α)) '' (F : Set (Finset α))) := by
    intro α inst1 inst2 F
    constructor
    · rintro ⟨h0, h1, h2⟩
      have hu : ∀ U V, U ∈ (fun V : Finset α => (V : Set α)) '' (F : Set (Finset α)) →
          V ∈ (fun V : Finset α => (V : Set α)) '' (F : Set (Finset α)) →
          U ∪ V ∈ (fun V : Finset α => (V : Set α)) '' (F : Set (Finset α)) := by
        rintro _ _ ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
        exact ⟨U ∪ V, (h2 U hU V hV).1, by simp⟩
      refine ⟨⟨∅, h0, by simp⟩, ⟨Finset.univ, h1, by simp⟩, ?_, fun S hS => ?_⟩
      · rintro _ _ ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
        exact ⟨U ∩ V, (h2 U hU V hV).2, by simp⟩
      · have hfin : S.Finite := Set.Finite.subset ((F.finite_toSet).image _) hS
        exact sUnion_mem_of_finite ((fun V : Finset α => (V : Set α)) '' (F : Set (Finset α))) ⟨∅, h0, by simp⟩
          hu S hfin hS
    · rintro ⟨⟨E, hE, hE'⟩, ⟨T, hT, hT'⟩, hi, hs⟩
      have hE'' : E = ∅ := Finset.coe_eq_empty.1 hE'
      have hT'' : T = Finset.univ := Finset.coe_eq_univ.1 hT'
      refine ⟨hE'' ▸ hE, hT'' ▸ hT, fun U hU V hV => ⟨?_, ?_⟩⟩
      · obtain ⟨W, hW, hW'⟩ := hs {(U : Set α), (V : Set α)} (by
          rintro _ (rfl | rfl); exacts [⟨U, hU, rfl⟩, ⟨V, hV, rfl⟩])
        have : W = U ∪ V := Finset.coe_inj.1 (by simpa using hW')
        exact this ▸ hW
      · obtain ⟨W, hW, hW'⟩ := hi U V ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
        have : W = U ∩ V := Finset.coe_inj.1 (by simpa using hW')
        exact this ▸ hW
  exact @key (Fin 0) (inferInstance) (inferInstance) (∅)

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.isFinTop_iff (Fin 3) (inferInstance) (inferInstance) ({∅, {0}, {1}, Finset.univ}) := by
  have key : ∀ {α : Type} [inst1 : Fintype α] [inst2 : DecidableEq α] (F : Finset (Finset α)), IsFinTop F ↔ IsTopologyOn α ((fun V : Finset α => (V : Set α)) '' (F : Set (Finset α))) := by
    intro α inst1 inst2 F
    constructor
    · rintro ⟨h0, h1, h2⟩
      have hu : ∀ U V, U ∈ (fun V : Finset α => (V : Set α)) '' (F : Set (Finset α)) →
          V ∈ (fun V : Finset α => (V : Set α)) '' (F : Set (Finset α)) →
          U ∪ V ∈ (fun V : Finset α => (V : Set α)) '' (F : Set (Finset α)) := by
        rintro _ _ ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
        exact ⟨U ∪ V, (h2 U hU V hV).1, by simp⟩
      refine ⟨⟨∅, h0, by simp⟩, ⟨Finset.univ, h1, by simp⟩, ?_, fun S hS => ?_⟩
      · rintro _ _ ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
        exact ⟨U ∩ V, (h2 U hU V hV).2, by simp⟩
      · have hfin : S.Finite := Set.Finite.subset ((F.finite_toSet).image _) hS
        exact sUnion_mem_of_finite ((fun V : Finset α => (V : Set α)) '' (F : Set (Finset α))) ⟨∅, h0, by simp⟩
          hu S hfin hS
    · rintro ⟨⟨E, hE, hE'⟩, ⟨T, hT, hT'⟩, hi, hs⟩
      have hE'' : E = ∅ := Finset.coe_eq_empty.1 hE'
      have hT'' : T = Finset.univ := Finset.coe_eq_univ.1 hT'
      refine ⟨hE'' ▸ hE, hT'' ▸ hT, fun U hU V hV => ⟨?_, ?_⟩⟩
      · obtain ⟨W, hW, hW'⟩ := hs {(U : Set α), (V : Set α)} (by
          rintro _ (rfl | rfl); exacts [⟨U, hU, rfl⟩, ⟨V, hV, rfl⟩])
        have : W = U ∪ V := Finset.coe_inj.1 (by simpa using hW')
        exact this ▸ hW
      · obtain ⟨W, hW, hW'⟩ := hi U V ⟨U, hU, rfl⟩ ⟨V, hV, rfl⟩
        have : W = U ∩ V := Finset.coe_inj.1 (by simpa using hW')
        exact this ▸ hW
  exact @key (Fin 3) (inferInstance) (inferInstance) ({∅, {0}, {1}, Finset.univ})

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.TopologicalSpaces.isFinTop_iff
-- test: set-isfintop-iff-test.sage

/-! ## exm-topology-rn -/

/-- The open sets of `ℝⁿ` (sets containing a ball around each of their points) form a topology. -/
theorem ball_opens_topology (n : ℕ) :
    IsTopologyOn (EuclideanSpace ℝ (Fin n)) {V | ∀ x ∈ V, ∃ ε > 0, ball x ε ⊆ V} := by
  have h : {V : Set (EuclideanSpace ℝ (Fin n)) | ∀ x ∈ V, ∃ ε > 0, ball x ε ⊆ V} = {V | IsOpen V} := by
    ext V; exact Metric.isOpen_iff.symm
  rw [h]
  exact ⟨show IsOpen (∅ : Set (EuclideanSpace ℝ (Fin n))) from isOpen_empty,
    show IsOpen (univ : Set (EuclideanSpace ℝ (Fin n))) from isOpen_univ,
    fun U V (hU : IsOpen U) (hV : IsOpen V) => show IsOpen (U ∩ V) from hU.inter hV,
    fun S hS => show IsOpen (⋃₀ S) from isOpen_sUnion fun U hU => (hS hU : IsOpen U)⟩

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.ball_opens_topology (0) := by
  have key : ∀ (n : ℕ), IsTopologyOn (EuclideanSpace ℝ (Fin n)) {V | ∀ x ∈ V, ∃ ε > 0, ball x ε ⊆ V} := by
    intro n
    have h : {V : Set (EuclideanSpace ℝ (Fin n)) | ∀ x ∈ V, ∃ ε > 0, ball x ε ⊆ V} = {V | IsOpen V} := by
      ext V; exact Metric.isOpen_iff.symm
    rw [h]
    exact ⟨show IsOpen (∅ : Set (EuclideanSpace ℝ (Fin n))) from isOpen_empty,
      show IsOpen (univ : Set (EuclideanSpace ℝ (Fin n))) from isOpen_univ,
      fun U V (hU : IsOpen U) (hV : IsOpen V) => show IsOpen (U ∩ V) from hU.inter hV,
      fun S hS => show IsOpen (⋃₀ S) from isOpen_sUnion fun U hU => (hS hU : IsOpen U)⟩
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.ball_opens_topology (1) := by
  have key : ∀ (n : ℕ), IsTopologyOn (EuclideanSpace ℝ (Fin n)) {V | ∀ x ∈ V, ∃ ε > 0, ball x ε ⊆ V} := by
    intro n
    have h : {V : Set (EuclideanSpace ℝ (Fin n)) | ∀ x ∈ V, ∃ ε > 0, ball x ε ⊆ V} = {V | IsOpen V} := by
      ext V; exact Metric.isOpen_iff.symm
    rw [h]
    exact ⟨show IsOpen (∅ : Set (EuclideanSpace ℝ (Fin n))) from isOpen_empty,
      show IsOpen (univ : Set (EuclideanSpace ℝ (Fin n))) from isOpen_univ,
      fun U V (hU : IsOpen U) (hV : IsOpen V) => show IsOpen (U ∩ V) from hU.inter hV,
      fun S hS => show IsOpen (⋃₀ S) from isOpen_sUnion fun U hU => (hS hU : IsOpen U)⟩
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.ball_opens_topology (2) := by
  have key : ∀ (n : ℕ), IsTopologyOn (EuclideanSpace ℝ (Fin n)) {V | ∀ x ∈ V, ∃ ε > 0, ball x ε ⊆ V} := by
    intro n
    have h : {V : Set (EuclideanSpace ℝ (Fin n)) | ∀ x ∈ V, ∃ ε > 0, ball x ε ⊆ V} = {V | IsOpen V} := by
      ext V; exact Metric.isOpen_iff.symm
    rw [h]
    exact ⟨show IsOpen (∅ : Set (EuclideanSpace ℝ (Fin n))) from isOpen_empty,
      show IsOpen (univ : Set (EuclideanSpace ℝ (Fin n))) from isOpen_univ,
      fun U V (hU : IsOpen U) (hV : IsOpen V) => show IsOpen (U ∩ V) from hU.inter hV,
      fun S hS => show IsOpen (⋃₀ S) from isOpen_sUnion fun U hU => (hS hU : IsOpen U)⟩
  exact @key (2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.TopologicalSpaces.ball_opens_topology
-- test: set-ball-opens-test.sage

/-! ## exm-topology-discrete -/

/-- On any set, all subsets (discrete) and `{∅, X}` (indiscrete) are topologies. -/
theorem discrete_indiscrete (X : Type) : IsTopologyOn X univ ∧ IsTopologyOn X {∅, univ} := by
  refine ⟨⟨trivial, trivial, fun _ _ _ _ => trivial, fun _ _ => trivial⟩, ?_⟩
  refine ⟨Or.inl rfl, Or.inr rfl, ?_, fun S hS => ?_⟩
  · rintro U V (rfl | rfl) (rfl | rfl) <;> simp
  · by_cases h : univ ∈ S
    · right; exact sUnion_eq_univ_iff.2 fun x => ⟨univ, h, trivial⟩
    · left
      rw [sUnion_eq_empty]
      intro U hU
      rcases hS hU with rfl | rfl
      · rfl
      · exact absurd hU h

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete (Fin 3) := by
  have key : ∀ (X : Type), IsTopologyOn X univ ∧ IsTopologyOn X {∅, univ} := by
    intro X
    refine ⟨⟨trivial, trivial, fun _ _ _ _ => trivial, fun _ _ => trivial⟩, ?_⟩
    refine ⟨Or.inl rfl, Or.inr rfl, ?_, fun S hS => ?_⟩
    · rintro U V (rfl | rfl) (rfl | rfl) <;> simp
    · by_cases h : univ ∈ S
      · right; exact sUnion_eq_univ_iff.2 fun x => ⟨univ, h, trivial⟩
      · left
        rw [sUnion_eq_empty]
        intro U hU
        rcases hS hU with rfl | rfl
        · rfl
        · exact absurd hU h
  exact @key (Fin 3)

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete (Empty) := by
  have key : ∀ (X : Type), IsTopologyOn X univ ∧ IsTopologyOn X {∅, univ} := by
    intro X
    refine ⟨⟨trivial, trivial, fun _ _ _ _ => trivial, fun _ _ => trivial⟩, ?_⟩
    refine ⟨Or.inl rfl, Or.inr rfl, ?_, fun S hS => ?_⟩
    · rintro U V (rfl | rfl) (rfl | rfl) <;> simp
    · by_cases h : univ ∈ S
      · right; exact sUnion_eq_univ_iff.2 fun x => ⟨univ, h, trivial⟩
      · left
        rw [sUnion_eq_empty]
        intro U hU
        rcases hS hU with rfl | rfl
        · rfl
        · exact absurd hU h
  exact @key (Empty)

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete (ℕ) := by
  have key : ∀ (X : Type), IsTopologyOn X univ ∧ IsTopologyOn X {∅, univ} := by
    intro X
    refine ⟨⟨trivial, trivial, fun _ _ _ _ => trivial, fun _ _ => trivial⟩, ?_⟩
    refine ⟨Or.inl rfl, Or.inr rfl, ?_, fun S hS => ?_⟩
    · rintro U V (rfl | rfl) (rfl | rfl) <;> simp
    · by_cases h : univ ∈ S
      · right; exact sUnion_eq_univ_iff.2 fun x => ⟨univ, h, trivial⟩
      · left
        rw [sUnion_eq_empty]
        intro U hU
        rcases hS hU with rfl | rfl
        · rfl
        · exact absurd hU h
  exact @key (ℕ)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete
-- test: set-discrete-indiscrete-test.sage

/-! ## exm-topology-finite -/

/-- A chain of subsets containing `∅` and the whole set is a topology: `{∅, {1}, {1, 2}, {1, 2, 3}}`. -/
theorem chain_isFinTop {α : Type} [Fintype α] [DecidableEq α] (F : Finset (Finset α))
    (hchain : ∀ U ∈ F, ∀ V ∈ F, U ⊆ V ∨ V ⊆ U) (h0 : ∅ ∈ F) (h1 : Finset.univ ∈ F) : IsFinTop F := by
  refine ⟨h0, h1, fun U hU V hV => ?_⟩
  rcases hchain U hU V hV with h | h
  · rw [Finset.union_eq_right.2 h, Finset.inter_eq_left.2 h]; exact ⟨hV, hU⟩
  · rw [Finset.union_eq_left.2 h, Finset.inter_eq_right.2 h]; exact ⟨hU, hV⟩

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.chain_isFinTop (Fin 3) (inferInstance) (inferInstance) ({∅, {0}, {0, 1}, Finset.univ}) ((by decide)) ((by decide)) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.chain_isFinTop (Fin 0) (inferInstance) (inferInstance) ({∅}) ((by decide)) ((by decide)) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.chain_isFinTop (Fin 2) (inferInstance) (inferInstance) ({∅, Finset.univ}) ((by decide)) ((by decide)) ((by decide)) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.TopologicalSpaces.chain_isFinTop
-- test: set-chain-test.sage

/-- `{∅, {1}, {2}, {1, 2, 3}}` is not a topology: two members whose union is missing (Sage finds `{1} ∪ {2}`). -/
theorem not_topology_union : ∃ U ∈ ({∅, {0}, {1}, Finset.univ} : Finset (Finset (Fin 3))),
    ∃ V ∈ ({∅, {0}, {1}, Finset.univ} : Finset (Finset (Fin 3))),
      U ∪ V ∉ ({∅, {0}, {1}, Finset.univ} : Finset (Finset (Fin 3))) := by
  veriphy_certified (exact ⟨{0}, by decide, { 1 }, by decide, by decide⟩)
  -- certificate: set-topology-missing-union.sage

/-! ## exr-topology-axioms-check -/

/-- `{∅, A, Aᶜ, X}` is a topology for any subset `A`: `{∅, {1}, {2, 3}, X}`. -/
theorem complement_pair_isFinTop {α : Type} [Fintype α] [DecidableEq α] (A : Finset α) :
    IsFinTop ({∅, A, Aᶜ, Finset.univ} : Finset (Finset α)) := by
  have hu : Aᶜ ∪ A = Finset.univ := by rw [Finset.union_comm]; exact Finset.union_compl A
  have hi : Aᶜ ∩ A = ∅ := by rw [Finset.inter_comm]; exact Finset.inter_compl A
  refine ⟨by simp, by simp, ?_⟩
  simp only [Finset.mem_insert, Finset.mem_singleton]
  rintro U (rfl | rfl | rfl | rfl) V (rfl | rfl | rfl | rfl) <;>
    simp [Finset.union_compl, Finset.inter_compl, hu, hi]

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.complement_pair_isFinTop (Fin 3) (inferInstance) (inferInstance) ({0}) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.complement_pair_isFinTop (Fin 3) (inferInstance) (inferInstance) (∅) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.complement_pair_isFinTop (Fin 4) (inferInstance) (inferInstance) ({1, 2}) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.TopologicalSpaces.complement_pair_isFinTop
-- test: set-complement-pair-test.sage

/-- `∅` together with all the sets containing a point `a` is a topology: `{∅, {1}, {1, 2}, {1, 3}, X}`. -/
theorem particular_point_isFinTop {α : Type} [Fintype α] [DecidableEq α] (a : α) :
    IsFinTop (insert ∅ (Finset.univ.filter (fun U : Finset α => a ∈ U))) := by
  refine ⟨Finset.mem_insert_self _ _, Finset.mem_insert_of_mem (by simp), ?_⟩
  intro U hU V hV
  simp only [Finset.mem_insert, Finset.mem_filter, Finset.mem_univ, true_and] at hU hV ⊢
  rcases hU with rfl | hU <;> rcases hV with rfl | hV <;> simp_all

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.particular_point_isFinTop (Fin 3) (inferInstance) (inferInstance) (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.particular_point_isFinTop (Fin 1) (inferInstance) (inferInstance) (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.particular_point_isFinTop (Fin 3) (inferInstance) (inferInstance) (2) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.TopologicalSpaces.particular_point_isFinTop
-- test: set-particular-point-test.sage

/-- `{∅, {1, 2}, {2, 3}, X}` is not a topology: two members whose intersection is missing (Sage finds `{2}`). -/
theorem not_topology_inter : ∃ U ∈ ({∅, {0, 1}, {1, 2}, Finset.univ} : Finset (Finset (Fin 3))),
    ∃ V ∈ ({∅, {0, 1}, {1, 2}, Finset.univ} : Finset (Finset (Fin 3))),
      U ∩ V ∉ ({∅, {0, 1}, {1, 2}, Finset.univ} : Finset (Finset (Fin 3))) := by
  veriphy_certified (exact ⟨{0, 1}, by decide, { 1, 2 }, by decide, by decide⟩)
  -- certificate: set-topology-missing-inter.sage

/-! ## exr-topology-two-points -/

/-- The topologies on `{1, 2}`: Sage lists four (indiscrete, the two Sierpiński topologies, discrete), and they are
exactly the families satisfying the axioms. -/
theorem topologies_two : ∃ L : List (Finset (Finset (Fin 2))), L.Nodup ∧ L.length = 4 ∧
    ∀ F : Finset (Finset (Fin 2)), F ∈ L ↔ IsFinTop F := by
  veriphy_certified
    (exact ⟨[{∅, {0, 1}}, {∅, {0}, {0, 1}}, {∅, { 1 }, {0, 1}}, {∅, {0}, { 1 }, {0, 1}}], by decide, rfl, by decide⟩)
  -- certificate: set-topologies-two.sage

/-! ## exr-topology-intersection -/

/-- The common open sets of two topologies form a topology. -/
theorem inter_isTopologyOn (X : Type) (O₁ O₂ : Set (Set X)) (h₁ : IsTopologyOn X O₁) (h₂ : IsTopologyOn X O₂) :
    IsTopologyOn X (O₁ ∩ O₂) :=
  ⟨⟨h₁.1, h₂.1⟩, ⟨h₁.2.1, h₂.2.1⟩, fun U V hU hV => ⟨h₁.2.2.1 U V hU.1 hV.1, h₂.2.2.1 U V hU.2 hV.2⟩,
    fun S hS => ⟨h₁.2.2.2 S (fun _ h => (hS h).1), h₂.2.2.2 S (fun _ h => (hS h).2)⟩⟩

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.inter_isTopologyOn (Fin 2) (Set.univ) ({∅, Set.univ}) ((KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete _).1) ((KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete _).2) := by
  have key : ∀ (X : Type) (O₁ O₂ : Set (Set X)) (h₁ : IsTopologyOn X O₁) (h₂ : IsTopologyOn X O₂), IsTopologyOn X (O₁ ∩ O₂) := by
    intro X O₁ O₂ h₁ h₂
    exact (⟨⟨h₁.1, h₂.1⟩, ⟨h₁.2.1, h₂.2.1⟩, fun U V hU hV => ⟨h₁.2.2.1 U V hU.1 hV.1, h₂.2.2.1 U V hU.2 hV.2⟩,
      fun S hS => ⟨h₁.2.2.2 S (fun _ h => (hS h).1), h₂.2.2.2 S (fun _ h => (hS h).2)⟩⟩)
  exact @key (Fin 2) (Set.univ) ({∅, Set.univ}) ((KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete _).1) ((KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete _).2)

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.inter_isTopologyOn (Fin 2) (Set.univ) (Set.univ) ((KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete _).1) ((KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete _).1) := by
  have key : ∀ (X : Type) (O₁ O₂ : Set (Set X)) (h₁ : IsTopologyOn X O₁) (h₂ : IsTopologyOn X O₂), IsTopologyOn X (O₁ ∩ O₂) := by
    intro X O₁ O₂ h₁ h₂
    exact (⟨⟨h₁.1, h₂.1⟩, ⟨h₁.2.1, h₂.2.1⟩, fun U V hU hV => ⟨h₁.2.2.1 U V hU.1 hV.1, h₂.2.2.1 U V hU.2 hV.2⟩,
      fun S hS => ⟨h₁.2.2.2 S (fun _ h => (hS h).1), h₂.2.2.2 S (fun _ h => (hS h).2)⟩⟩)
  exact @key (Fin 2) (Set.univ) (Set.univ) ((KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete _).1) ((KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete _).1)

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.inter_isTopologyOn (ℕ) ({∅, Set.univ}) (Set.univ) ((KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete _).2) ((KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete _).1) := by
  have key : ∀ (X : Type) (O₁ O₂ : Set (Set X)) (h₁ : IsTopologyOn X O₁) (h₂ : IsTopologyOn X O₂), IsTopologyOn X (O₁ ∩ O₂) := by
    intro X O₁ O₂ h₁ h₂
    exact (⟨⟨h₁.1, h₂.1⟩, ⟨h₁.2.1, h₂.2.1⟩, fun U V hU hV => ⟨h₁.2.2.1 U V hU.1 hV.1, h₂.2.2.1 U V hU.2 hV.2⟩,
      fun S hS => ⟨h₁.2.2.2 S (fun _ h => (hS h).1), h₂.2.2.2 S (fun _ h => (hS h).2)⟩⟩)
  exact @key (ℕ) ({∅, Set.univ}) (Set.univ) ((KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete _).2) ((KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete _).1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.TopologicalSpaces.inter_isTopologyOn
-- test: set-inter-topology-test.sage

set_option maxRecDepth 100000 in
/-- On two points the union of two topologies is again a topology (the union of the two Sierpiński topologies is the
discrete one). -/
theorem union_two_points (F G : Finset (Finset (Fin 2))) (hF : IsFinTop F) (hG : IsFinTop G) : IsFinTop (F ∪ G) := by
  revert F G
  decide

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.union_two_points ({∅, {0}, Finset.univ}) ({∅, {1}, Finset.univ}) ((by decide)) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.union_two_points ({∅, Finset.univ}) ({∅, Finset.univ}) ((by decide)) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.TopologicalSpaces.union_two_points (Finset.univ) ({∅, {0}, Finset.univ}) ((by decide)) ((by decide)) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.TopologicalSpaces.union_two_points
-- test: set-union-two-points-test.sage

/-- On three points the union of two topologies need not be one: Sage finds `{∅, {1}, X}` and `{∅, {2}, X}`, whose union
lacks `{1, 2}`. -/
theorem union_not_topology : ∃ F G : Finset (Finset (Fin 3)), IsFinTop F ∧ IsFinTop G ∧ ¬ IsFinTop (F ∪ G) := by
  veriphy_certified (exact ⟨{∅, {0}, {0, 1, 2}}, {∅, { 1 }, {0, 1, 2}}, by decide, by decide, by decide⟩)
  -- certificate: set-union-not-topology.sage

end KaluzaKlein.Sets.TopologicalSpaces
