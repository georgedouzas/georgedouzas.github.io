import Mathlib

/-!
# Countable facts for the primer chapters

Finite counting statements that the sets and topology primers display: the number of maps between finite sets, of
bijections, of elements of a quotient, and a bijection whose inverse is computed.
-/

namespace KaluzaKlein.Primers

/-- There are `2³ = 8` maps from a three-element set to a two-element set. -/
theorem card_maps_3_2 : Fintype.card (Fin 3 → Fin 2) = 8 := by simp

/-- There are `3! = 6` bijections of a three-element set. -/
theorem card_bijections_3 : Fintype.card (Equiv.Perm (Fin 3)) = 6 := by simp [Fintype.card_perm, Nat.factorial]

/-- No map from a three-element set to a two-element set is injective. -/
theorem no_injection_3_2 (f : Fin 3 → Fin 2) : ¬ Function.Injective f := by
  intro h
  have := Fintype.card_le_of_injective f h
  simp at this

/-- The quotient of `ℤ` by `3ℤ` has three elements. -/
theorem card_zmod3 : Fintype.card (ZMod 3) = 3 := by simp

/-- The map `k ↦ k + 1` on `ZMod 6` is a bijection with inverse `k ↦ k − 1`. -/
theorem shift_bijective : Function.Bijective (fun k : ZMod 6 => k + 1) := by
  refine ⟨fun a b h => by simpa using h, fun b => ⟨b - 1, by ring⟩⟩

end KaluzaKlein.Primers

namespace KaluzaKlein.Primers

/-- A map is injective if equal values force equal arguments (mirrors `Function.Injective`). -/
def IsInjective {X Y : Type*} (f : X → Y) : Prop := ∀ x x', f x = f x' → x = x'

/-- A map is surjective if every element of the codomain is a value. -/
def IsSurjective {X Y : Type*} (f : X → Y) : Prop := ∀ y, ∃ x, f x = y

/-- A bijection is a map that is both injective and surjective. -/
def IsBijective {X Y : Type*} (f : X → Y) : Prop := IsInjective f ∧ IsSurjective f

/-- An equivalence relation: reflexive, symmetric and transitive (mirrors mathlib's `Equivalence`). -/
def IsEquivalence {X : Type*} (r : X → X → Prop) : Prop :=
  (∀ x, r x x) ∧ (∀ x y, r x y → r y x) ∧ (∀ x y z, r x y → r y z → r x z)

/-- The mirrored notions agree with mathlib's. -/
theorem isInjective_iff {X Y : Type*} (f : X → Y) : IsInjective f ↔ Function.Injective f := Iff.rfl
theorem isSurjective_iff {X Y : Type*} (f : X → Y) : IsSurjective f ↔ Function.Surjective f := Iff.rfl

/-- There are `3⁴ = 81` maps from a four-element set to a three-element set. -/
theorem card_maps_4_3 : Fintype.card (Fin 4 → Fin 3) = 81 := by simp

/-- No map from a four-element set to a three-element set is injective. -/
theorem no_injection_4_3 (f : Fin 4 → Fin 3) : ¬ Function.Injective f := by
  intro h
  have := Fintype.card_le_of_injective f h
  simp at this

/-- There are `4 · 3 · 2 = 24` injective maps from a three-element set to a four-element set. -/
theorem card_injections_3_4 : Fintype.card (Fin 3 ↪ Fin 4) = 24 := by
  rw [Fintype.card_embedding_eq]; rfl

/-- There are `4! = 24` bijections of a four-element set. -/
theorem card_bijections_4 : Fintype.card (Equiv.Perm (Fin 4)) = 24 := by simp [Fintype.card_perm, Nat.factorial]

/-- `ℤ/5ℤ` has five elements. -/
theorem card_zmod5 : Fintype.card (ZMod 5) = 5 := by simp

/-- `ℤ/nℤ` has `n` elements for every `n ≥ 1`. -/
theorem card_zmod_n (n : ℕ) [NeZero n] : Fintype.card (ZMod n) = n := ZMod.card n

end KaluzaKlein.Primers

namespace KaluzaKlein.Primers

/-- A topology on `X`: a family of open sets containing `∅` and `X`, closed under finite intersections and
arbitrary unions (mirrors mathlib's `TopologicalSpace`). -/
def IsTopologyOn (X : Type*) (opens : Set (Set X)) : Prop :=
  ∅ ∈ opens ∧ Set.univ ∈ opens ∧ (∀ U V, U ∈ opens → V ∈ opens → U ∩ V ∈ opens) ∧
    (∀ S : Set (Set X), S ⊆ opens → ⋃₀ S ∈ opens)

/-- The open sets of a mathlib topological space satisfy the mirrored axioms. -/
theorem isTopologyOn_of_topologicalSpace (X : Type*) [TopologicalSpace X] :
    IsTopologyOn X {U | IsOpen U} := by
  refine ⟨isOpen_empty, isOpen_univ, fun U V hU hV => hU.inter hV, fun S hS => ?_⟩
  exact isOpen_sUnion fun U hU => hS hU

end KaluzaKlein.Primers
