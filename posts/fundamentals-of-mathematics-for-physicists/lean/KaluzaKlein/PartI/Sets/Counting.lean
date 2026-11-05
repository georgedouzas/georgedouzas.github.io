import Mathlib
import KaluzaKlein.PartI.Sets.Inverse
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Maps" (counting maps)

The claims of the blocks `prp-count-maps` to `exr-count-general` of `tutorial/part-i-01-sets-relations-topology.qmd`
not already stated in the earlier modules: maps out of a finite set as tuples of values, the tree of choices behind
`n!`, the pigeonhole inequality, the `36` surjections from four elements to three, and the general formulas for maps,
injections and bijections between finite sets.
-/

namespace KaluzaKlein.Sets.Counting

/-! ## exm-count-product -/

/-- A map `{1, 2, 3} → Y` is the triple of its values: `f ↦ (f 1, f 2, f 3)` is a bijection onto `Y × Y × Y`. -/
theorem maps_three_as_triples (β : Type) :
    Function.Bijective (fun f : Fin 3 → β => (f 0, f 1, f 2)) := by
  refine ⟨fun f g h => ?_, fun ⟨a, b, c⟩ => ⟨![a, b, c], rfl⟩⟩
  simp only [Prod.mk.injEq] at h
  funext i; fin_cases i
  · exact h.1
  · exact h.2.1
  · exact h.2.2

example : veriphy_instance% KaluzaKlein.Sets.Counting.maps_three_as_triples (Fin 2) := by
  have key : ∀ (β : Type), Function.Bijective (fun f : Fin 3 → β => (f 0, f 1, f 2)) := by
    intro β
    refine ⟨fun f g h => ?_, fun ⟨a, b, c⟩ => ⟨![a, b, c], rfl⟩⟩
    simp only [Prod.mk.injEq] at h
    funext i; fin_cases i
    · exact h.1
    · exact h.2.1
    · exact h.2.2
  exact @key (Fin 2)

example : veriphy_instance% KaluzaKlein.Sets.Counting.maps_three_as_triples (Unit) := by
  have key : ∀ (β : Type), Function.Bijective (fun f : Fin 3 → β => (f 0, f 1, f 2)) := by
    intro β
    refine ⟨fun f g h => ?_, fun ⟨a, b, c⟩ => ⟨![a, b, c], rfl⟩⟩
    simp only [Prod.mk.injEq] at h
    funext i; fin_cases i
    · exact h.1
    · exact h.2.1
    · exact h.2.2
  exact @key (Unit)

example : veriphy_instance% KaluzaKlein.Sets.Counting.maps_three_as_triples (ℕ) := by
  have key : ∀ (β : Type), Function.Bijective (fun f : Fin 3 → β => (f 0, f 1, f 2)) := by
    intro β
    refine ⟨fun f g h => ?_, fun ⟨a, b, c⟩ => ⟨![a, b, c], rfl⟩⟩
    simp only [Prod.mk.injEq] at h
    funext i; fin_cases i
    · exact h.1
    · exact h.2.1
    · exact h.2.2
  exact @key (ℕ)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Counting.maps_three_as_triples
-- test: set-maps-triples-test.sage

/-- The maps between finite sets `X → Y` number `|Y|^|X|`. -/
theorem card_fun (X Y : Type) [Fintype X] [DecidableEq X] [Fintype Y] :
    Fintype.card (X → Y) = Fintype.card Y ^ Fintype.card X :=
  Fintype.card_fun

example : veriphy_instance% KaluzaKlein.Sets.Counting.card_fun (Fin 3) (Fin 2) (inferInstance) (inferInstance) (inferInstance) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Counting.card_fun (Fin 0) (Fin 5) (inferInstance) (inferInstance) (inferInstance) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Counting.card_fun (Fin 4) (Fin 3) (inferInstance) (inferInstance) (inferInstance) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Counting.card_fun
-- test: set-card-fun-test.sage

/-! ## exm-count-factorial -/

/-- The tree of choices: a bijection of `n + 1` elements chooses the value of the first element among `n + 1`, then a
bijection of the remaining `n`: `(n + 1)! = (n + 1) · n!`, so `3! = 3 · 2 · 1 = 6`. -/
theorem card_perm_succ (n : ℕ) :
    Fintype.card (Equiv.Perm (Fin (n + 1))) = (n + 1) * Fintype.card (Equiv.Perm (Fin n)) := by
  rw [Fintype.card_congr Equiv.Perm.decomposeFin, Fintype.card_prod, Fintype.card_fin]

example : veriphy_instance% KaluzaKlein.Sets.Counting.card_perm_succ (2) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Counting.card_perm_succ (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Counting.card_perm_succ (3) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Counting.card_perm_succ
-- test: set-card-perm-succ-test.sage

/-! ## exm-count-pigeonhole -/

/-- An injective map between finite sets has `|X| ≤ |Y|`. -/
theorem card_le_of_injective (X Y : Type) [Fintype X] [Fintype Y] (f : X → Y) (hf : Function.Injective f) :
    Fintype.card X ≤ Fintype.card Y :=
  Fintype.card_le_of_injective f hf

example : veriphy_instance% KaluzaKlein.Sets.Counting.card_le_of_injective (Fin 2) (Fin 3) (inferInstance) (inferInstance) (Fin.castSucc) (Fin.castSucc_injective 2) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Counting.card_le_of_injective (Fin 0) (Fin 0) (inferInstance) (inferInstance) (id) (Function.injective_id) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Counting.card_le_of_injective (Fin 3) (Fin 4) (inferInstance) (inferInstance) (![3, 2, 1]) ((by decide)) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Counting.card_le_of_injective
-- test: set-card-le-injective-test.sage

/-! ## exr-count-maps-4-3 -/

/-- The surjective maps from four elements to three: Sage lists `36 = 81 - 3 · 2⁴ + 3`, without repetition, and the
list contains exactly the surjective maps. -/
theorem surjections_4_3 : ∃ L : List (Fin 4 → Fin 3), L.Nodup ∧ L.length = 36 ∧
    ∀ f : Fin 4 → Fin 3, f ∈ L ↔ Function.Surjective f := by
  veriphy_certified
    (exact
        ⟨[![0, 0, 1, 2], ![0, 0, 2, 1], ![0, 1, 0, 2], ![0, 1, 1, 2], ![0, 1, 2, 0], ![0, 1, 2, 1], ![0, 1, 2, 2],
            ![0, 2, 0, 1], ![0, 2, 1, 0], ![0, 2, 1, 1], ![0, 2, 1, 2], ![0, 2, 2, 1], ![1, 0, 0, 2], ![1, 0, 1, 2],
            ![1, 0, 2, 0], ![1, 0, 2, 1], ![1, 0, 2, 2], ![1, 1, 0, 2], ![1, 1, 2, 0], ![1, 2, 0, 0], ![1, 2, 0, 1],
            ![1, 2, 0, 2], ![1, 2, 1, 0], ![1, 2, 2, 0], ![2, 0, 0, 1], ![2, 0, 1, 0], ![2, 0, 1, 1], ![2, 0, 1, 2],
            ![2, 0, 2, 1], ![2, 1, 0, 0], ![2, 1, 0, 1], ![2, 1, 0, 2], ![2, 1, 1, 0], ![2, 1, 2, 0], ![2, 2, 0, 1],
            ![2, 2, 1, 0]],
          by decide, rfl, by decide⟩)
  -- certificate: set-surjections-4-3.sage

/-- Hence `36` of the `81` maps from four elements to three are surjective. -/
theorem card_surjections_4_3 :
    (Finset.univ.filter fun f : Fin 4 → Fin 3 => Function.Surjective f).card = 36 := by
  obtain ⟨L, hL, hlen, hmem⟩ := surjections_4_3
  rw [← hlen, ← List.toFinset_card_of_nodup hL]
  congr 1
  ext f
  simp [hmem]

/-! ## exr-count-general -/

/-- Bijections between an `m`-element and an `n`-element set: `n!` when `m = n` and none otherwise. -/
theorem card_bijections (m n : ℕ) : Fintype.card (Fin m ≃ Fin n) = if m = n then n.factorial else 0 := by
  split_ifs with h
  · subst h; rw [Fintype.card_equiv (Equiv.refl _), Fintype.card_fin]
  · rw [Fintype.card_eq_zero_iff]
    exact ⟨fun e => h (Fin.equiv_iff_eq.1 ⟨e⟩)⟩

example : veriphy_instance% KaluzaKlein.Sets.Counting.card_bijections (3) (3) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Counting.card_bijections (0) (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Counting.card_bijections (3) (2) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Counting.card_bijections
-- test: set-card-bijections-test.sage

/-- No injective map exists from `m` elements to `n < m`: the falling factorial `n(n-1)⋯(n-m+1)` vanishes. -/
theorem descFactorial_eq_zero (m n : ℕ) (h : n < m) : n.descFactorial m = 0 :=
  Nat.descFactorial_eq_zero_iff_lt.2 h

example : veriphy_instance% KaluzaKlein.Sets.Counting.descFactorial_eq_zero (3) (2) ((by norm_num)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Counting.descFactorial_eq_zero (1) (0) ((by norm_num)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Counting.descFactorial_eq_zero (4) (3) ((by norm_num)) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Counting.descFactorial_eq_zero
-- test: set-descfactorial-zero-test.sage

end KaluzaKlein.Sets.Counting
