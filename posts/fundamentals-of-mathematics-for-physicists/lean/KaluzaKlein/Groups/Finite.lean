import Mathlib

/-!
# Finite group facts displayed in the group chapters

Non-commutativity and element orders in `S₃`, Lagrange's theorem on a subgroup of `S₃`, the homomorphism
`ℤ₆ → ℤ₂` with its kernel, and the sign homomorphism of `S₃` with kernel `A₃`.
-/

namespace KaluzaKlein.Groups

open Equiv

/-- `(0 1)(0 2) ≠ (0 2)(0 1)` in `S₃`: the symmetric group is not abelian. -/
theorem s3_not_comm : swap (0 : Fin 3) 1 * swap 0 2 ≠ swap 0 2 * swap 0 1 := by decide

/-- A transposition has order two. -/
theorem transposition_order : orderOf (swap (0 : Fin 3) 1) = 2 :=
  orderOf_eq_prime (p := 2) (by decide) (by decide)

/-- The three-cycle `(0 1)(1 2)` has order three. -/
theorem three_cycle_order : orderOf (swap (0 : Fin 3) 1 * swap 1 2) = 3 :=
  orderOf_eq_prime (p := 3) (by decide) (by decide)

/-- `S₃` has six elements. -/
theorem card_s3 : Fintype.card (Perm (Fin 3)) = 6 := by simp [Fintype.card_perm, Nat.factorial]

/-- The alternating group `A₃` has three elements, and `3 ∣ 6` (Lagrange). -/
theorem card_a3 : Fintype.card (alternatingGroup (Fin 3)) = 3 := by decide

/-- The order of a subgroup divides the order of the group (Lagrange), instantiated for `A₃ ⊆ S₃`. -/
theorem lagrange_a3 : Nat.card (alternatingGroup (Fin 3)) ∣ Nat.card (Perm (Fin 3)) :=
  Subgroup.card_subgroup_dvd_card _

/-- Reduction modulo `2` as a map `ℤ₆ → ℤ₂`. -/
def mod2 (k : ZMod 6) : ZMod 2 := (k.val : ZMod 2)

/-- `mod2` is additive: a homomorphism of groups. -/
theorem mod2_add (a b : ZMod 6) : mod2 (a + b) = mod2 a + mod2 b := by
  fin_cases a <;> fin_cases b <;> decide

/-- The kernel of `mod2` is `{0, 2, 4}`. -/
theorem mod2_kernel : ∀ k : ZMod 6, mod2 k = 0 ↔ k = 0 ∨ k = 2 ∨ k = 4 := by decide

/-- The sign is a homomorphism `S₃ → {±1}`. -/
theorem sign_mul (g h : Perm (Fin 3)) : Perm.sign (g * h) = Perm.sign g * Perm.sign h := map_mul Perm.sign g h

/-- The kernel of the sign is the alternating group. -/
theorem sign_kernel (g : Perm (Fin 3)) : Perm.sign g = 1 ↔ g ∈ alternatingGroup (Fin 3) :=
  Equiv.Perm.mem_alternatingGroup.symm

/-- The sign of a transposition is `-1` and of a three-cycle `+1`. -/
theorem sign_examples : Perm.sign (swap (0 : Fin 3) 1) = -1 ∧ Perm.sign (swap (0 : Fin 3) 1 * swap 1 2) = 1 := by
  decide

end KaluzaKlein.Groups
