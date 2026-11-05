import KaluzaKlein.Conventions

/-!
# Groups, cosets and Schur's lemma on explicit examples

* Finite cosets and quotients: `ZMod 6 / ⟨2⟩ ≅ ZMod 2`: the cosets of the subgroup `{0, 2, 4}` are `{0,2,4}` and
  `{1,3,5}`, and the quotient multiplication is addition mod 2 (`decide`).
* Schur's lemma on the spin-1 (vector) representation of `so(3)`: a `3 × 3` matrix commuting with `L₀, L₁, L₂` is a
  multiple of the identity.
-/

namespace KaluzaKlein.Groups

/-- The subgroup `H = {0, 2, 4}` of `ZMod 6` is the image of `fun k => 2 k`. -/
def H6 : Finset (ZMod 6) := {0, 2, 4}

/-- The two cosets `0 + H` and `1 + H` partition `ZMod 6`. -/
theorem zmod6_cosets :
    (H6.image fun h => (0 : ZMod 6) + h) = {0, 2, 4} ∧ (H6.image fun h => (1 : ZMod 6) + h) = {1, 3, 5} ∧
      ({0, 2, 4} : Finset (ZMod 6)) ∪ {1, 3, 5} = Finset.univ := by
  decide

/-- The quotient is `Z₂`: the coset of `a + b` depends only on the cosets of `a` and `b`, and the sum of two odd
cosets is the even one. -/
theorem zmod6_quotient : ∀ a b : ZMod 6, (a + b).val % 2 = (a.val % 2 + b.val % 2) % 2 := by
  decide

/-- Schur's lemma for the vector representation of `so(3)`: a matrix commuting with the three generators is a
scalar. The nine commutation equations are read off entry by entry. -/
theorem schur_vector (X : Matrix (Fin 3) (Fin 3) ℚ) (h : ∀ a, X * L a = L a * X) :
    X = X 0 0 • (1 : Matrix (Fin 3) (Fin 3) ℚ) := by
  have h000 := congrFun (congrFun (h 0) 0) 0
  have h001 := congrFun (congrFun (h 0) 0) 1
  have h002 := congrFun (congrFun (h 0) 0) 2
  have h010 := congrFun (congrFun (h 0) 1) 0
  have h011 := congrFun (congrFun (h 0) 1) 1
  have h012 := congrFun (congrFun (h 0) 1) 2
  have h020 := congrFun (congrFun (h 0) 2) 0
  have h021 := congrFun (congrFun (h 0) 2) 1
  have h022 := congrFun (congrFun (h 0) 2) 2
  have h100 := congrFun (congrFun (h 1) 0) 0
  have h101 := congrFun (congrFun (h 1) 0) 1
  have h102 := congrFun (congrFun (h 1) 0) 2
  have h110 := congrFun (congrFun (h 1) 1) 0
  have h111 := congrFun (congrFun (h 1) 1) 1
  have h112 := congrFun (congrFun (h 1) 1) 2
  have h120 := congrFun (congrFun (h 1) 2) 0
  have h121 := congrFun (congrFun (h 1) 2) 1
  have h122 := congrFun (congrFun (h 1) 2) 2
  simp only [L, L0, L1] at h000 h001 h002 h010 h011 h012 h020 h021 h022 h100 h101 h102 h110 h111 h112 h120 h121 h122
  simp [Matrix.mul_apply, Fin.sum_univ_succ] at h000 h001 h002 h010 h011 h012 h020 h021 h022 h100 h101 h102 h110 h111 h112 h120 h121 h122
  ext i j; fin_cases i <;> fin_cases j <;> simp <;> linarith

end KaluzaKlein.Groups
