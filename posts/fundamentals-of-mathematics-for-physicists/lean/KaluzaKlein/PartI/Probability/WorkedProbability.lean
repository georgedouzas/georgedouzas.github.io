import KaluzaKlein.PartI.Probability.Conditioning

/-!
# Probability and statistics, section "Worked computations" of "Probability spaces"

The claims of the blocks `prp-probability-space-values` to `exr-prob-three-dice` of
`tutorial/part-i-08-probability.qmd` not already stated in `SampleSpaces` and `Conditioning`: the closure of power
sets and of the partition algebra, the exact Bayes cancellation, inclusion–exclusion for three events, and three dice.
-/

open Finset

namespace KaluzaKlein.Probability.WorkedProbability

open SampleSpaces Conditioning

/-! ## prp-probability-space-values -/

/-- The `2 ^ 3 = 8` subsets of a three-element set are closed under complement and union. -/
theorem powerset_three_closed : Fintype.card (Finset (Fin 3)) = 8 ∧
    ∀ A B : Finset (Fin 3), Aᶜ ∈ (univ : Finset (Finset (Fin 3))) ∧ A ∪ B ∈ (univ : Finset (Finset (Fin 3))) := by
  simp

/-! ## exm-prob-sage-encoding -/

/-- The encoding of the sample space as pairs `(i, j)`: `36` outcomes, `|A| / |Ω| = 6 / 36 = 1/6` exactly, and the
conditional probability is the ratio `(6/36) / (18/36) = 1/3`. -/
theorem encoding_values : Fintype.card Dice2 = 36 ∧ (A10.card : ℚ) / Fintype.card Dice2 = 1 / 6 ∧
    (A10.card : ℚ) / Fintype.card Dice2 = P (unif Dice2) A10 ∧
    ((A10 ∩ B4).card : ℚ) / Fintype.card Dice2 / ((B4.card : ℚ) / Fintype.card Dice2) = 1 / 3 := by
  decide +kernel

/-! ## exm-prob-sage-subsets -/

/-- A set with `n` elements has `2 ^ n` subsets and `4 ^ n` pairs of subsets. -/
theorem card_pairs_of_subsets (n : ℕ) :
    Fintype.card (Finset (Fin n)) = 2 ^ n ∧ Fintype.card (Finset (Fin n) × Finset (Fin n)) = 4 ^ n := by
  constructor
  · simp
  · rw [Fintype.card_prod, Fintype.card_finset, Fintype.card_fin, ← mul_pow]; norm_num

/-- For `n = 3`: `8` subsets and `64` pairs, each complement and each union again a subset. -/
theorem subsets_three : (univ : Finset (Finset (Fin 3))).card = 8 ∧
    (univ : Finset (Finset (Fin 3) × Finset (Fin 3))).card = 64 ∧
    (∀ A ∈ (univ : Finset (Finset (Fin 3))), Aᶜ ∈ (univ : Finset (Finset (Fin 3)))) ∧
    ∀ A ∈ (univ : Finset (Finset (Fin 3))), ∀ B ∈ (univ : Finset (Finset (Fin 3))),
      A ∪ B ∈ (univ : Finset (Finset (Fin 3))) := by
  simp

/-! ## exm-prob-sage-bayes-exact -/

/-- The exact posterior `(1/100 · 99/100) / (1/100 · 99/100 + 99/100 · 1/20) = 1/6`. -/
theorem bayes_exact : ((1 / 100 : ℚ) * (99 / 100)) / ((1 / 100) * (99 / 100) + (99 / 100) * (1 / 20)) = 1 / 6 ∧
    ((1 / 100 : ℚ)) / (1 / 100 + 1 / 20) = 1 / (1 + 5) := by
  norm_num

/-- The common factor cancels: `(s a) / (s a + s b) = a / (a + b)` for `s ≠ 0`, so the posterior depends only on the
ratio of prevalence to false-positive rate when the sensitivity is near one. -/
theorem common_factor_cancels (s a b : ℚ) (hs : s ≠ 0) (hab : a + b ≠ 0) :
    s * a / (s * a + s * b) = a / (a + b) := by
  VERIPHY[pr-identity]

/-! ## exr-prob-subsets-four -/

/-- The `16` subsets of a four-element set are closed under complement, union and intersection. -/
theorem subsets_four : Fintype.card (Finset (Fin 4)) = 16 ∧
    ∀ A B : Finset (Fin 4), Aᶜ ∈ (univ : Finset (Finset (Fin 4))) ∧ A ∪ B ∈ (univ : Finset (Finset (Fin 4))) ∧
      A ∩ B ∈ (univ : Finset (Finset (Fin 4))) := by
  simp

/-- The four blocks of `exr-prob-sigma-partition` generate exactly `16 = 2 ^ 4` events of `{1, …, 6}`, closed under
complement, union and intersection, among them "even" and "at most 2", and none separates `4` from `6`. -/
theorem partition_sixteen : genF.card = 16 ∧ (∀ A ∈ genF, Aᶜ ∈ genF) ∧
    (∀ A ∈ genF, ∀ B ∈ genF, A ∪ B ∈ genF) ∧ (∀ A ∈ genF, ∀ B ∈ genF, A ∩ B ∈ genF) ∧
    evenF ∈ genF ∧ le2 ∈ genF ∧ ∀ A ∈ genF, (3 ∈ A ↔ 5 ∈ A) := by
  decide +kernel

/-! ## exr-prob-inclusion-exclusion-three -/

/-- Inclusion–exclusion for three events. -/
theorem P_union_three {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (A B C : Finset Ω) :
    P p (A ∪ B ∪ C) = P p A + P p B + P p C - P p (A ∩ B) - P p (A ∩ C) - P p (B ∩ C) + P p (A ∩ B ∩ C) := by
  have h1 := P_union p (A ∪ B) C
  have h2 := P_union p A B
  have h3 := P_union p (A ∩ C) (B ∩ C)
  have e1 : (A ∪ B) ∩ C = A ∩ C ∪ B ∩ C := Finset.union_inter_distrib_right A B C
  have e2 : A ∩ C ∩ (B ∩ C) = A ∩ B ∩ C := by ext ω; simp; tauto
  rw [e1, e2] at h3
  linarith

/-- On two dice with `A` = first even, `B` = second even, `C` = sum at least `8`: the probabilities are
`1/2, 1/2, 5/12`, the intersections `1/4, 1/4, 1/4, 1/6`, and `P(A ∪ B ∪ C) = 5/6`. -/
theorem dice_three_events :
    let A : Finset Dice2 := univ.filter (fun w => 2 ∣ face w.1)
    let B : Finset Dice2 := univ.filter (fun w => 2 ∣ face w.2)
    let C : Finset Dice2 := univ.filter (fun w => 8 ≤ sum2 w)
    P (unif Dice2) A = 1 / 2 ∧ P (unif Dice2) B = 1 / 2 ∧ P (unif Dice2) C = 5 / 12 ∧
      P (unif Dice2) (A ∩ B) = 1 / 4 ∧ P (unif Dice2) (A ∩ C) = 1 / 4 ∧ P (unif Dice2) (B ∩ C) = 1 / 4 ∧
      P (unif Dice2) (A ∩ B ∩ C) = 1 / 6 ∧ P (unif Dice2) (A ∪ B ∪ C) = 5 / 6 ∧
      (1 / 2 : ℚ) + 1 / 2 + 5 / 12 - 1 / 4 - 1 / 4 - 1 / 4 + 1 / 6 = 5 / 6 := by
  decide +kernel

/-! ## exr-prob-three-dice -/

/-- The sample space of three dice. -/
abbrev Dice3 := Fin 6 × Fin 6 × Fin 6

/-- The sum of three dice. -/
def sum3 (w : Dice3) : ℕ := face w.1 + face w.2.1 + face w.2.2

/-- Three dice have `216` outcomes; `P(sum ≥ 15) = 20/216 = 5/54` and `P(sum ≥ 15 | first = 6) = 10/36 = 5/18`. -/
theorem three_dice : Fintype.card Dice3 = 216 ∧ (univ.filter (fun w : Dice3 => 15 ≤ sum3 w)).card = 20 ∧
    P (unif Dice3) (univ.filter (fun w => 15 ≤ sum3 w)) = 5 / 54 ∧
    cond (unif Dice3) (univ.filter (fun w => 15 ≤ sum3 w)) (univ.filter (fun w => w.1 = 5)) = 5 / 18 ∧
    (10 / 36 : ℚ) = 5 / 18 := by
  decide +kernel

/-- The counts of the sums `3, …, 18` are `1, 3, 6, 10, 15, 21, 25, 27, 27, 25, 21, 15, 10, 6, 3, 1`: the most
likely sums are `10` and `11`, with `27` outcomes each. -/
theorem three_dice_counts :
    (∀ s : Fin 16, (univ.filter (fun w : Dice3 => sum3 w = s + 3)).card =
      ![1, 3, 6, 10, 15, 21, 25, 27, 27, 25, 21, 15, 10, 6, 3, 1] s) ∧
    ∀ s : ℕ, (univ.filter (fun w : Dice3 => sum3 w = s)).card ≤ 27 := by
  refine ⟨by decide +kernel, fun s => ?_⟩
  by_cases h : 3 ≤ s ∧ s ≤ 18
  · obtain ⟨h3, h18⟩ := h
    interval_cases s <;> decide +kernel
  · have : (univ.filter (fun w : Dice3 => sum3 w = s)) = ∅ := by
      apply Finset.filter_eq_empty_iff.2
      intro w _
      simp only [sum3, face]
      have := w.1.isLt; have := w.2.1.isLt; have := w.2.2.isLt
      omega
    rw [this]; simp

end KaluzaKlein.Probability.WorkedProbability
