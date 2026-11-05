import KaluzaKlein.PartI.Probability.SampleSpaces

/-!
# Probability and statistics, sections "Conditional probability", "Independence" and "Bayes' rule"

The claims of the blocks `def-conditional-probability` to `exr-prob-bayes-coin-sage` of
`tutorial/part-i-08-probability.qmd`, in the finite model of `SampleSpaces` (weights `p : Ω → ℚ`, events
`Finset Ω`, `P p A`): conditional probabilities `cond p A B = P(A ∩ B) / P(B)`, the chain rule and the law of total
probability, independence and mutual independence, Bayes' rule, and the running examples (two dice, the diagnostic
test, cards, Monty Hall, the two-headed coin, the three urns).
-/

open Finset

namespace KaluzaKlein.Probability.Conditioning

open SampleSpaces

/-! ## def-conditional-probability -/

/-- The conditional probability of `A` given `B`: `P(A ∩ B) / P(B)`. -/
def cond {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (A B : Finset Ω) : ℚ := P p (A ∩ B) / P p B

/-- `A ↦ P(A | B)` is again a probability measure: non-negative, one on `B` (and on `Ω`), and additive. -/
theorem cond_isMeasure {Ω : Type} [Fintype Ω] [DecidableEq Ω] {p : Ω → ℚ} (hp : IsDist p) (A B : Finset Ω)
    (hB : 0 < P p B) {C D : Finset Ω} (hCD : Disjoint C D) :
    0 ≤ cond p A B ∧ cond p B B = 1 ∧ cond p univ B = 1 ∧ cond p (C ∪ D) B = cond p C B + cond p D B := by
  refine ⟨div_nonneg (P_nonneg hp _) hB.le, ?_, ?_, ?_⟩
  · simp [cond, div_self hB.ne']
  · simp [cond, div_self hB.ne']
  · simp only [cond, Finset.union_inter_distrib_right]
    rw [P_union_of_disjoint p (Finset.disjoint_of_subset_left Finset.inter_subset_left
      (Finset.disjoint_of_subset_right Finset.inter_subset_left hCD)), add_div]

/-- The chain rule `P(A ∩ B) = P(A | B) P(B)`. -/
theorem chain_rule {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (A B : Finset Ω) (hB : P p B ≠ 0) :
    P p (A ∩ B) = cond p A B * P p B := by
  simp [cond, div_mul_cancel₀ _ hB]

/-- The law of total probability: if the `B i` partition `Ω` (disjoint, covering, each of positive probability)
then `P(A) = ∑ P(A | B i) P(B i)`. -/
theorem total_probability {Ω ι : Type} [Fintype Ω] [DecidableEq Ω] [Fintype ι] (p : Ω → ℚ) (A : Finset Ω)
    (B : ι → Finset Ω) (hdisj : Pairwise (fun i j => Disjoint (B i) (B j))) (hcover : univ.biUnion B = univ)
    (hpos : ∀ i, P p (B i) ≠ 0) :
    P p A = ∑ i, cond p A (B i) * P p (B i) := by
  simp_rw [← chain_rule p A _ (hpos _)]
  have : A = univ.biUnion (fun i => A ∩ B i) := by
    rw [← Finset.inter_biUnion, hcover, Finset.inter_univ]
  conv_lhs => rw [this]
  exact Finset.sum_biUnion fun i _ j _ hij =>
    Finset.disjoint_of_subset_left Finset.inter_subset_right
      (Finset.disjoint_of_subset_right Finset.inter_subset_right (hdisj hij))

/-- mathlib's conditional measure: `μ[A | B] = μ(B)⁻¹ μ(B ∩ A)` and the chain rule `μ(B ∩ A) = μ[A | B] μ(B)`. -/
theorem cond_measure {Ω : Type} [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω)
    [MeasureTheory.IsFiniteMeasure μ] (A B : Set Ω) (hB : MeasurableSet B) :
    ProbabilityTheory.cond μ B A = (μ B)⁻¹ * μ (B ∩ A) ∧ ProbabilityTheory.cond μ B A * μ B = μ (B ∩ A) :=
  ⟨ProbabilityTheory.cond_apply hB μ A, ProbabilityTheory.cond_mul_eq_inter hB A μ⟩

/-! ## exm-prob-dice-conditional -/

/-- `P(A | B) = (6/36) / (18/36) = 1/3`, twice `P(A) = 1/6`; `P(C | B) = (3/36) / (18/36) = 1/6 = P(C)`. -/
theorem dice_conditional : cond (unif Dice2) A10 B4 = 1 / 3 ∧ P (unif Dice2) (A10 ∩ B4) = 6 / 36 ∧
    P (unif Dice2) B4 = 18 / 36 ∧ P (unif Dice2) A10 = 1 / 6 ∧ cond (unif Dice2) C7 B4 = 1 / 6 ∧
    P (unif Dice2) (C7 ∩ B4) = 3 / 36 ∧ P (unif Dice2) C7 = 1 / 6 := by
  decide +kernel

/-- `P(C | first die = i) = 1/6` for every `i`: each row contains exactly one outcome with sum `7`. -/
theorem seven_given_row (i : Fin 6) : cond (unif Dice2) C7 (row i) = 1 / 6 ∧ (C7 ∩ row i).card = 1 := by
  fin_cases i <;> decide +kernel

/-! ## exm-prob-cards-chain -/

/-- Two cards drawn without replacement: ordered pairs of distinct cards from `52`, each of weight `1/2652`; the aces
are the cards `0, 1, 2, 3`. -/
def cardW : Fin 52 × Fin 52 → ℚ := fun w => if w.1 = w.2 then 0 else 1 / 2652

/-- The first card is an ace. -/
def aceFirst : Finset (Fin 52 × Fin 52) := univ.filter (fun w => w.1.val < 4)

/-- The second card is an ace. -/
def aceSecond : Finset (Fin 52 × Fin 52) := univ.filter (fun w => w.2.val < 4)

/-- The probability of an event of a product space as a double sum (for the kernel to evaluate). -/
theorem P_prod_eq {α β : Type} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β] (p : α × β → ℚ) (A : Finset (α × β)) :
    P p A = ∑ a, ∑ b, if (a, b) ∈ A then p (a, b) else 0 := by
  rw [P, ← Finset.sum_filter, Fintype.sum_prod_type]

/-- The card weights form a probability distribution. -/
theorem cardW_isDist : IsDist cardW := by
  refine ⟨fun w => by unfold cardW; split_ifs <;> norm_num, ?_⟩
  rw [Fintype.sum_prod_type]; decide +kernel

/-- `P(ace first) = 4/52`, `P(ace second | ace first) = 3/51`, `P(two aces) = 1/221`, and by total probability
`P(ace second) = 4/52` although the two events are not independent. -/
theorem cards_chain : P cardW aceFirst = 4 / 52 ∧ cond cardW aceSecond aceFirst = 3 / 51 ∧
    P cardW (aceSecond ∩ aceFirst) = 1 / 221 ∧ (4 / 52 : ℚ) * (3 / 51) = 1 / 221 ∧
    P cardW aceSecond = 4 / 52 ∧
    cond cardW aceSecond aceFirstᶜ = 4 / 51 ∧ P cardW aceFirstᶜ = 48 / 52 ∧
    (3 / 51 : ℚ) * (4 / 52) + (4 / 51) * (48 / 52) = 4 / 52 ∧
    P cardW (aceSecond ∩ aceFirst) ≠ P cardW aceSecond * P cardW aceFirst := by
  simp only [cond, P_prod_eq]
  decide +kernel

/-! ## exm-prob-test-total -/

/-- The diagnostic-test model: an outcome is (ill?, positive?), with prevalence `prev`, sensitivity `sens` and
false-positive rate `fpr`. -/
def testW (prev sens fpr : ℚ) : Bool × Bool → ℚ := fun w =>
  (if w.1 then prev else 1 - prev) * (if w.2 then (if w.1 then sens else fpr) else 1 - (if w.1 then sens else fpr))

/-- The event "ill". -/
def ill : Finset (Bool × Bool) := univ.filter (fun w => w.1 = true)

/-- The event "positive". -/
def pos : Finset (Bool × Bool) := univ.filter (fun w => w.2 = true)

/-- Total probability over `{ill, healthy}`: `P(+) = sens · prev + fpr · (1 - prev)`. -/
theorem test_total (prev sens fpr : ℚ) : P (testW prev sens fpr) pos = sens * prev + fpr * (1 - prev) := by
  simp [P, pos, testW, Finset.sum_filter, Fintype.sum_prod_type]
  ring

/-- With prevalence `1/10`, sensitivity `9/10` and false-positive rate `1/5`: `P(+) = 9/100 + 18/100 = 27/100`, two
thirds of which comes from the healthy. -/
theorem test_total_values : P (testW (1 / 10) (9 / 10) (1 / 5)) pos = 27 / 100 ∧
    (9 / 10 : ℚ) * (1 / 10) = 9 / 100 ∧ (1 / 5 : ℚ) * (9 / 10) = 18 / 100 ∧
    P (testW (1 / 10) (9 / 10) (1 / 5)) (pos ∩ illᶜ) / P (testW (1 / 10) (9 / 10) (1 / 5)) pos = 2 / 3 := by
  decide +kernel

/-! ## exr-prob-conditional-table -/

/-- Row by row, `P(sum ≥ 10 | first = i) = 0, 0, 0, 1/6, 1/3, 1/2` and `P(sum = 7 | first = i) = 1/6`. -/
theorem conditional_table (i : Fin 6) :
    cond (unif Dice2) A10 (row i) = ![0, 0, 0, 1 / 6, 1 / 3, 1 / 2] i ∧ cond (unif Dice2) C7 (row i) = 1 / 6 := by
  fin_cases i <;> decide +kernel

/-- The rows partition the grid, each with probability `1/6`, and weighting the conditional probabilities by
`1/6` reproduces `P(A) = 1/6` and `P(C) = 1/6`. -/
theorem conditional_table_total :
    (∀ i, P (unif Dice2) (row i) = 1 / 6) ∧
      ∑ i, cond (unif Dice2) A10 (row i) * P (unif Dice2) (row i) = P (unif Dice2) A10 ∧
      ∑ i, cond (unif Dice2) C7 (row i) * P (unif Dice2) (row i) = P (unif Dice2) C7 := by
  decide +kernel

/-! ## exr-prob-conditional-measure -/

/-- The three-event chain rule `P(A ∩ B ∩ C) = P(A | B ∩ C) P(B | C) P(C)`. -/
theorem chain_rule_three {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (A B C : Finset Ω)
    (hBC : P p (B ∩ C) ≠ 0) (hC : P p C ≠ 0) :
    P p (A ∩ B ∩ C) = cond p A (B ∩ C) * cond p B C * P p C := by
  rw [mul_assoc, ← chain_rule p B C hC, ← chain_rule p A (B ∩ C) hBC, Finset.inter_assoc]

/-- Conditioning on `B` gives a probability distribution on `Ω`: the weights `ω ↦ P({ω} | B)` are non-negative and
sum to one. -/
theorem cond_isDist {Ω : Type} [Fintype Ω] [DecidableEq Ω] {p : Ω → ℚ} (hp : IsDist p) (B : Finset Ω)
    (hB : 0 < P p B) : IsDist (fun ω => cond p {ω} B) := by
  refine ⟨fun ω => div_nonneg (P_nonneg hp _) hB.le, ?_⟩
  simp only [cond, ← Finset.sum_div]
  rw [div_eq_one_iff_eq hB.ne']
  have : ∀ ω, P p ({ω} ∩ B) = if ω ∈ B then p ω else 0 := by
    intro ω; by_cases h : ω ∈ B
    · rw [Finset.singleton_inter_of_mem h, if_pos h]; simp [P]
    · rw [Finset.singleton_inter_of_notMem h, if_neg h]; simp [P]
  simp only [this, Finset.sum_ite_mem, Finset.univ_inter]; rfl

/-! ## exr-prob-monty-hall -/

/-- The Monty Hall game: outcomes (prize door, chosen door, opened door), the host opening a prize-free unchosen
door uniformly among the possibilities. -/
def mhW : Fin 3 × Fin 3 × Fin 3 → ℚ := fun w =>
  if w.2.2 = w.1 ∨ w.2.2 = w.2.1 then 0 else if w.1 = w.2.1 then 1 / 18 else 1 / 9

/-- The weights are a probability distribution on `12` outcomes of positive weight. -/
theorem mhW_isDist : IsDist mhW ∧ (univ.filter (fun w => mhW w ≠ 0)).card = 12 := by
  refine ⟨⟨fun w => ?_, ?_⟩, ?_⟩
  · unfold mhW; split_ifs <;> norm_num
  · decide +kernel
  · decide +kernel

/-- Switching wins exactly when the first choice was wrong: probability `2/3`; staying wins with `1/3`. -/
theorem monty_hall : P mhW (univ.filter (fun w => w.1 ≠ w.2.1)) = 2 / 3 ∧
    P mhW (univ.filter (fun w => w.1 = w.2.1)) = 1 / 3 := by
  decide +kernel

/-- Given that door `1` was chosen and door `3` opened, the prize is behind door `2` with probability `2/3` and behind
door `1` with probability `1/3`. -/
theorem monty_hall_conditional :
    cond mhW (univ.filter (fun w => w.1 = 1)) (univ.filter (fun w => w.2.1 = 0 ∧ w.2.2 = 2)) = 2 / 3 ∧
      cond mhW (univ.filter (fun w => w.1 = 0)) (univ.filter (fun w => w.2.1 = 0 ∧ w.2.2 = 2)) = 1 / 3 := by
  decide +kernel

/-! ## def-independence -/

/-- Two events are independent when `P(A ∩ B) = P(A) P(B)`. -/
def indep {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (A B : Finset Ω) : Prop :=
  P p (A ∩ B) = P p A * P p B

instance {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (A B : Finset Ω) : Decidable (indep p A B) := by
  unfold indep; infer_instance

/-- When `P(B) > 0`, independence is equivalent to `P(A | B) = P(A)`. -/
theorem indep_iff_cond {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (A B : Finset Ω) (hB : P p B ≠ 0) :
    indep p A B ↔ cond p A B = P p A := by
  simp only [indep, cond, div_eq_iff hB]

/-- A family of events is mutually independent when the probability of every sub-intersection is the product. -/
def mutIndep {Ω ι : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (A : ι → Finset Ω) : Prop :=
  ∀ S : Finset ι, P p (S.inf A) = ∏ i ∈ S, P p (A i)

/-- Mutual independence implies pairwise independence. -/
theorem mutIndep_pairwise {Ω ι : Type} [Fintype Ω] [DecidableEq Ω] [DecidableEq ι] {p : Ω → ℚ}
    {A : ι → Finset Ω} (h : mutIndep p A) {i j : ι} (hij : i ≠ j) : indep p (A i) (A j) := by
  have := h {i, j}
  rwa [Finset.inf_insert, Finset.inf_singleton, Finset.prod_pair hij] at this

/-- The product weights of two experiments run independently. -/
def prodW2 {Ω Ω' : Type} (p : Ω → ℚ) (q : Ω' → ℚ) : Ω × Ω' → ℚ := fun w => p w.1 * q w.2

/-- Under the product measure, `P(A × B) = P(A) P(B)`: events depending on different trials are independent. -/
theorem prodW2_prob {Ω Ω' : Type} [Fintype Ω] [Fintype Ω'] (p : Ω → ℚ) (q : Ω' → ℚ) (A : Finset Ω)
    (B : Finset Ω') : P (prodW2 p q) (A ×ˢ B) = P p A * P q B := by
  simp [P, prodW2, Finset.sum_product, Finset.sum_mul_sum]

/-- Independence is symmetric. -/
theorem indep_symm {Ω : Type} [Fintype Ω] [DecidableEq Ω] {p : Ω → ℚ} {A B : Finset Ω} (h : indep p A B) :
    indep p B A := by
  unfold indep at *; rw [Finset.inter_comm, h, mul_comm]

/-! ## exm-prob-independent-dice -/

/-- "Sum 7" and "first die at least 4" are independent: `P(C ∩ B) = 3/36 = (1/6)(1/2)`; likewise `C` and each row,
`P(C ∩ {first = i}) = 1/36 = (1/6)(1/6)`. -/
theorem seven_indep : indep (unif Dice2) C7 B4 ∧ P (unif Dice2) (C7 ∩ B4) = 3 / 36 ∧
    (1 / 6 : ℚ) * (1 / 2) = 3 / 36 ∧ ∀ i, indep (unif Dice2) C7 (row i) ∧ P (unif Dice2) (C7 ∩ row i) = 1 / 36 := by
  decide +kernel

/-- `A` and `B` are dependent: `P(A ∩ B) = 1/6 ≠ (1/6)(1/2)`. -/
theorem A_B_dependent : ¬ indep (unif Dice2) A10 B4 ∧ P (unif Dice2) (A10 ∩ B4) = 1 / 6 := by
  decide +kernel

/-- No other sum is independent of the first die: for every `s ≠ 7` some row `i` has
`P(sum = s ∩ first = i) ≠ P(sum = s) P(first = i)`. -/
theorem seven_unique (s : ℕ) (h2 : 2 ≤ s) (h12 : s ≤ 12) (hs : s ≠ 7) :
    ∃ i, ¬ indep (unif Dice2) (sumEq s) (row i) := by
  interval_cases s <;> first | exact absurd rfl hs | decide +kernel

/-! ## exm-prob-pairwise-not-mutual -/

/-- Two coins: `A` first heads, `B` second heads, `C` the two agree. -/
abbrev Coins2 := Fin 2 → Bool

/-- Each of `A`, `B`, `C` has probability `1/2`, each pair is independent with intersection `1/4`
(`A ∩ C` = "both heads"), but `A ∩ B ∩ C = {HH}` has probability `1/4 ≠ 1/8`. -/
theorem pairwise_not_mutual :
    let A : Finset Coins2 := univ.filter (fun w => w 0 = true)
    let B : Finset Coins2 := univ.filter (fun w => w 1 = true)
    let C : Finset Coins2 := univ.filter (fun w => w 0 = w 1)
    P (unif Coins2) A = 1 / 2 ∧ P (unif Coins2) B = 1 / 2 ∧ P (unif Coins2) C = 1 / 2 ∧
      indep (unif Coins2) A B ∧ indep (unif Coins2) A C ∧ indep (unif Coins2) B C ∧
      A ∩ C = {fun _ => true} ∧ A ∩ B ∩ C = {fun _ => true} ∧ P (unif Coins2) (A ∩ B ∩ C) = 1 / 4 ∧
      P (unif Coins2) (A ∩ B ∩ C) ≠ P (unif Coins2) A * P (unif Coins2) B * P (unif Coins2) C := by
  decide +kernel

/-! ## exm-prob-bernoulli-trials -/

/-- `n` independent trials with success probability `p`: a string with `k` successes has weight
`p ^ k (1 - p) ^ (n - k)`. -/
def bernW (p : ℚ) (n : ℕ) : (Fin n → Bool) → ℚ := fun w => p ^ heads w * (1 - p) ^ (n - heads w)

/-- The event "exactly `k` successes". -/
def headsEq (n k : ℕ) : Finset (Fin n → Bool) := univ.filter (fun w => heads w = k)

/-- The weight of a string is the product over the trials of `p` or `1 - p` (the product rule). -/
theorem bernW_eq_prod (p : ℚ) (n : ℕ) (w : Fin n → Bool) : bernW p n w = ∏ i, (if w i then p else 1 - p) := by
  rw [Finset.prod_ite, Finset.prod_const, Finset.prod_const]
  have := Finset.card_filter_add_card_filter_not (s := (univ : Finset (Fin n))) (p := fun i => w i = true)
  simp only [Finset.card_univ, Fintype.card_fin] at this
  simp only [bernW, heads]
  congr 2
  omega

/-- There are `n.choose k` strings with `k` successes. -/
theorem card_headsEq (n k : ℕ) : (headsEq n k).card = n.choose k := by
  have : (headsEq n k).card = (Finset.powersetCard k (univ : Finset (Fin n))).card := by
    refine Finset.card_bij' (fun w _ => univ.filter (fun i => w i = true)) (fun S _ => fun i => decide (i ∈ S))
      ?_ ?_ ?_ ?_
    · intro w hw
      rw [Finset.mem_powersetCard]
      exact ⟨Finset.subset_univ _, (Finset.mem_filter.1 hw).2⟩
    · intro S hS
      rw [headsEq, Finset.mem_filter]
      refine ⟨mem_univ _, ?_⟩
      rw [← (Finset.mem_powersetCard.1 hS).2]
      unfold heads
      congr 1; ext i; simp
    · intro w _; funext i; simp
    · intro S _; ext i; simp
  rw [this, Finset.card_powersetCard, Finset.card_univ, Fintype.card_fin]

/-- `P(k successes) = n.choose k · p ^ k (1 - p) ^ (n - k)`. -/
theorem binomial_prob (p : ℚ) (n k : ℕ) : P (bernW p n) (headsEq n k) = n.choose k * p ^ k * (1 - p) ^ (n - k) := by
  unfold P
  rw [Finset.sum_congr rfl (fun w hw => by
    simp only [headsEq, mem_filter] at hw; rw [bernW, hw.2]), Finset.sum_const, card_headsEq]
  simp [mul_assoc]

/-- Two fair coins: `1/4, 1/2, 1/4`; four fair coins: `1/16, 4/16, 6/16, 4/16, 1/16`. -/
theorem binomial_small : (∀ k : Fin 3, P (bernW (1 / 2) 2) (headsEq 2 k) = ![1 / 4, 1 / 2, 1 / 4] k) ∧
    ∀ k : Fin 5, P (bernW (1 / 2) 4) (headsEq 4 k) = ![1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16] k := by
  decide +kernel

/-! ## exr-prob-independent-complement -/

/-- If `A` and `B` are independent so are `A` and `Bᶜ`, and `Aᶜ` and `Bᶜ`. -/
theorem indep_compl {Ω : Type} [Fintype Ω] [DecidableEq Ω] {p : Ω → ℚ} (hp : IsDist p) {A B : Finset Ω}
    (h : indep p A B) : indep p A Bᶜ ∧ indep p Aᶜ Bᶜ := by
  have key : ∀ {A B : Finset Ω}, indep p A B → indep p A Bᶜ := by
    intro A B h
    unfold indep at *
    have hsplit : P p A = P p (A ∩ B) + P p (A ∩ Bᶜ) := by
      rw [← P_union_of_disjoint p (Finset.disjoint_of_subset_left Finset.inter_subset_right
        (Finset.disjoint_of_subset_right Finset.inter_subset_right disjoint_compl_right))]
      congr 1
      ext ω; simp; tauto
    rw [P_compl hp]
    linarith
  exact ⟨key h, indep_symm (key (indep_symm (key h)))⟩

/-- Replacing one event of a mutually independent family by its complement keeps the family mutually
independent. -/
theorem mutIndep_update {Ω ι : Type} [Fintype Ω] [DecidableEq Ω] [DecidableEq ι] {p : Ω → ℚ} (hp : IsDist p)
    {A : ι → Finset Ω} (h : mutIndep p A) (j : ι) : mutIndep p (Function.update A j (A j)ᶜ) := by
  intro S
  by_cases hj : j ∈ S
  · have hS : S = insert j (S.erase j) := (Finset.insert_erase hj).symm
    have hjT : j ∉ S.erase j := Finset.notMem_erase j S
    have hcongr : (S.erase j).inf (Function.update A j (A j)ᶜ) = (S.erase j).inf A :=
      Finset.inf_congr rfl fun i hi => Function.update_of_ne (Finset.ne_of_mem_erase hi) _ _
    have hcongr' : ∏ i ∈ S.erase j, P p (Function.update A j (A j)ᶜ i) = ∏ i ∈ S.erase j, P p (A i) :=
      Finset.prod_congr rfl fun i hi => by rw [Function.update_of_ne (Finset.ne_of_mem_erase hi)]
    rw [hS, Finset.inf_insert, Finset.prod_insert hjT, hcongr, hcongr', Function.update_self]
    have h1 := h (insert j (S.erase j))
    rw [Finset.inf_insert, Finset.prod_insert hjT] at h1
    have h2 := h (S.erase j)
    have hsplit : P p ((S.erase j).inf A) = P p (A j ⊓ (S.erase j).inf A) + P p ((A j)ᶜ ⊓ (S.erase j).inf A) := by
      rw [← P_union_of_disjoint]
      · congr 1; ext ω; simp; tauto
      · exact Finset.disjoint_of_subset_left Finset.inter_subset_left
          (Finset.disjoint_of_subset_right Finset.inter_subset_left disjoint_compl_right)
    rw [P_compl hp]
    linarith
  · have hcongr : S.inf (Function.update A j (A j)ᶜ) = S.inf A :=
      Finset.inf_congr rfl fun i hi => Function.update_of_ne (fun h => hj (h ▸ hi)) _ _
    have hcongr' : ∏ i ∈ S, P p (Function.update A j (A j)ᶜ i) = ∏ i ∈ S, P p (A i) :=
      Finset.prod_congr rfl fun i hi => by rw [Function.update_of_ne (fun h => hj (h ▸ hi))]
    rw [hcongr, hcongr']
    exact h S

/-- The events complemented on an index set `T`. -/
def complOn {Ω ι : Type} [Fintype Ω] [DecidableEq Ω] [DecidableEq ι] (A : ι → Finset Ω) (T : Finset ι) : ι → Finset Ω :=
  fun i => if i ∈ T then (A i)ᶜ else A i

/-- The complements of mutually independent events are mutually independent (any subfamily may be complemented). -/
theorem mutIndep_complOn {Ω ι : Type} [Fintype Ω] [DecidableEq Ω] [DecidableEq ι] {p : Ω → ℚ} (hp : IsDist p)
    {A : ι → Finset Ω} (h : mutIndep p A) (T : Finset ι) : mutIndep p (complOn A T) := by
  induction T using Finset.induction_on with
  | empty =>
    have : complOn A ∅ = A := by funext i; simp [complOn]
    rw [this]; exact h
  | insert a T ha ih =>
    have : complOn A (insert a T) = Function.update (complOn A T) a (complOn A T a)ᶜ := by
      funext i
      by_cases hi : i = a
      · subst hi; simp [complOn, ha]
      · simp [complOn, Function.update_of_ne hi, hi]
    rw [this]
    exact mutIndep_update hp ih a

/-- If `A₁, …, Aₙ` are mutually independent with `P(Aᵢ) = pᵢ` then `P(none occurs) = ∏ (1 - pᵢ)` and
`P(at least one occurs) = 1 - ∏ (1 - pᵢ)`. -/
theorem none_occurs {Ω ι : Type} [Fintype Ω] [DecidableEq Ω] [Fintype ι] [DecidableEq ι] {p : Ω → ℚ}
    (hp : IsDist p) {A : ι → Finset Ω} (h : mutIndep p A) :
    P p (univ.inf fun i => (A i)ᶜ) = ∏ i, (1 - P p (A i)) ∧
      P p (univ.sup A) = 1 - ∏ i, (1 - P p (A i)) := by
  have h1 := mutIndep_complOn hp h univ univ
  simp only [complOn, Finset.mem_univ, if_true, P_compl hp] at h1
  refine ⟨h1, ?_⟩
  have : univ.sup A = (univ.inf fun i => (A i)ᶜ)ᶜ := by
    ext ω; simp [Finset.mem_sup, Finset.mem_inf]
  rw [this, P_compl hp, h1]

/-! ## exr-prob-pairwise-sage -/

/-- Three coins with `A i` = "toss `i` is heads" and `D` = "even number of heads": every pair and every triple is
independent, but the four events together are not (`A₁ ∩ A₂ ∩ A₃` determines `D`). -/
theorem three_coins_parity :
    let A : Fin 4 → Finset Coins3 := ![univ.filter (fun w => w 0 = true), univ.filter (fun w => w 1 = true),
      univ.filter (fun w => w 2 = true), univ.filter (fun w => heads w % 2 = 0)]
    (∀ S : Finset (Fin 4), S.card ≤ 3 → P (unif Coins3) (S.inf A) = ∏ i ∈ S, P (unif Coins3) (A i)) ∧
      P (unif Coins3) (univ.inf A) ≠ ∏ i, P (unif Coins3) (A i) ∧ P (unif Coins3) (univ.inf A) = 0 := by
  decide +kernel

/-! ## exr-prob-rare-events -/

/-- An event of probability `1/n` tried `n` times never occurs with probability `(1 - 1/n) ^ n → e⁻¹`. -/
theorem rare_never : Filter.Tendsto (fun n : ℕ => (1 - 1 / (n : ℝ)) ^ n) Filter.atTop (nhds (Real.exp (-1))) := by
  have := Real.tendsto_one_add_div_pow_exp (-1)
  refine this.congr fun n => ?_
  congr 1; ring

/-- `m` tries of an event of probability `λ / m`: the probability of exactly `k` occurrences,
`m.choose k (λ/m) ^ k (1 - λ/m) ^ (m - k)`, tends to the Poisson probability `e^{-λ} λ^k / k!`. -/
theorem poisson_limit (lam : ℝ) (k : ℕ) :
    Filter.Tendsto (fun m : ℕ => (m.choose k : ℝ) * (lam / m) ^ k * (1 - lam / m) ^ (m - k)) Filter.atTop
      (nhds (Real.exp (-lam) * lam ^ k / k.factorial)) := by
  apply ProbabilityTheory.tendsto_choose_mul_pow_of_tendsto_mul_atTop k
  refine (tendsto_const_nhds (x := lam)).congr' ?_
  filter_upwards [Filter.eventually_ne_atTop 0] with m hm
  have : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  field_simp

/-! ## thm-bayes-rule -/

/-- Bayes' rule `P(A | B) = P(B | A) P(A) / P(B)`. -/
theorem bayes {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (A B : Finset Ω) (hA : P p A ≠ 0)
    (hB : P p B ≠ 0) : cond p A B = cond p B A * P p A / P p B := by
  rw [cond, cond, div_mul_cancel₀ _ hA, Finset.inter_comm]

/-- Bayes' rule for a partition `A i`: the posterior `P(Aᵢ | B) = P(B | Aᵢ) P(Aᵢ) / ∑ⱼ P(B | Aⱼ) P(Aⱼ)`. -/
theorem bayes_partition {Ω ι : Type} [Fintype Ω] [DecidableEq Ω] [Fintype ι] (p : Ω → ℚ) (A : ι → Finset Ω)
    (B : Finset Ω) (hdisj : Pairwise (fun i j => Disjoint (A i) (A j))) (hcover : univ.biUnion A = univ)
    (hpos : ∀ i, P p (A i) ≠ 0) (hB : P p B ≠ 0) (i : ι) :
    cond p (A i) B = cond p B (A i) * P p (A i) / ∑ j, cond p B (A j) * P p (A j) := by
  rw [← total_probability p B A hdisj hcover hpos, bayes p (A i) B (hpos i) hB]

/-- mathlib's Bayes' theorem: `μ[B | A] = μ(A)⁻¹ μ[A | B] μ(B)`. -/
theorem bayes_measure {Ω : Type} [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω)
    [MeasureTheory.IsFiniteMeasure μ] (A B : Set Ω) (hA : MeasurableSet A) (hB : MeasurableSet B) :
    ProbabilityTheory.cond μ A B = (μ A)⁻¹ * ProbabilityTheory.cond μ B A * μ B :=
  ProbabilityTheory.cond_eq_inv_mul_cond_mul hA hB μ

/-! ## exm-prob-bayes-test -/

/-- The posterior in the test model: `P(ill | +) = sens · prev / (sens · prev + fpr (1 - prev))`. -/
theorem test_posterior (prev sens fpr : ℚ) (h : sens * prev + fpr * (1 - prev) ≠ 0) :
    cond (testW prev sens fpr) ill pos = sens * prev / (sens * prev + fpr * (1 - prev)) := by
  rw [cond, test_total]
  congr 1
  simp [P, ill, pos, testW, Finset.sum_filter, Fintype.sum_prod_type]

/-- `P(ill | +) = (9/100) / (27/100) = 1/3`: a positive result leaves a two-thirds chance of being healthy. -/
theorem bayes_test : cond (testW (1 / 10) (9 / 10) (1 / 5)) ill pos = 1 / 3 ∧
    P (testW (1 / 10) (9 / 10) (1 / 5)) (ill ∩ pos) = 9 / 100 ∧ (9 / 100 : ℚ) / (27 / 100) = 1 / 3 ∧
    cond (testW (1 / 10) (9 / 10) (1 / 5)) illᶜ pos = 2 / 3 := by
  decide +kernel

/-! ## exm-prob-bayes-rare -/

/-- Prevalence `1/100`, sensitivity `99/100`, false-positive rate `1/20`: `P(+) = 99/10000 + 495/10000 = 594/10000`
and `P(ill | +) = 99/594 = 1/6`; five of six positives are false. -/
theorem bayes_rare : P (testW (1 / 100) (99 / 100) (1 / 20)) pos = 594 / 10000 ∧
    (99 / 100 : ℚ) * (1 / 20) = 495 / 10000 ∧ (99 / 10000 : ℚ) + 495 / 10000 = 594 / 10000 ∧
    cond (testW (1 / 100) (99 / 100) (1 / 20)) ill pos = 1 / 6 ∧ (99 / 594 : ℚ) = 1 / 6 ∧
    cond (testW (1 / 100) (99 / 100) (1 / 20)) illᶜ pos = 5 / 6 := by
  decide +kernel

/-- Two conditionally independent tests: outcomes (ill?, first positive?, second positive?). -/
def testW2 (prev sens fpr : ℚ) : Bool × Bool × Bool → ℚ := fun w =>
  (if w.1 then prev else 1 - prev) * (if w.2.1 then (if w.1 then sens else fpr) else 1 - (if w.1 then sens else fpr))
    * (if w.2.2 then (if w.1 then sens else fpr) else 1 - (if w.1 then sens else fpr))

/-- After two positive tests the posterior is `99/124 ≈ 0.80`. -/
theorem bayes_rare_two : cond (testW2 (1 / 100) (99 / 100) (1 / 20)) (univ.filter (fun w => w.1 = true))
    (univ.filter (fun w => w.2.1 = true ∧ w.2.2 = true)) = 99 / 124 ∧
    (99 / 124 : ℚ) = 99 / (99 + 25) ∧ |(99 / 124 : ℚ) - 0.80| < 0.005 := by
  decide +kernel

/-! ## exm-prob-bayes-coin -/

/-- The bag of coins: an outcome is (two-headed?, the string of `k` tosses); the two-headed coin always shows heads. -/
def coinW (k : ℕ) : Bool × (Fin k → Bool) → ℚ := fun w =>
  if w.1 then (if ∀ i, w.2 i = true then 1 / 1000 else 0) else (999 / 1000) / 2 ^ k

/-- The event "all `k` tosses are heads". -/
def allHeads (k : ℕ) : Finset (Bool × (Fin k → Bool)) := univ.filter (fun w => ∀ i, w.2 i = true)

/-- The event "the coin is two-headed". -/
def twoHeaded (k : ℕ) : Finset (Bool × (Fin k → Bool)) := univ.filter (fun w => w.1 = true)

/-- The posterior probability of the two-headed coin after `k` heads: `(1/1000) / (1/1000 + 0.999 · 2^{-k})`. -/
theorem coin_posterior (k : ℕ) :
    cond (coinW k) (twoHeaded k) (allHeads k) = (1 / 1000) / (1 / 1000 + (999 / 1000) * (1 / 2 ^ k)) := by
  have hall : allHeads k = univ ×ˢ {fun _ => true} := by
    ext ⟨b, w⟩
    simp only [allHeads, mem_filter, mem_univ, true_and, mem_product, mem_singleton]
    constructor
    · intro h; funext i; exact h i
    · intro h i; rw [h]
  have hboth : twoHeaded k ∩ allHeads k = {true} ×ˢ {fun _ => true} := by
    ext ⟨b, w⟩
    simp only [twoHeaded, allHeads, mem_inter, mem_filter, mem_univ, true_and, mem_product, mem_singleton]
    constructor
    · rintro ⟨hb, h⟩; exact ⟨hb, funext fun i => h i⟩
    · rintro ⟨hb, rfl⟩; exact ⟨hb, fun i => rfl⟩
  rw [cond, hboth, hall]
  simp [P, coinW, Finset.sum_product, Fintype.sum_bool]

/-- The posterior is `2/1001 ≈ 0.002` after one head, within `0.01` of `1/2` after ten, above `0.999` after
twenty; the prior of one in a thousand is overturned after about `log₂ 1000 ≈ 10` heads (`2⁹ < 1000 < 2¹⁰`). -/
theorem coin_posterior_values : cond (coinW 1) (twoHeaded 1) (allHeads 1) = 2 / 1001 ∧
    |cond (coinW 1) (twoHeaded 1) (allHeads 1) - 0.002| < 0.0001 ∧
    |cond (coinW 10) (twoHeaded 10) (allHeads 10) - 1 / 2| < 0.01 ∧
    cond (coinW 20) (twoHeaded 20) (allHeads 20) > 0.999 ∧ 2 ^ 9 < 1000 ∧ 1000 < 2 ^ 10 := by
  simp only [coin_posterior]
  norm_num

/-! ## exr-prob-bayes-sequential -/

/-- The odds form of Bayes' rule: posterior odds = likelihood ratio × prior odds. -/
theorem bayes_odds {Ω : Type} [Fintype Ω] [DecidableEq Ω] {p : Ω → ℚ} (hp : IsDist p) (A B : Finset Ω)
    (hA : P p A ≠ 0) (hAc : P p Aᶜ ≠ 0) (hB : P p B ≠ 0) (hAcB : P p (Aᶜ ∩ B) ≠ 0) :
    cond p A B / cond p Aᶜ B = (cond p B A / cond p B Aᶜ) * (P p A / P p Aᶜ) := by
  simp only [cond]
  rw [Finset.inter_comm B A, Finset.inter_comm B Aᶜ]
  field_simp

/-- Prior odds `1/99`, likelihood ratio `(99/100)/(1/20) = 99/5`; posterior odds `1/5` after one test (probability
`1/6`) and `99/25` after two (probability `99/124 ≈ 0.798`). -/
theorem two_tests_odds : (1 / 100 : ℚ) / (1 - 1 / 100) = 1 / 99 ∧ (99 / 100 : ℚ) / (1 / 20) = 99 / 5 ∧
    (1 / 99 : ℚ) * (99 / 5) = 1 / 5 ∧ (1 / 5 : ℚ) / (1 + 1 / 5) = 1 / 6 ∧
    (1 / 99 : ℚ) * (99 / 5) ^ 2 = 99 / 25 ∧ (99 ^ 2 : ℚ) / (25 * 99) = 99 / 25 ∧
    (99 / 25 : ℚ) / (1 + 99 / 25) = 99 / 124 ∧ |(99 / 124 : ℚ) - 0.798| < 0.0005 := by
  norm_num

/-! ## exr-prob-three-urns -/

/-- Three urns: urn `1` has `2` red and `1` blue, urn `2` has `1` red and `2` blue, urn `3` has `3` blue; an urn is
chosen uniformly and a ball drawn. Outcomes (urn, red?). -/
def urnW : Fin 3 × Bool → ℚ := fun w => (1 / 3) * (if w.2 then ![2, 1, 0] w.1 / 3 else ![1, 2, 3] w.1 / 3)

/-- `P(red) = 1/3`; the posteriors given red are `2/3, 1/3, 0`; `P(blue) = 2/3` and the posteriors given blue are
`1/6, 1/3, 1/2`, proportional to the numbers of blue balls. -/
theorem three_urns :
    let urn : Fin 3 → Finset (Fin 3 × Bool) := fun i => univ.filter (fun w => w.1 = i)
    let red : Finset (Fin 3 × Bool) := univ.filter (fun w => w.2 = true)
    IsDist urnW ∧ P urnW red = 1 / 3 ∧ cond urnW (urn 0) red = 2 / 3 ∧ cond urnW (urn 1) red = 1 / 3 ∧
      cond urnW (urn 2) red = 0 ∧ P urnW redᶜ = 2 / 3 ∧ cond urnW (urn 2) redᶜ = 1 / 2 ∧
      cond urnW (urn 0) redᶜ = 1 / 6 ∧ cond urnW (urn 1) redᶜ = 1 / 3 := by
  refine ⟨⟨fun w => ?_, by decide +kernel⟩, ?_⟩
  · unfold urnW; fin_cases w <;> simp
  · decide +kernel

/-! ## exr-prob-bayes-coin-sage -/

/-- The posterior exceeds `0.99` exactly when the posterior odds `2^k / 999` exceed `99`, that is `2^k > 98901`;
the smallest such `k` is `17`, and `2¹⁶ < 98901 < 2¹⁷` (so `log₂ 98901 ≈ 16.6`). -/
theorem coin_posterior_99 (k : ℕ) :
    (cond (coinW k) (twoHeaded k) (allHeads k) > 99 / 100 ↔ (2 : ℚ) ^ k / 999 > 99) ∧
      ((2 : ℚ) ^ k / 999 > 99 ↔ (2 : ℚ) ^ k > 98901) ∧ (2 : ℕ) ^ 16 < 98901 ∧ 98901 < (2 : ℕ) ^ 17 ∧
      cond (coinW 17) (twoHeaded 17) (allHeads 17) > 99 / 100 ∧
      ¬ cond (coinW 16) (twoHeaded 16) (allHeads 16) > 99 / 100 := by
  simp only [coin_posterior]
  have h2 : (0 : ℚ) < 2 ^ k := by positivity
  refine ⟨?_, ?_, by norm_num, by norm_num, by norm_num, by norm_num⟩
  · rw [gt_iff_lt, gt_iff_lt, div_lt_iff₀ (by positivity), lt_div_iff₀ (by norm_num)]
    constructor <;> intro h <;> nlinarith
  · rw [gt_iff_lt, gt_iff_lt, lt_div_iff₀ (by norm_num)]; norm_num

/-- The table of posteriors for `k = 0, 1, 5, 8, 10, 12, 15, 17, 20`. -/
theorem coin_posterior_table :
    cond (coinW 0) (twoHeaded 0) (allHeads 0) = 1 / 1000 ∧ cond (coinW 5) (twoHeaded 5) (allHeads 5) = 32 / 1031 ∧
      cond (coinW 8) (twoHeaded 8) (allHeads 8) = 256 / 1255 ∧
      cond (coinW 10) (twoHeaded 10) (allHeads 10) = 1024 / 2023 ∧
      cond (coinW 12) (twoHeaded 12) (allHeads 12) = 4096 / 5095 ∧
      cond (coinW 15) (twoHeaded 15) (allHeads 15) = 32768 / 33767 ∧
      cond (coinW 17) (twoHeaded 17) (allHeads 17) = 131072 / 132071 ∧
      cond (coinW 20) (twoHeaded 20) (allHeads 20) = 1048576 / 1049575 := by
  simp only [coin_posterior]; norm_num

end KaluzaKlein.Probability.Conditioning
