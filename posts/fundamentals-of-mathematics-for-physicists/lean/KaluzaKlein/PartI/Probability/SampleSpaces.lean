import Mathlib
import Veriphy.Kit.Levels

/-!
# Probability and statistics, sections "Sample spaces and events" and "Probability measures"

The claims of the blocks `def-sample-space` to `exr-prob-geometric-waiting` of `tutorial/part-i-08-probability.qmd`.
Finite experiments are modelled on a finite type `Ω` with a weight function `p : Ω → ℚ` (the uniform weights
`unif Ω` for equally likely outcomes); the probability of an event `A : Finset Ω` is `P p A = ∑ ω ∈ A, p ω`. The
continuous sample spaces use mathlib's Borel σ-algebra and Lebesgue measure.
-/

open Finset

namespace KaluzaKlein.Probability.SampleSpaces

/-! ## The finite model -/

/-- The probability of the event `A` under the weights `p`: the sum of the weights of its outcomes. -/
def P {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (A : Finset Ω) : ℚ := ∑ ω ∈ A, p ω

/-- The uniform weights on a finite sample space: every outcome has weight `1 / |Ω|`. -/
def unif (Ω : Type) [Fintype Ω] : Ω → ℚ := fun _ => (Fintype.card Ω : ℚ)⁻¹

/-- A weight function is a probability distribution: non-negative weights summing to one. -/
structure IsDist {Ω : Type} [Fintype Ω] (p : Ω → ℚ) : Prop where
  nonneg : ∀ ω, 0 ≤ p ω
  sum_one : ∑ ω, p ω = 1

/-- The face shown by a die, `1, …, 6`, as a function of the outcome `i : Fin 6`. -/
def face (i : Fin 6) : ℕ := i.val + 1

/-- The sample space of two dice: ordered pairs of faces. -/
abbrev Dice2 := Fin 6 × Fin 6

/-- The sum of two dice. -/
def sum2 (w : Dice2) : ℕ := face w.1 + face w.2

/-- The event "the sum is at least `10`" (the event `A` of the figure). -/
def A10 : Finset Dice2 := univ.filter (fun w => 10 ≤ sum2 w)

/-- The event "the first die shows at least `4`" (the event `B` of the figure). -/
def B4 : Finset Dice2 := univ.filter (fun w => 4 ≤ face w.1)

/-- The event "the sum is `7`" (the event `C` of the figure). -/
def C7 : Finset Dice2 := univ.filter (fun w => sum2 w = 7)

/-- The event "the sum is `s`". -/
def sumEq (s : ℕ) : Finset Dice2 := univ.filter (fun w => sum2 w = s)

/-- The event "the first die shows `i`" (a row of the grid). -/
def row (i : Fin 6) : Finset Dice2 := univ.filter (fun w => w.1 = i)

/-- The event "even" for one die. -/
def evenF : Finset (Fin 6) := univ.filter (fun i => 2 ∣ face i)

/-- The event "at least `5`" for one die. -/
def ge5 : Finset (Fin 6) := univ.filter (fun i => 5 ≤ face i)

/-! ## def-sample-space -/

/-- A σ-algebra closed under complements and countable unions is closed under countable intersections. -/
theorem sigma_iInter {Ω : Type} [MeasurableSpace Ω] (A : ℕ → Set Ω) (h : ∀ n, MeasurableSet (A n)) :
    MeasurableSet (⋂ n, A n) :=
  MeasurableSet.iInter h

/-- Every subset of a finite (or countable) sample space with the discrete σ-algebra is an event. -/
theorem measurableSet_of_discrete {Ω : Type} [MeasurableSpace Ω] [DiscreteMeasurableSpace Ω] (A : Set Ω) :
    MeasurableSet A :=
  MeasurableSet.of_discrete

/-- On `ℝ` the intervals are Borel sets, and so are their complements and countable unions. -/
theorem intervals_measurable (a b : ℝ) :
    MeasurableSet (Set.Icc a b) ∧ MeasurableSet (Set.Ioo a b) ∧ MeasurableSet (Set.Icc a b)ᶜ :=
  ⟨measurableSet_Icc, measurableSet_Ioo, measurableSet_Icc.compl⟩

/-! ## exm-prob-coin-die -/

/-- A sample space with `n` outcomes has `2 ^ n` events. -/
theorem card_events (n : ℕ) : Fintype.card (Finset (Fin n)) = 2 ^ n := by simp

/-- A coin has four events and a die has `64`. -/
theorem card_events_coin_die : Fintype.card (Finset (Fin 2)) = 4 ∧ Fintype.card (Finset (Fin 6)) = 64 := by
  simp

/-- "Even" and "at least 5" meet in the single outcome `6` (the face of index `5`). -/
theorem even_inter_ge5 : evenF ∩ ge5 = {5} := by decide +kernel

/-- The family of all subsets is closed under complement and union. -/
theorem powerset_closed (n : ℕ) (A B : Finset (Fin n)) : Aᶜ ∈ (univ : Finset (Finset (Fin n))) ∧
    A ∪ B ∈ (univ : Finset (Finset (Fin n))) := by
  simp

/-! ## exm-prob-two-dice-space -/

/-- Two dice have `36` outcomes. -/
theorem card_dice2 : Fintype.card Dice2 = 36 := by simp

/-- The events `A`, `B`, `C` of the figure have `6`, `18` and `6` outcomes, and `A ⊂ B`. -/
theorem dice_events_card : A10.card = 6 ∧ B4.card = 18 ∧ C7.card = 6 ∧ A10 ⊂ B4 := by decide +kernel

/-- The six outcomes with sum `7`. -/
theorem C7_eq : C7 = {(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)} := by decide +kernel

/-! ## exm-prob-continuous-space -/

/-- The three events of the example are Borel sets of `[0, 1]`. -/
theorem continuous_events_measurable :
    MeasurableSet (Set.Icc (0 : ℝ) (1 / 2)) ∧ MeasurableSet (Set.Ioo (1 / 3 : ℝ) (2 / 3)) ∧
      MeasurableSet {x : ℝ | x ^ 2 < 1 / 2} :=
  ⟨measurableSet_Icc, measurableSet_Ioo, measurableSet_lt (by fun_prop) measurable_const⟩

/-- Every open set of `ℝ` is a Borel set, and the complement of an open set (a closed set) too. -/
theorem open_closed_measurable (U : Set ℝ) (hU : IsOpen U) : MeasurableSet U ∧ MeasurableSet Uᶜ :=
  ⟨hU.measurableSet, hU.measurableSet.compl⟩

/-- Lebesgue measure assigns to an interval its length, and it is countably additive on disjoint Borel sets. -/
theorem length_additive (a b : ℝ) (hab : a ≤ b) (A : ℕ → Set ℝ) (hA : ∀ n, MeasurableSet (A n))
    (hd : Pairwise (fun i j => Disjoint (A i) (A j))) :
    MeasureTheory.volume (Set.Icc a b) = ENNReal.ofReal (b - a) ∧
      MeasureTheory.volume (⋃ n, A n) = ∑' n, MeasureTheory.volume (A n) :=
  ⟨Real.volume_Icc, MeasureTheory.measure_iUnion hd hA⟩

/-! ## exr-prob-three-coins -/

/-- The sample space of three coins: a head is `true`. -/
abbrev Coins3 := Fin 3 → Bool

/-- The number of heads among `n` coins. -/
def heads {n : ℕ} (w : Fin n → Bool) : ℕ := (univ.filter (fun i => w i = true)).card

/-- Three coins have `8` outcomes; "exactly two heads" has `3`, "at least one tail" `7` and "first and last
agree" `4`. -/
theorem three_coins_counts :
    Fintype.card Coins3 = 8 ∧ (univ.filter (fun w : Coins3 => heads w = 2)).card = 3 ∧
      (univ.filter (fun w : Coins3 => ∃ i, w i = false)).card = 7 ∧
      (univ.filter (fun w : Coins3 => w 0 = w 2)).card = 4 := by
  decide +kernel

/-- "At least one tail" is the complement of "all heads". -/
theorem at_least_one_tail_compl :
    univ.filter (fun w : Coins3 => ∃ i, w i = false) = ({fun _ => true} : Finset Coins3)ᶜ := by
  decide +kernel

/-! ## exr-prob-sigma-partition -/

/-- The four blocks `E ∩ L`, `E ∩ Lᶜ`, `Eᶜ ∩ L`, `Eᶜ ∩ Lᶜ` of the partition generated by `E` = even and
`L` = at most `2`, as subsets of `Fin 6` (faces `1, …, 6` are the indices `0, …, 5`). -/
def blocks : Finset (Finset (Fin 6)) := {{1}, {3, 5}, {0}, {2, 4}}

/-- The family generated by the blocks: all unions of blocks. -/
def genF : Finset (Finset (Fin 6)) := blocks.powerset.image (fun s => s.sup id)

/-- "At most 2" as an event of one die. -/
def le2 : Finset (Fin 6) := univ.filter (fun i => face i ≤ 2)

/-- The blocks are the intersections of `E`, `L` and their complements, and they partition the die. -/
theorem blocks_eq : evenF ∩ le2 = {1} ∧ evenF ∩ le2ᶜ = {3, 5} ∧ evenF ᶜ ∩ le2 = {0} ∧
    evenFᶜ ∩ le2ᶜ = {2, 4} ∧ blocks.sup id = univ ∧
    (∀ b ∈ blocks, ∀ b' ∈ blocks, b ≠ b' → Disjoint b b') := by
  decide +kernel

/-- The generated family has `16` elements, contains `E` and `L`, and is closed under complement and union. -/
theorem genF_card_closed : genF.card = 16 ∧ evenF ∈ genF ∧ le2 ∈ genF ∧
    (∀ A ∈ genF, Aᶜ ∈ genF) ∧ (∀ A ∈ genF, ∀ B ∈ genF, A ∪ B ∈ genF) := by
  decide +kernel

/-- Minimality: every family containing `E` and `L` and closed under complement and union contains the generated
family, which is therefore the smallest σ-algebra containing the two events. -/
theorem genF_minimal (G : Finset (Finset (Fin 6))) (hE : evenF ∈ G) (hL : le2 ∈ G)
    (hc : ∀ A ∈ G, Aᶜ ∈ G) (hu : ∀ A ∈ G, ∀ B ∈ G, A ∪ B ∈ G) : genF ⊆ G := by
  have hinter : ∀ A ∈ G, ∀ B ∈ G, A ∩ B ∈ G := by
    intro A hA B hB
    have := hc _ (hu _ (hc _ hA) _ (hc _ hB))
    simpa [Finset.compl_union] using this
  have hempty : (∅ : Finset (Fin 6)) ∈ G := by
    have := hc _ (hu _ hE _ (hc _ hE))
    simpa using this
  have hb : ∀ b ∈ blocks, b ∈ G := by
    intro b hb
    have h1 : ({1} : Finset (Fin 6)) ∈ G := blocks_eq.1 ▸ hinter _ hE _ hL
    have h2 : ({3, 5} : Finset (Fin 6)) ∈ G := blocks_eq.2.1 ▸ hinter _ hE _ (hc _ hL)
    have h3 : ({0} : Finset (Fin 6)) ∈ G := blocks_eq.2.2.1 ▸ hinter _ (hc _ hE) _ hL
    have h4 : ({2, 4} : Finset (Fin 6)) ∈ G := blocks_eq.2.2.2.1 ▸ hinter _ (hc _ hE) _ (hc _ hL)
    simp only [blocks, mem_insert, mem_singleton] at hb
    rcases hb with rfl | rfl | rfl | rfl <;> assumption
  intro A hA
  simp only [genF, mem_image, mem_powerset] at hA
  obtain ⟨s, hs, rfl⟩ := hA
  induction s using Finset.induction_on with
  | empty => simpa using hempty
  | insert a s ha ih =>
    rw [Finset.sup_insert]
    exact hu _ (hb a (hs (mem_insert_self a s))) _ (ih (fun x hx => hs (mem_insert_of_mem hx)))

/-- The generated family has `16 = 2 ^ 4` elements, fewer than the `64` subsets: no element of it separates the
faces `4` and `6` (indices `3` and `5`). -/
theorem genF_not_separating : genF.card = 2 ^ 4 ∧ genF.card < 64 ∧ ∀ A ∈ genF, (3 ∈ A ↔ 5 ∈ A) := by
  decide +kernel

/-! ## exr-prob-countable-union -/

/-- "At least once" is the countable union of the events `A n`. -/
theorem exists_eq_iUnion {Ω : Type} (A : ℕ → Set Ω) : {ω | ∃ n, ω ∈ A n} = ⋃ n, A n := by
  ext ω; simp

/-- "From some point on" is `⋃ N, ⋂ n ≥ N, A n`. -/
theorem eventually_eq_iUnion_iInter {Ω : Type} (A : ℕ → Set Ω) :
    {ω | ∃ N, ∀ n ≥ N, ω ∈ A n} = ⋃ N, ⋂ n, ⋂ (_ : N ≤ n), A n := by
  ext ω; simp

/-- "Infinitely often" is `⋂ N, ⋃ n ≥ N, A n`, the complement of "the complementary event from some point on". -/
theorem infinitely_often_eq {Ω : Type} (A : ℕ → Set Ω) :
    {ω | ∀ N, ∃ n ≥ N, ω ∈ A n} = ⋂ N, ⋃ n, ⋃ (_ : N ≤ n), A n ∧
      {ω | ∀ N, ∃ n ≥ N, ω ∈ A n} = {ω | ∃ N, ∀ n ≥ N, ω ∈ (A n)ᶜ}ᶜ := by
  constructor
  · ext ω; simp
  · ext ω; simp

/-- The three events are measurable in the σ-algebra generated by the `A n`. -/
theorem limit_events_measurable {Ω : Type} (A : ℕ → Set Ω) :
    @MeasurableSet _ (MeasurableSpace.generateFrom (Set.range A)) (⋃ n, A n) ∧
      @MeasurableSet _ (MeasurableSpace.generateFrom (Set.range A)) (⋃ N, ⋂ n, ⋂ (_ : N ≤ n), A n) ∧
      @MeasurableSet _ (MeasurableSpace.generateFrom (Set.range A)) (⋂ N, ⋃ n, ⋃ (_ : N ≤ n), A n) := by
  have h : ∀ n, @MeasurableSet _ (MeasurableSpace.generateFrom (Set.range A)) (A n) := fun n =>
    MeasurableSpace.measurableSet_generateFrom ⟨n, rfl⟩
  exact ⟨MeasurableSet.iUnion h, MeasurableSet.iUnion fun N => MeasurableSet.iInter fun n =>
    MeasurableSet.iInter fun _ => h n,
    MeasurableSet.iInter fun N => MeasurableSet.iUnion fun n => MeasurableSet.iUnion fun _ => h n⟩

/-! ## def-probability-measure -/

/-- The probability of the whole space is one and that of the empty event zero. -/
theorem P_univ_empty {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) :
    P p univ = 1 ∧ P p ∅ = 0 := ⟨hp.sum_one, by simp [P]⟩

/-- Probabilities are non-negative. -/
theorem P_nonneg {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (A : Finset Ω) : 0 ≤ P p A :=
  Finset.sum_nonneg fun ω _ => hp.nonneg ω

/-- Finite additivity: the probability of a disjoint union is the sum. -/
theorem P_union_of_disjoint {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) {A B : Finset Ω} (h : Disjoint A B) :
    P p (A ∪ B) = P p A + P p B :=
  Finset.sum_union h

/-- Countable additivity for a finite space is finite additivity over any finite family of disjoint events. -/
theorem P_biUnion {Ω ι : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (s : Finset ι) (A : ι → Finset Ω)
    (h : (s : Set ι).PairwiseDisjoint A) : P p (s.biUnion A) = ∑ i ∈ s, P p (A i) :=
  Finset.sum_biUnion h

/-- The complement rule `P(Aᶜ) = 1 - P(A)`. -/
theorem P_compl {Ω : Type} [Fintype Ω] [DecidableEq Ω] {p : Ω → ℚ} (hp : IsDist p) (A : Finset Ω) :
    P p Aᶜ = 1 - P p A := by
  have := Finset.sum_add_sum_compl A p
  simp only [P]; rw [← hp.sum_one]; linarith

/-- Monotonicity: `A ⊆ B` implies `P(A) ≤ P(B)`. -/
theorem P_mono {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) {A B : Finset Ω} (h : A ⊆ B) :
    P p A ≤ P p B :=
  Finset.sum_le_sum_of_subset_of_nonneg h fun ω _ _ => hp.nonneg ω

/-- Inclusion–exclusion for two events. -/
theorem P_union {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (A B : Finset Ω) :
    P p (A ∪ B) = P p A + P p B - P p (A ∩ B) := by
  have := Finset.sum_union_inter (s₁ := A) (s₂ := B) (f := p)
  simp only [P]; linarith

/-- Every probability is at most one. -/
theorem P_le_one {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (A : Finset Ω) : P p A ≤ 1 :=
  (P_mono hp (subset_univ A)).trans_eq hp.sum_one

/-- The uniform weights form a probability distribution. -/
theorem unif_isDist (Ω : Type) [Fintype Ω] [Nonempty Ω] : IsDist (unif Ω) :=
  ⟨fun _ => by simp [unif], by
    simp only [unif, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    have : (Fintype.card Ω : ℚ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
    field_simp⟩

/-- Under the uniform measure, probability is counting: `P(A) = |A| / N`. -/
theorem P_unif {Ω : Type} [Fintype Ω] (A : Finset Ω) : P (unif Ω) A = A.card / Fintype.card Ω := by
  simp [P, unif, Finset.sum_const, div_eq_mul_inv]

/-- The measure-theoretic form of the consequences: `P(∅) = 0`, `P(Aᶜ) = 1 - P(A)`, monotonicity and
inclusion–exclusion for a probability measure. -/
theorem measure_consequences {Ω : Type} [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω)
    [MeasureTheory.IsProbabilityMeasure μ] (A B : Set Ω) (hA : MeasurableSet A) (hB : MeasurableSet B)
    (hAB : A ⊆ B) :
    μ ∅ = 0 ∧ μ Aᶜ = 1 - μ A ∧ μ A ≤ μ B ∧ μ (A ∪ B) + μ (A ∩ B) = μ A + μ B :=
  ⟨MeasureTheory.measure_empty, MeasureTheory.prob_compl_eq_one_sub hA, MeasureTheory.measure_mono hAB,
    MeasureTheory.measure_union_add_inter A hB⟩

/-! ## exm-prob-fair-die -/

/-- Each face of a fair die has probability `1/6`, so `P(A) = |A| / 6`. -/
theorem die_face_prob (i : Fin 6) (A : Finset (Fin 6)) :
    P (unif (Fin 6)) {i} = 1 / 6 ∧ P (unif (Fin 6)) A = A.card / 6 := by
  constructor
  · simp [P, unif]
  · rw [P_unif]; simp

/-- `P(even) = 1/2`, `P(at least 5) = 1/3`, and by inclusion–exclusion `P(even or at least 5) = 1/2 + 1/3 - 1/6
= 2/3`, the probability of `{2, 4, 5, 6}`. -/
theorem die_even_ge5 : P (unif (Fin 6)) evenF = 1 / 2 ∧ P (unif (Fin 6)) ge5 = 1 / 3 ∧
    P (unif (Fin 6)) (evenF ∩ ge5) = 1 / 6 ∧ P (unif (Fin 6)) (evenF ∪ ge5) = 2 / 3 ∧
    (1 / 2 : ℚ) + 1 / 3 - 1 / 6 = 2 / 3 ∧ evenF ∪ ge5 = {1, 3, 4, 5} ∧ P (unif (Fin 6)) univ = 6 / 6 := by
  decide +kernel

/-! ## exm-prob-two-dice-sums -/

/-- The table of the sum of two dice: `P(sum = s) = (6 - |s - 7|) / 36` for `s = 2, …, 12`. -/
theorem sum_table (s : ℕ) (h2 : 2 ≤ s) (h12 : s ≤ 12) :
    P (unif Dice2) (sumEq s) = (6 - |(s : ℚ) - 7|) / 36 := by
  interval_cases s <;> decide +kernel

/-- No pair of dice has a sum outside `2, …, 12`. -/
theorem sumEq_empty (s : ℕ) (h : s < 2 ∨ 12 < s) : sumEq s = ∅ := by
  apply Finset.filter_eq_empty_iff.2
  intro w _
  simp only [sum2, face]
  have := w.1.isLt; have := w.2.isLt
  omega

/-- Sums outside `2, …, 12` have probability zero. -/
theorem sum_table_zero (s : ℕ) (h : s < 2 ∨ 12 < s) : P (unif Dice2) (sumEq s) = 0 := by
  rw [P_unif, sumEq_empty s h]; simp

/-- The sum `7` is the most likely, with probability `1/6`. -/
theorem seven_most_likely (s : ℕ) : P (unif Dice2) (sumEq s) ≤ 1 / 6 ∧ P (unif Dice2) C7 = 1 / 6 := by
  refine ⟨?_, by decide +kernel⟩
  by_cases h : 2 ≤ s ∧ s ≤ 12
  · obtain ⟨h2, h12⟩ := h
    interval_cases s <;> decide +kernel
  · rw [sum_table_zero s (by omega)]; norm_num

/-- `P(A) = 1/6` (the sums `10, 11, 12` have `3 + 2 + 1` outcomes), `P(B) = 1/2`, and
`P(A ∪ B) = P(B) = 1/2 = 1/6 + 1/2 - 1/6` since `A ⊂ B`. -/
theorem dice_A_B : P (unif Dice2) A10 = 1 / 6 ∧ (sumEq 10).card + (sumEq 11).card + (sumEq 12).card = 6 ∧
    P (unif Dice2) B4 = 1 / 2 ∧ P (unif Dice2) (A10 ∪ B4) = 1 / 2 ∧ P (unif Dice2) (A10 ∩ B4) = 1 / 6 := by
  decide +kernel

/-! ## exm-prob-uniform-interval -/

/-- The uniform probability on `[0, 1]`: `P([a, b]) = b - a`, the whole interval has probability one, and every
point has probability zero. -/
theorem uniform_interval (a b : ℝ) (h0 : 0 ≤ a) (hab : a ≤ b) (hb : b ≤ 1) (x : ℝ) :
    (MeasureTheory.volume.restrict (Set.Icc (0 : ℝ) 1)) (Set.Icc a b) = ENNReal.ofReal (b - a) ∧
      (MeasureTheory.volume.restrict (Set.Icc (0 : ℝ) 1)) (Set.Icc 0 1) = 1 ∧
      (MeasureTheory.volume.restrict (Set.Icc (0 : ℝ) 1)) {x} = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · rw [MeasureTheory.Measure.restrict_apply measurableSet_Icc,
      Set.inter_eq_left.2 (Set.Icc_subset_Icc h0 hb), Real.volume_Icc]
  · rw [MeasureTheory.Measure.restrict_apply measurableSet_Icc, Set.inter_self, Real.volume_Icc]; simp
  · rw [MeasureTheory.Measure.restrict_apply (measurableSet_singleton x)]
    exact MeasureTheory.measure_mono_null Set.inter_subset_left Real.volume_singleton

/-- Lengths of disjoint intervals add. -/
theorem lengths_add (a b c : ℝ) :
    MeasureTheory.volume (Set.Ioc a b ∪ Set.Ioc b c) = MeasureTheory.volume (Set.Ioc a b) +
      MeasureTheory.volume (Set.Ioc b c) ∧ MeasureTheory.volume (Set.Ioc a c) = ENNReal.ofReal (c - a) := by
  refine ⟨MeasureTheory.measure_union (Set.Ioc_disjoint_Ioc_of_le le_rfl) measurableSet_Ioc, Real.volume_Ioc⟩

/-- A non-negative density of total integral one defines a probability measure `P(A) = ∫_A ρ`. -/
theorem density_isProbabilityMeasure (ρ : ℝ → ℝ) (hρ : 0 ≤ ρ) (hm : Measurable ρ)
    (h1 : ∫ x in Set.Icc (0 : ℝ) 1, ρ x = 1) :
    MeasureTheory.IsProbabilityMeasure
      ((MeasureTheory.volume.restrict (Set.Icc (0 : ℝ) 1)).withDensity fun x => ENNReal.ofReal (ρ x)) := by
  constructor
  rw [MeasureTheory.withDensity_apply _ MeasurableSet.univ, MeasureTheory.Measure.restrict_univ]
  have hint : MeasureTheory.IntegrableOn ρ (Set.Icc 0 1) := by
    by_contra hc
    rw [MeasureTheory.integral_undef hc] at h1
    exact zero_ne_one h1
  rw [← MeasureTheory.ofReal_integral_eq_lintegral_ofReal hint (MeasureTheory.ae_of_all _ fun x => hρ x), h1]
  simp

/-! ## exr-prob-birthday -/

/-- The probability that `n` people with uniformly distributed birthdays do not all have different birthdays. -/
def coincidence (n : ℕ) : ℚ := 1 - ∏ k ∈ range n, (1 - (k : ℚ) / 365)

/-- The complement "all different" has `365 · 364 ⋯ (365 - n + 1)` of the `365 ^ n` outcomes, which is the product
`∏ (1 - k / 365)`. -/
theorem coincidence_eq (n : ℕ) (hn : n ≤ 365) :
    coincidence n = 1 - (Nat.descFactorial 365 n : ℚ) / 365 ^ n := by
  unfold coincidence
  congr 1
  induction n with
  | zero => simp
  | succ m ih =>
    rw [Finset.prod_range_succ, ih (by omega), Nat.descFactorial_succ, pow_succ]
    push_cast [Nat.cast_sub (by omega : m ≤ 365)]
    field_simp

/-- The coincidence probability first exceeds `1/2` at `n = 23`: it is below `1/2` for `n ≤ 22`. -/
theorem birthday_23 : (1 / 2 : ℚ) < coincidence 23 ∧ ∀ n ≤ 22, coincidence n ≤ 1 / 2 := by
  constructor
  · decide +kernel
  · intro n hn; interval_cases n <;> decide +kernel

/-- The values of the table: `0.0271` at `n = 5`, `0.4114` at `n = 20`, `0.5073` at `n = 23`, `0.7063` at `n = 30`,
`0.9704` at `n = 50`. -/
theorem birthday_values : |coincidence 5 - 0.0271| < 0.0001 ∧ |coincidence 20 - 0.4114| < 0.0001 ∧
    |coincidence 23 - 0.5073| < 0.0001 ∧ |coincidence 30 - 0.7063| < 0.0001 ∧
    |coincidence 50 - 0.9704| < 0.0001 := by
  decide +kernel

/-! ## exr-prob-axiom-consequences -/

/-- `Ω = Ω ∪ ∅` disjointly gives `P(∅) = 0`; `Ω = A ∪ Aᶜ` gives `P(Aᶜ) = 1 - P(A)`; `B = A ∪ (B \ A)` gives
monotonicity; and `A ∪ B = A ∪ (B \ A)`, `B = (A ∩ B) ∪ (B \ A)` give inclusion–exclusion. -/
theorem axiom_consequences {Ω : Type} [Fintype Ω] [DecidableEq Ω] {p : Ω → ℚ} (hp : IsDist p)
    (A B : Finset Ω) :
    P p ∅ = 0 ∧ P p Aᶜ = 1 - P p A ∧ (A ⊆ B → P p A ≤ P p B) ∧
      P p (A ∪ B) = P p A + P p B - P p (A ∩ B) ∧ P p B = P p (A ∩ B) + P p (B \ A) :=
  ⟨(P_univ_empty hp).2, P_compl hp A, P_mono hp, P_union p A B, by
    rw [Finset.inter_comm, ← P_union_of_disjoint p
      (Finset.disjoint_of_subset_left Finset.inter_subset_right Finset.disjoint_sdiff), Finset.union_comm,
      Finset.sdiff_union_inter]⟩

/-! ## exr-prob-geometric-waiting -/

/-- The strings of `k` tosses whose first head is at toss `k`: tails before, a head at the end. -/
def firstHeadAt (k : ℕ) : Finset (Fin (k + 1) → Bool) :=
  univ.filter (fun w => (∀ i, i.val < k → w i = false) ∧ w (Fin.last k) = true)

/-- "First head at toss `k + 1`" is one of the `2 ^ (k + 1)` equally likely strings, so it has probability
`2 ^ (-(k + 1))`. -/
theorem firstHeadAt_prob (k : ℕ) :
    (firstHeadAt k).card = 1 ∧ P (unif (Fin (k + 1) → Bool)) (firstHeadAt k) = 1 / 2 ^ (k + 1) := by
  have hcard : (firstHeadAt k).card = 1 := by
    rw [Finset.card_eq_one]
    refine ⟨fun i => decide (i = Fin.last k), ?_⟩
    ext w
    simp only [firstHeadAt, mem_filter, mem_univ, true_and, mem_singleton]
    constructor
    · rintro ⟨h1, h2⟩
      funext i
      by_cases hi : i = Fin.last k
      · subst hi; simp [h2]
      · have : i.val < k := by
          have := i.isLt
          have hne : i.val ≠ k := fun h => hi (Fin.ext (by simp [h]))
          omega
        simp [h1 i this, hi]
    · rintro rfl
      refine ⟨fun i hi => ?_, by simp⟩
      have : i ≠ Fin.last k := fun h => by simp [h] at hi
      simp [this]
  refine ⟨hcard, ?_⟩
  rw [P_unif, hcard]
  simp

/-- The partial sums `∑_{k=1}^m 2^{-k} = 1 - 2^{-m}`. -/
theorem geometric_partial (m : ℕ) : ∑ k ∈ range m, (1 / 2 : ℚ) ^ (k + 1) = 1 - (1 / 2) ^ m := by
  veriphy_certified
    (induction m with
      | zero => rw [Finset.sum_range_zero]; (try push_cast); ring1
      | succ m ih => rw [Finset.sum_range_succ, ih]; (try push_cast); ring1)
  -- certificate: pr-geometric-partial.sage

/-- The probabilities `2^{-k}`, `k ≥ 1`, sum to one. -/
theorem geometric_sum_one : ∑' k : ℕ, (1 / 2 : ℝ) ^ (k + 1) = 1 := by
  have := tsum_geometric_two' (1 : ℝ)
  convert this using 2 with k
  funext k; simp [one_div, pow_succ]; ring

/-- The probability of no head in the first `m` tosses is `2^{-m}`, which tends to zero: a head eventually appears
with probability one. -/
theorem no_head_tendsto_zero : (1 : ℚ) - ∑ k ∈ range 20, (1 / 2 : ℚ) ^ (k + 1) = 1 / 2 ^ 20 ∧
    Filter.Tendsto (fun m : ℕ => (1 / 2 : ℝ) ^ m) Filter.atTop (nhds 0) :=
  ⟨by rw [geometric_partial]; norm_num, tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)⟩

end KaluzaKlein.Probability.SampleSpaces
