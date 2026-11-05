import KaluzaKlein.PartI.GroupTheory.Groups

/-!
# Group theory, section "Subgroups"

The claims of the section "Subgroups" of `tutorial/part-i-10-group-theory.qmd`: the subgroups of `ℤ₆`, `S₃`, `D₄`
and `S₄`, the one-step subgroup test, generated and cyclic subgroups, normal subgroups and centres, the facts about
`S₃` and `S₄` (orders, cycle types, the alternating group) and the presentation of `S₃`.
-/

namespace KaluzaKlein.GroupTheory.Subgroups

open Equiv Matrix KaluzaKlein.GroupTheory.Basic KaluzaKlein.GroupTheory.Groups Real

/-- The multiples of `d` in `ℤₙ`, as a Finset. -/
def multiples (n : ℕ) [NeZero n] (d : ℕ) : Finset (ZMod n) :=
  Finset.univ.filter fun k => ∃ i : Fin n, k = (d : ZMod n) * (i : ℕ)

/-- The four-cycle `(1 2 3 4)` of `S₄`: `0 ↦ 1 ↦ 2 ↦ 3 ↦ 0`. -/
def c4 : Perm (Fin 4) := finRotate 4

/-- The Klein four-group inside `S₄`: the identity and the three double transpositions. -/
def klein4 : Finset (Perm (Fin 4)) := {1, swap 0 1 * swap 2 3, swap 0 2 * swap 1 3, swap 0 3 * swap 1 2}

/-- The even permutations of `S₄`, as a Finset. -/
def a4 : Finset (Perm (Fin 4)) := Finset.univ.filter fun g => Perm.sign g = 1

/-- A Finset is cyclic: some element's powers exhaust it. -/
def IsCyclicFinset {G : Type} [Group G] [Fintype G] [DecidableEq G] (H : Finset G) : Prop :=
  ∃ g ∈ H, ∀ h ∈ H, ∃ k < Fintype.card G, g ^ k = h

instance {G : Type} [Group G] [Fintype G] [DecidableEq G] (H : Finset G) : Decidable (IsCyclicFinset H) := by
  unfold IsCyclicFinset; infer_instance

/-! ## exm-subgroup-z6 -/

/-- The multiples of `d` in `ℤₙ` form a subgroup (`{0, 2, 4}` and `{0, 3}` in `ℤ₆`). -/
theorem multiples_subgroup (n : ℕ) [NeZero n] (d : ℕ) : IsAddSubgroupFinset (multiples n d) := by
  have hn : 0 < n := Nat.pos_of_neZero n
  refine ⟨?_, ?_, ?_⟩
  · simp only [multiples, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨⟨0, hn⟩, by simp⟩
  · intro a ha b hb
    simp only [multiples, Finset.mem_filter, Finset.mem_univ, true_and] at ha hb ⊢
    obtain ⟨i, rfl⟩ := ha
    obtain ⟨j, rfl⟩ := hb
    refine ⟨⟨(i + j) % n, Nat.mod_lt _ hn⟩, ?_⟩
    simp only
    rw [ZMod.natCast_mod]; push_cast; ring
  · intro a ha
    simp only [multiples, Finset.mem_filter, Finset.mem_univ, true_and] at ha ⊢
    obtain ⟨i, rfl⟩ := ha
    refine ⟨⟨(n - i) % n, Nat.mod_lt _ hn⟩, ?_⟩
    simp only
    rw [ZMod.natCast_mod, Nat.cast_sub i.isLt.le, ZMod.natCast_self]; ring

/-- The multiples of `d` depend on `d` modulo `n` only. -/
theorem multiples_mod (n : ℕ) [NeZero n] (d : ℕ) : multiples n d = multiples n (d % n) := by
  unfold multiples; congr 1; ext k; rw [ZMod.natCast_mod]

/-- The subgroups of multiples in `ℤ₆`: `{0}`, `{0, 3}`, `{0, 2, 4}` and `ℤ₆`, according to `d` modulo `6`. -/
theorem z6_multiples (d : ℕ) : multiples 6 d =
    if d % 6 = 0 then {0} else if d % 6 = 3 then {0, 3} else if d % 6 = 2 ∨ d % 6 = 4 then {0, 2, 4} else Finset.univ := by
  rw [multiples_mod]
  have : d % 6 < 6 := Nat.mod_lt _ (by norm_num)
  generalize d % 6 = r at *
  interval_cases r <;> decide

/-- `{0, 1}` is not a subgroup of `ℤₙ` for `n > 2`: `1 + 1 = 2` is missing. -/
theorem not_subgroup_zero_one (n : ℕ) (hn : 2 < n) : ¬ IsAddSubgroupFinset ({0, 1} : Finset (ZMod n)) := by
  intro h
  have hmem := h.2.1 1 (by simp) 1 (by simp)
  have h2 : (1 : ZMod n) + 1 = ((2 : ℕ) : ZMod n) := by push_cast; ring
  rw [h2, Finset.mem_insert, Finset.mem_singleton, ZMod.natCast_eq_zero_iff] at hmem
  rcases hmem with h0 | h1
  · exact absurd (Nat.le_of_dvd two_pos h0) (by omega)
  · have h3 : ((1 : ℕ) : ZMod n) = 0 := by
      have := h1; push_cast at this ⊢; linear_combination this
    rw [ZMod.natCast_eq_zero_iff] at h3
    exact absurd (Nat.le_of_dvd one_pos h3) (by omega)

/-- The order of a subgroup Finset divides the order of the group (Lagrange, in Finset form). -/
theorem card_dvd_of_isSubgroupFinset (G : Type) [Group G] [Fintype G] [DecidableEq G] (H : Finset G)
    (hH : IsSubgroupFinset H) : H.card ∣ Fintype.card G := by
  have := Subgroup.card_subgroup_dvd_card (toSubgroup H hH)
  rw [Nat.card_eq_fintype_card, Nat.card_eq_fintype_card] at this
  simpa [toSubgroup, Fintype.card_coe] using this

/-- The order of a subgroup of `ℤₙ` divides `n`. -/
theorem card_addSubgroup_zmod_dvd (n : ℕ) [NeZero n] (H : AddSubgroup (ZMod n)) : Nat.card H ∣ n := by
  have := AddSubgroup.card_addSubgroup_dvd_card H
  rwa [Nat.card_eq_fintype_card (α := ZMod n), ZMod.card] at this

/-- A cyclic group of order `n` has exactly one subgroup per divisor of `n`: `ℤₙ` has as many subgroups as `n` has
divisors (stated; checked by the oracle on `ℤ₆`, `ℤ₁₂`, `ℤ₈`, `ℤ₁`). -/
def CyclicSubgroupCount : Prop := ∀ n : ℕ, 0 < n → Nat.card (AddSubgroup (ZMod n)) = (Nat.divisors n).card

/-! ## exm-subgroup-s3 -/

/-- The subgroup Finsets of `S₃` are exactly `{e}`, `S₃`, `A₃ = {e, c, c²}` and the three `{e, (i j)}`. -/
theorem s3_subgroups (H : Finset (Perm (Fin 3))) :
    IsSubgroupFinset H ↔ H ∈ ({{1}, Finset.univ, {1, c, c ^ 2}, {1, s}, {1, t13}, {1, t23}} : Finset (Finset (Perm (Fin 3)))) := by
  revert H; decide

/-- The alternating group `Aₙ` has `n!/2` elements: `2 |Aₙ| = n!` for `n ≥ 2`. -/
theorem card_alternating (n : ℕ) (hn : 2 ≤ n) : 2 * Fintype.card (alternatingGroup (Fin n)) = n ! := by
  haveI : Nontrivial (Fin n) := Fin.nontrivial_iff_two_le.mpr hn
  rw [two_mul_card_alternatingGroup, Fintype.card_perm, Fintype.card_fin]

/-- `A₃ = {e, c, c²}`: the even permutations of three letters. -/
theorem a3_elements (g : Perm (Fin 3)) : g ∈ alternatingGroup (Fin 3) ↔ g = 1 ∨ g = c ∨ g = c ^ 2 := by
  rw [Perm.mem_alternatingGroup]; revert g; decide

/-! ## exm-subgroup-so2 -/

/-- Rotations about the `Z`-axis compose by adding angles: they form a subgroup `SO(2)` of `SO(3)`. -/
theorem Rz_mul (α β : ℝ) : Rz α * Rz β = Rz (α + β) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Rz, Matrix.mul_apply, Fin.sum_univ_succ, Real.cos_add, Real.sin_add] <;> ring

/-- A rotation about `Z` is orthogonal with determinant `1`: `Rz θ ∈ SO(3)`. -/
theorem Rz_orthogonal (θ : ℝ) : Rz θ * (Rz θ)ᵀ = 1 ∧ (Rz θ).det = 1 := by
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [Rz, Matrix.mul_apply, Fin.sum_univ_succ] <;>
      linear_combination Real.cos_sq_add_sin_sq θ
  · simp [Rz, Matrix.det_fin_three]
    linear_combination Real.cos_sq_add_sin_sq θ

/-- The phases `e^{tT₃}` compose by adding `t`: a subgroup `U(1)` of `SU(2)`. -/
theorem phase_mul (a b : ℝ) : phase a * phase b = phase (a + b) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [phase, Matrix.mul_apply, Fin.sum_univ_succ, ← Complex.exp_add] <;>
    congr 1 <;> push_cast <;> ring

/-- `e^{tT₃}` is unitary with determinant `1`: `phase t ∈ SU(2)`. -/
theorem phase_unitary (t : ℝ) : phase t * (phase t)ᴴ = 1 ∧ (phase t).det = 1 := by
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [phase, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.conjTranspose_apply,
      ← Complex.exp_conj, ← Complex.exp_add] <;> simp [Complex.ext_iff]
  · simp [phase, Matrix.det_fin_two, ← Complex.exp_add]

/-! ## exr-subgroups-s3 -/

/-- Among the subgroups of `S₃`, exactly `{e}`, `A₃` and `S₃` are normal. -/
theorem s3_normal_subgroups (H : Finset (Perm (Fin 3))) (hH : IsSubgroupFinset H) :
    IsNormalFinset H ↔ H = {1} ∨ H = Finset.univ ∨ H = {1, c, c ^ 2} := by
  revert H; decide

/-! ## exr-grp-subgroup-test -/

/-- The one-step subgroup test: a non-empty Finset `H` is a subgroup exactly when `g h⁻¹ ∈ H` for all `g, h ∈ H`. -/
theorem subgroup_test (G : Type) [Group G] [DecidableEq G] (H : Finset G) (hne : H.Nonempty) :
    (∀ g ∈ H, ∀ h ∈ H, g * h⁻¹ ∈ H) ↔ IsSubgroupFinset H := by
  constructor
  · intro h
    obtain ⟨x, hx⟩ := hne
    have h1 : (1 : G) ∈ H := by simpa using h x hx x hx
    have hinv : ∀ a ∈ H, a⁻¹ ∈ H := fun a ha => by simpa using h 1 h1 a ha
    exact ⟨h1, fun a ha b hb => by simpa using h a ha b⁻¹ (hinv b hb), hinv⟩
  · intro ⟨_, hmul, hinv⟩ g hg h hh
    exact hmul g hg h⁻¹ (hinv h hh)

/-- The multiples of `n` pass the test: `nℤ` is a subgroup of `(ℤ, +)`. -/
theorem zmultiples_test (n x y : ℤ) (hx : n ∣ x) (hy : n ∣ y) : n ∣ x - y := dvd_sub hx hy

/-- The inverse of a rotation is the rotation by the opposite angle. -/
theorem R_inv (θ : ℝ) : (R θ)⁻¹ = R (-θ) := by
  apply Matrix.inv_eq_right_inv
  rw [R_mul, add_neg_cancel]
  exact (R_eq_one_iff 0).mpr ⟨0, by simp⟩

/-- Rotations pass the test: `R α (R β)⁻¹ = R (α - β)`, so they form a subgroup of `GL(2, ℝ)`. -/
theorem R_test (α β : ℝ) : R α * (R β)⁻¹ = R (α - β) := by
  rw [R_inv, R_mul, sub_eq_add_neg]

/-! ## exr-grp-subgroups-d4 -/

/-- The ten subgroup Finsets of the dihedral group `D₄`. -/
theorem d4_subgroups (H : Finset (DihedralGroup 4)) :
    IsSubgroupFinset H ↔ H ∈ ({{1}, {1, DihedralGroup.r 2}, {1, DihedralGroup.sr 0}, {1, DihedralGroup.sr 1},
      {1, DihedralGroup.sr 2}, {1, DihedralGroup.sr 3}, {1, DihedralGroup.r 1, DihedralGroup.r 2, DihedralGroup.r 3},
      {1, DihedralGroup.r 2, DihedralGroup.sr 0, DihedralGroup.sr 2},
      {1, DihedralGroup.r 2, DihedralGroup.sr 1, DihedralGroup.sr 3}, Finset.univ} :
      Finset (Finset (DihedralGroup 4))) := by
  revert H; decide

/-- `D₄` has one subgroup of order `1`, five of order `2`, three of order `4` and one of order `8`. -/
theorem d4_subgroup_count (k : ℕ) :
    (Finset.univ.filter fun H : Finset (DihedralGroup 4) => IsSubgroupFinset H ∧ H.card = k).card =
      if k = 1 then 1 else if k = 2 then 5 else if k = 4 then 3 else if k = 8 then 1 else 0 := by
  by_cases hk : k ≤ 8
  · interval_cases k <;> decide
  · rw [Finset.card_eq_zero.mpr]
    · simp; omega
    · apply Finset.filter_eq_empty_iff.mpr
      intro H _ ⟨_, hc⟩
      have := Finset.card_le_univ H
      rw [hc, Fintype.card_fin.symm ▸ DihedralGroup.card] at this
      omega

/-- Of the three subgroups of order `4` of `D₄`, only the rotations form a cyclic group. -/
theorem d4_order_four_cyclic (H : Finset (DihedralGroup 4)) (hH : IsSubgroupFinset H) (h4 : H.card = 4) :
    IsCyclicFinset H ↔ H = {1, DihedralGroup.r 1, DihedralGroup.r 2, DihedralGroup.r 3} := by
  revert H; decide

/-! ## exm-generated-z6 -/

/-- The subgroup generated by `k` in `ℤ₆` is the set of multiples of `k`: `⟨2⟩ = {0, 2, 4}`, `⟨3⟩ = {0, 3}`,
`⟨1⟩ = ⟨5⟩ = ℤ₆`. -/
theorem z6_zmultiples (k x : ZMod 6) : x ∈ AddSubgroup.zmultiples k ↔ x ∈ multiples 6 k.val := by
  revert k x; decide

/-- A cyclic subgroup `⟨g⟩` has as many elements as the order of `g`. -/
theorem card_zpowers_eq (G : Type) [Group G] (g : G) : Nat.card (Subgroup.zpowers g) = orderOf g :=
  Nat.card_zpowers g

/-! ## exm-generated-s3 -/

/-- `⟨(1 2 3)⟩ = A₃`. -/
theorem zpowers_c (g : Perm (Fin 3)) : g ∈ Subgroup.zpowers c ↔ g ∈ alternatingGroup (Fin 3) := by
  rw [Perm.mem_alternatingGroup]; revert g; decide

/-- `⟨(1 2)⟩ = {e, (1 2)}`. -/
theorem zpowers_s (g : Perm (Fin 3)) : g ∈ Subgroup.zpowers s ↔ g = 1 ∨ g = s := by
  revert g; decide

/-- Every element of `S₃` is a word of length at most `3` in `(1 2)` and `(1 2 3)`: the two generate `S₃`. -/
theorem s3_words (g : Perm (Fin 3)) : g ∈ words {s, c} 3 := by
  revert g; decide

/-- `⟨(1 2), (1 2 3)⟩ = S₃`. -/
theorem s3_closure : Subgroup.closure ({s, c} : Set (Perm (Fin 3))) = ⊤ := by
  rw [eq_top_iff]
  intro g _
  exact words_subset_closure {s, c} 3 g (s3_words g)

/-! ## exm-generated-rotations -/

/-- The real generator `L₃` of rotations about the `Z`-axis. -/
def L3real : Matrix (Fin 3) (Fin 3) ℝ := !![0, -1, 0; 1, 0, 0; 0, 0, 0]

/-- The exponential of `t L₃` is the rotation by `t` about the `Z`-axis: `SO(2)` is generated by `L₃` through the
exponential map (stated; the matrix exponential is computed by the oracle). -/
def ExpL3 : Prop := ∀ t : ℝ, NormedSpace.exp ℝ (t • L3real) = Rz t

/-- A finite set of rotations by rational multiples of `2π` generates a finite subgroup, never all of `SO(2)`
(stated; the oracle computes the generated groups). -/
def RotationsNotFinitelyGenerated : Prop :=
  ∀ S : Finset ℚ, ∃ θ : ℝ, R θ ∉ Subgroup.closure ((fun q : ℚ => R (2 * π * q)) '' ↑S : Set (Matrix (Fin 2) (Fin 2) ℝ))

/-! ## exr-grp-generators-zn -/

/-- `⟨k⟩` in `ℤ₁₂` consists of the multiples of `gcd (k, 12)` and has `12 / gcd (k, 12)` elements. -/
theorem z12_zmultiples (k : ZMod 12) :
    (∀ x, x ∈ AddSubgroup.zmultiples k ↔ x ∈ multiples 12 (Nat.gcd 12 k.val)) ∧
      (multiples 12 (Nat.gcd 12 k.val)).card = 12 / Nat.gcd 12 k.val := by
  revert k; decide

/-- For a divisor `d` of `12`, the subgroup of multiples of `d` has `12 / d` elements: one cyclic subgroup per
divisor, of orders `12, 6, 4, 3, 2, 1`. -/
theorem z12_multiples_card (d : ℕ) (hd : d ∣ 12) : (multiples 12 d).card = 12 / d := by
  have : d ≤ 12 := Nat.le_of_dvd (by norm_num) hd
  interval_cases d <;> first | decide | (exfalso; norm_num at hd)

/-! ## exr-grp-generated-s4 -/

/-- A transposition and the `n`-cycle `(1 2 … n)` generate `Sₙ` (stated; checked for `n ≤ 4` by listing the words
of length at most six). -/
def TranspositionCycleGenerate : Prop :=
  ∀ n : ℕ, Subgroup.closure ({swap 0 1, finRotate (n + 2)} : Set (Perm (Fin (n + 2)))) = ⊤

/-- Two double transpositions on four distinct letters generate the Klein four-group: four elements, abelian, every
non-identity element of order `2`, so not cyclic. -/
theorem klein_generated (i j k l : Fin 4) (h : [i, j, k, l].Nodup) :
    words {swap i j * swap k l, swap i k * swap j l} 3 =
        {1, swap i j * swap k l, swap i k * swap j l, swap i j * swap k l * (swap i k * swap j l)} ∧
      swap i j * swap k l * (swap i k * swap j l) = swap i k * swap j l * (swap i j * swap k l) ∧
      (swap i j * swap k l) ^ 2 = 1 ∧ (swap i k * swap j l) ^ 2 = 1 ∧
      (swap i j * swap k l * (swap i k * swap j l)) ^ 2 = 1 ∧ swap i j * swap k l ≠ 1 ∧
      ¬ IsCyclicFinset ({1, swap i j * swap k l, swap i k * swap j l, swap i j * swap k l * (swap i k * swap j l)} :
        Finset (Perm (Fin 4))) := by
  revert i j k l; decide

/-- A four-cycle generates a cyclic group of order `4`. -/
theorem four_cycle_generates (g : Perm (Fin 4)) (hg : g.cycleType = {4}) :
    (words {g} 4).card = 4 ∧ g ∈ ofOrder (Perm (Fin 4)) 4 ∧ IsCyclicFinset (words {g} 4) := by
  revert g; decide

/-- A three-cycle and the double transposition `(1 2)(3 4)` generate the even permutations `A₄`, of order `12`. -/
theorem a4_generated (g : Perm (Fin 4)) (hg : g.cycleType = {3}) :
    words {g, swap 0 1 * swap 2 3} 6 = a4 ∧ a4.card = 12 := by
  revert g; decide

/-! ## exr-grp-cyclic-subgroups -/

/-- Every subgroup of a cyclic group is cyclic. -/
theorem subgroup_of_cyclic (G : Type) [Group G] [IsCyclic G] (H : Subgroup G) : IsCyclic H := inferInstance

/-- In a cyclic group `⟨g⟩` of order `n`, a subgroup `H` is generated by `g^m` for the least `m > 0` with `g^m ∈ H`,
and `m` divides `n` (stated; the oracle checks the subgroups of `ℤ₁₂` and `ℤ₆`). -/
def CyclicSubgroupGenerator : Prop :=
  ∀ (n : ℕ) [NeZero n] (H : AddSubgroup (ZMod n)), ∃ m : ℕ, 0 < m ∧ m ∣ n ∧
    H = AddSubgroup.zmultiples (m : ZMod n) ∧ ∀ m', 0 < m' → (m' : ZMod n) ∈ H → m ≤ m'

/-! ## def-normal-subgroup -/

/-- In an abelian group every subset is closed under conjugation: every subgroup is normal. -/
theorem normal_of_comm (G : Type) [CommGroup G] [Fintype G] [DecidableEq G] (H : Finset G) : IsNormalFinset H := by
  intro g h hh
  rwa [mul_comm g h, mul_assoc, mul_inv_cancel, mul_one]

/-! ## exm-normal-a3 -/

/-- Conjugation preserves the sign of a permutation. -/
theorem sign_conj (n : ℕ) (g k : Perm (Fin n)) : Perm.sign (k * g * k⁻¹) = Perm.sign g := by
  rw [map_mul, map_mul, mul_comm (Perm.sign k), mul_assoc, ← map_mul, mul_inv_cancel, map_one, mul_one]

/-- The conjugate of an even permutation is even: `A₃` is normal in `S₃`. -/
theorem a3_normal (g : Perm (Fin 3)) (hg : Perm.sign g = 1) (k : Perm (Fin 3)) : Perm.sign (k * g * k⁻¹) = 1 := by
  revert g k; decide

/-! ## exm-normal-not -/

/-- Conjugating `(i j)` by `(i k)` gives `(k j)`, which is not in `{e, (i j)}`: `{e, (1 2)}` is not normal. -/
theorem conj_swap (n : ℕ) (i j k : Fin n) (hij : i ≠ j) (hik : i ≠ k) (hjk : j ≠ k) :
    swap i k * swap i j * (swap i k)⁻¹ = swap k j ∧ swap k j ∉ ({1, swap i j} : Finset (Perm (Fin n))) := by
  refine ⟨?_, ?_⟩
  · rw [← swap_apply_apply, swap_apply_left, swap_apply_of_ne_of_ne hij.symm hjk]
  · simp only [Finset.mem_insert, Finset.mem_singleton, not_or]
    refine ⟨?_, ?_⟩
    · intro h
      have := congrArg (fun σ => σ k) h
      simp [swap_apply_left] at this
      exact hjk this
    · intro h
      have := congrArg (fun σ => σ k) h
      simp [swap_apply_left, swap_apply_of_ne_of_ne hik.symm hjk.symm] at this
      exact hjk this

/-- Conjugating a rotation about `Z` by the quarter turn about `X` tilts its axis: the result is no rotation
about `Z` (its `(0, 2)` entry is `-sin θ ≠ 0`), so `SO(2)` is not normal in `SO(3)`. -/
theorem so2_not_normal (θ : ℝ) (hθ : Real.sin θ ≠ 0) (ψ : ℝ) : Rx (π / 2) * Rz θ * Rx (-(π / 2)) ≠ Rz ψ := by
  intro h
  have := congrFun (congrFun h 0) 2
  simp [Rx, Rz, Matrix.mul_apply, Fin.sum_univ_succ] at this
  exact hθ this

/-! ## exm-normal-centre -/

/-- The centre is closed under conjugation: it is a normal subgroup. -/
theorem center_conj (G : Type) [Group G] (z : G) (hz : z ∈ Subgroup.center G) (g : G) :
    g * z * g⁻¹ ∈ Subgroup.center G := by
  rw [Subgroup.mem_center_iff] at hz
  rw [hz g, mul_assoc, mul_inv_cancel, mul_one]
  exact Subgroup.mem_center_iff.mpr hz

/-- The diagonal generator `diag (i, -i)` of `SU(2)`. -/
def su2D : Matrix (Fin 2) (Fin 2) ℂ := !![Complex.I, 0; 0, -Complex.I]

/-- The real rotation `!![0, 1; -1, 0]` of `SU(2)`. -/
def su2J : Matrix (Fin 2) (Fin 2) ℂ := !![0, 1; -1, 0]

/-- A matrix of determinant `1` commuting with the two generators `diag (i, -i)` and `!![0, 1; -1, 0]` of `SU(2)` is
`±1`: the centre of `SU(2)` is `{±1}`. -/
theorem su2_center (A : Matrix (Fin 2) (Fin 2) ℂ) :
    (A * su2D = su2D * A ∧ A * su2J = su2J * A ∧ A.det = 1) ↔ A = 1 ∨ A = -1 := by
  constructor
  · rintro ⟨hD, hJ, hdet⟩
    have h01 := congrFun (congrFun hD 0) 1
    have h10 := congrFun (congrFun hD 1) 0
    have h00 := congrFun (congrFun hJ 0) 0
    simp [su2D, su2J, Matrix.mul_apply, Fin.sum_univ_succ] at h01 h10 h00
    have hA01 : A 0 1 = 0 := by
      have : A 0 1 * (Complex.I + Complex.I) = 0 := by linear_combination -h01
      rcases mul_eq_zero.mp this with h | h
      · exact h
      · exfalso; have := congrArg Complex.im h; simp at this
    have hA10 : A 1 0 = 0 := by
      have : A 1 0 * (Complex.I + Complex.I) = 0 := by linear_combination h10
      rcases mul_eq_zero.mp this with h | h
      · exact h
      · exfalso; have := congrArg Complex.im h; simp at this
    rw [hA10] at h00
    have hA11 : A 1 1 = A 0 0 := by linear_combination -h00
    rw [Matrix.det_fin_two, hA01, hA10, hA11] at hdet
    simp at hdet
    rcases mul_self_eq_one_iff.mp hdet with h | h
    · left; ext i j; fin_cases i <;> fin_cases j <;> simp [hA01, hA10, hA11, h]
    · right; ext i j; fin_cases i <;> fin_cases j <;> simp [hA01, hA10, hA11, h]
  · rintro (rfl | rfl) <;> simp [su2D, su2J, Matrix.det_fin_two]

/-- The centre of `S₃` is trivial. -/
theorem s3_center (z : Perm (Fin 3)) : z ∈ Subgroup.center (Perm (Fin 3)) ↔ z = 1 := by
  revert z; decide

/-! ## exr-grp-normal-s4 -/

/-- The normal subgroups of `S₄` have orders `1, 4, 12, 24` (stated; the oracle lists the normal subgroups). -/
def S4NormalSubgroupOrders : Prop :=
  ∀ H : Subgroup (Perm (Fin 4)), H.Normal → Nat.card H ∈ ({1, 4, 12, 24} : Finset ℕ)

/-- The Klein four-group is normal in `S₄`: closed under every conjugation. -/
theorem klein_normal (g : Perm (Fin 4)) (hg : g ∈ klein4) (k : Perm (Fin 4)) : k * g * k⁻¹ ∈ klein4 := by
  revert g k; decide

/-- Conjugation preserves the cycle type. -/
theorem cycleType_conj_eq (n : ℕ) (g k : Perm (Fin n)) : (k * g * k⁻¹).cycleType = g.cycleType :=
  Perm.cycleType_conj

/-- A conjugate of the four-cycle `(1 2 3 4)` lies in `⟨(1 2 3 4)⟩` only when it is `(1 2 3 4)` or its inverse; the
conjugate by `(1 2)` is the four-cycle `(1 3 4 2)`, outside `⟨(1 2 3 4)⟩`: that subgroup is not normal. -/
theorem four_cycle_conj (k : Perm (Fin 4)) :
    (k * c4 * k⁻¹ ∈ Subgroup.zpowers c4 ↔ k * c4 * k⁻¹ = c4 ∨ k * c4 * k⁻¹ = c4⁻¹) ∧
      swap 0 1 * c4 * (swap 0 1)⁻¹ ∉ Subgroup.zpowers c4 ∧ (swap 0 1 * c4 * (swap 0 1)⁻¹).cycleType = {4} := by
  revert k; decide

/-! ## exr-grp-normal-det -/

/-- Conjugation preserves the determinant: `SL(n)` is normal in `GL(n)` and `SO(n)` in `O(n)`. -/
theorem det_conj_eq (n : ℕ) (g h A : Matrix (Fin n) (Fin n) ℚ) (hgh : g * h = 1) : (g * A * h).det = A.det := by
  rw [Matrix.det_mul, Matrix.det_mul, mul_comm, ← mul_assoc, ← Matrix.det_mul, mul_comm g h,
    Matrix.mul_eq_one_comm.mp hgh, Matrix.det_one, one_mul]

/-- Conjugating a shear by an invertible upper triangular matrix gives a shear: the unipotent matrices form a normal
subgroup of the triangular group. -/
theorem shear_conj (a b d x : ℚ) (ha : a ≠ 0) (hd : d ≠ 0) :
    !![a, b; 0, d] * !![1, x; 0, 1] * !![a⁻¹, -b / (a * d); 0, d⁻¹] = !![1, a * x / d; 0, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> field_simp <;> ring

/-- Conjugating `diag (a, 1)` by the shear `!![1, b; 0, 1]` gives `!![a, (1 - a) b; 0, 1]`. -/
theorem diag_conj (a b : ℚ) : !![1, b; 0, 1] * !![a, 0; 0, 1] * !![1, -b; 0, 1] = !![a, (1 - a) * b; 0, 1] := by
  VERIPHY[gt-matrix]

/-- The conjugate `!![a, (1 - a) b; 0, 1]` is not diagonal for `a ≠ 1`, `b ≠ 0`: the diagonal matrices are not
normal. -/
theorem diag_conj_not_diagonal (a b : ℚ) (ha : a ≠ 1) (hb : b ≠ 0) : (1 - a) * b ≠ 0 :=
  mul_ne_zero (sub_ne_zero.mpr ha.symm) hb

/-! ## exr-grp-centre -/

/-- The centre of `S₄` is trivial. -/
theorem s4_center (z : Perm (Fin 4)) : z ∈ Subgroup.center (Perm (Fin 4)) ↔ z = 1 := by
  revert z; decide

/-- The centre of `D₄` is `{e, r²}`, the half turn. -/
theorem d4_center (z : DihedralGroup 4) : z ∈ Subgroup.center (DihedralGroup 4) ↔ z = 1 ∨ z = DihedralGroup.r 2 := by
  revert z; decide

/-- The centre of `Q₈` is `{±1}`. -/
theorem q8_center (z : QuaternionGroup 2) : z ∈ Subgroup.center (QuaternionGroup 2) ↔ z = 1 ∨ z = QuaternionGroup.a 2 := by
  revert z; decide

/-- `ℤ₆` is its own centre. -/
theorem z6_center (z : Multiplicative (ZMod 6)) : z ∈ Subgroup.center (Multiplicative (ZMod 6)) := by
  revert z; decide

/-- The symmetric group `Sₙ` has trivial centre for `n ≥ 3` (stated; checked for `n = 3, 4`). -/
def SymmetricCenterTrivial : Prop := ∀ n : ℕ, ∀ z : Perm (Fin (n + 3)), z ∈ Subgroup.center (Perm (Fin (n + 3))) ↔ z = 1

/-! ## exm-grp-s3-alternating -/

/-- The powers of the three-cycle repeat with period `3`: `c^k = c^(k mod 3)`, so `c² = c⁻¹` and `c³ = e`. -/
theorem c_pow_mod (k : ℕ) : c ^ k = c ^ (k % 3) := by
  conv_lhs => rw [← Nat.mod_add_div k 3, pow_add, pow_mul]
  rw [show c ^ 3 = 1 by decide, one_pow, mul_one]

/-- `k ↦ c^k` matches the addition table of `ℤ₃` with the multiplication table of `A₃`. -/
theorem a3_table (j k : ZMod 3) : c ^ (j + k).val = c ^ j.val * c ^ k.val := by
  revert j k; decide

/-- `A₃` is abelian although `S₃` is not. -/
theorem a3_comm (g h : Perm (Fin 3)) (hg : g ∈ alternatingGroup (Fin 3)) (hh : h ∈ alternatingGroup (Fin 3)) :
    g * h = h * g := by
  rw [Perm.mem_alternatingGroup] at hg hh; revert g h; decide

/-! ## exm-grp-sn-facts -/

/-- `Sₙ` is not abelian for `n ≥ 3`: `(1 2)` and `(1 3)` do not commute. -/
theorem perm_nonabelian (n : ℕ) (hn : 3 ≤ n) : ∃ g h : Perm (Fin n), g * h ≠ h * g :=
  ⟨swap ⟨0, by omega⟩ ⟨1, by omega⟩, swap ⟨0, by omega⟩ ⟨2, by omega⟩,
    swap_swap_ne n _ _ _ (by simp [Fin.ext_iff]) (by simp [Fin.ext_iff]) (by simp [Fin.ext_iff])⟩

/-- The order of a permutation is the least common multiple of its cycle lengths: a `k`-cycle has order `k`. -/
theorem orderOf_eq_lcm_cycleType (n : ℕ) (g : Perm (Fin n)) : orderOf g = g.cycleType.lcm :=
  Perm.lcm_cycleType.symm

/-! ## exr-grp-s3-presentation -/

/-- Any transposition `s` and three-cycle `c` of `S₃` satisfy `s² = e`, `c³ = e`, `s c s⁻¹ = c⁻¹`. -/
theorem s3_relations (g h : Perm (Fin 3)) (hg : Perm.sign g = -1) (hh : h ∈ ofOrder (Perm (Fin 3)) 3) :
    g ^ 2 = 1 ∧ h ^ 3 = 1 ∧ g * h * g⁻¹ = h⁻¹ := by
  revert g h; decide

/-- Every element of `S₃` is `c^i s^j` with `i ∈ {0, 1, 2}`, `j ∈ {0, 1}`. -/
theorem s3_normal_form (g : Perm (Fin 3)) : ∃ i : Fin 3, ∃ j : Fin 2, g = c ^ (i : ℕ) * s ^ (j : ℕ) := by
  revert g; decide

/-- The products `c^i s`: `s = (1 2)`, `c s = (1 3)`, `c² s = (2 3)`. -/
theorem c_pow_mul_s (i : Fin 3) : c ^ (i : ℕ) * s = ![s, t13, t23] i := by
  revert i; decide

/-! ## exr-grp-s4-facts -/

/-- The cycle types of `S₄` and their counts: `1` identity, `6` transpositions, `3` double transpositions,
`8` three-cycles, `6` four-cycles. -/
theorem s4_cycleType_count (t : Multiset ℕ) :
    (Finset.univ.filter fun g : Perm (Fin 4) => g.cycleType = t).card =
      if t = {} then 1 else if t = {2} then 6 else if t = {2, 2} then 3 else if t = {3} then 8 else
        if t = {4} then 6 else 0 := by
  have hall : ∀ g : Perm (Fin 4), g.cycleType = {} ∨ g.cycleType = {2} ∨ g.cycleType = {2, 2} ∨ g.cycleType = {3} ∨
      g.cycleType = {4} := by decide
  by_cases h : t = {} ∨ t = {2} ∨ t = {2, 2} ∨ t = {3} ∨ t = {4}
  · rcases h with rfl | rfl | rfl | rfl | rfl <;> decide
  · push_neg at h
    obtain ⟨h0, h1, h2, h3, h4⟩ := h
    rw [Finset.card_eq_zero.mpr]
    · simp [h0, h1, h2, h3, h4]
    · apply Finset.filter_eq_empty_iff.mpr
      intro g _ hg
      rcases hall g with h' | h' | h' | h' | h' <;> rw [h'] at hg <;> simp_all

/-- The number of elements of `S₄` of each order: one of order `1`, nine of order `2`, eight of order `3`, six of
order `4`; all orders divide `24`. -/
theorem s4_order_count (k : ℕ) :
    (ofOrder (Perm (Fin 4)) k).card = if k = 1 then 1 else if k = 2 then 9 else if k = 3 then 8 else if k = 4 then 6 else 0 := by
  by_cases hk : k ≤ 24
  · interval_cases k <;> decide
  · have : ofOrder (Perm (Fin 4)) k = ∅ := by
      apply Finset.filter_eq_empty_iff.mpr
      intro g _ ⟨_, h⟩
      exact h 24 (by norm_num) (by omega) (by
        have h24 : Fintype.card (Perm (Fin 4)) = 24 := by decide
        rw [← h24]; exact pow_card_eq_one)
    rw [this]; simp; omega

/-- The even permutations of `S₄` are the identity, the double transpositions and the three-cycles. -/
theorem s4_even_iff (g : Perm (Fin 4)) : Perm.sign g = 1 ↔ g.cycleType = {} ∨ g.cycleType = {2, 2} ∨ g.cycleType = {3} := by
  revert g; decide

end KaluzaKlein.GroupTheory.Subgroups
