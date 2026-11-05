import KaluzaKlein.PartI.GroupTheory.Subgroups

/-!
# Group theory, section "Cosets"

The claims of the section "Cosets" of `tutorial/part-i-10-group-theory.qmd`: cosets as Finsets, equal or disjoint,
the cosets of `{0, 2, 4}` in `ℤ₆`, of `A₃` and `{e, (1 2)}` in `S₃`, of `⟨4⟩` and `⟨3⟩` in `ℤ₁₂`, the cosets of `SO(2)`
in `SO(3)` and the sphere, Lagrange's theorem with its consequences (groups of prime order, Fermat and Euler,
the failure of the converse for `A₄`).
-/

namespace KaluzaKlein.GroupTheory.Cosets

open Equiv Matrix KaluzaKlein.GroupTheory.Basic KaluzaKlein.GroupTheory.Groups KaluzaKlein.GroupTheory.Subgroups Real

/-- The even permutations of `S₃`, as a Finset. -/
def a3f : Finset (Perm (Fin 3)) := Finset.univ.filter fun g => Perm.sign g = 1

/-- The odd permutations of `S₃`, as a Finset. -/
def oddf : Finset (Perm (Fin 3)) := Finset.univ.filter fun g => Perm.sign g = -1

/-- The north pole `o = (0, 0, 1)`. -/
def northPole (K : Type) [Field K] : Fin 3 → K := ![0, 0, 1]

/-! ## def-left-coset -/

/-- Multiplying a subgroup Finset by one of its elements leaves it unchanged. -/
theorem image_mul_self {G : Type} [Group G] [DecidableEq G] (H : Finset G) (hH : IsSubgroupFinset H) (h : G) (hh : h ∈ H) :
    H.image (h * ·) = H := by
  apply Finset.eq_of_subset_of_card_le
  · intro x hx
    obtain ⟨y, hy, rfl⟩ := Finset.mem_image.mp hx
    exact hH.2.1 h hh y hy
  · rw [Finset.card_image_of_injective _ (mul_right_injective h)]

/-- Two left cosets are equal exactly when `g⁻¹ g' ∈ H`. -/
theorem leftCoset_eq_iff {G : Type} [Group G] [DecidableEq G] (H : Finset G) (hH : IsSubgroupFinset H) (g g' : G) :
    leftCoset g H = leftCoset g' H ↔ g⁻¹ * g' ∈ H := by
  constructor
  · intro h
    have : g' ∈ leftCoset g' H := Finset.mem_image.mpr ⟨1, hH.1, mul_one g'⟩
    rw [← h] at this
    obtain ⟨x, hx, hgx⟩ := Finset.mem_image.mp this
    rw [← hgx, ← mul_assoc, inv_mul_cancel, one_mul]; exact hx
  · intro h
    have hg' : g' = g * (g⁻¹ * g') := by rw [← mul_assoc, mul_inv_cancel, one_mul]
    rw [leftCoset, leftCoset, hg', ← image_mul_self H hH _ h, Finset.image_image]
    congr 1; ext x; simp [mul_assoc]

/-- Two left cosets are either equal or disjoint. -/
theorem leftCoset_eq_or_disjoint (G : Type) [Group G] [DecidableEq G] (H : Finset G) (hH : IsSubgroupFinset H) (g g' : G) :
    leftCoset g H = leftCoset g' H ∨ Disjoint (leftCoset g H) (leftCoset g' H) := by
  by_cases h : Disjoint (leftCoset g H) (leftCoset g' H)
  · exact Or.inr h
  · left
    rw [Finset.not_disjoint_iff] at h
    obtain ⟨x, hx, hx'⟩ := h
    obtain ⟨a, ha, rfl⟩ := Finset.mem_image.mp hx
    obtain ⟨b, hb, hab⟩ := Finset.mem_image.mp hx'
    rw [leftCoset_eq_iff H hH]
    have : g⁻¹ * g' = a * b⁻¹ := by
      rw [eq_mul_inv_iff_mul_eq, mul_assoc, hab, ← mul_assoc, inv_mul_cancel, one_mul]
    rw [this]; exact hH.2.1 a ha _ (hH.2.2 b hb)

/-- "`g ~ g'` iff `g⁻¹ g' ∈ H`" is an equivalence relation. -/
theorem coset_rel_equivalence (G : Type) [Group G] (H : Subgroup G) : Equivalence (fun g g' : G => g⁻¹ * g' ∈ H) :=
  ⟨fun g => by simp, fun h => by simpa using H.inv_mem h,
    fun h1 h2 => by simpa using H.mul_mem h1 h2⟩

/-- The classes of "`g⁻¹ g' ∈ H`" are the left cosets. -/
theorem coset_rel_class (G : Type) [Group G] [DecidableEq G] (H : Finset G) (hH : IsSubgroupFinset H) (g g' : G) :
    g⁻¹ * g' ∈ H ↔ g' ∈ leftCoset g H := by
  rw [← leftCoset_eq_iff H hH, eq_comm]
  constructor
  · intro h; rw [h]; exact Finset.mem_image.mpr ⟨1, hH.1, mul_one g'⟩
  · intro h
    obtain ⟨x, hx, rfl⟩ := Finset.mem_image.mp h
    rw [leftCoset_eq_iff H hH, mul_inv_rev, mul_assoc, ← mul_assoc g⁻¹, inv_mul_cancel, one_mul]
    exact hH.2.2 x hx

/-- The left cosets cover the group and are pairwise equal or disjoint: they partition `G`. -/
theorem leftCosets_partition (G : Type) [Group G] [Fintype G] [DecidableEq G] (H : Finset G) (hH : IsSubgroupFinset H) :
    (leftCosets G H).biUnion id = Finset.univ ∧
      ∀ C ∈ leftCosets G H, ∀ D ∈ leftCosets G H, C = D ∨ Disjoint C D := by
  refine ⟨?_, ?_⟩
  · apply Finset.eq_univ_of_forall
    intro g
    simp only [Finset.mem_biUnion, leftCosets, Finset.mem_image, Finset.mem_univ, true_and, id]
    exact ⟨leftCoset g H, ⟨g, rfl⟩, Finset.mem_image.mpr ⟨1, hH.1, mul_one g⟩⟩
  · intro C hC D hD
    simp only [leftCosets, Finset.mem_image, Finset.mem_univ, true_and] at hC hD
    obtain ⟨g, rfl⟩ := hC
    obtain ⟨g', rfl⟩ := hD
    exact leftCoset_eq_or_disjoint G H hH g g'

/-! ## exm-cosets-z6 -/

/-- The cosets of `H = {0, 2, 4}` in `ℤ₆`: `a + H` is `H` for even `a` and `{1, 3, 5}` for odd `a`. -/
theorem z6_cosets (a : ZMod 6) : addCoset a {0, 2, 4} = if a.val % 2 = 0 then {0, 2, 4} else {1, 3, 5} := by
  revert a; decide

/-- In an abelian group left and right cosets coincide. -/
theorem leftCoset_eq_rightCoset (G : Type) [CommGroup G] [DecidableEq G] (g : G) (H : Finset G) :
    leftCoset g H = rightCoset H g := by
  unfold leftCoset rightCoset; congr 1; ext x; exact mul_comm g x

/-! ## exm-cosets-s3 -/

/-- The cosets of `A₃` in `S₃`: `g A₃` is `A₃` for even `g` and the odd permutations for odd `g`; left and right
cosets agree, and there are two of them. -/
theorem s3_cosets_a3 (g : Perm (Fin 3)) :
    leftCoset g a3f = (if Perm.sign g = 1 then a3f else oddf) ∧ rightCoset a3f g = leftCoset g a3f ∧
      (leftCosets (Perm (Fin 3)) a3f).card = 2 := by
  revert g; decide

/-- The cosets of `H = {e, (1 2)}`: the left cosets are `{e, (1 2)}`, `{(1 2 3), (1 3)}`, `{(1 3 2), (2 3)}`, the right
cosets `{e, (1 2)}`, `{(1 2 3), (2 3)}`, `{(1 3 2), (1 3)}`; there are three of each and the two families differ. -/
theorem s3_cosets_h (g : Perm (Fin 3)) :
    leftCoset g {1, s} ∈ ({{1, s}, {c, t13}, {c ^ 2, t23}} : Finset (Finset (Perm (Fin 3)))) ∧
      rightCoset {1, s} g ∈ ({{1, s}, {c, t23}, {c ^ 2, t13}} : Finset (Finset (Perm (Fin 3)))) ∧
      (leftCosets (Perm (Fin 3)) {1, s}).card = 3 ∧ (rightCosets (Perm (Fin 3)) {1, s}).card = 3 ∧
      leftCosets (Perm (Fin 3)) {1, s} ≠ rightCosets (Perm (Fin 3)) {1, s} := by
  revert g; decide

/-- Left and right cosets agree for every element exactly when the subgroup is normal. -/
theorem leftCoset_eq_rightCoset_iff (G : Type) [Group G] [Fintype G] [DecidableEq G] (H : Finset G)
    (hH : IsSubgroupFinset H) : (∀ g, leftCoset g H = rightCoset H g) ↔ IsNormalFinset H := by
  constructor
  · intro h g x hx
    have : g * x ∈ rightCoset H g := by rw [← h g]; exact Finset.mem_image.mpr ⟨x, hx, rfl⟩
    obtain ⟨y, hy, hyx⟩ := Finset.mem_image.mp this
    have : g * x * g⁻¹ = y := by rw [hyx, mul_inv_cancel_right]
    rw [this]; exact hy
  · intro hN g
    apply Finset.eq_of_subset_of_card_le
    · intro x hx
      obtain ⟨y, hy, rfl⟩ := Finset.mem_image.mp hx
      refine Finset.mem_image.mpr ⟨g * y * g⁻¹, hN g y hy, ?_⟩
      simp [mul_assoc]
    · rw [Finset.card_image_of_injective _ (mul_right_injective g),
        Finset.card_image_of_injective _ (mul_left_injective g)]

/-! ## exm-cosets-so3 -/

/-- An orthogonal matrix preserves the dot product. -/
theorem orthogonal_preserves_dot (K : Type) [CommRing K] (n : ℕ) (A : Matrix (Fin n) (Fin n) K) (hA : Aᵀ * A = 1)
    (v w : Fin n → K) : (A *ᵥ v) ⬝ᵥ (A *ᵥ w) = v ⬝ᵥ w := by
  rw [Matrix.dotProduct_mulVec, Matrix.vecMul_mulVec, ← Matrix.mulVec_transpose, Matrix.mulVec_mulVec, hA,
    Matrix.one_mulVec]

/-- For orthogonal `g`, `g⁻¹ g' = gᵀ g'` fixes the north pole exactly when `g` and `g'` send it to the same point:
two rotations lie in the same coset of `SO(2)` exactly when they agree on the north pole. -/
theorem same_coset_iff_same_image (K : Type) [Field K] (g g' : Matrix (Fin 3) (Fin 3) K) (hg : gᵀ * g = 1) :
    (gᵀ * g') *ᵥ northPole K = northPole K ↔ g *ᵥ northPole K = g' *ᵥ northPole K := by
  have hg' : g * gᵀ = 1 := Matrix.mul_eq_one_comm.mp hg
  constructor
  · intro h
    have := congrArg (g *ᵥ ·) h
    simp only [Matrix.mulVec_mulVec, ← Matrix.mul_assoc, hg', Matrix.one_mul] at this
    exact this.symm
  · intro h
    rw [← Matrix.mulVec_mulVec, ← h, Matrix.mulVec_mulVec, hg, Matrix.one_mulVec]

/-- Every unit vector is the image of the north pole under a rotation: `SO(3)` is transitive on the sphere, and
the map `g SO(2) ↦ g · o` is onto `S²` (stated; the oracle constructs the rotations). -/
def SphereTransitive : Prop :=
  ∀ v : Fin 3 → ℝ, v ⬝ᵥ v = 1 → ∃ g : Matrix (Fin 3) (Fin 3) ℝ, gᵀ * g = 1 ∧ g.det = 1 ∧ g *ᵥ northPole ℝ = v

/-- The rotations fixing the north pole are exactly the rotations about the `Z`-axis: the stabiliser of `o` is
`SO(2)` (stated; the oracle checks the block form). -/
def StabilizerNorthPole : Prop :=
  ∀ A : Matrix (Fin 3) (Fin 3) ℝ, Aᵀ * A = 1 → A.det = 1 → (A *ᵥ northPole ℝ = northPole ℝ ↔ ∃ θ, A = Rz θ)

/-! ## exr-cosq-cosets-z12 -/

/-- The cosets of `⟨4⟩ = {0, 4, 8}` in `ℤ₁₂` are the residue classes modulo `4`. -/
theorem z12_cosets_four (a : ZMod 12) : addCoset a (multiples 12 4) = Finset.univ.filter fun x => x.val % 4 = a.val % 4 := by
  revert a; decide

/-- The cosets of `⟨3⟩ = {0, 3, 6, 9}` in `ℤ₁₂` are the residue classes modulo `3`. -/
theorem z12_cosets_three (a : ZMod 12) : addCoset a (multiples 12 3) = Finset.univ.filter fun x => x.val % 3 = a.val % 3 := by
  revert a; decide

/-- For a divisor `d` of `12`, the subgroup `⟨d⟩` has `d` cosets, each of `12 / d` elements, and they partition
`ℤ₁₂`. -/
theorem z12_cosets_count (d : ℕ) (hd : d ∣ 12) :
    (addCosets (ZMod 12) (multiples 12 d)).card = d ∧ (∀ C ∈ addCosets (ZMod 12) (multiples 12 d), C.card = 12 / d) ∧
      (addCosets (ZMod 12) (multiples 12 d)).biUnion id = Finset.univ := by
  have : d ≤ 12 := Nat.le_of_dvd (by norm_num) hd
  interval_cases d <;> first | decide | (exfalso; norm_num at hd)

/-! ## thm-lagrange -/

/-- Every left coset has as many elements as `H` (`h ↦ g h` is a bijection). -/
theorem card_leftCoset (G : Type) [Group G] [DecidableEq G] (g : G) (H : Finset G) : (leftCoset g H).card = H.card :=
  Finset.card_image_of_injective _ (mul_right_injective g)

/-- Lagrange's theorem: the order of a subgroup divides the order of the group. -/
theorem lagrange (G : Type) [Group G] (H : Subgroup G) : Nat.card H ∣ Nat.card G :=
  Subgroup.card_subgroup_dvd_card H

/-- The number of cosets times the order of the subgroup is the order of the group: the index is `|G| / |H|`. -/
theorem card_leftCosets_mul (G : Type) [Group G] [Fintype G] [DecidableEq G] (H : Finset G) (hH : IsSubgroupFinset H) :
    (leftCosets G H).card * H.card = Fintype.card G := by
  obtain ⟨hcover, hdisj⟩ := leftCosets_partition G H hH
  have hcard : ∀ C ∈ leftCosets G H, C.card = H.card := by
    intro C hC
    simp only [leftCosets, Finset.mem_image, Finset.mem_univ, true_and] at hC
    obtain ⟨g, rfl⟩ := hC
    exact card_leftCoset G g H
  rw [← Finset.card_univ, ← hcover, Finset.card_biUnion]
  · rw [Finset.sum_congr rfl hcard, Finset.sum_const, smul_eq_mul]
  · intro C hC D hD hCD
    rcases hdisj C hC D hD with h | h
    · exact absurd h hCD
    · exact h

/-- The index of a subgroup is `|G| / |H|`. -/
theorem index_eq_div (G : Type) [Group G] [Finite G] (H : Subgroup G) : H.index = Nat.card G / Nat.card H := by
  rw [← H.card_mul_index, Nat.mul_div_cancel_left _ Nat.card_pos]

/-! ## exm-lagrange-s3 -/

/-- `S₃` has no subgroup of order `4` (`4 ∤ 6`), and its subgroups have orders `1, 2, 3, 6`. -/
theorem s3_subgroup_orders (H : Finset (Perm (Fin 3))) (hH : IsSubgroupFinset H) :
    H.card ≠ 4 ∧ H.card ∈ ({1, 2, 3, 6} : Finset ℕ) := by
  revert H; decide

/-! ## exm-lagrange-prime -/

/-- A group of prime order has no subgroups besides `{e}` and itself. -/
theorem prime_card_subgroups (G : Type) [Group G] (p : ℕ) [Fact p.Prime] (h : Nat.card G = p) (H : Subgroup G) :
    H = ⊥ ∨ H = ⊤ := by
  rcases (Nat.dvd_prime (Fact.out)).mp (h ▸ Subgroup.card_subgroup_dvd_card H) with h1 | h1
  · left; exact Subgroup.eq_bot_of_card_eq H h1
  · right; exact Subgroup.eq_top_of_card_eq H (h1.trans h.symm)

/-- In a group of prime order every element other than `e` generates the group. -/
theorem prime_card_generates (G : Type) [Group G] (p : ℕ) [Fact p.Prime] (h : Nat.card G = p) (g : G) (hg : g ≠ 1) :
    ∀ x, x ∈ Subgroup.zpowers g := by
  rcases prime_card_subgroups G p h (Subgroup.zpowers g) with h1 | h1
  · exact absurd (Subgroup.zpowers_eq_bot.mp h1) hg
  · intro x; rw [h1]; exact Subgroup.mem_top x

/-- A group of prime order is cyclic. -/
theorem prime_card_isCyclic (G : Type) [Group G] (p : ℕ) [Fact p.Prime] (h : Nat.card G = p) : IsCyclic G :=
  isCyclic_of_prime_card h

/-- A group of prime order `p` is isomorphic to `ℤₚ` (stated; the oracle checks the isomorphism types). -/
def PrimeOrderIsZMod : Prop :=
  ∀ (G : Type) [Group G] (p : ℕ), p.Prime → Nat.card G = p → Nonempty (Multiplicative (ZMod p) ≃* G)

/-! ## exm-lagrange-infinite -/

/-- The antisymmetric `3 × 3` matrices (the Lie algebra `so(3)`), flattened to vectors of `ℚ⁹`, span a space of
dimension `3`. -/
theorem so3_dim : Module.finrank ℚ (Submodule.span ℚ (Set.range ![![(0 : ℚ), 0, 0, 0, 0, -1, 0, 1, 0],
    ![0, 0, 1, 0, 0, 0, -1, 0, 0], ![0, -1, 0, 1, 0, 0, 0, 0, 0]])) = 3 := by
  VERIPHY[gt-dim-so3]

/-- The generators of rotations about the `Z`-axis span a space of dimension `1`: `dim SO(3)/SO(2) = 3 - 1 = 2`. -/
theorem so2_dim : Module.finrank ℚ (Submodule.span ℚ (Set.range ![![(0 : ℚ), -1, 0, 1, 0, 0, 0, 0, 0]])) = 1 := by
  VERIPHY[gt-dim-so2]

/-! ## exr-cosq-fermat -/

/-- `g^|G| = e` in a finite group. -/
theorem pow_card (G : Type) [Group G] [Fintype G] (g : G) : g ^ Fintype.card G = 1 := pow_card_eq_one

/-- Fermat's little theorem: `a^(p-1) ≡ 1 (mod p)` for `p ∤ a`. -/
theorem fermat (p : ℕ) [Fact p.Prime] (a : ZMod p) (ha : a ≠ 0) : a ^ (p - 1) = 1 :=
  ZMod.pow_card_sub_one_eq_one ha

/-- Euler's theorem: `a^φ(n) ≡ 1 (mod n)` for `a` coprime to `n`. -/
theorem euler (n : ℕ) [NeZero n] (a : ℕ) (h : Nat.Coprime a n) : (a : ZMod n) ^ Nat.totient n = 1 := by
  have := ZMod.pow_totient (ZMod.unitOfCoprime a h)
  have h2 := congrArg (fun u : (ZMod n)ˣ => (u : ZMod n)) this
  simpa [ZMod.coe_unitOfCoprime] using h2

/-- `φ(n)` counts the residues coprime to `n`: `φ(6) = 2`. -/
theorem totient_card (n : ℕ) : Nat.totient n = ((Finset.range n).filter n.Coprime).card :=
  Nat.totient_eq_card_coprime n

/-! ## exr-cosq-a4-no-6 -/

/-- The subgroups of `A₄` have orders `1, 2, 3, 4, 12`: there is none of order `6` (stated; the oracle lists the
subgroups). -/
def A4SubgroupOrders : Prop :=
  ∀ H : Subgroup (alternatingGroup (Fin 4)), Nat.card H ∈ ({1, 2, 3, 4, 12} : Finset ℕ)

/-- A subgroup of index `2` is normal. -/
theorem normal_of_index_two (G : Type) [Group G] (H : Subgroup G) (h : H.index = 2) : H.Normal :=
  Subgroup.normal_of_index_eq_two h

/-- A normal subgroup of index `2` contains every square (the quotient is `ℤ₂`). -/
theorem sq_mem_of_index_two (G : Type) [Group G] (H : Subgroup G) (h : H.index = 2) (g : G) : g ^ 2 ∈ H := by
  have hN := Subgroup.normal_of_index_eq_two h
  have : ∀ x : G ⧸ H, x ^ 2 = 1 := by
    intro x
    have hc : Nat.card (G ⧸ H) = 2 := h
    have := pow_card_eq_one' (G := G ⧸ H) (x := x)
    rwa [hc] at this
  have := this (QuotientGroup.mk g)
  rwa [← QuotientGroup.mk_pow, QuotientGroup.eq_one_iff] at this

/-- Every three-cycle of `S₄` is the square of a three-cycle. -/
theorem three_cycle_is_square (g : Perm (Fin 4)) (hg : g.cycleType = {3}) : ∃ h : Perm (Fin 4), h.cycleType = {3} ∧ h ^ 2 = g := by
  revert g; decide

/-- `A₄` has `12` elements: the identity, three double transpositions and eight three-cycles. -/
theorem a4_composition (t : Multiset ℕ) :
    (a4.filter fun g => g.cycleType = t).card = if t = {} then 1 else if t = {2, 2} then 3 else if t = {3} then 8 else 0 := by
  have hall : ∀ g : Perm (Fin 4), g.cycleType = {} ∨ g.cycleType = {2} ∨ g.cycleType = {2, 2} ∨ g.cycleType = {3} ∨
      g.cycleType = {4} := by decide
  by_cases h : t = {} ∨ t = {2} ∨ t = {2, 2} ∨ t = {3} ∨ t = {4}
  · rcases h with rfl | rfl | rfl | rfl | rfl <;> decide
  · push_neg at h
    obtain ⟨h0, h1, h2, h3, h4⟩ := h
    rw [Finset.card_eq_zero.mpr]
    · simp [h0, h2, h3]
    · apply Finset.filter_eq_empty_iff.mpr
      intro g _ hg
      rcases hall g with h' | h' | h' | h' | h' <;> rw [h'] at hg <;> simp_all

end KaluzaKlein.GroupTheory.Cosets
