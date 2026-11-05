import Mathlib

/-!
# Group theory: shared objects

The running examples of `tutorial/part-i-10-group-theory.qmd`: the transposition `s = (1 2)` and the three-cycle
`c = (1 2 3)` of `S₃ = Perm (Fin 3)` (products are composition, "first `h`, then `g`" for `g * h`), the rotation
matrices `R θ` of the plane and the quarter turns of space, the dihedral group `D₄ = DihedralGroup 4`, the quaternion
group `Q₈ = QuaternionGroup 2`, the Klein four-group `ZMod 2 × ZMod 2`, and the counting functions used by the
exercises (commuting pairs, elements of a given order, conjugacy classes).
-/

namespace KaluzaKlein.GroupTheory.Basic

open Equiv Matrix

/-- The transposition `s = (1 2)` of `S₃`, on the labels `0, 1, 2`. -/
def s : Perm (Fin 3) := swap 0 1

/-- The three-cycle `c = (1 2 3)`: `0 ↦ 1 ↦ 2 ↦ 0`. -/
def c : Perm (Fin 3) := finRotate 3

/-- The transposition `(1 3)`. -/
def t13 : Perm (Fin 3) := swap 0 2

/-- The transposition `(2 3)`. -/
def t23 : Perm (Fin 3) := swap 1 2

/-- The rotation of the plane by the angle `θ`. -/
noncomputable def R (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]

/-- The rotation of space by `θ` about the `Z`-axis. -/
noncomputable def Rz (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![Real.cos θ, -Real.sin θ, 0; Real.sin θ, Real.cos θ, 0; 0, 0, 1]

/-- The rotation of space by `θ` about the `X`-axis. -/
noncomputable def Rx (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![1, 0, 0; 0, Real.cos θ, -Real.sin θ; 0, Real.sin θ, Real.cos θ]

/-- The quarter turn about the `Z`-axis, an integer matrix. -/
def qz : Matrix (Fin 3) (Fin 3) ℤ := !![0, -1, 0; 1, 0, 0; 0, 0, 1]

/-- The quarter turn about the `X`-axis, an integer matrix. -/
def qx : Matrix (Fin 3) (Fin 3) ℤ := !![1, 0, 0; 0, 0, -1; 0, 1, 0]

/-- The number of ordered pairs `(g, h)` of a finite group with `g * h = h * g`. -/
def commutingPairs (G : Type) [Group G] [Fintype G] [DecidableEq G] : ℕ :=
  (Finset.univ.filter fun p : G × G => p.1 * p.2 = p.2 * p.1).card

/-- The number of ordered pairs `(a, b)` of a finite additive group with `a + b = b + a`. -/
def addCommutingPairs (G : Type) [AddGroup G] [Fintype G] [DecidableEq G] : ℕ :=
  (Finset.univ.filter fun p : G × G => p.1 + p.2 = p.2 + p.1).card

/-- The elements of a finite group of order exactly `k` (`g ^ k = 1` and no smaller positive power is `1`),
as a Finset: a computable form of `orderOf g = k`. -/
def ofOrder (G : Type) [Group G] [Fintype G] [DecidableEq G] (k : ℕ) : Finset G :=
  Finset.univ.filter fun g => g ^ k = 1 ∧ ∀ m, 0 < m → m < k → g ^ m ≠ 1

/-- The elements of a finite additive group of order exactly `k`. -/
def ofAddOrder (G : Type) [AddGroup G] [Fintype G] [DecidableEq G] (k : ℕ) : Finset G :=
  Finset.univ.filter fun g => k • g = 0 ∧ ∀ m, 0 < m → m < k → m • g ≠ 0

/-- `g ∈ ofOrder G k` exactly when `orderOf g = k` (for `0 < k`). -/
theorem mem_ofOrder_iff {G : Type} [Group G] [Fintype G] [DecidableEq G] {k : ℕ} (hk : 0 < k) (g : G) :
    g ∈ ofOrder G k ↔ orderOf g = k := by
  rw [ofOrder, Finset.mem_filter, orderOf_eq_iff hk]
  simp only [Finset.mem_univ, true_and]
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨h1, fun m hm hm0 => h2 m hm0 hm⟩
  · rintro ⟨h1, h2⟩; exact ⟨h1, fun m hm0 hm => h2 m hm hm0⟩

/-- `g ∈ ofAddOrder G k` exactly when `addOrderOf g = k` (for `0 < k`). -/
theorem mem_ofAddOrder_iff {G : Type} [AddGroup G] [Fintype G] [DecidableEq G] {k : ℕ} (hk : 0 < k) (g : G) :
    g ∈ ofAddOrder G k ↔ addOrderOf g = k := by
  rw [ofAddOrder, Finset.mem_filter, addOrderOf_eq_iff hk]
  simp only [Finset.mem_univ, true_and]
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨h1, fun m hm hm0 => h2 m hm0 hm⟩
  · rintro ⟨h1, h2⟩; exact ⟨h1, fun m hm0 hm => h2 m hm hm0⟩

/-- The conjugacy class of `g` in a finite group, as a Finset. -/
def conjClass {G : Type} [Group G] [Fintype G] [DecidableEq G] (g : G) : Finset G :=
  Finset.univ.image fun h => h * g * h⁻¹

/-- The conjugacy classes of a finite group, as a Finset of Finsets. -/
def conjClasses (G : Type) [Group G] [Fintype G] [DecidableEq G] : Finset (Finset G) :=
  Finset.univ.image fun g : G => conjClass g

/-- The left coset `g * H` of a Finset `H`, as a Finset. -/
def leftCoset {G : Type} [Group G] [DecidableEq G] (g : G) (H : Finset G) : Finset G := H.image (g * ·)

/-- The right coset `H * g` of a Finset `H`, as a Finset. -/
def rightCoset {G : Type} [Group G] [DecidableEq G] (H : Finset G) (g : G) : Finset G := H.image (· * g)

/-- The family of left cosets of a Finset `H` in a finite group. -/
def leftCosets (G : Type) [Group G] [Fintype G] [DecidableEq G] (H : Finset G) : Finset (Finset G) :=
  Finset.univ.image fun g : G => leftCoset g H

/-- The family of right cosets of a Finset `H` in a finite group. -/
def rightCosets (G : Type) [Group G] [Fintype G] [DecidableEq G] (H : Finset G) : Finset (Finset G) :=
  Finset.univ.image fun g : G => rightCoset H g

/-- The additive coset `a + H` of a Finset `H`. -/
def addCoset {G : Type} [AddGroup G] [DecidableEq G] (a : G) (H : Finset G) : Finset G := H.image (a + ·)

/-- The family of additive cosets of a Finset `H` in a finite additive group. -/
def addCosets (G : Type) [AddGroup G] [Fintype G] [DecidableEq G] (H : Finset G) : Finset (Finset G) :=
  Finset.univ.image fun a : G => addCoset a H

/-- A Finset `H` of a group is a subgroup: it contains `1` and is closed under products and inverses. -/
def IsSubgroupFinset {G : Type} [Group G] (H : Finset G) : Prop :=
  (1 : G) ∈ H ∧ (∀ a ∈ H, ∀ b ∈ H, a * b ∈ H) ∧ ∀ a ∈ H, a⁻¹ ∈ H

/-- A Finset `H` of an additive group is a subgroup. -/
def IsAddSubgroupFinset {G : Type} [AddGroup G] (H : Finset G) : Prop :=
  (0 : G) ∈ H ∧ (∀ a ∈ H, ∀ b ∈ H, a + b ∈ H) ∧ ∀ a ∈ H, -a ∈ H

/-- A subgroup Finset is normal: closed under conjugation by every element of the group. -/
def IsNormalFinset {G : Type} [Group G] [Fintype G] (H : Finset G) : Prop :=
  ∀ g : G, ∀ h ∈ H, g * h * g⁻¹ ∈ H

instance {G : Type} [Group G] [DecidableEq G] (H : Finset G) : Decidable (IsSubgroupFinset H) := by
  unfold IsSubgroupFinset; infer_instance

instance {G : Type} [AddGroup G] [DecidableEq G] (H : Finset G) : Decidable (IsAddSubgroupFinset H) := by
  unfold IsAddSubgroupFinset; infer_instance

instance {G : Type} [Group G] [Fintype G] [DecidableEq G] (H : Finset G) : Decidable (IsNormalFinset H) := by
  unfold IsNormalFinset; infer_instance

/-- The subgroup of `G` carried by a subgroup Finset. -/
def toSubgroup {G : Type} [Group G] (H : Finset G) (h : IsSubgroupFinset H) : Subgroup G where
  carrier := ↑H
  one_mem' := h.1
  mul_mem' := fun ha hb => h.2.1 _ ha _ hb
  inv_mem' := fun ha => h.2.2 _ ha

/-- The words of length at most `k` in a generating Finset `S` and its inverses: `k` steps of closing `{1}` under
right multiplication by `S ∪ S⁻¹`. -/
def words {G : Type} [Group G] [DecidableEq G] (S : Finset G) : ℕ → Finset G
  | 0 => {1}
  | k + 1 => words S k ∪ ((words S k) ×ˢ (S ∪ S.image (·⁻¹))).image fun p => p.1 * p.2

/-- Every word in `S` and `S⁻¹` lies in the subgroup generated by `S`. -/
theorem words_subset_closure {G : Type} [Group G] [DecidableEq G] (S : Finset G) (k : ℕ) :
    ∀ g ∈ words S k, g ∈ Subgroup.closure (S : Set G) := by
  induction k with
  | zero => intro g hg; simp [words] at hg; subst hg; exact Subgroup.one_mem _
  | succ k ih =>
    intro g hg
    simp only [words, Finset.mem_union, Finset.mem_image, Finset.mem_product] at hg
    rcases hg with hg | ⟨⟨a, b⟩, ⟨ha, hb⟩, rfl⟩
    · exact ih g hg
    · refine Subgroup.mul_mem _ (ih a ha) ?_
      simp only [Finset.mem_union, Finset.mem_image] at hb
      rcases hb with hb | ⟨b', hb', rfl⟩
      · exact Subgroup.subset_closure hb
      · exact Subgroup.inv_mem _ (Subgroup.subset_closure hb')

/-- The subgroup generated by a Finset `S` is the subgroup Finset `H` as soon as `H` contains `S`, is closed under
the group operations and consists of words in `S`: the two inclusions of `⟨S⟩ = H`. -/
theorem closure_eq_of_words {G : Type} [Group G] [DecidableEq G] (S H : Finset G) (k : ℕ) (hH : IsSubgroupFinset H)
    (hS : S ⊆ H) (hw : H ⊆ words S k) : (Subgroup.closure (S : Set G) : Set G) = ↑H := by
  apply le_antisymm
  · have : Subgroup.closure (S : Set G) ≤ toSubgroup H hH := Subgroup.closure_le _ |>.mpr (by
      intro x hx; exact hS hx)
    intro x hx; exact this hx
  · intro x hx; exact words_subset_closure S k x (hw hx)

end KaluzaKlein.GroupTheory.Basic
