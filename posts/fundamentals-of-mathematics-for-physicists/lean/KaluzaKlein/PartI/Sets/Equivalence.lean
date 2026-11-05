import Mathlib
import KaluzaKlein.PartI.Sets.Composition
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Equivalence relations and quotients" (relations and classes)

The claims of the blocks `def-equivalence-relation` to `exr-class-partition-proof` of
`tutorial/part-i-01-sets-relations-topology.qmd`: congruence modulo `n` and modulo `2π`, "same value under a map",
the three relations on `{1, 2, 3, 4}`, parity, closeness and the equivalence relation it generates, the classes modulo
`3`, the classes of angles, of "same square" and of "same cosine", and the partition by classes.
-/

namespace KaluzaKlein.Sets.Equivalence

/-! ## exm-equivalence-mod -/

/-- Congruence modulo `n` is an equivalence relation on `ℤ` (parity for `n = 2`). -/
theorem modEq_equivalence (n : ℤ) : Equivalence (fun a b : ℤ => n ∣ a - b) :=
  ⟨fun a => by simp, fun h => by rw [← neg_sub]; exact (dvd_neg).2 h,
    fun h1 h2 => by have := dvd_add h1 h2; rwa [sub_add_sub_cancel] at this⟩

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.modEq_equivalence (3) := by
  have key : ∀ (n : ℤ), Equivalence (fun a b : ℤ => n ∣ a - b) := by
    intro n
    exact (⟨fun a => by simp, fun h => by rw [← neg_sub]; exact (dvd_neg).2 h,
      fun h1 h2 => by have := dvd_add h1 h2; rwa [sub_add_sub_cancel] at this⟩)
  exact @key (3)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.modEq_equivalence (0) := by
  have key : ∀ (n : ℤ), Equivalence (fun a b : ℤ => n ∣ a - b) := by
    intro n
    exact (⟨fun a => by simp, fun h => by rw [← neg_sub]; exact (dvd_neg).2 h,
      fun h1 h2 => by have := dvd_add h1 h2; rwa [sub_add_sub_cancel] at this⟩)
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.modEq_equivalence (2) := by
  have key : ∀ (n : ℤ), Equivalence (fun a b : ℤ => n ∣ a - b) := by
    intro n
    exact (⟨fun a => by simp, fun h => by rw [← neg_sub]; exact (dvd_neg).2 h,
      fun h1 h2 => by have := dvd_add h1 h2; rwa [sub_add_sub_cancel] at this⟩)
  exact @key (2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.modEq_equivalence
-- test: set-modeq-equivalence-test.sage

/-! ## exm-equivalence-angle -/

/-- Differing by a multiple of `T` is an equivalence relation on `ℝ`: the angles for `T = 2π`. -/
theorem angle_equivalence (T : ℝ) : Equivalence (fun x y : ℝ => ∃ k : ℤ, y - x = k * T) :=
  ⟨fun x => ⟨0, by simp⟩, fun ⟨k, hk⟩ => ⟨-k, by push_cast; linarith⟩,
    fun ⟨k, hk⟩ ⟨l, hl⟩ => ⟨k + l, by push_cast; linarith⟩⟩

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.angle_equivalence (2 * Real.pi) := by
  have key : ∀ (T : ℝ), Equivalence (fun x y : ℝ => ∃ k : ℤ, y - x = k * T) := by
    intro T
    exact (⟨fun x => ⟨0, by simp⟩, fun ⟨k, hk⟩ => ⟨-k, by push_cast; linarith⟩,
      fun ⟨k, hk⟩ ⟨l, hl⟩ => ⟨k + l, by push_cast; linarith⟩⟩)
  exact @key (2 * Real.pi)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.angle_equivalence (0) := by
  have key : ∀ (T : ℝ), Equivalence (fun x y : ℝ => ∃ k : ℤ, y - x = k * T) := by
    intro T
    exact (⟨fun x => ⟨0, by simp⟩, fun ⟨k, hk⟩ => ⟨-k, by push_cast; linarith⟩,
      fun ⟨k, hk⟩ ⟨l, hl⟩ => ⟨k + l, by push_cast; linarith⟩⟩)
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.angle_equivalence (1) := by
  have key : ∀ (T : ℝ), Equivalence (fun x y : ℝ => ∃ k : ℤ, y - x = k * T) := by
    intro T
    exact (⟨fun x => ⟨0, by simp⟩, fun ⟨k, hk⟩ => ⟨-k, by push_cast; linarith⟩,
      fun ⟨k, hk⟩ ⟨l, hl⟩ => ⟨k + l, by push_cast; linarith⟩⟩)
  exact @key (1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.angle_equivalence
-- test: set-angle-equivalence-test.sage

/-! ## exm-equivalence-coset -/

/-- For any map `f`, "same value under `f`" is an equivalence relation (for the orbit map on `SO(3)`, "same image of
the north pole"). -/
theorem sameValue_equivalence {X Y : Type} (f : X → Y) : Equivalence (fun x x' => f x = f x') :=
  ⟨fun _ => rfl, fun h => h.symm, fun h1 h2 => h1.trans h2⟩

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.sameValue_equivalence (ℝ) (ℝ) (fun x => x ^ 2) := by
  have key : ∀ {X Y : Type} (f : X → Y), Equivalence (fun x x' => f x = f x') := by
    intro X Y f
    exact (⟨fun _ => rfl, fun h => h.symm, fun h1 h2 => h1.trans h2⟩)
  exact @key (ℝ) (ℝ) (fun x => x ^ 2)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.sameValue_equivalence (Unit) (Unit) (id) := by
  have key : ∀ {X Y : Type} (f : X → Y), Equivalence (fun x x' => f x = f x') := by
    intro X Y f
    exact (⟨fun _ => rfl, fun h => h.symm, fun h1 h2 => h1.trans h2⟩)
  exact @key (Unit) (Unit) (id)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.sameValue_equivalence (Matrix (Fin 3) (Fin 3) ℝ) (Fin 3 → ℝ) (KaluzaKlein.Sets.Maps.orbit) := by
  have key : ∀ {X Y : Type} (f : X → Y), Equivalence (fun x x' => f x = f x') := by
    intro X Y f
    exact (⟨fun _ => rfl, fun h => h.symm, fun h1 h2 => h1.trans h2⟩)
  exact @key (Matrix (Fin 3) (Fin 3) ℝ) (Fin 3 → ℝ) (KaluzaKlein.Sets.Maps.orbit)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.sameValue_equivalence
-- test: set-samevalue-test.sage

/-! ## exr-equivalence-check -/

/-- `≤` is reflexive and transitive. -/
theorem le_refl_trans (a b c : ℤ) : a ≤ a ∧ (a ≤ b → b ≤ c → a ≤ c) :=
  ⟨le_refl a, le_trans⟩

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.le_refl_trans (1) (2) (3) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.le_refl_trans (4) (1) (2) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.le_refl_trans (2) (2) (2) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.le_refl_trans
-- test: set-le-refl-trans-test.sage

/-- `≤` is not symmetric on `{1, 2, 3, 4}` (Sage finds `1 ≤ 2`, `2 ≰ 1`). -/
theorem le_not_symm : ∃ a b : ℤ, 1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 4 ∧ a ≤ b ∧ ¬ b ≤ a := by
  veriphy_certified (exact ⟨1, 2, by decide⟩)
  -- certificate: set-le-not-symm.sage

/-- `|a - b| ≤ 1` is reflexive and symmetric. -/
theorem close_refl_symm (a b : ℤ) : |a - a| ≤ 1 ∧ (|a - b| ≤ 1 → |b - a| ≤ 1) :=
  ⟨by simp, fun h => by rwa [abs_sub_comm]⟩

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.close_refl_symm (1) (2) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.close_refl_symm (1) (3) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.close_refl_symm (4) (4) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.close_refl_symm
-- test: set-close-refl-symm-test.sage

/-- `|a - b| ≤ 1` is not transitive on `{1, 2, 3, 4}` (Sage finds `1 ∼ 2`, `2 ∼ 3`, `1 ≁ 3`). -/
theorem close_not_trans : ∃ a b c : ℤ, 1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 4 ∧ 1 ≤ c ∧ c ≤ 4 ∧
    |a - b| ≤ 1 ∧ |b - c| ≤ 1 ∧ ¬ |a - c| ≤ 1 := by
  veriphy_certified (exact ⟨1, 2, 3, by decide⟩)
  -- certificate: set-close-not-trans.sage

/-! ## exr-equivalence-parity -/

/-- "`a - b` is odd" is not reflexive (`a - a = 0` is even) and not transitive (odd plus odd is even). -/
theorem odd_diff_not_equivalence (a b c : ℤ) : ¬ Odd (a - a) ∧ (Odd (a - b) → Odd (b - c) → ¬ Odd (a - c)) := by
  refine ⟨by simp, fun h1 h2 h3 => ?_⟩
  have : Even ((a - b) + (b - c)) := Odd.add_odd h1 h2
  rw [sub_add_sub_cancel] at this
  exact (Int.not_even_iff_odd.2 h3) this

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.odd_diff_not_equivalence (1) (2) (3) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.odd_diff_not_equivalence (0) (5) (-2) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.odd_diff_not_equivalence (4) (4) (4) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.odd_diff_not_equivalence
-- test: set-odd-diff-test.sage

/-! ## exr-equivalence-closeness -/

/-- Closeness `|x - y| < 1` is not transitive (Sage finds `0 ∼ 0.9 ∼ 1.8`, `|0 - 1.8| ≥ 1`). -/
theorem closeness_not_trans : ∃ x y z : ℝ, |x - y| < 1 ∧ |y - z| < 1 ∧ ¬ |x - z| < 1 := by
  veriphy_certified (refine ⟨(0 : ℝ), (9 / 10 : ℝ), (9 / 5 : ℝ), ?_, ?_, ?_⟩ <;> norm_num [abs_lt])
  -- certificate: set-closeness-not-trans.sage

/-- The smallest equivalence relation containing `|x - y| < ε` (`ε > 0`) relates every pair of real numbers: one
class, all of `ℝ` (the line is connected). -/
theorem closeness_eqvGen (ε : ℝ) (hε : 0 < ε) (x y : ℝ) : Relation.EqvGen (fun a b : ℝ => |a - b| < ε) x y := by
  set S := {z : ℝ | Relation.EqvGen (fun a b : ℝ => |a - b| < ε) x z}
  have hopen : IsOpen S := by
    rw [Metric.isOpen_iff]
    intro z hz
    refine ⟨ε, hε, fun w hw => ?_⟩
    exact Relation.EqvGen.trans _ _ _ hz (Relation.EqvGen.rel _ _ (by rw [abs_sub_comm]; exact hw))
  have hclosed : IsClosed S := by
    rw [← isOpen_compl_iff, Metric.isOpen_iff]
    intro z hz
    refine ⟨ε, hε, fun w hw hwS => hz ?_⟩
    exact Relation.EqvGen.trans _ _ _ hwS (Relation.EqvGen.rel _ _ hw)
  have hne : S.Nonempty := ⟨x, Relation.EqvGen.refl x⟩
  have := IsClopen.eq_univ ⟨hclosed, hopen⟩ hne
  have hy : y ∈ S := by rw [this]; trivial
  exact hy

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.closeness_eqvGen (1) ((by norm_num)) (0) (1.8) := by
  have key : ∀ (ε : ℝ) (hε : 0 < ε) (x y : ℝ), Relation.EqvGen (fun a b : ℝ => |a - b| < ε) x y := by
    intro ε hε x y
    set S := {z : ℝ | Relation.EqvGen (fun a b : ℝ => |a - b| < ε) x z}
    have hopen : IsOpen S := by
      rw [Metric.isOpen_iff]
      intro z hz
      refine ⟨ε, hε, fun w hw => ?_⟩
      exact Relation.EqvGen.trans _ _ _ hz (Relation.EqvGen.rel _ _ (by rw [abs_sub_comm]; exact hw))
    have hclosed : IsClosed S := by
      rw [← isOpen_compl_iff, Metric.isOpen_iff]
      intro z hz
      refine ⟨ε, hε, fun w hw hwS => hz ?_⟩
      exact Relation.EqvGen.trans _ _ _ hwS (Relation.EqvGen.rel _ _ hw)
    have hne : S.Nonempty := ⟨x, Relation.EqvGen.refl x⟩
    have := IsClopen.eq_univ ⟨hclosed, hopen⟩ hne
    have hy : y ∈ S := by rw [this]; trivial
    exact hy
  exact @key (1) ((by norm_num)) (0) (1.8)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.closeness_eqvGen (1/2) ((by norm_num)) (-3) (7) := by
  have key : ∀ (ε : ℝ) (hε : 0 < ε) (x y : ℝ), Relation.EqvGen (fun a b : ℝ => |a - b| < ε) x y := by
    intro ε hε x y
    set S := {z : ℝ | Relation.EqvGen (fun a b : ℝ => |a - b| < ε) x z}
    have hopen : IsOpen S := by
      rw [Metric.isOpen_iff]
      intro z hz
      refine ⟨ε, hε, fun w hw => ?_⟩
      exact Relation.EqvGen.trans _ _ _ hz (Relation.EqvGen.rel _ _ (by rw [abs_sub_comm]; exact hw))
    have hclosed : IsClosed S := by
      rw [← isOpen_compl_iff, Metric.isOpen_iff]
      intro z hz
      refine ⟨ε, hε, fun w hw hwS => hz ?_⟩
      exact Relation.EqvGen.trans _ _ _ hwS (Relation.EqvGen.rel _ _ hw)
    have hne : S.Nonempty := ⟨x, Relation.EqvGen.refl x⟩
    have := IsClopen.eq_univ ⟨hclosed, hopen⟩ hne
    have hy : y ∈ S := by rw [this]; trivial
    exact hy
  exact @key (1/2) ((by norm_num)) (-3) (7)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.closeness_eqvGen (3) ((by norm_num)) (0) (100) := by
  have key : ∀ (ε : ℝ) (hε : 0 < ε) (x y : ℝ), Relation.EqvGen (fun a b : ℝ => |a - b| < ε) x y := by
    intro ε hε x y
    set S := {z : ℝ | Relation.EqvGen (fun a b : ℝ => |a - b| < ε) x z}
    have hopen : IsOpen S := by
      rw [Metric.isOpen_iff]
      intro z hz
      refine ⟨ε, hε, fun w hw => ?_⟩
      exact Relation.EqvGen.trans _ _ _ hz (Relation.EqvGen.rel _ _ (by rw [abs_sub_comm]; exact hw))
    have hclosed : IsClosed S := by
      rw [← isOpen_compl_iff, Metric.isOpen_iff]
      intro z hz
      refine ⟨ε, hε, fun w hw hwS => hz ?_⟩
      exact Relation.EqvGen.trans _ _ _ hwS (Relation.EqvGen.rel _ _ hw)
    have hne : S.Nonempty := ⟨x, Relation.EqvGen.refl x⟩
    have := IsClopen.eq_univ ⟨hclosed, hopen⟩ hne
    have hy : y ∈ S := by rw [this]; trivial
    exact hy
  exact @key (3) ((by norm_num)) (0) (100)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.closeness_eqvGen
-- test: set-closeness-eqvgen-test.sage

/-- On `(a, a + 1) ∪ (b, b + 1)` with a gap `b > a + 2`, the chains of closeness steps cannot cross the gap: two
points are related by the generated equivalence relation exactly when they lie in the same interval (two classes; for
`(0, 1) ∪ (5, 6)`, `a = 0`, `b = 5`). -/
theorem closeness_two_pieces (a b : ℝ) (hab : a + 2 < b) (x y : ℝ) (hx : x ∈ Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1))
    (hy : y ∈ Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)) :
    Relation.EqvGen (fun p q : ↥(Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)) => |(p : ℝ) - (q : ℝ)| < 1)
      ⟨x, hx⟩ ⟨y, hy⟩ ↔
      (x < a + 1 ↔ y < a + 1) := by
  have key : ∀ p q : ↥(Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)),
      Relation.EqvGen (fun p q : ↥(Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)) => |(p : ℝ) - (q : ℝ)| < 1) p q →
        ((p : ℝ) < a + 1 ↔ (q : ℝ) < a + 1) := by
    intro p q hpq
    induction hpq with
    | rel p q hpq =>
      obtain ⟨p, hp⟩ := p
      obtain ⟨q, hq⟩ := q
      simp only at hpq ⊢
      rw [abs_lt] at hpq
      rcases hp with hp | hp <;> rcases hq with hq | hq <;>
        simp only [Set.mem_Ioo] at hp hq <;> constructor <;> intro <;> linarith
    | refl => exact Iff.rfl
    | symm _ _ _ ih => exact ih.symm
    | trans _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
  constructor
  · intro h
    exact key _ _ h
  · intro h
    apply Relation.EqvGen.rel
    simp only
    rw [abs_lt]
    rcases hx with hx | hx <;> rcases hy with hy | hy <;> simp only [Set.mem_Ioo] at hx hy
    · constructor <;> linarith
    · exact absurd (h.1 hx.2) (by linarith)
    · exact absurd (h.2 hy.2) (by linarith)
    · constructor <;> linarith

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.closeness_two_pieces (0) (5) ((by norm_num)) (1/2) (11/2) ((Or.inl (by norm_num))) ((Or.inr (by norm_num))) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (a b : ℝ) (hab : a + 2 < b) (x y : ℝ) (hx : x ∈ Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)) (hy : y ∈ Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)), Relation.EqvGen (fun p q : ↥(Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)) => |(p : ℝ) - (q : ℝ)| < 1) ⟨x, hx⟩ ⟨y, hy⟩ ↔ (x < a + 1 ↔ y < a + 1) := by
    intro a b hab x y hx hy
    have key : ∀ p q : ↥(Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)),
        Relation.EqvGen (fun p q : ↥(Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)) => |(p : ℝ) - (q : ℝ)| < 1) p q →
          ((p : ℝ) < a + 1 ↔ (q : ℝ) < a + 1) := by
      intro p q hpq
      induction hpq with
      | rel p q hpq =>
        obtain ⟨p, hp⟩ := p
        obtain ⟨q, hq⟩ := q
        simp only at hpq ⊢
        rw [abs_lt] at hpq
        rcases hp with hp | hp <;> rcases hq with hq | hq <;>
          simp only [Set.mem_Ioo] at hp hq <;> constructor <;> intro <;> linarith
      | refl => exact Iff.rfl
      | symm _ _ _ ih => exact ih.symm
      | trans _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
    constructor
    · intro h
      exact key _ _ h
    · intro h
      apply Relation.EqvGen.rel
      simp only
      rw [abs_lt]
      rcases hx with hx | hx <;> rcases hy with hy | hy <;> simp only [Set.mem_Ioo] at hx hy
      · constructor <;> linarith
      · exact absurd (h.1 hx.2) (by linarith)
      · exact absurd (h.2 hy.2) (by linarith)
      · constructor <;> linarith
  exact @key (0) (5) ((by norm_num)) (1/2) (11/2) ((Or.inl (by norm_num))) ((Or.inr (by norm_num)))

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.closeness_two_pieces (0) (5) ((by norm_num)) (1/4) (3/4) ((Or.inl (by norm_num))) ((Or.inl (by norm_num))) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (a b : ℝ) (hab : a + 2 < b) (x y : ℝ) (hx : x ∈ Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)) (hy : y ∈ Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)), Relation.EqvGen (fun p q : ↥(Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)) => |(p : ℝ) - (q : ℝ)| < 1) ⟨x, hx⟩ ⟨y, hy⟩ ↔ (x < a + 1 ↔ y < a + 1) := by
    intro a b hab x y hx hy
    have key : ∀ p q : ↥(Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)),
        Relation.EqvGen (fun p q : ↥(Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)) => |(p : ℝ) - (q : ℝ)| < 1) p q →
          ((p : ℝ) < a + 1 ↔ (q : ℝ) < a + 1) := by
      intro p q hpq
      induction hpq with
      | rel p q hpq =>
        obtain ⟨p, hp⟩ := p
        obtain ⟨q, hq⟩ := q
        simp only at hpq ⊢
        rw [abs_lt] at hpq
        rcases hp with hp | hp <;> rcases hq with hq | hq <;>
          simp only [Set.mem_Ioo] at hp hq <;> constructor <;> intro <;> linarith
      | refl => exact Iff.rfl
      | symm _ _ _ ih => exact ih.symm
      | trans _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
    constructor
    · intro h
      exact key _ _ h
    · intro h
      apply Relation.EqvGen.rel
      simp only
      rw [abs_lt]
      rcases hx with hx | hx <;> rcases hy with hy | hy <;> simp only [Set.mem_Ioo] at hx hy
      · constructor <;> linarith
      · exact absurd (h.1 hx.2) (by linarith)
      · exact absurd (h.2 hy.2) (by linarith)
      · constructor <;> linarith
  exact @key (0) (5) ((by norm_num)) (1/4) (3/4) ((Or.inl (by norm_num))) ((Or.inl (by norm_num)))

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.closeness_two_pieces (-1) (3) ((by norm_num)) (7/2) (13/4) ((Or.inr (by norm_num))) ((Or.inr (by norm_num))) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (a b : ℝ) (hab : a + 2 < b) (x y : ℝ) (hx : x ∈ Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)) (hy : y ∈ Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)), Relation.EqvGen (fun p q : ↥(Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)) => |(p : ℝ) - (q : ℝ)| < 1) ⟨x, hx⟩ ⟨y, hy⟩ ↔ (x < a + 1 ↔ y < a + 1) := by
    intro a b hab x y hx hy
    have key : ∀ p q : ↥(Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)),
        Relation.EqvGen (fun p q : ↥(Set.Ioo a (a + 1) ∪ Set.Ioo b (b + 1)) => |(p : ℝ) - (q : ℝ)| < 1) p q →
          ((p : ℝ) < a + 1 ↔ (q : ℝ) < a + 1) := by
      intro p q hpq
      induction hpq with
      | rel p q hpq =>
        obtain ⟨p, hp⟩ := p
        obtain ⟨q, hq⟩ := q
        simp only at hpq ⊢
        rw [abs_lt] at hpq
        rcases hp with hp | hp <;> rcases hq with hq | hq <;>
          simp only [Set.mem_Ioo] at hp hq <;> constructor <;> intro <;> linarith
      | refl => exact Iff.rfl
      | symm _ _ _ ih => exact ih.symm
      | trans _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
    constructor
    · intro h
      exact key _ _ h
    · intro h
      apply Relation.EqvGen.rel
      simp only
      rw [abs_lt]
      rcases hx with hx | hx <;> rcases hy with hy | hy <;> simp only [Set.mem_Ioo] at hx hy
      · constructor <;> linarith
      · exact absurd (h.1 hx.2) (by linarith)
      · exact absurd (h.2 hy.2) (by linarith)
      · constructor <;> linarith
  exact @key (-1) (3) ((by norm_num)) (7/2) (13/4) ((Or.inr (by norm_num))) ((Or.inr (by norm_num)))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.closeness_two_pieces
-- test: set-closeness-pieces-test.sage

/-! ## def-equivalence-class -/

/-- The classes of an equivalence relation partition the set: two classes are equal or disjoint, and every element
lies in its own class, hence in exactly one. -/
theorem classes_partition {X : Type} (s : Setoid X) (x y : X) :
    ({z | s z x} = {z | s z y} ∨ Disjoint {z | s z x} {z | s z y}) ∧ x ∈ {z | s z x} ∧
      (∀ w, x ∈ {z | s z w} → {z | s z w} = {z | s z x}) := by
  refine ⟨?_, s.refl x, fun w hw => ?_⟩
  · by_cases h : s x y
    · left; ext z; exact ⟨fun hz => s.trans hz h, fun hz => s.trans hz (s.symm h)⟩
    · right
      rw [Set.disjoint_left]
      intro z hzx hzy
      exact h (s.trans (s.symm hzx) hzy)
  · ext z; exact ⟨fun hz => s.trans hz (s.symm hw), fun hz => s.trans hz hw⟩

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.classes_partition (ℤ) (⟨fun a b => (3 : ℤ) ∣ a - b, KaluzaKlein.Sets.Equivalence.modEq_equivalence 3⟩) (1) (4) := by
  have key : ∀ {X : Type} (s : Setoid X) (x y : X), ({z | s z x} = {z | s z y} ∨ Disjoint {z | s z x} {z | s z y}) ∧ x ∈ {z | s z x} ∧ (∀ w, x ∈ {z | s z w} → {z | s z w} = {z | s z x}) := by
    intro X s x y
    refine ⟨?_, s.refl x, fun w hw => ?_⟩
    · by_cases h : s x y
      · left; ext z; exact ⟨fun hz => s.trans hz h, fun hz => s.trans hz (s.symm h)⟩
      · right
        rw [Set.disjoint_left]
        intro z hzx hzy
        exact h (s.trans (s.symm hzx) hzy)
    · ext z; exact ⟨fun hz => s.trans hz (s.symm hw), fun hz => s.trans hz hw⟩
  exact @key (ℤ) (⟨fun a b => (3 : ℤ) ∣ a - b, KaluzaKlein.Sets.Equivalence.modEq_equivalence 3⟩) (1) (4)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.classes_partition (Unit) (⊥) (()) (()) := by
  have key : ∀ {X : Type} (s : Setoid X) (x y : X), ({z | s z x} = {z | s z y} ∨ Disjoint {z | s z x} {z | s z y}) ∧ x ∈ {z | s z x} ∧ (∀ w, x ∈ {z | s z w} → {z | s z w} = {z | s z x}) := by
    intro X s x y
    refine ⟨?_, s.refl x, fun w hw => ?_⟩
    · by_cases h : s x y
      · left; ext z; exact ⟨fun hz => s.trans hz h, fun hz => s.trans hz (s.symm h)⟩
      · right
        rw [Set.disjoint_left]
        intro z hzx hzy
        exact h (s.trans (s.symm hzx) hzy)
    · ext z; exact ⟨fun hz => s.trans hz (s.symm hw), fun hz => s.trans hz hw⟩
  exact @key (Unit) (⊥) (()) (())

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.classes_partition (ℤ) (⟨fun a b => (3 : ℤ) ∣ a - b, KaluzaKlein.Sets.Equivalence.modEq_equivalence 3⟩) (0) (2) := by
  have key : ∀ {X : Type} (s : Setoid X) (x y : X), ({z | s z x} = {z | s z y} ∨ Disjoint {z | s z x} {z | s z y}) ∧ x ∈ {z | s z x} ∧ (∀ w, x ∈ {z | s z w} → {z | s z w} = {z | s z x}) := by
    intro X s x y
    refine ⟨?_, s.refl x, fun w hw => ?_⟩
    · by_cases h : s x y
      · left; ext z; exact ⟨fun hz => s.trans hz h, fun hz => s.trans hz (s.symm h)⟩
      · right
        rw [Set.disjoint_left]
        intro z hzx hzy
        exact h (s.trans (s.symm hzx) hzy)
    · ext z; exact ⟨fun hz => s.trans hz (s.symm hw), fun hz => s.trans hz hw⟩
  exact @key (ℤ) (⟨fun a b => (3 : ℤ) ∣ a - b, KaluzaKlein.Sets.Equivalence.modEq_equivalence 3⟩) (0) (2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.classes_partition
-- test: set-classes-partition-test.sage

/-! ## exm-class-mod -/

/-- In `{0, …, nm - 1}` the class of the remainder `r < n` modulo `n` is `{r, r + n, …, r + (m - 1) n}`: modulo `3`
in `{0, …, 11}`, `{0, 3, 6, 9}`, `{1, 4, 7, 10}`, `{2, 5, 8, 11}`. -/
theorem class_mod (n m r : ℕ) (hr : r < n) :
    (Finset.range (n * m)).filter (fun k => k % n = r) = (Finset.range m).image (fun j => r + n * j) := by
  ext k
  simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_image]
  constructor
  · rintro ⟨hk, rfl⟩
    refine ⟨k / n, ?_, Nat.mod_add_div k n⟩
    rw [Nat.div_lt_iff_lt_mul (by omega)]; linarith [Nat.mul_comm n m]
  · rintro ⟨j, hj, rfl⟩
    refine ⟨?_, ?_⟩
    · nlinarith
    · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hr]

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_mod (3) (4) (0) ((by norm_num)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_mod (3) (4) (1) ((by norm_num)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_mod (3) (4) (2) ((by norm_num)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_mod (1) (0) (0) ((by norm_num)) := by
  decide

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.Sets.Equivalence.class_mod
-- test: set-class-mod-test.sage

/-! ## exm-class-angle -/

/-- Every interval `[a, a + T)` of length `T > 0` contains exactly one member of the class `{φ + kT}`: the class of an
angle has one angle in every interval of length `2π`. -/
theorem class_angle_unique (T a φ : ℝ) (hT : 0 < T) :
    ∃! ψ, ψ ∈ Set.Ico a (a + T) ∧ ∃ k : ℤ, ψ = φ + k * T := by
  refine ⟨toIcoMod hT a φ, ⟨toIcoMod_mem_Ico hT a φ, ⟨-toIcoDiv hT a φ, ?_⟩⟩, ?_⟩
  · rw [toIcoMod]; push_cast; rw [zsmul_eq_mul]; ring
  · rintro ψ ⟨hψ, k, rfl⟩
    exact ((toIcoMod_eq_iff hT).2 ⟨hψ, -k, by rw [zsmul_eq_mul]; push_cast; ring⟩).symm

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_angle_unique (2 * Real.pi) (0) (7) ((by positivity)) := by
  have key : ∀ (T a φ : ℝ) (hT : 0 < T), ∃! ψ, ψ ∈ Set.Ico a (a + T) ∧ ∃ k : ℤ, ψ = φ + k * T := by
    intro T a φ hT
    refine ⟨toIcoMod hT a φ, ⟨toIcoMod_mem_Ico hT a φ, ⟨-toIcoDiv hT a φ, ?_⟩⟩, ?_⟩
    · rw [toIcoMod]; push_cast; rw [zsmul_eq_mul]; ring
    · rintro ψ ⟨hψ, k, rfl⟩
      exact ((toIcoMod_eq_iff hT).2 ⟨hψ, -k, by rw [zsmul_eq_mul]; push_cast; ring⟩).symm
  exact @key (2 * Real.pi) (0) (7) ((by positivity))

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_angle_unique (1) (0) (0) ((by norm_num)) := by
  have key : ∀ (T a φ : ℝ) (hT : 0 < T), ∃! ψ, ψ ∈ Set.Ico a (a + T) ∧ ∃ k : ℤ, ψ = φ + k * T := by
    intro T a φ hT
    refine ⟨toIcoMod hT a φ, ⟨toIcoMod_mem_Ico hT a φ, ⟨-toIcoDiv hT a φ, ?_⟩⟩, ?_⟩
    · rw [toIcoMod]; push_cast; rw [zsmul_eq_mul]; ring
    · rintro ψ ⟨hψ, k, rfl⟩
      exact ((toIcoMod_eq_iff hT).2 ⟨hψ, -k, by rw [zsmul_eq_mul]; push_cast; ring⟩).symm
  exact @key (1) (0) (0) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_angle_unique (2 * Real.pi) (-Real.pi) (1) ((by positivity)) := by
  have key : ∀ (T a φ : ℝ) (hT : 0 < T), ∃! ψ, ψ ∈ Set.Ico a (a + T) ∧ ∃ k : ℤ, ψ = φ + k * T := by
    intro T a φ hT
    refine ⟨toIcoMod hT a φ, ⟨toIcoMod_mem_Ico hT a φ, ⟨-toIcoDiv hT a φ, ?_⟩⟩, ?_⟩
    · rw [toIcoMod]; push_cast; rw [zsmul_eq_mul]; ring
    · rintro ψ ⟨hψ, k, rfl⟩
      exact ((toIcoMod_eq_iff hT).2 ⟨hψ, -k, by rw [zsmul_eq_mul]; push_cast; ring⟩).symm
  exact @key (2 * Real.pi) (-Real.pi) (1) ((by positivity))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.class_angle_unique
-- test: set-class-angle-test.sage

/-! ## exr-class-square -/

/-- The class of `x` under "same square" is `{x, -x}`, two elements for `x ≠ 0` and one for `x = 0`. -/
theorem class_square (x : ℝ) : {y : ℝ | y ^ 2 = x ^ 2} = {x, -x} ∧ (x ≠ 0 → x ≠ -x) := by
  refine ⟨?_, fun hx h => hx (by linarith)⟩
  ext y
  simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]
  constructor
  · intro h
    have : (y - x) * (y + x) = 0 := by ring_nf; linarith
    rcases mul_eq_zero.1 this with h1 | h1
    · left; linarith
    · right; linarith
  · rintro (rfl | rfl) <;> ring

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_square (0) := by
  have key : ∀ (x : ℝ), {y : ℝ | y ^ 2 = x ^ 2} = {x, -x} ∧ (x ≠ 0 → x ≠ -x) := by
    intro x
    refine ⟨?_, fun hx h => hx (by linarith)⟩
    ext y
    simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]
    constructor
    · intro h
      have : (y - x) * (y + x) = 0 := by ring_nf; linarith
      rcases mul_eq_zero.1 this with h1 | h1
      · left; linarith
      · right; linarith
    · rintro (rfl | rfl) <;> ring
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_square (2) := by
  have key : ∀ (x : ℝ), {y : ℝ | y ^ 2 = x ^ 2} = {x, -x} ∧ (x ≠ 0 → x ≠ -x) := by
    intro x
    refine ⟨?_, fun hx h => hx (by linarith)⟩
    ext y
    simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]
    constructor
    · intro h
      have : (y - x) * (y + x) = 0 := by ring_nf; linarith
      rcases mul_eq_zero.1 this with h1 | h1
      · left; linarith
      · right; linarith
    · rintro (rfl | rfl) <;> ring
  exact @key (2)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_square (-3) := by
  have key : ∀ (x : ℝ), {y : ℝ | y ^ 2 = x ^ 2} = {x, -x} ∧ (x ≠ 0 → x ≠ -x) := by
    intro x
    refine ⟨?_, fun hx h => hx (by linarith)⟩
    ext y
    simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]
    constructor
    · intro h
      have : (y - x) * (y + x) = 0 := by ring_nf; linarith
      rcases mul_eq_zero.1 this with h1 | h1
      · left; linarith
      · right; linarith
    · rintro (rfl | rfl) <;> ring
  exact @key (-3)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.class_square
-- test: set-class-square-test.sage

/-- The class of `x` under "same cosine" is `{±x + 2πk}`. -/
theorem class_cos (x : ℝ) : {y : ℝ | Real.cos y = Real.cos x} = {y | ∃ k : ℤ, y = x + k * (2 * Real.pi) ∨
    y = -x + k * (2 * Real.pi)} := by
  ext y
  simp only [Set.mem_setOf_eq, Real.cos_eq_cos_iff]
  constructor
  · rintro ⟨k, h | h⟩
    · exact ⟨-k, Or.inl (by push_cast; linarith)⟩
    · exact ⟨k, Or.inr (by linarith)⟩
  · rintro ⟨k, h | h⟩
    · exact ⟨-k, Or.inl (by push_cast; linarith)⟩
    · exact ⟨k, Or.inr (by linarith)⟩

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_cos (0) := by
  have key : ∀ (x : ℝ), {y : ℝ | Real.cos y = Real.cos x} = {y | ∃ k : ℤ, y = x + k * (2 * Real.pi) ∨ y = -x + k * (2 * Real.pi)} := by
    intro x
    ext y
    simp only [Set.mem_setOf_eq, Real.cos_eq_cos_iff]
    constructor
    · rintro ⟨k, h | h⟩
      · exact ⟨-k, Or.inl (by push_cast; linarith)⟩
      · exact ⟨k, Or.inr (by linarith)⟩
    · rintro ⟨k, h | h⟩
      · exact ⟨-k, Or.inl (by push_cast; linarith)⟩
      · exact ⟨k, Or.inr (by linarith)⟩
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_cos (1) := by
  have key : ∀ (x : ℝ), {y : ℝ | Real.cos y = Real.cos x} = {y | ∃ k : ℤ, y = x + k * (2 * Real.pi) ∨ y = -x + k * (2 * Real.pi)} := by
    intro x
    ext y
    simp only [Set.mem_setOf_eq, Real.cos_eq_cos_iff]
    constructor
    · rintro ⟨k, h | h⟩
      · exact ⟨-k, Or.inl (by push_cast; linarith)⟩
      · exact ⟨k, Or.inr (by linarith)⟩
    · rintro ⟨k, h | h⟩
      · exact ⟨-k, Or.inl (by push_cast; linarith)⟩
      · exact ⟨k, Or.inr (by linarith)⟩
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.class_cos (Real.pi) := by
  have key : ∀ (x : ℝ), {y : ℝ | Real.cos y = Real.cos x} = {y | ∃ k : ℤ, y = x + k * (2 * Real.pi) ∨ y = -x + k * (2 * Real.pi)} := by
    intro x
    ext y
    simp only [Set.mem_setOf_eq, Real.cos_eq_cos_iff]
    constructor
    · rintro ⟨k, h | h⟩
      · exact ⟨-k, Or.inl (by push_cast; linarith)⟩
      · exact ⟨k, Or.inr (by linarith)⟩
    · rintro ⟨k, h | h⟩
      · exact ⟨-k, Or.inl (by push_cast; linarith)⟩
      · exact ⟨k, Or.inr (by linarith)⟩
  exact @key (Real.pi)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.class_cos
-- test: set-class-cos-test.sage

/-- `x` and `-x` lie in the same class modulo `2π` exactly when `x ∈ πℤ`: so a class of "same cosine" has two
elements per period except for `x ∈ πℤ`. -/
theorem neg_congruent_iff (x : ℝ) : (∃ k : ℤ, -x = x + k * (2 * Real.pi)) ↔ ∃ m : ℤ, x = m * Real.pi := by
  constructor
  · rintro ⟨k, hk⟩; exact ⟨-k, by push_cast; linarith⟩
  · rintro ⟨m, rfl⟩; exact ⟨-m, by push_cast; ring⟩

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.neg_congruent_iff (0) := by
  have key : ∀ (x : ℝ), (∃ k : ℤ, -x = x + k * (2 * Real.pi)) ↔ ∃ m : ℤ, x = m * Real.pi := by
    intro x
    constructor
    · rintro ⟨k, hk⟩; exact ⟨-k, by push_cast; linarith⟩
    · rintro ⟨m, rfl⟩; exact ⟨-m, by push_cast; ring⟩
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.neg_congruent_iff (Real.pi) := by
  have key : ∀ (x : ℝ), (∃ k : ℤ, -x = x + k * (2 * Real.pi)) ↔ ∃ m : ℤ, x = m * Real.pi := by
    intro x
    constructor
    · rintro ⟨k, hk⟩; exact ⟨-k, by push_cast; linarith⟩
    · rintro ⟨m, rfl⟩; exact ⟨-m, by push_cast; ring⟩
  exact @key (Real.pi)

example : veriphy_instance% KaluzaKlein.Sets.Equivalence.neg_congruent_iff (1) := by
  have key : ∀ (x : ℝ), (∃ k : ℤ, -x = x + k * (2 * Real.pi)) ↔ ∃ m : ℤ, x = m * Real.pi := by
    intro x
    constructor
    · rintro ⟨k, hk⟩; exact ⟨-k, by push_cast; linarith⟩
    · rintro ⟨m, rfl⟩; exact ⟨-m, by push_cast; ring⟩
  exact @key (1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Equivalence.neg_congruent_iff
-- test: set-neg-congruent-test.sage

end KaluzaKlein.Sets.Equivalence
