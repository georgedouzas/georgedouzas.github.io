import Mathlib
import KaluzaKlein.PartI.Sets.Equivalence
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Equivalence relations and quotients" (quotient sets)

The claims of the blocks `def-quotient-set` to `exr-quotient-square` of
`tutorial/part-i-01-sets-relations-topology.qmd`: the quotient map and descent of maps, `ℤ/nℤ` and its addition, the
circle as `ℝ/2πℤ`, the sphere as the quotient of `SO(3)` by "same image of the pole", the functions that descend to
the circle, and the square with its opposite edges glued.
-/

namespace KaluzaKlein.Sets.Quotients

open Matrix KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Maps

/-! ## def-quotient-set -/

/-- The quotient map `x ↦ [x]` is surjective, and a map constant on classes descends: `f = f̄ ∘ π`. -/
theorem quotient_descends {X Y : Type} (s : Setoid X) (f : X → Y) (hf : ∀ a b, s a b → f a = f b) :
    Function.Surjective (Quotient.mk s) ∧ (fun x => Quotient.lift f hf (Quotient.mk s x)) = f :=
  ⟨Quotient.mk_surjective, rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.Quotients.quotient_descends (ℤ) (ℤ) (⟨fun a b => (3 : ℤ) ∣ a - b, KaluzaKlein.Sets.Equivalence.modEq_equivalence 3⟩) (fun a => a % 3) ((fun a b h => by have h' : (3 : ℤ) ∣ a - b := h; show a % 3 = b % 3; omega)) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ {X Y : Type} (s : Setoid X) (f : X → Y) (hf : ∀ a b, s a b → f a = f b), Function.Surjective (Quotient.mk s) ∧ (fun x => Quotient.lift f hf (Quotient.mk s x)) = f := by
    intro X Y s f hf
    exact (⟨Quotient.mk_surjective, rfl⟩)
  exact @key (ℤ) (ℤ) (⟨fun a b => (3 : ℤ) ∣ a - b, KaluzaKlein.Sets.Equivalence.modEq_equivalence 3⟩) (fun a => a % 3) ((fun a b h => by have h' : (3 : ℤ) ∣ a - b := h; show a % 3 = b % 3; omega))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.quotient_descends (Unit) (Unit) (⊥) (id) ((fun a b _ => rfl)) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ {X Y : Type} (s : Setoid X) (f : X → Y) (hf : ∀ a b, s a b → f a = f b), Function.Surjective (Quotient.mk s) ∧ (fun x => Quotient.lift f hf (Quotient.mk s x)) = f := by
    intro X Y s f hf
    exact (⟨Quotient.mk_surjective, rfl⟩)
  exact @key (Unit) (Unit) (⊥) (id) ((fun a b _ => rfl))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.quotient_descends (ℝ) (ℝ) (⟨fun a b : ℝ => ∃ k : ℤ, b - a = k * (2 * Real.pi), KaluzaKlein.Sets.Equivalence.angle_equivalence _⟩) (Real.cos) ((fun a b ⟨k, hk⟩ => by rw [show b = a + k * (2 * Real.pi) by linarith, Real.cos_add_int_mul_two_pi])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ {X Y : Type} (s : Setoid X) (f : X → Y) (hf : ∀ a b, s a b → f a = f b), Function.Surjective (Quotient.mk s) ∧ (fun x => Quotient.lift f hf (Quotient.mk s x)) = f := by
    intro X Y s f hf
    exact (⟨Quotient.mk_surjective, rfl⟩)
  exact @key (ℝ) (ℝ) (⟨fun a b : ℝ => ∃ k : ℤ, b - a = k * (2 * Real.pi), KaluzaKlein.Sets.Equivalence.angle_equivalence _⟩) (Real.cos) ((fun a b ⟨k, hk⟩ => by rw [show b = a + k * (2 * Real.pi) by linarith, Real.cos_add_int_mul_two_pi]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Quotients.quotient_descends
-- test: set-quotient-descends-test.sage

/-! ## exm-quotient-zn -/

/-- Addition descends to the remainders: if `a ≡ a'` and `b ≡ b'` modulo `n`, then `a + b ≡ a' + b'`. -/
theorem add_descends (n a a' b b' : ℤ) (ha : a ≡ a' [ZMOD n]) (hb : b ≡ b' [ZMOD n]) : a + b ≡ a' + b' [ZMOD n] :=
  Int.ModEq.add ha hb

example : veriphy_instance% KaluzaKlein.Sets.Quotients.add_descends (5) (7) (2) (3) (8) ((by decide)) ((by decide)) := by
  have key : ∀ (n a a' b b' : ℤ) (ha : a ≡ a' [ZMOD n]) (hb : b ≡ b' [ZMOD n]), a + b ≡ a' + b' [ZMOD n] := by
    intro n a a' b b' ha hb
    exact (Int.ModEq.add ha hb)
  exact @key (5) (7) (2) (3) (8) ((by decide)) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.add_descends (0) (1) (1) (2) (2) ((by decide)) ((by decide)) := by
  have key : ∀ (n a a' b b' : ℤ) (ha : a ≡ a' [ZMOD n]) (hb : b ≡ b' [ZMOD n]), a + b ≡ a' + b' [ZMOD n] := by
    intro n a a' b b' ha hb
    exact (Int.ModEq.add ha hb)
  exact @key (0) (1) (1) (2) (2) ((by decide)) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.add_descends (3) (-1) (2) (4) (1) ((by decide)) ((by decide)) := by
  have key : ∀ (n a a' b b' : ℤ) (ha : a ≡ a' [ZMOD n]) (hb : b ≡ b' [ZMOD n]), a + b ≡ a' + b' [ZMOD n] := by
    intro n a a' b b' ha hb
    exact (Int.ModEq.add ha hb)
  exact @key (3) (-1) (2) (4) (1) ((by decide)) ((by decide))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Quotients.add_descends
-- test: set-add-descends-test.sage

/-- `ℤ/nℤ` has `n` elements (`n ≥ 1`): three for `n = 3`, five for `n = 5`. -/
theorem card_zmod (n : ℕ) [NeZero n] : Fintype.card (ZMod n) = n :=
  ZMod.card n

example : veriphy_instance% KaluzaKlein.Sets.Quotients.card_zmod (3) (inferInstance) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Quotients.card_zmod (5) (inferInstance) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Quotients.card_zmod (1) (inferInstance) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Quotients.card_zmod
-- test: set-card-zmod-test.sage

/-! ## exm-quotient-circle -/

/-- The relation `φ ∼ φ'` iff `φ' - φ ∈ Tℤ` (the angles for `T = 2π`). -/
def angleSetoid (T : ℝ) : Setoid ℝ where
  r x y := ∃ k : ℤ, y - x = k * T
  iseqv := Equivalence.angle_equivalence T

/-- `[φ] ↦ (cos(2πφ/T), sin(2πφ/T))` is a bijection from `ℝ/Tℤ` (`T > 0`) onto the circle: for `T = 2π`,
`[φ] ↦ (cos φ, sin φ)` identifies `ℝ/2πℤ` with the circle. -/
theorem circle_quotient_bijective (T : ℝ) (hT : 0 < T) :
    ∃ F : Quotient (angleSetoid T) → circle, Function.Bijective F ∧
      ∀ φ, (F ⟦φ⟧ : ℝ × ℝ) = wrap (2 * Real.pi * φ / T) := by
  have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
  have hpi := Real.pi_pos
  have key : ∀ a b : ℝ, wrap (2 * Real.pi * a / T) = wrap (2 * Real.pi * b / T) ↔ ∃ k : ℤ, b - a = k * T := by
    intro a b
    constructor
    · intro h
      have : 2 * Real.pi * b / T ∈ wrap ⁻¹' {wrap (2 * Real.pi * a / T)} := h.symm
      rw [wrap_preimage] at this
      obtain ⟨k, hk⟩ := this
      refine ⟨k, ?_⟩
      field_simp at hk
      nlinarith
    · rintro ⟨k, hk⟩
      have : 2 * Real.pi * b / T ∈ wrap ⁻¹' {wrap (2 * Real.pi * a / T)} := by
        rw [wrap_preimage]
        refine ⟨k, ?_⟩
        rw [show b = a + k * T by linarith]
        field_simp
        try ring
      exact this.symm
  refine ⟨Quotient.lift (fun φ => (⟨wrap (2 * Real.pi * φ / T), hw _⟩ : circle))
    (fun a b h => Subtype.ext ((key a b).2 h)), ⟨?_, ?_⟩, fun φ => rfl⟩
  · intro p q h
    induction p using Quotient.ind
    induction q using Quotient.ind
    exact Quotient.sound ((key _ _).1 (congrArg Subtype.val h))
  · rintro ⟨p, hp⟩
    have hp' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
      rw [range_wrap 1 zero_le_one]; simpa [circle] using hp
    obtain ⟨φ, hφ⟩ := hp'
    refine ⟨⟦φ * T / (2 * Real.pi)⟧, Subtype.ext ?_⟩
    simp only [Quotient.lift_mk]
    rw [← hφ]
    simp only [wrap, one_mul]
    congr 2 <;> field_simp

example : veriphy_instance% KaluzaKlein.Sets.Quotients.circle_quotient_bijective (2 * Real.pi) ((by positivity)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), ∃ F : Quotient (angleSetoid T) → circle, Function.Bijective F ∧ ∀ φ, (F ⟦φ⟧ : ℝ × ℝ) = wrap (2 * Real.pi * φ / T) := by
    intro T hT
    have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
    have hpi := Real.pi_pos
    have key : ∀ a b : ℝ, wrap (2 * Real.pi * a / T) = wrap (2 * Real.pi * b / T) ↔ ∃ k : ℤ, b - a = k * T := by
      intro a b
      constructor
      · intro h
        have : 2 * Real.pi * b / T ∈ wrap ⁻¹' {wrap (2 * Real.pi * a / T)} := h.symm
        rw [wrap_preimage] at this
        obtain ⟨k, hk⟩ := this
        refine ⟨k, ?_⟩
        field_simp at hk
        nlinarith
      · rintro ⟨k, hk⟩
        have : 2 * Real.pi * b / T ∈ wrap ⁻¹' {wrap (2 * Real.pi * a / T)} := by
          rw [wrap_preimage]
          refine ⟨k, ?_⟩
          rw [show b = a + k * T by linarith]
          field_simp
          try ring
        exact this.symm
    refine ⟨Quotient.lift (fun φ => (⟨wrap (2 * Real.pi * φ / T), hw _⟩ : circle))
      (fun a b h => Subtype.ext ((key a b).2 h)), ⟨?_, ?_⟩, fun φ => rfl⟩
    · intro p q h
      induction p using Quotient.ind
      induction q using Quotient.ind
      exact Quotient.sound ((key _ _).1 (congrArg Subtype.val h))
    · rintro ⟨p, hp⟩
      have hp' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hp
      obtain ⟨φ, hφ⟩ := hp'
      refine ⟨⟦φ * T / (2 * Real.pi)⟧, Subtype.ext ?_⟩
      simp only [Quotient.lift_mk]
      rw [← hφ]
      simp only [wrap, one_mul]
      congr 2 <;> field_simp
  exact @key (2 * Real.pi) ((by positivity))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.circle_quotient_bijective (1) ((by norm_num)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), ∃ F : Quotient (angleSetoid T) → circle, Function.Bijective F ∧ ∀ φ, (F ⟦φ⟧ : ℝ × ℝ) = wrap (2 * Real.pi * φ / T) := by
    intro T hT
    have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
    have hpi := Real.pi_pos
    have key : ∀ a b : ℝ, wrap (2 * Real.pi * a / T) = wrap (2 * Real.pi * b / T) ↔ ∃ k : ℤ, b - a = k * T := by
      intro a b
      constructor
      · intro h
        have : 2 * Real.pi * b / T ∈ wrap ⁻¹' {wrap (2 * Real.pi * a / T)} := h.symm
        rw [wrap_preimage] at this
        obtain ⟨k, hk⟩ := this
        refine ⟨k, ?_⟩
        field_simp at hk
        nlinarith
      · rintro ⟨k, hk⟩
        have : 2 * Real.pi * b / T ∈ wrap ⁻¹' {wrap (2 * Real.pi * a / T)} := by
          rw [wrap_preimage]
          refine ⟨k, ?_⟩
          rw [show b = a + k * T by linarith]
          field_simp
          try ring
        exact this.symm
    refine ⟨Quotient.lift (fun φ => (⟨wrap (2 * Real.pi * φ / T), hw _⟩ : circle))
      (fun a b h => Subtype.ext ((key a b).2 h)), ⟨?_, ?_⟩, fun φ => rfl⟩
    · intro p q h
      induction p using Quotient.ind
      induction q using Quotient.ind
      exact Quotient.sound ((key _ _).1 (congrArg Subtype.val h))
    · rintro ⟨p, hp⟩
      have hp' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hp
      obtain ⟨φ, hφ⟩ := hp'
      refine ⟨⟦φ * T / (2 * Real.pi)⟧, Subtype.ext ?_⟩
      simp only [Quotient.lift_mk]
      rw [← hφ]
      simp only [wrap, one_mul]
      congr 2 <;> field_simp
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.circle_quotient_bijective (3) ((by norm_num)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), ∃ F : Quotient (angleSetoid T) → circle, Function.Bijective F ∧ ∀ φ, (F ⟦φ⟧ : ℝ × ℝ) = wrap (2 * Real.pi * φ / T) := by
    intro T hT
    have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
    have hpi := Real.pi_pos
    have key : ∀ a b : ℝ, wrap (2 * Real.pi * a / T) = wrap (2 * Real.pi * b / T) ↔ ∃ k : ℤ, b - a = k * T := by
      intro a b
      constructor
      · intro h
        have : 2 * Real.pi * b / T ∈ wrap ⁻¹' {wrap (2 * Real.pi * a / T)} := h.symm
        rw [wrap_preimage] at this
        obtain ⟨k, hk⟩ := this
        refine ⟨k, ?_⟩
        field_simp at hk
        nlinarith
      · rintro ⟨k, hk⟩
        have : 2 * Real.pi * b / T ∈ wrap ⁻¹' {wrap (2 * Real.pi * a / T)} := by
          rw [wrap_preimage]
          refine ⟨k, ?_⟩
          rw [show b = a + k * T by linarith]
          field_simp
          try ring
        exact this.symm
    refine ⟨Quotient.lift (fun φ => (⟨wrap (2 * Real.pi * φ / T), hw _⟩ : circle))
      (fun a b h => Subtype.ext ((key a b).2 h)), ⟨?_, ?_⟩, fun φ => rfl⟩
    · intro p q h
      induction p using Quotient.ind
      induction q using Quotient.ind
      exact Quotient.sound ((key _ _).1 (congrArg Subtype.val h))
    · rintro ⟨p, hp⟩
      have hp' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hp
      obtain ⟨φ, hφ⟩ := hp'
      refine ⟨⟦φ * T / (2 * Real.pi)⟧, Subtype.ext ?_⟩
      simp only [Quotient.lift_mk]
      rw [← hφ]
      simp only [wrap, one_mul]
      congr 2 <;> field_simp
  exact @key (3) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Quotients.circle_quotient_bijective
-- test: set-circle-quotient-test.sage

/-- A function descends to `ℝ/Tℤ` (is constant on the classes of "differ by a multiple of `T`") exactly when it is
`T`-periodic: `cos` and `sin` descend to the circle. -/
theorem descends_iff_periodic {Y : Type} (T : ℝ) (f : ℝ → Y) :
    (∀ x y, (∃ k : ℤ, y - x = k * T) → f x = f y) ↔ Function.Periodic f T := by
  constructor
  · intro h x
    exact (h x (x + T) ⟨1, by simp⟩).symm
  · rintro h x y ⟨k, hk⟩
    rw [show y = x + k * T by linarith, ← zsmul_eq_mul]
    exact (h.zsmul k x).symm

example : veriphy_instance% KaluzaKlein.Sets.Quotients.descends_iff_periodic (ℝ) (2 * Real.pi) (Real.cos) := by
  have key : ∀ {Y : Type} (T : ℝ) (f : ℝ → Y), (∀ x y, (∃ k : ℤ, y - x = k * T) → f x = f y) ↔ Function.Periodic f T := by
    intro Y T f
    constructor
    · intro h x
      exact (h x (x + T) ⟨1, by simp⟩).symm
    · rintro h x y ⟨k, hk⟩
      rw [show y = x + k * T by linarith, ← zsmul_eq_mul]
      exact (h.zsmul k x).symm
  exact @key (ℝ) (2 * Real.pi) (Real.cos)

example : veriphy_instance% KaluzaKlein.Sets.Quotients.descends_iff_periodic (ℝ) (0) (id) := by
  have key : ∀ {Y : Type} (T : ℝ) (f : ℝ → Y), (∀ x y, (∃ k : ℤ, y - x = k * T) → f x = f y) ↔ Function.Periodic f T := by
    intro Y T f
    constructor
    · intro h x
      exact (h x (x + T) ⟨1, by simp⟩).symm
    · rintro h x y ⟨k, hk⟩
      rw [show y = x + k * T by linarith, ← zsmul_eq_mul]
      exact (h.zsmul k x).symm
  exact @key (ℝ) (0) (id)

example : veriphy_instance% KaluzaKlein.Sets.Quotients.descends_iff_periodic (ℝ) (1) (fun x => Real.sin (2 * Real.pi * x)) := by
  have key : ∀ {Y : Type} (T : ℝ) (f : ℝ → Y), (∀ x y, (∃ k : ℤ, y - x = k * T) → f x = f y) ↔ Function.Periodic f T := by
    intro Y T f
    constructor
    · intro h x
      exact (h x (x + T) ⟨1, by simp⟩).symm
    · rintro h x y ⟨k, hk⟩
      rw [show y = x + k * T by linarith, ← zsmul_eq_mul]
      exact (h.zsmul k x).symm
  exact @key (ℝ) (1) (fun x => Real.sin (2 * Real.pi * x))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Quotients.descends_iff_periodic
-- test: set-descends-periodic-test.sage

/-- `φ ↦ φ` does not descend: it is not periodic for any period `T ≠ 0`. -/
theorem id_not_periodic (T : ℝ) (hT : T ≠ 0) : ¬ Function.Periodic (fun φ : ℝ => φ) T := by
  intro h
  have := h 0
  simp at this
  exact hT this

example : veriphy_instance% KaluzaKlein.Sets.Quotients.id_not_periodic (2 * Real.pi) ((by positivity)) := by
  have key : ∀ (T : ℝ) (hT : T ≠ 0), ¬ Function.Periodic (fun φ : ℝ => φ) T := by
    intro T hT
    intro h
    have := h 0
    simp at this
    exact hT this
  exact @key (2 * Real.pi) ((by positivity))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.id_not_periodic (1) ((by norm_num)) := by
  have key : ∀ (T : ℝ) (hT : T ≠ 0), ¬ Function.Periodic (fun φ : ℝ => φ) T := by
    intro T hT
    intro h
    have := h 0
    simp at this
    exact hT this
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.id_not_periodic (-3) ((by norm_num)) := by
  have key : ∀ (T : ℝ) (hT : T ≠ 0), ¬ Function.Periodic (fun φ : ℝ => φ) T := by
    intro T hT
    intro h
    have := h 0
    simp at this
    exact hT this
  exact @key (-3) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Quotients.id_not_periodic
-- test: set-id-not-periodic-test.sage

/-! ## exm-quotient-sphere -/

/-- The rotations as a type. -/
abbrev SO3t := (Matrix.specialOrthogonalGroup (Fin 3) ℝ)

/-- A point is on the sphere exactly when `p · p = 1`. -/
theorem mem_sphere_iff_dot (p : Fin 3 → ℝ) : p ∈ sphere ↔ p ⬝ᵥ p = 1 := by
  simp [sphere, dotProduct, Fin.sum_univ_succ, sq, add_assoc]

/-- Rotations preserve the dot product of a vector with itself. -/
theorem rot_dot (g : Matrix (Fin 3) (Fin 3) ℝ) (hg : g ∈ SO3) (p : Fin 3 → ℝ) :
    (g *ᵥ p) ⬝ᵥ (g *ᵥ p) = p ⬝ᵥ p := by
  rw [mem_SO3_iff] at hg
  rw [Matrix.dotProduct_mulVec, ← Matrix.transpose_transpose g, Matrix.vecMul_transpose, Matrix.transpose_transpose,
    Matrix.mulVec_mulVec, hg.1, Matrix.one_mulVec]

/-- "Same image of `p₀`" on `SO(3)`. -/
def orbitSetoid (p₀ : Fin 3 → ℝ) : Setoid SO3t where
  r g g' := (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ = (g' : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀
  iseqv := Equivalence.sameValue_equivalence (fun g : SO3t => (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀)

/-- For `p₀` on the sphere, the quotient of `SO(3)` by "same image of `p₀`" is the sphere: `[g] ↦ g · p₀` is a
bijection onto it (for the north pole, the quotient by "same image of the pole"). -/
theorem orbit_quotient_bijective (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere) :
    ∃ F : Quotient (orbitSetoid p₀) → sphere, Function.Bijective F ∧
      ∀ g : SO3t, (F ⟦g⟧ : Fin 3 → ℝ) = (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ := by
  have hmem : ∀ g : SO3t, (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ ∈ sphere := fun g => by
    rw [mem_sphere_iff_dot, rot_dot _ g.2, ← mem_sphere_iff_dot]; exact hp₀
  refine ⟨Quotient.lift (fun g : SO3t => (⟨_, hmem g⟩ : sphere)) (fun _ _ h => Subtype.ext h), ⟨?_, ?_⟩,
    fun g => rfl⟩
  · intro p q h
    induction p using Quotient.ind
    induction q using Quotient.ind
    exact Quotient.sound (congrArg Subtype.val h)
  · rintro ⟨p, hp⟩
    have h1 := rotTo_mem p hp
    have h0 := rotTo_mem p₀ hp₀
    have h0' := h0
    rw [mem_SO3_iff] at h0'
    have hback : (rotTo p₀)ᵀ *ᵥ p₀ = northPole := by
      have h := rotTo_orbit p₀ hp₀
      unfold orbit at h
      calc (rotTo p₀)ᵀ *ᵥ p₀ = (rotTo p₀)ᵀ *ᵥ (rotTo p₀ *ᵥ northPole) := by rw [h]
        _ = northPole := by rw [Matrix.mulVec_mulVec, h0'.1, Matrix.one_mulVec]
    have hT : (rotTo p₀)ᵀ ∈ SO3 := by
      rw [mem_SO3_iff] at h0 ⊢
      refine ⟨?_, by rw [Matrix.det_transpose, h0.2]⟩
      rw [Matrix.transpose_transpose]; exact mul_eq_one_comm.1 h0.1
    refine ⟨⟦⟨rotTo p * (rotTo p₀)ᵀ, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).mul_mem h1 hT⟩⟧, Subtype.ext ?_⟩
    simp only [Quotient.lift_mk]
    rw [← Matrix.mulVec_mulVec, hback]
    exact rotTo_orbit p hp

example : veriphy_instance% KaluzaKlein.Sets.Quotients.orbit_quotient_bijective (KaluzaKlein.Sets.Sets.northPole) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere), ∃ F : Quotient (orbitSetoid p₀) → sphere, Function.Bijective F ∧ ∀ g : SO3t, (F ⟦g⟧ : Fin 3 → ℝ) = (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ := by
    intro p₀ hp₀
    have hmem : ∀ g : SO3t, (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ ∈ sphere := fun g => by
      rw [mem_sphere_iff_dot, rot_dot _ g.2, ← mem_sphere_iff_dot]; exact hp₀
    refine ⟨Quotient.lift (fun g : SO3t => (⟨_, hmem g⟩ : sphere)) (fun _ _ h => Subtype.ext h), ⟨?_, ?_⟩,
      fun g => rfl⟩
    · intro p q h
      induction p using Quotient.ind
      induction q using Quotient.ind
      exact Quotient.sound (congrArg Subtype.val h)
    · rintro ⟨p, hp⟩
      have h1 := rotTo_mem p hp
      have h0 := rotTo_mem p₀ hp₀
      have h0' := h0
      rw [mem_SO3_iff] at h0'
      have hback : (rotTo p₀)ᵀ *ᵥ p₀ = northPole := by
        have h := rotTo_orbit p₀ hp₀
        unfold orbit at h
        calc (rotTo p₀)ᵀ *ᵥ p₀ = (rotTo p₀)ᵀ *ᵥ (rotTo p₀ *ᵥ northPole) := by rw [h]
          _ = northPole := by rw [Matrix.mulVec_mulVec, h0'.1, Matrix.one_mulVec]
      have hT : (rotTo p₀)ᵀ ∈ SO3 := by
        rw [mem_SO3_iff] at h0 ⊢
        refine ⟨?_, by rw [Matrix.det_transpose, h0.2]⟩
        rw [Matrix.transpose_transpose]; exact mul_eq_one_comm.1 h0.1
      refine ⟨⟦⟨rotTo p * (rotTo p₀)ᵀ, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).mul_mem h1 hT⟩⟧, Subtype.ext ?_⟩
      simp only [Quotient.lift_mk]
      rw [← Matrix.mulVec_mulVec, hback]
      exact rotTo_orbit p hp
  exact @key (KaluzaKlein.Sets.Sets.northPole) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.orbit_quotient_bijective (![1, 0, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere), ∃ F : Quotient (orbitSetoid p₀) → sphere, Function.Bijective F ∧ ∀ g : SO3t, (F ⟦g⟧ : Fin 3 → ℝ) = (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ := by
    intro p₀ hp₀
    have hmem : ∀ g : SO3t, (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ ∈ sphere := fun g => by
      rw [mem_sphere_iff_dot, rot_dot _ g.2, ← mem_sphere_iff_dot]; exact hp₀
    refine ⟨Quotient.lift (fun g : SO3t => (⟨_, hmem g⟩ : sphere)) (fun _ _ h => Subtype.ext h), ⟨?_, ?_⟩,
      fun g => rfl⟩
    · intro p q h
      induction p using Quotient.ind
      induction q using Quotient.ind
      exact Quotient.sound (congrArg Subtype.val h)
    · rintro ⟨p, hp⟩
      have h1 := rotTo_mem p hp
      have h0 := rotTo_mem p₀ hp₀
      have h0' := h0
      rw [mem_SO3_iff] at h0'
      have hback : (rotTo p₀)ᵀ *ᵥ p₀ = northPole := by
        have h := rotTo_orbit p₀ hp₀
        unfold orbit at h
        calc (rotTo p₀)ᵀ *ᵥ p₀ = (rotTo p₀)ᵀ *ᵥ (rotTo p₀ *ᵥ northPole) := by rw [h]
          _ = northPole := by rw [Matrix.mulVec_mulVec, h0'.1, Matrix.one_mulVec]
      have hT : (rotTo p₀)ᵀ ∈ SO3 := by
        rw [mem_SO3_iff] at h0 ⊢
        refine ⟨?_, by rw [Matrix.det_transpose, h0.2]⟩
        rw [Matrix.transpose_transpose]; exact mul_eq_one_comm.1 h0.1
      refine ⟨⟦⟨rotTo p * (rotTo p₀)ᵀ, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).mul_mem h1 hT⟩⟧, Subtype.ext ?_⟩
      simp only [Quotient.lift_mk]
      rw [← Matrix.mulVec_mulVec, hback]
      exact rotTo_orbit p hp
  exact @key (![1, 0, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.orbit_quotient_bijective (![0, 0, -1]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere), ∃ F : Quotient (orbitSetoid p₀) → sphere, Function.Bijective F ∧ ∀ g : SO3t, (F ⟦g⟧ : Fin 3 → ℝ) = (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ := by
    intro p₀ hp₀
    have hmem : ∀ g : SO3t, (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ ∈ sphere := fun g => by
      rw [mem_sphere_iff_dot, rot_dot _ g.2, ← mem_sphere_iff_dot]; exact hp₀
    refine ⟨Quotient.lift (fun g : SO3t => (⟨_, hmem g⟩ : sphere)) (fun _ _ h => Subtype.ext h), ⟨?_, ?_⟩,
      fun g => rfl⟩
    · intro p q h
      induction p using Quotient.ind
      induction q using Quotient.ind
      exact Quotient.sound (congrArg Subtype.val h)
    · rintro ⟨p, hp⟩
      have h1 := rotTo_mem p hp
      have h0 := rotTo_mem p₀ hp₀
      have h0' := h0
      rw [mem_SO3_iff] at h0'
      have hback : (rotTo p₀)ᵀ *ᵥ p₀ = northPole := by
        have h := rotTo_orbit p₀ hp₀
        unfold orbit at h
        calc (rotTo p₀)ᵀ *ᵥ p₀ = (rotTo p₀)ᵀ *ᵥ (rotTo p₀ *ᵥ northPole) := by rw [h]
          _ = northPole := by rw [Matrix.mulVec_mulVec, h0'.1, Matrix.one_mulVec]
      have hT : (rotTo p₀)ᵀ ∈ SO3 := by
        rw [mem_SO3_iff] at h0 ⊢
        refine ⟨?_, by rw [Matrix.det_transpose, h0.2]⟩
        rw [Matrix.transpose_transpose]; exact mul_eq_one_comm.1 h0.1
      refine ⟨⟦⟨rotTo p * (rotTo p₀)ᵀ, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).mul_mem h1 hT⟩⟧, Subtype.ext ?_⟩
      simp only [Quotient.lift_mk]
      rw [← Matrix.mulVec_mulVec, hback]
      exact rotTo_orbit p hp
  exact @key (![0, 0, -1]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Quotients.orbit_quotient_bijective
-- test: set-orbit-quotient-test.sage

/-! ## exr-quotient-z5-table -/

/-- The operations of `ℤₙ` on remainders: `(a + b) mod n` and `(a b) mod n` (the tables of `ℤ₃` and `ℤ₅`). -/
theorem zmod_ops (n : ℕ) [NeZero n] (a b : ZMod n) :
    (a + b).val = (a.val + b.val) % n ∧ (a * b).val = (a.val * b.val) % n :=
  ⟨ZMod.val_add a b, ZMod.val_mul a b⟩

example : veriphy_instance% KaluzaKlein.Sets.Quotients.zmod_ops (3) (inferInstance) (1) (2) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Quotients.zmod_ops (3) (inferInstance) (2) (2) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Quotients.zmod_ops (5) (inferInstance) (2) (3) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Quotients.zmod_ops (5) (inferInstance) (4) (4) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Quotients.zmod_ops (1) (inferInstance) (0) (0) := by
  decide

attribute [veriphy_tested kernel_test 5 0] KaluzaKlein.Sets.Quotients.zmod_ops
-- test: set-zmod-ops-test.sage

/-! ## exr-descend -/

/-- `cos 2φ` is `2π`-periodic and descends to the circle. -/
theorem cos_two_mul_periodic (φ : ℝ) : Real.cos (2 * (φ + 2 * Real.pi)) = Real.cos (2 * φ) := by
  veriphy_certified
    (rw [show 2 * (φ + 2 * Real.pi) = 2 * φ + ((2 : ℕ) : ℝ) * (2 * Real.pi) by push_cast; ring,
        Real.cos_add_nat_mul_two_pi])
  -- certificate: set-trig-periodic.sage

/-- `e^{iφ}` is `2π`-periodic and descends to the circle. -/
theorem exp_I_periodic (φ : ℝ) : Complex.exp ((φ + 2 * Real.pi : ℝ) * Complex.I) = Complex.exp (φ * Complex.I) := by
  push_cast
  rw [add_mul, Complex.exp_add, Complex.exp_two_pi_mul_I, mul_one]

example : veriphy_instance% KaluzaKlein.Sets.Quotients.exp_I_periodic (0) := by
  have key : ∀ (φ : ℝ), Complex.exp ((φ + 2 * Real.pi : ℝ) * Complex.I) = Complex.exp (φ * Complex.I) := by
    intro φ
    push_cast
    rw [add_mul, Complex.exp_add, Complex.exp_two_pi_mul_I, mul_one]
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Quotients.exp_I_periodic (1) := by
  have key : ∀ (φ : ℝ), Complex.exp ((φ + 2 * Real.pi : ℝ) * Complex.I) = Complex.exp (φ * Complex.I) := by
    intro φ
    push_cast
    rw [add_mul, Complex.exp_add, Complex.exp_two_pi_mul_I, mul_one]
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Quotients.exp_I_periodic (Real.pi) := by
  have key : ∀ (φ : ℝ), Complex.exp ((φ + 2 * Real.pi : ℝ) * Complex.I) = Complex.exp (φ * Complex.I) := by
    intro φ
    push_cast
    rw [add_mul, Complex.exp_add, Complex.exp_two_pi_mul_I, mul_one]
  exact @key (Real.pi)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Quotients.exp_I_periodic
-- test: set-exp-i-periodic-test.sage

/-- `cos(φ/2)` changes sign under `φ ↦ φ + 2π`, so it does not descend. -/
theorem cos_half_antiperiodic (φ : ℝ) : Real.cos ((φ + 2 * Real.pi) / 2) = -Real.cos (φ / 2) := by
  veriphy_certified (rw [show (φ + 2 * Real.pi) / 2 = φ / 2 + Real.pi by ring, Real.cos_add_pi])
  -- certificate: set-trig-periodic.sage

/-- Wherever `cos(φ/2) ≠ 0`, the values at `φ` and `φ + 2π` differ: `cos(φ/2)` is not constant on classes (at
`φ = 0` it changes from `1` to `-1`). -/
theorem cos_half_changes (φ : ℝ) (h : Real.cos (φ / 2) ≠ 0) : Real.cos ((φ + 2 * Real.pi) / 2) ≠ Real.cos (φ / 2) := by
  rw [cos_half_antiperiodic φ]
  intro h'
  exact h (by linarith)

example : veriphy_instance% KaluzaKlein.Sets.Quotients.cos_half_changes (0) ((by simp)) := by
  have key : ∀ (φ : ℝ) (h : Real.cos (φ / 2) ≠ 0), Real.cos ((φ + 2 * Real.pi) / 2) ≠ Real.cos (φ / 2) := by
    intro φ h
    rw [cos_half_antiperiodic φ]
    intro h'
    exact h (by linarith)
  exact @key (0) ((by simp))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.cos_half_changes (2 * Real.pi / 3) ((by rw [show 2 * Real.pi / 3 / 2 = Real.pi / 3 by ring, Real.cos_pi_div_three]; norm_num)) := by
  have key : ∀ (φ : ℝ) (h : Real.cos (φ / 2) ≠ 0), Real.cos ((φ + 2 * Real.pi) / 2) ≠ Real.cos (φ / 2) := by
    intro φ h
    rw [cos_half_antiperiodic φ]
    intro h'
    exact h (by linarith)
  exact @key (2 * Real.pi / 3) ((by rw [show 2 * Real.pi / 3 / 2 = Real.pi / 3 by ring, Real.cos_pi_div_three]; norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.cos_half_changes (-(2 * Real.pi / 3)) ((by rw [show -(2 * Real.pi / 3) / 2 = -(Real.pi / 3) by ring, Real.cos_neg, Real.cos_pi_div_three]; norm_num)) := by
  have key : ∀ (φ : ℝ) (h : Real.cos (φ / 2) ≠ 0), Real.cos ((φ + 2 * Real.pi) / 2) ≠ Real.cos (φ / 2) := by
    intro φ h
    rw [cos_half_antiperiodic φ]
    intro h'
    exact h (by linarith)
  exact @key (-(2 * Real.pi / 3)) ((by rw [show -(2 * Real.pi / 3) / 2 = -(Real.pi / 3) by ring, Real.cos_neg, Real.cos_pi_div_three]; norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Quotients.cos_half_changes
-- test: set-cos-half-changes-test.sage

/-! ## exr-quotient-square -/

/-- The gluing of the opposite edges of the square `[0, L]²`: `(0, v) ∼ (L, v)` and `(u, 0) ∼ (u, L)` (`L = 1` in the
exercise, `L = 2π` for the torus). -/
def glue (L : ℝ) (p q : ℝ × ℝ) : Prop :=
  (p.1 = 0 ∧ q.1 = L ∧ p.2 = q.2) ∨ (p.2 = 0 ∧ q.2 = L ∧ p.1 = q.1)

/-- `L ↦ 0`, identity elsewhere: the label of a coordinate after gluing. -/
noncomputable def κ (L u : ℝ) : ℝ := if u = L then 0 else u

/-- The square `[0, L]²`. -/
def square (L : ℝ) : Set (ℝ × ℝ) := Set.Icc 0 L ×ˢ Set.Icc 0 L

/-- On the square `[0, L]²` (`L > 0`), the equivalence relation generated by the gluing identifies two points exactly
when their coordinates agree after `L ↦ 0`. -/
theorem glue_eqvGen_iff (L : ℝ) (hL : 0 < L) (p q : ℝ × ℝ) (hp : p ∈ square L) (hq : q ∈ square L) :
    Relation.EqvGen (glue L) p q ↔ κ L p.1 = κ L q.1 ∧ κ L p.2 = κ L q.2 := by
  constructor
  · intro h
    clear hp hq
    induction h with
    | rel a b hab =>
      rcases hab with ⟨h1, h2, h3⟩ | ⟨h1, h2, h3⟩ <;> simp [κ, h1, h2, h3, hL.ne']
    | refl => exact ⟨rfl, rfl⟩
    | symm _ _ _ ih => exact ⟨ih.1.symm, ih.2.symm⟩
    | trans _ _ _ _ _ ih1 ih2 => exact ⟨ih1.1.trans ih2.1, ih1.2.trans ih2.2⟩
  · rintro ⟨h1, h2⟩
    obtain ⟨u, v⟩ := p
    obtain ⟨u', v'⟩ := q
    simp only at h1 h2
    have step1 : Relation.EqvGen (glue L) (u, v) (u', v) := by
      unfold κ at h1
      split_ifs at h1 with ha hb hb
      · rw [ha, hb]; exact Relation.EqvGen.refl _
      · rw [ha, ← h1]; exact Relation.EqvGen.symm _ _ (Relation.EqvGen.rel _ _ (Or.inl ⟨rfl, rfl, rfl⟩))
      · rw [hb, h1]; exact Relation.EqvGen.rel _ _ (Or.inl ⟨rfl, rfl, rfl⟩)
      · rw [h1]; exact Relation.EqvGen.refl _
    have step2 : Relation.EqvGen (glue L) (u', v) (u', v') := by
      unfold κ at h2
      split_ifs at h2 with ha hb hb
      · rw [ha, hb]; exact Relation.EqvGen.refl _
      · rw [ha, ← h2]; exact Relation.EqvGen.symm _ _ (Relation.EqvGen.rel _ _ (Or.inr ⟨rfl, rfl, rfl⟩))
      · rw [hb, h2]; exact Relation.EqvGen.rel _ _ (Or.inr ⟨rfl, rfl, rfl⟩)
      · rw [h2]; exact Relation.EqvGen.refl _
    exact Relation.EqvGen.trans _ _ _ step1 step2

example : veriphy_instance% KaluzaKlein.Sets.Quotients.glue_eqvGen_iff (1) ((by norm_num)) ((0, 1/2)) ((1, 1/2)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L) (p q : ℝ × ℝ) (hp : p ∈ square L) (hq : q ∈ square L), Relation.EqvGen (glue L) p q ↔ κ L p.1 = κ L q.1 ∧ κ L p.2 = κ L q.2 := by
    intro L hL p q hp hq
    constructor
    · intro h
      clear hp hq
      induction h with
      | rel a b hab =>
        rcases hab with ⟨h1, h2, h3⟩ | ⟨h1, h2, h3⟩ <;> simp [κ, h1, h2, h3, hL.ne']
      | refl => exact ⟨rfl, rfl⟩
      | symm _ _ _ ih => exact ⟨ih.1.symm, ih.2.symm⟩
      | trans _ _ _ _ _ ih1 ih2 => exact ⟨ih1.1.trans ih2.1, ih1.2.trans ih2.2⟩
    · rintro ⟨h1, h2⟩
      obtain ⟨u, v⟩ := p
      obtain ⟨u', v'⟩ := q
      simp only at h1 h2
      have step1 : Relation.EqvGen (glue L) (u, v) (u', v) := by
        unfold κ at h1
        split_ifs at h1 with ha hb hb
        · rw [ha, hb]; exact Relation.EqvGen.refl _
        · rw [ha, ← h1]; exact Relation.EqvGen.symm _ _ (Relation.EqvGen.rel _ _ (Or.inl ⟨rfl, rfl, rfl⟩))
        · rw [hb, h1]; exact Relation.EqvGen.rel _ _ (Or.inl ⟨rfl, rfl, rfl⟩)
        · rw [h1]; exact Relation.EqvGen.refl _
      have step2 : Relation.EqvGen (glue L) (u', v) (u', v') := by
        unfold κ at h2
        split_ifs at h2 with ha hb hb
        · rw [ha, hb]; exact Relation.EqvGen.refl _
        · rw [ha, ← h2]; exact Relation.EqvGen.symm _ _ (Relation.EqvGen.rel _ _ (Or.inr ⟨rfl, rfl, rfl⟩))
        · rw [hb, h2]; exact Relation.EqvGen.rel _ _ (Or.inr ⟨rfl, rfl, rfl⟩)
        · rw [h2]; exact Relation.EqvGen.refl _
      exact Relation.EqvGen.trans _ _ _ step1 step2
  exact @key (1) ((by norm_num)) ((0, 1/2)) ((1, 1/2)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.glue_eqvGen_iff (1) ((by norm_num)) ((1/2, 1/3)) ((1/3, 1/2)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L) (p q : ℝ × ℝ) (hp : p ∈ square L) (hq : q ∈ square L), Relation.EqvGen (glue L) p q ↔ κ L p.1 = κ L q.1 ∧ κ L p.2 = κ L q.2 := by
    intro L hL p q hp hq
    constructor
    · intro h
      clear hp hq
      induction h with
      | rel a b hab =>
        rcases hab with ⟨h1, h2, h3⟩ | ⟨h1, h2, h3⟩ <;> simp [κ, h1, h2, h3, hL.ne']
      | refl => exact ⟨rfl, rfl⟩
      | symm _ _ _ ih => exact ⟨ih.1.symm, ih.2.symm⟩
      | trans _ _ _ _ _ ih1 ih2 => exact ⟨ih1.1.trans ih2.1, ih1.2.trans ih2.2⟩
    · rintro ⟨h1, h2⟩
      obtain ⟨u, v⟩ := p
      obtain ⟨u', v'⟩ := q
      simp only at h1 h2
      have step1 : Relation.EqvGen (glue L) (u, v) (u', v) := by
        unfold κ at h1
        split_ifs at h1 with ha hb hb
        · rw [ha, hb]; exact Relation.EqvGen.refl _
        · rw [ha, ← h1]; exact Relation.EqvGen.symm _ _ (Relation.EqvGen.rel _ _ (Or.inl ⟨rfl, rfl, rfl⟩))
        · rw [hb, h1]; exact Relation.EqvGen.rel _ _ (Or.inl ⟨rfl, rfl, rfl⟩)
        · rw [h1]; exact Relation.EqvGen.refl _
      have step2 : Relation.EqvGen (glue L) (u', v) (u', v') := by
        unfold κ at h2
        split_ifs at h2 with ha hb hb
        · rw [ha, hb]; exact Relation.EqvGen.refl _
        · rw [ha, ← h2]; exact Relation.EqvGen.symm _ _ (Relation.EqvGen.rel _ _ (Or.inr ⟨rfl, rfl, rfl⟩))
        · rw [hb, h2]; exact Relation.EqvGen.rel _ _ (Or.inr ⟨rfl, rfl, rfl⟩)
        · rw [h2]; exact Relation.EqvGen.refl _
      exact Relation.EqvGen.trans _ _ _ step1 step2
  exact @key (1) ((by norm_num)) ((1/2, 1/3)) ((1/3, 1/2)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.glue_eqvGen_iff (2 * Real.pi) ((by positivity)) ((0, 0)) ((2 * Real.pi, 2 * Real.pi)) ((by simp [KaluzaKlein.Sets.Quotients.square]; positivity)) ((by simp [KaluzaKlein.Sets.Quotients.square]; positivity)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L) (p q : ℝ × ℝ) (hp : p ∈ square L) (hq : q ∈ square L), Relation.EqvGen (glue L) p q ↔ κ L p.1 = κ L q.1 ∧ κ L p.2 = κ L q.2 := by
    intro L hL p q hp hq
    constructor
    · intro h
      clear hp hq
      induction h with
      | rel a b hab =>
        rcases hab with ⟨h1, h2, h3⟩ | ⟨h1, h2, h3⟩ <;> simp [κ, h1, h2, h3, hL.ne']
      | refl => exact ⟨rfl, rfl⟩
      | symm _ _ _ ih => exact ⟨ih.1.symm, ih.2.symm⟩
      | trans _ _ _ _ _ ih1 ih2 => exact ⟨ih1.1.trans ih2.1, ih1.2.trans ih2.2⟩
    · rintro ⟨h1, h2⟩
      obtain ⟨u, v⟩ := p
      obtain ⟨u', v'⟩ := q
      simp only at h1 h2
      have step1 : Relation.EqvGen (glue L) (u, v) (u', v) := by
        unfold κ at h1
        split_ifs at h1 with ha hb hb
        · rw [ha, hb]; exact Relation.EqvGen.refl _
        · rw [ha, ← h1]; exact Relation.EqvGen.symm _ _ (Relation.EqvGen.rel _ _ (Or.inl ⟨rfl, rfl, rfl⟩))
        · rw [hb, h1]; exact Relation.EqvGen.rel _ _ (Or.inl ⟨rfl, rfl, rfl⟩)
        · rw [h1]; exact Relation.EqvGen.refl _
      have step2 : Relation.EqvGen (glue L) (u', v) (u', v') := by
        unfold κ at h2
        split_ifs at h2 with ha hb hb
        · rw [ha, hb]; exact Relation.EqvGen.refl _
        · rw [ha, ← h2]; exact Relation.EqvGen.symm _ _ (Relation.EqvGen.rel _ _ (Or.inr ⟨rfl, rfl, rfl⟩))
        · rw [hb, h2]; exact Relation.EqvGen.rel _ _ (Or.inr ⟨rfl, rfl, rfl⟩)
        · rw [h2]; exact Relation.EqvGen.refl _
      exact Relation.EqvGen.trans _ _ _ step1 step2
  exact @key (2 * Real.pi) ((by positivity)) ((0, 0)) ((2 * Real.pi, 2 * Real.pi)) ((by simp [KaluzaKlein.Sets.Quotients.square]; positivity)) ((by simp [KaluzaKlein.Sets.Quotients.square]; positivity))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Quotients.glue_eqvGen_iff
-- test: set-glue-eqvgen-test.sage

/-- Two coordinates agree after `L ↦ 0` exactly when they are equal or are the two ends `0` and `L`. -/
theorem κ_eq_iff (L u u' : ℝ) (hL : 0 < L) : κ L u = κ L u' ↔ u = u' ∨ (u = 0 ∧ u' = L) ∨ (u = L ∧ u' = 0) := by
  unfold κ
  split_ifs with h1 h2 h2
  · subst h1; subst h2; simp
  · subst h1
    constructor
    · intro h; right; right; exact ⟨rfl, h.symm⟩
    · rintro (h | ⟨h, -⟩ | ⟨-, h⟩)
      · exact absurd h.symm h2
      · exact absurd h hL.ne'
      · exact h.symm
  · subst h2
    constructor
    · intro h; right; left; exact ⟨h, rfl⟩
    · rintro (h | ⟨h, -⟩ | ⟨-, h⟩)
      · exact absurd h h1
      · exact h
      · exact absurd h hL.ne'
  · constructor
    · intro h; left; exact h
    · rintro (h | ⟨-, h⟩ | ⟨h, -⟩)
      · exact h
      · exact absurd h h2
      · exact absurd h h1

/-- The four corners of `[0, 1]²` are all identified: any two corners are related, one class. -/
theorem corners_one_class (c c' : ℝ × ℝ) (hc : (c.1 = 0 ∨ c.1 = 1) ∧ (c.2 = 0 ∨ c.2 = 1))
    (hc' : (c'.1 = 0 ∨ c'.1 = 1) ∧ (c'.2 = 0 ∨ c'.2 = 1)) : Relation.EqvGen (glue 1) c c' := by
  have hsq : ∀ x : ℝ × ℝ, (x.1 = 0 ∨ x.1 = 1) ∧ (x.2 = 0 ∨ x.2 = 1) → x ∈ square 1 := by
    rintro ⟨a, b⟩ ⟨ha, hb⟩
    simp only [square, Set.mem_prod, Set.mem_Icc]
    simp only at ha hb
    rcases ha with rfl | rfl <;> rcases hb with rfl | rfl <;> norm_num
  rw [glue_eqvGen_iff 1 one_pos c c' (hsq c hc) (hsq c' hc'), κ_eq_iff 1 _ _ one_pos, κ_eq_iff 1 _ _ one_pos]
  obtain ⟨a, b⟩ := c
  obtain ⟨a', b'⟩ := c'
  simp only at hc hc' ⊢
  obtain ⟨h1 | h1, h2 | h2⟩ := hc <;> obtain ⟨h3 | h3, h4 | h4⟩ := hc' <;> subst_vars <;> norm_num

example : veriphy_instance% KaluzaKlein.Sets.Quotients.corners_one_class ((0, 0)) ((1, 1)) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (c c' : ℝ × ℝ) (hc : (c.1 = 0 ∨ c.1 = 1) ∧ (c.2 = 0 ∨ c.2 = 1)) (hc' : (c'.1 = 0 ∨ c'.1 = 1) ∧ (c'.2 = 0 ∨ c'.2 = 1)), Relation.EqvGen (glue 1) c c' := by
    intro c c' hc hc'
    have hsq : ∀ x : ℝ × ℝ, (x.1 = 0 ∨ x.1 = 1) ∧ (x.2 = 0 ∨ x.2 = 1) → x ∈ square 1 := by
      rintro ⟨a, b⟩ ⟨ha, hb⟩
      simp only [square, Set.mem_prod, Set.mem_Icc]
      simp only at ha hb
      rcases ha with rfl | rfl <;> rcases hb with rfl | rfl <;> norm_num
    rw [glue_eqvGen_iff 1 one_pos c c' (hsq c hc) (hsq c' hc'), κ_eq_iff 1 _ _ one_pos, κ_eq_iff 1 _ _ one_pos]
    obtain ⟨a, b⟩ := c
    obtain ⟨a', b'⟩ := c'
    simp only at hc hc' ⊢
    obtain ⟨h1 | h1, h2 | h2⟩ := hc <;> obtain ⟨h3 | h3, h4 | h4⟩ := hc' <;> subst_vars <;> norm_num
  exact @key ((0, 0)) ((1, 1)) ((by norm_num)) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.corners_one_class ((0, 0)) ((0, 0)) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (c c' : ℝ × ℝ) (hc : (c.1 = 0 ∨ c.1 = 1) ∧ (c.2 = 0 ∨ c.2 = 1)) (hc' : (c'.1 = 0 ∨ c'.1 = 1) ∧ (c'.2 = 0 ∨ c'.2 = 1)), Relation.EqvGen (glue 1) c c' := by
    intro c c' hc hc'
    have hsq : ∀ x : ℝ × ℝ, (x.1 = 0 ∨ x.1 = 1) ∧ (x.2 = 0 ∨ x.2 = 1) → x ∈ square 1 := by
      rintro ⟨a, b⟩ ⟨ha, hb⟩
      simp only [square, Set.mem_prod, Set.mem_Icc]
      simp only at ha hb
      rcases ha with rfl | rfl <;> rcases hb with rfl | rfl <;> norm_num
    rw [glue_eqvGen_iff 1 one_pos c c' (hsq c hc) (hsq c' hc'), κ_eq_iff 1 _ _ one_pos, κ_eq_iff 1 _ _ one_pos]
    obtain ⟨a, b⟩ := c
    obtain ⟨a', b'⟩ := c'
    simp only at hc hc' ⊢
    obtain ⟨h1 | h1, h2 | h2⟩ := hc <;> obtain ⟨h3 | h3, h4 | h4⟩ := hc' <;> subst_vars <;> norm_num
  exact @key ((0, 0)) ((0, 0)) ((by norm_num)) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.corners_one_class ((1, 0)) ((0, 1)) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (c c' : ℝ × ℝ) (hc : (c.1 = 0 ∨ c.1 = 1) ∧ (c.2 = 0 ∨ c.2 = 1)) (hc' : (c'.1 = 0 ∨ c'.1 = 1) ∧ (c'.2 = 0 ∨ c'.2 = 1)), Relation.EqvGen (glue 1) c c' := by
    intro c c' hc hc'
    have hsq : ∀ x : ℝ × ℝ, (x.1 = 0 ∨ x.1 = 1) ∧ (x.2 = 0 ∨ x.2 = 1) → x ∈ square 1 := by
      rintro ⟨a, b⟩ ⟨ha, hb⟩
      simp only [square, Set.mem_prod, Set.mem_Icc]
      simp only at ha hb
      rcases ha with rfl | rfl <;> rcases hb with rfl | rfl <;> norm_num
    rw [glue_eqvGen_iff 1 one_pos c c' (hsq c hc) (hsq c' hc'), κ_eq_iff 1 _ _ one_pos, κ_eq_iff 1 _ _ one_pos]
    obtain ⟨a, b⟩ := c
    obtain ⟨a', b'⟩ := c'
    simp only at hc hc' ⊢
    obtain ⟨h1 | h1, h2 | h2⟩ := hc <;> obtain ⟨h3 | h3, h4 | h4⟩ := hc' <;> subst_vars <;> norm_num
  exact @key ((1, 0)) ((0, 1)) ((by norm_num)) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Quotients.corners_one_class
-- test: set-corners-test.sage

/-- A point `(0, v)` of `[0, 1]²` with `0 < v < 1` is identified exactly with `(1, v)` (a two-element class), and an
interior point `(u, v)` is alone in its class. -/
theorem edge_interior_classes (v : ℝ) (hv : v ∈ Set.Ioo (0 : ℝ) 1) (q : ℝ × ℝ) (hq : q ∈ square 1) :
    (Relation.EqvGen (glue 1) (0, v) q ↔ q = (0, v) ∨ q = (1, v)) ∧
      (∀ u ∈ Set.Ioo (0 : ℝ) 1, Relation.EqvGen (glue 1) (u, v) q ↔ q = (u, v)) := by
  obtain ⟨hv0, hv1⟩ := hv
  have hmem : ∀ u : ℝ, 0 ≤ u → u ≤ 1 → (u, v) ∈ square 1 := fun u h1 h2 => ⟨⟨h1, h2⟩, ⟨hv0.le, hv1.le⟩⟩
  have hb : ∀ b : ℝ, (v = b ∨ (v = 0 ∧ b = 1) ∨ (v = 1 ∧ b = 0)) ↔ b = v := by
    intro b
    constructor
    · rintro (h | ⟨h, -⟩ | ⟨h, -⟩)
      · exact h.symm
      · linarith
      · linarith
    · intro h; left; exact h.symm
  constructor
  · rw [glue_eqvGen_iff 1 one_pos _ q (hmem 0 le_rfl zero_le_one) hq, κ_eq_iff 1 _ _ one_pos,
      κ_eq_iff 1 _ _ one_pos]
    obtain ⟨a, b⟩ := q
    simp only [Prod.mk.injEq]
    rw [hb b]
    constructor
    · rintro ⟨h | ⟨-, h⟩ | ⟨h, -⟩, hbv⟩
      · left; exact ⟨h.symm, hbv⟩
      · right; exact ⟨h, hbv⟩
      · norm_num at h
    · rintro (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩) <;> simp
  · intro u hu
    obtain ⟨hu0, hu1⟩ := hu
    rw [glue_eqvGen_iff 1 one_pos _ q (hmem u hu0.le hu1.le) hq, κ_eq_iff 1 _ _ one_pos,
      κ_eq_iff 1 _ _ one_pos]
    obtain ⟨a, b⟩ := q
    simp only [Prod.mk.injEq]
    rw [hb b]
    constructor
    · rintro ⟨h | ⟨h, -⟩ | ⟨h, -⟩, hbv⟩
      · exact ⟨h.symm, hbv⟩
      · linarith
      · linarith
    · rintro ⟨rfl, rfl⟩
      exact ⟨Or.inl rfl, by simp⟩

example : veriphy_instance% KaluzaKlein.Sets.Quotients.edge_interior_classes (1/2) ((by norm_num)) ((1, 1/2)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num)) := by
  have key : ∀ (v : ℝ) (hv : v ∈ Set.Ioo (0 : ℝ) 1) (q : ℝ × ℝ) (hq : q ∈ square 1), (Relation.EqvGen (glue 1) (0, v) q ↔ q = (0, v) ∨ q = (1, v)) ∧ (∀ u ∈ Set.Ioo (0 : ℝ) 1, Relation.EqvGen (glue 1) (u, v) q ↔ q = (u, v)) := by
    intro v hv q hq
    obtain ⟨hv0, hv1⟩ := hv
    have hmem : ∀ u : ℝ, 0 ≤ u → u ≤ 1 → (u, v) ∈ square 1 := fun u h1 h2 => ⟨⟨h1, h2⟩, ⟨hv0.le, hv1.le⟩⟩
    have hb : ∀ b : ℝ, (v = b ∨ (v = 0 ∧ b = 1) ∨ (v = 1 ∧ b = 0)) ↔ b = v := by
      intro b
      constructor
      · rintro (h | ⟨h, -⟩ | ⟨h, -⟩)
        · exact h.symm
        · linarith
        · linarith
      · intro h; left; exact h.symm
    constructor
    · rw [glue_eqvGen_iff 1 one_pos _ q (hmem 0 le_rfl zero_le_one) hq, κ_eq_iff 1 _ _ one_pos,
        κ_eq_iff 1 _ _ one_pos]
      obtain ⟨a, b⟩ := q
      simp only [Prod.mk.injEq]
      rw [hb b]
      constructor
      · rintro ⟨h | ⟨-, h⟩ | ⟨h, -⟩, hbv⟩
        · left; exact ⟨h.symm, hbv⟩
        · right; exact ⟨h, hbv⟩
        · norm_num at h
      · rintro (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩) <;> simp
    · intro u hu
      obtain ⟨hu0, hu1⟩ := hu
      rw [glue_eqvGen_iff 1 one_pos _ q (hmem u hu0.le hu1.le) hq, κ_eq_iff 1 _ _ one_pos,
        κ_eq_iff 1 _ _ one_pos]
      obtain ⟨a, b⟩ := q
      simp only [Prod.mk.injEq]
      rw [hb b]
      constructor
      · rintro ⟨h | ⟨h, -⟩ | ⟨h, -⟩, hbv⟩
        · exact ⟨h.symm, hbv⟩
        · linarith
        · linarith
      · rintro ⟨rfl, rfl⟩
        exact ⟨Or.inl rfl, by simp⟩
  exact @key (1/2) ((by norm_num)) ((1, 1/2)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.edge_interior_classes (1/2) ((by norm_num)) ((0, 1/2)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num)) := by
  have key : ∀ (v : ℝ) (hv : v ∈ Set.Ioo (0 : ℝ) 1) (q : ℝ × ℝ) (hq : q ∈ square 1), (Relation.EqvGen (glue 1) (0, v) q ↔ q = (0, v) ∨ q = (1, v)) ∧ (∀ u ∈ Set.Ioo (0 : ℝ) 1, Relation.EqvGen (glue 1) (u, v) q ↔ q = (u, v)) := by
    intro v hv q hq
    obtain ⟨hv0, hv1⟩ := hv
    have hmem : ∀ u : ℝ, 0 ≤ u → u ≤ 1 → (u, v) ∈ square 1 := fun u h1 h2 => ⟨⟨h1, h2⟩, ⟨hv0.le, hv1.le⟩⟩
    have hb : ∀ b : ℝ, (v = b ∨ (v = 0 ∧ b = 1) ∨ (v = 1 ∧ b = 0)) ↔ b = v := by
      intro b
      constructor
      · rintro (h | ⟨h, -⟩ | ⟨h, -⟩)
        · exact h.symm
        · linarith
        · linarith
      · intro h; left; exact h.symm
    constructor
    · rw [glue_eqvGen_iff 1 one_pos _ q (hmem 0 le_rfl zero_le_one) hq, κ_eq_iff 1 _ _ one_pos,
        κ_eq_iff 1 _ _ one_pos]
      obtain ⟨a, b⟩ := q
      simp only [Prod.mk.injEq]
      rw [hb b]
      constructor
      · rintro ⟨h | ⟨-, h⟩ | ⟨h, -⟩, hbv⟩
        · left; exact ⟨h.symm, hbv⟩
        · right; exact ⟨h, hbv⟩
        · norm_num at h
      · rintro (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩) <;> simp
    · intro u hu
      obtain ⟨hu0, hu1⟩ := hu
      rw [glue_eqvGen_iff 1 one_pos _ q (hmem u hu0.le hu1.le) hq, κ_eq_iff 1 _ _ one_pos,
        κ_eq_iff 1 _ _ one_pos]
      obtain ⟨a, b⟩ := q
      simp only [Prod.mk.injEq]
      rw [hb b]
      constructor
      · rintro ⟨h | ⟨h, -⟩ | ⟨h, -⟩, hbv⟩
        · exact ⟨h.symm, hbv⟩
        · linarith
        · linarith
      · rintro ⟨rfl, rfl⟩
        exact ⟨Or.inl rfl, by simp⟩
  exact @key (1/2) ((by norm_num)) ((0, 1/2)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Quotients.edge_interior_classes (1/3) ((by norm_num)) ((1/2, 1/3)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num)) := by
  have key : ∀ (v : ℝ) (hv : v ∈ Set.Ioo (0 : ℝ) 1) (q : ℝ × ℝ) (hq : q ∈ square 1), (Relation.EqvGen (glue 1) (0, v) q ↔ q = (0, v) ∨ q = (1, v)) ∧ (∀ u ∈ Set.Ioo (0 : ℝ) 1, Relation.EqvGen (glue 1) (u, v) q ↔ q = (u, v)) := by
    intro v hv q hq
    obtain ⟨hv0, hv1⟩ := hv
    have hmem : ∀ u : ℝ, 0 ≤ u → u ≤ 1 → (u, v) ∈ square 1 := fun u h1 h2 => ⟨⟨h1, h2⟩, ⟨hv0.le, hv1.le⟩⟩
    have hb : ∀ b : ℝ, (v = b ∨ (v = 0 ∧ b = 1) ∨ (v = 1 ∧ b = 0)) ↔ b = v := by
      intro b
      constructor
      · rintro (h | ⟨h, -⟩ | ⟨h, -⟩)
        · exact h.symm
        · linarith
        · linarith
      · intro h; left; exact h.symm
    constructor
    · rw [glue_eqvGen_iff 1 one_pos _ q (hmem 0 le_rfl zero_le_one) hq, κ_eq_iff 1 _ _ one_pos,
        κ_eq_iff 1 _ _ one_pos]
      obtain ⟨a, b⟩ := q
      simp only [Prod.mk.injEq]
      rw [hb b]
      constructor
      · rintro ⟨h | ⟨-, h⟩ | ⟨h, -⟩, hbv⟩
        · left; exact ⟨h.symm, hbv⟩
        · right; exact ⟨h, hbv⟩
        · norm_num at h
      · rintro (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩) <;> simp
    · intro u hu
      obtain ⟨hu0, hu1⟩ := hu
      rw [glue_eqvGen_iff 1 one_pos _ q (hmem u hu0.le hu1.le) hq, κ_eq_iff 1 _ _ one_pos,
        κ_eq_iff 1 _ _ one_pos]
      obtain ⟨a, b⟩ := q
      simp only [Prod.mk.injEq]
      rw [hb b]
      constructor
      · rintro ⟨h | ⟨h, -⟩ | ⟨h, -⟩, hbv⟩
        · exact ⟨h.symm, hbv⟩
        · linarith
        · linarith
      · rintro ⟨rfl, rfl⟩
        exact ⟨Or.inl rfl, by simp⟩
  exact @key (1/3) ((by norm_num)) ((1/2, 1/3)) ((by simp [KaluzaKlein.Sets.Quotients.square]; norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Quotients.edge_interior_classes
-- test: set-edge-interior-test.sage

end KaluzaKlein.Sets.Quotients
