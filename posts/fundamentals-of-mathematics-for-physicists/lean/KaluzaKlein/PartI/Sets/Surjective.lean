import Mathlib
import KaluzaKlein.PartI.Sets.Injective
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Maps" (surjective maps)

The claims of the blocks `def-surjective` to `exr-surjective-rotation-orbit` of
`tutorial/part-i-01-sets-relations-topology.qmd`: surjectivity as "image = codomain", the surjections between small
finite sets, odd powers, the square and the exponential, `sin` and `x² + 1`, the orbit map `SO(2) → S¹` (a bijection)
and the fibres of the orbit map `SO(3) → S²`, copies of the rotations about the `Z`-axis.
-/

namespace KaluzaKlein.Sets.Surjective

open Matrix KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Maps

/-- The rotation by `α` about the `Z`-axis. -/
noncomputable def Rz (α : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![Real.cos α, -Real.sin α, 0; Real.sin α, Real.cos α, 0; 0, 0, 1]

/-- The rotation group of the plane, `SO(2)`. -/
abbrev SO2 : Set (Matrix (Fin 2) (Fin 2) ℝ) := (Matrix.specialOrthogonalGroup (Fin 2) ℝ : Set _)

/-! ## def-surjective -/

/-- A map is surjective exactly when its image is the whole codomain. -/
theorem surjective_iff_range {X Y : Type} (f : X → Y) : Function.Surjective f ↔ Set.range f = Set.univ :=
  Set.range_eq_univ.symm

example : veriphy_instance% KaluzaKlein.Sets.Surjective.surjective_iff_range (Fin 2) (Fin 2) (id) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Surjective f ↔ Set.range f = Set.univ := by
    intro X Y f
    exact (Set.range_eq_univ.symm)
  exact @key (Fin 2) (Fin 2) (id)

example : veriphy_instance% KaluzaKlein.Sets.Surjective.surjective_iff_range (Fin 3) (Fin 2) (![0, 0, 1]) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Surjective f ↔ Set.range f = Set.univ := by
    intro X Y f
    exact (Set.range_eq_univ.symm)
  exact @key (Fin 3) (Fin 2) (![0, 0, 1])

example : veriphy_instance% KaluzaKlein.Sets.Surjective.surjective_iff_range (Fin 2) (Fin 3) (![0, 1]) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Surjective f ↔ Set.range f = Set.univ := by
    intro X Y f
    exact (Set.range_eq_univ.symm)
  exact @key (Fin 2) (Fin 3) (![0, 1])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.surjective_iff_range
-- test: set-surjective-iff-test.sage

/-! ## exm-surjective-finite -/

/-- `1, 2 ↦ x`, `3 ↦ y` is surjective onto a codomain whose elements are `x` and `y`. -/
theorem finite_map_surjective {β : Type} (x y : β) (h : ∀ b, b = x ∨ b = y) : Function.Surjective ![x, x, y] := by
  intro b
  rcases h b with rfl | rfl
  · exact ⟨0, rfl⟩
  · exact ⟨2, rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.Surjective.finite_map_surjective (Fin 2) (0) (1) ((by decide)) := by
  have key : ∀ {β : Type} (x y : β) (h : ∀ b, b = x ∨ b = y), Function.Surjective ![x, x, y] := by
    intro β x y h
    intro b
    rcases h b with rfl | rfl
    · exact ⟨0, rfl⟩
    · exact ⟨2, rfl⟩
  exact @key (Fin 2) (0) (1) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Surjective.finite_map_surjective (Bool) (true) (false) ((by decide)) := by
  have key : ∀ {β : Type} (x y : β) (h : ∀ b, b = x ∨ b = y), Function.Surjective ![x, x, y] := by
    intro β x y h
    intro b
    rcases h b with rfl | rfl
    · exact ⟨0, rfl⟩
    · exact ⟨2, rfl⟩
  exact @key (Bool) (true) (false) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Surjective.finite_map_surjective (Fin 2) (1) (0) ((by decide)) := by
  have key : ∀ {β : Type} (x y : β) (h : ∀ b, b = x ∨ b = y), Function.Surjective ![x, x, y] := by
    intro β x y h
    intro b
    rcases h b with rfl | rfl
    · exact ⟨0, rfl⟩
    · exact ⟨2, rfl⟩
  exact @key (Fin 2) (1) (0) ((by decide))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.finite_map_surjective
-- test: set-finite-surjective-test.sage

/-- A map to a two-element set that is not surjective is constant. -/
theorem not_surjective_two {m : ℕ} (f : Fin m → Fin 2) (hf : ¬ Function.Surjective f) :
    f = (fun _ => (0 : Fin 2)) ∨ f = (fun _ => (1 : Fin 2)) := by
  simp only [Function.Surjective, not_forall, not_exists] at hf
  obtain ⟨b, hb⟩ := hf
  fin_cases b
  · right; funext i; show f i = 1; have h := hb i; revert h; generalize f i = z; fin_cases z <;> simp
  · left; funext i; show f i = 0; have h := hb i; revert h; generalize f i = z; fin_cases z <;> simp

/-- Of the `2ᵐ` maps from an `m`-element set (`m ≥ 1`) to a two-element set, all but the two constant maps are
surjective: `8 - 2 = 6` for `m = 3`, `16 - 2 = 14` for `m = 4`. -/
theorem card_surjections_two (m : ℕ) (hm : 1 ≤ m) :
    (Finset.univ.filter fun f : Fin m → Fin 2 => Function.Surjective f).card = 2 ^ m - 2 := by
  have hc : (Finset.univ.filter fun f : Fin m → Fin 2 => ¬ Function.Surjective f) =
      {fun _ => (0 : Fin 2), fun _ => (1 : Fin 2)} := by
    ext f
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Finset.mem_singleton]
    constructor
    · exact not_surjective_two f
    · rintro (rfl | rfl) h
      · obtain ⟨i, hi⟩ := h 1; simp at hi
      · obtain ⟨i, hi⟩ := h 0; simp at hi
  have hne : (fun _ : Fin m => (0 : Fin 2)) ≠ fun _ => (1 : Fin 2) := fun h => by
    have := congrFun h ⟨0, hm⟩
    simp at this
  have h2 := Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin m → Fin 2)))
    (fun f : Fin m → Fin 2 => Function.Surjective f)
  rw [hc, Finset.card_pair hne, Finset.card_univ, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin] at h2
  omega

example : veriphy_instance% KaluzaKlein.Sets.Surjective.card_surjections_two (3) ((by norm_num)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Surjective.card_surjections_two (1) ((by norm_num)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Surjective.card_surjections_two (4) ((by norm_num)) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.card_surjections_two
-- test: set-card-surjections-two-test.sage

/-! ## exm-surjective-cube -/

/-- Odd powers are surjective onto `ℝ` (every real number has a cube root, `k = 1`). -/
theorem odd_pow_surjective (k : ℕ) : Function.Surjective (fun x : ℝ => x ^ (2 * k + 1)) := by
  intro y
  have hn : (2 * k + 1 : ℕ) ≠ 0 := by omega
  rcases le_or_gt 0 y with hy | hy
  · refine ⟨y ^ ((2 * k + 1 : ℕ) : ℝ)⁻¹, ?_⟩
    simp only
    rw [← Real.rpow_natCast, ← Real.rpow_mul hy, inv_mul_cancel₀ (by exact_mod_cast hn), Real.rpow_one]
  · refine ⟨-((-y) ^ ((2 * k + 1 : ℕ) : ℝ)⁻¹), ?_⟩
    simp only
    rw [Odd.neg_pow ⟨k, rfl⟩, ← Real.rpow_natCast, ← Real.rpow_mul (by linarith),
      inv_mul_cancel₀ (by exact_mod_cast hn), Real.rpow_one, neg_neg]

example : veriphy_instance% KaluzaKlein.Sets.Surjective.odd_pow_surjective (0) := by
  have key : ∀ (k : ℕ), Function.Surjective (fun x : ℝ => x ^ (2 * k + 1)) := by
    intro k
    intro y
    have hn : (2 * k + 1 : ℕ) ≠ 0 := by omega
    rcases le_or_gt 0 y with hy | hy
    · refine ⟨y ^ ((2 * k + 1 : ℕ) : ℝ)⁻¹, ?_⟩
      simp only
      rw [← Real.rpow_natCast, ← Real.rpow_mul hy, inv_mul_cancel₀ (by exact_mod_cast hn), Real.rpow_one]
    · refine ⟨-((-y) ^ ((2 * k + 1 : ℕ) : ℝ)⁻¹), ?_⟩
      simp only
      rw [Odd.neg_pow ⟨k, rfl⟩, ← Real.rpow_natCast, ← Real.rpow_mul (by linarith),
        inv_mul_cancel₀ (by exact_mod_cast hn), Real.rpow_one, neg_neg]
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Surjective.odd_pow_surjective (1) := by
  have key : ∀ (k : ℕ), Function.Surjective (fun x : ℝ => x ^ (2 * k + 1)) := by
    intro k
    intro y
    have hn : (2 * k + 1 : ℕ) ≠ 0 := by omega
    rcases le_or_gt 0 y with hy | hy
    · refine ⟨y ^ ((2 * k + 1 : ℕ) : ℝ)⁻¹, ?_⟩
      simp only
      rw [← Real.rpow_natCast, ← Real.rpow_mul hy, inv_mul_cancel₀ (by exact_mod_cast hn), Real.rpow_one]
    · refine ⟨-((-y) ^ ((2 * k + 1 : ℕ) : ℝ)⁻¹), ?_⟩
      simp only
      rw [Odd.neg_pow ⟨k, rfl⟩, ← Real.rpow_natCast, ← Real.rpow_mul (by linarith),
        inv_mul_cancel₀ (by exact_mod_cast hn), Real.rpow_one, neg_neg]
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Surjective.odd_pow_surjective (2) := by
  have key : ∀ (k : ℕ), Function.Surjective (fun x : ℝ => x ^ (2 * k + 1)) := by
    intro k
    intro y
    have hn : (2 * k + 1 : ℕ) ≠ 0 := by omega
    rcases le_or_gt 0 y with hy | hy
    · refine ⟨y ^ ((2 * k + 1 : ℕ) : ℝ)⁻¹, ?_⟩
      simp only
      rw [← Real.rpow_natCast, ← Real.rpow_mul hy, inv_mul_cancel₀ (by exact_mod_cast hn), Real.rpow_one]
    · refine ⟨-((-y) ^ ((2 * k + 1 : ℕ) : ℝ)⁻¹), ?_⟩
      simp only
      rw [Odd.neg_pow ⟨k, rfl⟩, ← Real.rpow_natCast, ← Real.rpow_mul (by linarith),
        inv_mul_cancel₀ (by exact_mod_cast hn), Real.rpow_one, neg_neg]
  exact @key (2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.odd_pow_surjective
-- test: set-odd-pow-test.sage

/-- `x ↦ a eˣ` (`a > 0`) has image the positive numbers: `exp` is not surjective onto `ℝ`. -/
theorem range_mul_exp (a : ℝ) (ha : 0 < a) : Set.range (fun x => a * Real.exp x) = Set.Ioi 0 := by
  ext y
  simp only [Set.mem_range, Set.mem_Ioi]
  constructor
  · rintro ⟨x, rfl⟩; positivity
  · intro hy
    refine ⟨Real.log (y / a), ?_⟩
    rw [Real.exp_log (div_pos hy ha)]; field_simp

example : veriphy_instance% KaluzaKlein.Sets.Surjective.range_mul_exp (1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 < a), Set.range (fun x => a * Real.exp x) = Set.Ioi 0 := by
    intro a ha
    ext y
    simp only [Set.mem_range, Set.mem_Ioi]
    constructor
    · rintro ⟨x, rfl⟩; positivity
    · intro hy
      refine ⟨Real.log (y / a), ?_⟩
      rw [Real.exp_log (div_pos hy ha)]; field_simp
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Surjective.range_mul_exp (2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 < a), Set.range (fun x => a * Real.exp x) = Set.Ioi 0 := by
    intro a ha
    ext y
    simp only [Set.mem_range, Set.mem_Ioi]
    constructor
    · rintro ⟨x, rfl⟩; positivity
    · intro hy
      refine ⟨Real.log (y / a), ?_⟩
      rw [Real.exp_log (div_pos hy ha)]; field_simp
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Surjective.range_mul_exp (1/2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 < a), Set.range (fun x => a * Real.exp x) = Set.Ioi 0 := by
    intro a ha
    ext y
    simp only [Set.mem_range, Set.mem_Ioi]
    constructor
    · rintro ⟨x, rfl⟩; positivity
    · intro hy
      refine ⟨Real.log (y / a), ?_⟩
      rw [Real.exp_log (div_pos hy ha)]; field_simp
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.range_mul_exp
-- test: set-range-exp-test.sage

/-! ## exr-surjective-count -/

/-- A map from a finite set to itself is surjective exactly when it is injective (a bijection). -/
theorem surjective_iff_injective_self (n : ℕ) (f : Fin n → Fin n) :
    Function.Surjective f ↔ Function.Injective f :=
  Finite.injective_iff_surjective.symm

example : veriphy_instance% KaluzaKlein.Sets.Surjective.surjective_iff_injective_self (3) (![1, 2, 0]) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Surjective.surjective_iff_injective_self (0) (![]) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Surjective.surjective_iff_injective_self (3) (![0, 0, 1]) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.surjective_iff_injective_self
-- test: set-surj-iff-inj-test.sage

/-- There are `n!` surjective maps from an `n`-element set to itself: `3! = 6`. -/
theorem card_surjections_self (n : ℕ) :
    (Finset.univ.filter fun f : Fin n → Fin n => Function.Surjective f).card = n.factorial := by
  have : (Finset.univ.filter fun f : Fin n → Fin n => Function.Surjective f) =
      Finset.univ.map ⟨fun σ : Equiv.Perm (Fin n) => (σ : Fin n → Fin n), DFunLike.coe_injective⟩ := by
    ext f
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_map, Function.Embedding.coeFn_mk]
    constructor
    · intro hf
      exact ⟨Equiv.ofBijective f ⟨Finite.injective_iff_surjective.2 hf, hf⟩, rfl⟩
    · rintro ⟨σ, rfl⟩
      exact σ.surjective
  rw [this, Finset.card_map, Finset.card_univ, Fintype.card_perm, Fintype.card_fin]

example : veriphy_instance% KaluzaKlein.Sets.Surjective.card_surjections_self (3) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Surjective.card_surjections_self (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Surjective.card_surjections_self (2) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.card_surjections_self
-- test: set-card-surj-self-test.sage

/-! ## exr-surjective-sin -/

/-- `x ↦ a sin x` (`a ≥ 0`) has image `[-a, a]`: `sin` has image `[-1, 1]`. -/
theorem range_mul_sin (a : ℝ) (ha : 0 ≤ a) : Set.range (fun x => a * Real.sin x) = Set.Icc (-a) a := by
  ext y
  simp only [Set.mem_range, Set.mem_Icc]
  constructor
  · rintro ⟨x, rfl⟩
    constructor <;> nlinarith [Real.neg_one_le_sin x, Real.sin_le_one x]
  · rintro ⟨h1, h2⟩
    rcases eq_or_lt_of_le ha with rfl | ha'
    · exact ⟨0, by linarith⟩
    · obtain ⟨x, -, hx⟩ := Real.bijOn_sin.surjOn
        (show y / a ∈ Set.Icc (-1 : ℝ) 1 from ⟨by rw [le_div_iff₀ ha']; linarith,
          by rw [div_le_iff₀ ha']; linarith⟩)
      exact ⟨x, by rw [hx]; field_simp⟩

example : veriphy_instance% KaluzaKlein.Sets.Surjective.range_mul_sin (1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 ≤ a), Set.range (fun x => a * Real.sin x) = Set.Icc (-a) a := by
    intro a ha
    ext y
    simp only [Set.mem_range, Set.mem_Icc]
    constructor
    · rintro ⟨x, rfl⟩
      constructor <;> nlinarith [Real.neg_one_le_sin x, Real.sin_le_one x]
    · rintro ⟨h1, h2⟩
      rcases eq_or_lt_of_le ha with rfl | ha'
      · exact ⟨0, by linarith⟩
      · obtain ⟨x, -, hx⟩ := Real.bijOn_sin.surjOn
          (show y / a ∈ Set.Icc (-1 : ℝ) 1 from ⟨by rw [le_div_iff₀ ha']; linarith,
            by rw [div_le_iff₀ ha']; linarith⟩)
        exact ⟨x, by rw [hx]; field_simp⟩
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Surjective.range_mul_sin (0) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 ≤ a), Set.range (fun x => a * Real.sin x) = Set.Icc (-a) a := by
    intro a ha
    ext y
    simp only [Set.mem_range, Set.mem_Icc]
    constructor
    · rintro ⟨x, rfl⟩
      constructor <;> nlinarith [Real.neg_one_le_sin x, Real.sin_le_one x]
    · rintro ⟨h1, h2⟩
      rcases eq_or_lt_of_le ha with rfl | ha'
      · exact ⟨0, by linarith⟩
      · obtain ⟨x, -, hx⟩ := Real.bijOn_sin.surjOn
          (show y / a ∈ Set.Icc (-1 : ℝ) 1 from ⟨by rw [le_div_iff₀ ha']; linarith,
            by rw [div_le_iff₀ ha']; linarith⟩)
        exact ⟨x, by rw [hx]; field_simp⟩
  exact @key (0) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Surjective.range_mul_sin (2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 ≤ a), Set.range (fun x => a * Real.sin x) = Set.Icc (-a) a := by
    intro a ha
    ext y
    simp only [Set.mem_range, Set.mem_Icc]
    constructor
    · rintro ⟨x, rfl⟩
      constructor <;> nlinarith [Real.neg_one_le_sin x, Real.sin_le_one x]
    · rintro ⟨h1, h2⟩
      rcases eq_or_lt_of_le ha with rfl | ha'
      · exact ⟨0, by linarith⟩
      · obtain ⟨x, -, hx⟩ := Real.bijOn_sin.surjOn
          (show y / a ∈ Set.Icc (-1 : ℝ) 1 from ⟨by rw [le_div_iff₀ ha']; linarith,
            by rw [div_le_iff₀ ha']; linarith⟩)
        exact ⟨x, by rw [hx]; field_simp⟩
  exact @key (2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.range_mul_sin
-- test: set-range-sin-test.sage

/-- `x ↦ a sin x` (`a ≥ 0`) is not surjective onto `ℝ` (it misses `a + 1`). -/
theorem mul_sin_not_surjective (a : ℝ) (ha : 0 ≤ a) : ¬ Function.Surjective (fun x => a * Real.sin x) := by
  intro h
  obtain ⟨x, hx⟩ := h (a + 1)
  nlinarith [Real.sin_le_one x]

example : veriphy_instance% KaluzaKlein.Sets.Surjective.mul_sin_not_surjective (1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 ≤ a), ¬ Function.Surjective (fun x => a * Real.sin x) := by
    intro a ha
    intro h
    obtain ⟨x, hx⟩ := h (a + 1)
    nlinarith [Real.sin_le_one x]
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Surjective.mul_sin_not_surjective (0) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 ≤ a), ¬ Function.Surjective (fun x => a * Real.sin x) := by
    intro a ha
    intro h
    obtain ⟨x, hx⟩ := h (a + 1)
    nlinarith [Real.sin_le_one x]
  exact @key (0) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Surjective.mul_sin_not_surjective (3) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 ≤ a), ¬ Function.Surjective (fun x => a * Real.sin x) := by
    intro a ha
    intro h
    obtain ⟨x, hx⟩ := h (a + 1)
    nlinarith [Real.sin_le_one x]
  exact @key (3) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.mul_sin_not_surjective
-- test: set-sin-not-surj-test.sage

/-- Every map is surjective onto its image. -/
theorem surjective_onto_range {X Y : Type} (f : X → Y) : Function.Surjective (Set.rangeFactorization f) :=
  Set.rangeFactorization_surjective

example : veriphy_instance% KaluzaKlein.Sets.Surjective.surjective_onto_range (Fin 2) (Fin 2) (id) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Surjective (Set.rangeFactorization f) := by
    intro X Y f
    exact (Set.rangeFactorization_surjective)
  exact @key (Fin 2) (Fin 2) (id)

example : veriphy_instance% KaluzaKlein.Sets.Surjective.surjective_onto_range (ℝ) (ℝ) (Real.sin) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Surjective (Set.rangeFactorization f) := by
    intro X Y f
    exact (Set.rangeFactorization_surjective)
  exact @key (ℝ) (ℝ) (Real.sin)

example : veriphy_instance% KaluzaKlein.Sets.Surjective.surjective_onto_range (ℝ) (ℝ) (fun x => x ^ 2 + 1) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Surjective (Set.rangeFactorization f) := by
    intro X Y f
    exact (Set.rangeFactorization_surjective)
  exact @key (ℝ) (ℝ) (fun x => x ^ 2 + 1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.surjective_onto_range
-- test: set-surj-onto-range-test.sage

/-! ## exr-surjective-rotation-orbit -/

/-- A plane rotation is determined by the image `(a, b)` of `(1, 0)`: it is `!![a, -b; b, a]`. -/
theorem SO2_eq_of_column (g : Matrix (Fin 2) (Fin 2) ℝ) (hg : g ∈ SO2) :
    g = !![g 0 0, -g 1 0; g 1 0, g 0 0] := by
  rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff'] at hg
  obtain ⟨ho, hd⟩ := hg
  have e00 := congrFun (congrFun ho 0) 0
  have e01 := congrFun (congrFun ho 0) 1
  have e11 := congrFun (congrFun ho 1) 1
  simp [Matrix.mul_apply, Fin.sum_univ_succ] at e00 e01 e11
  rw [Matrix.det_fin_two] at hd
  have h1 : g 0 1 = -g 1 0 := by nlinarith [sq_nonneg (g 0 1 + g 1 0), sq_nonneg (g 1 1 - g 0 0)]
  have h2 : g 1 1 = g 0 0 := by nlinarith [sq_nonneg (g 0 1 + g 1 0), sq_nonneg (g 1 1 - g 0 0)]
  ext i j; fin_cases i <;> fin_cases j <;> simp [h1, h2]

/-- `g ↦ g · (1, 0)` from `SO(2)` onto the circle is a bijection: each point of the circle is the image of exactly one
plane rotation. -/
theorem SO2_orbit_bijective (p : ℝ × ℝ) (hp : p ∈ circle) : ∃! g, g ∈ SO2 ∧ (g 0 0, g 1 0) = p := by
  simp only [circle, Set.mem_setOf_eq] at hp
  refine ⟨!![p.1, -p.2; p.2, p.1], ⟨?_, by simp⟩, ?_⟩
  · rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
    refine ⟨?_, ?_⟩
    · ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> linarith
    · rw [Matrix.det_fin_two_of]; linarith
  · rintro g ⟨hg, hgp⟩
    rw [SO2_eq_of_column g hg, ← hgp]

example : veriphy_instance% KaluzaKlein.Sets.Surjective.SO2_orbit_bijective ((1, 0)) ((by norm_num [KaluzaKlein.Sets.Sets.circle])) := by
  have key : ∀ (p : ℝ × ℝ) (hp : p ∈ circle), ∃! g, g ∈ SO2 ∧ (g 0 0, g 1 0) = p := by
    intro p hp
    simp only [circle, Set.mem_setOf_eq] at hp
    refine ⟨!![p.1, -p.2; p.2, p.1], ⟨?_, by simp⟩, ?_⟩
    · rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
      refine ⟨?_, ?_⟩
      · ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> linarith
      · rw [Matrix.det_fin_two_of]; linarith
    · rintro g ⟨hg, hgp⟩
      rw [SO2_eq_of_column g hg, ← hgp]
  exact @key ((1, 0)) ((by norm_num [KaluzaKlein.Sets.Sets.circle]))

example : veriphy_instance% KaluzaKlein.Sets.Surjective.SO2_orbit_bijective ((0, 1)) ((by norm_num [KaluzaKlein.Sets.Sets.circle])) := by
  have key : ∀ (p : ℝ × ℝ) (hp : p ∈ circle), ∃! g, g ∈ SO2 ∧ (g 0 0, g 1 0) = p := by
    intro p hp
    simp only [circle, Set.mem_setOf_eq] at hp
    refine ⟨!![p.1, -p.2; p.2, p.1], ⟨?_, by simp⟩, ?_⟩
    · rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
      refine ⟨?_, ?_⟩
      · ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> linarith
      · rw [Matrix.det_fin_two_of]; linarith
    · rintro g ⟨hg, hgp⟩
      rw [SO2_eq_of_column g hg, ← hgp]
  exact @key ((0, 1)) ((by norm_num [KaluzaKlein.Sets.Sets.circle]))

example : veriphy_instance% KaluzaKlein.Sets.Surjective.SO2_orbit_bijective ((3/5, 4/5)) ((by norm_num [KaluzaKlein.Sets.Sets.circle])) := by
  have key : ∀ (p : ℝ × ℝ) (hp : p ∈ circle), ∃! g, g ∈ SO2 ∧ (g 0 0, g 1 0) = p := by
    intro p hp
    simp only [circle, Set.mem_setOf_eq] at hp
    refine ⟨!![p.1, -p.2; p.2, p.1], ⟨?_, by simp⟩, ?_⟩
    · rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
      refine ⟨?_, ?_⟩
      · ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> linarith
      · rw [Matrix.det_fin_two_of]; linarith
    · rintro g ⟨hg, hgp⟩
      rw [SO2_eq_of_column g hg, ← hgp]
  exact @key ((3/5, 4/5)) ((by norm_num [KaluzaKlein.Sets.Sets.circle]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.SO2_orbit_bijective
-- test: set-so2-orbit-test.sage

/-- The rotations fixing the north pole are exactly the rotations `Rz α` about the `Z`-axis, a copy of `SO(2)`. -/
theorem stabilizer_eq (h : Matrix (Fin 3) (Fin 3) ℝ) : (h ∈ SO3 ∧ orbit h = northPole) ↔ ∃ α, h = Rz α := by
  constructor
  · rintro ⟨hh, ho⟩
    rw [mem_SO3_iff] at hh
    obtain ⟨hT, hd⟩ := hh
    have hT' : h * hᵀ = 1 := mul_eq_one_comm.1 hT
    have c0 := congrFun ho 0
    have c1 := congrFun ho 1
    have c2 := congrFun ho 2
    simp [orbit, northPole, Matrix.mulVec, dotProduct, Fin.sum_univ_succ] at c0 c1 c2
    have r22 := congrFun (congrFun hT' 2) 2
    simp [Matrix.mul_apply, Fin.sum_univ_succ, c2] at r22
    have h20 : h 2 0 = 0 := by nlinarith [sq_nonneg (h 2 0), sq_nonneg (h 2 1)]
    have h21 : h 2 1 = 0 := by nlinarith [sq_nonneg (h 2 0), sq_nonneg (h 2 1)]
    have e00 := congrFun (congrFun hT 0) 0
    have e01 := congrFun (congrFun hT 0) 1
    have e11 := congrFun (congrFun hT 1) 1
    simp [Matrix.mul_apply, Fin.sum_univ_succ, h20, h21] at e00 e01 e11
    rw [Matrix.det_fin_three, c0, c1, c2, h20, h21] at hd
    simp at hd
    have h01 : h 0 1 = -h 1 0 := by nlinarith [sq_nonneg (h 0 1 + h 1 0), sq_nonneg (h 1 1 - h 0 0)]
    have h11 : h 1 1 = h 0 0 := by nlinarith [sq_nonneg (h 0 1 + h 1 0), sq_nonneg (h 1 1 - h 0 0)]
    obtain ⟨φ, hφ⟩ := (range_wrap 1 zero_le_one).symm ▸ (show (h 0 0, h 1 0) ∈
      {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1 ^ 2} by simp; nlinarith)
    simp only [Prod.mk.injEq, one_mul] at hφ
    refine ⟨φ, ?_⟩
    ext i j; fin_cases i <;> fin_cases j <;> simp [Rz, hφ.1, hφ.2, h01, h11, h20, h21, c0, c1, c2]
  · rintro ⟨α, rfl⟩
    refine ⟨?_, ?_⟩
    · rw [mem_SO3_iff]
      refine ⟨?_, ?_⟩
      · ext i j; fin_cases i <;> fin_cases j <;> simp [Rz, Matrix.mul_apply, Fin.sum_univ_succ] <;>
          nlinarith [Real.sin_sq_add_cos_sq α]
      · simp [Rz, Matrix.det_fin_three]; nlinarith [Real.sin_sq_add_cos_sq α]
    · ext i; fin_cases i <;> simp [Rz, orbit, northPole, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.Sets.Surjective.stabilizer_eq (1) := by
  refine ⟨fun _ => ⟨0, ?_⟩, fun _ => ⟨(Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem, ?_⟩⟩
  · ext i j; fin_cases i <;> fin_cases j <;> simp [KaluzaKlein.Sets.Surjective.Rz]
  · ext i; fin_cases i <;> simp [KaluzaKlein.Sets.Maps.orbit, KaluzaKlein.Sets.Sets.northPole]

example : veriphy_instance% KaluzaKlein.Sets.Surjective.stabilizer_eq (!![0, -1, 0; 1, 0, 0; 0, 0, 1]) := by
  refine ⟨fun _ => ⟨Real.pi / 2, ?_⟩, fun _ => ⟨?_, ?_⟩⟩
  · ext i j; fin_cases i <;> fin_cases j <;> simp [KaluzaKlein.Sets.Surjective.Rz]
  · rw [KaluzaKlein.Sets.Sets.mem_SO3_iff]
    refine ⟨?_, by simp [Matrix.det_fin_three]⟩
    ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]
  · ext i; fin_cases i <;> simp [KaluzaKlein.Sets.Maps.orbit, KaluzaKlein.Sets.Sets.northPole, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.Sets.Surjective.stabilizer_eq (!![1, 0, 0; 0, 0, -1; 0, 1, 0]) := by
  constructor
  · rintro ⟨-, h⟩
    have := congrFun h 1
    simp [KaluzaKlein.Sets.Maps.orbit, KaluzaKlein.Sets.Sets.northPole, Matrix.mulVec, dotProduct, Fin.sum_univ_succ] at this
  · rintro ⟨α, h⟩
    have := congrFun (congrFun h 2) 2
    simp [KaluzaKlein.Sets.Surjective.Rz] at this

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.stabilizer_eq
-- test: set-stabilizer-test.sage

/-- The rotations sending the pole where `g₀` does are the `g₀ h` with `h` fixing the pole: the preimage of every
point is a copy of the rotations about the `Z`-axis. -/
theorem orbit_fiber (g₀ g : Matrix (Fin 3) (Fin 3) ℝ) (hg₀ : g₀ ∈ SO3) :
    (g ∈ SO3 ∧ orbit g = orbit g₀) ↔ ∃ α, g = g₀ * Rz α := by
  have hg₀' := hg₀
  rw [mem_SO3_iff] at hg₀'
  have hinv : g₀ * g₀ᵀ = 1 := mul_eq_one_comm.1 hg₀'.1
  constructor
  · rintro ⟨hg, ho⟩
    have hmem : g₀ᵀ * g ∈ SO3 := by
      have hg' := hg
      rw [mem_SO3_iff] at hg' ⊢
      refine ⟨?_, ?_⟩
      · rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc g₀,
          hinv, Matrix.one_mul, hg'.1]
      · rw [Matrix.det_mul, Matrix.det_transpose, hg'.2, hg₀'.2, mul_one]
    have hfix : orbit (g₀ᵀ * g) = northPole := by
      simp only [orbit] at ho ⊢
      rw [← Matrix.mulVec_mulVec, ho, Matrix.mulVec_mulVec, hg₀'.1, Matrix.one_mulVec]
    obtain ⟨α, hα⟩ := (stabilizer_eq (g₀ᵀ * g)).1 ⟨hmem, hfix⟩
    refine ⟨α, ?_⟩
    rw [← hα, ← Matrix.mul_assoc, hinv, Matrix.one_mul]
  · rintro ⟨α, rfl⟩
    have hR := (stabilizer_eq (Rz α)).2 ⟨α, rfl⟩
    refine ⟨?_, ?_⟩
    · exact (Matrix.specialOrthogonalGroup (Fin 3) ℝ).mul_mem hg₀ hR.1
    · simp only [orbit] at hR ⊢
      rw [← Matrix.mulVec_mulVec, hR.2]

example : veriphy_instance% KaluzaKlein.Sets.Surjective.orbit_fiber (1) (1) ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (g₀ g : Matrix (Fin 3) (Fin 3) ℝ) (hg₀ : g₀ ∈ SO3), (g ∈ SO3 ∧ orbit g = orbit g₀) ↔ ∃ α, g = g₀ * Rz α := by
    intro g₀ g hg₀
    have hg₀' := hg₀
    rw [mem_SO3_iff] at hg₀'
    have hinv : g₀ * g₀ᵀ = 1 := mul_eq_one_comm.1 hg₀'.1
    constructor
    · rintro ⟨hg, ho⟩
      have hmem : g₀ᵀ * g ∈ SO3 := by
        have hg' := hg
        rw [mem_SO3_iff] at hg' ⊢
        refine ⟨?_, ?_⟩
        · rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc g₀,
            hinv, Matrix.one_mul, hg'.1]
        · rw [Matrix.det_mul, Matrix.det_transpose, hg'.2, hg₀'.2, mul_one]
      have hfix : orbit (g₀ᵀ * g) = northPole := by
        simp only [orbit] at ho ⊢
        rw [← Matrix.mulVec_mulVec, ho, Matrix.mulVec_mulVec, hg₀'.1, Matrix.one_mulVec]
      obtain ⟨α, hα⟩ := (stabilizer_eq (g₀ᵀ * g)).1 ⟨hmem, hfix⟩
      refine ⟨α, ?_⟩
      rw [← hα, ← Matrix.mul_assoc, hinv, Matrix.one_mul]
    · rintro ⟨α, rfl⟩
      have hR := (stabilizer_eq (Rz α)).2 ⟨α, rfl⟩
      refine ⟨?_, ?_⟩
      · exact (Matrix.specialOrthogonalGroup (Fin 3) ℝ).mul_mem hg₀ hR.1
      · simp only [orbit] at hR ⊢
        rw [← Matrix.mulVec_mulVec, hR.2]
  exact @key (1) (1) ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem)

example : veriphy_instance% KaluzaKlein.Sets.Surjective.orbit_fiber (1) (!![0, -1, 0; 1, 0, 0; 0, 0, 1]) ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (g₀ g : Matrix (Fin 3) (Fin 3) ℝ) (hg₀ : g₀ ∈ SO3), (g ∈ SO3 ∧ orbit g = orbit g₀) ↔ ∃ α, g = g₀ * Rz α := by
    intro g₀ g hg₀
    have hg₀' := hg₀
    rw [mem_SO3_iff] at hg₀'
    have hinv : g₀ * g₀ᵀ = 1 := mul_eq_one_comm.1 hg₀'.1
    constructor
    · rintro ⟨hg, ho⟩
      have hmem : g₀ᵀ * g ∈ SO3 := by
        have hg' := hg
        rw [mem_SO3_iff] at hg' ⊢
        refine ⟨?_, ?_⟩
        · rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc g₀,
            hinv, Matrix.one_mul, hg'.1]
        · rw [Matrix.det_mul, Matrix.det_transpose, hg'.2, hg₀'.2, mul_one]
      have hfix : orbit (g₀ᵀ * g) = northPole := by
        simp only [orbit] at ho ⊢
        rw [← Matrix.mulVec_mulVec, ho, Matrix.mulVec_mulVec, hg₀'.1, Matrix.one_mulVec]
      obtain ⟨α, hα⟩ := (stabilizer_eq (g₀ᵀ * g)).1 ⟨hmem, hfix⟩
      refine ⟨α, ?_⟩
      rw [← hα, ← Matrix.mul_assoc, hinv, Matrix.one_mul]
    · rintro ⟨α, rfl⟩
      have hR := (stabilizer_eq (Rz α)).2 ⟨α, rfl⟩
      refine ⟨?_, ?_⟩
      · exact (Matrix.specialOrthogonalGroup (Fin 3) ℝ).mul_mem hg₀ hR.1
      · simp only [orbit] at hR ⊢
        rw [← Matrix.mulVec_mulVec, hR.2]
  exact @key (1) (!![0, -1, 0; 1, 0, 0; 0, 0, 1]) ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem)

example : veriphy_instance% KaluzaKlein.Sets.Surjective.orbit_fiber (!![1, 0, 0; 0, 0, -1; 0, 1, 0]) (!![1, 0, 0; 0, 0, -1; 0, 1, 0]) ((by rw [KaluzaKlein.Sets.Sets.mem_SO3_iff]; exact ⟨by ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ], by simp [Matrix.det_fin_three]⟩)) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (g₀ g : Matrix (Fin 3) (Fin 3) ℝ) (hg₀ : g₀ ∈ SO3), (g ∈ SO3 ∧ orbit g = orbit g₀) ↔ ∃ α, g = g₀ * Rz α := by
    intro g₀ g hg₀
    have hg₀' := hg₀
    rw [mem_SO3_iff] at hg₀'
    have hinv : g₀ * g₀ᵀ = 1 := mul_eq_one_comm.1 hg₀'.1
    constructor
    · rintro ⟨hg, ho⟩
      have hmem : g₀ᵀ * g ∈ SO3 := by
        have hg' := hg
        rw [mem_SO3_iff] at hg' ⊢
        refine ⟨?_, ?_⟩
        · rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc g₀,
            hinv, Matrix.one_mul, hg'.1]
        · rw [Matrix.det_mul, Matrix.det_transpose, hg'.2, hg₀'.2, mul_one]
      have hfix : orbit (g₀ᵀ * g) = northPole := by
        simp only [orbit] at ho ⊢
        rw [← Matrix.mulVec_mulVec, ho, Matrix.mulVec_mulVec, hg₀'.1, Matrix.one_mulVec]
      obtain ⟨α, hα⟩ := (stabilizer_eq (g₀ᵀ * g)).1 ⟨hmem, hfix⟩
      refine ⟨α, ?_⟩
      rw [← hα, ← Matrix.mul_assoc, hinv, Matrix.one_mul]
    · rintro ⟨α, rfl⟩
      have hR := (stabilizer_eq (Rz α)).2 ⟨α, rfl⟩
      refine ⟨?_, ?_⟩
      · exact (Matrix.specialOrthogonalGroup (Fin 3) ℝ).mul_mem hg₀ hR.1
      · simp only [orbit] at hR ⊢
        rw [← Matrix.mulVec_mulVec, hR.2]
  exact @key (!![1, 0, 0; 0, 0, -1; 0, 1, 0]) (!![1, 0, 0; 0, 0, -1; 0, 1, 0]) ((by rw [KaluzaKlein.Sets.Sets.mem_SO3_iff]; exact ⟨by ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ], by simp [Matrix.det_fin_three]⟩))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Surjective.orbit_fiber
-- test: set-orbit-fiber-test.sage

end KaluzaKlein.Sets.Surjective
