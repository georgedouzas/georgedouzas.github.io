import Mathlib
import KaluzaKlein.PartI.Sets.Surjective
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Maps" (bijections)

The claims of the blocks `def-bijection` to `exr-bijection-stereographic` of
`tutorial/part-i-01-sets-relations-topology.qmd`: bijections and sizes, permutations, bijections of the line (odd
powers, affine maps, `exp`, `tan`), the angle on the circle minus a point, the permutations of four elements fixing one,
the affine map on `ℝ` and on `ℤ`, and stereographic projection with its inverse.
-/

namespace KaluzaKlein.Sets.Bijective

open KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Maps KaluzaKlein.Sets.Injective

/-- Stereographic projection from the north pole, `(X, Y, Z) ↦ (X, Y)/(1 - Z)`. -/
noncomputable def stereo (p : Fin 3 → ℝ) : ℝ × ℝ := (p 0 / (1 - p 2), p 1 / (1 - p 2))

/-- Its inverse, `(u, v) ↦ (2u, 2v, u² + v² - 1)/(u² + v² + 1)`. -/
noncomputable def stereoInv (q : ℝ × ℝ) : Fin 3 → ℝ :=
  ![2 * q.1 / (q.1 ^ 2 + q.2 ^ 2 + 1), 2 * q.2 / (q.1 ^ 2 + q.2 ^ 2 + 1),
    (q.1 ^ 2 + q.2 ^ 2 - 1) / (q.1 ^ 2 + q.2 ^ 2 + 1)]

/-! ## def-bijection -/

/-- A map is a bijection exactly when every element of the codomain is the value of exactly one element. -/
theorem bijective_iff_existsUnique {X Y : Type} (f : X → Y) : Function.Bijective f ↔ ∀ y, ∃! x, f x = y :=
  Function.bijective_iff_existsUnique f

example : veriphy_instance% KaluzaKlein.Sets.Bijective.bijective_iff_existsUnique (Fin 2) (Fin 2) (id) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Bijective f ↔ ∀ y, ∃! x, f x = y := by
    intro X Y f
    exact (Function.bijective_iff_existsUnique f)
  exact @key (Fin 2) (Fin 2) (id)

example : veriphy_instance% KaluzaKlein.Sets.Bijective.bijective_iff_existsUnique (ℤ) (ℤ) (fun k => k + 1) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Bijective f ↔ ∀ y, ∃! x, f x = y := by
    intro X Y f
    exact (Function.bijective_iff_existsUnique f)
  exact @key (ℤ) (ℤ) (fun k => k + 1)

example : veriphy_instance% KaluzaKlein.Sets.Bijective.bijective_iff_existsUnique (Fin 3) (Fin 2) (![0, 0, 1]) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Bijective f ↔ ∀ y, ∃! x, f x = y := by
    intro X Y f
    exact (Function.bijective_iff_existsUnique f)
  exact @key (Fin 3) (Fin 2) (![0, 0, 1])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Bijective.bijective_iff_existsUnique
-- test: set-bijective-iff-test.sage

/-- Finite sets related by a bijection have the same number of elements. -/
theorem card_eq_of_bijective {X Y : Type} [Fintype X] [Fintype Y] (f : X → Y) (hf : Function.Bijective f) :
    Fintype.card X = Fintype.card Y :=
  Fintype.card_of_bijective hf

example : veriphy_instance% KaluzaKlein.Sets.Bijective.card_eq_of_bijective (Fin 3) (Fin 3) (inferInstance) (inferInstance) (id) (Function.bijective_id) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Bijective.card_eq_of_bijective (Bool) (Fin 2) (inferInstance) (inferInstance) (fun b => if b then 1 else 0) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Bijective.card_eq_of_bijective (Fin 3) (Fin 3) (inferInstance) (inferInstance) (![2, 0, 1]) ((by decide)) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Bijective.card_eq_of_bijective
-- test: set-card-bijective-test.sage

/-! ## exm-bijection-finite -/

/-- `1 ↦ x`, `2 ↦ y`, `3 ↦ z` is a bijection onto `{x, y, z}` (distinct labels exhausting the codomain). -/
theorem three_bijective {β : Type} (x y z : β) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
    (h : ∀ b, b = x ∨ b = y ∨ b = z) : Function.Bijective ![x, y, z] := by
  refine ⟨fun i j hij => ?_, fun b => ?_⟩
  · fin_cases i <;> fin_cases j <;> simp_all [eq_comm]
  · rcases h b with rfl | rfl | rfl
    · exact ⟨0, rfl⟩
    · exact ⟨1, rfl⟩
    · exact ⟨2, rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.Bijective.three_bijective (Fin 3) (0) (1) (2) ((by decide)) ((by decide)) ((by decide)) ((by decide)) := by
  have key : ∀ {β : Type} (x y z : β) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) (h : ∀ b, b = x ∨ b = y ∨ b = z), Function.Bijective ![x, y, z] := by
    intro β x y z hxy hxz hyz h
    refine ⟨fun i j hij => ?_, fun b => ?_⟩
    · fin_cases i <;> fin_cases j <;> simp_all [eq_comm]
    · rcases h b with rfl | rfl | rfl
      · exact ⟨0, rfl⟩
      · exact ⟨1, rfl⟩
      · exact ⟨2, rfl⟩
  exact @key (Fin 3) (0) (1) (2) ((by decide)) ((by decide)) ((by decide)) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Bijective.three_bijective (Fin 3) (2) (0) (1) ((by decide)) ((by decide)) ((by decide)) ((by decide)) := by
  have key : ∀ {β : Type} (x y z : β) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) (h : ∀ b, b = x ∨ b = y ∨ b = z), Function.Bijective ![x, y, z] := by
    intro β x y z hxy hxz hyz h
    refine ⟨fun i j hij => ?_, fun b => ?_⟩
    · fin_cases i <;> fin_cases j <;> simp_all [eq_comm]
    · rcases h b with rfl | rfl | rfl
      · exact ⟨0, rfl⟩
      · exact ⟨1, rfl⟩
      · exact ⟨2, rfl⟩
  exact @key (Fin 3) (2) (0) (1) ((by decide)) ((by decide)) ((by decide)) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Bijective.three_bijective (Fin 3) (1) (2) (0) ((by decide)) ((by decide)) ((by decide)) ((by decide)) := by
  have key : ∀ {β : Type} (x y z : β) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) (h : ∀ b, b = x ∨ b = y ∨ b = z), Function.Bijective ![x, y, z] := by
    intro β x y z hxy hxz hyz h
    refine ⟨fun i j hij => ?_, fun b => ?_⟩
    · fin_cases i <;> fin_cases j <;> simp_all [eq_comm]
    · rcases h b with rfl | rfl | rfl
      · exact ⟨0, rfl⟩
      · exact ⟨1, rfl⟩
      · exact ⟨2, rfl⟩
  exact @key (Fin 3) (1) (2) (0) ((by decide)) ((by decide)) ((by decide)) ((by decide))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Bijective.three_bijective
-- test: set-three-bijective-test.sage

/-- There are `n!` permutations of an `n`-element set: `3! = 6`, `4! = 24`. -/
theorem card_perm (n : ℕ) : Fintype.card (Equiv.Perm (Fin n)) = n.factorial := by
  rw [Fintype.card_perm, Fintype.card_fin]

example : veriphy_instance% KaluzaKlein.Sets.Bijective.card_perm (3) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Bijective.card_perm (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Bijective.card_perm (4) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Bijective.card_perm
-- test: set-card-perm-test.sage

/-! ## exm-bijection-cube -/

/-- Odd powers are bijections of `ℝ` (`x³` for `k = 1`). -/
theorem odd_pow_bijective (k : ℕ) : Function.Bijective (fun x : ℝ => x ^ (2 * k + 1)) :=
  ⟨(Odd.strictMono_pow (⟨k, rfl⟩ : Odd (2 * k + 1))).injective, Surjective.odd_pow_surjective k⟩

example : veriphy_instance% KaluzaKlein.Sets.Bijective.odd_pow_bijective (0) := by
  have key : ∀ (k : ℕ), Function.Bijective (fun x : ℝ => x ^ (2 * k + 1)) := by
    intro k
    exact (⟨(Odd.strictMono_pow (⟨k, rfl⟩ : Odd (2 * k + 1))).injective, Surjective.odd_pow_surjective k⟩)
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Bijective.odd_pow_bijective (1) := by
  have key : ∀ (k : ℕ), Function.Bijective (fun x : ℝ => x ^ (2 * k + 1)) := by
    intro k
    exact (⟨(Odd.strictMono_pow (⟨k, rfl⟩ : Odd (2 * k + 1))).injective, Surjective.odd_pow_surjective k⟩)
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Bijective.odd_pow_bijective (2) := by
  have key : ∀ (k : ℕ), Function.Bijective (fun x : ℝ => x ^ (2 * k + 1)) := by
    intro k
    exact (⟨(Odd.strictMono_pow (⟨k, rfl⟩ : Odd (2 * k + 1))).injective, Surjective.odd_pow_surjective k⟩)
  exact @key (2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Bijective.odd_pow_bijective
-- test: set-odd-pow-bijective-test.sage

/-- `x ↦ ax + b` (`a ≠ 0`) is a bijection of `ℝ`: `2x + 1` among them. -/
theorem affine_bijective (a b : ℝ) (ha : a ≠ 0) : Function.Bijective (fun x : ℝ => a * x + b) :=
  ⟨fun x y h => mul_left_cancel₀ ha (by simpa using h), fun y => ⟨(y - b) / a, by field_simp; ring⟩⟩

example : veriphy_instance% KaluzaKlein.Sets.Bijective.affine_bijective (2) (1) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (ha : a ≠ 0), Function.Bijective (fun x : ℝ => a * x + b) := by
    intro a b ha
    exact (⟨fun x y h => mul_left_cancel₀ ha (by simpa using h), fun y => ⟨(y - b) / a, by field_simp; ring⟩⟩)
  exact @key (2) (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Bijective.affine_bijective (1) (0) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (ha : a ≠ 0), Function.Bijective (fun x : ℝ => a * x + b) := by
    intro a b ha
    exact (⟨fun x y h => mul_left_cancel₀ ha (by simpa using h), fun y => ⟨(y - b) / a, by field_simp; ring⟩⟩)
  exact @key (1) (0) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Bijective.affine_bijective (-3) (5) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (ha : a ≠ 0), Function.Bijective (fun x : ℝ => a * x + b) := by
    intro a b ha
    exact (⟨fun x y h => mul_left_cancel₀ ha (by simpa using h), fun y => ⟨(y - b) / a, by field_simp; ring⟩⟩)
  exact @key (-3) (5) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Bijective.affine_bijective
-- test: set-affine-bijective-test.sage

/-- `x ↦ a eˣ` (`a > 0`) is a bijection of `ℝ` onto the positive numbers: `exp` for `a = 1`. -/
theorem mul_exp_bijOn (a : ℝ) (ha : 0 < a) : Set.BijOn (fun x => a * Real.exp x) Set.univ (Set.Ioi 0) := by
  refine ⟨fun x _ => mul_pos ha (Real.exp_pos x), fun x _ y _ h => ?_, fun y hy => ?_⟩
  · exact Real.exp_injective (mul_left_cancel₀ ha.ne' h)
  · obtain ⟨x, hx⟩ := (Surjective.range_mul_exp a ha).symm ▸ hy
    exact ⟨x, Set.mem_univ _, hx⟩

example : veriphy_instance% KaluzaKlein.Sets.Bijective.mul_exp_bijOn (1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 < a), Set.BijOn (fun x => a * Real.exp x) Set.univ (Set.Ioi 0) := by
    intro a ha
    refine ⟨fun x _ => mul_pos ha (Real.exp_pos x), fun x _ y _ h => ?_, fun y hy => ?_⟩
    · exact Real.exp_injective (mul_left_cancel₀ ha.ne' h)
    · obtain ⟨x, hx⟩ := (Surjective.range_mul_exp a ha).symm ▸ hy
      exact ⟨x, Set.mem_univ _, hx⟩
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Bijective.mul_exp_bijOn (2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 < a), Set.BijOn (fun x => a * Real.exp x) Set.univ (Set.Ioi 0) := by
    intro a ha
    refine ⟨fun x _ => mul_pos ha (Real.exp_pos x), fun x _ y _ h => ?_, fun y hy => ?_⟩
    · exact Real.exp_injective (mul_left_cancel₀ ha.ne' h)
    · obtain ⟨x, hx⟩ := (Surjective.range_mul_exp a ha).symm ▸ hy
      exact ⟨x, Set.mem_univ _, hx⟩
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Bijective.mul_exp_bijOn (1/2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 < a), Set.BijOn (fun x => a * Real.exp x) Set.univ (Set.Ioi 0) := by
    intro a ha
    refine ⟨fun x _ => mul_pos ha (Real.exp_pos x), fun x _ y _ h => ?_, fun y hy => ?_⟩
    · exact Real.exp_injective (mul_left_cancel₀ ha.ne' h)
    · obtain ⟨x, hx⟩ := (Surjective.range_mul_exp a ha).symm ▸ hy
      exact ⟨x, Set.mem_univ _, hx⟩
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Bijective.mul_exp_bijOn
-- test: set-exp-bijon-test.sage

/-- `x ↦ a tan x` (`a ≠ 0`) is a bijection of `(-π/2, π/2)` onto `ℝ`: `tan` for `a = 1`. -/
theorem mul_tan_bijOn (a : ℝ) (ha : a ≠ 0) :
    Set.BijOn (fun x => a * Real.tan x) (Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)) Set.univ := by
  refine ⟨fun _ _ => Set.mem_univ _, fun x hx y hy h => ?_, fun y _ => ?_⟩
  · have h' : Real.tan x = Real.tan y := mul_left_cancel₀ ha h
    exact Real.strictMonoOn_tan.injOn hx hy h'
  · refine ⟨Real.arctan (y / a), ⟨Real.neg_pi_div_two_lt_arctan _, Real.arctan_lt_pi_div_two _⟩, ?_⟩
    simp only
    rw [Real.tan_arctan]; field_simp

example : veriphy_instance% KaluzaKlein.Sets.Bijective.mul_tan_bijOn (1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : a ≠ 0), Set.BijOn (fun x => a * Real.tan x) (Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)) Set.univ := by
    intro a ha
    refine ⟨fun _ _ => Set.mem_univ _, fun x hx y hy h => ?_, fun y _ => ?_⟩
    · have h' : Real.tan x = Real.tan y := mul_left_cancel₀ ha h
      exact Real.strictMonoOn_tan.injOn hx hy h'
    · refine ⟨Real.arctan (y / a), ⟨Real.neg_pi_div_two_lt_arctan _, Real.arctan_lt_pi_div_two _⟩, ?_⟩
      simp only
      rw [Real.tan_arctan]; field_simp
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Bijective.mul_tan_bijOn (2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : a ≠ 0), Set.BijOn (fun x => a * Real.tan x) (Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)) Set.univ := by
    intro a ha
    refine ⟨fun _ _ => Set.mem_univ _, fun x hx y hy h => ?_, fun y _ => ?_⟩
    · have h' : Real.tan x = Real.tan y := mul_left_cancel₀ ha h
      exact Real.strictMonoOn_tan.injOn hx hy h'
    · refine ⟨Real.arctan (y / a), ⟨Real.neg_pi_div_two_lt_arctan _, Real.arctan_lt_pi_div_two _⟩, ?_⟩
      simp only
      rw [Real.tan_arctan]; field_simp
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Bijective.mul_tan_bijOn (-1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : a ≠ 0), Set.BijOn (fun x => a * Real.tan x) (Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)) Set.univ := by
    intro a ha
    refine ⟨fun _ _ => Set.mem_univ _, fun x hx y hy h => ?_, fun y _ => ?_⟩
    · have h' : Real.tan x = Real.tan y := mul_left_cancel₀ ha h
      exact Real.strictMonoOn_tan.injOn hx hy h'
    · refine ⟨Real.arctan (y / a), ⟨Real.neg_pi_div_two_lt_arctan _, Real.arctan_lt_pi_div_two _⟩, ?_⟩
      simp only
      rw [Real.tan_arctan]; field_simp
  exact @key (-1) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Bijective.mul_tan_bijOn
-- test: set-tan-bijon-test.sage

/-! ## exm-bijection-chart -/

/-- `φ ↦ (cos φ, sin φ)` restricted to `(a, a + 2π)` is a bijection onto the circle minus the point of angle `a`: on
`(0, 2π)` onto the circle minus `(1, 0)`. -/
theorem wrap_bijOn (a : ℝ) : Set.BijOn wrap (Set.Ioo a (a + 2 * Real.pi)) (circle \ {wrap a}) := by
  have hpi := Real.pi_pos
  refine ⟨fun φ hφ => ⟨?_, ?_⟩, wrap_injOn a, fun p hp => ?_⟩
  · simp [circle, wrap, Real.cos_sq_add_sin_sq]
  · intro h
    have h' : φ ∈ wrap ⁻¹' {wrap a} := h
    rw [wrap_preimage] at h'
    obtain ⟨k, hk⟩ := h'
    have h1 : 0 < (k : ℝ) * (2 * Real.pi) := by linarith [hφ.1]
    have h2 : (k : ℝ) * (2 * Real.pi) < 2 * Real.pi := by linarith [hφ.2]
    have hk1 : (0 : ℝ) < k := by nlinarith
    have hk2 : (k : ℝ) < 1 := by nlinarith
    have : (0 : ℤ) < k := by exact_mod_cast hk1
    have : k < (1 : ℤ) := by exact_mod_cast hk2
    omega
  · obtain ⟨hc, hne⟩ := hp
    have hc' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
      rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
    obtain ⟨ψ, hψ⟩ := hc'
    have hT : (0 : ℝ) < 2 * Real.pi := by positivity
    set ψ' := toIcoMod hT a ψ
    have hmem := toIcoMod_mem_Ico hT a ψ
    have hwrap : wrap ψ' = p := by
      rw [← hψ]
      simp only [wrap, one_mul, ψ', toIcoMod, zsmul_eq_mul]
      rw [show ψ - (toIcoDiv hT a ψ : ℝ) * (2 * Real.pi) = ψ + ((-toIcoDiv hT a ψ : ℤ) : ℝ) * (2 * Real.pi) by
        push_cast; ring]
      rw [Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
    refine ⟨ψ', ⟨lt_of_le_of_ne hmem.1 ?_, hmem.2⟩, hwrap⟩
    intro h
    apply hne
    rw [← hwrap, ← h]
    rfl

example : veriphy_instance% KaluzaKlein.Sets.Bijective.wrap_bijOn (0) := by
  have key : ∀ (a : ℝ), Set.BijOn wrap (Set.Ioo a (a + 2 * Real.pi)) (circle \ {wrap a}) := by
    intro a
    have hpi := Real.pi_pos
    refine ⟨fun φ hφ => ⟨?_, ?_⟩, wrap_injOn a, fun p hp => ?_⟩
    · simp [circle, wrap, Real.cos_sq_add_sin_sq]
    · intro h
      have h' : φ ∈ wrap ⁻¹' {wrap a} := h
      rw [wrap_preimage] at h'
      obtain ⟨k, hk⟩ := h'
      have h1 : 0 < (k : ℝ) * (2 * Real.pi) := by linarith [hφ.1]
      have h2 : (k : ℝ) * (2 * Real.pi) < 2 * Real.pi := by linarith [hφ.2]
      have hk1 : (0 : ℝ) < k := by nlinarith
      have hk2 : (k : ℝ) < 1 := by nlinarith
      have : (0 : ℤ) < k := by exact_mod_cast hk1
      have : k < (1 : ℤ) := by exact_mod_cast hk2
      omega
    · obtain ⟨hc, hne⟩ := hp
      have hc' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
      obtain ⟨ψ, hψ⟩ := hc'
      have hT : (0 : ℝ) < 2 * Real.pi := by positivity
      set ψ' := toIcoMod hT a ψ
      have hmem := toIcoMod_mem_Ico hT a ψ
      have hwrap : wrap ψ' = p := by
        rw [← hψ]
        simp only [wrap, one_mul, ψ', toIcoMod, zsmul_eq_mul]
        rw [show ψ - (toIcoDiv hT a ψ : ℝ) * (2 * Real.pi) = ψ + ((-toIcoDiv hT a ψ : ℤ) : ℝ) * (2 * Real.pi) by
          push_cast; ring]
        rw [Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
      refine ⟨ψ', ⟨lt_of_le_of_ne hmem.1 ?_, hmem.2⟩, hwrap⟩
      intro h
      apply hne
      rw [← hwrap, ← h]
      rfl
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Bijective.wrap_bijOn (Real.pi) := by
  have key : ∀ (a : ℝ), Set.BijOn wrap (Set.Ioo a (a + 2 * Real.pi)) (circle \ {wrap a}) := by
    intro a
    have hpi := Real.pi_pos
    refine ⟨fun φ hφ => ⟨?_, ?_⟩, wrap_injOn a, fun p hp => ?_⟩
    · simp [circle, wrap, Real.cos_sq_add_sin_sq]
    · intro h
      have h' : φ ∈ wrap ⁻¹' {wrap a} := h
      rw [wrap_preimage] at h'
      obtain ⟨k, hk⟩ := h'
      have h1 : 0 < (k : ℝ) * (2 * Real.pi) := by linarith [hφ.1]
      have h2 : (k : ℝ) * (2 * Real.pi) < 2 * Real.pi := by linarith [hφ.2]
      have hk1 : (0 : ℝ) < k := by nlinarith
      have hk2 : (k : ℝ) < 1 := by nlinarith
      have : (0 : ℤ) < k := by exact_mod_cast hk1
      have : k < (1 : ℤ) := by exact_mod_cast hk2
      omega
    · obtain ⟨hc, hne⟩ := hp
      have hc' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
      obtain ⟨ψ, hψ⟩ := hc'
      have hT : (0 : ℝ) < 2 * Real.pi := by positivity
      set ψ' := toIcoMod hT a ψ
      have hmem := toIcoMod_mem_Ico hT a ψ
      have hwrap : wrap ψ' = p := by
        rw [← hψ]
        simp only [wrap, one_mul, ψ', toIcoMod, zsmul_eq_mul]
        rw [show ψ - (toIcoDiv hT a ψ : ℝ) * (2 * Real.pi) = ψ + ((-toIcoDiv hT a ψ : ℤ) : ℝ) * (2 * Real.pi) by
          push_cast; ring]
        rw [Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
      refine ⟨ψ', ⟨lt_of_le_of_ne hmem.1 ?_, hmem.2⟩, hwrap⟩
      intro h
      apply hne
      rw [← hwrap, ← h]
      rfl
  exact @key (Real.pi)

example : veriphy_instance% KaluzaKlein.Sets.Bijective.wrap_bijOn (-1) := by
  have key : ∀ (a : ℝ), Set.BijOn wrap (Set.Ioo a (a + 2 * Real.pi)) (circle \ {wrap a}) := by
    intro a
    have hpi := Real.pi_pos
    refine ⟨fun φ hφ => ⟨?_, ?_⟩, wrap_injOn a, fun p hp => ?_⟩
    · simp [circle, wrap, Real.cos_sq_add_sin_sq]
    · intro h
      have h' : φ ∈ wrap ⁻¹' {wrap a} := h
      rw [wrap_preimage] at h'
      obtain ⟨k, hk⟩ := h'
      have h1 : 0 < (k : ℝ) * (2 * Real.pi) := by linarith [hφ.1]
      have h2 : (k : ℝ) * (2 * Real.pi) < 2 * Real.pi := by linarith [hφ.2]
      have hk1 : (0 : ℝ) < k := by nlinarith
      have hk2 : (k : ℝ) < 1 := by nlinarith
      have : (0 : ℤ) < k := by exact_mod_cast hk1
      have : k < (1 : ℤ) := by exact_mod_cast hk2
      omega
    · obtain ⟨hc, hne⟩ := hp
      have hc' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
      obtain ⟨ψ, hψ⟩ := hc'
      have hT : (0 : ℝ) < 2 * Real.pi := by positivity
      set ψ' := toIcoMod hT a ψ
      have hmem := toIcoMod_mem_Ico hT a ψ
      have hwrap : wrap ψ' = p := by
        rw [← hψ]
        simp only [wrap, one_mul, ψ', toIcoMod, zsmul_eq_mul]
        rw [show ψ - (toIcoDiv hT a ψ : ℝ) * (2 * Real.pi) = ψ + ((-toIcoDiv hT a ψ : ℤ) : ℝ) * (2 * Real.pi) by
          push_cast; ring]
        rw [Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
      refine ⟨ψ', ⟨lt_of_le_of_ne hmem.1 ?_, hmem.2⟩, hwrap⟩
      intro h
      apply hne
      rw [← hwrap, ← h]
      rfl
  exact @key (-1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Bijective.wrap_bijOn
-- test: set-wrap-bijon-test.sage

/-! ## exr-bijection-count-4 -/

/-- The permutations of an `(n + 1)`-element set fixing one element are the `n!` permutations of the others: the
`3! = 6` permutations of `{1, 2, 3, 4}` fixing `4`. -/
theorem card_perm_fixing (n : ℕ) (a : Fin (n + 1)) :
    Fintype.card {σ : Equiv.Perm (Fin (n + 1)) // σ a = a} = n.factorial := by
  have e := Equiv.Perm.subtypeEquivSubtypePerm (fun x : Fin (n + 1) => x ≠ a)
  have h : Fintype.card {σ : Equiv.Perm (Fin (n + 1)) // σ a = a} =
      Fintype.card {f : Equiv.Perm (Fin (n + 1)) // ∀ x, ¬ (x ≠ a) → f x = x} := by
    refine Fintype.card_congr (Equiv.subtypeEquivRight fun σ => ?_)
    simp
  rw [h, ← Fintype.card_congr e, Fintype.card_perm]
  simp [Fintype.card_subtype_compl, Fintype.card_fin]

example : veriphy_instance% KaluzaKlein.Sets.Bijective.card_perm_fixing (3) (3) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Bijective.card_perm_fixing (0) (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Bijective.card_perm_fixing (2) (0) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Bijective.card_perm_fixing
-- test: set-card-perm-fixing-test.sage

/-! ## exr-bijection-affine -/

/-- `ax + b = y` (`a ≠ 0`) has the unique solution `x = (y - b)/a`: `x = (y - 1)/2` for `2x + 1`. -/
theorem affine_unique_solution (a b y : ℝ) (ha : a ≠ 0) : ∃! x, a * x + b = y ∧ x = (y - b) / a := by
  refine ⟨(y - b) / a, ⟨by field_simp; ring, rfl⟩, fun x hx => hx.2⟩

example : veriphy_instance% KaluzaKlein.Sets.Bijective.affine_unique_solution (2) (1) (5) ((by norm_num)) := by
  have key : ∀ (a b y : ℝ) (ha : a ≠ 0), ∃! x, a * x + b = y ∧ x = (y - b) / a := by
    intro a b y ha
    refine ⟨(y - b) / a, ⟨by field_simp; ring, rfl⟩, fun x hx => hx.2⟩
  exact @key (2) (1) (5) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Bijective.affine_unique_solution (1) (0) (0) ((by norm_num)) := by
  have key : ∀ (a b y : ℝ) (ha : a ≠ 0), ∃! x, a * x + b = y ∧ x = (y - b) / a := by
    intro a b y ha
    refine ⟨(y - b) / a, ⟨by field_simp; ring, rfl⟩, fun x hx => hx.2⟩
  exact @key (1) (0) (0) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Bijective.affine_unique_solution (-3) (2) (7) ((by norm_num)) := by
  have key : ∀ (a b y : ℝ) (ha : a ≠ 0), ∃! x, a * x + b = y ∧ x = (y - b) / a := by
    intro a b y ha
    refine ⟨(y - b) / a, ⟨by field_simp; ring, rfl⟩, fun x hx => hx.2⟩
  exact @key (-3) (2) (7) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Bijective.affine_unique_solution
-- test: set-affine-solution-test.sage

/-- The solution of `2x + 1 = y`: `2 · (y - 1)/2 + 1 = y`. -/
theorem affine_solution_check (y w : ℝ) (hw : 2 * w = 1) : 2 * ((y - 1) * w) + 1 = y := by
  veriphy_certified (linear_combination (y - 1) * hw)
  -- certificate: ideal_membership.sage

/-- On `ℤ`, `k ↦ 2k + b` (`b` odd) is injective, not surjective, and its image is the odd integers: `2k + 1`. -/
theorem int_affine (b : ℤ) (hb : Odd b) :
    Function.Injective (fun k : ℤ => 2 * k + b) ∧ ¬ Function.Surjective (fun k : ℤ => 2 * k + b) ∧
      Set.range (fun k : ℤ => 2 * k + b) = {n | Odd n} := by
  obtain ⟨c, rfl⟩ := hb
  refine ⟨fun x y h => by simpa using h, fun h => ?_, ?_⟩
  · obtain ⟨k, hk⟩ := h 0
    simp only at hk
    omega
  · ext n
    simp only [Set.mem_range, Set.mem_setOf_eq, Odd]
    constructor
    · rintro ⟨k, rfl⟩; exact ⟨k + c, by ring⟩
    · rintro ⟨m, rfl⟩; exact ⟨m - c, by ring⟩

example : veriphy_instance% KaluzaKlein.Sets.Bijective.int_affine (1) ((by decide)) := by
  have key : ∀ (b : ℤ) (hb : Odd b), Function.Injective (fun k : ℤ => 2 * k + b) ∧ ¬ Function.Surjective (fun k : ℤ => 2 * k + b) ∧ Set.range (fun k : ℤ => 2 * k + b) = {n | Odd n} := by
    intro b hb
    obtain ⟨c, rfl⟩ := hb
    refine ⟨fun x y h => by simpa using h, fun h => ?_, ?_⟩
    · obtain ⟨k, hk⟩ := h 0
      simp only at hk
      omega
    · ext n
      simp only [Set.mem_range, Set.mem_setOf_eq, Odd]
      constructor
      · rintro ⟨k, rfl⟩; exact ⟨k + c, by ring⟩
      · rintro ⟨m, rfl⟩; exact ⟨m - c, by ring⟩
  exact @key (1) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Bijective.int_affine (3) ((by decide)) := by
  have key : ∀ (b : ℤ) (hb : Odd b), Function.Injective (fun k : ℤ => 2 * k + b) ∧ ¬ Function.Surjective (fun k : ℤ => 2 * k + b) ∧ Set.range (fun k : ℤ => 2 * k + b) = {n | Odd n} := by
    intro b hb
    obtain ⟨c, rfl⟩ := hb
    refine ⟨fun x y h => by simpa using h, fun h => ?_, ?_⟩
    · obtain ⟨k, hk⟩ := h 0
      simp only at hk
      omega
    · ext n
      simp only [Set.mem_range, Set.mem_setOf_eq, Odd]
      constructor
      · rintro ⟨k, rfl⟩; exact ⟨k + c, by ring⟩
      · rintro ⟨m, rfl⟩; exact ⟨m - c, by ring⟩
  exact @key (3) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Bijective.int_affine (-1) ((by decide)) := by
  have key : ∀ (b : ℤ) (hb : Odd b), Function.Injective (fun k : ℤ => 2 * k + b) ∧ ¬ Function.Surjective (fun k : ℤ => 2 * k + b) ∧ Set.range (fun k : ℤ => 2 * k + b) = {n | Odd n} := by
    intro b hb
    obtain ⟨c, rfl⟩ := hb
    refine ⟨fun x y h => by simpa using h, fun h => ?_, ?_⟩
    · obtain ⟨k, hk⟩ := h 0
      simp only at hk
      omega
    · ext n
      simp only [Set.mem_range, Set.mem_setOf_eq, Odd]
      constructor
      · rintro ⟨k, rfl⟩; exact ⟨k + c, by ring⟩
      · rintro ⟨m, rfl⟩; exact ⟨m - c, by ring⟩
  exact @key (-1) ((by decide))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Bijective.int_affine
-- test: set-int-affine-test.sage

/-! ## exr-bijection-stereographic

The identities are stated with `w = 1/(u² + v² + 1)` (respectively `w = 1/(1 - Z)`) as a variable with its defining
equation, so that they are polynomial identities modulo the hypotheses. -/

/-- `σ⁻¹(u, v)` lies on the sphere. -/
theorem stereoInv_sphere (u v w : ℝ) (hw : (u ^ 2 + v ^ 2 + 1) * w = 1) :
    (2 * u * w) ^ 2 + (2 * v * w) ^ 2 + ((u ^ 2 + v ^ 2 - 1) * w) ^ 2 = 1 := by
  veriphy_certified (linear_combination (u ^ 2 * w + w * v ^ 2 + w + 1) * hw)
  -- certificate: ideal_membership.sage

/-- `σ⁻¹(u, v)` is not the north pole: its height is `1 - 2w < 1`. -/
theorem stereoInv_height (u v w : ℝ) (hw : (u ^ 2 + v ^ 2 + 1) * w = 1) : (u ^ 2 + v ^ 2 - 1) * w = 1 - 2 * w := by
  veriphy_certified (linear_combination hw)
  -- certificate: ideal_membership.sage

/-- Projecting `σ⁻¹(u, v)` gives back `(u, v)`: `1 - Z = 2w`, and `X/(1 - Z) = u`, `Y/(1 - Z) = v`. -/
theorem stereo_stereoInv (u v w : ℝ) (hw : (u ^ 2 + v ^ 2 + 1) * w = 1) :
    1 - (u ^ 2 + v ^ 2 - 1) * w = 2 * w ∧ (2 * u * w) = u * (2 * w) ∧ (2 * v * w) = v * (2 * w) := by
  refine ⟨?_, ?_, ?_⟩
  · veriphy_certified (linear_combination (-1) * hw)
    -- certificate: ideal_membership.sage
  · ring
  · ring

/-- Applying `σ⁻¹` to the projection `(u, v) = (X, Y) t` of a point of the sphere, `t = 1/(1 - Z)`, gives back
`(X, Y, Z)`: `u² + v² + 1 = 2t`, so `σ⁻¹(u, v) = (X, Y, Z)`. -/
theorem stereoInv_stereo (X Y Z t : ℝ) (hp : X ^ 2 + Y ^ 2 + Z ^ 2 = 1) (ht : (1 - Z) * t = 1) :
    (X * t) ^ 2 + (Y * t) ^ 2 + 1 = 2 * t ∧ ((X * t) ^ 2 + (Y * t) ^ 2 - 1) = Z * (2 * t) := by
  refine ⟨?_, ?_⟩
  · veriphy_certified (linear_combination (t ^ 2) * hp + (t * Z + t - 1) * ht)
    -- certificate: ideal_membership.sage
  · veriphy_certified (linear_combination (t ^ 2) * hp + (t * Z + t + 1) * ht)
    -- certificate: ideal_membership.sage

/-- Stereographic projection is a bijection from the sphere minus the north pole onto the plane, with inverse
`σ⁻¹`. -/
theorem stereo_bijOn : Set.BijOn stereo (sphere \ {northPole}) Set.univ ∧
    (∀ q, stereo (stereoInv q) = q) ∧ ∀ p ∈ sphere \ {northPole}, stereoInv (stereo p) = p := by
  have hpos : ∀ q : ℝ × ℝ, 0 < q.1 ^ 2 + q.2 ^ 2 + 1 := fun q => by positivity
  have hZ : ∀ p ∈ sphere \ {northPole}, p 2 ≠ 1 := by
    rintro p ⟨hp, hN⟩ h2
    apply hN
    simp only [sphere, Set.mem_setOf_eq] at hp
    have h0 : p 0 = 0 := by nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]
    have h1 : p 1 = 0 := by nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]
    ext i; fin_cases i <;> simp [northPole, h0, h1, h2]
  have left : ∀ q, stereo (stereoInv q) = q := by
    rintro ⟨u, v⟩
    have hd := hpos (u, v)
    obtain ⟨h1, h2, h3⟩ := stereo_stereoInv u v (u ^ 2 + v ^ 2 + 1)⁻¹ (by field_simp)
    have hw : (2 * (u ^ 2 + v ^ 2 + 1)⁻¹) ≠ 0 := by positivity
    simp only [stereo, stereoInv, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      Matrix.cons_val_two, Matrix.tail_cons, Prod.mk.injEq]
    simp only [div_eq_mul_inv] at *
    rw [h1, h2, h3]
    exact ⟨by field_simp, by field_simp⟩
  have right : ∀ p ∈ sphere \ {northPole}, stereoInv (stereo p) = p := by
    intro p hp
    have hz := hZ p hp
    have hp' := hp.1
    simp only [sphere, Set.mem_setOf_eq] at hp'
    have hz' : 1 - p 2 ≠ 0 := sub_ne_zero.2 (Ne.symm hz)
    obtain ⟨h1, h2⟩ := stereoInv_stereo (p 0) (p 1) (p 2) (1 - p 2)⁻¹ hp' (by field_simp)
    have ht : (2 * (1 - p 2)⁻¹) ≠ 0 := by
      have : (1 - p 2)⁻¹ ≠ 0 := inv_ne_zero hz'
      positivity
    ext i
    fin_cases i <;> simp only [stereo, stereoInv, div_eq_mul_inv, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons, Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk]
    · rw [h1]; field_simp
    · rw [h1]; field_simp
    · rw [h1, h2]; field_simp
  refine ⟨⟨fun _ _ => Set.mem_univ _, fun p hp q hq h => ?_, fun q _ => ⟨stereoInv q, ⟨?_, ?_⟩, left q⟩⟩,
    left, right⟩
  · rw [← right p hp, ← right q hq, h]
  · simp only [sphere, Set.mem_setOf_eq, stereoInv, div_eq_mul_inv, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons]
    linear_combination stereoInv_sphere q.1 q.2 (q.1 ^ 2 + q.2 ^ 2 + 1)⁻¹ (by field_simp [(hpos q).ne'])
  · intro h
    have h2 := congrFun h 2
    simp only [stereoInv, northPole, Matrix.cons_val_two, Matrix.tail_cons, Matrix.head_cons] at h2
    have := stereoInv_height q.1 q.2 (q.1 ^ 2 + q.2 ^ 2 + 1)⁻¹ (by field_simp [(hpos q).ne'])
    rw [div_eq_mul_inv, this] at h2
    have : 0 < (q.1 ^ 2 + q.2 ^ 2 + 1)⁻¹ := inv_pos.2 (hpos q)
    linarith

end KaluzaKlein.Sets.Bijective
