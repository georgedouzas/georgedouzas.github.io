import Mathlib
import KaluzaKlein.PartI.Sets.Quotients
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Products"

The claims of the section "Products" of `tutorial/part-i-01-sets-relations-topology.qmd`: sizes and projections of
products, `ℝⁿ⁺¹ = ℝ × ℝⁿ`, the torus as pairs of points of the circle, a point of the torus in `ℝ⁴` and in `ℝ³`,
products of maps, and the quotients `ℤ/nℤ` (their sizes, remainders, operations, fields, cyclicity, and the map
`ℤ₁₂ → ℤ₄`).
-/

namespace KaluzaKlein.Sets.Products

open KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Maps

/-! ## def-product -/

/-- A product of finite sets has `|X| · |Y|` elements (`2 · 3 = 6` pairs, `2 · 3 · 2 = 12` triples). -/
theorem card_prod (X Y : Type) [Fintype X] [Fintype Y] : Fintype.card (X × Y) = Fintype.card X * Fintype.card Y :=
  Fintype.card_prod X Y

example : veriphy_instance% KaluzaKlein.Sets.Products.card_prod (Fin 2) (Fin 3) (inferInstance) (inferInstance) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Products.card_prod (Fin 0) (Fin 3) (inferInstance) (inferInstance) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Products.card_prod (Fin 2 × Fin 3) (Fin 2) (inferInstance) (inferInstance) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Products.card_prod
-- test: set-card-prod-test.sage

/-- The projections onto non-empty factors are surjective. -/
theorem projections_surjective (X Y : Type) [Nonempty X] [Nonempty Y] :
    Function.Surjective (Prod.fst : X × Y → X) ∧ Function.Surjective (Prod.snd : X × Y → Y) :=
  ⟨Prod.fst_surjective, Prod.snd_surjective⟩

example : veriphy_instance% KaluzaKlein.Sets.Products.projections_surjective (Fin 2) (Fin 3) (inferInstance) (inferInstance) := by
  have key : ∀ (X Y : Type) [inst1 : Nonempty X] [inst2 : Nonempty Y], Function.Surjective (Prod.fst : X × Y → X) ∧ Function.Surjective (Prod.snd : X × Y → Y) := by
    intro X Y inst1 inst2
    exact (⟨Prod.fst_surjective, Prod.snd_surjective⟩)
  exact @key (Fin 2) (Fin 3) (inferInstance) (inferInstance)

example : veriphy_instance% KaluzaKlein.Sets.Products.projections_surjective (Unit) (Unit) (inferInstance) (inferInstance) := by
  have key : ∀ (X Y : Type) [inst1 : Nonempty X] [inst2 : Nonempty Y], Function.Surjective (Prod.fst : X × Y → X) ∧ Function.Surjective (Prod.snd : X × Y → Y) := by
    intro X Y inst1 inst2
    exact (⟨Prod.fst_surjective, Prod.snd_surjective⟩)
  exact @key (Unit) (Unit) (inferInstance) (inferInstance)

example : veriphy_instance% KaluzaKlein.Sets.Products.projections_surjective (ℝ) (ℕ) (inferInstance) (inferInstance) := by
  have key : ∀ (X Y : Type) [inst1 : Nonempty X] [inst2 : Nonempty Y], Function.Surjective (Prod.fst : X × Y → X) ∧ Function.Surjective (Prod.snd : X × Y → Y) := by
    intro X Y inst1 inst2
    exact (⟨Prod.fst_surjective, Prod.snd_surjective⟩)
  exact @key (ℝ) (ℕ) (inferInstance) (inferInstance)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Products.projections_surjective
-- test: set-projections-test.sage

/-! ## exm-product-rn -/

/-- `ℝⁿ⁺¹ = ℝ × ℝⁿ`: a list of `n + 1` numbers is its first number and the list of the others. -/
theorem rn_succ_prod (n : ℕ) : Function.Bijective (fun x : Fin (n + 1) → ℝ => (x 0, Fin.tail x)) :=
  (Fin.consEquiv (fun _ : Fin (n + 1) => ℝ)).symm.bijective

example : veriphy_instance% KaluzaKlein.Sets.Products.rn_succ_prod (0) := by
  have key : ∀ (n : ℕ), Function.Bijective (fun x : Fin (n + 1) → ℝ => (x 0, Fin.tail x)) := by
    intro n
    exact ((Fin.consEquiv (fun _ : Fin (n + 1) => ℝ)).symm.bijective)
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Products.rn_succ_prod (1) := by
  have key : ∀ (n : ℕ), Function.Bijective (fun x : Fin (n + 1) → ℝ => (x 0, Fin.tail x)) := by
    intro n
    exact ((Fin.consEquiv (fun _ : Fin (n + 1) => ℝ)).symm.bijective)
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Products.rn_succ_prod (3) := by
  have key : ∀ (n : ℕ), Function.Bijective (fun x : Fin (n + 1) → ℝ => (x 0, Fin.tail x)) := by
    intro n
    exact ((Fin.consEquiv (fun _ : Fin (n + 1) => ℝ)).symm.bijective)
  exact @key (3)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Products.rn_succ_prod
-- test: set-rn-succ-test.sage

/-- The torus `S¹ × S¹` (radii `r₁`, `r₂ ≥ 0`) is the set of pairs of points of the two circles, each labelled by
an angle: a point of the torus is labelled by a pair of angles. -/
theorem torus_pairs_of_angles (r₁ r₂ : ℝ) (h₁ : 0 ≤ r₁) (h₂ : 0 ≤ r₂) :
    Set.range (fun q : ℝ × ℝ => ((r₁ * Real.cos q.1, r₁ * Real.sin q.1), (r₂ * Real.cos q.2, r₂ * Real.sin q.2))) =
      {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₁ ^ 2} ×ˢ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₂ ^ 2} := by
  rw [← range_wrap r₁ h₁, ← range_wrap r₂ h₂, ← Set.range_prodMap]
  rfl

example : veriphy_instance% KaluzaKlein.Sets.Products.torus_pairs_of_angles (1) (1) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (r₁ r₂ : ℝ) (h₁ : 0 ≤ r₁) (h₂ : 0 ≤ r₂), Set.range (fun q : ℝ × ℝ => ((r₁ * Real.cos q.1, r₁ * Real.sin q.1), (r₂ * Real.cos q.2, r₂ * Real.sin q.2))) = {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₁ ^ 2} ×ˢ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₂ ^ 2} := by
    intro r₁ r₂ h₁ h₂
    rw [← range_wrap r₁ h₁, ← range_wrap r₂ h₂, ← Set.range_prodMap]
    rfl
  exact @key (1) (1) ((by norm_num)) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Products.torus_pairs_of_angles (2) (1) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (r₁ r₂ : ℝ) (h₁ : 0 ≤ r₁) (h₂ : 0 ≤ r₂), Set.range (fun q : ℝ × ℝ => ((r₁ * Real.cos q.1, r₁ * Real.sin q.1), (r₂ * Real.cos q.2, r₂ * Real.sin q.2))) = {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₁ ^ 2} ×ˢ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₂ ^ 2} := by
    intro r₁ r₂ h₁ h₂
    rw [← range_wrap r₁ h₁, ← range_wrap r₂ h₂, ← Set.range_prodMap]
    rfl
  exact @key (2) (1) ((by norm_num)) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Products.torus_pairs_of_angles (0) (3) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (r₁ r₂ : ℝ) (h₁ : 0 ≤ r₁) (h₂ : 0 ≤ r₂), Set.range (fun q : ℝ × ℝ => ((r₁ * Real.cos q.1, r₁ * Real.sin q.1), (r₂ * Real.cos q.2, r₂ * Real.sin q.2))) = {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₁ ^ 2} ×ˢ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₂ ^ 2} := by
    intro r₁ r₂ h₁ h₂
    rw [← range_wrap r₁ h₁, ← range_wrap r₂ h₂, ← Set.range_prodMap]
    rfl
  exact @key (0) (3) ((by norm_num)) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Products.torus_pairs_of_angles
-- test: set-torus-pairs-test.sage

/-! ## exr-product-torus-point -/

/-- The point of `S¹ × S¹ ⊂ ℝ⁴` with angles `(π/3, π/2)` is `(½, √3/2, 0, 1)`. -/
theorem torus_point_r4 :
    Real.cos (Real.pi / 3) = 1 / 2 ∧ Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 ∧ Real.cos (Real.pi / 2) = 0 ∧
      Real.sin (Real.pi / 2) = 1 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · veriphy_certified
      (norm_num [Real.cos_pi_div_three, Real.sin_pi_div_three, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_zero,
          Real.sin_zero])
    -- certificate: set-trig-values.sage
  · veriphy_certified
      (norm_num [Real.cos_pi_div_three, Real.sin_pi_div_three, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_zero,
          Real.sin_zero])
    -- certificate: set-trig-values.sage
  · veriphy_certified
      (norm_num [Real.cos_pi_div_three, Real.sin_pi_div_three, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_zero,
          Real.sin_zero])
    -- certificate: set-trig-values.sage
  · veriphy_certified
      (norm_num [Real.cos_pi_div_three, Real.sin_pi_div_three, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_zero,
          Real.sin_zero])
    -- certificate: set-trig-values.sage

/-- The torus of revolution with radii `2` and `1`, `((2 + cos v) cos u, (2 + cos v) sin u, sin v)`, at
`(u, v) = (π/3, π/2)` is the point `(1, √3, 1)`. -/
theorem torus_point_r3 :
    (2 + Real.cos (Real.pi / 2)) * Real.cos (Real.pi / 3) = 1 ∧
      (2 + Real.cos (Real.pi / 2)) * Real.sin (Real.pi / 3) = Real.sqrt 3 ∧ Real.sin (Real.pi / 2) = 1 := by
  refine ⟨?_, ?_, ?_⟩
  · veriphy_certified
      (norm_num [Real.cos_pi_div_three, Real.sin_pi_div_three, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_zero,
          Real.sin_zero])
    -- certificate: set-trig-values.sage
  · veriphy_certified
      (norm_num [Real.cos_pi_div_three, Real.sin_pi_div_three, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_zero,
            Real.sin_zero] <;>
          ring_nf)
    -- certificate: set-trig-values.sage
  · veriphy_certified
      (norm_num [Real.cos_pi_div_three, Real.sin_pi_div_three, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_zero,
          Real.sin_zero])
    -- certificate: set-trig-values.sage

/-! ## exr-product-of-maps -/

/-- For non-empty sets, `f × g` is injective exactly when `f` and `g` are, and surjective exactly when `f` and `g`
are. -/
theorem prodMap_injective_surjective {X X' Y Y' : Type} [Nonempty X] [Nonempty Y] [Nonempty X'] [Nonempty Y']
    (f : X → X') (g : Y → Y') :
    (Function.Injective (Prod.map f g) ↔ Function.Injective f ∧ Function.Injective g) ∧
      (Function.Surjective (Prod.map f g) ↔ Function.Surjective f ∧ Function.Surjective g) :=
  ⟨Prod.map_injective, Prod.map_surjective⟩

example : veriphy_instance% KaluzaKlein.Sets.Products.prodMap_injective_surjective (Fin 2) (Fin 2) (Fin 3) (Fin 3) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (id) (id) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ {X X' Y Y' : Type} [inst1 : Nonempty X] [inst2 : Nonempty Y] [inst3 : Nonempty X'] [inst4 : Nonempty Y'] (f : X → X') (g : Y → Y'), (Function.Injective (Prod.map f g) ↔ Function.Injective f ∧ Function.Injective g) ∧ (Function.Surjective (Prod.map f g) ↔ Function.Surjective f ∧ Function.Surjective g) := by
    intro X X' Y Y' inst1 inst2 inst3 inst4 f g
    exact (⟨Prod.map_injective, Prod.map_surjective⟩)
  exact @key (Fin 2) (Fin 2) (Fin 3) (Fin 3) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (id) (id)

example : veriphy_instance% KaluzaKlein.Sets.Products.prodMap_injective_surjective (ℝ) (ℝ) (ℝ) (ℝ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (fun x => x ^ 2) (fun x => 2 * x + 1) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ {X X' Y Y' : Type} [inst1 : Nonempty X] [inst2 : Nonempty Y] [inst3 : Nonempty X'] [inst4 : Nonempty Y'] (f : X → X') (g : Y → Y'), (Function.Injective (Prod.map f g) ↔ Function.Injective f ∧ Function.Injective g) ∧ (Function.Surjective (Prod.map f g) ↔ Function.Surjective f ∧ Function.Surjective g) := by
    intro X X' Y Y' inst1 inst2 inst3 inst4 f g
    exact (⟨Prod.map_injective, Prod.map_surjective⟩)
  exact @key (ℝ) (ℝ) (ℝ) (ℝ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (fun x => x ^ 2) (fun x => 2 * x + 1)

example : veriphy_instance% KaluzaKlein.Sets.Products.prodMap_injective_surjective (ℕ) (ℤ) (ℕ) (ℤ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (fun n : ℕ => (n : ℤ)) (fun n : ℕ => (n : ℤ)) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ {X X' Y Y' : Type} [inst1 : Nonempty X] [inst2 : Nonempty Y] [inst3 : Nonempty X'] [inst4 : Nonempty Y'] (f : X → X') (g : Y → Y'), (Function.Injective (Prod.map f g) ↔ Function.Injective f ∧ Function.Injective g) ∧ (Function.Surjective (Prod.map f g) ↔ Function.Surjective f ∧ Function.Surjective g) := by
    intro X X' Y Y' inst1 inst2 inst3 inst4 f g
    exact (⟨Prod.map_injective, Prod.map_surjective⟩)
  exact @key (ℕ) (ℤ) (ℕ) (ℤ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (fun n : ℕ => (n : ℤ)) (fun n : ℕ => (n : ℤ))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Products.prodMap_injective_surjective
-- test: set-prodmap-test.sage

/-! ## exm-z3-elements -/

/-- Every integer has exactly one remainder `0 ≤ r < n` modulo `n ≥ 1`: the classes `[0], …, [n - 1]` cover `ℤ` and are
distinct (`[0] = 3ℤ`, `[1] = 1 + 3ℤ`, `[2] = 2 + 3ℤ` for `n = 3`). -/
theorem unique_remainder (n : ℕ) (hn : 1 ≤ n) (a : ℤ) : ∃! r : ℕ, r < n ∧ a ≡ r [ZMOD n] := by
  refine ⟨(a % n).toNat, ⟨?_, ?_⟩, ?_⟩
  · have h1 := Int.emod_nonneg a (by omega : (n : ℤ) ≠ 0)
    have h2 := Int.emod_lt_of_pos a (by omega : (0 : ℤ) < n)
    omega
  · have h1 := Int.emod_nonneg a (by omega : (n : ℤ) ≠ 0)
    rw [Int.toNat_of_nonneg h1]
    exact (Int.mod_modEq a n).symm
  · rintro r ⟨hr, har⟩
    have h1 := Int.emod_nonneg a (by omega : (n : ℤ) ≠ 0)
    have : (r : ℤ) = a % n := by
      rw [Int.ModEq] at har
      rw [har, Int.emod_eq_of_lt (by omega) (by omega)]
    omega

example : veriphy_instance% KaluzaKlein.Sets.Products.unique_remainder (3) ((by norm_num)) (7) := by
  have key : ∀ (n : ℕ) (hn : 1 ≤ n) (a : ℤ), ∃! r : ℕ, r < n ∧ a ≡ r [ZMOD n] := by
    intro n hn a
    refine ⟨(a % n).toNat, ⟨?_, ?_⟩, ?_⟩
    · have h1 := Int.emod_nonneg a (by omega : (n : ℤ) ≠ 0)
      have h2 := Int.emod_lt_of_pos a (by omega : (0 : ℤ) < n)
      omega
    · have h1 := Int.emod_nonneg a (by omega : (n : ℤ) ≠ 0)
      rw [Int.toNat_of_nonneg h1]
      exact (Int.mod_modEq a n).symm
    · rintro r ⟨hr, har⟩
      have h1 := Int.emod_nonneg a (by omega : (n : ℤ) ≠ 0)
      have : (r : ℤ) = a % n := by
        rw [Int.ModEq] at har
        rw [har, Int.emod_eq_of_lt (by omega) (by omega)]
      omega
  exact @key (3) ((by norm_num)) (7)

example : veriphy_instance% KaluzaKlein.Sets.Products.unique_remainder (1) ((by norm_num)) (0) := by
  have key : ∀ (n : ℕ) (hn : 1 ≤ n) (a : ℤ), ∃! r : ℕ, r < n ∧ a ≡ r [ZMOD n] := by
    intro n hn a
    refine ⟨(a % n).toNat, ⟨?_, ?_⟩, ?_⟩
    · have h1 := Int.emod_nonneg a (by omega : (n : ℤ) ≠ 0)
      have h2 := Int.emod_lt_of_pos a (by omega : (0 : ℤ) < n)
      omega
    · have h1 := Int.emod_nonneg a (by omega : (n : ℤ) ≠ 0)
      rw [Int.toNat_of_nonneg h1]
      exact (Int.mod_modEq a n).symm
    · rintro r ⟨hr, har⟩
      have h1 := Int.emod_nonneg a (by omega : (n : ℤ) ≠ 0)
      have : (r : ℤ) = a % n := by
        rw [Int.ModEq] at har
        rw [har, Int.emod_eq_of_lt (by omega) (by omega)]
      omega
  exact @key (1) ((by norm_num)) (0)

example : veriphy_instance% KaluzaKlein.Sets.Products.unique_remainder (5) ((by norm_num)) (-12) := by
  have key : ∀ (n : ℕ) (hn : 1 ≤ n) (a : ℤ), ∃! r : ℕ, r < n ∧ a ≡ r [ZMOD n] := by
    intro n hn a
    refine ⟨(a % n).toNat, ⟨?_, ?_⟩, ?_⟩
    · have h1 := Int.emod_nonneg a (by omega : (n : ℤ) ≠ 0)
      have h2 := Int.emod_lt_of_pos a (by omega : (0 : ℤ) < n)
      omega
    · have h1 := Int.emod_nonneg a (by omega : (n : ℤ) ≠ 0)
      rw [Int.toNat_of_nonneg h1]
      exact (Int.mod_modEq a n).symm
    · rintro r ⟨hr, har⟩
      have h1 := Int.emod_nonneg a (by omega : (n : ℤ) ≠ 0)
      have : (r : ℤ) = a % n := by
        rw [Int.ModEq] at har
        rw [har, Int.emod_eq_of_lt (by omega) (by omega)]
      omega
  exact @key (5) ((by norm_num)) (-12)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Products.unique_remainder
-- test: set-unique-remainder-test.sage

/-! ## exm-z3-table -/

/-- `ℤ_p` is a field for `p` prime (`ℤ₂`, `ℤ₃`, `ℤ₅`): every non-zero class has a multiplicative inverse. -/
theorem zmod_prime_field (p : ℕ) (hp : p.Prime) : ∀ a : ZMod p, a ≠ 0 → ∃ b : ZMod p, a * b = 1 := by
  haveI := Fact.mk hp
  intro a ha
  exact ⟨a⁻¹, mul_inv_cancel₀ ha⟩

example : veriphy_instance% KaluzaKlein.Sets.Products.zmod_prime_field (3) ((by norm_num)) (2) ((by decide)) := by
  exact ⟨2, by decide⟩

example : veriphy_instance% KaluzaKlein.Sets.Products.zmod_prime_field (2) ((by norm_num)) (1) ((by decide)) := by
  exact ⟨1, by decide⟩

example : veriphy_instance% KaluzaKlein.Sets.Products.zmod_prime_field (5) ((by norm_num)) (3) ((by decide)) := by
  exact ⟨2, by decide⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Products.zmod_prime_field
-- test: set-zmod-field-test.sage

/-! ## exm-z3-general -/

/-- `ℤₙ` is cyclic, generated by `1`: every element is a multiple of `1`. -/
theorem zmod_cyclic (n : ℕ) (a : ZMod n) : ∃ k : ℤ, a = k • (1 : ZMod n) :=
  ⟨a.cast, by simp⟩

example : veriphy_instance% KaluzaKlein.Sets.Products.zmod_cyclic (6) (4) := by
  exact ⟨4, by decide⟩

example : veriphy_instance% KaluzaKlein.Sets.Products.zmod_cyclic (1) (0) := by
  exact ⟨0, by decide⟩

example : veriphy_instance% KaluzaKlein.Sets.Products.zmod_cyclic (3) (2) := by
  exact ⟨2, by decide⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Products.zmod_cyclic
-- test: set-zmod-cyclic-test.sage

/-! ## exr-quotient-z5-card -/

/-- Two distinct remainders `r, s < n` are not congruent modulo `n`: the `n` classes are distinct. -/
theorem remainders_distinct (n r s : ℕ) (hr : r < n) (hs : s < n) (h : (r : ℤ) ≡ s [ZMOD n]) : r = s := by
  rw [Int.ModEq, Int.emod_eq_of_lt (by omega) (by omega), Int.emod_eq_of_lt (by omega) (by omega)] at h
  exact_mod_cast h

example : veriphy_instance% KaluzaKlein.Sets.Products.remainders_distinct (5) (2) (2) ((by norm_num)) ((by norm_num)) ((by decide)) := by
  have key : ∀ (n r s : ℕ) (hr : r < n) (hs : s < n) (h : (r : ℤ) ≡ s [ZMOD n]), r = s := by
    intro n r s hr hs h
    rw [Int.ModEq, Int.emod_eq_of_lt (by omega) (by omega), Int.emod_eq_of_lt (by omega) (by omega)] at h
    exact_mod_cast h
  exact @key (5) (2) (2) ((by norm_num)) ((by norm_num)) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Products.remainders_distinct (1) (0) (0) ((by norm_num)) ((by norm_num)) ((by decide)) := by
  have key : ∀ (n r s : ℕ) (hr : r < n) (hs : s < n) (h : (r : ℤ) ≡ s [ZMOD n]), r = s := by
    intro n r s hr hs h
    rw [Int.ModEq, Int.emod_eq_of_lt (by omega) (by omega), Int.emod_eq_of_lt (by omega) (by omega)] at h
    exact_mod_cast h
  exact @key (1) (0) (0) ((by norm_num)) ((by norm_num)) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Products.remainders_distinct (12) (7) (7) ((by norm_num)) ((by norm_num)) ((by decide)) := by
  have key : ∀ (n r s : ℕ) (hr : r < n) (hs : s < n) (h : (r : ℤ) ≡ s [ZMOD n]), r = s := by
    intro n r s hr hs h
    rw [Int.ModEq, Int.emod_eq_of_lt (by omega) (by omega), Int.emod_eq_of_lt (by omega) (by omega)] at h
    exact_mod_cast h
  exact @key (12) (7) (7) ((by norm_num)) ((by norm_num)) ((by decide))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Products.remainders_distinct
-- test: set-remainders-distinct-test.sage

/-- The inverses in `ℤ₅` (Sage: `[1]⁻¹ = [1]`, `[2]⁻¹ = [3]`, `[3]⁻¹ = [2]`, `[4]⁻¹ = [4]`). -/
theorem zmod5_inverses : ∃ b₁ b₂ b₃ b₄ : ZMod 5, 1 * b₁ = 1 ∧ 2 * b₂ = 1 ∧ 3 * b₃ = 1 ∧ 4 * b₄ = 1 := by
  veriphy_certified (exact ⟨1, 3, 2, 4, by decide⟩)
  -- certificate: set-zmod5-inverses.sage

/-! ## exr-quotient-z12-classes -/

/-- The multiples of `4` fall into the classes `[0], [4], [8]` of `ℤ₁₂`. -/
theorem multiples_of_four (k : ℤ) (hk : 4 ∣ k) : (k : ZMod 12) = 0 ∨ (k : ZMod 12) = 4 ∨ (k : ZMod 12) = 8 := by
  obtain ⟨m, rfl⟩ := hk
  rw [← ZMod.intCast_mod (4 * m) 12]
  have h : (4 * m) % ((12 : ℕ) : ℤ) = 0 ∨ (4 * m) % ((12 : ℕ) : ℤ) = 4 ∨ (4 * m) % ((12 : ℕ) : ℤ) = 8 := by omega
  rcases h with h | h | h <;> rw [h] <;> simp

example : veriphy_instance% KaluzaKlein.Sets.Products.multiples_of_four (0) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Products.multiples_of_four (4) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Products.multiples_of_four (20) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Products.multiples_of_four (-4) ((by decide)) := by
  decide

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.Sets.Products.multiples_of_four
-- test: set-multiples-four-test.sage

/-- `k mod 12 ↦ k mod 4` is a surjection `ℤ₁₂ → ℤ₄` whose preimages have three elements each. -/
theorem z12_to_z4 (y : ZMod 4) :
    (Finset.univ.filter fun x : ZMod 12 => ZMod.castHom (by norm_num : 4 ∣ 12) (ZMod 4) x = y).card = 3 := by
  fin_cases y <;> decide

example : veriphy_instance% KaluzaKlein.Sets.Products.z12_to_z4 (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Products.z12_to_z4 (1) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Products.z12_to_z4 (3) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Products.z12_to_z4
-- test: set-z12-z4-test.sage

end KaluzaKlein.Sets.Products
