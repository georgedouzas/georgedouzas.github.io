import Mathlib
import KaluzaKlein.PartI.Sets.Maps
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Maps" (injective maps)

The claims of the blocks `def-injective` to `exr-injective-sphere-parametrisation` of
`tutorial/part-i-01-sets-relations-topology.qmd`: injectivity through preimages, the pigeonhole principle, the
exponential and the square, the circle parametrisation on an arc, the count of injective maps, the cubics `x³ + x` and
`x³ - x`, and spherical coordinates.
-/

namespace KaluzaKlein.Sets.Injective

open KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Maps

/-- Spherical coordinates `(θ, φ) ↦ (sin θ cos φ, sin θ sin φ, cos θ)`. -/
noncomputable def sph (θ φ : ℝ) : Fin 3 → ℝ := ![Real.sin θ * Real.cos φ, Real.sin θ * Real.sin φ, Real.cos θ]

/-! ## def-injective -/

/-- A map is injective exactly when every point of the codomain has at most one preimage. -/
theorem injective_iff_preimage_subsingleton {X Y : Type} (f : X → Y) :
    Function.Injective f ↔ ∀ y, (f ⁻¹' {y}).Subsingleton :=
  ⟨fun h _ _ ha _ hb => h (ha.trans hb.symm), fun h _ b hab => h (f b) hab rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.Injective.injective_iff_preimage_subsingleton (Fin 2) (Fin 2) (id) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Injective f ↔ ∀ y, (f ⁻¹' {y}).Subsingleton := by
    intro X Y f
    exact (⟨fun h _ _ ha _ hb => h (ha.trans hb.symm), fun h _ b hab => h (f b) hab rfl⟩)
  exact @key (Fin 2) (Fin 2) (id)

example : veriphy_instance% KaluzaKlein.Sets.Injective.injective_iff_preimage_subsingleton (Fin 3) (ℕ) (![1, 2, 3]) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Injective f ↔ ∀ y, (f ⁻¹' {y}).Subsingleton := by
    intro X Y f
    exact (⟨fun h _ _ ha _ hb => h (ha.trans hb.symm), fun h _ b hab => h (f b) hab rfl⟩)
  exact @key (Fin 3) (ℕ) (![1, 2, 3])

example : veriphy_instance% KaluzaKlein.Sets.Injective.injective_iff_preimage_subsingleton (Fin 2) (ℕ) (![0, 0]) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Injective f ↔ ∀ y, (f ⁻¹' {y}).Subsingleton := by
    intro X Y f
    exact (⟨fun h _ _ ha _ hb => h (ha.trans hb.symm), fun h _ b hab => h (f b) hab rfl⟩)
  exact @key (Fin 2) (ℕ) (![0, 0])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Injective.injective_iff_preimage_subsingleton
-- test: set-injective-iff-test.sage

/-! ## exm-injective-finite -/

/-- `1 ↦ x`, `2 ↦ y` is injective when `x ≠ y`. -/
theorem pair_injective {β : Type} (x y : β) (hxy : x ≠ y) : Function.Injective ![x, y] := by
  intro i j h
  fin_cases i <;> fin_cases j <;> simp_all [eq_comm]

example : veriphy_instance% KaluzaKlein.Sets.Injective.pair_injective (Fin 3) (0) (1) ((by decide)) := by
  have key : ∀ {β : Type} (x y : β) (hxy : x ≠ y), Function.Injective ![x, y] := by
    intro β x y hxy
    intro i j h
    fin_cases i <;> fin_cases j <;> simp_all [eq_comm]
  exact @key (Fin 3) (0) (1) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Injective.pair_injective (ℕ) (4) (9) ((by decide)) := by
  have key : ∀ {β : Type} (x y : β) (hxy : x ≠ y), Function.Injective ![x, y] := by
    intro β x y hxy
    intro i j h
    fin_cases i <;> fin_cases j <;> simp_all [eq_comm]
  exact @key (ℕ) (4) (9) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Injective.pair_injective (Bool) (true) (false) ((by decide)) := by
  have key : ∀ {β : Type} (x y : β) (hxy : x ≠ y), Function.Injective ![x, y] := by
    intro β x y hxy
    intro i j h
    fin_cases i <;> fin_cases j <;> simp_all [eq_comm]
  exact @key (Bool) (true) (false) ((by decide))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Injective.pair_injective
-- test: set-pair-injective-test.sage

/-- The pigeonhole principle: no map from an `m`-element set to an `n`-element set with `n < m` is injective (none
from three elements to two). -/
theorem no_injection (m n : ℕ) (h : n < m) (f : Fin m → Fin n) : ¬ Function.Injective f := fun hf => by
  have := Fintype.card_le_of_injective f hf
  simp at this
  omega

example : veriphy_instance% KaluzaKlein.Sets.Injective.no_injection (3) (2) ((by norm_num)) (![0, 1, 1]) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Injective.no_injection (2) (1) ((by norm_num)) (![0, 0]) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Injective.no_injection (4) (3) ((by norm_num)) (![0, 1, 2, 0]) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Injective.no_injection
-- test: set-no-injection-test.sage

/-! ## exm-injective-exp -/

/-- `x ↦ e^{ax}` is injective for `a ≠ 0`: `exp` itself (`a = 1`). -/
theorem exp_mul_injective (a : ℝ) (ha : a ≠ 0) : Function.Injective (fun x => Real.exp (a * x)) :=
  Real.exp_injective.comp (mul_right_injective₀ ha)

example : veriphy_instance% KaluzaKlein.Sets.Injective.exp_mul_injective (1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : a ≠ 0), Function.Injective (fun x => Real.exp (a * x)) := by
    intro a ha
    exact (Real.exp_injective.comp (mul_right_injective₀ ha))
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Injective.exp_mul_injective (2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : a ≠ 0), Function.Injective (fun x => Real.exp (a * x)) := by
    intro a ha
    exact (Real.exp_injective.comp (mul_right_injective₀ ha))
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Injective.exp_mul_injective (-1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : a ≠ 0), Function.Injective (fun x => Real.exp (a * x)) := by
    intro a ha
    exact (Real.exp_injective.comp (mul_right_injective₀ ha))
  exact @key (-1) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Injective.exp_mul_injective
-- test: set-exp-injective-test.sage

/-- `x ↦ x²` is not injective: two distinct arguments with the same square (Sage finds `(-1)² = 1²`). -/
theorem sq_not_injective : ∃ x y : ℝ, x ≠ y ∧ x ^ 2 = y ^ 2 := by
  veriphy_certified (exact ⟨-1, 1, by norm_num⟩)
  -- certificate: set-witness-search.sage

/-! ## exm-injective-chart -/

/-- `φ ↦ (cos φ, sin φ)` is injective on every open interval `(a, a + 2π)` of length `2π`: on `(0, 2π)` for `a = 0`. -/
theorem wrap_injOn (a : ℝ) : Set.InjOn wrap (Set.Ioo a (a + 2 * Real.pi)) := by
  intro s hs t ht hst
  have h := wrap_preimage t
  have hs' : s ∈ wrap ⁻¹' {wrap t} := hst
  rw [h] at hs'
  obtain ⟨k, hk⟩ := hs'
  have hpi := Real.pi_pos
  have hk1 : (k : ℝ) * (2 * Real.pi) < 2 * Real.pi := by linarith [hs.1, hs.2, ht.1, ht.2]
  have hk2 : -(2 * Real.pi) < (k : ℝ) * (2 * Real.pi) := by linarith [hs.1, hs.2, ht.1, ht.2]
  have hk3 : (k : ℝ) < 1 := by nlinarith
  have hk4 : (-1 : ℝ) < k := by nlinarith
  have : k = 0 := by
    have h1 : k < 1 := by exact_mod_cast hk3
    have h2 : -1 < k := by exact_mod_cast hk4
    omega
  subst this
  simpa using hk

example : veriphy_instance% KaluzaKlein.Sets.Injective.wrap_injOn (0) := by
  have key : ∀ (a : ℝ), Set.InjOn wrap (Set.Ioo a (a + 2 * Real.pi)) := by
    intro a
    intro s hs t ht hst
    have h := wrap_preimage t
    have hs' : s ∈ wrap ⁻¹' {wrap t} := hst
    rw [h] at hs'
    obtain ⟨k, hk⟩ := hs'
    have hpi := Real.pi_pos
    have hk1 : (k : ℝ) * (2 * Real.pi) < 2 * Real.pi := by linarith [hs.1, hs.2, ht.1, ht.2]
    have hk2 : -(2 * Real.pi) < (k : ℝ) * (2 * Real.pi) := by linarith [hs.1, hs.2, ht.1, ht.2]
    have hk3 : (k : ℝ) < 1 := by nlinarith
    have hk4 : (-1 : ℝ) < k := by nlinarith
    have : k = 0 := by
      have h1 : k < 1 := by exact_mod_cast hk3
      have h2 : -1 < k := by exact_mod_cast hk4
      omega
    subst this
    simpa using hk
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Injective.wrap_injOn (-Real.pi) := by
  have key : ∀ (a : ℝ), Set.InjOn wrap (Set.Ioo a (a + 2 * Real.pi)) := by
    intro a
    intro s hs t ht hst
    have h := wrap_preimage t
    have hs' : s ∈ wrap ⁻¹' {wrap t} := hst
    rw [h] at hs'
    obtain ⟨k, hk⟩ := hs'
    have hpi := Real.pi_pos
    have hk1 : (k : ℝ) * (2 * Real.pi) < 2 * Real.pi := by linarith [hs.1, hs.2, ht.1, ht.2]
    have hk2 : -(2 * Real.pi) < (k : ℝ) * (2 * Real.pi) := by linarith [hs.1, hs.2, ht.1, ht.2]
    have hk3 : (k : ℝ) < 1 := by nlinarith
    have hk4 : (-1 : ℝ) < k := by nlinarith
    have : k = 0 := by
      have h1 : k < 1 := by exact_mod_cast hk3
      have h2 : -1 < k := by exact_mod_cast hk4
      omega
    subst this
    simpa using hk
  exact @key (-Real.pi)

example : veriphy_instance% KaluzaKlein.Sets.Injective.wrap_injOn (1) := by
  have key : ∀ (a : ℝ), Set.InjOn wrap (Set.Ioo a (a + 2 * Real.pi)) := by
    intro a
    intro s hs t ht hst
    have h := wrap_preimage t
    have hs' : s ∈ wrap ⁻¹' {wrap t} := hst
    rw [h] at hs'
    obtain ⟨k, hk⟩ := hs'
    have hpi := Real.pi_pos
    have hk1 : (k : ℝ) * (2 * Real.pi) < 2 * Real.pi := by linarith [hs.1, hs.2, ht.1, ht.2]
    have hk2 : -(2 * Real.pi) < (k : ℝ) * (2 * Real.pi) := by linarith [hs.1, hs.2, ht.1, ht.2]
    have hk3 : (k : ℝ) < 1 := by nlinarith
    have hk4 : (-1 : ℝ) < k := by nlinarith
    have : k = 0 := by
      have h1 : k < 1 := by exact_mod_cast hk3
      have h2 : -1 < k := by exact_mod_cast hk4
      omega
    subst this
    simpa using hk
  exact @key (1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Injective.wrap_injOn
-- test: set-wrap-injon-test.sage

/-- `φ` and `φ + 2π` give the same point, so the parametrisation is not injective on `ℝ`. -/
theorem wrap_periodic (φ : ℝ) : wrap (φ + 2 * Real.pi) = wrap φ := by
  simp [wrap]

example : veriphy_instance% KaluzaKlein.Sets.Injective.wrap_periodic (0) := by
  have key : ∀ (φ : ℝ), wrap (φ + 2 * Real.pi) = wrap φ := by
    intro φ
    simp [wrap]
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Injective.wrap_periodic (1) := by
  have key : ∀ (φ : ℝ), wrap (φ + 2 * Real.pi) = wrap φ := by
    intro φ
    simp [wrap]
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Injective.wrap_periodic (Real.pi / 3) := by
  have key : ∀ (φ : ℝ), wrap (φ + 2 * Real.pi) = wrap φ := by
    intro φ
    simp [wrap]
  exact @key (Real.pi / 3)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Injective.wrap_periodic
-- test: set-wrap-periodic-test.sage

/-! ## exr-injective-finite-check -/

/-- With distinct labels `a, b, c, d`: `(a, b, c)` and `(d, c, b)` are injective, `(a, a, b)` is not. -/
theorem three_maps_injectivity {β : Type} [DecidableEq β] (a b c d : β) (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d)
    (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
    Function.Injective ![a, b, c] ∧ ¬ Function.Injective ![a, a, b] ∧ Function.Injective ![d, c, b] := by
  refine ⟨?_, fun h => ?_, ?_⟩
  · intro i j h
    fin_cases i <;> fin_cases j <;> simp_all [eq_comm]
  · exact absurd (h (show ![a, a, b] 0 = ![a, a, b] 1 from rfl)) (by decide)
  · intro i j h
    fin_cases i <;> fin_cases j <;> simp_all [eq_comm]

example : veriphy_instance% KaluzaKlein.Sets.Injective.three_maps_injectivity (Fin 4) (inferInstance) (0) (1) (2) (3) ((by decide)) ((by decide)) ((by decide)) ((by decide)) ((by decide)) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Injective.three_maps_injectivity (ℕ) (inferInstance) (10) (20) (30) (40) ((by decide)) ((by decide)) ((by decide)) ((by decide)) ((by decide)) ((by decide)) := by
  have key : ∀ {β : Type} [inst1 : DecidableEq β] (a b c d : β) (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d), Function.Injective ![a, b, c] ∧ ¬ Function.Injective ![a, a, b] ∧ Function.Injective ![d, c, b] := by
    intro β inst1 a b c d hab hac had hbc hbd hcd
    refine ⟨?_, fun h => ?_, ?_⟩
    · intro i j h
      fin_cases i <;> fin_cases j <;> simp_all [eq_comm]
    · exact absurd (h (show ![a, a, b] 0 = ![a, a, b] 1 from rfl)) (by decide)
    · intro i j h
      fin_cases i <;> fin_cases j <;> simp_all [eq_comm]
  exact @key (ℕ) (inferInstance) (10) (20) (30) (40) ((by decide)) ((by decide)) ((by decide)) ((by decide)) ((by decide)) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Injective.three_maps_injectivity (Fin 5) (inferInstance) (4) (2) (0) (1) ((by decide)) ((by decide)) ((by decide)) ((by decide)) ((by decide)) ((by decide)) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Injective.three_maps_injectivity
-- test: set-three-maps-test.sage

/-- There are `n (n - 1) ⋯ (n - m + 1)` injective maps from an `m`-element set to an `n`-element set: `4 · 3 · 2 =
24` from three elements to four. -/
theorem card_injections (m n : ℕ) : Fintype.card (Fin m ↪ Fin n) = n.descFactorial m := by
  simp [Fintype.card_embedding_eq]

example : veriphy_instance% KaluzaKlein.Sets.Injective.card_injections (3) (4) := by
  rw [Fintype.card_embedding_eq]; decide

example : veriphy_instance% KaluzaKlein.Sets.Injective.card_injections (0) (5) := by
  rw [Fintype.card_embedding_eq]; decide

example : veriphy_instance% KaluzaKlein.Sets.Injective.card_injections (2) (2) := by
  rw [Fintype.card_embedding_eq]; decide

example : veriphy_instance% KaluzaKlein.Sets.Injective.card_injections (3) (2) := by
  rw [Fintype.card_embedding_eq]; decide

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.Sets.Injective.card_injections
-- test: set-card-injections-test.sage

/-! ## exr-injective-monotone -/

/-- `f'(x) = 3x² + 1` for `f(x) = x³ + x`. -/
theorem deriv_cubic (x : ℝ) : deriv (fun x : ℝ => x ^ 3 + x) x = 3 * x ^ 2 + 1 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x ^ 3 + x) ((3 * (x ^ 2) + 1 : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).pow 3).add (hasDerivAt_id' (x : ℝ))) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: set-deriv.sage

/-- `x ↦ x³ + ax` (`a ≥ 0`) is strictly increasing, hence injective: `x³ + x` among them. -/
theorem cubic_strictMono (a : ℝ) (ha : 0 ≤ a) : StrictMono (fun x : ℝ => x ^ 3 + a * x) ∧
    Function.Injective (fun x : ℝ => x ^ 3 + a * x) := by
  have h : StrictMono (fun x : ℝ => x ^ 3 + a * x) := by
    intro x y hxy
    simp only
    have h1 : 0 < y - x := sub_pos.2 hxy
    have h2 : 0 < y ^ 2 + x * y + x ^ 2 + a := by nlinarith [sq_nonneg (x + y), mul_pos h1 h1]
    nlinarith [mul_pos h1 h2]
  exact ⟨h, h.injective⟩

example : veriphy_instance% KaluzaKlein.Sets.Injective.cubic_strictMono (1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 ≤ a), StrictMono (fun x : ℝ => x ^ 3 + a * x) ∧ Function.Injective (fun x : ℝ => x ^ 3 + a * x) := by
    intro a ha
    have h : StrictMono (fun x : ℝ => x ^ 3 + a * x) := by
      intro x y hxy
      simp only
      have h1 : 0 < y - x := sub_pos.2 hxy
      have h2 : 0 < y ^ 2 + x * y + x ^ 2 + a := by nlinarith [sq_nonneg (x + y), mul_pos h1 h1]
      nlinarith [mul_pos h1 h2]
    exact ⟨h, h.injective⟩
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Injective.cubic_strictMono (0) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 ≤ a), StrictMono (fun x : ℝ => x ^ 3 + a * x) ∧ Function.Injective (fun x : ℝ => x ^ 3 + a * x) := by
    intro a ha
    have h : StrictMono (fun x : ℝ => x ^ 3 + a * x) := by
      intro x y hxy
      simp only
      have h1 : 0 < y - x := sub_pos.2 hxy
      have h2 : 0 < y ^ 2 + x * y + x ^ 2 + a := by nlinarith [sq_nonneg (x + y), mul_pos h1 h1]
      nlinarith [mul_pos h1 h2]
    exact ⟨h, h.injective⟩
  exact @key (0) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Injective.cubic_strictMono (2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 ≤ a), StrictMono (fun x : ℝ => x ^ 3 + a * x) ∧ Function.Injective (fun x : ℝ => x ^ 3 + a * x) := by
    intro a ha
    have h : StrictMono (fun x : ℝ => x ^ 3 + a * x) := by
      intro x y hxy
      simp only
      have h1 : 0 < y - x := sub_pos.2 hxy
      have h2 : 0 < y ^ 2 + x * y + x ^ 2 + a := by nlinarith [sq_nonneg (x + y), mul_pos h1 h1]
      nlinarith [mul_pos h1 h2]
    exact ⟨h, h.injective⟩
  exact @key (2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Injective.cubic_strictMono
-- test: set-cubic-strictmono-test.sage

/-- `g(x) = x³ - x` has three distinct roots (Sage finds `-1, 0, 1`). -/
theorem cubic_three_roots : ∃ x y z : ℝ, x < y ∧ y < z ∧ x ^ 3 - x = 0 ∧ y ^ 3 - y = 0 ∧ z ^ 3 - z = 0 := by
  veriphy_certified (exact ⟨-1, 0, 1, by norm_num⟩)
  -- certificate: set-witness-search.sage

/-- `g(x) = x³ - x` is not injective: two distinct arguments with the same value. -/
theorem cubic_not_injective : ∃ x y : ℝ, x ≠ y ∧ x ^ 3 - x = y ^ 3 - y := by
  veriphy_certified (exact ⟨0, -1, by norm_num⟩)
  -- certificate: set-witness-search.sage

/-! ## exr-injective-sphere-parametrisation -/

/-- Spherical coordinates are injective on `(0, π) × (a, a + 2π)`: on `(0, π) × (0, 2π)` for `a = 0`. -/
theorem sph_injOn (a : ℝ) :
    Set.InjOn (fun q : ℝ × ℝ => sph q.1 q.2) (Set.Ioo 0 Real.pi ×ˢ Set.Ioo a (a + 2 * Real.pi)) := by
  rintro ⟨θ, φ⟩ ⟨hθ, hφ⟩ ⟨θ', φ'⟩ ⟨hθ', hφ'⟩ h
  simp only at h hθ hφ hθ' hφ'
  have h0 := congrFun h 0
  have h1 := congrFun h 1
  have h2 := congrFun h 2
  simp only [sph, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two,
    Matrix.tail_cons] at h0 h1 h2
  have hθθ : θ = θ' := Real.injOn_cos ⟨hθ.1.le, hθ.2.le⟩ ⟨hθ'.1.le, hθ'.2.le⟩ h2
  subst hθθ
  have hs : Real.sin θ ≠ 0 := (Real.sin_pos_of_pos_of_lt_pi hθ.1 hθ.2).ne'
  have hc : Real.cos φ = Real.cos φ' := mul_left_cancel₀ hs h0
  have hsn : Real.sin φ = Real.sin φ' := mul_left_cancel₀ hs h1
  have : φ = φ' := wrap_injOn a hφ hφ' (by simp [wrap, hc, hsn])
  rw [this]

example : veriphy_instance% KaluzaKlein.Sets.Injective.sph_injOn (0) := by
  have key : ∀ (a : ℝ), Set.InjOn (fun q : ℝ × ℝ => sph q.1 q.2) (Set.Ioo 0 Real.pi ×ˢ Set.Ioo a (a + 2 * Real.pi)) := by
    intro a
    rintro ⟨θ, φ⟩ ⟨hθ, hφ⟩ ⟨θ', φ'⟩ ⟨hθ', hφ'⟩ h
    simp only at h hθ hφ hθ' hφ'
    have h0 := congrFun h 0
    have h1 := congrFun h 1
    have h2 := congrFun h 2
    simp only [sph, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two,
      Matrix.tail_cons] at h0 h1 h2
    have hθθ : θ = θ' := Real.injOn_cos ⟨hθ.1.le, hθ.2.le⟩ ⟨hθ'.1.le, hθ'.2.le⟩ h2
    subst hθθ
    have hs : Real.sin θ ≠ 0 := (Real.sin_pos_of_pos_of_lt_pi hθ.1 hθ.2).ne'
    have hc : Real.cos φ = Real.cos φ' := mul_left_cancel₀ hs h0
    have hsn : Real.sin φ = Real.sin φ' := mul_left_cancel₀ hs h1
    have : φ = φ' := wrap_injOn a hφ hφ' (by simp [wrap, hc, hsn])
    rw [this]
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Injective.sph_injOn (-Real.pi) := by
  have key : ∀ (a : ℝ), Set.InjOn (fun q : ℝ × ℝ => sph q.1 q.2) (Set.Ioo 0 Real.pi ×ˢ Set.Ioo a (a + 2 * Real.pi)) := by
    intro a
    rintro ⟨θ, φ⟩ ⟨hθ, hφ⟩ ⟨θ', φ'⟩ ⟨hθ', hφ'⟩ h
    simp only at h hθ hφ hθ' hφ'
    have h0 := congrFun h 0
    have h1 := congrFun h 1
    have h2 := congrFun h 2
    simp only [sph, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two,
      Matrix.tail_cons] at h0 h1 h2
    have hθθ : θ = θ' := Real.injOn_cos ⟨hθ.1.le, hθ.2.le⟩ ⟨hθ'.1.le, hθ'.2.le⟩ h2
    subst hθθ
    have hs : Real.sin θ ≠ 0 := (Real.sin_pos_of_pos_of_lt_pi hθ.1 hθ.2).ne'
    have hc : Real.cos φ = Real.cos φ' := mul_left_cancel₀ hs h0
    have hsn : Real.sin φ = Real.sin φ' := mul_left_cancel₀ hs h1
    have : φ = φ' := wrap_injOn a hφ hφ' (by simp [wrap, hc, hsn])
    rw [this]
  exact @key (-Real.pi)

example : veriphy_instance% KaluzaKlein.Sets.Injective.sph_injOn (1) := by
  have key : ∀ (a : ℝ), Set.InjOn (fun q : ℝ × ℝ => sph q.1 q.2) (Set.Ioo 0 Real.pi ×ˢ Set.Ioo a (a + 2 * Real.pi)) := by
    intro a
    rintro ⟨θ, φ⟩ ⟨hθ, hφ⟩ ⟨θ', φ'⟩ ⟨hθ', hφ'⟩ h
    simp only at h hθ hφ hθ' hφ'
    have h0 := congrFun h 0
    have h1 := congrFun h 1
    have h2 := congrFun h 2
    simp only [sph, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two,
      Matrix.tail_cons] at h0 h1 h2
    have hθθ : θ = θ' := Real.injOn_cos ⟨hθ.1.le, hθ.2.le⟩ ⟨hθ'.1.le, hθ'.2.le⟩ h2
    subst hθθ
    have hs : Real.sin θ ≠ 0 := (Real.sin_pos_of_pos_of_lt_pi hθ.1 hθ.2).ne'
    have hc : Real.cos φ = Real.cos φ' := mul_left_cancel₀ hs h0
    have hsn : Real.sin φ = Real.sin φ' := mul_left_cancel₀ hs h1
    have : φ = φ' := wrap_injOn a hφ hφ' (by simp [wrap, hc, hsn])
    rw [this]
  exact @key (1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Injective.sph_injOn
-- test: set-sph-injon-test.sage

/-- Every `(0, φ)` is sent to the north pole: on `[0, π] × [0, 2π)` the pole has infinitely many preimages. -/
theorem sph_pole (φ : ℝ) : sph 0 φ = northPole := by
  ext i; fin_cases i <;> simp [sph, northPole]

example : veriphy_instance% KaluzaKlein.Sets.Injective.sph_pole (0) := by
  have key : ∀ (φ : ℝ), sph 0 φ = northPole := by
    intro φ
    ext i; fin_cases i <;> simp [sph, northPole]
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Injective.sph_pole (1) := by
  have key : ∀ (φ : ℝ), sph 0 φ = northPole := by
    intro φ
    ext i; fin_cases i <;> simp [sph, northPole]
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Injective.sph_pole (Real.pi) := by
  have key : ∀ (φ : ℝ), sph 0 φ = northPole := by
    intro φ
    ext i; fin_cases i <;> simp [sph, northPole]
  exact @key (Real.pi)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Injective.sph_pole
-- test: set-sph-pole-test.sage

end KaluzaKlein.Sets.Injective
