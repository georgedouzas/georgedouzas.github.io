import Mathlib
import KaluzaKlein.PartI.Sets.Sets
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Maps" (maps, images and preimages)

The claims of the blocks `def-map` to `exr-map-height-cap` of `tutorial/part-i-01-sets-relations-topology.qmd`: images
and preimages of maps between finite sets, of `cos`, of the circle parametrisation, of the height function on the
sphere and of the orbit map `SO(3) → S²`, and of `x ↦ x²`. The rotation `rotTo p` sending the north pole to a point
`p` of the sphere is used again for surjectivity and quotients.
-/

namespace KaluzaKlein.Sets.Maps

open Matrix KaluzaKlein.Sets.Sets

/-- The parametrisation `φ ↦ (cos φ, sin φ)` of the circle. -/
noncomputable def wrap (φ : ℝ) : ℝ × ℝ := (Real.cos φ, Real.sin φ)

/-- The height function `(X, Y, Z) ↦ Z`. -/
def height (p : Fin 3 → ℝ) : ℝ := p 2

/-- The orbit map `g ↦ g · (0, 0, 1)`. -/
def orbit (g : Matrix (Fin 3) (Fin 3) ℝ) : Fin 3 → ℝ := g *ᵥ northPole

/-- The numerator of the rotation sending the north pole to `(a, b, c)`, `c ≠ -1`: `(1 + c)` times the rotation in
the plane of the pole and `(a, b, c)`, whose last column is `(a, b, c)`. -/
def rotNum (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![(1 + c) - a ^ 2, -(a * b), (1 + c) * a;
     -(a * b), (1 + c) - b ^ 2, (1 + c) * b;
     -((1 + c) * a), -((1 + c) * b), (1 + c) * c]

/-- A rotation sending the north pole to `p = (a, b, c)`: `rotNum a b c / (1 + c)` for `c ≠ -1`, and the half turn
about the `X`-axis for the south pole. -/
noncomputable def rotTo (p : Fin 3 → ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  if p 2 = -1 then !![1, 0, 0; 0, -1, 0; 0, 0, -1] else (1 + p 2)⁻¹ • rotNum (p 0) (p 1) (p 2)

/-! ## exm-map-finite -/

/-- `f(1) = f(2) = x`, `f(3) = y` (as `![x, x, y]` on `{0, 1, 2}`) has image `{x, y}` and `f⁻¹({x}) = {1, 2}`. -/
theorem finite_map_image_preimage {β : Type} (x y : β) (hxy : x ≠ y) :
    Set.range ![x, x, y] = {x, y} ∧ ![x, x, y] ⁻¹' {x} = {0, 1} := by
  constructor
  · ext z
    simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]
    constructor
    · rintro ⟨i, rfl⟩; fin_cases i <;> simp
    · rintro (rfl | rfl)
      · exact ⟨0, rfl⟩
      · exact ⟨2, rfl⟩
  · ext i
    fin_cases i <;> simp [hxy.symm]

example : veriphy_instance% KaluzaKlein.Sets.Maps.finite_map_image_preimage (Fin 2) (0) (1) ((by decide)) := by
  have key : ∀ {β : Type} (x y : β) (hxy : x ≠ y), Set.range ![x, x, y] = {x, y} ∧ ![x, x, y] ⁻¹' {x} = {0, 1} := by
    intro β x y hxy
    constructor
    · ext z
      simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]
      constructor
      · rintro ⟨i, rfl⟩; fin_cases i <;> simp
      · rintro (rfl | rfl)
        · exact ⟨0, rfl⟩
        · exact ⟨2, rfl⟩
    · ext i
      fin_cases i <;> simp [hxy.symm]
  exact @key (Fin 2) (0) (1) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Maps.finite_map_image_preimage (ℕ) (5) (7) ((by decide)) := by
  have key : ∀ {β : Type} (x y : β) (hxy : x ≠ y), Set.range ![x, x, y] = {x, y} ∧ ![x, x, y] ⁻¹' {x} = {0, 1} := by
    intro β x y hxy
    constructor
    · ext z
      simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]
      constructor
      · rintro ⟨i, rfl⟩; fin_cases i <;> simp
      · rintro (rfl | rfl)
        · exact ⟨0, rfl⟩
        · exact ⟨2, rfl⟩
    · ext i
      fin_cases i <;> simp [hxy.symm]
  exact @key (ℕ) (5) (7) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Maps.finite_map_image_preimage (Bool) (true) (false) ((by decide)) := by
  have key : ∀ {β : Type} (x y : β) (hxy : x ≠ y), Set.range ![x, x, y] = {x, y} ∧ ![x, x, y] ⁻¹' {x} = {0, 1} := by
    intro β x y hxy
    constructor
    · ext z
      simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]
      constructor
      · rintro ⟨i, rfl⟩; fin_cases i <;> simp
      · rintro (rfl | rfl)
        · exact ⟨0, rfl⟩
        · exact ⟨2, rfl⟩
    · ext i
      fin_cases i <;> simp [hxy.symm]
  exact @key (Bool) (true) (false) ((by decide))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Maps.finite_map_image_preimage
-- test: set-finite-map-test.sage

/-- There are `nᵐ` maps from an `m`-element set to an `n`-element set: `2³ = 8` from three elements to two. -/
theorem card_maps (m n : ℕ) : Fintype.card (Fin m → Fin n) = n ^ m := by
  simp

example : veriphy_instance% KaluzaKlein.Sets.Maps.card_maps (3) (2) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Maps.card_maps (0) (5) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Maps.card_maps (2) (3) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Maps.card_maps
-- test: set-card-maps-test.sage

/-! ## exm-map-functions -/

/-- `x ↦ a cos x` (`a ≥ 0`) has image `[-a, a]`: `cos` has image `[-1, 1]`. -/
theorem range_mul_cos (a : ℝ) (ha : 0 ≤ a) : Set.range (fun x => a * Real.cos x) = Set.Icc (-a) a := by
  ext y
  simp only [Set.mem_range, Set.mem_Icc]
  constructor
  · rintro ⟨x, rfl⟩
    constructor <;> nlinarith [Real.neg_one_le_cos x, Real.cos_le_one x]
  · rintro ⟨h1, h2⟩
    rcases eq_or_lt_of_le ha with rfl | ha'
    · exact ⟨0, by linarith⟩
    · obtain ⟨x, -, hx⟩ := Real.bijOn_cos.surjOn
        (show y / a ∈ Set.Icc (-1 : ℝ) 1 from ⟨by rw [le_div_iff₀ ha']; linarith,
          by rw [div_le_iff₀ ha']; linarith⟩)
      exact ⟨x, by rw [hx]; field_simp⟩

example : veriphy_instance% KaluzaKlein.Sets.Maps.range_mul_cos (1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 ≤ a), Set.range (fun x => a * Real.cos x) = Set.Icc (-a) a := by
    intro a ha
    ext y
    simp only [Set.mem_range, Set.mem_Icc]
    constructor
    · rintro ⟨x, rfl⟩
      constructor <;> nlinarith [Real.neg_one_le_cos x, Real.cos_le_one x]
    · rintro ⟨h1, h2⟩
      rcases eq_or_lt_of_le ha with rfl | ha'
      · exact ⟨0, by linarith⟩
      · obtain ⟨x, -, hx⟩ := Real.bijOn_cos.surjOn
          (show y / a ∈ Set.Icc (-1 : ℝ) 1 from ⟨by rw [le_div_iff₀ ha']; linarith,
            by rw [div_le_iff₀ ha']; linarith⟩)
        exact ⟨x, by rw [hx]; field_simp⟩
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Maps.range_mul_cos (0) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 ≤ a), Set.range (fun x => a * Real.cos x) = Set.Icc (-a) a := by
    intro a ha
    ext y
    simp only [Set.mem_range, Set.mem_Icc]
    constructor
    · rintro ⟨x, rfl⟩
      constructor <;> nlinarith [Real.neg_one_le_cos x, Real.cos_le_one x]
    · rintro ⟨h1, h2⟩
      rcases eq_or_lt_of_le ha with rfl | ha'
      · exact ⟨0, by linarith⟩
      · obtain ⟨x, -, hx⟩ := Real.bijOn_cos.surjOn
          (show y / a ∈ Set.Icc (-1 : ℝ) 1 from ⟨by rw [le_div_iff₀ ha']; linarith,
            by rw [div_le_iff₀ ha']; linarith⟩)
        exact ⟨x, by rw [hx]; field_simp⟩
  exact @key (0) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Maps.range_mul_cos (2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 ≤ a), Set.range (fun x => a * Real.cos x) = Set.Icc (-a) a := by
    intro a ha
    ext y
    simp only [Set.mem_range, Set.mem_Icc]
    constructor
    · rintro ⟨x, rfl⟩
      constructor <;> nlinarith [Real.neg_one_le_cos x, Real.cos_le_one x]
    · rintro ⟨h1, h2⟩
      rcases eq_or_lt_of_le ha with rfl | ha'
      · exact ⟨0, by linarith⟩
      · obtain ⟨x, -, hx⟩ := Real.bijOn_cos.surjOn
          (show y / a ∈ Set.Icc (-1 : ℝ) 1 from ⟨by rw [le_div_iff₀ ha']; linarith,
            by rw [div_le_iff₀ ha']; linarith⟩)
        exact ⟨x, by rw [hx]; field_simp⟩
  exact @key (2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Maps.range_mul_cos
-- test: set-range-cos-test.sage

/-- `φ ↦ (r cos φ, r sin φ)` (`r ≥ 0`) has image the circle `X² + Y² = r²`: for `r = 1`, the circle `S¹`. -/
theorem range_wrap (r : ℝ) (hr : 0 ≤ r) :
    Set.range (fun φ => (r * Real.cos φ, r * Real.sin φ)) = {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r ^ 2} := by
  ext ⟨x, y⟩
  simp only [Set.mem_range, Set.mem_setOf_eq, Prod.mk.injEq]
  constructor
  · rintro ⟨φ, rfl, rfl⟩
    nlinarith [Real.sin_sq_add_cos_sq φ]
  · intro h
    set z : ℂ := ⟨x, y⟩
    have hz : ‖z‖ = r := by
      rw [Complex.norm_def, Complex.normSq_mk, ← Real.sqrt_sq hr]
      congr 1; nlinarith
    refine ⟨Complex.arg z, ?_, ?_⟩
    · rw [← hz, Complex.norm_mul_cos_arg]
    · rw [← hz, Complex.norm_mul_sin_arg]

example : veriphy_instance% KaluzaKlein.Sets.Maps.range_wrap (1) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 ≤ r), Set.range (fun φ => (r * Real.cos φ, r * Real.sin φ)) = {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r ^ 2} := by
    intro r hr
    ext ⟨x, y⟩
    simp only [Set.mem_range, Set.mem_setOf_eq, Prod.mk.injEq]
    constructor
    · rintro ⟨φ, rfl, rfl⟩
      nlinarith [Real.sin_sq_add_cos_sq φ]
    · intro h
      set z : ℂ := ⟨x, y⟩
      have hz : ‖z‖ = r := by
        rw [Complex.norm_def, Complex.normSq_mk, ← Real.sqrt_sq hr]
        congr 1; nlinarith
      refine ⟨Complex.arg z, ?_, ?_⟩
      · rw [← hz, Complex.norm_mul_cos_arg]
      · rw [← hz, Complex.norm_mul_sin_arg]
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Maps.range_wrap (0) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 ≤ r), Set.range (fun φ => (r * Real.cos φ, r * Real.sin φ)) = {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r ^ 2} := by
    intro r hr
    ext ⟨x, y⟩
    simp only [Set.mem_range, Set.mem_setOf_eq, Prod.mk.injEq]
    constructor
    · rintro ⟨φ, rfl, rfl⟩
      nlinarith [Real.sin_sq_add_cos_sq φ]
    · intro h
      set z : ℂ := ⟨x, y⟩
      have hz : ‖z‖ = r := by
        rw [Complex.norm_def, Complex.normSq_mk, ← Real.sqrt_sq hr]
        congr 1; nlinarith
      refine ⟨Complex.arg z, ?_, ?_⟩
      · rw [← hz, Complex.norm_mul_cos_arg]
      · rw [← hz, Complex.norm_mul_sin_arg]
  exact @key (0) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Maps.range_wrap (3) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 ≤ r), Set.range (fun φ => (r * Real.cos φ, r * Real.sin φ)) = {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r ^ 2} := by
    intro r hr
    ext ⟨x, y⟩
    simp only [Set.mem_range, Set.mem_setOf_eq, Prod.mk.injEq]
    constructor
    · rintro ⟨φ, rfl, rfl⟩
      nlinarith [Real.sin_sq_add_cos_sq φ]
    · intro h
      set z : ℂ := ⟨x, y⟩
      have hz : ‖z‖ = r := by
        rw [Complex.norm_def, Complex.normSq_mk, ← Real.sqrt_sq hr]
        congr 1; nlinarith
      refine ⟨Complex.arg z, ?_, ?_⟩
      · rw [← hz, Complex.norm_mul_cos_arg]
      · rw [← hz, Complex.norm_mul_sin_arg]
  exact @key (3) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Maps.range_wrap
-- test: set-range-wrap-test.sage

/-- The preimage of the point `(cos a, sin a)` under `φ ↦ (cos φ, sin φ)` is `a + 2πℤ`: for `(1, 0)` (`a = 0`), the
multiples of `2π`. -/
theorem wrap_preimage (a : ℝ) : wrap ⁻¹' {wrap a} = {φ | ∃ k : ℤ, φ = a + k * (2 * Real.pi)} := by
  ext φ
  simp only [wrap, Set.mem_preimage, Set.mem_singleton_iff, Prod.mk.injEq, Set.mem_setOf_eq]
  have key : (Real.cos φ = Real.cos a ∧ Real.sin φ = Real.sin a) ↔
      Complex.exp (φ * Complex.I) = Complex.exp (a * Complex.I) := by
    rw [Complex.ext_iff, Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_im,
      Complex.exp_ofReal_mul_I_im]
  rw [key, Complex.exp_eq_exp_iff_exists_int]
  constructor
  · rintro ⟨n, hn⟩
    refine ⟨n, ?_⟩
    have := congrArg Complex.im hn
    simp at this
    linarith
  · rintro ⟨k, rfl⟩
    refine ⟨k, ?_⟩
    push_cast; ring

example : veriphy_instance% KaluzaKlein.Sets.Maps.wrap_preimage (0) := by
  have key : ∀ (a : ℝ), wrap ⁻¹' {wrap a} = {φ | ∃ k : ℤ, φ = a + k * (2 * Real.pi)} := by
    intro a
    ext φ
    simp only [wrap, Set.mem_preimage, Set.mem_singleton_iff, Prod.mk.injEq, Set.mem_setOf_eq]
    have key : (Real.cos φ = Real.cos a ∧ Real.sin φ = Real.sin a) ↔
        Complex.exp (φ * Complex.I) = Complex.exp (a * Complex.I) := by
      rw [Complex.ext_iff, Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_im,
        Complex.exp_ofReal_mul_I_im]
    rw [key, Complex.exp_eq_exp_iff_exists_int]
    constructor
    · rintro ⟨n, hn⟩
      refine ⟨n, ?_⟩
      have := congrArg Complex.im hn
      simp at this
      linarith
    · rintro ⟨k, rfl⟩
      refine ⟨k, ?_⟩
      push_cast; ring
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Maps.wrap_preimage (Real.pi / 2) := by
  have key : ∀ (a : ℝ), wrap ⁻¹' {wrap a} = {φ | ∃ k : ℤ, φ = a + k * (2 * Real.pi)} := by
    intro a
    ext φ
    simp only [wrap, Set.mem_preimage, Set.mem_singleton_iff, Prod.mk.injEq, Set.mem_setOf_eq]
    have key : (Real.cos φ = Real.cos a ∧ Real.sin φ = Real.sin a) ↔
        Complex.exp (φ * Complex.I) = Complex.exp (a * Complex.I) := by
      rw [Complex.ext_iff, Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_im,
        Complex.exp_ofReal_mul_I_im]
    rw [key, Complex.exp_eq_exp_iff_exists_int]
    constructor
    · rintro ⟨n, hn⟩
      refine ⟨n, ?_⟩
      have := congrArg Complex.im hn
      simp at this
      linarith
    · rintro ⟨k, rfl⟩
      refine ⟨k, ?_⟩
      push_cast; ring
  exact @key (Real.pi / 2)

example : veriphy_instance% KaluzaKlein.Sets.Maps.wrap_preimage (Real.pi) := by
  have key : ∀ (a : ℝ), wrap ⁻¹' {wrap a} = {φ | ∃ k : ℤ, φ = a + k * (2 * Real.pi)} := by
    intro a
    ext φ
    simp only [wrap, Set.mem_preimage, Set.mem_singleton_iff, Prod.mk.injEq, Set.mem_setOf_eq]
    have key : (Real.cos φ = Real.cos a ∧ Real.sin φ = Real.sin a) ↔
        Complex.exp (φ * Complex.I) = Complex.exp (a * Complex.I) := by
      rw [Complex.ext_iff, Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_im,
        Complex.exp_ofReal_mul_I_im]
    rw [key, Complex.exp_eq_exp_iff_exists_int]
    constructor
    · rintro ⟨n, hn⟩
      refine ⟨n, ?_⟩
      have := congrArg Complex.im hn
      simp at this
      linarith
    · rintro ⟨k, rfl⟩
      refine ⟨k, ?_⟩
      push_cast; ring
  exact @key (Real.pi)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Maps.wrap_preimage
-- test: set-wrap-preimage-test.sage

/-! ## exm-map-projection -/

/-- Each coordinate function on the sphere has image `[-1, 1]`: the height function `Z` (coordinate `2`) among them. -/
theorem coordinate_image_sphere (i : Fin 3) : (fun p : Fin 3 → ℝ => p i) '' sphere = Set.Icc (-1) 1 := by
  ext t
  simp only [Set.mem_image, Set.mem_Icc, sphere, Set.mem_setOf_eq]
  constructor
  · rintro ⟨p, hp, rfl⟩
    fin_cases i <;> simp <;> constructor <;> nlinarith [sq_nonneg (p 0), sq_nonneg (p 1), sq_nonneg (p 2)]
  · rintro ⟨h1, h2⟩
    have hs : Real.sqrt (1 - t ^ 2) ^ 2 = 1 - t ^ 2 := Real.sq_sqrt (by nlinarith)
    fin_cases i
    · exact ⟨![t, Real.sqrt (1 - t ^ 2), 0], by simp; linarith, by simp⟩
    · exact ⟨![Real.sqrt (1 - t ^ 2), t, 0], by simp; linarith, by simp⟩
    · exact ⟨![Real.sqrt (1 - t ^ 2), 0, t], by simp; linarith, by simp⟩

example : veriphy_instance% KaluzaKlein.Sets.Maps.coordinate_image_sphere (2) := by
  have key : ∀ (i : Fin 3), (fun p : Fin 3 → ℝ => p i) '' sphere = Set.Icc (-1) 1 := by
    intro i
    ext t
    simp only [Set.mem_image, Set.mem_Icc, sphere, Set.mem_setOf_eq]
    constructor
    · rintro ⟨p, hp, rfl⟩
      fin_cases i <;> simp <;> constructor <;> nlinarith [sq_nonneg (p 0), sq_nonneg (p 1), sq_nonneg (p 2)]
    · rintro ⟨h1, h2⟩
      have hs : Real.sqrt (1 - t ^ 2) ^ 2 = 1 - t ^ 2 := Real.sq_sqrt (by nlinarith)
      fin_cases i
      · exact ⟨![t, Real.sqrt (1 - t ^ 2), 0], by simp; linarith, by simp⟩
      · exact ⟨![Real.sqrt (1 - t ^ 2), t, 0], by simp; linarith, by simp⟩
      · exact ⟨![Real.sqrt (1 - t ^ 2), 0, t], by simp; linarith, by simp⟩
  exact @key (2)

example : veriphy_instance% KaluzaKlein.Sets.Maps.coordinate_image_sphere (0) := by
  have key : ∀ (i : Fin 3), (fun p : Fin 3 → ℝ => p i) '' sphere = Set.Icc (-1) 1 := by
    intro i
    ext t
    simp only [Set.mem_image, Set.mem_Icc, sphere, Set.mem_setOf_eq]
    constructor
    · rintro ⟨p, hp, rfl⟩
      fin_cases i <;> simp <;> constructor <;> nlinarith [sq_nonneg (p 0), sq_nonneg (p 1), sq_nonneg (p 2)]
    · rintro ⟨h1, h2⟩
      have hs : Real.sqrt (1 - t ^ 2) ^ 2 = 1 - t ^ 2 := Real.sq_sqrt (by nlinarith)
      fin_cases i
      · exact ⟨![t, Real.sqrt (1 - t ^ 2), 0], by simp; linarith, by simp⟩
      · exact ⟨![Real.sqrt (1 - t ^ 2), t, 0], by simp; linarith, by simp⟩
      · exact ⟨![Real.sqrt (1 - t ^ 2), 0, t], by simp; linarith, by simp⟩
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Maps.coordinate_image_sphere (1) := by
  have key : ∀ (i : Fin 3), (fun p : Fin 3 → ℝ => p i) '' sphere = Set.Icc (-1) 1 := by
    intro i
    ext t
    simp only [Set.mem_image, Set.mem_Icc, sphere, Set.mem_setOf_eq]
    constructor
    · rintro ⟨p, hp, rfl⟩
      fin_cases i <;> simp <;> constructor <;> nlinarith [sq_nonneg (p 0), sq_nonneg (p 1), sq_nonneg (p 2)]
    · rintro ⟨h1, h2⟩
      have hs : Real.sqrt (1 - t ^ 2) ^ 2 = 1 - t ^ 2 := Real.sq_sqrt (by nlinarith)
      fin_cases i
      · exact ⟨![t, Real.sqrt (1 - t ^ 2), 0], by simp; linarith, by simp⟩
      · exact ⟨![Real.sqrt (1 - t ^ 2), t, 0], by simp; linarith, by simp⟩
      · exact ⟨![Real.sqrt (1 - t ^ 2), 0, t], by simp; linarith, by simp⟩
  exact @key (1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Maps.coordinate_image_sphere
-- test: set-coordinate-image-test.sage

/-! The entries of `rotNumᵀ rotNum = (1 + c)² · 1` and `det rotNum = (1 + c)³` on the sphere `a² + b² + c² = 1`. -/

theorem rot_e00 (a b c : ℝ) (hp : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
    ((1 + c) - a ^ 2) * ((1 + c) - a ^ 2) + (-(a * b)) * (-(a * b)) + (-((1 + c) * a)) * (-((1 + c) * a)) =
      (1 + c) ^ 2 := by
  veriphy_certified (linear_combination (a ^ 2) * hp)
  -- certificate: ideal_membership.sage

theorem rot_e01 (a b c : ℝ) (hp : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
    ((1 + c) - a ^ 2) * (-(a * b)) + (-(a * b)) * ((1 + c) - b ^ 2) + (-((1 + c) * a)) * (-((1 + c) * b)) = 0 := by
  veriphy_certified (linear_combination (a * b) * hp)
  -- certificate: ideal_membership.sage

theorem rot_e02 (a b c : ℝ) (hp : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
    ((1 + c) - a ^ 2) * ((1 + c) * a) + (-(a * b)) * ((1 + c) * b) + (-((1 + c) * a)) * ((1 + c) * c) = 0 := by
  veriphy_certified (linear_combination (-c * a - a) * hp)
  -- certificate: ideal_membership.sage

theorem rot_e11 (a b c : ℝ) (hp : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
    (-(a * b)) * (-(a * b)) + ((1 + c) - b ^ 2) * ((1 + c) - b ^ 2) + (-((1 + c) * b)) * (-((1 + c) * b)) =
      (1 + c) ^ 2 := by
  veriphy_certified (linear_combination (b ^ 2) * hp)
  -- certificate: ideal_membership.sage

theorem rot_e12 (a b c : ℝ) (hp : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
    (-(a * b)) * ((1 + c) * a) + ((1 + c) - b ^ 2) * ((1 + c) * b) + (-((1 + c) * b)) * ((1 + c) * c) = 0 := by
  veriphy_certified (linear_combination (-b * c - b) * hp)
  -- certificate: ideal_membership.sage

theorem rot_e22 (a b c : ℝ) (hp : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
    ((1 + c) * a) * ((1 + c) * a) + ((1 + c) * b) * ((1 + c) * b) + ((1 + c) * c) * ((1 + c) * c) = (1 + c) ^ 2 := by
  veriphy_certified (linear_combination (c ^ 2 + 2 * c + 1) * hp)
  -- certificate: ideal_membership.sage

theorem rot_det (a b c : ℝ) (hp : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
    ((1 + c) - a ^ 2) * ((1 + c) - b ^ 2) * ((1 + c) * c) - ((1 + c) - a ^ 2) * ((1 + c) * b) * (-((1 + c) * b))
      - (-(a * b)) * (-(a * b)) * ((1 + c) * c) + (-(a * b)) * ((1 + c) * a) * (-((1 + c) * b))
      + ((1 + c) * a) * (-(a * b)) * (-((1 + c) * b)) - ((1 + c) * a) * ((1 + c) - b ^ 2) * (-((1 + c) * a)) =
      (1 + c) ^ 3 := by
  veriphy_certified (linear_combination (c ^ 2 + 2 * c + 1) * hp)
  -- certificate: ideal_membership.sage

theorem rotNum_orthogonal (a b c : ℝ) (hp : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
    (rotNum a b c)ᵀ * rotNum a b c = (1 + c) ^ 2 • (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [rotNum, Matrix.mul_apply, Fin.sum_univ_succ]
  · linear_combination rot_e00 a b c hp
  · linear_combination rot_e01 a b c hp
  · linear_combination rot_e02 a b c hp
  · linear_combination rot_e01 a b c hp
  · linear_combination rot_e11 a b c hp
  · linear_combination rot_e12 a b c hp
  · linear_combination rot_e02 a b c hp
  · linear_combination rot_e12 a b c hp
  · linear_combination rot_e22 a b c hp

theorem rotNum_det (a b c : ℝ) (hp : a ^ 2 + b ^ 2 + c ^ 2 = 1) : (rotNum a b c).det = (1 + c) ^ 3 := by
  simp [rotNum, Matrix.det_fin_three]
  linear_combination rot_det a b c hp

/-- `rotTo p` is a rotation for every point `p` of the sphere. -/
theorem rotTo_mem (p : Fin 3 → ℝ) (hp : p ∈ sphere) : rotTo p ∈ SO3 := by
  rw [mem_SO3_iff]
  simp only [sphere, Set.mem_setOf_eq] at hp
  unfold rotTo
  split_ifs with h
  · constructor
    · ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]
    · simp [Matrix.det_fin_three]
  · have hs : 1 + p 2 ≠ 0 := fun h' => h (by linarith)
    constructor
    · rw [Matrix.transpose_smul, Matrix.smul_mul, Matrix.mul_smul, rotNum_orthogonal _ _ _ hp, smul_smul,
        smul_smul]
      field_simp
      simp
    · rw [Matrix.det_smul, rotNum_det _ _ _ hp, Fintype.card_fin]
      field_simp

/-- `rotTo p` sends the north pole to `p`. -/
theorem rotTo_orbit (p : Fin 3 → ℝ) (hp : p ∈ sphere) : orbit (rotTo p) = p := by
  simp only [sphere, Set.mem_setOf_eq] at hp
  unfold orbit rotTo northPole
  split_ifs with h
  · have h0 : p 0 = 0 := by nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]
    have h1 : p 1 = 0 := by nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]
    ext i; fin_cases i <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ, h0, h1, h]
  · have hs : 1 + p 2 ≠ 0 := fun h' => h (by linarith)
    ext i; fin_cases i <;> simp [rotNum, Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;> field_simp

/-- The orbit map sends `SO(3)` onto the sphere: a point is on the sphere exactly when some rotation moves the north
pole to it. -/
theorem orbit_image (p : Fin 3 → ℝ) : p ∈ sphere ↔ ∃ g ∈ SO3, orbit g = p := by
  constructor
  · intro hp; exact ⟨rotTo p, rotTo_mem p hp, rotTo_orbit p hp⟩
  · rintro ⟨g, hg, rfl⟩
    rw [mem_SO3_iff] at hg
    have h22 := congrFun (congrFun hg.1 2) 2
    simp [Matrix.mul_apply, Fin.sum_univ_succ] at h22
    simp only [sphere, Set.mem_setOf_eq, orbit, northPole]
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
    nlinarith [h22]

example : veriphy_instance% KaluzaKlein.Sets.Maps.orbit_image (KaluzaKlein.Sets.Sets.northPole) := by
  have key : ∀ (p : Fin 3 → ℝ), p ∈ sphere ↔ ∃ g ∈ SO3, orbit g = p := by
    intro p
    constructor
    · intro hp; exact ⟨rotTo p, rotTo_mem p hp, rotTo_orbit p hp⟩
    · rintro ⟨g, hg, rfl⟩
      rw [mem_SO3_iff] at hg
      have h22 := congrFun (congrFun hg.1 2) 2
      simp [Matrix.mul_apply, Fin.sum_univ_succ] at h22
      simp only [sphere, Set.mem_setOf_eq, orbit, northPole]
      simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
      nlinarith [h22]
  exact @key (KaluzaKlein.Sets.Sets.northPole)

example : veriphy_instance% KaluzaKlein.Sets.Maps.orbit_image (![1, 0, 0]) := by
  have key : ∀ (p : Fin 3 → ℝ), p ∈ sphere ↔ ∃ g ∈ SO3, orbit g = p := by
    intro p
    constructor
    · intro hp; exact ⟨rotTo p, rotTo_mem p hp, rotTo_orbit p hp⟩
    · rintro ⟨g, hg, rfl⟩
      rw [mem_SO3_iff] at hg
      have h22 := congrFun (congrFun hg.1 2) 2
      simp [Matrix.mul_apply, Fin.sum_univ_succ] at h22
      simp only [sphere, Set.mem_setOf_eq, orbit, northPole]
      simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
      nlinarith [h22]
  exact @key (![1, 0, 0])

example : veriphy_instance% KaluzaKlein.Sets.Maps.orbit_image (![0, 0, -1]) := by
  have key : ∀ (p : Fin 3 → ℝ), p ∈ sphere ↔ ∃ g ∈ SO3, orbit g = p := by
    intro p
    constructor
    · intro hp; exact ⟨rotTo p, rotTo_mem p hp, rotTo_orbit p hp⟩
    · rintro ⟨g, hg, rfl⟩
      rw [mem_SO3_iff] at hg
      have h22 := congrFun (congrFun hg.1 2) 2
      simp [Matrix.mul_apply, Fin.sum_univ_succ] at h22
      simp only [sphere, Set.mem_setOf_eq, orbit, northPole]
      simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
      nlinarith [h22]
  exact @key (![0, 0, -1])

example : veriphy_instance% KaluzaKlein.Sets.Maps.orbit_image (![3/5, 0, 4/5]) := by
  have key : ∀ (p : Fin 3 → ℝ), p ∈ sphere ↔ ∃ g ∈ SO3, orbit g = p := by
    intro p
    constructor
    · intro hp; exact ⟨rotTo p, rotTo_mem p hp, rotTo_orbit p hp⟩
    · rintro ⟨g, hg, rfl⟩
      rw [mem_SO3_iff] at hg
      have h22 := congrFun (congrFun hg.1 2) 2
      simp [Matrix.mul_apply, Fin.sum_univ_succ] at h22
      simp only [sphere, Set.mem_setOf_eq, orbit, northPole]
      simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
      nlinarith [h22]
  exact @key (![3/5, 0, 4/5])

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.Sets.Maps.orbit_image
-- test: set-orbit-image-test.sage

/-! ## exr-map-image-preimage -/

/-- For `f = (a, c, a, a)` on `{1, 2, 3, 4}`: image `{a, c}`, `f⁻¹{a} = {1, 3, 4}`, `f⁻¹{b} = ∅`,
`f⁻¹{a, c}` everything (the elements `1, …, 4` as `0, …, 3`). -/
theorem finite_map_four {β : Type} (a b c : β) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    Set.range ![a, c, a, a] = {a, c} ∧ ![a, c, a, a] ⁻¹' {a} = {0, 2, 3} ∧ ![a, c, a, a] ⁻¹' {b} = ∅ ∧
      ![a, c, a, a] ⁻¹' {a, c} = Set.univ := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · ext z
    simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]
    constructor
    · rintro ⟨i, rfl⟩; fin_cases i <;> simp
    · rintro (rfl | rfl)
      · exact ⟨0, rfl⟩
      · exact ⟨1, rfl⟩
  · ext i; fin_cases i <;> simp [hac.symm]
  · ext i; fin_cases i <;> simp [hab, hbc.symm]
  · ext i; fin_cases i <;> simp

example : veriphy_instance% KaluzaKlein.Sets.Maps.finite_map_four (Fin 3) (0) (1) (2) ((by decide)) ((by decide)) ((by decide)) := by
  have key : ∀ {β : Type} (a b c : β) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c), Set.range ![a, c, a, a] = {a, c} ∧ ![a, c, a, a] ⁻¹' {a} = {0, 2, 3} ∧ ![a, c, a, a] ⁻¹' {b} = ∅ ∧ ![a, c, a, a] ⁻¹' {a, c} = Set.univ := by
    intro β a b c hab hac hbc
    refine ⟨?_, ?_, ?_, ?_⟩
    · ext z
      simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]
      constructor
      · rintro ⟨i, rfl⟩; fin_cases i <;> simp
      · rintro (rfl | rfl)
        · exact ⟨0, rfl⟩
        · exact ⟨1, rfl⟩
    · ext i; fin_cases i <;> simp [hac.symm]
    · ext i; fin_cases i <;> simp [hab, hbc.symm]
    · ext i; fin_cases i <;> simp
  exact @key (Fin 3) (0) (1) (2) ((by decide)) ((by decide)) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Maps.finite_map_four (ℕ) (1) (2) (3) ((by decide)) ((by decide)) ((by decide)) := by
  have key : ∀ {β : Type} (a b c : β) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c), Set.range ![a, c, a, a] = {a, c} ∧ ![a, c, a, a] ⁻¹' {a} = {0, 2, 3} ∧ ![a, c, a, a] ⁻¹' {b} = ∅ ∧ ![a, c, a, a] ⁻¹' {a, c} = Set.univ := by
    intro β a b c hab hac hbc
    refine ⟨?_, ?_, ?_, ?_⟩
    · ext z
      simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]
      constructor
      · rintro ⟨i, rfl⟩; fin_cases i <;> simp
      · rintro (rfl | rfl)
        · exact ⟨0, rfl⟩
        · exact ⟨1, rfl⟩
    · ext i; fin_cases i <;> simp [hac.symm]
    · ext i; fin_cases i <;> simp [hab, hbc.symm]
    · ext i; fin_cases i <;> simp
  exact @key (ℕ) (1) (2) (3) ((by decide)) ((by decide)) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Maps.finite_map_four (Char) ('a') ('b') ('c') ((by decide)) ((by decide)) ((by decide)) := by
  have key : ∀ {β : Type} (a b c : β) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c), Set.range ![a, c, a, a] = {a, c} ∧ ![a, c, a, a] ⁻¹' {a} = {0, 2, 3} ∧ ![a, c, a, a] ⁻¹' {b} = ∅ ∧ ![a, c, a, a] ⁻¹' {a, c} = Set.univ := by
    intro β a b c hab hac hbc
    refine ⟨?_, ?_, ?_, ?_⟩
    · ext z
      simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]
      constructor
      · rintro ⟨i, rfl⟩; fin_cases i <;> simp
      · rintro (rfl | rfl)
        · exact ⟨0, rfl⟩
        · exact ⟨1, rfl⟩
    · ext i; fin_cases i <;> simp [hac.symm]
    · ext i; fin_cases i <;> simp [hab, hbc.symm]
    · ext i; fin_cases i <;> simp
  exact @key (Char) ('a') ('b') ('c') ((by decide)) ((by decide)) ((by decide))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Maps.finite_map_four
-- test: set-finite-map-four-test.sage

/-! ## exr-map-square -/

/-- `x ↦ x² + c` has image `[c, ∞)`: `x²` has image `[0, ∞)`. -/
theorem range_sq_add (c : ℝ) : Set.range (fun x : ℝ => x ^ 2 + c) = Set.Ici c := by
  ext y
  simp only [Set.mem_range, Set.mem_Ici]
  constructor
  · rintro ⟨x, rfl⟩; nlinarith [sq_nonneg x]
  · intro h
    exact ⟨Real.sqrt (y - c), by rw [Real.sq_sqrt (by linarith)]; ring⟩

example : veriphy_instance% KaluzaKlein.Sets.Maps.range_sq_add (0) := by
  have key : ∀ (c : ℝ), Set.range (fun x : ℝ => x ^ 2 + c) = Set.Ici c := by
    intro c
    ext y
    simp only [Set.mem_range, Set.mem_Ici]
    constructor
    · rintro ⟨x, rfl⟩; nlinarith [sq_nonneg x]
    · intro h
      exact ⟨Real.sqrt (y - c), by rw [Real.sq_sqrt (by linarith)]; ring⟩
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Maps.range_sq_add (1) := by
  have key : ∀ (c : ℝ), Set.range (fun x : ℝ => x ^ 2 + c) = Set.Ici c := by
    intro c
    ext y
    simp only [Set.mem_range, Set.mem_Ici]
    constructor
    · rintro ⟨x, rfl⟩; nlinarith [sq_nonneg x]
    · intro h
      exact ⟨Real.sqrt (y - c), by rw [Real.sq_sqrt (by linarith)]; ring⟩
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Maps.range_sq_add (-3) := by
  have key : ∀ (c : ℝ), Set.range (fun x : ℝ => x ^ 2 + c) = Set.Ici c := by
    intro c
    ext y
    simp only [Set.mem_range, Set.mem_Ici]
    constructor
    · rintro ⟨x, rfl⟩; nlinarith [sq_nonneg x]
    · intro h
      exact ⟨Real.sqrt (y - c), by rw [Real.sq_sqrt (by linarith)]; ring⟩
  exact @key (-3)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Maps.range_sq_add
-- test: set-range-sq-test.sage

/-- For `0 ≤ a ≤ b`, the preimage of `[a², b²]` under `x ↦ x²` is `[-b, -a] ∪ [a, b]`: `[-2, -1] ∪ [1, 2]` for
`[1, 4]`. -/
theorem sq_preimage_Icc (a b : ℝ) (ha : 0 ≤ a) (hab : a ≤ b) :
    (fun x : ℝ => x ^ 2) ⁻¹' Set.Icc (a ^ 2) (b ^ 2) = Set.Icc (-b) (-a) ∪ Set.Icc a b := by
  ext x
  simp only [Set.mem_preimage, Set.mem_Icc, Set.mem_union]
  constructor
  · rintro ⟨h1, h2⟩
    have hb : |x| ≤ b := abs_le_of_sq_le_sq' h2 (by linarith) |>.elim (fun h1' h2' => abs_le.2 ⟨h1', h2'⟩)
    have ha' : a ≤ |x| := by
      rw [← Real.sqrt_sq ha, ← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt h1
    rcases le_or_gt 0 x with hx | hx
    · rw [abs_of_nonneg hx] at hb ha'; right; exact ⟨ha', hb⟩
    · rw [abs_of_neg hx] at hb ha'; left; constructor <;> linarith
  · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩)
    · constructor <;> nlinarith
    · constructor <;> nlinarith

example : veriphy_instance% KaluzaKlein.Sets.Maps.sq_preimage_Icc (1) (2) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (ha : 0 ≤ a) (hab : a ≤ b), (fun x : ℝ => x ^ 2) ⁻¹' Set.Icc (a ^ 2) (b ^ 2) = Set.Icc (-b) (-a) ∪ Set.Icc a b := by
    intro a b ha hab
    ext x
    simp only [Set.mem_preimage, Set.mem_Icc, Set.mem_union]
    constructor
    · rintro ⟨h1, h2⟩
      have hb : |x| ≤ b := abs_le_of_sq_le_sq' h2 (by linarith) |>.elim (fun h1' h2' => abs_le.2 ⟨h1', h2'⟩)
      have ha' : a ≤ |x| := by
        rw [← Real.sqrt_sq ha, ← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt h1
      rcases le_or_gt 0 x with hx | hx
      · rw [abs_of_nonneg hx] at hb ha'; right; exact ⟨ha', hb⟩
      · rw [abs_of_neg hx] at hb ha'; left; constructor <;> linarith
    · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩)
      · constructor <;> nlinarith
      · constructor <;> nlinarith
  exact @key (1) (2) ((by norm_num)) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Maps.sq_preimage_Icc (0) (1) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (ha : 0 ≤ a) (hab : a ≤ b), (fun x : ℝ => x ^ 2) ⁻¹' Set.Icc (a ^ 2) (b ^ 2) = Set.Icc (-b) (-a) ∪ Set.Icc a b := by
    intro a b ha hab
    ext x
    simp only [Set.mem_preimage, Set.mem_Icc, Set.mem_union]
    constructor
    · rintro ⟨h1, h2⟩
      have hb : |x| ≤ b := abs_le_of_sq_le_sq' h2 (by linarith) |>.elim (fun h1' h2' => abs_le.2 ⟨h1', h2'⟩)
      have ha' : a ≤ |x| := by
        rw [← Real.sqrt_sq ha, ← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt h1
      rcases le_or_gt 0 x with hx | hx
      · rw [abs_of_nonneg hx] at hb ha'; right; exact ⟨ha', hb⟩
      · rw [abs_of_neg hx] at hb ha'; left; constructor <;> linarith
    · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩)
      · constructor <;> nlinarith
      · constructor <;> nlinarith
  exact @key (0) (1) ((by norm_num)) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Maps.sq_preimage_Icc (3) (3) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (ha : 0 ≤ a) (hab : a ≤ b), (fun x : ℝ => x ^ 2) ⁻¹' Set.Icc (a ^ 2) (b ^ 2) = Set.Icc (-b) (-a) ∪ Set.Icc a b := by
    intro a b ha hab
    ext x
    simp only [Set.mem_preimage, Set.mem_Icc, Set.mem_union]
    constructor
    · rintro ⟨h1, h2⟩
      have hb : |x| ≤ b := abs_le_of_sq_le_sq' h2 (by linarith) |>.elim (fun h1' h2' => abs_le.2 ⟨h1', h2'⟩)
      have ha' : a ≤ |x| := by
        rw [← Real.sqrt_sq ha, ← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt h1
      rcases le_or_gt 0 x with hx | hx
      · rw [abs_of_nonneg hx] at hb ha'; right; exact ⟨ha', hb⟩
      · rw [abs_of_neg hx] at hb ha'; left; constructor <;> linarith
    · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩)
      · constructor <;> nlinarith
      · constructor <;> nlinarith
  exact @key (3) (3) ((by norm_num)) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Maps.sq_preimage_Icc
-- test: set-sq-preimage-test.sage

/-- No square is negative: the preimage of `{y}` under `x ↦ x²` is empty for `y < 0` (`y = -1`). -/
theorem sq_preimage_neg (y : ℝ) (hy : y < 0) : (fun x : ℝ => x ^ 2) ⁻¹' {y} = ∅ := by
  ext x
  simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_empty_iff_false, iff_false]
  intro h; nlinarith [sq_nonneg x]

example : veriphy_instance% KaluzaKlein.Sets.Maps.sq_preimage_neg (-1) ((by norm_num)) := by
  have key : ∀ (y : ℝ) (hy : y < 0), (fun x : ℝ => x ^ 2) ⁻¹' {y} = ∅ := by
    intro y hy
    ext x
    simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_empty_iff_false, iff_false]
    intro h; nlinarith [sq_nonneg x]
  exact @key (-1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Maps.sq_preimage_neg (-1/4) ((by norm_num)) := by
  have key : ∀ (y : ℝ) (hy : y < 0), (fun x : ℝ => x ^ 2) ⁻¹' {y} = ∅ := by
    intro y hy
    ext x
    simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_empty_iff_false, iff_false]
    intro h; nlinarith [sq_nonneg x]
  exact @key (-1/4) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Maps.sq_preimage_neg (-9) ((by norm_num)) := by
  have key : ∀ (y : ℝ) (hy : y < 0), (fun x : ℝ => x ^ 2) ⁻¹' {y} = ∅ := by
    intro y hy
    ext x
    simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_empty_iff_false, iff_false]
    intro h; nlinarith [sq_nonneg x]
  exact @key (-9) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Maps.sq_preimage_neg
-- test: set-sq-preimage-neg-test.sage

/-! ## exr-map-height-cap -/

/-- On the sphere, `h⁻¹((c, 1]) = {Z > c}` (since `Z ≤ 1`), and it contains the north pole for `c < 1`: the open cap
above `Z = ½`. -/
theorem height_cap (c : ℝ) (hc : c < 1) :
    sphere ∩ height ⁻¹' Set.Ioc c 1 = sphere ∩ {p | c < p 2} ∧ northPole ∈ sphere ∩ height ⁻¹' Set.Ioc c 1 := by
  constructor
  · ext p
    simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Ioc]
    constructor
    · rintro ⟨h, h1, -⟩; exact ⟨h, h1⟩
    · rintro ⟨h, h1⟩; exact ⟨h, h1, by nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]⟩
  · simp [sphere, height, northPole, hc]

example : veriphy_instance% KaluzaKlein.Sets.Maps.height_cap (1/2) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : c < 1), sphere ∩ height ⁻¹' Set.Ioc c 1 = sphere ∩ {p | c < p 2} ∧ northPole ∈ sphere ∩ height ⁻¹' Set.Ioc c 1 := by
    intro c hc
    constructor
    · ext p
      simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Ioc]
      constructor
      · rintro ⟨h, h1, -⟩; exact ⟨h, h1⟩
      · rintro ⟨h, h1⟩; exact ⟨h, h1, by nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]⟩
    · simp [sphere, height, northPole, hc]
  exact @key (1/2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Maps.height_cap (0) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : c < 1), sphere ∩ height ⁻¹' Set.Ioc c 1 = sphere ∩ {p | c < p 2} ∧ northPole ∈ sphere ∩ height ⁻¹' Set.Ioc c 1 := by
    intro c hc
    constructor
    · ext p
      simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Ioc]
      constructor
      · rintro ⟨h, h1, -⟩; exact ⟨h, h1⟩
      · rintro ⟨h, h1⟩; exact ⟨h, h1, by nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]⟩
    · simp [sphere, height, northPole, hc]
  exact @key (0) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Maps.height_cap (-1/2) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : c < 1), sphere ∩ height ⁻¹' Set.Ioc c 1 = sphere ∩ {p | c < p 2} ∧ northPole ∈ sphere ∩ height ⁻¹' Set.Ioc c 1 := by
    intro c hc
    constructor
    · ext p
      simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Ioc]
      constructor
      · rintro ⟨h, h1, -⟩; exact ⟨h, h1⟩
      · rintro ⟨h, h1⟩; exact ⟨h, h1, by nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]⟩
    · simp [sphere, height, northPole, hc]
  exact @key (-1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Maps.height_cap
-- test: set-height-cap-test.sage

/-- `h⁻¹(0)` is the equator, the circle `X² + Y² = 1` in the plane `Z = 0`. -/
theorem height_equator : sphere ∩ height ⁻¹' {0} = {p | p 0 ^ 2 + p 1 ^ 2 = 1 ∧ p 2 = 0} := by
  ext p
  simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_singleton_iff]
  constructor
  · rintro ⟨h, h2⟩; rw [h2] at h; exact ⟨by linarith, h2⟩
  · rintro ⟨h, h2⟩; exact ⟨by rw [h2]; linarith, h2⟩

/-- `|Z| ≤ 1` on the sphere, so `h⁻¹([a, b])` is empty for `1 < a`: `h⁻¹([2, 3]) = ∅`. -/
theorem height_preimage_empty (a b : ℝ) (ha : 1 < a) : sphere ∩ height ⁻¹' Set.Icc a b = ∅ := by
  ext p
  simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Icc,
    Set.mem_empty_iff_false, iff_false, not_and]
  intro h h1 _
  nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]

example : veriphy_instance% KaluzaKlein.Sets.Maps.height_preimage_empty (2) (3) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (ha : 1 < a), sphere ∩ height ⁻¹' Set.Icc a b = ∅ := by
    intro a b ha
    ext p
    simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Icc,
      Set.mem_empty_iff_false, iff_false, not_and]
    intro h h1 _
    nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]
  exact @key (2) (3) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Maps.height_preimage_empty (3/2) (1) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (ha : 1 < a), sphere ∩ height ⁻¹' Set.Icc a b = ∅ := by
    intro a b ha
    ext p
    simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Icc,
      Set.mem_empty_iff_false, iff_false, not_and]
    intro h h1 _
    nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]
  exact @key (3/2) (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Maps.height_preimage_empty (5) (10) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (ha : 1 < a), sphere ∩ height ⁻¹' Set.Icc a b = ∅ := by
    intro a b ha
    ext p
    simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Icc,
      Set.mem_empty_iff_false, iff_false, not_and]
    intro h h1 _
    nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]
  exact @key (5) (10) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Maps.height_preimage_empty
-- test: set-height-empty-test.sage

end KaluzaKlein.Sets.Maps
