import Mathlib
import KaluzaKlein.PartI.Sets.TopologicalSpaces
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Topological spaces" (subspaces)

The claims of the blocks `def-subspace-topology` to `exr-subspace-transitive` of
`tutorial/part-i-01-sets-relations-topology.qmd`: open arcs of the circle (traces of open half-planes), caps of the
sphere and open sets of `SO(3)`, a half-open interval as a subspace, the halves of the circle, discrete subspaces of
the line, and subspaces of subspaces. A subset `S` carries the subspace topology as the type `↥S`; a set `A` is open
in `S` when `(↑) ⁻¹' A` is open.
-/

namespace KaluzaKlein.Sets.Subspace

open Set Metric Matrix KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Maps

/-- `wrap` is `2π`-periodic under integer shifts. -/
theorem wrap_add_int (ψ : ℝ) (k : ℤ) : wrap (ψ + k * (2 * Real.pi)) = wrap ψ := by
  simp [wrap, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]

/-- For `0 < δ ≤ π`: `cos x > cos δ` exactly when `x` is within `δ` of a multiple of `2π`. -/
theorem cos_gt_iff (δ x : ℝ) (hδ : 0 < δ) (hδπ : δ ≤ Real.pi) :
    Real.cos δ < Real.cos x ↔ ∃ k : ℤ, |x - k * (2 * Real.pi)| < δ := by
  have hpi := Real.pi_pos
  have hT : (0 : ℝ) < 2 * Real.pi := by positivity
  constructor
  · intro h
    set y := toIocMod hT (-Real.pi) x
    have hy := toIocMod_mem_Ioc hT (-Real.pi) x
    have hy' : y = x + (-(toIocDiv hT (-Real.pi) x) : ℤ) * (2 * Real.pi) := by
      simp only [y, toIocMod, zsmul_eq_mul]; push_cast; ring
    refine ⟨toIocDiv hT (-Real.pi) x, ?_⟩
    have hcos : Real.cos y = Real.cos x := by rw [hy', Real.cos_add_int_mul_two_pi]
    have hxy : x - (toIocDiv hT (-Real.pi) x : ℝ) * (2 * Real.pi) = y := by rw [hy']; push_cast; ring
    rw [hxy]
    by_contra hge
    push_neg at hge
    have habs : |y| ≤ Real.pi := abs_le.2 ⟨by linarith [hy.1], by linarith [hy.2]⟩
    have : Real.cos |y| ≤ Real.cos δ :=
      Real.cos_le_cos_of_nonneg_of_le_pi hδ.le habs hge
    rw [Real.cos_abs, hcos] at this
    linarith
  · rintro ⟨k, hk⟩
    have : Real.cos x = Real.cos |x - k * (2 * Real.pi)| := by
      rw [Real.cos_abs, show x - k * (2 * Real.pi) = x + ((-k : ℤ) : ℝ) * (2 * Real.pi) by push_cast; ring,
        Real.cos_add_int_mul_two_pi]
    rw [this]
    exact Real.cos_lt_cos_of_nonneg_of_le_pi (abs_nonneg _) hδπ hk

/-! ## exm-subspace-circle -/

/-- An open arc of the circle, the angles within `δ ≤ π` of `c`, is the trace on the circle of the open half-plane
`{p · (cos c, sin c) > cos δ}`, hence open in the circle. -/
theorem arc_open (c δ : ℝ) (hδ : 0 < δ) (hδπ : δ ≤ Real.pi) :
    wrap '' Ioo (c - δ) (c + δ) = circle ∩ {p | Real.cos δ < p.1 * Real.cos c + p.2 * Real.sin c} ∧
      IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' (wrap '' Ioo (c - δ) (c + δ))) := by
  have hpi := Real.pi_pos
  have key : wrap '' Ioo (c - δ) (c + δ) = circle ∩ {p | Real.cos δ < p.1 * Real.cos c + p.2 * Real.sin c} := by
    ext p
    simp only [mem_image, mem_Ioo, mem_inter_iff, mem_setOf_eq]
    constructor
    · rintro ⟨φ, ⟨h1, h2⟩, rfl⟩
      refine ⟨by simp [circle, wrap, Real.cos_sq_add_sin_sq], ?_⟩
      simp only [wrap]
      rw [← Real.cos_sub]
      exact (cos_gt_iff δ (φ - c) hδ hδπ).2 ⟨0, by rw [abs_lt]; constructor <;> push_cast <;> linarith⟩
    · rintro ⟨hc, hp⟩
      have hp' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
      obtain ⟨ψ, rfl⟩ := hp'
      simp only [one_mul] at hp
      rw [← Real.cos_sub] at hp
      obtain ⟨k, hk⟩ := (cos_gt_iff δ (ψ - c) hδ hδπ).1 hp
      rw [abs_lt] at hk
      refine ⟨ψ - k * (2 * Real.pi), ⟨by linarith [hk.1], by linarith [hk.2]⟩, ?_⟩
      rw [show ψ - k * (2 * Real.pi) = ψ + ((-k : ℤ) : ℝ) * (2 * Real.pi) by push_cast; ring, wrap_add_int]
      simp [wrap]
  refine ⟨key, ?_⟩
  rw [key, preimage_inter]
  have h1 : (Subtype.val : circle → ℝ × ℝ) ⁻¹' circle = univ := by ext x; simp [x.2]
  rw [h1, univ_inter]
  exact (isOpen_lt continuous_const (by fun_prop)).preimage continuous_subtype_val

example : veriphy_instance% KaluzaKlein.Sets.Subspace.arc_open (0) (Real.pi / 2) ((by positivity)) ((by linarith [Real.pi_pos])) := by
  have key : ∀ (c δ : ℝ) (hδ : 0 < δ) (hδπ : δ ≤ Real.pi), wrap '' Ioo (c - δ) (c + δ) = circle ∩ {p | Real.cos δ < p.1 * Real.cos c + p.2 * Real.sin c} ∧ IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' (wrap '' Ioo (c - δ) (c + δ))) := by
    intro c δ hδ hδπ
    have hpi := Real.pi_pos
    have key : wrap '' Ioo (c - δ) (c + δ) = circle ∩ {p | Real.cos δ < p.1 * Real.cos c + p.2 * Real.sin c} := by
      ext p
      simp only [mem_image, mem_Ioo, mem_inter_iff, mem_setOf_eq]
      constructor
      · rintro ⟨φ, ⟨h1, h2⟩, rfl⟩
        refine ⟨by simp [circle, wrap, Real.cos_sq_add_sin_sq], ?_⟩
        simp only [wrap]
        rw [← Real.cos_sub]
        exact (cos_gt_iff δ (φ - c) hδ hδπ).2 ⟨0, by rw [abs_lt]; constructor <;> push_cast <;> linarith⟩
      · rintro ⟨hc, hp⟩
        have hp' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
          rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
        obtain ⟨ψ, rfl⟩ := hp'
        simp only [one_mul] at hp
        rw [← Real.cos_sub] at hp
        obtain ⟨k, hk⟩ := (cos_gt_iff δ (ψ - c) hδ hδπ).1 hp
        rw [abs_lt] at hk
        refine ⟨ψ - k * (2 * Real.pi), ⟨by linarith [hk.1], by linarith [hk.2]⟩, ?_⟩
        rw [show ψ - k * (2 * Real.pi) = ψ + ((-k : ℤ) : ℝ) * (2 * Real.pi) by push_cast; ring, wrap_add_int]
        simp [wrap]
    refine ⟨key, ?_⟩
    rw [key, preimage_inter]
    have h1 : (Subtype.val : circle → ℝ × ℝ) ⁻¹' circle = univ := by ext x; simp [x.2]
    rw [h1, univ_inter]
    exact (isOpen_lt continuous_const (by fun_prop)).preimage continuous_subtype_val
  exact @key (0) (Real.pi / 2) ((by positivity)) ((by linarith [Real.pi_pos]))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.arc_open (Real.pi / 2) (Real.pi / 2) ((by positivity)) ((by linarith [Real.pi_pos])) := by
  have key : ∀ (c δ : ℝ) (hδ : 0 < δ) (hδπ : δ ≤ Real.pi), wrap '' Ioo (c - δ) (c + δ) = circle ∩ {p | Real.cos δ < p.1 * Real.cos c + p.2 * Real.sin c} ∧ IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' (wrap '' Ioo (c - δ) (c + δ))) := by
    intro c δ hδ hδπ
    have hpi := Real.pi_pos
    have key : wrap '' Ioo (c - δ) (c + δ) = circle ∩ {p | Real.cos δ < p.1 * Real.cos c + p.2 * Real.sin c} := by
      ext p
      simp only [mem_image, mem_Ioo, mem_inter_iff, mem_setOf_eq]
      constructor
      · rintro ⟨φ, ⟨h1, h2⟩, rfl⟩
        refine ⟨by simp [circle, wrap, Real.cos_sq_add_sin_sq], ?_⟩
        simp only [wrap]
        rw [← Real.cos_sub]
        exact (cos_gt_iff δ (φ - c) hδ hδπ).2 ⟨0, by rw [abs_lt]; constructor <;> push_cast <;> linarith⟩
      · rintro ⟨hc, hp⟩
        have hp' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
          rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
        obtain ⟨ψ, rfl⟩ := hp'
        simp only [one_mul] at hp
        rw [← Real.cos_sub] at hp
        obtain ⟨k, hk⟩ := (cos_gt_iff δ (ψ - c) hδ hδπ).1 hp
        rw [abs_lt] at hk
        refine ⟨ψ - k * (2 * Real.pi), ⟨by linarith [hk.1], by linarith [hk.2]⟩, ?_⟩
        rw [show ψ - k * (2 * Real.pi) = ψ + ((-k : ℤ) : ℝ) * (2 * Real.pi) by push_cast; ring, wrap_add_int]
        simp [wrap]
    refine ⟨key, ?_⟩
    rw [key, preimage_inter]
    have h1 : (Subtype.val : circle → ℝ × ℝ) ⁻¹' circle = univ := by ext x; simp [x.2]
    rw [h1, univ_inter]
    exact (isOpen_lt continuous_const (by fun_prop)).preimage continuous_subtype_val
  exact @key (Real.pi / 2) (Real.pi / 2) ((by positivity)) ((by linarith [Real.pi_pos]))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.arc_open (1) (Real.pi) ((by positivity)) (le_rfl) := by
  have key : ∀ (c δ : ℝ) (hδ : 0 < δ) (hδπ : δ ≤ Real.pi), wrap '' Ioo (c - δ) (c + δ) = circle ∩ {p | Real.cos δ < p.1 * Real.cos c + p.2 * Real.sin c} ∧ IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' (wrap '' Ioo (c - δ) (c + δ))) := by
    intro c δ hδ hδπ
    have hpi := Real.pi_pos
    have key : wrap '' Ioo (c - δ) (c + δ) = circle ∩ {p | Real.cos δ < p.1 * Real.cos c + p.2 * Real.sin c} := by
      ext p
      simp only [mem_image, mem_Ioo, mem_inter_iff, mem_setOf_eq]
      constructor
      · rintro ⟨φ, ⟨h1, h2⟩, rfl⟩
        refine ⟨by simp [circle, wrap, Real.cos_sq_add_sin_sq], ?_⟩
        simp only [wrap]
        rw [← Real.cos_sub]
        exact (cos_gt_iff δ (φ - c) hδ hδπ).2 ⟨0, by rw [abs_lt]; constructor <;> push_cast <;> linarith⟩
      · rintro ⟨hc, hp⟩
        have hp' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
          rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
        obtain ⟨ψ, rfl⟩ := hp'
        simp only [one_mul] at hp
        rw [← Real.cos_sub] at hp
        obtain ⟨k, hk⟩ := (cos_gt_iff δ (ψ - c) hδ hδπ).1 hp
        rw [abs_lt] at hk
        refine ⟨ψ - k * (2 * Real.pi), ⟨by linarith [hk.1], by linarith [hk.2]⟩, ?_⟩
        rw [show ψ - k * (2 * Real.pi) = ψ + ((-k : ℤ) : ℝ) * (2 * Real.pi) by push_cast; ring, wrap_add_int]
        simp [wrap]
    refine ⟨key, ?_⟩
    rw [key, preimage_inter]
    have h1 : (Subtype.val : circle → ℝ × ℝ) ⁻¹' circle = univ := by ext x; simp [x.2]
    rw [h1, univ_inter]
    exact (isOpen_lt continuous_const (by fun_prop)).preimage continuous_subtype_val
  exact @key (1) (Real.pi) ((by positivity)) (le_rfl)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Subspace.arc_open
-- test: set-arc-open-test.sage

/-! ## exm-subspace-sphere -/

/-- A cap `{p · n > t}` of the sphere is open in the sphere, and the rotations whose `(i, j)` entry lies in an open
range `(a, b)` form an open set of `SO(3)`. -/
theorem caps_and_entries_open (n : Fin 3 → ℝ) (t a b : ℝ) (i j : Fin 3) :
    IsOpen ((Subtype.val : sphere → (Fin 3 → ℝ)) ⁻¹' {p | t < p ⬝ᵥ n}) ∧
      IsOpen ((Subtype.val : SO3 → Matrix (Fin 3) (Fin 3) ℝ) ⁻¹' {g | g i j ∈ Ioo a b}) := by
  constructor
  · refine (isOpen_lt continuous_const ?_).preimage continuous_subtype_val
    simp only [dotProduct]
    fun_prop
  · refine IsOpen.preimage continuous_subtype_val ?_
    exact isOpen_Ioo.preimage (continuous_id.matrix_elem i j)

example : veriphy_instance% KaluzaKlein.Sets.Subspace.caps_and_entries_open (KaluzaKlein.Sets.Sets.northPole) (1/2) (0) (1) (0) (0) := by
  have key : ∀ (n : Fin 3 → ℝ) (t a b : ℝ) (i j : Fin 3), IsOpen ((Subtype.val : sphere → (Fin 3 → ℝ)) ⁻¹' {p | t < p ⬝ᵥ n}) ∧ IsOpen ((Subtype.val : SO3 → Matrix (Fin 3) (Fin 3) ℝ) ⁻¹' {g | g i j ∈ Ioo a b}) := by
    intro n t a b i j
    constructor
    · refine (isOpen_lt continuous_const ?_).preimage continuous_subtype_val
      simp only [dotProduct]
      fun_prop
    · refine IsOpen.preimage continuous_subtype_val ?_
      exact isOpen_Ioo.preimage (continuous_id.matrix_elem i j)
  exact @key (KaluzaKlein.Sets.Sets.northPole) (1/2) (0) (1) (0) (0)

example : veriphy_instance% KaluzaKlein.Sets.Subspace.caps_and_entries_open (0) (0) (0) (0) (0) (0) := by
  have key : ∀ (n : Fin 3 → ℝ) (t a b : ℝ) (i j : Fin 3), IsOpen ((Subtype.val : sphere → (Fin 3 → ℝ)) ⁻¹' {p | t < p ⬝ᵥ n}) ∧ IsOpen ((Subtype.val : SO3 → Matrix (Fin 3) (Fin 3) ℝ) ⁻¹' {g | g i j ∈ Ioo a b}) := by
    intro n t a b i j
    constructor
    · refine (isOpen_lt continuous_const ?_).preimage continuous_subtype_val
      simp only [dotProduct]
      fun_prop
    · refine IsOpen.preimage continuous_subtype_val ?_
      exact isOpen_Ioo.preimage (continuous_id.matrix_elem i j)
  exact @key (0) (0) (0) (0) (0) (0)

example : veriphy_instance% KaluzaKlein.Sets.Subspace.caps_and_entries_open (![1, 0, 0]) (-1) (-1/2) (1/2) (2) (1) := by
  have key : ∀ (n : Fin 3 → ℝ) (t a b : ℝ) (i j : Fin 3), IsOpen ((Subtype.val : sphere → (Fin 3 → ℝ)) ⁻¹' {p | t < p ⬝ᵥ n}) ∧ IsOpen ((Subtype.val : SO3 → Matrix (Fin 3) (Fin 3) ℝ) ⁻¹' {g | g i j ∈ Ioo a b}) := by
    intro n t a b i j
    constructor
    · refine (isOpen_lt continuous_const ?_).preimage continuous_subtype_val
      simp only [dotProduct]
      fun_prop
    · refine IsOpen.preimage continuous_subtype_val ?_
      exact isOpen_Ioo.preimage (continuous_id.matrix_elem i j)
  exact @key (![1, 0, 0]) (-1) (-1/2) (1/2) (2) (1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Subspace.caps_and_entries_open
-- test: set-caps-entries-test.sage

/-! ## exm-subspace-interval -/

/-- For `0 < c ≤ 1`: `[0, c) = (-c, c) ∩ [0, 1)` is open in the subspace `[0, 1)` but not open in `ℝ`
(`c = ½`). -/
theorem half_open_relative (c : ℝ) (hc : 0 < c) (hc1 : c ≤ 1) :
    Ico 0 c = Ioo (-c) c ∩ Ico 0 1 ∧ IsOpen ((Subtype.val : Ico (0 : ℝ) 1 → ℝ) ⁻¹' Ico 0 c) ∧
      ¬ IsOpen (Ico (0 : ℝ) c) := by
  have heq : Ico 0 c = Ioo (-c) c ∩ Ico 0 1 := by
    ext x; simp only [mem_Ico, mem_inter_iff, mem_Ioo]
    constructor
    · rintro ⟨h1, h2⟩; exact ⟨⟨by linarith, h2⟩, h1, by linarith⟩
    · rintro ⟨⟨_, h2⟩, h1, _⟩; exact ⟨h1, h2⟩
  refine ⟨heq, ?_, (OpenSets.intervals_open_closed 0 c hc).2.2.1⟩
  rw [heq, preimage_inter]
  have h1 : (Subtype.val : Ico (0 : ℝ) 1 → ℝ) ⁻¹' Ico 0 1 = univ := by ext x; simp [x.2]
  rw [h1, inter_univ]
  exact isOpen_Ioo.preimage continuous_subtype_val

example : veriphy_instance% KaluzaKlein.Sets.Subspace.half_open_relative (1/2) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c) (hc1 : c ≤ 1), Ico 0 c = Ioo (-c) c ∩ Ico 0 1 ∧ IsOpen ((Subtype.val : Ico (0 : ℝ) 1 → ℝ) ⁻¹' Ico 0 c) ∧ ¬ IsOpen (Ico (0 : ℝ) c) := by
    intro c hc hc1
    have heq : Ico 0 c = Ioo (-c) c ∩ Ico 0 1 := by
      ext x; simp only [mem_Ico, mem_inter_iff, mem_Ioo]
      constructor
      · rintro ⟨h1, h2⟩; exact ⟨⟨by linarith, h2⟩, h1, by linarith⟩
      · rintro ⟨⟨_, h2⟩, h1, _⟩; exact ⟨h1, h2⟩
    refine ⟨heq, ?_, (OpenSets.intervals_open_closed 0 c hc).2.2.1⟩
    rw [heq, preimage_inter]
    have h1 : (Subtype.val : Ico (0 : ℝ) 1 → ℝ) ⁻¹' Ico 0 1 = univ := by ext x; simp [x.2]
    rw [h1, inter_univ]
    exact isOpen_Ioo.preimage continuous_subtype_val
  exact @key (1/2) ((by norm_num)) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.half_open_relative (1) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c) (hc1 : c ≤ 1), Ico 0 c = Ioo (-c) c ∩ Ico 0 1 ∧ IsOpen ((Subtype.val : Ico (0 : ℝ) 1 → ℝ) ⁻¹' Ico 0 c) ∧ ¬ IsOpen (Ico (0 : ℝ) c) := by
    intro c hc hc1
    have heq : Ico 0 c = Ioo (-c) c ∩ Ico 0 1 := by
      ext x; simp only [mem_Ico, mem_inter_iff, mem_Ioo]
      constructor
      · rintro ⟨h1, h2⟩; exact ⟨⟨by linarith, h2⟩, h1, by linarith⟩
      · rintro ⟨⟨_, h2⟩, h1, _⟩; exact ⟨h1, h2⟩
    refine ⟨heq, ?_, (OpenSets.intervals_open_closed 0 c hc).2.2.1⟩
    rw [heq, preimage_inter]
    have h1 : (Subtype.val : Ico (0 : ℝ) 1 → ℝ) ⁻¹' Ico 0 1 = univ := by ext x; simp [x.2]
    rw [h1, inter_univ]
    exact isOpen_Ioo.preimage continuous_subtype_val
  exact @key (1) ((by norm_num)) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.half_open_relative (1/4) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c) (hc1 : c ≤ 1), Ico 0 c = Ioo (-c) c ∩ Ico 0 1 ∧ IsOpen ((Subtype.val : Ico (0 : ℝ) 1 → ℝ) ⁻¹' Ico 0 c) ∧ ¬ IsOpen (Ico (0 : ℝ) c) := by
    intro c hc hc1
    have heq : Ico 0 c = Ioo (-c) c ∩ Ico 0 1 := by
      ext x; simp only [mem_Ico, mem_inter_iff, mem_Ioo]
      constructor
      · rintro ⟨h1, h2⟩; exact ⟨⟨by linarith, h2⟩, h1, by linarith⟩
      · rintro ⟨⟨_, h2⟩, h1, _⟩; exact ⟨h1, h2⟩
    refine ⟨heq, ?_, (OpenSets.intervals_open_closed 0 c hc).2.2.1⟩
    rw [heq, preimage_inter]
    have h1 : (Subtype.val : Ico (0 : ℝ) 1 → ℝ) ⁻¹' Ico 0 1 = univ := by ext x; simp [x.2]
    rw [h1, inter_univ]
    exact isOpen_Ioo.preimage continuous_subtype_val
  exact @key (1/4) ((by norm_num)) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Subspace.half_open_relative
-- test: set-half-open-relative-test.sage

/-! ## exr-subspace-open -/

/-- The part `{Y > t}` of the circle is open in the circle (the trace of an open half-plane): the open upper half for
`t = 0`. -/
theorem upper_part_open (t : ℝ) : IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t < p.2}) :=
  (isOpen_lt continuous_const continuous_snd).preimage continuous_subtype_val

example : veriphy_instance% KaluzaKlein.Sets.Subspace.upper_part_open (0) := by
  have key : ∀ (t : ℝ), IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t < p.2}) := by
    intro t
    exact ((isOpen_lt continuous_const continuous_snd).preimage continuous_subtype_val)
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Subspace.upper_part_open (1/2) := by
  have key : ∀ (t : ℝ), IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t < p.2}) := by
    intro t
    exact ((isOpen_lt continuous_const continuous_snd).preimage continuous_subtype_val)
  exact @key (1/2)

example : veriphy_instance% KaluzaKlein.Sets.Subspace.upper_part_open (-1) := by
  have key : ∀ (t : ℝ), IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t < p.2}) := by
    intro t
    exact ((isOpen_lt continuous_const continuous_snd).preimage continuous_subtype_val)
  exact @key (-1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Subspace.upper_part_open
-- test: set-upper-part-test.sage

/-- For `-1 < t < 1` the part `{Y ≥ t}` of the circle is not open in the circle: every arc around the point
`(√(1 - t²), t)` contains points with `Y < t` (the closed upper half for `t = 0`). -/
theorem closed_part_not_open (t : ℝ) (ht : t ∈ Ioo (-1 : ℝ) 1) :
    ¬ IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t ≤ p.2}) := by
  intro h
  set θ := Real.arcsin t
  have hθ1 : -(Real.pi / 2) < θ := by
    rw [Real.neg_pi_div_two_lt_arcsin]; exact ht.1
  have hθ2 : θ < Real.pi / 2 := by
    rw [Real.arcsin_lt_pi_div_two]; exact ht.2
  have hsin : Real.sin θ = t := Real.sin_arcsin ht.1.le ht.2.le
  let γ : ℝ → circle := fun s => ⟨wrap (θ - s), by simp [circle, wrap, Real.cos_sq_add_sin_sq]⟩
  have hγ : Continuous γ := by
    refine Continuous.subtype_mk ?_ _
    simp only [wrap]; fun_prop
  have hopen := h.preimage hγ
  have h0 : (0 : ℝ) ∈ γ ⁻¹' ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t ≤ p.2}) := by
    simp [γ, wrap, hsin]
  obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 hopen 0 h0
  set s := min (ε / 2) ((θ + Real.pi / 2) / 2)
  have hs0 : 0 < s := lt_min (by linarith) (by linarith)
  have hsε : s < ε := lt_of_le_of_lt (min_le_left _ _) (by linarith)
  have hs1 : s < θ + Real.pi / 2 := lt_of_le_of_lt (min_le_right _ _) (by linarith)
  have hmem := hball (show s ∈ ball (0 : ℝ) ε by
    rw [mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos hs0]; exact hsε)
  simp only [γ, wrap, mem_preimage, mem_setOf_eq] at hmem
  have : Real.sin (θ - s) < Real.sin θ :=
    Real.sin_lt_sin_of_lt_of_le_pi_div_two (by linarith) hθ2.le (by linarith)
  linarith

example : veriphy_instance% KaluzaKlein.Sets.Subspace.closed_part_not_open (0) ((by constructor <;> norm_num)) := by
  have key : ∀ (t : ℝ) (ht : t ∈ Ioo (-1 : ℝ) 1), ¬ IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t ≤ p.2}) := by
    intro t ht
    intro h
    set θ := Real.arcsin t
    have hθ1 : -(Real.pi / 2) < θ := by
      rw [Real.neg_pi_div_two_lt_arcsin]; exact ht.1
    have hθ2 : θ < Real.pi / 2 := by
      rw [Real.arcsin_lt_pi_div_two]; exact ht.2
    have hsin : Real.sin θ = t := Real.sin_arcsin ht.1.le ht.2.le
    let γ : ℝ → circle := fun s => ⟨wrap (θ - s), by simp [circle, wrap, Real.cos_sq_add_sin_sq]⟩
    have hγ : Continuous γ := by
      refine Continuous.subtype_mk ?_ _
      simp only [wrap]; fun_prop
    have hopen := h.preimage hγ
    have h0 : (0 : ℝ) ∈ γ ⁻¹' ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t ≤ p.2}) := by
      simp [γ, wrap, hsin]
    obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 hopen 0 h0
    set s := min (ε / 2) ((θ + Real.pi / 2) / 2)
    have hs0 : 0 < s := lt_min (by linarith) (by linarith)
    have hsε : s < ε := lt_of_le_of_lt (min_le_left _ _) (by linarith)
    have hs1 : s < θ + Real.pi / 2 := lt_of_le_of_lt (min_le_right _ _) (by linarith)
    have hmem := hball (show s ∈ ball (0 : ℝ) ε by
      rw [mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos hs0]; exact hsε)
    simp only [γ, wrap, mem_preimage, mem_setOf_eq] at hmem
    have : Real.sin (θ - s) < Real.sin θ :=
      Real.sin_lt_sin_of_lt_of_le_pi_div_two (by linarith) hθ2.le (by linarith)
    linarith
  exact @key (0) ((by constructor <;> norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.closed_part_not_open (1/2) ((by constructor <;> norm_num)) := by
  have key : ∀ (t : ℝ) (ht : t ∈ Ioo (-1 : ℝ) 1), ¬ IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t ≤ p.2}) := by
    intro t ht
    intro h
    set θ := Real.arcsin t
    have hθ1 : -(Real.pi / 2) < θ := by
      rw [Real.neg_pi_div_two_lt_arcsin]; exact ht.1
    have hθ2 : θ < Real.pi / 2 := by
      rw [Real.arcsin_lt_pi_div_two]; exact ht.2
    have hsin : Real.sin θ = t := Real.sin_arcsin ht.1.le ht.2.le
    let γ : ℝ → circle := fun s => ⟨wrap (θ - s), by simp [circle, wrap, Real.cos_sq_add_sin_sq]⟩
    have hγ : Continuous γ := by
      refine Continuous.subtype_mk ?_ _
      simp only [wrap]; fun_prop
    have hopen := h.preimage hγ
    have h0 : (0 : ℝ) ∈ γ ⁻¹' ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t ≤ p.2}) := by
      simp [γ, wrap, hsin]
    obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 hopen 0 h0
    set s := min (ε / 2) ((θ + Real.pi / 2) / 2)
    have hs0 : 0 < s := lt_min (by linarith) (by linarith)
    have hsε : s < ε := lt_of_le_of_lt (min_le_left _ _) (by linarith)
    have hs1 : s < θ + Real.pi / 2 := lt_of_le_of_lt (min_le_right _ _) (by linarith)
    have hmem := hball (show s ∈ ball (0 : ℝ) ε by
      rw [mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos hs0]; exact hsε)
    simp only [γ, wrap, mem_preimage, mem_setOf_eq] at hmem
    have : Real.sin (θ - s) < Real.sin θ :=
      Real.sin_lt_sin_of_lt_of_le_pi_div_two (by linarith) hθ2.le (by linarith)
    linarith
  exact @key (1/2) ((by constructor <;> norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.closed_part_not_open (-1/2) ((by constructor <;> norm_num)) := by
  have key : ∀ (t : ℝ) (ht : t ∈ Ioo (-1 : ℝ) 1), ¬ IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t ≤ p.2}) := by
    intro t ht
    intro h
    set θ := Real.arcsin t
    have hθ1 : -(Real.pi / 2) < θ := by
      rw [Real.neg_pi_div_two_lt_arcsin]; exact ht.1
    have hθ2 : θ < Real.pi / 2 := by
      rw [Real.arcsin_lt_pi_div_two]; exact ht.2
    have hsin : Real.sin θ = t := Real.sin_arcsin ht.1.le ht.2.le
    let γ : ℝ → circle := fun s => ⟨wrap (θ - s), by simp [circle, wrap, Real.cos_sq_add_sin_sq]⟩
    have hγ : Continuous γ := by
      refine Continuous.subtype_mk ?_ _
      simp only [wrap]; fun_prop
    have hopen := h.preimage hγ
    have h0 : (0 : ℝ) ∈ γ ⁻¹' ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t ≤ p.2}) := by
      simp [γ, wrap, hsin]
    obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 hopen 0 h0
    set s := min (ε / 2) ((θ + Real.pi / 2) / 2)
    have hs0 : 0 < s := lt_min (by linarith) (by linarith)
    have hsε : s < ε := lt_of_le_of_lt (min_le_left _ _) (by linarith)
    have hs1 : s < θ + Real.pi / 2 := lt_of_le_of_lt (min_le_right _ _) (by linarith)
    have hmem := hball (show s ∈ ball (0 : ℝ) ε by
      rw [mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos hs0]; exact hsε)
    simp only [γ, wrap, mem_preimage, mem_setOf_eq] at hmem
    have : Real.sin (θ - s) < Real.sin θ :=
      Real.sin_lt_sin_of_lt_of_le_pi_div_two (by linarith) hθ2.le (by linarith)
    linarith
  exact @key (-1/2) ((by constructor <;> norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Subspace.closed_part_not_open
-- test: set-closed-part-test.sage

/-! ## exr-subspace-discrete -/

/-- The multiples `Tℤ` (`T > 0`) are a discrete subspace of `ℝ`: `ℤ` for `T = 1`. -/
theorem multiples_discrete (T : ℝ) (hT : 0 < T) : DiscreteTopology (range (fun k : ℤ => (k : ℝ) * T)) := by
  refine DiscreteTopology.of_forall_le_dist hT ?_
  rintro ⟨x, k, rfl⟩ ⟨y, l, rfl⟩ hxy
  have hkl : k ≠ l := fun h => hxy (by subst h; rfl)
  rw [Subtype.dist_eq, Real.dist_eq, ← sub_mul, abs_mul, abs_of_pos hT]
  have : (1 : ℝ) ≤ |(k : ℝ) - l| := by
    rw [← Int.cast_sub, ← Int.cast_abs]
    exact_mod_cast Int.one_le_abs (sub_ne_zero.2 hkl)
  nlinarith

example : veriphy_instance% KaluzaKlein.Sets.Subspace.multiples_discrete (1) ((by norm_num)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), DiscreteTopology (range (fun k : ℤ => (k : ℝ) * T)) := by
    intro T hT
    refine DiscreteTopology.of_forall_le_dist hT ?_
    rintro ⟨x, k, rfl⟩ ⟨y, l, rfl⟩ hxy
    have hkl : k ≠ l := fun h => hxy (by subst h; rfl)
    rw [Subtype.dist_eq, Real.dist_eq, ← sub_mul, abs_mul, abs_of_pos hT]
    have : (1 : ℝ) ≤ |(k : ℝ) - l| := by
      rw [← Int.cast_sub, ← Int.cast_abs]
      exact_mod_cast Int.one_le_abs (sub_ne_zero.2 hkl)
    nlinarith
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.multiples_discrete (2 * Real.pi) ((by positivity)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), DiscreteTopology (range (fun k : ℤ => (k : ℝ) * T)) := by
    intro T hT
    refine DiscreteTopology.of_forall_le_dist hT ?_
    rintro ⟨x, k, rfl⟩ ⟨y, l, rfl⟩ hxy
    have hkl : k ≠ l := fun h => hxy (by subst h; rfl)
    rw [Subtype.dist_eq, Real.dist_eq, ← sub_mul, abs_mul, abs_of_pos hT]
    have : (1 : ℝ) ≤ |(k : ℝ) - l| := by
      rw [← Int.cast_sub, ← Int.cast_abs]
      exact_mod_cast Int.one_le_abs (sub_ne_zero.2 hkl)
    nlinarith
  exact @key (2 * Real.pi) ((by positivity))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.multiples_discrete (1/2) ((by norm_num)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), DiscreteTopology (range (fun k : ℤ => (k : ℝ) * T)) := by
    intro T hT
    refine DiscreteTopology.of_forall_le_dist hT ?_
    rintro ⟨x, k, rfl⟩ ⟨y, l, rfl⟩ hxy
    have hkl : k ≠ l := fun h => hxy (by subst h; rfl)
    rw [Subtype.dist_eq, Real.dist_eq, ← sub_mul, abs_mul, abs_of_pos hT]
    have : (1 : ℝ) ≤ |(k : ℝ) - l| := by
      rw [← Int.cast_sub, ← Int.cast_abs]
      exact_mod_cast Int.one_le_abs (sub_ne_zero.2 hkl)
    nlinarith
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Subspace.multiples_discrete
-- test: set-multiples-discrete-test.sage

/-- The points `c/n` (`n ≥ 1`, `c > 0`) form a discrete subspace of `ℝ`: each is cut out by a small interval
(`{1/n}` for `c = 1`). -/
theorem reciprocals_discrete (c : ℝ) (hc : 0 < c) : DiscreteTopology (range (fun n : ℕ => c / (n + 1))) := by
  rw [discreteTopology_iff_isOpen_singleton]
  rintro ⟨x, n, rfl⟩
  have hn : (0 : ℝ) < n + 1 := by positivity
  set r := c / ((n + 1) * (n + 2))
  have hr : 0 < r := by positivity
  rw [isOpen_induced_iff]
  refine ⟨ball (c / (n + 1)) r, isOpen_ball, ?_⟩
  ext ⟨y, m, rfl⟩
  simp only [mem_preimage, mem_ball, Real.dist_eq, mem_singleton_iff, Subtype.mk.injEq]
  constructor
  · intro h
    by_contra hmn
    have hm : (0 : ℝ) < m + 1 := by positivity
    have hne : m ≠ n := fun h' => hmn (by rw [h'])
    have key : c / ((n + 1) * (n + 2)) ≤ |c / (m + 1) - c / (n + 1)| := by
      rcases lt_or_gt_of_ne hne with hlt | hgt
      · have hlt' : (m : ℝ) + 1 ≤ n := by exact_mod_cast hlt
        rw [abs_of_pos (by rw [sub_pos]; exact div_lt_div_of_pos_left hc hm (by linarith))]
        rw [div_sub_div _ _ hm.ne' hn.ne', div_le_div_iff₀ (by positivity) (by positivity)]
        nlinarith [mul_nonneg (sub_nonneg.2 hlt') (by positivity : (0 : ℝ) ≤ (n : ℝ) + 2), mul_pos hc hn,
          mul_le_mul_of_nonneg_left (show (m : ℝ) + 1 ≤ ((n : ℝ) - m) * ((n : ℝ) + 2) by
            nlinarith [mul_nonneg (sub_nonneg.2 hlt') (by positivity : (0 : ℝ) ≤ (n : ℝ) + 2)])
            (le_of_lt (mul_pos hc hn))]
      · have hgt' : (n : ℝ) + 1 ≤ m := by exact_mod_cast hgt
        rw [abs_sub_comm, abs_of_pos (by rw [sub_pos]; exact div_lt_div_of_pos_left hc hn (by linarith))]
        rw [div_sub_div _ _ hn.ne' hm.ne', div_le_div_iff₀ (by positivity) (by positivity)]
        nlinarith [mul_le_mul_of_nonneg_left (show (m : ℝ) + 1 ≤ ((m : ℝ) - n) * ((n : ℝ) + 2) by
            nlinarith [mul_nonneg (sub_nonneg.2 hgt') hn.le]) (le_of_lt (mul_pos hc hn))]
    linarith
  · intro h
    have h' := congrArg Subtype.val h
    simp only at h'
    rw [h', sub_self, abs_zero]; exact hr

example : veriphy_instance% KaluzaKlein.Sets.Subspace.reciprocals_discrete (1) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), DiscreteTopology (range (fun n : ℕ => c / (n + 1))) := by
    intro c hc
    rw [discreteTopology_iff_isOpen_singleton]
    rintro ⟨x, n, rfl⟩
    have hn : (0 : ℝ) < n + 1 := by positivity
    set r := c / ((n + 1) * (n + 2))
    have hr : 0 < r := by positivity
    rw [isOpen_induced_iff]
    refine ⟨ball (c / (n + 1)) r, isOpen_ball, ?_⟩
    ext ⟨y, m, rfl⟩
    simp only [mem_preimage, mem_ball, Real.dist_eq, mem_singleton_iff, Subtype.mk.injEq]
    constructor
    · intro h
      by_contra hmn
      have hm : (0 : ℝ) < m + 1 := by positivity
      have hne : m ≠ n := fun h' => hmn (by rw [h'])
      have key : c / ((n + 1) * (n + 2)) ≤ |c / (m + 1) - c / (n + 1)| := by
        rcases lt_or_gt_of_ne hne with hlt | hgt
        · have hlt' : (m : ℝ) + 1 ≤ n := by exact_mod_cast hlt
          rw [abs_of_pos (by rw [sub_pos]; exact div_lt_div_of_pos_left hc hm (by linarith))]
          rw [div_sub_div _ _ hm.ne' hn.ne', div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith [mul_nonneg (sub_nonneg.2 hlt') (by positivity : (0 : ℝ) ≤ (n : ℝ) + 2), mul_pos hc hn,
            mul_le_mul_of_nonneg_left (show (m : ℝ) + 1 ≤ ((n : ℝ) - m) * ((n : ℝ) + 2) by
              nlinarith [mul_nonneg (sub_nonneg.2 hlt') (by positivity : (0 : ℝ) ≤ (n : ℝ) + 2)])
              (le_of_lt (mul_pos hc hn))]
        · have hgt' : (n : ℝ) + 1 ≤ m := by exact_mod_cast hgt
          rw [abs_sub_comm, abs_of_pos (by rw [sub_pos]; exact div_lt_div_of_pos_left hc hn (by linarith))]
          rw [div_sub_div _ _ hn.ne' hm.ne', div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith [mul_le_mul_of_nonneg_left (show (m : ℝ) + 1 ≤ ((m : ℝ) - n) * ((n : ℝ) + 2) by
              nlinarith [mul_nonneg (sub_nonneg.2 hgt') hn.le]) (le_of_lt (mul_pos hc hn))]
      linarith
    · intro h
      have h' := congrArg Subtype.val h
      simp only at h'
      rw [h', sub_self, abs_zero]; exact hr
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.reciprocals_discrete (2) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), DiscreteTopology (range (fun n : ℕ => c / (n + 1))) := by
    intro c hc
    rw [discreteTopology_iff_isOpen_singleton]
    rintro ⟨x, n, rfl⟩
    have hn : (0 : ℝ) < n + 1 := by positivity
    set r := c / ((n + 1) * (n + 2))
    have hr : 0 < r := by positivity
    rw [isOpen_induced_iff]
    refine ⟨ball (c / (n + 1)) r, isOpen_ball, ?_⟩
    ext ⟨y, m, rfl⟩
    simp only [mem_preimage, mem_ball, Real.dist_eq, mem_singleton_iff, Subtype.mk.injEq]
    constructor
    · intro h
      by_contra hmn
      have hm : (0 : ℝ) < m + 1 := by positivity
      have hne : m ≠ n := fun h' => hmn (by rw [h'])
      have key : c / ((n + 1) * (n + 2)) ≤ |c / (m + 1) - c / (n + 1)| := by
        rcases lt_or_gt_of_ne hne with hlt | hgt
        · have hlt' : (m : ℝ) + 1 ≤ n := by exact_mod_cast hlt
          rw [abs_of_pos (by rw [sub_pos]; exact div_lt_div_of_pos_left hc hm (by linarith))]
          rw [div_sub_div _ _ hm.ne' hn.ne', div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith [mul_nonneg (sub_nonneg.2 hlt') (by positivity : (0 : ℝ) ≤ (n : ℝ) + 2), mul_pos hc hn,
            mul_le_mul_of_nonneg_left (show (m : ℝ) + 1 ≤ ((n : ℝ) - m) * ((n : ℝ) + 2) by
              nlinarith [mul_nonneg (sub_nonneg.2 hlt') (by positivity : (0 : ℝ) ≤ (n : ℝ) + 2)])
              (le_of_lt (mul_pos hc hn))]
        · have hgt' : (n : ℝ) + 1 ≤ m := by exact_mod_cast hgt
          rw [abs_sub_comm, abs_of_pos (by rw [sub_pos]; exact div_lt_div_of_pos_left hc hn (by linarith))]
          rw [div_sub_div _ _ hn.ne' hm.ne', div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith [mul_le_mul_of_nonneg_left (show (m : ℝ) + 1 ≤ ((m : ℝ) - n) * ((n : ℝ) + 2) by
              nlinarith [mul_nonneg (sub_nonneg.2 hgt') hn.le]) (le_of_lt (mul_pos hc hn))]
      linarith
    · intro h
      have h' := congrArg Subtype.val h
      simp only at h'
      rw [h', sub_self, abs_zero]; exact hr
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.reciprocals_discrete (1/3) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), DiscreteTopology (range (fun n : ℕ => c / (n + 1))) := by
    intro c hc
    rw [discreteTopology_iff_isOpen_singleton]
    rintro ⟨x, n, rfl⟩
    have hn : (0 : ℝ) < n + 1 := by positivity
    set r := c / ((n + 1) * (n + 2))
    have hr : 0 < r := by positivity
    rw [isOpen_induced_iff]
    refine ⟨ball (c / (n + 1)) r, isOpen_ball, ?_⟩
    ext ⟨y, m, rfl⟩
    simp only [mem_preimage, mem_ball, Real.dist_eq, mem_singleton_iff, Subtype.mk.injEq]
    constructor
    · intro h
      by_contra hmn
      have hm : (0 : ℝ) < m + 1 := by positivity
      have hne : m ≠ n := fun h' => hmn (by rw [h'])
      have key : c / ((n + 1) * (n + 2)) ≤ |c / (m + 1) - c / (n + 1)| := by
        rcases lt_or_gt_of_ne hne with hlt | hgt
        · have hlt' : (m : ℝ) + 1 ≤ n := by exact_mod_cast hlt
          rw [abs_of_pos (by rw [sub_pos]; exact div_lt_div_of_pos_left hc hm (by linarith))]
          rw [div_sub_div _ _ hm.ne' hn.ne', div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith [mul_nonneg (sub_nonneg.2 hlt') (by positivity : (0 : ℝ) ≤ (n : ℝ) + 2), mul_pos hc hn,
            mul_le_mul_of_nonneg_left (show (m : ℝ) + 1 ≤ ((n : ℝ) - m) * ((n : ℝ) + 2) by
              nlinarith [mul_nonneg (sub_nonneg.2 hlt') (by positivity : (0 : ℝ) ≤ (n : ℝ) + 2)])
              (le_of_lt (mul_pos hc hn))]
        · have hgt' : (n : ℝ) + 1 ≤ m := by exact_mod_cast hgt
          rw [abs_sub_comm, abs_of_pos (by rw [sub_pos]; exact div_lt_div_of_pos_left hc hn (by linarith))]
          rw [div_sub_div _ _ hn.ne' hm.ne', div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith [mul_le_mul_of_nonneg_left (show (m : ℝ) + 1 ≤ ((m : ℝ) - n) * ((n : ℝ) + 2) by
              nlinarith [mul_nonneg (sub_nonneg.2 hgt') hn.le]) (le_of_lt (mul_pos hc hn))]
      linarith
    · intro h
      have h' := congrArg Subtype.val h
      simp only at h'
      rw [h', sub_self, abs_zero]; exact hr
  exact @key (1/3) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Subspace.reciprocals_discrete
-- test: set-reciprocals-discrete-test.sage

/-- With `0` added, the subspace is not discrete: every open set of `ℝ` containing `0` contains infinitely many
`c/n`, so `{0}` is not open in it. -/
theorem reciprocals_zero_not_discrete (c : ℝ) (hc : 0 < c) :
    ¬ DiscreteTopology ↥(insert (0 : ℝ) (range (fun n : ℕ => c / (n + 1))) : Set ℝ) := by
  set S : Set ℝ := insert (0 : ℝ) (range (fun n : ℕ => c / (n + 1))) with hS
  intro h
  have h0 := (discreteTopology_iff_isOpen_singleton.1 h) ⟨0, mem_insert _ _⟩
  rw [isOpen_induced_iff] at h0
  obtain ⟨t, ht, hpre⟩ := h0
  have h0t : (0 : ℝ) ∈ t := by
    have : (⟨0, mem_insert _ _⟩ : S) ∈ (Subtype.val ⁻¹' t) := by rw [hpre]; rfl
    exact this
  obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 ht 0 h0t
  obtain ⟨n, hn⟩ := exists_nat_gt (c / ε)
  have hpos : (0 : ℝ) < c / (n + 1) := by positivity
  have hlt : c / (n + 1) < ε := by
    rw [div_lt_iff₀ (by positivity)]
    rw [div_lt_iff₀ hε] at hn
    nlinarith
  have hmem : (⟨c / (n + 1), mem_insert_of_mem _ ⟨n, rfl⟩⟩ : S) ∈ (Subtype.val ⁻¹' t) := by
    apply hball
    rw [mem_ball, Real.dist_eq, sub_zero, abs_of_pos hpos]; exact hlt
  rw [hpre] at hmem
  have := congrArg Subtype.val hmem
  simp at this
  rcases this with h1 | h1
  · linarith
  · have : (0 : ℝ) < n + 1 := by positivity
    linarith

example : veriphy_instance% KaluzaKlein.Sets.Subspace.reciprocals_zero_not_discrete (1) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), ¬ DiscreteTopology ↥(insert (0 : ℝ) (range (fun n : ℕ => c / (n + 1))) : Set ℝ) := by
    intro c hc
    set S : Set ℝ := insert (0 : ℝ) (range (fun n : ℕ => c / (n + 1))) with hS
    intro h
    have h0 := (discreteTopology_iff_isOpen_singleton.1 h) ⟨0, mem_insert _ _⟩
    rw [isOpen_induced_iff] at h0
    obtain ⟨t, ht, hpre⟩ := h0
    have h0t : (0 : ℝ) ∈ t := by
      have : (⟨0, mem_insert _ _⟩ : S) ∈ (Subtype.val ⁻¹' t) := by rw [hpre]; rfl
      exact this
    obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 ht 0 h0t
    obtain ⟨n, hn⟩ := exists_nat_gt (c / ε)
    have hpos : (0 : ℝ) < c / (n + 1) := by positivity
    have hlt : c / (n + 1) < ε := by
      rw [div_lt_iff₀ (by positivity)]
      rw [div_lt_iff₀ hε] at hn
      nlinarith
    have hmem : (⟨c / (n + 1), mem_insert_of_mem _ ⟨n, rfl⟩⟩ : S) ∈ (Subtype.val ⁻¹' t) := by
      apply hball
      rw [mem_ball, Real.dist_eq, sub_zero, abs_of_pos hpos]; exact hlt
    rw [hpre] at hmem
    have := congrArg Subtype.val hmem
    simp at this
    rcases this with h1 | h1
    · linarith
    · have : (0 : ℝ) < n + 1 := by positivity
      linarith
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.reciprocals_zero_not_discrete (2) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), ¬ DiscreteTopology ↥(insert (0 : ℝ) (range (fun n : ℕ => c / (n + 1))) : Set ℝ) := by
    intro c hc
    set S : Set ℝ := insert (0 : ℝ) (range (fun n : ℕ => c / (n + 1))) with hS
    intro h
    have h0 := (discreteTopology_iff_isOpen_singleton.1 h) ⟨0, mem_insert _ _⟩
    rw [isOpen_induced_iff] at h0
    obtain ⟨t, ht, hpre⟩ := h0
    have h0t : (0 : ℝ) ∈ t := by
      have : (⟨0, mem_insert _ _⟩ : S) ∈ (Subtype.val ⁻¹' t) := by rw [hpre]; rfl
      exact this
    obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 ht 0 h0t
    obtain ⟨n, hn⟩ := exists_nat_gt (c / ε)
    have hpos : (0 : ℝ) < c / (n + 1) := by positivity
    have hlt : c / (n + 1) < ε := by
      rw [div_lt_iff₀ (by positivity)]
      rw [div_lt_iff₀ hε] at hn
      nlinarith
    have hmem : (⟨c / (n + 1), mem_insert_of_mem _ ⟨n, rfl⟩⟩ : S) ∈ (Subtype.val ⁻¹' t) := by
      apply hball
      rw [mem_ball, Real.dist_eq, sub_zero, abs_of_pos hpos]; exact hlt
    rw [hpre] at hmem
    have := congrArg Subtype.val hmem
    simp at this
    rcases this with h1 | h1
    · linarith
    · have : (0 : ℝ) < n + 1 := by positivity
      linarith
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.reciprocals_zero_not_discrete (1/3) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), ¬ DiscreteTopology ↥(insert (0 : ℝ) (range (fun n : ℕ => c / (n + 1))) : Set ℝ) := by
    intro c hc
    set S : Set ℝ := insert (0 : ℝ) (range (fun n : ℕ => c / (n + 1))) with hS
    intro h
    have h0 := (discreteTopology_iff_isOpen_singleton.1 h) ⟨0, mem_insert _ _⟩
    rw [isOpen_induced_iff] at h0
    obtain ⟨t, ht, hpre⟩ := h0
    have h0t : (0 : ℝ) ∈ t := by
      have : (⟨0, mem_insert _ _⟩ : S) ∈ (Subtype.val ⁻¹' t) := by rw [hpre]; rfl
      exact this
    obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 ht 0 h0t
    obtain ⟨n, hn⟩ := exists_nat_gt (c / ε)
    have hpos : (0 : ℝ) < c / (n + 1) := by positivity
    have hlt : c / (n + 1) < ε := by
      rw [div_lt_iff₀ (by positivity)]
      rw [div_lt_iff₀ hε] at hn
      nlinarith
    have hmem : (⟨c / (n + 1), mem_insert_of_mem _ ⟨n, rfl⟩⟩ : S) ∈ (Subtype.val ⁻¹' t) := by
      apply hball
      rw [mem_ball, Real.dist_eq, sub_zero, abs_of_pos hpos]; exact hlt
    rw [hpre] at hmem
    have := congrArg Subtype.val hmem
    simp at this
    rcases this with h1 | h1
    · linarith
    · have : (0 : ℝ) < n + 1 := by positivity
      linarith
  exact @key (1/3) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Subspace.reciprocals_zero_not_discrete
-- test: set-reciprocals-zero-test.sage

/-! ## exr-subspace-transitive -/

/-- For `Z ⊆ Y ⊆ X`, the topology `Z` inherits from the subspace `Y` (through the inclusion `Z → Y`) is the one it
inherits from `X` directly. -/
theorem subspace_transitive (X : Type) [TopologicalSpace X] (Y Z : Set X) (h : Z ⊆ Y) :
    TopologicalSpace.induced (Set.inclusion h) (inferInstance : TopologicalSpace Y) =
      (inferInstance : TopologicalSpace Z) := by
  change TopologicalSpace.induced (Set.inclusion h) (TopologicalSpace.induced Subtype.val _) =
    TopologicalSpace.induced Subtype.val _
  rw [induced_compose]
  rfl

example : veriphy_instance% KaluzaKlein.Sets.Subspace.subspace_transitive (ℝ) (inferInstance) (Set.univ) (Set.Icc 0 1) ((Set.subset_univ _)) := by
  have key : ∀ (X : Type) [inst1 : TopologicalSpace X] (Y Z : Set X) (h : Z ⊆ Y), TopologicalSpace.induced (Set.inclusion h) (inferInstance : TopologicalSpace Y) = (inferInstance : TopologicalSpace Z) := by
    intro X inst1 Y Z h
    change TopologicalSpace.induced (Set.inclusion h) (TopologicalSpace.induced Subtype.val _) =
      TopologicalSpace.induced Subtype.val _
    rw [induced_compose]
    rfl
  exact @key (ℝ) (inferInstance) (Set.univ) (Set.Icc 0 1) ((Set.subset_univ _))

example : veriphy_instance% KaluzaKlein.Sets.Subspace.subspace_transitive (ℝ) (inferInstance) (∅) (∅) (le_rfl) := by
  have key : ∀ (X : Type) [inst1 : TopologicalSpace X] (Y Z : Set X) (h : Z ⊆ Y), TopologicalSpace.induced (Set.inclusion h) (inferInstance : TopologicalSpace Y) = (inferInstance : TopologicalSpace Z) := by
    intro X inst1 Y Z h
    change TopologicalSpace.induced (Set.inclusion h) (TopologicalSpace.induced Subtype.val _) =
      TopologicalSpace.induced Subtype.val _
    rw [induced_compose]
    rfl
  exact @key (ℝ) (inferInstance) (∅) (∅) (le_rfl)

example : veriphy_instance% KaluzaKlein.Sets.Subspace.subspace_transitive (ℝ × ℝ) (inferInstance) (Set.univ) (KaluzaKlein.Sets.Sets.circle) ((Set.subset_univ _)) := by
  have key : ∀ (X : Type) [inst1 : TopologicalSpace X] (Y Z : Set X) (h : Z ⊆ Y), TopologicalSpace.induced (Set.inclusion h) (inferInstance : TopologicalSpace Y) = (inferInstance : TopologicalSpace Z) := by
    intro X inst1 Y Z h
    change TopologicalSpace.induced (Set.inclusion h) (TopologicalSpace.induced Subtype.val _) =
      TopologicalSpace.induced Subtype.val _
    rw [induced_compose]
    rfl
  exact @key (ℝ × ℝ) (inferInstance) (Set.univ) (KaluzaKlein.Sets.Sets.circle) ((Set.subset_univ _))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Subspace.subspace_transitive
-- test: set-subspace-transitive-test.sage

end KaluzaKlein.Sets.Subspace
