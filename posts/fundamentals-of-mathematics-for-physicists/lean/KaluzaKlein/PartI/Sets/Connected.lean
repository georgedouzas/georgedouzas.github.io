import Mathlib
import KaluzaKlein.PartI.Sets.Compact
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Connected spaces"

The claims of the section "Connected spaces" of `tutorial/part-i-01-sets-relations-topology.qmd`: components, the
connected circle, sphere, torus and `ℝⁿ`, the circle minus one or two points, the components of `O(3)` and `O(2)`
(through Euler's theorem that every rotation has an axis, which makes `SO(3)` path-connected), finite graphs, the
invertible `n × n` matrices, connected images and the intermediate value theorem, path-connectedness of the sphere,
of `ℝⁿ` and of `SO(3)`, and a path along a great circle.
-/

namespace KaluzaKlein.Sets.Connected

open Set Metric Filter Topology Matrix KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Maps KaluzaKlein.Sets.Surjective

/-! ## def-connected -/

/-- The connected components partition a space: each point lies in its component, and two components are equal or
disjoint. -/
theorem components_partition (X : Type) [TopologicalSpace X] (x y : X) :
    x ∈ connectedComponent x ∧
      (connectedComponent x = connectedComponent y ∨ Disjoint (connectedComponent x) (connectedComponent y)) := by
  refine ⟨mem_connectedComponent, ?_⟩
  by_cases h : y ∈ connectedComponent x
  · left; exact (connectedComponent_eq h)
  · right; exact connectedComponent_disjoint (fun h' => h (h' ▸ mem_connectedComponent))

example : veriphy_instance% KaluzaKlein.Sets.Connected.components_partition (ℝ) (inferInstance) (0) (1) := by
  have key : ∀ (X : Type) [inst1 : TopologicalSpace X] (x y : X), x ∈ connectedComponent x ∧ (connectedComponent x = connectedComponent y ∨ Disjoint (connectedComponent x) (connectedComponent y)) := by
    intro X inst1 x y
    refine ⟨mem_connectedComponent, ?_⟩
    by_cases h : y ∈ connectedComponent x
    · left; exact (connectedComponent_eq h)
    · right; exact connectedComponent_disjoint (fun h' => h (h' ▸ mem_connectedComponent))
  exact @key (ℝ) (inferInstance) (0) (1)

example : veriphy_instance% KaluzaKlein.Sets.Connected.components_partition (Unit) (inferInstance) (()) (()) := by
  have key : ∀ (X : Type) [inst1 : TopologicalSpace X] (x y : X), x ∈ connectedComponent x ∧ (connectedComponent x = connectedComponent y ∨ Disjoint (connectedComponent x) (connectedComponent y)) := by
    intro X inst1 x y
    refine ⟨mem_connectedComponent, ?_⟩
    by_cases h : y ∈ connectedComponent x
    · left; exact (connectedComponent_eq h)
    · right; exact connectedComponent_disjoint (fun h' => h (h' ▸ mem_connectedComponent))
  exact @key (Unit) (inferInstance) (()) (())

example : veriphy_instance% KaluzaKlein.Sets.Connected.components_partition (ℤ) (inferInstance) (0) (1) := by
  have key : ∀ (X : Type) [inst1 : TopologicalSpace X] (x y : X), x ∈ connectedComponent x ∧ (connectedComponent x = connectedComponent y ∨ Disjoint (connectedComponent x) (connectedComponent y)) := by
    intro X inst1 x y
    refine ⟨mem_connectedComponent, ?_⟩
    by_cases h : y ∈ connectedComponent x
    · left; exact (connectedComponent_eq h)
    · right; exact connectedComponent_disjoint (fun h' => h (h' ▸ mem_connectedComponent))
  exact @key (ℤ) (inferInstance) (0) (1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.components_partition
-- test: set-components-partition-test.sage

/-! ## exm-connected-circle -/

/-- A set that is the image of a connected space under a continuous map is connected: the circle (image of `ℝ`), the
sphere and the torus (images of `ℝ²`), `ℝⁿ` (image of itself). -/
theorem connected_of_range (X Y : Type) [TopologicalSpace X] [TopologicalSpace Y] [ConnectedSpace X] (S : Set Y)
    (f : X → Y) (hf : Continuous f) (hS : range f = S) : IsConnected S :=
  hS ▸ isConnected_range hf

example : veriphy_instance% KaluzaKlein.Sets.Connected.connected_of_range (ℝ) (ℝ × ℝ) (inferInstance) (inferInstance) (inferInstance) ({p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1 ^ 2}) (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) ((by fun_prop)) ((KaluzaKlein.Sets.Maps.range_wrap 1 zero_le_one)) := by
  have key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] [inst3 : ConnectedSpace X] (S : Set Y) (f : X → Y) (hf : Continuous f) (hS : range f = S), IsConnected S := by
    intro X Y inst1 inst2 inst3 S f hf hS
    exact (hS ▸ isConnected_range hf)
  exact @key (ℝ) (ℝ × ℝ) (inferInstance) (inferInstance) (inferInstance) ({p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1 ^ 2}) (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) ((by fun_prop)) ((KaluzaKlein.Sets.Maps.range_wrap 1 zero_le_one))

example : veriphy_instance% KaluzaKlein.Sets.Connected.connected_of_range (ℝ × ℝ) ((ℝ × ℝ) × (ℝ × ℝ)) (inferInstance) (inferInstance) (inferInstance) ({p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1 ^ 2} ×ˢ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1 ^ 2}) (fun q : ℝ × ℝ => ((1 * Real.cos q.1, 1 * Real.sin q.1), (1 * Real.cos q.2, 1 * Real.sin q.2))) ((by fun_prop)) ((KaluzaKlein.Sets.Products.torus_pairs_of_angles 1 1 zero_le_one zero_le_one)) := by
  have key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] [inst3 : ConnectedSpace X] (S : Set Y) (f : X → Y) (hf : Continuous f) (hS : range f = S), IsConnected S := by
    intro X Y inst1 inst2 inst3 S f hf hS
    exact (hS ▸ isConnected_range hf)
  exact @key (ℝ × ℝ) ((ℝ × ℝ) × (ℝ × ℝ)) (inferInstance) (inferInstance) (inferInstance) ({p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1 ^ 2} ×ˢ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1 ^ 2}) (fun q : ℝ × ℝ => ((1 * Real.cos q.1, 1 * Real.sin q.1), (1 * Real.cos q.2, 1 * Real.sin q.2))) ((by fun_prop)) ((KaluzaKlein.Sets.Products.torus_pairs_of_angles 1 1 zero_le_one zero_le_one))

example : veriphy_instance% KaluzaKlein.Sets.Connected.connected_of_range (EuclideanSpace ℝ (Fin 2)) (EuclideanSpace ℝ (Fin 2)) (inferInstance) (inferInstance) (inferInstance) (Set.univ) (id) (continuous_id) (Set.range_id) := by
  have key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] [inst3 : ConnectedSpace X] (S : Set Y) (f : X → Y) (hf : Continuous f) (hS : range f = S), IsConnected S := by
    intro X Y inst1 inst2 inst3 S f hf hS
    exact (hS ▸ isConnected_range hf)
  exact @key (EuclideanSpace ℝ (Fin 2)) (EuclideanSpace ℝ (Fin 2)) (inferInstance) (inferInstance) (inferInstance) (Set.univ) (id) (continuous_id) (Set.range_id)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.connected_of_range
-- test: set-connected-range-test.sage

/-- Two angles less than a full turn apart give different points of the circle. -/
theorem wrap_ne (x y : ℝ) (h : 0 < y - x) (h' : y - x < 2 * Real.pi) : wrap y ≠ wrap x := by
  intro hw
  have : y ∈ wrap ⁻¹' {wrap x} := hw
  rw [wrap_preimage] at this
  obtain ⟨k, hk⟩ := this
  have hpi := Real.pi_pos
  have h1 : (0 : ℝ) < k := by nlinarith
  have h2 : (k : ℝ) < 1 := by nlinarith
  have : (0 : ℤ) < k := by exact_mod_cast h1
  have : k < (1 : ℤ) := by exact_mod_cast h2
  omega

/-- Removing two points `wrap a`, `wrap b` (`a < b < a + 2π`) disconnects the circle into the two open arcs between
them. -/
theorem circle_minus_two_points (a b : ℝ) (hab : a < b) (hb : b < a + 2 * Real.pi) :
    ¬ IsPreconnected (circle \ {wrap a, wrap b}) := by
  have hpi := Real.pi_pos
  intro hpc
  set c₁ := (a + b) / 2 with hc₁
  set δ₁ := (b - a) / 2 with hδ₁
  set c₂ := (b + (a + 2 * Real.pi)) / 2 with hc₂
  set δ₂ := (a + 2 * Real.pi - b) / 2 with hδ₂
  have h₁ := (Subspace.arc_open c₁ δ₁ (by linarith) (by linarith)).1
  have h₂ := (Subspace.arc_open c₂ δ₂ (by linarith) (by linarith)).1
  have e₁ : Ioo (c₁ - δ₁) (c₁ + δ₁) = Ioo a b := by congr 1 <;> linarith
  have e₂ : Ioo (c₂ - δ₂) (c₂ + δ₂) = Ioo b (a + 2 * Real.pi) := by congr 1 <;> linarith
  rw [e₁] at h₁
  rw [e₂] at h₂
  set H₁ := {p : ℝ × ℝ | Real.cos δ₁ < p.1 * Real.cos c₁ + p.2 * Real.sin c₁}
  set H₂ := {p : ℝ × ℝ | Real.cos δ₂ < p.1 * Real.cos c₂ + p.2 * Real.sin c₂}
  have hU : IsOpen H₁ := isOpen_lt continuous_const (by fun_prop)
  have hV : IsOpen H₂ := isOpen_lt continuous_const (by fun_prop)
  have hT : (0 : ℝ) < 2 * Real.pi := by positivity
  have hcover : circle \ {wrap a, wrap b} ⊆ H₁ ∪ H₂ := by
    rintro p ⟨hp, hne⟩
    simp only [mem_insert_iff, mem_singleton_iff, not_or] at hne
    have hp' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
      rw [range_wrap 1 zero_le_one]; simpa [circle] using hp
    obtain ⟨φ, rfl⟩ := hp'
    set ψ := toIcoMod hT a φ
    have hψ := toIcoMod_mem_Ico hT a φ
    have hwψ : wrap ψ = (1 * Real.cos φ, 1 * Real.sin φ) := by
      simp only [ψ, toIcoMod, zsmul_eq_mul, wrap, one_mul]
      rw [show φ - (toIcoDiv hT a φ : ℝ) * (2 * Real.pi) = φ + ((-toIcoDiv hT a φ : ℤ) : ℝ) * (2 * Real.pi) by
        push_cast; ring, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
    simp only at hne ⊢
    rw [← hwψ] at hne ⊢
    have ha' : ψ ≠ a := fun h => hne.1 (by rw [h])
    have hb' : ψ ≠ b := fun h => hne.2 (by rw [h])
    rcases lt_or_gt_of_ne hb' with hlt | hgt
    · left
      have : wrap ψ ∈ wrap '' Ioo a b := ⟨ψ, ⟨lt_of_le_of_ne hψ.1 (Ne.symm ha'), hlt⟩, rfl⟩
      rw [h₁] at this; exact this.2
    · right
      have : wrap ψ ∈ wrap '' Ioo b (a + 2 * Real.pi) := ⟨ψ, ⟨hgt, hψ.2⟩, rfl⟩
      rw [h₂] at this; exact this.2
  have hmem₁ : (circle \ {wrap a, wrap b} ∩ H₁).Nonempty := by
    have : wrap c₁ ∈ wrap '' Ioo a b := ⟨c₁, ⟨by linarith, by linarith⟩, rfl⟩
    rw [h₁] at this
    refine ⟨wrap c₁, ⟨this.1, ?_⟩, this.2⟩
    simp only [mem_insert_iff, mem_singleton_iff, not_or]
    exact ⟨wrap_ne a c₁ (by linarith) (by linarith), fun h => wrap_ne c₁ b (by linarith) (by linarith) h.symm⟩
  have hmem₂ : (circle \ {wrap a, wrap b} ∩ H₂).Nonempty := by
    have : wrap c₂ ∈ wrap '' Ioo b (a + 2 * Real.pi) := ⟨c₂, ⟨by linarith, by linarith⟩, rfl⟩
    rw [h₂] at this
    refine ⟨wrap c₂, ⟨this.1, ?_⟩, this.2⟩
    simp only [mem_insert_iff, mem_singleton_iff, not_or]
    exact ⟨wrap_ne a c₂ (by linarith) (by linarith), wrap_ne b c₂ (by linarith) (by linarith)⟩
  obtain ⟨p, ⟨hp, -⟩, hp1, hp2⟩ := hpc H₁ H₂ hU hV hcover hmem₁ hmem₂
  have hA : p ∈ wrap '' Ioo a b := by rw [h₁]; exact ⟨hp, hp1⟩
  have hB : p ∈ wrap '' Ioo b (a + 2 * Real.pi) := by rw [h₂]; exact ⟨hp, hp2⟩
  obtain ⟨φ₁, hφ₁, rfl⟩ := hA
  obtain ⟨φ₂, hφ₂, h⟩ := hB
  have := Injective.wrap_injOn a ⟨hφ₂.1.trans' hab, hφ₂.2⟩ ⟨hφ₁.1, hφ₁.2.trans hb⟩ h
  linarith [hφ₁.2, hφ₂.1]

example : veriphy_instance% KaluzaKlein.Sets.Connected.circle_minus_two_points (0) (Real.pi) (Real.pi_pos) ((by linarith [Real.pi_gt_three])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (a b : ℝ) (hab : a < b) (hb : b < a + 2 * Real.pi), ¬ IsPreconnected (circle \ {wrap a, wrap b}) := by
    intro a b hab hb
    have hpi := Real.pi_pos
    intro hpc
    set c₁ := (a + b) / 2 with hc₁
    set δ₁ := (b - a) / 2 with hδ₁
    set c₂ := (b + (a + 2 * Real.pi)) / 2 with hc₂
    set δ₂ := (a + 2 * Real.pi - b) / 2 with hδ₂
    have h₁ := (Subspace.arc_open c₁ δ₁ (by linarith) (by linarith)).1
    have h₂ := (Subspace.arc_open c₂ δ₂ (by linarith) (by linarith)).1
    have e₁ : Ioo (c₁ - δ₁) (c₁ + δ₁) = Ioo a b := by congr 1 <;> linarith
    have e₂ : Ioo (c₂ - δ₂) (c₂ + δ₂) = Ioo b (a + 2 * Real.pi) := by congr 1 <;> linarith
    rw [e₁] at h₁
    rw [e₂] at h₂
    set H₁ := {p : ℝ × ℝ | Real.cos δ₁ < p.1 * Real.cos c₁ + p.2 * Real.sin c₁}
    set H₂ := {p : ℝ × ℝ | Real.cos δ₂ < p.1 * Real.cos c₂ + p.2 * Real.sin c₂}
    have hU : IsOpen H₁ := isOpen_lt continuous_const (by fun_prop)
    have hV : IsOpen H₂ := isOpen_lt continuous_const (by fun_prop)
    have hT : (0 : ℝ) < 2 * Real.pi := by positivity
    have hcover : circle \ {wrap a, wrap b} ⊆ H₁ ∪ H₂ := by
      rintro p ⟨hp, hne⟩
      simp only [mem_insert_iff, mem_singleton_iff, not_or] at hne
      have hp' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hp
      obtain ⟨φ, rfl⟩ := hp'
      set ψ := toIcoMod hT a φ
      have hψ := toIcoMod_mem_Ico hT a φ
      have hwψ : wrap ψ = (1 * Real.cos φ, 1 * Real.sin φ) := by
        simp only [ψ, toIcoMod, zsmul_eq_mul, wrap, one_mul]
        rw [show φ - (toIcoDiv hT a φ : ℝ) * (2 * Real.pi) = φ + ((-toIcoDiv hT a φ : ℤ) : ℝ) * (2 * Real.pi) by
          push_cast; ring, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
      simp only at hne ⊢
      rw [← hwψ] at hne ⊢
      have ha' : ψ ≠ a := fun h => hne.1 (by rw [h])
      have hb' : ψ ≠ b := fun h => hne.2 (by rw [h])
      rcases lt_or_gt_of_ne hb' with hlt | hgt
      · left
        have : wrap ψ ∈ wrap '' Ioo a b := ⟨ψ, ⟨lt_of_le_of_ne hψ.1 (Ne.symm ha'), hlt⟩, rfl⟩
        rw [h₁] at this; exact this.2
      · right
        have : wrap ψ ∈ wrap '' Ioo b (a + 2 * Real.pi) := ⟨ψ, ⟨hgt, hψ.2⟩, rfl⟩
        rw [h₂] at this; exact this.2
    have hmem₁ : (circle \ {wrap a, wrap b} ∩ H₁).Nonempty := by
      have : wrap c₁ ∈ wrap '' Ioo a b := ⟨c₁, ⟨by linarith, by linarith⟩, rfl⟩
      rw [h₁] at this
      refine ⟨wrap c₁, ⟨this.1, ?_⟩, this.2⟩
      simp only [mem_insert_iff, mem_singleton_iff, not_or]
      exact ⟨wrap_ne a c₁ (by linarith) (by linarith), fun h => wrap_ne c₁ b (by linarith) (by linarith) h.symm⟩
    have hmem₂ : (circle \ {wrap a, wrap b} ∩ H₂).Nonempty := by
      have : wrap c₂ ∈ wrap '' Ioo b (a + 2 * Real.pi) := ⟨c₂, ⟨by linarith, by linarith⟩, rfl⟩
      rw [h₂] at this
      refine ⟨wrap c₂, ⟨this.1, ?_⟩, this.2⟩
      simp only [mem_insert_iff, mem_singleton_iff, not_or]
      exact ⟨wrap_ne a c₂ (by linarith) (by linarith), wrap_ne b c₂ (by linarith) (by linarith)⟩
    obtain ⟨p, ⟨hp, -⟩, hp1, hp2⟩ := hpc H₁ H₂ hU hV hcover hmem₁ hmem₂
    have hA : p ∈ wrap '' Ioo a b := by rw [h₁]; exact ⟨hp, hp1⟩
    have hB : p ∈ wrap '' Ioo b (a + 2 * Real.pi) := by rw [h₂]; exact ⟨hp, hp2⟩
    obtain ⟨φ₁, hφ₁, rfl⟩ := hA
    obtain ⟨φ₂, hφ₂, h⟩ := hB
    have := Injective.wrap_injOn a ⟨hφ₂.1.trans' hab, hφ₂.2⟩ ⟨hφ₁.1, hφ₁.2.trans hb⟩ h
    linarith [hφ₁.2, hφ₂.1]
  exact @key (0) (Real.pi) (Real.pi_pos) ((by linarith [Real.pi_gt_three]))

example : veriphy_instance% KaluzaKlein.Sets.Connected.circle_minus_two_points (0) (Real.pi / 2) ((by positivity)) ((by linarith [Real.pi_gt_three])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (a b : ℝ) (hab : a < b) (hb : b < a + 2 * Real.pi), ¬ IsPreconnected (circle \ {wrap a, wrap b}) := by
    intro a b hab hb
    have hpi := Real.pi_pos
    intro hpc
    set c₁ := (a + b) / 2 with hc₁
    set δ₁ := (b - a) / 2 with hδ₁
    set c₂ := (b + (a + 2 * Real.pi)) / 2 with hc₂
    set δ₂ := (a + 2 * Real.pi - b) / 2 with hδ₂
    have h₁ := (Subspace.arc_open c₁ δ₁ (by linarith) (by linarith)).1
    have h₂ := (Subspace.arc_open c₂ δ₂ (by linarith) (by linarith)).1
    have e₁ : Ioo (c₁ - δ₁) (c₁ + δ₁) = Ioo a b := by congr 1 <;> linarith
    have e₂ : Ioo (c₂ - δ₂) (c₂ + δ₂) = Ioo b (a + 2 * Real.pi) := by congr 1 <;> linarith
    rw [e₁] at h₁
    rw [e₂] at h₂
    set H₁ := {p : ℝ × ℝ | Real.cos δ₁ < p.1 * Real.cos c₁ + p.2 * Real.sin c₁}
    set H₂ := {p : ℝ × ℝ | Real.cos δ₂ < p.1 * Real.cos c₂ + p.2 * Real.sin c₂}
    have hU : IsOpen H₁ := isOpen_lt continuous_const (by fun_prop)
    have hV : IsOpen H₂ := isOpen_lt continuous_const (by fun_prop)
    have hT : (0 : ℝ) < 2 * Real.pi := by positivity
    have hcover : circle \ {wrap a, wrap b} ⊆ H₁ ∪ H₂ := by
      rintro p ⟨hp, hne⟩
      simp only [mem_insert_iff, mem_singleton_iff, not_or] at hne
      have hp' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hp
      obtain ⟨φ, rfl⟩ := hp'
      set ψ := toIcoMod hT a φ
      have hψ := toIcoMod_mem_Ico hT a φ
      have hwψ : wrap ψ = (1 * Real.cos φ, 1 * Real.sin φ) := by
        simp only [ψ, toIcoMod, zsmul_eq_mul, wrap, one_mul]
        rw [show φ - (toIcoDiv hT a φ : ℝ) * (2 * Real.pi) = φ + ((-toIcoDiv hT a φ : ℤ) : ℝ) * (2 * Real.pi) by
          push_cast; ring, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
      simp only at hne ⊢
      rw [← hwψ] at hne ⊢
      have ha' : ψ ≠ a := fun h => hne.1 (by rw [h])
      have hb' : ψ ≠ b := fun h => hne.2 (by rw [h])
      rcases lt_or_gt_of_ne hb' with hlt | hgt
      · left
        have : wrap ψ ∈ wrap '' Ioo a b := ⟨ψ, ⟨lt_of_le_of_ne hψ.1 (Ne.symm ha'), hlt⟩, rfl⟩
        rw [h₁] at this; exact this.2
      · right
        have : wrap ψ ∈ wrap '' Ioo b (a + 2 * Real.pi) := ⟨ψ, ⟨hgt, hψ.2⟩, rfl⟩
        rw [h₂] at this; exact this.2
    have hmem₁ : (circle \ {wrap a, wrap b} ∩ H₁).Nonempty := by
      have : wrap c₁ ∈ wrap '' Ioo a b := ⟨c₁, ⟨by linarith, by linarith⟩, rfl⟩
      rw [h₁] at this
      refine ⟨wrap c₁, ⟨this.1, ?_⟩, this.2⟩
      simp only [mem_insert_iff, mem_singleton_iff, not_or]
      exact ⟨wrap_ne a c₁ (by linarith) (by linarith), fun h => wrap_ne c₁ b (by linarith) (by linarith) h.symm⟩
    have hmem₂ : (circle \ {wrap a, wrap b} ∩ H₂).Nonempty := by
      have : wrap c₂ ∈ wrap '' Ioo b (a + 2 * Real.pi) := ⟨c₂, ⟨by linarith, by linarith⟩, rfl⟩
      rw [h₂] at this
      refine ⟨wrap c₂, ⟨this.1, ?_⟩, this.2⟩
      simp only [mem_insert_iff, mem_singleton_iff, not_or]
      exact ⟨wrap_ne a c₂ (by linarith) (by linarith), wrap_ne b c₂ (by linarith) (by linarith)⟩
    obtain ⟨p, ⟨hp, -⟩, hp1, hp2⟩ := hpc H₁ H₂ hU hV hcover hmem₁ hmem₂
    have hA : p ∈ wrap '' Ioo a b := by rw [h₁]; exact ⟨hp, hp1⟩
    have hB : p ∈ wrap '' Ioo b (a + 2 * Real.pi) := by rw [h₂]; exact ⟨hp, hp2⟩
    obtain ⟨φ₁, hφ₁, rfl⟩ := hA
    obtain ⟨φ₂, hφ₂, h⟩ := hB
    have := Injective.wrap_injOn a ⟨hφ₂.1.trans' hab, hφ₂.2⟩ ⟨hφ₁.1, hφ₁.2.trans hb⟩ h
    linarith [hφ₁.2, hφ₂.1]
  exact @key (0) (Real.pi / 2) ((by positivity)) ((by linarith [Real.pi_gt_three]))

example : veriphy_instance% KaluzaKlein.Sets.Connected.circle_minus_two_points (0) (1) ((by norm_num)) ((by linarith [Real.pi_gt_three])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (a b : ℝ) (hab : a < b) (hb : b < a + 2 * Real.pi), ¬ IsPreconnected (circle \ {wrap a, wrap b}) := by
    intro a b hab hb
    have hpi := Real.pi_pos
    intro hpc
    set c₁ := (a + b) / 2 with hc₁
    set δ₁ := (b - a) / 2 with hδ₁
    set c₂ := (b + (a + 2 * Real.pi)) / 2 with hc₂
    set δ₂ := (a + 2 * Real.pi - b) / 2 with hδ₂
    have h₁ := (Subspace.arc_open c₁ δ₁ (by linarith) (by linarith)).1
    have h₂ := (Subspace.arc_open c₂ δ₂ (by linarith) (by linarith)).1
    have e₁ : Ioo (c₁ - δ₁) (c₁ + δ₁) = Ioo a b := by congr 1 <;> linarith
    have e₂ : Ioo (c₂ - δ₂) (c₂ + δ₂) = Ioo b (a + 2 * Real.pi) := by congr 1 <;> linarith
    rw [e₁] at h₁
    rw [e₂] at h₂
    set H₁ := {p : ℝ × ℝ | Real.cos δ₁ < p.1 * Real.cos c₁ + p.2 * Real.sin c₁}
    set H₂ := {p : ℝ × ℝ | Real.cos δ₂ < p.1 * Real.cos c₂ + p.2 * Real.sin c₂}
    have hU : IsOpen H₁ := isOpen_lt continuous_const (by fun_prop)
    have hV : IsOpen H₂ := isOpen_lt continuous_const (by fun_prop)
    have hT : (0 : ℝ) < 2 * Real.pi := by positivity
    have hcover : circle \ {wrap a, wrap b} ⊆ H₁ ∪ H₂ := by
      rintro p ⟨hp, hne⟩
      simp only [mem_insert_iff, mem_singleton_iff, not_or] at hne
      have hp' : p ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hp
      obtain ⟨φ, rfl⟩ := hp'
      set ψ := toIcoMod hT a φ
      have hψ := toIcoMod_mem_Ico hT a φ
      have hwψ : wrap ψ = (1 * Real.cos φ, 1 * Real.sin φ) := by
        simp only [ψ, toIcoMod, zsmul_eq_mul, wrap, one_mul]
        rw [show φ - (toIcoDiv hT a φ : ℝ) * (2 * Real.pi) = φ + ((-toIcoDiv hT a φ : ℤ) : ℝ) * (2 * Real.pi) by
          push_cast; ring, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
      simp only at hne ⊢
      rw [← hwψ] at hne ⊢
      have ha' : ψ ≠ a := fun h => hne.1 (by rw [h])
      have hb' : ψ ≠ b := fun h => hne.2 (by rw [h])
      rcases lt_or_gt_of_ne hb' with hlt | hgt
      · left
        have : wrap ψ ∈ wrap '' Ioo a b := ⟨ψ, ⟨lt_of_le_of_ne hψ.1 (Ne.symm ha'), hlt⟩, rfl⟩
        rw [h₁] at this; exact this.2
      · right
        have : wrap ψ ∈ wrap '' Ioo b (a + 2 * Real.pi) := ⟨ψ, ⟨hgt, hψ.2⟩, rfl⟩
        rw [h₂] at this; exact this.2
    have hmem₁ : (circle \ {wrap a, wrap b} ∩ H₁).Nonempty := by
      have : wrap c₁ ∈ wrap '' Ioo a b := ⟨c₁, ⟨by linarith, by linarith⟩, rfl⟩
      rw [h₁] at this
      refine ⟨wrap c₁, ⟨this.1, ?_⟩, this.2⟩
      simp only [mem_insert_iff, mem_singleton_iff, not_or]
      exact ⟨wrap_ne a c₁ (by linarith) (by linarith), fun h => wrap_ne c₁ b (by linarith) (by linarith) h.symm⟩
    have hmem₂ : (circle \ {wrap a, wrap b} ∩ H₂).Nonempty := by
      have : wrap c₂ ∈ wrap '' Ioo b (a + 2 * Real.pi) := ⟨c₂, ⟨by linarith, by linarith⟩, rfl⟩
      rw [h₂] at this
      refine ⟨wrap c₂, ⟨this.1, ?_⟩, this.2⟩
      simp only [mem_insert_iff, mem_singleton_iff, not_or]
      exact ⟨wrap_ne a c₂ (by linarith) (by linarith), wrap_ne b c₂ (by linarith) (by linarith)⟩
    obtain ⟨p, ⟨hp, -⟩, hp1, hp2⟩ := hpc H₁ H₂ hU hV hcover hmem₁ hmem₂
    have hA : p ∈ wrap '' Ioo a b := by rw [h₁]; exact ⟨hp, hp1⟩
    have hB : p ∈ wrap '' Ioo b (a + 2 * Real.pi) := by rw [h₂]; exact ⟨hp, hp2⟩
    obtain ⟨φ₁, hφ₁, rfl⟩ := hA
    obtain ⟨φ₂, hφ₂, h⟩ := hB
    have := Injective.wrap_injOn a ⟨hφ₂.1.trans' hab, hφ₂.2⟩ ⟨hφ₁.1, hφ₁.2.trans hb⟩ h
    linarith [hφ₁.2, hφ₂.1]
  exact @key (0) (1) ((by norm_num)) ((by linarith [Real.pi_gt_three]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.circle_minus_two_points
-- test: set-circle-two-points-test.sage

/-! ## exm-connected-groups and exm-path-rotations: every rotation has an axis -/

/-- `det (R - 1) = 0` for a rotation `R`, since `det (R - 1) = det (1 - R) = -det (R - 1)` in dimension three. -/
theorem det_sub_one (R : Matrix (Fin 3) (Fin 3) ℝ) (hR : R ∈ SO3) : (R - 1).det = 0 := by
  rw [mem_SO3_iff] at hR
  have hRT : R * Rᵀ = 1 := mul_eq_one_comm.1 hR.1
  have h1 : R - 1 = R * (1 - Rᵀ) := by rw [Matrix.mul_sub, hRT, Matrix.mul_one]
  have h2 : (R - 1).det = (1 - R).det := by
    rw [h1, Matrix.det_mul, hR.2, one_mul, ← Matrix.det_transpose, Matrix.transpose_sub, Matrix.transpose_one,
      Matrix.transpose_transpose]
  have h3 : (1 - R).det = -(R - 1).det := by
    rw [show (1 - R) = (-1 : ℝ) • (R - 1) by rw [neg_one_smul, neg_sub], Matrix.det_smul, Fintype.card_fin]
    norm_num
  linarith

/-- Euler's theorem: every rotation is a rotation `P R_z(α) Pᵀ` about some axis `P (0, 0, 1)`. -/
theorem rotation_axis (R : Matrix (Fin 3) (Fin 3) ℝ) (hR : R ∈ SO3) :
    ∃ P ∈ SO3, ∃ α : ℝ, R = P * Rz α * Pᵀ := by
  obtain ⟨v, hv0, hv⟩ := Matrix.exists_mulVec_eq_zero_iff.2 (det_sub_one R hR)
  rw [Matrix.sub_mulVec, Matrix.one_mulVec, sub_eq_zero] at hv
  -- normalise the fixed vector to a point `n` of the sphere
  have hd : 0 < v ⬝ᵥ v := by
    rw [dotProduct]
    have : ∃ i, v i ≠ 0 := by by_contra h; push_neg at h; exact hv0 (funext h)
    obtain ⟨i, hi⟩ := this
    calc 0 < v i * v i := mul_self_pos.2 hi
      _ ≤ ∑ j, v j * v j := Finset.single_le_sum (fun j _ => mul_self_nonneg (v j)) (Finset.mem_univ i)
  set n := (Real.sqrt (v ⬝ᵥ v))⁻¹ • v
  have hn : n ∈ sphere := by
    rw [Quotients.mem_sphere_iff_dot]
    simp only [n, dotProduct_smul, smul_dotProduct, smul_eq_mul]
    rw [← mul_assoc, ← sq, inv_pow, Real.sq_sqrt hd.le, inv_mul_cancel₀ hd.ne']
  have hRn : R *ᵥ n = n := by simp only [n, Matrix.mulVec_smul, hv]
  set P := rotTo n
  have hP := rotTo_mem n hn
  have hP' := hP
  rw [mem_SO3_iff] at hP'
  have hPPT : P * Pᵀ = 1 := mul_eq_one_comm.1 hP'.1
  have hPe : P *ᵥ northPole = n := rotTo_orbit n hn
  have hPTn : Pᵀ *ᵥ n = northPole := by rw [← hPe, Matrix.mulVec_mulVec, hP'.1, Matrix.one_mulVec]
  have hPT : Pᵀ ∈ SO3 := by
    rw [mem_SO3_iff]; exact ⟨by rw [Matrix.transpose_transpose]; exact hPPT, by rw [Matrix.det_transpose, hP'.2]⟩
  have hQ : Pᵀ * R * P ∈ SO3 :=
    (Matrix.specialOrthogonalGroup (Fin 3) ℝ).mul_mem ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).mul_mem hPT hR) hP
  have hQe : orbit (Pᵀ * R * P) = northPole := by
    simp only [orbit]
    rw [← Matrix.mulVec_mulVec, hPe, ← Matrix.mulVec_mulVec, hRn, hPTn]
  obtain ⟨α, hα⟩ := (stabilizer_eq _).1 ⟨hQ, hQe⟩
  refine ⟨P, hP, α, ?_⟩
  rw [← hα, ← Matrix.mul_assoc, ← Matrix.mul_assoc, hPPT, Matrix.one_mul, Matrix.mul_assoc, hPPT, Matrix.mul_one]

/-- The rotations about a common axis by the angles `tα` form a continuous path of rotations from the identity
(`t = 0`) to `P R_z(α) Pᵀ` (`t = 1`): `SO(3)` is path-connected, every rotation being joined to the identity. -/
theorem rotation_path (R : Matrix (Fin 3) (Fin 3) ℝ) (hR : R ∈ SO3) : JoinedIn SO3 1 R := by
  obtain ⟨P, hP, α, rfl⟩ := rotation_axis R hR
  have hP' := hP
  rw [mem_SO3_iff] at hP'
  have hPPT : P * Pᵀ = 1 := mul_eq_one_comm.1 hP'.1
  have hPT : Pᵀ ∈ SO3 := by
    rw [mem_SO3_iff]; exact ⟨by rw [Matrix.transpose_transpose]; exact hPPT, by rw [Matrix.det_transpose, hP'.2]⟩
  refine JoinedIn.ofLine (f := fun t : ℝ => P * Rz (t * α) * Pᵀ) ?_ ?_ ?_ ?_
  · refine Continuous.continuousOn ?_
    refine (continuous_const.matrix_mul ?_).matrix_mul continuous_const
    refine continuous_pi fun i => continuous_pi fun j => ?_
    fin_cases i <;> fin_cases j <;> simp [Rz] <;> fun_prop
  · simp only [zero_mul]
    have : Rz 0 = 1 := by ext i j; fin_cases i <;> fin_cases j <;> simp [Rz]
    rw [this, Matrix.mul_one, hPPT]
  · simp
  · rintro _ ⟨t, -, rfl⟩
    exact (Matrix.specialOrthogonalGroup (Fin 3) ℝ).mul_mem ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).mul_mem hP
      ((stabilizer_eq _).2 ⟨t * α, rfl⟩).1) hPT

/-- `SO(3)` is path-connected, hence connected. -/
theorem SO3_pathConnected : IsPathConnected SO3 :=
  ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem, fun {y} hy => rotation_path y hy⟩

instance SO3_connectedSpace : ConnectedSpace Quotients.SO3t :=
  isConnected_iff_connectedSpace.1 SO3_pathConnected.isConnected

/-- The orbit map sends `SO(3)` onto the sphere. -/
theorem range_orbit_SO3 : Set.range (fun g : Quotients.SO3t => orbit (g : Matrix (Fin 3) (Fin 3) ℝ)) = sphere := by
  ext p
  constructor
  · rintro ⟨g, rfl⟩; exact (orbit_image _).2 ⟨g, g.2, rfl⟩
  · intro hp
    obtain ⟨g, hg, rfl⟩ := (orbit_image p).1 hp
    exact ⟨⟨g, hg⟩, rfl⟩

/-- The orthogonal group `O(3)`. -/
abbrev O3 : Set (Matrix (Fin 3) (Fin 3) ℝ) := (Matrix.orthogonalGroup (Fin 3) ℝ : Set _)

theorem mem_O3_iff (g : Matrix (Fin 3) (Fin 3) ℝ) : g ∈ O3 ↔ gᵀ * g = 1 := by
  rw [SetLike.mem_coe, Matrix.mem_orthogonalGroup_iff']

/-- `O(3)` has two connected components, the rotations (`det = 1`) and the reflections (`det = -1`): the component of
`g` is the set of orthogonal matrices with the determinant of `g`, since the determinant is continuous with values
`±1`, and each of the two sets is a translate of the connected `SO(3)`. -/
theorem O3_components (g : Matrix (Fin 3) (Fin 3) ℝ) (hg : g ∈ O3) :
    connectedComponentIn O3 g = {h ∈ O3 | h.det = g.det} := by
  have hdet : ∀ h ∈ O3, h.det = 1 ∨ h.det = -1 := by
    intro h hh
    rw [mem_O3_iff] at hh
    have := congrArg Matrix.det hh
    rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
    rcases mul_self_eq_one_iff.1 this with h1 | h1
    · left; exact h1
    · right; exact h1
  have hg' := hg
  rw [mem_O3_iff] at hg'
  have hggT : g * gᵀ = 1 := mul_eq_one_comm.1 hg'
  apply Subset.antisymm
  · -- the determinant is constant on the (preconnected) component
    intro h hh
    refine ⟨connectedComponentIn_subset _ _ hh, ?_⟩
    by_contra hne
    have hpc := (isPreconnected_connectedComponentIn (F := O3) (x := g)).image Matrix.det
      continuous_id.matrix_det.continuousOn
    have hg1 := hdet g hg
    have hh1 := hdet h (connectedComponentIn_subset _ _ hh)
    have h0 : (0 : ℝ) ∈ Matrix.det '' connectedComponentIn O3 g := by
      have hgmem : g.det ∈ Matrix.det '' connectedComponentIn O3 g :=
        ⟨g, mem_connectedComponentIn hg, rfl⟩
      have hhmem : h.det ∈ Matrix.det '' connectedComponentIn O3 g := ⟨h, hh, rfl⟩
      rcases hg1 with hg1 | hg1 <;> rcases hh1 with hh1 | hh1
      · exact absurd (hh1.trans hg1.symm) hne
      · rw [hg1] at hgmem; rw [hh1] at hhmem
        exact hpc.Icc_subset hhmem hgmem ⟨by norm_num, by norm_num⟩
      · rw [hg1] at hgmem; rw [hh1] at hhmem
        exact hpc.Icc_subset hgmem hhmem ⟨by norm_num, by norm_num⟩
      · exact absurd (hh1.trans hg1.symm) hne
    obtain ⟨k, hk, hk0⟩ := h0
    rcases hdet k (connectedComponentIn_subset _ _ hk) with h1 | h1 <;> linarith
  · -- the set is `g · SO(3)`, connected, containing `g`
    have hset : {h ∈ O3 | h.det = g.det} = (g * ·) '' SO3 := by
      ext h
      simp only [mem_setOf_eq, mem_image]
      constructor
      · rintro ⟨hh, hd⟩
        refine ⟨gᵀ * h, ?_, by rw [← Matrix.mul_assoc, hggT, Matrix.one_mul]⟩
        rw [mem_SO3_iff]
        rw [mem_O3_iff] at hh
        refine ⟨?_, ?_⟩
        · rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc g, hggT,
            Matrix.one_mul, hh]
        · rw [Matrix.det_mul, Matrix.det_transpose, hd]
          have := congrArg Matrix.det hg'
          rwa [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
      · rintro ⟨k, hk, rfl⟩
        rw [mem_SO3_iff] at hk
        refine ⟨?_, by rw [Matrix.det_mul, hk.2, mul_one]⟩
        rw [mem_O3_iff, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul, hk.1]
    rw [hset]
    have hSO3 : IsPreconnected SO3 := by
      have : IsPathConnected SO3 := ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem,
        fun {y} hy => rotation_path y hy⟩
      exact this.isConnected.isPreconnected
    refine (hSO3.image _ (continuous_const.matrix_mul continuous_id).continuousOn).subset_connectedComponentIn
      ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem, Matrix.mul_one g⟩ ?_
    rintro _ ⟨k, hk, rfl⟩
    rw [mem_SO3_iff] at hk
    rw [mem_O3_iff, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul]
    exact hk.1

example : veriphy_instance% KaluzaKlein.Sets.Connected.O3_components (1) ((Matrix.orthogonalGroup (Fin 3) ℝ).one_mem) := by
  have key : ∀ (g : Matrix (Fin 3) (Fin 3) ℝ) (hg : g ∈ O3), connectedComponentIn O3 g = {h ∈ O3 | h.det = g.det} := by
    intro g hg
    have hdet : ∀ h ∈ O3, h.det = 1 ∨ h.det = -1 := by
      intro h hh
      rw [mem_O3_iff] at hh
      have := congrArg Matrix.det hh
      rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
      rcases mul_self_eq_one_iff.1 this with h1 | h1
      · left; exact h1
      · right; exact h1
    have hg' := hg
    rw [mem_O3_iff] at hg'
    have hggT : g * gᵀ = 1 := mul_eq_one_comm.1 hg'
    apply Subset.antisymm
    · -- the determinant is constant on the (preconnected) component
      intro h hh
      refine ⟨connectedComponentIn_subset _ _ hh, ?_⟩
      by_contra hne
      have hpc := (isPreconnected_connectedComponentIn (F := O3) (x := g)).image Matrix.det
        continuous_id.matrix_det.continuousOn
      have hg1 := hdet g hg
      have hh1 := hdet h (connectedComponentIn_subset _ _ hh)
      have h0 : (0 : ℝ) ∈ Matrix.det '' connectedComponentIn O3 g := by
        have hgmem : g.det ∈ Matrix.det '' connectedComponentIn O3 g :=
          ⟨g, mem_connectedComponentIn hg, rfl⟩
        have hhmem : h.det ∈ Matrix.det '' connectedComponentIn O3 g := ⟨h, hh, rfl⟩
        rcases hg1 with hg1 | hg1 <;> rcases hh1 with hh1 | hh1
        · exact absurd (hh1.trans hg1.symm) hne
        · rw [hg1] at hgmem; rw [hh1] at hhmem
          exact hpc.Icc_subset hhmem hgmem ⟨by norm_num, by norm_num⟩
        · rw [hg1] at hgmem; rw [hh1] at hhmem
          exact hpc.Icc_subset hgmem hhmem ⟨by norm_num, by norm_num⟩
        · exact absurd (hh1.trans hg1.symm) hne
      obtain ⟨k, hk, hk0⟩ := h0
      rcases hdet k (connectedComponentIn_subset _ _ hk) with h1 | h1 <;> linarith
    · -- the set is `g · SO(3)`, connected, containing `g`
      have hset : {h ∈ O3 | h.det = g.det} = (g * ·) '' SO3 := by
        ext h
        simp only [mem_setOf_eq, mem_image]
        constructor
        · rintro ⟨hh, hd⟩
          refine ⟨gᵀ * h, ?_, by rw [← Matrix.mul_assoc, hggT, Matrix.one_mul]⟩
          rw [mem_SO3_iff]
          rw [mem_O3_iff] at hh
          refine ⟨?_, ?_⟩
          · rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc g, hggT,
              Matrix.one_mul, hh]
          · rw [Matrix.det_mul, Matrix.det_transpose, hd]
            have := congrArg Matrix.det hg'
            rwa [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
        · rintro ⟨k, hk, rfl⟩
          rw [mem_SO3_iff] at hk
          refine ⟨?_, by rw [Matrix.det_mul, hk.2, mul_one]⟩
          rw [mem_O3_iff, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul, hk.1]
      rw [hset]
      have hSO3 : IsPreconnected SO3 := by
        have : IsPathConnected SO3 := ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem,
          fun {y} hy => rotation_path y hy⟩
        exact this.isConnected.isPreconnected
      refine (hSO3.image _ (continuous_const.matrix_mul continuous_id).continuousOn).subset_connectedComponentIn
        ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem, Matrix.mul_one g⟩ ?_
      rintro _ ⟨k, hk, rfl⟩
      rw [mem_SO3_iff] at hk
      rw [mem_O3_iff, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul]
      exact hk.1
  exact @key (1) ((Matrix.orthogonalGroup (Fin 3) ℝ).one_mem)

example : veriphy_instance% KaluzaKlein.Sets.Connected.O3_components (!![1, 0, 0; 0, 1, 0; 0, 0, -1]) ((by rw [KaluzaKlein.Sets.Connected.mem_O3_iff]; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ])) := by
  have key : ∀ (g : Matrix (Fin 3) (Fin 3) ℝ) (hg : g ∈ O3), connectedComponentIn O3 g = {h ∈ O3 | h.det = g.det} := by
    intro g hg
    have hdet : ∀ h ∈ O3, h.det = 1 ∨ h.det = -1 := by
      intro h hh
      rw [mem_O3_iff] at hh
      have := congrArg Matrix.det hh
      rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
      rcases mul_self_eq_one_iff.1 this with h1 | h1
      · left; exact h1
      · right; exact h1
    have hg' := hg
    rw [mem_O3_iff] at hg'
    have hggT : g * gᵀ = 1 := mul_eq_one_comm.1 hg'
    apply Subset.antisymm
    · -- the determinant is constant on the (preconnected) component
      intro h hh
      refine ⟨connectedComponentIn_subset _ _ hh, ?_⟩
      by_contra hne
      have hpc := (isPreconnected_connectedComponentIn (F := O3) (x := g)).image Matrix.det
        continuous_id.matrix_det.continuousOn
      have hg1 := hdet g hg
      have hh1 := hdet h (connectedComponentIn_subset _ _ hh)
      have h0 : (0 : ℝ) ∈ Matrix.det '' connectedComponentIn O3 g := by
        have hgmem : g.det ∈ Matrix.det '' connectedComponentIn O3 g :=
          ⟨g, mem_connectedComponentIn hg, rfl⟩
        have hhmem : h.det ∈ Matrix.det '' connectedComponentIn O3 g := ⟨h, hh, rfl⟩
        rcases hg1 with hg1 | hg1 <;> rcases hh1 with hh1 | hh1
        · exact absurd (hh1.trans hg1.symm) hne
        · rw [hg1] at hgmem; rw [hh1] at hhmem
          exact hpc.Icc_subset hhmem hgmem ⟨by norm_num, by norm_num⟩
        · rw [hg1] at hgmem; rw [hh1] at hhmem
          exact hpc.Icc_subset hgmem hhmem ⟨by norm_num, by norm_num⟩
        · exact absurd (hh1.trans hg1.symm) hne
      obtain ⟨k, hk, hk0⟩ := h0
      rcases hdet k (connectedComponentIn_subset _ _ hk) with h1 | h1 <;> linarith
    · -- the set is `g · SO(3)`, connected, containing `g`
      have hset : {h ∈ O3 | h.det = g.det} = (g * ·) '' SO3 := by
        ext h
        simp only [mem_setOf_eq, mem_image]
        constructor
        · rintro ⟨hh, hd⟩
          refine ⟨gᵀ * h, ?_, by rw [← Matrix.mul_assoc, hggT, Matrix.one_mul]⟩
          rw [mem_SO3_iff]
          rw [mem_O3_iff] at hh
          refine ⟨?_, ?_⟩
          · rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc g, hggT,
              Matrix.one_mul, hh]
          · rw [Matrix.det_mul, Matrix.det_transpose, hd]
            have := congrArg Matrix.det hg'
            rwa [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
        · rintro ⟨k, hk, rfl⟩
          rw [mem_SO3_iff] at hk
          refine ⟨?_, by rw [Matrix.det_mul, hk.2, mul_one]⟩
          rw [mem_O3_iff, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul, hk.1]
      rw [hset]
      have hSO3 : IsPreconnected SO3 := by
        have : IsPathConnected SO3 := ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem,
          fun {y} hy => rotation_path y hy⟩
        exact this.isConnected.isPreconnected
      refine (hSO3.image _ (continuous_const.matrix_mul continuous_id).continuousOn).subset_connectedComponentIn
        ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem, Matrix.mul_one g⟩ ?_
      rintro _ ⟨k, hk, rfl⟩
      rw [mem_SO3_iff] at hk
      rw [mem_O3_iff, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul]
      exact hk.1
  exact @key (!![1, 0, 0; 0, 1, 0; 0, 0, -1]) ((by rw [KaluzaKlein.Sets.Connected.mem_O3_iff]; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]))

example : veriphy_instance% KaluzaKlein.Sets.Connected.O3_components (!![0, -1, 0; 1, 0, 0; 0, 0, 1]) ((by rw [KaluzaKlein.Sets.Connected.mem_O3_iff]; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ])) := by
  have key : ∀ (g : Matrix (Fin 3) (Fin 3) ℝ) (hg : g ∈ O3), connectedComponentIn O3 g = {h ∈ O3 | h.det = g.det} := by
    intro g hg
    have hdet : ∀ h ∈ O3, h.det = 1 ∨ h.det = -1 := by
      intro h hh
      rw [mem_O3_iff] at hh
      have := congrArg Matrix.det hh
      rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
      rcases mul_self_eq_one_iff.1 this with h1 | h1
      · left; exact h1
      · right; exact h1
    have hg' := hg
    rw [mem_O3_iff] at hg'
    have hggT : g * gᵀ = 1 := mul_eq_one_comm.1 hg'
    apply Subset.antisymm
    · -- the determinant is constant on the (preconnected) component
      intro h hh
      refine ⟨connectedComponentIn_subset _ _ hh, ?_⟩
      by_contra hne
      have hpc := (isPreconnected_connectedComponentIn (F := O3) (x := g)).image Matrix.det
        continuous_id.matrix_det.continuousOn
      have hg1 := hdet g hg
      have hh1 := hdet h (connectedComponentIn_subset _ _ hh)
      have h0 : (0 : ℝ) ∈ Matrix.det '' connectedComponentIn O3 g := by
        have hgmem : g.det ∈ Matrix.det '' connectedComponentIn O3 g :=
          ⟨g, mem_connectedComponentIn hg, rfl⟩
        have hhmem : h.det ∈ Matrix.det '' connectedComponentIn O3 g := ⟨h, hh, rfl⟩
        rcases hg1 with hg1 | hg1 <;> rcases hh1 with hh1 | hh1
        · exact absurd (hh1.trans hg1.symm) hne
        · rw [hg1] at hgmem; rw [hh1] at hhmem
          exact hpc.Icc_subset hhmem hgmem ⟨by norm_num, by norm_num⟩
        · rw [hg1] at hgmem; rw [hh1] at hhmem
          exact hpc.Icc_subset hgmem hhmem ⟨by norm_num, by norm_num⟩
        · exact absurd (hh1.trans hg1.symm) hne
      obtain ⟨k, hk, hk0⟩ := h0
      rcases hdet k (connectedComponentIn_subset _ _ hk) with h1 | h1 <;> linarith
    · -- the set is `g · SO(3)`, connected, containing `g`
      have hset : {h ∈ O3 | h.det = g.det} = (g * ·) '' SO3 := by
        ext h
        simp only [mem_setOf_eq, mem_image]
        constructor
        · rintro ⟨hh, hd⟩
          refine ⟨gᵀ * h, ?_, by rw [← Matrix.mul_assoc, hggT, Matrix.one_mul]⟩
          rw [mem_SO3_iff]
          rw [mem_O3_iff] at hh
          refine ⟨?_, ?_⟩
          · rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc g, hggT,
              Matrix.one_mul, hh]
          · rw [Matrix.det_mul, Matrix.det_transpose, hd]
            have := congrArg Matrix.det hg'
            rwa [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
        · rintro ⟨k, hk, rfl⟩
          rw [mem_SO3_iff] at hk
          refine ⟨?_, by rw [Matrix.det_mul, hk.2, mul_one]⟩
          rw [mem_O3_iff, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul, hk.1]
      rw [hset]
      have hSO3 : IsPreconnected SO3 := by
        have : IsPathConnected SO3 := ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem,
          fun {y} hy => rotation_path y hy⟩
        exact this.isConnected.isPreconnected
      refine (hSO3.image _ (continuous_const.matrix_mul continuous_id).continuousOn).subset_connectedComponentIn
        ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem, Matrix.mul_one g⟩ ?_
      rintro _ ⟨k, hk, rfl⟩
      rw [mem_SO3_iff] at hk
      rw [mem_O3_iff, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul]
      exact hk.1
  exact @key (!![0, -1, 0; 1, 0, 0; 0, 0, 1]) ((by rw [KaluzaKlein.Sets.Connected.mem_O3_iff]; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.O3_components
-- test: set-o3-components-test.sage

/-! ## exm-connected-finite and exr-connected-components -/

/-- The graph on `{0, …, n - 1}` with adjacency `e a b ∨ e b a` (`a ≠ b`). -/
def graphOf {n : ℕ} (e : Fin n → Fin n → Bool) : SimpleGraph (Fin n) := SimpleGraph.fromRel (fun a b => e a b = true)

instance {n : ℕ} (e : Fin n → Fin n → Bool) : DecidableRel (graphOf e).Adj := fun a b => by
  unfold graphOf; rw [SimpleGraph.fromRel_adj]; infer_instance

/-- The graph with edges `AB`, `BC`, `DE` (`A, …, E = 0, …, 4`). -/
def graphABCDE : SimpleGraph (Fin 5) := graphOf (fun a b => decide ((a, b) ∈ [((0 : Fin 5), (1 : Fin 5)), (1, 2), (3, 4)]))

/-- The six-cycle with vertex `5` removed (its edges deleted). -/
def cycleMinusOne : SimpleGraph (Fin 6) := graphOf (fun a b => decide (b = a + 1 ∧ a ≠ 5 ∧ b ≠ 5))

/-- The six-cycle with the opposite vertices `0` and `3` removed (their edges deleted). -/
def cycleMinusTwo : SimpleGraph (Fin 6) :=
  graphOf (fun a b => decide (b = a + 1 ∧ a ≠ 0 ∧ b ≠ 0 ∧ a ≠ 3 ∧ b ≠ 3))

instance : DecidableRel graphABCDE.Adj := inferInstanceAs (DecidableRel (graphOf _).Adj)
instance : DecidableRel cycleMinusOne.Adj := inferInstanceAs (DecidableRel (graphOf _).Adj)
instance : DecidableRel cycleMinusTwo.Adj := inferInstanceAs (DecidableRel (graphOf _).Adj)

/-- The graph `AB, BC, DE` has the two components `{A, B, C}` and `{D, E}`. -/
theorem graph_components (u v : Fin 5) : graphABCDE.Reachable u v ↔ (u.val < 3 ↔ v.val < 3) := by
  revert u v
  decide

example : veriphy_instance% KaluzaKlein.Sets.Connected.graph_components (0) (2) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Connected.graph_components (0) (3) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Connected.graph_components (3) (4) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Connected.graph_components (1) (1) := by
  decide

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.Sets.Connected.graph_components
-- test: set-graph-components-test.sage

/-- The six-cycle minus one vertex stays connected: any two remaining vertices are joined. -/
theorem cycle_minus_one (u v : Fin 6) (hu : u ≠ 5) (hv : v ≠ 5) : cycleMinusOne.Reachable u v := by
  revert u v
  decide

example : veriphy_instance% KaluzaKlein.Sets.Connected.cycle_minus_one (0) (4) ((by decide)) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Connected.cycle_minus_one (2) (2) ((by decide)) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Connected.cycle_minus_one (1) (3) ((by decide)) ((by decide)) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.cycle_minus_one
-- test: set-cycle-minus-one-test.sage

/-- The six-cycle minus two opposite vertices splits into the two components `{1, 2}` and `{4, 5}`. -/
theorem cycle_minus_two (u v : Fin 6) (hu : u ≠ 0 ∧ u ≠ 3) (hv : v ≠ 0 ∧ v ≠ 3) :
    cycleMinusTwo.Reachable u v ↔ (u.val < 3 ↔ v.val < 3) := by
  revert u v
  decide

example : veriphy_instance% KaluzaKlein.Sets.Connected.cycle_minus_two (1) (2) ((by decide)) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Connected.cycle_minus_two (1) (4) ((by decide)) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Connected.cycle_minus_two (5) (5) ((by decide)) ((by decide)) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.cycle_minus_two
-- test: set-cycle-minus-two-test.sage

/-! ## exm-path-graph -/

/-- A graph is connected exactly when it has a vertex and every pair of vertices is joined by a walk. -/
theorem graph_connected_iff (n : ℕ) (G : SimpleGraph (Fin n)) : G.Connected ↔ Nonempty (Fin n) ∧ ∀ u v, G.Reachable u v :=
  ⟨fun h => ⟨h.nonempty, h.preconnected⟩, fun ⟨h1, h2⟩ => ⟨h2⟩⟩

example : veriphy_instance% KaluzaKlein.Sets.Connected.graph_connected_iff (0) (⊥) := by
  have key : ∀ (n : ℕ) (G : SimpleGraph (Fin n)), G.Connected ↔ Nonempty (Fin n) ∧ ∀ u v, G.Reachable u v := by
    intro n G
    exact (⟨fun h => ⟨h.nonempty, h.preconnected⟩, fun ⟨h1, h2⟩ => ⟨h2⟩⟩)
  exact @key (0) (⊥)

example : veriphy_instance% KaluzaKlein.Sets.Connected.graph_connected_iff (3) (⊤) := by
  have key : ∀ (n : ℕ) (G : SimpleGraph (Fin n)), G.Connected ↔ Nonempty (Fin n) ∧ ∀ u v, G.Reachable u v := by
    intro n G
    exact (⟨fun h => ⟨h.nonempty, h.preconnected⟩, fun ⟨h1, h2⟩ => ⟨h2⟩⟩)
  exact @key (3) (⊤)

example : veriphy_instance% KaluzaKlein.Sets.Connected.graph_connected_iff (5) (KaluzaKlein.Sets.Connected.graphABCDE) := by
  have key : ∀ (n : ℕ) (G : SimpleGraph (Fin n)), G.Connected ↔ Nonempty (Fin n) ∧ ∀ u v, G.Reachable u v := by
    intro n G
    exact (⟨fun h => ⟨h.nonempty, h.preconnected⟩, fun ⟨h1, h2⟩ => ⟨h2⟩⟩)
  exact @key (5) (KaluzaKlein.Sets.Connected.graphABCDE)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.graph_connected_iff
-- test: set-graph-connected-test.sage

/-! ## exr-connected-gl2 -/

/-- The invertible `n × n` matrices (`n ≥ 1`) are not connected: `{det > 0}` and `{det < 0}` are disjoint non-empty
open sets covering them (`GL(2, ℝ)` for `n = 2`). -/
theorem GL_not_connected (n : ℕ) (hn : 1 ≤ n) : ¬ IsPreconnected {A : Matrix (Fin n) (Fin n) ℝ | A.det ≠ 0} := by
  intro h
  have hc : Continuous (fun A : Matrix (Fin n) (Fin n) ℝ => A.det) := continuous_id.matrix_det
  set D : Matrix (Fin n) (Fin n) ℝ := Matrix.diagonal (fun i => if i.val = 0 then -1 else 1)
  have hD : D.det = -1 := by
    rw [Matrix.det_diagonal]
    rw [Finset.prod_eq_single (⟨0, hn⟩ : Fin n)]
    · simp
    · intro b _ hb
      have : b.val ≠ 0 := fun h' => hb (Fin.ext h')
      simp [this]
    · simp
  obtain ⟨A, -, hA1, hA2⟩ := h {A | 0 < A.det} {A | A.det < 0} (isOpen_lt continuous_const hc)
    (isOpen_lt hc continuous_const)
    (fun A hA => by
      rcases lt_or_gt_of_ne hA with h' | h'
      · right; exact h'
      · left; exact h')
    ⟨1, by simp, by simp⟩ ⟨D, by simp [hD], by simp [hD]⟩
  simp only [mem_setOf_eq] at hA1 hA2
  linarith

example : veriphy_instance% KaluzaKlein.Sets.Connected.GL_not_connected (1) ((by norm_num)) := by
  have key : ∀ (n : ℕ) (hn : 1 ≤ n), ¬ IsPreconnected {A : Matrix (Fin n) (Fin n) ℝ | A.det ≠ 0} := by
    intro n hn
    intro h
    have hc : Continuous (fun A : Matrix (Fin n) (Fin n) ℝ => A.det) := continuous_id.matrix_det
    set D : Matrix (Fin n) (Fin n) ℝ := Matrix.diagonal (fun i => if i.val = 0 then -1 else 1)
    have hD : D.det = -1 := by
      rw [Matrix.det_diagonal]
      rw [Finset.prod_eq_single (⟨0, hn⟩ : Fin n)]
      · simp
      · intro b _ hb
        have : b.val ≠ 0 := fun h' => hb (Fin.ext h')
        simp [this]
      · simp
    obtain ⟨A, -, hA1, hA2⟩ := h {A | 0 < A.det} {A | A.det < 0} (isOpen_lt continuous_const hc)
      (isOpen_lt hc continuous_const)
      (fun A hA => by
        rcases lt_or_gt_of_ne hA with h' | h'
        · right; exact h'
        · left; exact h')
      ⟨1, by simp, by simp⟩ ⟨D, by simp [hD], by simp [hD]⟩
    simp only [mem_setOf_eq] at hA1 hA2
    linarith
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Connected.GL_not_connected (2) ((by norm_num)) := by
  have key : ∀ (n : ℕ) (hn : 1 ≤ n), ¬ IsPreconnected {A : Matrix (Fin n) (Fin n) ℝ | A.det ≠ 0} := by
    intro n hn
    intro h
    have hc : Continuous (fun A : Matrix (Fin n) (Fin n) ℝ => A.det) := continuous_id.matrix_det
    set D : Matrix (Fin n) (Fin n) ℝ := Matrix.diagonal (fun i => if i.val = 0 then -1 else 1)
    have hD : D.det = -1 := by
      rw [Matrix.det_diagonal]
      rw [Finset.prod_eq_single (⟨0, hn⟩ : Fin n)]
      · simp
      · intro b _ hb
        have : b.val ≠ 0 := fun h' => hb (Fin.ext h')
        simp [this]
      · simp
    obtain ⟨A, -, hA1, hA2⟩ := h {A | 0 < A.det} {A | A.det < 0} (isOpen_lt continuous_const hc)
      (isOpen_lt hc continuous_const)
      (fun A hA => by
        rcases lt_or_gt_of_ne hA with h' | h'
        · right; exact h'
        · left; exact h')
      ⟨1, by simp, by simp⟩ ⟨D, by simp [hD], by simp [hD]⟩
    simp only [mem_setOf_eq] at hA1 hA2
    linarith
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Connected.GL_not_connected (3) ((by norm_num)) := by
  have key : ∀ (n : ℕ) (hn : 1 ≤ n), ¬ IsPreconnected {A : Matrix (Fin n) (Fin n) ℝ | A.det ≠ 0} := by
    intro n hn
    intro h
    have hc : Continuous (fun A : Matrix (Fin n) (Fin n) ℝ => A.det) := continuous_id.matrix_det
    set D : Matrix (Fin n) (Fin n) ℝ := Matrix.diagonal (fun i => if i.val = 0 then -1 else 1)
    have hD : D.det = -1 := by
      rw [Matrix.det_diagonal]
      rw [Finset.prod_eq_single (⟨0, hn⟩ : Fin n)]
      · simp
      · intro b _ hb
        have : b.val ≠ 0 := fun h' => hb (Fin.ext h')
        simp [this]
      · simp
    obtain ⟨A, -, hA1, hA2⟩ := h {A | 0 < A.det} {A | A.det < 0} (isOpen_lt continuous_const hc)
      (isOpen_lt hc continuous_const)
      (fun A hA => by
        rcases lt_or_gt_of_ne hA with h' | h'
        · right; exact h'
        · left; exact h')
      ⟨1, by simp, by simp⟩ ⟨D, by simp [hD], by simp [hD]⟩
    simp only [mem_setOf_eq] at hA1 hA2
    linarith
  exact @key (3) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.GL_not_connected
-- test: set-gl-test.sage

/-- The orthogonal group `O(2)`. -/
abbrev O2 : Set (Matrix (Fin 2) (Fin 2) ℝ) := (Matrix.orthogonalGroup (Fin 2) ℝ : Set _)

/-- The plane rotation by `α`. -/
noncomputable def rot2 (α : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![Real.cos α, -Real.sin α; Real.sin α, Real.cos α]

/-- `SO(2)` is the circle of rotations `rot2 α`, a connected set. -/
theorem SO2_eq_range : SO2 = range rot2 ∧ IsConnected SO2 := by
  have heq : SO2 = range rot2 := by
    ext g
    constructor
    · intro hg
      have hcol : (g 0 0, g 1 0) ∈ circle := by
        have h := SO2_orbit_bijective
        rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff'] at hg
        have e00 := congrFun (congrFun hg.1 0) 0
        simp [Matrix.mul_apply, Fin.sum_univ_succ] at e00
        simp only [circle, mem_setOf_eq]; nlinarith
      have hc' : (g 0 0, g 1 0) ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hcol
      obtain ⟨φ, hφ⟩ := hc'
      simp only [Prod.mk.injEq, one_mul] at hφ
      refine ⟨φ, ?_⟩
      rw [SO2_eq_of_column g hg]
      ext i j; fin_cases i <;> fin_cases j <;> simp [rot2, hφ.1, hφ.2]
    · rintro ⟨α, rfl⟩
      rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
      refine ⟨?_, ?_⟩
      · ext i j; fin_cases i <;> fin_cases j <;> simp [rot2, Matrix.mul_apply, Fin.sum_univ_succ] <;>
          nlinarith [Real.sin_sq_add_cos_sq α]
      · rw [rot2, Matrix.det_fin_two_of]; nlinarith [Real.sin_sq_add_cos_sq α]
  refine ⟨heq, heq ▸ isConnected_range ?_⟩
  refine continuous_pi fun i => continuous_pi fun j => ?_
  fin_cases i <;> fin_cases j <;> simp [rot2] <;> fun_prop

/-- `O(2)` has exactly two components, the rotations and the reflections, each a circle: the component of `g` is the
set of orthogonal matrices with the determinant of `g`. -/
theorem O2_components (g : Matrix (Fin 2) (Fin 2) ℝ) (hg : g ∈ O2) :
    connectedComponentIn O2 g = {h ∈ O2 | h.det = g.det} := by
  have hmem : ∀ h : Matrix (Fin 2) (Fin 2) ℝ, h ∈ O2 ↔ hᵀ * h = 1 := fun h => by
    rw [SetLike.mem_coe, Matrix.mem_orthogonalGroup_iff']
  have hmemSO : ∀ h : Matrix (Fin 2) (Fin 2) ℝ, h ∈ SO2 ↔ hᵀ * h = 1 ∧ h.det = 1 := fun h => by
    rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
  have hdet : ∀ h ∈ O2, h.det = 1 ∨ h.det = -1 := by
    intro h hh
    rw [hmem] at hh
    have := congrArg Matrix.det hh
    rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
    rcases mul_self_eq_one_iff.1 this with h1 | h1
    · left; exact h1
    · right; exact h1
  have hg' := (hmem g).1 hg
  have hggT : g * gᵀ = 1 := mul_eq_one_comm.1 hg'
  apply Subset.antisymm
  · intro h hh
    refine ⟨connectedComponentIn_subset _ _ hh, ?_⟩
    by_contra hne
    have hpc := (isPreconnected_connectedComponentIn (F := O2) (x := g)).image Matrix.det
      continuous_id.matrix_det.continuousOn
    have h0 : (0 : ℝ) ∈ Matrix.det '' connectedComponentIn O2 g := by
      have hgmem : g.det ∈ Matrix.det '' connectedComponentIn O2 g := ⟨g, mem_connectedComponentIn hg, rfl⟩
      have hhmem : h.det ∈ Matrix.det '' connectedComponentIn O2 g := ⟨h, hh, rfl⟩
      rcases hdet g hg with hg1 | hg1 <;> rcases hdet h (connectedComponentIn_subset _ _ hh) with hh1 | hh1
      · exact absurd (hh1.trans hg1.symm) hne
      · rw [hg1] at hgmem; rw [hh1] at hhmem
        exact hpc.Icc_subset hhmem hgmem ⟨by norm_num, by norm_num⟩
      · rw [hg1] at hgmem; rw [hh1] at hhmem
        exact hpc.Icc_subset hgmem hhmem ⟨by norm_num, by norm_num⟩
      · exact absurd (hh1.trans hg1.symm) hne
    obtain ⟨k, hk, hk0⟩ := h0
    rcases hdet k (connectedComponentIn_subset _ _ hk) with h1 | h1 <;> linarith
  · have hset : {h ∈ O2 | h.det = g.det} = (g * ·) '' SO2 := by
      ext h
      simp only [mem_setOf_eq, mem_image]
      constructor
      · rintro ⟨hh, hd⟩
        refine ⟨gᵀ * h, ?_, by rw [← Matrix.mul_assoc, hggT, Matrix.one_mul]⟩
        rw [hmemSO]
        rw [hmem] at hh
        refine ⟨?_, ?_⟩
        · rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc g, hggT,
            Matrix.one_mul, hh]
        · rw [Matrix.det_mul, Matrix.det_transpose, hd]
          have := congrArg Matrix.det hg'
          rwa [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
      · rintro ⟨k, hk, rfl⟩
        rw [hmemSO] at hk
        refine ⟨?_, by rw [Matrix.det_mul, hk.2, mul_one]⟩
        rw [hmem, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul, hk.1]
    rw [hset]
    refine (SO2_eq_range.2.isPreconnected.image _
      (continuous_const.matrix_mul continuous_id).continuousOn).subset_connectedComponentIn
      ⟨1, (Matrix.specialOrthogonalGroup (Fin 2) ℝ).one_mem, Matrix.mul_one g⟩ ?_
    rintro _ ⟨k, hk, rfl⟩
    rw [hmemSO] at hk
    rw [hmem, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul]
    exact hk.1

example : veriphy_instance% KaluzaKlein.Sets.Connected.O2_components (1) ((Matrix.orthogonalGroup (Fin 2) ℝ).one_mem) := by
  have key : ∀ (g : Matrix (Fin 2) (Fin 2) ℝ) (hg : g ∈ O2), connectedComponentIn O2 g = {h ∈ O2 | h.det = g.det} := by
    intro g hg
    have hmem : ∀ h : Matrix (Fin 2) (Fin 2) ℝ, h ∈ O2 ↔ hᵀ * h = 1 := fun h => by
      rw [SetLike.mem_coe, Matrix.mem_orthogonalGroup_iff']
    have hmemSO : ∀ h : Matrix (Fin 2) (Fin 2) ℝ, h ∈ SO2 ↔ hᵀ * h = 1 ∧ h.det = 1 := fun h => by
      rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
    have hdet : ∀ h ∈ O2, h.det = 1 ∨ h.det = -1 := by
      intro h hh
      rw [hmem] at hh
      have := congrArg Matrix.det hh
      rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
      rcases mul_self_eq_one_iff.1 this with h1 | h1
      · left; exact h1
      · right; exact h1
    have hg' := (hmem g).1 hg
    have hggT : g * gᵀ = 1 := mul_eq_one_comm.1 hg'
    apply Subset.antisymm
    · intro h hh
      refine ⟨connectedComponentIn_subset _ _ hh, ?_⟩
      by_contra hne
      have hpc := (isPreconnected_connectedComponentIn (F := O2) (x := g)).image Matrix.det
        continuous_id.matrix_det.continuousOn
      have h0 : (0 : ℝ) ∈ Matrix.det '' connectedComponentIn O2 g := by
        have hgmem : g.det ∈ Matrix.det '' connectedComponentIn O2 g := ⟨g, mem_connectedComponentIn hg, rfl⟩
        have hhmem : h.det ∈ Matrix.det '' connectedComponentIn O2 g := ⟨h, hh, rfl⟩
        rcases hdet g hg with hg1 | hg1 <;> rcases hdet h (connectedComponentIn_subset _ _ hh) with hh1 | hh1
        · exact absurd (hh1.trans hg1.symm) hne
        · rw [hg1] at hgmem; rw [hh1] at hhmem
          exact hpc.Icc_subset hhmem hgmem ⟨by norm_num, by norm_num⟩
        · rw [hg1] at hgmem; rw [hh1] at hhmem
          exact hpc.Icc_subset hgmem hhmem ⟨by norm_num, by norm_num⟩
        · exact absurd (hh1.trans hg1.symm) hne
      obtain ⟨k, hk, hk0⟩ := h0
      rcases hdet k (connectedComponentIn_subset _ _ hk) with h1 | h1 <;> linarith
    · have hset : {h ∈ O2 | h.det = g.det} = (g * ·) '' SO2 := by
        ext h
        simp only [mem_setOf_eq, mem_image]
        constructor
        · rintro ⟨hh, hd⟩
          refine ⟨gᵀ * h, ?_, by rw [← Matrix.mul_assoc, hggT, Matrix.one_mul]⟩
          rw [hmemSO]
          rw [hmem] at hh
          refine ⟨?_, ?_⟩
          · rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc g, hggT,
              Matrix.one_mul, hh]
          · rw [Matrix.det_mul, Matrix.det_transpose, hd]
            have := congrArg Matrix.det hg'
            rwa [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
        · rintro ⟨k, hk, rfl⟩
          rw [hmemSO] at hk
          refine ⟨?_, by rw [Matrix.det_mul, hk.2, mul_one]⟩
          rw [hmem, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul, hk.1]
      rw [hset]
      refine (SO2_eq_range.2.isPreconnected.image _
        (continuous_const.matrix_mul continuous_id).continuousOn).subset_connectedComponentIn
        ⟨1, (Matrix.specialOrthogonalGroup (Fin 2) ℝ).one_mem, Matrix.mul_one g⟩ ?_
      rintro _ ⟨k, hk, rfl⟩
      rw [hmemSO] at hk
      rw [hmem, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul]
      exact hk.1
  exact @key (1) ((Matrix.orthogonalGroup (Fin 2) ℝ).one_mem)

example : veriphy_instance% KaluzaKlein.Sets.Connected.O2_components (!![1, 0; 0, -1]) ((by rw [SetLike.mem_coe, Matrix.mem_orthogonalGroup_iff']; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ])) := by
  have key : ∀ (g : Matrix (Fin 2) (Fin 2) ℝ) (hg : g ∈ O2), connectedComponentIn O2 g = {h ∈ O2 | h.det = g.det} := by
    intro g hg
    have hmem : ∀ h : Matrix (Fin 2) (Fin 2) ℝ, h ∈ O2 ↔ hᵀ * h = 1 := fun h => by
      rw [SetLike.mem_coe, Matrix.mem_orthogonalGroup_iff']
    have hmemSO : ∀ h : Matrix (Fin 2) (Fin 2) ℝ, h ∈ SO2 ↔ hᵀ * h = 1 ∧ h.det = 1 := fun h => by
      rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
    have hdet : ∀ h ∈ O2, h.det = 1 ∨ h.det = -1 := by
      intro h hh
      rw [hmem] at hh
      have := congrArg Matrix.det hh
      rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
      rcases mul_self_eq_one_iff.1 this with h1 | h1
      · left; exact h1
      · right; exact h1
    have hg' := (hmem g).1 hg
    have hggT : g * gᵀ = 1 := mul_eq_one_comm.1 hg'
    apply Subset.antisymm
    · intro h hh
      refine ⟨connectedComponentIn_subset _ _ hh, ?_⟩
      by_contra hne
      have hpc := (isPreconnected_connectedComponentIn (F := O2) (x := g)).image Matrix.det
        continuous_id.matrix_det.continuousOn
      have h0 : (0 : ℝ) ∈ Matrix.det '' connectedComponentIn O2 g := by
        have hgmem : g.det ∈ Matrix.det '' connectedComponentIn O2 g := ⟨g, mem_connectedComponentIn hg, rfl⟩
        have hhmem : h.det ∈ Matrix.det '' connectedComponentIn O2 g := ⟨h, hh, rfl⟩
        rcases hdet g hg with hg1 | hg1 <;> rcases hdet h (connectedComponentIn_subset _ _ hh) with hh1 | hh1
        · exact absurd (hh1.trans hg1.symm) hne
        · rw [hg1] at hgmem; rw [hh1] at hhmem
          exact hpc.Icc_subset hhmem hgmem ⟨by norm_num, by norm_num⟩
        · rw [hg1] at hgmem; rw [hh1] at hhmem
          exact hpc.Icc_subset hgmem hhmem ⟨by norm_num, by norm_num⟩
        · exact absurd (hh1.trans hg1.symm) hne
      obtain ⟨k, hk, hk0⟩ := h0
      rcases hdet k (connectedComponentIn_subset _ _ hk) with h1 | h1 <;> linarith
    · have hset : {h ∈ O2 | h.det = g.det} = (g * ·) '' SO2 := by
        ext h
        simp only [mem_setOf_eq, mem_image]
        constructor
        · rintro ⟨hh, hd⟩
          refine ⟨gᵀ * h, ?_, by rw [← Matrix.mul_assoc, hggT, Matrix.one_mul]⟩
          rw [hmemSO]
          rw [hmem] at hh
          refine ⟨?_, ?_⟩
          · rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc g, hggT,
              Matrix.one_mul, hh]
          · rw [Matrix.det_mul, Matrix.det_transpose, hd]
            have := congrArg Matrix.det hg'
            rwa [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
        · rintro ⟨k, hk, rfl⟩
          rw [hmemSO] at hk
          refine ⟨?_, by rw [Matrix.det_mul, hk.2, mul_one]⟩
          rw [hmem, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul, hk.1]
      rw [hset]
      refine (SO2_eq_range.2.isPreconnected.image _
        (continuous_const.matrix_mul continuous_id).continuousOn).subset_connectedComponentIn
        ⟨1, (Matrix.specialOrthogonalGroup (Fin 2) ℝ).one_mem, Matrix.mul_one g⟩ ?_
      rintro _ ⟨k, hk, rfl⟩
      rw [hmemSO] at hk
      rw [hmem, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul]
      exact hk.1
  exact @key (!![1, 0; 0, -1]) ((by rw [SetLike.mem_coe, Matrix.mem_orthogonalGroup_iff']; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]))

example : veriphy_instance% KaluzaKlein.Sets.Connected.O2_components (!![0, -1; 1, 0]) ((by rw [SetLike.mem_coe, Matrix.mem_orthogonalGroup_iff']; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ])) := by
  have key : ∀ (g : Matrix (Fin 2) (Fin 2) ℝ) (hg : g ∈ O2), connectedComponentIn O2 g = {h ∈ O2 | h.det = g.det} := by
    intro g hg
    have hmem : ∀ h : Matrix (Fin 2) (Fin 2) ℝ, h ∈ O2 ↔ hᵀ * h = 1 := fun h => by
      rw [SetLike.mem_coe, Matrix.mem_orthogonalGroup_iff']
    have hmemSO : ∀ h : Matrix (Fin 2) (Fin 2) ℝ, h ∈ SO2 ↔ hᵀ * h = 1 ∧ h.det = 1 := fun h => by
      rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
    have hdet : ∀ h ∈ O2, h.det = 1 ∨ h.det = -1 := by
      intro h hh
      rw [hmem] at hh
      have := congrArg Matrix.det hh
      rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
      rcases mul_self_eq_one_iff.1 this with h1 | h1
      · left; exact h1
      · right; exact h1
    have hg' := (hmem g).1 hg
    have hggT : g * gᵀ = 1 := mul_eq_one_comm.1 hg'
    apply Subset.antisymm
    · intro h hh
      refine ⟨connectedComponentIn_subset _ _ hh, ?_⟩
      by_contra hne
      have hpc := (isPreconnected_connectedComponentIn (F := O2) (x := g)).image Matrix.det
        continuous_id.matrix_det.continuousOn
      have h0 : (0 : ℝ) ∈ Matrix.det '' connectedComponentIn O2 g := by
        have hgmem : g.det ∈ Matrix.det '' connectedComponentIn O2 g := ⟨g, mem_connectedComponentIn hg, rfl⟩
        have hhmem : h.det ∈ Matrix.det '' connectedComponentIn O2 g := ⟨h, hh, rfl⟩
        rcases hdet g hg with hg1 | hg1 <;> rcases hdet h (connectedComponentIn_subset _ _ hh) with hh1 | hh1
        · exact absurd (hh1.trans hg1.symm) hne
        · rw [hg1] at hgmem; rw [hh1] at hhmem
          exact hpc.Icc_subset hhmem hgmem ⟨by norm_num, by norm_num⟩
        · rw [hg1] at hgmem; rw [hh1] at hhmem
          exact hpc.Icc_subset hgmem hhmem ⟨by norm_num, by norm_num⟩
        · exact absurd (hh1.trans hg1.symm) hne
      obtain ⟨k, hk, hk0⟩ := h0
      rcases hdet k (connectedComponentIn_subset _ _ hk) with h1 | h1 <;> linarith
    · have hset : {h ∈ O2 | h.det = g.det} = (g * ·) '' SO2 := by
        ext h
        simp only [mem_setOf_eq, mem_image]
        constructor
        · rintro ⟨hh, hd⟩
          refine ⟨gᵀ * h, ?_, by rw [← Matrix.mul_assoc, hggT, Matrix.one_mul]⟩
          rw [hmemSO]
          rw [hmem] at hh
          refine ⟨?_, ?_⟩
          · rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc g, hggT,
              Matrix.one_mul, hh]
          · rw [Matrix.det_mul, Matrix.det_transpose, hd]
            have := congrArg Matrix.det hg'
            rwa [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
        · rintro ⟨k, hk, rfl⟩
          rw [hmemSO] at hk
          refine ⟨?_, by rw [Matrix.det_mul, hk.2, mul_one]⟩
          rw [hmem, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul, hk.1]
      rw [hset]
      refine (SO2_eq_range.2.isPreconnected.image _
        (continuous_const.matrix_mul continuous_id).continuousOn).subset_connectedComponentIn
        ⟨1, (Matrix.specialOrthogonalGroup (Fin 2) ℝ).one_mem, Matrix.mul_one g⟩ ?_
      rintro _ ⟨k, hk, rfl⟩
      rw [hmemSO] at hk
      rw [hmem, Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc gᵀ, hg', Matrix.one_mul]
      exact hk.1
  exact @key (!![0, -1; 1, 0]) ((by rw [SetLike.mem_coe, Matrix.mem_orthogonalGroup_iff']; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.O2_components
-- test: set-o2-components-test.sage

/-- A path of plane rotations `t ↦ rot2 (tα)` from the identity to `rot2 α`, with determinant `1` throughout. -/
theorem rot2_path (α t : ℝ) : (rot2 (t * α)).det = 1 ∧ rot2 (0 * α) = 1 ∧ rot2 (1 * α) = rot2 α := by
  refine ⟨?_, ?_, by rw [one_mul]⟩
  · rw [rot2, Matrix.det_fin_two_of]; nlinarith [Real.sin_sq_add_cos_sq (t * α)]
  · ext i j; fin_cases i <;> fin_cases j <;> simp [rot2]

example : veriphy_instance% KaluzaKlein.Sets.Connected.rot2_path (0) (0) := by
  have key : ∀ (α t : ℝ), (rot2 (t * α)).det = 1 ∧ rot2 (0 * α) = 1 ∧ rot2 (1 * α) = rot2 α := by
    intro α t
    refine ⟨?_, ?_, by rw [one_mul]⟩
    · rw [rot2, Matrix.det_fin_two_of]; nlinarith [Real.sin_sq_add_cos_sq (t * α)]
    · ext i j; fin_cases i <;> fin_cases j <;> simp [rot2]
  exact @key (0) (0)

example : veriphy_instance% KaluzaKlein.Sets.Connected.rot2_path (Real.pi) (1/2) := by
  have key : ∀ (α t : ℝ), (rot2 (t * α)).det = 1 ∧ rot2 (0 * α) = 1 ∧ rot2 (1 * α) = rot2 α := by
    intro α t
    refine ⟨?_, ?_, by rw [one_mul]⟩
    · rw [rot2, Matrix.det_fin_two_of]; nlinarith [Real.sin_sq_add_cos_sq (t * α)]
    · ext i j; fin_cases i <;> fin_cases j <;> simp [rot2]
  exact @key (Real.pi) (1/2)

example : veriphy_instance% KaluzaKlein.Sets.Connected.rot2_path (1) (2) := by
  have key : ∀ (α t : ℝ), (rot2 (t * α)).det = 1 ∧ rot2 (0 * α) = 1 ∧ rot2 (1 * α) = rot2 α := by
    intro α t
    refine ⟨?_, ?_, by rw [one_mul]⟩
    · rw [rot2, Matrix.det_fin_two_of]; nlinarith [Real.sin_sq_add_cos_sq (t * α)]
    · ext i j; fin_cases i <;> fin_cases j <;> simp [rot2]
  exact @key (1) (2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.rot2_path
-- test: set-rot2-path-test.sage

/-! ## exr-connected-image -/

/-- The image of a connected space under a continuous map is connected; for `f : [0, 1] → ℝ` continuous this gives the
intermediate value theorem; and the connected subsets of `ℝ` are the intervals. -/
theorem connected_image_ivt (f : ℝ → ℝ) (hf : ContinuousOn f (Icc 0 1)) (y : ℝ) (hy : y ∈ Icc (f 0) (f 1)) :
    IsConnected (f '' Icc 0 1) ∧ (∃ x ∈ Icc (0 : ℝ) 1, f x = y) ∧ (f '' Icc 0 1).OrdConnected := by
  have hc : IsConnected (f '' Icc 0 1) := (isConnected_Icc zero_le_one).image f hf
  exact ⟨hc, intermediate_value_Icc zero_le_one hf hy, hc.isPreconnected.ordConnected⟩

example : veriphy_instance% KaluzaKlein.Sets.Connected.connected_image_ivt (id) (continuous_id.continuousOn) (1/2) ((by simp; norm_num)) := by
  have key : ∀ (f : ℝ → ℝ) (hf : ContinuousOn f (Icc 0 1)) (y : ℝ) (hy : y ∈ Icc (f 0) (f 1)), IsConnected (f '' Icc 0 1) ∧ (∃ x ∈ Icc (0 : ℝ) 1, f x = y) ∧ (f '' Icc 0 1).OrdConnected := by
    intro f hf y hy
    have hc : IsConnected (f '' Icc 0 1) := (isConnected_Icc zero_le_one).image f hf
    exact ⟨hc, intermediate_value_Icc zero_le_one hf hy, hc.isPreconnected.ordConnected⟩
  exact @key (id) (continuous_id.continuousOn) (1/2) ((by simp; norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Connected.connected_image_ivt (fun _ => 0) (continuousOn_const) (0) ((by simp)) := by
  have key : ∀ (f : ℝ → ℝ) (hf : ContinuousOn f (Icc 0 1)) (y : ℝ) (hy : y ∈ Icc (f 0) (f 1)), IsConnected (f '' Icc 0 1) ∧ (∃ x ∈ Icc (0 : ℝ) 1, f x = y) ∧ (f '' Icc 0 1).OrdConnected := by
    intro f hf y hy
    have hc : IsConnected (f '' Icc 0 1) := (isConnected_Icc zero_le_one).image f hf
    exact ⟨hc, intermediate_value_Icc zero_le_one hf hy, hc.isPreconnected.ordConnected⟩
  exact @key (fun _ => 0) (continuousOn_const) (0) ((by simp))

example : veriphy_instance% KaluzaKlein.Sets.Connected.connected_image_ivt (fun x => x ^ 2) ((continuous_pow 2).continuousOn) (1/4) ((by norm_num)) := by
  have key : ∀ (f : ℝ → ℝ) (hf : ContinuousOn f (Icc 0 1)) (y : ℝ) (hy : y ∈ Icc (f 0) (f 1)), IsConnected (f '' Icc 0 1) ∧ (∃ x ∈ Icc (0 : ℝ) 1, f x = y) ∧ (f '' Icc 0 1).OrdConnected := by
    intro f hf y hy
    have hc : IsConnected (f '' Icc 0 1) := (isConnected_Icc zero_le_one).image f hf
    exact ⟨hc, intermediate_value_Icc zero_le_one hf hy, hc.isPreconnected.ordConnected⟩
  exact @key (fun x => x ^ 2) ((continuous_pow 2).continuousOn) (1/4) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.connected_image_ivt
-- test: set-ivt-test.sage

/-! ## def-path-connected and exr-path-connected-implies -/

/-- A path-connected set is connected. -/
theorem pathConnected_connected (X : Type) [TopologicalSpace X] (S : Set X) (h : IsPathConnected S) :
    IsConnected S :=
  h.isConnected

example : veriphy_instance% KaluzaKlein.Sets.Connected.pathConnected_connected (ℝ) (inferInstance) (Set.univ) (isPathConnected_univ) := by
  have key : ∀ (X : Type) [inst1 : TopologicalSpace X] (S : Set X) (h : IsPathConnected S), IsConnected S := by
    intro X inst1 S h
    exact (h.isConnected)
  exact @key (ℝ) (inferInstance) (Set.univ) (isPathConnected_univ)

example : veriphy_instance% KaluzaKlein.Sets.Connected.pathConnected_connected (ℝ) (inferInstance) (Set.Icc 0 1) (((convex_Icc 0 1).isPathConnected (Set.nonempty_Icc.2 zero_le_one))) := by
  have key : ∀ (X : Type) [inst1 : TopologicalSpace X] (S : Set X) (h : IsPathConnected S), IsConnected S := by
    intro X inst1 S h
    exact (h.isConnected)
  exact @key (ℝ) (inferInstance) (Set.Icc 0 1) (((convex_Icc 0 1).isPathConnected (Set.nonempty_Icc.2 zero_le_one)))

example : veriphy_instance% KaluzaKlein.Sets.Connected.pathConnected_connected (Fin 3 → ℝ) (inferInstance) (Set.univ) (isPathConnected_univ) := by
  have key : ∀ (X : Type) [inst1 : TopologicalSpace X] (S : Set X) (h : IsPathConnected S), IsConnected S := by
    intro X inst1 S h
    exact (h.isConnected)
  exact @key (Fin 3 → ℝ) (inferInstance) (Set.univ) (isPathConnected_univ)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.pathConnected_connected
-- test: set-path-connected-test.sage

/-! ## exm-path-sphere -/

/-- Any two points of the sphere are joined by a path in the sphere (an arc of a great circle). -/
theorem sphere_joined (p q : Fin 3 → ℝ) (hp : p ∈ KaluzaKlein.Sets.Sets.sphere)
    (hq : q ∈ KaluzaKlein.Sets.Sets.sphere) : JoinedIn KaluzaKlein.Sets.Sets.sphere p q := by
  -- the sphere is the image of the Euclidean unit sphere under the identification `EuclideanSpace ℝ (Fin 3) ≃ ℝ³`
  set e := EuclideanSpace.equiv (Fin 3) ℝ
  have himg : sphere = e '' Metric.sphere (0 : EuclideanSpace ℝ (Fin 3)) 1 := by
    ext x
    simp only [KaluzaKlein.Sets.Sets.sphere, Set.mem_setOf_eq, Set.mem_image, mem_sphere_iff_norm, sub_zero]
    constructor
    · intro hx
      refine ⟨e.symm x, ?_, by simp⟩
      rw [EuclideanSpace.norm_eq, Real.sqrt_eq_one]
      simp [Fin.sum_univ_succ, e]
      linarith
    · rintro ⟨y, hy, rfl⟩
      rw [EuclideanSpace.norm_eq, Real.sqrt_eq_one] at hy
      simp [Fin.sum_univ_succ, e] at hy ⊢
      linarith
  have hrank : 1 < Module.rank ℝ (EuclideanSpace ℝ (Fin 3)) := by
    rw [← Module.finrank_eq_rank, finrank_euclideanSpace_fin]; norm_num
  have hpc : IsPathConnected sphere := by
    rw [himg]
    exact (isPathConnected_sphere hrank 0 zero_le_one).image e.continuous
  exact hpc.joinedIn p hp q hq

example : veriphy_instance% KaluzaKlein.Sets.Connected.sphere_joined (KaluzaKlein.Sets.Sets.northPole) (![1, 0, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (p q : Fin 3 → ℝ) (hp : p ∈ KaluzaKlein.Sets.Sets.sphere) (hq : q ∈ KaluzaKlein.Sets.Sets.sphere), JoinedIn KaluzaKlein.Sets.Sets.sphere p q := by
    intro p q hp hq
    -- the sphere is the image of the Euclidean unit sphere under the identification `EuclideanSpace ℝ (Fin 3) ≃ ℝ³`
    set e := EuclideanSpace.equiv (Fin 3) ℝ
    have himg : sphere = e '' Metric.sphere (0 : EuclideanSpace ℝ (Fin 3)) 1 := by
      ext x
      simp only [KaluzaKlein.Sets.Sets.sphere, Set.mem_setOf_eq, Set.mem_image, mem_sphere_iff_norm, sub_zero]
      constructor
      · intro hx
        refine ⟨e.symm x, ?_, by simp⟩
        rw [EuclideanSpace.norm_eq, Real.sqrt_eq_one]
        simp [Fin.sum_univ_succ, e]
        linarith
      · rintro ⟨y, hy, rfl⟩
        rw [EuclideanSpace.norm_eq, Real.sqrt_eq_one] at hy
        simp [Fin.sum_univ_succ, e] at hy ⊢
        linarith
    have hrank : 1 < Module.rank ℝ (EuclideanSpace ℝ (Fin 3)) := by
      rw [← Module.finrank_eq_rank, finrank_euclideanSpace_fin]; norm_num
    have hpc : IsPathConnected sphere := by
      rw [himg]
      exact (isPathConnected_sphere hrank 0 zero_le_one).image e.continuous
    exact hpc.joinedIn p hp q hq
  exact @key (KaluzaKlein.Sets.Sets.northPole) (![1, 0, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))

example : veriphy_instance% KaluzaKlein.Sets.Connected.sphere_joined (KaluzaKlein.Sets.Sets.northPole) (KaluzaKlein.Sets.Sets.northPole) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (p q : Fin 3 → ℝ) (hp : p ∈ KaluzaKlein.Sets.Sets.sphere) (hq : q ∈ KaluzaKlein.Sets.Sets.sphere), JoinedIn KaluzaKlein.Sets.Sets.sphere p q := by
    intro p q hp hq
    -- the sphere is the image of the Euclidean unit sphere under the identification `EuclideanSpace ℝ (Fin 3) ≃ ℝ³`
    set e := EuclideanSpace.equiv (Fin 3) ℝ
    have himg : sphere = e '' Metric.sphere (0 : EuclideanSpace ℝ (Fin 3)) 1 := by
      ext x
      simp only [KaluzaKlein.Sets.Sets.sphere, Set.mem_setOf_eq, Set.mem_image, mem_sphere_iff_norm, sub_zero]
      constructor
      · intro hx
        refine ⟨e.symm x, ?_, by simp⟩
        rw [EuclideanSpace.norm_eq, Real.sqrt_eq_one]
        simp [Fin.sum_univ_succ, e]
        linarith
      · rintro ⟨y, hy, rfl⟩
        rw [EuclideanSpace.norm_eq, Real.sqrt_eq_one] at hy
        simp [Fin.sum_univ_succ, e] at hy ⊢
        linarith
    have hrank : 1 < Module.rank ℝ (EuclideanSpace ℝ (Fin 3)) := by
      rw [← Module.finrank_eq_rank, finrank_euclideanSpace_fin]; norm_num
    have hpc : IsPathConnected sphere := by
      rw [himg]
      exact (isPathConnected_sphere hrank 0 zero_le_one).image e.continuous
    exact hpc.joinedIn p hp q hq
  exact @key (KaluzaKlein.Sets.Sets.northPole) (KaluzaKlein.Sets.Sets.northPole) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))

example : veriphy_instance% KaluzaKlein.Sets.Connected.sphere_joined (![0, 0, -1]) (![0, 1, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (p q : Fin 3 → ℝ) (hp : p ∈ KaluzaKlein.Sets.Sets.sphere) (hq : q ∈ KaluzaKlein.Sets.Sets.sphere), JoinedIn KaluzaKlein.Sets.Sets.sphere p q := by
    intro p q hp hq
    -- the sphere is the image of the Euclidean unit sphere under the identification `EuclideanSpace ℝ (Fin 3) ≃ ℝ³`
    set e := EuclideanSpace.equiv (Fin 3) ℝ
    have himg : sphere = e '' Metric.sphere (0 : EuclideanSpace ℝ (Fin 3)) 1 := by
      ext x
      simp only [KaluzaKlein.Sets.Sets.sphere, Set.mem_setOf_eq, Set.mem_image, mem_sphere_iff_norm, sub_zero]
      constructor
      · intro hx
        refine ⟨e.symm x, ?_, by simp⟩
        rw [EuclideanSpace.norm_eq, Real.sqrt_eq_one]
        simp [Fin.sum_univ_succ, e]
        linarith
      · rintro ⟨y, hy, rfl⟩
        rw [EuclideanSpace.norm_eq, Real.sqrt_eq_one] at hy
        simp [Fin.sum_univ_succ, e] at hy ⊢
        linarith
    have hrank : 1 < Module.rank ℝ (EuclideanSpace ℝ (Fin 3)) := by
      rw [← Module.finrank_eq_rank, finrank_euclideanSpace_fin]; norm_num
    have hpc : IsPathConnected sphere := by
      rw [himg]
      exact (isPathConnected_sphere hrank 0 zero_le_one).image e.continuous
    exact hpc.joinedIn p hp q hq
  exact @key (![0, 0, -1]) (![0, 1, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.sphere_joined
-- test: set-sphere-joined-test.sage

/-- Any two points of `ℝⁿ` are joined by the straight segment `γ(t) = (1 - t) p + t q`. -/
theorem segment_joined (n : ℕ) (p q : Fin n → ℝ) : JoinedIn univ p q :=
  JoinedIn.ofLine (f := fun t : ℝ => (1 - t) • p + t • q) (by fun_prop) (by simp) (by simp) (subset_univ _)

example : veriphy_instance% KaluzaKlein.Sets.Connected.segment_joined (0) (![]) (![]) := by
  have key : ∀ (n : ℕ) (p q : Fin n → ℝ), JoinedIn univ p q := by
    intro n p q
    exact (JoinedIn.ofLine (f := fun t : ℝ => (1 - t) • p + t • q) (by fun_prop) (by simp) (by simp) (subset_univ _))
  exact @key (0) (![]) (![])

example : veriphy_instance% KaluzaKlein.Sets.Connected.segment_joined (2) (![0, 0]) (![1, 1]) := by
  have key : ∀ (n : ℕ) (p q : Fin n → ℝ), JoinedIn univ p q := by
    intro n p q
    exact (JoinedIn.ofLine (f := fun t : ℝ => (1 - t) • p + t • q) (by fun_prop) (by simp) (by simp) (subset_univ _))
  exact @key (2) (![0, 0]) (![1, 1])

example : veriphy_instance% KaluzaKlein.Sets.Connected.segment_joined (3) (![1, 2, 3]) (![0, 0, 0]) := by
  have key : ∀ (n : ℕ) (p q : Fin n → ℝ), JoinedIn univ p q := by
    intro n p q
    exact (JoinedIn.ofLine (f := fun t : ℝ => (1 - t) • p + t • q) (by fun_prop) (by simp) (by simp) (subset_univ _))
  exact @key (3) (![1, 2, 3]) (![0, 0, 0])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.segment_joined
-- test: set-segment-test.sage

/-! ## exm-path-rotations -/

/-- Every rotation is joined to the identity by a path of rotations about its axis, and no path in `O(3)` joins a
rotation to a reflection (the determinant would have to jump from `1` to `-1`). -/
theorem rotations_joined (R g : Matrix (Fin 3) (Fin 3) ℝ) (hR : R ∈ SO3) (hg : g ∈ O3) (hdg : g.det = -1) :
    JoinedIn SO3 1 R ∧ ¬ JoinedIn O3 R g := by
  refine ⟨rotation_path R hR, fun h => ?_⟩
  have hRO : R ∈ O3 := by rw [mem_SO3_iff] at hR; rw [mem_O3_iff]; exact hR.1
  have hcomp := O3_components R hRO
  obtain ⟨γ, hγ⟩ := h
  have hsub := (isPreconnected_range γ.continuous).subset_connectedComponentIn (x := R) ⟨0, by simp⟩
    (by rintro _ ⟨t, rfl⟩; exact hγ t)
  have hmem : g ∈ connectedComponentIn O3 R := hsub ⟨1, by simp⟩
  rw [hcomp] at hmem
  rw [mem_SO3_iff] at hR
  have := hmem.2
  rw [hdg, hR.2] at this
  norm_num at this

example : veriphy_instance% KaluzaKlein.Sets.Connected.rotations_joined (1) (!![1, 0, 0; 0, 1, 0; 0, 0, -1]) ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem) ((by rw [KaluzaKlein.Sets.Connected.mem_O3_iff]; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ])) ((by simp [Matrix.det_fin_three])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (R g : Matrix (Fin 3) (Fin 3) ℝ) (hR : R ∈ SO3) (hg : g ∈ O3) (hdg : g.det = -1), JoinedIn SO3 1 R ∧ ¬ JoinedIn O3 R g := by
    intro R g hR hg hdg
    refine ⟨rotation_path R hR, fun h => ?_⟩
    have hRO : R ∈ O3 := by rw [mem_SO3_iff] at hR; rw [mem_O3_iff]; exact hR.1
    have hcomp := O3_components R hRO
    obtain ⟨γ, hγ⟩ := h
    have hsub := (isPreconnected_range γ.continuous).subset_connectedComponentIn (x := R) ⟨0, by simp⟩
      (by rintro _ ⟨t, rfl⟩; exact hγ t)
    have hmem : g ∈ connectedComponentIn O3 R := hsub ⟨1, by simp⟩
    rw [hcomp] at hmem
    rw [mem_SO3_iff] at hR
    have := hmem.2
    rw [hdg, hR.2] at this
    norm_num at this
  exact @key (1) (!![1, 0, 0; 0, 1, 0; 0, 0, -1]) ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem) ((by rw [KaluzaKlein.Sets.Connected.mem_O3_iff]; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ])) ((by simp [Matrix.det_fin_three]))

example : veriphy_instance% KaluzaKlein.Sets.Connected.rotations_joined (KaluzaKlein.Sets.Surjective.Rz (Real.pi / 2)) (!![1, 0, 0; 0, 1, 0; 0, 0, -1]) (((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1) ((by rw [KaluzaKlein.Sets.Connected.mem_O3_iff]; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ])) ((by simp [Matrix.det_fin_three])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (R g : Matrix (Fin 3) (Fin 3) ℝ) (hR : R ∈ SO3) (hg : g ∈ O3) (hdg : g.det = -1), JoinedIn SO3 1 R ∧ ¬ JoinedIn O3 R g := by
    intro R g hR hg hdg
    refine ⟨rotation_path R hR, fun h => ?_⟩
    have hRO : R ∈ O3 := by rw [mem_SO3_iff] at hR; rw [mem_O3_iff]; exact hR.1
    have hcomp := O3_components R hRO
    obtain ⟨γ, hγ⟩ := h
    have hsub := (isPreconnected_range γ.continuous).subset_connectedComponentIn (x := R) ⟨0, by simp⟩
      (by rintro _ ⟨t, rfl⟩; exact hγ t)
    have hmem : g ∈ connectedComponentIn O3 R := hsub ⟨1, by simp⟩
    rw [hcomp] at hmem
    rw [mem_SO3_iff] at hR
    have := hmem.2
    rw [hdg, hR.2] at this
    norm_num at this
  exact @key (KaluzaKlein.Sets.Surjective.Rz (Real.pi / 2)) (!![1, 0, 0; 0, 1, 0; 0, 0, -1]) (((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1) ((by rw [KaluzaKlein.Sets.Connected.mem_O3_iff]; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ])) ((by simp [Matrix.det_fin_three]))

example : veriphy_instance% KaluzaKlein.Sets.Connected.rotations_joined (KaluzaKlein.Sets.Surjective.Rz 1) (!![1, 0, 0; 0, 1, 0; 0, 0, -1]) (((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1) ((by rw [KaluzaKlein.Sets.Connected.mem_O3_iff]; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ])) ((by simp [Matrix.det_fin_three])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (R g : Matrix (Fin 3) (Fin 3) ℝ) (hR : R ∈ SO3) (hg : g ∈ O3) (hdg : g.det = -1), JoinedIn SO3 1 R ∧ ¬ JoinedIn O3 R g := by
    intro R g hR hg hdg
    refine ⟨rotation_path R hR, fun h => ?_⟩
    have hRO : R ∈ O3 := by rw [mem_SO3_iff] at hR; rw [mem_O3_iff]; exact hR.1
    have hcomp := O3_components R hRO
    obtain ⟨γ, hγ⟩ := h
    have hsub := (isPreconnected_range γ.continuous).subset_connectedComponentIn (x := R) ⟨0, by simp⟩
      (by rintro _ ⟨t, rfl⟩; exact hγ t)
    have hmem : g ∈ connectedComponentIn O3 R := hsub ⟨1, by simp⟩
    rw [hcomp] at hmem
    rw [mem_SO3_iff] at hR
    have := hmem.2
    rw [hdg, hR.2] at this
    norm_num at this
  exact @key (KaluzaKlein.Sets.Surjective.Rz 1) (!![1, 0, 0; 0, 1, 0; 0, 0, -1]) (((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1) ((by rw [KaluzaKlein.Sets.Connected.mem_O3_iff]; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ])) ((by simp [Matrix.det_fin_three]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.rotations_joined
-- test: set-rotations-joined-test.sage

/-! ## exr-path-great-circle -/

/-- `γ(t) = (sin(πt/2), 0, cos(πt/2))` stays on the sphere: `sin² + 0² + cos² = 1`. -/
theorem great_circle_on_sphere (t : ℝ) :
    Real.sin (Real.pi * t / 2) ^ 2 + 0 ^ 2 + Real.cos (Real.pi * t / 2) ^ 2 = 1 := by
  veriphy_certified
    (simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, add_zero, Real.sin_sq_add_cos_sq])
  -- certificate: set-trig-pythagoras.sage

/-- `γ(0)` is the north pole and `γ(1) = (1, 0, 0)`. -/
theorem great_circle_endpoints :
    Real.sin (Real.pi * 0 / 2) = 0 ∧ Real.cos (Real.pi * 0 / 2) = 1 ∧ Real.sin (Real.pi * 1 / 2) = 1 ∧
      Real.cos (Real.pi * 1 / 2) = 0 := by
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

/-! ## exr-path-rotations-connect -/

/-- Any two rotations are joined by a path of rotations: `t ↦ R₁ R(t)` with `R(t)` the path from the identity to
`R₁⁻¹ R₂`. -/
theorem rotations_connect (R₁ R₂ : Matrix (Fin 3) (Fin 3) ℝ) (h₁ : R₁ ∈ SO3) (h₂ : R₂ ∈ SO3) :
    JoinedIn SO3 R₁ R₂ := by
  have hpc : IsPathConnected SO3 := ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem,
    fun {y} hy => rotation_path y hy⟩
  exact hpc.joinedIn R₁ h₁ R₂ h₂

example : veriphy_instance% KaluzaKlein.Sets.Connected.rotations_connect (1) (KaluzaKlein.Sets.Surjective.Rz (Real.pi / 2)) ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem) (((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1) := by
  have key : ∀ (R₁ R₂ : Matrix (Fin 3) (Fin 3) ℝ) (h₁ : R₁ ∈ SO3) (h₂ : R₂ ∈ SO3), JoinedIn SO3 R₁ R₂ := by
    intro R₁ R₂ h₁ h₂
    have hpc : IsPathConnected SO3 := ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem,
      fun {y} hy => rotation_path y hy⟩
    exact hpc.joinedIn R₁ h₁ R₂ h₂
  exact @key (1) (KaluzaKlein.Sets.Surjective.Rz (Real.pi / 2)) ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem) (((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1)

example : veriphy_instance% KaluzaKlein.Sets.Connected.rotations_connect (1) (1) ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem) ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem) := by
  have key : ∀ (R₁ R₂ : Matrix (Fin 3) (Fin 3) ℝ) (h₁ : R₁ ∈ SO3) (h₂ : R₂ ∈ SO3), JoinedIn SO3 R₁ R₂ := by
    intro R₁ R₂ h₁ h₂
    have hpc : IsPathConnected SO3 := ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem,
      fun {y} hy => rotation_path y hy⟩
    exact hpc.joinedIn R₁ h₁ R₂ h₂
  exact @key (1) (1) ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem) ((Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem)

example : veriphy_instance% KaluzaKlein.Sets.Connected.rotations_connect (KaluzaKlein.Sets.Surjective.Rz 1) (KaluzaKlein.Sets.Surjective.Rz 2) (((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1) (((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1) := by
  have key : ∀ (R₁ R₂ : Matrix (Fin 3) (Fin 3) ℝ) (h₁ : R₁ ∈ SO3) (h₂ : R₂ ∈ SO3), JoinedIn SO3 R₁ R₂ := by
    intro R₁ R₂ h₁ h₂
    have hpc : IsPathConnected SO3 := ⟨1, (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem,
      fun {y} hy => rotation_path y hy⟩
    exact hpc.joinedIn R₁ h₁ R₂ h₂
  exact @key (KaluzaKlein.Sets.Surjective.Rz 1) (KaluzaKlein.Sets.Surjective.Rz 2) (((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1) (((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Connected.rotations_connect
-- test: set-rotations-connect-test.sage

end KaluzaKlein.Sets.Connected
