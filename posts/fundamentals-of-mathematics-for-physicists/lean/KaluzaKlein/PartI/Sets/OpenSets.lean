import Mathlib
import KaluzaKlein.PartI.Sets.Products
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Open sets in ℝⁿ"

The claims of the section "Open sets in `ℝⁿ`" of `tutorial/part-i-01-sets-relations-topology.qmd`: intervals, discs
and the circle, coordinate domains and punctured spaces, the five sets of the exercise, open balls, and unions and
intersections of open sets. The plane is `ℝ × ℝ` with its product topology, which is the topology of the Euclidean
distance.
-/

namespace KaluzaKlein.Sets.OpenSets

open Set Metric

/-! ## exm-open-intervals -/

/-- For `a < b`: `(a, b)` is open, `[a, b]` is closed and not open, `[a, b)` is neither (`a = 0`, `b = 2π`). -/
theorem intervals_open_closed (a b : ℝ) (hab : a < b) :
    IsOpen (Ioo a b) ∧ (IsClosed (Icc a b) ∧ ¬ IsOpen (Icc a b)) ∧ (¬ IsOpen (Ico a b) ∧ ¬ IsClosed (Ico a b)) := by
  refine ⟨isOpen_Ioo, ⟨isClosed_Icc, fun h => ?_⟩, fun h => ?_, fun h => ?_⟩
  · have := (h.mem_nhds (left_mem_Icc.2 hab.le))
    have h2 := Icc_mem_nhds_iff.1 this
    simp at h2
  · have := h.mem_nhds (left_mem_Ico.2 hab)
    rw [mem_nhds_iff_exists_Ioo_subset] at this
    obtain ⟨l, u, ⟨hl, hu⟩, hsub⟩ := this
    have := hsub (show (max l (a - 1) + a) / 2 ∈ Ioo l u from ⟨by
      cases max_cases l (a - 1) <;> linarith, by cases max_cases l (a - 1) <;> linarith⟩)
    have := this.1
    cases max_cases l (a - 1) <;> linarith
  · have hcl := h.closure_eq
    have hb : b ∈ closure (Ico a b) := by
      rw [closure_Ico hab.ne]; exact right_mem_Icc.2 hab.le
    rw [hcl] at hb
    exact lt_irrefl b hb.2

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.intervals_open_closed (0) (2 * Real.pi) ((by positivity)) := by
  have key : ∀ (a b : ℝ) (hab : a < b), IsOpen (Ioo a b) ∧ (IsClosed (Icc a b) ∧ ¬ IsOpen (Icc a b)) ∧ (¬ IsOpen (Ico a b) ∧ ¬ IsClosed (Ico a b)) := by
    intro a b hab
    refine ⟨isOpen_Ioo, ⟨isClosed_Icc, fun h => ?_⟩, fun h => ?_, fun h => ?_⟩
    · have := (h.mem_nhds (left_mem_Icc.2 hab.le))
      have h2 := Icc_mem_nhds_iff.1 this
      simp at h2
    · have := h.mem_nhds (left_mem_Ico.2 hab)
      rw [mem_nhds_iff_exists_Ioo_subset] at this
      obtain ⟨l, u, ⟨hl, hu⟩, hsub⟩ := this
      have := hsub (show (max l (a - 1) + a) / 2 ∈ Ioo l u from ⟨by
        cases max_cases l (a - 1) <;> linarith, by cases max_cases l (a - 1) <;> linarith⟩)
      have := this.1
      cases max_cases l (a - 1) <;> linarith
    · have hcl := h.closure_eq
      have hb : b ∈ closure (Ico a b) := by
        rw [closure_Ico hab.ne]; exact right_mem_Icc.2 hab.le
      rw [hcl] at hb
      exact lt_irrefl b hb.2
  exact @key (0) (2 * Real.pi) ((by positivity))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.intervals_open_closed (0) (1) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (hab : a < b), IsOpen (Ioo a b) ∧ (IsClosed (Icc a b) ∧ ¬ IsOpen (Icc a b)) ∧ (¬ IsOpen (Ico a b) ∧ ¬ IsClosed (Ico a b)) := by
    intro a b hab
    refine ⟨isOpen_Ioo, ⟨isClosed_Icc, fun h => ?_⟩, fun h => ?_, fun h => ?_⟩
    · have := (h.mem_nhds (left_mem_Icc.2 hab.le))
      have h2 := Icc_mem_nhds_iff.1 this
      simp at h2
    · have := h.mem_nhds (left_mem_Ico.2 hab)
      rw [mem_nhds_iff_exists_Ioo_subset] at this
      obtain ⟨l, u, ⟨hl, hu⟩, hsub⟩ := this
      have := hsub (show (max l (a - 1) + a) / 2 ∈ Ioo l u from ⟨by
        cases max_cases l (a - 1) <;> linarith, by cases max_cases l (a - 1) <;> linarith⟩)
      have := this.1
      cases max_cases l (a - 1) <;> linarith
    · have hcl := h.closure_eq
      have hb : b ∈ closure (Ico a b) := by
        rw [closure_Ico hab.ne]; exact right_mem_Icc.2 hab.le
      rw [hcl] at hb
      exact lt_irrefl b hb.2
  exact @key (0) (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.intervals_open_closed (-1) (1) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (hab : a < b), IsOpen (Ioo a b) ∧ (IsClosed (Icc a b) ∧ ¬ IsOpen (Icc a b)) ∧ (¬ IsOpen (Ico a b) ∧ ¬ IsClosed (Ico a b)) := by
    intro a b hab
    refine ⟨isOpen_Ioo, ⟨isClosed_Icc, fun h => ?_⟩, fun h => ?_, fun h => ?_⟩
    · have := (h.mem_nhds (left_mem_Icc.2 hab.le))
      have h2 := Icc_mem_nhds_iff.1 this
      simp at h2
    · have := h.mem_nhds (left_mem_Ico.2 hab)
      rw [mem_nhds_iff_exists_Ioo_subset] at this
      obtain ⟨l, u, ⟨hl, hu⟩, hsub⟩ := this
      have := hsub (show (max l (a - 1) + a) / 2 ∈ Ioo l u from ⟨by
        cases max_cases l (a - 1) <;> linarith, by cases max_cases l (a - 1) <;> linarith⟩)
      have := this.1
      cases max_cases l (a - 1) <;> linarith
    · have hcl := h.closure_eq
      have hb : b ∈ closure (Ico a b) := by
        rw [closure_Ico hab.ne]; exact right_mem_Icc.2 hab.le
      rw [hcl] at hb
      exact lt_irrefl b hb.2
  exact @key (-1) (1) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.OpenSets.intervals_open_closed
-- test: set-intervals-test.sage

/-! ## exm-open-disc -/

/-- For radius `√r`: the open disc is open, the closed disc and the circle are closed, and for `r > 0` the circle is
not open: every ball around a point of the circle contains points off it. -/
theorem discs_open_closed (r : ℝ) :
    IsOpen {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} ∧ IsClosed {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧
      IsClosed {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ∧ (0 < r → ¬ IsOpen {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r}) := by
  refine ⟨isOpen_lt (by fun_prop) (by fun_prop), isClosed_le (by fun_prop) (by fun_prop),
    isClosed_eq (by fun_prop) (by fun_prop), fun hr h => ?_⟩
  -- the point (√r, 0) is on the circle; the points (√r + t, 0) are not, for t > 0
  set s := Real.sqrt r
  have hs : s ^ 2 = r := Real.sq_sqrt hr.le
  have hs0 : 0 < s := Real.sqrt_pos.2 hr
  have hmem : ((s, 0) : ℝ × ℝ) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} := by simp [hs]
  have hn := h.mem_nhds hmem
  have hc : Filter.Tendsto (fun t : ℝ => ((s + t, 0) : ℝ × ℝ)) (nhdsWithin 0 (Ioi 0)) (nhds (s, 0)) := by
    have : Continuous (fun t : ℝ => ((s + t, 0) : ℝ × ℝ)) := by fun_prop
    simpa using (this.tendsto 0).mono_left nhdsWithin_le_nhds
  obtain ⟨t, ht, ht0⟩ := ((hc.eventually hn).and self_mem_nhdsWithin).exists
  simp only [mem_setOf_eq, mem_Ioi] at ht ht0
  nlinarith

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.discs_open_closed (1) := by
  have key : ∀ (r : ℝ), IsOpen {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} ∧ IsClosed {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ IsClosed {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ∧ (0 < r → ¬ IsOpen {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r}) := by
    intro r
    refine ⟨isOpen_lt (by fun_prop) (by fun_prop), isClosed_le (by fun_prop) (by fun_prop),
      isClosed_eq (by fun_prop) (by fun_prop), fun hr h => ?_⟩
    -- the point (√r, 0) is on the circle; the points (√r + t, 0) are not, for t > 0
    set s := Real.sqrt r
    have hs : s ^ 2 = r := Real.sq_sqrt hr.le
    have hs0 : 0 < s := Real.sqrt_pos.2 hr
    have hmem : ((s, 0) : ℝ × ℝ) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} := by simp [hs]
    have hn := h.mem_nhds hmem
    have hc : Filter.Tendsto (fun t : ℝ => ((s + t, 0) : ℝ × ℝ)) (nhdsWithin 0 (Ioi 0)) (nhds (s, 0)) := by
      have : Continuous (fun t : ℝ => ((s + t, 0) : ℝ × ℝ)) := by fun_prop
      simpa using (this.tendsto 0).mono_left nhdsWithin_le_nhds
    obtain ⟨t, ht, ht0⟩ := ((hc.eventually hn).and self_mem_nhdsWithin).exists
    simp only [mem_setOf_eq, mem_Ioi] at ht ht0
    nlinarith
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.discs_open_closed (0) := by
  have key : ∀ (r : ℝ), IsOpen {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} ∧ IsClosed {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ IsClosed {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ∧ (0 < r → ¬ IsOpen {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r}) := by
    intro r
    refine ⟨isOpen_lt (by fun_prop) (by fun_prop), isClosed_le (by fun_prop) (by fun_prop),
      isClosed_eq (by fun_prop) (by fun_prop), fun hr h => ?_⟩
    -- the point (√r, 0) is on the circle; the points (√r + t, 0) are not, for t > 0
    set s := Real.sqrt r
    have hs : s ^ 2 = r := Real.sq_sqrt hr.le
    have hs0 : 0 < s := Real.sqrt_pos.2 hr
    have hmem : ((s, 0) : ℝ × ℝ) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} := by simp [hs]
    have hn := h.mem_nhds hmem
    have hc : Filter.Tendsto (fun t : ℝ => ((s + t, 0) : ℝ × ℝ)) (nhdsWithin 0 (Ioi 0)) (nhds (s, 0)) := by
      have : Continuous (fun t : ℝ => ((s + t, 0) : ℝ × ℝ)) := by fun_prop
      simpa using (this.tendsto 0).mono_left nhdsWithin_le_nhds
    obtain ⟨t, ht, ht0⟩ := ((hc.eventually hn).and self_mem_nhdsWithin).exists
    simp only [mem_setOf_eq, mem_Ioi] at ht ht0
    nlinarith
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.discs_open_closed (4) := by
  have key : ∀ (r : ℝ), IsOpen {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} ∧ IsClosed {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ IsClosed {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ∧ (0 < r → ¬ IsOpen {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r}) := by
    intro r
    refine ⟨isOpen_lt (by fun_prop) (by fun_prop), isClosed_le (by fun_prop) (by fun_prop),
      isClosed_eq (by fun_prop) (by fun_prop), fun hr h => ?_⟩
    -- the point (√r, 0) is on the circle; the points (√r + t, 0) are not, for t > 0
    set s := Real.sqrt r
    have hs : s ^ 2 = r := Real.sq_sqrt hr.le
    have hs0 : 0 < s := Real.sqrt_pos.2 hr
    have hmem : ((s, 0) : ℝ × ℝ) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} := by simp [hs]
    have hn := h.mem_nhds hmem
    have hc : Filter.Tendsto (fun t : ℝ => ((s + t, 0) : ℝ × ℝ)) (nhdsWithin 0 (Ioi 0)) (nhds (s, 0)) := by
      have : Continuous (fun t : ℝ => ((s + t, 0) : ℝ × ℝ)) := by fun_prop
      simpa using (this.tendsto 0).mono_left nhdsWithin_le_nhds
    obtain ⟨t, ht, ht0⟩ := ((hc.eventually hn).and self_mem_nhdsWithin).exists
    simp only [mem_setOf_eq, mem_Ioi] at ht ht0
    nlinarith
  exact @key (4)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.OpenSets.discs_open_closed
-- test: set-discs-test.sage

/-! ## exm-open-rectangle -/

/-- Open rectangles `(a, b) × (c, d)` are open in `ℝ²` (the domain `(0, π) × (0, 2π)` of spherical coordinates), and
`ℝ²` itself is open. -/
theorem rectangle_open (a b c d : ℝ) : IsOpen (Ioo a b ×ˢ Ioo c d) ∧ IsOpen (univ : Set (ℝ × ℝ)) :=
  ⟨isOpen_Ioo.prod isOpen_Ioo, isOpen_univ⟩

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.rectangle_open (0) (Real.pi) (0) (2 * Real.pi) := by
  have key : ∀ (a b c d : ℝ), IsOpen (Ioo a b ×ˢ Ioo c d) ∧ IsOpen (univ : Set (ℝ × ℝ)) := by
    intro a b c d
    exact (⟨isOpen_Ioo.prod isOpen_Ioo, isOpen_univ⟩)
  exact @key (0) (Real.pi) (0) (2 * Real.pi)

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.rectangle_open (0) (0) (0) (0) := by
  have key : ∀ (a b c d : ℝ), IsOpen (Ioo a b ×ˢ Ioo c d) ∧ IsOpen (univ : Set (ℝ × ℝ)) := by
    intro a b c d
    exact (⟨isOpen_Ioo.prod isOpen_Ioo, isOpen_univ⟩)
  exact @key (0) (0) (0) (0)

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.rectangle_open (-1) (1) (2) (3) := by
  have key : ∀ (a b c d : ℝ), IsOpen (Ioo a b ×ˢ Ioo c d) ∧ IsOpen (univ : Set (ℝ × ℝ)) := by
    intro a b c d
    exact (⟨isOpen_Ioo.prod isOpen_Ioo, isOpen_univ⟩)
  exact @key (-1) (1) (2) (3)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.OpenSets.rectangle_open
-- test: set-rectangle-test.sage

/-- `ℝⁿ` minus a point is open. -/
theorem punctured_open (n : ℕ) (p : Fin n → ℝ) : IsOpen ({p}ᶜ : Set (Fin n → ℝ)) :=
  isOpen_compl_singleton

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.punctured_open (2) (0) := by
  have key : ∀ (n : ℕ) (p : Fin n → ℝ), IsOpen ({p}ᶜ : Set (Fin n → ℝ)) := by
    intro n p
    exact (isOpen_compl_singleton)
  exact @key (2) (0)

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.punctured_open (0) (![]) := by
  have key : ∀ (n : ℕ) (p : Fin n → ℝ), IsOpen ({p}ᶜ : Set (Fin n → ℝ)) := by
    intro n p
    exact (isOpen_compl_singleton)
  exact @key (0) (![])

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.punctured_open (3) (![1, 2, 3]) := by
  have key : ∀ (n : ℕ) (p : Fin n → ℝ), IsOpen ({p}ᶜ : Set (Fin n → ℝ)) := by
    intro n p
    exact (isOpen_compl_singleton)
  exact @key (3) (![1, 2, 3])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.OpenSets.punctured_open
-- test: set-punctured-test.sage

/-! ## exr-open-decide -/

/-- `(a, b]` (`a < b`) is neither open nor closed: `(1, 2]`. -/
theorem Ioc_neither (a b : ℝ) (hab : a < b) : ¬ IsOpen (Ioc a b) ∧ ¬ IsClosed (Ioc a b) := by
  constructor
  · intro h
    have := h.mem_nhds (right_mem_Ioc.2 hab)
    rw [mem_nhds_iff_exists_Ioo_subset] at this
    obtain ⟨l, u, ⟨hl, hu⟩, hsub⟩ := this
    have := (hsub (show (b + min u (b + 1)) / 2 ∈ Ioo l u from ⟨by
      cases min_cases u (b + 1) <;> linarith, by cases min_cases u (b + 1) <;> linarith⟩)).2
    cases min_cases u (b + 1) <;> linarith
  · intro h
    have hcl := h.closure_eq
    have ha : a ∈ closure (Ioc a b) := by
      rw [closure_Ioc hab.ne]; exact left_mem_Icc.2 hab.le
    rw [hcl] at ha
    exact lt_irrefl a ha.1

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.Ioc_neither (1) (2) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (hab : a < b), ¬ IsOpen (Ioc a b) ∧ ¬ IsClosed (Ioc a b) := by
    intro a b hab
    constructor
    · intro h
      have := h.mem_nhds (right_mem_Ioc.2 hab)
      rw [mem_nhds_iff_exists_Ioo_subset] at this
      obtain ⟨l, u, ⟨hl, hu⟩, hsub⟩ := this
      have := (hsub (show (b + min u (b + 1)) / 2 ∈ Ioo l u from ⟨by
        cases min_cases u (b + 1) <;> linarith, by cases min_cases u (b + 1) <;> linarith⟩)).2
      cases min_cases u (b + 1) <;> linarith
    · intro h
      have hcl := h.closure_eq
      have ha : a ∈ closure (Ioc a b) := by
        rw [closure_Ioc hab.ne]; exact left_mem_Icc.2 hab.le
      rw [hcl] at ha
      exact lt_irrefl a ha.1
  exact @key (1) (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.Ioc_neither (0) (1) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (hab : a < b), ¬ IsOpen (Ioc a b) ∧ ¬ IsClosed (Ioc a b) := by
    intro a b hab
    constructor
    · intro h
      have := h.mem_nhds (right_mem_Ioc.2 hab)
      rw [mem_nhds_iff_exists_Ioo_subset] at this
      obtain ⟨l, u, ⟨hl, hu⟩, hsub⟩ := this
      have := (hsub (show (b + min u (b + 1)) / 2 ∈ Ioo l u from ⟨by
        cases min_cases u (b + 1) <;> linarith, by cases min_cases u (b + 1) <;> linarith⟩)).2
      cases min_cases u (b + 1) <;> linarith
    · intro h
      have hcl := h.closure_eq
      have ha : a ∈ closure (Ioc a b) := by
        rw [closure_Ioc hab.ne]; exact left_mem_Icc.2 hab.le
      rw [hcl] at ha
      exact lt_irrefl a ha.1
  exact @key (0) (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.Ioc_neither (-3) (5) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (hab : a < b), ¬ IsOpen (Ioc a b) ∧ ¬ IsClosed (Ioc a b) := by
    intro a b hab
    constructor
    · intro h
      have := h.mem_nhds (right_mem_Ioc.2 hab)
      rw [mem_nhds_iff_exists_Ioo_subset] at this
      obtain ⟨l, u, ⟨hl, hu⟩, hsub⟩ := this
      have := (hsub (show (b + min u (b + 1)) / 2 ∈ Ioo l u from ⟨by
        cases min_cases u (b + 1) <;> linarith, by cases min_cases u (b + 1) <;> linarith⟩)).2
      cases min_cases u (b + 1) <;> linarith
    · intro h
      have hcl := h.closure_eq
      have ha : a ∈ closure (Ioc a b) := by
        rw [closure_Ioc hab.ne]; exact left_mem_Icc.2 hab.le
      rw [hcl] at ha
      exact lt_irrefl a ha.1
  exact @key (-3) (5) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.OpenSets.Ioc_neither
-- test: set-ioc-neither-test.sage

/-- The multiples `Tℤ` (`T > 0`) form a closed, not open, subset of `ℝ`: `ℤ` for `T = 1`. -/
theorem multiples_closed_not_open (T : ℝ) (hT : 0 < T) :
    IsClosed (range (fun k : ℤ => (k : ℝ) * T)) ∧ ¬ IsOpen (range (fun k : ℤ => (k : ℝ) * T)) := by
  constructor
  · have : range (fun k : ℤ => (k : ℝ) * T) = (AddSubgroup.zmultiples T : Set ℝ) := by
      ext x; simp [AddSubgroup.mem_zmultiples_iff, zsmul_eq_mul, eq_comm]
    rw [this]
    exact AddSubgroup.isClosed_of_discrete
  · intro h
    have hmem : (0 : ℝ) ∈ range (fun k : ℤ => (k : ℝ) * T) := ⟨0, by simp⟩
    obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 h 0 hmem
    obtain ⟨k, hk⟩ := hball (show min (ε / 2) (T / 2) ∈ ball (0 : ℝ) ε by
      rw [mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos (lt_min (by linarith) (by linarith))]
      exact lt_of_le_of_lt (min_le_left _ _) (by linarith))
    simp only at hk
    have h1 : 0 < (k : ℝ) * T := by rw [hk]; exact lt_min (by linarith) (by linarith)
    have h2 : (k : ℝ) * T < T := by rw [hk]; exact lt_of_le_of_lt (min_le_right _ _) (by linarith)
    have hk1 : (0 : ℝ) < k := pos_of_mul_pos_left h1 hT.le
    have hk2 : (k : ℝ) < 1 := by nlinarith
    have : (0 : ℤ) < k := by exact_mod_cast hk1
    have : k < (1 : ℤ) := by exact_mod_cast hk2
    omega

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.multiples_closed_not_open (1) ((by norm_num)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), IsClosed (range (fun k : ℤ => (k : ℝ) * T)) ∧ ¬ IsOpen (range (fun k : ℤ => (k : ℝ) * T)) := by
    intro T hT
    constructor
    · have : range (fun k : ℤ => (k : ℝ) * T) = (AddSubgroup.zmultiples T : Set ℝ) := by
        ext x; simp [AddSubgroup.mem_zmultiples_iff, zsmul_eq_mul, eq_comm]
      rw [this]
      exact AddSubgroup.isClosed_of_discrete
    · intro h
      have hmem : (0 : ℝ) ∈ range (fun k : ℤ => (k : ℝ) * T) := ⟨0, by simp⟩
      obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 h 0 hmem
      obtain ⟨k, hk⟩ := hball (show min (ε / 2) (T / 2) ∈ ball (0 : ℝ) ε by
        rw [mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos (lt_min (by linarith) (by linarith))]
        exact lt_of_le_of_lt (min_le_left _ _) (by linarith))
      simp only at hk
      have h1 : 0 < (k : ℝ) * T := by rw [hk]; exact lt_min (by linarith) (by linarith)
      have h2 : (k : ℝ) * T < T := by rw [hk]; exact lt_of_le_of_lt (min_le_right _ _) (by linarith)
      have hk1 : (0 : ℝ) < k := pos_of_mul_pos_left h1 hT.le
      have hk2 : (k : ℝ) < 1 := by nlinarith
      have : (0 : ℤ) < k := by exact_mod_cast hk1
      have : k < (1 : ℤ) := by exact_mod_cast hk2
      omega
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.multiples_closed_not_open (2 * Real.pi) ((by positivity)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), IsClosed (range (fun k : ℤ => (k : ℝ) * T)) ∧ ¬ IsOpen (range (fun k : ℤ => (k : ℝ) * T)) := by
    intro T hT
    constructor
    · have : range (fun k : ℤ => (k : ℝ) * T) = (AddSubgroup.zmultiples T : Set ℝ) := by
        ext x; simp [AddSubgroup.mem_zmultiples_iff, zsmul_eq_mul, eq_comm]
      rw [this]
      exact AddSubgroup.isClosed_of_discrete
    · intro h
      have hmem : (0 : ℝ) ∈ range (fun k : ℤ => (k : ℝ) * T) := ⟨0, by simp⟩
      obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 h 0 hmem
      obtain ⟨k, hk⟩ := hball (show min (ε / 2) (T / 2) ∈ ball (0 : ℝ) ε by
        rw [mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos (lt_min (by linarith) (by linarith))]
        exact lt_of_le_of_lt (min_le_left _ _) (by linarith))
      simp only at hk
      have h1 : 0 < (k : ℝ) * T := by rw [hk]; exact lt_min (by linarith) (by linarith)
      have h2 : (k : ℝ) * T < T := by rw [hk]; exact lt_of_le_of_lt (min_le_right _ _) (by linarith)
      have hk1 : (0 : ℝ) < k := pos_of_mul_pos_left h1 hT.le
      have hk2 : (k : ℝ) < 1 := by nlinarith
      have : (0 : ℤ) < k := by exact_mod_cast hk1
      have : k < (1 : ℤ) := by exact_mod_cast hk2
      omega
  exact @key (2 * Real.pi) ((by positivity))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.multiples_closed_not_open (1/2) ((by norm_num)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), IsClosed (range (fun k : ℤ => (k : ℝ) * T)) ∧ ¬ IsOpen (range (fun k : ℤ => (k : ℝ) * T)) := by
    intro T hT
    constructor
    · have : range (fun k : ℤ => (k : ℝ) * T) = (AddSubgroup.zmultiples T : Set ℝ) := by
        ext x; simp [AddSubgroup.mem_zmultiples_iff, zsmul_eq_mul, eq_comm]
      rw [this]
      exact AddSubgroup.isClosed_of_discrete
    · intro h
      have hmem : (0 : ℝ) ∈ range (fun k : ℤ => (k : ℝ) * T) := ⟨0, by simp⟩
      obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 h 0 hmem
      obtain ⟨k, hk⟩ := hball (show min (ε / 2) (T / 2) ∈ ball (0 : ℝ) ε by
        rw [mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos (lt_min (by linarith) (by linarith))]
        exact lt_of_le_of_lt (min_le_left _ _) (by linarith))
      simp only at hk
      have h1 : 0 < (k : ℝ) * T := by rw [hk]; exact lt_min (by linarith) (by linarith)
      have h2 : (k : ℝ) * T < T := by rw [hk]; exact lt_of_le_of_lt (min_le_right _ _) (by linarith)
      have hk1 : (0 : ℝ) < k := pos_of_mul_pos_left h1 hT.le
      have hk2 : (k : ℝ) < 1 := by nlinarith
      have : (0 : ℤ) < k := by exact_mod_cast hk1
      have : k < (1 : ℤ) := by exact_mod_cast hk2
      omega
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.OpenSets.multiples_closed_not_open
-- test: set-multiples-closed-test.sage

/-- The region `v > u² + c` above a parabola is open; the hyperbola `uv = c` is closed; `ℝ² ∖ {p}` is open (for
`c = 0`, `c = 1` and `p = 0`). -/
theorem parabola_hyperbola (c : ℝ) (p : ℝ × ℝ) :
    IsOpen {q : ℝ × ℝ | q.2 > q.1 ^ 2 + c} ∧ IsClosed {q : ℝ × ℝ | q.1 * q.2 = c} ∧ IsOpen ({p}ᶜ : Set (ℝ × ℝ)) :=
  ⟨isOpen_lt (by fun_prop) (by fun_prop), isClosed_eq (by fun_prop) (by fun_prop), isOpen_compl_singleton⟩

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.parabola_hyperbola (0) ((0, 0)) := by
  have key : ∀ (c : ℝ) (p : ℝ × ℝ), IsOpen {q : ℝ × ℝ | q.2 > q.1 ^ 2 + c} ∧ IsClosed {q : ℝ × ℝ | q.1 * q.2 = c} ∧ IsOpen ({p}ᶜ : Set (ℝ × ℝ)) := by
    intro c p
    exact (⟨isOpen_lt (by fun_prop) (by fun_prop), isClosed_eq (by fun_prop) (by fun_prop), isOpen_compl_singleton⟩)
  exact @key (0) ((0, 0))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.parabola_hyperbola (1) ((1, 2)) := by
  have key : ∀ (c : ℝ) (p : ℝ × ℝ), IsOpen {q : ℝ × ℝ | q.2 > q.1 ^ 2 + c} ∧ IsClosed {q : ℝ × ℝ | q.1 * q.2 = c} ∧ IsOpen ({p}ᶜ : Set (ℝ × ℝ)) := by
    intro c p
    exact (⟨isOpen_lt (by fun_prop) (by fun_prop), isClosed_eq (by fun_prop) (by fun_prop), isOpen_compl_singleton⟩)
  exact @key (1) ((1, 2))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.parabola_hyperbola (-2) ((3, -1)) := by
  have key : ∀ (c : ℝ) (p : ℝ × ℝ), IsOpen {q : ℝ × ℝ | q.2 > q.1 ^ 2 + c} ∧ IsClosed {q : ℝ × ℝ | q.1 * q.2 = c} ∧ IsOpen ({p}ᶜ : Set (ℝ × ℝ)) := by
    intro c p
    exact (⟨isOpen_lt (by fun_prop) (by fun_prop), isClosed_eq (by fun_prop) (by fun_prop), isOpen_compl_singleton⟩)
  exact @key (-2) ((3, -1))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.OpenSets.parabola_hyperbola
-- test: set-parabola-test.sage

/-- Around a point `(u₀, v₀)` above the parabola, the ball of radius `δ = min(1, (v₀ - u₀²)/(2|u₀| + 2))` stays above
it: the explicit radius behind the openness (the solution samples such balls). -/
theorem parabola_ball (u₀ v₀ u v : ℝ) (h : u₀ ^ 2 < v₀) (hu : |u - u₀| < min 1 ((v₀ - u₀ ^ 2) / (2 * |u₀| + 2)))
    (hv : |v - v₀| < min 1 ((v₀ - u₀ ^ 2) / (2 * |u₀| + 2))) : u ^ 2 < v := by
  set δ := min 1 ((v₀ - u₀ ^ 2) / (2 * |u₀| + 2))
  have hd : 0 < 2 * |u₀| + 2 := by positivity
  have hδ1 : δ ≤ 1 := min_le_left _ _
  have hδ2 : δ ≤ (v₀ - u₀ ^ 2) / (2 * |u₀| + 2) := min_le_right _ _
  have hδ2' : δ * (2 * |u₀| + 2) ≤ v₀ - u₀ ^ 2 := by rwa [le_div_iff₀ hd] at hδ2
  rw [abs_lt] at hu hv
  have hu0 := abs_nonneg u₀
  have e1 : u ^ 2 - u₀ ^ 2 = (u - u₀) * (u + u₀) := by ring
  have : |u + u₀| ≤ 2 * |u₀| + 1 := by
    calc |u + u₀| = |(u - u₀) + 2 * u₀| := by ring_nf
      _ ≤ |u - u₀| + |2 * u₀| := abs_add_le _ _
      _ ≤ 1 + 2 * |u₀| := by rw [abs_mul, abs_two]; nlinarith [abs_lt.2 hu]
      _ = 2 * |u₀| + 1 := by ring
  have h3 : u ^ 2 - u₀ ^ 2 ≤ |u - u₀| * (2 * |u₀| + 1) := by
    rw [e1]; calc (u - u₀) * (u + u₀) ≤ |(u - u₀) * (u + u₀)| := le_abs_self _
      _ = |u - u₀| * |u + u₀| := abs_mul _ _
      _ ≤ |u - u₀| * (2 * |u₀| + 1) := by gcongr
  have h4 : |u - u₀| < δ := abs_lt.2 hu
  nlinarith [abs_nonneg (u - u₀)]

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.parabola_ball (0) (1) (0) (1) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.parabola_ball (1) (3) (6/5) (14/5) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.parabola_ball (-1) (2) (-11/10) (19/10) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.OpenSets.parabola_ball
-- test: set-parabola-ball-test.sage

/-! ## exr-open-ball-is-open -/

/-- Around `x ∈ B_ε(p)` the ball of radius `δ = ε - |x - p|` lies in `B_ε(p)`, so open balls are open. -/
theorem ball_radius_subset (n : ℕ) (p x : EuclideanSpace ℝ (Fin n)) (ε : ℝ) (hx : x ∈ ball p ε) :
    0 < ε - dist x p ∧ ball x (ε - dist x p) ⊆ ball p ε ∧ IsOpen (ball p ε) := by
  refine ⟨sub_pos.2 hx, fun y hy => ?_, isOpen_ball⟩
  rw [mem_ball] at hy ⊢
  linarith [dist_triangle y x p]

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.ball_radius_subset (0) (0) (0) (1) ((by simp)) := by
  have key : ∀ (n : ℕ) (p x : EuclideanSpace ℝ (Fin n)) (ε : ℝ) (hx : x ∈ ball p ε), 0 < ε - dist x p ∧ ball x (ε - dist x p) ⊆ ball p ε ∧ IsOpen (ball p ε) := by
    intro n p x ε hx
    refine ⟨sub_pos.2 hx, fun y hy => ?_, isOpen_ball⟩
    rw [mem_ball] at hy ⊢
    linarith [dist_triangle y x p]
  exact @key (0) (0) (0) (1) ((by simp))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.ball_radius_subset (1) (0) (0) (2) ((by simp)) := by
  have key : ∀ (n : ℕ) (p x : EuclideanSpace ℝ (Fin n)) (ε : ℝ) (hx : x ∈ ball p ε), 0 < ε - dist x p ∧ ball x (ε - dist x p) ⊆ ball p ε ∧ IsOpen (ball p ε) := by
    intro n p x ε hx
    refine ⟨sub_pos.2 hx, fun y hy => ?_, isOpen_ball⟩
    rw [mem_ball] at hy ⊢
    linarith [dist_triangle y x p]
  exact @key (1) (0) (0) (2) ((by simp))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.ball_radius_subset (3) (0) (0) (1/2) ((by simp)) := by
  have key : ∀ (n : ℕ) (p x : EuclideanSpace ℝ (Fin n)) (ε : ℝ) (hx : x ∈ ball p ε), 0 < ε - dist x p ∧ ball x (ε - dist x p) ⊆ ball p ε ∧ IsOpen (ball p ε) := by
    intro n p x ε hx
    refine ⟨sub_pos.2 hx, fun y hy => ?_, isOpen_ball⟩
    rw [mem_ball] at hy ⊢
    linarith [dist_triangle y x p]
  exact @key (3) (0) (0) (1/2) ((by simp))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.OpenSets.ball_radius_subset
-- test: set-ball-open-test.sage

/-! ## exr-open-union-intersection -/

/-- Any union of open sets of `ℝⁿ` is open, and any finite intersection is open. -/
theorem union_inter_open (n : ℕ) (S : Set (Set (EuclideanSpace ℝ (Fin n)))) (hS : ∀ U ∈ S, IsOpen U)
    (F : Finset (Set (EuclideanSpace ℝ (Fin n)))) (hF : ∀ U ∈ F, IsOpen U) :
    IsOpen (⋃₀ S) ∧ IsOpen (⋂₀ (F : Set (Set (EuclideanSpace ℝ (Fin n))))) :=
  ⟨isOpen_sUnion hS, (F.finite_toSet).isOpen_sInter hF⟩

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.union_inter_open (1) ({Metric.ball 0 1, Metric.ball 0 2}) ((by rintro U (rfl | rfl) <;> exact Metric.isOpen_ball)) ({Metric.ball 0 1}) ((by simp [Metric.isOpen_ball])) := by
  have key : ∀ (n : ℕ) (S : Set (Set (EuclideanSpace ℝ (Fin n)))) (hS : ∀ U ∈ S, IsOpen U) (F : Finset (Set (EuclideanSpace ℝ (Fin n)))) (hF : ∀ U ∈ F, IsOpen U), IsOpen (⋃₀ S) ∧ IsOpen (⋂₀ (F : Set (Set (EuclideanSpace ℝ (Fin n))))) := by
    intro n S hS F hF
    exact (⟨isOpen_sUnion hS, (F.finite_toSet).isOpen_sInter hF⟩)
  exact @key (1) ({Metric.ball 0 1, Metric.ball 0 2}) ((by rintro U (rfl | rfl) <;> exact Metric.isOpen_ball)) ({Metric.ball 0 1}) ((by simp [Metric.isOpen_ball]))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.union_inter_open (0) (∅) ((by simp)) (∅) ((by simp)) := by
  have key : ∀ (n : ℕ) (S : Set (Set (EuclideanSpace ℝ (Fin n)))) (hS : ∀ U ∈ S, IsOpen U) (F : Finset (Set (EuclideanSpace ℝ (Fin n)))) (hF : ∀ U ∈ F, IsOpen U), IsOpen (⋃₀ S) ∧ IsOpen (⋂₀ (F : Set (Set (EuclideanSpace ℝ (Fin n))))) := by
    intro n S hS F hF
    exact (⟨isOpen_sUnion hS, (F.finite_toSet).isOpen_sInter hF⟩)
  exact @key (0) (∅) ((by simp)) (∅) ((by simp))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.union_inter_open (2) ({Set.univ}) ((by simp)) ({Metric.ball 0 3}) ((by simp [Metric.isOpen_ball])) := by
  have key : ∀ (n : ℕ) (S : Set (Set (EuclideanSpace ℝ (Fin n)))) (hS : ∀ U ∈ S, IsOpen U) (F : Finset (Set (EuclideanSpace ℝ (Fin n)))) (hF : ∀ U ∈ F, IsOpen U), IsOpen (⋃₀ S) ∧ IsOpen (⋂₀ (F : Set (Set (EuclideanSpace ℝ (Fin n))))) := by
    intro n S hS F hF
    exact (⟨isOpen_sUnion hS, (F.finite_toSet).isOpen_sInter hF⟩)
  exact @key (2) ({Set.univ}) ((by simp)) ({Metric.ball 0 3}) ((by simp [Metric.isOpen_ball]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.OpenSets.union_inter_open
-- test: set-union-inter-test.sage

/-- The intervals `(-c/(n + 1), c/(n + 1))` (`c > 0`) intersect in `{0}`, which is not open: an infinite intersection
of open sets need not be open (`c = 1`: the intervals `(-1/n, 1/n)`). -/
theorem infinite_inter_not_open (c : ℝ) (hc : 0 < c) :
    (⋂ n : ℕ, Ioo (-c / (n + 1)) (c / (n + 1))) = {0} ∧ ¬ IsOpen ({0} : Set ℝ) := by
  refine ⟨?_, not_isOpen_singleton 0⟩
  ext x
  simp only [mem_iInter, mem_Ioo, mem_singleton_iff]
  constructor
  · intro h
    by_contra hx
    obtain ⟨n, hn⟩ := exists_nat_gt (c / |x|)
    have hx' : 0 < |x| := abs_pos.2 hx
    have h1 := h n
    have h2 : c / (n + 1) < |x| := by
      rw [div_lt_iff₀ (by positivity)]
      rw [div_lt_iff₀ hx'] at hn
      nlinarith
    rw [neg_div] at h1
    rcases abs_cases x with ⟨h3, -⟩ | ⟨h3, -⟩ <;> rw [h3] at h2 <;> linarith [h1.1, h1.2]
  · rintro rfl n
    exact ⟨by rw [neg_div]; exact neg_lt_zero.2 (by positivity), by positivity⟩

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.infinite_inter_not_open (1) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), (⋂ n : ℕ, Ioo (-c / (n + 1)) (c / (n + 1))) = {0} ∧ ¬ IsOpen ({0} : Set ℝ) := by
    intro c hc
    refine ⟨?_, not_isOpen_singleton 0⟩
    ext x
    simp only [mem_iInter, mem_Ioo, mem_singleton_iff]
    constructor
    · intro h
      by_contra hx
      obtain ⟨n, hn⟩ := exists_nat_gt (c / |x|)
      have hx' : 0 < |x| := abs_pos.2 hx
      have h1 := h n
      have h2 : c / (n + 1) < |x| := by
        rw [div_lt_iff₀ (by positivity)]
        rw [div_lt_iff₀ hx'] at hn
        nlinarith
      rw [neg_div] at h1
      rcases abs_cases x with ⟨h3, -⟩ | ⟨h3, -⟩ <;> rw [h3] at h2 <;> linarith [h1.1, h1.2]
    · rintro rfl n
      exact ⟨by rw [neg_div]; exact neg_lt_zero.2 (by positivity), by positivity⟩
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.infinite_inter_not_open (2) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), (⋂ n : ℕ, Ioo (-c / (n + 1)) (c / (n + 1))) = {0} ∧ ¬ IsOpen ({0} : Set ℝ) := by
    intro c hc
    refine ⟨?_, not_isOpen_singleton 0⟩
    ext x
    simp only [mem_iInter, mem_Ioo, mem_singleton_iff]
    constructor
    · intro h
      by_contra hx
      obtain ⟨n, hn⟩ := exists_nat_gt (c / |x|)
      have hx' : 0 < |x| := abs_pos.2 hx
      have h1 := h n
      have h2 : c / (n + 1) < |x| := by
        rw [div_lt_iff₀ (by positivity)]
        rw [div_lt_iff₀ hx'] at hn
        nlinarith
      rw [neg_div] at h1
      rcases abs_cases x with ⟨h3, -⟩ | ⟨h3, -⟩ <;> rw [h3] at h2 <;> linarith [h1.1, h1.2]
    · rintro rfl n
      exact ⟨by rw [neg_div]; exact neg_lt_zero.2 (by positivity), by positivity⟩
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.OpenSets.infinite_inter_not_open (1/2) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), (⋂ n : ℕ, Ioo (-c / (n + 1)) (c / (n + 1))) = {0} ∧ ¬ IsOpen ({0} : Set ℝ) := by
    intro c hc
    refine ⟨?_, not_isOpen_singleton 0⟩
    ext x
    simp only [mem_iInter, mem_Ioo, mem_singleton_iff]
    constructor
    · intro h
      by_contra hx
      obtain ⟨n, hn⟩ := exists_nat_gt (c / |x|)
      have hx' : 0 < |x| := abs_pos.2 hx
      have h1 := h n
      have h2 : c / (n + 1) < |x| := by
        rw [div_lt_iff₀ (by positivity)]
        rw [div_lt_iff₀ hx'] at hn
        nlinarith
      rw [neg_div] at h1
      rcases abs_cases x with ⟨h3, -⟩ | ⟨h3, -⟩ <;> rw [h3] at h2 <;> linarith [h1.1, h1.2]
    · rintro rfl n
      exact ⟨by rw [neg_div]; exact neg_lt_zero.2 (by positivity), by positivity⟩
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.OpenSets.infinite_inter_not_open
-- test: set-infinite-inter-test.sage

end KaluzaKlein.Sets.OpenSets
