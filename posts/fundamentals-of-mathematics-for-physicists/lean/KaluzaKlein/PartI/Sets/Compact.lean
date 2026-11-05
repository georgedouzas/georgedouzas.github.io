import Mathlib
import KaluzaKlein.PartI.Sets.Hausdorff
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Compact spaces"

The claims of the section "Compact spaces" of `tutorial/part-i-01-sets-relations-topology.qmd`: Heine-Borel and
extreme values, closed and open intervals, the circle, the sphere, the torus and `SO(3)`, quotients of compact spaces,
the non-compact `ℝⁿ`, open disc and hyperboloid, the cover `(1/n, 1)` of `(0, 1)`, the extremes of `x e^{-x}`, and
closed subsets of compact sets.
-/

namespace KaluzaKlein.Sets.Compact

open Set Metric Filter Topology Matrix KaluzaKlein.Sets.Sets

/-! ## def-compact -/

/-- Heine-Borel: a subset of a proper metric space such as `ℝⁿ` is compact exactly when it is closed and bounded. -/
theorem heine_borel (E : Type) [MetricSpace E] [ProperSpace E] (K : Set E) :
    IsCompact K ↔ IsClosed K ∧ Bornology.IsBounded K :=
  Metric.isCompact_iff_isClosed_bounded

example : veriphy_instance% KaluzaKlein.Sets.Compact.heine_borel (ℝ) (inferInstance) (inferInstance) (Set.Icc 0 1) := by
  have key : ∀ (E : Type) [inst1 : MetricSpace E] [inst2 : ProperSpace E] (K : Set E), IsCompact K ↔ IsClosed K ∧ Bornology.IsBounded K := by
    intro E inst1 inst2 K
    exact (Metric.isCompact_iff_isClosed_bounded)
  exact @key (ℝ) (inferInstance) (inferInstance) (Set.Icc 0 1)

example : veriphy_instance% KaluzaKlein.Sets.Compact.heine_borel (ℝ) (inferInstance) (inferInstance) (∅) := by
  have key : ∀ (E : Type) [inst1 : MetricSpace E] [inst2 : ProperSpace E] (K : Set E), IsCompact K ↔ IsClosed K ∧ Bornology.IsBounded K := by
    intro E inst1 inst2 K
    exact (Metric.isCompact_iff_isClosed_bounded)
  exact @key (ℝ) (inferInstance) (inferInstance) (∅)

example : veriphy_instance% KaluzaKlein.Sets.Compact.heine_borel (ℝ × ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.circle) := by
  have key : ∀ (E : Type) [inst1 : MetricSpace E] [inst2 : ProperSpace E] (K : Set E), IsCompact K ↔ IsClosed K ∧ Bornology.IsBounded K := by
    intro E inst1 inst2 K
    exact (Metric.isCompact_iff_isClosed_bounded)
  exact @key (ℝ × ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.circle)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Compact.heine_borel
-- test: set-heine-borel-test.sage

/-- A continuous real function on a non-empty compact set is bounded and attains its maximum and minimum. -/
theorem extreme_values (E : Type) [TopologicalSpace E] (K : Set E) (hK : IsCompact K) (hne : K.Nonempty) (f : E → ℝ)
    (hf : ContinuousOn f K) :
    BddAbove (f '' K) ∧ BddBelow (f '' K) ∧ (∃ a ∈ K, IsMaxOn f K a) ∧ ∃ b ∈ K, IsMinOn f K b :=
  ⟨(hK.image_of_continuousOn hf).bddAbove, (hK.image_of_continuousOn hf).bddBelow, hK.exists_isMaxOn hne hf,
    hK.exists_isMinOn hne hf⟩

example : veriphy_instance% KaluzaKlein.Sets.Compact.extreme_values (ℝ) (inferInstance) (Set.Icc 0 2) (isCompact_Icc) (⟨0, by norm_num⟩) (fun x => x * Real.exp (-x)) ((by fun_prop : Continuous fun x : ℝ => x * Real.exp (-x)).continuousOn) := by
  have key : ∀ (E : Type) [inst1 : TopologicalSpace E] (K : Set E) (hK : IsCompact K) (hne : K.Nonempty) (f : E → ℝ) (hf : ContinuousOn f K), BddAbove (f '' K) ∧ BddBelow (f '' K) ∧ (∃ a ∈ K, IsMaxOn f K a) ∧ ∃ b ∈ K, IsMinOn f K b := by
    intro E inst1 K hK hne f hf
    exact (⟨(hK.image_of_continuousOn hf).bddAbove, (hK.image_of_continuousOn hf).bddBelow, hK.exists_isMaxOn hne hf,
      hK.exists_isMinOn hne hf⟩)
  exact @key (ℝ) (inferInstance) (Set.Icc 0 2) (isCompact_Icc) (⟨0, by norm_num⟩) (fun x => x * Real.exp (-x)) ((by fun_prop : Continuous fun x : ℝ => x * Real.exp (-x)).continuousOn)

example : veriphy_instance% KaluzaKlein.Sets.Compact.extreme_values (ℝ) (inferInstance) ({0}) (isCompact_singleton) (⟨0, rfl⟩) (id) (continuousOn_id) := by
  have key : ∀ (E : Type) [inst1 : TopologicalSpace E] (K : Set E) (hK : IsCompact K) (hne : K.Nonempty) (f : E → ℝ) (hf : ContinuousOn f K), BddAbove (f '' K) ∧ BddBelow (f '' K) ∧ (∃ a ∈ K, IsMaxOn f K a) ∧ ∃ b ∈ K, IsMinOn f K b := by
    intro E inst1 K hK hne f hf
    exact (⟨(hK.image_of_continuousOn hf).bddAbove, (hK.image_of_continuousOn hf).bddBelow, hK.exists_isMaxOn hne hf,
      hK.exists_isMinOn hne hf⟩)
  exact @key (ℝ) (inferInstance) ({0}) (isCompact_singleton) (⟨0, rfl⟩) (id) (continuousOn_id)

example : veriphy_instance% KaluzaKlein.Sets.Compact.extreme_values (ℝ) (inferInstance) (Set.Icc (-1) 1) (isCompact_Icc) (⟨0, by norm_num⟩) (Real.cos) (Real.continuous_cos.continuousOn) := by
  have key : ∀ (E : Type) [inst1 : TopologicalSpace E] (K : Set E) (hK : IsCompact K) (hne : K.Nonempty) (f : E → ℝ) (hf : ContinuousOn f K), BddAbove (f '' K) ∧ BddBelow (f '' K) ∧ (∃ a ∈ K, IsMaxOn f K a) ∧ ∃ b ∈ K, IsMinOn f K b := by
    intro E inst1 K hK hne f hf
    exact (⟨(hK.image_of_continuousOn hf).bddAbove, (hK.image_of_continuousOn hf).bddBelow, hK.exists_isMaxOn hne hf,
      hK.exists_isMinOn hne hf⟩)
  exact @key (ℝ) (inferInstance) (Set.Icc (-1) 1) (isCompact_Icc) (⟨0, by norm_num⟩) (Real.cos) (Real.continuous_cos.continuousOn)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Compact.extreme_values
-- test: set-extreme-values-test.sage

/-! ## exm-compact-interval -/

/-- For `a < b`: `[a, b]` is compact, `(a, b)` is not, and `ℝ` is not (`[0, 1]` and `(0, 1)`). -/
theorem intervals_compact (a b : ℝ) (hab : a < b) :
    IsCompact (Icc a b) ∧ ¬ IsCompact (Ioo a b) ∧ ¬ IsCompact (univ : Set ℝ) := by
  refine ⟨isCompact_Icc, fun h => ?_, noncompact_univ ℝ⟩
  have hcl := h.isClosed.closure_eq
  rw [closure_Ioo hab.ne] at hcl
  have : a ∈ Ioo a b := hcl ▸ left_mem_Icc.2 hab.le
  exact lt_irrefl a this.1

example : veriphy_instance% KaluzaKlein.Sets.Compact.intervals_compact (0) (1) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (hab : a < b), IsCompact (Icc a b) ∧ ¬ IsCompact (Ioo a b) ∧ ¬ IsCompact (univ : Set ℝ) := by
    intro a b hab
    refine ⟨isCompact_Icc, fun h => ?_, noncompact_univ ℝ⟩
    have hcl := h.isClosed.closure_eq
    rw [closure_Ioo hab.ne] at hcl
    have : a ∈ Ioo a b := hcl ▸ left_mem_Icc.2 hab.le
    exact lt_irrefl a this.1
  exact @key (0) (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Compact.intervals_compact (-1) (1) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (hab : a < b), IsCompact (Icc a b) ∧ ¬ IsCompact (Ioo a b) ∧ ¬ IsCompact (univ : Set ℝ) := by
    intro a b hab
    refine ⟨isCompact_Icc, fun h => ?_, noncompact_univ ℝ⟩
    have hcl := h.isClosed.closure_eq
    rw [closure_Ioo hab.ne] at hcl
    have : a ∈ Ioo a b := hcl ▸ left_mem_Icc.2 hab.le
    exact lt_irrefl a this.1
  exact @key (-1) (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Compact.intervals_compact (2) (5) ((by norm_num)) := by
  have key : ∀ (a b : ℝ) (hab : a < b), IsCompact (Icc a b) ∧ ¬ IsCompact (Ioo a b) ∧ ¬ IsCompact (univ : Set ℝ) := by
    intro a b hab
    refine ⟨isCompact_Icc, fun h => ?_, noncompact_univ ℝ⟩
    have hcl := h.isClosed.closure_eq
    rw [closure_Ioo hab.ne] at hcl
    have : a ∈ Ioo a b := hcl ▸ left_mem_Icc.2 hab.le
    exact lt_irrefl a this.1
  exact @key (2) (5) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Compact.intervals_compact
-- test: set-intervals-compact-test.sage

/-! ## exm-compact-spheres -/

/-- The circle is closed and bounded in `ℝ²`. -/
theorem circle_closed_bounded : IsClosed circle ∧ Bornology.IsBounded circle := by
  refine ⟨isClosed_eq (by fun_prop) continuous_const, (Metric.isBounded_closedBall (x := (0 : ℝ × ℝ)) (r := 1)).subset
    fun p hp => ?_⟩
  simp only [circle, mem_setOf_eq] at hp
  rw [Metric.mem_closedBall, dist_zero_right, Prod.norm_def, max_le_iff, Real.norm_eq_abs, Real.norm_eq_abs]
  constructor <;> rw [abs_le] <;> constructor <;> nlinarith [sq_nonneg p.1, sq_nonneg p.2]

/-- The sphere is closed and bounded in `ℝ³`. -/
theorem sphere_closed_bounded : IsClosed sphere ∧ Bornology.IsBounded sphere := by
  refine ⟨isClosed_eq (by fun_prop) continuous_const, (Metric.isBounded_closedBall (x := (0 : Fin 3 → ℝ)) (r := 1)).subset
    fun p hp => ?_⟩
  simp only [KaluzaKlein.Sets.Sets.sphere, mem_setOf_eq] at hp
  rw [Metric.mem_closedBall, dist_zero_right, pi_norm_le_iff_of_nonneg zero_le_one]
  intro i
  rw [Real.norm_eq_abs, abs_le]
  fin_cases i <;> simp <;> constructor <;> nlinarith [sq_nonneg (p 0), sq_nonneg (p 1), sq_nonneg (p 2)]

/-- The torus `S¹ × S¹ ⊂ ℝ⁴` is closed and bounded. -/
theorem torus_closed_bounded : IsClosed (circle ×ˢ circle) ∧ Bornology.IsBounded (circle ×ˢ circle) :=
  ⟨circle_closed_bounded.1.prod circle_closed_bounded.1, circle_closed_bounded.2.prod circle_closed_bounded.2⟩

/-- `SO(3)`, as arrays of entries in `ℝ^{3×3}`, is closed and bounded (entries in `[-1, 1]`). -/
def SO3pi : Set (Fin 3 → Fin 3 → ℝ) := {A | Matrix.of A ∈ SO3}

theorem SO3pi_compact : IsCompact SO3pi := by
  have h := QuotientTopology.SO3_compact.image
    (f := fun A : Matrix (Fin 3) (Fin 3) ℝ => (Matrix.of.symm A : Fin 3 → Fin 3 → ℝ)) continuous_id
  convert h using 1
  ext A
  constructor
  · intro hA; exact ⟨Matrix.of A, hA, rfl⟩
  · rintro ⟨B, hB, rfl⟩; simpa [SO3pi] using hB

theorem SO3_closed_bounded : IsClosed SO3pi ∧ Bornology.IsBounded SO3pi :=
  ⟨SO3pi_compact.isClosed, SO3pi_compact.isBounded⟩

/-- A closed and bounded subset of a proper metric space is compact: the circle, the sphere, the torus and `SO(3)`. -/
theorem compact_of_closed_bounded (E : Type) [MetricSpace E] [ProperSpace E] (K : Set E) (hc : IsClosed K)
    (hb : Bornology.IsBounded K) : IsCompact K :=
  Metric.isCompact_of_isClosed_isBounded hc hb

example : veriphy_instance% KaluzaKlein.Sets.Compact.compact_of_closed_bounded (ℝ × ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.circle) (KaluzaKlein.Sets.Compact.circle_closed_bounded.1) (KaluzaKlein.Sets.Compact.circle_closed_bounded.2) := by
  have key : ∀ (E : Type) [inst1 : MetricSpace E] [inst2 : ProperSpace E] (K : Set E) (hc : IsClosed K) (hb : Bornology.IsBounded K), IsCompact K := by
    intro E inst1 inst2 K hc hb
    exact (Metric.isCompact_of_isClosed_isBounded hc hb)
  exact @key (ℝ × ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.circle) (KaluzaKlein.Sets.Compact.circle_closed_bounded.1) (KaluzaKlein.Sets.Compact.circle_closed_bounded.2)

example : veriphy_instance% KaluzaKlein.Sets.Compact.compact_of_closed_bounded (Fin 3 → ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.sphere) (KaluzaKlein.Sets.Compact.sphere_closed_bounded.1) (KaluzaKlein.Sets.Compact.sphere_closed_bounded.2) := by
  have key : ∀ (E : Type) [inst1 : MetricSpace E] [inst2 : ProperSpace E] (K : Set E) (hc : IsClosed K) (hb : Bornology.IsBounded K), IsCompact K := by
    intro E inst1 inst2 K hc hb
    exact (Metric.isCompact_of_isClosed_isBounded hc hb)
  exact @key (Fin 3 → ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.sphere) (KaluzaKlein.Sets.Compact.sphere_closed_bounded.1) (KaluzaKlein.Sets.Compact.sphere_closed_bounded.2)

example : veriphy_instance% KaluzaKlein.Sets.Compact.compact_of_closed_bounded ((ℝ × ℝ) × (ℝ × ℝ)) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.circle ×ˢ KaluzaKlein.Sets.Sets.circle) (KaluzaKlein.Sets.Compact.torus_closed_bounded.1) (KaluzaKlein.Sets.Compact.torus_closed_bounded.2) := by
  have key : ∀ (E : Type) [inst1 : MetricSpace E] [inst2 : ProperSpace E] (K : Set E) (hc : IsClosed K) (hb : Bornology.IsBounded K), IsCompact K := by
    intro E inst1 inst2 K hc hb
    exact (Metric.isCompact_of_isClosed_isBounded hc hb)
  exact @key ((ℝ × ℝ) × (ℝ × ℝ)) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.circle ×ˢ KaluzaKlein.Sets.Sets.circle) (KaluzaKlein.Sets.Compact.torus_closed_bounded.1) (KaluzaKlein.Sets.Compact.torus_closed_bounded.2)

example : veriphy_instance% KaluzaKlein.Sets.Compact.compact_of_closed_bounded (Fin 3 → Fin 3 → ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Compact.SO3pi) (KaluzaKlein.Sets.Compact.SO3_closed_bounded.1) (KaluzaKlein.Sets.Compact.SO3_closed_bounded.2) := by
  have key : ∀ (E : Type) [inst1 : MetricSpace E] [inst2 : ProperSpace E] (K : Set E) (hc : IsClosed K) (hb : Bornology.IsBounded K), IsCompact K := by
    intro E inst1 inst2 K hc hb
    exact (Metric.isCompact_of_isClosed_isBounded hc hb)
  exact @key (Fin 3 → Fin 3 → ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Compact.SO3pi) (KaluzaKlein.Sets.Compact.SO3_closed_bounded.1) (KaluzaKlein.Sets.Compact.SO3_closed_bounded.2)

example : veriphy_instance% KaluzaKlein.Sets.Compact.compact_of_closed_bounded (ℝ) (inferInstance) (inferInstance) (∅) (isClosed_empty) (Bornology.isBounded_empty) := by
  have key : ∀ (E : Type) [inst1 : MetricSpace E] [inst2 : ProperSpace E] (K : Set E) (hc : IsClosed K) (hb : Bornology.IsBounded K), IsCompact K := by
    intro E inst1 inst2 K hc hb
    exact (Metric.isCompact_of_isClosed_isBounded hc hb)
  exact @key (ℝ) (inferInstance) (inferInstance) (∅) (isClosed_empty) (Bornology.isBounded_empty)

attribute [veriphy_tested kernel_test 5 0] KaluzaKlein.Sets.Compact.compact_of_closed_bounded
-- test: set-compact-closed-bounded-test.sage

/-- The closed unit disc is compact. -/
theorem closedDisc_compact : IsCompact closedDisc := by
  refine Metric.isCompact_of_isClosed_isBounded (isClosed_le (by fun_prop) continuous_const)
    ((Metric.isBounded_closedBall (x := (0 : ℝ × ℝ)) (r := 1)).subset fun p hp => ?_)
  simp only [closedDisc, mem_setOf_eq] at hp
  rw [Metric.mem_closedBall, dist_zero_right, Prod.norm_def, max_le_iff, Real.norm_eq_abs, Real.norm_eq_abs]
  constructor <;> rw [abs_le] <;> constructor <;> nlinarith [sq_nonneg p.1, sq_nonneg p.2]

/-- A quotient of a compact space is compact: every coset space `S/R` of a compact group, `SO(3)/SO(2)` among them. -/
theorem quotient_compact (X : Type) [TopologicalSpace X] [CompactSpace X] (s : Setoid X) :
    CompactSpace (Quotient s) :=
  Quotient.compactSpace

example : veriphy_instance% KaluzaKlein.Sets.Compact.quotient_compact (Fin 3) (inferInstance) (inferInstance) (⊤) := by
  exact Quotient.compactSpace

example : veriphy_instance% KaluzaKlein.Sets.Compact.quotient_compact (Unit) (inferInstance) (inferInstance) (⊥) := by
  exact Quotient.compactSpace

example : veriphy_instance% KaluzaKlein.Sets.Compact.quotient_compact (Set.Icc (0 : ℝ) 1) (inferInstance) ((isCompact_iff_compactSpace.1 isCompact_Icc)) (⊤) := by
  exact Quotient.compactSpace

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Compact.quotient_compact
-- test: set-quotient-compact-test.sage

/-! ## exm-compact-plane -/

/-- `ℝⁿ` (`n ≥ 1`) is not compact, nor is the open disc of radius `√r` (`r > 0`). -/
theorem plane_disc_not_compact (n : ℕ) (hn : 1 ≤ n) (r : ℝ) (hr : 0 < r) :
    ¬ IsCompact (univ : Set (EuclideanSpace ℝ (Fin n))) ∧ ¬ IsCompact {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} := by
  haveI : Nontrivial (EuclideanSpace ℝ (Fin n)) := by
    haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
    infer_instance
  refine ⟨noncompact_univ _, fun h => ?_⟩
  -- the boundary point `(√r, 0)` is a limit of points `((1 - t)√r, 0)` of the disc
  set s := Real.sqrt r
  have hs : s ^ 2 = r := Real.sq_sqrt hr.le
  have hs0 : 0 < s := Real.sqrt_pos.2 hr
  have hcl := h.isClosed
  have hlim : Tendsto (fun t : ℝ => (((1 - t) * s, 0) : ℝ × ℝ)) (𝓝[>] 0) (𝓝 (s, 0)) := by
    have : Continuous (fun t : ℝ => (((1 - t) * s, 0) : ℝ × ℝ)) := by fun_prop
    simpa using (this.tendsto 0).mono_left nhdsWithin_le_nhds
  have hev : ∀ᶠ t in 𝓝[>] (0 : ℝ), (((1 - t) * s, 0) : ℝ × ℝ) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} := by
    filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with t ht
    obtain ⟨h0, h1⟩ := ht
    have : (1 - t) ^ 2 < 1 := by nlinarith
    nlinarith
  have := hcl.mem_of_tendsto hlim hev
  simp only [mem_setOf_eq] at this
  nlinarith

example : veriphy_instance% KaluzaKlein.Sets.Compact.plane_disc_not_compact (1) ((by norm_num)) (1) ((by norm_num)) := by
  have key : ∀ (n : ℕ) (hn : 1 ≤ n) (r : ℝ) (hr : 0 < r), ¬ IsCompact (univ : Set (EuclideanSpace ℝ (Fin n))) ∧ ¬ IsCompact {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} := by
    intro n hn r hr
    haveI : Nontrivial (EuclideanSpace ℝ (Fin n)) := by
      haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
      infer_instance
    refine ⟨noncompact_univ _, fun h => ?_⟩
    -- the boundary point `(√r, 0)` is a limit of points `((1 - t)√r, 0)` of the disc
    set s := Real.sqrt r
    have hs : s ^ 2 = r := Real.sq_sqrt hr.le
    have hs0 : 0 < s := Real.sqrt_pos.2 hr
    have hcl := h.isClosed
    have hlim : Tendsto (fun t : ℝ => (((1 - t) * s, 0) : ℝ × ℝ)) (𝓝[>] 0) (𝓝 (s, 0)) := by
      have : Continuous (fun t : ℝ => (((1 - t) * s, 0) : ℝ × ℝ)) := by fun_prop
      simpa using (this.tendsto 0).mono_left nhdsWithin_le_nhds
    have hev : ∀ᶠ t in 𝓝[>] (0 : ℝ), (((1 - t) * s, 0) : ℝ × ℝ) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} := by
      filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with t ht
      obtain ⟨h0, h1⟩ := ht
      have : (1 - t) ^ 2 < 1 := by nlinarith
      nlinarith
    have := hcl.mem_of_tendsto hlim hev
    simp only [mem_setOf_eq] at this
    nlinarith
  exact @key (1) ((by norm_num)) (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Compact.plane_disc_not_compact (2) ((by norm_num)) (1) ((by norm_num)) := by
  have key : ∀ (n : ℕ) (hn : 1 ≤ n) (r : ℝ) (hr : 0 < r), ¬ IsCompact (univ : Set (EuclideanSpace ℝ (Fin n))) ∧ ¬ IsCompact {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} := by
    intro n hn r hr
    haveI : Nontrivial (EuclideanSpace ℝ (Fin n)) := by
      haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
      infer_instance
    refine ⟨noncompact_univ _, fun h => ?_⟩
    -- the boundary point `(√r, 0)` is a limit of points `((1 - t)√r, 0)` of the disc
    set s := Real.sqrt r
    have hs : s ^ 2 = r := Real.sq_sqrt hr.le
    have hs0 : 0 < s := Real.sqrt_pos.2 hr
    have hcl := h.isClosed
    have hlim : Tendsto (fun t : ℝ => (((1 - t) * s, 0) : ℝ × ℝ)) (𝓝[>] 0) (𝓝 (s, 0)) := by
      have : Continuous (fun t : ℝ => (((1 - t) * s, 0) : ℝ × ℝ)) := by fun_prop
      simpa using (this.tendsto 0).mono_left nhdsWithin_le_nhds
    have hev : ∀ᶠ t in 𝓝[>] (0 : ℝ), (((1 - t) * s, 0) : ℝ × ℝ) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} := by
      filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with t ht
      obtain ⟨h0, h1⟩ := ht
      have : (1 - t) ^ 2 < 1 := by nlinarith
      nlinarith
    have := hcl.mem_of_tendsto hlim hev
    simp only [mem_setOf_eq] at this
    nlinarith
  exact @key (2) ((by norm_num)) (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Compact.plane_disc_not_compact (4) ((by norm_num)) (4) ((by norm_num)) := by
  have key : ∀ (n : ℕ) (hn : 1 ≤ n) (r : ℝ) (hr : 0 < r), ¬ IsCompact (univ : Set (EuclideanSpace ℝ (Fin n))) ∧ ¬ IsCompact {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} := by
    intro n hn r hr
    haveI : Nontrivial (EuclideanSpace ℝ (Fin n)) := by
      haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
      infer_instance
    refine ⟨noncompact_univ _, fun h => ?_⟩
    -- the boundary point `(√r, 0)` is a limit of points `((1 - t)√r, 0)` of the disc
    set s := Real.sqrt r
    have hs : s ^ 2 = r := Real.sq_sqrt hr.le
    have hs0 : 0 < s := Real.sqrt_pos.2 hr
    have hcl := h.isClosed
    have hlim : Tendsto (fun t : ℝ => (((1 - t) * s, 0) : ℝ × ℝ)) (𝓝[>] 0) (𝓝 (s, 0)) := by
      have : Continuous (fun t : ℝ => (((1 - t) * s, 0) : ℝ × ℝ)) := by fun_prop
      simpa using (this.tendsto 0).mono_left nhdsWithin_le_nhds
    have hev : ∀ᶠ t in 𝓝[>] (0 : ℝ), (((1 - t) * s, 0) : ℝ × ℝ) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} := by
      filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with t ht
      obtain ⟨h0, h1⟩ := ht
      have : (1 - t) ^ 2 < 1 := by nlinarith
      nlinarith
    have := hcl.mem_of_tendsto hlim hev
    simp only [mem_setOf_eq] at this
    nlinarith
  exact @key (4) ((by norm_num)) (4) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Compact.plane_disc_not_compact
-- test: set-plane-disc-test.sage

/-- The hyperboloid `-t² + x² + y² + z² = ℓ²` (de Sitter space) is not compact: it is unbounded, containing
`(t, √(ℓ² + t²), 0, 0)` for every `t`. -/
theorem hyperboloid_not_compact (ℓ : ℝ) :
    ¬ IsCompact {p : Fin 4 → ℝ | -p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 = ℓ ^ 2} := by
  intro h
  obtain ⟨R, hR⟩ := h.isBounded.subset_closedBall 0
  have hmem : (![R + 1, Real.sqrt (ℓ ^ 2 + (R + 1) ^ 2), 0, 0] : Fin 4 → ℝ) ∈
      {p : Fin 4 → ℝ | -p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 = ℓ ^ 2} := by
    simp only [mem_setOf_eq, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two,
      Matrix.tail_cons, Matrix.cons_val_three]
    rw [Real.sq_sqrt (by positivity)]; ring
  have := hR hmem
  rw [mem_closedBall, dist_zero_right] at this
  have h0 := norm_le_pi_norm (![R + 1, Real.sqrt (ℓ ^ 2 + (R + 1) ^ 2), 0, 0] : Fin 4 → ℝ) 0
  simp only [Matrix.cons_val_zero, Real.norm_eq_abs] at h0
  have hR0 : 0 ≤ R := le_trans (norm_nonneg _) this
  rw [abs_of_pos (by linarith)] at h0
  linarith

example : veriphy_instance% KaluzaKlein.Sets.Compact.hyperboloid_not_compact (1) := by
  have key : ∀ (ℓ : ℝ), ¬ IsCompact {p : Fin 4 → ℝ | -p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 = ℓ ^ 2} := by
    intro ℓ
    intro h
    obtain ⟨R, hR⟩ := h.isBounded.subset_closedBall 0
    have hmem : (![R + 1, Real.sqrt (ℓ ^ 2 + (R + 1) ^ 2), 0, 0] : Fin 4 → ℝ) ∈
        {p : Fin 4 → ℝ | -p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 = ℓ ^ 2} := by
      simp only [mem_setOf_eq, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two,
        Matrix.tail_cons, Matrix.cons_val_three]
      rw [Real.sq_sqrt (by positivity)]; ring
    have := hR hmem
    rw [mem_closedBall, dist_zero_right] at this
    have h0 := norm_le_pi_norm (![R + 1, Real.sqrt (ℓ ^ 2 + (R + 1) ^ 2), 0, 0] : Fin 4 → ℝ) 0
    simp only [Matrix.cons_val_zero, Real.norm_eq_abs] at h0
    have hR0 : 0 ≤ R := le_trans (norm_nonneg _) this
    rw [abs_of_pos (by linarith)] at h0
    linarith
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Compact.hyperboloid_not_compact (0) := by
  have key : ∀ (ℓ : ℝ), ¬ IsCompact {p : Fin 4 → ℝ | -p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 = ℓ ^ 2} := by
    intro ℓ
    intro h
    obtain ⟨R, hR⟩ := h.isBounded.subset_closedBall 0
    have hmem : (![R + 1, Real.sqrt (ℓ ^ 2 + (R + 1) ^ 2), 0, 0] : Fin 4 → ℝ) ∈
        {p : Fin 4 → ℝ | -p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 = ℓ ^ 2} := by
      simp only [mem_setOf_eq, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two,
        Matrix.tail_cons, Matrix.cons_val_three]
      rw [Real.sq_sqrt (by positivity)]; ring
    have := hR hmem
    rw [mem_closedBall, dist_zero_right] at this
    have h0 := norm_le_pi_norm (![R + 1, Real.sqrt (ℓ ^ 2 + (R + 1) ^ 2), 0, 0] : Fin 4 → ℝ) 0
    simp only [Matrix.cons_val_zero, Real.norm_eq_abs] at h0
    have hR0 : 0 ≤ R := le_trans (norm_nonneg _) this
    rw [abs_of_pos (by linarith)] at h0
    linarith
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Compact.hyperboloid_not_compact (2) := by
  have key : ∀ (ℓ : ℝ), ¬ IsCompact {p : Fin 4 → ℝ | -p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 = ℓ ^ 2} := by
    intro ℓ
    intro h
    obtain ⟨R, hR⟩ := h.isBounded.subset_closedBall 0
    have hmem : (![R + 1, Real.sqrt (ℓ ^ 2 + (R + 1) ^ 2), 0, 0] : Fin 4 → ℝ) ∈
        {p : Fin 4 → ℝ | -p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 + p 3 ^ 2 = ℓ ^ 2} := by
      simp only [mem_setOf_eq, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two,
        Matrix.tail_cons, Matrix.cons_val_three]
      rw [Real.sq_sqrt (by positivity)]; ring
    have := hR hmem
    rw [mem_closedBall, dist_zero_right] at this
    have h0 := norm_le_pi_norm (![R + 1, Real.sqrt (ℓ ^ 2 + (R + 1) ^ 2), 0, 0] : Fin 4 → ℝ) 0
    simp only [Matrix.cons_val_zero, Real.norm_eq_abs] at h0
    have hR0 : 0 ≤ R := le_trans (norm_nonneg _) this
    rw [abs_of_pos (by linarith)] at h0
    linarith
  exact @key (2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Compact.hyperboloid_not_compact
-- test: set-hyperboloid-test.sage

/-! ## exr-compact-proof -/

/-- The sets `U_n = (c/n, c)`, `n ≥ 2` (`c > 0`), cover `(0, c)`, but no finite subfamily does: the members with
`n < N + 2` all miss `c/(N + 2)` (the cover of `(0, 1)` by `(1/n, 1)` for `c = 1`). -/
theorem cover_no_finite_subcover (c : ℝ) (hc : 0 < c) :
    (⋃ n : ℕ, Ioo (c / (n + 2)) c) = Ioo 0 c ∧
      ∀ N : ℕ, c / (N + 2) ∈ Ioo 0 c ∧ c / (N + 2) ∉ ⋃ n ∈ Finset.range N, Ioo (c / (n + 2)) c := by
  refine ⟨?_, fun N => ⟨⟨by positivity, by rw [div_lt_iff₀ (by positivity)]; nlinarith⟩, ?_⟩⟩
  · ext x
    simp only [mem_iUnion, mem_Ioo]
    constructor
    · rintro ⟨n, h1, h2⟩; exact ⟨lt_trans (by positivity) h1, h2⟩
    · rintro ⟨h1, h2⟩
      obtain ⟨n, hn⟩ := exists_nat_gt (c / x)
      refine ⟨n, ?_, h2⟩
      rw [div_lt_iff₀ (by positivity)]
      rw [div_lt_iff₀ h1] at hn
      nlinarith
  · simp only [mem_iUnion, mem_Ioo, Finset.mem_range, not_exists, not_and]
    intro n hn h
    exfalso
    have : c / (N + 2) ≤ c / (n + 2) := by
      apply div_le_div_of_nonneg_left hc.le (by positivity)
      have : (n : ℝ) < N := by exact_mod_cast hn
      linarith
    linarith

example : veriphy_instance% KaluzaKlein.Sets.Compact.cover_no_finite_subcover (1) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), (⋃ n : ℕ, Ioo (c / (n + 2)) c) = Ioo 0 c ∧ ∀ N : ℕ, c / (N + 2) ∈ Ioo 0 c ∧ c / (N + 2) ∉ ⋃ n ∈ Finset.range N, Ioo (c / (n + 2)) c := by
    intro c hc
    refine ⟨?_, fun N => ⟨⟨by positivity, by rw [div_lt_iff₀ (by positivity)]; nlinarith⟩, ?_⟩⟩
    · ext x
      simp only [mem_iUnion, mem_Ioo]
      constructor
      · rintro ⟨n, h1, h2⟩; exact ⟨lt_trans (by positivity) h1, h2⟩
      · rintro ⟨h1, h2⟩
        obtain ⟨n, hn⟩ := exists_nat_gt (c / x)
        refine ⟨n, ?_, h2⟩
        rw [div_lt_iff₀ (by positivity)]
        rw [div_lt_iff₀ h1] at hn
        nlinarith
    · simp only [mem_iUnion, mem_Ioo, Finset.mem_range, not_exists, not_and]
      intro n hn h
      exfalso
      have : c / (N + 2) ≤ c / (n + 2) := by
        apply div_le_div_of_nonneg_left hc.le (by positivity)
        have : (n : ℝ) < N := by exact_mod_cast hn
        linarith
      linarith
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Compact.cover_no_finite_subcover (2) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), (⋃ n : ℕ, Ioo (c / (n + 2)) c) = Ioo 0 c ∧ ∀ N : ℕ, c / (N + 2) ∈ Ioo 0 c ∧ c / (N + 2) ∉ ⋃ n ∈ Finset.range N, Ioo (c / (n + 2)) c := by
    intro c hc
    refine ⟨?_, fun N => ⟨⟨by positivity, by rw [div_lt_iff₀ (by positivity)]; nlinarith⟩, ?_⟩⟩
    · ext x
      simp only [mem_iUnion, mem_Ioo]
      constructor
      · rintro ⟨n, h1, h2⟩; exact ⟨lt_trans (by positivity) h1, h2⟩
      · rintro ⟨h1, h2⟩
        obtain ⟨n, hn⟩ := exists_nat_gt (c / x)
        refine ⟨n, ?_, h2⟩
        rw [div_lt_iff₀ (by positivity)]
        rw [div_lt_iff₀ h1] at hn
        nlinarith
    · simp only [mem_iUnion, mem_Ioo, Finset.mem_range, not_exists, not_and]
      intro n hn h
      exfalso
      have : c / (N + 2) ≤ c / (n + 2) := by
        apply div_le_div_of_nonneg_left hc.le (by positivity)
        have : (n : ℝ) < N := by exact_mod_cast hn
        linarith
      linarith
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Compact.cover_no_finite_subcover (1/2) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : 0 < c), (⋃ n : ℕ, Ioo (c / (n + 2)) c) = Ioo 0 c ∧ ∀ N : ℕ, c / (N + 2) ∈ Ioo 0 c ∧ c / (N + 2) ∉ ⋃ n ∈ Finset.range N, Ioo (c / (n + 2)) c := by
    intro c hc
    refine ⟨?_, fun N => ⟨⟨by positivity, by rw [div_lt_iff₀ (by positivity)]; nlinarith⟩, ?_⟩⟩
    · ext x
      simp only [mem_iUnion, mem_Ioo]
      constructor
      · rintro ⟨n, h1, h2⟩; exact ⟨lt_trans (by positivity) h1, h2⟩
      · rintro ⟨h1, h2⟩
        obtain ⟨n, hn⟩ := exists_nat_gt (c / x)
        refine ⟨n, ?_, h2⟩
        rw [div_lt_iff₀ (by positivity)]
        rw [div_lt_iff₀ h1] at hn
        nlinarith
    · simp only [mem_iUnion, mem_Ioo, Finset.mem_range, not_exists, not_and]
      intro n hn h
      exfalso
      have : c / (N + 2) ≤ c / (n + 2) := by
        apply div_le_div_of_nonneg_left hc.le (by positivity)
        have : (n : ℝ) < N := by exact_mod_cast hn
        linarith
      linarith
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Compact.cover_no_finite_subcover
-- test: set-cover-test.sage

/-- For `[0, c]` the family misses the endpoints `0` and `c`; open sets around them contain `(-ε, ε)` and
`(c - ε, c + ε)`, and together with one `U_N` (`c/(N + 2) < ε`) they cover `[0, c]`. -/
theorem closed_interval_cover (c ε : ℝ) (hc : 0 < c) (hε : 0 < ε) (N : ℕ) (hN : c / (N + 2) < ε) :
    (0 : ℝ) ∉ ⋃ n : ℕ, Ioo (c / (n + 2)) c ∧ c ∉ ⋃ n : ℕ, Ioo (c / (n + 2)) c ∧
      Icc 0 c ⊆ Ioo (-ε) ε ∪ Ioo (c - ε) (c + ε) ∪ Ioo (c / (N + 2)) c := by
  refine ⟨?_, ?_, ?_⟩
  · simp only [mem_iUnion, mem_Ioo, not_exists, not_and]
    intro n h; exact absurd h (not_lt.2 (by positivity))
  · simp only [mem_iUnion, mem_Ioo, not_exists, not_and]
    intro n _ h; exact lt_irrefl c h
  · intro x ⟨h0, h1⟩
    simp only [mem_union, mem_Ioo]
    by_cases hx : x < ε
    · left; left; exact ⟨by linarith, hx⟩
    · rcases eq_or_lt_of_le h1 with rfl | h1'
      · left; right; exact ⟨by linarith, by linarith⟩
      · right; exact ⟨by linarith, h1'⟩

example : veriphy_instance% KaluzaKlein.Sets.Compact.closed_interval_cover (1) (1/2) ((by norm_num)) ((by norm_num)) (2) ((by norm_num)) := by
  have key : ∀ (c ε : ℝ) (hc : 0 < c) (hε : 0 < ε) (N : ℕ) (hN : c / (N + 2) < ε), (0 : ℝ) ∉ ⋃ n : ℕ, Ioo (c / (n + 2)) c ∧ c ∉ ⋃ n : ℕ, Ioo (c / (n + 2)) c ∧ Icc 0 c ⊆ Ioo (-ε) ε ∪ Ioo (c - ε) (c + ε) ∪ Ioo (c / (N + 2)) c := by
    intro c ε hc hε N hN
    refine ⟨?_, ?_, ?_⟩
    · simp only [mem_iUnion, mem_Ioo, not_exists, not_and]
      intro n h; exact absurd h (not_lt.2 (by positivity))
    · simp only [mem_iUnion, mem_Ioo, not_exists, not_and]
      intro n _ h; exact lt_irrefl c h
    · intro x ⟨h0, h1⟩
      simp only [mem_union, mem_Ioo]
      by_cases hx : x < ε
      · left; left; exact ⟨by linarith, hx⟩
      · rcases eq_or_lt_of_le h1 with rfl | h1'
        · left; right; exact ⟨by linarith, by linarith⟩
        · right; exact ⟨by linarith, h1'⟩
  exact @key (1) (1/2) ((by norm_num)) ((by norm_num)) (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Compact.closed_interval_cover (1) (1/10) ((by norm_num)) ((by norm_num)) (10) ((by norm_num)) := by
  have key : ∀ (c ε : ℝ) (hc : 0 < c) (hε : 0 < ε) (N : ℕ) (hN : c / (N + 2) < ε), (0 : ℝ) ∉ ⋃ n : ℕ, Ioo (c / (n + 2)) c ∧ c ∉ ⋃ n : ℕ, Ioo (c / (n + 2)) c ∧ Icc 0 c ⊆ Ioo (-ε) ε ∪ Ioo (c - ε) (c + ε) ∪ Ioo (c / (N + 2)) c := by
    intro c ε hc hε N hN
    refine ⟨?_, ?_, ?_⟩
    · simp only [mem_iUnion, mem_Ioo, not_exists, not_and]
      intro n h; exact absurd h (not_lt.2 (by positivity))
    · simp only [mem_iUnion, mem_Ioo, not_exists, not_and]
      intro n _ h; exact lt_irrefl c h
    · intro x ⟨h0, h1⟩
      simp only [mem_union, mem_Ioo]
      by_cases hx : x < ε
      · left; left; exact ⟨by linarith, hx⟩
      · rcases eq_or_lt_of_le h1 with rfl | h1'
        · left; right; exact ⟨by linarith, by linarith⟩
        · right; exact ⟨by linarith, h1'⟩
  exact @key (1) (1/10) ((by norm_num)) ((by norm_num)) (10) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Compact.closed_interval_cover (2) (1) ((by norm_num)) ((by norm_num)) (1) ((by norm_num)) := by
  have key : ∀ (c ε : ℝ) (hc : 0 < c) (hε : 0 < ε) (N : ℕ) (hN : c / (N + 2) < ε), (0 : ℝ) ∉ ⋃ n : ℕ, Ioo (c / (n + 2)) c ∧ c ∉ ⋃ n : ℕ, Ioo (c / (n + 2)) c ∧ Icc 0 c ⊆ Ioo (-ε) ε ∪ Ioo (c - ε) (c + ε) ∪ Ioo (c / (N + 2)) c := by
    intro c ε hc hε N hN
    refine ⟨?_, ?_, ?_⟩
    · simp only [mem_iUnion, mem_Ioo, not_exists, not_and]
      intro n h; exact absurd h (not_lt.2 (by positivity))
    · simp only [mem_iUnion, mem_Ioo, not_exists, not_and]
      intro n _ h; exact lt_irrefl c h
    · intro x ⟨h0, h1⟩
      simp only [mem_union, mem_Ioo]
      by_cases hx : x < ε
      · left; left; exact ⟨by linarith, hx⟩
      · rcases eq_or_lt_of_le h1 with rfl | h1'
        · left; right; exact ⟨by linarith, by linarith⟩
        · right; exact ⟨by linarith, h1'⟩
  exact @key (2) (1) ((by norm_num)) ((by norm_num)) (1) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Compact.closed_interval_cover
-- test: set-closed-cover-test.sage

/-! ## exr-compact-extremes -/

/-- `f'(x) = (1 - x) e^{-x}` for `f(x) = x e^{-x}`. -/
theorem deriv_x_exp (x : ℝ) : deriv (fun x : ℝ => x * Real.exp (-x)) x = (1 - x) * Real.exp (-x) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x * Real.exp (-x)) ((-x * (Real.exp (-x)) + (Real.exp (-x)) : ℝ)) (x) :=
        by
        convert ((hasDerivAt_id' (x : ℝ)).mul (((hasDerivAt_id' (x : ℝ)).const_mul (-1 : ℝ)).exp)) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.mul_apply]; ring_nf; done)
          | ((ring_nf; done))
      exact hd.deriv.trans (by ring1))
  -- certificate: set-deriv.sage

/-- `x e^{-x} ≤ e^{-1}` for every `x`, with equality at `x = 1`: the maximum on `[0, 2]` (and on `(0, 2)`) is `e^{-1}`,
at `1`. -/
theorem x_exp_le (x : ℝ) : x * Real.exp (-x) ≤ Real.exp (-1) ∧ 1 * Real.exp (-1) = Real.exp (-1) := by
  refine ⟨?_, one_mul _⟩
  have h := Real.add_one_le_exp (x - 1)
  have hx : Real.exp (-x) = Real.exp (-1) * Real.exp (-(x - 1)) := by rw [← Real.exp_add]; ring_nf
  rw [hx]
  have hpos := Real.exp_pos (-(x - 1))
  have : x * Real.exp (-(x - 1)) ≤ 1 := by
    rw [Real.exp_neg]
    rw [mul_inv_le_iff₀ (Real.exp_pos _)]
    linarith
  nlinarith [Real.exp_pos (-1)]

example : veriphy_instance% KaluzaKlein.Sets.Compact.x_exp_le (1) := by
  have key : ∀ (x : ℝ), x * Real.exp (-x) ≤ Real.exp (-1) ∧ 1 * Real.exp (-1) = Real.exp (-1) := by
    intro x
    refine ⟨?_, one_mul _⟩
    have h := Real.add_one_le_exp (x - 1)
    have hx : Real.exp (-x) = Real.exp (-1) * Real.exp (-(x - 1)) := by rw [← Real.exp_add]; ring_nf
    rw [hx]
    have hpos := Real.exp_pos (-(x - 1))
    have : x * Real.exp (-(x - 1)) ≤ 1 := by
      rw [Real.exp_neg]
      rw [mul_inv_le_iff₀ (Real.exp_pos _)]
      linarith
    nlinarith [Real.exp_pos (-1)]
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Compact.x_exp_le (0) := by
  have key : ∀ (x : ℝ), x * Real.exp (-x) ≤ Real.exp (-1) ∧ 1 * Real.exp (-1) = Real.exp (-1) := by
    intro x
    refine ⟨?_, one_mul _⟩
    have h := Real.add_one_le_exp (x - 1)
    have hx : Real.exp (-x) = Real.exp (-1) * Real.exp (-(x - 1)) := by rw [← Real.exp_add]; ring_nf
    rw [hx]
    have hpos := Real.exp_pos (-(x - 1))
    have : x * Real.exp (-(x - 1)) ≤ 1 := by
      rw [Real.exp_neg]
      rw [mul_inv_le_iff₀ (Real.exp_pos _)]
      linarith
    nlinarith [Real.exp_pos (-1)]
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Compact.x_exp_le (2) := by
  have key : ∀ (x : ℝ), x * Real.exp (-x) ≤ Real.exp (-1) ∧ 1 * Real.exp (-1) = Real.exp (-1) := by
    intro x
    refine ⟨?_, one_mul _⟩
    have h := Real.add_one_le_exp (x - 1)
    have hx : Real.exp (-x) = Real.exp (-1) * Real.exp (-(x - 1)) := by rw [← Real.exp_add]; ring_nf
    rw [hx]
    have hpos := Real.exp_pos (-(x - 1))
    have : x * Real.exp (-(x - 1)) ≤ 1 := by
      rw [Real.exp_neg]
      rw [mul_inv_le_iff₀ (Real.exp_pos _)]
      linarith
    nlinarith [Real.exp_pos (-1)]
  exact @key (2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Compact.x_exp_le
-- test: set-x-exp-le-test.sage

/-- On `[0, ∞)`, `x e^{-x} ≥ 0 = f(0)`: the minimum on `[0, 2]` is `f(0) = 0`. -/
theorem x_exp_nonneg (x : ℝ) (hx : 0 ≤ x) : 0 ≤ x * Real.exp (-x) ∧ 0 * Real.exp (-0) = (0 : ℝ) :=
  ⟨mul_nonneg hx (Real.exp_pos _).le, by simp⟩

example : veriphy_instance% KaluzaKlein.Sets.Compact.x_exp_nonneg (0) (le_rfl) := by
  have key : ∀ (x : ℝ) (hx : 0 ≤ x), 0 ≤ x * Real.exp (-x) ∧ 0 * Real.exp (-0) = (0 : ℝ) := by
    intro x hx
    exact (⟨mul_nonneg hx (Real.exp_pos _).le, by simp⟩)
  exact @key (0) (le_rfl)

example : veriphy_instance% KaluzaKlein.Sets.Compact.x_exp_nonneg (1) ((by norm_num)) := by
  have key : ∀ (x : ℝ) (hx : 0 ≤ x), 0 ≤ x * Real.exp (-x) ∧ 0 * Real.exp (-0) = (0 : ℝ) := by
    intro x hx
    exact (⟨mul_nonneg hx (Real.exp_pos _).le, by simp⟩)
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Compact.x_exp_nonneg (2) ((by norm_num)) := by
  have key : ∀ (x : ℝ) (hx : 0 ≤ x), 0 ≤ x * Real.exp (-x) ∧ 0 * Real.exp (-0) = (0 : ℝ) := by
    intro x hx
    exact (⟨mul_nonneg hx (Real.exp_pos _).le, by simp⟩)
  exact @key (2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Compact.x_exp_nonneg
-- test: set-x-exp-nonneg-test.sage

/-- On `(0, 2)`, `f > 0` and no value is a minimum: below every value there is a smaller one, and the infimum `0` is
not attained. -/
theorem x_exp_no_min (x : ℝ) (hx : x ∈ Ioo (0 : ℝ) 2) (ε : ℝ) (hε : 0 < ε) :
    0 < x * Real.exp (-x) ∧ (∃ y ∈ Ioo (0 : ℝ) 2, y * Real.exp (-y) < x * Real.exp (-x)) ∧
      ∃ y ∈ Ioo (0 : ℝ) 2, y * Real.exp (-y) < ε := by
  have hf : ∀ y : ℝ, 0 < y → y * Real.exp (-y) < y := fun y hy => by
    have : Real.exp (-y) < 1 := Real.exp_lt_one_iff.2 (by linarith)
    nlinarith
  have hpos : 0 < x * Real.exp (-x) := mul_pos hx.1 (Real.exp_pos _)
  refine ⟨hpos, ⟨min (x * Real.exp (-x)) 1 / 2, ⟨by positivity, ?_⟩, ?_⟩,
    ⟨min ε 1 / 2, ⟨by positivity, ?_⟩, ?_⟩⟩
  · have := min_le_right (x * Real.exp (-x)) 1; linarith
  · have h1 := hf (min (x * Real.exp (-x)) 1 / 2) (by positivity)
    have h2 := min_le_left (x * Real.exp (-x)) 1
    linarith
  · have := min_le_right ε 1; linarith
  · have h1 := hf (min ε 1 / 2) (by positivity)
    have h2 := min_le_left ε 1
    linarith

example : veriphy_instance% KaluzaKlein.Sets.Compact.x_exp_no_min (1) ((by constructor <;> norm_num)) (1/2) ((by norm_num)) := by
  have key : ∀ (x : ℝ) (hx : x ∈ Ioo (0 : ℝ) 2) (ε : ℝ) (hε : 0 < ε), 0 < x * Real.exp (-x) ∧ (∃ y ∈ Ioo (0 : ℝ) 2, y * Real.exp (-y) < x * Real.exp (-x)) ∧ ∃ y ∈ Ioo (0 : ℝ) 2, y * Real.exp (-y) < ε := by
    intro x hx ε hε
    have hf : ∀ y : ℝ, 0 < y → y * Real.exp (-y) < y := fun y hy => by
      have : Real.exp (-y) < 1 := Real.exp_lt_one_iff.2 (by linarith)
      nlinarith
    have hpos : 0 < x * Real.exp (-x) := mul_pos hx.1 (Real.exp_pos _)
    refine ⟨hpos, ⟨min (x * Real.exp (-x)) 1 / 2, ⟨by positivity, ?_⟩, ?_⟩,
      ⟨min ε 1 / 2, ⟨by positivity, ?_⟩, ?_⟩⟩
    · have := min_le_right (x * Real.exp (-x)) 1; linarith
    · have h1 := hf (min (x * Real.exp (-x)) 1 / 2) (by positivity)
      have h2 := min_le_left (x * Real.exp (-x)) 1
      linarith
    · have := min_le_right ε 1; linarith
    · have h1 := hf (min ε 1 / 2) (by positivity)
      have h2 := min_le_left ε 1
      linarith
  exact @key (1) ((by constructor <;> norm_num)) (1/2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Compact.x_exp_no_min (1/10) ((by constructor <;> norm_num)) (1) ((by norm_num)) := by
  have key : ∀ (x : ℝ) (hx : x ∈ Ioo (0 : ℝ) 2) (ε : ℝ) (hε : 0 < ε), 0 < x * Real.exp (-x) ∧ (∃ y ∈ Ioo (0 : ℝ) 2, y * Real.exp (-y) < x * Real.exp (-x)) ∧ ∃ y ∈ Ioo (0 : ℝ) 2, y * Real.exp (-y) < ε := by
    intro x hx ε hε
    have hf : ∀ y : ℝ, 0 < y → y * Real.exp (-y) < y := fun y hy => by
      have : Real.exp (-y) < 1 := Real.exp_lt_one_iff.2 (by linarith)
      nlinarith
    have hpos : 0 < x * Real.exp (-x) := mul_pos hx.1 (Real.exp_pos _)
    refine ⟨hpos, ⟨min (x * Real.exp (-x)) 1 / 2, ⟨by positivity, ?_⟩, ?_⟩,
      ⟨min ε 1 / 2, ⟨by positivity, ?_⟩, ?_⟩⟩
    · have := min_le_right (x * Real.exp (-x)) 1; linarith
    · have h1 := hf (min (x * Real.exp (-x)) 1 / 2) (by positivity)
      have h2 := min_le_left (x * Real.exp (-x)) 1
      linarith
    · have := min_le_right ε 1; linarith
    · have h1 := hf (min ε 1 / 2) (by positivity)
      have h2 := min_le_left ε 1
      linarith
  exact @key (1/10) ((by constructor <;> norm_num)) (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Compact.x_exp_no_min (19/10) ((by constructor <;> norm_num)) (1/100) ((by norm_num)) := by
  have key : ∀ (x : ℝ) (hx : x ∈ Ioo (0 : ℝ) 2) (ε : ℝ) (hε : 0 < ε), 0 < x * Real.exp (-x) ∧ (∃ y ∈ Ioo (0 : ℝ) 2, y * Real.exp (-y) < x * Real.exp (-x)) ∧ ∃ y ∈ Ioo (0 : ℝ) 2, y * Real.exp (-y) < ε := by
    intro x hx ε hε
    have hf : ∀ y : ℝ, 0 < y → y * Real.exp (-y) < y := fun y hy => by
      have : Real.exp (-y) < 1 := Real.exp_lt_one_iff.2 (by linarith)
      nlinarith
    have hpos : 0 < x * Real.exp (-x) := mul_pos hx.1 (Real.exp_pos _)
    refine ⟨hpos, ⟨min (x * Real.exp (-x)) 1 / 2, ⟨by positivity, ?_⟩, ?_⟩,
      ⟨min ε 1 / 2, ⟨by positivity, ?_⟩, ?_⟩⟩
    · have := min_le_right (x * Real.exp (-x)) 1; linarith
    · have h1 := hf (min (x * Real.exp (-x)) 1 / 2) (by positivity)
      have h2 := min_le_left (x * Real.exp (-x)) 1
      linarith
    · have := min_le_right ε 1; linarith
    · have h1 := hf (min ε 1 / 2) (by positivity)
      have h2 := min_le_left ε 1
      linarith
  exact @key (19/10) ((by constructor <;> norm_num)) (1/100) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Compact.x_exp_no_min
-- test: set-x-exp-no-min-test.sage

/-! ## exr-compact-closed-subset -/

/-- A closed subset of a compact set is compact: the circle, closed in the compact closed disc. -/
theorem closed_subset_compact (E : Type) [TopologicalSpace E] (K C : Set E) (hK : IsCompact K) (hC : IsClosed C)
    (hCK : C ⊆ K) : IsCompact C :=
  hK.of_isClosed_subset hC hCK

example : veriphy_instance% KaluzaKlein.Sets.Compact.closed_subset_compact (ℝ × ℝ) (inferInstance) (KaluzaKlein.Sets.Sets.closedDisc) (KaluzaKlein.Sets.Sets.circle) (KaluzaKlein.Sets.Compact.closedDisc_compact) (KaluzaKlein.Sets.Compact.circle_closed_bounded.1) ((fun p hp => le_of_eq hp)) := by
  have key : ∀ (E : Type) [inst1 : TopologicalSpace E] (K C : Set E) (hK : IsCompact K) (hC : IsClosed C) (hCK : C ⊆ K), IsCompact C := by
    intro E inst1 K C hK hC hCK
    exact (hK.of_isClosed_subset hC hCK)
  exact @key (ℝ × ℝ) (inferInstance) (KaluzaKlein.Sets.Sets.closedDisc) (KaluzaKlein.Sets.Sets.circle) (KaluzaKlein.Sets.Compact.closedDisc_compact) (KaluzaKlein.Sets.Compact.circle_closed_bounded.1) ((fun p hp => le_of_eq hp))

example : veriphy_instance% KaluzaKlein.Sets.Compact.closed_subset_compact (ℝ) (inferInstance) (Set.Icc 0 1) (∅) (isCompact_Icc) (isClosed_empty) ((Set.empty_subset _)) := by
  have key : ∀ (E : Type) [inst1 : TopologicalSpace E] (K C : Set E) (hK : IsCompact K) (hC : IsClosed C) (hCK : C ⊆ K), IsCompact C := by
    intro E inst1 K C hK hC hCK
    exact (hK.of_isClosed_subset hC hCK)
  exact @key (ℝ) (inferInstance) (Set.Icc 0 1) (∅) (isCompact_Icc) (isClosed_empty) ((Set.empty_subset _))

example : veriphy_instance% KaluzaKlein.Sets.Compact.closed_subset_compact (ℝ) (inferInstance) (Set.Icc 0 2) (Set.Icc 0 1) (isCompact_Icc) (isClosed_Icc) ((Set.Icc_subset_Icc le_rfl (by norm_num))) := by
  have key : ∀ (E : Type) [inst1 : TopologicalSpace E] (K C : Set E) (hK : IsCompact K) (hC : IsClosed C) (hCK : C ⊆ K), IsCompact C := by
    intro E inst1 K C hK hC hCK
    exact (hK.of_isClosed_subset hC hCK)
  exact @key (ℝ) (inferInstance) (Set.Icc 0 2) (Set.Icc 0 1) (isCompact_Icc) (isClosed_Icc) ((Set.Icc_subset_Icc le_rfl (by norm_num)))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Compact.closed_subset_compact
-- test: set-closed-subset-test.sage

end KaluzaKlein.Sets.Compact
