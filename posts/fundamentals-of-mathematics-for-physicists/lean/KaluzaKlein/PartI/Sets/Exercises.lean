import Mathlib
import KaluzaKlein.PartI.Sets.Connected
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Exercises"

The claims of the three closing exercises of `tutorial/part-i-01-sets-relations-topology.qmd`: `ℝ/Tℤ` is compact and
Hausdorff though `ℝ` is not compact, a continuous bijection from a compact space onto a Hausdorff space is a
homeomorphism (which gives `SO(3)/SO(2) ≅ S²`), and continuous periodic functions descend to continuous functions on
the circle.
-/

namespace KaluzaKlein.Sets.Exercises

open Set Topology Matrix KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Quotients

/-! ## exr-circle-quotient-compact -/

/-- `ℝ/Tℤ` (`T > 0`) with the quotient topology is compact and Hausdorff, although `ℝ` is not compact (`T = 2π`). -/
theorem circle_quotient_compact_hausdorff (T : ℝ) (hT : 0 < T) :
    CompactSpace (Quotient (angleSetoid T)) ∧ T2Space (Quotient (angleSetoid T)) ∧ ¬ CompactSpace ℝ := by
  obtain ⟨e, -⟩ := QuotientTopology.circle_quotient_homeomorph T hT
  haveI : CompactSpace circle := isCompact_iff_compactSpace.1 (by
    refine Metric.isCompact_of_isClosed_isBounded (isClosed_eq (by fun_prop) continuous_const) ?_
    refine (Metric.isBounded_closedBall (x := (0 : ℝ × ℝ)) (r := 1)).subset fun p hp => ?_
    simp only [circle, mem_setOf_eq] at hp
    rw [Metric.mem_closedBall, dist_zero_right, Prod.norm_def, max_le_iff, Real.norm_eq_abs, Real.norm_eq_abs]
    constructor <;> rw [abs_le] <;> constructor <;> nlinarith [sq_nonneg p.1, sq_nonneg p.2])
  exact ⟨e.symm.compactSpace, e.isEmbedding.t2Space, fun h => noncompact_univ ℝ isCompact_univ⟩

example : veriphy_instance% KaluzaKlein.Sets.Exercises.circle_quotient_compact_hausdorff (2 * Real.pi) ((by positivity)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), CompactSpace (Quotient (angleSetoid T)) ∧ T2Space (Quotient (angleSetoid T)) ∧ ¬ CompactSpace ℝ := by
    intro T hT
    obtain ⟨e, -⟩ := QuotientTopology.circle_quotient_homeomorph T hT
    haveI : CompactSpace circle := isCompact_iff_compactSpace.1 (by
      refine Metric.isCompact_of_isClosed_isBounded (isClosed_eq (by fun_prop) continuous_const) ?_
      refine (Metric.isBounded_closedBall (x := (0 : ℝ × ℝ)) (r := 1)).subset fun p hp => ?_
      simp only [circle, mem_setOf_eq] at hp
      rw [Metric.mem_closedBall, dist_zero_right, Prod.norm_def, max_le_iff, Real.norm_eq_abs, Real.norm_eq_abs]
      constructor <;> rw [abs_le] <;> constructor <;> nlinarith [sq_nonneg p.1, sq_nonneg p.2])
    exact ⟨e.symm.compactSpace, e.isEmbedding.t2Space, fun h => noncompact_univ ℝ isCompact_univ⟩
  exact @key (2 * Real.pi) ((by positivity))

example : veriphy_instance% KaluzaKlein.Sets.Exercises.circle_quotient_compact_hausdorff (1) ((by norm_num)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), CompactSpace (Quotient (angleSetoid T)) ∧ T2Space (Quotient (angleSetoid T)) ∧ ¬ CompactSpace ℝ := by
    intro T hT
    obtain ⟨e, -⟩ := QuotientTopology.circle_quotient_homeomorph T hT
    haveI : CompactSpace circle := isCompact_iff_compactSpace.1 (by
      refine Metric.isCompact_of_isClosed_isBounded (isClosed_eq (by fun_prop) continuous_const) ?_
      refine (Metric.isBounded_closedBall (x := (0 : ℝ × ℝ)) (r := 1)).subset fun p hp => ?_
      simp only [circle, mem_setOf_eq] at hp
      rw [Metric.mem_closedBall, dist_zero_right, Prod.norm_def, max_le_iff, Real.norm_eq_abs, Real.norm_eq_abs]
      constructor <;> rw [abs_le] <;> constructor <;> nlinarith [sq_nonneg p.1, sq_nonneg p.2])
    exact ⟨e.symm.compactSpace, e.isEmbedding.t2Space, fun h => noncompact_univ ℝ isCompact_univ⟩
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Exercises.circle_quotient_compact_hausdorff (3) ((by norm_num)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), CompactSpace (Quotient (angleSetoid T)) ∧ T2Space (Quotient (angleSetoid T)) ∧ ¬ CompactSpace ℝ := by
    intro T hT
    obtain ⟨e, -⟩ := QuotientTopology.circle_quotient_homeomorph T hT
    haveI : CompactSpace circle := isCompact_iff_compactSpace.1 (by
      refine Metric.isCompact_of_isClosed_isBounded (isClosed_eq (by fun_prop) continuous_const) ?_
      refine (Metric.isBounded_closedBall (x := (0 : ℝ × ℝ)) (r := 1)).subset fun p hp => ?_
      simp only [circle, mem_setOf_eq] at hp
      rw [Metric.mem_closedBall, dist_zero_right, Prod.norm_def, max_le_iff, Real.norm_eq_abs, Real.norm_eq_abs]
      constructor <;> rw [abs_le] <;> constructor <;> nlinarith [sq_nonneg p.1, sq_nonneg p.2])
    exact ⟨e.symm.compactSpace, e.isEmbedding.t2Space, fun h => noncompact_univ ℝ isCompact_univ⟩
  exact @key (3) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Exercises.circle_quotient_compact_hausdorff
-- test: set-circle-compact-test.sage

/-! ## exr-sphere-quotient-homeomorphism -/

/-- A continuous bijection from a compact space onto a Hausdorff space is a homeomorphism: it sends closed sets to
closed sets, so its inverse is continuous. -/
theorem compact_to_hausdorff (X Y : Type) [TopologicalSpace X] [TopologicalSpace Y] [CompactSpace X] [T2Space Y]
    (f : X → Y) (hf : Continuous f) (hb : Function.Bijective f) :
    IsClosedMap f ∧ ∃ e : X ≃ₜ Y, ∀ x, e x = f x :=
  ⟨hf.isClosedMap, ⟨hf.homeoOfEquivCompactToT2 (f := Equiv.ofBijective f hb), fun _ => rfl⟩⟩

example : veriphy_instance% KaluzaKlein.Sets.Exercises.compact_to_hausdorff (Unit) (Unit) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (id) (continuous_id) (Function.bijective_id) := by
  have key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] [inst3 : CompactSpace X] [inst4 : T2Space Y] (f : X → Y) (hf : Continuous f) (hb : Function.Bijective f), IsClosedMap f ∧ ∃ e : X ≃ₜ Y, ∀ x, e x = f x := by
    intro X Y inst1 inst2 inst3 inst4 f hf hb
    exact (⟨hf.isClosedMap, ⟨hf.homeoOfEquivCompactToT2 (f := Equiv.ofBijective f hb), fun _ => rfl⟩⟩)
  exact @key (Unit) (Unit) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (id) (continuous_id) (Function.bijective_id)

example : veriphy_instance% KaluzaKlein.Sets.Exercises.compact_to_hausdorff (Fin 3) (Fin 3) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (![1, 2, 0]) (continuous_of_discreteTopology) ((by decide)) := by
  have key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] [inst3 : CompactSpace X] [inst4 : T2Space Y] (f : X → Y) (hf : Continuous f) (hb : Function.Bijective f), IsClosedMap f ∧ ∃ e : X ≃ₜ Y, ∀ x, e x = f x := by
    intro X Y inst1 inst2 inst3 inst4 f hf hb
    exact (⟨hf.isClosedMap, ⟨hf.homeoOfEquivCompactToT2 (f := Equiv.ofBijective f hb), fun _ => rfl⟩⟩)
  exact @key (Fin 3) (Fin 3) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (![1, 2, 0]) (continuous_of_discreteTopology) ((by decide))

example : veriphy_instance% KaluzaKlein.Sets.Exercises.compact_to_hausdorff (Bool) (Bool) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (not) (continuous_of_discreteTopology) ((by decide)) := by
  have key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] [inst3 : CompactSpace X] [inst4 : T2Space Y] (f : X → Y) (hf : Continuous f) (hb : Function.Bijective f), IsClosedMap f ∧ ∃ e : X ≃ₜ Y, ∀ x, e x = f x := by
    intro X Y inst1 inst2 inst3 inst4 f hf hb
    exact (⟨hf.isClosedMap, ⟨hf.homeoOfEquivCompactToT2 (f := Equiv.ofBijective f hb), fun _ => rfl⟩⟩)
  exact @key (Bool) (Bool) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (not) (continuous_of_discreteTopology) ((by decide))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Exercises.compact_to_hausdorff
-- test: set-compact-to-hausdorff-test.sage

/-! ## exr-descend-continuity -/

/-- A continuous `T`-periodic function on `ℝ` descends to a continuous function on `ℝ/Tℤ`, with `f = f̄ ∘ π` (`cos`
and the circle, `T = 2π`). -/
theorem periodic_descends_continuous (T : ℝ) (f : ℝ → ℝ) (hf : Continuous f) (hp : Function.Periodic f T) :
    ∃ g : Quotient (angleSetoid T) → ℝ, Continuous g ∧ ∀ x, g (Quotient.mk _ x) = f x := by
  have hc : ∀ a b, angleSetoid T a b → f a = f b := fun a b h => (descends_iff_periodic T f).2 hp a b h
  refine ⟨Quotient.lift f hc, ?_, fun x => rfl⟩
  rw [isQuotientMap_quotient_mk'.continuous_iff]
  exact hf

example : veriphy_instance% KaluzaKlein.Sets.Exercises.periodic_descends_continuous (2 * Real.pi) (Real.cos) (Real.continuous_cos) (Real.cos_periodic) := by
  have key : ∀ (T : ℝ) (f : ℝ → ℝ) (hf : Continuous f) (hp : Function.Periodic f T), ∃ g : Quotient (angleSetoid T) → ℝ, Continuous g ∧ ∀ x, g (Quotient.mk _ x) = f x := by
    intro T f hf hp
    have hc : ∀ a b, angleSetoid T a b → f a = f b := fun a b h => (descends_iff_periodic T f).2 hp a b h
    refine ⟨Quotient.lift f hc, ?_, fun x => rfl⟩
    rw [isQuotientMap_quotient_mk'.continuous_iff]
    exact hf
  exact @key (2 * Real.pi) (Real.cos) (Real.continuous_cos) (Real.cos_periodic)

example : veriphy_instance% KaluzaKlein.Sets.Exercises.periodic_descends_continuous (1) (fun _ => 0) (continuous_const) ((fun _ => rfl)) := by
  have key : ∀ (T : ℝ) (f : ℝ → ℝ) (hf : Continuous f) (hp : Function.Periodic f T), ∃ g : Quotient (angleSetoid T) → ℝ, Continuous g ∧ ∀ x, g (Quotient.mk _ x) = f x := by
    intro T f hf hp
    have hc : ∀ a b, angleSetoid T a b → f a = f b := fun a b h => (descends_iff_periodic T f).2 hp a b h
    refine ⟨Quotient.lift f hc, ?_, fun x => rfl⟩
    rw [isQuotientMap_quotient_mk'.continuous_iff]
    exact hf
  exact @key (1) (fun _ => 0) (continuous_const) ((fun _ => rfl))

example : veriphy_instance% KaluzaKlein.Sets.Exercises.periodic_descends_continuous (2 * Real.pi) (Real.sin) (Real.continuous_sin) (Real.sin_periodic) := by
  have key : ∀ (T : ℝ) (f : ℝ → ℝ) (hf : Continuous f) (hp : Function.Periodic f T), ∃ g : Quotient (angleSetoid T) → ℝ, Continuous g ∧ ∀ x, g (Quotient.mk _ x) = f x := by
    intro T f hf hp
    have hc : ∀ a b, angleSetoid T a b → f a = f b := fun a b h => (descends_iff_periodic T f).2 hp a b h
    refine ⟨Quotient.lift f hc, ?_, fun x => rfl⟩
    rw [isQuotientMap_quotient_mk'.continuous_iff]
    exact hf
  exact @key (2 * Real.pi) (Real.sin) (Real.continuous_sin) (Real.sin_periodic)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Exercises.periodic_descends_continuous
-- test: set-periodic-descends-test.sage

end KaluzaKlein.Sets.Exercises
