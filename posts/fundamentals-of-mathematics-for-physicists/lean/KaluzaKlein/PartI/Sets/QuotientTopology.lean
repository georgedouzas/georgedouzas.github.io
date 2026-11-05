import Mathlib
import KaluzaKlein.PartI.Sets.Homeomorphism
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Topological spaces" (quotient topology)

The claims of the blocks `def-quotient-topology` to `exr-quotient-moebius` of
`tutorial/part-i-01-sets-relations-topology.qmd`: continuity of maps out of a quotient, `ℝ/Tℤ`, `SO(3)/SO(2)` and the
square with glued edges as the circle, the sphere and the torus (continuous bijections from compact spaces onto
Hausdorff ones), the circle from an interval, the open sets of `ℝ/Tℤ` and the openness of its quotient map, and the
boundaries of the Möbius band and the cylinder. `SO(3)` is shown compact here (closed and bounded).
-/

namespace KaluzaKlein.Sets.QuotientTopology

open Set Metric Matrix KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Maps KaluzaKlein.Sets.Quotients

/-! ## def-quotient-topology -/

/-- A map out of a quotient is continuous exactly when its composition with the quotient map is. -/
theorem quotient_continuous_iff (X Y : Type) [TopologicalSpace X] [TopologicalSpace Y] (s : Setoid X)
    (g : Quotient s → Y) : Continuous g ↔ Continuous (g ∘ Quotient.mk s) :=
  isQuotientMap_quotient_mk'.continuous_iff

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.quotient_continuous_iff (Unit) (Unit) (inferInstance) (inferInstance) (⊥) (fun _ => ()) := by
  have key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] (s : Setoid X) (g : Quotient s → Y), Continuous g ↔ Continuous (g ∘ Quotient.mk s) := by
    intro X Y inst1 inst2 s g
    exact (isQuotientMap_quotient_mk'.continuous_iff)
  exact @key (Unit) (Unit) (inferInstance) (inferInstance) (⊥) (fun _ => ())

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.quotient_continuous_iff (ℝ) (ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Quotients.angleSetoid 1) (fun _ => 0) := by
  have key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] (s : Setoid X) (g : Quotient s → Y), Continuous g ↔ Continuous (g ∘ Quotient.mk s) := by
    intro X Y inst1 inst2 s g
    exact (isQuotientMap_quotient_mk'.continuous_iff)
  exact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Quotients.angleSetoid 1) (fun _ => 0)

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.quotient_continuous_iff (ℝ) (ℝ) (inferInstance) (inferInstance) (⊤) (fun _ => 1) := by
  have key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] (s : Setoid X) (g : Quotient s → Y), Continuous g ↔ Continuous (g ∘ Quotient.mk s) := by
    intro X Y inst1 inst2 s g
    exact (isQuotientMap_quotient_mk'.continuous_iff)
  exact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (⊤) (fun _ => 1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.QuotientTopology.quotient_continuous_iff
-- test: set-quotient-continuous-test.sage

/-! ## exm-quotient-circle-top -/

/-- `ℝ/Tℤ` (`T > 0`) is homeomorphic to the circle by `[φ] ↦ (cos(2πφ/T), sin(2πφ/T))`: a continuous bijection from
the compact quotient onto the Hausdorff circle (for `T = 2π`, `[φ] ↦ (cos φ, sin φ)`). -/
theorem circle_quotient_homeomorph (T : ℝ) (hT : 0 < T) :
    ∃ e : Quotient (angleSetoid T) ≃ₜ circle, ∀ φ, (e ⟦φ⟧ : ℝ × ℝ) = wrap (2 * Real.pi * φ / T) := by
  obtain ⟨F, hF, hFφ⟩ := circle_quotient_bijective T hT
  have hFc : Continuous F := by
    rw [isQuotientMap_quotient_mk'.continuous_iff]
    refine Continuous.subtype_mk ?_ _
    convert (show Continuous fun φ : ℝ => wrap (2 * Real.pi * φ / T) by unfold wrap; fun_prop) using 1
    funext φ
    exact hFφ φ
  haveI : CompactSpace (Quotient (angleSetoid T)) := by
    refine ⟨?_⟩
    have himg : (Quotient.mk (angleSetoid T)) '' Icc 0 T = univ := by
      refine eq_univ_of_forall fun q => ?_
      induction q using Quotient.ind
      rename_i φ
      have hm := toIcoMod_mem_Ico hT 0 φ
      rw [zero_add] at hm
      refine ⟨toIcoMod hT 0 φ, Ico_subset_Icc_self hm, Quotient.sound ?_⟩
      refine ⟨toIcoDiv hT 0 φ, ?_⟩
      rw [toIcoMod, zsmul_eq_mul]; ring
    rw [← himg]
    exact isCompact_Icc.image continuous_quotient_mk'
  exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F hF), fun φ => hFφ φ⟩

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.circle_quotient_homeomorph (2 * Real.pi) ((by positivity)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), ∃ e : Quotient (angleSetoid T) ≃ₜ circle, ∀ φ, (e ⟦φ⟧ : ℝ × ℝ) = wrap (2 * Real.pi * φ / T) := by
    intro T hT
    obtain ⟨F, hF, hFφ⟩ := circle_quotient_bijective T hT
    have hFc : Continuous F := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      refine Continuous.subtype_mk ?_ _
      convert (show Continuous fun φ : ℝ => wrap (2 * Real.pi * φ / T) by unfold wrap; fun_prop) using 1
      funext φ
      exact hFφ φ
    haveI : CompactSpace (Quotient (angleSetoid T)) := by
      refine ⟨?_⟩
      have himg : (Quotient.mk (angleSetoid T)) '' Icc 0 T = univ := by
        refine eq_univ_of_forall fun q => ?_
        induction q using Quotient.ind
        rename_i φ
        have hm := toIcoMod_mem_Ico hT 0 φ
        rw [zero_add] at hm
        refine ⟨toIcoMod hT 0 φ, Ico_subset_Icc_self hm, Quotient.sound ?_⟩
        refine ⟨toIcoDiv hT 0 φ, ?_⟩
        rw [toIcoMod, zsmul_eq_mul]; ring
      rw [← himg]
      exact isCompact_Icc.image continuous_quotient_mk'
    exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F hF), fun φ => hFφ φ⟩
  exact @key (2 * Real.pi) ((by positivity))

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.circle_quotient_homeomorph (1) ((by norm_num)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), ∃ e : Quotient (angleSetoid T) ≃ₜ circle, ∀ φ, (e ⟦φ⟧ : ℝ × ℝ) = wrap (2 * Real.pi * φ / T) := by
    intro T hT
    obtain ⟨F, hF, hFφ⟩ := circle_quotient_bijective T hT
    have hFc : Continuous F := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      refine Continuous.subtype_mk ?_ _
      convert (show Continuous fun φ : ℝ => wrap (2 * Real.pi * φ / T) by unfold wrap; fun_prop) using 1
      funext φ
      exact hFφ φ
    haveI : CompactSpace (Quotient (angleSetoid T)) := by
      refine ⟨?_⟩
      have himg : (Quotient.mk (angleSetoid T)) '' Icc 0 T = univ := by
        refine eq_univ_of_forall fun q => ?_
        induction q using Quotient.ind
        rename_i φ
        have hm := toIcoMod_mem_Ico hT 0 φ
        rw [zero_add] at hm
        refine ⟨toIcoMod hT 0 φ, Ico_subset_Icc_self hm, Quotient.sound ?_⟩
        refine ⟨toIcoDiv hT 0 φ, ?_⟩
        rw [toIcoMod, zsmul_eq_mul]; ring
      rw [← himg]
      exact isCompact_Icc.image continuous_quotient_mk'
    exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F hF), fun φ => hFφ φ⟩
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.circle_quotient_homeomorph (3) ((by norm_num)) := by
  have key : ∀ (T : ℝ) (hT : 0 < T), ∃ e : Quotient (angleSetoid T) ≃ₜ circle, ∀ φ, (e ⟦φ⟧ : ℝ × ℝ) = wrap (2 * Real.pi * φ / T) := by
    intro T hT
    obtain ⟨F, hF, hFφ⟩ := circle_quotient_bijective T hT
    have hFc : Continuous F := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      refine Continuous.subtype_mk ?_ _
      convert (show Continuous fun φ : ℝ => wrap (2 * Real.pi * φ / T) by unfold wrap; fun_prop) using 1
      funext φ
      exact hFφ φ
    haveI : CompactSpace (Quotient (angleSetoid T)) := by
      refine ⟨?_⟩
      have himg : (Quotient.mk (angleSetoid T)) '' Icc 0 T = univ := by
        refine eq_univ_of_forall fun q => ?_
        induction q using Quotient.ind
        rename_i φ
        have hm := toIcoMod_mem_Ico hT 0 φ
        rw [zero_add] at hm
        refine ⟨toIcoMod hT 0 φ, Ico_subset_Icc_self hm, Quotient.sound ?_⟩
        refine ⟨toIcoDiv hT 0 φ, ?_⟩
        rw [toIcoMod, zsmul_eq_mul]; ring
      rw [← himg]
      exact isCompact_Icc.image continuous_quotient_mk'
    exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F hF), fun φ => hFφ φ⟩
  exact @key (3) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.QuotientTopology.circle_quotient_homeomorph
-- test: set-circle-homeomorph-test.sage

/-! ## exm-quotient-sphere-top -/

/-- `SO(3)` is compact: closed in the `3 × 3` matrices, with entries in `[-1, 1]`. -/
theorem SO3_compact : IsCompact SO3 := by
  have hbox : IsCompact (Set.pi univ fun _ : Fin 3 => Set.pi univ fun _ : Fin 3 => Icc (-1 : ℝ) 1) :=
    isCompact_univ_pi fun _ => isCompact_univ_pi fun _ => isCompact_Icc
  have hcl : IsClosed SO3 := by
    have : SO3 = {A : Matrix (Fin 3) (Fin 3) ℝ | Aᵀ * A = 1} ∩ {A | A.det = 1} := by
      ext A; exact mem_SO3_iff A
    rw [this]
    exact (isClosed_eq ((continuous_id.matrix_transpose).matrix_mul continuous_id) continuous_const).inter
      (isClosed_eq continuous_id.matrix_det continuous_const)
  have hsub : ∀ A ∈ SO3, ∀ i j, A i j ∈ Icc (-1 : ℝ) 1 := by
    intro A hA i j
    rw [mem_SO3_iff] at hA
    have h := congrFun (congrFun hA.1 j) j
    simp [Matrix.mul_apply, Fin.sum_univ_succ] at h
    have : A i j ^ 2 ≤ 1 := by
      have hi : i = 0 ∨ i = 1 ∨ i = 2 := by fin_cases i <;> simp
      rcases hi with rfl | rfl | rfl <;> nlinarith [sq_nonneg (A 0 j), sq_nonneg (A 1 j), sq_nonneg (A 2 j)]
    constructor <;> nlinarith [sq_nonneg (A i j + 1), sq_nonneg (A i j - 1)]
  exact hbox.of_isClosed_subset hcl (fun A hA => by
    simp only [Set.mem_pi, Set.mem_univ, true_implies]
    exact hsub A hA)

/-- For `p₀` on the sphere, the quotient of `SO(3)` by "same image of `p₀`" is homeomorphic to the sphere by
`[g] ↦ g · p₀`: a continuous bijection from a compact space onto a Hausdorff one (for the north pole, this is
`SO(3)/SO(2) ≅ S²`). -/
theorem sphere_quotient_homeomorph (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere) :
    ∃ e : Quotient (orbitSetoid p₀) ≃ₜ sphere, ∀ g : SO3t, (e ⟦g⟧ : Fin 3 → ℝ) =
      (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ := by
  obtain ⟨F, hF, hFg⟩ := orbit_quotient_bijective p₀ hp₀
  have hFc : Continuous F := by
    rw [isQuotientMap_quotient_mk'.continuous_iff]
    refine Continuous.subtype_mk ?_ _
    convert (continuous_subtype_val.matrix_mulVec continuous_const :
      Continuous fun g : SO3t => (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀) using 1
    funext g
    exact hFg g
  haveI : CompactSpace SO3t := isCompact_iff_compactSpace.1 SO3_compact
  exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F hF), fun g => hFg g⟩

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.sphere_quotient_homeomorph (KaluzaKlein.Sets.Sets.northPole) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere), ∃ e : Quotient (orbitSetoid p₀) ≃ₜ sphere, ∀ g : SO3t, (e ⟦g⟧ : Fin 3 → ℝ) = (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ := by
    intro p₀ hp₀
    obtain ⟨F, hF, hFg⟩ := orbit_quotient_bijective p₀ hp₀
    have hFc : Continuous F := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      refine Continuous.subtype_mk ?_ _
      convert (continuous_subtype_val.matrix_mulVec continuous_const :
        Continuous fun g : SO3t => (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀) using 1
      funext g
      exact hFg g
    haveI : CompactSpace SO3t := isCompact_iff_compactSpace.1 SO3_compact
    exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F hF), fun g => hFg g⟩
  exact @key (KaluzaKlein.Sets.Sets.northPole) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.sphere_quotient_homeomorph (![1, 0, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere), ∃ e : Quotient (orbitSetoid p₀) ≃ₜ sphere, ∀ g : SO3t, (e ⟦g⟧ : Fin 3 → ℝ) = (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ := by
    intro p₀ hp₀
    obtain ⟨F, hF, hFg⟩ := orbit_quotient_bijective p₀ hp₀
    have hFc : Continuous F := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      refine Continuous.subtype_mk ?_ _
      convert (continuous_subtype_val.matrix_mulVec continuous_const :
        Continuous fun g : SO3t => (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀) using 1
      funext g
      exact hFg g
    haveI : CompactSpace SO3t := isCompact_iff_compactSpace.1 SO3_compact
    exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F hF), fun g => hFg g⟩
  exact @key (![1, 0, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.sphere_quotient_homeomorph (![0, 0, -1]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere), ∃ e : Quotient (orbitSetoid p₀) ≃ₜ sphere, ∀ g : SO3t, (e ⟦g⟧ : Fin 3 → ℝ) = (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀ := by
    intro p₀ hp₀
    obtain ⟨F, hF, hFg⟩ := orbit_quotient_bijective p₀ hp₀
    have hFc : Continuous F := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      refine Continuous.subtype_mk ?_ _
      convert (continuous_subtype_val.matrix_mulVec continuous_const :
        Continuous fun g : SO3t => (g : Matrix (Fin 3) (Fin 3) ℝ) *ᵥ p₀) using 1
      funext g
      exact hFg g
    haveI : CompactSpace SO3t := isCompact_iff_compactSpace.1 SO3_compact
    exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F hF), fun g => hFg g⟩
  exact @key (![0, 0, -1]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.QuotientTopology.sphere_quotient_homeomorph
-- test: set-sphere-homeomorph-test.sage

/-! ## exm-quotient-torus -/

/-- The square `[0, L]²` as a space, with the gluing of its opposite edges. -/
def squareSetoid (L : ℝ) : Setoid (square L) where
  r p q := Relation.EqvGen (glue L) p.1 q.1
  iseqv := ⟨fun _ => Relation.EqvGen.refl _, fun h => Relation.EqvGen.symm _ _ h,
    fun h1 h2 => Relation.EqvGen.trans _ _ _ h1 h2⟩

/-- For `u, u' ∈ [0, L]`: the angles `2πu/L`, `2πu'/L` give the same point of the circle exactly when `u` and `u'`
agree after `L ↦ 0`. -/
theorem wrap_scaled_eq_iff (L u u' : ℝ) (hL : 0 < L) (hu : u ∈ Icc 0 L) (hu' : u' ∈ Icc 0 L) :
    wrap (2 * Real.pi * u / L) = wrap (2 * Real.pi * u' / L) ↔ κ L u = κ L u' := by
  have hpi := Real.pi_pos
  constructor
  · intro h
    have : 2 * Real.pi * u' / L ∈ wrap ⁻¹' {wrap (2 * Real.pi * u / L)} := h.symm
    rw [wrap_preimage] at this
    obtain ⟨k, hk⟩ := this
    have hk' : u' = u + k * L := by field_simp at hk; nlinarith
    have hk1 : (k : ℝ) * L ≤ L := by linarith [hu.1, hu'.2]
    have hk2 : -L ≤ (k : ℝ) * L := by linarith [hu.2, hu'.1]
    have hk3 : (k : ℝ) ≤ 1 := by nlinarith
    have hk4 : (-1 : ℝ) ≤ k := by nlinarith
    have : k = -1 ∨ k = 0 ∨ k = 1 := by
      have a1 : k ≤ 1 := by exact_mod_cast hk3
      have a2 : -1 ≤ k := by exact_mod_cast hk4
      omega
    rcases this with rfl | rfl | rfl
    · have h1 : u = L := by push_cast at hk'; linarith [hu'.1, hu.2]
      have h2 : u' = 0 := by push_cast at hk'; linarith
      simp [κ, h1, h2]
    · simp at hk'; rw [hk']
    · have h1 : u = 0 := by push_cast at hk'; linarith [hu'.2, hu.1]
      have h2 : u' = L := by push_cast at hk'; linarith
      simp [κ, h1, h2]
  · intro h
    unfold κ at h
    split_ifs at h with h1 h2 h2
    · rw [h1, h2]
    · rw [h1, ← h]
      rw [show 2 * Real.pi * L / L = 0 + ((1 : ℤ) : ℝ) * (2 * Real.pi) by field_simp; ring]
      simp [wrap, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
    · rw [h2, h]
      rw [show 2 * Real.pi * L / L = 0 + ((1 : ℤ) : ℝ) * (2 * Real.pi) by field_simp; ring]
      simp [wrap, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
    · rw [h]

/-- Identifying the opposite edges of `[0, L]²` (`L > 0`) gives a space homeomorphic to the torus `S¹ × S¹` by
`(u, v) ↦ ((cos(2πu/L), sin(2πu/L)), (cos(2πv/L), sin(2πv/L)))` (for `L = 2π`, `(u, v) ↦ ((cos u, sin u), (cos v,
sin v))`). -/
theorem torus_quotient_homeomorph (L : ℝ) (hL : 0 < L) :
    ∃ e : Quotient (squareSetoid L) ≃ₜ (circle ×ˢ circle : Set ((ℝ × ℝ) × (ℝ × ℝ))),
      ∀ p : square L, (e ⟦p⟧ : (ℝ × ℝ) × (ℝ × ℝ)) =
        (wrap (2 * Real.pi * p.1.1 / L), wrap (2 * Real.pi * p.1.2 / L)) := by
  have hpi := Real.pi_pos
  have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
  let f : square L → (circle ×ˢ circle : Set ((ℝ × ℝ) × (ℝ × ℝ))) := fun p =>
    ⟨(wrap (2 * Real.pi * p.1.1 / L), wrap (2 * Real.pi * p.1.2 / L)), hw _, hw _⟩
  have hf : ∀ p q : square L, squareSetoid L p q → f p = f q := by
    intro p q h
    have h' := (glue_eqvGen_iff L hL p.1 q.1 p.2 q.2).1 h
    apply Subtype.ext
    simp only [f, Prod.mk.injEq]
    exact ⟨(wrap_scaled_eq_iff L _ _ hL p.2.1 q.2.1).2 h'.1, (wrap_scaled_eq_iff L _ _ hL p.2.2 q.2.2).2 h'.2⟩
  let F : Quotient (squareSetoid L) → (circle ×ˢ circle : Set ((ℝ × ℝ) × (ℝ × ℝ))) := Quotient.lift f hf
  have hFc : Continuous F := by
    rw [isQuotientMap_quotient_mk'.continuous_iff]
    refine Continuous.subtype_mk ?_ _
    simp only [Function.comp_def, F, f, Quotient.lift_mk, wrap]
    fun_prop
  have hinj : Function.Injective F := by
    intro p q h
    induction p using Quotient.ind
    induction q using Quotient.ind
    rename_i p q
    apply Quotient.sound
    change Relation.EqvGen (glue L) p.1 q.1
    rw [glue_eqvGen_iff L hL p.1 q.1 p.2 q.2]
    have h' := congrArg Subtype.val h
    simp only [F, f, Quotient.lift_mk, Prod.mk.injEq] at h'
    exact ⟨(wrap_scaled_eq_iff L _ _ hL p.2.1 q.2.1).1 h'.1, (wrap_scaled_eq_iff L _ _ hL p.2.2 q.2.2).1 h'.2⟩
  have hsurj : Function.Surjective F := by
    rintro ⟨⟨a, b⟩, ha, hb⟩
    have hT : (0 : ℝ) < 2 * Real.pi := by positivity
    have pre : ∀ c ∈ circle, ∃ u ∈ Icc 0 L, wrap (2 * Real.pi * u / L) = c := by
      intro c hc
      have hc' : c ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
      obtain ⟨φ, rfl⟩ := hc'
      set ψ := toIcoMod hT 0 φ
      have hψ := toIcoMod_mem_Ico hT 0 φ
      refine ⟨ψ * L / (2 * Real.pi), ⟨by have := hψ.1; positivity, ?_⟩, ?_⟩
      · rw [div_le_iff₀ hT]; nlinarith [hψ.2]
      · rw [show 2 * Real.pi * (ψ * L / (2 * Real.pi)) / L = ψ by field_simp]
        simp only [ψ, toIcoMod, zsmul_eq_mul, wrap, one_mul]
        rw [show φ - (toIcoDiv hT 0 φ : ℝ) * (2 * Real.pi) = φ + ((-toIcoDiv hT 0 φ : ℤ) : ℝ) * (2 * Real.pi) by
          push_cast; ring, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
    obtain ⟨u, hu, hua⟩ := pre a ha
    obtain ⟨v, hv, hvb⟩ := pre b hb
    exact ⟨⟦⟨(u, v), hu, hv⟩⟧, Subtype.ext (by simp [F, f, hua, hvb])⟩
  haveI : CompactSpace (square L) := isCompact_iff_compactSpace.1 (isCompact_Icc.prod isCompact_Icc)
  exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F ⟨hinj, hsurj⟩), fun p => rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.torus_quotient_homeomorph (2 * Real.pi) ((by positivity)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L), ∃ e : Quotient (squareSetoid L) ≃ₜ (circle ×ˢ circle : Set ((ℝ × ℝ) × (ℝ × ℝ))), ∀ p : square L, (e ⟦p⟧ : (ℝ × ℝ) × (ℝ × ℝ)) = (wrap (2 * Real.pi * p.1.1 / L), wrap (2 * Real.pi * p.1.2 / L)) := by
    intro L hL
    have hpi := Real.pi_pos
    have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
    let f : square L → (circle ×ˢ circle : Set ((ℝ × ℝ) × (ℝ × ℝ))) := fun p =>
      ⟨(wrap (2 * Real.pi * p.1.1 / L), wrap (2 * Real.pi * p.1.2 / L)), hw _, hw _⟩
    have hf : ∀ p q : square L, squareSetoid L p q → f p = f q := by
      intro p q h
      have h' := (glue_eqvGen_iff L hL p.1 q.1 p.2 q.2).1 h
      apply Subtype.ext
      simp only [f, Prod.mk.injEq]
      exact ⟨(wrap_scaled_eq_iff L _ _ hL p.2.1 q.2.1).2 h'.1, (wrap_scaled_eq_iff L _ _ hL p.2.2 q.2.2).2 h'.2⟩
    let F : Quotient (squareSetoid L) → (circle ×ˢ circle : Set ((ℝ × ℝ) × (ℝ × ℝ))) := Quotient.lift f hf
    have hFc : Continuous F := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      refine Continuous.subtype_mk ?_ _
      simp only [Function.comp_def, F, f, Quotient.lift_mk, wrap]
      fun_prop
    have hinj : Function.Injective F := by
      intro p q h
      induction p using Quotient.ind
      induction q using Quotient.ind
      rename_i p q
      apply Quotient.sound
      change Relation.EqvGen (glue L) p.1 q.1
      rw [glue_eqvGen_iff L hL p.1 q.1 p.2 q.2]
      have h' := congrArg Subtype.val h
      simp only [F, f, Quotient.lift_mk, Prod.mk.injEq] at h'
      exact ⟨(wrap_scaled_eq_iff L _ _ hL p.2.1 q.2.1).1 h'.1, (wrap_scaled_eq_iff L _ _ hL p.2.2 q.2.2).1 h'.2⟩
    have hsurj : Function.Surjective F := by
      rintro ⟨⟨a, b⟩, ha, hb⟩
      have hT : (0 : ℝ) < 2 * Real.pi := by positivity
      have pre : ∀ c ∈ circle, ∃ u ∈ Icc 0 L, wrap (2 * Real.pi * u / L) = c := by
        intro c hc
        have hc' : c ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
          rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
        obtain ⟨φ, rfl⟩ := hc'
        set ψ := toIcoMod hT 0 φ
        have hψ := toIcoMod_mem_Ico hT 0 φ
        refine ⟨ψ * L / (2 * Real.pi), ⟨by have := hψ.1; positivity, ?_⟩, ?_⟩
        · rw [div_le_iff₀ hT]; nlinarith [hψ.2]
        · rw [show 2 * Real.pi * (ψ * L / (2 * Real.pi)) / L = ψ by field_simp]
          simp only [ψ, toIcoMod, zsmul_eq_mul, wrap, one_mul]
          rw [show φ - (toIcoDiv hT 0 φ : ℝ) * (2 * Real.pi) = φ + ((-toIcoDiv hT 0 φ : ℤ) : ℝ) * (2 * Real.pi) by
            push_cast; ring, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
      obtain ⟨u, hu, hua⟩ := pre a ha
      obtain ⟨v, hv, hvb⟩ := pre b hb
      exact ⟨⟦⟨(u, v), hu, hv⟩⟧, Subtype.ext (by simp [F, f, hua, hvb])⟩
    haveI : CompactSpace (square L) := isCompact_iff_compactSpace.1 (isCompact_Icc.prod isCompact_Icc)
    exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F ⟨hinj, hsurj⟩), fun p => rfl⟩
  exact @key (2 * Real.pi) ((by positivity))

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.torus_quotient_homeomorph (1) ((by norm_num)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L), ∃ e : Quotient (squareSetoid L) ≃ₜ (circle ×ˢ circle : Set ((ℝ × ℝ) × (ℝ × ℝ))), ∀ p : square L, (e ⟦p⟧ : (ℝ × ℝ) × (ℝ × ℝ)) = (wrap (2 * Real.pi * p.1.1 / L), wrap (2 * Real.pi * p.1.2 / L)) := by
    intro L hL
    have hpi := Real.pi_pos
    have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
    let f : square L → (circle ×ˢ circle : Set ((ℝ × ℝ) × (ℝ × ℝ))) := fun p =>
      ⟨(wrap (2 * Real.pi * p.1.1 / L), wrap (2 * Real.pi * p.1.2 / L)), hw _, hw _⟩
    have hf : ∀ p q : square L, squareSetoid L p q → f p = f q := by
      intro p q h
      have h' := (glue_eqvGen_iff L hL p.1 q.1 p.2 q.2).1 h
      apply Subtype.ext
      simp only [f, Prod.mk.injEq]
      exact ⟨(wrap_scaled_eq_iff L _ _ hL p.2.1 q.2.1).2 h'.1, (wrap_scaled_eq_iff L _ _ hL p.2.2 q.2.2).2 h'.2⟩
    let F : Quotient (squareSetoid L) → (circle ×ˢ circle : Set ((ℝ × ℝ) × (ℝ × ℝ))) := Quotient.lift f hf
    have hFc : Continuous F := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      refine Continuous.subtype_mk ?_ _
      simp only [Function.comp_def, F, f, Quotient.lift_mk, wrap]
      fun_prop
    have hinj : Function.Injective F := by
      intro p q h
      induction p using Quotient.ind
      induction q using Quotient.ind
      rename_i p q
      apply Quotient.sound
      change Relation.EqvGen (glue L) p.1 q.1
      rw [glue_eqvGen_iff L hL p.1 q.1 p.2 q.2]
      have h' := congrArg Subtype.val h
      simp only [F, f, Quotient.lift_mk, Prod.mk.injEq] at h'
      exact ⟨(wrap_scaled_eq_iff L _ _ hL p.2.1 q.2.1).1 h'.1, (wrap_scaled_eq_iff L _ _ hL p.2.2 q.2.2).1 h'.2⟩
    have hsurj : Function.Surjective F := by
      rintro ⟨⟨a, b⟩, ha, hb⟩
      have hT : (0 : ℝ) < 2 * Real.pi := by positivity
      have pre : ∀ c ∈ circle, ∃ u ∈ Icc 0 L, wrap (2 * Real.pi * u / L) = c := by
        intro c hc
        have hc' : c ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
          rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
        obtain ⟨φ, rfl⟩ := hc'
        set ψ := toIcoMod hT 0 φ
        have hψ := toIcoMod_mem_Ico hT 0 φ
        refine ⟨ψ * L / (2 * Real.pi), ⟨by have := hψ.1; positivity, ?_⟩, ?_⟩
        · rw [div_le_iff₀ hT]; nlinarith [hψ.2]
        · rw [show 2 * Real.pi * (ψ * L / (2 * Real.pi)) / L = ψ by field_simp]
          simp only [ψ, toIcoMod, zsmul_eq_mul, wrap, one_mul]
          rw [show φ - (toIcoDiv hT 0 φ : ℝ) * (2 * Real.pi) = φ + ((-toIcoDiv hT 0 φ : ℤ) : ℝ) * (2 * Real.pi) by
            push_cast; ring, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
      obtain ⟨u, hu, hua⟩ := pre a ha
      obtain ⟨v, hv, hvb⟩ := pre b hb
      exact ⟨⟦⟨(u, v), hu, hv⟩⟧, Subtype.ext (by simp [F, f, hua, hvb])⟩
    haveI : CompactSpace (square L) := isCompact_iff_compactSpace.1 (isCompact_Icc.prod isCompact_Icc)
    exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F ⟨hinj, hsurj⟩), fun p => rfl⟩
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.torus_quotient_homeomorph (3) ((by norm_num)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L), ∃ e : Quotient (squareSetoid L) ≃ₜ (circle ×ˢ circle : Set ((ℝ × ℝ) × (ℝ × ℝ))), ∀ p : square L, (e ⟦p⟧ : (ℝ × ℝ) × (ℝ × ℝ)) = (wrap (2 * Real.pi * p.1.1 / L), wrap (2 * Real.pi * p.1.2 / L)) := by
    intro L hL
    have hpi := Real.pi_pos
    have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
    let f : square L → (circle ×ˢ circle : Set ((ℝ × ℝ) × (ℝ × ℝ))) := fun p =>
      ⟨(wrap (2 * Real.pi * p.1.1 / L), wrap (2 * Real.pi * p.1.2 / L)), hw _, hw _⟩
    have hf : ∀ p q : square L, squareSetoid L p q → f p = f q := by
      intro p q h
      have h' := (glue_eqvGen_iff L hL p.1 q.1 p.2 q.2).1 h
      apply Subtype.ext
      simp only [f, Prod.mk.injEq]
      exact ⟨(wrap_scaled_eq_iff L _ _ hL p.2.1 q.2.1).2 h'.1, (wrap_scaled_eq_iff L _ _ hL p.2.2 q.2.2).2 h'.2⟩
    let F : Quotient (squareSetoid L) → (circle ×ˢ circle : Set ((ℝ × ℝ) × (ℝ × ℝ))) := Quotient.lift f hf
    have hFc : Continuous F := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      refine Continuous.subtype_mk ?_ _
      simp only [Function.comp_def, F, f, Quotient.lift_mk, wrap]
      fun_prop
    have hinj : Function.Injective F := by
      intro p q h
      induction p using Quotient.ind
      induction q using Quotient.ind
      rename_i p q
      apply Quotient.sound
      change Relation.EqvGen (glue L) p.1 q.1
      rw [glue_eqvGen_iff L hL p.1 q.1 p.2 q.2]
      have h' := congrArg Subtype.val h
      simp only [F, f, Quotient.lift_mk, Prod.mk.injEq] at h'
      exact ⟨(wrap_scaled_eq_iff L _ _ hL p.2.1 q.2.1).1 h'.1, (wrap_scaled_eq_iff L _ _ hL p.2.2 q.2.2).1 h'.2⟩
    have hsurj : Function.Surjective F := by
      rintro ⟨⟨a, b⟩, ha, hb⟩
      have hT : (0 : ℝ) < 2 * Real.pi := by positivity
      have pre : ∀ c ∈ circle, ∃ u ∈ Icc 0 L, wrap (2 * Real.pi * u / L) = c := by
        intro c hc
        have hc' : c ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
          rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
        obtain ⟨φ, rfl⟩ := hc'
        set ψ := toIcoMod hT 0 φ
        have hψ := toIcoMod_mem_Ico hT 0 φ
        refine ⟨ψ * L / (2 * Real.pi), ⟨by have := hψ.1; positivity, ?_⟩, ?_⟩
        · rw [div_le_iff₀ hT]; nlinarith [hψ.2]
        · rw [show 2 * Real.pi * (ψ * L / (2 * Real.pi)) / L = ψ by field_simp]
          simp only [ψ, toIcoMod, zsmul_eq_mul, wrap, one_mul]
          rw [show φ - (toIcoDiv hT 0 φ : ℝ) * (2 * Real.pi) = φ + ((-toIcoDiv hT 0 φ : ℤ) : ℝ) * (2 * Real.pi) by
            push_cast; ring, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
      obtain ⟨u, hu, hua⟩ := pre a ha
      obtain ⟨v, hv, hvb⟩ := pre b hb
      exact ⟨⟦⟨(u, v), hu, hv⟩⟧, Subtype.ext (by simp [F, f, hua, hvb])⟩
    haveI : CompactSpace (square L) := isCompact_iff_compactSpace.1 (isCompact_Icc.prod isCompact_Icc)
    exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F ⟨hinj, hsurj⟩), fun p => rfl⟩
  exact @key (3) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.QuotientTopology.torus_quotient_homeomorph
-- test: set-torus-homeomorph-test.sage

/-! ## exr-quotient-interval-circle -/

/-- `[0, L]` with the endpoints identified. -/
def intervalSetoid (L : ℝ) : Setoid (Icc (0 : ℝ) L) where
  r p q := κ L p.1 = κ L q.1
  iseqv := ⟨fun _ => rfl, fun h => h.symm, fun h1 h2 => h1.trans h2⟩

/-- `[0, L]` with `0 ∼ L` (`L > 0`) is homeomorphic to the circle by `[φ] ↦ (cos(2πφ/L), sin(2πφ/L))` (for `L = 2π`,
`[φ] ↦ (cos φ, sin φ)`). -/
theorem interval_quotient_homeomorph (L : ℝ) (hL : 0 < L) :
    ∃ e : Quotient (intervalSetoid L) ≃ₜ circle, ∀ p : Icc (0 : ℝ) L, (e ⟦p⟧ : ℝ × ℝ) =
      wrap (2 * Real.pi * p.1 / L) := by
  have hpi := Real.pi_pos
  have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
  let f : Icc (0 : ℝ) L → circle := fun p => ⟨wrap (2 * Real.pi * p.1 / L), hw _⟩
  have hf : ∀ p q : Icc (0 : ℝ) L, intervalSetoid L p q → f p = f q := fun p q h =>
    Subtype.ext ((wrap_scaled_eq_iff L _ _ hL p.2 q.2).2 h)
  let F : Quotient (intervalSetoid L) → circle := Quotient.lift f hf
  have hFc : Continuous F := by
    rw [isQuotientMap_quotient_mk'.continuous_iff]
    refine Continuous.subtype_mk ?_ _
    simp only [Function.comp_def, F, f, Quotient.lift_mk, wrap]
    fun_prop
  have hinj : Function.Injective F := by
    intro p q h
    induction p using Quotient.ind
    induction q using Quotient.ind
    rename_i a b
    exact Quotient.sound ((wrap_scaled_eq_iff L _ _ hL a.2 b.2).1 (congrArg Subtype.val h))
  have hsurj : Function.Surjective F := by
    rintro ⟨c, hc⟩
    have hT : (0 : ℝ) < 2 * Real.pi := by positivity
    have hc' : c ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
      rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
    obtain ⟨φ, rfl⟩ := hc'
    set ψ := toIcoMod hT 0 φ
    have hψ := toIcoMod_mem_Ico hT 0 φ
    refine ⟨⟦⟨ψ * L / (2 * Real.pi), by have := hψ.1; positivity, ?_⟩⟧, Subtype.ext ?_⟩
    · rw [div_le_iff₀ hT]; nlinarith [hψ.2]
    · simp only [F, f, Quotient.lift_mk]
      rw [show 2 * Real.pi * (ψ * L / (2 * Real.pi)) / L = ψ by field_simp]
      simp only [ψ, toIcoMod, zsmul_eq_mul, wrap, one_mul]
      rw [show φ - (toIcoDiv hT 0 φ : ℝ) * (2 * Real.pi) = φ + ((-toIcoDiv hT 0 φ : ℤ) : ℝ) * (2 * Real.pi) by
        push_cast; ring, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
  haveI : CompactSpace (Icc (0 : ℝ) L) := isCompact_iff_compactSpace.1 isCompact_Icc
  exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F ⟨hinj, hsurj⟩), fun p => rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.interval_quotient_homeomorph (2 * Real.pi) ((by positivity)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L), ∃ e : Quotient (intervalSetoid L) ≃ₜ circle, ∀ p : Icc (0 : ℝ) L, (e ⟦p⟧ : ℝ × ℝ) = wrap (2 * Real.pi * p.1 / L) := by
    intro L hL
    have hpi := Real.pi_pos
    have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
    let f : Icc (0 : ℝ) L → circle := fun p => ⟨wrap (2 * Real.pi * p.1 / L), hw _⟩
    have hf : ∀ p q : Icc (0 : ℝ) L, intervalSetoid L p q → f p = f q := fun p q h =>
      Subtype.ext ((wrap_scaled_eq_iff L _ _ hL p.2 q.2).2 h)
    let F : Quotient (intervalSetoid L) → circle := Quotient.lift f hf
    have hFc : Continuous F := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      refine Continuous.subtype_mk ?_ _
      simp only [Function.comp_def, F, f, Quotient.lift_mk, wrap]
      fun_prop
    have hinj : Function.Injective F := by
      intro p q h
      induction p using Quotient.ind
      induction q using Quotient.ind
      rename_i a b
      exact Quotient.sound ((wrap_scaled_eq_iff L _ _ hL a.2 b.2).1 (congrArg Subtype.val h))
    have hsurj : Function.Surjective F := by
      rintro ⟨c, hc⟩
      have hT : (0 : ℝ) < 2 * Real.pi := by positivity
      have hc' : c ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
      obtain ⟨φ, rfl⟩ := hc'
      set ψ := toIcoMod hT 0 φ
      have hψ := toIcoMod_mem_Ico hT 0 φ
      refine ⟨⟦⟨ψ * L / (2 * Real.pi), by have := hψ.1; positivity, ?_⟩⟧, Subtype.ext ?_⟩
      · rw [div_le_iff₀ hT]; nlinarith [hψ.2]
      · simp only [F, f, Quotient.lift_mk]
        rw [show 2 * Real.pi * (ψ * L / (2 * Real.pi)) / L = ψ by field_simp]
        simp only [ψ, toIcoMod, zsmul_eq_mul, wrap, one_mul]
        rw [show φ - (toIcoDiv hT 0 φ : ℝ) * (2 * Real.pi) = φ + ((-toIcoDiv hT 0 φ : ℤ) : ℝ) * (2 * Real.pi) by
          push_cast; ring, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
    haveI : CompactSpace (Icc (0 : ℝ) L) := isCompact_iff_compactSpace.1 isCompact_Icc
    exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F ⟨hinj, hsurj⟩), fun p => rfl⟩
  exact @key (2 * Real.pi) ((by positivity))

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.interval_quotient_homeomorph (1) ((by norm_num)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L), ∃ e : Quotient (intervalSetoid L) ≃ₜ circle, ∀ p : Icc (0 : ℝ) L, (e ⟦p⟧ : ℝ × ℝ) = wrap (2 * Real.pi * p.1 / L) := by
    intro L hL
    have hpi := Real.pi_pos
    have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
    let f : Icc (0 : ℝ) L → circle := fun p => ⟨wrap (2 * Real.pi * p.1 / L), hw _⟩
    have hf : ∀ p q : Icc (0 : ℝ) L, intervalSetoid L p q → f p = f q := fun p q h =>
      Subtype.ext ((wrap_scaled_eq_iff L _ _ hL p.2 q.2).2 h)
    let F : Quotient (intervalSetoid L) → circle := Quotient.lift f hf
    have hFc : Continuous F := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      refine Continuous.subtype_mk ?_ _
      simp only [Function.comp_def, F, f, Quotient.lift_mk, wrap]
      fun_prop
    have hinj : Function.Injective F := by
      intro p q h
      induction p using Quotient.ind
      induction q using Quotient.ind
      rename_i a b
      exact Quotient.sound ((wrap_scaled_eq_iff L _ _ hL a.2 b.2).1 (congrArg Subtype.val h))
    have hsurj : Function.Surjective F := by
      rintro ⟨c, hc⟩
      have hT : (0 : ℝ) < 2 * Real.pi := by positivity
      have hc' : c ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
      obtain ⟨φ, rfl⟩ := hc'
      set ψ := toIcoMod hT 0 φ
      have hψ := toIcoMod_mem_Ico hT 0 φ
      refine ⟨⟦⟨ψ * L / (2 * Real.pi), by have := hψ.1; positivity, ?_⟩⟧, Subtype.ext ?_⟩
      · rw [div_le_iff₀ hT]; nlinarith [hψ.2]
      · simp only [F, f, Quotient.lift_mk]
        rw [show 2 * Real.pi * (ψ * L / (2 * Real.pi)) / L = ψ by field_simp]
        simp only [ψ, toIcoMod, zsmul_eq_mul, wrap, one_mul]
        rw [show φ - (toIcoDiv hT 0 φ : ℝ) * (2 * Real.pi) = φ + ((-toIcoDiv hT 0 φ : ℤ) : ℝ) * (2 * Real.pi) by
          push_cast; ring, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
    haveI : CompactSpace (Icc (0 : ℝ) L) := isCompact_iff_compactSpace.1 isCompact_Icc
    exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F ⟨hinj, hsurj⟩), fun p => rfl⟩
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.interval_quotient_homeomorph (1/2) ((by norm_num)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L), ∃ e : Quotient (intervalSetoid L) ≃ₜ circle, ∀ p : Icc (0 : ℝ) L, (e ⟦p⟧ : ℝ × ℝ) = wrap (2 * Real.pi * p.1 / L) := by
    intro L hL
    have hpi := Real.pi_pos
    have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
    let f : Icc (0 : ℝ) L → circle := fun p => ⟨wrap (2 * Real.pi * p.1 / L), hw _⟩
    have hf : ∀ p q : Icc (0 : ℝ) L, intervalSetoid L p q → f p = f q := fun p q h =>
      Subtype.ext ((wrap_scaled_eq_iff L _ _ hL p.2 q.2).2 h)
    let F : Quotient (intervalSetoid L) → circle := Quotient.lift f hf
    have hFc : Continuous F := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      refine Continuous.subtype_mk ?_ _
      simp only [Function.comp_def, F, f, Quotient.lift_mk, wrap]
      fun_prop
    have hinj : Function.Injective F := by
      intro p q h
      induction p using Quotient.ind
      induction q using Quotient.ind
      rename_i a b
      exact Quotient.sound ((wrap_scaled_eq_iff L _ _ hL a.2 b.2).1 (congrArg Subtype.val h))
    have hsurj : Function.Surjective F := by
      rintro ⟨c, hc⟩
      have hT : (0 : ℝ) < 2 * Real.pi := by positivity
      have hc' : c ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using hc
      obtain ⟨φ, rfl⟩ := hc'
      set ψ := toIcoMod hT 0 φ
      have hψ := toIcoMod_mem_Ico hT 0 φ
      refine ⟨⟦⟨ψ * L / (2 * Real.pi), by have := hψ.1; positivity, ?_⟩⟧, Subtype.ext ?_⟩
      · rw [div_le_iff₀ hT]; nlinarith [hψ.2]
      · simp only [F, f, Quotient.lift_mk]
        rw [show 2 * Real.pi * (ψ * L / (2 * Real.pi)) / L = ψ by field_simp]
        simp only [ψ, toIcoMod, zsmul_eq_mul, wrap, one_mul]
        rw [show φ - (toIcoDiv hT 0 φ : ℝ) * (2 * Real.pi) = φ + ((-toIcoDiv hT 0 φ : ℤ) : ℝ) * (2 * Real.pi) by
          push_cast; ring, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
    haveI : CompactSpace (Icc (0 : ℝ) L) := isCompact_iff_compactSpace.1 isCompact_Icc
    exact ⟨hFc.homeoOfEquivCompactToT2 (f := Equiv.ofBijective F ⟨hinj, hsurj⟩), fun p => rfl⟩
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.QuotientTopology.interval_quotient_homeomorph
-- test: set-interval-homeomorph-test.sage

/-! ## exr-quotient-open-sets -/

/-- In `ℝ/Tℤ`: a set of classes is open exactly when its preimage, a `T`-periodic set, is open in `ℝ`; and the
quotient map is open, since the preimage of the image of `U` is the union of the translates `U + kT`. -/
theorem quotient_open_sets (T : ℝ) (V : Set (Quotient (angleSetoid T))) (U : Set ℝ) (hU : IsOpen U) :
    (IsOpen V ↔ IsOpen (Quotient.mk (angleSetoid T) ⁻¹' V)) ∧
      (∀ x ∈ Quotient.mk (angleSetoid T) ⁻¹' V, x + T ∈ Quotient.mk (angleSetoid T) ⁻¹' V) ∧
      Quotient.mk (angleSetoid T) ⁻¹' (Quotient.mk (angleSetoid T) '' U) = ⋃ k : ℤ, (fun x => x + k * T) '' U ∧
      IsOpen (Quotient.mk (angleSetoid T) '' U) := by
  have hpre : Quotient.mk (angleSetoid T) ⁻¹' (Quotient.mk (angleSetoid T) '' U) =
      ⋃ k : ℤ, (fun x => x + k * T) '' U := by
    ext y
    simp only [mem_preimage, mem_image, mem_iUnion]
    constructor
    · rintro ⟨x, hx, hxy⟩
      obtain ⟨k, hk⟩ := Quotient.exact hxy
      exact ⟨k, x, hx, by linarith⟩
    · rintro ⟨k, x, hx, rfl⟩
      exact ⟨x, hx, Quotient.sound ⟨k, by ring⟩⟩
  refine ⟨Iff.symm isQuotientMap_quotient_mk'.isOpen_preimage, fun x hx => ?_, hpre, ?_⟩
  · simp only [mem_preimage] at hx ⊢
    rwa [Quotient.sound (show angleSetoid T x (x + T) from ⟨1, by simp⟩)] at hx
  · have h : IsOpen (Quotient.mk (angleSetoid T) ⁻¹' (Quotient.mk (angleSetoid T) '' U)) := by
      rw [hpre]
      exact isOpen_iUnion fun k => (Homeomorph.addRight ((k : ℝ) * T)).isOpenMap U hU
    exact isQuotientMap_quotient_mk'.isOpen_preimage.1 h

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.quotient_open_sets (2 * Real.pi) (Set.univ) (Set.Ioo 0 1) (isOpen_Ioo) := by
  have key : ∀ (T : ℝ) (V : Set (Quotient (angleSetoid T))) (U : Set ℝ) (hU : IsOpen U), (IsOpen V ↔ IsOpen (Quotient.mk (angleSetoid T) ⁻¹' V)) ∧ (∀ x ∈ Quotient.mk (angleSetoid T) ⁻¹' V, x + T ∈ Quotient.mk (angleSetoid T) ⁻¹' V) ∧ Quotient.mk (angleSetoid T) ⁻¹' (Quotient.mk (angleSetoid T) '' U) = ⋃ k : ℤ, (fun x => x + k * T) '' U ∧ IsOpen (Quotient.mk (angleSetoid T) '' U) := by
    intro T V U hU
    have hpre : Quotient.mk (angleSetoid T) ⁻¹' (Quotient.mk (angleSetoid T) '' U) =
        ⋃ k : ℤ, (fun x => x + k * T) '' U := by
      ext y
      simp only [mem_preimage, mem_image, mem_iUnion]
      constructor
      · rintro ⟨x, hx, hxy⟩
        obtain ⟨k, hk⟩ := Quotient.exact hxy
        exact ⟨k, x, hx, by linarith⟩
      · rintro ⟨k, x, hx, rfl⟩
        exact ⟨x, hx, Quotient.sound ⟨k, by ring⟩⟩
    refine ⟨Iff.symm isQuotientMap_quotient_mk'.isOpen_preimage, fun x hx => ?_, hpre, ?_⟩
    · simp only [mem_preimage] at hx ⊢
      rwa [Quotient.sound (show angleSetoid T x (x + T) from ⟨1, by simp⟩)] at hx
    · have h : IsOpen (Quotient.mk (angleSetoid T) ⁻¹' (Quotient.mk (angleSetoid T) '' U)) := by
        rw [hpre]
        exact isOpen_iUnion fun k => (Homeomorph.addRight ((k : ℝ) * T)).isOpenMap U hU
      exact isQuotientMap_quotient_mk'.isOpen_preimage.1 h
  exact @key (2 * Real.pi) (Set.univ) (Set.Ioo 0 1) (isOpen_Ioo)

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.quotient_open_sets (0) (∅) (∅) (isOpen_empty) := by
  have key : ∀ (T : ℝ) (V : Set (Quotient (angleSetoid T))) (U : Set ℝ) (hU : IsOpen U), (IsOpen V ↔ IsOpen (Quotient.mk (angleSetoid T) ⁻¹' V)) ∧ (∀ x ∈ Quotient.mk (angleSetoid T) ⁻¹' V, x + T ∈ Quotient.mk (angleSetoid T) ⁻¹' V) ∧ Quotient.mk (angleSetoid T) ⁻¹' (Quotient.mk (angleSetoid T) '' U) = ⋃ k : ℤ, (fun x => x + k * T) '' U ∧ IsOpen (Quotient.mk (angleSetoid T) '' U) := by
    intro T V U hU
    have hpre : Quotient.mk (angleSetoid T) ⁻¹' (Quotient.mk (angleSetoid T) '' U) =
        ⋃ k : ℤ, (fun x => x + k * T) '' U := by
      ext y
      simp only [mem_preimage, mem_image, mem_iUnion]
      constructor
      · rintro ⟨x, hx, hxy⟩
        obtain ⟨k, hk⟩ := Quotient.exact hxy
        exact ⟨k, x, hx, by linarith⟩
      · rintro ⟨k, x, hx, rfl⟩
        exact ⟨x, hx, Quotient.sound ⟨k, by ring⟩⟩
    refine ⟨Iff.symm isQuotientMap_quotient_mk'.isOpen_preimage, fun x hx => ?_, hpre, ?_⟩
    · simp only [mem_preimage] at hx ⊢
      rwa [Quotient.sound (show angleSetoid T x (x + T) from ⟨1, by simp⟩)] at hx
    · have h : IsOpen (Quotient.mk (angleSetoid T) ⁻¹' (Quotient.mk (angleSetoid T) '' U)) := by
        rw [hpre]
        exact isOpen_iUnion fun k => (Homeomorph.addRight ((k : ℝ) * T)).isOpenMap U hU
      exact isQuotientMap_quotient_mk'.isOpen_preimage.1 h
  exact @key (0) (∅) (∅) (isOpen_empty)

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.quotient_open_sets (1) (Set.univ) (Set.Ioo (-1) 1) (isOpen_Ioo) := by
  have key : ∀ (T : ℝ) (V : Set (Quotient (angleSetoid T))) (U : Set ℝ) (hU : IsOpen U), (IsOpen V ↔ IsOpen (Quotient.mk (angleSetoid T) ⁻¹' V)) ∧ (∀ x ∈ Quotient.mk (angleSetoid T) ⁻¹' V, x + T ∈ Quotient.mk (angleSetoid T) ⁻¹' V) ∧ Quotient.mk (angleSetoid T) ⁻¹' (Quotient.mk (angleSetoid T) '' U) = ⋃ k : ℤ, (fun x => x + k * T) '' U ∧ IsOpen (Quotient.mk (angleSetoid T) '' U) := by
    intro T V U hU
    have hpre : Quotient.mk (angleSetoid T) ⁻¹' (Quotient.mk (angleSetoid T) '' U) =
        ⋃ k : ℤ, (fun x => x + k * T) '' U := by
      ext y
      simp only [mem_preimage, mem_image, mem_iUnion]
      constructor
      · rintro ⟨x, hx, hxy⟩
        obtain ⟨k, hk⟩ := Quotient.exact hxy
        exact ⟨k, x, hx, by linarith⟩
      · rintro ⟨k, x, hx, rfl⟩
        exact ⟨x, hx, Quotient.sound ⟨k, by ring⟩⟩
    refine ⟨Iff.symm isQuotientMap_quotient_mk'.isOpen_preimage, fun x hx => ?_, hpre, ?_⟩
    · simp only [mem_preimage] at hx ⊢
      rwa [Quotient.sound (show angleSetoid T x (x + T) from ⟨1, by simp⟩)] at hx
    · have h : IsOpen (Quotient.mk (angleSetoid T) ⁻¹' (Quotient.mk (angleSetoid T) '' U)) := by
        rw [hpre]
        exact isOpen_iUnion fun k => (Homeomorph.addRight ((k : ℝ) * T)).isOpenMap U hU
      exact isQuotientMap_quotient_mk'.isOpen_preimage.1 h
  exact @key (1) (Set.univ) (Set.Ioo (-1) 1) (isOpen_Ioo)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.QuotientTopology.quotient_open_sets
-- test: set-quotient-open-test.sage

/-! ## exr-quotient-moebius -/

/-- The gluings of the vertical edges of `[0, L]²`: with a half twist, `(0, v) ∼ (L, L - v)` (Möbius band), or
straight, `(0, v) ∼ (L, v)` (cylinder). -/
def edgeGlue (L : ℝ) (twist : Bool) (p q : square L) : Prop :=
  p.1.1 = 0 ∧ q.1.1 = L ∧ q.1.2 = (if twist then L - p.1.2 else p.1.2)

/-- The quotient by the equivalence relation generated by a gluing. -/
def edgeSetoid (L : ℝ) (twist : Bool) : Setoid (square L) := Relation.EqvGen.setoid (edgeGlue L twist)

/-- The boundary of the square, its top and bottom edges. -/
def topBottom (L : ℝ) : Set (square L) := {p | p.1.2 = 0 ∨ p.1.2 = L}

/-- The boundary of the Möbius band is a single circle: the image of the top and bottom edges is connected (the half
twist joins the top edge to the bottom one). -/
theorem moebius_boundary_connected (L : ℝ) (hL : 0 < L) :
    IsConnected (Quotient.mk (edgeSetoid L true) '' topBottom L) := by
  set π := Quotient.mk (edgeSetoid L true)
  have hπ : Continuous π := continuous_quotient_mk'
  let γ : Bool → ℝ → square L := fun top t =>
    ⟨(max 0 (min t L), if top then L else 0), ⟨le_max_left _ _, max_le hL.le (min_le_right _ _)⟩,
      by cases top <;> simp [hL.le]⟩
  have hγ : ∀ top, Continuous (γ top) := fun top => by
    refine Continuous.subtype_mk ?_ _
    fun_prop
  have hedge : ∀ top, (γ top) '' Icc 0 L = {p : square L | p.1.2 = if top then L else 0} := by
    intro top
    ext ⟨⟨u, v⟩, hu, hv⟩
    simp only [mem_setOf_eq]
    constructor
    · rintro ⟨t, -, ht⟩
      have := congrArg (fun p : square L => p.1.2) ht
      simpa [γ] using this.symm
    · intro h
      refine ⟨u, hu, Subtype.ext ?_⟩
      simp only [γ, Prod.mk.injEq]
      exact ⟨by rw [min_eq_left hu.2, max_eq_right hu.1], h.symm⟩
  have hB : topBottom L = (γ true) '' Icc 0 L ∪ (γ false) '' Icc 0 L := by
    rw [hedge, hedge]; ext p; simp [topBottom, or_comm]
  have hc : ∀ top, IsConnected (π '' ((γ top) '' Icc 0 L)) := fun top =>
    ((isConnected_Icc hL.le).image _ (hγ top).continuousOn).image _ hπ.continuousOn
  rw [hB, image_union]
  refine IsConnected.union ?_ (hc true) (hc false)
  -- `(0, L)` on the top edge is glued to `(L, 0)` on the bottom edge
  refine ⟨π (γ true 0), ⟨γ true 0, ⟨0, ⟨le_rfl, hL.le⟩, rfl⟩, rfl⟩, ⟨γ false L, ⟨L, ⟨hL.le, le_rfl⟩, rfl⟩, ?_⟩⟩
  apply Quotient.sound
  apply Relation.EqvGen.symm
  apply Relation.EqvGen.rel
  simp [edgeGlue, γ, hL.le]

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.moebius_boundary_connected (1) ((by norm_num)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L), IsConnected (Quotient.mk (edgeSetoid L true) '' topBottom L) := by
    intro L hL
    set π := Quotient.mk (edgeSetoid L true)
    have hπ : Continuous π := continuous_quotient_mk'
    let γ : Bool → ℝ → square L := fun top t =>
      ⟨(max 0 (min t L), if top then L else 0), ⟨le_max_left _ _, max_le hL.le (min_le_right _ _)⟩,
        by cases top <;> simp [hL.le]⟩
    have hγ : ∀ top, Continuous (γ top) := fun top => by
      refine Continuous.subtype_mk ?_ _
      fun_prop
    have hedge : ∀ top, (γ top) '' Icc 0 L = {p : square L | p.1.2 = if top then L else 0} := by
      intro top
      ext ⟨⟨u, v⟩, hu, hv⟩
      simp only [mem_setOf_eq]
      constructor
      · rintro ⟨t, -, ht⟩
        have := congrArg (fun p : square L => p.1.2) ht
        simpa [γ] using this.symm
      · intro h
        refine ⟨u, hu, Subtype.ext ?_⟩
        simp only [γ, Prod.mk.injEq]
        exact ⟨by rw [min_eq_left hu.2, max_eq_right hu.1], h.symm⟩
    have hB : topBottom L = (γ true) '' Icc 0 L ∪ (γ false) '' Icc 0 L := by
      rw [hedge, hedge]; ext p; simp [topBottom, or_comm]
    have hc : ∀ top, IsConnected (π '' ((γ top) '' Icc 0 L)) := fun top =>
      ((isConnected_Icc hL.le).image _ (hγ top).continuousOn).image _ hπ.continuousOn
    rw [hB, image_union]
    refine IsConnected.union ?_ (hc true) (hc false)
    -- `(0, L)` on the top edge is glued to `(L, 0)` on the bottom edge
    refine ⟨π (γ true 0), ⟨γ true 0, ⟨0, ⟨le_rfl, hL.le⟩, rfl⟩, rfl⟩, ⟨γ false L, ⟨L, ⟨hL.le, le_rfl⟩, rfl⟩, ?_⟩⟩
    apply Quotient.sound
    apply Relation.EqvGen.symm
    apply Relation.EqvGen.rel
    simp [edgeGlue, γ, hL.le]
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.moebius_boundary_connected (2) ((by norm_num)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L), IsConnected (Quotient.mk (edgeSetoid L true) '' topBottom L) := by
    intro L hL
    set π := Quotient.mk (edgeSetoid L true)
    have hπ : Continuous π := continuous_quotient_mk'
    let γ : Bool → ℝ → square L := fun top t =>
      ⟨(max 0 (min t L), if top then L else 0), ⟨le_max_left _ _, max_le hL.le (min_le_right _ _)⟩,
        by cases top <;> simp [hL.le]⟩
    have hγ : ∀ top, Continuous (γ top) := fun top => by
      refine Continuous.subtype_mk ?_ _
      fun_prop
    have hedge : ∀ top, (γ top) '' Icc 0 L = {p : square L | p.1.2 = if top then L else 0} := by
      intro top
      ext ⟨⟨u, v⟩, hu, hv⟩
      simp only [mem_setOf_eq]
      constructor
      · rintro ⟨t, -, ht⟩
        have := congrArg (fun p : square L => p.1.2) ht
        simpa [γ] using this.symm
      · intro h
        refine ⟨u, hu, Subtype.ext ?_⟩
        simp only [γ, Prod.mk.injEq]
        exact ⟨by rw [min_eq_left hu.2, max_eq_right hu.1], h.symm⟩
    have hB : topBottom L = (γ true) '' Icc 0 L ∪ (γ false) '' Icc 0 L := by
      rw [hedge, hedge]; ext p; simp [topBottom, or_comm]
    have hc : ∀ top, IsConnected (π '' ((γ top) '' Icc 0 L)) := fun top =>
      ((isConnected_Icc hL.le).image _ (hγ top).continuousOn).image _ hπ.continuousOn
    rw [hB, image_union]
    refine IsConnected.union ?_ (hc true) (hc false)
    -- `(0, L)` on the top edge is glued to `(L, 0)` on the bottom edge
    refine ⟨π (γ true 0), ⟨γ true 0, ⟨0, ⟨le_rfl, hL.le⟩, rfl⟩, rfl⟩, ⟨γ false L, ⟨L, ⟨hL.le, le_rfl⟩, rfl⟩, ?_⟩⟩
    apply Quotient.sound
    apply Relation.EqvGen.symm
    apply Relation.EqvGen.rel
    simp [edgeGlue, γ, hL.le]
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.moebius_boundary_connected (1/2) ((by norm_num)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L), IsConnected (Quotient.mk (edgeSetoid L true) '' topBottom L) := by
    intro L hL
    set π := Quotient.mk (edgeSetoid L true)
    have hπ : Continuous π := continuous_quotient_mk'
    let γ : Bool → ℝ → square L := fun top t =>
      ⟨(max 0 (min t L), if top then L else 0), ⟨le_max_left _ _, max_le hL.le (min_le_right _ _)⟩,
        by cases top <;> simp [hL.le]⟩
    have hγ : ∀ top, Continuous (γ top) := fun top => by
      refine Continuous.subtype_mk ?_ _
      fun_prop
    have hedge : ∀ top, (γ top) '' Icc 0 L = {p : square L | p.1.2 = if top then L else 0} := by
      intro top
      ext ⟨⟨u, v⟩, hu, hv⟩
      simp only [mem_setOf_eq]
      constructor
      · rintro ⟨t, -, ht⟩
        have := congrArg (fun p : square L => p.1.2) ht
        simpa [γ] using this.symm
      · intro h
        refine ⟨u, hu, Subtype.ext ?_⟩
        simp only [γ, Prod.mk.injEq]
        exact ⟨by rw [min_eq_left hu.2, max_eq_right hu.1], h.symm⟩
    have hB : topBottom L = (γ true) '' Icc 0 L ∪ (γ false) '' Icc 0 L := by
      rw [hedge, hedge]; ext p; simp [topBottom, or_comm]
    have hc : ∀ top, IsConnected (π '' ((γ top) '' Icc 0 L)) := fun top =>
      ((isConnected_Icc hL.le).image _ (hγ top).continuousOn).image _ hπ.continuousOn
    rw [hB, image_union]
    refine IsConnected.union ?_ (hc true) (hc false)
    -- `(0, L)` on the top edge is glued to `(L, 0)` on the bottom edge
    refine ⟨π (γ true 0), ⟨γ true 0, ⟨0, ⟨le_rfl, hL.le⟩, rfl⟩, rfl⟩, ⟨γ false L, ⟨L, ⟨hL.le, le_rfl⟩, rfl⟩, ?_⟩⟩
    apply Quotient.sound
    apply Relation.EqvGen.symm
    apply Relation.EqvGen.rel
    simp [edgeGlue, γ, hL.le]
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.QuotientTopology.moebius_boundary_connected
-- test: set-moebius-boundary-test.sage

/-- The boundary of the cylinder is two circles: the height `v` descends to the quotient and separates the images of
the top and bottom edges, so their union is not connected. -/
theorem cylinder_boundary_disconnected (L : ℝ) (hL : 0 < L) :
    ¬ IsPreconnected (Quotient.mk (edgeSetoid L false) '' topBottom L) := by
  set π := Quotient.mk (edgeSetoid L false)
  have hv : ∀ a b : square L, edgeSetoid L false a b → a.1.2 = b.1.2 := by
    intro a b h
    induction h with
    | rel x y hxy => simp only [edgeGlue] at hxy; simp [hxy.2.2]
    | refl => rfl
    | symm _ _ _ ih => exact ih.symm
    | trans _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
  let h : Quotient (edgeSetoid L false) → ℝ := Quotient.lift (fun p : square L => p.1.2) hv
  have hc : Continuous h := by
    rw [isQuotientMap_quotient_mk'.continuous_iff]
    exact continuous_snd.comp continuous_subtype_val
  intro hpc
  have hU : IsOpen (h ⁻¹' Iio (L / 2)) := isOpen_Iio.preimage hc
  have hV : IsOpen (h ⁻¹' Ioi (L / 2)) := isOpen_Ioi.preimage hc
  obtain ⟨q, -, hq1, hq2⟩ := hpc _ _ hU hV (by
    rintro _ ⟨p, hp, rfl⟩
    rcases hp with hp | hp
    · left; show p.1.2 < L / 2; rw [hp]; linarith
    · right; show L / 2 < p.1.2; rw [hp]; linarith)
    ⟨π ⟨(0, 0), ⟨le_rfl, hL.le⟩, ⟨le_rfl, hL.le⟩⟩, ⟨_, Or.inl rfl, rfl⟩, by
      simp only [mem_preimage, mem_Iio, h, π, Quotient.lift_mk]; linarith⟩
    ⟨π ⟨(0, L), ⟨le_rfl, hL.le⟩, ⟨hL.le, le_rfl⟩⟩, ⟨_, Or.inr rfl, rfl⟩, by
      simp only [mem_preimage, mem_Ioi, h, π, Quotient.lift_mk]; linarith⟩
  simp only [mem_preimage, mem_Iio, mem_Ioi] at hq1 hq2
  linarith

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.cylinder_boundary_disconnected (1) ((by norm_num)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L), ¬ IsPreconnected (Quotient.mk (edgeSetoid L false) '' topBottom L) := by
    intro L hL
    set π := Quotient.mk (edgeSetoid L false)
    have hv : ∀ a b : square L, edgeSetoid L false a b → a.1.2 = b.1.2 := by
      intro a b h
      induction h with
      | rel x y hxy => simp only [edgeGlue] at hxy; simp [hxy.2.2]
      | refl => rfl
      | symm _ _ _ ih => exact ih.symm
      | trans _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
    let h : Quotient (edgeSetoid L false) → ℝ := Quotient.lift (fun p : square L => p.1.2) hv
    have hc : Continuous h := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      exact continuous_snd.comp continuous_subtype_val
    intro hpc
    have hU : IsOpen (h ⁻¹' Iio (L / 2)) := isOpen_Iio.preimage hc
    have hV : IsOpen (h ⁻¹' Ioi (L / 2)) := isOpen_Ioi.preimage hc
    obtain ⟨q, -, hq1, hq2⟩ := hpc _ _ hU hV (by
      rintro _ ⟨p, hp, rfl⟩
      rcases hp with hp | hp
      · left; show p.1.2 < L / 2; rw [hp]; linarith
      · right; show L / 2 < p.1.2; rw [hp]; linarith)
      ⟨π ⟨(0, 0), ⟨le_rfl, hL.le⟩, ⟨le_rfl, hL.le⟩⟩, ⟨_, Or.inl rfl, rfl⟩, by
        simp only [mem_preimage, mem_Iio, h, π, Quotient.lift_mk]; linarith⟩
      ⟨π ⟨(0, L), ⟨le_rfl, hL.le⟩, ⟨hL.le, le_rfl⟩⟩, ⟨_, Or.inr rfl, rfl⟩, by
        simp only [mem_preimage, mem_Ioi, h, π, Quotient.lift_mk]; linarith⟩
    simp only [mem_preimage, mem_Iio, mem_Ioi] at hq1 hq2
    linarith
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.cylinder_boundary_disconnected (2) ((by norm_num)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L), ¬ IsPreconnected (Quotient.mk (edgeSetoid L false) '' topBottom L) := by
    intro L hL
    set π := Quotient.mk (edgeSetoid L false)
    have hv : ∀ a b : square L, edgeSetoid L false a b → a.1.2 = b.1.2 := by
      intro a b h
      induction h with
      | rel x y hxy => simp only [edgeGlue] at hxy; simp [hxy.2.2]
      | refl => rfl
      | symm _ _ _ ih => exact ih.symm
      | trans _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
    let h : Quotient (edgeSetoid L false) → ℝ := Quotient.lift (fun p : square L => p.1.2) hv
    have hc : Continuous h := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      exact continuous_snd.comp continuous_subtype_val
    intro hpc
    have hU : IsOpen (h ⁻¹' Iio (L / 2)) := isOpen_Iio.preimage hc
    have hV : IsOpen (h ⁻¹' Ioi (L / 2)) := isOpen_Ioi.preimage hc
    obtain ⟨q, -, hq1, hq2⟩ := hpc _ _ hU hV (by
      rintro _ ⟨p, hp, rfl⟩
      rcases hp with hp | hp
      · left; show p.1.2 < L / 2; rw [hp]; linarith
      · right; show L / 2 < p.1.2; rw [hp]; linarith)
      ⟨π ⟨(0, 0), ⟨le_rfl, hL.le⟩, ⟨le_rfl, hL.le⟩⟩, ⟨_, Or.inl rfl, rfl⟩, by
        simp only [mem_preimage, mem_Iio, h, π, Quotient.lift_mk]; linarith⟩
      ⟨π ⟨(0, L), ⟨le_rfl, hL.le⟩, ⟨hL.le, le_rfl⟩⟩, ⟨_, Or.inr rfl, rfl⟩, by
        simp only [mem_preimage, mem_Ioi, h, π, Quotient.lift_mk]; linarith⟩
    simp only [mem_preimage, mem_Iio, mem_Ioi] at hq1 hq2
    linarith
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.QuotientTopology.cylinder_boundary_disconnected (1/2) ((by norm_num)) := by
  have key : ∀ (L : ℝ) (hL : 0 < L), ¬ IsPreconnected (Quotient.mk (edgeSetoid L false) '' topBottom L) := by
    intro L hL
    set π := Quotient.mk (edgeSetoid L false)
    have hv : ∀ a b : square L, edgeSetoid L false a b → a.1.2 = b.1.2 := by
      intro a b h
      induction h with
      | rel x y hxy => simp only [edgeGlue] at hxy; simp [hxy.2.2]
      | refl => rfl
      | symm _ _ _ ih => exact ih.symm
      | trans _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
    let h : Quotient (edgeSetoid L false) → ℝ := Quotient.lift (fun p : square L => p.1.2) hv
    have hc : Continuous h := by
      rw [isQuotientMap_quotient_mk'.continuous_iff]
      exact continuous_snd.comp continuous_subtype_val
    intro hpc
    have hU : IsOpen (h ⁻¹' Iio (L / 2)) := isOpen_Iio.preimage hc
    have hV : IsOpen (h ⁻¹' Ioi (L / 2)) := isOpen_Ioi.preimage hc
    obtain ⟨q, -, hq1, hq2⟩ := hpc _ _ hU hV (by
      rintro _ ⟨p, hp, rfl⟩
      rcases hp with hp | hp
      · left; show p.1.2 < L / 2; rw [hp]; linarith
      · right; show L / 2 < p.1.2; rw [hp]; linarith)
      ⟨π ⟨(0, 0), ⟨le_rfl, hL.le⟩, ⟨le_rfl, hL.le⟩⟩, ⟨_, Or.inl rfl, rfl⟩, by
        simp only [mem_preimage, mem_Iio, h, π, Quotient.lift_mk]; linarith⟩
      ⟨π ⟨(0, L), ⟨le_rfl, hL.le⟩, ⟨hL.le, le_rfl⟩⟩, ⟨_, Or.inr rfl, rfl⟩, by
        simp only [mem_preimage, mem_Ioi, h, π, Quotient.lift_mk]; linarith⟩
    simp only [mem_preimage, mem_Iio, mem_Ioi] at hq1 hq2
    linarith
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.QuotientTopology.cylinder_boundary_disconnected
-- test: set-cylinder-boundary-test.sage

end KaluzaKlein.Sets.QuotientTopology
