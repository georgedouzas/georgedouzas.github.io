import Mathlib
import KaluzaKlein.PartI.Sets.Continuous
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Topological spaces" (homeomorphisms)

The claims of the blocks `def-homeomorphism` to `exr-homeomorphism-invariance` of
`tutorial/part-i-01-sets-relations-topology.qmd`: `tan` from an open interval onto the line, closed versus open
intervals, the continuity of stereographic projection and of its inverse, the continuous bijection `[0, 2π) → S¹` with
discontinuous inverse, the boundary of a square and the figure eight against the circle, `x/(1 - x²)`, `[0, 1)` against
`(0, 1)`, and homeomorphisms as open and closed maps.
-/

namespace KaluzaKlein.Sets.Homeomorphism

open Set Metric Matrix KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Maps KaluzaKlein.Sets.Bijective

/-! ## exm-homeomorphism-interval -/

/-- `x ↦ a tan x` (`a > 0`) is a homeomorphism from `(-π/2, π/2)` onto `ℝ`: an open interval and the whole line are
homeomorphic (`tan` for `a = 1`). -/
theorem tan_homeomorph (a : ℝ) (ha : 0 < a) :
    ∃ e : Ioo (-(Real.pi / 2)) (Real.pi / 2) ≃ₜ ℝ, ∀ x, e x = a * Real.tan x := by
  exact ⟨Real.tanOrderIso.toHomeomorph.trans (Homeomorph.mulLeft₀ a ha.ne'), fun x => rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.tan_homeomorph (1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 < a), ∃ e : Ioo (-(Real.pi / 2)) (Real.pi / 2) ≃ₜ ℝ, ∀ x, e x = a * Real.tan x := by
    intro a ha
    exact ⟨Real.tanOrderIso.toHomeomorph.trans (Homeomorph.mulLeft₀ a ha.ne'), fun x => rfl⟩
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.tan_homeomorph (2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 < a), ∃ e : Ioo (-(Real.pi / 2)) (Real.pi / 2) ≃ₜ ℝ, ∀ x, e x = a * Real.tan x := by
    intro a ha
    exact ⟨Real.tanOrderIso.toHomeomorph.trans (Homeomorph.mulLeft₀ a ha.ne'), fun x => rfl⟩
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.tan_homeomorph (1/2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 < a), ∃ e : Ioo (-(Real.pi / 2)) (Real.pi / 2) ≃ₜ ℝ, ∀ x, e x = a * Real.tan x := by
    intro a ha
    exact ⟨Real.tanOrderIso.toHomeomorph.trans (Homeomorph.mulLeft₀ a ha.ne'), fun x => rfl⟩
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.tan_homeomorph
-- test: set-tan-homeomorph-test.sage

/-- A closed interval is not homeomorphic to an open one (the first is compact, the second is not). -/
theorem Icc_not_homeomorph_Ioo (a b c d : ℝ) (hab : a ≤ b) (hcd : c < d) : IsEmpty (Icc a b ≃ₜ Ioo c d) := by
  refine ⟨fun e => ?_⟩
  haveI : CompactSpace (Icc a b) := isCompact_iff_compactSpace.1 isCompact_Icc
  haveI : CompactSpace (Ioo c d) := e.compactSpace
  have hc : IsCompact (Ioo c d) := isCompact_iff_compactSpace.2 inferInstance
  have hcl := hc.isClosed.closure_eq
  rw [closure_Ioo hcd.ne] at hcl
  have : c ∈ Ioo c d := hcl ▸ left_mem_Icc.2 hcd.le
  exact lt_irrefl c this.1

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.Icc_not_homeomorph_Ioo (0) (1) (0) (1) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (a b c d : ℝ) (hab : a ≤ b) (hcd : c < d), IsEmpty (Icc a b ≃ₜ Ioo c d) := by
    intro a b c d hab hcd
    refine ⟨fun e => ?_⟩
    haveI : CompactSpace (Icc a b) := isCompact_iff_compactSpace.1 isCompact_Icc
    haveI : CompactSpace (Ioo c d) := e.compactSpace
    have hc : IsCompact (Ioo c d) := isCompact_iff_compactSpace.2 inferInstance
    have hcl := hc.isClosed.closure_eq
    rw [closure_Ioo hcd.ne] at hcl
    have : c ∈ Ioo c d := hcl ▸ left_mem_Icc.2 hcd.le
    exact lt_irrefl c this.1
  exact @key (0) (1) (0) (1) ((by norm_num)) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.Icc_not_homeomorph_Ioo (0) (0) (0) (1) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (a b c d : ℝ) (hab : a ≤ b) (hcd : c < d), IsEmpty (Icc a b ≃ₜ Ioo c d) := by
    intro a b c d hab hcd
    refine ⟨fun e => ?_⟩
    haveI : CompactSpace (Icc a b) := isCompact_iff_compactSpace.1 isCompact_Icc
    haveI : CompactSpace (Ioo c d) := e.compactSpace
    have hc : IsCompact (Ioo c d) := isCompact_iff_compactSpace.2 inferInstance
    have hcl := hc.isClosed.closure_eq
    rw [closure_Ioo hcd.ne] at hcl
    have : c ∈ Ioo c d := hcl ▸ left_mem_Icc.2 hcd.le
    exact lt_irrefl c this.1
  exact @key (0) (0) (0) (1) ((by norm_num)) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.Icc_not_homeomorph_Ioo (-1) (2) (3) (5) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (a b c d : ℝ) (hab : a ≤ b) (hcd : c < d), IsEmpty (Icc a b ≃ₜ Ioo c d) := by
    intro a b c d hab hcd
    refine ⟨fun e => ?_⟩
    haveI : CompactSpace (Icc a b) := isCompact_iff_compactSpace.1 isCompact_Icc
    haveI : CompactSpace (Ioo c d) := e.compactSpace
    have hc : IsCompact (Ioo c d) := isCompact_iff_compactSpace.2 inferInstance
    have hcl := hc.isClosed.closure_eq
    rw [closure_Ioo hcd.ne] at hcl
    have : c ∈ Ioo c d := hcl ▸ left_mem_Icc.2 hcd.le
    exact lt_irrefl c this.1
  exact @key (-1) (2) (3) (5) ((by norm_num)) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.Icc_not_homeomorph_Ioo
-- test: set-icc-ioo-test.sage

/-! ## exm-homeomorphism-sphere -/

/-- Projection from the point of height `c` on the `Z`-axis, `(X, Y, Z) ↦ (X, Y)/(c - Z)`, is continuous where
`Z ≠ c`: stereographic projection (`c = 1`) is continuous on the sphere minus the north pole. -/
theorem stereo_continuousOn (c : ℝ) :
    ContinuousOn (fun p : Fin 3 → ℝ => (p 0 / (c - p 2), p 1 / (c - p 2))) {p | p 2 ≠ c} := by
  have h : ∀ p ∈ {p : Fin 3 → ℝ | p 2 ≠ c}, c - p 2 ≠ 0 := fun p hp => sub_ne_zero.2 (Ne.symm hp)
  refine ContinuousOn.prodMk ?_ ?_
  · exact ((continuous_apply 0).continuousOn).div (by fun_prop) h
  · exact ((continuous_apply 1).continuousOn).div (by fun_prop) h

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.stereo_continuousOn (1) := by
  have key : ∀ (c : ℝ), ContinuousOn (fun p : Fin 3 → ℝ => (p 0 / (c - p 2), p 1 / (c - p 2))) {p | p 2 ≠ c} := by
    intro c
    have h : ∀ p ∈ {p : Fin 3 → ℝ | p 2 ≠ c}, c - p 2 ≠ 0 := fun p hp => sub_ne_zero.2 (Ne.symm hp)
    refine ContinuousOn.prodMk ?_ ?_
    · exact ((continuous_apply 0).continuousOn).div (by fun_prop) h
    · exact ((continuous_apply 1).continuousOn).div (by fun_prop) h
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.stereo_continuousOn (0) := by
  have key : ∀ (c : ℝ), ContinuousOn (fun p : Fin 3 → ℝ => (p 0 / (c - p 2), p 1 / (c - p 2))) {p | p 2 ≠ c} := by
    intro c
    have h : ∀ p ∈ {p : Fin 3 → ℝ | p 2 ≠ c}, c - p 2 ≠ 0 := fun p hp => sub_ne_zero.2 (Ne.symm hp)
    refine ContinuousOn.prodMk ?_ ?_
    · exact ((continuous_apply 0).continuousOn).div (by fun_prop) h
    · exact ((continuous_apply 1).continuousOn).div (by fun_prop) h
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.stereo_continuousOn (-1) := by
  have key : ∀ (c : ℝ), ContinuousOn (fun p : Fin 3 → ℝ => (p 0 / (c - p 2), p 1 / (c - p 2))) {p | p 2 ≠ c} := by
    intro c
    have h : ∀ p ∈ {p : Fin 3 → ℝ | p 2 ≠ c}, c - p 2 ≠ 0 := fun p hp => sub_ne_zero.2 (Ne.symm hp)
    refine ContinuousOn.prodMk ?_ ?_
    · exact ((continuous_apply 0).continuousOn).div (by fun_prop) h
    · exact ((continuous_apply 1).continuousOn).div (by fun_prop) h
  exact @key (-1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.stereo_continuousOn
-- test: set-stereo-continuous-test.sage

/-- The inverse `(u, v) ↦ (2u, 2v, u² + v² - r)/(u² + v² + r)` (`r > 0`) is continuous on the whole plane: `σ⁻¹` for
`r = 1`. -/
theorem stereoInv_continuous (r : ℝ) (hr : 0 < r) :
    Continuous (fun q : ℝ × ℝ => ![2 * q.1 / (q.1 ^ 2 + q.2 ^ 2 + r), 2 * q.2 / (q.1 ^ 2 + q.2 ^ 2 + r),
      (q.1 ^ 2 + q.2 ^ 2 - r) / (q.1 ^ 2 + q.2 ^ 2 + r)]) := by
  have h : ∀ q : ℝ × ℝ, q.1 ^ 2 + q.2 ^ 2 + r ≠ 0 := fun q => by positivity
  refine continuous_pi fun i => ?_
  fin_cases i <;> simp only [Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons] <;>
    exact Continuous.div (by fun_prop) (by fun_prop) h

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.stereoInv_continuous (1) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 < r), Continuous (fun q : ℝ × ℝ => ![2 * q.1 / (q.1 ^ 2 + q.2 ^ 2 + r), 2 * q.2 / (q.1 ^ 2 + q.2 ^ 2 + r), (q.1 ^ 2 + q.2 ^ 2 - r) / (q.1 ^ 2 + q.2 ^ 2 + r)]) := by
    intro r hr
    have h : ∀ q : ℝ × ℝ, q.1 ^ 2 + q.2 ^ 2 + r ≠ 0 := fun q => by positivity
    refine continuous_pi fun i => ?_
    fin_cases i <;> simp only [Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons] <;>
      exact Continuous.div (by fun_prop) (by fun_prop) h
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.stereoInv_continuous (2) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 < r), Continuous (fun q : ℝ × ℝ => ![2 * q.1 / (q.1 ^ 2 + q.2 ^ 2 + r), 2 * q.2 / (q.1 ^ 2 + q.2 ^ 2 + r), (q.1 ^ 2 + q.2 ^ 2 - r) / (q.1 ^ 2 + q.2 ^ 2 + r)]) := by
    intro r hr
    have h : ∀ q : ℝ × ℝ, q.1 ^ 2 + q.2 ^ 2 + r ≠ 0 := fun q => by positivity
    refine continuous_pi fun i => ?_
    fin_cases i <;> simp only [Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons] <;>
      exact Continuous.div (by fun_prop) (by fun_prop) h
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.stereoInv_continuous (1/2) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 < r), Continuous (fun q : ℝ × ℝ => ![2 * q.1 / (q.1 ^ 2 + q.2 ^ 2 + r), 2 * q.2 / (q.1 ^ 2 + q.2 ^ 2 + r), (q.1 ^ 2 + q.2 ^ 2 - r) / (q.1 ^ 2 + q.2 ^ 2 + r)]) := by
    intro r hr
    have h : ∀ q : ℝ × ℝ, q.1 ^ 2 + q.2 ^ 2 + r ≠ 0 := fun q => by positivity
    refine continuous_pi fun i => ?_
    fin_cases i <;> simp only [Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons] <;>
      exact Continuous.div (by fun_prop) (by fun_prop) h
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.stereoInv_continuous
-- test: set-stereoinv-continuous-test.sage

/-- `φ ↦ (cos φ, sin φ)` is a continuous bijection from `[a, a + 2π)` onto the circle, but it has no continuous
inverse: a continuous `g` with `g(cos φ, sin φ) = φ` on `[a, a + 2π)` would give `g(wrap a) = a + 2π` by continuity
(`a = 0`: the inverse jumps at `(1, 0)`). -/
theorem wrap_Ico_not_homeomorph (a : ℝ) :
    BijOn wrap (Ico a (a + 2 * Real.pi)) circle ∧
      ¬ ∃ g : ℝ × ℝ → ℝ, ContinuousOn g circle ∧ ∀ φ ∈ Ico a (a + 2 * Real.pi), g (wrap φ) = φ := by
  have hpi := Real.pi_pos
  have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
  constructor
  · obtain ⟨hmaps, hinj, hsurj⟩ := wrap_bijOn a
    refine ⟨fun φ _ => hw φ, ?_, ?_⟩
    · intro s hs t ht hst
      rcases eq_or_lt_of_le hs.1 with rfl | hs1 <;> rcases eq_or_lt_of_le ht.1 with h | ht1
      · exact h
      · exfalso
        exact (hmaps ⟨ht1, ht.2⟩).2 (by rw [← hst]; rfl)
      · exfalso
        exact (hmaps ⟨hs1, hs.2⟩).2 (by rw [hst, ← h]; rfl)
      · exact hinj ⟨hs1, hs.2⟩ ⟨ht1, ht.2⟩ hst
    · intro p hp
      by_cases h : p = wrap a
      · exact ⟨a, ⟨le_rfl, by linarith⟩, h.symm⟩
      · obtain ⟨φ, hφ, rfl⟩ := hsurj ⟨hp, h⟩
        exact ⟨φ, ⟨hφ.1.le, hφ.2⟩, rfl⟩
  · rintro ⟨g, hg, hinv⟩
    have hcont : Continuous (fun φ => wrap φ) := by unfold wrap; fun_prop
    have hh : ContinuousOn (fun φ => g (wrap φ)) univ :=
      hg.comp hcont.continuousOn (fun φ _ => hw φ)
    have heq : EqOn (fun φ => g (wrap φ)) id (Ico a (a + 2 * Real.pi)) := fun φ hφ => hinv φ hφ
    have hcl : EqOn (fun φ => g (wrap φ)) id (closure (Ico a (a + 2 * Real.pi))) :=
      heq.of_subset_closure (hh.mono (subset_univ _)) continuousOn_id subset_closure subset_rfl
    rw [closure_Ico (by linarith)] at hcl
    have h1 := hcl (right_mem_Icc.2 (by linarith : a ≤ a + 2 * Real.pi))
    have h2 : wrap (a + 2 * Real.pi) = wrap a := Injective.wrap_periodic a
    simp only [id] at h1
    rw [h2, hinv a ⟨le_rfl, by linarith⟩] at h1
    linarith

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.wrap_Ico_not_homeomorph (0) := by
  have key : ∀ (a : ℝ), BijOn wrap (Ico a (a + 2 * Real.pi)) circle ∧ ¬ ∃ g : ℝ × ℝ → ℝ, ContinuousOn g circle ∧ ∀ φ ∈ Ico a (a + 2 * Real.pi), g (wrap φ) = φ := by
    intro a
    have hpi := Real.pi_pos
    have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
    constructor
    · obtain ⟨hmaps, hinj, hsurj⟩ := wrap_bijOn a
      refine ⟨fun φ _ => hw φ, ?_, ?_⟩
      · intro s hs t ht hst
        rcases eq_or_lt_of_le hs.1 with rfl | hs1 <;> rcases eq_or_lt_of_le ht.1 with h | ht1
        · exact h
        · exfalso
          exact (hmaps ⟨ht1, ht.2⟩).2 (by rw [← hst]; rfl)
        · exfalso
          exact (hmaps ⟨hs1, hs.2⟩).2 (by rw [hst, ← h]; rfl)
        · exact hinj ⟨hs1, hs.2⟩ ⟨ht1, ht.2⟩ hst
      · intro p hp
        by_cases h : p = wrap a
        · exact ⟨a, ⟨le_rfl, by linarith⟩, h.symm⟩
        · obtain ⟨φ, hφ, rfl⟩ := hsurj ⟨hp, h⟩
          exact ⟨φ, ⟨hφ.1.le, hφ.2⟩, rfl⟩
    · rintro ⟨g, hg, hinv⟩
      have hcont : Continuous (fun φ => wrap φ) := by unfold wrap; fun_prop
      have hh : ContinuousOn (fun φ => g (wrap φ)) univ :=
        hg.comp hcont.continuousOn (fun φ _ => hw φ)
      have heq : EqOn (fun φ => g (wrap φ)) id (Ico a (a + 2 * Real.pi)) := fun φ hφ => hinv φ hφ
      have hcl : EqOn (fun φ => g (wrap φ)) id (closure (Ico a (a + 2 * Real.pi))) :=
        heq.of_subset_closure (hh.mono (subset_univ _)) continuousOn_id subset_closure subset_rfl
      rw [closure_Ico (by linarith)] at hcl
      have h1 := hcl (right_mem_Icc.2 (by linarith : a ≤ a + 2 * Real.pi))
      have h2 : wrap (a + 2 * Real.pi) = wrap a := Injective.wrap_periodic a
      simp only [id] at h1
      rw [h2, hinv a ⟨le_rfl, by linarith⟩] at h1
      linarith
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.wrap_Ico_not_homeomorph (Real.pi) := by
  have key : ∀ (a : ℝ), BijOn wrap (Ico a (a + 2 * Real.pi)) circle ∧ ¬ ∃ g : ℝ × ℝ → ℝ, ContinuousOn g circle ∧ ∀ φ ∈ Ico a (a + 2 * Real.pi), g (wrap φ) = φ := by
    intro a
    have hpi := Real.pi_pos
    have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
    constructor
    · obtain ⟨hmaps, hinj, hsurj⟩ := wrap_bijOn a
      refine ⟨fun φ _ => hw φ, ?_, ?_⟩
      · intro s hs t ht hst
        rcases eq_or_lt_of_le hs.1 with rfl | hs1 <;> rcases eq_or_lt_of_le ht.1 with h | ht1
        · exact h
        · exfalso
          exact (hmaps ⟨ht1, ht.2⟩).2 (by rw [← hst]; rfl)
        · exfalso
          exact (hmaps ⟨hs1, hs.2⟩).2 (by rw [hst, ← h]; rfl)
        · exact hinj ⟨hs1, hs.2⟩ ⟨ht1, ht.2⟩ hst
      · intro p hp
        by_cases h : p = wrap a
        · exact ⟨a, ⟨le_rfl, by linarith⟩, h.symm⟩
        · obtain ⟨φ, hφ, rfl⟩ := hsurj ⟨hp, h⟩
          exact ⟨φ, ⟨hφ.1.le, hφ.2⟩, rfl⟩
    · rintro ⟨g, hg, hinv⟩
      have hcont : Continuous (fun φ => wrap φ) := by unfold wrap; fun_prop
      have hh : ContinuousOn (fun φ => g (wrap φ)) univ :=
        hg.comp hcont.continuousOn (fun φ _ => hw φ)
      have heq : EqOn (fun φ => g (wrap φ)) id (Ico a (a + 2 * Real.pi)) := fun φ hφ => hinv φ hφ
      have hcl : EqOn (fun φ => g (wrap φ)) id (closure (Ico a (a + 2 * Real.pi))) :=
        heq.of_subset_closure (hh.mono (subset_univ _)) continuousOn_id subset_closure subset_rfl
      rw [closure_Ico (by linarith)] at hcl
      have h1 := hcl (right_mem_Icc.2 (by linarith : a ≤ a + 2 * Real.pi))
      have h2 : wrap (a + 2 * Real.pi) = wrap a := Injective.wrap_periodic a
      simp only [id] at h1
      rw [h2, hinv a ⟨le_rfl, by linarith⟩] at h1
      linarith
  exact @key (Real.pi)

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.wrap_Ico_not_homeomorph (-1) := by
  have key : ∀ (a : ℝ), BijOn wrap (Ico a (a + 2 * Real.pi)) circle ∧ ¬ ∃ g : ℝ × ℝ → ℝ, ContinuousOn g circle ∧ ∀ φ ∈ Ico a (a + 2 * Real.pi), g (wrap φ) = φ := by
    intro a
    have hpi := Real.pi_pos
    have hw : ∀ φ, wrap φ ∈ circle := fun φ => by simp [circle, wrap, Real.cos_sq_add_sin_sq]
    constructor
    · obtain ⟨hmaps, hinj, hsurj⟩ := wrap_bijOn a
      refine ⟨fun φ _ => hw φ, ?_, ?_⟩
      · intro s hs t ht hst
        rcases eq_or_lt_of_le hs.1 with rfl | hs1 <;> rcases eq_or_lt_of_le ht.1 with h | ht1
        · exact h
        · exfalso
          exact (hmaps ⟨ht1, ht.2⟩).2 (by rw [← hst]; rfl)
        · exfalso
          exact (hmaps ⟨hs1, hs.2⟩).2 (by rw [hst, ← h]; rfl)
        · exact hinj ⟨hs1, hs.2⟩ ⟨ht1, ht.2⟩ hst
      · intro p hp
        by_cases h : p = wrap a
        · exact ⟨a, ⟨le_rfl, by linarith⟩, h.symm⟩
        · obtain ⟨φ, hφ, rfl⟩ := hsurj ⟨hp, h⟩
          exact ⟨φ, ⟨hφ.1.le, hφ.2⟩, rfl⟩
    · rintro ⟨g, hg, hinv⟩
      have hcont : Continuous (fun φ => wrap φ) := by unfold wrap; fun_prop
      have hh : ContinuousOn (fun φ => g (wrap φ)) univ :=
        hg.comp hcont.continuousOn (fun φ _ => hw φ)
      have heq : EqOn (fun φ => g (wrap φ)) id (Ico a (a + 2 * Real.pi)) := fun φ hφ => hinv φ hφ
      have hcl : EqOn (fun φ => g (wrap φ)) id (closure (Ico a (a + 2 * Real.pi))) :=
        heq.of_subset_closure (hh.mono (subset_univ _)) continuousOn_id subset_closure subset_rfl
      rw [closure_Ico (by linarith)] at hcl
      have h1 := hcl (right_mem_Icc.2 (by linarith : a ≤ a + 2 * Real.pi))
      have h2 : wrap (a + 2 * Real.pi) = wrap a := Injective.wrap_periodic a
      simp only [id] at h1
      rw [h2, hinv a ⟨le_rfl, by linarith⟩] at h1
      linarith
  exact @key (-1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.wrap_Ico_not_homeomorph
-- test: set-wrap-ico-test.sage

/-! ## exm-homeomorphism-square -/

/-- The boundary of the square `[-r, r]²`. -/
def squareBoundary (r : ℝ) : Set (ℝ × ℝ) := {p | max |p.1| |p.2| = r}

/-- The boundary of a square is homeomorphic to the circle by radial projection `x ↦ x/|x|`, with inverse
`y ↦ r y / max(|y₁|, |y₂|)`. -/
theorem square_homeomorph_circle (r : ℝ) (hr : 0 < r) :
    ∃ e : squareBoundary r ≃ₜ circle, ∀ p, (e p : ℝ × ℝ) =
      (p.1.1 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2), p.1.2 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2)) := by
  -- the Euclidean norm is positive on the square boundary, the max norm on the circle
  have hs : ∀ p : squareBoundary r, 0 < Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2) := by
    rintro ⟨⟨x, y⟩, hp⟩
    simp only [squareBoundary, mem_setOf_eq] at hp
    apply Real.sqrt_pos.2
    rcases max_cases |x| |y| with ⟨h1, -⟩ | ⟨h1, -⟩
    · have : |x| = r := h1 ▸ hp; have : x ^ 2 = r ^ 2 := by rw [← sq_abs, this]
      nlinarith [sq_nonneg y]
    · have : |y| = r := h1 ▸ hp; have : y ^ 2 = r ^ 2 := by rw [← sq_abs, this]
      nlinarith [sq_nonneg x]
  have hm : ∀ q : circle, 0 < max |q.1.1| |q.1.2| := by
    rintro ⟨⟨x, y⟩, hq⟩
    simp only [circle, mem_setOf_eq] at hq
    by_contra h
    push_neg at h
    have hx : |x| ≤ 0 := le_trans (le_max_left _ _) h
    have hy : |y| ≤ 0 := le_trans (le_max_right _ _) h
    have : x = 0 := abs_nonpos_iff.1 hx
    have : y = 0 := abs_nonpos_iff.1 hy
    subst_vars; simp at hq
  let F : squareBoundary r → circle := fun p =>
    ⟨(p.1.1 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2), p.1.2 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2)), by
      have h := hs p
      simp only [circle, mem_setOf_eq, div_pow, Real.sq_sqrt (le_of_lt (Real.sqrt_pos.1 h))]
      rw [← add_div, div_self (Real.sqrt_pos.1 h).ne']⟩
  let G : circle → squareBoundary r := fun q =>
    ⟨(r * q.1.1 / max |q.1.1| |q.1.2|, r * q.1.2 / max |q.1.1| |q.1.2|), by
      have h := hm q
      simp only [squareBoundary, mem_setOf_eq, abs_div, abs_mul, abs_of_pos hr, abs_of_pos h]
      rw [max_div_div_right h.le, ← mul_max_of_nonneg _ _ hr.le, mul_div_assoc, div_self h.ne', mul_one]⟩
  have hF : Continuous F := by
    refine Continuous.subtype_mk ?_ _
    refine Continuous.prodMk ?_ ?_ <;>
      exact Continuous.div (by fun_prop) (by fun_prop) (fun p => (hs p).ne')
  have hG : Continuous G := by
    refine Continuous.subtype_mk ?_ _
    refine Continuous.prodMk ?_ ?_ <;>
      exact Continuous.div (by fun_prop) (by fun_prop) (fun q => (hm q).ne')
  have hGF : ∀ p, G (F p) = p := by
    rintro ⟨⟨x, y⟩, hp⟩
    have h := hs ⟨⟨x, y⟩, hp⟩
    simp only [squareBoundary, mem_setOf_eq] at hp
    apply Subtype.ext
    simp only [F, G, abs_div, abs_of_pos h]
    rw [max_div_div_right h.le, hp]
    simp only [Prod.mk.injEq]
    constructor <;> field_simp
  have hFG : ∀ q, F (G q) = q := by
    rintro ⟨⟨x, y⟩, hq⟩
    have h := hm ⟨⟨x, y⟩, hq⟩
    simp only [circle, mem_setOf_eq] at hq
    simp only at h
    apply Subtype.ext
    simp only [F, G]
    have hsq : Real.sqrt ((r * x / max |x| |y|) ^ 2 + (r * y / max |x| |y|) ^ 2) = r / max |x| |y| := by
      rw [div_pow, div_pow, mul_pow, mul_pow, ← add_div, ← mul_add, hq, mul_one, Real.sqrt_div' _ (sq_nonneg _),
        Real.sqrt_sq hr.le, Real.sqrt_sq h.le]
    rw [hsq]
    simp only [Prod.mk.injEq]
    constructor <;> field_simp
  exact ⟨⟨⟨F, G, hGF, hFG⟩, hF, hG⟩, fun p => rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.square_homeomorph_circle (1) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 < r), ∃ e : squareBoundary r ≃ₜ circle, ∀ p, (e p : ℝ × ℝ) = (p.1.1 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2), p.1.2 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2)) := by
    intro r hr
    -- the Euclidean norm is positive on the square boundary, the max norm on the circle
    have hs : ∀ p : squareBoundary r, 0 < Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2) := by
      rintro ⟨⟨x, y⟩, hp⟩
      simp only [squareBoundary, mem_setOf_eq] at hp
      apply Real.sqrt_pos.2
      rcases max_cases |x| |y| with ⟨h1, -⟩ | ⟨h1, -⟩
      · have : |x| = r := h1 ▸ hp; have : x ^ 2 = r ^ 2 := by rw [← sq_abs, this]
        nlinarith [sq_nonneg y]
      · have : |y| = r := h1 ▸ hp; have : y ^ 2 = r ^ 2 := by rw [← sq_abs, this]
        nlinarith [sq_nonneg x]
    have hm : ∀ q : circle, 0 < max |q.1.1| |q.1.2| := by
      rintro ⟨⟨x, y⟩, hq⟩
      simp only [circle, mem_setOf_eq] at hq
      by_contra h
      push_neg at h
      have hx : |x| ≤ 0 := le_trans (le_max_left _ _) h
      have hy : |y| ≤ 0 := le_trans (le_max_right _ _) h
      have : x = 0 := abs_nonpos_iff.1 hx
      have : y = 0 := abs_nonpos_iff.1 hy
      subst_vars; simp at hq
    let F : squareBoundary r → circle := fun p =>
      ⟨(p.1.1 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2), p.1.2 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2)), by
        have h := hs p
        simp only [circle, mem_setOf_eq, div_pow, Real.sq_sqrt (le_of_lt (Real.sqrt_pos.1 h))]
        rw [← add_div, div_self (Real.sqrt_pos.1 h).ne']⟩
    let G : circle → squareBoundary r := fun q =>
      ⟨(r * q.1.1 / max |q.1.1| |q.1.2|, r * q.1.2 / max |q.1.1| |q.1.2|), by
        have h := hm q
        simp only [squareBoundary, mem_setOf_eq, abs_div, abs_mul, abs_of_pos hr, abs_of_pos h]
        rw [max_div_div_right h.le, ← mul_max_of_nonneg _ _ hr.le, mul_div_assoc, div_self h.ne', mul_one]⟩
    have hF : Continuous F := by
      refine Continuous.subtype_mk ?_ _
      refine Continuous.prodMk ?_ ?_ <;>
        exact Continuous.div (by fun_prop) (by fun_prop) (fun p => (hs p).ne')
    have hG : Continuous G := by
      refine Continuous.subtype_mk ?_ _
      refine Continuous.prodMk ?_ ?_ <;>
        exact Continuous.div (by fun_prop) (by fun_prop) (fun q => (hm q).ne')
    have hGF : ∀ p, G (F p) = p := by
      rintro ⟨⟨x, y⟩, hp⟩
      have h := hs ⟨⟨x, y⟩, hp⟩
      simp only [squareBoundary, mem_setOf_eq] at hp
      apply Subtype.ext
      simp only [F, G, abs_div, abs_of_pos h]
      rw [max_div_div_right h.le, hp]
      simp only [Prod.mk.injEq]
      constructor <;> field_simp
    have hFG : ∀ q, F (G q) = q := by
      rintro ⟨⟨x, y⟩, hq⟩
      have h := hm ⟨⟨x, y⟩, hq⟩
      simp only [circle, mem_setOf_eq] at hq
      simp only at h
      apply Subtype.ext
      simp only [F, G]
      have hsq : Real.sqrt ((r * x / max |x| |y|) ^ 2 + (r * y / max |x| |y|) ^ 2) = r / max |x| |y| := by
        rw [div_pow, div_pow, mul_pow, mul_pow, ← add_div, ← mul_add, hq, mul_one, Real.sqrt_div' _ (sq_nonneg _),
          Real.sqrt_sq hr.le, Real.sqrt_sq h.le]
      rw [hsq]
      simp only [Prod.mk.injEq]
      constructor <;> field_simp
    exact ⟨⟨⟨F, G, hGF, hFG⟩, hF, hG⟩, fun p => rfl⟩
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.square_homeomorph_circle (2) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 < r), ∃ e : squareBoundary r ≃ₜ circle, ∀ p, (e p : ℝ × ℝ) = (p.1.1 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2), p.1.2 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2)) := by
    intro r hr
    -- the Euclidean norm is positive on the square boundary, the max norm on the circle
    have hs : ∀ p : squareBoundary r, 0 < Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2) := by
      rintro ⟨⟨x, y⟩, hp⟩
      simp only [squareBoundary, mem_setOf_eq] at hp
      apply Real.sqrt_pos.2
      rcases max_cases |x| |y| with ⟨h1, -⟩ | ⟨h1, -⟩
      · have : |x| = r := h1 ▸ hp; have : x ^ 2 = r ^ 2 := by rw [← sq_abs, this]
        nlinarith [sq_nonneg y]
      · have : |y| = r := h1 ▸ hp; have : y ^ 2 = r ^ 2 := by rw [← sq_abs, this]
        nlinarith [sq_nonneg x]
    have hm : ∀ q : circle, 0 < max |q.1.1| |q.1.2| := by
      rintro ⟨⟨x, y⟩, hq⟩
      simp only [circle, mem_setOf_eq] at hq
      by_contra h
      push_neg at h
      have hx : |x| ≤ 0 := le_trans (le_max_left _ _) h
      have hy : |y| ≤ 0 := le_trans (le_max_right _ _) h
      have : x = 0 := abs_nonpos_iff.1 hx
      have : y = 0 := abs_nonpos_iff.1 hy
      subst_vars; simp at hq
    let F : squareBoundary r → circle := fun p =>
      ⟨(p.1.1 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2), p.1.2 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2)), by
        have h := hs p
        simp only [circle, mem_setOf_eq, div_pow, Real.sq_sqrt (le_of_lt (Real.sqrt_pos.1 h))]
        rw [← add_div, div_self (Real.sqrt_pos.1 h).ne']⟩
    let G : circle → squareBoundary r := fun q =>
      ⟨(r * q.1.1 / max |q.1.1| |q.1.2|, r * q.1.2 / max |q.1.1| |q.1.2|), by
        have h := hm q
        simp only [squareBoundary, mem_setOf_eq, abs_div, abs_mul, abs_of_pos hr, abs_of_pos h]
        rw [max_div_div_right h.le, ← mul_max_of_nonneg _ _ hr.le, mul_div_assoc, div_self h.ne', mul_one]⟩
    have hF : Continuous F := by
      refine Continuous.subtype_mk ?_ _
      refine Continuous.prodMk ?_ ?_ <;>
        exact Continuous.div (by fun_prop) (by fun_prop) (fun p => (hs p).ne')
    have hG : Continuous G := by
      refine Continuous.subtype_mk ?_ _
      refine Continuous.prodMk ?_ ?_ <;>
        exact Continuous.div (by fun_prop) (by fun_prop) (fun q => (hm q).ne')
    have hGF : ∀ p, G (F p) = p := by
      rintro ⟨⟨x, y⟩, hp⟩
      have h := hs ⟨⟨x, y⟩, hp⟩
      simp only [squareBoundary, mem_setOf_eq] at hp
      apply Subtype.ext
      simp only [F, G, abs_div, abs_of_pos h]
      rw [max_div_div_right h.le, hp]
      simp only [Prod.mk.injEq]
      constructor <;> field_simp
    have hFG : ∀ q, F (G q) = q := by
      rintro ⟨⟨x, y⟩, hq⟩
      have h := hm ⟨⟨x, y⟩, hq⟩
      simp only [circle, mem_setOf_eq] at hq
      simp only at h
      apply Subtype.ext
      simp only [F, G]
      have hsq : Real.sqrt ((r * x / max |x| |y|) ^ 2 + (r * y / max |x| |y|) ^ 2) = r / max |x| |y| := by
        rw [div_pow, div_pow, mul_pow, mul_pow, ← add_div, ← mul_add, hq, mul_one, Real.sqrt_div' _ (sq_nonneg _),
          Real.sqrt_sq hr.le, Real.sqrt_sq h.le]
      rw [hsq]
      simp only [Prod.mk.injEq]
      constructor <;> field_simp
    exact ⟨⟨⟨F, G, hGF, hFG⟩, hF, hG⟩, fun p => rfl⟩
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.square_homeomorph_circle (1/2) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 < r), ∃ e : squareBoundary r ≃ₜ circle, ∀ p, (e p : ℝ × ℝ) = (p.1.1 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2), p.1.2 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2)) := by
    intro r hr
    -- the Euclidean norm is positive on the square boundary, the max norm on the circle
    have hs : ∀ p : squareBoundary r, 0 < Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2) := by
      rintro ⟨⟨x, y⟩, hp⟩
      simp only [squareBoundary, mem_setOf_eq] at hp
      apply Real.sqrt_pos.2
      rcases max_cases |x| |y| with ⟨h1, -⟩ | ⟨h1, -⟩
      · have : |x| = r := h1 ▸ hp; have : x ^ 2 = r ^ 2 := by rw [← sq_abs, this]
        nlinarith [sq_nonneg y]
      · have : |y| = r := h1 ▸ hp; have : y ^ 2 = r ^ 2 := by rw [← sq_abs, this]
        nlinarith [sq_nonneg x]
    have hm : ∀ q : circle, 0 < max |q.1.1| |q.1.2| := by
      rintro ⟨⟨x, y⟩, hq⟩
      simp only [circle, mem_setOf_eq] at hq
      by_contra h
      push_neg at h
      have hx : |x| ≤ 0 := le_trans (le_max_left _ _) h
      have hy : |y| ≤ 0 := le_trans (le_max_right _ _) h
      have : x = 0 := abs_nonpos_iff.1 hx
      have : y = 0 := abs_nonpos_iff.1 hy
      subst_vars; simp at hq
    let F : squareBoundary r → circle := fun p =>
      ⟨(p.1.1 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2), p.1.2 / Real.sqrt (p.1.1 ^ 2 + p.1.2 ^ 2)), by
        have h := hs p
        simp only [circle, mem_setOf_eq, div_pow, Real.sq_sqrt (le_of_lt (Real.sqrt_pos.1 h))]
        rw [← add_div, div_self (Real.sqrt_pos.1 h).ne']⟩
    let G : circle → squareBoundary r := fun q =>
      ⟨(r * q.1.1 / max |q.1.1| |q.1.2|, r * q.1.2 / max |q.1.1| |q.1.2|), by
        have h := hm q
        simp only [squareBoundary, mem_setOf_eq, abs_div, abs_mul, abs_of_pos hr, abs_of_pos h]
        rw [max_div_div_right h.le, ← mul_max_of_nonneg _ _ hr.le, mul_div_assoc, div_self h.ne', mul_one]⟩
    have hF : Continuous F := by
      refine Continuous.subtype_mk ?_ _
      refine Continuous.prodMk ?_ ?_ <;>
        exact Continuous.div (by fun_prop) (by fun_prop) (fun p => (hs p).ne')
    have hG : Continuous G := by
      refine Continuous.subtype_mk ?_ _
      refine Continuous.prodMk ?_ ?_ <;>
        exact Continuous.div (by fun_prop) (by fun_prop) (fun q => (hm q).ne')
    have hGF : ∀ p, G (F p) = p := by
      rintro ⟨⟨x, y⟩, hp⟩
      have h := hs ⟨⟨x, y⟩, hp⟩
      simp only [squareBoundary, mem_setOf_eq] at hp
      apply Subtype.ext
      simp only [F, G, abs_div, abs_of_pos h]
      rw [max_div_div_right h.le, hp]
      simp only [Prod.mk.injEq]
      constructor <;> field_simp
    have hFG : ∀ q, F (G q) = q := by
      rintro ⟨⟨x, y⟩, hq⟩
      have h := hm ⟨⟨x, y⟩, hq⟩
      simp only [circle, mem_setOf_eq] at hq
      simp only at h
      apply Subtype.ext
      simp only [F, G]
      have hsq : Real.sqrt ((r * x / max |x| |y|) ^ 2 + (r * y / max |x| |y|) ^ 2) = r / max |x| |y| := by
        rw [div_pow, div_pow, mul_pow, mul_pow, ← add_div, ← mul_add, hq, mul_one, Real.sqrt_div' _ (sq_nonneg _),
          Real.sqrt_sq hr.le, Real.sqrt_sq h.le]
      rw [hsq]
      simp only [Prod.mk.injEq]
      constructor <;> field_simp
    exact ⟨⟨⟨F, G, hGF, hFG⟩, hF, hG⟩, fun p => rfl⟩
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.square_homeomorph_circle
-- test: set-square-circle-test.sage

/-- The figure eight: two circles of radius `r` touching at the origin. -/
def figureEight (r : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - r) ^ 2 + p.2 ^ 2 = r ^ 2} ∪ {p | (p.1 + r) ^ 2 + p.2 ^ 2 = r ^ 2}

/-- The circle minus any point is connected: it is the image of an open interval of angles. -/
theorem circle_minus_point_connected (a : ℝ) : IsConnected (circle \ {wrap a}) := by
  rw [← (wrap_bijOn a).image_eq]
  exact (isConnected_Ioo (by linarith [Real.pi_pos])).image _
    (show Continuous wrap by unfold wrap; fun_prop).continuousOn

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.circle_minus_point_connected (0) := by
  have key : ∀ (a : ℝ), IsConnected (circle \ {wrap a}) := by
    intro a
    rw [← (wrap_bijOn a).image_eq]
    exact (isConnected_Ioo (by linarith [Real.pi_pos])).image _
      (show Continuous wrap by unfold wrap; fun_prop).continuousOn
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.circle_minus_point_connected (Real.pi) := by
  have key : ∀ (a : ℝ), IsConnected (circle \ {wrap a}) := by
    intro a
    rw [← (wrap_bijOn a).image_eq]
    exact (isConnected_Ioo (by linarith [Real.pi_pos])).image _
      (show Continuous wrap by unfold wrap; fun_prop).continuousOn
  exact @key (Real.pi)

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.circle_minus_point_connected (1) := by
  have key : ∀ (a : ℝ), IsConnected (circle \ {wrap a}) := by
    intro a
    rw [← (wrap_bijOn a).image_eq]
    exact (isConnected_Ioo (by linarith [Real.pi_pos])).image _
      (show Continuous wrap by unfold wrap; fun_prop).continuousOn
  exact @key (1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.circle_minus_point_connected
-- test: set-circle-minus-point-test.sage

/-- The figure eight minus its crossing point is not connected: it splits into the parts `X > 0` and `X < 0`. -/
theorem figureEight_minus_origin_not_connected (r : ℝ) (hr : 0 < r) :
    ¬ IsPreconnected (figureEight r \ {(0, 0)}) := by
  intro h
  have hU : IsOpen {p : ℝ × ℝ | 0 < p.1} := isOpen_lt continuous_const continuous_fst
  have hV : IsOpen {p : ℝ × ℝ | p.1 < 0} := isOpen_lt continuous_fst continuous_const
  have hcover : figureEight r \ {(0, 0)} ⊆ {p : ℝ × ℝ | 0 < p.1} ∪ {p | p.1 < 0} := by
    rintro ⟨x, y⟩ ⟨hp, hne⟩
    simp only [mem_singleton_iff, Prod.mk.injEq, not_and] at hne
    simp only [mem_union, mem_setOf_eq]
    rcases lt_trichotomy x 0 with hx | rfl | hx
    · right; exact hx
    · exfalso
      rcases hp with hp | hp <;> simp only [mem_setOf_eq] at hp <;>
        exact hne rfl (by nlinarith [sq_nonneg y])
    · left; exact hx
  obtain ⟨p, -, hp1, hp2⟩ := h _ _ hU hV hcover
    ⟨(2 * r, 0), ⟨Or.inl (by simp only [mem_setOf_eq]; ring), by
      simp only [mem_singleton_iff, Prod.mk.injEq, not_and]; intro h; linarith⟩, by
      simp only [mem_setOf_eq]; linarith⟩
    ⟨(-(2 * r), 0), ⟨Or.inr (by simp only [mem_setOf_eq]; ring), by
      simp only [mem_singleton_iff, Prod.mk.injEq, not_and]; intro h; linarith⟩, by
      simp only [mem_setOf_eq]; linarith⟩
  have h1 : 0 < p.1 := hp1
  have h2 : p.1 < 0 := hp2
  linarith

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.figureEight_minus_origin_not_connected (1) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 < r), ¬ IsPreconnected (figureEight r \ {(0, 0)}) := by
    intro r hr
    intro h
    have hU : IsOpen {p : ℝ × ℝ | 0 < p.1} := isOpen_lt continuous_const continuous_fst
    have hV : IsOpen {p : ℝ × ℝ | p.1 < 0} := isOpen_lt continuous_fst continuous_const
    have hcover : figureEight r \ {(0, 0)} ⊆ {p : ℝ × ℝ | 0 < p.1} ∪ {p | p.1 < 0} := by
      rintro ⟨x, y⟩ ⟨hp, hne⟩
      simp only [mem_singleton_iff, Prod.mk.injEq, not_and] at hne
      simp only [mem_union, mem_setOf_eq]
      rcases lt_trichotomy x 0 with hx | rfl | hx
      · right; exact hx
      · exfalso
        rcases hp with hp | hp <;> simp only [mem_setOf_eq] at hp <;>
          exact hne rfl (by nlinarith [sq_nonneg y])
      · left; exact hx
    obtain ⟨p, -, hp1, hp2⟩ := h _ _ hU hV hcover
      ⟨(2 * r, 0), ⟨Or.inl (by simp only [mem_setOf_eq]; ring), by
        simp only [mem_singleton_iff, Prod.mk.injEq, not_and]; intro h; linarith⟩, by
        simp only [mem_setOf_eq]; linarith⟩
      ⟨(-(2 * r), 0), ⟨Or.inr (by simp only [mem_setOf_eq]; ring), by
        simp only [mem_singleton_iff, Prod.mk.injEq, not_and]; intro h; linarith⟩, by
        simp only [mem_setOf_eq]; linarith⟩
    have h1 : 0 < p.1 := hp1
    have h2 : p.1 < 0 := hp2
    linarith
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.figureEight_minus_origin_not_connected (2) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 < r), ¬ IsPreconnected (figureEight r \ {(0, 0)}) := by
    intro r hr
    intro h
    have hU : IsOpen {p : ℝ × ℝ | 0 < p.1} := isOpen_lt continuous_const continuous_fst
    have hV : IsOpen {p : ℝ × ℝ | p.1 < 0} := isOpen_lt continuous_fst continuous_const
    have hcover : figureEight r \ {(0, 0)} ⊆ {p : ℝ × ℝ | 0 < p.1} ∪ {p | p.1 < 0} := by
      rintro ⟨x, y⟩ ⟨hp, hne⟩
      simp only [mem_singleton_iff, Prod.mk.injEq, not_and] at hne
      simp only [mem_union, mem_setOf_eq]
      rcases lt_trichotomy x 0 with hx | rfl | hx
      · right; exact hx
      · exfalso
        rcases hp with hp | hp <;> simp only [mem_setOf_eq] at hp <;>
          exact hne rfl (by nlinarith [sq_nonneg y])
      · left; exact hx
    obtain ⟨p, -, hp1, hp2⟩ := h _ _ hU hV hcover
      ⟨(2 * r, 0), ⟨Or.inl (by simp only [mem_setOf_eq]; ring), by
        simp only [mem_singleton_iff, Prod.mk.injEq, not_and]; intro h; linarith⟩, by
        simp only [mem_setOf_eq]; linarith⟩
      ⟨(-(2 * r), 0), ⟨Or.inr (by simp only [mem_setOf_eq]; ring), by
        simp only [mem_singleton_iff, Prod.mk.injEq, not_and]; intro h; linarith⟩, by
        simp only [mem_setOf_eq]; linarith⟩
    have h1 : 0 < p.1 := hp1
    have h2 : p.1 < 0 := hp2
    linarith
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.figureEight_minus_origin_not_connected (1/2) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 < r), ¬ IsPreconnected (figureEight r \ {(0, 0)}) := by
    intro r hr
    intro h
    have hU : IsOpen {p : ℝ × ℝ | 0 < p.1} := isOpen_lt continuous_const continuous_fst
    have hV : IsOpen {p : ℝ × ℝ | p.1 < 0} := isOpen_lt continuous_fst continuous_const
    have hcover : figureEight r \ {(0, 0)} ⊆ {p : ℝ × ℝ | 0 < p.1} ∪ {p | p.1 < 0} := by
      rintro ⟨x, y⟩ ⟨hp, hne⟩
      simp only [mem_singleton_iff, Prod.mk.injEq, not_and] at hne
      simp only [mem_union, mem_setOf_eq]
      rcases lt_trichotomy x 0 with hx | rfl | hx
      · right; exact hx
      · exfalso
        rcases hp with hp | hp <;> simp only [mem_setOf_eq] at hp <;>
          exact hne rfl (by nlinarith [sq_nonneg y])
      · left; exact hx
    obtain ⟨p, -, hp1, hp2⟩ := h _ _ hU hV hcover
      ⟨(2 * r, 0), ⟨Or.inl (by simp only [mem_setOf_eq]; ring), by
        simp only [mem_singleton_iff, Prod.mk.injEq, not_and]; intro h; linarith⟩, by
        simp only [mem_setOf_eq]; linarith⟩
      ⟨(-(2 * r), 0), ⟨Or.inr (by simp only [mem_setOf_eq]; ring), by
        simp only [mem_singleton_iff, Prod.mk.injEq, not_and]; intro h; linarith⟩, by
        simp only [mem_setOf_eq]; linarith⟩
    have h1 : 0 < p.1 := hp1
    have h2 : p.1 < 0 := hp2
    linarith
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.figureEight_minus_origin_not_connected
-- test: set-eight-minus-origin-test.sage

/-- A figure eight is not homeomorphic to the circle: removing the crossing point leaves two pieces, while removing a
point from the circle leaves one. -/
theorem figureEight_not_homeomorph_circle (r : ℝ) (hr : 0 < r) : IsEmpty (figureEight r ≃ₜ circle) := by
  refine ⟨fun e => ?_⟩
  have ho : ((0 : ℝ), (0 : ℝ)) ∈ figureEight r := Or.inl (by simp only [mem_setOf_eq]; ring)
  set o : figureEight r := ⟨(0, 0), ho⟩
  obtain ⟨a, ha⟩ : ∃ a, wrap a = (e o : ℝ × ℝ) := by
    have h := (e o).2
    have h' : (e o : ℝ × ℝ) ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
      rw [range_wrap 1 zero_le_one]; simpa [circle] using h
    obtain ⟨φ, hφ⟩ := h'
    exact ⟨φ, by simpa [wrap] using hφ⟩
  have hconn := circle_minus_point_connected a
  have hsub : IsPreconnected ((Subtype.val : circle → ℝ × ℝ) ⁻¹' (circle \ {wrap a})) := by
    rw [← Topology.IsInducing.subtypeVal.isPreconnected_image]
    rw [Subtype.image_preimage_coe, inter_eq_right.2 diff_subset]
    exact hconn.isPreconnected
  have himg := hsub.image _ ((continuous_subtype_val.comp e.symm.continuous).continuousOn)
  apply figureEight_minus_origin_not_connected r hr
  convert himg using 1
  ext p
  simp only [mem_diff, mem_singleton_iff, mem_image, mem_preimage, Function.comp_apply]
  constructor
  · rintro ⟨hp, hne⟩
    refine ⟨e ⟨p, hp⟩, ⟨(e ⟨p, hp⟩).2, ?_⟩, by simp⟩
    rw [ha]
    intro h
    apply hne
    have := e.injective (Subtype.ext h)
    exact congrArg Subtype.val this
  · rintro ⟨q, ⟨-, hq⟩, rfl⟩
    refine ⟨(e.symm q).2, fun h => hq ?_⟩
    rw [ha]
    have : e.symm q = o := Subtype.ext h
    rw [← this, e.apply_symm_apply]

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.figureEight_not_homeomorph_circle (1) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 < r), IsEmpty (figureEight r ≃ₜ circle) := by
    intro r hr
    refine ⟨fun e => ?_⟩
    have ho : ((0 : ℝ), (0 : ℝ)) ∈ figureEight r := Or.inl (by simp only [mem_setOf_eq]; ring)
    set o : figureEight r := ⟨(0, 0), ho⟩
    obtain ⟨a, ha⟩ : ∃ a, wrap a = (e o : ℝ × ℝ) := by
      have h := (e o).2
      have h' : (e o : ℝ × ℝ) ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using h
      obtain ⟨φ, hφ⟩ := h'
      exact ⟨φ, by simpa [wrap] using hφ⟩
    have hconn := circle_minus_point_connected a
    have hsub : IsPreconnected ((Subtype.val : circle → ℝ × ℝ) ⁻¹' (circle \ {wrap a})) := by
      rw [← Topology.IsInducing.subtypeVal.isPreconnected_image]
      rw [Subtype.image_preimage_coe, inter_eq_right.2 diff_subset]
      exact hconn.isPreconnected
    have himg := hsub.image _ ((continuous_subtype_val.comp e.symm.continuous).continuousOn)
    apply figureEight_minus_origin_not_connected r hr
    convert himg using 1
    ext p
    simp only [mem_diff, mem_singleton_iff, mem_image, mem_preimage, Function.comp_apply]
    constructor
    · rintro ⟨hp, hne⟩
      refine ⟨e ⟨p, hp⟩, ⟨(e ⟨p, hp⟩).2, ?_⟩, by simp⟩
      rw [ha]
      intro h
      apply hne
      have := e.injective (Subtype.ext h)
      exact congrArg Subtype.val this
    · rintro ⟨q, ⟨-, hq⟩, rfl⟩
      refine ⟨(e.symm q).2, fun h => hq ?_⟩
      rw [ha]
      have : e.symm q = o := Subtype.ext h
      rw [← this, e.apply_symm_apply]
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.figureEight_not_homeomorph_circle (2) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 < r), IsEmpty (figureEight r ≃ₜ circle) := by
    intro r hr
    refine ⟨fun e => ?_⟩
    have ho : ((0 : ℝ), (0 : ℝ)) ∈ figureEight r := Or.inl (by simp only [mem_setOf_eq]; ring)
    set o : figureEight r := ⟨(0, 0), ho⟩
    obtain ⟨a, ha⟩ : ∃ a, wrap a = (e o : ℝ × ℝ) := by
      have h := (e o).2
      have h' : (e o : ℝ × ℝ) ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using h
      obtain ⟨φ, hφ⟩ := h'
      exact ⟨φ, by simpa [wrap] using hφ⟩
    have hconn := circle_minus_point_connected a
    have hsub : IsPreconnected ((Subtype.val : circle → ℝ × ℝ) ⁻¹' (circle \ {wrap a})) := by
      rw [← Topology.IsInducing.subtypeVal.isPreconnected_image]
      rw [Subtype.image_preimage_coe, inter_eq_right.2 diff_subset]
      exact hconn.isPreconnected
    have himg := hsub.image _ ((continuous_subtype_val.comp e.symm.continuous).continuousOn)
    apply figureEight_minus_origin_not_connected r hr
    convert himg using 1
    ext p
    simp only [mem_diff, mem_singleton_iff, mem_image, mem_preimage, Function.comp_apply]
    constructor
    · rintro ⟨hp, hne⟩
      refine ⟨e ⟨p, hp⟩, ⟨(e ⟨p, hp⟩).2, ?_⟩, by simp⟩
      rw [ha]
      intro h
      apply hne
      have := e.injective (Subtype.ext h)
      exact congrArg Subtype.val this
    · rintro ⟨q, ⟨-, hq⟩, rfl⟩
      refine ⟨(e.symm q).2, fun h => hq ?_⟩
      rw [ha]
      have : e.symm q = o := Subtype.ext h
      rw [← this, e.apply_symm_apply]
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.figureEight_not_homeomorph_circle (1/2) ((by norm_num)) := by
  have key : ∀ (r : ℝ) (hr : 0 < r), IsEmpty (figureEight r ≃ₜ circle) := by
    intro r hr
    refine ⟨fun e => ?_⟩
    have ho : ((0 : ℝ), (0 : ℝ)) ∈ figureEight r := Or.inl (by simp only [mem_setOf_eq]; ring)
    set o : figureEight r := ⟨(0, 0), ho⟩
    obtain ⟨a, ha⟩ : ∃ a, wrap a = (e o : ℝ × ℝ) := by
      have h := (e o).2
      have h' : (e o : ℝ × ℝ) ∈ Set.range (fun φ => (1 * Real.cos φ, 1 * Real.sin φ)) := by
        rw [range_wrap 1 zero_le_one]; simpa [circle] using h
      obtain ⟨φ, hφ⟩ := h'
      exact ⟨φ, by simpa [wrap] using hφ⟩
    have hconn := circle_minus_point_connected a
    have hsub : IsPreconnected ((Subtype.val : circle → ℝ × ℝ) ⁻¹' (circle \ {wrap a})) := by
      rw [← Topology.IsInducing.subtypeVal.isPreconnected_image]
      rw [Subtype.image_preimage_coe, inter_eq_right.2 diff_subset]
      exact hconn.isPreconnected
    have himg := hsub.image _ ((continuous_subtype_val.comp e.symm.continuous).continuousOn)
    apply figureEight_minus_origin_not_connected r hr
    convert himg using 1
    ext p
    simp only [mem_diff, mem_singleton_iff, mem_image, mem_preimage, Function.comp_apply]
    constructor
    · rintro ⟨hp, hne⟩
      refine ⟨e ⟨p, hp⟩, ⟨(e ⟨p, hp⟩).2, ?_⟩, by simp⟩
      rw [ha]
      intro h
      apply hne
      have := e.injective (Subtype.ext h)
      exact congrArg Subtype.val this
    · rintro ⟨q, ⟨-, hq⟩, rfl⟩
      refine ⟨(e.symm q).2, fun h => hq ?_⟩
      rw [ha]
      have : e.symm q = o := Subtype.ext h
      rw [← this, e.apply_symm_apply]
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.figureEight_not_homeomorph_circle
-- test: set-eight-circle-test.sage

/-! ## exr-homeomorphism-tan -/

/-- `x ↦ x/(1 - x²)` has derivative `(1 + x²)/(1 - x²)²` where `x² ≠ 1`. -/
theorem deriv_x_div (x : ℝ) (hx : 1 - x ^ 2 ≠ 0) :
    deriv (fun x : ℝ => x / (1 - x ^ 2)) x = (1 + x ^ 2) / (1 - x ^ 2) ^ 2 := by
  have h := (hasDerivAt_id x).div ((hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_pow 2 x)) hx
  refine h.deriv.trans ?_
  simp only [Pi.sub_apply, id]
  field_simp
  ring

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.deriv_x_div (0) ((by norm_num)) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (x : ℝ) (hx : 1 - x ^ 2 ≠ 0), deriv (fun x : ℝ => x / (1 - x ^ 2)) x = (1 + x ^ 2) / (1 - x ^ 2) ^ 2 := by
    intro x hx
    have h := (hasDerivAt_id x).div ((hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_pow 2 x)) hx
    refine h.deriv.trans ?_
    simp only [Pi.sub_apply, id]
    field_simp
    ring
  exact @key (0) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.deriv_x_div (1/2) ((by norm_num)) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (x : ℝ) (hx : 1 - x ^ 2 ≠ 0), deriv (fun x : ℝ => x / (1 - x ^ 2)) x = (1 + x ^ 2) / (1 - x ^ 2) ^ 2 := by
    intro x hx
    have h := (hasDerivAt_id x).div ((hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_pow 2 x)) hx
    refine h.deriv.trans ?_
    simp only [Pi.sub_apply, id]
    field_simp
    ring
  exact @key (1/2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.deriv_x_div (2) ((by norm_num)) := by
  set_option maxHeartbeats 2000000 in
  have key : ∀ (x : ℝ) (hx : 1 - x ^ 2 ≠ 0), deriv (fun x : ℝ => x / (1 - x ^ 2)) x = (1 + x ^ 2) / (1 - x ^ 2) ^ 2 := by
    intro x hx
    have h := (hasDerivAt_id x).div ((hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_pow 2 x)) hx
    refine h.deriv.trans ?_
    simp only [Pi.sub_apply, id]
    field_simp
    ring
  exact @key (2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.deriv_x_div
-- test: set-deriv-x-div-test.sage

/-- The derivative `(1 + x²)/(1 - x²)²` is positive on `(-1, 1)`. -/
theorem deriv_x_div_pos (x : ℝ) (hx : x ∈ Ioo (-1 : ℝ) 1) : 0 < (1 + x ^ 2) / (1 - x ^ 2) ^ 2 := by
  have : 0 < 1 - x ^ 2 := by nlinarith [hx.1, hx.2]
  positivity

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.deriv_x_div_pos (0) ((by constructor <;> norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.deriv_x_div_pos (1/2) ((by constructor <;> norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.deriv_x_div_pos (-9/10) ((by constructor <;> norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.deriv_x_div_pos
-- test: set-deriv-pos-test.sage

/-- `x ↦ a x/(1 - x²)` (`a > 0`) is a homeomorphism from `(-1, 1)` onto `ℝ`, with inverse
`y ↦ 2y/(a + √(a² + 4y²))` (`a = 1`). -/
theorem x_div_homeomorph (a : ℝ) (ha : 0 < a) :
    ∃ e : Ioo (-1 : ℝ) 1 ≃ₜ ℝ, ∀ x, e x = a * x / (1 - (x : ℝ) ^ 2) := by
  have hden : ∀ x : Ioo (-1 : ℝ) 1, 0 < 1 - (x : ℝ) ^ 2 := fun x => by nlinarith [x.2.1, x.2.2]
  have hroot : ∀ y : ℝ, 0 < a + Real.sqrt (a ^ 2 + 4 * y ^ 2) := fun y => by positivity
  have hsq : ∀ y : ℝ, Real.sqrt (a ^ 2 + 4 * y ^ 2) ^ 2 = a ^ 2 + 4 * y ^ 2 := fun y => Real.sq_sqrt (by positivity)
  have hlt : ∀ y : ℝ, |2 * y| < a + Real.sqrt (a ^ 2 + 4 * y ^ 2) := by
    intro y
    have h1 : |2 * y| ≤ Real.sqrt (a ^ 2 + 4 * y ^ 2) := by
      rw [← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt (by nlinarith)
    linarith
  let G : ℝ → Ioo (-1 : ℝ) 1 := fun y => ⟨2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2)), by
    have h := hroot y
    have h2 := abs_lt.1 (hlt y)
    constructor
    · rw [lt_div_iff₀ h]; linarith
    · rw [div_lt_iff₀ h]; linarith⟩
  have hF : Continuous (fun x : Ioo (-1 : ℝ) 1 => a * x / (1 - (x : ℝ) ^ 2)) :=
    Continuous.div (by fun_prop) (by fun_prop) (fun x => (hden x).ne')
  have hG : Continuous G := by
    refine Continuous.subtype_mk ?_ _
    exact Continuous.div (by fun_prop) (by fun_prop) (fun y => (hroot y).ne')
  have hFG : ∀ y, a * (G y : ℝ) / (1 - (G y : ℝ) ^ 2) = y := by
    intro y
    have h := hroot y
    have hs := hsq y
    have h1 : 1 - (2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2))) ^ 2 =
        2 * a / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2)) := by
      have hs' : Real.sqrt (a ^ 2 + y ^ 2 * 4) ^ 2 = a ^ 2 + y ^ 2 * 4 := Real.sq_sqrt (by positivity)
      field_simp
      nlinarith [hs, hs']
    show a * (2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2))) /
      (1 - (2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2))) ^ 2) = y
    rw [h1]
    field_simp
  have hGF : ∀ x : Ioo (-1 : ℝ) 1, G (a * x / (1 - (x : ℝ) ^ 2)) = x := by
    intro x
    apply Subtype.ext
    have hd := hden x
    set t : ℝ := (x : ℝ)
    simp only [G]
    have hs : Real.sqrt (a ^ 2 + 4 * (a * t / (1 - t ^ 2)) ^ 2) = a * (1 + t ^ 2) / (1 - t ^ 2) := by
      rw [Real.sqrt_eq_iff_mul_self_eq (by positivity) (by positivity)]
      field_simp
      ring
    rw [hs]
    field_simp
    ring
  exact ⟨⟨⟨fun x => a * x / (1 - (x : ℝ) ^ 2), G, hGF, hFG⟩, hF, hG⟩, fun x => rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.x_div_homeomorph (1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 < a), ∃ e : Ioo (-1 : ℝ) 1 ≃ₜ ℝ, ∀ x, e x = a * x / (1 - (x : ℝ) ^ 2) := by
    intro a ha
    have hden : ∀ x : Ioo (-1 : ℝ) 1, 0 < 1 - (x : ℝ) ^ 2 := fun x => by nlinarith [x.2.1, x.2.2]
    have hroot : ∀ y : ℝ, 0 < a + Real.sqrt (a ^ 2 + 4 * y ^ 2) := fun y => by positivity
    have hsq : ∀ y : ℝ, Real.sqrt (a ^ 2 + 4 * y ^ 2) ^ 2 = a ^ 2 + 4 * y ^ 2 := fun y => Real.sq_sqrt (by positivity)
    have hlt : ∀ y : ℝ, |2 * y| < a + Real.sqrt (a ^ 2 + 4 * y ^ 2) := by
      intro y
      have h1 : |2 * y| ≤ Real.sqrt (a ^ 2 + 4 * y ^ 2) := by
        rw [← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt (by nlinarith)
      linarith
    let G : ℝ → Ioo (-1 : ℝ) 1 := fun y => ⟨2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2)), by
      have h := hroot y
      have h2 := abs_lt.1 (hlt y)
      constructor
      · rw [lt_div_iff₀ h]; linarith
      · rw [div_lt_iff₀ h]; linarith⟩
    have hF : Continuous (fun x : Ioo (-1 : ℝ) 1 => a * x / (1 - (x : ℝ) ^ 2)) :=
      Continuous.div (by fun_prop) (by fun_prop) (fun x => (hden x).ne')
    have hG : Continuous G := by
      refine Continuous.subtype_mk ?_ _
      exact Continuous.div (by fun_prop) (by fun_prop) (fun y => (hroot y).ne')
    have hFG : ∀ y, a * (G y : ℝ) / (1 - (G y : ℝ) ^ 2) = y := by
      intro y
      have h := hroot y
      have hs := hsq y
      have h1 : 1 - (2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2))) ^ 2 =
          2 * a / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2)) := by
        have hs' : Real.sqrt (a ^ 2 + y ^ 2 * 4) ^ 2 = a ^ 2 + y ^ 2 * 4 := Real.sq_sqrt (by positivity)
        field_simp
        nlinarith [hs, hs']
      show a * (2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2))) /
        (1 - (2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2))) ^ 2) = y
      rw [h1]
      field_simp
    have hGF : ∀ x : Ioo (-1 : ℝ) 1, G (a * x / (1 - (x : ℝ) ^ 2)) = x := by
      intro x
      apply Subtype.ext
      have hd := hden x
      set t : ℝ := (x : ℝ)
      simp only [G]
      have hs : Real.sqrt (a ^ 2 + 4 * (a * t / (1 - t ^ 2)) ^ 2) = a * (1 + t ^ 2) / (1 - t ^ 2) := by
        rw [Real.sqrt_eq_iff_mul_self_eq (by positivity) (by positivity)]
        field_simp
        ring
      rw [hs]
      field_simp
      ring
    exact ⟨⟨⟨fun x => a * x / (1 - (x : ℝ) ^ 2), G, hGF, hFG⟩, hF, hG⟩, fun x => rfl⟩
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.x_div_homeomorph (2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 < a), ∃ e : Ioo (-1 : ℝ) 1 ≃ₜ ℝ, ∀ x, e x = a * x / (1 - (x : ℝ) ^ 2) := by
    intro a ha
    have hden : ∀ x : Ioo (-1 : ℝ) 1, 0 < 1 - (x : ℝ) ^ 2 := fun x => by nlinarith [x.2.1, x.2.2]
    have hroot : ∀ y : ℝ, 0 < a + Real.sqrt (a ^ 2 + 4 * y ^ 2) := fun y => by positivity
    have hsq : ∀ y : ℝ, Real.sqrt (a ^ 2 + 4 * y ^ 2) ^ 2 = a ^ 2 + 4 * y ^ 2 := fun y => Real.sq_sqrt (by positivity)
    have hlt : ∀ y : ℝ, |2 * y| < a + Real.sqrt (a ^ 2 + 4 * y ^ 2) := by
      intro y
      have h1 : |2 * y| ≤ Real.sqrt (a ^ 2 + 4 * y ^ 2) := by
        rw [← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt (by nlinarith)
      linarith
    let G : ℝ → Ioo (-1 : ℝ) 1 := fun y => ⟨2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2)), by
      have h := hroot y
      have h2 := abs_lt.1 (hlt y)
      constructor
      · rw [lt_div_iff₀ h]; linarith
      · rw [div_lt_iff₀ h]; linarith⟩
    have hF : Continuous (fun x : Ioo (-1 : ℝ) 1 => a * x / (1 - (x : ℝ) ^ 2)) :=
      Continuous.div (by fun_prop) (by fun_prop) (fun x => (hden x).ne')
    have hG : Continuous G := by
      refine Continuous.subtype_mk ?_ _
      exact Continuous.div (by fun_prop) (by fun_prop) (fun y => (hroot y).ne')
    have hFG : ∀ y, a * (G y : ℝ) / (1 - (G y : ℝ) ^ 2) = y := by
      intro y
      have h := hroot y
      have hs := hsq y
      have h1 : 1 - (2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2))) ^ 2 =
          2 * a / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2)) := by
        have hs' : Real.sqrt (a ^ 2 + y ^ 2 * 4) ^ 2 = a ^ 2 + y ^ 2 * 4 := Real.sq_sqrt (by positivity)
        field_simp
        nlinarith [hs, hs']
      show a * (2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2))) /
        (1 - (2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2))) ^ 2) = y
      rw [h1]
      field_simp
    have hGF : ∀ x : Ioo (-1 : ℝ) 1, G (a * x / (1 - (x : ℝ) ^ 2)) = x := by
      intro x
      apply Subtype.ext
      have hd := hden x
      set t : ℝ := (x : ℝ)
      simp only [G]
      have hs : Real.sqrt (a ^ 2 + 4 * (a * t / (1 - t ^ 2)) ^ 2) = a * (1 + t ^ 2) / (1 - t ^ 2) := by
        rw [Real.sqrt_eq_iff_mul_self_eq (by positivity) (by positivity)]
        field_simp
        ring
      rw [hs]
      field_simp
      ring
    exact ⟨⟨⟨fun x => a * x / (1 - (x : ℝ) ^ 2), G, hGF, hFG⟩, hF, hG⟩, fun x => rfl⟩
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.x_div_homeomorph (1/3) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : 0 < a), ∃ e : Ioo (-1 : ℝ) 1 ≃ₜ ℝ, ∀ x, e x = a * x / (1 - (x : ℝ) ^ 2) := by
    intro a ha
    have hden : ∀ x : Ioo (-1 : ℝ) 1, 0 < 1 - (x : ℝ) ^ 2 := fun x => by nlinarith [x.2.1, x.2.2]
    have hroot : ∀ y : ℝ, 0 < a + Real.sqrt (a ^ 2 + 4 * y ^ 2) := fun y => by positivity
    have hsq : ∀ y : ℝ, Real.sqrt (a ^ 2 + 4 * y ^ 2) ^ 2 = a ^ 2 + 4 * y ^ 2 := fun y => Real.sq_sqrt (by positivity)
    have hlt : ∀ y : ℝ, |2 * y| < a + Real.sqrt (a ^ 2 + 4 * y ^ 2) := by
      intro y
      have h1 : |2 * y| ≤ Real.sqrt (a ^ 2 + 4 * y ^ 2) := by
        rw [← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt (by nlinarith)
      linarith
    let G : ℝ → Ioo (-1 : ℝ) 1 := fun y => ⟨2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2)), by
      have h := hroot y
      have h2 := abs_lt.1 (hlt y)
      constructor
      · rw [lt_div_iff₀ h]; linarith
      · rw [div_lt_iff₀ h]; linarith⟩
    have hF : Continuous (fun x : Ioo (-1 : ℝ) 1 => a * x / (1 - (x : ℝ) ^ 2)) :=
      Continuous.div (by fun_prop) (by fun_prop) (fun x => (hden x).ne')
    have hG : Continuous G := by
      refine Continuous.subtype_mk ?_ _
      exact Continuous.div (by fun_prop) (by fun_prop) (fun y => (hroot y).ne')
    have hFG : ∀ y, a * (G y : ℝ) / (1 - (G y : ℝ) ^ 2) = y := by
      intro y
      have h := hroot y
      have hs := hsq y
      have h1 : 1 - (2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2))) ^ 2 =
          2 * a / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2)) := by
        have hs' : Real.sqrt (a ^ 2 + y ^ 2 * 4) ^ 2 = a ^ 2 + y ^ 2 * 4 := Real.sq_sqrt (by positivity)
        field_simp
        nlinarith [hs, hs']
      show a * (2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2))) /
        (1 - (2 * y / (a + Real.sqrt (a ^ 2 + 4 * y ^ 2))) ^ 2) = y
      rw [h1]
      field_simp
    have hGF : ∀ x : Ioo (-1 : ℝ) 1, G (a * x / (1 - (x : ℝ) ^ 2)) = x := by
      intro x
      apply Subtype.ext
      have hd := hden x
      set t : ℝ := (x : ℝ)
      simp only [G]
      have hs : Real.sqrt (a ^ 2 + 4 * (a * t / (1 - t ^ 2)) ^ 2) = a * (1 + t ^ 2) / (1 - t ^ 2) := by
        rw [Real.sqrt_eq_iff_mul_self_eq (by positivity) (by positivity)]
        field_simp
        ring
      rw [hs]
      field_simp
      ring
    exact ⟨⟨⟨fun x => a * x / (1 - (x : ℝ) ^ 2), G, hGF, hFG⟩, hF, hG⟩, fun x => rfl⟩
  exact @key (1/3) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.x_div_homeomorph
-- test: set-x-div-homeomorph-test.sage

/-! ## exr-homeomorphism-half-open -/

/-- `[a, b)` and `(c, d)` are not homeomorphic: removing `a` leaves the connected `(a, b)`, while removing any point
`f(a)` from `(c, d)` leaves two pieces (`[0, 1)` and `(0, 1)`). -/
theorem Ico_not_homeomorph_Ioo (a b c d : ℝ) (hab : a < b) (hcd : c < d) : IsEmpty (Ico a b ≃ₜ Ioo c d) := by
  refine ⟨fun e => ?_⟩
  set o : Ico a b := ⟨a, left_mem_Ico.2 hab⟩
  set m : ℝ := ((e o : Ioo c d) : ℝ)
  have hm : m ∈ Ioo c d := (e o).2
  -- `(a, b)` is connected; its image in `(c, d)` (in `ℝ`) is `(c, d)` minus `m`
  have hconn : IsPreconnected ((Subtype.val : Ico a b → ℝ) ⁻¹' Ioo a b) := by
    rw [← Topology.IsInducing.subtypeVal.isPreconnected_image, Subtype.image_preimage_coe,
      inter_eq_right.2 Ioo_subset_Ico_self]
    exact isPreconnected_Ioo
  have himg := hconn.image _ ((continuous_subtype_val.comp e.continuous).continuousOn)
  have hset : (Subtype.val ∘ e) '' ((Subtype.val : Ico a b → ℝ) ⁻¹' Ioo a b) = Ioo c d \ {m} := by
    ext y
    simp only [mem_image, mem_preimage, Function.comp_apply, mem_diff, mem_singleton_iff]
    constructor
    · rintro ⟨x, hx, rfl⟩
      refine ⟨(e x).2, fun h => ?_⟩
      have : e x = e o := Subtype.ext h
      have := e.injective this
      rw [this] at hx
      exact lt_irrefl a hx.1
    · rintro ⟨hy, hne⟩
      refine ⟨e.symm ⟨y, hy⟩, ⟨lt_of_le_of_ne (e.symm ⟨y, hy⟩).2.1 ?_, (e.symm ⟨y, hy⟩).2.2⟩, by simp⟩
      intro h
      apply hne
      have : e.symm ⟨y, hy⟩ = o := Subtype.ext h.symm
      have := congrArg (fun z => (e z : ℝ)) this
      simpa using this
  rw [hset] at himg
  have h1 : (c + m) / 2 ∈ Ioo c d \ {m} := ⟨⟨by linarith [hm.1], by linarith [hm.1, hm.2]⟩, by
    simp only [mem_singleton_iff]; intro h; linarith [hm.1]⟩
  have h2 : (m + d) / 2 ∈ Ioo c d \ {m} := ⟨⟨by linarith [hm.1, hm.2], by linarith [hm.2]⟩, by
    simp only [mem_singleton_iff]; intro h; linarith [hm.2]⟩
  have := himg.Icc_subset h1 h2 (show m ∈ Icc ((c + m) / 2) ((m + d) / 2) from
    ⟨by linarith [hm.1], by linarith [hm.2]⟩)
  exact this.2 (Set.mem_singleton m)

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.Ico_not_homeomorph_Ioo (0) (1) (0) (1) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (a b c d : ℝ) (hab : a < b) (hcd : c < d), IsEmpty (Ico a b ≃ₜ Ioo c d) := by
    intro a b c d hab hcd
    refine ⟨fun e => ?_⟩
    set o : Ico a b := ⟨a, left_mem_Ico.2 hab⟩
    set m : ℝ := ((e o : Ioo c d) : ℝ)
    have hm : m ∈ Ioo c d := (e o).2
    -- `(a, b)` is connected; its image in `(c, d)` (in `ℝ`) is `(c, d)` minus `m`
    have hconn : IsPreconnected ((Subtype.val : Ico a b → ℝ) ⁻¹' Ioo a b) := by
      rw [← Topology.IsInducing.subtypeVal.isPreconnected_image, Subtype.image_preimage_coe,
        inter_eq_right.2 Ioo_subset_Ico_self]
      exact isPreconnected_Ioo
    have himg := hconn.image _ ((continuous_subtype_val.comp e.continuous).continuousOn)
    have hset : (Subtype.val ∘ e) '' ((Subtype.val : Ico a b → ℝ) ⁻¹' Ioo a b) = Ioo c d \ {m} := by
      ext y
      simp only [mem_image, mem_preimage, Function.comp_apply, mem_diff, mem_singleton_iff]
      constructor
      · rintro ⟨x, hx, rfl⟩
        refine ⟨(e x).2, fun h => ?_⟩
        have : e x = e o := Subtype.ext h
        have := e.injective this
        rw [this] at hx
        exact lt_irrefl a hx.1
      · rintro ⟨hy, hne⟩
        refine ⟨e.symm ⟨y, hy⟩, ⟨lt_of_le_of_ne (e.symm ⟨y, hy⟩).2.1 ?_, (e.symm ⟨y, hy⟩).2.2⟩, by simp⟩
        intro h
        apply hne
        have : e.symm ⟨y, hy⟩ = o := Subtype.ext h.symm
        have := congrArg (fun z => (e z : ℝ)) this
        simpa using this
    rw [hset] at himg
    have h1 : (c + m) / 2 ∈ Ioo c d \ {m} := ⟨⟨by linarith [hm.1], by linarith [hm.1, hm.2]⟩, by
      simp only [mem_singleton_iff]; intro h; linarith [hm.1]⟩
    have h2 : (m + d) / 2 ∈ Ioo c d \ {m} := ⟨⟨by linarith [hm.1, hm.2], by linarith [hm.2]⟩, by
      simp only [mem_singleton_iff]; intro h; linarith [hm.2]⟩
    have := himg.Icc_subset h1 h2 (show m ∈ Icc ((c + m) / 2) ((m + d) / 2) from
      ⟨by linarith [hm.1], by linarith [hm.2]⟩)
    exact this.2 (Set.mem_singleton m)
  exact @key (0) (1) (0) (1) ((by norm_num)) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.Ico_not_homeomorph_Ioo (-1) (1) (2) (5) ((by norm_num)) ((by norm_num)) := by
  have key : ∀ (a b c d : ℝ) (hab : a < b) (hcd : c < d), IsEmpty (Ico a b ≃ₜ Ioo c d) := by
    intro a b c d hab hcd
    refine ⟨fun e => ?_⟩
    set o : Ico a b := ⟨a, left_mem_Ico.2 hab⟩
    set m : ℝ := ((e o : Ioo c d) : ℝ)
    have hm : m ∈ Ioo c d := (e o).2
    -- `(a, b)` is connected; its image in `(c, d)` (in `ℝ`) is `(c, d)` minus `m`
    have hconn : IsPreconnected ((Subtype.val : Ico a b → ℝ) ⁻¹' Ioo a b) := by
      rw [← Topology.IsInducing.subtypeVal.isPreconnected_image, Subtype.image_preimage_coe,
        inter_eq_right.2 Ioo_subset_Ico_self]
      exact isPreconnected_Ioo
    have himg := hconn.image _ ((continuous_subtype_val.comp e.continuous).continuousOn)
    have hset : (Subtype.val ∘ e) '' ((Subtype.val : Ico a b → ℝ) ⁻¹' Ioo a b) = Ioo c d \ {m} := by
      ext y
      simp only [mem_image, mem_preimage, Function.comp_apply, mem_diff, mem_singleton_iff]
      constructor
      · rintro ⟨x, hx, rfl⟩
        refine ⟨(e x).2, fun h => ?_⟩
        have : e x = e o := Subtype.ext h
        have := e.injective this
        rw [this] at hx
        exact lt_irrefl a hx.1
      · rintro ⟨hy, hne⟩
        refine ⟨e.symm ⟨y, hy⟩, ⟨lt_of_le_of_ne (e.symm ⟨y, hy⟩).2.1 ?_, (e.symm ⟨y, hy⟩).2.2⟩, by simp⟩
        intro h
        apply hne
        have : e.symm ⟨y, hy⟩ = o := Subtype.ext h.symm
        have := congrArg (fun z => (e z : ℝ)) this
        simpa using this
    rw [hset] at himg
    have h1 : (c + m) / 2 ∈ Ioo c d \ {m} := ⟨⟨by linarith [hm.1], by linarith [hm.1, hm.2]⟩, by
      simp only [mem_singleton_iff]; intro h; linarith [hm.1]⟩
    have h2 : (m + d) / 2 ∈ Ioo c d \ {m} := ⟨⟨by linarith [hm.1, hm.2], by linarith [hm.2]⟩, by
      simp only [mem_singleton_iff]; intro h; linarith [hm.2]⟩
    have := himg.Icc_subset h1 h2 (show m ∈ Icc ((c + m) / 2) ((m + d) / 2) from
      ⟨by linarith [hm.1], by linarith [hm.2]⟩)
    exact this.2 (Set.mem_singleton m)
  exact @key (-1) (1) (2) (5) ((by norm_num)) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.Ico_not_homeomorph_Ioo (0) (2 * Real.pi) (0) (2 * Real.pi) ((by positivity)) ((by positivity)) := by
  have key : ∀ (a b c d : ℝ) (hab : a < b) (hcd : c < d), IsEmpty (Ico a b ≃ₜ Ioo c d) := by
    intro a b c d hab hcd
    refine ⟨fun e => ?_⟩
    set o : Ico a b := ⟨a, left_mem_Ico.2 hab⟩
    set m : ℝ := ((e o : Ioo c d) : ℝ)
    have hm : m ∈ Ioo c d := (e o).2
    -- `(a, b)` is connected; its image in `(c, d)` (in `ℝ`) is `(c, d)` minus `m`
    have hconn : IsPreconnected ((Subtype.val : Ico a b → ℝ) ⁻¹' Ioo a b) := by
      rw [← Topology.IsInducing.subtypeVal.isPreconnected_image, Subtype.image_preimage_coe,
        inter_eq_right.2 Ioo_subset_Ico_self]
      exact isPreconnected_Ioo
    have himg := hconn.image _ ((continuous_subtype_val.comp e.continuous).continuousOn)
    have hset : (Subtype.val ∘ e) '' ((Subtype.val : Ico a b → ℝ) ⁻¹' Ioo a b) = Ioo c d \ {m} := by
      ext y
      simp only [mem_image, mem_preimage, Function.comp_apply, mem_diff, mem_singleton_iff]
      constructor
      · rintro ⟨x, hx, rfl⟩
        refine ⟨(e x).2, fun h => ?_⟩
        have : e x = e o := Subtype.ext h
        have := e.injective this
        rw [this] at hx
        exact lt_irrefl a hx.1
      · rintro ⟨hy, hne⟩
        refine ⟨e.symm ⟨y, hy⟩, ⟨lt_of_le_of_ne (e.symm ⟨y, hy⟩).2.1 ?_, (e.symm ⟨y, hy⟩).2.2⟩, by simp⟩
        intro h
        apply hne
        have : e.symm ⟨y, hy⟩ = o := Subtype.ext h.symm
        have := congrArg (fun z => (e z : ℝ)) this
        simpa using this
    rw [hset] at himg
    have h1 : (c + m) / 2 ∈ Ioo c d \ {m} := ⟨⟨by linarith [hm.1], by linarith [hm.1, hm.2]⟩, by
      simp only [mem_singleton_iff]; intro h; linarith [hm.1]⟩
    have h2 : (m + d) / 2 ∈ Ioo c d \ {m} := ⟨⟨by linarith [hm.1, hm.2], by linarith [hm.2]⟩, by
      simp only [mem_singleton_iff]; intro h; linarith [hm.2]⟩
    have := himg.Icc_subset h1 h2 (show m ∈ Icc ((c + m) / 2) ((m + d) / 2) from
      ⟨by linarith [hm.1], by linarith [hm.2]⟩)
    exact this.2 (Set.mem_singleton m)
  exact @key (0) (2 * Real.pi) (0) (2 * Real.pi) ((by positivity)) ((by positivity))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.Ico_not_homeomorph_Ioo
-- test: set-ico-ioo-test.sage

/-! ## exr-homeomorphism-invariance -/

/-- A homeomorphism sends open sets to open sets and closed sets to closed sets. -/
theorem homeomorph_open_closed (X Y : Type) [TopologicalSpace X] [TopologicalSpace Y] (f : X ≃ₜ Y) (U : Set X) :
    (IsOpen U → IsOpen (f '' U)) ∧ (IsClosed U → IsClosed (f '' U)) :=
  ⟨fun h => f.isOpenMap U h, fun h => f.isClosedMap U h⟩

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.homeomorph_open_closed (ℝ) (ℝ) (inferInstance) (inferInstance) (Homeomorph.refl ℝ) (Set.Ioo 0 1) := by
  have key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] (f : X ≃ₜ Y) (U : Set X), (IsOpen U → IsOpen (f '' U)) ∧ (IsClosed U → IsClosed (f '' U)) := by
    intro X Y inst1 inst2 f U
    exact (⟨fun h => f.isOpenMap U h, fun h => f.isClosedMap U h⟩)
  exact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (Homeomorph.refl ℝ) (Set.Ioo 0 1)

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.homeomorph_open_closed (ℝ) (ℝ) (inferInstance) (inferInstance) (Homeomorph.neg ℝ) (Set.Icc 0 1) := by
  have key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] (f : X ≃ₜ Y) (U : Set X), (IsOpen U → IsOpen (f '' U)) ∧ (IsClosed U → IsClosed (f '' U)) := by
    intro X Y inst1 inst2 f U
    exact (⟨fun h => f.isOpenMap U h, fun h => f.isClosedMap U h⟩)
  exact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (Homeomorph.neg ℝ) (Set.Icc 0 1)

example : veriphy_instance% KaluzaKlein.Sets.Homeomorphism.homeomorph_open_closed (ℝ) (ℝ) (inferInstance) (inferInstance) (Homeomorph.addRight 1) (Set.univ) := by
  have key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] (f : X ≃ₜ Y) (U : Set X), (IsOpen U → IsOpen (f '' U)) ∧ (IsClosed U → IsClosed (f '' U)) := by
    intro X Y inst1 inst2 f U
    exact (⟨fun h => f.isOpenMap U h, fun h => f.isClosedMap U h⟩)
  exact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (Homeomorph.addRight 1) (Set.univ)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Homeomorphism.homeomorph_open_closed
-- test: set-homeomorph-open-test.sage

end KaluzaKlein.Sets.Homeomorphism
