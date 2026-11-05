import Mathlib
import KaluzaKlein.PartI.Sets.Subspace
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Topological spaces" (continuous maps)

The claims of the blocks `def-continuous-map` to `exr-continuous-composition` of
`tutorial/part-i-01-sets-relations-topology.qmd`: the ε-δ form of continuity, elementary functions and the step
function, the parametrisations of the circle and the sphere, matrix maps, the preimages of `(½, 2)` under the step
function and `cos`, the floor function, and compositions with the orbit map.
-/

namespace KaluzaKlein.Sets.Continuous

open Set Metric Matrix KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Maps KaluzaKlein.Sets.Injective

/-- The step function at `a`: `0` for `x < a`, `1` for `x ≥ a`. -/
noncomputable def step (a x : ℝ) : ℝ := if x < a then 0 else 1

/-! ## def-continuous-map -/

/-- Between metric spaces (subsets of `ℝⁿ`), continuity is the ε-δ condition. -/
theorem continuous_iff_eps_delta (X Y : Type) [MetricSpace X] [MetricSpace Y] (f : X → Y) :
    Continuous f ↔ ∀ p, ∀ ε > 0, ∃ δ > 0, ∀ x, dist x p < δ → dist (f x) (f p) < ε :=
  Metric.continuous_iff

example : veriphy_instance% KaluzaKlein.Sets.Continuous.continuous_iff_eps_delta (ℝ) (ℝ) (inferInstance) (inferInstance) (Real.sin) := by
  have key : ∀ (X Y : Type) [inst1 : MetricSpace X] [inst2 : MetricSpace Y] (f : X → Y), Continuous f ↔ ∀ p, ∀ ε > 0, ∃ δ > 0, ∀ x, dist x p < δ → dist (f x) (f p) < ε := by
    intro X Y inst1 inst2 f
    exact (Metric.continuous_iff)
  exact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (Real.sin)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.continuous_iff_eps_delta (ℝ) (ℝ) (inferInstance) (inferInstance) (id) := by
  have key : ∀ (X Y : Type) [inst1 : MetricSpace X] [inst2 : MetricSpace Y] (f : X → Y), Continuous f ↔ ∀ p, ∀ ε > 0, ∃ δ > 0, ∀ x, dist x p < δ → dist (f x) (f p) < ε := by
    intro X Y inst1 inst2 f
    exact (Metric.continuous_iff)
  exact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (id)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.continuous_iff_eps_delta (ℝ) (ℝ) (inferInstance) (inferInstance) (fun x => x ^ 2) := by
  have key : ∀ (X Y : Type) [inst1 : MetricSpace X] [inst2 : MetricSpace Y] (f : X → Y), Continuous f ↔ ∀ p, ∀ ε > 0, ∃ δ > 0, ∀ x, dist x p < δ → dist (f x) (f p) < ε := by
    intro X Y inst1 inst2 f
    exact (Metric.continuous_iff)
  exact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (fun x => x ^ 2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Continuous.continuous_iff_eps_delta
-- test: set-eps-delta-test.sage

/-! ## exm-continuous-elementary -/

/-- Combinations of `cos`, `sin`, `exp` and a polynomial are continuous on `ℝ`. -/
theorem elementary_continuous (a b c : ℝ) (p : Polynomial ℝ) :
    Continuous (fun x => a * Real.cos x + b * Real.sin x + c * Real.exp x + p.eval x) := by
  have := p.continuous
  fun_prop

example : veriphy_instance% KaluzaKlein.Sets.Continuous.elementary_continuous (1) (1) (1) (Polynomial.X) := by
  have key : ∀ (a b c : ℝ) (p : Polynomial ℝ), Continuous (fun x => a * Real.cos x + b * Real.sin x + c * Real.exp x + p.eval x) := by
    intro a b c p
    have := p.continuous
    fun_prop
  exact @key (1) (1) (1) (Polynomial.X)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.elementary_continuous (0) (0) (0) (0) := by
  have key : ∀ (a b c : ℝ) (p : Polynomial ℝ), Continuous (fun x => a * Real.cos x + b * Real.sin x + c * Real.exp x + p.eval x) := by
    intro a b c p
    have := p.continuous
    fun_prop
  exact @key (0) (0) (0) (0)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.elementary_continuous (2) (-1) (3) (Polynomial.X ^ 3 - Polynomial.X) := by
  have key : ∀ (a b c : ℝ) (p : Polynomial ℝ), Continuous (fun x => a * Real.cos x + b * Real.sin x + c * Real.exp x + p.eval x) := by
    intro a b c p
    have := p.continuous
    fun_prop
  exact @key (2) (-1) (3) (Polynomial.X ^ 3 - Polynomial.X)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Continuous.elementary_continuous
-- test: set-elementary-test.sage

/-- The step function at `a` is not continuous: the preimage of the open interval `(½, 2)` is `[a, ∞)`, which is not
open (`a = 0`). -/
theorem step_not_continuous (a : ℝ) :
    step a ⁻¹' Ioo (1 / 2) 2 = Ici a ∧ ¬ IsOpen (Ici a) ∧ ¬ Continuous (step a) := by
  have hpre : step a ⁻¹' Ioo (1 / 2) 2 = Ici a := by
    ext x
    simp only [step, mem_preimage, mem_Ioo, mem_Ici]
    split_ifs with h
    · constructor
      · rintro ⟨h1, -⟩; norm_num at h1
      · intro h2; linarith
    · constructor
      · intro _; linarith
      · intro _; norm_num
  have hno : ¬ IsOpen (Ici a) := fun h => by
    have := h.mem_nhds (self_mem_Ici : a ∈ Ici a)
    rw [mem_nhds_iff_exists_Ioo_subset] at this
    obtain ⟨l, u, ⟨hl, hu⟩, hsub⟩ := this
    have := hsub (show (l + a) / 2 ∈ Ioo l u from ⟨by linarith, by linarith⟩)
    simp only [mem_Ici] at this
    linarith
  refine ⟨hpre, hno, fun hc => hno ?_⟩
  rw [← hpre]
  exact isOpen_Ioo.preimage hc

example : veriphy_instance% KaluzaKlein.Sets.Continuous.step_not_continuous (0) := by
  have key : ∀ (a : ℝ), step a ⁻¹' Ioo (1 / 2) 2 = Ici a ∧ ¬ IsOpen (Ici a) ∧ ¬ Continuous (step a) := by
    intro a
    have hpre : step a ⁻¹' Ioo (1 / 2) 2 = Ici a := by
      ext x
      simp only [step, mem_preimage, mem_Ioo, mem_Ici]
      split_ifs with h
      · constructor
        · rintro ⟨h1, -⟩; norm_num at h1
        · intro h2; linarith
      · constructor
        · intro _; linarith
        · intro _; norm_num
    have hno : ¬ IsOpen (Ici a) := fun h => by
      have := h.mem_nhds (self_mem_Ici : a ∈ Ici a)
      rw [mem_nhds_iff_exists_Ioo_subset] at this
      obtain ⟨l, u, ⟨hl, hu⟩, hsub⟩ := this
      have := hsub (show (l + a) / 2 ∈ Ioo l u from ⟨by linarith, by linarith⟩)
      simp only [mem_Ici] at this
      linarith
    refine ⟨hpre, hno, fun hc => hno ?_⟩
    rw [← hpre]
    exact isOpen_Ioo.preimage hc
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.step_not_continuous (1) := by
  have key : ∀ (a : ℝ), step a ⁻¹' Ioo (1 / 2) 2 = Ici a ∧ ¬ IsOpen (Ici a) ∧ ¬ Continuous (step a) := by
    intro a
    have hpre : step a ⁻¹' Ioo (1 / 2) 2 = Ici a := by
      ext x
      simp only [step, mem_preimage, mem_Ioo, mem_Ici]
      split_ifs with h
      · constructor
        · rintro ⟨h1, -⟩; norm_num at h1
        · intro h2; linarith
      · constructor
        · intro _; linarith
        · intro _; norm_num
    have hno : ¬ IsOpen (Ici a) := fun h => by
      have := h.mem_nhds (self_mem_Ici : a ∈ Ici a)
      rw [mem_nhds_iff_exists_Ioo_subset] at this
      obtain ⟨l, u, ⟨hl, hu⟩, hsub⟩ := this
      have := hsub (show (l + a) / 2 ∈ Ioo l u from ⟨by linarith, by linarith⟩)
      simp only [mem_Ici] at this
      linarith
    refine ⟨hpre, hno, fun hc => hno ?_⟩
    rw [← hpre]
    exact isOpen_Ioo.preimage hc
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.step_not_continuous (-2) := by
  have key : ∀ (a : ℝ), step a ⁻¹' Ioo (1 / 2) 2 = Ici a ∧ ¬ IsOpen (Ici a) ∧ ¬ Continuous (step a) := by
    intro a
    have hpre : step a ⁻¹' Ioo (1 / 2) 2 = Ici a := by
      ext x
      simp only [step, mem_preimage, mem_Ioo, mem_Ici]
      split_ifs with h
      · constructor
        · rintro ⟨h1, -⟩; norm_num at h1
        · intro h2; linarith
      · constructor
        · intro _; linarith
        · intro _; norm_num
    have hno : ¬ IsOpen (Ici a) := fun h => by
      have := h.mem_nhds (self_mem_Ici : a ∈ Ici a)
      rw [mem_nhds_iff_exists_Ioo_subset] at this
      obtain ⟨l, u, ⟨hl, hu⟩, hsub⟩ := this
      have := hsub (show (l + a) / 2 ∈ Ioo l u from ⟨by linarith, by linarith⟩)
      simp only [mem_Ici] at this
      linarith
    refine ⟨hpre, hno, fun hc => hno ?_⟩
    rw [← hpre]
    exact isOpen_Ioo.preimage hc
  exact @key (-2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Continuous.step_not_continuous
-- test: set-step-test.sage

/-! ## exm-continuous-chart -/

/-- The parametrisations `φ ↦ r (cos φ, sin φ)` of a circle and `(θ, φ) ↦ r (sin θ cos φ, sin θ sin φ, cos θ)` of a
sphere are continuous, since their components are (`r = 1`). -/
theorem parametrisations_continuous (r : ℝ) :
    Continuous (fun φ : ℝ => (r * Real.cos φ, r * Real.sin φ)) ∧
      Continuous (fun q : ℝ × ℝ => r • sph q.1 q.2) := by
  have hs : Continuous (fun q : ℝ × ℝ => sph q.1 q.2) := by
    refine continuous_pi fun i => ?_
    fin_cases i <;> simp [sph] <;> fun_prop
  exact ⟨by fun_prop, hs.const_smul r⟩

example : veriphy_instance% KaluzaKlein.Sets.Continuous.parametrisations_continuous (1) := by
  have key : ∀ (r : ℝ), Continuous (fun φ : ℝ => (r * Real.cos φ, r * Real.sin φ)) ∧ Continuous (fun q : ℝ × ℝ => r • sph q.1 q.2) := by
    intro r
    have hs : Continuous (fun q : ℝ × ℝ => sph q.1 q.2) := by
      refine continuous_pi fun i => ?_
      fin_cases i <;> simp [sph] <;> fun_prop
    exact ⟨by fun_prop, hs.const_smul r⟩
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.parametrisations_continuous (0) := by
  have key : ∀ (r : ℝ), Continuous (fun φ : ℝ => (r * Real.cos φ, r * Real.sin φ)) ∧ Continuous (fun q : ℝ × ℝ => r • sph q.1 q.2) := by
    intro r
    have hs : Continuous (fun q : ℝ × ℝ => sph q.1 q.2) := by
      refine continuous_pi fun i => ?_
      fin_cases i <;> simp [sph] <;> fun_prop
    exact ⟨by fun_prop, hs.const_smul r⟩
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.parametrisations_continuous (2) := by
  have key : ∀ (r : ℝ), Continuous (fun φ : ℝ => (r * Real.cos φ, r * Real.sin φ)) ∧ Continuous (fun q : ℝ × ℝ => r • sph q.1 q.2) := by
    intro r
    have hs : Continuous (fun q : ℝ × ℝ => sph q.1 q.2) := by
      refine continuous_pi fun i => ?_
      fin_cases i <;> simp [sph] <;> fun_prop
    exact ⟨by fun_prop, hs.const_smul r⟩
  exact @key (2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Continuous.parametrisations_continuous
-- test: set-parametrisations-test.sage

/-! ## exm-continuous-matrix -/

/-- Matrix multiplication, the determinant and `g ↦ g · v` are continuous (each entry is a polynomial in the
entries); `v = (0, 0, 1)` gives the orbit map. -/
theorem matrix_maps_continuous (n : ℕ) (v : Fin n → ℝ) :
    Continuous (fun p : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ => p.1 * p.2) ∧
      Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g.det) ∧
      Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g *ᵥ v) :=
  ⟨continuous_fst.matrix_mul continuous_snd, continuous_id.matrix_det, continuous_id.matrix_mulVec continuous_const⟩

example : veriphy_instance% KaluzaKlein.Sets.Continuous.matrix_maps_continuous (3) (KaluzaKlein.Sets.Sets.northPole) := by
  have key : ∀ (n : ℕ) (v : Fin n → ℝ), Continuous (fun p : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ => p.1 * p.2) ∧ Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g.det) ∧ Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g *ᵥ v) := by
    intro n v
    exact (⟨continuous_fst.matrix_mul continuous_snd, continuous_id.matrix_det, continuous_id.matrix_mulVec continuous_const⟩)
  exact @key (3) (KaluzaKlein.Sets.Sets.northPole)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.matrix_maps_continuous (0) (![]) := by
  have key : ∀ (n : ℕ) (v : Fin n → ℝ), Continuous (fun p : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ => p.1 * p.2) ∧ Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g.det) ∧ Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g *ᵥ v) := by
    intro n v
    exact (⟨continuous_fst.matrix_mul continuous_snd, continuous_id.matrix_det, continuous_id.matrix_mulVec continuous_const⟩)
  exact @key (0) (![])

example : veriphy_instance% KaluzaKlein.Sets.Continuous.matrix_maps_continuous (2) (![1, 1]) := by
  have key : ∀ (n : ℕ) (v : Fin n → ℝ), Continuous (fun p : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ => p.1 * p.2) ∧ Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g.det) ∧ Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g *ᵥ v) := by
    intro n v
    exact (⟨continuous_fst.matrix_mul continuous_snd, continuous_id.matrix_det, continuous_id.matrix_mulVec continuous_const⟩)
  exact @key (2) (![1, 1])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Continuous.matrix_maps_continuous
-- test: set-matrix-maps-test.sage

/-! ## exr-preimages -/

/-- `cos(π/3) = ½`. -/
theorem cos_pi_third_value : Real.cos (Real.pi / 3) = 1 / 2 := by
  veriphy_certified
    (norm_num [Real.cos_pi_div_three, Real.sin_pi_div_three, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_zero,
        Real.sin_zero])
  -- certificate: set-trig-values.sage

/-- For `0 < δ ≤ π`, the preimage of `(cos δ, 2)` under `cos` is the union of the open intervals
`(-δ + 2πk, δ + 2πk)`: for `δ = π/3`, the preimage of `(½, 2)` is `⋃ (-π/3 + 2πk, π/3 + 2πk)`, open. -/
theorem cos_preimage (δ : ℝ) (hδ : 0 < δ) (hδπ : δ ≤ Real.pi) :
    Real.cos ⁻¹' Ioo (Real.cos δ) 2 = ⋃ k : ℤ, Ioo (-δ + k * (2 * Real.pi)) (δ + k * (2 * Real.pi)) ∧
      IsOpen (Real.cos ⁻¹' Ioo (Real.cos δ) 2) := by
  refine ⟨?_, isOpen_Ioo.preimage Real.continuous_cos⟩
  ext x
  simp only [mem_preimage, mem_Ioo, mem_iUnion]
  constructor
  · rintro ⟨h1, -⟩
    obtain ⟨k, hk⟩ := (Subspace.cos_gt_iff δ x hδ hδπ).1 h1
    rw [abs_lt] at hk
    exact ⟨k, by linarith [hk.1], by linarith [hk.2]⟩
  · rintro ⟨k, h1, h2⟩
    refine ⟨(Subspace.cos_gt_iff δ x hδ hδπ).2 ⟨k, by rw [abs_lt]; constructor <;> linarith⟩, ?_⟩
    linarith [Real.cos_le_one x]

example : veriphy_instance% KaluzaKlein.Sets.Continuous.cos_preimage (Real.pi / 3) ((by positivity)) ((by linarith [Real.pi_pos])) := by
  have key : ∀ (δ : ℝ) (hδ : 0 < δ) (hδπ : δ ≤ Real.pi), Real.cos ⁻¹' Ioo (Real.cos δ) 2 = ⋃ k : ℤ, Ioo (-δ + k * (2 * Real.pi)) (δ + k * (2 * Real.pi)) ∧ IsOpen (Real.cos ⁻¹' Ioo (Real.cos δ) 2) := by
    intro δ hδ hδπ
    refine ⟨?_, isOpen_Ioo.preimage Real.continuous_cos⟩
    ext x
    simp only [mem_preimage, mem_Ioo, mem_iUnion]
    constructor
    · rintro ⟨h1, -⟩
      obtain ⟨k, hk⟩ := (Subspace.cos_gt_iff δ x hδ hδπ).1 h1
      rw [abs_lt] at hk
      exact ⟨k, by linarith [hk.1], by linarith [hk.2]⟩
    · rintro ⟨k, h1, h2⟩
      refine ⟨(Subspace.cos_gt_iff δ x hδ hδπ).2 ⟨k, by rw [abs_lt]; constructor <;> linarith⟩, ?_⟩
      linarith [Real.cos_le_one x]
  exact @key (Real.pi / 3) ((by positivity)) ((by linarith [Real.pi_pos]))

example : veriphy_instance% KaluzaKlein.Sets.Continuous.cos_preimage (Real.pi) ((by positivity)) (le_rfl) := by
  have key : ∀ (δ : ℝ) (hδ : 0 < δ) (hδπ : δ ≤ Real.pi), Real.cos ⁻¹' Ioo (Real.cos δ) 2 = ⋃ k : ℤ, Ioo (-δ + k * (2 * Real.pi)) (δ + k * (2 * Real.pi)) ∧ IsOpen (Real.cos ⁻¹' Ioo (Real.cos δ) 2) := by
    intro δ hδ hδπ
    refine ⟨?_, isOpen_Ioo.preimage Real.continuous_cos⟩
    ext x
    simp only [mem_preimage, mem_Ioo, mem_iUnion]
    constructor
    · rintro ⟨h1, -⟩
      obtain ⟨k, hk⟩ := (Subspace.cos_gt_iff δ x hδ hδπ).1 h1
      rw [abs_lt] at hk
      exact ⟨k, by linarith [hk.1], by linarith [hk.2]⟩
    · rintro ⟨k, h1, h2⟩
      refine ⟨(Subspace.cos_gt_iff δ x hδ hδπ).2 ⟨k, by rw [abs_lt]; constructor <;> linarith⟩, ?_⟩
      linarith [Real.cos_le_one x]
  exact @key (Real.pi) ((by positivity)) (le_rfl)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.cos_preimage (Real.pi / 2) ((by positivity)) ((by linarith [Real.pi_pos])) := by
  have key : ∀ (δ : ℝ) (hδ : 0 < δ) (hδπ : δ ≤ Real.pi), Real.cos ⁻¹' Ioo (Real.cos δ) 2 = ⋃ k : ℤ, Ioo (-δ + k * (2 * Real.pi)) (δ + k * (2 * Real.pi)) ∧ IsOpen (Real.cos ⁻¹' Ioo (Real.cos δ) 2) := by
    intro δ hδ hδπ
    refine ⟨?_, isOpen_Ioo.preimage Real.continuous_cos⟩
    ext x
    simp only [mem_preimage, mem_Ioo, mem_iUnion]
    constructor
    · rintro ⟨h1, -⟩
      obtain ⟨k, hk⟩ := (Subspace.cos_gt_iff δ x hδ hδπ).1 h1
      rw [abs_lt] at hk
      exact ⟨k, by linarith [hk.1], by linarith [hk.2]⟩
    · rintro ⟨k, h1, h2⟩
      refine ⟨(Subspace.cos_gt_iff δ x hδ hδπ).2 ⟨k, by rw [abs_lt]; constructor <;> linarith⟩, ?_⟩
      linarith [Real.cos_le_one x]
  exact @key (Real.pi / 2) ((by positivity)) ((by linarith [Real.pi_pos]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Continuous.cos_preimage
-- test: set-cos-preimage-test.sage

/-! ## exr-continuous-floor -/

/-- The preimage of `(k - ½, k + ½)` under the floor function is `[k, k + 1)`, not open; the floor function is
discontinuous at every integer `k` and continuous on `(k, k + 1)`. -/
theorem floor_discontinuity (k : ℤ) :
    (fun x : ℝ => (⌊x⌋ : ℝ)) ⁻¹' Ioo (k - 1 / 2) (k + 1 / 2) = Ico (k : ℝ) (k + 1) ∧
      ¬ ContinuousAt (fun x : ℝ => (⌊x⌋ : ℝ)) k ∧ ContinuousOn (fun x : ℝ => (⌊x⌋ : ℝ)) (Ioo (k : ℝ) (k + 1)) := by
  have hpre : (fun x : ℝ => (⌊x⌋ : ℝ)) ⁻¹' Ioo (k - 1 / 2) (k + 1 / 2) = Ico (k : ℝ) (k + 1) := by
    ext x
    simp only [mem_preimage, mem_Ioo, mem_Ico]
    constructor
    · rintro ⟨h1, h2⟩
      have e1 : (k : ℝ) - 1 < ⌊x⌋ := by linarith
      have e2 : (⌊x⌋ : ℝ) < k + 1 := by linarith
      have : ⌊x⌋ = k := by
        have a1 : k - 1 < ⌊x⌋ := by exact_mod_cast e1
        have a2 : ⌊x⌋ < k + 1 := by exact_mod_cast e2
        omega
      exact ⟨by rw [← this]; exact Int.floor_le x, by rw [← this]; exact Int.lt_floor_add_one x⟩
    · rintro ⟨h1, h2⟩
      have : ⌊x⌋ = k := by rw [Int.floor_eq_iff]; exact ⟨h1, h2⟩
      rw [this]; constructor <;> linarith
  refine ⟨hpre, fun h => ?_, ?_⟩
  · have hk : (⌊(k : ℝ)⌋ : ℝ) ∈ Ioo ((k : ℝ) - 1 / 2) (k + 1 / 2) := by
      rw [Int.floor_intCast]; constructor <;> linarith
    have := h.preimage_mem_nhds (isOpen_Ioo.mem_nhds hk)
    rw [hpre, Ico_mem_nhds_iff] at this
    exact lt_irrefl _ this.1
  · refine (continuousOn_const (c := (k : ℝ))).congr fun x hx => ?_
    have : ⌊x⌋ = k := by rw [Int.floor_eq_iff]; exact ⟨hx.1.le, hx.2⟩
    simp [this]

example : veriphy_instance% KaluzaKlein.Sets.Continuous.floor_discontinuity (0) := by
  have key : ∀ (k : ℤ), (fun x : ℝ => (⌊x⌋ : ℝ)) ⁻¹' Ioo (k - 1 / 2) (k + 1 / 2) = Ico (k : ℝ) (k + 1) ∧ ¬ ContinuousAt (fun x : ℝ => (⌊x⌋ : ℝ)) k ∧ ContinuousOn (fun x : ℝ => (⌊x⌋ : ℝ)) (Ioo (k : ℝ) (k + 1)) := by
    intro k
    have hpre : (fun x : ℝ => (⌊x⌋ : ℝ)) ⁻¹' Ioo (k - 1 / 2) (k + 1 / 2) = Ico (k : ℝ) (k + 1) := by
      ext x
      simp only [mem_preimage, mem_Ioo, mem_Ico]
      constructor
      · rintro ⟨h1, h2⟩
        have e1 : (k : ℝ) - 1 < ⌊x⌋ := by linarith
        have e2 : (⌊x⌋ : ℝ) < k + 1 := by linarith
        have : ⌊x⌋ = k := by
          have a1 : k - 1 < ⌊x⌋ := by exact_mod_cast e1
          have a2 : ⌊x⌋ < k + 1 := by exact_mod_cast e2
          omega
        exact ⟨by rw [← this]; exact Int.floor_le x, by rw [← this]; exact Int.lt_floor_add_one x⟩
      · rintro ⟨h1, h2⟩
        have : ⌊x⌋ = k := by rw [Int.floor_eq_iff]; exact ⟨h1, h2⟩
        rw [this]; constructor <;> linarith
    refine ⟨hpre, fun h => ?_, ?_⟩
    · have hk : (⌊(k : ℝ)⌋ : ℝ) ∈ Ioo ((k : ℝ) - 1 / 2) (k + 1 / 2) := by
        rw [Int.floor_intCast]; constructor <;> linarith
      have := h.preimage_mem_nhds (isOpen_Ioo.mem_nhds hk)
      rw [hpre, Ico_mem_nhds_iff] at this
      exact lt_irrefl _ this.1
    · refine (continuousOn_const (c := (k : ℝ))).congr fun x hx => ?_
      have : ⌊x⌋ = k := by rw [Int.floor_eq_iff]; exact ⟨hx.1.le, hx.2⟩
      simp [this]
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.floor_discontinuity (1) := by
  have key : ∀ (k : ℤ), (fun x : ℝ => (⌊x⌋ : ℝ)) ⁻¹' Ioo (k - 1 / 2) (k + 1 / 2) = Ico (k : ℝ) (k + 1) ∧ ¬ ContinuousAt (fun x : ℝ => (⌊x⌋ : ℝ)) k ∧ ContinuousOn (fun x : ℝ => (⌊x⌋ : ℝ)) (Ioo (k : ℝ) (k + 1)) := by
    intro k
    have hpre : (fun x : ℝ => (⌊x⌋ : ℝ)) ⁻¹' Ioo (k - 1 / 2) (k + 1 / 2) = Ico (k : ℝ) (k + 1) := by
      ext x
      simp only [mem_preimage, mem_Ioo, mem_Ico]
      constructor
      · rintro ⟨h1, h2⟩
        have e1 : (k : ℝ) - 1 < ⌊x⌋ := by linarith
        have e2 : (⌊x⌋ : ℝ) < k + 1 := by linarith
        have : ⌊x⌋ = k := by
          have a1 : k - 1 < ⌊x⌋ := by exact_mod_cast e1
          have a2 : ⌊x⌋ < k + 1 := by exact_mod_cast e2
          omega
        exact ⟨by rw [← this]; exact Int.floor_le x, by rw [← this]; exact Int.lt_floor_add_one x⟩
      · rintro ⟨h1, h2⟩
        have : ⌊x⌋ = k := by rw [Int.floor_eq_iff]; exact ⟨h1, h2⟩
        rw [this]; constructor <;> linarith
    refine ⟨hpre, fun h => ?_, ?_⟩
    · have hk : (⌊(k : ℝ)⌋ : ℝ) ∈ Ioo ((k : ℝ) - 1 / 2) (k + 1 / 2) := by
        rw [Int.floor_intCast]; constructor <;> linarith
      have := h.preimage_mem_nhds (isOpen_Ioo.mem_nhds hk)
      rw [hpre, Ico_mem_nhds_iff] at this
      exact lt_irrefl _ this.1
    · refine (continuousOn_const (c := (k : ℝ))).congr fun x hx => ?_
      have : ⌊x⌋ = k := by rw [Int.floor_eq_iff]; exact ⟨hx.1.le, hx.2⟩
      simp [this]
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.floor_discontinuity (-2) := by
  have key : ∀ (k : ℤ), (fun x : ℝ => (⌊x⌋ : ℝ)) ⁻¹' Ioo (k - 1 / 2) (k + 1 / 2) = Ico (k : ℝ) (k + 1) ∧ ¬ ContinuousAt (fun x : ℝ => (⌊x⌋ : ℝ)) k ∧ ContinuousOn (fun x : ℝ => (⌊x⌋ : ℝ)) (Ioo (k : ℝ) (k + 1)) := by
    intro k
    have hpre : (fun x : ℝ => (⌊x⌋ : ℝ)) ⁻¹' Ioo (k - 1 / 2) (k + 1 / 2) = Ico (k : ℝ) (k + 1) := by
      ext x
      simp only [mem_preimage, mem_Ioo, mem_Ico]
      constructor
      · rintro ⟨h1, h2⟩
        have e1 : (k : ℝ) - 1 < ⌊x⌋ := by linarith
        have e2 : (⌊x⌋ : ℝ) < k + 1 := by linarith
        have : ⌊x⌋ = k := by
          have a1 : k - 1 < ⌊x⌋ := by exact_mod_cast e1
          have a2 : ⌊x⌋ < k + 1 := by exact_mod_cast e2
          omega
        exact ⟨by rw [← this]; exact Int.floor_le x, by rw [← this]; exact Int.lt_floor_add_one x⟩
      · rintro ⟨h1, h2⟩
        have : ⌊x⌋ = k := by rw [Int.floor_eq_iff]; exact ⟨h1, h2⟩
        rw [this]; constructor <;> linarith
    refine ⟨hpre, fun h => ?_, ?_⟩
    · have hk : (⌊(k : ℝ)⌋ : ℝ) ∈ Ioo ((k : ℝ) - 1 / 2) (k + 1 / 2) := by
        rw [Int.floor_intCast]; constructor <;> linarith
      have := h.preimage_mem_nhds (isOpen_Ioo.mem_nhds hk)
      rw [hpre, Ico_mem_nhds_iff] at this
      exact lt_irrefl _ this.1
    · refine (continuousOn_const (c := (k : ℝ))).congr fun x hx => ?_
      have : ⌊x⌋ = k := by rw [Int.floor_eq_iff]; exact ⟨hx.1.le, hx.2⟩
      simp [this]
  exact @key (-2)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Continuous.floor_discontinuity
-- test: set-floor-test.sage

/-! ## exr-continuous-composition -/

/-- The composition of continuous maps is continuous: `(g ∘ f)⁻¹(W) = f⁻¹(g⁻¹(W))`. -/
theorem comp_continuous (X Y Z : Type) [TopologicalSpace X] [TopologicalSpace Y] [TopologicalSpace Z] (f : X → Y)
    (g : Y → Z) (hf : Continuous f) (hg : Continuous g) : Continuous (g ∘ f) ∧ ∀ W : Set Z, (g ∘ f) ⁻¹' W =
      f ⁻¹' (g ⁻¹' W) :=
  ⟨hg.comp hf, fun _ => rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.Continuous.comp_continuous (ℝ) (ℝ) (ℝ) (inferInstance) (inferInstance) (inferInstance) (Real.sin) (Real.exp) (Real.continuous_sin) (Real.continuous_exp) := by
  have key : ∀ (X Y Z : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] [inst3 : TopologicalSpace Z] (f : X → Y) (g : Y → Z) (hf : Continuous f) (hg : Continuous g), Continuous (g ∘ f) ∧ ∀ W : Set Z, (g ∘ f) ⁻¹' W = f ⁻¹' (g ⁻¹' W) := by
    intro X Y Z inst1 inst2 inst3 f g hf hg
    exact (⟨hg.comp hf, fun _ => rfl⟩)
  exact @key (ℝ) (ℝ) (ℝ) (inferInstance) (inferInstance) (inferInstance) (Real.sin) (Real.exp) (Real.continuous_sin) (Real.continuous_exp)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.comp_continuous (ℝ) (ℝ) (ℝ) (inferInstance) (inferInstance) (inferInstance) (id) (id) (continuous_id) (continuous_id) := by
  have key : ∀ (X Y Z : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] [inst3 : TopologicalSpace Z] (f : X → Y) (g : Y → Z) (hf : Continuous f) (hg : Continuous g), Continuous (g ∘ f) ∧ ∀ W : Set Z, (g ∘ f) ⁻¹' W = f ⁻¹' (g ⁻¹' W) := by
    intro X Y Z inst1 inst2 inst3 f g hf hg
    exact (⟨hg.comp hf, fun _ => rfl⟩)
  exact @key (ℝ) (ℝ) (ℝ) (inferInstance) (inferInstance) (inferInstance) (id) (id) (continuous_id) (continuous_id)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.comp_continuous (ℝ) (ℝ × ℝ) (ℝ) (inferInstance) (inferInstance) (inferInstance) (fun t => (t, t)) (fun p => p.1 * p.2) ((by fun_prop)) ((by fun_prop)) := by
  have key : ∀ (X Y Z : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] [inst3 : TopologicalSpace Z] (f : X → Y) (g : Y → Z) (hf : Continuous f) (hg : Continuous g), Continuous (g ∘ f) ∧ ∀ W : Set Z, (g ∘ f) ⁻¹' W = f ⁻¹' (g ⁻¹' W) := by
    intro X Y Z inst1 inst2 inst3 f g hf hg
    exact (⟨hg.comp hf, fun _ => rfl⟩)
  exact @key (ℝ) (ℝ × ℝ) (ℝ) (inferInstance) (inferInstance) (inferInstance) (fun t => (t, t)) (fun p => p.1 * p.2) ((by fun_prop)) ((by fun_prop))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Continuous.comp_continuous
-- test: set-comp-continuous-test.sage

/-- For a continuous family of rotations `R(t)`, `t ↦ R(t) · (0, 0, 1)` is a continuous curve on the sphere. -/
theorem rotation_curve (R : ℝ → Matrix (Fin 3) (Fin 3) ℝ) (hR : Continuous R) (hSO : ∀ t, R t ∈ SO3) :
    Continuous (fun t => orbit (R t)) ∧ ∀ t, orbit (R t) ∈ sphere := by
  refine ⟨hR.matrix_mulVec continuous_const, fun t => (orbit_image _).2 ⟨R t, hSO t, rfl⟩⟩

example : veriphy_instance% KaluzaKlein.Sets.Continuous.rotation_curve (fun t => KaluzaKlein.Sets.Surjective.Rz t) ((by refine continuous_pi fun i => continuous_pi fun j => ?_; fin_cases i <;> fin_cases j <;> simp [KaluzaKlein.Sets.Surjective.Rz] <;> fun_prop)) (fun t => ((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1) := by
  have key : ∀ (R : ℝ → Matrix (Fin 3) (Fin 3) ℝ) (hR : Continuous R) (hSO : ∀ t, R t ∈ SO3), Continuous (fun t => orbit (R t)) ∧ ∀ t, orbit (R t) ∈ sphere := by
    intro R hR hSO
    refine ⟨hR.matrix_mulVec continuous_const, fun t => (orbit_image _).2 ⟨R t, hSO t, rfl⟩⟩
  exact @key (fun t => KaluzaKlein.Sets.Surjective.Rz t) ((by refine continuous_pi fun i => continuous_pi fun j => ?_; fin_cases i <;> fin_cases j <;> simp [KaluzaKlein.Sets.Surjective.Rz] <;> fun_prop)) (fun t => ((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.rotation_curve (fun _ => 1) (continuous_const) (fun _ => (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem) := by
  have key : ∀ (R : ℝ → Matrix (Fin 3) (Fin 3) ℝ) (hR : Continuous R) (hSO : ∀ t, R t ∈ SO3), Continuous (fun t => orbit (R t)) ∧ ∀ t, orbit (R t) ∈ sphere := by
    intro R hR hSO
    refine ⟨hR.matrix_mulVec continuous_const, fun t => (orbit_image _).2 ⟨R t, hSO t, rfl⟩⟩
  exact @key (fun _ => 1) (continuous_const) (fun _ => (Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem)

example : veriphy_instance% KaluzaKlein.Sets.Continuous.rotation_curve (fun t => KaluzaKlein.Sets.Surjective.Rz (2 * t)) ((by refine continuous_pi fun i => continuous_pi fun j => ?_; fin_cases i <;> fin_cases j <;> simp [KaluzaKlein.Sets.Surjective.Rz] <;> fun_prop)) (fun t => ((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1) := by
  have key : ∀ (R : ℝ → Matrix (Fin 3) (Fin 3) ℝ) (hR : Continuous R) (hSO : ∀ t, R t ∈ SO3), Continuous (fun t => orbit (R t)) ∧ ∀ t, orbit (R t) ∈ sphere := by
    intro R hR hSO
    refine ⟨hR.matrix_mulVec continuous_const, fun t => (orbit_image _).2 ⟨R t, hSO t, rfl⟩⟩
  exact @key (fun t => KaluzaKlein.Sets.Surjective.Rz (2 * t)) ((by refine continuous_pi fun i => continuous_pi fun j => ?_; fin_cases i <;> fin_cases j <;> simp [KaluzaKlein.Sets.Surjective.Rz] <;> fun_prop)) (fun t => ((KaluzaKlein.Sets.Surjective.stabilizer_eq _).2 ⟨_, rfl⟩).1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Continuous.rotation_curve
-- test: set-rotation-curve-test.sage

end KaluzaKlein.Sets.Continuous
