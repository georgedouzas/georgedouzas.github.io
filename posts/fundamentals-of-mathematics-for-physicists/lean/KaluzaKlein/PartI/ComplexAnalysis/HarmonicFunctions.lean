import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.CauchyRiemann
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Harmonic functions"

The claims of the section "Harmonic functions" of `tutorial/part-i-04-complex-analysis.qmd`. Harmonicity is
mathlib's `InnerProductSpace.HarmonicAt` (twice continuously differentiable with vanishing Laplacian); explicit
Laplacians `u_xx + u_yy` are computed with `deriv` of the one-variable sections. The section shows: the real and
imaginary parts of a holomorphic function are harmonic; on a disc a harmonic function has a harmonic conjugate,
unique up to a constant; `x² - y²`, `eˣ cos y`, `x³ - 3xy²` are harmonic and `x² + y²` is not (it is the real part of
no entire function); `log |z|` is harmonic off `0`, with a conjugate on every disc avoiding `0` but none on the
punctured plane; the mean value property and the absence of strict local maxima; the polar Laplacian and the
harmonicity of `rⁿ cos nθ`.
-/

namespace KaluzaKlein.ComplexAnalysis.HarmonicFunctions

open Complex Filter Topology ComplexConjugate InnerProductSpace Real Metric

/-! ## def-harmonic -/

/-- The real and imaginary parts of a holomorphic function on an open set are harmonic there. -/
theorem re_im_harmonic (f : ℂ → ℂ) (U : Set ℂ) (hU : IsOpen U) (hf : DifferentiableOn ℂ f U) (z : ℂ) (hz : z ∈ U) :
    HarmonicAt (fun w => (f w).re) z ∧ HarmonicAt (fun w => (f w).im) z := by
  have h : AnalyticAt ℂ f z := hf.analyticAt (hU.mem_nhds hz)
  exact ⟨h.harmonicAt_re, h.harmonicAt_im⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.re_im_harmonic ((fun w : ℂ => w)) (Set.univ) (isOpen_univ) ((by fun_prop)) (0) ((Set.mem_univ _)) := by
  have h : AnalyticAt ℂ (fun w : ℂ => w) (0) := DifferentiableOn.analyticAt (s := Set.univ) (by fun_prop) (Filter.univ_mem)
  exact ⟨h.harmonicAt_re, h.harmonicAt_im⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.re_im_harmonic ((fun w : ℂ => w ^ 2)) (Set.univ) (isOpen_univ) ((by fun_prop)) ((1 + Complex.I)) ((Set.mem_univ _)) := by
  have h : AnalyticAt ℂ (fun w : ℂ => w ^ 2) (1 + Complex.I) := DifferentiableOn.analyticAt (s := Set.univ) (by fun_prop) (Filter.univ_mem)
  exact ⟨h.harmonicAt_re, h.harmonicAt_im⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.re_im_harmonic ((fun w : ℂ => w ^ 3 + w)) ((Metric.ball 0 1)) (Metric.isOpen_ball) ((by fun_prop)) (0) ((Metric.mem_ball_self one_pos)) := by
  have h : AnalyticAt ℂ (fun w : ℂ => w ^ 3 + w) (0) := DifferentiableOn.analyticAt (s := Metric.ball 0 1) (by fun_prop) (Metric.isOpen_ball.mem_nhds (Metric.mem_ball_self one_pos))
  exact ⟨h.harmonicAt_re, h.harmonicAt_im⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.re_im_harmonic
-- test: ca-holo-re-im-harmonic-test.sage

/-- On a disc every harmonic function is the real part of a holomorphic function: it has a harmonic conjugate. -/
theorem conjugate_exists (u : ℂ → ℝ) (c : ℂ) (R : ℝ) (hu : HarmonicOnNhd u (ball c R)) :
    ∃ F : ℂ → ℂ, DifferentiableOn ℂ F (ball c R) ∧ ∀ z ∈ ball c R, (F z).re = u z := by
  obtain ⟨F, hF, hre⟩ := HarmonicOnNhd.exists_analyticOnNhd_ball_re_eq hu
  exact ⟨F, hF.differentiableOn, fun z hz => hre hz⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.conjugate_exists ((fun z : ℂ => z.re)) (0) (0) ((fun z hz => absurd hz (by simp))) := by
  exact ⟨fun z => z, by fun_prop, fun _ _ => rfl⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.conjugate_exists ((fun z : ℂ => z.re)) (0) (1) ((fun _ _ => analyticAt_id.harmonicAt_re)) := by
  exact ⟨fun z => z, by fun_prop, fun _ _ => rfl⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.conjugate_exists ((fun z : ℂ => z.im)) (Complex.I) (2) ((fun _ _ => analyticAt_id.harmonicAt_im)) := by
  exact ⟨fun z => -Complex.I * z, by fun_prop, fun _ _ => by simp⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.conjugate_exists
-- test: ca-holo-conj-exists-test.sage

/-- Two harmonic conjugates of `u` on a connected open set differ by a constant. -/
theorem conjugate_unique (u v₁ v₂ : ℂ → ℝ) (U : Set ℂ) (hU : IsOpen U) (hc : IsPreconnected U)
    (h₁ : DifferentiableOn ℂ (fun z => (u z : ℂ) + v₁ z * I) U)
    (h₂ : DifferentiableOn ℂ (fun z => (u z : ℂ) + v₂ z * I) U) :
    ∀ z ∈ U, ∀ w ∈ U, v₁ z - v₂ z = v₁ w - v₂ w := by
  have hd : DifferentiableOn ℂ (fun z => -I * (((u z : ℂ) + v₁ z * I) - ((u z : ℂ) + v₂ z * I))) U :=
    (h₁.sub h₂).const_mul (-I)
  have hr : ∀ z ∈ U, (-I * (((u z : ℂ) + v₁ z * I) - ((u z : ℂ) + v₂ z * I))).im = 0 := fun z _ => by simp
  intro z hz w hw
  have := CauchyRiemann.real_valued_const _ U hU hc hd hr z hz w hw
  have e := congrArg Complex.re this
  simp at e
  linarith

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.conjugate_unique ((fun z : ℂ => z.re)) ((fun z : ℂ => z.im)) ((fun z : ℂ => z.im)) (Set.univ) (isOpen_univ) (isPreconnected_univ) ((by simp only [Complex.re_add_im]; fun_prop)) ((by simp only [Complex.re_add_im]; fun_prop)) (0) ((Set.mem_univ _)) (Complex.I) ((Set.mem_univ _)) := by
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.conjugate_unique ((fun z : ℂ => z.re)) ((fun z : ℂ => z.im)) ((fun z : ℂ => z.im + 1)) (Set.univ) (isOpen_univ) (isPreconnected_univ) ((by simp only [Complex.re_add_im]; fun_prop)) ((by simp only [Complex.ofReal_add, Complex.ofReal_one, add_mul, ← add_assoc, Complex.re_add_im, one_mul]; fun_prop)) (1) ((Set.mem_univ _)) ((2 * Complex.I)) ((Set.mem_univ _)) := by
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.conjugate_unique ((fun z : ℂ => z.re)) ((fun z : ℂ => z.im)) ((fun z : ℂ => z.im - 2)) ((Metric.ball 0 1)) (Metric.isOpen_ball) ((convex_ball (0 : ℂ) 1).isPreconnected) ((by simp only [Complex.re_add_im]; fun_prop)) ((by simp only [Complex.ofReal_sub, Complex.ofReal_ofNat, sub_mul, ← add_sub_assoc, Complex.re_add_im]; fun_prop)) (0) ((Metric.mem_ball_self one_pos)) ((Complex.I / 2)) ((by simp; norm_num)) := by
  ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.conjugate_unique
-- test: ca-holo-conj-unique-test.sage

/-! ## exm-harmonic-polynomial -/

/-- `u_xx = 2` for `u = x² - y²`. -/
theorem sq_uxx (x y : ℝ) : deriv (fun x => deriv (fun x => x ^ 2 - y ^ 2) x) x = 2 := by
  have e : (fun x => deriv (fun x => x ^ 2 - y ^ 2) x) = fun x => 2 * x := funext fun x => CauchyRiemann.sq_ux x y
  rw [e]
  show deriv (fun x => 2 * x) x = 2
  veriphy_certified
    ( have hd : HasDerivAt (fun x => 2 * x) ((2 : ℝ)) (x) :=
        by
        convert ((hasDerivAt_id' (x : ℝ)).const_mul (2 : ℝ)) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `u_yy = -2` for `u = x² - y²`. -/
theorem sq_uyy (x y : ℝ) : deriv (fun y => deriv (fun y => x ^ 2 - y ^ 2) y) y = -2 := by
  have e : (fun y => deriv (fun y => x ^ 2 - y ^ 2) y) = fun y => -2 * y := funext fun y => CauchyRiemann.sq_uy x y
  rw [e]
  show deriv (fun y => -2 * y) y = -2
  veriphy_certified
    ( have hd : HasDerivAt (fun y => -2 * y) ((-2 : ℝ)) (y) :=
        by
        convert ((hasDerivAt_id' (y : ℝ)).const_mul (-2 : ℝ)) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `Δ(x² - y²) = 2 - 2 = 0`. -/
theorem sq_laplacian (x y : ℝ) :
    deriv (fun x => deriv (fun x => x ^ 2 - y ^ 2) x) x + deriv (fun y => deriv (fun y => x ^ 2 - y ^ 2) y) y = 0 := by
  rw [sq_uxx, sq_uyy]
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `u_x = 2x` for `u = x² + y²`. -/
theorem normSq_ux (x y : ℝ) : deriv (fun x => x ^ 2 + y ^ 2) x = 2 * x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x ^ 2 + y ^ 2) ((2 * x : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (y ^ 2 : ℝ))) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `u_y = 2y` for `u = x² + y²`. -/
theorem normSq_uy (x y : ℝ) : deriv (fun y => x ^ 2 + y ^ 2) y = 2 * y := by
  veriphy_certified
    ( have hd : HasDerivAt (fun y => x ^ 2 + y ^ 2) ((2 * y : ℝ)) (y) :=
        by
        convert (((hasDerivAt_id' (y : ℝ)).pow 2).add (hasDerivAt_const (y : ℝ) (x ^ 2 : ℝ))) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply, Pi.pow_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `u_xx = 2` for `u = x² + y²`. -/
theorem normSq_uxx (x y : ℝ) : deriv (fun x => deriv (fun x => x ^ 2 + y ^ 2) x) x = 2 := by
  have e : (fun x => deriv (fun x => x ^ 2 + y ^ 2) x) = fun x => 2 * x := funext fun x => normSq_ux x y
  rw [e]
  show deriv (fun x => 2 * x) x = 2
  veriphy_certified
    ( have hd : HasDerivAt (fun x => 2 * x) ((2 : ℝ)) (x) :=
        by
        convert ((hasDerivAt_id' (x : ℝ)).const_mul (2 : ℝ)) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `u_yy = 2` for `u = x² + y²`. -/
theorem normSq_uyy (x y : ℝ) : deriv (fun y => deriv (fun y => x ^ 2 + y ^ 2) y) y = 2 := by
  have e : (fun y => deriv (fun y => x ^ 2 + y ^ 2) y) = fun y => 2 * y := funext fun y => normSq_uy x y
  rw [e]
  show deriv (fun y => 2 * y) y = 2
  veriphy_certified
    ( have hd : HasDerivAt (fun y => 2 * y) ((2 : ℝ)) (y) :=
        by
        convert ((hasDerivAt_id' (y : ℝ)).const_mul (2 : ℝ)) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `Δ(x² + y²) = 4`. -/
theorem normSq_laplacian (x y : ℝ) :
    deriv (fun x => deriv (fun x => x ^ 2 + y ^ 2) x) x + deriv (fun y => deriv (fun y => x ^ 2 + y ^ 2) y) y = 4 := by
  rw [normSq_uxx, normSq_uyy]
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `x² + y² + c` is the real part of no entire function (its circle average over `|z| = 1` is `1 + c`, its value at
the centre `c`, against the mean value property). -/
theorem normSq_not_re (c : ℝ) :
    ¬ ∃ f : ℂ → ℂ, Differentiable ℂ f ∧ ∀ z, (f z).re = z.re ^ 2 + z.im ^ 2 + c := by
  rintro ⟨f, hf, hre⟩
  have hH : HarmonicOnNhd (fun z => (f z).re) (closedBall 0 |1|) := fun z _ => (hf.analyticAt z).harmonicAt_re
  have h1 := hH.circleAverage_eq
  have h2 : circleAverage (fun z => (f z).re) 0 1 = 1 + c := by
    apply circleAverage_const_on_circle
    intro z hz
    rw [hre]
    have hz' : ‖z‖ = 1 := by simpa using hz
    have : z.re ^ 2 + z.im ^ 2 = 1 := by
      have h1 : Complex.normSq z = 1 := by rw [Complex.normSq_eq_norm_sq, hz']; norm_num
      rw [Complex.normSq_apply] at h1
      nlinarith [h1]
    linarith
  rw [h1, hre] at h2
  simp at h2

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.normSq_not_re (0) := by
  rintro ⟨f, hf, hre⟩
  have hH : HarmonicOnNhd (fun z => (f z).re) (closedBall 0 |1|) := fun z _ => (hf.analyticAt z).harmonicAt_re
  have h1 := hH.circleAverage_eq
  have h2 : circleAverage (fun z => (f z).re) 0 1 = 1 + (0 : ℝ) := by
    apply circleAverage_const_on_circle
    intro z hz
    rw [hre]
    have hz' : ‖z‖ = 1 := by simpa using hz
    have : z.re ^ 2 + z.im ^ 2 = 1 := by
      have h1 : Complex.normSq z = 1 := by rw [Complex.normSq_eq_norm_sq, hz']; norm_num
      rw [Complex.normSq_apply] at h1
      nlinarith [h1]
    linarith
  rw [h1, hre] at h2
  simp at h2

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.normSq_not_re (1) := by
  rintro ⟨f, hf, hre⟩
  have hH : HarmonicOnNhd (fun z => (f z).re) (closedBall 0 |1|) := fun z _ => (hf.analyticAt z).harmonicAt_re
  have h1 := hH.circleAverage_eq
  have h2 : circleAverage (fun z => (f z).re) 0 1 = 1 + (1 : ℝ) := by
    apply circleAverage_const_on_circle
    intro z hz
    rw [hre]
    have hz' : ‖z‖ = 1 := by simpa using hz
    have : z.re ^ 2 + z.im ^ 2 = 1 := by
      have h1 : Complex.normSq z = 1 := by rw [Complex.normSq_eq_norm_sq, hz']; norm_num
      rw [Complex.normSq_apply] at h1
      nlinarith [h1]
    linarith
  rw [h1, hre] at h2
  simp at h2

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.normSq_not_re ((-3 / 2)) := by
  rintro ⟨f, hf, hre⟩
  have hH : HarmonicOnNhd (fun z => (f z).re) (closedBall 0 |1|) := fun z _ => (hf.analyticAt z).harmonicAt_re
  have h1 := hH.circleAverage_eq
  have h2 : circleAverage (fun z => (f z).re) 0 1 = 1 + (-3 / 2 : ℝ) := by
    apply circleAverage_const_on_circle
    intro z hz
    rw [hre]
    have hz' : ‖z‖ = 1 := by simpa using hz
    have : z.re ^ 2 + z.im ^ 2 = 1 := by
      have h1 : Complex.normSq z = 1 := by rw [Complex.normSq_eq_norm_sq, hz']; norm_num
      rw [Complex.normSq_apply] at h1
      nlinarith [h1]
    linarith
  rw [h1, hre] at h2
  simp at h2

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.normSq_not_re
-- test: ca-holo-normsq-not-re-test.sage

/-! ## exm-harmonic-exp -/

/-- `u_xx = eˣ cos y` for `u = eˣ cos y`. -/
theorem exp_uxx (x y : ℝ) :
    deriv (fun x => deriv (fun x => Real.exp x * Real.cos y) x) x = Real.exp x * Real.cos y := by
  have e : (fun x => deriv (fun x => Real.exp x * Real.cos y) x) = fun x => Real.exp x * Real.cos y :=
    funext fun x => CauchyRiemann.exp_ux x y
  rw [e]
  show deriv (fun x => Real.exp x * Real.cos y) x = Real.exp x * Real.cos y
  veriphy_certified
    ( have hd : HasDerivAt (fun x => rexp x * Real.cos y) (((Real.cos y) * (Real.exp x) : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).exp).const_mul (Real.cos y : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ca-holo-deriv.sage

/-- `u_yy = -eˣ cos y` for `u = eˣ cos y`. -/
theorem exp_uyy (x y : ℝ) :
    deriv (fun y => deriv (fun y => Real.exp x * Real.cos y) y) y = -(Real.exp x * Real.cos y) := by
  have e : (fun y => deriv (fun y => Real.exp x * Real.cos y) y) = fun y => -(Real.exp x * Real.sin y) :=
    funext fun y => CauchyRiemann.exp_uy x y
  rw [e]
  show deriv (fun y => -(Real.exp x * Real.sin y)) y = -(Real.exp x * Real.cos y)
  veriphy_certified
    ( have hd : HasDerivAt (fun y => -(rexp x * Real.sin y)) ((-(Real.cos y) * (Real.exp x) : ℝ)) (y) :=
        by
        convert (((hasDerivAt_id' (y : ℝ)).sin).const_mul (-(Real.exp x) : ℝ)) using 1
        all_goals ((ring_nf; done))
      exact hd.deriv.trans (by ring1))
  -- certificate: ca-holo-deriv.sage

/-- `Δ(eˣ cos y) = 0`. -/
theorem exp_laplacian (x y : ℝ) :
    deriv (fun x => deriv (fun x => Real.exp x * Real.cos y) x) x +
      deriv (fun y => deriv (fun y => Real.exp x * Real.cos y) y) y = 0 := by
  rw [exp_uxx, exp_uyy]
  generalize Real.exp x * Real.cos y = a
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- The real part of an entire function is harmonic on the whole plane. -/
theorem entire_re_harmonic (f : ℂ → ℂ) (hf : Differentiable ℂ f) (z : ℂ) : HarmonicAt (fun w => (f w).re) z :=
  (hf.analyticAt z).harmonicAt_re

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.entire_re_harmonic ((fun w : ℂ => w)) ((by fun_prop)) (0) := by
  exact ((by fun_prop : Differentiable ℂ (fun w : ℂ => w)).analyticAt _).harmonicAt_re

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.entire_re_harmonic (Complex.exp) (Complex.differentiable_exp) (1) := by
  exact (Complex.differentiable_exp.analyticAt _).harmonicAt_re

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.entire_re_harmonic ((fun w : ℂ => w ^ 2)) ((by fun_prop)) (Complex.I) := by
  exact ((by fun_prop : Differentiable ℂ (fun w : ℂ => w ^ 2)).analyticAt _).harmonicAt_re

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.entire_re_harmonic
-- test: ca-holo-entire-re-test.sage

/-! ## exm-harmonic-log -/

/-- `ln |z| = ln √(x² + y²)` is the real part of every branch `Log z + 2πik` of `log z`. -/
theorem log_re_eq (k : ℤ) (x y : ℝ) :
    (Complex.log ((x : ℂ) + y * I) + 2 * π * I * k).re = Real.log (Real.sqrt (x ^ 2 + y ^ 2)) := by
  simp [Complex.log_re, Complex.norm_def, Complex.normSq_apply, sq]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_re_eq (0) (0) (0) := by
  simp [Complex.log_re, Complex.norm_def, Complex.normSq_apply, sq]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_re_eq (-1) (-5) ((-5 / 2)) := by
  simp [Complex.log_re, Complex.norm_def, Complex.normSq_apply, sq]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_re_eq (0) (0) (1) := by
  simp [Complex.log_re, Complex.norm_def, Complex.normSq_apply, sq]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_re_eq (0) (0) (-5) := by
  simp [Complex.log_re, Complex.norm_def, Complex.normSq_apply, sq]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_re_eq
-- test: ca-holo-log-re-test.sage

/-- `ln |z|` is harmonic on `ℂ ∖ {0}`. -/
theorem log_norm_harmonic (z : ℂ) (hz : z ≠ 0) : HarmonicAt (fun w : ℂ => Real.log ‖w‖) z :=
  analyticAt_id.harmonicAt_log_norm hz

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_norm_harmonic (1) ((by simp)) := by
  exact analyticAt_id.harmonicAt_log_norm (by simp)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_norm_harmonic (Complex.I) ((by simp)) := by
  exact analyticAt_id.harmonicAt_log_norm (by simp)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_norm_harmonic ((-2)) ((by norm_num)) := by
  exact analyticAt_id.harmonicAt_log_norm (by norm_num)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_norm_harmonic
-- test: ca-holo-log-harmonic-test.sage

/-- On every disc `|z - z₀| < |z₀|` (not containing `0`) `ln |z|` has a conjugate: it is the real part of the
holomorphic `Log(z/z₀) + Log z₀`. -/
theorem log_conjugate_disc (z₀ : ℂ) :
    ∃ F : ℂ → ℂ, DifferentiableOn ℂ F (ball z₀ ‖z₀‖) ∧ ∀ z ∈ ball z₀ ‖z₀‖, (F z).re = Real.log ‖z‖ := by
  by_cases h0 : z₀ = 0
  · exact ⟨0, differentiableOn_const 0, by simp [h0]⟩
  have hn : 0 < ‖z₀‖ := norm_pos_iff.mpr h0
  have hslit : ∀ z ∈ ball z₀ ‖z₀‖, z / z₀ ∈ slitPlane := by
    intro z hz
    have e : z / z₀ = 1 + (z - z₀) / z₀ := by field_simp; ring
    rw [e]
    apply mem_slitPlane_of_norm_lt_one
    rw [norm_div, div_lt_one hn, ← dist_eq_norm]
    exact hz
  refine ⟨fun z => Complex.log (z / z₀) + Complex.log z₀, ?_, ?_⟩
  · intro z hz
    exact (((differentiableAt_id.div_const z₀).clog (hslit z hz)).add_const _).differentiableWithinAt
  · intro z hz
    have hz0 : z ≠ 0 := by
      rintro rfl
      simp at hz
    simp only [add_re, log_re, norm_div]
    rw [Real.log_div (norm_ne_zero_iff.mpr hz0) (norm_ne_zero_iff.mpr h0)]
    ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_conjugate_disc (0) := by
  by_cases h0 : (0 : ℂ) = 0
  · exact ⟨0, differentiableOn_const 0, by simp [h0]⟩
  have hn : 0 < ‖(0 : ℂ)‖ := norm_pos_iff.mpr h0
  have hslit : ∀ z ∈ Metric.ball (0 : ℂ) ‖(0 : ℂ)‖, z / (0) ∈ Complex.slitPlane := by
    intro z hz
    have e : z / (0) = 1 + (z - (0)) / (0) := by field_simp; ring
    rw [e]
    apply Complex.mem_slitPlane_of_norm_lt_one
    rw [norm_div, div_lt_one hn, ← dist_eq_norm]
    exact hz
  refine ⟨fun z => Complex.log (z / (0)) + Complex.log (0), ?_, ?_⟩
  · intro z hz
    exact (((differentiableAt_id.div_const (0)).clog (hslit z hz)).add_const _).differentiableWithinAt
  · intro z hz
    have hz0 : z ≠ 0 := by
      rintro rfl
      simp at hz
    simp only [Complex.add_re, Complex.log_re, norm_div]
    rw [Real.log_div (norm_ne_zero_iff.mpr hz0) (norm_ne_zero_iff.mpr h0)]
    ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_conjugate_disc (1) := by
  by_cases h0 : (1 : ℂ) = 0
  · exact ⟨0, differentiableOn_const 0, by simp [h0]⟩
  have hn : 0 < ‖(1 : ℂ)‖ := norm_pos_iff.mpr h0
  have hslit : ∀ z ∈ Metric.ball (1 : ℂ) ‖(1 : ℂ)‖, z / (1) ∈ Complex.slitPlane := by
    intro z hz
    have e : z / (1) = 1 + (z - (1)) / (1) := by field_simp; ring
    rw [e]
    apply Complex.mem_slitPlane_of_norm_lt_one
    rw [norm_div, div_lt_one hn, ← dist_eq_norm]
    exact hz
  refine ⟨fun z => Complex.log (z / (1)) + Complex.log (1), ?_, ?_⟩
  · intro z hz
    exact (((differentiableAt_id.div_const (1)).clog (hslit z hz)).add_const _).differentiableWithinAt
  · intro z hz
    have hz0 : z ≠ 0 := by
      rintro rfl
      simp at hz
    simp only [Complex.add_re, Complex.log_re, norm_div]
    rw [Real.log_div (norm_ne_zero_iff.mpr hz0) (norm_ne_zero_iff.mpr h0)]
    ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_conjugate_disc (Complex.I) := by
  by_cases h0 : (Complex.I : ℂ) = 0
  · exact ⟨0, differentiableOn_const 0, by simp [h0]⟩
  have hn : 0 < ‖(Complex.I : ℂ)‖ := norm_pos_iff.mpr h0
  have hslit : ∀ z ∈ Metric.ball (Complex.I : ℂ) ‖(Complex.I : ℂ)‖, z / (Complex.I) ∈ Complex.slitPlane := by
    intro z hz
    have e : z / (Complex.I) = 1 + (z - (Complex.I)) / (Complex.I) := by field_simp; ring
    rw [e]
    apply Complex.mem_slitPlane_of_norm_lt_one
    rw [norm_div, div_lt_one hn, ← dist_eq_norm]
    exact hz
  refine ⟨fun z => Complex.log (z / (Complex.I)) + Complex.log (Complex.I), ?_, ?_⟩
  · intro z hz
    exact (((differentiableAt_id.div_const (Complex.I)).clog (hslit z hz)).add_const _).differentiableWithinAt
  · intro z hz
    have hz0 : z ≠ 0 := by
      rintro rfl
      simp at hz
    simp only [Complex.add_re, Complex.log_re, norm_div]
    rw [Real.log_div (norm_ne_zero_iff.mpr hz0) (norm_ne_zero_iff.mpr h0)]
    ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_conjugate_disc
-- test: ca-holo-log-conj-disc-test.sage

/-- Auxiliary form of `no_global_log_conjugate` (its proof); the claim below restates it. -/
theorem no_global_log_conjugate_aux (c : ℂ) :
    ¬ ∃ F : ℂ → ℂ, DifferentiableOn ℂ F {c}ᶜ ∧ ∀ z ≠ c, (F z).re = Real.log ‖z - c‖ := by
  rintro ⟨F, hF, hre⟩
  -- G w = F (w + c) - Log w is holomorphic on the slit plane with zero real part, hence constant
  set U := slitPlane with hU
  have hUo : IsOpen U := isOpen_slitPlane
  have hUc : IsPreconnected U := (StarConvex.isPathConnected Complex.starConvex_one_slitPlane one_mem_slitPlane).isConnected.isPreconnected
  have hmem : ∀ w ∈ U, w + c ≠ c := fun w hw h => slitPlane_ne_zero hw (by simpa using h)
  have hG : DifferentiableOn ℂ (fun w => -I * (F (w + c) - Complex.log w)) U := by
    intro w hw
    have h1 : DifferentiableAt ℂ (fun w => F (w + c)) w :=
      ((hF.differentiableAt (isOpen_compl_singleton.mem_nhds (hmem w hw))).comp w
        (differentiableAt_id.add_const c))
    exact ((h1.sub (differentiableAt_log hw)).const_mul (-I)).differentiableWithinAt
  have hGr : ∀ w ∈ U, (-I * (F (w + c) - Complex.log w)).im = 0 := by
    intro w hw
    have := hre (w + c) (hmem w hw)
    simp only [add_sub_cancel_right] at this
    simp [log_re, this]
  have hconst := CauchyRiemann.real_valued_const _ U hUo hUc hG hGr
  -- so F (w + c) = Log w + K on the slit plane
  set K := F (1 + c) - Complex.log 1 with hK
  have hFU : ∀ w ∈ U, F (w + c) = Complex.log w + K := by
    intro w hw
    have := hconst w hw 1 (one_mem_slitPlane)
    have h' : F (w + c) - Complex.log w = F (1 + c) - Complex.log 1 := by
      have := congrArg (fun t => I * t) this
      simpa [← mul_assoc] using this
    rw [hK]
    linear_combination h'
  -- continuity of w ↦ F (w + c) at -1
  have hcont : ContinuousAt (fun w => F (w + c)) (-1) := by
    have : (-1 : ℂ) + c ≠ c := by simp
    exact ContinuousAt.comp (g := F) (f := fun w : ℂ => w + c)
      (hF.differentiableAt (isOpen_compl_singleton.mem_nhds this)).continuousAt
      (continuous_id.add continuous_const : Continuous fun w : ℂ => w + c).continuousAt
  -- limits along the lower and upper half-planes
  have hlow : (-1 : ℂ) ∈ closure {z : ℂ | z.im < 0} := by rw [closure_setOfPred_im_lt]; simp
  have hup : (-1 : ℂ) ∈ closure {z : ℂ | 0 < z.im} := by rw [closure_setOfPred_lt_im]; simp
  haveI : (𝓝[{z : ℂ | z.im < 0}] (-1 : ℂ)).NeBot := mem_closure_iff_nhdsWithin_neBot.mp hlow
  haveI : (𝓝[{z : ℂ | 0 < z.im}] (-1 : ℂ)).NeBot := mem_closure_iff_nhdsWithin_neBot.mp hup
  have hl1 : Tendsto (fun w => Complex.log w + K) (𝓝[{z : ℂ | z.im < 0}] (-1 : ℂ))
      (𝓝 (Real.log ‖(-1 : ℂ)‖ - π * I + K)) :=
    (tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by norm_num)).add_const K
  have hl2 : Tendsto (fun w => Complex.log w + K) (𝓝[{z : ℂ | 0 < z.im}] (-1 : ℂ))
      (𝓝 (Complex.log (-1) + K)) := by
    have hc := (continuousWithinAt_log_of_re_neg_of_im_zero (z := (-1 : ℂ)) (by norm_num) (by norm_num))
    exact ((hc.mono (fun z (hz : 0 < z.im) => le_of_lt hz)).tendsto).add_const K
  have hev1 : (fun w => F (w + c)) =ᶠ[𝓝[{z : ℂ | z.im < 0}] (-1 : ℂ)] fun w => Complex.log w + K :=
    eventually_nhdsWithin_of_forall fun w (hw : w.im < 0) => hFU w (Or.inr hw.ne)
  have hev2 : (fun w => F (w + c)) =ᶠ[𝓝[{z : ℂ | 0 < z.im}] (-1 : ℂ)] fun w => Complex.log w + K :=
    eventually_nhdsWithin_of_forall fun w (hw : 0 < w.im) => hFU w (Or.inr hw.ne')
  have e1 := tendsto_nhds_unique ((hcont.tendsto.mono_left nhdsWithin_le_nhds).congr' hev1) hl1
  have e2 := tendsto_nhds_unique ((hcont.tendsto.mono_left nhdsWithin_le_nhds).congr' hev2) hl2
  rw [e1, Complex.log_neg_one] at e2
  have := congrArg Complex.im e2
  simp at this
  linarith [Real.pi_pos]

/-- `ln |z - c|` has no harmonic conjugate on the whole punctured plane `ℂ ∖ {c}`: a holomorphic `F` there with
`Re F = ln |z - c|` would differ from `Log(z - c)` by a constant on the slit plane, and so jump by `2πi` across the
cut, against its continuity. -/
theorem no_global_log_conjugate (c : ℂ) :
    ¬ ∃ F : ℂ → ℂ, DifferentiableOn ℂ F {c}ᶜ ∧ ∀ z ≠ c, (F z).re = Real.log ‖z - c‖ :=
  no_global_log_conjugate_aux c

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_global_log_conjugate (0) := by
  exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_global_log_conjugate_aux 0

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_global_log_conjugate (1) := by
  exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_global_log_conjugate_aux 1

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_global_log_conjugate (Complex.I) := by
  exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_global_log_conjugate_aux Complex.I

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_global_log_conjugate
-- test: ca-holo-no-global-log-test.sage

/-! ## exr-harmonic-conjugate -/

/-- `u_x = 3x² - 3y²` for `u = x³ - 3xy²`. -/
theorem cube_ux (x y : ℝ) : deriv (fun x => x ^ 3 - 3 * x * y ^ 2) x = 3 * x ^ 2 - 3 * y ^ 2 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x ^ 3 - 3 * x * y ^ 2) ((3 * (x ^ 2) - (3 * (y ^ 2)) : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).pow 3).add ((hasDerivAt_id' (x : ℝ)).const_mul (-3 * (y ^ 2) : ℝ))) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply, Pi.pow_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `u_y = -6xy` for `u = x³ - 3xy²`. -/
theorem cube_uy (x y : ℝ) : deriv (fun y => x ^ 3 - 3 * x * y ^ 2) y = -6 * x * y := by
  have h := (hasDerivAt_const y (x ^ 3)).sub ((hasDerivAt_pow 2 y).const_mul (3 * x))
  refine h.deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_uy (0) (0) := by
  refine ((hasDerivAt_const _ _).sub ((hasDerivAt_pow 2 _).const_mul _)).deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_uy (1) (-5) := by
  refine ((hasDerivAt_const _ _).sub ((hasDerivAt_pow 2 _).const_mul _)).deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_uy (0) (1) := by
  refine ((hasDerivAt_const _ _).sub ((hasDerivAt_pow 2 _).const_mul _)).deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_uy (0) (0) := by
  refine ((hasDerivAt_const _ _).sub ((hasDerivAt_pow 2 _).const_mul _)).deriv.trans ?_
  norm_num
  try ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_uy
-- test: ca-holo-cube-uy-test.sage

/-- `v_x = 6xy` for `v = 3x²y - y³`. -/
theorem cube_vx (x y : ℝ) : deriv (fun x => 3 * x ^ 2 * y - y ^ 3) x = 6 * x * y := by
  have h := (((hasDerivAt_pow 2 x).const_mul 3).mul_const y).sub_const (y ^ 3)
  refine h.deriv.trans ?_
  simp only [Nat.cast_ofNat, show (2 : ℕ) - 1 = 1 from rfl, pow_one]
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_vx (0) (0) := by
  refine ((((hasDerivAt_pow 2 _).const_mul 3).mul_const _).sub_const _).deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_vx (1) (-5) := by
  refine ((((hasDerivAt_pow 2 _).const_mul 3).mul_const _).sub_const _).deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_vx (0) (1) := by
  refine ((((hasDerivAt_pow 2 _).const_mul 3).mul_const _).sub_const _).deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_vx (0) (0) := by
  refine ((((hasDerivAt_pow 2 _).const_mul 3).mul_const _).sub_const _).deriv.trans ?_
  norm_num
  try ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_vx
-- test: ca-holo-cube-vx-test.sage

/-- `v_y = 3x² - 3y²` for `v = 3x²y - y³`. -/
theorem cube_vy (x y : ℝ) : deriv (fun y => 3 * x ^ 2 * y - y ^ 3) y = 3 * x ^ 2 - 3 * y ^ 2 := by
  have h := ((hasDerivAt_id' y).const_mul (3 * x ^ 2)).sub (hasDerivAt_pow 3 y)
  refine h.deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_vy (0) (0) := by
  refine (((hasDerivAt_id' _).const_mul _).sub (hasDerivAt_pow 3 _)).deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_vy (1) (-5) := by
  refine (((hasDerivAt_id' _).const_mul _).sub (hasDerivAt_pow 3 _)).deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_vy (0) (1) := by
  refine (((hasDerivAt_id' _).const_mul _).sub (hasDerivAt_pow 3 _)).deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_vy (0) (0) := by
  refine (((hasDerivAt_id' _).const_mul _).sub (hasDerivAt_pow 3 _)).deriv.trans ?_
  norm_num
  try ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_vy
-- test: ca-holo-cube-vy-test.sage

/-- `u_xx = 6x`. -/
theorem cube_uxx (x y : ℝ) : deriv (fun x => deriv (fun x => x ^ 3 - 3 * x * y ^ 2) x) x = 6 * x := by
  have e : (fun x => deriv (fun x => x ^ 3 - 3 * x * y ^ 2) x) = fun x => 3 * x ^ 2 - 3 * y ^ 2 :=
    funext fun x => cube_ux x y
  rw [e]
  show deriv (fun x => 3 * x ^ 2 - 3 * y ^ 2) x = 6 * x
  have h := ((hasDerivAt_pow 2 x).const_mul 3).sub_const (3 * y ^ 2)
  refine h.deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_uxx (0) (0) := by
  rw [show (fun x => deriv (fun x => x ^ 3 - 3 * x * (0 : ℝ) ^ 2) x) = fun x => 3 * x ^ 2 - 3 * (0 : ℝ) ^ 2 from
    funext fun x => KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_ux x (0)]
  refine (((hasDerivAt_pow 2 (0 : ℝ)).const_mul 3).sub_const (3 * (0 : ℝ) ^ 2)).deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_uxx (1) ((-2)) := by
  rw [show (fun x => deriv (fun x => x ^ 3 - 3 * x * (-2 : ℝ) ^ 2) x) = fun x => 3 * x ^ 2 - 3 * (-2 : ℝ) ^ 2 from
    funext fun x => KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_ux x (-2)]
  refine (((hasDerivAt_pow 2 (1 : ℝ)).const_mul 3).sub_const (3 * (-2 : ℝ) ^ 2)).deriv.trans ?_
  norm_num
  try ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_uxx ((-1 / 2)) (3) := by
  rw [show (fun x => deriv (fun x => x ^ 3 - 3 * x * (3 : ℝ) ^ 2) x) = fun x => 3 * x ^ 2 - 3 * (3 : ℝ) ^ 2 from
    funext fun x => KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_ux x (3)]
  refine (((hasDerivAt_pow 2 (-1 / 2 : ℝ)).const_mul 3).sub_const (3 * (3 : ℝ) ^ 2)).deriv.trans ?_
  norm_num
  try ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_uxx
-- test: ca-holo-cube-uxx-test.sage

/-- `u_yy = -6x`. -/
theorem cube_uyy (x y : ℝ) : deriv (fun y => deriv (fun y => x ^ 3 - 3 * x * y ^ 2) y) y = -6 * x := by
  have e : (fun y => deriv (fun y => x ^ 3 - 3 * x * y ^ 2) y) = fun y => -6 * x * y := funext fun y => cube_uy x y
  rw [e]
  show deriv (fun y => -6 * x * y) y = -6 * x
  veriphy_certified
    ( have hd : HasDerivAt (fun y => -6 * x * y) ((-6 * x : ℝ)) (y) :=
        by
        convert ((hasDerivAt_id' (y : ℝ)).const_mul (-6 * x : ℝ)) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ca-holo-deriv.sage

/-- `Δ(x³ - 3xy²) = 6x - 6x = 0`. -/
theorem cube_laplacian (x y : ℝ) :
    deriv (fun x => deriv (fun x => x ^ 3 - 3 * x * y ^ 2) x) x +
      deriv (fun y => deriv (fun y => x ^ 3 - 3 * x * y ^ 2) y) y = 0 := by
  rw [cube_uxx, cube_uyy]
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- The normalisation `v(0, 0) = 3·0²·0 - 0³ = 0`. -/
theorem cube_v_origin : (3 : ℝ) * 0 ^ 2 * 0 - 0 ^ 3 = 0 := by
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-holo-ring.sage

/-- `u + iv = x³ - 3xy² + i(3x²y - y³) = (x + iy)³ = z³`. -/
theorem cube_parts (x y : ℝ) :
    ((x : ℂ) + y * I) ^ 3 = ((x ^ 3 - 3 * x * y ^ 2 : ℝ) : ℂ) + ((3 * x ^ 2 * y - y ^ 3 : ℝ) : ℂ) * I := by
  have hI : I ^ 2 = -1 := I_sq
  have e1 : ((x ^ 3 - 3 * x * y ^ 2 : ℝ) : ℂ) = (x : ℂ) ^ 3 - 3 * (x : ℂ) * (y : ℂ) ^ 2 := by push_cast <;> ring
  have e2 : ((3 * x ^ 2 * y - y ^ 3 : ℝ) : ℂ) = 3 * (x : ℂ) ^ 2 * (y : ℂ) - (y : ℂ) ^ 3 := by push_cast <;> ring
  rw [e1, e2]
  clear e1 e2
  generalize ((x : ℝ) : ℂ) = xC at *
  generalize ((y : ℝ) : ℂ) = yC at *
  generalize Complex.I = i at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (3 * xC * yC ^ 2 + yC ^ 3 * i) * hI)
  -- certificate: ca-holo-ring.sage

/-! ## exr-harmonic-mean-value -/

/-- Auxiliary form of `circle_param_deriv` (its proof); the claim below restates it. -/
theorem circle_param_deriv_aux (z₀ : ℂ) (r θ : ℝ) :
    HasDerivAt (fun θ : ℝ => z₀ + r * Complex.exp (θ * I)) (I * r * Complex.exp (θ * I)) θ := by
  have h := hasDerivAt_circleMap z₀ r θ
  have e : circleMap z₀ r = fun t : ℝ => z₀ + r * Complex.exp (t * I) := by
    funext t
    simp [circleMap]
  rw [e] at h
  convert h using 1
  simp [circleMap]
  try ring

/-- `z = z₀ + r e^{iθ}` has `dz/dθ = i r e^{iθ}`. -/
theorem circle_param_deriv (z₀ : ℂ) (r θ : ℝ) :
    HasDerivAt (fun θ : ℝ => z₀ + r * Complex.exp (θ * I)) (I * r * Complex.exp (θ * I)) θ :=
  circle_param_deriv_aux z₀ r θ

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.circle_param_deriv (0) (0) (0) := by
  exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.circle_param_deriv_aux 0 0 0

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.circle_param_deriv (1) (2) (1) := by
  exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.circle_param_deriv_aux 1 2 1

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.circle_param_deriv (Complex.I) ((1 / 2)) ((-1)) := by
  exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.circle_param_deriv_aux Complex.I (1 / 2) (-1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.circle_param_deriv
-- test: ca-holo-circle-param-test.sage

/-- Cauchy's integral formula on a circle as an average: `f(z₀) = (1/2π) ∫₀^{2π} f(z₀ + r e^{iθ}) dθ`. -/
theorem holomorphic_mean_value (f : ℂ → ℂ) (z₀ : ℂ) (r : ℝ) (hf : DiffContOnCl ℂ f (ball z₀ |r|)) :
    circleAverage f z₀ r = f z₀ :=
  hf.circleAverage

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.holomorphic_mean_value ((fun w : ℂ => w)) (0) (1) ((by fun_prop : Differentiable ℂ (fun w : ℂ => w)).diffContOnCl) := by
  exact (by fun_prop : Differentiable ℂ (fun w : ℂ => w)).diffContOnCl.circleAverage

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.holomorphic_mean_value (Complex.exp) (1) (2) (Complex.differentiable_exp.diffContOnCl) := by
  exact Complex.differentiable_exp.diffContOnCl.circleAverage

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.holomorphic_mean_value ((fun w : ℂ => w ^ 2)) (Complex.I) ((1 / 2)) ((by fun_prop : Differentiable ℂ (fun w : ℂ => w ^ 2)).diffContOnCl) := by
  exact (by fun_prop : Differentiable ℂ (fun w : ℂ => w ^ 2)).diffContOnCl.circleAverage

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.holomorphic_mean_value
-- test: ca-holo-mean-value-holo-test.sage

/-- The mean value property of harmonic functions (the real part of the previous one). -/
theorem harmonic_mean_value (u : ℂ → ℝ) (z₀ : ℂ) (r : ℝ) (hu : HarmonicOnNhd u (closedBall z₀ |r|)) :
    circleAverage u z₀ r = u z₀ :=
  hu.circleAverage_eq

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.harmonic_mean_value ((fun z : ℂ => z.re)) (0) (0) ((fun _ _ => analyticAt_id.harmonicAt_re)) := by
  exact InnerProductSpace.HarmonicOnNhd.circleAverage_eq (fun _ _ => analyticAt_id.harmonicAt_re)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.harmonic_mean_value ((fun z : ℂ => z.im)) (1) (2) ((fun _ _ => analyticAt_id.harmonicAt_im)) := by
  exact InnerProductSpace.HarmonicOnNhd.circleAverage_eq (fun _ _ => analyticAt_id.harmonicAt_im)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.harmonic_mean_value ((fun z : ℂ => (z ^ 2).re)) (Complex.I) (1) ((fun _ _ => (analyticAt_id.pow 2).harmonicAt_re)) := by
  exact InnerProductSpace.HarmonicOnNhd.circleAverage_eq (fun _ _ => (analyticAt_id.pow 2).harmonicAt_re)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.harmonic_mean_value
-- test: ca-holo-mean-value-test.sage

/-- Auxiliary form of `no_strict_local_max` (its proof); the claim below restates it. -/
theorem no_strict_local_max_aux (u : ℂ → ℝ) (U : Set ℂ) (hU : IsOpen U) (hu : HarmonicOnNhd u U) (z₀ : ℂ)
    (hz : z₀ ∈ U) : ¬ ∃ ε > 0, ∀ z, z ≠ z₀ → dist z z₀ < ε → u z < u z₀ := by
  rintro ⟨ε, hε, hmax⟩
  obtain ⟨δ, hδ, hball⟩ := Metric.isOpen_iff.mp hU z₀ hz
  set r := min (ε / 2) (δ / 2) with hr
  have hr0 : 0 < r := lt_min (half_pos hε) (half_pos hδ)
  have hsub : closedBall z₀ |r| ⊆ U := by
    rw [abs_of_pos hr0]
    exact (closedBall_subset_ball (by rw [hr]; linarith [min_le_right (ε / 2) (δ / 2)])).trans hball
  have hmv := harmonic_mean_value u z₀ r (HarmonicOnNhd.mono hu hsub)
  have hcont : ContinuousOn u (sphere z₀ |r|) :=
    (HarmonicOnNhd.contDiffOn (HarmonicOnNhd.mono hu hsub)).continuousOn.mono sphere_subset_closedBall
  have hne : (sphere z₀ |r|).Nonempty := NormedSpace.sphere_nonempty.mpr (abs_nonneg r)
  obtain ⟨w, hw, hwmax⟩ := (isCompact_sphere z₀ |r|).exists_isMaxOn hne hcont
  have hlt : u w < u z₀ := by
    have hwd : dist w z₀ = r := by rw [dist_eq_norm]; simpa [abs_of_pos hr0] using hw
    refine hmax w ?_ ?_
    · intro h
      rw [h, dist_self] at hwd
      linarith
    · rw [hwd, hr]
      linarith [min_le_left (ε / 2) (δ / 2)]
  have hle : circleAverage u z₀ r ≤ u w :=
    circleAverage_mono_on_of_le_circle (hcont.circleIntegrable') fun x hx => hwmax hx
  linarith

/-- A harmonic function on an open set has no strict local maximum. -/
theorem no_strict_local_max (u : ℂ → ℝ) (U : Set ℂ) (hU : IsOpen U) (hu : HarmonicOnNhd u U) (z₀ : ℂ)
    (hz : z₀ ∈ U) : ¬ ∃ ε > 0, ∀ z, z ≠ z₀ → dist z z₀ < ε → u z < u z₀ :=
  no_strict_local_max_aux u U hU hu z₀ hz

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_strict_local_max ((fun z : ℂ => z.re)) (Set.univ) (isOpen_univ) ((fun _ _ => analyticAt_id.harmonicAt_re)) (0) ((Set.mem_univ _)) := by
  exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_strict_local_max_aux (fun z : ℂ => z.re) Set.univ isOpen_univ (fun _ _ => analyticAt_id.harmonicAt_re) 0 (Set.mem_univ _)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_strict_local_max ((fun z : ℂ => z.im)) ((Metric.ball 0 1)) (Metric.isOpen_ball) ((fun _ _ => analyticAt_id.harmonicAt_im)) ((1 / 2)) ((by simp; norm_num)) := by
  exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_strict_local_max_aux (fun z : ℂ => z.im) (Metric.ball 0 1) Metric.isOpen_ball (fun _ _ => analyticAt_id.harmonicAt_im) (1 / 2) (by simp; norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_strict_local_max ((fun z : ℂ => (z ^ 2).re)) (Set.univ) (isOpen_univ) ((fun _ _ => (analyticAt_id.pow 2).harmonicAt_re)) (Complex.I) ((Set.mem_univ _)) := by
  exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_strict_local_max_aux (fun z : ℂ => (z ^ 2).re) Set.univ isOpen_univ (fun _ _ => (analyticAt_id.pow 2).harmonicAt_re) Complex.I (Set.mem_univ _)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_strict_local_max
-- test: ca-holo-no-max-test.sage

/-! ## exr-harmonic-polar

With `c = cos θ`, `s = sin θ`, the chain rule gives `u_r = c p + s q`, `u_θ = -r s p + r c q` for the gradient
`(p, q) = (u_x, u_y)`, and, for the Hessian `(a, b; b, d) = (u_xx, u_xy; u_xy, u_yy)`,
`u_rr = c² a + 2cs b + s² d`, `u_θθ = r²(s² a - 2sc b + c² d) - r(c p + s q)`. -/

/-- Inverting the first-order chain rule: `r u_x = r cos θ u_r - sin θ u_θ` and `r u_y = r sin θ u_r + cos θ u_θ`. -/
theorem polar_first (c s r p q : ℝ) (hcs : c ^ 2 + s ^ 2 = 1) :
    r * p = r * c * (c * p + s * q) - s * (-r * s * p + r * c * q) ∧
      r * q = r * s * (c * p + s * q) + c * (-r * s * p + r * c * q) := by
  refine ⟨?_, ?_⟩
  · rw [← sub_eq_zero]
    veriphy_certified (linear_combination (-r * p) * hcs)
    -- certificate: ca-holo-ring.sage
  · rw [← sub_eq_zero]
    veriphy_certified (linear_combination (-r * q) * hcs)
    -- certificate: ca-holo-ring.sage

/-- The cross terms cancel: `r²(u_rr + u_r/r + u_θθ/r²) = r²(u_xx + u_yy)`. -/
theorem polar_laplacian (c s r p q a b d : ℝ) (hcs : c ^ 2 + s ^ 2 = 1) :
    r ^ 2 * (c ^ 2 * a + 2 * c * s * b + s ^ 2 * d) + r * (c * p + s * q) +
        (r ^ 2 * (s ^ 2 * a - 2 * s * c * b + c ^ 2 * d) - r * (c * p + s * q)) = r ^ 2 * (a + d) := by
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (r ^ 2 * a + r ^ 2 * d) * hcs)
  -- certificate: ca-holo-ring.sage

/-- Auxiliary form of `rn_cos_derivs` (its proof); the claim below restates it. -/
theorem rn_cos_derivs_aux (n : ℤ) (r θ : ℝ) (hr : r ≠ 0) :
    deriv (fun r => r ^ n * Real.cos (n * θ)) r = n * r ^ (n - 1) * Real.cos (n * θ) ∧
      deriv (fun r => deriv (fun r => r ^ n * Real.cos (n * θ)) r) r =
        n * (n - 1) * r ^ (n - 2) * Real.cos (n * θ) ∧
      deriv (fun θ => deriv (fun θ => r ^ n * Real.cos (n * θ)) θ) θ = -(n ^ 2 * r ^ n * Real.cos (n * θ)) := by
  have h1 : ∀ s : ℝ, s ≠ 0 →
      deriv (fun q => q ^ n * Real.cos (n * θ)) s = n * s ^ (n - 1) * Real.cos (n * θ) := fun s hs =>
    ((hasDerivAt_zpow n s (Or.inl hs)).mul_const _).deriv
  refine ⟨h1 r hr, ?_, ?_⟩
  · have e : deriv (fun s => deriv (fun q => q ^ n * Real.cos (n * θ)) s) r =
        deriv (fun s => n * s ^ (n - 1) * Real.cos (n * θ)) r :=
      Filter.EventuallyEq.deriv_eq ((eventually_ne_nhds hr).mono h1)
    rw [e]
    have h2 := (((hasDerivAt_zpow (n - 1) r (Or.inl hr)).const_mul (n : ℝ)).mul_const (Real.cos (n * θ))).deriv
    rw [h2]
    have : n - 1 - 1 = n - 2 := by ring
    rw [this]
    push_cast
    ring
  · have e : (fun t => deriv (fun q => r ^ n * Real.cos (n * q)) t) = fun t => -(r ^ n * (n * Real.sin (n * t))) := by
      funext t
      refine ((((hasDerivAt_id' t).const_mul (n : ℝ)).cos).const_mul (r ^ n)).deriv.trans ?_
      ring
    rw [e]
    refine (((((hasDerivAt_id' θ).const_mul (n : ℝ)).sin).const_mul (n : ℝ)).const_mul (r ^ n)).neg.deriv.trans ?_
    ring

/-- The derivatives of `u = rⁿ cos nθ` (`n ∈ ℤ`, `r ≠ 0`): `u_r = n rⁿ⁻¹ cos nθ`, `u_rr = n(n - 1) rⁿ⁻² cos nθ`,
`u_θθ = -n² rⁿ cos nθ`. -/
theorem rn_cos_derivs (n : ℤ) (r θ : ℝ) (hr : r ≠ 0) :
    deriv (fun r => r ^ n * Real.cos (n * θ)) r = n * r ^ (n - 1) * Real.cos (n * θ) ∧
      deriv (fun r => deriv (fun r => r ^ n * Real.cos (n * θ)) r) r =
        n * (n - 1) * r ^ (n - 2) * Real.cos (n * θ) ∧
      deriv (fun θ => deriv (fun θ => r ^ n * Real.cos (n * θ)) θ) θ = -(n ^ 2 * r ^ n * Real.cos (n * θ)) :=
  rn_cos_derivs_aux n r θ hr

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_derivs (0) (1) (0) ((by norm_num)) := by
  exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_derivs_aux 0 1 0 (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_derivs (2) (2) (1) ((by norm_num)) := by
  exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_derivs_aux 2 2 1 (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_derivs ((-1)) ((1 / 2)) ((-1)) ((by norm_num)) := by
  exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_derivs_aux (-1) (1 / 2) (-1) (by norm_num)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_derivs
-- test: ca-holo-rn-derivs-test.sage

/-- With these derivatives, `u_rr + u_r/r + u_θθ/r² = 0`: `rⁿ cos nθ` is harmonic for every integer `n`. -/
theorem rn_cos_laplacian (n : ℤ) (r θ : ℝ) (hr : r ≠ 0) :
    n * (n - 1) * r ^ (n - 2) * Real.cos (n * θ) + (1 / r) * (n * r ^ (n - 1) * Real.cos (n * θ)) +
      (1 / r ^ 2) * (-(n ^ 2 * r ^ n * Real.cos (n * θ))) = 0 := by
  have hB : r ^ (n - 1) = r ^ (n - 2) * r := by
    rw [← zpow_add_one₀ hr]; ring_nf
  have hC : r ^ n = r ^ (n - 2) * r ^ 2 := by
    rw [← zpow_natCast, ← zpow_add₀ hr]; push_cast; ring_nf
  have hinv : r * r⁻¹ = 1 := mul_inv_cancel₀ hr
  rw [one_div, one_div, ← inv_pow, hB, hC]
  generalize r ^ (n - 2) = A at *
  generalize r⁻¹ = ri at *
  generalize Real.cos (n * θ) = C at *
  generalize (n : ℝ) = m at *
  generalize ((r : ℝ) : ℂ) = rC at *
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-m ^ 2 * A * C * ri * r - m ^ 2 * A * C + m * A * C) * hinv)
  -- certificate: ca-holo-ring.sage

/-- Indeed `rⁿ cos nθ = Re zⁿ` for `z = r e^{iθ}`. -/
theorem rn_cos_eq_re (n : ℕ) (r θ : ℝ) :
    (((r : ℂ) * Complex.exp (θ * I)) ^ n).re = r ^ n * Real.cos (n * θ) := by
  rw [mul_pow, ← Complex.exp_nat_mul, ← Complex.ofReal_pow, Complex.re_ofReal_mul]
  congr 1
  rw [← Complex.exp_ofReal_mul_I_re]
  congr 1
  congr 1
  push_cast <;> ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_eq_re (0) (0) (0) := by
  rw [mul_pow, ← Complex.exp_nat_mul, ← Complex.ofReal_pow, Complex.re_ofReal_mul]
  congr 1
  rw [← Complex.exp_ofReal_mul_I_re]
  congr 1
  congr 1
  push_cast <;> ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_eq_re (4) (-5) ((-5 / 2)) := by
  rw [mul_pow, ← Complex.exp_nat_mul, ← Complex.ofReal_pow, Complex.re_ofReal_mul]
  congr 1
  rw [← Complex.exp_ofReal_mul_I_re]
  congr 1
  congr 1
  push_cast <;> ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_eq_re (0) (0) (1) := by
  rw [mul_pow, ← Complex.exp_nat_mul, ← Complex.ofReal_pow, Complex.re_ofReal_mul]
  congr 1
  rw [← Complex.exp_ofReal_mul_I_re]
  congr 1
  congr 1
  push_cast <;> ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_eq_re (0) (0) (-5) := by
  rw [mul_pow, ← Complex.exp_nat_mul, ← Complex.ofReal_pow, Complex.re_ofReal_mul]
  congr 1
  rw [← Complex.exp_ofReal_mul_I_re]
  congr 1
  congr 1
  push_cast <;> ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_eq_re
-- test: ca-holo-rn-cos-re-test.sage

end KaluzaKlein.ComplexAnalysis.HarmonicFunctions
