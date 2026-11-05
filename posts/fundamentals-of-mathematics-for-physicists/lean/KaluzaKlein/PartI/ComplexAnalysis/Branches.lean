import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Branches"

The claims of the section "Branches" of `tutorial/part-i-04-complex-analysis.qmd`: branches as continuous selections
of a many-valued function, the branches `Log z + 2πik` of the logarithm on the slit plane, the principal `n`-th root,
the branch points of `log` and `√z` (no branch on the punctured plane), the branch of `log` with the cut `[0, ∞)`,
the jumps of `√z` and `Log z` across the negative axis, and branches of `√(z² - 1)`.
-/

namespace KaluzaKlein.ComplexAnalysis.Branches

open Complex Real Filter Topology Set

/-- A branch of a many-valued function `F` on `D`: a function continuous on `D` with `f z ∈ F z` on `D`. -/
def IsBranch (F : ℂ → Set ℂ) (D : Set ℂ) (f : ℂ → ℂ) : Prop :=
  ContinuousOn f D ∧ ∀ z ∈ D, f z ∈ F z

/-- The many-valued logarithm: `log z = {w : e^w = z}`. -/
def logSet (z : ℂ) : Set ℂ := {w | Complex.exp w = z}

/-- The many-valued `n`-th root: `z^{1/n} = {w : wⁿ = z}`. -/
def rootSet (n : ℕ) (z : ℂ) : Set ℂ := {w | w ^ n = z}

/-! ## def-branch, exm-branch-log -/

/-- On the slit plane `ℂ ∖ (-∞, 0]` every `Log z + 2πik` is a branch of `log` (`k = 0`: the principal branch). -/
theorem log_branches (k : ℤ) : IsBranch logSet slitPlane (fun z => Complex.log z + 2 * π * I * k) := by
  refine ⟨fun z hz => ((continuousAt_clog hz).add continuousAt_const).continuousWithinAt, fun z hz => ?_⟩
  show Complex.exp _ = z
  rw [Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), mul_comm (2 * π * I) _,
    Complex.exp_int_mul_two_pi_mul_I, mul_one]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.log_branches (0) := by
  refine ⟨fun z hz => ((continuousAt_clog hz).add continuousAt_const).continuousWithinAt, fun z hz => ?_⟩
  show Complex.exp _ = z
  rw [Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), mul_comm (2 * π * I) _,
    Complex.exp_int_mul_two_pi_mul_I, mul_one]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.log_branches (1) := by
  refine ⟨fun z hz => ((continuousAt_clog hz).add continuousAt_const).continuousWithinAt, fun z hz => ?_⟩
  show Complex.exp _ = z
  rw [Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), mul_comm (2 * π * I) _,
    Complex.exp_int_mul_two_pi_mul_I, mul_one]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.log_branches ((-2)) := by
  refine ⟨fun z hz => ((continuousAt_clog hz).add continuousAt_const).continuousWithinAt, fun z hz => ?_⟩
  show Complex.exp _ = z
  rw [Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), mul_comm (2 * π * I) _,
    Complex.exp_int_mul_two_pi_mul_I, mul_one]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Branches.log_branches
-- test: ca-num-log-branches-test.sage

/-- The principal `n`-th root `z^{1/n}` is a branch of `z^{1/n}` on the slit plane, with `1^{1/n} = 1`
(`n = 2`: the principal square root). -/
theorem root_branch (n : ℕ) (hn : n ≠ 0) :
    IsBranch (rootSet n) slitPlane (fun z => z ^ (1 / n : ℂ)) ∧ (1 : ℂ) ^ (1 / n : ℂ) = 1 := by
  refine ⟨⟨fun z hz => (continuousAt_cpow_const hz).continuousWithinAt, fun z _ => ?_⟩, Complex.one_cpow _⟩
  show (z ^ (1 / n : ℂ)) ^ n = z
  rw [one_div]; exact Complex.cpow_nat_inv_pow z hn
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.root_branch (1) (one_ne_zero) := by
  refine ⟨⟨fun z hz => (continuousAt_cpow_const hz).continuousWithinAt, fun z _ => ?_⟩, Complex.one_cpow _⟩
  show (z ^ (1 / _ : ℂ)) ^ _ = z
  rw [one_div]
  exact Complex.cpow_nat_inv_pow z (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.root_branch (2) (two_ne_zero) := by
  refine ⟨⟨fun z hz => (continuousAt_cpow_const hz).continuousWithinAt, fun z _ => ?_⟩, Complex.one_cpow _⟩
  show (z ^ (1 / _ : ℂ)) ^ _ = z
  rw [one_div]
  exact Complex.cpow_nat_inv_pow z (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.root_branch (3) ((by norm_num)) := by
  refine ⟨⟨fun z hz => (continuousAt_cpow_const hz).continuousWithinAt, fun z _ => ?_⟩, Complex.one_cpow _⟩
  show (z ^ (1 / _ : ℂ)) ^ _ = z
  rw [one_div]
  exact Complex.cpow_nat_inv_pow z (by norm_num)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Branches.root_branch
-- test: ca-num-root-branch-test.sage

/-- `0` is a branch point of `log`: no continuous `f` on `ℂ ∖ {0}` has `e^{f z} = z` everywhere (following `ln|z| + iθ`
once around the circle would have to increase it by `2πi`). -/
theorem no_log_branch (f : ℂ → ℂ) (hf : ContinuousOn f {0}ᶜ) : ¬ ∀ z : ℂ, z ≠ 0 → Complex.exp (f z) = z := by
  intro h
  -- `g θ = (f(e^{iθ}) - iθ)/(2πi)` is continuous with integer values
  set g : ℝ → ℂ := fun θ => (f (Complex.exp (θ * I)) - θ * I) / (2 * π * I) with hg
  have hπ : (2 * π * I : ℂ) ≠ 0 := by simp [Real.pi_ne_zero, I_ne_zero]
  have hint : ∀ θ : ℝ, ∃ n : ℤ, g θ = n := by
    intro θ
    have h1 := h (Complex.exp (θ * I)) (Complex.exp_ne_zero _)
    obtain ⟨n, hn⟩ := Complex.exp_eq_exp_iff_exists_int.mp h1
    exact ⟨n, by simp only [hg]; rw [hn, add_sub_cancel_left, mul_div_cancel_right₀ _ hπ]⟩
  have hcont : Continuous fun θ : ℝ => (g θ).re := by
    apply Complex.continuous_re.comp
    apply Continuous.div_const
    apply Continuous.sub _ (by fun_prop)
    refine hf.comp_continuous (by fun_prop) fun θ => ?_
    exact Complex.exp_ne_zero _
  have hg0 : g (2 * π) = g 0 - 1 := by
    simp only [hg]
    rw [show ((2 * π : ℝ) : ℂ) * I = 2 * π * I by push_cast; ring, Complex.exp_two_pi_mul_I]
    simp only [Complex.ofReal_zero, zero_mul, Complex.exp_zero, sub_zero]
    rw [sub_div, div_self hπ]
  obtain ⟨n, hn⟩ := hint 0
  have hre0 : (g 0).re = n := by rw [hn]; simp
  have hre1 : (g (2 * π)).re = n - 1 := by rw [hg0, hn]; simp
  obtain ⟨θ, -, hθ⟩ := intermediate_value_Icc' (by positivity : (0 : ℝ) ≤ 2 * π) hcont.continuousOn
    (show (n : ℝ) - 1 / 2 ∈ Icc (g (2 * π)).re (g 0).re by rw [hre0, hre1]; constructor <;> linarith)
  obtain ⟨m, hm⟩ := hint θ
  have hθ' : (g θ).re = n - 1 / 2 := hθ
  rw [hm] at hθ'
  simp at hθ'
  have hθ := hθ'
  have : (m : ℝ) - n = -1 / 2 := by linarith
  have h2 : (2 : ℝ) * (m - n) = -1 := by linarith
  have h3 : (2 * (m - n) : ℤ) = -1 := by exact_mod_cast h2
  omega
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.no_log_branch ((fun _ => 0)) (continuousOn_const) := by
  intro h
  have h1 := h 2 two_ne_zero
  beta_reduce at h1
  norm_num at h1

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.no_log_branch ((fun z => z)) (continuousOn_id) := by
  intro h
  have h1 := h 1 one_ne_zero
  beta_reduce at h1
  have h2 : Real.exp 1 = 1 := by exact_mod_cast (show ((Real.exp 1 : ℝ) : ℂ) = 1 by push_cast; exact h1)
  have := Real.add_one_lt_exp (x := 1) one_ne_zero
  linarith

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.no_log_branch ((fun z => z + 1)) ((by fun_prop)) := by
  intro h
  have h1 := h (-1) (by norm_num)
  beta_reduce at h1
  norm_num at h1

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Branches.no_log_branch
-- test: ca-num-no-log-branch-test.sage

/-- `0` is a branch point of `√z`: no continuous `f` on `ℂ ∖ {0}` has `f(z)² = z` everywhere. -/
theorem no_sqrt_branch (f : ℂ → ℂ) (hf : ContinuousOn f {0}ᶜ) : ¬ ∀ z : ℂ, z ≠ 0 → f z ^ 2 = z := by
  intro h
  set g : ℝ → ℂ := fun θ => f (Complex.exp (θ * I)) * Complex.exp (-(θ / 2 * I)) with hg
  have hsq : ∀ θ : ℝ, g θ = 1 ∨ g θ = -1 := by
    intro θ
    have h1 := h (Complex.exp (θ * I)) (Complex.exp_ne_zero _)
    have : g θ ^ 2 = 1 := by
      simp only [hg]
      rw [mul_pow, h1, ← Complex.exp_nat_mul, ← Complex.exp_add]
      rw [show (θ : ℂ) * I + ((2 : ℕ) : ℂ) * -(θ / 2 * I) = 0 by push_cast; ring, Complex.exp_zero]
    have h2 : (g θ - 1) * (g θ + 1) = 0 := by linear_combination this
    rcases mul_eq_zero.mp h2 with h3 | h3
    · exact Or.inl (by linear_combination h3)
    · exact Or.inr (by linear_combination h3)
  have hcont : Continuous fun θ : ℝ => (g θ).re := by
    apply Complex.continuous_re.comp
    apply Continuous.mul _ (by fun_prop)
    refine hf.comp_continuous (by fun_prop) fun θ => ?_
    exact Complex.exp_ne_zero _
  have hg0 : g (2 * π) = -g 0 := by
    simp only [hg]
    rw [show ((2 * π : ℝ) : ℂ) * I = 2 * π * I by push_cast; ring, Complex.exp_two_pi_mul_I,
      show -(((2 * π : ℝ) : ℂ) / 2 * I) = -(π * I) by push_cast; ring, Complex.exp_neg, Complex.exp_pi_mul_I]
    simp
  have hre : ∀ θ : ℝ, (g θ).re = 1 ∨ (g θ).re = -1 := fun θ => by
    rcases hsq θ with h | h <;> simp [h]
  rcases hsq 0 with h0 | h0
  · obtain ⟨θ, -, hθ⟩ := intermediate_value_Icc' (by positivity : (0 : ℝ) ≤ 2 * π) hcont.continuousOn
      (show (0 : ℝ) ∈ Icc (g (2 * π)).re (g 0).re by rw [hg0, h0]; simp)
    rcases hre θ with h | h <;> linarith
  · obtain ⟨θ, -, hθ⟩ := intermediate_value_Icc (by positivity : (0 : ℝ) ≤ 2 * π) hcont.continuousOn
      (show (0 : ℝ) ∈ Icc (g 0).re (g (2 * π)).re by rw [hg0, h0]; simp)
    rcases hre θ with h | h <;> linarith
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.no_sqrt_branch ((fun _ => 1)) (continuousOn_const) := by
  intro h
  have h1 := h 2 two_ne_zero
  beta_reduce at h1
  norm_num at h1

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.no_sqrt_branch ((fun z => z)) (continuousOn_id) := by
  intro h
  have h1 := h 2 two_ne_zero
  beta_reduce at h1
  norm_num at h1

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.no_sqrt_branch ((fun z => -z)) ((by fun_prop)) := by
  intro h
  have h1 := h 2 two_ne_zero
  beta_reduce at h1
  norm_num at h1

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Branches.no_sqrt_branch
-- test: ca-num-no-sqrt-branch-test.sage

/-- The other cut: `log₊ z = Log(-z) + iπ`, with argument in `(0, 2π)`, is a branch of `log` on `ℂ ∖ [0, ∞)`. -/
noncomputable def logPos (z : ℂ) : ℂ := Complex.log (-z) + π * I

/-- `log₊ + 2πik` is a branch of `log` off the cut `[0, ∞)` (continuous across the negative axis); `k = 0` is
`log₊` itself. -/
theorem logPos_branch (k : ℤ) : IsBranch logSet {z | -z ∈ slitPlane} (fun z => logPos z + 2 * π * I * k) := by
  refine ⟨fun z hz => ?_, fun z hz => ?_⟩
  · exact (((continuousAt_clog hz).comp continuous_neg.continuousAt |>.add continuousAt_const).add
      continuousAt_const).continuousWithinAt
  · show Complex.exp (Complex.log (-z) + π * I + 2 * π * I * k) = z
    rw [Complex.exp_add, Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), Complex.exp_pi_mul_I,
      mul_comm (2 * π * I) _, Complex.exp_int_mul_two_pi_mul_I]; ring
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.logPos_branch (0) := by
  refine ⟨fun z hz => ?_, fun z hz => ?_⟩
  · exact (((continuousAt_clog hz).comp continuous_neg.continuousAt |>.add continuousAt_const).add
      continuousAt_const).continuousWithinAt
  · show Complex.exp (Complex.log (-z) + π * I + 2 * π * I * _) = z
    rw [Complex.exp_add, Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), Complex.exp_pi_mul_I,
      mul_comm (2 * π * I) _, Complex.exp_int_mul_two_pi_mul_I]
    ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.logPos_branch (1) := by
  refine ⟨fun z hz => ?_, fun z hz => ?_⟩
  · exact (((continuousAt_clog hz).comp continuous_neg.continuousAt |>.add continuousAt_const).add
      continuousAt_const).continuousWithinAt
  · show Complex.exp (Complex.log (-z) + π * I + 2 * π * I * _) = z
    rw [Complex.exp_add, Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), Complex.exp_pi_mul_I,
      mul_comm (2 * π * I) _, Complex.exp_int_mul_two_pi_mul_I]
    ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.logPos_branch ((-1)) := by
  refine ⟨fun z hz => ?_, fun z hz => ?_⟩
  · exact (((continuousAt_clog hz).comp continuous_neg.continuousAt |>.add continuousAt_const).add
      continuousAt_const).continuousWithinAt
  · show Complex.exp (Complex.log (-z) + π * I + 2 * π * I * _) = z
    rw [Complex.exp_add, Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), Complex.exp_pi_mul_I,
      mul_comm (2 * π * I) _, Complex.exp_int_mul_two_pi_mul_I]
    ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Branches.logPos_branch
-- test: ca-num-logpos-branch-test.sage

/-- The imaginary part of `log₊` lies in `(0, 2π)` off the cut. -/
theorem logPos_im (z : ℂ) (hz : -z ∈ slitPlane) : 0 < (logPos z).im ∧ (logPos z).im < 2 * π := by
  have h1 := neg_pi_lt_arg (-z)
  have h2 : arg (-z) < π := by
    rcases (mem_slitPlane_iff.mp hz) with h | h
    · exact arg_lt_pi_iff.mpr (Or.inl h.le)
    · exact arg_lt_pi_iff.mpr (Or.inr h)
  simp only [logPos, Complex.add_im, Complex.log_im, Complex.mul_im, Complex.ofReal_re, Complex.I_im,
    Complex.ofReal_im, Complex.I_re]
  constructor <;> nlinarith [Real.pi_pos]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.logPos_im ((-1)) ((by simp [Complex.mem_slitPlane_iff])) := by
  simp [logPos, Real.pi_pos]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.logPos_im (Complex.I) ((by simp [Complex.mem_slitPlane_iff])) := by
  simp [logPos, Complex.log_neg_I]
  constructor <;> linarith [Real.pi_pos]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.logPos_im ((-Complex.I)) ((by simp [Complex.mem_slitPlane_iff])) := by
  simp [logPos, Complex.log_I]
  constructor <;> linarith [Real.pi_pos]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Branches.logPos_im
-- test: ca-num-logpos-im-test.sage

/-! ## exm-branch-sqrt -/

/-- Along `z = e^{iθ}` the continuous choice `√z = e^{iθ/2}` returns to `e^{iπ} = -1` after one turn and to `1`
after two. -/
theorem sqrt_monodromy : Complex.exp ((2 * π : ℝ) / 2 * I) = -1 ∧ Complex.exp ((4 * π : ℝ) / 2 * I) = 1 := by
  constructor
  · rw [show ((2 * π : ℝ) : ℂ) / 2 * I = π * I by push_cast; ring]
    veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage
  · rw [show ((4 * π : ℝ) : ℂ) / 2 * I = 2 * π * I by push_cast; ring]
    veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage

/-! ## exm-branch-sqrt-z2-minus-1 -/

/-- `√(z² - 1) = √(z - 1) √(z + 1)` as many-valued functions: a product of square roots of `z - 1` and `z + 1` is a
square root of `z² - 1`. -/
theorem sqrt_product (z a b : ℂ) (ha : a ^ 2 = z - 1) (hb : b ^ 2 = z + 1) : (a * b) ^ 2 = z ^ 2 - 1 := by
  veriphy_certified (linear_combination (z + 1) * ha + (a ^ 2) * hb)
  -- certificate: ideal_membership.sage

/-- A branch of `√(z² - c)` where `c - z²` avoids the cut: `i (c - z²)^{1/2}`; for `c = 1` a branch of `√(z² - 1)` on
`ℂ ∖ ((-∞, -1] ∪ [1, ∞))`. -/
theorem sqrt_branch_outer_cuts (c : ℂ) :
    IsBranch (fun z => {w | w ^ 2 = z ^ 2 - c}) {z | c - z ^ 2 ∈ slitPlane}
      (fun z => I * (c - z ^ 2) ^ (1 / 2 : ℂ)) := by
  refine ⟨fun z hz => ?_, fun z _ => ?_⟩
  · exact (continuousAt_const.mul ((continuousAt_cpow_const hz).comp (f := fun z : ℂ => c - z ^ 2) (x := z)
      (by fun_prop))).continuousWithinAt
  · show (I * (c - z ^ 2) ^ (1 / 2 : ℂ)) ^ 2 = z ^ 2 - c
    rw [mul_pow, one_div, Complex.cpow_ofNat_inv_pow, I_sq]; ring
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.sqrt_branch_outer_cuts (0) := by
  refine ⟨fun z hz => ?_, fun z _ => ?_⟩
  · exact (continuousAt_const.mul ((continuousAt_cpow_const hz).comp (f := fun w : ℂ => 0 - w ^ 2) (x := z)
      (by fun_prop))).continuousWithinAt
  · show (I * (_ - z ^ 2) ^ (1 / 2 : ℂ)) ^ 2 = z ^ 2 - _
    rw [mul_pow, one_div, Complex.cpow_ofNat_inv_pow, I_sq]
    ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.sqrt_branch_outer_cuts (1) := by
  refine ⟨fun z hz => ?_, fun z _ => ?_⟩
  · exact (continuousAt_const.mul ((continuousAt_cpow_const hz).comp (f := fun w : ℂ => 1 - w ^ 2) (x := z)
      (by fun_prop))).continuousWithinAt
  · show (I * (_ - z ^ 2) ^ (1 / 2 : ℂ)) ^ 2 = z ^ 2 - _
    rw [mul_pow, one_div, Complex.cpow_ofNat_inv_pow, I_sq]
    ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.sqrt_branch_outer_cuts (Complex.I) := by
  refine ⟨fun z hz => ?_, fun z _ => ?_⟩
  · exact (continuousAt_const.mul ((continuousAt_cpow_const hz).comp (f := fun w : ℂ => Complex.I - w ^ 2) (x := z)
      (by fun_prop))).continuousWithinAt
  · show (I * (_ - z ^ 2) ^ (1 / 2 : ℂ)) ^ 2 = z ^ 2 - _
    rw [mul_pow, one_div, Complex.cpow_ofNat_inv_pow, I_sq]
    ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Branches.sqrt_branch_outer_cuts
-- test: ca-num-sqrt-outer-test.sage

/-! ## exr-branch-sqrt-continuity -/

/-- The principal `Log` just above and just below a negative real `x` tends to `ln|x| + iπ` and `ln|x| - iπ`: it
jumps by `2πi`. -/
theorem log_jump (x : ℝ) (hx : x < 0) :
    Tendsto Complex.log (𝓝[{z : ℂ | 0 ≤ z.im}] x) (𝓝 (Real.log |x| + π * I)) ∧
      Tendsto Complex.log (𝓝[{z : ℂ | z.im < 0}] x) (𝓝 (Real.log |x| - π * I)) := by
  constructor
  · have h := continuousWithinAt_log_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)
    have hv : Complex.log x = Real.log |x| + π * I := by
      apply Complex.ext <;> simp [Complex.log_re, Complex.log_im, arg_ofReal_of_neg hx]
    rw [ContinuousWithinAt, hv] at h
    exact h
  · have h := tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)
    simpa using h
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.log_jump ((-4)) ((by norm_num)) := by
  set x : ℝ := (-4) with hxv
  have hx : x < 0 := by rw [hxv]; norm_num
  constructor
  · have h := continuousWithinAt_log_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)
    have hv : Complex.log x = Real.log |x| + π * I := by
      apply Complex.ext <;> simp [Complex.log_re, Complex.log_im, arg_ofReal_of_neg hx]
    rw [ContinuousWithinAt, hv] at h
    exact h
  · have h := tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)
    simpa using h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.log_jump ((-1)) ((by norm_num)) := by
  set x : ℝ := (-1) with hxv
  have hx : x < 0 := by rw [hxv]; norm_num
  constructor
  · have h := continuousWithinAt_log_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)
    have hv : Complex.log x = Real.log |x| + π * I := by
      apply Complex.ext <;> simp [Complex.log_re, Complex.log_im, arg_ofReal_of_neg hx]
    rw [ContinuousWithinAt, hv] at h
    exact h
  · have h := tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)
    simpa using h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.log_jump ((-1 / 2)) ((by norm_num)) := by
  set x : ℝ := (-1 / 2) with hxv
  have hx : x < 0 := by rw [hxv]; norm_num
  constructor
  · have h := continuousWithinAt_log_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)
    have hv : Complex.log x = Real.log |x| + π * I := by
      apply Complex.ext <;> simp [Complex.log_re, Complex.log_im, arg_ofReal_of_neg hx]
    rw [ContinuousWithinAt, hv] at h
    exact h
  · have h := tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)
    simpa using h

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Branches.log_jump
-- test: ca-num-log-jump-test.sage

/-- The principal `√` just above and just below a negative real `x` tends to `±i√|x|`: it jumps by `2i√|x|`
(by `4i` at `x = -4`). -/
theorem sqrt_jump (x : ℝ) (hx : x < 0) :
    Tendsto (fun z : ℂ => z ^ (1 / 2 : ℂ)) (𝓝[{z : ℂ | 0 ≤ z.im}] x) (𝓝 (√|x| * I)) ∧
      Tendsto (fun z : ℂ => z ^ (1 / 2 : ℂ)) (𝓝[{z : ℂ | z.im < 0}] x) (𝓝 (-(√|x| * I))) := by
  have hx0 : (x : ℂ) ≠ 0 := by exact_mod_cast hx.ne
  have hsq : Real.exp (Real.log |x| / 2) = √|x| := by
    rw [Real.sqrt_eq_rpow, Real.rpow_def_of_pos (abs_pos.mpr hx.ne)]; ring_nf
  have hA : Complex.exp ((Real.log |x| + π * I) * (1 / 2)) = √|x| * I := by
    rw [show ((Real.log |x| : ℝ) + π * I : ℂ) * (1 / 2) = ((Real.log |x| / 2 : ℝ) : ℂ) + π / 2 * I by
      push_cast; ring, Complex.exp_add, Complex.exp_pi_div_two_mul_I, ← Complex.ofReal_exp, hsq]
  have hB : Complex.exp ((Real.log |x| - π * I) * (1 / 2)) = -(√|x| * I) := by
    rw [show ((Real.log |x| : ℝ) - π * I : ℂ) * (1 / 2) = ((Real.log |x| / 2 : ℝ) : ℂ) + -(π / 2 * I) by
      push_cast; ring, Complex.exp_add, Complex.exp_neg, Complex.exp_pi_div_two_mul_I, Complex.inv_I,
      ← Complex.ofReal_exp, hsq]
    ring
  have hev : ∀ᶠ z in 𝓝 (x : ℂ), z ^ (1 / 2 : ℂ) = Complex.exp (Complex.log z * (1 / 2)) := by
    filter_upwards [isOpen_compl_singleton.mem_nhds hx0] with z hz
    exact Complex.cpow_def_of_ne_zero hz _
  constructor
  · have h := ((Complex.continuous_exp.tendsto _).comp
      (((log_jump x hx).1).mul_const (1 / 2 : ℂ)))
    rw [hA] at h
    exact h.congr' (eventually_nhdsWithin_of_eventually_nhds (hev.mono fun z hz => hz.symm))
  · have h := ((Complex.continuous_exp.tendsto _).comp
      (((log_jump x hx).2).mul_const (1 / 2 : ℂ)))
    rw [hB] at h
    exact h.congr' (eventually_nhdsWithin_of_eventually_nhds (hev.mono fun z hz => hz.symm))
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.sqrt_jump ((-4)) ((by norm_num)) := by
  set x : ℝ := (-4) with hxv
  have hx : x < 0 := by rw [hxv]; norm_num
  have hx0 : (x : ℂ) ≠ 0 := by exact_mod_cast hx.ne
  have hsq : Real.exp (Real.log |x| / 2) = √|x| := by
    rw [Real.sqrt_eq_rpow, Real.rpow_def_of_pos (abs_pos.mpr hx.ne)]; ring_nf
  have hA : Complex.exp ((Real.log |x| + π * I) * (1 / 2)) = √|x| * I := by
    rw [show ((Real.log |x| : ℝ) + π * I : ℂ) * (1 / 2) = ((Real.log |x| / 2 : ℝ) : ℂ) + π / 2 * I by
      push_cast; ring, Complex.exp_add, Complex.exp_pi_div_two_mul_I, ← Complex.ofReal_exp, hsq]
  have hB : Complex.exp ((Real.log |x| - π * I) * (1 / 2)) = -(√|x| * I) := by
    rw [show ((Real.log |x| : ℝ) - π * I : ℂ) * (1 / 2) = ((Real.log |x| / 2 : ℝ) : ℂ) + -(π / 2 * I) by
      push_cast; ring, Complex.exp_add, Complex.exp_neg, Complex.exp_pi_div_two_mul_I, Complex.inv_I,
      ← Complex.ofReal_exp, hsq]
    ring
  have hev : ∀ᶠ z in 𝓝 (x : ℂ), z ^ (1 / 2 : ℂ) = Complex.exp (Complex.log z * (1 / 2)) := by
    filter_upwards [isOpen_compl_singleton.mem_nhds hx0] with z hz
    exact Complex.cpow_def_of_ne_zero hz _
  constructor
  · have h := ((Complex.continuous_exp.tendsto _).comp
      (((log_jump x hx).1).mul_const (1 / 2 : ℂ)))
    rw [hA] at h
    exact h.congr' (eventually_nhdsWithin_of_eventually_nhds (hev.mono fun z hz => hz.symm))
  · have h := ((Complex.continuous_exp.tendsto _).comp
      (((log_jump x hx).2).mul_const (1 / 2 : ℂ)))
    rw [hB] at h
    exact h.congr' (eventually_nhdsWithin_of_eventually_nhds (hev.mono fun z hz => hz.symm))

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.sqrt_jump ((-1)) ((by norm_num)) := by
  set x : ℝ := (-1) with hxv
  have hx : x < 0 := by rw [hxv]; norm_num
  have hx0 : (x : ℂ) ≠ 0 := by exact_mod_cast hx.ne
  have hsq : Real.exp (Real.log |x| / 2) = √|x| := by
    rw [Real.sqrt_eq_rpow, Real.rpow_def_of_pos (abs_pos.mpr hx.ne)]; ring_nf
  have hA : Complex.exp ((Real.log |x| + π * I) * (1 / 2)) = √|x| * I := by
    rw [show ((Real.log |x| : ℝ) + π * I : ℂ) * (1 / 2) = ((Real.log |x| / 2 : ℝ) : ℂ) + π / 2 * I by
      push_cast; ring, Complex.exp_add, Complex.exp_pi_div_two_mul_I, ← Complex.ofReal_exp, hsq]
  have hB : Complex.exp ((Real.log |x| - π * I) * (1 / 2)) = -(√|x| * I) := by
    rw [show ((Real.log |x| : ℝ) - π * I : ℂ) * (1 / 2) = ((Real.log |x| / 2 : ℝ) : ℂ) + -(π / 2 * I) by
      push_cast; ring, Complex.exp_add, Complex.exp_neg, Complex.exp_pi_div_two_mul_I, Complex.inv_I,
      ← Complex.ofReal_exp, hsq]
    ring
  have hev : ∀ᶠ z in 𝓝 (x : ℂ), z ^ (1 / 2 : ℂ) = Complex.exp (Complex.log z * (1 / 2)) := by
    filter_upwards [isOpen_compl_singleton.mem_nhds hx0] with z hz
    exact Complex.cpow_def_of_ne_zero hz _
  constructor
  · have h := ((Complex.continuous_exp.tendsto _).comp
      (((log_jump x hx).1).mul_const (1 / 2 : ℂ)))
    rw [hA] at h
    exact h.congr' (eventually_nhdsWithin_of_eventually_nhds (hev.mono fun z hz => hz.symm))
  · have h := ((Complex.continuous_exp.tendsto _).comp
      (((log_jump x hx).2).mul_const (1 / 2 : ℂ)))
    rw [hB] at h
    exact h.congr' (eventually_nhdsWithin_of_eventually_nhds (hev.mono fun z hz => hz.symm))

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.sqrt_jump ((-1 / 2)) ((by norm_num)) := by
  set x : ℝ := (-1 / 2) with hxv
  have hx : x < 0 := by rw [hxv]; norm_num
  have hx0 : (x : ℂ) ≠ 0 := by exact_mod_cast hx.ne
  have hsq : Real.exp (Real.log |x| / 2) = √|x| := by
    rw [Real.sqrt_eq_rpow, Real.rpow_def_of_pos (abs_pos.mpr hx.ne)]; ring_nf
  have hA : Complex.exp ((Real.log |x| + π * I) * (1 / 2)) = √|x| * I := by
    rw [show ((Real.log |x| : ℝ) + π * I : ℂ) * (1 / 2) = ((Real.log |x| / 2 : ℝ) : ℂ) + π / 2 * I by
      push_cast; ring, Complex.exp_add, Complex.exp_pi_div_two_mul_I, ← Complex.ofReal_exp, hsq]
  have hB : Complex.exp ((Real.log |x| - π * I) * (1 / 2)) = -(√|x| * I) := by
    rw [show ((Real.log |x| : ℝ) - π * I : ℂ) * (1 / 2) = ((Real.log |x| / 2 : ℝ) : ℂ) + -(π / 2 * I) by
      push_cast; ring, Complex.exp_add, Complex.exp_neg, Complex.exp_pi_div_two_mul_I, Complex.inv_I,
      ← Complex.ofReal_exp, hsq]
    ring
  have hev : ∀ᶠ z in 𝓝 (x : ℂ), z ^ (1 / 2 : ℂ) = Complex.exp (Complex.log z * (1 / 2)) := by
    filter_upwards [isOpen_compl_singleton.mem_nhds hx0] with z hz
    exact Complex.cpow_def_of_ne_zero hz _
  constructor
  · have h := ((Complex.continuous_exp.tendsto _).comp
      (((log_jump x hx).1).mul_const (1 / 2 : ℂ)))
    rw [hA] at h
    exact h.congr' (eventually_nhdsWithin_of_eventually_nhds (hev.mono fun z hz => hz.symm))
  · have h := ((Complex.continuous_exp.tendsto _).comp
      (((log_jump x hx).2).mul_const (1 / 2 : ℂ)))
    rw [hB] at h
    exact h.congr' (eventually_nhdsWithin_of_eventually_nhds (hev.mono fun z hz => hz.symm))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Branches.sqrt_jump
-- test: ca-num-sqrt-jump-test.sage

/-- Across the positive axis both are continuous: `Log` and `√` are continuous at every `x > 0`. -/
theorem continuous_positive_axis (x : ℝ) (hx : 0 < x) :
    ContinuousAt Complex.log x ∧ ContinuousAt (fun z : ℂ => z ^ (1 / 2 : ℂ)) x := by
  have hs : (x : ℂ) ∈ slitPlane := by simpa [mem_slitPlane_iff] using hx
  exact ⟨continuousAt_clog hs, continuousAt_cpow_const hs⟩
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.continuous_positive_axis (4) ((by norm_num)) := by
  set x : ℝ := 4 with hxv
  have hx : 0 < x := by rw [hxv]; norm_num
  have hs : (x : ℂ) ∈ slitPlane := by simpa [mem_slitPlane_iff] using hx
  exact ⟨continuousAt_clog hs, continuousAt_cpow_const hs⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.continuous_positive_axis (1) ((by norm_num)) := by
  set x : ℝ := 1 with hxv
  have hx : 0 < x := by rw [hxv]; norm_num
  have hs : (x : ℂ) ∈ slitPlane := by simpa [mem_slitPlane_iff] using hx
  exact ⟨continuousAt_clog hs, continuousAt_cpow_const hs⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.continuous_positive_axis ((1 / 2)) ((by norm_num)) := by
  set x : ℝ := (1 / 2) with hxv
  have hx : 0 < x := by rw [hxv]; norm_num
  have hs : (x : ℂ) ∈ slitPlane := by simpa [mem_slitPlane_iff] using hx
  exact ⟨continuousAt_clog hs, continuousAt_cpow_const hs⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Branches.continuous_positive_axis
-- test: ca-num-positive-axis-test.sage

/-- On the negative axis the principal root takes the value from above: `(-a²)^{1/2} = ai` for `a > 0`
(at `a = 2`: `(-4)^{1/2} = 2i`). -/
theorem sqrt_neg_sq (a : ℝ) (ha : 0 < a) : (-(a : ℂ) ^ 2) ^ (1 / 2 : ℂ) = a * I := by
  have hlog : Complex.log (-(a : ℂ) ^ 2) = Real.log (a ^ 2) + π * I := by
    apply Complex.ext
    · simp [Complex.log_re]
    · rw [Complex.log_im, show (-(a : ℂ) ^ 2) = ((-a ^ 2 : ℝ) : ℂ) by push_cast; ring,
        arg_ofReal_of_neg (by nlinarith)]
      simp
  have ha0 : (-(a : ℂ) ^ 2) ≠ 0 := by
    rw [show (-(a : ℂ) ^ 2) = ((-a ^ 2 : ℝ) : ℂ) by push_cast; ring]; exact_mod_cast (by nlinarith : -a ^ 2 ≠ 0)
  rw [Complex.cpow_def_of_ne_zero ha0, hlog,
    show ((Real.log (a ^ 2) : ℂ) + π * I) * (1 / 2) = ((Real.log a : ℝ) : ℂ) + π / 2 * I by
      rw [Real.log_pow]; push_cast; ring,
    Complex.exp_add, ← Complex.ofReal_exp, Real.exp_log ha, Complex.exp_pi_div_two_mul_I]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.sqrt_neg_sq (1) (one_pos) := by
  rw [show (-((1 : ℝ) : ℂ) ^ 2) = -1 by push_cast; ring]
  rw [Complex.cpow_def_of_ne_zero (by norm_num), Complex.log_neg_one]
  push_cast
  rw [show (Real.pi : ℂ) * Complex.I * (1 / 2) = Real.pi / 2 * Complex.I by ring, Complex.exp_pi_div_two_mul_I]
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.Branches.sqrt_neg_sq
-- test: ca-num-sqrt-neg-sq-test.sage

/-! ## exr-branch-cut-choice -/

/-- `log₊(-1) = iπ`, `log₊(i) = iπ/2 = Log i`, `log₊(-i) = 3πi/2`, whereas `Log(-i) = -iπ/2`. -/
theorem logPos_values :
    logPos (-1) = π * I ∧ logPos I = π / 2 * I ∧ logPos (-I) = 3 * π / 2 * I ∧
      Complex.log (-I) = -(π / 2) * I := by
  simp only [logPos, neg_neg]
  refine ⟨?_, ?_, ?_, ?_⟩
  · veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage
  · veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage
  · veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage
  · veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage

/-- The two branches agree in the upper half-plane and differ by `2πi` in the lower one. -/
theorem logPos_vs_log (z : ℂ) :
    (0 < z.im → logPos z = Complex.log z) ∧ (z.im < 0 → logPos z = Complex.log z + 2 * π * I) := by
  constructor
  · intro h
    apply Complex.ext <;> simp [logPos, Complex.log_re, Complex.log_im, arg_neg_eq_arg_sub_pi_of_im_pos h]
  · intro h
    apply Complex.ext <;> simp [logPos, Complex.log_re, Complex.log_im, arg_neg_eq_arg_add_pi_of_im_neg h] <;>
      ring
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.logPos_vs_log (1) := by
  constructor <;> intro h <;> simp at h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.logPos_vs_log (Complex.I) := by
  constructor
  · intro _
    simp [logPos, Complex.log_neg_I, Complex.log_I]
    ring
  · intro h
    norm_num at h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.logPos_vs_log ((-Complex.I)) := by
  constructor
  · intro h
    norm_num at h
  · intro _
    simp [logPos, Complex.log_neg_I, Complex.log_I]
    ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Branches.logPos_vs_log
-- test: ca-num-logpos-vs-log-test.sage

/-! ## exr-branch-sqrt-z2-1 -/

/-- `f(z) = z √(1 - 1/z²)` with the principal root. -/
noncomputable def fOut (z : ℂ) : ℂ := z * (1 - 1 / z ^ 2) ^ (1 / 2 : ℂ)

/-- `f(z)² = z² - 1` for `z ≠ 0`. -/
theorem fOut_sq (z : ℂ) (hz : z ≠ 0) : fOut z ^ 2 = z ^ 2 - 1 := by
  rw [fOut, mul_pow, one_div (2 : ℂ), Complex.cpow_ofNat_inv_pow]
  field_simp
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.fOut_sq (1) (one_ne_zero) := by
  simp [fOut]

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.Branches.fOut_sq
-- test: ca-num-fout-sq-test.sage

/-- `1 - 1/z²` lies on the cut `(-∞, 0]` only for `z ∈ [-1, 1]`: off that segment `f` is continuous. -/
theorem fOut_slit (z : ℂ) (hz : ¬ (z.im = 0 ∧ |z.re| ≤ 1)) : 1 - 1 / z ^ 2 ∈ slitPlane := by
  rw [mem_slitPlane_iff]
  by_contra hcon
  push_neg at hcon
  obtain ⟨hre, him⟩ := hcon
  have hz0 : z ≠ 0 := by rintro rfl; exact hz ⟨by simp, by simp⟩
  set u := z ^ 2 with hu
  have hu0 : u ≠ 0 := pow_ne_zero 2 hz0
  have hn : 0 < Complex.normSq u := Complex.normSq_pos.mpr hu0
  have him' : u.im = 0 := by
    have : (1 / u).im = 0 := by simpa using him
    rw [one_div, Complex.inv_im] at this
    rcases div_eq_zero_iff.mp this with h | h
    · linarith
    · exact absurd h hn.ne'
  have hre' : 1 ≤ u.re / Complex.normSq u := by
    have : 1 ≤ (1 / u).re := by rw [Complex.sub_re, Complex.one_re] at hre; linarith
    rwa [one_div, Complex.inv_re] at this
  have hnorm : Complex.normSq u = u.re * u.re := by rw [Complex.normSq_apply, him']; ring
  have hupos : 0 < u.re := by
    rcases lt_trichotomy u.re 0 with h | h | h
    · have : u.re / Complex.normSq u < 0 := div_neg_of_neg_of_pos h hn
      linarith
    · rw [h] at hre'; simp at hre'; linarith
    · exact h
  have hule : u.re ≤ 1 := by
    rw [hnorm, div_mul_cancel_right₀ hupos.ne'] at hre'
    exact (one_le_inv₀ hupos).mp hre'
  have hre2 : u.re = z.re ^ 2 - z.im ^ 2 := by rw [hu]; simp [sq, Complex.mul_re]
  have him2 : u.im = 2 * z.re * z.im := by rw [hu]; simp [sq, Complex.mul_im]; ring
  have hxy : z.re * z.im = 0 := by linarith
  rcases mul_eq_zero.mp hxy with h | h
  · rw [h] at hre2; nlinarith [sq_nonneg z.im]
  · apply hz
    refine ⟨h, ?_⟩
    rw [h] at hre2
    have : z.re ^ 2 ≤ 1 := by nlinarith
    rw [abs_le]; constructor <;> nlinarith [sq_nonneg (z.re + 1), sq_nonneg (z.re - 1)]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.fOut_slit (2) ((by norm_num)) := by
  rw [Complex.mem_slitPlane_iff]
  left
  norm_num

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.Branches.fOut_slit
-- test: ca-num-fout-slit-test.sage

/-- `f` is a branch of `√(z² - 1)` on `ℂ ∖ [-1, 1]`: at every point off the segment it is continuous and squares
to `z² - 1`. -/
theorem fOut_branch_at (z : ℂ) (hz : ¬ (z.im = 0 ∧ |z.re| ≤ 1)) :
    ContinuousAt fOut z ∧ fOut z ^ 2 = z ^ 2 - 1 := by
  have hz0 : z ≠ 0 := fun h => hz (by simp [h])
  refine ⟨?_, fOut_sq z hz0⟩
  exact continuousAt_id.mul ((continuousAt_cpow_const (fOut_slit z hz)).comp
    (f := fun z : ℂ => 1 - 1 / z ^ 2) (x := z) (by fun_prop (disch := exact pow_ne_zero 2 hz0)))
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.fOut_branch_at (2) ((by norm_num)) := by
  have hz0 : (2 : ℂ) ≠ 0 := by norm_num
  refine ⟨?_, fOut_sq _ hz0⟩
  exact continuousAt_id.mul ((continuousAt_cpow_const (fOut_slit _ (by norm_num))).comp
    (f := fun w : ℂ => 1 - 1 / w ^ 2) (x := _) (by fun_prop (disch := exact pow_ne_zero 2 hz0)))

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.fOut_branch_at ((-3)) ((by norm_num)) := by
  have hz0 : (-3 : ℂ) ≠ 0 := by norm_num
  refine ⟨?_, fOut_sq _ hz0⟩
  exact continuousAt_id.mul ((continuousAt_cpow_const (fOut_slit _ (by norm_num))).comp
    (f := fun w : ℂ => 1 - 1 / w ^ 2) (x := _) (by fun_prop (disch := exact pow_ne_zero 2 hz0)))

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.fOut_branch_at ((2 * Complex.I)) ((by simp)) := by
  have hz0 : (2 * Complex.I : ℂ) ≠ 0 := by simp
  refine ⟨?_, fOut_sq _ hz0⟩
  exact continuousAt_id.mul ((continuousAt_cpow_const (fOut_slit _ (by simp))).comp
    (f := fun w : ℂ => 1 - 1 / w ^ 2) (x := _) (by fun_prop (disch := exact pow_ne_zero 2 hz0)))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Branches.fOut_branch_at
-- test: ca-num-fout-branch-test.sage

/-- `f(z) = z + O(1/z)`: `|f(z) - z| ≤ 1/|z|` for every `z ≠ 0`. -/
theorem fOut_asymptotic (z : ℂ) (hz : z ≠ 0) : ‖fOut z - z‖ ≤ 1 / ‖z‖ := by
  set s := (1 - 1 / z ^ 2) ^ (1 / 2 : ℂ) with hs
  have hs2 : s ^ 2 = 1 - 1 / z ^ 2 := by rw [hs, one_div (2 : ℂ), Complex.cpow_ofNat_inv_pow]
  have hsre : 0 ≤ s.re := by
    rw [hs]
    by_cases h0 : (1 - 1 / z ^ 2) = 0
    · rw [h0, Complex.zero_cpow (by norm_num)]; simp
    · rw [Complex.cpow_def_of_ne_zero h0, Complex.exp_re]
      apply mul_nonneg (Real.exp_pos _).le
      apply Real.cos_nonneg_of_mem_Icc
      have h1 := neg_pi_lt_arg (1 - 1 / z ^ 2)
      have h2 := arg_le_pi (1 - 1 / z ^ 2)
      have him : (Complex.log (1 - 1 / z ^ 2) * (1 / 2)).im = arg (1 - 1 / z ^ 2) / 2 := by
        simp [Complex.log_im]; ring
      rw [him]; constructor <;> linarith
  have hz' : 0 < ‖z‖ := norm_pos_iff.mpr hz
  have hprod : (s - 1) * (s + 1) = -(1 / z ^ 2) := by linear_combination hs2
  have hge : 1 ≤ ‖s + 1‖ := by
    have := Complex.re_le_norm (s + 1)
    simp at this; linarith
  have h1 : ‖s - 1‖ * ‖s + 1‖ = 1 / ‖z‖ ^ 2 := by
    rw [← norm_mul, hprod, norm_neg, norm_div, norm_one, norm_pow]
  have h2 : ‖s - 1‖ ≤ 1 / ‖z‖ ^ 2 := by
    rw [← h1]; exact le_mul_of_one_le_right (norm_nonneg _) hge
  have : fOut z - z = z * (s - 1) := by rw [fOut, ← hs]; ring
  rw [this, norm_mul]
  calc ‖z‖ * ‖s - 1‖ ≤ ‖z‖ * (1 / ‖z‖ ^ 2) := mul_le_mul_of_nonneg_left h2 hz'.le
    _ = 1 / ‖z‖ := by field_simp
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Branches.fOut_asymptotic (1) (one_ne_zero) := by
  simp [fOut]

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.Branches.fOut_asymptotic
-- test: ca-num-fout-asym-test.sage

/-- Just above and below the cut at `0`: `f(±iε) = ±i√(ε² + 1)`, which tends to `±i` (a jump of `2i`). -/
theorem fOut_imaginary (ε : ℝ) (hε : 0 < ε) :
    fOut (ε * I) = √(ε ^ 2 + 1) * I ∧ fOut (-(ε * I)) = -(√(ε ^ 2 + 1) * I) := by
  have hε2 : (1 : ℂ) - 1 / (ε * I) ^ 2 = ((1 + 1 / ε ^ 2 : ℝ) : ℂ) := by
    rw [mul_pow, I_sq, mul_neg_one, div_neg, sub_neg_eq_add]; push_cast; ring
  have hroot : ((1 + 1 / ε ^ 2 : ℝ) : ℂ) ^ (1 / 2 : ℂ) = ((√(1 + 1 / ε ^ 2) : ℝ) : ℂ) := by
    rw [Real.sqrt_eq_rpow, Complex.ofReal_cpow (by positivity)]; norm_num
  have hmul : ε * √(1 + 1 / ε ^ 2) = √(ε ^ 2 + 1) := by
    rw [show ε ^ 2 + 1 = ε ^ 2 * (1 + 1 / ε ^ 2) by field_simp, Real.sqrt_mul (sq_nonneg ε),
      Real.sqrt_sq hε.le]
  constructor
  · rw [fOut, hε2, hroot, ← hmul]; push_cast; ring
  · rw [fOut, show (-((ε : ℂ) * I)) ^ 2 = ((ε : ℂ) * I) ^ 2 by ring, hε2, hroot, ← hmul]; push_cast; ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Branches.fOut_imaginary
-- test: ca-num-fout-imag-test.sage

end KaluzaKlein.ComplexAnalysis.Branches
