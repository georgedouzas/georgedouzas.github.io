import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Modulus"

The claims of the section "Modulus" of `tutorial/part-i-04-complex-analysis.qmd`: the modulus `|x + iy| = √(x² + y²)`,
multiplicativity, the triangle inequality and its reverse, distances and the topology of `ℂ = ℝ²`, explicit moduli,
the unit circle as a group, an Apollonius circle and the parallelogram law.
-/

namespace KaluzaKlein.ComplexAnalysis.Modulus

open Complex ComplexConjugate Metric

/-! ## def-modulus -/

/-- `|x + iy| = √(x² + y²)`. -/
theorem modulus_formula (x y : ℝ) : ‖(x : ℂ) + y * I‖ = √(x ^ 2 + y ^ 2) := norm_add_mul_I x y
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.modulus_formula (0) (0) := by
  rw [Complex.norm_def, Complex.normSq_apply]
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.modulus_formula (3) (4) := by
  rw [Complex.norm_def, Complex.normSq_apply]
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.modulus_formula (1) ((-2)) := by
  rw [Complex.norm_def, Complex.normSq_apply]
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Modulus.modulus_formula
-- test: ca-num-modulus-test.sage

/-- `|z|² = z z̄`. -/
theorem sq_norm_eq_mul_conj (z : ℂ) : ((‖z‖ ^ 2 : ℝ) : ℂ) = z * conj z := by
  rw [Complex.mul_conj']; push_cast; ring
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.sq_norm_eq_mul_conj (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.sq_norm_eq_mul_conj (1) := by
  simp

attribute [veriphy_tested oracle_test 2 1] KaluzaKlein.ComplexAnalysis.Modulus.sq_norm_eq_mul_conj
-- test: ca-num-sq-norm-conj-test.sage

/-- The modulus is multiplicative: `|zw| = |z||w|`. -/
theorem norm_mul_eq (z w : ℂ) : ‖z * w‖ = ‖z‖ * ‖w‖ := Complex.norm_mul z w
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.norm_mul_eq (0) ((2 + Complex.I)) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.Modulus.norm_mul_eq
-- test: ca-num-norm-mul-test.sage

/-- The squared form of multiplicativity is Lagrange's identity `(xu - yv)² + (xv + yu)² = (x² + y²)(u² + v²)`. -/
theorem lagrange_identity (x y u v : ℝ) :
    (x * u - y * v) ^ 2 + (x * v + y * u) ^ 2 = (x ^ 2 + y ^ 2) * (u ^ 2 + v ^ 2) := by
  veriphy_certified (ring1)
  -- certificate: ca-num-ring.sage

/-- The triangle inequality `|z + w| ≤ |z| + |w|`. -/
theorem triangle (z w : ℂ) : ‖z + w‖ ≤ ‖z‖ + ‖w‖ := norm_add_le z w
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.triangle (0) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.Modulus.triangle
-- test: ca-num-triangle-test.sage

/-- `|z - w|` is the distance between `z` and `w`. -/
theorem dist_eq_norm_sub (z w : ℂ) : dist z w = ‖z - w‖ := Complex.dist_eq z w
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.dist_eq_norm_sub (0) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.dist_eq_norm_sub (1) (Complex.I) := by
  exact dist_eq_norm _ _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.dist_eq_norm_sub ((2 + Complex.I)) (3) := by
  exact dist_eq_norm _ _

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Modulus.dist_eq_norm_sub
-- test: ca-num-dist-test.sage

/-- The open disc `D(z₀, r) = {z : |z - z₀| < r}` is open. -/
theorem disc_isOpen (z₀ : ℂ) (r : ℝ) : IsOpen {z : ℂ | ‖z - z₀‖ < r} := by
  have : {z : ℂ | ‖z - z₀‖ < r} = ball z₀ r := by ext z; simp [dist_eq_norm]
  rw [this]; exact isOpen_ball
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.disc_isOpen (0) (0) := by
  exact isOpen_lt (by fun_prop) continuous_const

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.disc_isOpen (0) (1) := by
  exact isOpen_lt (by fun_prop) continuous_const

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.disc_isOpen (Complex.I) (2) := by
  exact isOpen_lt (by fun_prop) continuous_const

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Modulus.disc_isOpen
-- test: ca-num-disc-open-test.sage

/-- A set is open, closed, compact or connected in `ℂ` exactly when its image in `ℝ²` is. -/
theorem topology_as_R2 (s : Set ℂ) :
    (IsOpen s ↔ IsOpen (equivRealProdCLM '' s)) ∧ (IsClosed s ↔ IsClosed (equivRealProdCLM '' s)) ∧
      (IsCompact s ↔ IsCompact (equivRealProdCLM '' s)) ∧
      (IsConnected s ↔ IsConnected (equivRealProdCLM '' s)) := by
  let e := equivRealProdCLM.toHomeomorph
  have he : ∀ t : Set ℂ, e '' t = equivRealProdCLM '' t := fun t => rfl
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [← he, e.isOpen_image]
  · rw [← he, e.isClosed_image]
  · rw [← he, e.isCompact_image]
  · rw [← he, e.isConnected_image]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.topology_as_R2 (∅) := by
  let e := Complex.equivRealProdCLM.toHomeomorph
  have he : ∀ t : Set ℂ, e '' t = Complex.equivRealProdCLM '' t := fun t => rfl
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [← he, e.isOpen_image]
  · rw [← he, e.isClosed_image]
  · rw [← he, e.isCompact_image]
  · rw [← he, e.isConnected_image]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.topology_as_R2 (Set.univ) := by
  let e := Complex.equivRealProdCLM.toHomeomorph
  have he : ∀ t : Set ℂ, e '' t = Complex.equivRealProdCLM '' t := fun t => rfl
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [← he, e.isOpen_image]
  · rw [← he, e.isClosed_image]
  · rw [← he, e.isCompact_image]
  · rw [← he, e.isConnected_image]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.topology_as_R2 (Metric.ball 0 1) := by
  let e := Complex.equivRealProdCLM.toHomeomorph
  have he : ∀ t : Set ℂ, e '' t = Complex.equivRealProdCLM '' t := fun t => rfl
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [← he, e.isOpen_image]
  · rw [← he, e.isClosed_image]
  · rw [← he, e.isCompact_image]
  · rw [← he, e.isConnected_image]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Modulus.topology_as_R2
-- test: ca-num-topology-test.sage

/-! ## exm-modulus-values -/

/-- `|3 + 4i| = 5`, `|i| = 1`, `|2 + i| = √5`. -/
theorem moduli_values : ‖(3 : ℂ) + 4 * I‖ = 5 ∧ ‖I‖ = 1 ∧ ‖(2 : ℂ) + I‖ = √5 := by
  refine ⟨?_, ?_, ?_⟩
  · rw [show (3 : ℂ) + 4 * I = ((3 : ℝ) : ℂ) + ((4 : ℝ) : ℂ) * I by push_cast; ring, norm_add_mul_I]
    veriphy_certified
      ( conv_rhs => rw [show (5 : ℝ) = √((5) ^ 2) from (Real.sqrt_sq (by norm_num)).symm]
        norm_num)
    -- certificate: ca-num-sqrt.sage
  · rw [show I = ((0 : ℝ) : ℂ) + ((1 : ℝ) : ℂ) * I by push_cast; ring, norm_add_mul_I]
    veriphy_certified
      ( conv_rhs => rw [show (1 : ℝ) = √((1) ^ 2) from (Real.sqrt_sq (by norm_num)).symm]
        norm_num)
    -- certificate: ca-num-sqrt.sage
  · rw [show (2 : ℂ) + I = ((2 : ℝ) : ℂ) + ((1 : ℝ) : ℂ) * I by push_cast; ring, norm_add_mul_I]
    veriphy_certified (norm_num)
    -- certificate: ca-num-sqrt.sage

/-- `|(2 + i)(3 - i)| = |7 + i| = √50 = √5 √10`. -/
theorem norm_product_example :
    ‖(2 + I) * (3 - I)‖ = √50 ∧ ‖(7 : ℂ) + I‖ = √50 ∧ √50 = √5 * √10 := by
  have h7 : ‖(7 : ℂ) + I‖ = √50 := by
    rw [show (7 : ℂ) + I = ((7 : ℝ) : ℂ) + ((1 : ℝ) : ℂ) * I by push_cast; ring, norm_add_mul_I]
    veriphy_certified (norm_num)
    -- certificate: ca-num-sqrt.sage
  have hp : (2 + I) * (3 - I) = (7 : ℂ) + I := by
    linear_combination (-1 : ℂ) * I_sq
  refine ⟨by rw [hp, h7], h7, ?_⟩
  veriphy_certified
    ( rw [← Real.sqrt_mul (by norm_num)]
      norm_num)
  -- certificate: ca-num-sqrt.sage

/-! ## exm-modulus-circle -/

/-- The unit circle is closed under multiplication and inversion. -/
theorem circle_mul_inv (z w : ℂ) (hz : ‖z‖ = 1) (hw : ‖w‖ = 1) : ‖z * w‖ = 1 ∧ ‖z⁻¹‖ = 1 := by
  refine ⟨by rw [Complex.norm_mul, hz, hw, one_mul], by rw [norm_inv, hz, inv_one]⟩
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.circle_mul_inv (1) (1) (norm_one) (norm_one) := by
  constructor <;> simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.circle_mul_inv (Complex.I) ((-1)) (Complex.norm_I) ((by simp)) := by
  constructor <;> simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.circle_mul_inv ((-Complex.I)) (Complex.I) ((by simp)) (Complex.norm_I) := by
  constructor <;> simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Modulus.circle_mul_inv
-- test: ca-num-circle-test.sage

/-- `|1/z| = 1/|z|`. -/
theorem norm_one_div (z : ℂ) : ‖1 / z‖ = 1 / ‖z‖ := by rw [norm_div, norm_one]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.norm_one_div (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.norm_one_div (2) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.norm_one_div (Complex.I) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Modulus.norm_one_div
-- test: ca-num-norm-inv-test.sage

/-- Open discs (the open unit disc among them) are domains: open and connected. -/
theorem disc_domain (c : ℂ) (r : ℝ) (hr : 0 < r) : IsOpen (ball c r) ∧ IsConnected (ball c r) :=
  ⟨isOpen_ball, (convex_ball _ _).isConnected (nonempty_ball.mpr hr)⟩
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.disc_domain (0) (1) (one_pos) := by
  exact ⟨Metric.isOpen_ball, (convex_ball _ _).isConnected (Metric.nonempty_ball.mpr (by norm_num))⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.disc_domain (Complex.I) (2) (two_pos) := by
  exact ⟨Metric.isOpen_ball, (convex_ball _ _).isConnected (Metric.nonempty_ball.mpr (by norm_num))⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.disc_domain (1) ((1 / 2)) ((by norm_num)) := by
  exact ⟨Metric.isOpen_ball, (convex_ball _ _).isConnected (Metric.nonempty_ball.mpr (by norm_num))⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Modulus.disc_domain
-- test: ca-num-disc-domain-test.sage

/-- Closed discs (the closed unit disc among them) are compact. -/
theorem closed_disc_compact (c : ℂ) (r : ℝ) : IsCompact (closedBall c r) := isCompact_closedBall _ _
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.closed_disc_compact (0) (0) := by
  exact isCompact_closedBall _ _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.closed_disc_compact (0) (1) := by
  exact isCompact_closedBall _ _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.closed_disc_compact (Complex.I) (2) := by
  exact isCompact_closedBall _ _

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Modulus.closed_disc_compact
-- test: ca-num-disc-compact-test.sage

/-! ## exm-modulus-reverse-triangle -/

/-- The reverse triangle inequality `||z| - |w|| ≤ |z - w|`. -/
theorem reverse_triangle (z w : ℂ) : |‖z‖ - ‖w‖| ≤ ‖z - w‖ := abs_norm_sub_norm_le z w
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.reverse_triangle (0) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.Modulus.reverse_triangle
-- test: ca-num-reverse-triangle-test.sage

/-- On the circle `|z| = 2`, `|z² + 1| ≥ |z|² - 1 = 3`. -/
theorem lower_bound_on_circle (z : ℂ) (hz : ‖z‖ = 2) : 3 ≤ ‖z ^ 2 + 1‖ := by
  have h := norm_sub_norm_le (z ^ 2) (-1)
  rw [norm_pow, hz, norm_neg, norm_one, sub_neg_eq_add] at h
  linarith
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.lower_bound_on_circle (2) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.lower_bound_on_circle ((2 * Complex.I)) ((by simp)) := by
  rw [show (2 * Complex.I) ^ 2 + 1 = (-3 : ℂ) by linear_combination 4 * Complex.I_sq]
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.lower_bound_on_circle ((-2)) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Modulus.lower_bound_on_circle
-- test: ca-num-lower-bound-test.sage

/-! ## exr-modulus-triangle -/

/-- `|z + w|² = |z|² + |w|² + 2 Re(z w̄)`. -/
theorem sq_norm_add (z w : ℂ) : ‖z + w‖ ^ 2 = ‖z‖ ^ 2 + ‖w‖ ^ 2 + 2 * (z * conj w).re := by
  rw [← Complex.normSq_eq_norm_sq, ← Complex.normSq_eq_norm_sq, ← Complex.normSq_eq_norm_sq,
    Complex.normSq_add]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.sq_norm_add (0) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.Modulus.sq_norm_add
-- test: ca-num-sq-norm-add-test.sage

/-- `Re(z w̄) ≤ |z||w|`. -/
theorem re_mul_conj_le (z w : ℂ) : (z * conj w).re ≤ ‖z‖ * ‖w‖ := by
  have := Complex.re_le_norm (z * conj w)
  rwa [Complex.norm_mul, Complex.norm_conj] at this
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.re_mul_conj_le (0) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.Modulus.re_mul_conj_le
-- test: ca-num-re-le-test.sage

/-- Equality holds in the triangle inequality exactly when `z` and `w` point in the same direction. -/
theorem triangle_eq_iff (z w : ℂ) : ‖z + w‖ = ‖z‖ + ‖w‖ ↔ z = 0 ∨ w = 0 ∨ z.arg = w.arg :=
  Complex.norm_add_eq_iff
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.triangle_eq_iff (0) (1) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.triangle_eq_iff (1) (1) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.triangle_eq_iff (1) ((-1)) := by
  norm_num [Complex.arg_neg_one, Real.pi_pos.ne]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Modulus.triangle_eq_iff
-- test: ca-num-triangle-eq-test.sage

/-! ## exr-modulus-circle-equation -/

/-- `|z - 1| = 2|z + 1|` exactly when `3x² + 3y² + 10x + 3 = 0`, the circle of centre `-5/3` and radius `4/3`. -/
theorem apollonius (x y : ℝ) :
    ‖(x : ℂ) + y * I - 1‖ = 2 * ‖(x : ℂ) + y * I + 1‖ ↔ (x + 5 / 3) ^ 2 + y ^ 2 = (4 / 3) ^ 2 := by
  have e1 : (x : ℂ) + y * I - 1 = ((x - 1 : ℝ) : ℂ) + (y : ℂ) * I := by push_cast; ring
  have e2 : (x : ℂ) + y * I + 1 = ((x + 1 : ℝ) : ℂ) + (y : ℂ) * I := by push_cast; ring
  rw [e1, e2, norm_add_mul_I, norm_add_mul_I]
  constructor
  · intro h
    have h2 := congrArg (· ^ 2) h
    simp only [mul_pow, Real.sq_sqrt (by positivity : (0 : ℝ) ≤ (x - 1) ^ 2 + y ^ 2),
      Real.sq_sqrt (by positivity : (0 : ℝ) ≤ (x + 1) ^ 2 + y ^ 2)] at h2
    linarith
  · intro h
    rw [show (2 : ℝ) = √4 by rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)],
      ← Real.sqrt_mul (by norm_num)]
    congr 1
    linarith
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.apollonius (0) (0) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.apollonius ((-3)) (0) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.apollonius (5) (0) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Modulus.apollonius
-- test: ca-num-apollonius-test.sage

/-- Completing the square: `3x² + 3y² + 10x + 3 = 3((x + 5/3)² + y² - 16/9)`. -/
theorem apollonius_square (x y : ℝ) :
    3 * x ^ 2 + 3 * y ^ 2 + 10 * x + 3 = 3 * ((x + 5 / 3) ^ 2 + y ^ 2 - (4 / 3) ^ 2) := by
  veriphy_certified (ring1)
  -- certificate: ca-num-ring.sage

/-! ## exr-modulus-parallelogram -/

/-- The parallelogram law `|z + w|² + |z - w|² = 2|z|² + 2|w|²`. -/
theorem parallelogram (z w : ℂ) : ‖z + w‖ ^ 2 + ‖z - w‖ ^ 2 = 2 * ‖z‖ ^ 2 + 2 * ‖w‖ ^ 2 := by
  simp only [← Complex.normSq_eq_norm_sq, Complex.normSq_apply, Complex.add_re, Complex.add_im,
    Complex.sub_re, Complex.sub_im]
  ring
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.parallelogram (0) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.Modulus.parallelogram
-- test: ca-num-parallelogram-test.sage

/-- `Re(z w̄) = xu + yv` is the Euclidean dot product. -/
theorem re_mul_conj_dot (x y u v : ℝ) :
    (((x : ℂ) + y * I) * conj ((u : ℂ) + v * I)).re = x * u + y * v := by
  simp [Complex.mul_re]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.re_mul_conj_dot (0) (0) (0) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.re_mul_conj_dot (1) (2) (3) (4) := by
  simp [Complex.mul_re]
  all_goals ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.re_mul_conj_dot ((1 / 2)) ((-1)) (2) (3) := by
  simp [Complex.mul_re]
  all_goals ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Modulus.re_mul_conj_dot
-- test: ca-num-dot-test.sage

/-- `⟨z, z⟩ = Re(z z̄) = |z|²`: the modulus is the norm of the Euclidean inner product. -/
theorem inner_self_eq (z : ℂ) : (z * conj z).re = ‖z‖ ^ 2 := by
  rw [Complex.mul_conj, Complex.ofReal_re, Complex.normSq_eq_norm_sq]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.inner_self_eq (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Modulus.inner_self_eq (1) := by
  simp

attribute [veriphy_tested oracle_test 2 1] KaluzaKlein.ComplexAnalysis.Modulus.inner_self_eq
-- test: ca-num-inner-self-test.sage

end KaluzaKlein.ComplexAnalysis.Modulus
