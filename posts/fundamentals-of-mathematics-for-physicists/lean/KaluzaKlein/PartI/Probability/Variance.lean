import KaluzaKlein.PartI.Probability.RandomVariables

/-!
# Probability and statistics, section "Variance"

The claims of the blocks `def-variance` to `exr-rv-skewness` of `tutorial/part-i-08-probability.qmd`: variance,
covariance and their rules in the finite model, the variance of sums of independent coordinates on the product
space, the running examples (dice, indicators, binomial counts) and the continuous variances (uniform, exponential).
-/

open Finset

namespace KaluzaKlein.Probability.Variance

open SampleSpaces Conditioning RandomVariables

/-! ## def-variance -/

/-- The variance `E (X - E X)²`. -/
def Var {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X : Ω → ℚ) : ℚ := E p (fun ω => (X ω - E p X) ^ 2)

/-- The covariance `E[(X - E X)(Y - E Y)]`. -/
def Cov {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X Y : Ω → ℚ) : ℚ := E p (fun ω => (X ω - E p X) * (Y ω - E p Y))

/-- `Var X = E X² - (E X)²`. -/
theorem Var_eq {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (X : Ω → ℚ) :
    Var p X = E p (fun ω => X ω ^ 2) - (E p X) ^ 2 := by
  have h := hp.sum_one
  simp only [Var, E]
  have : ∀ ω, p ω * (X ω - ∑ ω, p ω * X ω) ^ 2 =
      p ω * X ω ^ 2 - (2 * ∑ ω, p ω * X ω) * (p ω * X ω) + (∑ ω, p ω * X ω) ^ 2 * p ω := fun ω => by ring
  simp only [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.sum_mul, h]
  ring

/-- The variance is non-negative, so the standard deviation `√(Var X)` is defined. -/
theorem Var_nonneg {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (X : Ω → ℚ) : 0 ≤ Var p X :=
  Finset.sum_nonneg fun ω _ => mul_nonneg (hp.nonneg ω) (sq_nonneg _)

/-- The expectation of an affine image: `E(aX + b) = a E X + b`. -/
theorem E_affine {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (a b : ℚ) (X : Ω → ℚ) :
    E p (fun ω => a * X ω + b) = a * E p X + b := by
  have := E_linear p a b X (fun _ => 1)
  simp only [mul_one] at this
  rw [this, E_const hp]; ring

/-- The scaling rule `Var(aX + b) = a² Var X`. -/
theorem Var_affine {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (a b : ℚ) (X : Ω → ℚ) :
    Var p (fun ω => a * X ω + b) = a ^ 2 * Var p X := by
  simp only [Var, E_affine hp]
  simp only [E, Finset.mul_sum]
  exact Finset.sum_congr rfl fun ω _ => by ring

/-- `Var(X + Y) = Var X + Var Y + 2 Cov(X, Y)`. -/
theorem Var_add {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (X Y : Ω → ℚ) :
    Var p (fun ω => X ω + Y ω) = Var p X + Var p Y + 2 * Cov p X Y := by
  have hE : E p (fun ω => X ω + Y ω) = E p X + E p Y := (E_eq_sum_points p X Y).2
  simp only [Var, Cov, hE]
  simp only [E, Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun ω _ => by ring

/-- `Cov(X, Y) = E(XY) - E X E Y`, so independent variables have covariance zero. -/
theorem Cov_eq {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (X Y : Ω → ℚ) :
    Cov p X Y = E p (fun ω => X ω * Y ω) - E p X * E p Y := by
  have h := hp.sum_one
  simp only [Cov, E]
  have : ∀ ω, p ω * ((X ω - ∑ ω, p ω * X ω) * (Y ω - ∑ ω, p ω * Y ω)) =
      p ω * (X ω * Y ω) - (∑ ω, p ω * Y ω) * (p ω * X ω) - (∑ ω, p ω * X ω) * (p ω * Y ω) +
        ((∑ ω, p ω * X ω) * ∑ ω, p ω * Y ω) * p ω := fun ω => by ring
  simp only [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.sum_mul, h]
  ring

/-- Independent variables have covariance zero, so their variances add. -/
theorem Var_add_of_indep {Ω : Type} [Fintype Ω] [DecidableEq Ω] {p : Ω → ℚ} (hp : IsDist p) {X Y : Ω → ℚ}
    (h : indepRV p X Y) : Cov p X Y = 0 ∧ Var p (fun ω => X ω + Y ω) = Var p X + Var p Y := by
  have hc : Cov p X Y = 0 := by rw [Cov_eq hp, E_mul_of_indep p X Y h]; ring
  exact ⟨hc, by rw [Var_add hp, hc]; ring⟩

/-- The variance of the sum of `n` independent copies is `n` times the variance: on the product space,
`Var(∑ᵢ X(wᵢ)) = n Var X`, and the mean is `n E X`. -/
theorem Var_sum_coord {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (n : ℕ) (X : Ω → ℚ) :
    E (prodW p n) (fun w => ∑ i, X (w i)) = n * E p X ∧
      Var (prodW p n) (fun w => ∑ i, X (w i)) = n * Var p X := by
  have hpn := prodW_isDist hp n
  have hmean : E (prodW p n) (fun w => ∑ i, X (w i)) = n * E p X := by
    rw [E_sum]; simp [E_coord hp n X]
  refine ⟨hmean, ?_⟩
  rw [Var_eq hpn, Var_eq hp, hmean]
  have hsq : ∀ w : Fin n → Ω, (∑ i, X (w i)) ^ 2 = ∑ i, ∑ j, X (w i) * X (w j) := fun w => by
    rw [sq, Finset.sum_mul_sum]
  simp only [hsq]
  rw [E_sum]
  have hinner : ∀ i : Fin n, E (prodW p n) (fun w => ∑ j, X (w i) * X (w j)) =
      E p (fun ω => X ω ^ 2) + (n - 1) * (E p X) ^ 2 := by
    intro i
    rw [E_sum]
    have : ∀ j : Fin n, E (prodW p n) (fun w => X (w i) * X (w j)) =
        if j = i then E p (fun ω => X ω ^ 2) else (E p X) ^ 2 := by
      intro j
      split_ifs with hij
      · subst hij; rw [← E_coord hp n (fun ω => X ω ^ 2) j]; congr 1; funext w; ring
      · rw [E_coord_mul hp n X X (Ne.symm hij), sq]
    simp only [this, Finset.sum_ite_eq', mem_univ, if_true, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
    have hn : (n : ℚ) = (univ.filter (fun j : Fin n => ¬ j = i)).card + 1 := by
      rw [Finset.filter_ne', Finset.card_erase_of_mem (mem_univ i), Finset.card_univ, Fintype.card_fin]
      have : 0 < n := Fin.pos i
      push_cast [Nat.cast_sub this]; ring
    rw [Finset.sum_ite, Finset.sum_const, Finset.sum_const, nsmul_eq_mul, nsmul_eq_mul]
    simp only [Finset.filter_eq', mem_univ, if_true, Finset.card_singleton]
    rw [hn]; ring
  simp only [hinner, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

/-- mathlib's variance obeys the same scaling rule `Var(cX + b) = c² Var X`. -/
theorem variance_affine_measure {Ω : Type} [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω)
    [MeasureTheory.IsProbabilityMeasure μ] (X : Ω → ℝ) (hX : MeasureTheory.AEStronglyMeasurable X μ) (c b : ℝ) :
    ProbabilityTheory.variance (fun ω => c * X ω + b) μ = c ^ 2 * ProbabilityTheory.variance X μ := by
  rw [ProbabilityTheory.variance_add_const (hX.const_mul c) b, ProbabilityTheory.variance_const_mul]

/-! ## exm-rv-variance-dice -/

/-- `Var X = 91/6 - 49/4 = 35/12` for one die, and `1.7075 < σ < 1.7085`, so `σ = 1.708`. -/
theorem die_variance : Var (unif (Fin 6)) (fun i => (face i : ℚ)) = 35 / 12 ∧ (91 / 6 : ℚ) - 49 / 4 = 35 / 12 ∧
    (1.7075 : ℝ) < Real.sqrt (35 / 12) ∧ Real.sqrt (35 / 12) < 1.7085 := by
  refine ⟨by decide +kernel, by norm_num, ?_, ?_⟩
  · rw [Real.lt_sqrt (by norm_num)]; norm_num
  · rw [Real.sqrt_lt' (by norm_num)]; norm_num

/-- The two dice are independent, so `Var S = 35/12 + 35/12 = 35/6`, which the table reproduces; `2.414 < σ_S < 2.416`. -/
theorem two_dice_variance : Var (unif Dice2) (fun w => (sum2 w : ℚ)) = 35 / 6 ∧
    Var (unif Dice2) (fun w => (face w.1 : ℚ)) + Var (unif Dice2) (fun w => (face w.2 : ℚ)) = 35 / 6 ∧
    ∑ s ∈ Finset.Icc (2 : ℕ) 12, ((s : ℚ) - 7) ^ 2 * ((6 - |(s : ℚ) - 7|) / 36) = 35 / 6 ∧
    (2.414 : ℝ) < Real.sqrt (35 / 6) ∧ Real.sqrt (35 / 6) < 2.416 := by
  refine ⟨by decide +kernel, by decide +kernel, by decide +kernel, ?_, ?_⟩
  · rw [Real.lt_sqrt (by norm_num)]; norm_num
  · rw [Real.sqrt_lt' (by norm_num)]; norm_num

/-- For `n` dice, `E Sₙ = 7n/2` and `Var Sₙ = 35n/12`. -/
theorem n_dice_moments (n : ℕ) :
    E (prodW (unif (Fin 6)) n) (fun w => ∑ i, (face (w i) : ℚ)) = 7 * n / 2 ∧
      Var (prodW (unif (Fin 6)) n) (fun w => ∑ i, (face (w i) : ℚ)) = 35 * n / 12 := by
  have h := Var_sum_coord (unif_isDist (Fin 6)) n (fun i => (face i : ℚ))
  rw [die_moments.1, die_variance.1] at h
  exact ⟨by rw [h.1]; ring, by rw [h.2]; ring⟩

/-- The relative spread `σₙ / μₙ = √(35n/12) / (7n/2) = (2√(35/12)/7) / √n` shrinks like `1/√n`. -/
theorem relative_spread (n : ℕ) (hn : 0 < n) :
    Real.sqrt (35 * n / 12) / (7 * n / 2) = (2 * Real.sqrt (35 / 12) / 7) / Real.sqrt n := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  rw [show (35 * (n : ℝ) / 12) = (35 / 12) * n by ring, Real.sqrt_mul (by norm_num)]
  have hs : Real.sqrt (n : ℝ) ≠ 0 := (Real.sqrt_pos.2 hn').ne'
  field_simp
  rw [Real.mul_self_sqrt hn'.le]
  ring

/-! ## exm-rv-variance-continuous -/

/-- `Var U = ∫_0^1 (u - 1/2)² du = 1/12 = 1/3 - 1/4` for the uniform variable. -/
theorem uniform_variance : (∫ u in (0 : ℝ)..1, (u - 1 / 2) ^ 2) = 1 / 12 ∧ (1 / 3 : ℝ) - 1 / 4 = 1 / 12 := by
  refine ⟨?_, by norm_num⟩
  VERIPHY[pr-integral]

/-- Uniform on `[a, a + L]` is `a + L U`, whose variance is `L² Var U = L²/12` by the scaling rule. -/
theorem uniform_interval_variance {Ω : Type} [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω)
    [MeasureTheory.IsProbabilityMeasure μ] (U : Ω → ℝ) (hU : MeasureTheory.AEStronglyMeasurable U μ) (a L : ℝ)
    (hV : ProbabilityTheory.variance U μ = 1 / 12) :
    ProbabilityTheory.variance (fun ω => L * U ω + a) μ = L ^ 2 / 12 := by
  rw [variance_affine_measure μ U hU L a, hV]; ring

/-- The exponential variable: `E T² = 2/λ²`, so `Var T = 2/λ² - 1/λ² = 1/λ²` and the standard deviation `1/λ` equals
the mean. -/
theorem exponential_variance (lam : ℝ) (hlam : 0 < lam) :
    (∫ t in Set.Ioi (0 : ℝ), t ^ 2 * (lam * Real.exp (-lam * t))) = 2 / lam ^ 2 ∧
      2 / lam ^ 2 - (1 / lam) ^ 2 = 1 / lam ^ 2 ∧ Real.sqrt (1 / lam ^ 2) = 1 / lam := by
  refine ⟨?_, by ring, ?_⟩
  · have := exponential_moment lam hlam 2
    rw [this]; norm_num [Nat.factorial]
  · rw [one_div, Real.sqrt_inv, Real.sqrt_sq hlam.le, one_div]

/-! ## exm-rv-covariance -/

/-- `Var(X + X) = Var(2X) = 4 Var X`, `Cov(X, X) = Var X`, while for an independent copy `Var(X + Y) = 2 Var X`. -/
theorem correlated_vs_independent {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (X : Ω → ℚ) :
    Var p (fun ω => X ω + X ω) = 4 * Var p X ∧ Cov p X X = Var p X ∧
      Var (prodW p 2) (fun w => X (w 0) + X (w 1)) = 2 * Var p X := by
  refine ⟨?_, rfl, ?_⟩
  · have := Var_affine hp 2 0 X
    simp only [add_zero] at this
    rw [← this]; congr 1; funext ω; ring
  · have := (Var_sum_coord hp 2 X).2
    rw [← this]; congr 1; funext w; simp [Fin.sum_univ_two]

/-- For a die `X` and `Y = 7 - X`: `Cov(X, Y) = -Var X` and `X + Y = 7` has variance `0`. -/
theorem anticorrelated {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (X : Ω → ℚ) (c : ℚ) :
    Cov p X (fun ω => c - X ω) = -Var p X ∧ Var p (fun ω => X ω + (c - X ω)) = 0 := by
  constructor
  · have hE : E p (fun ω => c - X ω) = c - E p X := by
      have := E_affine hp (-1) c X
      rw [← this]; congr 1; funext ω; ring
    simp only [Cov, Var, hE]
    simp only [E, Finset.mul_sum, ← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun ω _ => by ring
  · have : (fun ω => X ω + (c - X ω)) = fun _ => c := by funext ω; ring
    rw [this, Var, E_const hp]
    simp [E, E_const hp]

/-- For the die, `Var(X + (7 - X)) = 0` and `Cov(X, 7 - X) = -35/12`. -/
theorem die_anticorrelated : Cov (unif (Fin 6)) (fun i => (face i : ℚ)) (fun i => 7 - (face i : ℚ)) = -(35 / 12) ∧
    Var (unif (Fin 6)) (fun i => (face i : ℚ) + (7 - (face i : ℚ))) = 0 := by
  decide +kernel

/-! ## exr-rv-binomial-variance -/

/-- An indicator with `P(A) = p` has `1_A² = 1_A`, so `Var 1_A = p - p² = p(1 - p)`. -/
theorem Var_ind {Ω : Type} [Fintype Ω] [DecidableEq Ω] {p : Ω → ℚ} (hp : IsDist p) (A : Finset Ω) :
    (∀ ω, ind A ω ^ 2 = ind A ω) ∧ Var p (ind A) = P p A * (1 - P p A) := by
  have hsq : ∀ ω, ind A ω ^ 2 = ind A ω := fun ω => by unfold ind; split_ifs <;> norm_num
  refine ⟨hsq, ?_⟩
  rw [Var_eq hp]
  simp only [hsq, E_ind]; ring

/-- The Bernoulli weights on one trial (`true` is a success). -/
def bern (p : ℚ) : Bool → ℚ := fun b => if b then p else 1 - p

/-- The value of one trial. -/
def bval : Bool → ℚ := fun b => if b then 1 else 0

/-- The `n` trials are independent: `bernW p n` is the product of the Bernoulli weights, and the number of
successes is the sum of the coordinates. -/
theorem bernW_eq_prodW (p : ℚ) (n : ℕ) : bernW p n = prodW (bern p) n ∧
    ∀ w : Fin n → Bool, (heads w : ℚ) = ∑ i, bval (w i) := by
  refine ⟨funext fun w => ?_, fun w => ?_⟩
  · rw [bernW_eq_prod]; rfl
  · simp [heads, bval, Finset.card_filter]

/-- The Bernoulli weights are a distribution with `Var = p(1 - p)`. -/
theorem bern_variance (p : ℚ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : IsDist (bern p) ∧ Var (bern p) bval = p * (1 - p) := by
  refine ⟨⟨fun b => by cases b <;> simp [bern] <;> linarith, by simp [bern]⟩, ?_⟩
  simp [Var, E, bern, bval, Fintype.sum_bool]; ring

/-- The number of successes in `n` independent trials has mean `np` and variance `np(1 - p)`. -/
theorem binomial_mean_variance (p : ℚ) (h0 : 0 ≤ p) (h1 : p ≤ 1) (n : ℕ) :
    E (bernW p n) (fun w => (heads w : ℚ)) = n * p ∧ Var (bernW p n) (fun w => (heads w : ℚ)) = n * (p * (1 - p)) := by
  obtain ⟨hw, hh⟩ := bernW_eq_prodW p n
  obtain ⟨hd, hv⟩ := bern_variance p h0 h1
  have hE : E (bern p) bval = p := by simp [E, bern, bval, Fintype.sum_bool]
  have := Var_sum_coord hd n bval
  rw [hv, hE] at this
  simp only [hw]
  have hfun : (fun w : Fin n → Bool => (heads w : ℚ)) = fun w => ∑ i, bval (w i) := funext hh
  rw [hfun]; exact this

/-- From the exact tables, `Var S = 35/6` for two dice and `Var N = 10/4 = 5/2` for ten coin flips, as the formulas
`2 · 35/12` and `np(1 - p)` require. -/
theorem variance_tables : Var (unif Dice2) (fun w => (sum2 w : ℚ)) = 2 * (35 / 12) ∧
    Var (bernW (1 / 2) 10) (fun w => (heads w : ℚ)) = 5 / 2 ∧ (10 : ℚ) * ((1 / 2) * (1 - 1 / 2)) = 5 / 2 := by
  refine ⟨by rw [two_dice_variance.1]; norm_num, ?_, by norm_num⟩
  rw [(binomial_mean_variance (1 / 2) (by norm_num) (by norm_num) 10).2]; norm_num

/-! ## exr-rv-skewness -/

/-- A density symmetric about `μ` has all odd central moments zero: the integrand `y^{2k+1} ρ(μ + y)` is odd. -/
theorem odd_central_moments_zero (ρ : ℝ → ℝ) (μ : ℝ) (hsym : ∀ y, ρ (μ + y) = ρ (μ - y)) (k : ℕ) :
    (∫ x : ℝ, (x - μ) ^ (2 * k + 1) * ρ x) = 0 := by
  have h1 : (∫ x : ℝ, (x - μ) ^ (2 * k + 1) * ρ x) = ∫ y : ℝ, y ^ (2 * k + 1) * ρ (μ + y) := by
    rw [← MeasureTheory.integral_add_left_eq_self μ (fun x => (x - μ) ^ (2 * k + 1) * ρ x)]
    simp
  have h2 : (∫ y : ℝ, y ^ (2 * k + 1) * ρ (μ + y)) = -∫ y : ℝ, y ^ (2 * k + 1) * ρ (μ + y) := by
    conv_lhs => rw [← MeasureTheory.integral_comp_neg (fun y => y ^ (2 * k + 1) * ρ (μ + y))]
    rw [← MeasureTheory.integral_neg]
    congr 1; funext y
    rw [Odd.neg_pow ⟨k, rfl⟩, show μ + -y = μ - y by ring, ← hsym]; ring
  rw [h1]; linarith

/-- `E T³ = 6/λ³` for the exponential distribution. -/
theorem exponential_third_moment (lam : ℝ) (hlam : 0 < lam) :
    (∫ t in Set.Ioi (0 : ℝ), t ^ 3 * (lam * Real.exp (-lam * t))) = 6 / lam ^ 3 := by
  rw [exponential_moment lam hlam 3]; norm_num [Nat.factorial]

/-- Expanding `(T - 1/λ)³` with the moments `1/λ, 2/λ², 6/λ³` gives `E(T - μ)³ = 6 - 6 + 3 - 1 = 2` in units of `1/λ³`. -/
theorem exponential_central_third (L m1 m2 m3 : ℝ) (h1 : m1 = L) (h2 : m2 = 2 * L ^ 2) (h3 : m3 = 6 * L ^ 3) :
    m3 - 3 * m1 * m2 + 3 * m1 ^ 2 * m1 - m1 ^ 3 = 2 * L ^ 3 := by
  VERIPHY[ideal_membership]

/-- With `σ = 1/λ` the skewness `E(T - μ)³ / σ³ = (2/λ³) / (1/λ)³ = 2`, positive. -/
theorem exponential_skewness (lam : ℝ) (hlam : 0 < lam) : (2 / lam ^ 3) / (1 / lam) ^ 3 = 2 := by
  VERIPHY[pr-identity]

end KaluzaKlein.Probability.Variance
