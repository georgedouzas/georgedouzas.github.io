import KaluzaKlein.PartI.Probability.Conditioning

/-!
# Probability and statistics, sections "Random variables" and "Expectation"

The claims of the blocks `def-random-variable` to `exr-rv-cauchy-no-mean` of `tutorial/part-i-08-probability.qmd`.
A random variable on the finite model is a function `X : Ω → ℚ`; its distribution is the probability mass function
`pmf p X`, its expectation `E p X = ∑ ω, p ω * X ω`. Independent repetitions live on the product space
`Fin n → Ω` with the product weights `prodW p n`, whose marginals and product rule are proved here and used by every
later section. Continuous variables are handled with Lebesgue measure and mathlib's integrals.
-/

open Finset

namespace KaluzaKlein.Probability.RandomVariables

open SampleSpaces Conditioning

/-! ## The finite model: random variables, distributions, expectation -/

/-- The expectation of a random variable `X` under the weights `p`. -/
def E {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X : Ω → ℚ) : ℚ := ∑ ω, p ω * X ω

/-- The probability mass function `x ↦ P(X = x)`. -/
def pmf {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X : Ω → ℚ) (x : ℚ) : ℚ := P p (univ.filter (fun ω => X ω = x))

/-- The distribution function `x ↦ P(X ≤ x)`. -/
def cdf {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X : Ω → ℚ) (x : ℚ) : ℚ := P p (univ.filter (fun ω => X ω ≤ x))

/-- The indicator of an event. -/
def ind {Ω : Type} [Fintype Ω] [DecidableEq Ω] (A : Finset Ω) : Ω → ℚ := fun ω => if ω ∈ A then 1 else 0

/-- The product weights of `n` independent repetitions of the experiment with weights `p`. -/
def prodW {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (n : ℕ) : (Fin n → Ω) → ℚ := fun w => ∏ i, p (w i)

/-- The fair coin: weights `1/2` on `Fin 2` (`1` is heads). -/
def coin : Fin 2 → ℚ := fun _ => 1 / 2

/-- Linearity of the expectation. -/
theorem E_linear {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (a b : ℚ) (X Y : Ω → ℚ) :
    E p (fun ω => a * X ω + b * Y ω) = a * E p X + b * E p Y := by
  simp only [E, Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun ω _ => by ring

/-- The expectation of a finite sum of random variables is the sum of the expectations. -/
theorem E_sum {Ω ι : Type} [Fintype Ω] (p : Ω → ℚ) (s : Finset ι) (X : ι → Ω → ℚ) :
    E p (fun ω => ∑ i ∈ s, X i ω) = ∑ i ∈ s, E p (X i) := by
  simp only [E, Finset.mul_sum]
  exact Finset.sum_comm

/-- The expectation of a constant is the constant (for a probability distribution). -/
theorem E_const {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (c : ℚ) : E p (fun _ => c) = c := by
  simp [E, ← Finset.sum_mul, hp.sum_one]

/-- Monotonicity: `X ≤ Y` implies `E X ≤ E Y`. -/
theorem E_mono {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) {X Y : Ω → ℚ} (h : ∀ ω, X ω ≤ Y ω) :
    E p X ≤ E p Y :=
  Finset.sum_le_sum fun ω _ => mul_le_mul_of_nonneg_left (h ω) (hp.nonneg ω)

/-- The expectation of an indicator is the probability of the event. -/
theorem E_ind {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (A : Finset Ω) : E p (ind A) = P p A := by
  simp [E, ind, P, Finset.sum_ite_mem]

/-- The expectation is the sum over the values of `x P(X = x)`. -/
theorem E_eq_sum_pmf {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X : Ω → ℚ) :
    E p X = ∑ x ∈ univ.image X, x * pmf p X x := by
  simp only [pmf, P, Finset.mul_sum]
  rw [E, ← Finset.sum_fiberwise_of_maps_to (g := X) (t := univ.image X) (fun ω _ => Finset.mem_image_of_mem X
    (mem_univ ω))]
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun ω hω => ?_
  rw [(Finset.mem_filter.1 hω).2]; ring

/-- The probabilities of the values sum to one. -/
theorem pmf_sum_one {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (X : Ω → ℚ) :
    ∑ x ∈ univ.image X, pmf p X x = 1 := by
  simp only [pmf, P]
  rw [Finset.sum_fiberwise_of_maps_to (fun ω _ => Finset.mem_image_of_mem X (mem_univ ω))]
  exact hp.sum_one

/-- The distribution function is non-decreasing and `P(a < X ≤ b) = F(b) - F(a)`. -/
theorem cdf_mono_sub {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (X : Ω → ℚ) {a b : ℚ} (hab : a ≤ b) :
    cdf p X a ≤ cdf p X b ∧ P p (univ.filter (fun ω => a < X ω ∧ X ω ≤ b)) = cdf p X b - cdf p X a := by
  have hsplit : univ.filter (fun ω => X ω ≤ b) =
      univ.filter (fun ω => X ω ≤ a) ∪ univ.filter (fun ω => a < X ω ∧ X ω ≤ b) := by
    ext ω; simp; constructor
    · intro h; by_cases h' : X ω ≤ a
      · exact Or.inl h'
      · exact Or.inr ⟨lt_of_not_ge h', h⟩
    · rintro (h | ⟨_, h⟩); exacts [h.trans hab, h]
  have hdisj : Disjoint (univ.filter (fun ω => X ω ≤ a)) (univ.filter (fun ω => a < X ω ∧ X ω ≤ b)) := by
    rw [Finset.disjoint_filter]; intro ω _ h h'; exact absurd h (not_le.2 h'.1)
  refine ⟨P_mono hp (Finset.monotone_filter_right _ fun ω h => h.trans hab), ?_⟩
  simp only [cdf]
  rw [hsplit, P_union_of_disjoint p hdisj]; ring

/-- `F` is between `0` and `1`. -/
theorem cdf_bounds {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (X : Ω → ℚ) (x : ℚ) :
    0 ≤ cdf p X x ∧ cdf p X x ≤ 1 :=
  ⟨P_nonneg hp _, P_le_one hp _⟩

/-- A random variable with density `ρ` has `P(a ≤ X ≤ b) = ∫_a^b ρ`, and its distribution function
`F(x) = ∫_{-∞}^x ρ` has derivative `ρ` at every point where `ρ` is continuous. -/
theorem density_cdf_deriv (ρ : ℝ → ℝ) (hρ : Continuous ρ) (a x : ℝ) :
    HasDerivAt (fun u => ∫ t in a..u, ρ t) (ρ x) x :=
  intervalIntegral.integral_hasDerivAt_right (hρ.intervalIntegrable a x)
    (hρ.stronglyMeasurableAtFilter _ _) hρ.continuousAt

/-- A measure given by a density assigns probability zero to every single point. -/
theorem density_singleton (ρ : ℝ → ℝ≥0∞) (hρ : Measurable ρ) (x : ℝ) :
    MeasureTheory.volume.withDensity ρ {x} = 0 := by
  rw [MeasureTheory.withDensity_apply _ (measurableSet_singleton x)]
  simp [MeasureTheory.Measure.restrict_singleton]

/-! ## The product space: marginals and the product rule -/

/-- The expectation of a product of functions of distinct coordinates is the product of the expectations. -/
theorem E_prod_coord {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (n : ℕ) (g : Fin n → Ω → ℚ) :
    E (prodW p n) (fun w => ∏ i, g i (w i)) = ∏ i, E p (g i) := by
  simp only [E, prodW]
  rw [Finset.prod_univ_sum]
  simp [Finset.prod_mul_distrib]

/-- The product weights form a probability distribution. -/
theorem prodW_isDist {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (n : ℕ) : IsDist (prodW p n) := by
  refine ⟨fun w => Finset.prod_nonneg fun i _ => hp.nonneg _, ?_⟩
  have := E_prod_coord p n (fun _ _ => 1)
  simpa [E, hp.sum_one] using this

/-- The marginal of a coordinate: `E[X(wᵢ)] = E[X]`. -/
theorem E_coord {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (n : ℕ) (X : Ω → ℚ) (i : Fin n) :
    E (prodW p n) (fun w => X (w i)) = E p X := by
  have := E_prod_coord p n (Function.update (fun _ _ => 1) i X)
  rw [Finset.prod_update_of_mem (mem_univ i)] at this
  simp only [Finset.prod_const_one, mul_one] at this
  rw [← this]
  congr 1; funext w
  rw [Finset.prod_update_of_mem (mem_univ i)]
  simp

/-- The product rule for two distinct coordinates: `E[X(wᵢ) Y(wⱼ)] = E[X] E[Y]` for `i ≠ j`. -/
theorem E_coord_mul {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (n : ℕ) (X Y : Ω → ℚ) {i j : Fin n}
    (hij : i ≠ j) : E (prodW p n) (fun w => X (w i) * Y (w j)) = E p X * E p Y := by
  have := E_prod_coord p n (Function.update (Function.update (fun _ _ => 1) i X) j Y)
  rw [Finset.prod_update_of_mem (mem_univ j), Finset.prod_update_of_mem (Finset.mem_sdiff.2 ⟨mem_univ i,
    by simpa using hij⟩)] at this
  have h1 : ∀ k ∈ (univ \ {j}) \ {i}, E p (Function.update (fun _ _ => 1) i X k) = 1 := by
    intro k hk
    simp only [Finset.mem_sdiff, Finset.mem_singleton] at hk
    rw [Function.update_of_ne hk.2]; simp [E, hp.sum_one]
  rw [Finset.prod_eq_one h1, mul_one, Function.update_self, ← this, mul_comm]
  congr 1; funext w
  rw [Finset.prod_update_of_mem (mem_univ j), Finset.prod_update_of_mem (Finset.mem_sdiff.2 ⟨mem_univ i,
    by simpa using hij⟩), Function.update_self]
  have h2 : ∀ k ∈ (univ \ {j}) \ {i}, Function.update (fun _ _ => 1) i X k (w k) = 1 := by
    intro k hk
    simp only [Finset.mem_sdiff, Finset.mem_singleton] at hk
    rw [Function.update_of_ne hk.2]
  rw [Finset.prod_eq_one h2]; ring

/-- The product weights of two spaces are the special case `n = 2` in disguise: `P(A × B) = P(A) P(B)` was
`prodW2_prob`; here the marginal of an event of one coordinate. -/
theorem P_coord {Ω : Type} [Fintype Ω] [DecidableEq Ω] {p : Ω → ℚ} (hp : IsDist p) (n : ℕ) (A : Finset Ω)
    (i : Fin n) : P (prodW p n) (univ.filter (fun w => w i ∈ A)) = P p A := by
  rw [← E_ind, ← E_ind]
  have := E_coord hp n (ind A) i
  rw [← this]
  congr 1; funext w; simp [ind]

/-! ## exm-rv-die-indicator -/

/-- The face of a fair die has `P(X = k) = 1/6` for `k = 1, …, 6`, and its distribution function jumps by `1/6` at
each integer: `F(k) = k/6`. -/
theorem die_pmf_cdf (k : Fin 6) : pmf (unif (Fin 6)) (fun i => (face i : ℚ)) (face k) = 1 / 6 ∧
    cdf (unif (Fin 6)) (fun i => (face i : ℚ)) (face k) = (face k : ℚ) / 6 := by
  fin_cases k <;> decide +kernel

/-- The indicator of `A` takes the value `1` with probability `P(A)` and `0` with probability `1 - P(A)`. -/
theorem ind_pmf {Ω : Type} [Fintype Ω] [DecidableEq Ω] {p : Ω → ℚ} (hp : IsDist p) (A : Finset Ω) :
    pmf p (ind A) 1 = P p A ∧ pmf p (ind A) 0 = 1 - P p A := by
  have h1 : univ.filter (fun ω => ind A ω = 1) = A := by ext ω; simp [ind]
  have h0 : univ.filter (fun ω => ind A ω = 0) = Aᶜ := by ext ω; simp [ind]
  simp only [pmf, h1, h0, P_compl hp]
  exact ⟨rfl, rfl⟩

/-- The number of heads in `n` tosses is the sum of the indicators of "toss `i` is heads". -/
theorem heads_eq_sum_ind (n : ℕ) (w : Fin n → Bool) :
    (heads w : ℚ) = ∑ i, ind (univ.filter (fun v : Fin n → Bool => v i = true)) w := by
  simp [heads, ind, Finset.card_filter]

/-! ## exm-rv-two-dice-sum -/

/-- The sum of two dice has `P(S = s) = (6 - |s - 7|) / 36` for `s = 2, …, 12`. -/
theorem sum_pmf (s : ℕ) (h2 : 2 ≤ s) (h12 : s ≤ 12) :
    pmf (unif Dice2) (fun w => (sum2 w : ℚ)) s = (6 - |(s : ℚ) - 7|) / 36 := by
  interval_cases s <;> decide +kernel

/-- `S'(i, j) = i + (7 - j)` is a different map on `Ω` with the same distribution as `S`. -/
theorem sum_other_same_distribution :
    (fun w : Dice2 => (face w.1 + (7 - face w.2) : ℚ)) ≠ (fun w => (sum2 w : ℚ)) ∧
      ∀ s : ℕ, s ≤ 13 → pmf (unif Dice2) (fun w => (face w.1 + (7 - face w.2) : ℚ)) s =
        pmf (unif Dice2) (fun w => (sum2 w : ℚ)) s := by
  refine ⟨fun h => ?_, fun s hs => ?_⟩
  · have := congrFun h (0, 0)
    simp [face, sum2] at this; norm_num at this
  · interval_cases s <;> decide +kernel

/-! ## exm-rv-uniform-exponential -/

/-- The uniform variable on `[0, 1]`: `F(u) = ∫_0^u 1 = u` and `P(1/4 < U ≤ 3/4) = 1/2`. -/
theorem uniform_cdf (u : ℝ) : (∫ t in (0 : ℝ)..u, (1 : ℝ)) = u ∧ (∫ t in (1 / 4 : ℝ)..(3 / 4), (1 : ℝ)) = 1 / 2 := by
  constructor
  · simp
  · VERIPHY[pr-integral]

/-- The exponential density `λ e^{-λt}` has distribution function `F(t) = 1 - e^{-λt}`. -/
theorem exponential_cdf (lam : ℝ) (s : ℝ) :
    ∫ t in (0 : ℝ)..s, lam * Real.exp (-lam * t) = 1 - Real.exp (-lam * s) := by
  have hF : ∀ t, HasDerivAt (fun t => -Real.exp (-lam * t)) (lam * Real.exp (-lam * t)) t := by
    intro t
    have := ((Real.hasDerivAt_exp (-lam * t)).comp t ((hasDerivAt_id t).const_mul (-lam))).neg
    convert this using 1; simp; ring
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => hF t) (by fun_prop)]
  simp

/-- The survival probability `P(T > t) = ∫_t^∞ λ e^{-λx} dx = e^{-λt}`; in particular the density integrates to
one. -/
theorem exponential_survival (lam : ℝ) (hlam : 0 < lam) (t : ℝ) :
    (∫ x in Set.Ioi t, lam * Real.exp (-lam * x)) = Real.exp (-lam * t) ∧
      (∫ x in Set.Ioi (0 : ℝ), lam * Real.exp (-lam * x)) = 1 := by
  have key : ∀ t, (∫ x in Set.Ioi t, lam * Real.exp (-lam * x)) = Real.exp (-lam * t) := by
    intro t
    rw [MeasureTheory.integral_const_mul]
    have := MeasureTheory.integral_comp_mul_left_Ioi (fun x => Real.exp (-x)) t hlam
    simp only [smul_eq_mul] at this
    rw [show (fun x => Real.exp (-lam * x)) = fun x => Real.exp (-(lam * x)) by funext x; ring_nf, this,
      integral_exp_neg_Ioi]
    field_simp
  exact ⟨key t, by simpa using key 0⟩

/-- The uniform density integrates to one. -/
theorem uniform_total : (∫ t in (0 : ℝ)..1, (1 : ℝ)) = 1 := by simp

/-! ## exr-rv-max-two-dice -/

/-- The maximum of two dice: `F(m) = m²/36` and `P(M = m) = (2m - 1)/36`; `P(M = 6) = 11/36` exceeds
`P(S = 7) = 6/36`, and `E M = 161/36`. -/
theorem max_two_dice (m : Fin 6) :
    cdf (unif Dice2) (fun w => (max (face w.1) (face w.2) : ℚ)) (face m) = (face m : ℚ) ^ 2 / 36 ∧
      pmf (unif Dice2) (fun w => (max (face w.1) (face w.2) : ℚ)) (face m) = (2 * (face m : ℚ) - 1) / 36 ∧
      pmf (unif Dice2) (fun w => (max (face w.1) (face w.2) : ℚ)) 6 = 11 / 36 ∧ (11 / 36 : ℚ) > 6 / 36 ∧
      E (unif Dice2) (fun w => (max (face w.1) (face w.2) : ℚ)) = 161 / 36 := by
  fin_cases m <;> decide +kernel

/-! ## exr-rv-density-transform -/

/-- For `X` uniform on `[-1, 1]` and `0 ≤ y ≤ 1`, `P(X² ≤ y) = P(-√y ≤ X ≤ √y) = √y`. -/
theorem square_cdf (y : ℝ) (h0 : 0 ≤ y) (h1 : y ≤ 1) :
    (MeasureTheory.volume (Set.Icc (-1 : ℝ) 1 ∩ {x | x ^ 2 ≤ y})).toReal / 2 = Real.sqrt y := by
  have hs : {x : ℝ | x ^ 2 ≤ y} = Set.Icc (-Real.sqrt y) (Real.sqrt y) := by
    ext x; simp only [Set.mem_setOf_eq, Set.mem_Icc]; exact Real.sq_le h0
  have hle : Real.sqrt y ≤ 1 := Real.sqrt_le_one.2 h1 |> fun h => by simpa using h
  rw [hs, Set.inter_eq_right.2 (Set.Icc_subset_Icc (by linarith) hle), Real.volume_Icc,
    ENNReal.toReal_ofReal (by linarith [Real.sqrt_nonneg y])]
  ring

/-- The density `1 / (2√y)` of `X²` integrates to one over `(0, 1]` (an improper but convergent integral). -/
theorem square_density_total : ∫ y in (0 : ℝ)..1, 1 / (2 * Real.sqrt y) = 1 := by
  have : ∀ y ∈ Set.uIcc (0 : ℝ) 1, 1 / (2 * Real.sqrt y) = (1 / 2) * y ^ (-(1 / 2) : ℝ) := by
    intro y hy
    rw [Set.uIcc_of_le zero_le_one] at hy
    rw [Real.sqrt_eq_rpow, Real.rpow_neg hy.1]; field_simp
  rw [intervalIntegral.integral_congr this, intervalIntegral.integral_const_mul,
    integral_rpow (Or.inl (by norm_num))]
  norm_num

/-- The distribution function `√y` has derivative `1 / (2√y)` for `y > 0`. -/
theorem sqrt_hasDerivAt (y : ℝ) (hy : 0 < y) : HasDerivAt Real.sqrt (1 / (2 * Real.sqrt y)) y :=
  Real.hasDerivAt_sqrt hy.ne'

/-- The density is unbounded near `0`: it exceeds any `M` at some `y ∈ (0, 1]`. -/
theorem square_density_unbounded (M : ℝ) : ∃ y ∈ Set.Ioc (0 : ℝ) 1, M < 1 / (2 * Real.sqrt y) := by
  refine ⟨1 / (4 * (|M| + 1) ^ 2), ⟨by positivity, ?_⟩, ?_⟩
  · rw [div_le_one (by positivity)]; nlinarith [abs_nonneg M]
  · have hpos : 0 < |M| + 1 := by positivity
    rw [show (1 : ℝ) / (4 * (|M| + 1) ^ 2) = (1 / (2 * (|M| + 1))) ^ 2 by field_simp; ring,
      Real.sqrt_sq (by positivity)]
    calc M ≤ |M| := le_abs_self M
      _ < |M| + 1 := by linarith
      _ = 1 / (2 * (1 / (2 * (|M| + 1)))) := by field_simp

/-! ## exr-rv-convolution -/

/-- The convolution formula: for independent `X`, `Y` (on a product space),
`P(X + Y = s) = ∑ₖ P(X = k) P(Y = s - k)`, the sum running over the values of `X`. -/
theorem pmf_add_prod {Ω Ω' : Type} [Fintype Ω] [Fintype Ω'] (p : Ω → ℚ) (q : Ω' → ℚ) (X : Ω → ℚ) (Y : Ω' → ℚ)
    (s : ℚ) : pmf (prodW2 p q) (fun w => X w.1 + Y w.2) s = ∑ k ∈ univ.image X, pmf p X k * pmf q Y (s - k) := by
  simp only [pmf, P, prodW2]
  rw [Finset.sum_filter, Fintype.sum_prod_type]
  simp_rw [← Finset.sum_filter]
  have : ∀ a : Ω, (univ.filter (fun b => X a + Y b = s)) = univ.filter (fun b => Y b = s - X a) := by
    intro a; ext b; simp; constructor <;> intro h <;> linarith
  simp_rw [this, ← Finset.mul_sum]
  rw [← Finset.sum_fiberwise_of_maps_to (g := X) (t := univ.image X) (fun ω _ => Finset.mem_image_of_mem X
    (mem_univ ω))]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun a ha => ?_
  rw [(Finset.mem_filter.1 ha).2]

/-- One die convolved with itself gives the triangular table, and three dice: `P(S₃ = 10) = ∑ₖ (1/6) P(S₂ = 10 - k)
= (3 + 4 + 5 + 6 + 5 + 4) / 216 = 27/216`. -/
theorem convolution_dice :
    (∀ s : ℕ, 2 ≤ s → s ≤ 12 → ∑ k ∈ univ.image (fun i : Fin 6 => (face i : ℚ)),
      pmf (unif (Fin 6)) (fun i => (face i : ℚ)) k * pmf (unif (Fin 6)) (fun i => (face i : ℚ)) (s - k) =
        (6 - |(s : ℚ) - 7|) / 36) ∧
    ∑ k ∈ Finset.Icc (1 : ℕ) 6, (1 / 6 : ℚ) * ((6 - |((10 - k : ℕ) : ℚ) - 7|) / 36) = 27 / 216 ∧
    (3 + 4 + 5 + 6 + 5 + 4 : ℚ) / 216 = 27 / 216 := by
  refine ⟨fun s h2 h12 => ?_, by decide +kernel, by norm_num⟩
  interval_cases s <;> decide +kernel

/-! ## def-expectation -/

/-- Two random variables are independent when the events `{X = x}` and `{Y = y}` are independent for all `x, y`. -/
def indepRV {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (X Y : Ω → ℚ) : Prop :=
  ∀ x y, P p (univ.filter (fun ω => X ω = x ∧ Y ω = y)) = pmf p X x * pmf p Y y

/-- The product rule `E(XY) = E X · E Y` for independent random variables. -/
theorem E_mul_of_indep {Ω : Type} [Fintype Ω] [DecidableEq Ω] (p : Ω → ℚ) (X Y : Ω → ℚ) (h : indepRV p X Y) :
    E p (fun ω => X ω * Y ω) = E p X * E p Y := by
  rw [E_eq_sum_pmf p X, E_eq_sum_pmf p Y, Finset.sum_mul_sum]
  simp only [E]
  rw [← Finset.sum_fiberwise_of_maps_to (g := X) (t := univ.image X) (fun ω _ => Finset.mem_image_of_mem X
    (mem_univ ω))]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [← Finset.sum_fiberwise_of_maps_to (g := Y) (t := univ.image Y) (fun ω _ => Finset.mem_image_of_mem Y
    (mem_univ ω))]
  refine Finset.sum_congr rfl fun y _ => ?_
  have hxy : ∑ ω ∈ (univ.filter fun ω => X ω = x).filter (fun ω => Y ω = y), p ω * (X ω * Y ω) =
      x * y * P p (univ.filter (fun ω => X ω = x ∧ Y ω = y)) := by
    rw [Finset.filter_filter, P, Finset.mul_sum]
    refine Finset.sum_congr rfl fun ω hω => ?_
    obtain ⟨_, h1, h2⟩ := Finset.mem_filter.1 hω
    rw [h1, h2]; ring
  rw [hxy, h x y]; ring

/-- Expectations with respect to a measure are linear and monotone (mathlib's integral). -/
theorem integral_linear_mono {Ω : Type} [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω) (a b : ℝ)
    (X Y : Ω → ℝ) (hX : MeasureTheory.Integrable X μ) (hY : MeasureTheory.Integrable Y μ) (hXY : X ≤ Y) :
    (∫ ω, (a * X ω + b * Y ω) ∂μ) = a * ∫ ω, X ω ∂μ + b * ∫ ω, Y ω ∂μ ∧ (∫ ω, X ω ∂μ) ≤ ∫ ω, Y ω ∂μ := by
  constructor
  · rw [MeasureTheory.integral_add (hX.const_mul a) (hY.const_mul b), MeasureTheory.integral_const_mul,
      MeasureTheory.integral_const_mul]
  · exact MeasureTheory.integral_mono hX hY hXY

/-! ## exm-rv-expectation-dice -/

/-- `∑_{k=1}^n k = n(n+1)/2`. -/
theorem sum_Icc_id (n : ℕ) : ∑ k ∈ Finset.Icc 1 n, (k : ℚ) = n * (n + 1) / 2 := by
  VERIPHY[pr-power-sum]

/-- `∑_{k=1}^n k² = n(n+1)(2n+1)/6`. -/
theorem sum_Icc_sq (n : ℕ) : ∑ k ∈ Finset.Icc 1 n, (k : ℚ) ^ 2 = n * (n + 1) * (2 * n + 1) / 6 := by
  VERIPHY[pr-power-sum]

/-- `E X = (1 + ⋯ + 6)/6 = 7/2` and `E X² = (1 + 4 + ⋯ + 36)/6 = 91/6` for one die. -/
theorem die_moments : E (unif (Fin 6)) (fun i => (face i : ℚ)) = 7 / 2 ∧
    E (unif (Fin 6)) (fun i => (face i : ℚ) ^ 2) = 91 / 6 ∧
    (∑ k ∈ Finset.Icc 1 6, (k : ℚ)) / 6 = 7 / 2 ∧ (∑ k ∈ Finset.Icc 1 6, (k : ℚ) ^ 2) / 6 = 91 / 6 := by
  decide +kernel

/-- For two independent dice, `E S = 7/2 + 7/2 = 7` by linearity, the table gives `∑ s (6 - |s - 7|)/36 = 7`, and
independence gives `E(XY) = 49/4`. -/
theorem two_dice_expectations : E (unif Dice2) (fun w => (sum2 w : ℚ)) = 7 ∧
    E (unif Dice2) (fun w => (face w.1 : ℚ)) + E (unif Dice2) (fun w => (face w.2 : ℚ)) = 7 ∧
    ∑ s ∈ Finset.Icc (2 : ℕ) 12, (s : ℚ) * ((6 - |(s : ℚ) - 7|) / 36) = 7 ∧
    E (unif Dice2) (fun w => (face w.1 : ℚ) * face w.2) = 49 / 4 ∧
    E (unif Dice2) (fun w => (face w.1 : ℚ) * face w.2) =
      E (unif Dice2) (fun w => (face w.1 : ℚ)) * E (unif Dice2) (fun w => (face w.2 : ℚ)) := by
  decide +kernel

/-! ## exm-rv-sum-of-indicators -/

/-- The expected number of heads in `n` fair tosses is `n/2`: the sum of `n` indicators of probability `1/2`. -/
theorem E_heads (n : ℕ) : E (prodW coin n) (fun w => ∑ i, ((w i).val : ℚ)) = n / 2 := by
  rw [E_sum]
  have : ∀ i : Fin n, E (prodW coin n) (fun w => ((w i).val : ℚ)) = 1 / 2 := fun i => by
    rw [E_coord ⟨fun _ => by norm_num [coin], by simp [coin]; norm_num⟩ n (fun b : Fin 2 => (b.val : ℚ)) i]
    decide +kernel
  simp [this]; ring

/-- For a uniformly random permutation of `n ≥ 1` objects, `P(σ(i) = i) = 1/n` for each `i`, so the expected number
of fixed points is `n · (1/n) = 1`, although the indicators are not independent. -/
theorem E_fixed_points (n : ℕ) (hn : 0 < n) :
    (∀ i : Fin n, P (unif (Equiv.Perm (Fin n))) (univ.filter (fun σ => σ i = i)) = 1 / n) ∧
      E (unif (Equiv.Perm (Fin n))) (fun σ => ∑ i, ind (univ.filter (fun τ : Equiv.Perm (Fin n) => τ i = i)) σ) = 1 := by
  have key : ∀ i : Fin n, P (unif (Equiv.Perm (Fin n))) (univ.filter (fun σ => σ i = i)) = 1 / n := by
    intro i
    rw [P_unif]
    have hfib : ∑ a : Fin n, (univ.filter (fun σ : Equiv.Perm (Fin n) => σ i = a)).card =
        Fintype.card (Equiv.Perm (Fin n)) := by
      rw [← Finset.card_univ, Finset.card_eq_sum_card_fiberwise (f := fun σ : Equiv.Perm (Fin n) => σ i)
        (t := univ) (fun _ _ => mem_univ _)]
    have hsame : ∀ a : Fin n, (univ.filter (fun σ : Equiv.Perm (Fin n) => σ i = a)).card =
        (univ.filter (fun σ : Equiv.Perm (Fin n) => σ i = i)).card := by
      intro a
      refine Finset.card_equiv (Equiv.mulLeft (Equiv.swap a i)) fun σ => ?_
      simp only [Finset.mem_filter, mem_univ, true_and, Equiv.coe_mulLeft, Equiv.Perm.mul_apply]
      constructor
      · intro h; rw [h, Equiv.swap_apply_left]
      · intro h
        have := congrArg (Equiv.swap a i) h
        rwa [Equiv.swap_apply_self, Equiv.swap_apply_right] at this
    simp only [hsame, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul] at hfib
    have hn' : (n : ℚ) ≠ 0 := by exact_mod_cast hn.ne'
    have hc : (Fintype.card (Equiv.Perm (Fin n)) : ℚ) = n * (univ.filter (fun σ : Equiv.Perm (Fin n) => σ i = i)).card := by
      exact_mod_cast hfib.symm
    rw [hc]
    have hpos : ((univ.filter (fun σ : Equiv.Perm (Fin n) => σ i = i)).card : ℚ) ≠ 0 := by
      have : (1 : Equiv.Perm (Fin n)) ∈ univ.filter (fun σ : Equiv.Perm (Fin n) => σ i = i) := by simp
      exact_mod_cast (Finset.card_pos.2 ⟨_, this⟩).ne'
    field_simp
  refine ⟨key, ?_⟩
  rw [E_sum]
  simp only [E_ind, key, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-! ## exm-rv-expectation-continuous -/

/-- `E U = ∫_0^1 u du = 1/2` and `E U² = ∫_0^1 u² du = 1/3`. -/
theorem uniform_moments : (∫ u in (0 : ℝ)..1, u) = 1 / 2 ∧ (∫ u in (0 : ℝ)..1, u ^ 2) = 1 / 3 := by
  constructor
  · VERIPHY[pr-integral]
  · VERIPHY[pr-integral]

/-- The moments of the exponential distribution: `∫_0^∞ tⁿ λ e^{-λt} dt = n! / λⁿ`; in particular the mean is
`1/λ`. -/
theorem exponential_moment (lam : ℝ) (hlam : 0 < lam) (n : ℕ) :
    (∫ t in Set.Ioi (0 : ℝ), t ^ n * (lam * Real.exp (-lam * t))) = n.factorial / lam ^ n := by
  have h := integral_rpow_mul_exp_neg_mul_rpow (p := 1) (q := n) (b := lam) one_pos (by
    have : (0 : ℝ) ≤ n := Nat.cast_nonneg n; linarith) hlam
  have hcongr : ∀ x ∈ Set.Ioi (0 : ℝ), x ^ n * (lam * Real.exp (-lam * x)) =
      lam * (x ^ (n : ℝ) * Real.exp (-lam * x ^ (1 : ℝ))) := by
    intro x hx
    rw [Real.rpow_natCast, Real.rpow_one]; ring
  rw [MeasureTheory.setIntegral_congr_fun measurableSet_Ioi hcongr, MeasureTheory.integral_const_mul, h]
  have hG : Real.Gamma ((n + 1) / 1) = n.factorial := by rw [div_one, Real.Gamma_nat_eq_factorial]
  rw [hG, show -((n : ℝ) + 1) / 1 = -((n + 1 : ℕ) : ℝ) by push_cast; ring, Real.rpow_neg hlam.le,
    Real.rpow_natCast, pow_succ]
  field_simp

/-- The mean lifetime of an exponential variable with rate `λ` is `1/λ`. -/
theorem exponential_mean (lam : ℝ) (hlam : 0 < lam) :
    (∫ t in Set.Ioi (0 : ℝ), t * (lam * Real.exp (-lam * t))) = 1 / lam := by
  have := exponential_moment lam hlam 1
  simpa using this

/-- The Cauchy density `1 / (π (1 + x²))` integrates to one over the line. -/
theorem cauchy_total : (∫ x : ℝ, 1 / (Real.pi * (1 + x ^ 2))) = 1 := by
  have := ProbabilityTheory.integral_cauchyPDFReal_eq_one 0 (γ := 1) one_ne_zero
  convert this using 2 with x
  simp [ProbabilityTheory.cauchyPDFReal_def]

/-- `∫_0^R x / (1 + x²) dx = log(1 + R²) / 2`, which is unbounded in `R`: the Cauchy variable has no expectation. -/
theorem cauchy_no_mean (R : ℝ) :
    (∫ x in (0 : ℝ)..R, x / (1 + x ^ 2)) = Real.log (1 + R ^ 2) / 2 ∧
      ∀ C : ℝ, ∃ R : ℝ, C < ∫ x in (0 : ℝ)..R, x / (1 + x ^ 2) := by
  have key : ∀ R : ℝ, (∫ x in (0 : ℝ)..R, x / (1 + x ^ 2)) = Real.log (1 + R ^ 2) / 2 := by
    intro R
    have hF : ∀ x : ℝ, HasDerivAt (fun x => Real.log (1 + x ^ 2) / 2) (x / (1 + x ^ 2)) x := by
      intro x
      have h1 : HasDerivAt (fun x : ℝ => 1 + x ^ 2) (2 * x) x := by
        simpa using ((hasDerivAt_pow 2 x).const_add 1)
      have h2 := (h1.log (by positivity)).div_const 2
      convert h2 using 1; field_simp; ring
    rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hF x) (by
      apply Continuous.intervalIntegrable; fun_prop (disch := positivity))]
    simp
  refine ⟨key R, fun C => ?_⟩
  refine ⟨Real.exp (2 * C), ?_⟩
  rw [key]
  have h1 : Real.exp (2 * C) ^ 2 < 1 + Real.exp (2 * C) ^ 2 := by linarith
  have h2 : Real.log (Real.exp (2 * C) ^ 2) < Real.log (1 + Real.exp (2 * C) ^ 2) :=
    Real.log_lt_log (by positivity) h1
  rw [Real.log_pow, Real.log_exp] at h2
  push_cast at h2
  linarith

/-! ## exr-rv-linearity-proof -/

/-- On a finite space `E X = ∑ X(ω) P({ω})`, which is visibly linear. -/
theorem E_eq_sum_points {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X Y : Ω → ℚ) :
    E p X = ∑ ω, X ω * P p {ω} ∧ E p (fun ω => X ω + Y ω) = E p X + E p Y := by
  constructor
  · simp [E, P, mul_comm]
  · have := E_linear p 1 1 X Y; simpa using this

/-! ## exr-rv-st-petersburg -/

/-- `∑_{n≥0} n rⁿ = r / (1 - r)²` for `|r| < 1`. -/
theorem tsum_mul_geometric (r : ℝ) (hr : |r| < 1) : ∑' n : ℕ, (n : ℝ) * r ^ n = r / (1 - r) ^ 2 :=
  tsum_coe_mul_geometric_of_norm_lt_one (by simpa using hr)

/-- `∑_{n≥0} n² rⁿ = r (1 + r) / (1 - r)³` for `|r| < 1`. -/
theorem tsum_sq_mul_geometric (r : ℝ) (hr : |r| < 1) : ∑' n : ℕ, (n : ℝ) ^ 2 * r ^ n = r * (1 + r) / (1 - r) ^ 3 := by
  have hr' : ‖r‖ < 1 := by simpa using hr
  have hs2 : Summable (fun n : ℕ => (n : ℝ) ^ 2 * r ^ n) := summable_pow_mul_geometric_of_norm_lt_one 2 hr'
  have hs1 : Summable (fun n : ℕ => (n : ℝ) * r ^ n) := by
    simpa using summable_pow_mul_geometric_of_norm_lt_one 1 hr'
  have hs0 : Summable (fun n : ℕ => r ^ n) := summable_geometric_of_norm_lt_one hr'
  have hshift := tsum_eq_zero_add hs2
  simp only [Nat.cast_zero, zero_pow two_ne_zero, zero_mul, zero_add] at hshift
  have hexp : ∀ n : ℕ, ((n + 1 : ℕ) : ℝ) ^ 2 * r ^ (n + 1) = r * ((n : ℝ) ^ 2 * r ^ n) + 2 * r * ((n : ℝ) * r ^ n) +
      r * r ^ n := by intro n; push_cast; ring
  simp_rw [hexp] at hshift
  rw [tsum_add (hs2.mul_left r |>.add (hs1.mul_left (2 * r))) (hs0.mul_left r), tsum_add (hs2.mul_left r)
    (hs1.mul_left (2 * r)), tsum_mul_left, tsum_mul_left, tsum_mul_left, tsum_mul_geometric r hr,
    tsum_geometric_of_norm_lt_one hr'] at hshift
  have h1 : (1 : ℝ) - r ≠ 0 := by
    have := abs_lt.1 hr; intro h; linarith
  have : (∑' n : ℕ, (n : ℝ) ^ 2 * r ^ n) * (1 - r) = r * (1 + r) / (1 - r) ^ 2 := by
    field_simp at hshift ⊢; linarith
  rw [← this]; field_simp; ring

/-- The waiting time `K` for the first head, `P(K = k) = 2^{-k}`: `E K = ∑ k 2^{-k} = 2` and `E K² = 6`. -/
theorem waiting_time_moments : ∑' k : ℕ, ((k + 1 : ℕ) : ℝ) * (1 / 2) ^ (k + 1) = 2 ∧
    ∑' k : ℕ, ((k + 1 : ℕ) : ℝ) ^ 2 * (1 / 2) ^ (k + 1) = 6 := by
  have hr : |(1 / 2 : ℝ)| < 1 := by rw [abs_of_pos (by norm_num)]; norm_num
  have hr' : ‖(1 / 2 : ℝ)‖ < 1 := by simpa using hr
  constructor
  · have h := tsum_eq_zero_add (f := fun n : ℕ => (n : ℝ) * (1 / 2) ^ n)
      (by simpa using summable_pow_mul_geometric_of_norm_lt_one 1 hr')
    rw [tsum_mul_geometric _ hr] at h
    simp only [Nat.cast_zero, zero_mul, zero_add] at h
    rw [← h]; norm_num
  · have h := tsum_eq_zero_add (f := fun n : ℕ => (n : ℝ) ^ 2 * (1 / 2) ^ n)
      (summable_pow_mul_geometric_of_norm_lt_one 2 hr')
    rw [tsum_sq_mul_geometric _ hr] at h
    simp only [Nat.cast_zero, zero_pow two_ne_zero, zero_mul, zero_add] at h
    rw [← h]; norm_num

/-- The payoff `2^K` has infinite expectation: the terms `2^k 2^{-k}` are all `1`, so the series diverges. -/
theorem st_petersburg_diverges : (∀ k : ℕ, (2 : ℝ) ^ (k + 1) * (1 / 2) ^ (k + 1) = 1) ∧
    ¬ Summable (fun k : ℕ => (2 : ℝ) ^ (k + 1) * (1 / 2) ^ (k + 1)) := by
  have h : ∀ k : ℕ, (2 : ℝ) ^ (k + 1) * (1 / 2) ^ (k + 1) = 1 := fun k => by
    rw [← mul_pow]; norm_num
  refine ⟨h, fun hs => ?_⟩
  simp only [h] at hs
  exact one_ne_zero (summable_const_iff.1 hs)

/-- Capped at `2^30` the expected payoff is `30 + 2^30 · ∑_{k > 30} 2^{-k} = 30 + 1 = 31`. -/
theorem st_petersburg_capped : (∑ k ∈ range 30, (2 : ℝ) ^ (k + 1) * (1 / 2) ^ (k + 1)) +
    (2 : ℝ) ^ 30 * ∑' k : ℕ, (1 / 2 : ℝ) ^ (k + 31) = 31 := by
  have h : ∀ k : ℕ, (2 : ℝ) ^ (k + 1) * (1 / 2) ^ (k + 1) = 1 := fun k => by rw [← mul_pow]; norm_num
  simp only [h, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one]
  have : ∑' k : ℕ, (1 / 2 : ℝ) ^ (k + 31) = (1 / 2) ^ 31 * ∑' k : ℕ, (1 / 2 : ℝ) ^ k := by
    rw [← tsum_mul_left]; congr 1; funext k; rw [pow_add]; ring
  rw [this, tsum_geometric_of_lt_one (by norm_num) (by norm_num)]
  norm_num

/-! ## exr-rv-cauchy-no-mean -/

/-- `∫_{-1}^1 dx / (π (1 + x²)) = 1/2`, so `P(|X| > 1) = 1 - 1/2 = 1/2` for a Cauchy variable. -/
theorem cauchy_tail : (∫ x in (-1 : ℝ)..1, 1 / (Real.pi * (1 + x ^ 2))) = 1 / 2 ∧ (1 : ℝ) - 1 / 2 = 1 / 2 := by
  refine ⟨?_, by norm_num⟩
  have : (fun x : ℝ => 1 / (Real.pi * (1 + x ^ 2))) = fun x => Real.pi⁻¹ * (1 + x ^ 2)⁻¹ := by
    funext x; field_simp
  rw [this, intervalIntegral.integral_const_mul, integral_inv_one_add_sq, Real.arctan_one, Real.arctan_neg,
    Real.arctan_one]
  field_simp; ring

/-- For `Θ` uniform on `(-π/2, π/2)`, `P(tan Θ ≤ x) = (arctan x + π/2) / π`. -/
theorem tan_uniform_cdf (x : ℝ) :
    (MeasureTheory.volume {θ ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2) | Real.tan θ ≤ x}).toReal / Real.pi =
      (Real.arctan x + Real.pi / 2) / Real.pi := by
  have hset : {θ ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2) | Real.tan θ ≤ x} =
      Set.Ioc (-(Real.pi / 2)) (Real.arctan x) := by
    ext θ
    simp only [Set.mem_setOf_eq, Set.mem_Ioo, Set.mem_Ioc]
    constructor
    · rintro ⟨⟨h1, h2⟩, h3⟩
      refine ⟨h1, ?_⟩
      rw [← Real.arctan_tan h1 h2]
      exact Real.arctan_le_arctan h3
    · rintro ⟨h1, h2⟩
      have hlt : θ < Real.pi / 2 := h2.trans_lt (Real.arctan_lt_pi_div_two x)
      refine ⟨⟨h1, hlt⟩, ?_⟩
      rw [← Real.tan_arctan x]
      exact Real.strictMonoOn_tan.monotoneOn ⟨h1, hlt⟩ ⟨Real.neg_pi_div_two_lt_arctan x,
        Real.arctan_lt_pi_div_two x⟩ h2
  rw [hset, Real.volume_Ioc, ENNReal.toReal_ofReal (by linarith [Real.neg_pi_div_two_lt_arctan x])]
  ring

/-- The derivative of `(arctan x + π/2) / π` is the Cauchy density `1 / (π (1 + x²))`. -/
theorem cauchy_cdf_deriv (x : ℝ) :
    HasDerivAt (fun x => (Real.arctan x + Real.pi / 2) / Real.pi) (1 / (Real.pi * (1 + x ^ 2))) x := by
  have := ((Real.hasDerivAt_arctan x).add_const (Real.pi / 2)).div_const Real.pi
  convert this using 1
  field_simp

/-- The ratio of two independent standard Gaussians is Cauchy: `P(X / Y ≤ t) = arctan t / π + 1/2` (the event
`{X ≤ tY, Y > 0} ∪ {X ≥ tY, Y < 0}` under the product Gaussian measure). -/
def ratio_gaussians_cauchy : Prop := ∀ t : ℝ,
  ((ProbabilityTheory.gaussianReal 0 1).prod (ProbabilityTheory.gaussianReal 0 1)).real
    ({w : ℝ × ℝ | w.1 ≤ t * w.2 ∧ 0 < w.2} ∪ {w : ℝ × ℝ | t * w.2 ≤ w.1 ∧ w.2 < 0}) = Real.arctan t / Real.pi + 1 / 2

end KaluzaKlein.Probability.RandomVariables
