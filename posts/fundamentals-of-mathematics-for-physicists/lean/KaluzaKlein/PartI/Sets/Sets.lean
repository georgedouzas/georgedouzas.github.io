import Mathlib
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Sets"

The claims of the section "Sets" of `tutorial/part-i-01-sets-relations-topology.qmd`: finite sets and their sizes,
the number systems and their inclusions, the circle, the sphere and `SO(3)` as sets cut out by equations, the
subsets of a three-element set, the strict inclusions `ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ` and the discs, and the slices of the circle
and the sphere by a line and a plane. The geometric sets defined here are used by every later section.
-/

namespace KaluzaKlein.Sets.Sets

open Matrix

/-- The unit circle `S¹ = {(X, Y) : X² + Y² = 1}` in `ℝ²`. -/
def circle : Set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 = 1}

/-- The unit sphere `S² = {(X, Y, Z) : X² + Y² + Z² = 1}` in `ℝ³`. -/
def sphere : Set (Fin 3 → ℝ) := {p | p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 = 1}

/-- The closed unit disc `{X² + Y² ≤ 1}`. -/
def closedDisc : Set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 ≤ 1}

/-- The open unit disc `{X² + Y² < 1}`. -/
def openDisc : Set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 < 1}

/-- The rotation group `SO(3)`, mathlib's special orthogonal group of `3 × 3` real matrices. -/
abbrev SO3 : Set (Matrix (Fin 3) (Fin 3) ℝ) := (Matrix.specialOrthogonalGroup (Fin 3) ℝ : Set _)

/-- The north pole `(0, 0, 1)`. -/
def northPole : Fin 3 → ℝ := ![0, 0, 1]

/-! ## exm-set-finite -/

/-- A set listed by three distinct elements has three elements (`{1, 2, 3}` among the instances). -/
theorem card_three {α : Type} [DecidableEq α] (a b c : α) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    ({a, b, c} : Finset α).card = 3 := by
  rw [Finset.card_insert_of_notMem (by simp [hab, hac]), Finset.card_pair hbc]

example : veriphy_instance% KaluzaKlein.Sets.Sets.card_three (ℕ) (inferInstance) (1) (2) (3) ((by decide)) ((by decide)) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Sets.card_three (Fin 3) (inferInstance) (0) (1) (2) ((by decide)) ((by decide)) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Sets.card_three (ℤ) (inferInstance) (-1) (0) (5) ((by decide)) ((by decide)) ((by decide)) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.card_three
-- test: set-card-three-test.sage

/-- The doubles of `0, …, n - 1` lie in `{0, …, 2n - 1}`: for `n = 3`, `{0, 2, 4} ⊆ {0, 1, 2, 3, 4, 5}`. -/
theorem doubles_subset (n : ℕ) : (Finset.range n).image (2 * ·) ⊆ Finset.range (2 * n) := by
  intro x hx
  simp only [Finset.mem_image, Finset.mem_range] at hx ⊢
  omega

example : veriphy_instance% KaluzaKlein.Sets.Sets.doubles_subset (3) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Sets.doubles_subset (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Sets.doubles_subset (5) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.doubles_subset
-- test: set-doubles-subset-test.sage

/-- The rotations of the plane by the multiples of `360°/n`, the `n`-th roots of unity, are `n` in number: six for
the hexagon. -/
theorem card_rotations (n : ℕ) [NeZero n] : Nat.card (rootsOfUnity n ℂ) = n :=
  Complex.card_rootsOfUnity n

example : veriphy_instance% KaluzaKlein.Sets.Sets.card_rotations (6) (inferInstance) := by
  exact Complex.card_rootsOfUnity 6

example : veriphy_instance% KaluzaKlein.Sets.Sets.card_rotations (1) (inferInstance) := by
  exact Complex.card_rootsOfUnity 1

example : veriphy_instance% KaluzaKlein.Sets.Sets.card_rotations (4) (inferInstance) := by
  exact Complex.card_rootsOfUnity 4

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.card_rotations
-- test: set-card-rotations-test.sage

/-! ## exm-set-numbers -/

/-- A ring of characteristic zero (`ℤ`, `ℝ`, `ℂ`) is infinite. -/
theorem infinite_of_charZero (K : Type) [AddMonoidWithOne K] [CharZero K] : Infinite K :=
  Infinite.of_injective Nat.cast Nat.cast_injective

example : veriphy_instance% KaluzaKlein.Sets.Sets.infinite_of_charZero (ℤ) (inferInstance) (inferInstance) := by
  infer_instance

example : veriphy_instance% KaluzaKlein.Sets.Sets.infinite_of_charZero (ℝ) (inferInstance) (inferInstance) := by
  exact Infinite.of_injective Nat.cast Nat.cast_injective

example : veriphy_instance% KaluzaKlein.Sets.Sets.infinite_of_charZero (ℂ) (inferInstance) (inferInstance) := by
  exact Infinite.of_injective Nat.cast Nat.cast_injective

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.infinite_of_charZero
-- test: set-infinite-charzero-test.sage

/-- `ℝⁿ` is infinite for `n ≥ 1`. -/
theorem infinite_rn (n : ℕ) (hn : 1 ≤ n) : Infinite (Fin n → ℝ) := by
  haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
  infer_instance

example : veriphy_instance% KaluzaKlein.Sets.Sets.infinite_rn (1) ((by norm_num)) := by
  infer_instance

example : veriphy_instance% KaluzaKlein.Sets.Sets.infinite_rn (2) ((by norm_num)) := by
  infer_instance

example : veriphy_instance% KaluzaKlein.Sets.Sets.infinite_rn (4) ((by norm_num)) := by
  infer_instance

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.infinite_rn
-- test: set-infinite-rn-test.sage

/-- `ℤ ⊆ ℝ ⊆ ℂ`: the inclusions are injective (distinct integers are distinct reals, distinct reals distinct complex
numbers). -/
theorem int_real_complex_inclusions (a b : ℤ) (x y : ℝ) :
    ((a : ℝ) = b → a = b) ∧ ((x : ℂ) = y → x = y) :=
  ⟨fun h => by exact_mod_cast h, fun h => by exact_mod_cast h⟩

example : veriphy_instance% KaluzaKlein.Sets.Sets.int_real_complex_inclusions (0) (0) (0) (0) := by
  have key : ∀ (a b : ℤ) (x y : ℝ), ((a : ℝ) = b → a = b) ∧ ((x : ℂ) = y → x = y) := by
    intro a b x y
    exact (⟨fun h => by exact_mod_cast h, fun h => by exact_mod_cast h⟩)
  exact @key (0) (0) (0) (0)

example : veriphy_instance% KaluzaKlein.Sets.Sets.int_real_complex_inclusions (1) (-2) (1/2) (3) := by
  have key : ∀ (a b : ℤ) (x y : ℝ), ((a : ℝ) = b → a = b) ∧ ((x : ℂ) = y → x = y) := by
    intro a b x y
    exact (⟨fun h => by exact_mod_cast h, fun h => by exact_mod_cast h⟩)
  exact @key (1) (-2) (1/2) (3)

example : veriphy_instance% KaluzaKlein.Sets.Sets.int_real_complex_inclusions (5) (5) (-1) (-1) := by
  have key : ∀ (a b : ℤ) (x y : ℝ), ((a : ℝ) = b → a = b) ∧ ((x : ℂ) = y → x = y) := by
    intro a b x y
    exact (⟨fun h => by exact_mod_cast h, fun h => by exact_mod_cast h⟩)
  exact @key (5) (5) (-1) (-1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.int_real_complex_inclusions
-- test: set-inclusions-test.sage

/-! ## exm-set-geometric -/

/-- `SO(3)` is cut out of the `3 × 3` matrices by `Rᵀ R = 1` and `det R = 1`. -/
theorem mem_SO3_iff (R : Matrix (Fin 3) (Fin 3) ℝ) : R ∈ SO3 ↔ Rᵀ * R = 1 ∧ R.det = 1 := by
  rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']

example : veriphy_instance% KaluzaKlein.Sets.Sets.mem_SO3_iff (1) := by
  have key : ∀ (R : Matrix (Fin 3) (Fin 3) ℝ), R ∈ SO3 ↔ Rᵀ * R = 1 ∧ R.det = 1 := by
    intro R
    rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Sets.mem_SO3_iff (!![1, 0, 0; 0, -1, 0; 0, 0, -1]) := by
  have key : ∀ (R : Matrix (Fin 3) (Fin 3) ℝ), R ∈ SO3 ↔ Rᵀ * R = 1 ∧ R.det = 1 := by
    intro R
    rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
  exact @key (!![1, 0, 0; 0, -1, 0; 0, 0, -1])

example : veriphy_instance% KaluzaKlein.Sets.Sets.mem_SO3_iff (!![2, 0, 0; 0, 1, 0; 0, 0, 1]) := by
  have key : ∀ (R : Matrix (Fin 3) (Fin 3) ℝ), R ∈ SO3 ↔ Rᵀ * R = 1 ∧ R.det = 1 := by
    intro R
    rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']
  exact @key (!![2, 0, 0; 0, 1, 0; 0, 0, 1])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.mem_SO3_iff
-- test: set-mem-so3-test.sage

/-- The circle and the sphere are cut out by one equation each (membership is the equation). -/
theorem mem_circle_sphere (X Y Z : ℝ) :
    ((X, Y) ∈ circle ↔ X ^ 2 + Y ^ 2 = 1) ∧ (![X, Y, Z] ∈ sphere ↔ X ^ 2 + Y ^ 2 + Z ^ 2 = 1) := by
  simp [circle, sphere]

example : veriphy_instance% KaluzaKlein.Sets.Sets.mem_circle_sphere (1) (0) (0) := by
  have key : ∀ (X Y Z : ℝ), ((X, Y) ∈ circle ↔ X ^ 2 + Y ^ 2 = 1) ∧ (![X, Y, Z] ∈ sphere ↔ X ^ 2 + Y ^ 2 + Z ^ 2 = 1) := by
    intro X Y Z
    simp [circle, sphere]
  exact @key (1) (0) (0)

example : veriphy_instance% KaluzaKlein.Sets.Sets.mem_circle_sphere (3/5) (4/5) (0) := by
  have key : ∀ (X Y Z : ℝ), ((X, Y) ∈ circle ↔ X ^ 2 + Y ^ 2 = 1) ∧ (![X, Y, Z] ∈ sphere ↔ X ^ 2 + Y ^ 2 + Z ^ 2 = 1) := by
    intro X Y Z
    simp [circle, sphere]
  exact @key (3/5) (4/5) (0)

example : veriphy_instance% KaluzaKlein.Sets.Sets.mem_circle_sphere (1) (1) (1) := by
  have key : ∀ (X Y Z : ℝ), ((X, Y) ∈ circle ↔ X ^ 2 + Y ^ 2 = 1) ∧ (![X, Y, Z] ∈ sphere ↔ X ^ 2 + Y ^ 2 + Z ^ 2 = 1) := by
    intro X Y Z
    simp [circle, sphere]
  exact @key (1) (1) (1)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.mem_circle_sphere
-- test: set-mem-circle-sphere-test.sage

/-! ## exr-set-subsets -/

/-- A set with `k` elements has `2ᵏ` subsets: `2³ = 8` for `{a, b, c}`. -/
theorem card_powerset {α : Type} (s : Finset α) : s.powerset.card = 2 ^ s.card :=
  Finset.card_powerset s

example : veriphy_instance% KaluzaKlein.Sets.Sets.card_powerset (Fin 3) (Finset.univ) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Sets.card_powerset (ℕ) (∅) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Sets.card_powerset (ℕ) ({1, 2, 3, 4}) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.card_powerset
-- test: set-card-powerset-test.sage

/-! ## exr-set-inclusions -/

/-- A negative integer is not a natural number (`-1 ∉ ℕ`): `ℕ ⊊ ℤ`. -/
theorem neg_not_nat (z : ℤ) (hz : z < 0) : ¬ ∃ n : ℕ, (n : ℤ) = z := by
  rintro ⟨n, rfl⟩
  omega

example : veriphy_instance% KaluzaKlein.Sets.Sets.neg_not_nat (-1) ((by norm_num)) := by
  have key : ∀ (z : ℤ) (hz : z < 0), ¬ ∃ n : ℕ, (n : ℤ) = z := by
    intro z hz
    rintro ⟨n, rfl⟩
    omega
  exact @key (-1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Sets.neg_not_nat (-5) ((by norm_num)) := by
  have key : ∀ (z : ℤ) (hz : z < 0), ¬ ∃ n : ℕ, (n : ℤ) = z := by
    intro z hz
    rintro ⟨n, rfl⟩
    omega
  exact @key (-5) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Sets.neg_not_nat (-100) ((by norm_num)) := by
  have key : ∀ (z : ℤ) (hz : z < 0), ¬ ∃ n : ℕ, (n : ℤ) = z := by
    intro z hz
    rintro ⟨n, rfl⟩
    omega
  exact @key (-100) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.neg_not_nat
-- test: set-neg-not-nat-test.sage

/-- A half-odd number is not an integer (`½ ∉ ℤ`): `ℤ ⊊ ℚ`. -/
theorem half_odd_not_int (m : ℤ) : ¬ ∃ k : ℤ, (k : ℚ) = (2 * m + 1) / 2 := by
  rintro ⟨k, hk⟩
  have h : ((2 * k : ℤ) : ℚ) = ((2 * m + 1 : ℤ) : ℚ) := by push_cast; linarith
  have h' : 2 * k = 2 * m + 1 := by exact_mod_cast h
  omega

example : veriphy_instance% KaluzaKlein.Sets.Sets.half_odd_not_int (0) := by
  have key : ∀ (m : ℤ), ¬ ∃ k : ℤ, (k : ℚ) = (2 * m + 1) / 2 := by
    intro m
    rintro ⟨k, hk⟩
    have h : ((2 * k : ℤ) : ℚ) = ((2 * m + 1 : ℤ) : ℚ) := by push_cast; linarith
    have h' : 2 * k = 2 * m + 1 := by exact_mod_cast h
    omega
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Sets.half_odd_not_int (1) := by
  have key : ∀ (m : ℤ), ¬ ∃ k : ℤ, (k : ℚ) = (2 * m + 1) / 2 := by
    intro m
    rintro ⟨k, hk⟩
    have h : ((2 * k : ℤ) : ℚ) = ((2 * m + 1 : ℤ) : ℚ) := by push_cast; linarith
    have h' : 2 * k = 2 * m + 1 := by exact_mod_cast h
    omega
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Sets.half_odd_not_int (-3) := by
  have key : ∀ (m : ℤ), ¬ ∃ k : ℤ, (k : ℚ) = (2 * m + 1) / 2 := by
    intro m
    rintro ⟨k, hk⟩
    have h : ((2 * k : ℤ) : ℚ) = ((2 * m + 1 : ℤ) : ℚ) := by push_cast; linarith
    have h' : 2 * k = 2 * m + 1 := by exact_mod_cast h
    omega
  exact @key (-3)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.half_odd_not_int
-- test: set-half-odd-test.sage

/-- The square root of a prime is irrational (`√2 ∉ ℚ`): `ℚ ⊊ ℝ`. -/
theorem sqrt_prime_irrational (p : ℕ) (hp : p.Prime) : Irrational (Real.sqrt p) :=
  hp.irrational_sqrt

example : veriphy_instance% KaluzaKlein.Sets.Sets.sqrt_prime_irrational (2) ((by norm_num)) := by
  exact Nat.prime_two.irrational_sqrt

example : veriphy_instance% KaluzaKlein.Sets.Sets.sqrt_prime_irrational (3) ((by norm_num)) := by
  exact Nat.prime_three.irrational_sqrt

example : veriphy_instance% KaluzaKlein.Sets.Sets.sqrt_prime_irrational (5) ((by norm_num)) := by
  exact Nat.prime_five.irrational_sqrt

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.sqrt_prime_irrational
-- test: set-sqrt-prime-test.sage

/-- A non-zero multiple of `i` is not real (`i ∉ ℝ`): `ℝ ⊊ ℂ`. -/
theorem imag_not_real (y : ℝ) (hy : y ≠ 0) : ¬ ∃ x : ℝ, (x : ℂ) = y * Complex.I := by
  rintro ⟨x, hx⟩
  have := congrArg Complex.im hx
  simp at this
  exact hy this.symm

example : veriphy_instance% KaluzaKlein.Sets.Sets.imag_not_real (1) ((by norm_num)) := by
  have key : ∀ (y : ℝ) (hy : y ≠ 0), ¬ ∃ x : ℝ, (x : ℂ) = y * Complex.I := by
    intro y hy
    rintro ⟨x, hx⟩
    have := congrArg Complex.im hx
    simp at this
    exact hy this.symm
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Sets.imag_not_real (-2) ((by norm_num)) := by
  have key : ∀ (y : ℝ) (hy : y ≠ 0), ¬ ∃ x : ℝ, (x : ℂ) = y * Complex.I := by
    intro y hy
    rintro ⟨x, hx⟩
    have := congrArg Complex.im hx
    simp at this
    exact hy this.symm
  exact @key (-2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Sets.imag_not_real (1/2) ((by norm_num)) := by
  have key : ∀ (y : ℝ) (hy : y ≠ 0), ¬ ∃ x : ℝ, (x : ℂ) = y * Complex.I := by
    intro y hy
    rintro ⟨x, hx⟩
    have := congrArg Complex.im hx
    simp at this
    exact hy this.symm
  exact @key (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.imag_not_real
-- test: set-imag-not-real-test.sage

/-- For the discs and circle of radius `√r`: the open disc lies in the closed one, the circle lies in the closed disc
and misses the open one (`r = 1` for `S¹ ⊆ D`, `S¹ ∩ {X² + Y² < 1} = ∅`). -/
theorem disc_inclusions (r : ℝ) :
    {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧
      {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧
      {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ∩ {p | p.1 ^ 2 + p.2 ^ 2 < r} = ∅ := by
  refine ⟨fun p (hp : _ < r) => (le_of_lt hp : _ ≤ r), fun p (hp : _ = r) => (le_of_eq hp : _ ≤ r), ?_⟩
  ext p
  simp only [Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_and]
  intro h1 h2
  linarith

example : veriphy_instance% KaluzaKlein.Sets.Sets.disc_inclusions (1) := by
  have key : ∀ (r : ℝ), {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ∩ {p | p.1 ^ 2 + p.2 ^ 2 < r} = ∅ := by
    intro r
    refine ⟨fun p (hp : _ < r) => (le_of_lt hp : _ ≤ r), fun p (hp : _ = r) => (le_of_eq hp : _ ≤ r), ?_⟩
    ext p
    simp only [Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_and]
    intro h1 h2
    linarith
  exact @key (1)

example : veriphy_instance% KaluzaKlein.Sets.Sets.disc_inclusions (0) := by
  have key : ∀ (r : ℝ), {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ∩ {p | p.1 ^ 2 + p.2 ^ 2 < r} = ∅ := by
    intro r
    refine ⟨fun p (hp : _ < r) => (le_of_lt hp : _ ≤ r), fun p (hp : _ = r) => (le_of_eq hp : _ ≤ r), ?_⟩
    ext p
    simp only [Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_and]
    intro h1 h2
    linarith
  exact @key (0)

example : veriphy_instance% KaluzaKlein.Sets.Sets.disc_inclusions (4) := by
  have key : ∀ (r : ℝ), {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ∩ {p | p.1 ^ 2 + p.2 ^ 2 < r} = ∅ := by
    intro r
    refine ⟨fun p (hp : _ < r) => (le_of_lt hp : _ ≤ r), fun p (hp : _ = r) => (le_of_eq hp : _ ≤ r), ?_⟩
    ext p
    simp only [Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_and]
    intro h1 h2
    linarith
  exact @key (4)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.disc_inclusions
-- test: set-disc-inclusions-test.sage

/-! ## exr-set-circle-line -/

/-- The circle meets the line `Y = c` (`|c| < 1`) in the two points `(±√(1 - c²), c)`: for `c = ½`, `(±√3/2, ½)`. -/
theorem circle_slice (c : ℝ) (hc : c ^ 2 < 1) :
    circle ∩ {p | p.2 = c} = {(Real.sqrt (1 - c ^ 2), c), (-Real.sqrt (1 - c ^ 2), c)} := by
  have hs : Real.sqrt (1 - c ^ 2) ^ 2 = 1 - c ^ 2 := Real.sq_sqrt (by linarith)
  ext ⟨x, y⟩
  simp only [circle, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff,
    Prod.mk.injEq]
  constructor
  · rintro ⟨h1, rfl⟩
    have : (x - Real.sqrt (1 - y ^ 2)) * (x + Real.sqrt (1 - y ^ 2)) = 0 := by nlinarith
    rcases mul_eq_zero.1 this with h | h
    · left; exact ⟨by linarith, rfl⟩
    · right; exact ⟨by linarith, rfl⟩
  · rintro (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
    · exact ⟨by linarith, rfl⟩
    · exact ⟨by nlinarith, rfl⟩

example : veriphy_instance% KaluzaKlein.Sets.Sets.circle_slice (1/2) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : c ^ 2 < 1), circle ∩ {p | p.2 = c} = {(Real.sqrt (1 - c ^ 2), c), (-Real.sqrt (1 - c ^ 2), c)} := by
    intro c hc
    have hs : Real.sqrt (1 - c ^ 2) ^ 2 = 1 - c ^ 2 := Real.sq_sqrt (by linarith)
    ext ⟨x, y⟩
    simp only [circle, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff,
      Prod.mk.injEq]
    constructor
    · rintro ⟨h1, rfl⟩
      have : (x - Real.sqrt (1 - y ^ 2)) * (x + Real.sqrt (1 - y ^ 2)) = 0 := by nlinarith
      rcases mul_eq_zero.1 this with h | h
      · left; exact ⟨by linarith, rfl⟩
      · right; exact ⟨by linarith, rfl⟩
    · rintro (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
      · exact ⟨by linarith, rfl⟩
      · exact ⟨by nlinarith, rfl⟩
  exact @key (1/2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Sets.circle_slice (0) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : c ^ 2 < 1), circle ∩ {p | p.2 = c} = {(Real.sqrt (1 - c ^ 2), c), (-Real.sqrt (1 - c ^ 2), c)} := by
    intro c hc
    have hs : Real.sqrt (1 - c ^ 2) ^ 2 = 1 - c ^ 2 := Real.sq_sqrt (by linarith)
    ext ⟨x, y⟩
    simp only [circle, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff,
      Prod.mk.injEq]
    constructor
    · rintro ⟨h1, rfl⟩
      have : (x - Real.sqrt (1 - y ^ 2)) * (x + Real.sqrt (1 - y ^ 2)) = 0 := by nlinarith
      rcases mul_eq_zero.1 this with h | h
      · left; exact ⟨by linarith, rfl⟩
      · right; exact ⟨by linarith, rfl⟩
    · rintro (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
      · exact ⟨by linarith, rfl⟩
      · exact ⟨by nlinarith, rfl⟩
  exact @key (0) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Sets.circle_slice (3/5) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : c ^ 2 < 1), circle ∩ {p | p.2 = c} = {(Real.sqrt (1 - c ^ 2), c), (-Real.sqrt (1 - c ^ 2), c)} := by
    intro c hc
    have hs : Real.sqrt (1 - c ^ 2) ^ 2 = 1 - c ^ 2 := Real.sq_sqrt (by linarith)
    ext ⟨x, y⟩
    simp only [circle, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff,
      Prod.mk.injEq]
    constructor
    · rintro ⟨h1, rfl⟩
      have : (x - Real.sqrt (1 - y ^ 2)) * (x + Real.sqrt (1 - y ^ 2)) = 0 := by nlinarith
      rcases mul_eq_zero.1 this with h | h
      · left; exact ⟨by linarith, rfl⟩
      · right; exact ⟨by linarith, rfl⟩
    · rintro (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
      · exact ⟨by linarith, rfl⟩
      · exact ⟨by nlinarith, rfl⟩
  exact @key (3/5) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.circle_slice
-- test: set-circle-slice-test.sage

/-- `√(1 - (½)²) = √3/2`: the two points are `(±√3/2, ½)`. -/
theorem sqrt_three_quarters : Real.sqrt (1 - (1 / 2 : ℝ) ^ 2) = Real.sqrt 3 / 2 := by
  veriphy_certified
    (rw [show (1 : ℝ) - (1 / 2) ^ 2 = 3 / 2 ^ 2 by norm_num, Real.sqrt_div' _ (by norm_num),
        Real.sqrt_sq (by norm_num)])
  -- certificate: set-sqrt-three-quarters.sage

/-- The sphere meets the plane `Z = c` (`|c| < 1`) in a circle of radius `√(1 - c²)`, a set with infinitely many
elements: for `c = ½`, the circle `X² + Y² = ¾`. -/
theorem sphere_slice (c : ℝ) (hc : c ^ 2 < 1) :
    sphere ∩ {p | p 2 = c} = {p | p 0 ^ 2 + p 1 ^ 2 = 1 - c ^ 2 ∧ p 2 = c} ∧
      (sphere ∩ {p | p 2 = c}).Infinite := by
  have hEq : sphere ∩ {p | p 2 = c} = {p | p 0 ^ 2 + p 1 ^ 2 = 1 - c ^ 2 ∧ p 2 = c} := by
    ext p
    simp only [sphere, Set.mem_inter_iff, Set.mem_setOf_eq]
    constructor
    · rintro ⟨h1, h2⟩; exact ⟨by rw [h2] at h1; linarith, h2⟩
    · rintro ⟨h1, h2⟩; exact ⟨by rw [h2]; linarith, h2⟩
  refine ⟨hEq, ?_⟩
  set r := Real.sqrt (1 - c ^ 2)
  have hr : 0 < r := Real.sqrt_pos.2 (by linarith)
  have hr2 : r ^ 2 = 1 - c ^ 2 := Real.sq_sqrt (by linarith)
  let f : ℝ → Fin 3 → ℝ := fun t => ![r * Real.cos t, r * Real.sin t, c]
  have hf : Set.InjOn f (Set.Ioo 0 Real.pi) := by
    intro s hs t ht hst
    have h0 := congrFun hst 0
    simp only [f, Matrix.cons_val_zero] at h0
    have hcos : Real.cos s = Real.cos t := by
      have := mul_left_cancel₀ hr.ne' h0; exact this
    exact Real.injOn_cos ⟨hs.1.le, hs.2.le⟩ ⟨ht.1.le, ht.2.le⟩ hcos
  have himg : f '' Set.Ioo 0 Real.pi ⊆ sphere ∩ {p | p 2 = c} := by
    rintro _ ⟨t, -, rfl⟩
    rw [hEq]
    refine ⟨?_, by simp [f]⟩
    simp only [f, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    nlinarith [Real.sin_sq_add_cos_sq t]
  exact Set.Infinite.mono himg ((Set.Ioo_infinite Real.pi_pos).image hf)

example : veriphy_instance% KaluzaKlein.Sets.Sets.sphere_slice (1/2) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : c ^ 2 < 1), sphere ∩ {p | p 2 = c} = {p | p 0 ^ 2 + p 1 ^ 2 = 1 - c ^ 2 ∧ p 2 = c} ∧ (sphere ∩ {p | p 2 = c}).Infinite := by
    intro c hc
    have hEq : sphere ∩ {p | p 2 = c} = {p | p 0 ^ 2 + p 1 ^ 2 = 1 - c ^ 2 ∧ p 2 = c} := by
      ext p
      simp only [sphere, Set.mem_inter_iff, Set.mem_setOf_eq]
      constructor
      · rintro ⟨h1, h2⟩; exact ⟨by rw [h2] at h1; linarith, h2⟩
      · rintro ⟨h1, h2⟩; exact ⟨by rw [h2]; linarith, h2⟩
    refine ⟨hEq, ?_⟩
    set r := Real.sqrt (1 - c ^ 2)
    have hr : 0 < r := Real.sqrt_pos.2 (by linarith)
    have hr2 : r ^ 2 = 1 - c ^ 2 := Real.sq_sqrt (by linarith)
    let f : ℝ → Fin 3 → ℝ := fun t => ![r * Real.cos t, r * Real.sin t, c]
    have hf : Set.InjOn f (Set.Ioo 0 Real.pi) := by
      intro s hs t ht hst
      have h0 := congrFun hst 0
      simp only [f, Matrix.cons_val_zero] at h0
      have hcos : Real.cos s = Real.cos t := by
        have := mul_left_cancel₀ hr.ne' h0; exact this
      exact Real.injOn_cos ⟨hs.1.le, hs.2.le⟩ ⟨ht.1.le, ht.2.le⟩ hcos
    have himg : f '' Set.Ioo 0 Real.pi ⊆ sphere ∩ {p | p 2 = c} := by
      rintro _ ⟨t, -, rfl⟩
      rw [hEq]
      refine ⟨?_, by simp [f]⟩
      simp only [f, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
      nlinarith [Real.sin_sq_add_cos_sq t]
    exact Set.Infinite.mono himg ((Set.Ioo_infinite Real.pi_pos).image hf)
  exact @key (1/2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Sets.sphere_slice (0) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : c ^ 2 < 1), sphere ∩ {p | p 2 = c} = {p | p 0 ^ 2 + p 1 ^ 2 = 1 - c ^ 2 ∧ p 2 = c} ∧ (sphere ∩ {p | p 2 = c}).Infinite := by
    intro c hc
    have hEq : sphere ∩ {p | p 2 = c} = {p | p 0 ^ 2 + p 1 ^ 2 = 1 - c ^ 2 ∧ p 2 = c} := by
      ext p
      simp only [sphere, Set.mem_inter_iff, Set.mem_setOf_eq]
      constructor
      · rintro ⟨h1, h2⟩; exact ⟨by rw [h2] at h1; linarith, h2⟩
      · rintro ⟨h1, h2⟩; exact ⟨by rw [h2]; linarith, h2⟩
    refine ⟨hEq, ?_⟩
    set r := Real.sqrt (1 - c ^ 2)
    have hr : 0 < r := Real.sqrt_pos.2 (by linarith)
    have hr2 : r ^ 2 = 1 - c ^ 2 := Real.sq_sqrt (by linarith)
    let f : ℝ → Fin 3 → ℝ := fun t => ![r * Real.cos t, r * Real.sin t, c]
    have hf : Set.InjOn f (Set.Ioo 0 Real.pi) := by
      intro s hs t ht hst
      have h0 := congrFun hst 0
      simp only [f, Matrix.cons_val_zero] at h0
      have hcos : Real.cos s = Real.cos t := by
        have := mul_left_cancel₀ hr.ne' h0; exact this
      exact Real.injOn_cos ⟨hs.1.le, hs.2.le⟩ ⟨ht.1.le, ht.2.le⟩ hcos
    have himg : f '' Set.Ioo 0 Real.pi ⊆ sphere ∩ {p | p 2 = c} := by
      rintro _ ⟨t, -, rfl⟩
      rw [hEq]
      refine ⟨?_, by simp [f]⟩
      simp only [f, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
      nlinarith [Real.sin_sq_add_cos_sq t]
    exact Set.Infinite.mono himg ((Set.Ioo_infinite Real.pi_pos).image hf)
  exact @key (0) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Sets.sphere_slice (-3/5) ((by norm_num)) := by
  have key : ∀ (c : ℝ) (hc : c ^ 2 < 1), sphere ∩ {p | p 2 = c} = {p | p 0 ^ 2 + p 1 ^ 2 = 1 - c ^ 2 ∧ p 2 = c} ∧ (sphere ∩ {p | p 2 = c}).Infinite := by
    intro c hc
    have hEq : sphere ∩ {p | p 2 = c} = {p | p 0 ^ 2 + p 1 ^ 2 = 1 - c ^ 2 ∧ p 2 = c} := by
      ext p
      simp only [sphere, Set.mem_inter_iff, Set.mem_setOf_eq]
      constructor
      · rintro ⟨h1, h2⟩; exact ⟨by rw [h2] at h1; linarith, h2⟩
      · rintro ⟨h1, h2⟩; exact ⟨by rw [h2]; linarith, h2⟩
    refine ⟨hEq, ?_⟩
    set r := Real.sqrt (1 - c ^ 2)
    have hr : 0 < r := Real.sqrt_pos.2 (by linarith)
    have hr2 : r ^ 2 = 1 - c ^ 2 := Real.sq_sqrt (by linarith)
    let f : ℝ → Fin 3 → ℝ := fun t => ![r * Real.cos t, r * Real.sin t, c]
    have hf : Set.InjOn f (Set.Ioo 0 Real.pi) := by
      intro s hs t ht hst
      have h0 := congrFun hst 0
      simp only [f, Matrix.cons_val_zero] at h0
      have hcos : Real.cos s = Real.cos t := by
        have := mul_left_cancel₀ hr.ne' h0; exact this
      exact Real.injOn_cos ⟨hs.1.le, hs.2.le⟩ ⟨ht.1.le, ht.2.le⟩ hcos
    have himg : f '' Set.Ioo 0 Real.pi ⊆ sphere ∩ {p | p 2 = c} := by
      rintro _ ⟨t, -, rfl⟩
      rw [hEq]
      refine ⟨?_, by simp [f]⟩
      simp only [f, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
      nlinarith [Real.sin_sq_add_cos_sq t]
    exact Set.Infinite.mono himg ((Set.Ioo_infinite Real.pi_pos).image hf)
  exact @key (-3/5) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Sets.sphere_slice
-- test: set-sphere-slice-test.sage

end KaluzaKlein.Sets.Sets
