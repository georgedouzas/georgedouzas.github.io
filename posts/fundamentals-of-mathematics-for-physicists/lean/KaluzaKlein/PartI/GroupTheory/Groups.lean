import KaluzaKlein.PartI.GroupTheory.Basic

/-!
# Group theory, section "Groups"

The claims of the section "Groups" of `tutorial/part-i-10-group-theory.qmd`: the integers modulo `n`, the symmetric
group, the matrix groups and the circle group as groups; the axioms and their failures (subtraction, the zero
matrix, the orthogonal group `O(2)`); the units modulo `5` and `6`; uniqueness of identity and inverses; abelian
and non-abelian groups with the counts of commuting pairs; orders of elements in `S₃`, `ℤ₆`, `ℤ₁₂`, `SO(2)` and
`SU(2)`, and the order of a power.
-/

namespace KaluzaKlein.GroupTheory.Groups

open Equiv Matrix KaluzaKlein.GroupTheory.Basic Real

/-! ## exm-group-zn -/

/-- In `ℤₙ` the inverse of `k` is `n - k`: `k + (n - k) = 0`. -/
theorem zmod_inverse (n : ℕ) [NeZero n] (k : ZMod n) : k + ((n - k.val : ℕ) : ZMod n) = 0 := by
  rw [Nat.cast_sub k.val_lt.le, ZMod.natCast_self, ZMod.natCast_zmod_val]; ring

/-- Rotations of the plane compose by adding angles: `R α R β = R (α + β)`. -/
theorem R_mul (α β : ℝ) : R α * R β = R (α + β) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [R, Matrix.mul_apply, Fin.sum_univ_succ, Real.cos_add, Real.sin_add] <;> ring

/-- `R` is `2π`-periodic: `R (θ + q · 2π) = R θ`. -/
theorem R_add_nat_mul_two_pi (θ : ℝ) (q : ℕ) : R (θ + q * (2 * π)) = R θ := by
  simp [R, Real.cos_add_nat_mul_two_pi, Real.sin_add_nat_mul_two_pi]

/-- `R (2πn/m)` depends on `n` modulo `m` only. -/
theorem R_nat_mod (n m : ℕ) (hm : 0 < m) : R (2 * π * n / m) = R (2 * π * (n % m) / m) := by
  conv_lhs => rw [← Nat.mod_add_div n m]
  rw [← R_add_nat_mul_two_pi (2 * π * (n % m) / m) (n / m)]
  congr 1; push_cast; field_simp; ring

/-- The rotation of the hexagon by `k · 60°`, for `k ∈ ℤ₆`. -/
noncomputable def hexRot (k : ZMod 6) : Matrix (Fin 2) (Fin 2) ℝ := R (2 * π * k.val / 6)

/-- The rotations of the hexagon multiply as `ℤ₆` adds: `hexRot (j + k) = hexRot j · hexRot k`. -/
theorem hexRot_add (j k : ZMod 6) : hexRot (j + k) = hexRot j * hexRot k := by
  unfold hexRot
  rw [R_mul, ZMod.val_add, ← R_nat_mod _ _ (by norm_num)]
  congr 1; push_cast; ring

/-- The six rotations of the hexagon are distinct: the multiplication tables of `ℤ₆` and of the rotations agree
element by element (stated; the six matrices are compared by the oracle). -/
def HexRotFaithful : Prop := ∀ j k : ZMod 6, hexRot j = hexRot k → j = k

/-! ## exm-group-s3 -/

/-- The permutations of `n` objects satisfy the group axioms under composition. -/
theorem perm_group_axioms (n : ℕ) (g h k : Perm (Fin n)) :
    g * h * k = g * (h * k) ∧ 1 * g = g ∧ g * 1 = g ∧ g * g⁻¹ = 1 ∧ g⁻¹ * g = 1 := by
  simp [mul_assoc]

/-- `Sₙ` has `n!` elements (`6` for `n = 3`, `24` for `n = 4`). -/
theorem card_perm (n : ℕ) : Fintype.card (Perm (Fin n)) = n ! := by
  simp [Fintype.card_perm]

/-! ## exm-group-matrix -/

/-- An invertible matrix has a two-sided inverse: `GL(n)` is a group under matrix multiplication. -/
theorem gl_inv (n : ℕ) (A : Matrix (Fin n) (Fin n) ℚ) (hA : A.det ≠ 0) : ∃ B, A * B = 1 ∧ B * A = 1 :=
  ⟨A⁻¹, Matrix.mul_nonsing_inv A (isUnit_iff_ne_zero.mpr hA), Matrix.nonsing_inv_mul A (isUnit_iff_ne_zero.mpr hA)⟩

/-- A rotation is orthogonal with determinant `1`: `R θ ∈ SO(2)`. -/
theorem R_orthogonal (θ : ℝ) : R θ * (R θ)ᵀ = 1 ∧ (R θ).det = 1 := by
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [R, Matrix.mul_apply, Fin.sum_univ_succ] <;>
      linear_combination Real.cos_sq_add_sin_sq θ
  · simp [R, Matrix.det_fin_two]
    linear_combination Real.cos_sq_add_sin_sq θ

/-- `SO(n)` is closed under products and inverses (the inverse being the transpose). -/
theorem so_closed (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℚ) (hA : Aᵀ * A = 1 ∧ A.det = 1) (hB : Bᵀ * B = 1 ∧ B.det = 1) :
    (A * B)ᵀ * (A * B) = 1 ∧ (A * B).det = 1 ∧ Aᵀᵀ * Aᵀ = 1 ∧ Aᵀ.det = 1 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Aᵀ, hA.1, Matrix.one_mul, hB.1]
  · rw [Matrix.det_mul, hA.2, hB.2, one_mul]
  · rw [Matrix.transpose_transpose]; exact Matrix.mul_eq_one_comm.mp hA.1
  · rw [Matrix.det_transpose]; exact hA.2

/-- The `SU(2)` matrices `!![a, b; -b̄, ā]` multiply to a matrix of the same form. -/
theorem su2_mul (a b a' b' : ℂ) :
    !![a, b; -star b, star a] * !![a', b'; -star b', star a'] =
      !![a * a' - b * star b', a * b' + b * star a'; -star (a * b' + b * star a'), star (a * a' - b * star b')] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, star_mul', star_star] <;> ring

/-- The determinant of `!![a, b; -b̄, ā]` is `|a|² + |b|²`. -/
theorem su2_det (a b : ℂ) : (!![a, b; -star b, star a]).det = a * star a + b * star b := by
  simp [Matrix.det_fin_two]; ring

/-! ## exr-group-axioms -/

/-- The non-zero reals form an abelian group under multiplication: closed, with inverses, commutative. -/
theorem units_real (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : x * y ≠ 0 ∧ x * x⁻¹ = 1 ∧ x * y = y * x :=
  ⟨mul_ne_zero hx hy, mul_inv_cancel₀ hx, mul_comm x y⟩

/-- Subtraction of integers is not associative: `(a - b) - c ≠ a - (b - c)` whenever `c ≠ 0`. -/
theorem sub_not_assoc (a b c : ℤ) (hc : c ≠ 0) : a - b - c ≠ a - (b - c) := by omega

/-- Subtraction has no identity: `a - e = a` forces `e = 0`, but `0 - a ≠ a` for `a ≠ 0`. -/
theorem sub_no_identity (a e : ℤ) (ha : a ≠ 0) (he : a - e = a) : e = 0 ∧ 0 - a ≠ a := by omega

/-- The zero matrix has no inverse: `0 * B ≠ 1` for every `B`. -/
theorem zero_mul_ne_one (R : Type) [Ring R] [Nontrivial R] (B : Matrix (Fin 2) (Fin 2) R) : 0 * B ≠ 1 := by
  rw [Matrix.zero_mul]; exact zero_ne_one

/-- The reflection of the plane in the line at angle `θ/2`. -/
noncomputable def S (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![Real.cos θ, Real.sin θ; Real.sin θ, -Real.cos θ]

/-- Two reflections compose to a rotation, a rotation and a reflection to a reflection: `O(2)` is closed. -/
theorem o2_products (θ φ : ℝ) : S θ * S φ = R (θ - φ) ∧ R θ * S φ = S (θ + φ) ∧ S φ * R θ = S (φ - θ) := by
  refine ⟨?_, ?_, ?_⟩ <;> ext i j <;> fin_cases i <;> fin_cases j <;>
    simp [R, S, Matrix.mul_apply, Fin.sum_univ_succ, Real.cos_add, Real.sin_add, Real.cos_sub, Real.sin_sub] <;> ring

/-- `O(2)` is not abelian: a rotation and a reflection commute only when `sin θ = 0`. -/
theorem o2_nonabelian (θ : ℝ) (hθ : Real.sin θ ≠ 0) : R θ * S 0 ≠ S 0 * R θ := by
  intro h
  have h01 := congrFun (congrFun h 0) 1
  simp [R, S, Matrix.mul_apply, Fin.sum_univ_succ] at h01
  exact hθ (by linarith)

/-- Conjugating a rotation by a reflection inverts it: `SO(2)` is normal in `O(2)`. -/
theorem S_conj_R (θ φ : ℝ) : S φ * R θ * S φ = R (-θ) := by
  rw [(o2_products θ φ).2.2, (o2_products _ _).1]; congr 1; ring

/-- An orthogonal matrix has determinant `±1`: `O(n)` has two cosets of `SO(n)`. -/
theorem o_det (n : ℕ) (A : Matrix (Fin n) (Fin n) ℚ) (h : Aᵀ * A = 1) : A.det = 1 ∨ A.det = -1 := by
  have h2 : A.det * A.det = 1 := by rw [← Matrix.det_transpose A, Matrix.det_transpose, ← Matrix.det_mul, h, Matrix.det_one]
  rcases mul_self_eq_one_iff.mp h2 with h1 | h1
  · exact Or.inl h1
  · exact Or.inr h1

/-! ## exr-grp-units-mod5 -/

/-- Modulo a prime the product of two non-zero residues is non-zero. -/
theorem zmod_prime_mul_ne_zero (p : ℕ) [Fact p.Prime] (a b : ZMod p) (ha : a ≠ 0) (hb : b ≠ 0) : a * b ≠ 0 :=
  mul_ne_zero ha hb

/-- The inverses modulo `5`: `1⁻¹ = 1`, `2⁻¹ = 3`, `3⁻¹ = 2`, `4⁻¹ = 4`, and `a a⁻¹ = 1` for `a ≠ 0`. -/
theorem units5_inverse_table (a : ZMod 5) (ha : a ≠ 0) :
    a * a⁻¹ = 1 ∧ (a = 1 → a⁻¹ = 1) ∧ (a = 2 → a⁻¹ = 3) ∧ (a = 3 → a⁻¹ = 2) ∧ (a = 4 → a⁻¹ = 4) := by
  revert a; decide

/-- `a b = 0` in `ℤₙ` whenever `a b = n`: `2 · 3 = 0` modulo `6`. -/
theorem mul_eq_zero_of_mul_eq (n a b : ℕ) (h : a * b = n) : (a : ZMod n) * b = 0 := by
  rw [← Nat.cast_mul, h, ZMod.natCast_self]

/-- Modulo `6`, the residues `2, 3, 4` have no inverse. -/
theorem mod6_no_inverse (a : ZMod 6) (ha : a = 2 ∨ a = 3 ∨ a = 4) : ∀ b, a * b ≠ 1 := by
  revert a; decide

/-- Modulo `6` exactly the residues coprime to `6`, namely `1` and `5`, are invertible. -/
theorem mod6_units (a : ZMod 6) : (∃ b, a * b = 1) ↔ a = 1 ∨ a = 5 := by
  revert a; decide

/-! ## exr-grp-uniqueness -/

/-- A group has only one identity. -/
theorem unique_identity (G : Type) [Group G] (e : G) (h : ∀ g, e * g = g) : e = 1 := by
  simpa using h 1

/-- Each element has only one inverse. -/
theorem unique_inverse (G : Type) [Group G] (g h : G) (hh : g * h = 1) : h = g⁻¹ :=
  eq_inv_of_mul_eq_one_right hh

/-- The inverse of a product reverses the order: `(gh)⁻¹ = h⁻¹ g⁻¹`. -/
theorem inv_of_product (G : Type) [Group G] (g h : G) : (g * h)⁻¹ = h⁻¹ * g⁻¹ :=
  mul_inv_rev g h

/-- `g x = k` has exactly one solution, `x = g⁻¹ k`. -/
theorem unique_solution (G : Type) [Group G] (g k : G) : ∃! x, g * x = k :=
  ⟨g⁻¹ * k, by simp, fun x hx => by rw [← hx]; simp⟩

/-- Every row of a multiplication table contains each element exactly once: `x ↦ g x` is a bijection. -/
theorem row_bijective (G : Type) [Group G] (g : G) : Function.Bijective (g * ·) :=
  (Equiv.mulLeft g).bijective

/-! ## exm-abelian-zn -/

/-- `ℤₙ` is abelian. -/
theorem zmod_add_comm (n : ℕ) (a b : ZMod n) : a + b = b + a := add_comm a b

/-- `SO(2)` is abelian: `R α R β = R β R α`. -/
theorem R_comm (α β : ℝ) : R α * R β = R β * R α := by rw [R_mul, R_mul, add_comm]

/-- The unit complex numbers multiply by adding angles: `e^{iα} e^{iβ} = e^{i(α + β)}`. -/
theorem u1_mul (α β : ℝ) : Complex.exp (α * Complex.I) * Complex.exp (β * Complex.I) = Complex.exp ((α + β) * Complex.I) := by
  rw [← Complex.exp_add]; ring_nf

/-- `e^{iα} ↦ R α` identifies `U(1)` with `SO(2)`: the real and imaginary parts of `e^{iα}` are the entries of `R α`. -/
theorem u1_to_R (α : ℝ) : (Complex.exp (α * Complex.I)).re = R α 0 0 ∧ (Complex.exp (α * Complex.I)).im = R α 1 0 := by
  simp [R, Complex.exp_mul_I]

/-! ## exm-nonabelian-s3 -/

/-- Two transpositions with one common point do not commute: `(i j)(i k) ≠ (i k)(i j)`. -/
theorem swap_swap_ne (n : ℕ) (i j k : Fin n) (hij : i ≠ j) (hik : i ≠ k) (hjk : j ≠ k) :
    swap i j * swap i k ≠ swap i k * swap i j := by
  intro h
  have := congrArg (fun σ => σ j) h
  simp [swap_apply_of_ne_of_ne hjk.symm hij.symm, swap_apply_left, swap_apply_right] at this
  exact hik this

/-- The products `(i j)(i k)` and `(i k)(i j)` are inverse to each other, and `(i j)(i k)` has order `3`:
the two products of two transpositions with a common point are the two three-cycles. -/
theorem swap_swap_cycle (n : ℕ) (i j k : Fin n) (hij : i ≠ j) (hik : i ≠ k) (hjk : j ≠ k) :
    swap i j * swap i k * (swap i k * swap i j) = 1 ∧ orderOf (swap i j * swap i k) = 3 ∧
      (swap i j * swap i k) ^ 2 ≠ 1 := by
  have h3 : (swap i j * swap i k) ^ 3 = 1 := by
    ext x
    simp only [pow_succ, pow_zero, one_mul, Perm.mul_apply, Perm.one_apply]
    by_cases hx : x = i
    · subst hx; simp [swap_apply_left, swap_apply_right, swap_apply_of_ne_of_ne hjk hik.symm]
    by_cases hx' : x = j
    · subst hx'; simp [swap_apply_left, swap_apply_right, swap_apply_of_ne_of_ne hij.symm hjk.symm,
        swap_apply_of_ne_of_ne hik.symm hjk]
    by_cases hx'' : x = k
    · subst hx''; simp [swap_apply_left, swap_apply_right, swap_apply_of_ne_of_ne hjk.symm hij.symm,
        swap_apply_of_ne_of_ne hik hjk]
    · simp [swap_apply_of_ne_of_ne hx hx', swap_apply_of_ne_of_ne hx hx'']
  have hne : swap i j * swap i k ≠ 1 := by
    intro h
    have := congrArg (fun σ => σ k) h
    simp [swap_apply_right, swap_apply_of_ne_of_ne hik.symm hjk.symm] at this
    exact hij this.symm
  refine ⟨?_, orderOf_eq_prime h3 hne, ?_⟩
  · rw [mul_assoc, ← mul_assoc (swap i k), swap_mul_self, one_mul, swap_mul_self]
  · intro h2
    have : (swap i j * swap i k) ^ 3 = swap i j * swap i k := by rw [pow_succ, h2, one_mul]
    exact hne (this.symm.trans h3)

/-- Every group with fewer than six elements is abelian: `S₃` is the smallest non-abelian group (stated; checked
on the groups of orders `1` to `5`). -/
def SmallestNonabelian : Prop :=
  ∀ (G : Type) [Group G] [Fintype G], Fintype.card G < 6 → ∀ a b : G, a * b = b * a

/-! ## exm-nonabelian-so3 -/

/-- Rotations about the `X`- and `Z`-axes by the same angle `θ` do not commute unless `sin θ = 0`. -/
theorem rot_xz_ne (θ : ℝ) (hθ : Real.sin θ ≠ 0) : Rx θ * Rz θ ≠ Rz θ * Rx θ := by
  intro h
  have h01 := congrFun (congrFun h 0) 1
  simp [Rx, Rz, Matrix.mul_apply, Fin.sum_univ_succ] at h01
  have hc : Real.cos θ = 1 := by
    have : Real.sin θ * (Real.cos θ - 1) = 0 := by linarith
    rcases mul_eq_zero.mp this with h | h
    · exact absurd h hθ
    · linarith
  have := Real.sin_sq_add_cos_sq θ
  rw [hc] at this
  exact hθ (by nlinarith)

/-- The diagonal phase `e^{tT₃} = diag (e^{-it/2}, e^{it/2})` of `SU(2)`. -/
noncomputable def phase (t : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![Complex.exp (((-(t / 2) : ℝ) : ℂ) * Complex.I), 0; 0, Complex.exp (((t / 2 : ℝ) : ℂ) * Complex.I)]

/-- `SU(2)` is not abelian: the phase `e^{tT₃}` and the real rotation `!![0, 1; -1, 0]` commute only when
`sin t = 0`. -/
theorem su2_nonabelian (t : ℝ) (ht : Real.sin t ≠ 0) :
    phase t * !![0, 1; -1, 0] ≠ !![(0 : ℂ), 1; -1, 0] * phase t := by
  intro h
  have h01 := congrFun (congrFun h 0) 1
  simp [phase, Matrix.mul_apply, Fin.sum_univ_succ] at h01
  have := congrArg Complex.im h01
  simp [Complex.exp_ofReal_mul_I_im] at this
  have hs : Real.sin (t / 2) = 0 := by linarith
  apply ht
  have ht2 : t = 2 * (t / 2) := by ring
  rw [ht2, Real.sin_two_mul, hs]; ring

/-! ## exr-grp-abelian-test -/

/-- In an abelian group every ordered pair commutes: `|G|²` commuting pairs (`36` for `ℤ₆`, `16` for `ℤ₂ × ℤ₂`). -/
theorem addCommutingPairs_abelian (G : Type) [AddCommGroup G] [Fintype G] [DecidableEq G] :
    addCommutingPairs G = Fintype.card G ^ 2 := by
  unfold addCommutingPairs
  rw [Finset.filter_true_of_mem (fun p _ => add_comm p.1 p.2), Finset.card_univ, Fintype.card_prod, sq]

/-- The number of commuting pairs is the sum over `g` of the number of elements commuting with `g`. -/
theorem commutingPairs_eq_sum (G : Type) [Group G] [Fintype G] [DecidableEq G] :
    commutingPairs G = ∑ g : G, (Finset.univ.filter fun h => g * h = h * g).card := by
  unfold commutingPairs
  rw [Finset.card_filter, ← Finset.univ_product_univ, Finset.sum_product]
  simp [Finset.card_filter]

/-- In `S₃` the identity commutes with all `6` elements, a transposition with `2` and a three-cycle with `3`. -/
theorem s3_centralizer_card (g : Perm (Fin 3)) :
    (Finset.univ.filter fun h => g * h = h * g).card = if g = 1 then 6 else if Perm.sign g = -1 then 2 else 3 := by
  revert g; decide

/-- All pairs commute exactly when the group is abelian. -/
theorem commutingPairs_eq_sq_iff (G : Type) [Group G] [Fintype G] [DecidableEq G] :
    commutingPairs G = Fintype.card G ^ 2 ↔ ∀ a b : G, a * b = b * a := by
  unfold commutingPairs
  rw [sq, ← Fintype.card_prod, ← Finset.card_univ]
  constructor
  · intro h a b
    have := (Finset.filter_card_eq h (a, b) (Finset.mem_univ _))
    simpa using this
  · intro h; rw [Finset.filter_true_of_mem (fun p _ => h p.1 p.2)]

/-- The number of commuting pairs of `Sₙ` is `n!` times the number of cycle types (conjugacy classes): `18` for
`n = 3` (stated; checked for `n ≤ 4`). -/
def SymmetricCommutingPairs : Prop :=
  ∀ n : ℕ, commutingPairs (Perm (Fin n)) = n ! * (Finset.univ.image fun g : Perm (Fin n) => g.cycleType).card

/-- The number of commuting pairs of the dihedral group `Dₙ` (order `2n`) is `n (n + 6)` for even `n` and
`n (n + 3)` for odd `n`: `40` for the square (stated; checked for `n ≤ 6`). -/
def DihedralCommutingPairs : Prop :=
  ∀ n : ℕ, 0 < n → commutingPairs (DihedralGroup n) = if Even n then n * (n + 6) else n * (n + 3)

/-- The number of commuting pairs of the dicyclic group `Q₄ₙ` is `4n (n + 3)`: `40` for the quaternion group
`Q₈ = Q₄·₂` (stated; checked for `n ≤ 3`). -/
def QuaternionCommutingPairs : Prop :=
  ∀ n : ℕ, 0 < n → commutingPairs (QuaternionGroup n) = 4 * n * (n + 3)

/-! ## exr-grp-square-identity -/

/-- A group in which every element squares to the identity is abelian. -/
theorem abelian_of_sq_eq_one (G : Type) [Group G] (h : ∀ g : G, g ^ 2 = 1) (a b : G) : a * b = b * a := by
  have hinv : ∀ g : G, g⁻¹ = g := fun g => inv_eq_of_mul_eq_one_right (by rw [← sq]; exact h g)
  calc a * b = (a * b)⁻¹ := (hinv _).symm
    _ = b⁻¹ * a⁻¹ := mul_inv_rev a b
    _ = b * a := by rw [hinv, hinv]

/-- In the Klein four-group `ℤ₂ × ℤ₂` every element is its own inverse: `x + x = 0`. -/
theorem klein_sq (x : ZMod 2 × ZMod 2) : x + x = 0 := by
  revert x; decide

/-! ## exr-grp-commutator -/

/-- Upper triangular matrices multiply to upper triangular matrices, with diagonal entries multiplied. -/
theorem tri_mul (a b d a' b' d' : ℚ) : !![a, b; 0, d] * !![a', b'; 0, d'] = !![a * a', a * b' + b * d'; 0, d * d'] := by
  VERIPHY[gt-matrix]

/-- The inverse of an upper triangular matrix with `a d ≠ 0` is upper triangular. -/
theorem tri_inv (a b d : ℚ) (ha : a ≠ 0) (hd : d ≠ 0) : !![a, b; 0, d] * !![1 / a, -b / (a * d); 0, 1 / d] = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> field_simp

/-- The commutator of `g = diag (a, 1)` and the shear `h = !![1, b; 0, 1]` is the shear by `(a - 1) b`. -/
theorem commutator_tri (a b : ℚ) (ha : a ≠ 0) :
    !![a, 0; 0, 1] * !![1, b; 0, 1] * !![a⁻¹, 0; 0, 1] * !![1, -b; 0, 1] = !![1, (a - 1) * b; 0, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> field_simp <;> ring

/-- The commutator shear is the identity exactly when `a = 1` or `b = 0`: the triangular group is not abelian. -/
theorem commutator_eq_one_iff (a b : ℚ) : (!![1, (a - 1) * b; 0, 1] : Matrix (Fin 2) (Fin 2) ℚ) = 1 ↔ a = 1 ∨ b = 0 := by
  constructor
  · intro h
    have := congrFun (congrFun h 0) 1
    simp at this
    rcases this with h | h
    · left; linarith
    · right; exact h
  · rintro (h | h) <;> simp [h]

/-- Shears multiply by adding the entries `b`, and commute: an abelian group isomorphic to `(ℝ, +)`. -/
theorem shear_mul (b b' : ℚ) : !![1, b; 0, 1] * !![1, b'; 0, 1] = !![1, b + b'; 0, 1] ∧
    !![1, b; 0, 1] * !![1, b'; 0, 1] = !![1, b'; 0, 1] * !![1, b; 0, 1] := by
  refine ⟨?_, ?_⟩
  · VERIPHY[gt-matrix]
  · VERIPHY[gt-matrix]

/-! ## exm-order-s3 -/

/-- The identity has order `1`. -/
theorem orderOf_one_eq (G : Type) [Group G] : orderOf (1 : G) = 1 := orderOf_one

/-- A transposition has order `2`. -/
theorem orderOf_swap (n : ℕ) (i j : Fin n) (hij : i ≠ j) : orderOf (swap i j) = 2 := by
  refine orderOf_eq_prime (by rw [sq, swap_mul_self]) ?_
  intro h
  have := congrArg (fun σ => σ i) h
  simp [swap_apply_left] at this
  exact hij this.symm

/-- The orders of the elements of `S₃`: `1` for `e`, `2` for the three transpositions, `3` for the two three-cycles
(the order read off `ofOrder`, computably). -/
theorem s3_orders (g : Perm (Fin 3)) : g ∈ ofOrder (Perm (Fin 3)) (if g = 1 then 1 else if Perm.sign g = -1 then 2 else 3) := by
  revert g; decide

/-- The number of elements of `S₃` of each order: one of order `1`, three of order `2`, two of order `3`. -/
theorem s3_order_count (k : ℕ) :
    (ofOrder (Perm (Fin 3)) k).card = if k = 1 then 1 else if k = 2 then 3 else if k = 3 then 2 else 0 := by
  by_cases hk : k ≤ 6
  · interval_cases k <;> decide
  · have : ofOrder (Perm (Fin 3)) k = ∅ := by
      apply Finset.filter_eq_empty_iff.mpr
      intro g _ ⟨_, h⟩
      exact h 6 (by norm_num) (by omega) (by
        have h6 : Fintype.card (Perm (Fin 3)) = 6 := by decide
        rw [← h6]; exact pow_card_eq_one)
    rw [this]; simp; omega

/-- The order of every element of a finite group divides the order of the group (`6` for `S₃`). -/
theorem orderOf_dvd_card_group (G : Type) [Group G] [Fintype G] (g : G) : orderOf g ∣ Fintype.card G :=
  orderOf_dvd_card

/-! ## exm-order-zn -/

/-- `ℤₙ` has `n` elements. -/
theorem card_zmod (n : ℕ) [NeZero n] : Fintype.card (ZMod n) = n := ZMod.card n

/-- The order of `k` in `ℤ₆` is `6 / gcd (6, k)`: `1` has order `6`, `2, 4` order `3`, `3` order `2`, `0` order `1`. -/
theorem z6_orders (k : ZMod 6) : k ∈ ofAddOrder (ZMod 6) (6 / Nat.gcd 6 k.val) := by
  revert k; decide

/-- `1` generates `ℤₙ`: every `k` is the multiple `k · 1`. -/
theorem one_generates (n : ℕ) [NeZero n] (k : ZMod n) : k.val • (1 : ZMod n) = k := by
  simp [nsmul_eq_mul, ZMod.natCast_zmod_val]

/-- `ℤₙ` is cyclic. -/
theorem zmod_isAddCyclic (n : ℕ) : IsAddCyclic (ZMod n) := inferInstance

/-! ## exm-order-infinite -/

/-- Powers of a rotation: `(R θ)^m = R (m θ)`. -/
theorem R_pow (θ : ℝ) (m : ℕ) : (R θ) ^ m = R (m * θ) := by
  induction m with
  | zero => ext i j; fin_cases i <;> fin_cases j <;> simp [R]
  | succ m ih => rw [pow_succ, ih, R_mul]; congr 1; push_cast; ring

/-- `R θ = 1` exactly when `θ` is an integer multiple of `2π`. -/
theorem R_eq_one_iff (θ : ℝ) : R θ = 1 ↔ ∃ k : ℤ, θ = k * (2 * π) := by
  constructor
  · intro h
    have := congrFun (congrFun h 0) 0
    simp [R] at this
    rcases Real.cos_eq_one_iff θ |>.mp this with ⟨k, hk⟩
    exact ⟨k, by linarith⟩
  · rintro ⟨k, rfl⟩
    ext i j; fin_cases i <;> fin_cases j <;> simp [R, Real.cos_int_mul_two_pi, Real.sin_int_mul_two_pi]

/-- The rotation by `2π/n` has order `n`: `(R (2π/n))^n = 1` and no smaller positive power is `1`. -/
theorem rot_order (n : ℕ) (hn : 0 < n) :
    (R (2 * π / n)) ^ n = 1 ∧ ∀ m, 0 < m → m < n → (R (2 * π / n)) ^ m ≠ 1 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  refine ⟨?_, ?_⟩
  · rw [R_pow, R_eq_one_iff]; exact ⟨1, by field_simp; ring⟩
  · intro m hm hmn h
    rw [R_pow, R_eq_one_iff] at h
    obtain ⟨k, hk⟩ := h
    have hpi : (0 : ℝ) < 2 * π := by positivity
    have : (m : ℝ) = k * n := by
      have := hk
      field_simp at this
      nlinarith [Real.pi_pos]
    have h1 : (0 : ℝ) < k * n := by rw [← this]; exact_mod_cast hm
    have h2 : (k : ℝ) * n < n := by rw [← this]; exact_mod_cast hmn
    have hk0 : (0 : ℝ) < k := by
      by_contra hc; push_neg at hc
      have : (k : ℝ) * n ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hc (by positivity)
      linarith
    have hk1 : (k : ℝ) < 1 := by
      by_contra hc; push_neg at hc
      have : (n : ℝ) ≤ k * n := by nlinarith
      linarith
    have : (0 : ℤ) < k := by exact_mod_cast hk0
    have : k < (1 : ℤ) := by exact_mod_cast hk1
    omega

/-- The rotation by an irrational multiple of `π` has infinite order: no positive power is the identity. -/
theorem rot_irrational (θ : ℝ) (hθ : Irrational (θ / π)) (m : ℕ) (hm : 0 < m) : (R θ) ^ m ≠ 1 := by
  intro h
  rw [R_pow, R_eq_one_iff] at h
  obtain ⟨k, hk⟩ := h
  apply hθ
  refine ⟨(2 * k : ℚ) / m, ?_⟩
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  have hpi := Real.pi_pos.ne'
  push_cast
  field_simp
  linear_combination hk

/-- `R` is injective on `[0, 2π)`: the rotations of the plane are pairwise distinct, so `SO(2)` is infinite. -/
theorem R_injOn (a b : ℝ) (ha : a ∈ Set.Ico 0 (2 * π)) (hb : b ∈ Set.Ico 0 (2 * π)) (h : R a = R b) : a = b := by
  have h1 : R (a - b) = 1 := by
    have : R a * R (-b) = R b * R (-b) := by rw [h]
    rw [R_mul, R_mul, add_neg_cancel] at this
    rw [sub_eq_add_neg, this]
    exact (R_eq_one_iff 0).mpr ⟨0, by simp⟩
  obtain ⟨k, hk⟩ := (R_eq_one_iff _).mp h1
  have hpi : 0 < 2 * π := by positivity
  have hk1 : (k : ℝ) < 1 := by
    by_contra hc; push_neg at hc
    have : (1 : ℝ) * (2 * π) ≤ k * (2 * π) := by nlinarith
    linarith [ha.2, hb.1]
  have hk2 : (-1 : ℝ) < k := by
    by_contra hc; push_neg at hc
    have : (k : ℝ) * (2 * π) ≤ (-1) * (2 * π) := by nlinarith
    linarith [ha.1, hb.2]
  have : k = 0 := by
    have h1' : k < (1 : ℤ) := by exact_mod_cast hk1
    have h2' : (-1 : ℤ) < k := by exact_mod_cast hk2
    omega
  subst this; simp at hk; linarith

/-- `-1 ∈ SU(2)` has order `2`: `(-1)² = 1` and `-1 ≠ 1`. -/
theorem neg_one_order_two (n : ℕ) [NeZero n] : (-1 : Matrix (Fin n) (Fin n) ℂ) ^ 2 = 1 ∧ (-1 : Matrix (Fin n) (Fin n) ℂ) ≠ 1 := by
  refine ⟨by simp, ?_⟩
  intro h
  have := congrFun (congrFun h ⟨0, Nat.pos_of_neZero n⟩) ⟨0, Nat.pos_of_neZero n⟩
  simp at this
  norm_num at this

/-- Powers of the phase: `(e^{tT₃})^m = e^{mtT₃}`. -/
theorem phase_pow (t : ℝ) (m : ℕ) : (phase t) ^ m = phase (m * t) := by
  induction m with
  | zero => ext i j; fin_cases i <;> fin_cases j <;> simp [phase]
  | succ m ih =>
    rw [pow_succ, ih]
    ext i j; fin_cases i <;> fin_cases j <;> simp [phase, Matrix.mul_apply, Fin.sum_univ_succ, ← Complex.exp_add] <;>
      congr 1 <;> push_cast <;> ring

/-- `e^{tT₃} = 1` exactly when `t` is an integer multiple of `4π`. -/
theorem phase_eq_one_iff (t : ℝ) : phase t = 1 ↔ ∃ k : ℤ, t = k * (4 * π) := by
  constructor
  · intro h
    have := congrFun (congrFun h 1) 1
    simp [phase] at this
    rw [Complex.exp_eq_one_iff] at this
    obtain ⟨k, hk⟩ := this
    refine ⟨k, ?_⟩
    have := congrArg Complex.im hk
    simp at this
    linarith
  · rintro ⟨k, rfl⟩
    ext i j; fin_cases i <;> fin_cases j <;> simp [phase]
    · rw [Complex.exp_eq_one_iff]; exact ⟨-k, by push_cast; ring⟩
    · rw [Complex.exp_eq_one_iff]; exact ⟨k, by push_cast; ring⟩

/-- When `4π/t = N` is an integer, `e^{tT₃}` has order `N`. -/
theorem phase_order (N : ℕ) (hN : 0 < N) :
    (phase (4 * π / N)) ^ N = 1 ∧ ∀ m, 0 < m → m < N → (phase (4 * π / N)) ^ m ≠ 1 := by
  have hN' : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  refine ⟨?_, ?_⟩
  · rw [phase_pow, phase_eq_one_iff]; exact ⟨1, by field_simp; ring⟩
  · intro m hm hmN h
    rw [phase_pow, phase_eq_one_iff] at h
    obtain ⟨k, hk⟩ := h
    have : (m : ℝ) = k * N := by
      have := hk
      field_simp at this
      nlinarith [Real.pi_pos]
    have hk0 : (0 : ℝ) < k := by
      by_contra hc; push_neg at hc
      have : (k : ℝ) * N ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hc (by positivity)
      have : (0 : ℝ) < m := by exact_mod_cast hm
      linarith
    have hk1 : (k : ℝ) < 1 := by
      by_contra hc; push_neg at hc
      have : (N : ℝ) ≤ k * N := by nlinarith
      have : (m : ℝ) < N := by exact_mod_cast hmN
      linarith
    have : (0 : ℤ) < k := by exact_mod_cast hk0
    have : k < (1 : ℤ) := by exact_mod_cast hk1
    omega

/-- `ℤ` is infinite. -/
theorem int_infinite (n : ℕ) : ∃ S : Finset ℤ, S.card = n := ⟨Finset.range n |>.map Nat.castEmbedding, by simp⟩

/-! ## exr-grp-orders-z12 -/

/-- The order of `k` in `ℤ₁₂` is `12 / gcd (k, 12)`. -/
theorem z12_orders (k : ZMod 12) : k ∈ ofAddOrder (ZMod 12) (12 / Nat.gcd 12 k.val) := by
  revert k; decide

/-- The number of elements of `ℤ₁₂` of order `d` is `φ(d)` for `d ∣ 12` (one of order `1`, one of order `2`, two of
order `3`, two of order `4`, two of order `6`, four of order `12`) and `0` otherwise. -/
theorem z12_order_count (d : ℕ) :
    (ofAddOrder (ZMod 12) d).card = if d ∣ 12 ∧ 0 < d then Nat.totient d else 0 := by
  by_cases hd : d ≤ 12
  · interval_cases d <;> decide
  · have : ofAddOrder (ZMod 12) d = ∅ := by
      apply Finset.filter_eq_empty_iff.mpr
      intro g _ ⟨_, h⟩
      exact h 12 (by norm_num) (by omega) (by
        have h12 : Fintype.card (ZMod 12) = 12 := by decide
        rw [← h12]; exact card_nsmul_eq_zero)
    rw [this]
    have : ¬ (d ∣ 12 ∧ 0 < d) := fun ⟨h, _⟩ => hd (Nat.le_of_dvd (by norm_num) h)
    simp [this]

/-- In `ℤₙ` the order of `k` is `n / gcd (k, n)`. -/
theorem zmod_addOrderOf (n : ℕ) (hn : n ≠ 0) (k : ℕ) : addOrderOf (k : ZMod n) = n / Nat.gcd n k :=
  ZMod.addOrderOf_coe k hn

/-- In a cyclic group of order `n` the number of elements of order `d ∣ n` is Euler's `φ(d)`. -/
theorem cyclic_card_orderOf (G : Type) [Group G] [IsCyclic G] [Fintype G] (d : ℕ) (hd : d ∣ Fintype.card G) :
    (Finset.univ.filter fun a : G => orderOf a = d).card = Nat.totient d :=
  IsCyclic.card_orderOf_eq_totient hd

/-! ## exr-grp-order-inverse -/

/-- `g⁻¹` has the order of `g`. -/
theorem orderOf_inv_eq (G : Type) [Group G] (g : G) : orderOf g⁻¹ = orderOf g := orderOf_inv g

/-- A conjugate `k g k⁻¹` has the order of `g`. -/
theorem orderOf_conj_eq (G : Type) [Group G] (g k : G) : orderOf (k * g * k⁻¹) = orderOf g := by
  apply orderOf_eq_orderOf_iff.mpr
  intro n
  rw [conj_pow, conj_eq_one_iff]

/-- `gh` and `hg` have the same order, since `hg = g⁻¹ (gh) g`. -/
theorem orderOf_mul_comm (G : Type) [Group G] (g h : G) : orderOf (g * h) = orderOf (h * g) := by
  have : h * g = g⁻¹ * (g * h) * g⁻¹⁻¹ := by group
  rw [this, orderOf_conj_eq]

/-- In `S₃`, `(1 2)(1 3)` and `(1 3)(1 2)` are different elements of the same order `3`. -/
theorem s3_products_order : s * t13 ≠ t13 * s ∧ orderOf (s * t13) = 3 ∧ orderOf (t13 * s) = 3 := by
  refine ⟨by decide, ?_, ?_⟩ <;> exact (mem_ofOrder_iff (by norm_num) _).mp (by decide)

/-! ## exr-grp-order-power -/

/-- `g^m = e` exactly when the order of `g` divides `m`. -/
theorem pow_eq_one_iff_dvd (G : Type) [Group G] (g : G) (m : ℕ) : g ^ m = 1 ↔ orderOf g ∣ m :=
  orderOf_dvd_iff_pow_eq_one.symm

/-- `g^k` has order `n / gcd (n, k)` when `g` has order `n`. -/
theorem orderOf_pow_eq (G : Type) [Group G] [Finite G] (g : G) (k : ℕ) : orderOf (g ^ k) = orderOf g / Nat.gcd (orderOf g) k :=
  orderOf_pow g

/-- A cyclic group of order `n` has `φ(n)` generators (elements of order `n`). -/
theorem cyclic_card_generators (G : Type) [Group G] [IsCyclic G] [Fintype G] :
    (Finset.univ.filter fun a : G => orderOf a = Fintype.card G).card = Nat.totient (Fintype.card G) :=
  IsCyclic.card_orderOf_eq_totient dvd_rfl

/-- The generators of `ℤ₁₂` are `1, 5, 7, 11`: the residues coprime to `12`. -/
theorem z12_generators (k : ZMod 12) : k ∈ ofAddOrder (ZMod 12) 12 ↔ k = 1 ∨ k = 5 ∨ k = 7 ∨ k = 11 := by
  revert k; decide

end KaluzaKlein.GroupTheory.Groups
