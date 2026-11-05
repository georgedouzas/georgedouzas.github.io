import Mathlib
import KaluzaKlein.PartI.Sets.Bijective
import Veriphy.Kit.Levels

/-!
# Sets, relations and topology, section "Maps" (inverse maps)

The claims of the blocks `def-inverse-map` to `exr-inverse-rotation-matrix` of
`tutorial/part-i-01-sets-relations-topology.qmd`: bijections are the maps with a two-sided inverse, the shift and the
doubling of `ℤ₆`, logarithm, arctangent and cube root, the Möbius map `(x - 1)/(x + 1)`, and the inverse of a rotation
about the `Z`-axis as its transpose.
-/

namespace KaluzaKlein.Sets.Inverse

open Matrix KaluzaKlein.Sets.Sets KaluzaKlein.Sets.Surjective

/-- The cube root, `y^{1/3}` for `y ≥ 0` and `-(-y)^{1/3}` for `y < 0`. -/
noncomputable def cbrt (y : ℝ) : ℝ := if 0 ≤ y then y ^ ((3 : ℕ) : ℝ)⁻¹ else -((-y) ^ ((3 : ℕ) : ℝ)⁻¹)

/-! ## def-inverse-map -/

/-- A map is a bijection exactly when it has an inverse `g`, with `g (f x) = x` and `f (g y) = y`. -/
theorem bijective_iff_has_inverse {X Y : Type} (f : X → Y) :
    Function.Bijective f ↔ ∃ g : Y → X, Function.LeftInverse g f ∧ Function.RightInverse g f :=
  Function.bijective_iff_has_inverse

example : veriphy_instance% KaluzaKlein.Sets.Inverse.bijective_iff_has_inverse (Fin 2) (Fin 2) (id) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Bijective f ↔ ∃ g : Y → X, Function.LeftInverse g f ∧ Function.RightInverse g f := by
    intro X Y f
    exact (Function.bijective_iff_has_inverse)
  exact @key (Fin 2) (Fin 2) (id)

example : veriphy_instance% KaluzaKlein.Sets.Inverse.bijective_iff_has_inverse (ℤ) (ℤ) (fun k => k + 1) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Bijective f ↔ ∃ g : Y → X, Function.LeftInverse g f ∧ Function.RightInverse g f := by
    intro X Y f
    exact (Function.bijective_iff_has_inverse)
  exact @key (ℤ) (ℤ) (fun k => k + 1)

example : veriphy_instance% KaluzaKlein.Sets.Inverse.bijective_iff_has_inverse (Fin 3) (Fin 2) (![0, 0, 1]) := by
  have key : ∀ {X Y : Type} (f : X → Y), Function.Bijective f ↔ ∃ g : Y → X, Function.LeftInverse g f ∧ Function.RightInverse g f := by
    intro X Y f
    exact (Function.bijective_iff_has_inverse)
  exact @key (Fin 3) (Fin 2) (![0, 0, 1])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Inverse.bijective_iff_has_inverse
-- test: set-has-inverse-test.sage

/-! ## exm-inverse-shift -/

/-- The shift `k ↦ k + a` of `ℤₙ` is undone by `k ↦ k - a` in both orders, so it is a bijection: `k ↦ k + 1` on
`ℤ₆`. -/
theorem shift_inverse (n : ℕ) (a : ZMod n) :
    Function.LeftInverse (fun k => k - a) (fun k => k + a) ∧ Function.RightInverse (fun k => k - a) (fun k => k + a) ∧
      Function.Bijective (fun k : ZMod n => k + a) :=
  ⟨fun k => by simp, fun k => by simp, ⟨fun x y h => by simpa using h, fun y => ⟨y - a, by simp⟩⟩⟩

example : veriphy_instance% KaluzaKlein.Sets.Inverse.shift_inverse (6) (1) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Inverse.shift_inverse (1) (0) := by
  decide

example : veriphy_instance% KaluzaKlein.Sets.Inverse.shift_inverse (5) (3) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Inverse.shift_inverse
-- test: set-shift-inverse-test.sage

/-- Doubling on `ℤ₆` is not injective: two distinct elements with the same double (Sage finds `0` and `3`). -/
theorem double_not_injective : ∃ a b : ZMod 6, a ≠ b ∧ 2 * a = 2 * b := by
  veriphy_certified (exact ⟨0, 3, by decide⟩)
  -- certificate: set-zmod6-double-noninjective.sage

/-- Doubling on `ℤ₆` is not surjective: some element is no double (Sage finds `1`, outside the image `{0, 2, 4}`). -/
theorem double_not_surjective : ∃ y : ZMod 6, ∀ k : ZMod 6, 2 * k ≠ y := by
  veriphy_certified (exact ⟨1, by decide⟩)
  -- certificate: set-zmod6-double-nonsurjective.sage

/-! ## exm-inverse-log -/

/-- `log` inverts `exp : ℝ → (0, ∞)`. -/
theorem log_exp_inverse (x y : ℝ) (hy : 0 < y) : Real.log (Real.exp x) = x ∧ Real.exp (Real.log y) = y :=
  ⟨Real.log_exp x, Real.exp_log hy⟩

example : veriphy_instance% KaluzaKlein.Sets.Inverse.log_exp_inverse (0) (1) ((by norm_num)) := by
  have key : ∀ (x y : ℝ) (hy : 0 < y), Real.log (Real.exp x) = x ∧ Real.exp (Real.log y) = y := by
    intro x y hy
    exact (⟨Real.log_exp x, Real.exp_log hy⟩)
  exact @key (0) (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Inverse.log_exp_inverse (2) (3) ((by norm_num)) := by
  have key : ∀ (x y : ℝ) (hy : 0 < y), Real.log (Real.exp x) = x ∧ Real.exp (Real.log y) = y := by
    intro x y hy
    exact (⟨Real.log_exp x, Real.exp_log hy⟩)
  exact @key (2) (3) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Inverse.log_exp_inverse (-1) (1/2) ((by norm_num)) := by
  have key : ∀ (x y : ℝ) (hy : 0 < y), Real.log (Real.exp x) = x ∧ Real.exp (Real.log y) = y := by
    intro x y hy
    exact (⟨Real.log_exp x, Real.exp_log hy⟩)
  exact @key (-1) (1/2) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Inverse.log_exp_inverse
-- test: set-log-exp-test.sage

/-- `arctan` inverts `tan : (-π/2, π/2) → ℝ`. -/
theorem arctan_tan_inverse (x y : ℝ) (hx : x ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)) :
    Real.arctan (Real.tan x) = x ∧ Real.tan (Real.arctan y) = y :=
  ⟨Real.arctan_tan hx.1 hx.2, Real.tan_arctan y⟩

example : veriphy_instance% KaluzaKlein.Sets.Inverse.arctan_tan_inverse (0) (0) ((by constructor <;> linarith [Real.pi_pos])) := by
  have key : ∀ (x y : ℝ) (hx : x ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)), Real.arctan (Real.tan x) = x ∧ Real.tan (Real.arctan y) = y := by
    intro x y hx
    exact (⟨Real.arctan_tan hx.1 hx.2, Real.tan_arctan y⟩)
  exact @key (0) (0) ((by constructor <;> linarith [Real.pi_pos]))

example : veriphy_instance% KaluzaKlein.Sets.Inverse.arctan_tan_inverse (1) (5) ((by constructor <;> linarith [Real.pi_gt_three])) := by
  have key : ∀ (x y : ℝ) (hx : x ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)), Real.arctan (Real.tan x) = x ∧ Real.tan (Real.arctan y) = y := by
    intro x y hx
    exact (⟨Real.arctan_tan hx.1 hx.2, Real.tan_arctan y⟩)
  exact @key (1) (5) ((by constructor <;> linarith [Real.pi_gt_three]))

example : veriphy_instance% KaluzaKlein.Sets.Inverse.arctan_tan_inverse (-1) (-2) ((by constructor <;> linarith [Real.pi_gt_three])) := by
  have key : ∀ (x y : ℝ) (hx : x ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)), Real.arctan (Real.tan x) = x ∧ Real.tan (Real.arctan y) = y := by
    intro x y hx
    exact (⟨Real.arctan_tan hx.1 hx.2, Real.tan_arctan y⟩)
  exact @key (-1) (-2) ((by constructor <;> linarith [Real.pi_gt_three]))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Inverse.arctan_tan_inverse
-- test: set-arctan-tan-test.sage

/-- The cube root inverts `x ↦ x³`. -/
theorem cbrt_inverse (x y : ℝ) : cbrt (x ^ 3) = x ∧ cbrt y ^ 3 = y := by
  have h3 : ((3 : ℕ) : ℝ) ≠ 0 := by norm_num
  have key : ∀ y : ℝ, cbrt y ^ 3 = y := by
    intro y
    unfold cbrt
    split_ifs with hy
    · rw [← Real.rpow_natCast, ← Real.rpow_mul hy, inv_mul_cancel₀ h3, Real.rpow_one]
    · rw [neg_pow, ← Real.rpow_natCast ((-y) ^ _), ← Real.rpow_mul (by linarith), inv_mul_cancel₀ h3,
        Real.rpow_one]
      norm_num
  refine ⟨?_, key y⟩
  have hinj : Function.Injective (fun x : ℝ => x ^ 3) := (Odd.strictMono_pow (by decide : Odd 3)).injective
  exact hinj (key (x ^ 3))

example : veriphy_instance% KaluzaKlein.Sets.Inverse.cbrt_inverse (0) (0) := by
  have key : ∀ (x y : ℝ), cbrt (x ^ 3) = x ∧ cbrt y ^ 3 = y := by
    intro x y
    have h3 : ((3 : ℕ) : ℝ) ≠ 0 := by norm_num
    have key : ∀ y : ℝ, cbrt y ^ 3 = y := by
      intro y
      unfold cbrt
      split_ifs with hy
      · rw [← Real.rpow_natCast, ← Real.rpow_mul hy, inv_mul_cancel₀ h3, Real.rpow_one]
      · rw [neg_pow, ← Real.rpow_natCast ((-y) ^ _), ← Real.rpow_mul (by linarith), inv_mul_cancel₀ h3,
          Real.rpow_one]
        norm_num
    refine ⟨?_, key y⟩
    have hinj : Function.Injective (fun x : ℝ => x ^ 3) := (Odd.strictMono_pow (by decide : Odd 3)).injective
    exact hinj (key (x ^ 3))
  exact @key (0) (0)

example : veriphy_instance% KaluzaKlein.Sets.Inverse.cbrt_inverse (2) (8) := by
  have key : ∀ (x y : ℝ), cbrt (x ^ 3) = x ∧ cbrt y ^ 3 = y := by
    intro x y
    have h3 : ((3 : ℕ) : ℝ) ≠ 0 := by norm_num
    have key : ∀ y : ℝ, cbrt y ^ 3 = y := by
      intro y
      unfold cbrt
      split_ifs with hy
      · rw [← Real.rpow_natCast, ← Real.rpow_mul hy, inv_mul_cancel₀ h3, Real.rpow_one]
      · rw [neg_pow, ← Real.rpow_natCast ((-y) ^ _), ← Real.rpow_mul (by linarith), inv_mul_cancel₀ h3,
          Real.rpow_one]
        norm_num
    refine ⟨?_, key y⟩
    have hinj : Function.Injective (fun x : ℝ => x ^ 3) := (Odd.strictMono_pow (by decide : Odd 3)).injective
    exact hinj (key (x ^ 3))
  exact @key (2) (8)

example : veriphy_instance% KaluzaKlein.Sets.Inverse.cbrt_inverse (-2) (-27) := by
  have key : ∀ (x y : ℝ), cbrt (x ^ 3) = x ∧ cbrt y ^ 3 = y := by
    intro x y
    have h3 : ((3 : ℕ) : ℝ) ≠ 0 := by norm_num
    have key : ∀ y : ℝ, cbrt y ^ 3 = y := by
      intro y
      unfold cbrt
      split_ifs with hy
      · rw [← Real.rpow_natCast, ← Real.rpow_mul hy, inv_mul_cancel₀ h3, Real.rpow_one]
      · rw [neg_pow, ← Real.rpow_natCast ((-y) ^ _), ← Real.rpow_mul (by linarith), inv_mul_cancel₀ h3,
          Real.rpow_one]
        norm_num
    refine ⟨?_, key y⟩
    have hinj : Function.Injective (fun x : ℝ => x ^ 3) := (Odd.strictMono_pow (by decide : Odd 3)).injective
    exact hinj (key (x ^ 3))
  exact @key (-2) (-27)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Inverse.cbrt_inverse
-- test: set-cbrt-test.sage

/-! ## exm-inverse-rotation -/

/-- `R_z(α) R_z(-α) = 1`: the rotation by `-α` undoes the rotation by `α`. -/
theorem Rz_mul_Rz_neg (α : ℝ) : Rz α * Rz (-α) = 1 := by
  unfold Rz
  rw [Real.cos_neg, Real.sin_neg]
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ, Matrix.mul_apply,
            Fin.sum_univ_succ, Matrix.transpose_apply]
        all_goals
          first
          | ring1
          | (norm_num; done)
          | linear_combination ((1 : ℝ)) * (Real.cos_sq_add_sin_sq (α : ℝ))
          | linear_combination -(((1 : ℝ)) * (Real.cos_sq_add_sin_sq (α : ℝ)))))
  -- certificate: set-matrix-trig.sage

/-- `R_z(-α) = R_z(α)ᵀ`: the inverse of a rotation is its transpose. -/
theorem Rz_neg_eq_transpose (α : ℝ) : Rz (-α) = (Rz α)ᵀ := by
  unfold Rz
  rw [Real.cos_neg, Real.sin_neg]
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ, Matrix.mul_apply,
            Fin.sum_univ_succ, Matrix.transpose_apply]
        all_goals ring1))
  -- certificate: set-matrix-trig.sage

/-- `R_z(α)ᵀ R_z(α) = 1`: `R_z(α)` is orthogonal. -/
theorem Rz_transpose_mul (α : ℝ) : (Rz α)ᵀ * Rz α = 1 := by
  unfold Rz
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ, Matrix.mul_apply,
            Fin.sum_univ_succ, Matrix.transpose_apply]
        all_goals
          first
          | ring1
          | (norm_num; done)
          | linear_combination ((1 : ℝ)) * (Real.cos_sq_add_sin_sq (α : ℝ))
          | linear_combination -(((1 : ℝ)) * (Real.cos_sq_add_sin_sq (α : ℝ)))))
  -- certificate: set-matrix-trig.sage

/-! ## exr-inverse-moebius

The compositions, stated with `w = 1/(1 - y)` (respectively `t = 1/(x + 1)`) as a variable with its defining equation:
`f(f⁻¹(y)) = y` says `x - 1 = y (x + 1)` for `x = (1 + y) w`, and `f⁻¹(f(x)) = x` says `1 + y = x (1 - y)` for
`y = (x - 1) t`. -/

theorem moebius_f_finv (y w : ℝ) (hw : (1 - y) * w = 1) : (1 + y) * w - 1 = y * ((1 + y) * w + 1) := by
  veriphy_certified (linear_combination (y + 1) * hw)
  -- certificate: ideal_membership.sage

theorem moebius_finv_f (x t : ℝ) (ht : (x + 1) * t = 1) : 1 + (x - 1) * t = x * (1 - (x - 1) * t) := by
  veriphy_certified (linear_combination (x - 1) * ht)
  -- certificate: ideal_membership.sage

/-- `x ↦ (x - a)/(x + a)` (`a ≠ 0`) is a bijection of `ℝ ∖ {-a}` onto `ℝ ∖ {1}` with inverse
`y ↦ a (1 + y)/(1 - y)`: for `a = 1`, `f(x) = (x - 1)/(x + 1)` and `f⁻¹(y) = (1 + y)/(1 - y)`. -/
theorem moebius_bijOn (a : ℝ) (ha : a ≠ 0) :
    Set.BijOn (fun x => (x - a) / (x + a)) {x | x ≠ -a} {y | y ≠ 1} ∧
      (∀ y, y ≠ 1 → (a * (1 + y) / (1 - y) - a) / (a * (1 + y) / (1 - y) + a) = y) ∧
      (∀ x, x ≠ -a → a * (1 + (x - a) / (x + a)) / (1 - (x - a) / (x + a)) = x) := by
  have back : ∀ y, y ≠ 1 → (a * (1 + y) / (1 - y) - a) / (a * (1 + y) / (1 - y) + a) = y := by
    intro y hy
    have h1 : 1 - y ≠ 0 := sub_ne_zero.2 (Ne.symm hy)
    rw [div_sub' h1, div_add' _ _ _ h1, div_div_div_cancel_right₀ h1]
    rw [div_eq_iff (by intro h; apply ha; linear_combination h / 2)]
    ring
  have forth : ∀ x, x ≠ -a → a * (1 + (x - a) / (x + a)) / (1 - (x - a) / (x + a)) = x := by
    intro x hx
    have h1 : x + a ≠ 0 := fun h => hx (by linarith)
    rw [one_add_div h1, one_sub_div h1, mul_div_assoc', div_div_div_cancel_right₀ h1]
    rw [div_eq_iff (by intro h; apply ha; linear_combination h / 2)]
    ring
  refine ⟨⟨fun x hx => ?_, fun x hx x' hx' h => ?_, fun y hy => ⟨a * (1 + y) / (1 - y), ?_, back y hy⟩⟩, back, forth⟩
  · intro h
    have h1 : x + a ≠ 0 := fun h' => hx (by linarith)
    rw [div_eq_one_iff_eq h1] at h
    exact ha (by linarith)
  · rw [← forth x hx, ← forth x' hx']
    simp only at h
    rw [h]
  · intro h
    have h1 : 1 - y ≠ 0 := sub_ne_zero.2 (Ne.symm hy)
    rw [div_eq_iff h1] at h
    exact ha (by linarith)

example : veriphy_instance% KaluzaKlein.Sets.Inverse.moebius_bijOn (1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : a ≠ 0), Set.BijOn (fun x => (x - a) / (x + a)) {x | x ≠ -a} {y | y ≠ 1} ∧ (∀ y, y ≠ 1 → (a * (1 + y) / (1 - y) - a) / (a * (1 + y) / (1 - y) + a) = y) ∧ (∀ x, x ≠ -a → a * (1 + (x - a) / (x + a)) / (1 - (x - a) / (x + a)) = x) := by
    intro a ha
    have back : ∀ y, y ≠ 1 → (a * (1 + y) / (1 - y) - a) / (a * (1 + y) / (1 - y) + a) = y := by
      intro y hy
      have h1 : 1 - y ≠ 0 := sub_ne_zero.2 (Ne.symm hy)
      rw [div_sub' h1, div_add' _ _ _ h1, div_div_div_cancel_right₀ h1]
      rw [div_eq_iff (by intro h; apply ha; linear_combination h / 2)]
      ring
    have forth : ∀ x, x ≠ -a → a * (1 + (x - a) / (x + a)) / (1 - (x - a) / (x + a)) = x := by
      intro x hx
      have h1 : x + a ≠ 0 := fun h => hx (by linarith)
      rw [one_add_div h1, one_sub_div h1, mul_div_assoc', div_div_div_cancel_right₀ h1]
      rw [div_eq_iff (by intro h; apply ha; linear_combination h / 2)]
      ring
    refine ⟨⟨fun x hx => ?_, fun x hx x' hx' h => ?_, fun y hy => ⟨a * (1 + y) / (1 - y), ?_, back y hy⟩⟩, back, forth⟩
    · intro h
      have h1 : x + a ≠ 0 := fun h' => hx (by linarith)
      rw [div_eq_one_iff_eq h1] at h
      exact ha (by linarith)
    · rw [← forth x hx, ← forth x' hx']
      simp only at h
      rw [h]
    · intro h
      have h1 : 1 - y ≠ 0 := sub_ne_zero.2 (Ne.symm hy)
      rw [div_eq_iff h1] at h
      exact ha (by linarith)
  exact @key (1) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Inverse.moebius_bijOn (2) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : a ≠ 0), Set.BijOn (fun x => (x - a) / (x + a)) {x | x ≠ -a} {y | y ≠ 1} ∧ (∀ y, y ≠ 1 → (a * (1 + y) / (1 - y) - a) / (a * (1 + y) / (1 - y) + a) = y) ∧ (∀ x, x ≠ -a → a * (1 + (x - a) / (x + a)) / (1 - (x - a) / (x + a)) = x) := by
    intro a ha
    have back : ∀ y, y ≠ 1 → (a * (1 + y) / (1 - y) - a) / (a * (1 + y) / (1 - y) + a) = y := by
      intro y hy
      have h1 : 1 - y ≠ 0 := sub_ne_zero.2 (Ne.symm hy)
      rw [div_sub' h1, div_add' _ _ _ h1, div_div_div_cancel_right₀ h1]
      rw [div_eq_iff (by intro h; apply ha; linear_combination h / 2)]
      ring
    have forth : ∀ x, x ≠ -a → a * (1 + (x - a) / (x + a)) / (1 - (x - a) / (x + a)) = x := by
      intro x hx
      have h1 : x + a ≠ 0 := fun h => hx (by linarith)
      rw [one_add_div h1, one_sub_div h1, mul_div_assoc', div_div_div_cancel_right₀ h1]
      rw [div_eq_iff (by intro h; apply ha; linear_combination h / 2)]
      ring
    refine ⟨⟨fun x hx => ?_, fun x hx x' hx' h => ?_, fun y hy => ⟨a * (1 + y) / (1 - y), ?_, back y hy⟩⟩, back, forth⟩
    · intro h
      have h1 : x + a ≠ 0 := fun h' => hx (by linarith)
      rw [div_eq_one_iff_eq h1] at h
      exact ha (by linarith)
    · rw [← forth x hx, ← forth x' hx']
      simp only at h
      rw [h]
    · intro h
      have h1 : 1 - y ≠ 0 := sub_ne_zero.2 (Ne.symm hy)
      rw [div_eq_iff h1] at h
      exact ha (by linarith)
  exact @key (2) ((by norm_num))

example : veriphy_instance% KaluzaKlein.Sets.Inverse.moebius_bijOn (-1) ((by norm_num)) := by
  have key : ∀ (a : ℝ) (ha : a ≠ 0), Set.BijOn (fun x => (x - a) / (x + a)) {x | x ≠ -a} {y | y ≠ 1} ∧ (∀ y, y ≠ 1 → (a * (1 + y) / (1 - y) - a) / (a * (1 + y) / (1 - y) + a) = y) ∧ (∀ x, x ≠ -a → a * (1 + (x - a) / (x + a)) / (1 - (x - a) / (x + a)) = x) := by
    intro a ha
    have back : ∀ y, y ≠ 1 → (a * (1 + y) / (1 - y) - a) / (a * (1 + y) / (1 - y) + a) = y := by
      intro y hy
      have h1 : 1 - y ≠ 0 := sub_ne_zero.2 (Ne.symm hy)
      rw [div_sub' h1, div_add' _ _ _ h1, div_div_div_cancel_right₀ h1]
      rw [div_eq_iff (by intro h; apply ha; linear_combination h / 2)]
      ring
    have forth : ∀ x, x ≠ -a → a * (1 + (x - a) / (x + a)) / (1 - (x - a) / (x + a)) = x := by
      intro x hx
      have h1 : x + a ≠ 0 := fun h => hx (by linarith)
      rw [one_add_div h1, one_sub_div h1, mul_div_assoc', div_div_div_cancel_right₀ h1]
      rw [div_eq_iff (by intro h; apply ha; linear_combination h / 2)]
      ring
    refine ⟨⟨fun x hx => ?_, fun x hx x' hx' h => ?_, fun y hy => ⟨a * (1 + y) / (1 - y), ?_, back y hy⟩⟩, back, forth⟩
    · intro h
      have h1 : x + a ≠ 0 := fun h' => hx (by linarith)
      rw [div_eq_one_iff_eq h1] at h
      exact ha (by linarith)
    · rw [← forth x hx, ← forth x' hx']
      simp only at h
      rw [h]
    · intro h
      have h1 : 1 - y ≠ 0 := sub_ne_zero.2 (Ne.symm hy)
      rw [div_eq_iff h1] at h
      exact ha (by linarith)
  exact @key (-1) ((by norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Sets.Inverse.moebius_bijOn
-- test: set-moebius-test.sage

end KaluzaKlein.Sets.Inverse
