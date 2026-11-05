import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.PartI.LinearAlgebra.VectorSpaceResults
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Linear functionals"

The claims of the section "Linear functionals" of `tutorial/part-i-02-linear-algebra.qmd`: the functional `2x + y`,
coordinate functionals, rows, evaluation and integration, the kernel of `3x - y + 2z`, and the integral over the
circle evaluated on Fourier modes.
-/

namespace KaluzaKlein.LinearAlgebra.LinearFunctionals

open Matrix Module KaluzaKlein.LinearAlgebra.VectorSpaceResults

/-! ## def-linear-functional, exm-functional-coordinate -/

/-- A linear functional on `V` is a linear map `V → ℝ` (mathlib's `Module.Dual ℝ V`). -/
abbrev LinearFunctional (V : Type*) [AddCommGroup V] [Module ℝ V] := Module.Dual ℝ V

/-- The functional `(x, y) ↦ 2x + y`. -/
noncomputable def alpha : LinearFunctional (Fin 2 → ℝ) :=
  2 • LinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 0 + LinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 1

theorem alpha_apply (v : Fin 2 → ℝ) : alpha v = 2 * v 0 + v 1 := by simp [alpha]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.alpha_apply (0) := by
  norm_num [alpha]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.alpha_apply (![1, 2]) := by
  norm_num [alpha]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.alpha_apply (![-3, 1 / 2]) := by
  norm_num [alpha]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearFunctionals.alpha_apply
-- test: la-alpha-apply-test.sage

/-- The kernel of `2x + y` is the line `y = -2x`. -/
theorem mem_ker_alpha (v : Fin 2 → ℝ) : v ∈ LinearMap.ker alpha ↔ v 1 = -2 * v 0 := by
  rw [LinearMap.mem_ker, alpha_apply]
  constructor <;> intro h <;> linarith

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.mem_ker_alpha (0) := by
  rw [LinearMap.mem_ker]
  norm_num [alpha]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.mem_ker_alpha (![1, -2]) := by
  rw [LinearMap.mem_ker]
  norm_num [alpha]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.mem_ker_alpha (![1, 1]) := by
  rw [LinearMap.mem_ker]
  norm_num [alpha]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearFunctionals.mem_ker_alpha
-- test: la-mem-ker-alpha-test.sage

/-- `(1, -2)` spans the kernel: the row `(2, 1)` pairs it to zero. -/
theorem alpha_kernel_vector : !![(2 : ℝ), 1] * !![(1 : ℝ); -2] = !![(0 : ℝ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-- The level sets are parallel: `α(v + w) = α(v)` for `w` in the kernel. -/
theorem alpha_level_sets (v w : Fin 2 → ℝ) (hw : w ∈ LinearMap.ker alpha) : alpha (v + w) = alpha v := by
  rw [map_add, LinearMap.mem_ker.mp hw, add_zero]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.alpha_level_sets (0) (0) ((by rw [LinearMap.mem_ker]; norm_num [alpha])) := by
  norm_num [alpha]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.alpha_level_sets (![1, 2]) (![1, -2]) ((by rw [LinearMap.mem_ker]; norm_num [alpha])) := by
  norm_num [alpha]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.alpha_level_sets (![-3, 1 / 2]) (![-1 / 2, 1]) ((by rw [LinearMap.mem_ker]; norm_num [alpha])) := by
  norm_num [alpha]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearFunctionals.alpha_level_sets
-- test: la-alpha-level-sets-test.sage

/-- The `i`-th coordinate is a linear functional on `ℝⁿ`. -/
theorem coordinate_functional (n : ℕ) (i : Fin n) (v : Fin n → ℝ) :
    LinearMap.proj (R := ℝ) (φ := fun _ : Fin n => ℝ) i v = v i := rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.coordinate_functional (1) (0) (![5]) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.coordinate_functional (3) (1) (![5, 7, 9]) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.coordinate_functional (4) (3) (![1, -2, 1 / 2, 3]) := by
  rfl

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearFunctionals.coordinate_functional
-- test: la-coordinate-functional-test.sage

/-! ## exm-functional-row, exr-functional-rows -/

/-- Every functional on `ℝⁿ` is `v ↦ a_i v^i` with `a_i = α(e_i)`. -/
theorem functional_is_row {n : ℕ} (α : Module.Dual ℝ (Fin n → ℝ)) (v : Fin n → ℝ) :
    α v = ∑ i, v i * α (fun j => if i = j then 1 else 0) := by
  simpa [smul_eq_mul] using LinearMap.pi_apply_eq_sum_univ α v

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.functional_is_row (2) (0) (![1, 2]) := by
  simp [Fin.sum_univ_succ, Matrix.mulVec, dotProduct]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.functional_is_row (2) (LinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 1) (![1, 2]) := by
  simp [Fin.sum_univ_succ, Matrix.mulVec, dotProduct]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.functional_is_row (3) (LinearMap.proj (R := ℝ) (φ := fun _ : Fin 1 => ℝ) 0 ∘ₗ Matrix.toLin' !![3, -1, 2]) (![1, 0, -2]) := by
  simp [Fin.sum_univ_succ, Matrix.mulVec, dotProduct]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearFunctionals.functional_is_row
-- test: la-functional-is-row-test.sage

/-- The row is unique: two rows giving the same functional are equal (evaluate at `e_j`). -/
theorem row_unique {n : ℕ} (a b : Fin n → ℝ) (h : ∀ v : Fin n → ℝ, a ⬝ᵥ v = b ⬝ᵥ v) : a = b := by
  ext j
  simpa [dotProduct, Pi.single_apply] using h (Pi.single j 1)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.row_unique (0) (![]) (![]) ((fun _ => rfl)) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.row_unique (2) (![2, 1]) (![2, 1]) ((fun _ => rfl)) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.row_unique (3) (![3, -1, 2]) (![3, -1, 2]) ((fun _ => rfl)) := by
  rfl

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearFunctionals.row_unique
-- test: la-row-unique-test.sage

/-! ## exm-functional-evaluation -/

/-- Evaluation at `0` is linear. -/
theorem evaluation_linear (f g : ℝ → ℝ) (c : ℝ) : (f + c • g) 0 = f 0 + c * g 0 := by simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.evaluation_linear (0) (0) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.evaluation_linear ((fun x => x ^ 2 + 1)) (Real.cos) (3) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.evaluation_linear (Real.exp) ((fun x => 2 * x - 5)) (-1 / 2) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearFunctionals.evaluation_linear
-- test: la-evaluation-linear-test.sage

/-- Integration over `[0, 2π]` is linear on continuous functions. -/
theorem integration_linear (f g : ℝ → ℝ) (c : ℝ) (hf : Continuous f) (hg : Continuous g) :
    ∫ φ in (0 : ℝ)..(2 * Real.pi), (f φ + c * g φ) =
      (∫ φ in (0 : ℝ)..(2 * Real.pi), f φ) + c * ∫ φ in (0 : ℝ)..(2 * Real.pi), g φ := by
  rw [intervalIntegral.integral_add (hf.intervalIntegrable _ _) ((hg.intervalIntegrable _ _).const_mul c),
    intervalIntegral.integral_const_mul]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.integration_linear ((fun _ => 0)) ((fun _ => 0)) (0) (continuous_const) (continuous_const) := by
  simp

attribute [veriphy_tested oracle_test 1 3] KaluzaKlein.LinearAlgebra.LinearFunctionals.integration_linear
-- test: la-integration-linear-test.sage

/-! ## exr-functional-kernel -/

/-- The functional `3x - y + 2z`. -/
noncomputable def beta : Module.Dual ℝ (Fin 3 → ℝ) := Matrix.toLin' !![(3 : ℝ), -1, 2] |>.comp LinearMap.id
  |> fun f => (LinearMap.proj (R := ℝ) (φ := fun _ : Fin 1 => ℝ) 0).comp f

theorem beta_apply (v : Fin 3 → ℝ) : beta v = 3 * v 0 - v 1 + 2 * v 2 := by
  simp [beta, Matrix.mulVec, dotProduct, Fin.sum_univ_three]
  ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.beta_apply (0) := by
  norm_num [beta, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.beta_apply (![1, 3, 0]) := by
  norm_num [beta, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.beta_apply (![1 / 2, -1, 2]) := by
  norm_num [beta, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearFunctionals.beta_apply
-- test: la-beta-apply-test.sage

/-- Its kernel, the plane `3x - y + 2z = 0`, has dimension `3 - 1 = 2`. -/
theorem finrank_ker_beta : finrank ℝ (LinearMap.ker beta) = 2 := by
  veriphy_certified
    ( have h :=
        Veriphy.Linear.finrank_ker_add_one (KaluzaKlein.LinearAlgebra.LinearFunctionals.beta) (![(1 / 3), 0, 0])
          (by rw [KaluzaKlein.LinearAlgebra.LinearFunctionals.beta_apply]; norm_num)
      simp [Module.finrank_matrix, Module.finrank_fin_fun] at h
      omega)
  -- certificate: la-dim-ker.sage

/-- `(1, 3, 0)` and `(0, 2, 1)` lie in the kernel. -/
theorem beta_kernel_vectors : !![(3 : ℝ), -1, 2] * !![(1 : ℝ), 0; 3, 2; 0, 1] = !![(0 : ℝ), 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-functional-kernel.sage

/-- The level set `β = c` is the kernel translated: `β(p + w) = c` for `β(p) = c`, `w` in the kernel. -/
theorem beta_level_set (p w : Fin 3 → ℝ) (c : ℝ) (hp : beta p = c) (hw : w ∈ LinearMap.ker beta) :
    beta (p + w) = c := by
  rw [map_add, LinearMap.mem_ker.mp hw, add_zero, hp]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.beta_level_set (0) (0) (0) ((by rw [beta_apply]; norm_num)) ((by rw [LinearMap.mem_ker, beta_apply]; norm_num)) := by
  rw [beta_apply]
  norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.beta_level_set (![1, 0, 0]) (![1, 3, 0]) (3) ((by rw [beta_apply]; norm_num)) ((by rw [LinearMap.mem_ker, beta_apply]; norm_num)) := by
  rw [beta_apply]
  norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearFunctionals.beta_level_set (![0, 0, 1]) (![0, 2, 1]) (2) ((by rw [beta_apply]; norm_num)) ((by rw [LinearMap.mem_ker, beta_apply]; norm_num)) := by
  rw [beta_apply]
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearFunctionals.beta_level_set
-- test: la-beta-level-set-test.sage

/-! ## exr-functional-integration -/

theorem integral_one : ∫ _φ in (0 : ℝ)..(2 * Real.pi), (1 : ℝ) = 2 * Real.pi := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * Real.pi : ℝ),
          HasDerivAt (fun x_veriphy => (x_veriphy : ℝ)) ((fun _φ => 1) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((hasDerivAt_id' (x_veriphy : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (x_veriphy : ℝ)) (f' := fun _φ => 1) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: la-integral.sage

theorem integral_cos_circle : ∫ φ in (0 : ℝ)..(2 * Real.pi), Real.cos φ = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * Real.pi : ℝ),
          HasDerivAt (fun x_veriphy => (Real.sin x_veriphy : ℝ)) ((fun φ => Real.cos φ) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((hasDerivAt_id' (x_veriphy : ℝ)).sin)).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (Real.sin x_veriphy : ℝ)) (f' :=
          fun φ => Real.cos φ) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he1 : Real.sin (((2 * Real.pi) : ℝ)) = 0 := by
        rw [show (((2 * Real.pi) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: la-integral.sage

theorem integral_sin_circle : ∫ φ in (0 : ℝ)..(2 * Real.pi), Real.sin φ = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * Real.pi : ℝ),
          HasDerivAt (fun x_veriphy => (-(Real.cos x_veriphy) : ℝ)) ((fun φ => Real.sin φ) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-1 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; ring1
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-(Real.cos x_veriphy) : ℝ)) (f' :=
          fun φ => Real.sin φ) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((2 * Real.pi) : ℝ)) = 1 := by
        rw [show (((2 * Real.pi) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: la-integral.sage

theorem integral_sin_two : ∫ φ in (0 : ℝ)..(2 * Real.pi), Real.sin (2 * φ) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * Real.pi : ℝ),
          HasDerivAt (fun x_veriphy => (-(Real.cos (2 * x_veriphy)) / 2 : ℝ)) ((fun φ => Real.sin (2 * φ)) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).cos).const_mul (-(1 / 2) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; ring1
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-(Real.cos (2 * x_veriphy)) / 2 : ℝ))
          (f' := fun φ => Real.sin (2 * φ)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos ((2 * (0) : ℝ)) = 1 := by simp
      have he1 : Real.cos ((2 * (2 * Real.pi) : ℝ)) = 1 := by
        rw [show ((2 * (2 * Real.pi) : ℝ) : ℝ) = ((4 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi];
        norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: la-integral.sage

theorem integral_cos_sq_circle : ∫ φ in (0 : ℝ)..(2 * Real.pi), Real.cos φ ^ 2 = Real.pi := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * Real.pi : ℝ),
          HasDerivAt (fun x_veriphy => (x_veriphy / 2 + ((Real.sin (2 * x_veriphy)) / 4) : ℝ))
            ((fun φ => Real.cos φ ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / 2 : ℝ)).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).sin).const_mul
                      (1 / 4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (simp only [Real.cos_sq, Real.sin_sq_eq_half_sub]; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (x_veriphy / 2 + ((Real.sin (2 * x_veriphy)) / 4) : ℝ)) (f' := fun φ => Real.cos φ ^ 2) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin ((2 * (0) : ℝ)) = 0 := by simp
      have he1 : Real.sin ((2 * (2 * Real.pi) : ℝ)) = 0 := by
        rw [show ((2 * (2 * Real.pi) : ℝ) : ℝ) = ((4 : ℕ) : ℝ) * Real.pi by push_cast; ring];
        exact Real.sin_nat_mul_pi 4
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: la-integral.sage

/-- Linearity on `f = 3 + 2 cos φ`, `g = 1 - sin 2φ`: `I(f) = 6π`, `I(g) = 2π`, `I(f + g) = 8π`. -/
theorem integral_f_g :
    (∫ φ in (0 : ℝ)..(2 * Real.pi), (3 + 2 * Real.cos φ)) = 6 * Real.pi ∧
      (∫ φ in (0 : ℝ)..(2 * Real.pi), (1 - Real.sin (2 * φ))) = 2 * Real.pi ∧
      (∫ φ in (0 : ℝ)..(2 * Real.pi), ((3 + 2 * Real.cos φ) + (1 - Real.sin (2 * φ)))) = 8 * Real.pi := by
  refine ⟨?_, ?_, ?_⟩
  · veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * Real.pi : ℝ),
            HasDerivAt (fun x_veriphy => (3 * x_veriphy + (2 * (Real.sin x_veriphy)) : ℝ))
              ((fun φ => 3 + 2 * Real.cos φ) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 : ℝ)).add
                      (((hasDerivAt_id' (x_veriphy : ℝ)).sin).const_mul (2 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
            (3 * x_veriphy + (2 * (Real.sin x_veriphy)) : ℝ)) (f' := fun φ => 3 + 2 * Real.cos φ) hF
            (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        have he0 : Real.sin (((0) : ℝ)) = 0 := by simp
        have he1 : Real.sin (((2 * Real.pi) : ℝ)) = 0 := by
          rw [show (((2 * Real.pi) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
        simp only [he0, he1]
        all_goals ring1)
    -- certificate: la-integral.sage
  · veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * Real.pi : ℝ),
            HasDerivAt (fun x_veriphy => (x_veriphy + ((Real.cos (2 * x_veriphy)) / 2) : ℝ))
              ((fun φ => 1 - Real.sin (2 * φ)) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            ((((hasDerivAt_id' (x_veriphy : ℝ)).add
                      ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).cos).const_mul
                        (1 / 2 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
            (x_veriphy + ((Real.cos (2 * x_veriphy)) / 2) : ℝ)) (f' := fun φ => 1 - Real.sin (2 * φ)) hF
            (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        have he0 : Real.cos ((2 * (0) : ℝ)) = 1 := by simp
        have he1 : Real.cos ((2 * (2 * Real.pi) : ℝ)) = 1 := by
          rw [show ((2 * (2 * Real.pi) : ℝ) : ℝ) = ((4 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi];
          norm_num
        simp only [he0, he1]
        all_goals (norm_num; done))
    -- certificate: la-integral.sage
  · veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * Real.pi : ℝ),
            HasDerivAt
              (fun x_veriphy => (4 * x_veriphy + ((Real.cos (2 * x_veriphy)) / 2) + (2 * (Real.sin x_veriphy)) : ℝ))
              ((fun φ => 3 + 2 * Real.cos φ + (1 - Real.sin (2 * φ))) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            ((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (4 : ℝ)).add
                          ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).cos).const_mul (1 / 2 : ℝ))).add
                      (((hasDerivAt_id' (x_veriphy : ℝ)).sin).const_mul (2 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
            (4 * x_veriphy + ((Real.cos (2 * x_veriphy)) / 2) + (2 * (Real.sin x_veriphy)) : ℝ)) (f' := fun φ =>
            3 + 2 * Real.cos φ + (1 - Real.sin (2 * φ))) hF (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        have he0 : Real.cos ((2 * (0) : ℝ)) = 1 := by simp
        have he1 : Real.cos ((2 * (2 * Real.pi) : ℝ)) = 1 := by
          rw [show ((2 * (2 * Real.pi) : ℝ) : ℝ) = ((4 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi];
          norm_num
        have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
        have he3 : Real.sin (((2 * Real.pi) : ℝ)) = 0 := by
          rw [show (((2 * Real.pi) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
        simp only [he0, he1, he2, he3]
        all_goals ring1)
    -- certificate: la-integral.sage

end KaluzaKlein.LinearAlgebra.LinearFunctionals
