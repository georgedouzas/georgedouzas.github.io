import Mathlib
import Veriphy.Kit.Levels

/-!
# Differential equations, closing exercises

The claims of the section "Exercises" of `tutorial/part-i-05-differential-equations.qmd`: the damped pendulum near
its equilibria, the string from its modes to its Green's function, and the Klein-Gordon equation from plane waves to
the Yukawa potential and the one-dimensional retarded Green's function (with `J₀` truncated to a polynomial).
-/

open Real Filter Topology

namespace KaluzaKlein.DifferentialEquations

/-! Helper lemmas of the hand proofs below (private: each module carries its own copy). -/

private lemma int_x_sin (n ξ : ℝ) (hn : n ≠ 0) :
    ∫ x in (0 : ℝ)..ξ, x * Real.sin (n * x) = Real.sin (n * ξ) / n ^ 2 - ξ * Real.cos (n * ξ) / n := by
  have hF : ∀ x : ℝ, HasDerivAt (fun x => -x * Real.cos (n * x) / n + Real.sin (n * x) / n ^ 2) (x * Real.sin (n * x)) x := by
    intro x
    have hneg : HasDerivAt (fun x : ℝ => -x) (-1) x := hasDerivAt_neg x
    have h1' : HasDerivAt (fun x => -x * Real.cos (n * x)) _ x := hneg.mul ((hasDerivAt_id' x).const_mul n).cos
    have h1 : HasDerivAt (fun x => -x * Real.cos (n * x) / n) _ x := h1'.div_const n
    have h2 : HasDerivAt (fun x => Real.sin (n * x) / n ^ 2) _ x := (((hasDerivAt_id' x).const_mul n).sin).div_const (n ^ 2)
    refine (h1.add h2).congr_deriv ?_
    first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hF x) (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [mul_zero, neg_zero, zero_mul, zero_div, Real.sin_zero, add_zero, sub_zero]
  ring

end KaluzaKlein.DifferentialEquations

namespace KaluzaKlein.DifferentialEquations.Exercises

/-! ## exr-de-pendulum-linearised -/

/-- The Jacobian's entry `∂_θ(-sin θ - γω) = -cos θ`. -/
theorem pendulum_jacobian (θ ω γ : ℝ) : deriv (fun s => -Real.sin s - γ * ω) θ = -Real.cos θ := by
  veriphy_certified
    ( have hd : HasDerivAt (fun s => -sin s - γ * ω) ((-(Real.cos θ) : ℝ)) (θ) :=
        by
        convert ((hasDerivAt_const (θ : ℝ) (-γ * ω : ℝ)).add (((hasDerivAt_id' (θ : ℝ)).sin).const_mul (-1 : ℝ))) using
          1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: de-derivative.sage

/-- At `(0, 0)` (with `γ = 0.3`): trace `-0.3` … -/
theorem pendulum_trace : Matrix.trace !![(0 : ℝ), 1; -Real.cos 0, -0.3] = -0.3 := by
  veriphy_certified (norm_num [Matrix.trace_fin_two_of] <;> ring1)
  -- certificate: la-matrix.sage

/-- … determinant `1`: a spiral sink … -/
theorem pendulum_det : Matrix.det !![(0 : ℝ), 1; -Real.cos 0, -0.3] = 1 := by
  veriphy_certified (norm_num [Matrix.det_fin_two_of] <;> ring1)
  -- certificate: la-matrix.sage

/-- … with eigenvalues `-0.15 ± i√0.9775` (`0.15 = 3/20`, `0.9775 = 391/400`, `γ = 0.3 = 3/10`). -/
theorem pendulum_eigenvalues (i s : ℂ) (hi : i ^ 2 = -1) (hs : s ^ 2 = 391 / 400) :
    (-3 / 20 + i * s) ^ 2 + 3 / 10 * (-3 / 20 + i * s) + 1 = 0 := by
  veriphy_certified (linear_combination (391 / 400) * hi + (i ^ 2) * hs)
  -- certificate: de-poly.sage

/-- `ω_d = √(1 - 0.0225) ≈ 0.989`. -/
theorem pendulum_damped_frequency : |Real.sqrt (1 - 9 / 400) - 989 / 1000| < (1 / 2000 : ℝ) := by
  veriphy_certified
    ( rw [abs_lt]
      constructor
      · rw [lt_sub_iff_add_lt', Real.lt_sqrt (by norm_num)]; norm_num
      · rw [sub_lt_iff_lt_add', Real.sqrt_lt' (by norm_num)]; norm_num)
  -- certificate: de-rewrite.sage

/-- At `(π, 0)` the determinant is `-1`: a saddle. -/
theorem pendulum_upright_det : Matrix.det !![(0 : ℝ), 1; -Real.cos Real.pi, -0.3] = -1 := by
  veriphy_certified (norm_num [Matrix.det_fin_two_of] <;> ring1)
  -- certificate: la-matrix.sage

/-- `e^{tA}` by the spiral formula: `B = A + 0.15` has `B² = -0.9775`. -/
theorem pendulum_spiral_B :
    !![(3 / 20 : ℝ), 1; -1, -3 / 20] * !![(3 / 20 : ℝ), 1; -1, -3 / 20] = !![-391 / 400, 0; 0, -391 / 400] := by
  veriphy_certified
    ( ext i j
      fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply])
  -- certificate: la-matrix.sage

/-- The linearisation error: `θ - θ³/6 < sin θ ≤ θ` for `θ > 0`, an error of order `θ³/6`. -/
theorem pendulum_sine_error (θ : ℝ) (h0 : 0 < θ) : θ - θ ^ 3 / 6 < Real.sin θ ∧ Real.sin θ ≤ θ :=
  ⟨Real.sin_gt_sub_cube h0, Real.sin_le h0.le⟩
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.Exercises.pendulum_sine_error
-- test: de-pendulum-sine-test.sage

/-! ## exr-de-string-greens -/

/-- (a) On `[0, π]`: `u_a = x`, `u_b = π - x`, `W = x·(-1) - 1·(π - x) = -π`. -/
theorem string_wronskian (x : ℝ) : x * (-1) - 1 * (Real.pi - x) = -Real.pi := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- The sine coefficients of `G(·, ξ) = x_<(π - x_>)/π`, `n = 1`: `∫₀^ξ x(π - ξ)/π sin x dx` … -/
theorem string_coeff1_left (ξ : ℝ) :
    ∫ x in (0 : ℝ)..ξ, x * (Real.pi - ξ) / Real.pi * Real.sin (1 * x) =
      (Real.pi - ξ) / Real.pi * (Real.sin (1 * ξ) / 1 ^ 2 - ξ * Real.cos (1 * ξ) / 1) := by
  rw [show (fun x => x * (Real.pi - ξ) / Real.pi * Real.sin (1 * x)) = fun x => (Real.pi - ξ) / Real.pi * (x * Real.sin (1 * x)) from
    funext fun x => by ring, intervalIntegral.integral_const_mul, int_x_sin 1 ξ one_ne_zero]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.Exercises.string_coeff1_left
-- test: de-string-coeff1-left-test.sage

/-- … `∫_ξ^π ξ(π - x)/π sin x dx` … -/
theorem string_coeff1_right (ξ : ℝ) :
    ∫ x in ξ..Real.pi, ξ * (Real.pi - x) / Real.pi * Real.sin (1 * x) =
      ξ / Real.pi * ((Real.pi - ξ) * Real.cos (1 * ξ) / 1 + Real.sin (1 * ξ) / 1 ^ 2) := by
  veriphy_certified
    ( have hF :
        ∀ i0 ∈ Set.uIcc (ξ : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (-(Real.pi * (Real.cos i0) - (i0 * (Real.cos i0)) + (Real.sin i0)) * ξ / Real.pi : ℝ))
            ((fun x => ξ * (π - x) / π * sin (1 * x)) i0) i0 :=
        by
        intro i0 _
        refine
          ((((((((hasDerivAt_id' (i0 : ℝ)).cos).const_mul (Real.pi : ℝ)).add
                            (((hasDerivAt_id' (i0 : ℝ)).mul ((hasDerivAt_id' (i0 : ℝ)).cos)).const_mul (-1 : ℝ))).add
                        ((hasDerivAt_id' (i0 : ℝ)).sin)).const_mul
                    (-ξ / Real.pi : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (field_simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero, Real.cos_zero, Real.sin_two_pi,
        Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero, zero_add]
      ring1)
  -- certificate: de-integral-trig.sage

/-- `n = 2`: `∫₀^ξ x(π - ξ)/π sin 2x dx` … -/
theorem string_coeff2_left (ξ : ℝ) :
    ∫ x in (0 : ℝ)..ξ, x * (Real.pi - ξ) / Real.pi * Real.sin (2 * x) =
      (Real.pi - ξ) / Real.pi * (Real.sin (2 * ξ) / 2 ^ 2 - ξ * Real.cos (2 * ξ) / 2) := by
  rw [show (fun x => x * (Real.pi - ξ) / Real.pi * Real.sin (2 * x)) = fun x => (Real.pi - ξ) / Real.pi * (x * Real.sin (2 * x)) from
    funext fun x => by ring, intervalIntegral.integral_const_mul, int_x_sin 2 ξ two_ne_zero]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.Exercises.string_coeff2_left
-- test: de-string-coeff2-left-test.sage

/-- … `∫_ξ^π ξ(π - x)/π sin 2x dx` … -/
theorem string_coeff2_right (ξ : ℝ) :
    ∫ x in ξ..Real.pi, ξ * (Real.pi - x) / Real.pi * Real.sin (2 * x) =
      ξ / Real.pi * ((Real.pi - ξ) * Real.cos (2 * ξ) / 2 + Real.sin (2 * ξ) / 2 ^ 2) := by
  veriphy_certified
    ( have c2 : ∀ y : ℝ, Real.cos (2 * y) = Real.cos y ^ 2 - Real.sin y ^ 2 :=
        by
        intro y
        rw [show (2 : ℝ) * y = y + y by ring, Real.cos_add]
        ring
      have s2 : ∀ y : ℝ, Real.sin (2 * y) = 2 * Real.cos y * Real.sin y :=
        by
        intro y
        rw [show (2 : ℝ) * y = y + y by ring, Real.sin_add]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (ξ : ℝ) (π : ℝ),
          HasDerivAt
            (fun i0 =>
              (-(2 * Real.pi * (Real.cos (2 * i0)) - (2 * i0 * (Real.cos (2 * i0))) + (Real.sin (2 * i0))) * ξ /
                  (4 * Real.pi) :
                ℝ))
            ((fun x => ξ * (π - x) / π * sin (2 * x)) i0) i0 :=
        by
        intro i0 _
        refine
          (((((((((hasDerivAt_id' (i0 : ℝ)).const_mul (2 : ℝ)).cos).const_mul (2 * Real.pi : ℝ)).add
                            (((hasDerivAt_id' (i0 : ℝ)).mul
                                  (((hasDerivAt_id' (i0 : ℝ)).const_mul (2 : ℝ)).cos)).const_mul
                              (-2 : ℝ))).add
                        (((hasDerivAt_id' (i0 : ℝ)).const_mul (2 : ℝ)).sin)).const_mul
                    (-ξ / (4 * Real.pi) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero, Real.cos_zero, Real.sin_two_pi,
        Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero, zero_add]
      ring1)
  -- certificate: de-integral-trig.sage

/-- … and the two pieces add to `sin nξ/n²`, so `(2/π)∫₀^π G sin nx = (2/π) sin nξ/n²`, the coefficient of the
eigenfunction series `(2/π)Σ sin nx sin nξ/n²`. -/
theorem string_coeff_sum (n ξ : ℝ) (hn : n ≠ 0) :
    (Real.pi - ξ) / Real.pi * (Real.sin (n * ξ) / n ^ 2 - ξ * Real.cos (n * ξ) / n) +
      ξ / Real.pi * ((Real.pi - ξ) * Real.cos (n * ξ) / n + Real.sin (n * ξ) / n ^ 2) = Real.sin (n * ξ) / n ^ 2 := by
  veriphy_certified
    ( field_simp
      ring)
  -- certificate: de-rewrite.sage

/-- (b) The static shape under `f = 1`: `∫₀ˣ ξ(π - x)/π dξ + ∫ₓ^π x(π - ξ)/π dξ = x(π - x)/2`; the first piece … -/
theorem string_static_left (x : ℝ) : ∫ ξ in (0 : ℝ)..x, ξ * (Real.pi - x) / Real.pi = x ^ 2 * (Real.pi - x) / (2 * Real.pi) := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (x : ℝ),
          HasDerivAt (fun x_veriphy => ((Real.pi - x) * (x_veriphy ^ 2) / (2 * Real.pi) : ℝ))
            ((fun ξ => ξ * (π - x) / π) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul
                    ((Real.pi - x) / (2 * Real.pi) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((Real.pi - x) * (x_veriphy ^ 2) / (2 * Real.pi) : ℝ)) (f' := fun ξ => ξ * (π - x) / π) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: de-integral.sage

/-- … the second piece … -/
theorem string_static_right (x : ℝ) :
    ∫ ξ in x..Real.pi, x * (Real.pi - ξ) / Real.pi = x * (Real.pi - x) ^ 2 / (2 * Real.pi) := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (x : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => ((2 * Real.pi * x_veriphy - (x_veriphy ^ 2)) * x / (2 * Real.pi) : ℝ))
            ((fun ξ => x * (π - ξ) / π) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 * Real.pi : ℝ)).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (-1 : ℝ))).const_mul
                    (x / (2 * Real.pi) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((2 * Real.pi * x_veriphy - (x_veriphy ^ 2)) * x / (2 * Real.pi) : ℝ)) (f' := fun ξ => x * (π - ξ) / π) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: de-integral.sage

/-- … and their sum. -/
theorem string_static_sum (x : ℝ) :
    x ^ 2 * (Real.pi - x) / (2 * Real.pi) + x * (Real.pi - x) ^ 2 / (2 * Real.pi) = x * (Real.pi - x) / 2 := by
  veriphy_certified
    ( field_simp
      ring)
  -- certificate: de-rewrite.sage

/-- (c) Released from `x(π - x)/2`: the coefficients are half those of `x(π - x)`, `4/(πn³)`. -/
theorem string_release_coefficient (n : ℝ) (hn : n ≠ 0) : 1 / 2 * (8 / (Real.pi * n ^ 3)) = 4 / (Real.pi * n ^ 3) := by
  veriphy_certified ((field_simp; ring))
  -- certificate: de-rewrite.sage

/-- (d) The energy of a mode `b sin x cos t`: `½∫₀^π(u_t² + u_x²) = (π/4)b²(sin²t + cos²t)` … -/
theorem string_mode_energy (b t : ℝ) :
    1 / 2 * ∫ x in (0 : ℝ)..Real.pi, ((b * Real.sin x * Real.sin t) ^ 2 + (b * Real.cos x * Real.cos t) ^ 2) =
      Real.pi / 4 * b ^ 2 * (Real.sin t ^ 2 + Real.cos t ^ 2) := by
  have e : (fun x => (b * Real.sin x * Real.sin t) ^ 2 + (b * Real.cos x * Real.cos t) ^ 2) =
      fun x => (b ^ 2 * Real.sin t ^ 2) * Real.sin x ^ 2 + (b ^ 2 * Real.cos t ^ 2) * Real.cos x ^ 2 := by
    funext x; ring
  rw [e, intervalIntegral.integral_add (by apply Continuous.intervalIntegrable; fun_prop)
    (by apply Continuous.intervalIntegrable; fun_prop), intervalIntegral.integral_const_mul,
    intervalIntegral.integral_const_mul, integral_sin_sq, integral_cos_sq]
  simp only [Real.sin_pi, Real.sin_zero, Real.cos_pi, Real.cos_zero, mul_zero, zero_mul, sub_zero, zero_sub]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.Exercises.string_mode_energy
-- test: de-string-mode-energy-test.sage

/-- … `= (π/4)b²`, independent of `t`. -/
theorem string_mode_energy_constant (b t : ℝ) :
    Real.pi / 4 * b ^ 2 * (Real.sin t ^ 2 + Real.cos t ^ 2) = Real.pi / 4 * b ^ 2 := by
  veriphy_certified (rw [Real.sin_sq_add_cos_sq, mul_one])
  -- certificate: de-rewrite.sage

/-! ## exr-de-klein-gordon-modes -/

/-- (a) With `u = e^{-iωt}v`: `(-ω² - ∇² + m²)v = 0` is `-∇²v = (ω² - m²)v`. -/
theorem kg_helmholtz (ω m v lv : ℝ) (h : -ω ^ 2 * v - lv + m ^ 2 * v = 0) : -lv = (ω ^ 2 - m ^ 2) * v := by
  veriphy_certified (linear_combination h)
  -- certificate: de-poly.sage

/-- (b) The box modes `sin n₁x sin n₂y sin n₃z` have `-∇² = n₁² + n₂² + n₃²`. -/
theorem kg_box_mode (n₁ n₂ n₃ x y z : ℝ) :
    -(deriv (fun s => deriv (fun r => Real.sin (n₁ * r) * Real.sin (n₂ * y) * Real.sin (n₃ * z)) s) x +
      deriv (fun s => deriv (fun r => Real.sin (n₁ * x) * Real.sin (n₂ * r) * Real.sin (n₃ * z)) s) y +
      deriv (fun s => deriv (fun r => Real.sin (n₁ * x) * Real.sin (n₂ * y) * Real.sin (n₃ * r)) s) z) =
      (n₁ ^ 2 + n₂ ^ 2 + n₃ ^ 2) * (Real.sin (n₁ * x) * Real.sin (n₂ * y) * Real.sin (n₃ * z)) := by
  veriphy_certified
    ( have i0 :
        deriv (fun (r : ℝ) => (sin (n₁ * r) * sin (n₂ * y) * sin (n₃ * z) : ℝ)) = fun (d1 : ℝ) =>
          (n₁ * (Real.cos (d1 * n₁)) * (Real.sin (z * n₃)) * (Real.sin (n₂ * y)) : ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun (r : ℝ) => (sin (n₁ * r) * sin (n₂ * y) * sin (n₃ * z) : ℝ))
            ((n₁ * (Real.cos (d1 * n₁)) * (Real.sin (z * n₃)) * (Real.sin (n₂ * y)) : ℝ)) (d1) :=
          by
          refine
            ((((((hasDerivAt_id' (d1 : ℝ)).const_mul (n₁ : ℝ)).sin).const_mul
                      ((Real.sin (z * n₃)) * (Real.sin (n₂ * y)) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce;
            first
            | rfl
            |
              (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                  Pi.const_apply, Pi.smul_apply, smul_eq_mul];
                ring_nf; done)
            | (ring_nf; done)
            | (simp; ring_nf; done)
            | (simp; done)
            | (field_simp; ring_nf; done)
          ·
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; done)
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (simp; ring_nf; done)
            | (norm_num; done)
            | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt
          (fun s => (fun (d1 : ℝ) => (n₁ * (Real.cos (d1 * n₁)) * (Real.sin (z * n₃)) * (Real.sin (n₂ * y)) : ℝ)) s)
          ((-(n₁ ^ 2) * (Real.sin (x * n₁)) * (Real.sin (z * n₃)) * (Real.sin (n₂ * y)) : ℝ)) (x) :=
        by
        refine
          ((((((hasDerivAt_id' (x : ℝ)).const_mul (n₁ : ℝ)).cos).const_mul
                    (n₁ * (Real.sin (z * n₃)) * (Real.sin (n₂ * y)) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce;
          first
          | rfl
          |
            (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                Pi.const_apply, Pi.smul_apply, smul_eq_mul];
              ring_nf; done)
          | (ring_nf; done)
          | (simp; ring_nf; done)
          | (simp; done)
          | (field_simp; ring_nf; done)
        ·
          first
          | rfl
          | ring1
          | (ring_nf; done)
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (simp; ring_nf; done)
          | (norm_num; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      try rw [h1.deriv]
      have i2 :
        deriv (fun (r : ℝ) => (sin (n₁ * x) * sin (n₂ * r) * sin (n₃ * z) : ℝ)) = fun (d1 : ℝ) =>
          (n₂ * (Real.cos (d1 * n₂)) * (Real.sin (x * n₁)) * (Real.sin (z * n₃)) : ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun (r : ℝ) => (sin (n₁ * x) * sin (n₂ * r) * sin (n₃ * z) : ℝ))
            ((n₂ * (Real.cos (d1 * n₂)) * (Real.sin (x * n₁)) * (Real.sin (z * n₃)) : ℝ)) (d1) :=
          by
          refine
            ((((((hasDerivAt_id' (d1 : ℝ)).const_mul (n₂ : ℝ)).sin).const_mul
                      ((Real.sin (x * n₁)) * (Real.sin (z * n₃)) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce;
            first
            | rfl
            |
              (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                  Pi.const_apply, Pi.smul_apply, smul_eq_mul];
                ring_nf; done)
            | (ring_nf; done)
            | (simp; ring_nf; done)
            | (simp; done)
            | (field_simp; ring_nf; done)
          ·
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; done)
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (simp; ring_nf; done)
            | (norm_num; done)
            | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
        exact hd.deriv
      rw [i2]
      have h3 :
        HasDerivAt
          (fun s => (fun (d1 : ℝ) => (n₂ * (Real.cos (d1 * n₂)) * (Real.sin (x * n₁)) * (Real.sin (z * n₃)) : ℝ)) s)
          ((-(n₂ ^ 2) * (Real.sin (x * n₁)) * (Real.sin (z * n₃)) * (Real.sin (n₂ * y)) : ℝ)) (y) :=
        by
        refine
          ((((((hasDerivAt_id' (y : ℝ)).const_mul (n₂ : ℝ)).cos).const_mul
                    (n₂ * (Real.sin (x * n₁)) * (Real.sin (z * n₃)) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce;
          first
          | rfl
          |
            (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                Pi.const_apply, Pi.smul_apply, smul_eq_mul];
              ring_nf; done)
          | (ring_nf; done)
          | (simp; ring_nf; done)
          | (simp; done)
          | (field_simp; ring_nf; done)
        ·
          first
          | rfl
          | ring1
          | (ring_nf; done)
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (simp; ring_nf; done)
          | (norm_num; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      try rw [h3.deriv]
      have i4 :
        deriv (fun (r : ℝ) => (sin (n₁ * x) * sin (n₂ * y) * sin (n₃ * r) : ℝ)) = fun (d1 : ℝ) =>
          (n₃ * (Real.cos (d1 * n₃)) * (Real.sin (x * n₁)) * (Real.sin (n₂ * y)) : ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun (r : ℝ) => (sin (n₁ * x) * sin (n₂ * y) * sin (n₃ * r) : ℝ))
            ((n₃ * (Real.cos (d1 * n₃)) * (Real.sin (x * n₁)) * (Real.sin (n₂ * y)) : ℝ)) (d1) :=
          by
          refine
            ((((((hasDerivAt_id' (d1 : ℝ)).const_mul (n₃ : ℝ)).sin).const_mul
                      ((Real.sin (x * n₁)) * (Real.sin (n₂ * y)) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce;
            first
            | rfl
            |
              (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                  Pi.const_apply, Pi.smul_apply, smul_eq_mul];
                ring_nf; done)
            | (ring_nf; done)
            | (simp; ring_nf; done)
            | (simp; done)
            | (field_simp; ring_nf; done)
          ·
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; done)
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (simp; ring_nf; done)
            | (norm_num; done)
            | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
        exact hd.deriv
      rw [i4]
      have h5 :
        HasDerivAt
          (fun s => (fun (d1 : ℝ) => (n₃ * (Real.cos (d1 * n₃)) * (Real.sin (x * n₁)) * (Real.sin (n₂ * y)) : ℝ)) s)
          ((-(n₃ ^ 2) * (Real.sin (x * n₁)) * (Real.sin (z * n₃)) * (Real.sin (n₂ * y)) : ℝ)) (z) :=
        by
        refine
          ((((((hasDerivAt_id' (z : ℝ)).const_mul (n₃ : ℝ)).cos).const_mul
                    (n₃ * (Real.sin (x * n₁)) * (Real.sin (n₂ * y)) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce;
          first
          | rfl
          |
            (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                Pi.const_apply, Pi.smul_apply, smul_eq_mul];
              ring_nf; done)
          | (ring_nf; done)
          | (simp; ring_nf; done)
          | (simp; done)
          | (field_simp; ring_nf; done)
        ·
          first
          | rfl
          | ring1
          | (ring_nf; done)
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (simp; ring_nf; done)
          | (norm_num; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      try rw [h5.deriv]
      try beta_reduce
      all_goals
        (first
          | ring1
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
          | (norm_num; done)
          |
            (ext i j;
              fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, dotProduct] <;>
                ring_nf)))
  -- certificate: de-partials.sage

/-- The lowest frequency: `n₁² + n₂² + n₃² ≥ 3` for integers `nᵢ ≥ 1`, with equality at `(1, 1, 1)`, so
`ω_min = √(3 + m²)`. -/
theorem kg_lowest (n₁ n₂ n₃ : ℕ) (h₁ : 1 ≤ n₁) (h₂ : 1 ≤ n₂) (h₃ : 1 ≤ n₃) : 3 ≤ n₁ ^ 2 + n₂ ^ 2 + n₃ ^ 2 := by
  nlinarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.Exercises.kg_lowest
-- test: de-kg-lowest-test.sage

/-- (c) `J₀(m√(t² - x²))`, with `J₀` truncated at `k = 4` (a polynomial in `w = m²(t² - x²)`), solves
`(∂_t² - ∂_x² + m²)u = 0` up to the single truncation term `m²w⁴/147456`: Bessel's equation in the variable
`m√(t² - x²)`. -/
theorem kg_bessel_retarded (m t x : ℝ) :
    deriv (fun s => deriv (fun r => 1 - m ^ 2 * (r ^ 2 - x ^ 2) / 4 + (m ^ 2 * (r ^ 2 - x ^ 2)) ^ 2 / 64 -
        (m ^ 2 * (r ^ 2 - x ^ 2)) ^ 3 / 2304 + (m ^ 2 * (r ^ 2 - x ^ 2)) ^ 4 / 147456) s) t -
      deriv (fun s => deriv (fun r => 1 - m ^ 2 * (t ^ 2 - r ^ 2) / 4 + (m ^ 2 * (t ^ 2 - r ^ 2)) ^ 2 / 64 -
        (m ^ 2 * (t ^ 2 - r ^ 2)) ^ 3 / 2304 + (m ^ 2 * (t ^ 2 - r ^ 2)) ^ 4 / 147456) s) x +
      m ^ 2 * (1 - m ^ 2 * (t ^ 2 - x ^ 2) / 4 + (m ^ 2 * (t ^ 2 - x ^ 2)) ^ 2 / 64 -
        (m ^ 2 * (t ^ 2 - x ^ 2)) ^ 3 / 2304 + (m ^ 2 * (t ^ 2 - x ^ 2)) ^ 4 / 147456) =
      m ^ 2 * (m ^ 2 * (t ^ 2 - x ^ 2)) ^ 4 / 147456 := by
  veriphy_certified
    ( have i0 :
        deriv
            (fun (r : ℝ) =>
              (1 - m ^ 2 * (r ^ 2 - x ^ 2) / 4 + (m ^ 2 * (r ^ 2 - x ^ 2)) ^ 2 / 64 -
                    (m ^ 2 * (r ^ 2 - x ^ 2)) ^ 3 / 2304 +
                  (m ^ 2 * (r ^ 2 - x ^ 2)) ^ 4 / 147456 :
                ℝ)) =
          fun (d1 : ℝ) =>
          (((d1 ^ 2 - (x ^ 2)) ^ 3) * d1 * (m ^ 8) / 18432 - (((d1 ^ 2 - (x ^ 2)) ^ 2) * d1 * (m ^ 6) / 384) +
                ((d1 ^ 2 - (x ^ 2)) * d1 * (m ^ 4) / 16) -
              (d1 * (m ^ 2) / 2) :
            ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt
            (fun (r : ℝ) =>
              (1 - m ^ 2 * (r ^ 2 - x ^ 2) / 4 + (m ^ 2 * (r ^ 2 - x ^ 2)) ^ 2 / 64 -
                    (m ^ 2 * (r ^ 2 - x ^ 2)) ^ 3 / 2304 +
                  (m ^ 2 * (r ^ 2 - x ^ 2)) ^ 4 / 147456 :
                ℝ))
            ((((d1 ^ 2 - (x ^ 2)) ^ 3) * d1 * (m ^ 8) / 18432 - (((d1 ^ 2 - (x ^ 2)) ^ 2) * d1 * (m ^ 6) / 384) +
                  ((d1 ^ 2 - (x ^ 2)) * d1 * (m ^ 4) / 16) -
                (d1 * (m ^ 2) / 2) :
              ℝ))
            (d1) :=
          by
          refine
            (((((((((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-(x ^ 2) : ℝ))).pow 4).const_mul
                                      ((m ^ 8) / 147456 : ℝ)).add
                                  (((((hasDerivAt_id' (d1 : ℝ)).pow 2).add
                                            (hasDerivAt_const (d1 : ℝ) (-(x ^ 2) : ℝ))).pow
                                        3).const_mul
                                    (-(m ^ 6) / 2304 : ℝ))).add
                              (((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-(x ^ 2) : ℝ))).pow
                                    2).const_mul
                                ((m ^ 4) / 64 : ℝ))).add
                          ((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-(x ^ 2) : ℝ))).const_mul
                            (-(m ^ 2) / 4 : ℝ))).add
                      (hasDerivAt_const (d1 : ℝ) (1 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce;
            first
            | rfl
            |
              (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                  Pi.const_apply, Pi.smul_apply, smul_eq_mul];
                ring_nf; done)
            | (ring_nf; done)
            | (simp; ring_nf; done)
            | (simp; done)
            | (field_simp; ring_nf; done)
          ·
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; done)
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (simp; ring_nf; done)
            | (norm_num; done)
            |
              (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                  Real.exp_neg];
                field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt
          (fun s =>
            (fun (d1 : ℝ) =>
                (((d1 ^ 2 - (x ^ 2)) ^ 3) * d1 * (m ^ 8) / 18432 - (((d1 ^ 2 - (x ^ 2)) ^ 2) * d1 * (m ^ 6) / 384) +
                      ((d1 ^ 2 - (x ^ 2)) * d1 * (m ^ 4) / 16) -
                    (d1 * (m ^ 2) / 2) :
                  ℝ))
              s)
          ((((t ^ 2 - (x ^ 2)) ^ 2) * (m ^ 8) * (t ^ 2) / 3072 + (((t ^ 2 - (x ^ 2)) ^ 3) * (m ^ 8) / 18432) -
                      ((t ^ 2 - (x ^ 2)) * (m ^ 6) * (t ^ 2) / 96) -
                    (((t ^ 2 - (x ^ 2)) ^ 2) * (m ^ 6) / 384) +
                  ((m ^ 4) * (t ^ 2) / 8) +
                ((t ^ 2 - (x ^ 2)) * (m ^ 4) / 16) -
              ((m ^ 2) / 2) :
            ℝ))
          (t) :=
        by
        refine
          (((((((((((hasDerivAt_id' (t : ℝ)).pow 2).add (hasDerivAt_const (t : ℝ) (-(x ^ 2) : ℝ))).pow 3).mul
                                    (hasDerivAt_id' (t : ℝ))).const_mul
                                ((m ^ 8) / 18432 : ℝ)).add
                            ((((((hasDerivAt_id' (t : ℝ)).pow 2).add (hasDerivAt_const (t : ℝ) (-(x ^ 2) : ℝ))).pow
                                      2).mul
                                  (hasDerivAt_id' (t : ℝ))).const_mul
                              (-(m ^ 6) / 384 : ℝ))).add
                        (((((hasDerivAt_id' (t : ℝ)).pow 2).add (hasDerivAt_const (t : ℝ) (-(x ^ 2) : ℝ))).mul
                              (hasDerivAt_id' (t : ℝ))).const_mul
                          ((m ^ 4) / 16 : ℝ))).add
                    ((hasDerivAt_id' (t : ℝ)).const_mul (-(m ^ 2) / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce;
          first
          | rfl
          |
            (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                Pi.const_apply, Pi.smul_apply, smul_eq_mul];
              ring_nf; done)
          | (ring_nf; done)
          | (simp; ring_nf; done)
          | (simp; done)
          | (field_simp; ring_nf; done)
        ·
          first
          | rfl
          | ring1
          | (ring_nf; done)
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (simp; ring_nf; done)
          | (norm_num; done)
          |
            (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                Real.exp_neg];
              field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      try rw [h1.deriv]
      have i2 :
        deriv
            (fun (r : ℝ) =>
              (1 - m ^ 2 * (t ^ 2 - r ^ 2) / 4 + (m ^ 2 * (t ^ 2 - r ^ 2)) ^ 2 / 64 -
                    (m ^ 2 * (t ^ 2 - r ^ 2)) ^ 3 / 2304 +
                  (m ^ 2 * (t ^ 2 - r ^ 2)) ^ 4 / 147456 :
                ℝ)) =
          fun (d1 : ℝ) =>
          (((d1 ^ 2 - (t ^ 2)) ^ 3) * d1 * (m ^ 8) / 18432 + (((d1 ^ 2 - (t ^ 2)) ^ 2) * d1 * (m ^ 6) / 384) +
                ((d1 ^ 2 - (t ^ 2)) * d1 * (m ^ 4) / 16) +
              (d1 * (m ^ 2) / 2) :
            ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt
            (fun (r : ℝ) =>
              (1 - m ^ 2 * (t ^ 2 - r ^ 2) / 4 + (m ^ 2 * (t ^ 2 - r ^ 2)) ^ 2 / 64 -
                    (m ^ 2 * (t ^ 2 - r ^ 2)) ^ 3 / 2304 +
                  (m ^ 2 * (t ^ 2 - r ^ 2)) ^ 4 / 147456 :
                ℝ))
            ((((d1 ^ 2 - (t ^ 2)) ^ 3) * d1 * (m ^ 8) / 18432 + (((d1 ^ 2 - (t ^ 2)) ^ 2) * d1 * (m ^ 6) / 384) +
                  ((d1 ^ 2 - (t ^ 2)) * d1 * (m ^ 4) / 16) +
                (d1 * (m ^ 2) / 2) :
              ℝ))
            (d1) :=
          by
          refine
            (((((((((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-(t ^ 2) : ℝ))).pow 4).const_mul
                                      ((m ^ 8) / 147456 : ℝ)).add
                                  (((((hasDerivAt_id' (d1 : ℝ)).pow 2).add
                                            (hasDerivAt_const (d1 : ℝ) (-(t ^ 2) : ℝ))).pow
                                        3).const_mul
                                    ((m ^ 6) / 2304 : ℝ))).add
                              (((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-(t ^ 2) : ℝ))).pow
                                    2).const_mul
                                ((m ^ 4) / 64 : ℝ))).add
                          ((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-(t ^ 2) : ℝ))).const_mul
                            ((m ^ 2) / 4 : ℝ))).add
                      (hasDerivAt_const (d1 : ℝ) (1 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce;
            first
            | rfl
            |
              (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                  Pi.const_apply, Pi.smul_apply, smul_eq_mul];
                ring_nf; done)
            | (ring_nf; done)
            | (simp; ring_nf; done)
            | (simp; done)
            | (field_simp; ring_nf; done)
          ·
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; done)
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (simp; ring_nf; done)
            | (norm_num; done)
            |
              (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                  Real.exp_neg];
                field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
        exact hd.deriv
      rw [i2]
      have h3 :
        HasDerivAt
          (fun s =>
            (fun (d1 : ℝ) =>
                (((d1 ^ 2 - (t ^ 2)) ^ 3) * d1 * (m ^ 8) / 18432 + (((d1 ^ 2 - (t ^ 2)) ^ 2) * d1 * (m ^ 6) / 384) +
                      ((d1 ^ 2 - (t ^ 2)) * d1 * (m ^ 4) / 16) +
                    (d1 * (m ^ 2) / 2) :
                  ℝ))
              s)
          ((((t ^ 2 - (x ^ 2)) ^ 2) * (m ^ 8) * (x ^ 2) / 3072 - (((t ^ 2 - (x ^ 2)) ^ 3) * (m ^ 8) / 18432) -
                      ((t ^ 2 - (x ^ 2)) * (m ^ 6) * (x ^ 2) / 96) +
                    (((t ^ 2 - (x ^ 2)) ^ 2) * (m ^ 6) / 384) +
                  ((m ^ 4) * (x ^ 2) / 8) -
                ((t ^ 2 - (x ^ 2)) * (m ^ 4) / 16) +
              ((m ^ 2) / 2) :
            ℝ))
          (x) :=
        by
        refine
          (((((((((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-(t ^ 2) : ℝ))).pow 3).mul
                                    (hasDerivAt_id' (x : ℝ))).const_mul
                                ((m ^ 8) / 18432 : ℝ)).add
                            ((((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-(t ^ 2) : ℝ))).pow
                                      2).mul
                                  (hasDerivAt_id' (x : ℝ))).const_mul
                              ((m ^ 6) / 384 : ℝ))).add
                        (((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-(t ^ 2) : ℝ))).mul
                              (hasDerivAt_id' (x : ℝ))).const_mul
                          ((m ^ 4) / 16 : ℝ))).add
                    ((hasDerivAt_id' (x : ℝ)).const_mul ((m ^ 2) / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce;
          first
          | rfl
          |
            (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                Pi.const_apply, Pi.smul_apply, smul_eq_mul];
              ring_nf; done)
          | (ring_nf; done)
          | (simp; ring_nf; done)
          | (simp; done)
          | (field_simp; ring_nf; done)
        ·
          first
          | rfl
          | ring1
          | (ring_nf; done)
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (simp; ring_nf; done)
          | (norm_num; done)
          |
            (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                Real.exp_neg];
              field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      try rw [h3.deriv]
      try beta_reduce
      all_goals
        (first
          | ring1
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
          | (norm_num; done)
          |
            (ext i j;
              fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, dotProduct] <;>
                ring_nf)))
  -- certificate: de-partials.sage

end KaluzaKlein.DifferentialEquations.Exercises
