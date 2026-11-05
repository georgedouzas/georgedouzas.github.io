import Mathlib
import Veriphy.Kit.Levels

/-!
# Classical mechanics, section "Energy"

The claims of the section "Energy" of `tutorial/part-ii-01-classical-mechanics.qmd`. Work, kinetic and potential
energy and their conservation are identities in the jet variables (position, velocity, acceleration, force, and the
derivatives of the potential); gradients and curls of explicit forces are derivatives of explicit functions; the
curl criterion on a simply connected region is stated as a proposition about `C¹` vector fields on `ℝ³` and tested.
-/

open Real Filter Topology

namespace KaluzaKlein.ClassicalMechanics.Energy

/-! ## def-clm-conservative-force -/

/-- `dT/dt = m v a = F v` along a motion with `m a = F` (one component; the vector statement is the sum). -/
theorem kinetic_rate (m v a F : ℝ) (h : m * a = F) : m * v * a = F * v := by
  veriphy_certified (linear_combination (v) * h)
  -- certificate: cm-poly.sage

/-- For `F = -V'`, `dE/dt = m v a + V' v = 0`: the energy `T + V` is conserved. -/
theorem energy_conserved (m v a V' : ℝ) (h : m * a = -V') : m * v * a + V' * v = 0 := by
  veriphy_certified (linear_combination (v) * h)
  -- certificate: cm-poly.sage

/-- A gradient has zero curl: with symmetric second derivatives, `(∇ × ∇V)_x = V_zy - V_yz = 0` … -/
theorem gradient_curl_x (Vyz Vzy : ℝ) (h : Vyz = Vzy) : Vzy - Vyz = 0 := by
  veriphy_certified (linear_combination (-1) * h)
  -- certificate: cm-poly.sage

/-- … `(∇ × ∇V)_y = V_xz - V_zx = 0` … -/
theorem gradient_curl_y (Vxz Vzx : ℝ) (h : Vxz = Vzx) : Vxz - Vzx = 0 := by
  veriphy_certified (linear_combination h)
  -- certificate: cm-poly.sage

/-- … `(∇ × ∇V)_z = V_yx - V_xy = 0`. -/
theorem gradient_curl_z (Vxy Vyx : ℝ) (h : Vxy = Vyx) : Vyx - Vxy = 0 := by
  veriphy_certified (linear_combination (-1) * h)
  -- certificate: cm-poly.sage

/-- The partial derivative `∂_i f` of a function on `ℝ³`. -/
noncomputable def pd (f : (Fin 3 → ℝ) → ℝ) (i : Fin 3) (x : Fin 3 → ℝ) : ℝ := fderiv ℝ f x (Pi.single i 1)

/-- The curl of a vector field on `ℝ³`. -/
noncomputable def curl (F : (Fin 3 → ℝ) → Fin 3 → ℝ) (x : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![pd (fun y => F y 2) 1 x - pd (fun y => F y 1) 2 x, pd (fun y => F y 0) 2 x - pd (fun y => F y 2) 0 x,
    pd (fun y => F y 1) 0 x - pd (fun y => F y 0) 1 x]

/-- On the simply connected region `ℝ³`, a `C¹` force is conservative exactly when its curl vanishes (the Poincaré
lemma; a statement about every vector field, recorded as a proposition and tested on explicit fields). -/
def curl_criterion : Prop :=
  ∀ F : (Fin 3 → ℝ) → Fin 3 → ℝ, ContDiff ℝ 1 F →
    ((∃ V : (Fin 3 → ℝ) → ℝ, ∀ x i, F x i = -pd V i x) ↔ ∀ x, curl F x = 0)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ClassicalMechanics.Energy.curl_criterion
-- test: cm-curl-criterion-test.sage

/-! ## exm-clm-conservative-spring -/

/-- `-∂_x (k|x|²/2) = -k x`: the spring force is the gradient of `V = k|x|²/2` (`x`-component). -/
theorem spring_potential_grad (k x y z : ℝ) : deriv (fun x => k * (x ^ 2 + y ^ 2 + z ^ 2) / 2) x = k * x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => k * (x ^ 2 + y ^ 2 + z ^ 2) / 2) ((k * x : ℝ)) (x) :=
        by
        refine
          (((((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (y ^ 2 : ℝ))).add
                        (hasDerivAt_const (x : ℝ) (z ^ 2 : ℝ))).const_mul
                    (k / 2 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      exact hd.deriv.trans (by rfl))
  -- certificate: cm-derivative.sage

/-- `E = m v²/2 + k x²/2` is conserved along `m a = -k x`: `dE/dt = m v a + k x v = 0`. -/
theorem spring_energy_conserved (m k x v a : ℝ) (h : m * a = -k * x) : m * v * a + k * x * v = 0 := by
  veriphy_certified (linear_combination (v) * h)
  -- certificate: cm-poly.sage

/-- The motion is bounded: `k x² ≤ 2E`, that is `|x| ≤ √(2E/k)`. -/
theorem spring_bounded (m k x v E : ℝ) (hm : 0 ≤ m) (hE : E = m * v ^ 2 / 2 + k * x ^ 2 / 2) : k * x ^ 2 ≤ 2 * E := by
  nlinarith [mul_nonneg hm (sq_nonneg v)]
example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Energy.spring_bounded (1) (1) (0) (0) (0) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Energy.spring_bounded (2) (3) (1) (2) (11/2) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Energy.spring_bounded (1) (1/2) (2) (1) (3/2) ((by norm_num)) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ClassicalMechanics.Energy.spring_bounded
-- test: cm-spring-bounded-test.sage

/-- On a line the motion is the harmonic oscillator `x = A cos (ω t + δ)`: `ẍ + ω² x = 0` … -/
theorem spring_solution (A ω δ : ℝ) (x : ℝ → ℝ) (hx : x = fun s => A * Real.cos (ω * s + δ)) (t : ℝ) :
    deriv (deriv x) t + ω ^ 2 * x t = 0 := by
  VERIPHY[cm-ode]

/-- … which is `m ẍ = -k x` when `ω² = k/m` (`m ω² = k`). -/
theorem spring_frequency (m k ω x : ℝ) (hω : m * ω ^ 2 = k) : m * (-(ω ^ 2 * x)) = -k * x := by
  veriphy_certified (linear_combination (-x) * hω)
  -- certificate: cm-poly.sage

/-! ## exm-clm-conservative-gravity -/

/-- `-∂_x (-GMm/|x|) = -GMm x/|x|³` (`x`-component): Newtonian gravity is the gradient of `V = -GMm/|x|`. -/
theorem gravity_potential_grad (G M m x y z : ℝ) (h : 0 < x ^ 2 + y ^ 2 + z ^ 2) :
    deriv (fun x => -(G * M * m) / Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2)) x =
      G * M * m * x / Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2) ^ 3 := by
  VERIPHY[cm-derivative]

/-- `E = m v²/2 - GMm/R ≥ 0` exactly when `v² ≥ 2GM/R`. -/
theorem escape_iff (G M m R v : ℝ) (hm : 0 < m) (hR : 0 < R) :
    0 ≤ m * v ^ 2 / 2 - G * M * m / R ↔ 2 * G * M / R ≤ v ^ 2 := by
  constructor
  · intro h
    have h1 : G * M * m / R ≤ m * v ^ 2 / 2 := by linarith
    rw [div_le_iff₀ hR]
    have : G * M * m ≤ m * v ^ 2 / 2 * R := (div_le_iff₀ hR).mp h1
    nlinarith
  · intro h
    have : 2 * G * M ≤ v ^ 2 * R := (div_le_iff₀ hR).mp h
    have h2 : G * M * m / R ≤ m * v ^ 2 / 2 := by rw [div_le_iff₀ hR]; nlinarith
    linarith
example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Energy.escape_iff (0) (1) (1) (1) (0) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Energy.escape_iff (1) (1) (1) (1) (1) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Energy.escape_iff (1) (1) (1) (1) (2) ((by norm_num)) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ClassicalMechanics.Energy.escape_iff
-- test: cm-escape-iff-test.sage

/-- The escape speed `√(2GM/R)` has `v² = 2GM/R`. -/
theorem escape_speed_sq (G M R : ℝ) (hGM : 0 ≤ G * M) (hR : 0 < R) : Real.sqrt (2 * G * M / R) ^ 2 = 2 * G * M / R := by
  VERIPHY[cm-rewrite]

/-! ## exm-clm-conservative-curl -/

/-- `∇ × (-y, x, 0)` has `z`-component `∂_x x - ∂_y (-y) = 2 ≠ 0`. -/
theorem rotation_curl (x y : ℝ) : deriv (fun x : ℝ => x) x - deriv (fun y : ℝ => -y) y = 2 := by
  veriphy_certified (simp only [deriv_id'', deriv_neg'']; norm_num)
  -- certificate: cm-rewrite.sage

/-- Its work around the unit circle is `∫₀^{2π} (sin² s + cos² s) ds = 2π` … -/
theorem rotation_work : ∫ s in (0 : ℝ)..(2 * Real.pi), (Real.sin s ^ 2 + Real.cos s ^ 2) = 2 * Real.pi := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt (fun x_veriphy => (x_veriphy : ℝ)) ((fun s => sin s ^ 2 + cos s ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((hasDerivAt_id' (x_veriphy : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · (simp only [Real.cos_sq, Real.sin_sq_eq_half_sub]; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (x_veriphy : ℝ)) (f' := fun s =>
          sin s ^ 2 + cos s ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: cm-integral.sage

/-- … which is not zero. -/
theorem rotation_work_ne : (2 * Real.pi : ℝ) ≠ 0 := by
  veriphy_certified (positivity)
  -- certificate: cm-rewrite.sage

/-- A gradient does no work around a closed loop: for `g = V ∘ γ` with `γ(2π) = γ(0)`, `∫₀^{2π} g' = g(2π) - g(0) = 0`. -/
theorem gradient_no_work (g : ℝ → ℝ) (hg : Differentiable ℝ g) (hint : IntervalIntegrable (deriv g) MeasureTheory.volume 0 (2 * Real.pi))
    (hper : g (2 * Real.pi) = g 0) : ∫ s in (0 : ℝ)..(2 * Real.pi), deriv g s = 0 := by
  rw [intervalIntegral.integral_deriv_eq_sub (fun x _ => hg x) hint, hper, sub_self]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ClassicalMechanics.Energy.gradient_no_work
-- test: cm-gradient-no-work-test.sage

/-- The drag force `-γ v` (`γ ≠ 0`) is not conservative: it is not a function of the position alone, so it is no
gradient `-V'(x)`. -/
theorem drag_not_conservative (γ : ℝ) (hγ : γ ≠ 0) : ¬ ∃ V : ℝ → ℝ, ∀ x v : ℝ, -γ * v = -deriv V x := by
  rintro ⟨V, h⟩
  have h0 := h 0 0
  have h1 := h 0 1
  simp only [mul_zero, mul_one] at h0 h1
  exact hγ (by linarith)
example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Energy.drag_not_conservative (1) ((by norm_num)) := by
  rintro ⟨V, h⟩
  have h0 := h 0 0
  have h1 := h 0 1
  norm_num at h0 h1
  linarith

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Energy.drag_not_conservative (-2) ((by norm_num)) := by
  rintro ⟨V, h⟩
  have h0 := h 0 0
  have h1 := h 0 1
  norm_num at h0 h1
  linarith

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Energy.drag_not_conservative (1/3) ((by norm_num)) := by
  rintro ⟨V, h⟩
  have h0 := h 0 0
  have h1 := h 0 1
  norm_num at h0 h1
  linarith

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ClassicalMechanics.Energy.drag_not_conservative
-- test: cm-drag-not-conservative-test.sage

/-! ## exr-clm-curl-test -/

/-- `F₁ = (2xy, x², 0)`: `∂_x F₁ʸ = ∂_x x² = 2x` … -/
theorem F1_dx (x : ℝ) : deriv (fun x : ℝ => x ^ 2) x = 2 * x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x ^ 2) ((2 * x : ℝ)) (x) :=
        by
        refine
          ((((hasDerivAt_id' (x : ℝ)).pow 2)).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      exact hd.deriv.trans (by rfl))
  -- certificate: cm-derivative.sage

/-- … `∂_y F₁ˣ = ∂_y (2xy) = 2x`, so `(∇ × F₁)_z = 0` (the other components vanish since `F₁` does not depend on `z`). -/
theorem F1_dy (x y : ℝ) : deriv (fun y : ℝ => 2 * x * y) y = 2 * x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun y => 2 * x * y) ((2 * x : ℝ)) (y) :=
        by
        refine
          ((((hasDerivAt_id' (y : ℝ)).const_mul (2 * x : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      exact hd.deriv.trans (by rfl))
  -- certificate: cm-derivative.sage

/-- `V₁ = -x² y`: `-∂_x V₁ = 2xy` … -/
theorem V1_x (x y : ℝ) : deriv (fun x : ℝ => -(x ^ 2 * y)) x = -(2 * x * y) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => -(x ^ 2 * y)) ((-2 * x * y : ℝ)) (x) :=
        by
        refine
          (((((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (-y : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      exact hd.deriv.trans (by ring1))
  -- certificate: cm-derivative.sage

/-- … and `-∂_y V₁ = x²`: `-∇V₁ = F₁`. -/
theorem V1_y (x y : ℝ) : deriv (fun y : ℝ => -(x ^ 2 * y)) y = -(x ^ 2) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun y => -(x ^ 2 * y)) ((-(x ^ 2) : ℝ)) (y) :=
        by
        refine
          ((((hasDerivAt_id' (y : ℝ)).const_mul (-(x ^ 2) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (ring_nf; done)
        · ring1
      exact hd.deriv.trans (by rfl))
  -- certificate: cm-derivative.sage

/-- `F₂ = (y, -x, 0)`: `(∇ × F₂)_z = ∂_x (-x) - ∂_y y = -2`, not conservative. -/
theorem F2_curl (x y : ℝ) : deriv (fun x : ℝ => -x) x - deriv (fun y : ℝ => y) y = -2 := by
  veriphy_certified (simp only [deriv_id'', deriv_neg'']; norm_num)
  -- certificate: cm-rewrite.sage

/-- `F₃ = -k x/|x|³`: `∂_z F₃ʸ = ∂_z (-k y/|x|³) = 3 k y z/|x|⁵` … -/
theorem F3_dz (k x y z : ℝ) (h : 0 < x ^ 2 + y ^ 2 + z ^ 2) :
    deriv (fun z => -k * y / Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2) ^ 3) z =
      3 * k * y * z / Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2) ^ 5 := by
  VERIPHY[cm-derivative]

/-- … `∂_y F₃ᶻ = ∂_y (-k z/|x|³) = 3 k y z/|x|⁵`, so `(∇ × F₃)_x = 0` (the other components by symmetry). -/
theorem F3_dy (k x y z : ℝ) (h : 0 < x ^ 2 + y ^ 2 + z ^ 2) :
    deriv (fun y => -k * z / Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2) ^ 3) y =
      3 * k * y * z / Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2) ^ 5 := by
  VERIPHY[cm-derivative]

/-- `V₃ = -k/|x|`: `-∂_x V₃ = -k x/|x|³ = F₃ˣ` (the other components by symmetry). -/
theorem V3_x (k x y z : ℝ) (h : 0 < x ^ 2 + y ^ 2 + z ^ 2) :
    deriv (fun x => -k / Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2)) x = k * x / Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2) ^ 3 := by
  VERIPHY[cm-derivative]

/-! ## exr-clm-energy-1d -/

/-- From `E = m ẋ²/2 + V` with `ẋ > 0`: `ẋ = √(2(E - V)/m)`, so `dt = dx/√(2(E - V)/m)`. -/
theorem speed_from_energy (m E V v : ℝ) (hm : 0 < m) (hv : 0 < v) (hE : E = m * v ^ 2 / 2 + V) :
    v = Real.sqrt (2 * (E - V) / m) := by
  rw [hE, show 2 * (m * v ^ 2 / 2 + V - V) / m = v ^ 2 by field_simp, Real.sqrt_sq hv.le]
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.Energy.speed_from_energy, cm-speed-from-energy-test]

/-- For the oscillator `V = k x²/2`, `√(2(E - V)/m) = ω √(A² - x²)` with `ω = √(k/m)` and `A² = 2E/k`. -/
theorem oscillator_speed (m k E x : ℝ) (hm : 0 < m) (hk : 0 < k) :
    Real.sqrt (2 * (E - k * x ^ 2 / 2) / m) = Real.sqrt (k / m) * Real.sqrt (2 * E / k - x ^ 2) := by
  VERIPHY[cm-rewrite]

/-- The motion `x = A sin (ω t)` is at `-A` at `t = -π/(2ω)` … -/
theorem oscillator_turning_left (A ω : ℝ) (hω : ω ≠ 0) : A * Real.sin (ω * (-(Real.pi / (2 * ω)))) = -A := by
  VERIPHY[cm-rewrite]

/-- … and at `A` at `t = π/(2ω)`: the time from `-A` to `A` is `π/ω`, the period `2π/ω` … -/
theorem oscillator_turning_right (A ω : ℝ) (hω : ω ≠ 0) : A * Real.sin (ω * (Real.pi / (2 * ω))) = A := by
  VERIPHY[cm-rewrite]

/-- … which is `2π √(m/k)` for `ω = √(k/m)`, independent of the amplitude. -/
theorem oscillator_period (m k : ℝ) (hm : 0 < m) (hk : 0 < k) : 2 * Real.pi / Real.sqrt (k / m) = 2 * Real.pi * Real.sqrt (m / k) := by
  veriphy_certified
    ( rw [show m / k = (k / m)⁻¹ by rw [inv_div], Real.sqrt_inv]
      ring)
  -- certificate: cm-rewrite.sage

/-! ## exr-clm-escape -/

/-- `E = m v²/2 - GMm/R ≥ 0` exactly when `v ≥ √(2GM/R)` (for `v ≥ 0`). -/
theorem escape_speed_iff (G M m R v : ℝ) (hm : 0 < m) (hR : 0 < R) (hv : 0 ≤ v) (hGM : 0 ≤ G * M) :
    0 ≤ m * v ^ 2 / 2 - G * M * m / R ↔ Real.sqrt (2 * G * M / R) ≤ v := by
  rw [escape_iff G M m R v hm hR, Real.sqrt_le_left hv]
example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Energy.escape_speed_iff (0) (1) (1) (1) (0) ((by norm_num)) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Energy.escape_speed_iff (1) (1) (1) (2) (3) ((by norm_num)) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  rw [show (2 : ℝ) * 1 * 1 / 2 = 1 by norm_num, Real.sqrt_one]
  norm_num

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Energy.escape_speed_iff (2) (1) (1) (1) (1) ((by norm_num)) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  rw [show (2 : ℝ) * 2 * 1 / 1 = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ClassicalMechanics.Energy.escape_speed_iff
-- test: cm-escape-speed-iff-test.sage

/-- With `g = GM/R²`, `2GM/R = 2gR`. -/
theorem escape_speed_g (G M R g : ℝ) (hR : R ≠ 0) (hg : g = G * M / R ^ 2) : 2 * G * M / R = 2 * g * R := by
  rw [hg]
  veriphy_certified ((field_simp; done))
  -- certificate: cm-poly.sage

/-- `2 × 9.81 × 6.371 × 10⁶ ≈ 1.250 × 10⁸`. -/
theorem escape_speed_sq_value : |2 * 9.81 * 6371000 - 125000000| < (100000 : ℝ) := by
  VERIPHY[cm-rewrite]

/-- `√(2 × 9.81 × 6.371 × 10⁶) ≈ 1.12 × 10⁴ m/s`: it lies between `11180` and `11181`. -/
theorem escape_speed_value_lower : (11180 : ℝ) < Real.sqrt (2 * 9.81 * 6371000) := by
  VERIPHY[cm-rewrite]

/-- … and below `11181`, that is about `11.2 km/s`. -/
theorem escape_speed_value_upper : Real.sqrt (2 * 9.81 * 6371000) < (11181 : ℝ) := by
  VERIPHY[cm-rewrite]

end KaluzaKlein.ClassicalMechanics.Energy
