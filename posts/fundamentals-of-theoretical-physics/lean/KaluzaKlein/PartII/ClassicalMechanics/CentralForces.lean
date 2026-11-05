import Mathlib
import Veriphy.Kit.Levels

/-!
# Classical mechanics, section "Central forces"

The claims of the section "Central forces" of `tutorial/part-ii-01-classical-mechanics.qmd`. Angular momentum, the
plane of the motion, the polar form of the energy and the effective potential are identities in the jet variables
(positions, momenta, forces, `cos φ`, `sin φ` with `cos² + sin² = 1`); the effective potentials of gravity and of
the isotropic oscillator are explicit functions whose derivatives and critical values are computed; Bertrand's
theorem is stated as a proposition about the orbits of the power-law forces and tested numerically.
-/

open Real Filter Topology

namespace KaluzaKlein.ClassicalMechanics.CentralForces

/-! ## def-clm-central-force -/

/-- A central force is conservative: for `V' = -f`, `∂_x V(|x|) = -f(|x|) x/|x|`, that is `-∇V = f(r) x/r`. -/
theorem central_potential_grad (V f : ℝ → ℝ) (hV : ∀ r, HasDerivAt V (-f r) r) (x y z : ℝ)
    (h : 0 < x ^ 2 + y ^ 2 + z ^ 2) :
    deriv (fun x => V (Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2))) x =
      -(f (Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2)) * x / Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2)) := by
  have h1 : HasDerivAt (fun x => x ^ 2 + y ^ 2 + z ^ 2) (2 * x) x := by
    have := ((hasDerivAt_id' x).pow 2).add_const (y ^ 2 + z ^ 2)
    convert this using 1
    · funext u; ring
    · push_cast; ring
  have hs : HasDerivAt (fun x => Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2)) (x / Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2)) x := by
    have := h1.sqrt h.ne'
    convert this using 1
    field_simp
  have := ((hV _).comp x hs).deriv
  rw [show (fun x => V (Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2))) = V ∘ fun x => Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2) from rfl,
    this]
  ring
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.CentralForces.central_potential_grad, cm-central-potential-grad-test]

/-- `dL_z/dt = ẋ p_y + x ṗ_y - ẏ p_x - y ṗ_x = x F_y - y F_x` with `p = m ẋ`, `ṗ = F`. -/
theorem angular_momentum_rate (m x y vx vy Fx Fy : ℝ) :
    vx * (m * vy) + x * Fy - vy * (m * vx) - y * Fx = x * Fy - y * Fx := by
  veriphy_certified (ring1)
  -- certificate: cm-poly.sage

/-- A central force `F = f(r) x/r` exerts no torque: `x F_y - y F_x = 0`, so `L` is conserved. -/
theorem central_torque_zero (f r x y : ℝ) (hr : r ≠ 0) : x * (f * y / r) - y * (f * x / r) = 0 := by
  veriphy_certified (ring1)
  -- certificate: cm-poly.sage

/-- The motion lies in the plane orthogonal to `L = x × p`: `x · L = 0` … -/
theorem position_orthogonal_L (x y z px py pz : ℝ) :
    x * (y * pz - z * py) + y * (z * px - x * pz) + z * (x * py - y * px) = 0 := by
  veriphy_certified (ring1)
  -- certificate: cm-poly.sage

/-- … and `p · L = 0`. -/
theorem momentum_orthogonal_L (x y z px py pz : ℝ) :
    px * (y * pz - z * py) + py * (z * px - x * pz) + pz * (x * py - y * px) = 0 := by
  VERIPHY[cm-poly]

/-- In polar coordinates (`x = r c`, `y = r s`, `c² + s² = 1`, `ẋ = ṙ c - r s φ̇`, `ẏ = ṙ s + r c φ̇`),
`L = m (x ẏ - y ẋ) = m r² φ̇`. -/
theorem polar_angular_momentum (m r c s ṙ φ̇ : ℝ) (hcs : c ^ 2 + s ^ 2 = 1) :
    m * (r * c) * (ṙ * s + r * c * φ̇) - m * (r * s) * (ṙ * c - r * s * φ̇) = m * r ^ 2 * φ̇ := by
  VERIPHY[cm-poly]

/-- The energy in polar coordinates: `m(ẋ² + ẏ²)/2 + V = m ṙ²/2 + m r² φ̇²/2 + V` … -/
theorem polar_energy (m r c s ṙ φ̇ V : ℝ) (hcs : c ^ 2 + s ^ 2 = 1) :
    m / 2 * ((ṙ * c - r * s * φ̇) ^ 2 + (ṙ * s + r * c * φ̇) ^ 2) + V = m / 2 * ṙ ^ 2 + m * r ^ 2 * φ̇ ^ 2 / 2 + V := by
  VERIPHY[cm-poly]

/-- … and `m r² φ̇²/2 = L²/(2 m r²)` with `L = m r² φ̇`: the effective potential `U = L²/(2mr²) + V`. -/
theorem effective_potential_term (m r φ̇ : ℝ) (hm : m ≠ 0) (hr : r ≠ 0) :
    (m * r ^ 2 * φ̇) ^ 2 / (2 * m * r ^ 2) = m * r ^ 2 * φ̇ ^ 2 / 2 := by
  VERIPHY[cm-poly]

/-! ## exm-clm-central-areal -/

/-- `dA/dt = r² φ̇/2 = L/(2m)` with `L = m r² φ̇`. -/
theorem areal_velocity (m r φ̇ : ℝ) (hm : m ≠ 0) : r ^ 2 * φ̇ / 2 = m * r ^ 2 * φ̇ / (2 * m) := by
  VERIPHY[cm-poly]

/-! ## exm-clm-central-effective -/

/-- `U = L²/(2mr²) - k/r` has `U' = -L²/(m r³) + k/r²`. -/
theorem kepler_U_deriv (L m k r : ℝ) (hr : r ≠ 0) (hm : m ≠ 0) :
    deriv (fun r => L ^ 2 / (2 * m * r ^ 2) - k / r) r = -L ^ 2 / (m * r ^ 3) + k / r ^ 2 := by
  VERIPHY[cm-derivative]

/-- `U'` vanishes at `r₀ = L²/(mk)`. -/
theorem kepler_U_deriv_zero (L m k : ℝ) (hL : L ≠ 0) (hm : m ≠ 0) (hk : k ≠ 0) :
    -L ^ 2 / (m * (L ^ 2 / (m * k)) ^ 3) + k / (L ^ 2 / (m * k)) ^ 2 = 0 := by
  veriphy_certified ((field_simp; ring1))
  -- certificate: cm-poly.sage

/-- `U(r₀) = -m k²/(2L²)`. -/
theorem kepler_U_min_value (L m k : ℝ) (hL : L ≠ 0) (hm : m ≠ 0) (hk : k ≠ 0) :
    L ^ 2 / (2 * m * (L ^ 2 / (m * k)) ^ 2) - k / (L ^ 2 / (m * k)) = -(m * k ^ 2 / (2 * L ^ 2)) := by
  veriphy_certified ((field_simp; ring1))
  -- certificate: cm-poly.sage

/-- `U'' = 3L²/(m r⁴) - 2k/r³` … -/
theorem kepler_U_second_deriv (L m k r : ℝ) (hr : r ≠ 0) (hm : m ≠ 0) :
    deriv (fun r => -L ^ 2 / (m * r ^ 3) + k / r ^ 2) r = 3 * L ^ 2 / (m * r ^ 4) - 2 * k / r ^ 3 := by
  VERIPHY[cm-derivative]

/-- … equal to `m³k⁴/L⁶` at `r₀` … -/
theorem kepler_U_second_at_min (L m k : ℝ) (hL : L ≠ 0) (hm : m ≠ 0) (hk : k ≠ 0) :
    3 * L ^ 2 / (m * (L ^ 2 / (m * k)) ^ 4) - 2 * k / (L ^ 2 / (m * k)) ^ 3 = m ^ 3 * k ^ 4 / L ^ 6 := by
  veriphy_certified ((field_simp; ring1))
  -- certificate: cm-poly.sage

/-- … which is positive: `r₀` is the minimum of `U`. -/
theorem kepler_U_second_pos (L m k : ℝ) (hL : L ≠ 0) (hm : 0 < m) (hk : 0 < k) : 0 < m ^ 3 * k ^ 4 / L ^ 6 := by
  positivity
example : veriphy_instance% KaluzaKlein.ClassicalMechanics.CentralForces.kepler_U_second_pos (1) (1) (1) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  positivity

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.CentralForces.kepler_U_second_pos (-2) (3) (1/2) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  positivity

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.CentralForces.kepler_U_second_pos (1/3) (1/2) (5) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  positivity

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ClassicalMechanics.CentralForces.kepler_U_second_pos
-- test: cm-kepler-U-second-pos-test.sage

/-- The turning points `U(r) = E` are the roots of `L² u² - 2mk u - 2mE = 0` in `u = 1/r`: with
`s² = m²k² + 2mEL²`, `u₊ = (mk + s)/L²` is a root (the equation times `L²`) … -/
theorem turning_point_plus (m k E L s : ℝ) (hs : s ^ 2 = m ^ 2 * k ^ 2 + 2 * m * E * L ^ 2) :
    (m * k + s) ^ 2 - 2 * m * k * (m * k + s) - 2 * m * E * L ^ 2 = 0 := by
  veriphy_certified (linear_combination hs)
  -- certificate: cm-poly.sage

/-- … and so is `u₋ = (mk - s)/L²`. -/
theorem turning_point_minus (m k E L s : ℝ) (hs : s ^ 2 = m ^ 2 * k ^ 2 + 2 * m * E * L ^ 2) :
    (m * k - s) ^ 2 - 2 * m * k * (m * k - s) - 2 * m * E * L ^ 2 = 0 := by
  VERIPHY[cm-poly]

/-- Both turning points are at finite radius (`u₋ > 0`) exactly when `E < 0`: the radius oscillates between them;
for `E ≥ 0` the particle comes in from infinity and returns to it. -/
theorem two_turning_points_iff (m k E L s : ℝ) (hm : 0 < m) (hk : 0 < k) (hL : 0 < L) (hs0 : 0 ≤ s)
    (hs : s ^ 2 = m ^ 2 * k ^ 2 + 2 * m * E * L ^ 2) : 0 < m * k - s ↔ E < 0 := by
  have hmk : 0 < m * k := mul_pos hm hk
  constructor
  · intro h
    have h2 : s ^ 2 < (m * k) ^ 2 := by nlinarith
    have h3 : 2 * m * E * L ^ 2 < 0 := by nlinarith
    have h4 : 0 < 2 * m * L ^ 2 := by positivity
    nlinarith
  · intro h
    have h3 : 2 * m * E * L ^ 2 < 0 := by
      have : 0 < 2 * m * L ^ 2 := by positivity
      nlinarith
    have h2 : s ^ 2 < (m * k) ^ 2 := by nlinarith
    nlinarith [abs_lt_abs_of_sq_lt_sq' h2 hmk.le]
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.CentralForces.two_turning_points_iff, cm-two-turning-points-test]

/-- Turning points exist (`s² > 0`) exactly when `E > U(r₀) = -mk²/(2L²)`. -/
theorem energy_above_minimum_iff (m k E L : ℝ) (hm : 0 < m) (hL : 0 < L) :
    -(m * k ^ 2 / (2 * L ^ 2)) < E ↔ 0 < m ^ 2 * k ^ 2 + 2 * m * E * L ^ 2 := by
  have key : m ^ 2 * k ^ 2 + 2 * m * E * L ^ 2 = 2 * m * L ^ 2 * (E + m * k ^ 2 / (2 * L ^ 2)) := by
    field_simp; ring
  rw [key, mul_pos_iff_of_pos_left (by positivity)]
  constructor <;> intro h <;> linarith
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.CentralForces.energy_above_minimum_iff, cm-energy-above-minimum-test]

/-! ## exm-clm-central-oscillator -/

/-- For `V = k r²/2` every orbit is bounded: `k r² ≤ 2E`. -/
theorem oscillator_bounded (m k E L r ṙ : ℝ) (hm : 0 < m) (hE : E = m * ṙ ^ 2 / 2 + L ^ 2 / (2 * m * r ^ 2) + k * r ^ 2 / 2) :
    k * r ^ 2 ≤ 2 * E := by
  have h1 : 0 ≤ L ^ 2 / (2 * m * r ^ 2) := by positivity
  nlinarith [mul_nonneg hm.le (sq_nonneg ṙ)]
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.CentralForces.oscillator_bounded, cm-oscillator-bounded-test]

/-- `U = L²/(2mr²) + k r²/2 → ∞` as `r → ∞` … -/
theorem oscillator_U_tendsto_atTop (m k L : ℝ) (hm : 0 < m) (hk : 0 < k) :
    Tendsto (fun r : ℝ => L ^ 2 / (2 * m * r ^ 2) + k * r ^ 2 / 2) atTop atTop := by
  have h1 : Tendsto (fun r : ℝ => k * r ^ 2 / 2) atTop atTop :=
    ((tendsto_pow_atTop two_ne_zero).const_mul_atTop hk).atTop_div_const two_pos
  exact tendsto_atTop_mono (fun r => le_add_of_nonneg_left (by positivity)) h1
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.CentralForces.oscillator_U_tendsto_atTop, cm-oscillator-U-atTop-test]

/-- … and as `r → 0⁺`. -/
theorem oscillator_U_tendsto_zero (m k L : ℝ) (hm : 0 < m) (hk : 0 ≤ k) (hL : L ≠ 0) :
    Tendsto (fun r : ℝ => L ^ 2 / (2 * m * r ^ 2) + k * r ^ 2 / 2) (𝓝[>] 0) atTop := by
  have h1 : Tendsto (fun r : ℝ => r ^ 2) (𝓝[>] 0) (𝓝[>] 0) := by
    apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within
    · have : Tendsto (fun r : ℝ => r ^ 2) (𝓝 0) (𝓝 ((0 : ℝ) ^ 2)) := (continuous_pow 2).tendsto 0
      simpa using this.mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin] with r hr
      exact pow_pos hr 2
  have h2 : Tendsto (fun r : ℝ => (r ^ 2)⁻¹) (𝓝[>] 0) atTop := tendsto_inv_nhdsGT_zero.comp h1
  have h3 : Tendsto (fun r : ℝ => L ^ 2 / (2 * m) * (r ^ 2)⁻¹) (𝓝[>] 0) atTop :=
    h2.const_mul_atTop (by positivity)
  refine tendsto_atTop_mono (fun r => ?_) h3
  have : L ^ 2 / (2 * m) * (r ^ 2)⁻¹ = L ^ 2 / (2 * m * r ^ 2) := by ring
  rw [this]
  exact le_add_of_nonneg_right (by positivity)
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.CentralForces.oscillator_U_tendsto_zero, cm-oscillator-U-zero-test]

/-- In Cartesian coordinates `x = a cos ωt` solves `ẍ + ω² x = 0` … -/
theorem isotropic_x_solves (a ω : ℝ) (x : ℝ → ℝ) (hx : x = fun t => a * Real.cos (ω * t)) (t : ℝ) :
    deriv (deriv x) t + ω ^ 2 * x t = 0 := by
  veriphy_certified
    ( subst hx
      have d1 : deriv (fun (t : ℝ) => (a * cos (ω * t) : ℝ)) = fun (s : ℝ) => (-ω * a * (Real.sin (s * ω)) : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (t : ℝ) => (a * cos (ω * t) : ℝ)) ((-ω * a * (Real.sin (s * ω)) : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (ω : ℝ)).cos).const_mul (a : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · (ring_nf; done)
        exact hd.deriv
      have d2 :
        deriv (fun (s : ℝ) => (-ω * a * (Real.sin (s * ω)) : ℝ)) = fun (s : ℝ) =>
          (-(ω ^ 2) * a * (Real.cos (s * ω)) : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => (-ω * a * (Real.sin (s * ω)) : ℝ)) ((-(ω ^ 2) * a * (Real.cos (s * ω)) : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (ω : ℝ)).sin).const_mul (-ω * a : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (ring_nf; done)
        exact hd.deriv
      simp only [d1, d2]
      all_goals ((ring_nf; done)))
  -- certificate: cm-ode.sage

/-- … and `y = b cos (ωt + δ)` solves `ÿ + ω² y = 0` (with `ω² = k/m`). -/
theorem isotropic_y_solves (b ω δ : ℝ) (y : ℝ → ℝ) (hy : y = fun t => b * Real.cos (ω * t + δ)) (t : ℝ) :
    deriv (deriv y) t + ω ^ 2 * y t = 0 := by
  VERIPHY[cm-ode]

/-- The orbit is an ellipse centred at the origin: `(x/a)² - 2 (x/a)(y/b) cos δ + (y/b)² = sin² δ` (times `a²b²`),
with `x = a cos θ`, `y = b cos (θ + δ)`, `θ = ωt`. -/
theorem isotropic_ellipse (a b cθ sθ cδ sδ : ℝ) (h1 : cθ ^ 2 + sθ ^ 2 = 1) (h2 : cδ ^ 2 + sδ ^ 2 = 1) :
    (b * (a * cθ)) ^ 2 - 2 * (b * (a * cθ)) * (a * (b * (cθ * cδ - sθ * sδ))) * cδ + (a * (b * (cθ * cδ - sθ * sδ))) ^ 2
      = (a * b * sδ) ^ 2 := by
  veriphy_certified (linear_combination (b ^ 2 * a ^ 2 * sδ ^ 2) * h1 + (-b ^ 2 * a ^ 2 * cθ ^ 2) * h2)
  -- certificate: cm-poly.sage

/-- For the power-law force `f(r) = -k r^α`, every bounded orbit of a particle in the plane (with
`ẍ = -k r^{α-1} x`, `ÿ = -k r^{α-1} y`) closes: it is periodic. -/
def closes_all_bounded_orbits (α : ℝ) : Prop :=
  ∀ k : ℝ, 0 < k → ∀ x y : ℝ → ℝ,
    (∀ t, deriv (deriv x) t = -k * Real.sqrt (x t ^ 2 + y t ^ 2) ^ (α - 1) * x t) →
    (∀ t, deriv (deriv y) t = -k * Real.sqrt (x t ^ 2 + y t ^ 2) ^ (α - 1) * y t) →
    (∀ t, 0 < x t ^ 2 + y t ^ 2) → (∃ R, ∀ t, x t ^ 2 + y t ^ 2 ≤ R) →
    ∃ T, 0 < T ∧ ∀ t, x (t + T) = x t ∧ y (t + T) = y t

/-- Bertrand's theorem: `-k/r` (`α = -2`) and `k r²` (`α = 1`) are the only power-law central potentials all of
whose bounded orbits close (a theorem of the literature, stated as a proposition and tested numerically). -/
def bertrand_theorem : Prop := ∀ α : ℝ, closes_all_bounded_orbits α ↔ (α = -2 ∨ α = 1)
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.CentralForces.bertrand_theorem, cm-bertrand-test]

/-! ## exr-clm-circular-orbit -/

/-- Kepler: `ω_r² = U''(r₀)/m = k/(m r₀³)` with `r₀ = L²/(mk)` … -/
theorem kepler_radial_freq_sq (L m k : ℝ) (hL : L ≠ 0) (hm : m ≠ 0) (hk : k ≠ 0) :
    m ^ 3 * k ^ 4 / L ^ 6 / m = k / (m * (L ^ 2 / (m * k)) ^ 3) := by
  veriphy_certified ((field_simp; done))
  -- certificate: cm-poly.sage

/-- … and `ω_φ² = (L/(m r₀²))² = k/(m r₀³)`: one radial oscillation per revolution, the orbit closes. -/
theorem kepler_angular_freq_sq (L m k : ℝ) (hL : L ≠ 0) (hm : m ≠ 0) (hk : k ≠ 0) :
    (L / (m * (L ^ 2 / (m * k)) ^ 2)) ^ 2 = k / (m * (L ^ 2 / (m * k)) ^ 3) := by
  veriphy_certified ((field_simp; done))
  -- certificate: cm-poly.sage

/-- Oscillator: `U = L²/(2mr²) + k r²/2` has `U' = -L²/(m r³) + k r` … -/
theorem osc_U_deriv (L m k r : ℝ) (hr : r ≠ 0) (hm : m ≠ 0) :
    deriv (fun r => L ^ 2 / (2 * m * r ^ 2) + k * r ^ 2 / 2) r = -L ^ 2 / (m * r ^ 3) + k * r := by
  VERIPHY[cm-derivative]

/-- … vanishing at `r₀⁴ = L²/(mk)` … -/
theorem osc_circular_radius (L m k r₀ : ℝ) (hm : m ≠ 0) (hr : r₀ ≠ 0) (hr₀ : m * k * r₀ ^ 4 = L ^ 2) :
    -L ^ 2 / (m * r₀ ^ 3) + k * r₀ = 0 := by
  rw [← hr₀]
  veriphy_certified ((field_simp; ring1))
  -- certificate: cm-poly.sage

/-- … with `U'' = 3L²/(m r⁴) + k` … -/
theorem osc_U_second_deriv (L m k r : ℝ) (hr : r ≠ 0) (hm : m ≠ 0) :
    deriv (fun r => -L ^ 2 / (m * r ^ 3) + k * r) r = 3 * L ^ 2 / (m * r ^ 4) + k := by
  VERIPHY[cm-derivative]

/-- … so `ω_r² = U''(r₀)/m = 4k/m` … -/
theorem osc_radial_freq_sq (L m k r₀ : ℝ) (hm : m ≠ 0) (hr : r₀ ≠ 0) (hr₀ : m * k * r₀ ^ 4 = L ^ 2) :
    (3 * L ^ 2 / (m * r₀ ^ 4) + k) / m = 4 * k / m := by
  rw [← hr₀]
  veriphy_certified ((field_simp; ring1))
  -- certificate: cm-poly.sage

/-- … and `ω_φ² = (L/(m r₀²))² = k/m`: two radial oscillations per revolution. -/
theorem osc_angular_freq_sq (L m k r₀ : ℝ) (hm : m ≠ 0) (hr : r₀ ≠ 0) (hr₀ : m * k * r₀ ^ 4 = L ^ 2) :
    (L / (m * r₀ ^ 2)) ^ 2 = k / m := by
  rw [div_pow, ← hr₀]
  veriphy_certified ((field_simp; done))
  -- certificate: cm-poly.sage

/-! ## exr-clm-binet -/

/-- `ṙ = -u̇/u² = -u' φ̇/u² = -(L/m) u'` with `φ̇ = L u²/m`. -/
theorem binet_rdot (L m u u' : ℝ) (hu : u ≠ 0) (hm : m ≠ 0) : -(u' * (L * u ^ 2 / m)) / u ^ 2 = -(L / m) * u' := by
  veriphy_certified ((field_simp; done))
  -- certificate: cm-poly.sage

/-- `r̈ = -(L/m) u'' φ̇ = -(L²/m²) u² u''`. -/
theorem binet_rddot (L m u u'' : ℝ) : -(L / m) * u'' * (L * u ^ 2 / m) = -(L ^ 2 / m ^ 2) * u ^ 2 * u'' := by
  veriphy_certified (ring1)
  -- certificate: cm-poly.sage

/-- The radial equation `m r̈ - m r φ̇² = f`, that is `-(L²/m) u² u'' - (L²/m) u³ = f`, is Binet's equation
`u'' + u = -m f/(L² u²)`. -/
theorem binet_equation (L m u u'' f : ℝ) (hu : u ≠ 0) (hm : m ≠ 0) (hL : L ≠ 0)
    (h : -(L ^ 2 / m) * u ^ 2 * u'' - (L ^ 2 / m) * u ^ 3 = f) : u'' + u = -(m / (L ^ 2 * u ^ 2)) * f := by
  rw [← h]
  veriphy_certified ((field_simp; ring1))
  -- certificate: cm-poly.sage

/-- For `f = -k u²` the right side is the constant `mk/L²`. -/
theorem binet_kepler_rhs (m k L u : ℝ) (hu : u ≠ 0) (hL : L ≠ 0) :
    -(m / (L ^ 2 * u ^ 2)) * (-k * u ^ 2) = m * k / L ^ 2 := by
  veriphy_certified ((field_simp; done))
  -- certificate: cm-poly.sage

/-- `u = (mk/L²)(1 + e cos (φ - φ₀))` solves `u'' + u = mk/L²`. -/
theorem binet_kepler_solution (m k L e φ₀ : ℝ) (hL : L ≠ 0) (u : ℝ → ℝ)
    (hu : u = fun φ => m * k / L ^ 2 * (1 + e * Real.cos (φ - φ₀))) (φ : ℝ) :
    deriv (deriv u) φ + u φ = m * k / L ^ 2 := by
  VERIPHY[cm-ode]

/-! ## exr-clm-circular-stability -/

/-- For `f = -k r^α` (`α ≠ -1`), `U = L²/(2mr²) + k r^{α+1}/(α+1)` has `U' = -L²/(m r³) + k r^α = -L²/(m r³) - f`. -/
theorem powerlaw_U_deriv (L m k α r : ℝ) (hr : 0 < r) (hm : m ≠ 0) (hα : α + 1 ≠ 0) :
    deriv (fun r => L ^ 2 / (2 * m * r ^ 2) + k * r ^ (α + 1) / (α + 1)) r = -L ^ 2 / (m * r ^ 3) + k * r ^ α := by
  VERIPHY[cm-derivative]

/-- `f'(r) = -k α r^{α-1}`. -/
theorem powerlaw_f_deriv (k α r : ℝ) (hr : 0 < r) : deriv (fun r => -k * r ^ α) r = -k * α * r ^ (α - 1) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun r => -k * r ^ α) ((-(r ^ (α - 1)) * k * α : ℝ)) (r) :=
        by
        refine
          (((((hasDerivAt_id' (r : ℝ)).rpow_const (p := ((α : ℝ) : ℝ)) (Or.inl (by exact hr.ne'))).const_mul
                    (-k : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      exact hd.deriv.trans (by ring1))
  -- certificate: cm-derivative.sage

/-- At a circular orbit (`k r₀^α = L²/(m r₀³)`, with `p = r₀^α`, `p₁ = r₀^{α-1}`, `r₀ p₁ = p`),
`U''(r₀) = 3L²/(m r₀⁴) + k α r₀^{α-1} = (3 + α) L²/(m r₀⁴)`. -/
theorem powerlaw_U_second (L m k α r₀ p p₁ : ℝ) (hm : m ≠ 0) (hr : r₀ ≠ 0) (h1 : k * p * (m * r₀ ^ 3) = L ^ 2)
    (h2 : r₀ * p₁ = p) : 3 * L ^ 2 / (m * r₀ ^ 4) + k * α * p₁ = (3 + α) * L ^ 2 / (m * r₀ ^ 4) := by
  rw [← h1, ← h2]
  veriphy_certified ((field_simp; done))
  -- certificate: cm-poly.sage

/-- `U''(r₀) > 0` exactly when `α > -3`: gravity (`α = -2`) and the oscillator (`α = 1`) are stable, the inverse
cube (`α = -3`) is marginal, steeper forces are unstable. -/
theorem powerlaw_stable_iff (L m r₀ α : ℝ) (hL : L ≠ 0) (hm : 0 < m) (hr : 0 < r₀) :
    0 < (3 + α) * L ^ 2 / (m * r₀ ^ 4) ↔ -3 < α := by
  rw [div_pos_iff_of_pos_right (by positivity), mul_pos_iff_of_pos_right (by positivity)]
  constructor <;> intro h <;> linarith
example : veriphy_instance% KaluzaKlein.ClassicalMechanics.CentralForces.powerlaw_stable_iff (1) (1) (1) (-3) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.CentralForces.powerlaw_stable_iff (1) (1) (1) (-2) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.CentralForces.powerlaw_stable_iff (1) (1) (1) (1) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.CentralForces.powerlaw_stable_iff (2) (1) (1) (-4) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ClassicalMechanics.CentralForces.powerlaw_stable_iff
-- test: cm-powerlaw-stable-test.sage

end KaluzaKlein.ClassicalMechanics.CentralForces
