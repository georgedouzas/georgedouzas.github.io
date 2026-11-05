import Mathlib

/-!
# Classical mechanics, section "Kepler's problem"

The claims of the section "Kepler's problem" of `tutorial/part-ii-01-classical-mechanics.qmd`. The conic is checked
against Binet's equation; the energy, the conservation, the orthogonality and the length of the Laplace-Runge-Lenz
vector are identities in the jet variables (positions, momenta, `r² = x² + y² + z²`, `cos² + sin² = 1`); the period
and the numerical values are explicit computations.
-/

open Real Filter Topology

namespace KaluzaKlein.ClassicalMechanics.Kepler

/-! ## prp-clm-kepler-orbits -/

/-- The conic `u = 1/r = (1 + e cos (φ - φ₀))/p` solves Binet's equation `u'' + u = 1/p` (`p = L²/(mk)`). -/
theorem conic_solves_binet (p e φ₀ : ℝ) (hp : p ≠ 0) (u : ℝ → ℝ) (hu : u = fun φ => (1 + e * Real.cos (φ - φ₀)) / p)
    (φ : ℝ) : deriv (deriv u) φ + u φ = 1 / p := by
  VERIPHY[cm-ode]

/-- The energy along the conic: `E = (L²/2m)(u'² + u²) - k u = k(e² - 1)/(2p)`, with `u' = -e sin ψ/p`,
`u = (1 + e cos ψ)/p`, `cos² ψ + sin² ψ = 1` and `L² = mkp`. -/
theorem conic_energy (L m k p e c s : ℝ) (hm : m ≠ 0) (hp : p ≠ 0) (hcs : c ^ 2 + s ^ 2 = 1) (hL : L ^ 2 = m * k * p) :
    L ^ 2 / (2 * m) * ((-e * s / p) ^ 2 + ((1 + e * c) / p) ^ 2) - k * (1 + e * c) / p = k * (e ^ 2 - 1) / (2 * p) := by
  rw [hL]
  field_simp
  VERIPHY[cm-poly]

/-- With `E = k(e² - 1)/(2p)` (`k, p > 0`, `e ≥ 0`): an ellipse for `E < 0` (`e < 1`), a parabola for `E = 0`
(`e = 1`), a hyperbola for `E > 0` (`e > 1`). -/
theorem conic_type_iff (k p e E : ℝ) (hk : 0 < k) (hp : 0 < p) (he : 0 ≤ e) (hE : E = k * (e ^ 2 - 1) / (2 * p)) :
    (E < 0 ↔ e < 1) ∧ (E = 0 ↔ e = 1) ∧ (0 < E ↔ 1 < e) := by
  have hc : 0 < k / (2 * p) := by positivity
  have hE' : E = k / (2 * p) * (e ^ 2 - 1) := by rw [hE]; ring
  rw [hE']
  refine ⟨?_, ?_, ?_⟩
  · rw [mul_neg_iff_of_pos_left hc, sub_neg]
    constructor <;> intro h <;> nlinarith
  · rw [mul_eq_zero, or_iff_right hc.ne', sub_eq_zero, pow_eq_one_iff_of_nonneg he two_ne_zero]
  · rw [mul_pos_iff_of_pos_left hc, sub_pos]
    constructor <;> intro h <;> nlinarith
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.Kepler.conic_type_iff, cm-conic-type-test]

/-- The Laplace-Runge-Lenz vector `A = p × L - mk x/r` is conserved: with `ẋ = p/m`, `ṗ = -k x/r³`, `L̇ = 0`,
`ṙ = x·p/(m r)`, the `x`-component of `r³ Ȧ` is `-k[y L_z - z L_y + p_x r² - x (x·p)] = 0` … -/
theorem lrl_rate_x (k x y z px py pz r : ℝ) (hr : r ^ 2 = x ^ 2 + y ^ 2 + z ^ 2) :
    -k * (y * (x * py - y * px) - z * (z * px - x * pz)) - k * (px * r ^ 2 - x * (x * px + y * py + z * pz)) = 0 := by
  VERIPHY[cm-poly]

/-- … the `y`-component `-k[z L_x - x L_z + p_y r² - y (x·p)] = 0` … -/
theorem lrl_rate_y (k x y z px py pz r : ℝ) (hr : r ^ 2 = x ^ 2 + y ^ 2 + z ^ 2) :
    -k * (z * (y * pz - z * py) - x * (x * py - y * px)) - k * (py * r ^ 2 - y * (x * px + y * py + z * pz)) = 0 := by
  VERIPHY[cm-poly]

/-- … and the `z`-component `-k[x L_y - y L_x + p_z r² - z (x·p)] = 0`. -/
theorem lrl_rate_z (k x y z px py pz r : ℝ) (hr : r ^ 2 = x ^ 2 + y ^ 2 + z ^ 2) :
    -k * (x * (z * px - x * pz) - y * (y * pz - z * py)) - k * (pz * r ^ 2 - z * (x * px + y * py + z * pz)) = 0 := by
  VERIPHY[cm-poly]

/-- `A` is orthogonal to `L`: `(rA) · L = 0` with `L = x × p` and `rA = (p × L) r - mk x`. -/
theorem lrl_orthogonal_L (m k x y z px py pz r : ℝ) :
    ((py * (x * py - y * px) - pz * (z * px - x * pz)) * r - m * k * x) * (y * pz - z * py) +
      ((pz * (y * pz - z * py) - px * (x * py - y * px)) * r - m * k * y) * (z * px - x * pz) +
      ((px * (z * px - x * pz) - py * (y * pz - z * py)) * r - m * k * z) * (x * py - y * px) = 0 := by
  VERIPHY[cm-poly]

/-- `|A|² = m²k² + 2mEL²`: `|rA|² = r² (m²k² + 2mE|L|²)` with `2mrE = r|p|² - 2mk` (`E = p²/2m - k/r`). -/
theorem lrl_norm_sq (m k E x y z px py pz r : ℝ) (hr : r ^ 2 = x ^ 2 + y ^ 2 + z ^ 2)
    (hE : 2 * m * r * E = r * (px ^ 2 + py ^ 2 + pz ^ 2) - 2 * m * k) :
    ((py * (x * py - y * px) - pz * (z * px - x * pz)) * r - m * k * x) ^ 2 +
      ((pz * (y * pz - z * py) - px * (x * py - y * px)) * r - m * k * y) ^ 2 +
      ((px * (z * px - x * pz) - py * (y * pz - z * py)) * r - m * k * z) ^ 2 =
      r ^ 2 * (m ^ 2 * k ^ 2 + 2 * m * E * ((y * pz - z * py) ^ 2 + (z * px - x * pz) ^ 2 + (x * py - y * px) ^ 2)) := by
  VERIPHY[cm-poly]

/-- `|A| = mke`: `m²k² + 2mEL² = (mke)²` with `2pE = k(e² - 1)` and `L² = mkp`. -/
theorem lrl_norm_eccentricity (m k E L p e : ℝ) (hE : 2 * p * E = k * (e ^ 2 - 1)) (hL : L ^ 2 = m * k * p) :
    m ^ 2 * k ^ 2 + 2 * m * E * L ^ 2 = (m * k * e) ^ 2 := by
  VERIPHY[cm-poly]

/-- Kepler's third law: `T = π a b · 2m/L` with `b = a s`, `s = √(1 - e²)`, `L² = mka(1 - e²) = mkas²` gives
`T² = 4π² m a³/k`. -/
theorem kepler_third_law (m k a s L : ℝ) (hk : k ≠ 0) (hL : L ≠ 0) (hL2 : L ^ 2 = m * k * a * s ^ 2) :
    (Real.pi * a * (a * s) * 2 * m / L) ^ 2 = 4 * Real.pi ^ 2 * m * a ^ 3 / k := by
  rw [div_pow, hL2]
  VERIPHY[cm-poly]

/-- `T = √(T²) = √(4π² m a³/k) = 2π √(m a³/k)`. -/
theorem kepler_period (m k a : ℝ) (hm : 0 < m) (hk : 0 < k) (ha : 0 < a) :
    Real.sqrt (4 * Real.pi ^ 2 * m * a ^ 3 / k) = 2 * Real.pi * Real.sqrt (m * a ^ 3 / k) := by
  VERIPHY[cm-rewrite]

/-! ## exm-clm-kepler-circle -/

/-- For `e = 0` the conic is the circle `r = p`: `u = (1 + 0 · cos ψ)/p = 1/p`. -/
theorem circle_radius (p c : ℝ) : (1 + 0 * c) / p = 1 / p := by
  VERIPHY[cm-poly]

/-- Its energy is `E = k(0² - 1)/(2p) = -k/(2p)`. -/
theorem circle_energy (k p : ℝ) : k * ((0 : ℝ) ^ 2 - 1) / (2 * p) = -k / (2 * p) := by
  VERIPHY[cm-poly]

/-- From `m v²/p = k/p²` (that is `m v² p = k`): `v² = k/(mp)` … -/
theorem circle_speed_sq (m k p v : ℝ) (hm : m ≠ 0) (hp : p ≠ 0) (hv : m * v ^ 2 * p = k) : v ^ 2 = k / (m * p) := by
  rw [← hv]
  VERIPHY[cm-poly]

/-- … so the kinetic energy `m v²/2 = k/(2p)` … -/
theorem circle_kinetic (m k p v : ℝ) (hp : p ≠ 0) (hv : m * v ^ 2 * p = k) : m * v ^ 2 / 2 = k / (2 * p) := by
  rw [← hv]
  VERIPHY[cm-poly]

/-- … is half the potential energy `k/p` in absolute value (the virial theorem for `1/r`). -/
theorem circle_virial (k p : ℝ) : k / (2 * p) = (k / p) / 2 := by
  VERIPHY[cm-poly]

/-- The period `2πp/v = 2π √(m p³/k)` with `v = √(k/(mp))`: the third law with `a = p`. -/
theorem circle_period (m k p : ℝ) (hm : 0 < m) (hk : 0 < k) (hp : 0 < p) :
    2 * Real.pi * p / Real.sqrt (k / (m * p)) = 2 * Real.pi * Real.sqrt (m * p ^ 3 / k) := by
  VERIPHY[cm-rewrite]

/-! ## exm-clm-kepler-ellipse -/

/-- Perihelion `r₊ = p/(1 + e) = a(1 - e)` with `a = p/(1 - e²)` … -/
theorem perihelion (p e : ℝ) (h1 : 1 + e ≠ 0) (h2 : 1 - e ^ 2 ≠ 0) : p / (1 + e) = p / (1 - e ^ 2) * (1 - e) := by
  VERIPHY[cm-poly]

/-- … and aphelion `r₋ = p/(1 - e) = a(1 + e)`. -/
theorem aphelion (p e : ℝ) (h1 : 1 - e ≠ 0) (h2 : 1 - e ^ 2 ≠ 0) : p / (1 - e) = p / (1 - e ^ 2) * (1 + e) := by
  VERIPHY[cm-poly]

/-- For the Earth, `e ≈ 0.0167`, the ratio `(1 - e)/(1 + e) ≈ 0.967`. -/
theorem earth_ratio : |(1 - 0.0167) / (1 + 0.0167) - 0.967| < (0.0005 : ℝ) := by
  VERIPHY[cm-rewrite]

/-- With `k = GMm`, `T²/a³ = 4π²/(GM)`, independent of the planet's mass. -/
theorem third_law_constant (G M m a k : ℝ) (hG : G * M ≠ 0) (hm : m ≠ 0) (ha : a ≠ 0) (hk : k = G * M * m) :
    4 * Real.pi ^ 2 * m * a ^ 3 / k / a ^ 3 = 4 * Real.pi ^ 2 / (G * M) := by
  rw [hk]
  VERIPHY[cm-poly]

/-- The same constant for all planets: `T₁² a₂³ = T₂² a₁³` (in years and astronomical units, `T² = a³`). -/
theorem same_constant (C T₁ a₁ T₂ a₂ : ℝ) (h₁ : T₁ ^ 2 = C * a₁ ^ 3) (h₂ : T₂ ^ 2 = C * a₂ ^ 3) :
    T₁ ^ 2 * a₂ ^ 3 = T₂ ^ 2 * a₁ ^ 3 := by
  VERIPHY[cm-poly]

/-! ## exm-clm-kepler-hyperbola -/

/-- For `e > 1`, `r → ∞` where `1 + e cos ψ = 0`, that is `cos ψ = -1/e`. -/
theorem hyperbola_asymptote (e : ℝ) (he : e ≠ 0) : 1 + e * (-1 / e) = 0 := by
  VERIPHY[cm-poly]

/-- The polar angle sweeps `2 arccos (-1/e) > π` between the asymptotes. -/
theorem hyperbola_sweep (e : ℝ) (he : 1 < e) : Real.pi < 2 * Real.arccos (-1 / e) := by
  rw [Real.arccos_eq_pi_div_two_sub_arcsin, neg_div, Real.arcsin_neg]
  have : 0 < Real.arcsin (1 / e) := Real.arcsin_pos.mpr (by positivity)
  linarith
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.Kepler.hyperbola_sweep, cm-hyperbola-sweep-test]

/-- The deflection `χ = 2 arccos (-1/e) - π` has `sin (χ/2) = 1/e`. -/
theorem hyperbola_deflection (e : ℝ) (he : 1 < e) : Real.sin (Real.arccos (-1 / e) - Real.pi / 2) = 1 / e := by
  VERIPHY[cm-rewrite]

/-- Fast or distant particles (`e` large) are deflected little: `χ → 0` as `e → ∞` (for `-c/e` with any `c`,
`c = 1` being the conic's `-1/e`). -/
theorem deflection_limit (c : ℝ) : Tendsto (fun e : ℝ => 2 * Real.arccos (-c / e) - Real.pi) atTop (𝓝 0) := by
  have h1 : Tendsto (fun e : ℝ => -c / e) atTop (𝓝 0) := by
    have : (fun e : ℝ => -c / e) = fun e => -c * e⁻¹ := by funext e; rw [div_eq_mul_inv]
    rw [this]
    simpa using tendsto_inv_atTop_zero.const_mul (-c)
  have h2 : Tendsto (fun e : ℝ => Real.arccos (-c / e)) atTop (𝓝 (Real.arccos 0)) :=
    (Real.continuous_arccos.tendsto 0).comp h1
  rw [Real.arccos_zero] at h2
  have h3 := (h2.const_mul 2).sub_const Real.pi
  simpa [mul_div_cancel₀] using h3
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.Kepler.deflection_limit, cm-deflection-limit-test]

/-! ## exr-clm-lrl-orbit -/

/-- `x · (p × L) = L · (x × p) = L²` with `L = x × p`. -/
theorem triple_product (x y z px py pz Lx Ly Lz : ℝ) (h1 : Lx = y * pz - z * py) (h2 : Ly = z * px - x * pz)
    (h3 : Lz = x * py - y * px) :
    x * (py * Lz - pz * Ly) + y * (pz * Lx - px * Lz) + z * (px * Ly - py * Lx) = Lx ^ 2 + Ly ^ 2 + Lz ^ 2 := by
  VERIPHY[cm-poly]

/-- `A · x = L² - mkr`: `(rA) · x = r (L² - mkr)`. -/
theorem lrl_dot_position (m k x y z px py pz r Lx Ly Lz : ℝ) (hr : r ^ 2 = x ^ 2 + y ^ 2 + z ^ 2)
    (h1 : Lx = y * pz - z * py) (h2 : Ly = z * px - x * pz) (h3 : Lz = x * py - y * px) :
    ((py * Lz - pz * Ly) * r - m * k * x) * x + ((pz * Lx - px * Lz) * r - m * k * y) * y +
      ((px * Ly - py * Lx) * r - m * k * z) * z = r * (Lx ^ 2 + Ly ^ 2 + Lz ^ 2 - m * k * r) := by
  VERIPHY[cm-poly]

/-- Writing `A · x = |A| r cos φ`: `r (|A| cos φ + mk) = L²`. -/
theorem orbit_from_lrl (a r c L2 m k : ℝ) (h : a * r * c = L2 - m * k * r) : r * (a * c + m * k) = L2 := by
  VERIPHY[cm-poly]

/-- That is `r = (L²/(mk))/(1 + (|A|/(mk)) cos φ)`: the conic with `e = |A|/(mk)`, `φ` measured from `A`. -/
theorem orbit_conic_form (a r c L2 m k : ℝ) (hmk : m * k ≠ 0) (hd : 1 + a / (m * k) * c ≠ 0)
    (h : r * (a * c + m * k) = L2) : r = (L2 / (m * k)) / (1 + a / (m * k) * c) := by
  rw [eq_div_iff hd, ← h]
  VERIPHY[cm-poly]

/-! ## exr-clm-kepler-third -/

/-- The period is the area over the areal velocity: `T = π a b / (L/(2m)) = π a b · 2m/L`. -/
theorem period_from_area (a b m L : ℝ) (hL : L ≠ 0) (hm : m ≠ 0) :
    Real.pi * a * b / (L / (2 * m)) = Real.pi * a * b * 2 * m / L := by
  VERIPHY[cm-poly]

/-! ## exr-clm-halley -/

/-- Halley's comet: `a = 75.3^{2/3} ≈ 17.8` (between `17.8` and `17.85`) … -/
theorem halley_axis_lower : (17.8 : ℝ) < (75.3 : ℝ) ^ ((2 : ℝ) / 3) := by
  VERIPHY[cm-rewrite]

/-- … and below `17.85`. -/
theorem halley_axis_upper : (75.3 : ℝ) ^ ((2 : ℝ) / 3) < (17.85 : ℝ) := by
  VERIPHY[cm-rewrite]

/-- Perihelion `a(1 - e) ≈ 17.8 × 0.033 ≈ 0.59 AU`. -/
theorem halley_perihelion : |17.8 * (1 - 0.967) - 0.59| < (0.005 : ℝ) := by
  VERIPHY[cm-rewrite]

/-- Aphelion `a(1 + e) ≈ 17.8 × 1.967 ≈ 35 AU`. -/
theorem halley_aphelion : |17.8 * (1 + 0.967) - 35| < (0.05 : ℝ) := by
  VERIPHY[cm-rewrite]

end KaluzaKlein.ClassicalMechanics.Kepler
