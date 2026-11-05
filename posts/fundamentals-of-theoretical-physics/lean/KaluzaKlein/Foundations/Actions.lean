import Mathlib

/-!
# Actions of fields: displayed computations in jet variables

The partial derivatives of the Klein-Gordon Lagrangian density with respect to the jet variables, the resulting
Euler-Lagrange expression, and the divergence of the canonical energy-momentum tensor of a real scalar in `1 + 1`
dimensions, which is proportional to the field equation.
-/

namespace KaluzaKlein.Actions

/-- The Klein-Gordon Lagrangian density in `1 + 1` dimensions as a function of the jet variables `p` (the field),
`pt` and `px` (its time and space derivatives): `L = ½ pt² − ½ px² − ½ m² p²`. -/
noncomputable def lagrangian (m p pt px : ℝ) : ℝ := (1 / 2) * pt ^ 2 - (1 / 2) * px ^ 2 - (1 / 2) * m ^ 2 * p ^ 2

/-- `∂L/∂p = −m² p`. -/
theorem dL_dp (m p pt px : ℝ) : deriv (fun p => lagrangian m p pt px) p = -m ^ 2 * p := by
  unfold lagrangian
  simp
  ring

/-- `∂L/∂pt = pt` and `∂L/∂px = −px`. -/
theorem dL_dpt_dpx (m p pt px : ℝ) :
    deriv (fun pt => lagrangian m p pt px) pt = pt ∧ deriv (fun px => lagrangian m p pt px) px = -px := by
  unfold lagrangian
  constructor <;> simp

/-- The Euler-Lagrange expression `∂_t(∂L/∂pt) + ∂_x(∂L/∂px) − ∂L/∂p`, with the second derivatives `ptt`, `pxx`
of the field substituted, is `ptt − pxx + m² p = (□ + m²) p` with `□ = ∂²_t − ∂²_x` (signature `(+, −)`): the
Klein-Gordon equation. -/
theorem euler_lagrange_kg (m p ptt pxx : ℝ) : ptt + (-pxx) - (-m ^ 2 * p) = (ptt - pxx) + m ^ 2 * p := by ring

/-- The canonical energy-momentum tensor of the scalar in `1 + 1` dimensions, `T^{μν} = ∂^μ p ∂^ν p − η^{μν} L`,
with `η = diag(1, −1)`: its divergence `∂_μ T^{μν}` equals `(□p + m² p) ∂^ν p` identically in the jet variables
(`ptx` is the mixed second derivative, symmetric), hence vanishes on the field equation. Stated for `ν = t`, where
`∂^t p = pt`. -/
theorem energy_momentum_divergence_t (m p pt px ptt ptx pxx : ℝ) :
    -- ∂_t T^{tt} + ∂_x T^{xt}, with T^{tt} = ½ pt² + ½ px² + ½ m² p² and T^{xt} = −pt px
    (pt * ptt + px * ptx + m ^ 2 * p * pt) + (-(ptx * px) - pt * pxx) = ((ptt - pxx) + m ^ 2 * p) * pt := by
  ring

/-- The same for `ν = x`: `∂_t T^{tx} + ∂_x T^{xx} = (□p + m² p) ∂^x p` with `∂^x p = −px`, `T^{tx} = −pt px` and
`T^{xx} = ½ pt² + ½ px² − ½ m² p²`. -/
theorem energy_momentum_divergence_x (m p pt px ptt ptx pxx : ℝ) :
    (-(ptt * px) - pt * ptx) + (pt * ptx + px * pxx - m ^ 2 * p * px) = ((ptt - pxx) + m ^ 2 * p) * (-px) := by
  ring

end KaluzaKlein.Actions
