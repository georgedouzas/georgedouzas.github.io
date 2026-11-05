import KaluzaKlein.Geometry.Killing

/-!
# The non-abelian gauge transformation from an internal diffeomorphism (D5, task 5.2)

On `M_4 × S²` with the ansatz `ĝ_{μm} = h_{mn} K^n_a A^a_μ`, the internal coordinate change `y^m ↦ y^m + ε^a(x) K^m_a(y)`
changes the metric, to first order, by the Lie derivative along `ξ = ε^a K_a` together with the term `∂_μ ξ^n h_{nm}`
from the `x`-dependence of the parameter. Both sides are linear in the internal metric, so the sign of the internal
block (`h = -γ` in the mostly-minus signature of the tutorial, `γ = diag(1, sin²θ)` the round metric) drops out;
the identity is stated with the round metric `sphere`. The result is again of the ansatz form with
`δA^c_μ = ∂_μ ε^c + ε_{abc} A^a_μ ε^b`, the Yang-Mills transformation with structure constants `f_{ab}{}^c = -ε_{abc}`,
the brackets of the sphere's Killing vectors in Part I. The identity is proved on the chart domain `sin θ ≠ 0` as
an identity of functions of `(θ, φ)` with the jets `A^a_μ`, `ε^a`, `∂_μ ε^a` as parameters, for each fixed `μ`.
-/

namespace KaluzaKlein.NonAbelianKK.Gauge

open KaluzaKlein.Geometry CoordinateGeometry

/-- The three Killing vectors of the sphere as one family, `K 0 = K₁`, `K 1 = K₂`, `K 2 = K₃`. -/
noncomputable def K : Fin 3 → Fin 2 → (Fin 2 → ℝ) → ℝ := ![killingX, killingY, killingZ]

/-- The Levi-Civita symbol `ε_{abc}` on three indices. -/
def levi : Fin 3 → Fin 3 → Fin 3 → ℝ :=
  ![![![0, 0, 0], ![0, 0, 1], ![0, -1, 0]],
    ![![0, 0, -1], ![0, 0, 0], ![1, 0, 0]],
    ![![0, 1, 0], ![-1, 0, 0], ![0, 0, 0]]]

/-- The mixed component `ĝ_{μm} = Σ_a h_{mn} K^n_a A^a` of the ansatz, for a fixed spacetime index `μ`
(`A a = A^a_μ`), as a function on the sphere, written with the round metric `γ` in place of `h = -γ` (the overall
sign cancels in `gauge_transformation`). -/
noncomputable def mixed (A : Fin 3 → ℝ) (m : Fin 2) : (Fin 2 → ℝ) → ℝ :=
  fun y => ∑ a, ∑ n, sphere m n y * K a n y * A a

/-- The internal vector field `ξ^n = Σ_a ε^a K^n_a` of the transformation. -/
noncomputable def xi (ε : Fin 3 → ℝ) (n : Fin 2) : (Fin 2 → ℝ) → ℝ :=
  fun y => ∑ a, ε a * K a n y

/-- The first-order change of `ĝ_{μm}` under `y ↦ y + ξ`: the Lie derivative `ξ^n ∂_n ĝ_{μm} + ∂_m ξ^n ĝ_{μn}`
plus `(∂_μ ξ^n) h_{nm}` with `∂_μ ξ^n = Σ_a (∂_μ ε^a) K^n_a` (`dε a = ∂_μ ε^a`). -/
noncomputable def deltaMixed (A ε dε : Fin 3 → ℝ) (m : Fin 2) (y : Fin 2 → ℝ) : ℝ :=
  ∑ n, xi ε n y * pd n (mixed A m) y + ∑ n, pd m (xi ε n) y * mixed A n y
    + ∑ n, (∑ a, dε a * K a n y) * sphere n m y

/-- `δĝ_{μm} = Σ_c h_{mn} K^n_c (∂_μ ε^c + ε_{abc} A^a_μ ε^b)`: the transformed metric is the ansatz for
`A + δA` with `δA^c = ∂_μ ε^c + ε_{abc} A^a ε^b`. -/
theorem gauge_transformation (A ε dε : Fin 3 → ℝ) (m : Fin 2) (y : Fin 2 → ℝ) (h : Real.sin (y 0) ≠ 0) :
    deltaMixed A ε dε m y =
      ∑ c, (∑ n, sphere m n y * K c n y) * (dε c + ∑ a, ∑ b, levi a b c * A a * ε b) := by
  have hc3 : ∀ x : ℝ, Real.cos x ^ 3 = Real.cos x * (1 - Real.sin x ^ 2) := fun x => by
    rw [show (3 : ℕ) = 2 + 1 from rfl, pow_succ, Real.cos_sq']; ring
  have hc4 : ∀ x : ℝ, Real.cos x ^ 4 = (1 - Real.sin x ^ 2) ^ 2 := fun x => by
    rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, Real.cos_sq']
  fin_cases m <;>
    simp (disch :=
      first
      | assumption
      | fun_prop) [deltaMixed, mixed, xi, K, levi, CoordinateGeometry.pd, KaluzaKlein.Geometry.sphere,
      KaluzaKlein.Geometry.killingX, KaluzaKlein.Geometry.killingY, KaluzaKlein.Geometry.killingZ,
      Fin.sum_univ_succ, Function.update_apply, deriv_fun_pow, deriv_fun_mul, deriv_fun_add,
      deriv.fun_neg, deriv_fun_div, deriv_const_mul, deriv_mul_const, Real.deriv_sin, Real.deriv_cos]
  all_goals
    (try field_simp)
    ring_nf
    simp only [Real.cos_sq', hc3, hc4]
    ring1

end KaluzaKlein.NonAbelianKK.Gauge
