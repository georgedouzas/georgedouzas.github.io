# name: ca-holo-cayley-disc-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_disc
# description: |f(z)| < 1 at z = i, 1 + i, -3 + 2i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['Complex.I', '(by simp)'], 'have h2 : (Complex.I) + Complex.I ≠ 0 := by norm_num [Complex.ext_iff]\nrw [norm_div, div_lt_one (norm_pos_iff.mpr h2), ← sq_lt_sq₀ (norm_nonneg _) (norm_nonneg _),\n  ← Complex.normSq_eq_norm_sq, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply, Complex.normSq_apply]\nsimp <;> nlinarith', True),
    (['(1 + Complex.I)', '(by simp)'], 'have h2 : (1 + Complex.I) + Complex.I ≠ 0 := by norm_num [Complex.ext_iff]\nrw [norm_div, div_lt_one (norm_pos_iff.mpr h2), ← sq_lt_sq₀ (norm_nonneg _) (norm_nonneg _),\n  ← Complex.normSq_eq_norm_sq, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply, Complex.normSq_apply]\nsimp <;> nlinarith', False),
    (['(-3 + 2 * Complex.I)', '(by simp)'], 'have h2 : (-3 + 2 * Complex.I) + Complex.I ≠ 0 := by norm_num [Complex.ext_iff]\nrw [norm_div, div_lt_one (norm_pos_iff.mpr h2), ← sq_lt_sq₀ (norm_nonneg _) (norm_nonneg _),\n  ← Complex.normSq_eq_norm_sq, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply, Complex.normSq_apply]\nsimp <;> nlinarith', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
