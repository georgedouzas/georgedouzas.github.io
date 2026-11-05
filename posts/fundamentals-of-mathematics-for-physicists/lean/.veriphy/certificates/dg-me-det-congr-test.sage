# name: dg-me-det-congr-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.DifferentialGeometry.Metric.det_congr file "dg-me-det-congr-test.sage"
# description: det (Jᵀ g J) = (det J)² det g at explicit 2×2 and 3×3 matrices (the polar Jacobian at r = 2, φ = 0 with
#   the identity; a singular J (edge); a 3×3 pair), by the closed determinant formulas.
SEED = 0
PROOF = 'norm_num [Matrix.det_fin_two_of, Matrix.det_fin_three, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]'
CASES = [
    (['2', '!![(1 : ℝ), 0; 0, 2]', '!![(1 : ℝ), 0; 0, 1]'], False),
    (['2', '!![(1 : ℝ), 1; 1, 1]', '!![(1 : ℝ), 0; 0, -1]'], False),
    (['2', '!![(1 : ℝ), 2; 2, 4]', '!![(3 : ℝ), 1; 1, 2]'], True),
    (['3', '!![(1 : ℝ), 0, 0; 0, 2, 0; 0, 0, 3]', '!![(1 : ℝ), 0, 0; 0, -1, 0; 0, 0, -1]'], False),
]
cases = CASES
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
