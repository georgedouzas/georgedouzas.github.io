# name: ca-holo-polynomial-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.polynomial_entire
# description: the polynomials 0, X, X^2 + 1 are entire; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(0 : Polynomial ℂ)'], 'simp only [Polynomial.eval_add, Polynomial.eval_pow, Polynomial.eval_X, Polynomial.eval_one, Polynomial.eval_zero]\nfun_prop', True),
    (['(Polynomial.X : Polynomial ℂ)'], 'simp only [Polynomial.eval_add, Polynomial.eval_pow, Polynomial.eval_X, Polynomial.eval_one, Polynomial.eval_zero]\nfun_prop', False),
    (['(Polynomial.X ^ 2 + 1 : Polynomial ℂ)'], 'simp only [Polynomial.eval_add, Polynomial.eval_pow, Polynomial.eval_X, Polynomial.eval_one, Polynomial.eval_zero]\nfun_prop', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
