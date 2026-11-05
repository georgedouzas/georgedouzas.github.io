# name: la-span-three-reduce-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Spans.span_three_reduce file "la-span-three-reduce-test.sage"
# description: a combination of v₁, v₂, v₃ = a v₁ + b v₂ is a combination of v₁, v₂, at explicit vectors of ℝ² and ℝ³.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'funext i\nfin_cases i <;> norm_num'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['(Fin 2 → ℝ)', 'inferInstance', 'inferInstance', '0', '0', '0', '0', '0', '0', '0', '0', '(by simp)'], True),
    (['(Fin 2 → ℝ)', 'inferInstance', 'inferInstance', '![1, 0]', '![0, 1]', '![2, 3]', '2', '3', '1', '-1', '1 / 2', '(by funext i; fin_cases i <;> norm_num)'], False),
    (['(Fin 3 → ℝ)', 'inferInstance', 'inferInstance', '![1, 2, 0]', '![0, 1, 1]', '![1, 3, 1]', '1', '1', '2', '0', '-3', '(by funext i; fin_cases i <;> norm_num)'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
