# name: la-elementary-sum-left-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.ElementaryTensors.elementary_sum_left file "la-elementary-sum-left-test.sage"
# description: u ⊗ v + c u ⊗ z = u ⊗ (v + c z) at explicit vectors, entry by entry.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'ext i j\nfin_cases i <;> fin_cases j <;> norm_num [Matrix.vecMulVec_apply]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['1', '1', '![0]', '![0]', '![0]', '0'], True),
    (['2', '3', '![1, 2]', '![1, 0, -1]', '![0, 1, 1]', '2'], False),
    (['3', '2', '![1, -1, 0]', '![1 / 2, 3]', '![2, 2]', '-1'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
