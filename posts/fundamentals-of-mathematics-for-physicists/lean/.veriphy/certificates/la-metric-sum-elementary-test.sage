# name: la-metric-sum-elementary-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.ElementaryTensors.metric_sum_elementary file "la-metric-sum-elementary-test.sage"
# description: g = Σ g_{ij} eⁱ ⊗ eʲ at n = 0 and explicit 2 × 2 and 3 × 3 metrics, entry by entry.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'ext a b\nfin_cases a <;> fin_cases b <;> simp [Matrix.sum_apply, Matrix.vecMulVec_apply, Pi.single_apply, Fin.sum_univ_succ]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0'], True, 'simp'),
    (['2', '!![1, 2; 2, 4]'], False),
    (['3', '!![1, 0, 0; 0, -1, 0; 0, 0, 2]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
