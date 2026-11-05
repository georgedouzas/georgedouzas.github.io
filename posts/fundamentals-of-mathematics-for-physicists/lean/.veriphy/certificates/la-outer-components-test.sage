# name: la-outer-components-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.ElementaryTensors.outer_components file "la-outer-components-test.sage"
# description: (v ⊗ w)^{ia} = v^i w^a for explicit vectors, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [Matrix.vecMulVec_apply]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['1', '1', '![0]', '![5]', '0', '0'], True),
    (['3', '3', '![1, 2, 3]', '![1, 0, -1]', '2', '0'], False),
    (['2', '3', '![1 / 2, -1]', '![2, 0, 4]', '1', '2'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
