# name: la-determined-by-basis-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearMaps.determined_by_basis file "la-determined-by-basis-test.sage"
# description: two linear maps agreeing on a basis are equal, at the zero map, the identity and a matrix map on ℝ².
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'rfl'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['(Fin 2)', '(Fin 2 → ℝ)', '(Fin 2 → ℝ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℝ (Fin 2)', '0', '0', '(fun _ => rfl)'], True),
    (['(Fin 2)', '(Fin 2 → ℝ)', '(Fin 2 → ℝ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℝ (Fin 2)', 'LinearMap.id', 'LinearMap.id', '(fun _ => rfl)'], False),
    (['(Fin 2)', '(Fin 2 → ℝ)', '(Fin 2 → ℝ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℝ (Fin 2)', "Matrix.toLin' !![2, -1; 0, 0]", "Matrix.toLin' !![2, -1; 0, 0]", '(fun _ => rfl)'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
