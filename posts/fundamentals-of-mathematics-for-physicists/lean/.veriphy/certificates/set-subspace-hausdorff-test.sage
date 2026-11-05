# name: set-subspace-hausdorff-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Hausdorff.subspace_hausdorff
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℝ × ℝ', 'inferInstance', 'inferInstance', 'KaluzaKlein.Sets.Sets.circle'], False, 'infer_instance'),
    (['Fin 3 → ℝ', 'inferInstance', 'inferInstance', 'KaluzaKlein.Sets.Sets.sphere'], False, 'infer_instance'),
    (['(ℝ × ℝ) × (ℝ × ℝ)', 'inferInstance', 'inferInstance', 'KaluzaKlein.Sets.Sets.circle ×ˢ KaluzaKlein.Sets.Sets.circle'], False, 'infer_instance'),
    (['Matrix (Fin 3) (Fin 3) ℝ', 'inferInstance', 'inferInstance', 'KaluzaKlein.Sets.Sets.SO3'], False, 'infer_instance'),
    (['ℝ', 'inferInstance', 'inferInstance', '∅'], True, 'infer_instance'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
