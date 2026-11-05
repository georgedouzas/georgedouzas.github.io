# name: set-rn-succ-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Products.rn_succ_prod
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ), Function.Bijective (fun x : Fin (n + 1) → ℝ => (x 0, Fin.tail x)) := by\n  intro n\n  exact ((Fin.consEquiv (fun _ : Fin (n + 1) => ℝ)).symm.bijective)\nexact @key (0)'),
    (['1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ), Function.Bijective (fun x : Fin (n + 1) → ℝ => (x 0, Fin.tail x)) := by\n  intro n\n  exact ((Fin.consEquiv (fun _ : Fin (n + 1) => ℝ)).symm.bijective)\nexact @key (1)'),
    (['3'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ), Function.Bijective (fun x : Fin (n + 1) → ℝ => (x 0, Fin.tail x)) := by\n  intro n\n  exact ((Fin.consEquiv (fun _ : Fin (n + 1) => ℝ)).symm.bijective)\nexact @key (3)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
