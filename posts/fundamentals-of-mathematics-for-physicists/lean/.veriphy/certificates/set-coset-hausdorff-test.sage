# name: set-coset-hausdorff-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Hausdorff.coset_space_hausdorff
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['KaluzaKlein.Sets.Sets.northPole', '(by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere), T2Space (Quotient (orbitSetoid p₀)) := by\n  intro p₀ hp₀\n  obtain ⟨e, -⟩ := QuotientTopology.sphere_quotient_homeomorph p₀ hp₀\n  exact e.isEmbedding.t2Space\nexact @key (KaluzaKlein.Sets.Sets.northPole) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))'),
    (['![1, 0, 0]', '(by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere), T2Space (Quotient (orbitSetoid p₀)) := by\n  intro p₀ hp₀\n  obtain ⟨e, -⟩ := QuotientTopology.sphere_quotient_homeomorph p₀ hp₀\n  exact e.isEmbedding.t2Space\nexact @key (![1, 0, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))'),
    (['![0, 1, 0]', '(by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole])'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (p₀ : Fin 3 → ℝ) (hp₀ : p₀ ∈ sphere), T2Space (Quotient (orbitSetoid p₀)) := by\n  intro p₀ hp₀\n  obtain ⟨e, -⟩ := QuotientTopology.sphere_quotient_homeomorph p₀ hp₀\n  exact e.isEmbedding.t2Space\nexact @key (![0, 1, 0]) ((by norm_num [KaluzaKlein.Sets.Sets.sphere, KaluzaKlein.Sets.Sets.northPole]))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
