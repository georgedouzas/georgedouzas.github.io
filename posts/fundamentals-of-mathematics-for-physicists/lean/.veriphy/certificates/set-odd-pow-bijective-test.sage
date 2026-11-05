# name: set-odd-pow-bijective-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Bijective.odd_pow_bijective
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (k : ℕ), Function.Bijective (fun x : ℝ => x ^ (2 * k + 1)) := by\n  intro k\n  exact (⟨(Odd.strictMono_pow (⟨k, rfl⟩ : Odd (2 * k + 1))).injective, Surjective.odd_pow_surjective k⟩)\nexact @key (0)'),
    (['1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (k : ℕ), Function.Bijective (fun x : ℝ => x ^ (2 * k + 1)) := by\n  intro k\n  exact (⟨(Odd.strictMono_pow (⟨k, rfl⟩ : Odd (2 * k + 1))).injective, Surjective.odd_pow_surjective k⟩)\nexact @key (1)'),
    (['2'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (k : ℕ), Function.Bijective (fun x : ℝ => x ^ (2 * k + 1)) := by\n  intro k\n  exact (⟨(Odd.strictMono_pow (⟨k, rfl⟩ : Odd (2 * k + 1))).injective, Surjective.odd_pow_surjective k⟩)\nexact @key (2)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
