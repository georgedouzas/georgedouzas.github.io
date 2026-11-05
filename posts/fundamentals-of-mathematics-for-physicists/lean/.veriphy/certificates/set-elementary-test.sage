# name: set-elementary-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Continuous.elementary_continuous
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '1', '1', 'Polynomial.X'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b c : ℝ) (p : Polynomial ℝ), Continuous (fun x => a * Real.cos x + b * Real.sin x + c * Real.exp x + p.eval x) := by\n  intro a b c p\n  have := p.continuous\n  fun_prop\nexact @key (1) (1) (1) (Polynomial.X)'),
    (['0', '0', '0', '0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b c : ℝ) (p : Polynomial ℝ), Continuous (fun x => a * Real.cos x + b * Real.sin x + c * Real.exp x + p.eval x) := by\n  intro a b c p\n  have := p.continuous\n  fun_prop\nexact @key (0) (0) (0) (0)'),
    (['2', '-1', '3', 'Polynomial.X ^ 3 - Polynomial.X'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b c : ℝ) (p : Polynomial ℝ), Continuous (fun x => a * Real.cos x + b * Real.sin x + c * Real.exp x + p.eval x) := by\n  intro a b c p\n  have := p.continuous\n  fun_prop\nexact @key (2) (-1) (3) (Polynomial.X ^ 3 - Polynomial.X)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
