# name: set-parabola-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.OpenSets.parabola_hyperbola
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '(0, 0)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (c : ℝ) (p : ℝ × ℝ), IsOpen {q : ℝ × ℝ | q.2 > q.1 ^ 2 + c} ∧ IsClosed {q : ℝ × ℝ | q.1 * q.2 = c} ∧ IsOpen ({p}ᶜ : Set (ℝ × ℝ)) := by\n  intro c p\n  exact (⟨isOpen_lt (by fun_prop) (by fun_prop), isClosed_eq (by fun_prop) (by fun_prop), isOpen_compl_singleton⟩)\nexact @key (0) ((0, 0))'),
    (['1', '(1, 2)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (c : ℝ) (p : ℝ × ℝ), IsOpen {q : ℝ × ℝ | q.2 > q.1 ^ 2 + c} ∧ IsClosed {q : ℝ × ℝ | q.1 * q.2 = c} ∧ IsOpen ({p}ᶜ : Set (ℝ × ℝ)) := by\n  intro c p\n  exact (⟨isOpen_lt (by fun_prop) (by fun_prop), isClosed_eq (by fun_prop) (by fun_prop), isOpen_compl_singleton⟩)\nexact @key (1) ((1, 2))'),
    (['-2', '(3, -1)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (c : ℝ) (p : ℝ × ℝ), IsOpen {q : ℝ × ℝ | q.2 > q.1 ^ 2 + c} ∧ IsClosed {q : ℝ × ℝ | q.1 * q.2 = c} ∧ IsOpen ({p}ᶜ : Set (ℝ × ℝ)) := by\n  intro c p\n  exact (⟨isOpen_lt (by fun_prop) (by fun_prop), isClosed_eq (by fun_prop) (by fun_prop), isOpen_compl_singleton⟩)\nexact @key (-2) ((3, -1))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
