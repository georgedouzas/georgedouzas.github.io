# name: set-rectangle-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.OpenSets.rectangle_open
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', 'Real.pi', '0', '2 * Real.pi'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b c d : ℝ), IsOpen (Ioo a b ×ˢ Ioo c d) ∧ IsOpen (univ : Set (ℝ × ℝ)) := by\n  intro a b c d\n  exact (⟨isOpen_Ioo.prod isOpen_Ioo, isOpen_univ⟩)\nexact @key (0) (Real.pi) (0) (2 * Real.pi)'),
    (['0', '0', '0', '0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b c d : ℝ), IsOpen (Ioo a b ×ˢ Ioo c d) ∧ IsOpen (univ : Set (ℝ × ℝ)) := by\n  intro a b c d\n  exact (⟨isOpen_Ioo.prod isOpen_Ioo, isOpen_univ⟩)\nexact @key (0) (0) (0) (0)'),
    (['-1', '1', '2', '3'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b c d : ℝ), IsOpen (Ioo a b ×ˢ Ioo c d) ∧ IsOpen (univ : Set (ℝ × ℝ)) := by\n  intro a b c d\n  exact (⟨isOpen_Ioo.prod isOpen_Ioo, isOpen_univ⟩)\nexact @key (-1) (1) (2) (3)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
