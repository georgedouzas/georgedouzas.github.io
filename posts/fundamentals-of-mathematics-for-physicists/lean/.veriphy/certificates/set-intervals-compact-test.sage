# name: set-intervals-compact-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Compact.intervals_compact
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '1', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b : ℝ) (hab : a < b), IsCompact (Icc a b) ∧ ¬ IsCompact (Ioo a b) ∧ ¬ IsCompact (univ : Set ℝ) := by\n  intro a b hab\n  refine ⟨isCompact_Icc, fun h => ?_, noncompact_univ ℝ⟩\n  have hcl := h.isClosed.closure_eq\n  rw [closure_Ioo hab.ne] at hcl\n  have : a ∈ Ioo a b := hcl ▸ left_mem_Icc.2 hab.le\n  exact lt_irrefl a this.1\nexact @key (0) (1) ((by norm_num))'),
    (['-1', '1', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b : ℝ) (hab : a < b), IsCompact (Icc a b) ∧ ¬ IsCompact (Ioo a b) ∧ ¬ IsCompact (univ : Set ℝ) := by\n  intro a b hab\n  refine ⟨isCompact_Icc, fun h => ?_, noncompact_univ ℝ⟩\n  have hcl := h.isClosed.closure_eq\n  rw [closure_Ioo hab.ne] at hcl\n  have : a ∈ Ioo a b := hcl ▸ left_mem_Icc.2 hab.le\n  exact lt_irrefl a this.1\nexact @key (-1) (1) ((by norm_num))'),
    (['2', '5', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b : ℝ) (hab : a < b), IsCompact (Icc a b) ∧ ¬ IsCompact (Ioo a b) ∧ ¬ IsCompact (univ : Set ℝ) := by\n  intro a b hab\n  refine ⟨isCompact_Icc, fun h => ?_, noncompact_univ ℝ⟩\n  have hcl := h.isClosed.closure_eq\n  rw [closure_Ioo hab.ne] at hcl\n  have : a ∈ Ioo a b := hcl ▸ left_mem_Icc.2 hab.le\n  exact lt_irrefl a this.1\nexact @key (2) (5) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
