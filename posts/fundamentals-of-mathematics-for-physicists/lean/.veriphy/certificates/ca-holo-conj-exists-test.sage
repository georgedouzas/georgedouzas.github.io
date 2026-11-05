# name: ca-holo-conj-exists-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.conjugate_exists
# description: harmonic conjugates of Re z and Im z on discs; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun z : ℂ => z.re)', '0', '0', '(fun z hz => absurd hz (by simp))'], 'exact ⟨fun z => z, by fun_prop, fun _ _ => rfl⟩', True),
    (['(fun z : ℂ => z.re)', '0', '1', '(fun _ _ => analyticAt_id.harmonicAt_re)'], 'exact ⟨fun z => z, by fun_prop, fun _ _ => rfl⟩', False),
    (['(fun z : ℂ => z.im)', 'Complex.I', '2', '(fun _ _ => analyticAt_id.harmonicAt_im)'], 'exact ⟨fun z => -Complex.I * z, by fun_prop, fun _ _ => by simp⟩', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
