# name: ca-holo-exp-family-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.exp_family_entire
# description: e^{az}, cos az, sin az, cosh az, sinh az entire at a = 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0'], 'refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> fun_prop', True),
    (['1'], 'refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> fun_prop', False),
    (['Complex.I'], 'refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> fun_prop', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
