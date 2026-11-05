# name: ca-holo-conformal-iff-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.conformal_iff
# description: conformal iff holomorphic or antiholomorphic, for w, conj w, exp; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', '0'], 'exact conformalAt_iff_differentiableAt_or_differentiableAt_comp_conj', True),
    (['(fun w : ℂ => (starRingEnd ℂ) w)', '1'], 'exact conformalAt_iff_differentiableAt_or_differentiableAt_comp_conj', False),
    (['Complex.exp', 'Complex.I'], 'exact conformalAt_iff_differentiableAt_or_differentiableAt_comp_conj', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
