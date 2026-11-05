# name: ca-num-omega-pow-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.NumbersWorked.ω_pow_eq file "ca-num-omega-pow-test.sage"
# description: e^{2πik/6} = ω^k at k = 0 (edge), 1, 2, 5; each instance closed by exp_nat_mul, congr and ring
SEED = 0
PROOF = 'show Complex.exp _ = Complex.exp (2 * Real.pi * Complex.I / 6) ^ _\nrw [← Complex.exp_nat_mul]\ncongr 1\npush_cast\nring'
CASES = [(['0'], True), (['1'], False), (['2'], False), (['5'], False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]
print([(x.values, x.kind, x.verdict) for x in instances])
